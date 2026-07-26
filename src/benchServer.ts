import { isMainToWorker, type WorkerToMain } from "./protocol";
import type { InferenceAdapter } from "./adapters/types";
import type { DeviceProbe, BenchCell, LoadReport, StackId, WarmupPolicy } from "./schema";
import { computeGenMetrics, aggregateReplicates } from "./metrics";
import { PROMPT_512 } from "./promptset";
import { STACK_FIXED_QUANT } from "./stacks";

const DEFAULT_REPLICATE_COUNT = 3;
const HIGH_VARIANCE_THRESHOLD = 0.15; // stdev/mean sul tok/s aggregato

/**
 * Politica di riscaldamento (ruling PI, docket #5b). Le due modalità **misurano cose
 * diverse e non vanno confrontate fra loro**:
 *
 * - `"always"` (default, uso benchmark) — una generazione di riscaldamento a carico
 *   identico viene eseguita e scartata prima di ogni cella. Misura lo *steady state*.
 *   Senza, la prima cella di ogni sessione browser è ~7-14% più veloce delle successive
 *   e il confronto cross-stack dipende dall'ordine dei run (misurato su 4090, vedi
 *   `results/methodology/`). Residuo dichiarato: ~3% di dipendenza dall'ordine.
 * - `"never"` (uso divulgativo) — nessun riscaldamento. È ciò che un utente vero
 *   sperimenta al primo colpo, che è l'informazione utile su una pagina pubblica.
 * - `"cold-only"` — riscalda solo a cache fredda. Prima formulazione del ruling,
 *   conservata perché è ciò che ha prodotto i dati in `results/methodology/`.
 *
 * `cacheState: "unknown"` è trattato come freddo: non sapendo, si riscalda.
 * (`WarmupPolicy` vive in `schema.ts`: è parte della forma di `BenchCell.protocol`, v3 docket #7.)
 */
export const DEFAULT_WARMUP_POLICY: WarmupPolicy = "always";

function needsWarmup(policy: WarmupPolicy, cacheState: LoadReport["cacheState"]): boolean {
  if (policy === "never") return false;
  if (policy === "always") return true;
  return cacheState !== "warm";
}

export class BenchServer {
  private deps: {
    adapters: Record<StackId, () => InferenceAdapter>;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
    replicateCount?: number;
    warmup?: WarmupPolicy;
  };

  constructor(deps: {
    adapters: Record<StackId, () => InferenceAdapter>;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
    replicateCount?: number;
    warmup?: WarmupPolicy;
  }) {
    this.deps = deps;
  }

  async handle(msg: unknown): Promise<void> {
    if (!isMainToWorker(msg)) {
      this.deps.post({ type: "error", message: "invalid message" });
      return;
    }
    try {
      if (msg.type === "probe") {
        this.deps.post({ type: "probe:result", probe: await this.deps.probe() });
      } else if (msg.type === "bench") {
        const fixedQuant = STACK_FIXED_QUANT[msg.stack];
        if (fixedQuant !== undefined && fixedQuant !== msg.quant) {
          this.deps.post({
            type: "error",
            message: `quant mismatch for stack "${msg.stack}": requested "${msg.quant}" but this stack is pinned to "${fixedQuant}"`,
          });
          return;
        }
        const adapter = this.deps.adapters[msg.stack]();
        const replicateCount = this.deps.replicateCount ?? DEFAULT_REPLICATE_COUNT;
        try {
          const load = await adapter.load(msg.modelId, (text, progress) =>
            this.deps.post({ type: "progress", text, progress }),
          );
          const anomalies: string[] = [];
          // Per-run (dal messaggio) prevale sulla politica del server: la pagina pubblica
          // sceglierà per singolo run fra steady-state e prima-esperienza.
          const policy = msg.warmup ?? this.deps.warmup ?? DEFAULT_WARMUP_POLICY;
          const warmupApplied = needsWarmup(policy, load.cacheState);
          if (warmupApplied) {
            this.deps.post({ type: "progress", text: "warm-up (discarded)…", progress: 0 });
            await adapter.generate({ prompt: PROMPT_512.text, maxTokens: 256 });
          }
          const replicates = [];
          for (let i = 0; i < replicateCount; i++) {
            this.deps.post({
              type: "progress",
              text: `generating (replica ${i + 1}/${replicateCount})…`,
              progress: (i + 1) / replicateCount,
            });
            const timeline = await adapter.generate({ prompt: PROMPT_512.text, maxTokens: 256 });
            replicates.push(computeGenMetrics(timeline));
          }
          const gen = aggregateReplicates(replicates);
          if (
            gen.decodeToksPerSec &&
            gen.decodeToksPerSec.mean > 0 &&
            gen.decodeToksPerSec.stdev / gen.decodeToksPerSec.mean > HIGH_VARIANCE_THRESHOLD
          ) {
            anomalies.push(
              `high-variance: decodeToksPerSec stdev/mean=${(gen.decodeToksPerSec.stdev / gen.decodeToksPerSec.mean).toFixed(2)} > ${HIGH_VARIANCE_THRESHOLD}`,
            );
          }
          const cell: BenchCell = {
            stack: adapter.id,
            modelId: msg.modelId,
            quant: msg.quant,
            promptId: PROMPT_512.id,
            load,
            gen,
            replicates,
            protocol: { warmupPolicy: policy, warmupApplied, replicateCount },
            anomalies,
          };
          this.deps.post({ type: "bench:result", cell });
        } finally {
          await adapter.dispose().catch(() => {});
        }
      }
    } catch (e) {
      this.deps.post({ type: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }
}
