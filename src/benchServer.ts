import { isMainToWorker, type WorkerToMain } from "./protocol";
import type { InferenceAdapter } from "./adapters/types";
import type { DeviceProbe, BenchCell } from "./schema";
import { computeGenMetrics, aggregateReplicates } from "./metrics";
import { PROMPT_512 } from "./promptset";

const DEFAULT_REPLICATE_COUNT = 3;
const HIGH_VARIANCE_THRESHOLD = 0.15; // stdev/mean sul tok/s aggregato

export class BenchServer {
  private deps: {
    adapterFactory: () => InferenceAdapter;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
    replicateCount?: number;
  };

  constructor(deps: {
    adapterFactory: () => InferenceAdapter;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
    replicateCount?: number;
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
        const adapter = this.deps.adapterFactory();
        const replicateCount = this.deps.replicateCount ?? DEFAULT_REPLICATE_COUNT;
        try {
          const load = await adapter.load(msg.modelId, (text, progress) =>
            this.deps.post({ type: "progress", text, progress }),
          );
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
          const anomalies: string[] = [];
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
