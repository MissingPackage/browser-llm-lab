import { probeWebGPU } from "../probe";
import { MICROBENCH_SCHEMA_VERSION, type MicrobenchRunFile } from "./mbSchema";
import { runMicrobench, DEFAULT_SIZES } from "./runner";
import { createEngineDevice } from "../engine/gpudevice";

// Protocollo minimale pagina<->worker, sullo stile di bench.worker.ts.
export type MicrobenchWorkerIn = { type: "run"; deviceLabel: string };
export type MicrobenchWorkerOut =
  | { type: "progress"; message: string }
  | { type: "done"; runFile: MicrobenchRunFile }
  | { type: "error"; message: string };

const post = (m: MicrobenchWorkerOut) => self.postMessage(m);

self.onmessage = async (ev: MessageEvent<MicrobenchWorkerIn>) => {
  if (ev.data.type !== "run") return;
  try {
    const probe = await probeWebGPU(navigator.gpu, navigator as { userAgent: string; deviceMemory?: number });
    if (!probe.webgpu) throw new Error("WebGPU non disponibile nel worker");

    // Limiti DERIVATI dal consumatore (fase 4d): il binding più grande dello
    // sweep è la matrice f32 della cella massima di DEFAULT_SIZES (16384² =
    // 1 GiB). Prima si chiedeva il massimo dell'adapter — il difetto che
    // gpulimits.ts documenta "nell'altra direzione"; senza requiredLimits il
    // device nasce coi default (binding 128 MiB) e le taglie grandi falliscono
    // la validazione in silenzio (osservato: primo run 4090, f32 8192² garbage).
    // Le celle oltre i limiti CONCESSI restano skip espliciti nel runner.
    const maxCellBytes = Math.max(...DEFAULT_SIZES.map((s) => s.rowsN * s.colsK * 4));
    const { device } = await createEngineDevice({
      label: "microbench",
      // feature SOLO se l'adapter le espone (strumento di misura, non serve girare ovunque)
      optionalFeatures: ["timestamp-query", "shader-f16"],
      needs: {
        ctxMax: 1, mlaAttention: false, kvBytesPerLayer: 0,
        extraBindings: [{ bytes: maxCellBytes, consumer: "microbench: matrice f32 della cella massima del sweep" }],
      },
    });

    const { cells, skipped } = await runMicrobench(device, (message) =>
      post({ type: "progress", message }),
    );
    const runFile: MicrobenchRunFile = {
      schemaVersion: MICROBENCH_SCHEMA_VERSION,
      kind: "microbench-matmul",
      deviceLabel: ev.data.deviceLabel,
      ts: new Date().toISOString(),
      probe,
      cells,
      skipped,
    };
    post({ type: "done", runFile });
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
