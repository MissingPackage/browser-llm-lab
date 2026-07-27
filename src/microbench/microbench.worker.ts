import { probeWebGPU } from "../probe";
import { MICROBENCH_SCHEMA_VERSION, type MicrobenchRunFile } from "./mbSchema";
import { runMicrobench } from "./runner";

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

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("requestAdapter ha restituito null");
    // Chiediamo le feature che servono alla misura, SOLO se l'adapter le espone
    // (a differenza di WebLLM non serve girare ovunque: è uno strumento di misura).
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
    if (adapter.features.has("shader-f16")) requiredFeatures.push("shader-f16");
    // Limiti al massimo dell'adapter: senza requiredLimits il device nasce coi default
    // WebGPU (binding 128 MiB) e le taglie grandi falliscono la validazione in silenzio
    // — osservato dal vivo al primo run 4090 (cella f32 8192² garbage). Stesso errore
    // concettuale del cap hardcoded documentato in buffer-limit-2gb.md.
    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
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
