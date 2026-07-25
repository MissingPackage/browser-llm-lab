import { isMainToWorker, type WorkerToMain } from "./protocol";
import type { InferenceAdapter } from "./adapters/types";
import type { DeviceProbe, BenchCell } from "./schema";
import { computeGenMetrics } from "./metrics";
import { PROMPT_512 } from "./promptset";

export class BenchServer {
  private deps: {
    adapterFactory: () => InferenceAdapter;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
  };

  constructor(deps: {
    adapterFactory: () => InferenceAdapter;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
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
        try {
          const load = await adapter.load(msg.modelId, (text, progress) =>
            this.deps.post({ type: "progress", text, progress }),
          );
          this.deps.post({ type: "progress", text: "generating…", progress: 1 });
          const timeline = await adapter.generate({ prompt: PROMPT_512.text, maxTokens: 256 });
          const cell: BenchCell = {
            stack: adapter.id,
            modelId: msg.modelId,
            quant: msg.quant,
            promptId: PROMPT_512.id,
            load,
            gen: computeGenMetrics(timeline),
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
