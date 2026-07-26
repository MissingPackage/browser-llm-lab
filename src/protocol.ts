import type { DeviceProbe, BenchCell, StackId } from "./schema";
import type { WarmupPolicy } from "./benchServer";

export type MainToWorker =
  | { type: "probe" }
  | { type: "bench"; stack: StackId; modelId: string; quant: string; warmup?: WarmupPolicy };

export type WorkerToMain =
  | { type: "probe:result"; probe: DeviceProbe }
  | { type: "progress"; text: string; progress: number }
  | { type: "bench:result"; cell: BenchCell }
  | { type: "error"; message: string };

const STACK_IDS: readonly StackId[] = ["webllm", "transformersjs"];
const WARMUP_POLICIES: readonly WarmupPolicy[] = ["always", "cold-only", "never"];

export function isMainToWorker(x: unknown): x is MainToWorker {
  if (typeof x !== "object" || x === null || !("type" in x)) return false;
  const m = x as Record<string, unknown>;
  if (m.type === "probe") return true;
  if (m.type === "bench") {
    return (
      typeof m.modelId === "string" &&
      typeof m.quant === "string" &&
      typeof m.stack === "string" &&
      (STACK_IDS as string[]).includes(m.stack) &&
      (m.warmup === undefined ||
        (typeof m.warmup === "string" && (WARMUP_POLICIES as string[]).includes(m.warmup)))
    );
  }
  return false;
}
