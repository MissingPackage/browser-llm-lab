import type { DeviceProbe, BenchCell } from "./schema";

export type MainToWorker =
  | { type: "probe" }
  | { type: "bench"; modelId: string; quant: string };

export type WorkerToMain =
  | { type: "probe:result"; probe: DeviceProbe }
  | { type: "progress"; text: string; progress: number }
  | { type: "bench:result"; cell: BenchCell }
  | { type: "error"; message: string };

export function isMainToWorker(x: unknown): x is MainToWorker {
  if (typeof x !== "object" || x === null || !("type" in x)) return false;
  const m = x as Record<string, unknown>;
  if (m.type === "probe") return true;
  if (m.type === "bench") return typeof m.modelId === "string" && typeof m.quant === "string";
  return false;
}
