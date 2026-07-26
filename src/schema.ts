import type { GenMetricsAgg } from "./metrics";

export const SCHEMA_VERSION = 2 as const;

export type StackId = "webllm" | "transformersjs" | "wllama";

export interface DeviceProbe {
  webgpu: boolean;
  adapterInfo: { vendor: string; architecture: string; device: string; description: string } | null;
  limits: Record<string, number> | null;
  features: string[]; // v2: GPUSupportedFeatures enumerate (es. "shader-f16")
  userAgent: string;
  deviceMemoryGB: number | null;
  browser: { name: string; version: string }; // v2: parsed da userAgent
  anomalies: string[]; // v2: es. rilevazione software-adapter
}

export interface LoadReport {
  loadMs: number;
  cacheState: "cold" | "warm" | "unknown";
}

export interface GenMetrics {
  ttftMs: number;
  decodeToksPerSec: number | null; // null se <2 chunk o token count assente
  totalMs: number;
  promptTokens: number | null;
  completionTokens: number | null;
}

export interface BenchCell {
  stack: StackId;
  modelId: string;
  quant: string;
  promptId: string;
  load: LoadReport;
  gen: GenMetricsAgg;
  replicates: GenMetrics[];
  anomalies: string[]; // es. "high-variance"
}

export interface RunFile {
  schemaVersion: typeof SCHEMA_VERSION;
  deviceLabel: string;
  ts: string; // ISO, momento di creazione run
  probe: DeviceProbe;
  cells: BenchCell[];
}

export function newRunFile(deviceLabel: string, probe: DeviceProbe, ts: string): RunFile {
  return { schemaVersion: SCHEMA_VERSION, deviceLabel, ts, probe, cells: [] };
}

export function addCell(run: RunFile, cell: BenchCell): RunFile {
  return { ...run, cells: [...run.cells, cell] };
}
