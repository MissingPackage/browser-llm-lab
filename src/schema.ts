export const SCHEMA_VERSION = 1 as const;

export interface DeviceProbe {
  webgpu: boolean;
  adapterInfo: { vendor: string; architecture: string; device: string; description: string } | null;
  limits: Record<string, number> | null;
  userAgent: string;
  deviceMemoryGB: number | null;
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
  stack: "webllm"; // union estesa in 1b
  modelId: string;
  quant: string;
  promptId: string;
  load: LoadReport;
  gen: GenMetrics;
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
