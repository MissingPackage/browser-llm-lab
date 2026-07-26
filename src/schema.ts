import type { GenMetricsAgg } from "./metrics";
import type { QualityScore } from "./quality";

export const SCHEMA_VERSION = 3 as const;

export type StackId = "webllm" | "transformersjs" | "wllama";

// Politica di riscaldamento fra celle (ruling PI, docket #5b) — vive in schema.ts perché è
// parte della forma dei dati (BenchCell.protocol), non del comportamento del server.
export type WarmupPolicy = "always" | "cold-only" | "never";

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

// v3 (docket #7): registra la politica di misura applicata alla cella, esplicitamente —
// prima era una nota testuale dentro `anomalies`, che significa "cosa è andato storto".
// Un warm-up applicato è il protocollo che funziona come previsto, non un'anomalia.
export interface BenchProtocol {
  warmupPolicy: WarmupPolicy;
  warmupApplied: boolean;
  replicateCount: number;
}

export interface BenchCell {
  stack: StackId;
  modelId: string;
  quant: string;
  promptId: string;
  load: LoadReport;
  gen: GenMetricsAgg;
  replicates: GenMetrics[];
  protocol: BenchProtocol; // v3, docket #7
  // v3: non ancora popolato dalla pipeline di bench reale (vedi docket #10) — il modulo
  // qualità è pronto e testato, il collegamento a benchServer.ts è un passo separato.
  qualityScore?: QualityScore;
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
