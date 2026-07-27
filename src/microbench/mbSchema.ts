import type { DeviceProbe } from "../schema";
import type { MetricAggregate } from "../metrics";

// Schema dei run micro-bench: versionato a parte rispetto ai RunFile di bench
// (results/microbench/ vs results/), stesso stile: label manuale, niente fingerprinting.
export const MICROBENCH_SCHEMA_VERSION = 1 as const;

export type MicrobenchKernelId = "gemv-q4f32" | "gemv-f32" | "gemv-f16";

export type TimingSource = "timestamp-query" | "cpu";

export interface MicrobenchCell {
  kernel: MicrobenchKernelId;
  rowsN: number; // righe di output (dimensione GEMV)
  colsK: number; // profondità della riduzione
  bytesRead: number; // byte letti dalla GPU per un dispatch (pesi+scale+input)
  repeats: number;
  timingSource: TimingSource;
  // Tempo per dispatch in ms. gpuMs presente solo con timestamp-query.
  gpuMs: MetricAggregate | null;
  cpuMs: MetricAggregate; // sempre presente (submit -> onSubmittedWorkDone)
  // Banda effettiva calcolata dalla sorgente di timing migliore disponibile.
  effectiveGBps: number;
  checksum: number; // somma degli output: sanity che il kernel abbia lavorato davvero
}

export interface SkippedCell {
  kernel: MicrobenchKernelId;
  rowsN: number;
  colsK: number;
  reason: string; // es. errore di validazione WebGPU — buchi loggati, mai silenziosi
}

export interface MicrobenchRunFile {
  schemaVersion: typeof MICROBENCH_SCHEMA_VERSION;
  kind: "microbench-matmul";
  deviceLabel: string;
  ts: string;
  probe: DeviceProbe;
  cells: MicrobenchCell[];
  skipped: SkippedCell[];
}
