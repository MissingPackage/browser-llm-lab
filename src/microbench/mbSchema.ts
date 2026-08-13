import type { DeviceProbe } from "../schema";
import type { MetricAggregate } from "../metrics";

// Schema dei run micro-bench: versionato a parte rispetto ai RunFile di bench
// (results/microbench/ vs results/), stesso stile: label manuale, niente fingerprinting.
// v2 (goal engine-kernel-decode, fase 0): si aggiunge il kind
// "microbench-kernel-decode" — celle per (kernel, variante, forma) con p50/IQR e
// blocco sonda delle feature. Il kind "microbench-matmul" resta invariato: i
// JSON gia' committati a v1 restano leggibili come dati.
export const MICROBENCH_SCHEMA_VERSION = 2 as const;

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

// ===========================================================================
// kind "microbench-kernel-decode" (goal engine-kernel-decode, fase 0)
// ===========================================================================

/** Dispersione di un campione: p50 e' la statistica di confronto pre-registrata. */
export interface SampleStats {
  p50: number;
  min: number;
  max: number;
  iqr: number;
  n: number;
  samples: number[];
}

/** Esito della sonda di una feature: dichiarata presente/assente per MISURA. */
export interface FeatureProbeResult {
  /** esposta dall'adapter (o dalla lista wgslLanguageFeatures per le feature di linguaggio) */
  exposed: boolean;
  /** concessa sul device richiedendola (null = non richiesta perche' non esposta) */
  granted: boolean | null;
  /** uno shader che la USA compila? (null = non provato) */
  compiles: boolean | null;
  /** ...e produce il risultato atteso? (null = non eseguito) */
  correct: boolean | null;
  note: string;
}

export interface KernelDecodeProbe {
  adapterFeatures: string[];
  deviceFeatures: string[];
  wgslLanguageFeatures: string[];
  adapterInfo: Record<string, unknown>;
  grantedLimits: Record<string, number>;
  subgroupSizeObserved: number | null;
  features: Record<string, FeatureProbeResult>;
}

export interface KernelDecodeCell {
  kernel: "attn-decode" | "gemv-q4_0";
  variant: string;
  /** forma dichiarata: la cella non si legge senza il suo contesto */
  shape: Record<string, number>;
  /** descrizione a parole di cosa fa la variante — nessuna cella senza contesto */
  context: string;
  dispatchesPerOp: number;
  opsPerSample: number;
  warmupDiscarded: number;
  timingSource: TimingSource;
  /** ms per OP (un token di attenzione / un GEMV), dalla sorgente di timing dichiarata */
  msPerOp: SampleStats;
  /** stessa misura via CPU fence: sempre presente, per confronto */
  cpuMsPerOp: SampleStats;
  /** solo attenzione: la stessa cella misurata in batch da 16 (riportata, non usata dal grade) */
  msPerOpBatched: SampleStats | null;
  bytesUnique: number;
  bytesEmitted: number;
  effectiveGBps: number;
  emittedGBps: number;
  weightsPerSecond: number | null;
  checksum: number;
  checksumAbs: number;
  /** scarto relativo del checksum rispetto alla cella "base" della stessa forma */
  checksumRelDiff: number | null;
  notes: string;
}

export interface KernelDecodeSkipped {
  kernel: string;
  variant: string;
  shape: Record<string, number>;
  reason: string;
}

// ===========================================================================
// kind "microbench-ttft-riga1" (goal engine-ttft, riga 1 — sonde del prefill)
//
// Cella distinta da KernelDecodeCell perche' porta tre grandezze che la fase 0
// non aveva e che sono IL done-when di questa riga: M (righe del chunk),
// i byte di PESO emessi per token prefillato, e il workgroup storage LETTO dal
// testo WGSL generato. Il kind precedente resta invariato: i JSON gia'
// committati restano leggibili come dati.
// ===========================================================================

export interface TtftCell {
  kernel: "gemm-dense-f32" | "gemm-q4_0-multirow" | "attn-prefill-chunk";
  variant: string;
  shape: Record<string, number>;
  /** righe del chunk di prefill trattate insieme (1 = forma sequenziale) */
  M: number;
  context: string;
  dispatchesPerOp: number;
  opsPerSample: number;
  warmupDiscarded: number;
  timingSource: TimingSource;
  msPerOp: SampleStats;
  cpuMsPerOp: SampleStats;
  bytesUnique: number;
  bytesEmitted: number;
  effectiveGBps: number;
  emittedGBps: number;
  weightsPerSecond: number | null;
  /** byte di PESO (qs + scale) letti UNA volta dal tensore */
  weightBytesUnique: number | null;
  /** byte di PESO che la forma EMETTE verso la memoria in un'esecuzione */
  weightBytesEmitted: number | null;
  /** la metrica del done-when: weightBytesEmitted / M */
  weightBytesPerToken: number | null;
  /** M / msPerOp.p50, in token al secondo */
  tokensPerSecond: number | null;
  /** solo GEMM densa: 2·M·N·K / msPerOp */
  tflops: number | null;
  /** LETTO dal testo WGSL generato (var<workgroup> con allineamento di spec) */
  workgroupStorageBytes: number;
  checksum: number;
  checksumAbs: number;
  checksumRelDiff: number | null;
  hostState: string;
  notes: string;
}

/** Una tacca della spazzata del tetto negoziabile (done-when d, predizione P6). */
export interface TtftLimitSweepPoint {
  requestedWorkgroupStorage: number;
  grantedWorkgroupStorage: number | null;
  deviceCreated: boolean;
  /** la pipeline della forma in esame si crea a questo tetto? */
  pipelineCreated: boolean;
  error: string | null;
  msPerOpP50: number | null;
  samples: number[];
  note: string;
}

export interface TtftLimitSweep {
  variant: string;
  shape: Record<string, number>;
  M: number;
  workgroupStorageBytes: number;
  points: TtftLimitSweepPoint[];
}

export interface TtftRunFile {
  schemaVersion: typeof MICROBENCH_SCHEMA_VERSION;
  kind: "microbench-ttft-riga1";
  goal: string;
  prereg: string;
  deviceLabel: string;
  hostState: { declared: string };
  ts: string;
  probe: DeviceProbe;
  kdProbe: KernelDecodeProbe;
  /** i limiti dell'ADAPTER, accanto ai numeri (done-when d, testuale) */
  adapterLimits: Record<string, number>;
  cells: TtftCell[];
  skipped: KernelDecodeSkipped[];
  /** riempita dal driver: richiede device distinti con requiredLimits espliciti */
  limitSweep: TtftLimitSweep[] | null;
}

export interface KernelDecodeRunFile {
  schemaVersion: typeof MICROBENCH_SCHEMA_VERSION;
  kind: "microbench-kernel-decode";
  goal: string;
  prereg: string;
  deviceLabel: string;
  hostState: { declared: string };
  ts: string;
  probe: DeviceProbe;
  kdProbe: KernelDecodeProbe;
  cells: KernelDecodeCell[];
  skipped: KernelDecodeSkipped[];
}
