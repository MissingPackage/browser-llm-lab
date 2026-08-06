// Schema di telemetria UNICO dei due path (C3a fase 4d, emendamento 6).
//
// Prima della 4d i path Qwen e GLM avevano due telemetrie non confrontabili:
// GLM cumulativa (GlmTelemetry), Qwen a medie per-token con drenaggio
// (EngineTelemetry) — e due DEFAULT diversi (GLM off, Qwen on). Questo modulo
// fissa il nucleo comune: contatori CUMULATIVI, mai medie, cosi' una finestra
// si misura per differenza (`diffCounters(dopo, prima)`) senza reset nascosti
// — lo stesso pattern dei TelemDelta di glmbench. Regole ereditate dai due
// path e ora uniche:
//   - default OFF, zero-overhead da spenta (niente performance.now nel forward);
//   - submits e dispatches si contano SEMPRE (incremento intero ~ogni 120 µs
//     di lavoro GPU: sotto il rumore, e il numero vero in ogni report vale
//     piu' della purezza — razionale it.9);
//   - gpuBusyMs e' null quando il livello 2 (timestamp-query) e' assente o
//     spento: null = "non misurato", MAI 0 (che e' una misura).
// Le estensioni per-path (GlmTelemetry, EngineTelemetry) AGGIUNGONO campi,
// non ridefiniscono questi.

export interface CoreCounters {
  /** forward completati (token per il decode; posizioni per il prefill GLM). */
  forwards: number;
  /** ms JS di encode/bind/writeBuffer, await esclusi — cumulativo, 0 da spenta. */
  encodeCpuMs: number;
  /** queue.submit effettivi — contati sempre. */
  submits: number;
  /** dispatchWorkgroups effettivi — contati sempre. */
  dispatches: number;
  /** somma delle durate dei pass GPU (liv.2) — null se assente o spento. */
  gpuBusyMs: number | null;
  /** pass strumentati che compongono gpuBusyMs. */
  gpuPasses: number;
}

export const zeroCounters = (): CoreCounters => ({
  forwards: 0, encodeCpuMs: 0, submits: 0, dispatches: 0, gpuBusyMs: null, gpuPasses: 0,
});

/**
 * Finestra `after − before`. gpuBusyMs: se uno dei due estremi non ha il
 * livello 2 la finestra non ce l'ha (null contagia — una differenza con un
 * estremo non misurato non e' una misura).
 */
export const diffCounters = (after: CoreCounters, before: CoreCounters): CoreCounters => ({
  forwards: after.forwards - before.forwards,
  encodeCpuMs: after.encodeCpuMs - before.encodeCpuMs,
  submits: after.submits - before.submits,
  dispatches: after.dispatches - before.dispatches,
  gpuBusyMs: after.gpuBusyMs === null || before.gpuBusyMs === null
    ? null
    : after.gpuBusyMs - before.gpuBusyMs,
  gpuPasses: after.gpuPasses - before.gpuPasses,
});
