import type { MetricAggregate } from "../metrics";
import type { MicrobenchKernelId } from "./mbSchema";

export function aggregateSamples(samples: number[]): MetricAggregate {
  if (samples.length === 0) throw new Error("aggregateSamples: empty sample list");
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const stdev =
    samples.length < 2
      ? 0
      : Math.sqrt(samples.reduce((a, x) => a + (x - mean) ** 2, 0) / (samples.length - 1));
  return { mean, stdev, samples };
}

// Byte letti da un dispatch GEMV y[N] = W[N,K] · x[K], per variante di kernel.
// q4: pesi packed 4-bit (K/2 byte per riga) + scale f32 una ogni 32 (K/32 * 4 per riga)
// + input x f32. L'output scritto (N*4) non è contato: misuriamo la banda di lettura,
// coerente col roofline dei doc (i pesi dominano).
export function bytesReadFor(kernel: MicrobenchKernelId, rowsN: number, colsK: number): number {
  if (rowsN <= 0 || colsK <= 0) throw new Error("bytesReadFor: dimensioni non positive");
  if (colsK % 32 !== 0) throw new Error("bytesReadFor: colsK deve essere multiplo di 32 (group size q4)");
  const xBytes = colsK * 4;
  switch (kernel) {
    case "gemv-q4f32":
      return rowsN * (colsK / 2) + rowsN * (colsK / 32) * 4 + xBytes;
    case "gemv-f32":
      return rowsN * colsK * 4 + xBytes;
    case "gemv-f16":
      return rowsN * colsK * 2 + colsK * 2; // input f16
  }
}

// Banda effettiva in GB/s (decimali) da byte letti e tempo in ms.
export function effectiveGBps(bytesRead: number, timeMs: number): number {
  if (timeMs <= 0) throw new Error("effectiveGBps: tempo non positivo");
  return bytesRead / 1e6 / timeMs; // (bytes/1e9) / (ms/1e3)
}
