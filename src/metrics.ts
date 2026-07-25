import type { GenMetrics } from "./schema";

export interface GenTimeline {
  tRequestStart: number;
  chunkTimestamps: number[]; // performance.now() di ogni chunk ricevuto, in-worker
  promptTokens: number | null;
  completionTokens: number | null;
}

export function computeGenMetrics(t: GenTimeline): GenMetrics {
  if (t.chunkTimestamps.length === 0) throw new Error("empty timeline: no chunks received");
  const first = t.chunkTimestamps[0];
  const last = t.chunkTimestamps[t.chunkTimestamps.length - 1];
  const completionTokens = t.completionTokens ?? t.chunkTimestamps.length;
  const span = last - first;
  // Steady-state richiede >=2 token generati: con 0/1 token il rate non è misurabile -> null (decisione: null, non 0, per non inquinare le medie).
  const decodeToksPerSec =
    t.chunkTimestamps.length >= 2 && span > 0 && completionTokens >= 2
      ? ((completionTokens - 1) / span) * 1000
      : null;
  return {
    ttftMs: first - t.tRequestStart,
    decodeToksPerSec,
    totalMs: last - t.tRequestStart,
    promptTokens: t.promptTokens,
    completionTokens,
  };
}
