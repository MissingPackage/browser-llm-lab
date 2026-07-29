import type { GenMetrics } from "./schema";

export interface GenTimeline {
  tRequestStart: number;
  chunkTimestamps: number[]; // performance.now() di ogni chunk ricevuto, in-worker
  promptTokens: number | null;
  completionTokens: number | null;
}

export function computeGenMetrics(t: GenTimeline): GenMetrics {
  // I token di usage nel messaggio discriminano tre cause diverse di timeline vuota
  // (visto sul campo, S22 2026-07-29): completionTokens=0 -> il modello ha emesso EOS
  // subito; >0 -> token generati ma delta senza contenuto; n/d -> stream chiuso senza
  // nemmeno il chunk di usage (abort lato engine senza eccezione).
  if (t.chunkTimestamps.length === 0) {
    throw new Error(
      `empty timeline: no chunks received (promptTokens=${t.promptTokens ?? "n/d"}, ` +
        `completionTokens=${t.completionTokens ?? "n/d"})`,
    );
  }
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

export interface MetricAggregate {
  mean: number;
  stdev: number;
  samples: number[];
}

export interface GenMetricsAgg {
  ttftMs: MetricAggregate;
  decodeToksPerSec: MetricAggregate | null;
  totalMs: MetricAggregate;
  promptTokens: number | null;
  completionTokens: number | null;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function stdev(xs: number[], m: number): number {
  if (xs.length < 2) return 0;
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
}

function aggregate(xs: number[]): MetricAggregate {
  const m = mean(xs);
  return { mean: m, stdev: stdev(xs, m), samples: xs };
}

export function aggregateReplicates(reps: GenMetrics[]): GenMetricsAgg {
  if (reps.length === 0) throw new Error("aggregateReplicates: empty replicate list");
  const rates = reps.map((r) => r.decodeToksPerSec).filter((r): r is number => r !== null);
  return {
    ttftMs: aggregate(reps.map((r) => r.ttftMs)),
    decodeToksPerSec: rates.length > 0 ? aggregate(rates) : null,
    totalMs: aggregate(reps.map((r) => r.totalMs)),
    promptTokens: reps[0].promptTokens,
    completionTokens: reps[0].completionTokens,
  };
}
