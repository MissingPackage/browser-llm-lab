import { describe, it, expect } from "vitest";
import { computeGenMetrics, aggregateReplicates, type GenTimeline } from "../src/metrics";

describe("computeGenMetrics", () => {
  it("computes ttft, steady-state decode rate, total", () => {
    // 5 chunk: primo a t=1100 (ttft 100ms), poi ogni 50ms → 4 token in 200ms = 20 tok/s
    const t: GenTimeline = {
      tRequestStart: 1000,
      chunkTimestamps: [1100, 1150, 1200, 1250, 1300],
      promptTokens: 512,
      completionTokens: 5,
    };
    const m = computeGenMetrics(t);
    expect(m.ttftMs).toBe(100);
    expect(m.decodeToksPerSec).toBeCloseTo(20, 5);
    expect(m.totalMs).toBe(300);
    expect(m.promptTokens).toBe(512);
    expect(m.completionTokens).toBe(5);
  });

  it("falls back to chunk count when completionTokens is null", () => {
    const m = computeGenMetrics({
      tRequestStart: 0,
      chunkTimestamps: [100, 200, 300],
      promptTokens: null,
      completionTokens: null,
    });
    expect(m.completionTokens).toBe(3);
    expect(m.decodeToksPerSec).toBeCloseTo(10, 5); // (3-1)/200ms
  });

  it("returns null rate when fewer than 2 chunks", () => {
    const m = computeGenMetrics({ tRequestStart: 0, chunkTimestamps: [50], promptTokens: null, completionTokens: 1 });
    expect(m.ttftMs).toBe(50);
    expect(m.decodeToksPerSec).toBeNull();
  });

  it("throws on empty timeline", () => {
    expect(() => computeGenMetrics({ tRequestStart: 0, chunkTimestamps: [], promptTokens: null, completionTokens: null })).toThrow();
  });

  it("returns null rate when completionTokens is 1 despite multiple chunks", () => {
    const m = computeGenMetrics({ tRequestStart: 0, chunkTimestamps: [100, 200], promptTokens: null, completionTokens: 1 });
    expect(m.decodeToksPerSec).toBeNull();
    expect(m.completionTokens).toBe(1);
  });

  it("returns null rate when span is zero", () => {
    const m = computeGenMetrics({ tRequestStart: 0, chunkTimestamps: [100, 100], promptTokens: null, completionTokens: 2 });
    expect(m.decodeToksPerSec).toBeNull();
  });
});

describe("aggregateReplicates", () => {
  it("computes mean/stdev over replicate GenMetrics", () => {
    const reps = [
      { ttftMs: 100, decodeToksPerSec: 40, totalMs: 6000, promptTokens: 512, completionTokens: 256 },
      { ttftMs: 110, decodeToksPerSec: 44, totalMs: 6100, promptTokens: 512, completionTokens: 256 },
      { ttftMs: 90, decodeToksPerSec: 42, totalMs: 5900, promptTokens: 512, completionTokens: 256 },
    ];
    const agg = aggregateReplicates(reps);
    expect(agg.ttftMs.mean).toBeCloseTo(100, 5);
    expect(agg.decodeToksPerSec?.mean).toBeCloseTo(42, 5);
    expect(agg.decodeToksPerSec?.samples).toEqual([40, 44, 42]);
    expect(agg.promptTokens).toBe(512);
    expect(agg.completionTokens).toBe(256);
  });

  it("returns decodeToksPerSec: null when every replicate rate is null", () => {
    const reps = [
      { ttftMs: 50, decodeToksPerSec: null, totalMs: 50, promptTokens: null, completionTokens: 1 },
      { ttftMs: 55, decodeToksPerSec: null, totalMs: 55, promptTokens: null, completionTokens: 1 },
    ];
    const agg = aggregateReplicates(reps);
    expect(agg.decodeToksPerSec).toBeNull();
  });

  it("throws on empty replicate list", () => {
    expect(() => aggregateReplicates([])).toThrow();
  });
});
