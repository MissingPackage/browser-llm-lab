import { describe, it, expect } from "vitest";
import { computeGenMetrics, type GenTimeline } from "../src/metrics";

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
});
