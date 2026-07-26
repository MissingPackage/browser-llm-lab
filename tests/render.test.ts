import { describe, it, expect } from "vitest";
import { renderResultsTable } from "../src/render";
import { newRunFile, addCell } from "../src/schema";

const probe = {
  webgpu: true,
  adapterInfo: null,
  limits: null,
  features: [],
  userAgent: "ua",
  deviceMemoryGB: null,
  browser: { name: "chrome", version: "150.0" },
  anomalies: [],
};

function cellWith(overrides: { modelId?: string; anomalies?: string[] } = {}) {
  return {
    stack: "webllm" as const,
    modelId: overrides.modelId ?? "M",
    quant: "q4f16_1",
    promptId: "bench-512-v1",
    load: { loadMs: 1500, cacheState: "cold" as const },
    gen: {
      ttftMs: { mean: 123.4, stdev: 3, samples: [120, 123.4, 127] },
      decodeToksPerSec: { mean: 41.7, stdev: 2.1, samples: [40, 41.7, 43.4] },
      totalMs: { mean: 6000, stdev: 50, samples: [5950, 6000, 6050] },
      promptTokens: 512,
      completionTokens: 256,
    },
    replicates: [],
    protocol: { warmupPolicy: "always" as const, warmupApplied: true, replicateCount: 3 },
    anomalies: overrides.anomalies ?? [],
  };
}

describe("renderResultsTable", () => {
  it("renders one row per cell with mean±stdev tok/s", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, cellWith());
    const html = renderResultsTable(run);
    expect(html).toContain("M");
    expect(html).toContain("41.7");
    expect(html).toContain("±2.1");
    expect(html).toContain("cold");
  });

  it("renders anomaly badges when present, escaped", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, cellWith({ anomalies: ["high-variance: stdev/mean=0.20 > 0.15"] }));
    const html = renderResultsTable(run);
    expect(html).toContain('<span class="anomaly">high-variance: stdev/mean=0.20 &gt; 0.15</span>');
    expect(html).not.toContain("0.20 > 0.15"); // il '>' letterale nell'anomalia deve uscire escaped
  });

  it("renders empty state", () => {
    const html = renderResultsTable(newRunFile("x", probe, "2026-07-25T12:00:00Z"));
    expect(html).toContain("Nessun risultato");
  });

  it("escapes html-unsafe characters in modelId", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, cellWith({ modelId: "<img src=x onerror=alert(1)>" }));
    const html = renderResultsTable(run);
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("&lt;img src=x onerror=alert(1)&gt;");
  });
});
