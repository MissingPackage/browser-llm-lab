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

describe("renderResultsTable", () => {
  it("renders one row per cell with key metrics", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, {
      stack: "webllm", modelId: "M", quant: "q4f16_1", promptId: "bench-512-v1",
      load: { loadMs: 1500, cacheState: "cold" },
      gen: { ttftMs: 123.4, decodeToksPerSec: 41.7, totalMs: 6000, promptTokens: 512, completionTokens: 256 },
    });
    const html = renderResultsTable(run);
    expect(html).toContain("M");
    expect(html).toContain("41.7");
    expect(html).toContain("cold");
  });

  it("renders empty state", () => {
    const html = renderResultsTable(newRunFile("x", probe, "2026-07-25T12:00:00Z"));
    expect(html).toContain("Nessun risultato");
  });

  it("escapes html-unsafe characters in modelId", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, {
      stack: "webllm",
      modelId: "<img src=x onerror=alert(1)>",
      quant: "q4f16_1",
      promptId: "bench-512-v1",
      load: { loadMs: 1, cacheState: "cold" },
      gen: { ttftMs: 1, decodeToksPerSec: 1, totalMs: 1, promptTokens: 1, completionTokens: 1 },
    });
    const html = renderResultsTable(run);
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("&lt;img src=x onerror=alert(1)&gt;");
  });
});
