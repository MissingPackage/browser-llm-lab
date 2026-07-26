import { describe, it, expect } from "vitest";
import { SCHEMA_VERSION, newRunFile, addCell, type DeviceProbe, type BenchCell } from "../src/schema";

const probe: DeviceProbe = {
  webgpu: true,
  adapterInfo: { vendor: "nvidia", architecture: "", device: "", description: "" },
  limits: { maxStorageBufferBindingSize: 2147483644 },
  features: ["shader-f16"],
  userAgent: "test-ua",
  deviceMemoryGB: 8,
  browser: { name: "chrome", version: "150.0" },
  anomalies: [],
};

const cell: BenchCell = {
  stack: "webllm",
  modelId: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  quant: "q4f16_1",
  promptId: "bench-512-v1",
  load: { loadMs: 1234, cacheState: "cold" },
  gen: { ttftMs: 100, decodeToksPerSec: 42.5, totalMs: 6000, promptTokens: 512, completionTokens: 256 },
};

describe("schema", () => {
  it("builds a versioned run file", () => {
    const run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    expect(run.schemaVersion).toBe(SCHEMA_VERSION);
    expect(run.deviceLabel).toBe("4090-linux");
    expect(run.cells).toEqual([]);
  });

  it("addCell is immutable", () => {
    const run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    const run2 = addCell(run, cell);
    expect(run.cells.length).toBe(0);
    expect(run2.cells.length).toBe(1);
    expect(run2.cells[0].gen.decodeToksPerSec).toBe(42.5);
  });
});
