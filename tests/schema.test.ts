import { describe, it, expect } from "vitest";
import {
  SCHEMA_VERSION,
  newRunFile,
  addCell,
  normalizeDeviceLabel,
  type DeviceProbe,
  type BenchCell,
} from "../src/schema";

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
  gen: {
    ttftMs: { mean: 100, stdev: 5, samples: [95, 100, 105] },
    decodeToksPerSec: { mean: 42.5, stdev: 1.2, samples: [41, 42.5, 44] },
    totalMs: { mean: 6000, stdev: 100, samples: [5900, 6000, 6100] },
    promptTokens: 512,
    completionTokens: 256,
  },
  replicates: [
    { ttftMs: 95, decodeToksPerSec: 41, totalMs: 5900, promptTokens: 512, completionTokens: 256 },
    { ttftMs: 100, decodeToksPerSec: 42.5, totalMs: 6000, promptTokens: 512, completionTokens: 256 },
    { ttftMs: 105, decodeToksPerSec: 44, totalMs: 6100, promptTokens: 512, completionTokens: 256 },
  ],
  protocol: { warmupPolicy: "always", warmupApplied: true, replicateCount: 3 },
  anomalies: [],
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
    expect(run2.cells[0].gen.decodeToksPerSec?.mean).toBe(42.5);
  });
});

describe("normalizeDeviceLabel", () => {
  it("keeps a real label as typed", () => {
    expect(normalizeDeviceLabel("s22-ultra")).toBe("s22-ultra");
  });

  it("trims surrounding whitespace", () => {
    expect(normalizeDeviceLabel("  m4-pro \n")).toBe("m4-pro");
  });

  it("falls back to unknown-device on empty or whitespace-only input", () => {
    // Meglio un'etichetta palesemente assente che una sbagliata: un file "unknown-device"
    // si nota, uno che si dichiara "4090-linux" girando su un telefono no.
    expect(normalizeDeviceLabel("")).toBe("unknown-device");
    expect(normalizeDeviceLabel("   ")).toBe("unknown-device");
  });
});
