import { describe, it, expect } from "vitest";
import { BenchServer } from "../src/benchServer";
import type { WorkerToMain } from "../src/protocol";
import type { InferenceAdapter } from "../src/adapters/types";

const fakeProbe = async () => ({
  webgpu: true,
  adapterInfo: null,
  limits: null,
  features: [],
  userAgent: "ua",
  deviceMemoryGB: null,
  browser: { name: "chrome", version: "150.0" },
  anomalies: [],
});

function fakeAdapter(): InferenceAdapter {
  return {
    id: "webllm",
    capabilities: () => ({ logprobs: false, streaming: true, seed: false }),
    load: async (_m, onProgress) => { onProgress("loading", 0.5); return { loadMs: 100, cacheState: "cold" as const }; },
    generate: async () => ({ tRequestStart: 0, chunkTimestamps: [10, 20, 30], promptTokens: 512, completionTokens: 3 }),
    dispose: async () => {},
  };
}

describe("BenchServer", () => {
  it("probe message → probe:result", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapterFactory: fakeAdapter, probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "probe" });
    expect(out[0].type).toBe("probe:result");
  });

  it("bench message → progress then bench:result with metrics", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapterFactory: fakeAdapter, probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "bench", modelId: "test-model", quant: "q4f16_1" });
    const types = out.map((m) => m.type);
    expect(types).toContain("progress");
    const result = out.find((m) => m.type === "bench:result");
    expect(result).toBeDefined();
    if (result?.type === "bench:result") {
      expect(result.cell.modelId).toBe("test-model");
      expect(result.cell.gen.ttftMs).toBe(10);
      expect(result.cell.load.cacheState).toBe("cold");
      expect(result.cell.promptId).toBe("bench-512-v1");
    }
  });

  it("adapter failure → error message, no throw", async () => {
    const broken = { ...fakeAdapter(), load: async () => { throw new Error("boom"); } };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapterFactory: () => broken, probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "bench", modelId: "m", quant: "q" });
    expect(out.some((m) => m.type === "error" && m.message.includes("boom"))).toBe(true);
  });

  it("invalid message → error", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapterFactory: fakeAdapter, probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ garbage: true });
    expect(out[0].type).toBe("error");
  });
});
