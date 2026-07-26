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

const fakeAdapters = () => ({ webllm: fakeAdapter, transformersjs: fakeAdapter });

describe("BenchServer", () => {
  it("probe message → probe:result", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapters: fakeAdapters(), probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "probe" });
    expect(out[0].type).toBe("probe:result");
  });

  it("bench message → progress then bench:result with aggregated metrics over replicates", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapters: fakeAdapters(), probe: fakeProbe, post: (m) => out.push(m), replicateCount: 2 });
    await s.handle({ type: "bench", stack: "webllm", modelId: "test-model", quant: "q4f16_1" });
    const types = out.map((m) => m.type);
    expect(types.filter((t) => t === "progress").length).toBeGreaterThanOrEqual(2);
    const result = out.find((m) => m.type === "bench:result");
    expect(result).toBeDefined();
    if (result?.type === "bench:result") {
      expect(result.cell.modelId).toBe("test-model");
      expect(result.cell.stack).toBe("webllm");
      expect(result.cell.replicates.length).toBe(2);
      expect(result.cell.gen.ttftMs.mean).toBe(10);
      expect(result.cell.load.cacheState).toBe("cold");
      expect(result.cell.promptId).toBe("bench-512-v1");
      expect(result.cell.anomalies).toEqual([]);
    }
  });

  it("dispatches to the adapter registered for the requested stack", async () => {
    const out: WorkerToMain[] = [];
    const calls: string[] = [];
    const adapters = {
      webllm: () => { calls.push("webllm"); return fakeAdapter(); },
      transformersjs: () => { calls.push("transformersjs"); return { ...fakeAdapter(), id: "transformersjs" as const }; },
    };
    const s = new BenchServer({ adapters, probe: fakeProbe, post: (m) => out.push(m), replicateCount: 1 });
    await s.handle({ type: "bench", stack: "transformersjs", modelId: "m", quant: "q4" });
    expect(calls).toEqual(["transformersjs"]);
    const result = out.find((m) => m.type === "bench:result");
    if (result?.type === "bench:result") expect(result.cell.stack).toBe("transformersjs");
  });

  it("flags high-variance when decode rate spreads across replicates beyond threshold", async () => {
    let call = 0;
    const varyingAdapter: InferenceAdapter = {
      ...fakeAdapter(),
      generate: async () => {
        call++;
        const chunks = call === 1 ? [0, 100] : [0, 25];
        return { tRequestStart: 0, chunkTimestamps: chunks, promptTokens: 512, completionTokens: 5 };
      },
    };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => varyingAdapter, transformersjs: () => varyingAdapter },
      probe: fakeProbe,
      post: (m) => out.push(m),
      replicateCount: 2,
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q" });
    const result = out.find((m) => m.type === "bench:result");
    if (result?.type === "bench:result") {
      expect(result.cell.anomalies.some((a) => a.includes("high-variance"))).toBe(true);
    } else {
      throw new Error("expected bench:result");
    }
  });

  it("adapter failure → error message, no throw", async () => {
    const broken = { ...fakeAdapter(), load: async () => { throw new Error("boom"); } };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => broken, transformersjs: () => broken },
      probe: fakeProbe,
      post: (m) => out.push(m),
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q" });
    expect(out.some((m) => m.type === "error" && m.message.includes("boom"))).toBe(true);
  });

  it("invalid message → error", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapters: fakeAdapters(), probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ garbage: true });
    expect(out[0].type).toBe("error");
  });

  it("quant mismatch for a stack pinned to a fixed quant → error, no bench:result", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapters: fakeAdapters(), probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "bench", stack: "transformersjs", modelId: "m", quant: "q4f16" });
    expect(out.some((m) => m.type === "error" && m.message.includes("q4f16") && m.message.includes("q4"))).toBe(true);
    expect(out.some((m) => m.type === "bench:result")).toBe(false);
  });

  it("quant matching the pinned value for transformersjs → runs normally", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapters: fakeAdapters(), probe: fakeProbe, post: (m) => out.push(m), replicateCount: 1 });
    await s.handle({ type: "bench", stack: "transformersjs", modelId: "m", quant: "q4" });
    const result = out.find((m) => m.type === "bench:result");
    expect(result).toBeDefined();
    if (result?.type === "bench:result") expect(result.cell.quant).toBe("q4");
  });
});
