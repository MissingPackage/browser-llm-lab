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
      expect(result.cell.anomalies).toEqual([
        "protocol: warm-up run discarded (policy=always, cacheState=cold)",
      ]);
    }
  });

  it("default policy warms up even on a warm cache (steady-state measurement)", async () => {
    let calls = 0;
    const warm: InferenceAdapter = {
      ...fakeAdapter(),
      load: async () => ({ loadMs: 100, cacheState: "warm" as const }),
      generate: async () => { calls++; return { tRequestStart: 0, chunkTimestamps: [10, 20, 30], promptTokens: 512, completionTokens: 3 }; },
    };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => warm, transformersjs: () => warm },
      probe: fakeProbe,
      post: (m) => out.push(m),
      replicateCount: 3,
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q" });
    expect(calls).toBe(4); // 3 misurate + 1 di riscaldamento, anche a cache calda
  });

  it('policy "never" → nessun riscaldamento, misura la prima esperienza reale', async () => {
    let calls = 0;
    const counting: InferenceAdapter = {
      ...fakeAdapter(),
      generate: async () => { calls++; return { tRequestStart: 0, chunkTimestamps: [10, 20, 30], promptTokens: 512, completionTokens: 3 }; },
    };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => counting, transformersjs: () => counting },
      probe: fakeProbe,
      post: (m) => out.push(m),
      replicateCount: 3,
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q", warmup: "never" });
    expect(calls).toBe(3);
    const result = out.find((m) => m.type === "bench:result");
    if (result?.type === "bench:result") {
      expect(result.cell.anomalies.some((a) => a.startsWith("protocol: warm-up"))).toBe(false);
    } else {
      throw new Error("expected bench:result");
    }
  });

  it("la policy del messaggio prevale su quella del server", async () => {
    let calls = 0;
    const counting: InferenceAdapter = {
      ...fakeAdapter(),
      generate: async () => { calls++; return { tRequestStart: 0, chunkTimestamps: [10, 20, 30], promptTokens: 512, completionTokens: 3 }; },
    };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => counting, transformersjs: () => counting },
      probe: fakeProbe,
      post: (m) => out.push(m),
      replicateCount: 1,
      warmup: "always",
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q", warmup: "never" });
    expect(calls).toBe(1);
  });

  it("cold cache → runs one extra warm-up generation and discards it", async () => {
    let calls = 0;
    const counting: InferenceAdapter = {
      ...fakeAdapter(),
      generate: async () => { calls++; return { tRequestStart: 0, chunkTimestamps: [10, 20, 30], promptTokens: 512, completionTokens: 3 }; },
    };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => counting, transformersjs: () => counting },
      probe: fakeProbe,
      post: (m) => out.push(m),
      replicateCount: 3,
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q" });
    expect(calls).toBe(4); // 3 misurate + 1 di riscaldamento
    const result = out.find((m) => m.type === "bench:result");
    if (result?.type === "bench:result") {
      expect(result.cell.replicates.length).toBe(3);
      expect(result.cell.anomalies).toContain(
        "protocol: warm-up run discarded (policy=always, cacheState=cold)",
      );
    } else {
      throw new Error("expected bench:result");
    }
  });

  it('policy "cold-only" + cache calda → nessun riscaldamento, nessuna nota di protocollo', async () => {
    let calls = 0;
    const warm: InferenceAdapter = {
      ...fakeAdapter(),
      load: async () => ({ loadMs: 100, cacheState: "warm" as const }),
      generate: async () => { calls++; return { tRequestStart: 0, chunkTimestamps: [10, 20, 30], promptTokens: 512, completionTokens: 3 }; },
    };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => warm, transformersjs: () => warm },
      probe: fakeProbe,
      post: (m) => out.push(m),
      replicateCount: 3,
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q", warmup: "cold-only" });
    expect(calls).toBe(3);
    const result = out.find((m) => m.type === "bench:result");
    if (result?.type === "bench:result") {
      expect(result.cell.anomalies.some((a) => a.startsWith("protocol: warm-up"))).toBe(false);
    } else {
      throw new Error("expected bench:result");
    }
  });

  it("the slow first-run penalty lands on the warm-up, not on the reported metrics", async () => {
    let call = 0;
    // Prima generazione lenta (2 chunk in 200ms), successive veloci (2 chunk in 20ms):
    // riproduce l'effetto di docket #5b. Senza lo scarto, la media sarebbe contaminata.
    const rampUp: InferenceAdapter = {
      ...fakeAdapter(),
      generate: async () => {
        call++;
        return { tRequestStart: 0, chunkTimestamps: call === 1 ? [10, 210] : [10, 30], promptTokens: 512, completionTokens: 2 };
      },
    };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => rampUp, transformersjs: () => rampUp },
      probe: fakeProbe,
      post: (m) => out.push(m),
      replicateCount: 2,
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q" });
    const result = out.find((m) => m.type === "bench:result");
    if (result?.type === "bench:result") {
      // ogni replica misurata vede la finestra veloce: nessuna varianza residua
      expect(result.cell.replicates.every((r) => r.totalMs === 30)).toBe(true);
      expect(result.cell.anomalies.some((a) => a.includes("high-variance"))).toBe(false);
    } else {
      throw new Error("expected bench:result");
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
    // warmup "never": senza, il run di riscaldamento assorbirebbe la generazione lenta
    // che è proprio la sorgente di spread che questo test verifica.
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q", warmup: "never" });
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
