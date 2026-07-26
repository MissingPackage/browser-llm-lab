import { describe, it, expect } from "vitest";
import { WebLLMAdapter } from "../src/adapters/webllm";

function fakeEngine() {
  return {
    chat: {
      completions: {
        create: async function* mock() {
          yield { choices: [{ delta: { content: "a" } }], usage: null };
          yield { choices: [{ delta: { content: "b" } }], usage: null };
          yield { choices: [], usage: { prompt_tokens: 512, completion_tokens: 2 } };
        },
      },
    },
    unload: async () => {},
  };
}

describe("WebLLMAdapter", () => {
  it("load reports cold/warm from cache probe and measures loadMs", async () => {
    let t = 0;
    const a = new WebLLMAdapter({
      engineFactory: async () => fakeEngine() as never,
      hasCache: async () => false,
      now: () => (t += 500), // load: t0=500, t1=1000 → 500ms
    });
    const r = await a.load("test-model", () => {});
    expect(r.cacheState).toBe("cold");
    expect(r.loadMs).toBe(500);
  });

  it("generate builds a timeline with usage tokens", async () => {
    let t = 0;
    const a = new WebLLMAdapter({
      engineFactory: async () => fakeEngine() as never,
      hasCache: async () => true,
      now: () => (t += 100),
    });
    const r = await a.load("test-model", () => {});
    expect(r.cacheState).toBe("warm");
    const tl = await a.generate({ prompt: "hi", maxTokens: 8 });
    expect(tl.chunkTimestamps.length).toBe(2); // 2 chunk con contenuto; il chunk usage-only non conta
    expect(tl.promptTokens).toBe(512);
    expect(tl.completionTokens).toBe(2);
  });

  it("generate before load throws", async () => {
    const a = new WebLLMAdapter({ engineFactory: async () => fakeEngine() as never });
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow("not loaded");
  });

  it("sends greedy decode params and usage streaming to the engine", async () => {
    let captured: Record<string, unknown> | null = null;
    const engine = {
      chat: {
        completions: {
          create: async function* (req: Record<string, unknown>) {
            captured = req;
            yield { choices: [{ delta: { content: "x" } }], usage: null };
          },
        },
      },
      unload: async () => {},
    };
    const a = new WebLLMAdapter({
      engineFactory: async () => engine as never,
      hasCache: async () => false,
    });
    await a.load("test-model", () => {});
    const tl = await a.generate({ prompt: "hi", maxTokens: 8 });
    expect(tl.chunkTimestamps.length).toBe(1);
    expect(captured).toMatchObject({
      messages: [{ role: "user", content: "hi" }],
      temperature: 0,
      max_tokens: 8,
      stream: true,
      stream_options: { include_usage: true },
    });
  });

  it("dispose unloads the engine; generate after dispose throws not loaded", async () => {
    let unloaded = false;
    const engine = {
      chat: {
        completions: {
          create: async function* () {
            yield { choices: [{ delta: { content: "x" } }], usage: null };
          },
        },
      },
      unload: async () => {
        unloaded = true;
      },
    };
    const a = new WebLLMAdapter({ engineFactory: async () => engine as never, hasCache: async () => false });
    await a.load("test-model", () => {});
    await a.dispose();
    expect(unloaded).toBe(true);
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow("not loaded");
  });
});
