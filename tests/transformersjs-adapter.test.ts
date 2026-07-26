import { describe, it, expect } from "vitest";
import { TransformersJsAdapter } from "../src/adapters/transformersjs";

function fakeEngine(tokenCount: number) {
  let disposed = false;
  return {
    engine: {
      generate: async (_messages: unknown, _maxTokens: number, onToken: () => void) => {
        for (let i = 0; i < tokenCount; i++) onToken();
      },
      dispose: async () => {
        disposed = true;
      },
    },
    wasDisposed: () => disposed,
  };
}

describe("TransformersJsAdapter", () => {
  it("load reports cold/warm from cache probe and measures loadMs", async () => {
    let t = 0;
    const { engine } = fakeEngine(3);
    const a = new TransformersJsAdapter({
      engineFactory: async () => engine,
      isCached: async () => false,
      now: () => (t += 500), // load: t0=500, t1=1000 → 500ms
    });
    const r = await a.load("test-model", () => {});
    expect(r.cacheState).toBe("cold");
    expect(r.loadMs).toBe(500);
  });

  it("generate builds a timeline with one timestamp per generated token", async () => {
    let t = 0;
    const { engine } = fakeEngine(3);
    const a = new TransformersJsAdapter({
      engineFactory: async () => engine,
      isCached: async () => true,
      now: () => (t += 100),
    });
    const r = await a.load("test-model", () => {});
    expect(r.cacheState).toBe("warm");
    const tl = await a.generate({ prompt: "hi", maxTokens: 8 });
    expect(tl.chunkTimestamps.length).toBe(3);
    expect(tl.promptTokens).toBeNull();
  });

  it("generate before load throws", async () => {
    const { engine } = fakeEngine(1);
    const a = new TransformersJsAdapter({ engineFactory: async () => engine });
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow("not loaded");
  });

  it("passes greedy decode params and the prompt as a chat message to the engine", async () => {
    let captured: unknown[] = [];
    const engine = {
      generate: async (messages: unknown, maxTokens: number, onToken: () => void) => {
        captured = [messages, maxTokens];
        onToken();
      },
      dispose: async () => {},
    };
    const a = new TransformersJsAdapter({ engineFactory: async () => engine, isCached: async () => false });
    await a.load("test-model", () => {});
    await a.generate({ prompt: "hi", maxTokens: 8 });
    expect(captured).toEqual([[{ role: "user", content: "hi" }], 8]);
  });

  it("dispose disposes the engine; generate after dispose throws not loaded", async () => {
    const { engine, wasDisposed } = fakeEngine(1);
    const a = new TransformersJsAdapter({ engineFactory: async () => engine, isCached: async () => false });
    await a.load("test-model", () => {});
    await a.dispose();
    expect(wasDisposed()).toBe(true);
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow("not loaded");
  });
});
