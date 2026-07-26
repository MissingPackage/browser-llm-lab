import { describe, it, expect } from "vitest";
import { WllamaAdapter, hfUrlFromModelId, chunkIsToken } from "../src/adapters/wllama";
import type { ChatCompletionChunk } from "@wllama/wllama/esm/types/oai-compat.js";

function chunk(delta: { content?: string | null }, extra?: Partial<ChatCompletionChunk>): ChatCompletionChunk {
  return {
    id: "c",
    object: "chat.completion.chunk",
    created: 0,
    model: "m",
    choices: [{ index: 0, delta, finish_reason: null, logprobs: null }],
    ...extra,
  };
}

// Engine finto: gli unit test verificano la forma della DI e il cablaggio dell'adapter.
// Che l'adapter sia cablato *correttamente* alla libreria reale lo verifica il contratto
// di conformance in browser (src/conformance/contract.ts), non questi test.
function fakeEngine(opts?: { tokens?: number; promptTokens?: number | null; completionTokens?: number | null }) {
  const tokens = opts?.tokens ?? 3;
  return {
    generate: async (_m: unknown, _max: number, onToken: () => void) => {
      for (let i = 0; i < tokens; i++) onToken();
      // `in` e non `??`: un null esplicito è un caso da testare (usage assente), non
      // un "non passato" da rimpiazzare col default.
      return {
        promptTokens: opts && "promptTokens" in opts ? (opts.promptTokens ?? null) : 512,
        completionTokens: opts && "completionTokens" in opts ? (opts.completionTokens ?? null) : tokens,
      };
    },
    dispose: async () => {},
  };
}

const MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf";

describe("hfUrlFromModelId", () => {
  it("costruisce l'URL di risoluzione HF da owner/repo/file", () => {
    expect(hfUrlFromModelId(MODEL_ID)).toBe(
      "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );
  });

  it("supporta file annidati in sottocartelle (modelli split)", () => {
    expect(hfUrlFromModelId("org/repo/sub/dir/model-Q4_K_M.gguf")).toBe(
      "https://huggingface.co/org/repo/resolve/main/sub/dir/model-Q4_K_M.gguf",
    );
  });

  it("rifiuta un modelId senza file esplicito — niente fallback silenzioso di quant", () => {
    expect(() => hfUrlFromModelId("Qwen/Qwen2.5-0.5B-Instruct-GGUF")).toThrow(/must be/);
  });

  it("rifiuta un file che non è .gguf", () => {
    expect(() => hfUrlFromModelId("org/repo/model.safetensors")).toThrow(/\.gguf/);
  });
});

describe("chunkIsToken", () => {
  it("conta un chunk che porta contenuto", () => {
    expect(chunkIsToken(chunk({ content: "ciao" }))).toBe(true);
  });

  it("NON conta il chunk finale di chiusura (delta vuoto + usage)", () => {
    // Il caso reale: wllama chiude lo stream con un chunk senza contenuto che porta usage.
    // Contarlo dava 17 timestamp per 16 token — rilevato dal contratto di conformance.
    const closing = chunk({}, {
      choices: [{ index: 0, delta: {}, finish_reason: "length", logprobs: null }],
      usage: { prompt_tokens: 46, completion_tokens: 16, total_tokens: 62 },
    });
    expect(chunkIsToken(closing)).toBe(false);
  });

  it("NON conta content null o stringa vuota", () => {
    expect(chunkIsToken(chunk({ content: null }))).toBe(false);
    expect(chunkIsToken(chunk({ content: "" }))).toBe(false);
  });

  it("NON conta un chunk senza choices", () => {
    expect(chunkIsToken(chunk({}, { choices: [] }))).toBe(false);
  });
});

describe("WllamaAdapter", () => {
  it("dichiara l'id dello stack", () => {
    expect(new WllamaAdapter().id).toBe("wllama");
  });

  it("capabilities() non dichiara seed: l'adapter genera a temperature 0 e non lo usa", () => {
    const caps = new WllamaAdapter().capabilities();
    expect(caps).toEqual({ logprobs: false, streaming: true, seed: false });
  });

  it("load() riporta cold quando il modello non è in cache", async () => {
    const a = new WllamaAdapter({
      engineFactory: async () => fakeEngine(),
      isCached: async () => false,
      now: (() => { let t = 0; return () => (t += 100); })(),
    });
    const r = await a.load(MODEL_ID, () => {});
    expect(r.cacheState).toBe("cold");
    expect(r.loadMs).toBeGreaterThan(0);
  });

  it("load() riporta warm quando il modello è già in cache", async () => {
    const a = new WllamaAdapter({
      engineFactory: async () => fakeEngine(),
      isCached: async () => true,
      now: (() => { let t = 0; return () => (t += 100); })(),
    });
    expect((await a.load(MODEL_ID, () => {})).cacheState).toBe("warm");
  });

  it("load() passa all'engine l'URL risolto, non il modelId grezzo", async () => {
    let seen = "";
    const a = new WllamaAdapter({
      engineFactory: async (url) => { seen = url; return fakeEngine(); },
      isCached: async () => false,
    });
    await a.load(MODEL_ID, () => {});
    expect(seen).toBe(hfUrlFromModelId(MODEL_ID));
  });

  it("generate() registra un timestamp per token e propaga i conteggi di usage", async () => {
    const a = new WllamaAdapter({
      engineFactory: async () => fakeEngine({ tokens: 5, promptTokens: 480, completionTokens: 5 }),
      isCached: async () => false,
      now: (() => { let t = 0; return () => (t += 10); })(),
    });
    await a.load(MODEL_ID, () => {});
    const timeline = await a.generate({ prompt: "ciao", maxTokens: 5 });
    expect(timeline.chunkTimestamps).toHaveLength(5);
    // I conteggi vengono da llama.cpp (usage), non dal numero di callback ricevute.
    expect(timeline.promptTokens).toBe(480);
    expect(timeline.completionTokens).toBe(5);
  });

  it("generate() lascia i conteggi a null quando usage è assente", async () => {
    const a = new WllamaAdapter({
      engineFactory: async () => fakeEngine({ tokens: 2, promptTokens: null, completionTokens: null }),
      isCached: async () => false,
    });
    await a.load(MODEL_ID, () => {});
    const timeline = await a.generate({ prompt: "ciao", maxTokens: 2 });
    expect(timeline.promptTokens).toBeNull();
    expect(timeline.completionTokens).toBeNull();
  });

  it("generate() prima di load() rigetta", async () => {
    const a = new WllamaAdapter({ engineFactory: async () => fakeEngine(), isCached: async () => false });
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow(/not loaded/);
  });

  it("generate() dopo dispose() rigetta", async () => {
    const a = new WllamaAdapter({ engineFactory: async () => fakeEngine(), isCached: async () => false });
    await a.load(MODEL_ID, () => {});
    await a.dispose();
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow(/not loaded/);
  });

  it("dispose() rilascia l'engine ed è sicuro chiamarlo senza load()", async () => {
    let disposed = 0;
    const a = new WllamaAdapter({
      engineFactory: async () => ({ ...fakeEngine(), dispose: async () => { disposed++; } }),
      isCached: async () => false,
    });
    await a.dispose(); // nessun engine: non deve lanciare
    await a.load(MODEL_ID, () => {});
    await a.dispose();
    expect(disposed).toBe(1);
  });
});
