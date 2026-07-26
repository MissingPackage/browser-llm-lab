import { describe, it, expect } from "vitest";
import { computePerplexity, evaluateExactMatch, selectQualityMethod } from "../src/quality";
import { QUALITY_PROMPTS } from "../src/qualityPrompts";

describe("selectQualityMethod", () => {
  it("picks perplexity when the adapter exposes logprobs", () => {
    expect(selectQualityMethod({ logprobs: true })).toBe("perplexity");
  });

  it("falls back to exact-match when logprobs are unavailable", () => {
    expect(selectQualityMethod({ logprobs: false })).toBe("exact-match");
  });
});

describe("computePerplexity", () => {
  it("returns 1 when every token had logprob 0 (probability 1)", () => {
    const score = computePerplexity([0, 0, 0]);
    expect(score).toEqual({ kind: "perplexity", value: 1 });
  });

  it("computes exp(-mean(logprobs)) for a mixed sequence", () => {
    const logprobs = [-0.1, -0.2, -0.3];
    const mean = logprobs.reduce((a, b) => a + b, 0) / logprobs.length;
    const score = computePerplexity(logprobs);
    expect(score.kind).toBe("perplexity");
    if (score.kind === "perplexity") expect(score.value).toBeCloseTo(Math.exp(-mean), 10);
  });

  it("throws on an empty logprobs array", () => {
    expect(() => computePerplexity([])).toThrow();
  });
});

describe("evaluateExactMatch", () => {
  it("scores 0/12 when no responses are provided", () => {
    const score = evaluateExactMatch([]);
    expect(score).toEqual({ kind: "exact-match", value: 0, total: 12 });
  });

  it("scores 12/12 when every prompt gets its canonical correct answer", () => {
    const canonical: Record<string, string> = {
      "arith-1": "19",
      "arith-2": "54",
      "arith-3": "63",
      "factual-1": "Paris",
      "factual-2": "7",
      "factual-3": "green",
      "format-1": "banana, banana, banana",
      "format-2": "1 2 3 4 5",
      "format-3": "yes",
      "json-1": '{"answer": 4}',
      "json-2": '{"ok": true}',
      "json-3": '["a", "b", "c"]',
    };
    const responses = QUALITY_PROMPTS.map((p) => ({ promptId: p.id, response: canonical[p.id] }));
    expect(responses.length).toBe(12);
    const score = evaluateExactMatch(responses);
    expect(score).toEqual({ kind: "exact-match", value: 12, total: 12 });
  });

  it("counts only the prompts that got a matching response, missing ones score as wrong", () => {
    const score = evaluateExactMatch([
      { promptId: "arith-1", response: "19" },
      { promptId: "arith-2", response: "wrong" },
      // le altre 10 non hanno risposta: contano come sbagliate, non lanciano
    ]);
    expect(score).toEqual({ kind: "exact-match", value: 1, total: 12 });
  });

  it("is tolerant to a trailing period and surrounding whitespace", () => {
    const score = evaluateExactMatch([{ promptId: "arith-1", response: "  19.\n" }]);
    expect(score.value).toBe(1);
  });

  it("rejects malformed JSON for the json-category prompts", () => {
    const score = evaluateExactMatch([{ promptId: "json-1", response: "not json" }]);
    expect(score.value).toBe(0);
  });
});

describe("QUALITY_PROMPTS", () => {
  it("has exactly 12 deterministic prompts with unique ids", () => {
    expect(QUALITY_PROMPTS.length).toBe(12);
    expect(new Set(QUALITY_PROMPTS.map((p) => p.id)).size).toBe(12);
  });

  it("covers all four categories", () => {
    const categories = new Set(QUALITY_PROMPTS.map((p) => p.category));
    expect(categories).toEqual(new Set(["arithmetic", "factual", "format", "json"]));
  });
});
