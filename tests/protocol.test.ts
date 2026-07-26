import { describe, it, expect } from "vitest";
import { isMainToWorker } from "../src/protocol";

describe("protocol guards", () => {
  it("accepts valid messages", () => {
    expect(isMainToWorker({ type: "probe" })).toBe(true);
    expect(isMainToWorker({ type: "bench", stack: "webllm", modelId: "m", quant: "q4f16_1" })).toBe(true);
    expect(isMainToWorker({ type: "bench", stack: "transformersjs", modelId: "m", quant: "q4" })).toBe(true);
  });
  it("rejects invalid messages", () => {
    expect(isMainToWorker(null)).toBe(false);
    expect(isMainToWorker({ type: "bench" })).toBe(false); // manca modelId/quant/stack
    expect(isMainToWorker({ type: "bench", modelId: "m", quant: "q" })).toBe(false); // manca stack
    expect(isMainToWorker({ type: "bench", stack: "not-a-stack", modelId: "m", quant: "q" })).toBe(false);
    expect(isMainToWorker({ type: "nope" })).toBe(false);
  });
});
