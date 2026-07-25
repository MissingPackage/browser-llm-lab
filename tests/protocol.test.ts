import { describe, it, expect } from "vitest";
import { isMainToWorker } from "../src/protocol";

describe("protocol guards", () => {
  it("accepts valid messages", () => {
    expect(isMainToWorker({ type: "probe" })).toBe(true);
    expect(isMainToWorker({ type: "bench", modelId: "m", quant: "q4f16_1" })).toBe(true);
  });
  it("rejects invalid messages", () => {
    expect(isMainToWorker(null)).toBe(false);
    expect(isMainToWorker({ type: "bench" })).toBe(false); // manca modelId
    expect(isMainToWorker({ type: "nope" })).toBe(false);
  });
});
