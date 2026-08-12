import { describe, it, expect } from "vitest";
import { aggregateSamples, bytesReadFor, effectiveGBps } from "../src/microbench/stats";
import { kernelSource, WORKGROUP_SIZE } from "../src/microbench/kernels";
import { seededData } from "../src/microbench/runner";
import { MICROBENCH_SCHEMA_VERSION } from "../src/microbench/mbSchema";

describe("microbench stats", () => {
  it("aggregateSamples: media e stdev campionaria", () => {
    const a = aggregateSamples([2, 4, 6]);
    expect(a.mean).toBe(4);
    expect(a.stdev).toBeCloseTo(2, 10);
    expect(a.samples).toEqual([2, 4, 6]);
  });

  it("aggregateSamples: un solo campione -> stdev 0", () => {
    expect(aggregateSamples([5]).stdev).toBe(0);
  });

  it("aggregateSamples: lista vuota -> throw", () => {
    expect(() => aggregateSamples([])).toThrow();
  });

  it("bytesReadFor q4: pesi packed + scale + input", () => {
    // 896x896: pesi 896*448, scale 896*28*4, x 896*4
    expect(bytesReadFor("gemv-q4f32", 896, 896)).toBe(896 * 448 + 896 * 28 * 4 + 896 * 4);
  });

  it("bytesReadFor f32/f16", () => {
    expect(bytesReadFor("gemv-f32", 100, 32)).toBe(100 * 32 * 4 + 32 * 4);
    expect(bytesReadFor("gemv-f16", 100, 32)).toBe(100 * 32 * 2 + 32 * 2);
  });

  it("bytesReadFor: K non multiplo di 32 -> throw", () => {
    expect(() => bytesReadFor("gemv-q4f32", 10, 33)).toThrow();
  });

  it("effectiveGBps: 1 GB in 1 s = 1 GB/s", () => {
    expect(effectiveGBps(1e9, 1000)).toBeCloseTo(1, 10);
    expect(() => effectiveGBps(100, 0)).toThrow();
  });
});

describe("microbench kernels (sorgente WGSL)", () => {
  it("q4: schema di dequant osservato nel dump (nibble, offset -7, scale/32)", () => {
    const src = kernelSource("gemv-q4f32", 896);
    expect(src).toContain("& 15u");
    expect(src).toContain("- 7.0");
    expect(src).toContain("(w >> 2u)"); // 4 u32 = 32 pesi per scale
    expect(src).toContain(`@workgroup_size(${WORKGROUP_SIZE}, 1, 1)`);
  });

  it("f16: abilita l'estensione; f32 no", () => {
    expect(kernelSource("gemv-f16", 128)).toContain("enable f16;");
    expect(kernelSource("gemv-f32", 128)).not.toContain("enable f16;");
  });

  it("K non multiplo di 32 -> throw", () => {
    expect(() => kernelSource("gemv-q4f32", 100)).toThrow();
  });
});

describe("microbench runner (parti pure)", () => {
  it("seededData: deterministico e in [-1, 1]", () => {
    const a = seededData(64, 42);
    const b = seededData(64, 42);
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(Math.max(...a)).toBeLessThanOrEqual(1);
    expect(Math.min(...a)).toBeGreaterThanOrEqual(-1);
    expect(Array.from(seededData(64, 43))).not.toEqual(Array.from(a));
  });

  it("schema version = 2 (kind microbench-kernel-decode, fase 0 engine-kernel-decode)", () => {
    expect(MICROBENCH_SCHEMA_VERSION).toBe(2);
  });
});
