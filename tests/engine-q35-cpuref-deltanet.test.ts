import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { deltaNetStepCore, softplusGgml } from "../src/engine/q35cpuref";

// Test del cpuref-f64 DeltaNet (fase 3 q1, spec §4) su DUE piani:
// 1. proprietà MATEMATICHE del core in aritmetica esatta (la delta rule ha
//    identità algebriche verificabili a mano: fit con β=1, wipe del decay,
//    lettura pura con β=0) — se una regressione cambia la semantica, salta
//    qui, non nel confronto col kernel;
// 2. campione sintetico DETERMINISTICO pinnato su fixture (il campione che
//    il ktest WGSL di fase 3 bersaglia; guardia anti-drift del riferimento).

const HD = 8;

function unit(vals: number[]): Float64Array {
  const v = Float64Array.from(vals);
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s);
  return v.map((x) => x / n) as Float64Array;
}

describe("deltaNetStepCore: proprietà algebriche", () => {
  it("β=1, g=0, S=0: la coppia (k,v) viene FITTATA — o(q=k) = v/√hd", () => {
    const S = new Float64Array(HD * HD);
    const k = unit([1, 2, -1, 0.5, 0, 3, -2, 1]);
    const v = Float64Array.from({ length: HD }, (_, j) => j - 3.5);
    const scaleQ = 1 / Math.sqrt(HD);
    const q = k.map((x) => x * scaleQ) as Float64Array;
    const o = deltaNetStepCore(S, q, k, v, 0, 1, HD);
    // S = k⊗v (S era 0) ⇒ o[j] = (kᵀk)·v[j]/√hd = v[j]/√hd (k unitario)
    for (let j = 0; j < HD; j++) expect(o[j]).toBeCloseTo(v[j] * scaleQ, 12);
  });

  it("β=1: una seconda coppia su k ORTOGONALE non disturba la prima", () => {
    const S = new Float64Array(HD * HD);
    const k1 = unit([1, 0, 0, 0, 0, 0, 0, 0]);
    const k2 = unit([0, 1, 0, 0, 0, 0, 0, 0]);
    const v1 = Float64Array.from({ length: HD }, (_, j) => j + 1);
    const v2 = Float64Array.from({ length: HD }, (_, j) => -j);
    const scaleQ = 1 / Math.sqrt(HD);
    deltaNetStepCore(S, k1.map((x) => x * scaleQ) as Float64Array, k1, v1, 0, 1, HD);
    const o = deltaNetStepCore(S, k1.map((x) => x * scaleQ) as Float64Array, k2, v2, 0, 1, HD);
    // lettura su q=k1 dopo update su k2⊥k1: ancora v1/√hd
    for (let j = 0; j < HD; j++) expect(o[j]).toBeCloseTo(v1[j] * scaleQ, 12);
  });

  it("β=0: lo stato non cambia, la lettura è S·q puro", () => {
    const S = new Float64Array(HD * HD);
    const k1 = unit([1, 1, 1, 1, 1, 1, 1, 1]);
    const v1 = Float64Array.from({ length: HD }, (_, j) => 2 * j);
    const scaleQ = 1 / Math.sqrt(HD);
    deltaNetStepCore(S, k1.map((x) => x * scaleQ) as Float64Array, k1, v1, 0, 1, HD);
    const before = S.slice();
    const k2 = unit([1, -1, 1, -1, 1, -1, 1, -1]);
    deltaNetStepCore(S, k2.map((x) => x * scaleQ) as Float64Array, k2, Float64Array.from({ length: HD }, () => 99), 0, 0, HD);
    expect(Array.from(S)).toEqual(Array.from(before));
  });

  it("decay g→−∞ azzera la memoria: resta solo la coppia nuova", () => {
    const S = new Float64Array(HD * HD);
    const k1 = unit([3, 1, 4, 1, 5, 9, 2, 6]);
    const v1 = Float64Array.from({ length: HD }, () => 7);
    const k2 = unit([2, 7, 1, 8, 2, 8, 1, 8]);
    const v2 = Float64Array.from({ length: HD }, (_, j) => j);
    const scaleQ = 1 / Math.sqrt(HD);
    deltaNetStepCore(S, k1.map((x) => x * scaleQ) as Float64Array, k1, v1, 0, 1, HD);
    const o = deltaNetStepCore(S, k2.map((x) => x * scaleQ) as Float64Array, k2, v2, -1e4, 1, HD);
    for (let j = 0; j < HD; j++) expect(o[j]).toBeCloseTo(v2[j] * scaleQ, 12);
  });

  it("ordine decay-prima-della-lettura: g agisce sulla lettura dello STESSO step", () => {
    // se il decay fosse applicato DOPO la lettura, sk vedrebbe S non decaduto
    // e la delta rule produrrebbe un residuo diverso: il test distingue i due
    // ordinamenti con g finito.
    const S = new Float64Array(HD * HD);
    const k = unit([1, 0, 0, 0, 0, 0, 0, 0]);
    const v1 = Float64Array.from({ length: HD }, () => 1);
    const scaleQ = 1 / Math.sqrt(HD);
    deltaNetStepCore(S, k.map((x) => x * scaleQ) as Float64Array, k, v1, 0, 1, HD);
    const g = -0.7;
    const v2 = Float64Array.from({ length: HD }, () => 0);
    // step 2: v=0, β=1 ⇒ d = −e^g·v1 (lettura su S decaduto) ⇒ S finale = 0
    const o = deltaNetStepCore(S, k.map((x) => x * scaleQ) as Float64Array, k, v2, g, 1, HD);
    for (let j = 0; j < HD; j++) expect(o[j]).toBeCloseTo(0, 12);
    for (let i = 0; i < HD * HD; i++) expect(Math.abs(S[i])).toBeLessThan(1e-12);
  });

  it("softplus ggml: continuità al gomito x=20 e identità oltre", () => {
    expect(softplusGgml(21)).toBe(21);
    expect(softplusGgml(0)).toBeCloseTo(Math.log(2), 15);
    expect(softplusGgml(20) - 20).toBeLessThan(1e-8);
  });
});

// --- campione sintetico deterministico (il bersaglio del ktest WGSL) ---

import { runSample, SAMPLE_DIMS, SAMPLE_T } from "./helpers/q35-deltanet-sample";

const FIXTURE = join(process.cwd(), "tests/fixtures/q35-deltanet-sample.json");

describe("catena DeltaNet: campione sintetico pinnato", () => {
  it("T=12 deterministico: output stabile vs fixture (guardia anti-drift)", () => {
    const { outputs, ref } = runSample();
    // sanity: niente NaN/Inf, stato non degenere
    for (const o of outputs) for (const v of o) expect(Number.isFinite(v)).toBe(true);
    let sAbs = 0;
    for (const v of ref.S) sAbs += Math.abs(v);
    expect(sAbs).toBeGreaterThan(0);

    if (!existsSync(FIXTURE)) {
      // il fixture non si auto-pinna in silenzio: si genera esplicitamente
      throw new Error("fixture q35-deltanet-sample.json assente: generarla con scripts/q35-deltanet-fixture-gen.mjs (vite-node)");
    }
    const fx = JSON.parse(readFileSync(FIXTURE, "utf8")) as { outputs: number[][] };
    expect(fx.outputs.length).toBe(SAMPLE_T);
    for (let t = 0; t < SAMPLE_T; t++) {
      for (let i = 0; i < SAMPLE_DIMS.d; i++) {
        expect(Math.abs(outputs[t][i] - fx.outputs[t][i]), `t=${t} i=${i}`).toBeLessThan(1e-12);
      }
    }
  });
});
