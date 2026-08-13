import { describe, it, expect } from "vitest";
import {
  Q35_ATTN_MIN_CHUNK, Q35_ATTN_MAX_SPLITS,
  q35AttnSplitPlan, q35AttnPartialsFloats, q35AttnLseReduce,
} from "../src/engine/q35attnsplit";

// Unit CPU-side sul piano dello split streaming dell'attention Qwen (goal
// engine-kernel-decode). Stessa convenzione di attnsplit.ts / mlasplit.ts:
// l'aritmetica del piano e' testabile in CI SENZA GPU, e i kernel WGSL la
// importano invece di ricopiarla.

const CTX_CASES = [525, 2048, 6400, 32768, 131072];

describe("q35AttnSplitPlan — geometria dello split", () => {
  it("[f] chunkLen multiplo di 64, splits <= MAX, copertura completa di ctxMax", () => {
    for (const ctxMax of CTX_CASES) {
      const { splits, chunkLen } = q35AttnSplitPlan(ctxMax);
      expect(chunkLen % 64, `chunkLen ${chunkLen} @ctxMax ${ctxMax}`).toBe(0);
      expect(splits, `splits @ctxMax ${ctxMax}`).toBeLessThanOrEqual(Q35_ATTN_MAX_SPLITS);
      expect(splits, `splits @ctxMax ${ctxMax}`).toBeGreaterThanOrEqual(1);
      expect(splits * chunkLen, `copertura @ctxMax ${ctxMax}`).toBeGreaterThanOrEqual(ctxMax);
      expect(splits).toBe(Math.ceil(ctxMax / chunkLen));
      expect(chunkLen).toBeGreaterThanOrEqual(Q35_ATTN_MIN_CHUNK);
    }
  });

  it("[f] la granularita' minima e' MIN_CHUNK finche' MAX_SPLITS lo consente", () => {
    expect(Q35_ATTN_MIN_CHUNK % 64).toBe(0);
    const { chunkLen, splits } = q35AttnSplitPlan(Q35_ATTN_MIN_CHUNK * Q35_ATTN_MAX_SPLITS);
    expect(chunkLen).toBe(Q35_ATTN_MIN_CHUNK);
    expect(splits).toBe(Q35_ATTN_MAX_SPLITS);
    // oltre quella soglia il chunk cresce, il numero di workgroup no
    const big = q35AttnSplitPlan(Q35_ATTN_MIN_CHUNK * Q35_ATTN_MAX_SPLITS * 4);
    expect(big.splits).toBe(Q35_ATTN_MAX_SPLITS);
    expect(big.chunkLen).toBe(Q35_ATTN_MIN_CHUNK * 4);
  });

  it("il piano e' monotono e deterministico", () => {
    let prev = 0;
    for (const ctxMax of CTX_CASES) {
      const a = q35AttnSplitPlan(ctxMax);
      const b = q35AttnSplitPlan(ctxMax);
      expect(a).toEqual(b);
      expect(a.splits * a.chunkLen).toBeGreaterThanOrEqual(prev);
      prev = ctxMax;
    }
  });
});

describe("q35AttnPartialsFloats — sizing dei buffer parziali", () => {
  it("out = nHead*splits*headDim, ms = nHead*splits*2", () => {
    for (const ctxMax of CTX_CASES) {
      const { splits } = q35AttnSplitPlan(ctxMax);
      const nHead = 16, headDim = 256;
      const f = q35AttnPartialsFloats({ nHead, headDim, ctxMax });
      expect(f.out).toBe(nHead * splits * headDim);
      expect(f.ms).toBe(nHead * splits * 2);
    }
  });
});

describe("q35AttnLseReduce — equivalenza alla softmax monolitica", () => {
  // riferimento: softmax monolitica sugli score CONCATENATI, output Σ p_i·v_i
  const monolithic = (scores: number[], vs: number[][]): Float64Array => {
    const m = Math.max(...scores);
    const es = scores.map((s) => Math.exp(s - m));
    const sum = es.reduce((a, b) => a + b, 0);
    const out = new Float64Array(vs[0].length);
    es.forEach((e, i) => vs[i].forEach((v, d) => { out[d] += (e / sum) * v; }));
    return out;
  };
  // pass 1 di riferimento su uno slice: {m, s, out non normalizzato} — e'
  // esattamente cio' che scrive il kernel streaming in partMS/partOut.
  const partial = (scores: number[], vs: number[][]) => {
    const m = Math.max(...scores);
    let s = 0;
    const out = new Float64Array(vs[0].length);
    scores.forEach((sc, i) => {
      const e = Math.exp(sc - m);
      s += e;
      vs[i].forEach((v, d) => { out[d] += e * v; });
    });
    return { m, s, out };
  };

  // LCG deterministico: niente flake, stessa sequenza a ogni run
  const rnd = (seed: number) => {
    let x = seed >>> 0;
    return () => { x = (x * 1664525 + 1013904223) >>> 0; return x / 4294967296; };
  };

  it("[g] parti casuali === softmax monolitica entro 1e-12 (f64)", () => {
    for (const seed of [1, 7, 31, 1234, 99991]) {
      const r = rnd(seed);
      const dim = 8;
      const chunks = 1 + Math.floor(r() * 6);
      const allScores: number[] = [];
      const allVs: number[][] = [];
      const parts: { m: number; s: number; out: Float64Array }[] = [];
      for (let c = 0; c < chunks; c++) {
        const n = 1 + Math.floor(r() * 9);
        const sc: number[] = [];
        const vs: number[][] = [];
        for (let i = 0; i < n; i++) {
          // scale ampia: costringe la log-sum-exp a fare il suo lavoro
          sc.push((r() - 0.5) * 40);
          vs.push(Array.from({ length: dim }, () => (r() - 0.5) * 3));
        }
        allScores.push(...sc);
        allVs.push(...vs);
        parts.push(partial(sc, vs));
      }
      const got = q35AttnLseReduce(parts);
      const want = monolithic(allScores, allVs);
      expect(got.length).toBe(dim);
      for (let d = 0; d < dim; d++) {
        expect(Math.abs(got[d] - want[d]), `seed ${seed} dim ${d}`).toBeLessThan(1e-12);
      }
    }
  });

  it("[g] i chunk VUOTI del kernel (m = -3e38, s = 0, out = 0) non alterano il risultato", () => {
    // e' il caso dei chunk oltre nPast+1: escono col loop vuoto e devono
    // annullarsi nel combine, senza NaN.
    const scores = [1.5, -0.25, 3.0];
    const vs = [[1, 2], [3, -1], [0.5, 0.25]];
    const real = partial(scores, vs);
    const empty = { m: -3.0e38, s: 0, out: new Float64Array(2) };
    const got = q35AttnLseReduce([real, empty, empty]);
    const want = monolithic(scores, vs);
    for (let d = 0; d < 2; d++) {
      expect(Number.isNaN(got[d])).toBe(false);
      expect(Math.abs(got[d] - want[d])).toBeLessThan(1e-12);
    }
  });
});
