import { describe, it, expect } from "vitest";
import {
  ATTN_CHUNK_P, attnSMax, attnPartialsLen, attnActiveParts, attnOwnerPart,
  attnPartRange, lseReduce,
} from "../src/engine/attnsplit";

// Unit CPU-side sul piano dell'attention split (spec B2 §Attention split e §Unit)
// — convenzione CI-senza-GPU: geometria delle partizioni + proprietà della
// riduzione log-sum-exp; la matematica dei kernel la verificano microbench
// (attn-bench, vs CPU f64) e conformance (gate doppio).

describe("geometria partizioni", () => {
  it("sMax e partials per la shape del motore (ctxMax 1024, 14 head, dim 64)", () => {
    expect(attnSMax(1024)).toBe(16);
    expect(attnPartialsLen(14, 64, 1024)).toBe(14 * 16 * 66);
  });

  it("partizioni attive: n=pos+1 coperto esattamente, senza buchi né overlap", () => {
    for (const pos of [0, 1, 63, 64, 127, 128, 469, 575, 1023]) {
      const n = pos + 1;
      const nParts = attnActiveParts(pos);
      expect(nParts).toBe(Math.ceil(n / ATTN_CHUNK_P));
      let next = 0;
      for (let p = 0; p < nParts; p++) {
        const { begin, end } = attnPartRange(p, n);
        expect(begin).toBe(next);
        expect(end).toBeGreaterThan(begin); // ogni partizione attiva è non-vuota
        next = end;
      }
      expect(next).toBe(n);
    }
  });

  it("owner = partizione che contiene pos (rope k_cur + append)", () => {
    for (const pos of [0, 63, 64, 468, 575, 1023]) {
      const owner = attnOwnerPart(pos);
      const { begin, end } = attnPartRange(owner, pos + 1);
      expect(pos).toBeGreaterThanOrEqual(begin);
      expect(pos).toBeLessThan(end === begin + ATTN_CHUNK_P ? end : begin + ATTN_CHUNK_P);
      expect(end).toBe(pos + 1); // l'owner è sempre l'ULTIMA partizione attiva
      expect(owner).toBe(attnActiveParts(pos) - 1);
    }
  });
});

describe("lseReduce — equivalenza alla softmax monolitica", () => {
  // riferimento: softmax monolitica su score concatenati, output Σ p_i·v_i
  const monolithic = (scores: number[], vs: number[][]): Float64Array => {
    const m = Math.max(...scores);
    const es = scores.map((s) => Math.exp(s - m));
    const sum = es.reduce((a, b) => a + b, 0);
    const out = new Float64Array(vs[0].length);
    es.forEach((e, i) => vs[i].forEach((v, d) => { out[d] += (e / sum) * v; }));
    return out;
  };
  // pass 1 di riferimento: partial {m, l, out non normalizzato} su uno slice
  const partial = (scores: number[], vs: number[][]) => {
    const m = Math.max(...scores);
    let l = 0;
    const out = new Float64Array(vs[0].length);
    scores.forEach((s, i) => {
      const e = Math.exp(s - m);
      l += e;
      vs[i].forEach((v, d) => { out[d] += e * v; });
    });
    return { m, l, out };
  };
  // LCG deterministico
  let seed = 7;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };

  it("split in 1..5 partizioni di taglie disuguali ≡ monolitica (1e-12)", () => {
    const N = 137, D = 8;
    const scores = Array.from({ length: N }, () => rnd() * 10);
    const vs = Array.from({ length: N }, () => Array.from({ length: D }, rnd));
    const ref = monolithic(scores, vs);
    for (const cuts of [[N], [64, N], [64, 128, N], [1, 65, 129, 130, N]]) {
      const parts = [];
      let prev = 0;
      for (const c of cuts) { parts.push(partial(scores.slice(prev, c), vs.slice(prev, c))); prev = c; }
      const got = lseReduce(parts);
      for (let d = 0; d < D; d++) expect(Math.abs(got[d] - ref[d])).toBeLessThan(1e-12);
    }
  });

  it("robusta a score estremi (max in partizioni diverse, niente overflow)", () => {
    const scores = [1000, -1000, 999.5, 3, 1001];
    const vs = scores.map((_, i) => [i + 1, -(i + 1)]);
    const ref = monolithic(scores, vs);
    const parts = [partial(scores.slice(0, 2), vs.slice(0, 2)), partial(scores.slice(2), vs.slice(2))];
    const got = lseReduce(parts);
    expect(Number.isFinite(got[0])).toBe(true);
    for (let d = 0; d < 2; d++) expect(Math.abs(got[d] - ref[d])).toBeLessThan(1e-9);
  });
});
