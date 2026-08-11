// Test del piano MoE del prefill batched (C3a fase 5, primo slice — it.26).
//
// Due gruppi: (1) proprietà STRUTTURALI del piano — biiezione delle selezioni,
// ordine deterministico, pesi allineati; (2) IDENTITÀ M=1 vs M>1 sul percorso
// CPU f32: eseguire il piano (per-expert sull'unione + combine in ordine k)
// produce BIT-IDENTICO al percorso per-token del decode (accumulo k crescente).
// È la condizione di identità della fase ridotta al suo nucleo ordinale: i
// kernel GPU replicheranno questa struttura (slot y[m][k] + combine k-order).
//
// DUE FAMIGLIE dal goal fase-D fase 4 (it.20): le stesse proprietà si
// verificano su GLM (64 expert, top-4) e su Qwen 3.6 (256 expert, top-8). Non
// è ridondanza — è ciò che tiene onesta la parametrizzazione: un piano che
// funziona solo col parco e col top-K di GLM passerebbe metà di questi test.
import { describe, expect, it } from "vitest";
import { planMoeChunk, combineMoeRow, GLM_PREFILL_M, type MoePlanShape } from "../src/engine/moeprefillplan";
import type { RouterSelection } from "../src/engine/moe";
import { GLM47_FLASH as G } from "../src/engine/shape";

// PRNG deterministico (mulberry32, convenzione repo)
const rng = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/** GLM è il default del piano; q35 è la seconda famiglia, con parco 4× e top-K 2×. */
const SHAPES: { name: string; cfg: MoePlanShape }[] = [
  { name: "glm-4.7-flash (64 expert, top-4)", cfg: { nExpert: G.nExpert, nExpertUsed: G.nExpertUsed } },
  { name: "qwen3.6-35b (256 expert, top-8)", cfg: { nExpert: 256, nExpertUsed: 8 } },
];

function randSelections(m: number, seed: number, cfg: MoePlanShape): RouterSelection[] {
  const r = rng(seed);
  return Array.from({ length: m }, () => {
    // top-K expert DISTINTI per token (contratto del router)
    const ids = new Set<number>();
    while (ids.size < cfg.nExpertUsed) ids.add(Math.floor(r() * cfg.nExpert));
    return {
      experts: Int32Array.from(ids),
      weights: Float64Array.from({ length: cfg.nExpertUsed }, () => r() * 1.8),
    };
  });
}

describe.each(SHAPES)("planMoeChunk — proprietà strutturali [$name]", ({ cfg }) => {
  const K = cfg.nExpertUsed;

  it("biiezione: ogni (riga, k) compare esattamente una volta nell'unione", () => {
    const sels = randSelections(16, 42, cfg);
    const plan = planMoeChunk(sels, GLM_PREFILL_M, cfg);
    const seen = new Set<string>();
    for (const b of plan.experts) {
      expect(b.rows.length).toBe(b.slots.length);
      for (let i = 0; i < b.rows.length; i++) {
        const key = `${b.rows[i]}:${b.slots[i]}`;
        expect(seen.has(key)).toBe(false); // nessun duplicato
        seen.add(key);
        // coerenza: quello slot di quella riga seleziona DAVVERO questo expert
        expect(sels[b.rows[i]].experts[b.slots[i]]).toBe(b.expert);
      }
    }
    expect(seen.size).toBe(16 * K); // copertura totale
  });

  it("ordine deterministico: expert crescenti, righe crescenti dentro l'expert", () => {
    const plan = planMoeChunk(randSelections(16, 7, cfg), GLM_PREFILL_M, cfg);
    for (let i = 1; i < plan.experts.length; i++) {
      expect(plan.experts[i].expert).toBeGreaterThan(plan.experts[i - 1].expert);
    }
    for (const b of plan.experts) {
      for (let i = 1; i < b.rows.length; i++) expect(b.rows[i]).toBeGreaterThan(b.rows[i - 1]);
    }
  });

  it("l'unione raggruppa davvero: |unione| ≤ min(K·M, nExpert), e i pesi sono selF32", () => {
    const sels = randSelections(16, 99, cfg);
    const plan = planMoeChunk(sels, GLM_PREFILL_M, cfg);
    expect(plan.experts.length).toBeLessThanOrEqual(Math.min(K * 16, cfg.nExpert));
    // con K·16 selezioni su nExpert id le collisioni sono certe in entrambe le
    // famiglie (GLM: 64 su 64; q35: 128 su 256 — compleanno)
    expect(plan.experts.length).toBeLessThan(K * 16);
    for (let m = 0; m < 16; m++) {
      for (let k = 0; k < K; k++) {
        expect(plan.weights[m * K + k]).toBe(Math.fround(sels[m].weights[k]));
      }
    }
  });

  it("validazione hard: chunk vuoto/oltre mMax, selezione monca, expert fuori range", () => {
    expect(() => planMoeChunk([], GLM_PREFILL_M, cfg)).toThrow(/1\.\./);
    expect(() => planMoeChunk(randSelections(GLM_PREFILL_M + 1, 1, cfg), GLM_PREFILL_M, cfg)).toThrow(/1\.\./);
    const bad = randSelections(2, 3, cfg);
    bad[1] = { experts: Int32Array.from([1, 2]), weights: Float64Array.from([0.1, 0.2]) };
    if (K !== 2) expect(() => planMoeChunk(bad, GLM_PREFILL_M, cfg)).toThrow(new RegExp(`attesi ${K}`));
    const oob = randSelections(1, 4, cfg);
    oob[0].experts[2] = cfg.nExpert;
    expect(() => planMoeChunk(oob, GLM_PREFILL_M, cfg)).toThrow(/fuori range/);
  });
});

describe.each(SHAPES)("identità M=1 vs M>1 (percorso CPU f32) [$name]", ({ cfg }) => {
  const D = 32; // dimensione ridotta: l'identità è ordinale, non dimensionale
  const K = cfg.nExpertUsed;

  // Expert finto ma non banale: y = fround(c_e · x[i]) + fround(x[perm]) — f32 a ogni op
  const fakeExpert = (e: number, x: Float32Array): Float32Array => {
    const c = Math.fround(0.013 * (e + 1));
    const out = new Float32Array(D);
    for (let i = 0; i < D; i++) {
      out[i] = Math.fround(Math.fround(c * x[i]) + x[(i * 7 + e) % D]);
    }
    return out;
  };

  // shexp finto: la base dell'accumulatore moeOut nel decode vero (glmmodel)
  const fakeShexp = (x: Float32Array): Float32Array => {
    const out = new Float32Array(D);
    for (let i = 0; i < D; i++) out[i] = Math.fround(0.5 * x[(i + 3) % D]);
    return out;
  };

  it("eseguire il piano (unione + combine k-order) ≡ catena del decode, BIT-IDENTICO", () => {
    const M = 16;
    const sels = randSelections(M, 1234, cfg);
    const r = rng(555);
    const xs = Array.from({ length: M }, () =>
      Float32Array.from({ length: D }, () => Math.fround(r() * 2 - 1)));
    const plan = planMoeChunk(sels, GLM_PREFILL_M, cfg);

    // percorso DECODE (riferimento, catena di glmmodel): moeOut = shexp;
    // += w_k·y_k in ordine k; poi x += moeOut
    const refOut = xs.map((x, m) => {
      const moeOut = fakeShexp(x);
      for (let k = 0; k < K; k++) {
        const w = Math.fround(sels[m].weights[k]); // selF32
        const y = fakeExpert(sels[m].experts[k], x);
        for (let i = 0; i < D; i++) moeOut[i] = Math.fround(moeOut[i] + Math.fround(w * y[i]));
      }
      const out = new Float32Array(D);
      for (let i = 0; i < D; i++) out[i] = Math.fround(x[i] + moeOut[i]);
      return out;
    });

    // percorso BATCHED: per expert dell'unione, scrivi y[m][k] negli slot;
    // l'ORDINE di esecuzione degli expert è quello dell'unione (diverso da k!)
    const slots: Float32Array[][] = Array.from({ length: M }, () => new Array<Float32Array>(K));
    for (const b of plan.experts) {
      for (let i = 0; i < b.rows.length; i++) {
        slots[b.rows[i]][b.slots[i]] = fakeExpert(b.expert, xs[b.rows[i]]);
      }
    }
    const gotOut = xs.map((x, m) => combineMoeRow(x, fakeShexp(x), slots[m], plan.weights, m, K));

    for (let m = 0; m < M; m++) {
      // uguaglianza ESATTA, elemento per elemento (Float32Array bit-uguali)
      expect(gotOut[m]).toEqual(refOut[m]);
    }
  });

  it("controprova: accumulare nell'ordine dell'UNIONE (senza slot) NON è identico", () => {
    // Se questa controprova passasse (cioè fosse identico), la struttura a
    // slot sarebbe complessità inutile: il test la tiene onesta.
    const M = 8;
    const sels = randSelections(M, 77, cfg);
    const r = rng(888);
    const xs = Array.from({ length: M }, () =>
      Float32Array.from({ length: D }, () => Math.fround(r() * 2 - 1)));
    const plan = planMoeChunk(sels, GLM_PREFILL_M, cfg);

    const refOut = xs.map((x, m) => {
      const moeOut = fakeShexp(x);
      for (let k = 0; k < K; k++) {
        const w = Math.fround(sels[m].weights[k]);
        const y = fakeExpert(sels[m].experts[k], x);
        for (let i = 0; i < D; i++) moeOut[i] = Math.fround(moeOut[i] + Math.fround(w * y[i]));
      }
      const out = new Float32Array(D);
      for (let i = 0; i < D; i++) out[i] = Math.fround(x[i] + moeOut[i]);
      return out;
    });
    const unionOut = xs.map((x) => fakeShexp(x));
    for (const b of plan.experts) {
      for (let i = 0; i < b.rows.length; i++) {
        const m = b.rows[i];
        const w = plan.weights[m * K + b.slots[i]];
        const y = fakeExpert(b.expert, xs[m]);
        for (let j = 0; j < D; j++) unionOut[m][j] = Math.fround(unionOut[m][j] + Math.fround(w * y[j]));
      }
    }
    for (let m = 0; m < M; m++) {
      for (let j = 0; j < D; j++) unionOut[m][j] = Math.fround(xs[m][j] + unionOut[m][j]);
    }
    const anyDiff = unionOut.some((o, m) => o.some((v, i) => v !== refOut[m][i]));
    expect(anyDiff).toBe(true); // le somme riordinate DIVERGONO in f32
  });
});

describe("default = GLM: chi non passa la config ottiene la famiglia-tesi", () => {
  it("planMoeChunk senza cfg ≡ planMoeChunk con la cfg di GLM", () => {
    const sels = randSelections(12, 2026, SHAPES[0].cfg);
    const a = planMoeChunk(sels);
    const b = planMoeChunk(sels, GLM_PREFILL_M, SHAPES[0].cfg);
    expect(a.m).toBe(b.m);
    expect(a.weights).toEqual(b.weights);
    expect(a.experts.map((e) => e.expert)).toEqual(b.experts.map((e) => e.expert));
    for (let i = 0; i < a.experts.length; i++) {
      expect(a.experts[i].rows).toEqual(b.experts[i].rows);
      expect(a.experts[i].slots).toEqual(b.experts[i].slots);
    }
  });
});
