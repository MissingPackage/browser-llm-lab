import { describe, expect, it } from "vitest";
import { LruFast, simulateOptimistic, simulateBelady, crossTokenLocality } from "../tools/oracle-moe/sim/wp0.js";
import type { TraceRow } from "../tools/oracle-moe/sim/simulate.js";

// Mini-modello: 2 layer MoE, 8 expert per layer, top-4.
const N_MOE = 2, N_EXPERT = 8;

/** riga di traccia con selezioni [layer0, layer1] e predizioni opzionali */
const row = (p: number, ph: "p" | "d", sel: number[][], pred?: number[][]): TraceRow => ({
  p, i: 0, ph,
  e: sel.flat(),
  pr: pred ? pred.flat() : undefined,
});

const PRED8 = [
  [0, 1, 2, 3, 4, 5, 6, 7],
  [0, 1, 2, 3, 4, 5, 6, 7],
];

describe("wp0: LruFast", () => {
  it("evicta il least-recently-inserted/touched, O(1) sull'ordine della Map", () => {
    const c = new LruFast(2);
    c.insert(1); c.insert(2);
    c.touch(1);           // 2 diventa il piu' vecchio
    expect(c.insert(3)).toBe(1);
    expect(c.has(2)).toBe(false);
    expect(c.has(1)).toBe(true);
  });
});

describe("wp0: semantica differita del decode ottimistico", () => {
  it("un miss al token t NON diventa hit dentro t, ma e' residente a t+1 (repair)", () => {
    const sel = [[0, 1, 2, 3], [0, 1, 2, 3]];
    const rows = [row(0, "d", sel, PRED8), row(0, "d", sel, PRED8)];
    const r = simulateOptimistic(rows, { budget: 100, nExpert: N_EXPERT, nMoe: N_MOE, prefetchK: 0, warmupTokens: 0 });
    // token 1: cache vuota congelata => 8 miss; token 2: tutto riparato => 0 miss
    expect(r.all.decodeTokens).toBe(2);
    expect(r.all.dirtyTokens).toBe(1);
    expect(r.decodeHits).toBe(8);
    expect(r.all.firstMissLayerHist[0]).toBe(1);
  });

  it("il prefill e' sincrono e scalda la cache del decode", () => {
    const sel = [[0, 1, 2, 3], [0, 1, 2, 3]];
    const rows = [row(0, "p", sel), row(0, "d", sel, PRED8)];
    const r = simulateOptimistic(rows, { budget: 100, nExpert: N_EXPERT, nMoe: N_MOE, prefetchK: 0, warmupTokens: 0 });
    expect(r.all.dirtyTokens).toBe(0);
    expect(r.decodeHitRate).toBe(1);
  });

  it("il prefetch LOOKA entra al confine: salva t+1, mai t", () => {
    // token 1 usa 0-3 e predice 4-7 (K=4); token 2 usa 4-7
    const pred47 = [[4, 5, 6, 7, 0, 1, 2, 3], [4, 5, 6, 7, 0, 1, 2, 3]];
    const rows = [
      row(0, "d", [[0, 1, 2, 3], [0, 1, 2, 3]], pred47),
      row(0, "d", [[4, 5, 6, 7], [4, 5, 6, 7]], pred47),
    ];
    const noPf = simulateOptimistic(rows, { budget: 100, nExpert: N_EXPERT, nMoe: N_MOE, prefetchK: 0, warmupTokens: 0 });
    const pf = simulateOptimistic(rows, { budget: 100, nExpert: N_EXPERT, nMoe: N_MOE, prefetchK: 4, warmupTokens: 0 });
    expect(noPf.all.dirtyTokens).toBe(2);   // entrambi sporchi
    expect(pf.all.dirtyTokens).toBe(1);     // il secondo e' salvato dal prefetch del primo
    expect(pf.prefetchInserted).toBeGreaterThan(0);
  });

  it("firstMissLayer e la quota late-half sono coerenti", () => {
    // token 2: layer 0 tutto hit, layer 1 (= seconda meta') ha un miss
    const rows = [
      row(0, "d", [[0, 1, 2, 3], [0, 1, 2, 3]], PRED8),
      row(0, "d", [[0, 1, 2, 3], [0, 1, 2, 4]], PRED8),
    ];
    const r = simulateOptimistic(rows, { budget: 100, nExpert: N_EXPERT, nMoe: N_MOE, prefetchK: 0, warmupTokens: 0 });
    expect(r.all.dirtyTokens).toBe(2);
    expect(r.all.firstMissLayerHist[1]).toBe(1);      // il secondo token parte sporco dal layer 1
    expect(r.all.dirtyLateHalfFraction).toBe(0.5);
  });
});

describe("wp0: belady e localita'", () => {
  it("belady non e' mai sotto una LRU sincrona a pari budget (ceiling)", () => {
    // pattern avverso alla LRU: scansione ciclica di 3 working set con budget 8
    const sets = [
      [[0, 1, 2, 3], [0, 1, 2, 3]],
      [[4, 5, 6, 7], [4, 5, 6, 7]],
      [[0, 1, 4, 5], [2, 3, 6, 7]],
    ];
    const rows: TraceRow[] = [];
    for (let t = 0; t < 30; t++) rows.push(row(0, "d", sets[t % 3], PRED8));
    const opt = simulateOptimistic(rows, { budget: 8, nExpert: N_EXPERT, nMoe: N_MOE, prefetchK: 0, warmupTokens: 0 });
    const bel = simulateBelady(rows, 8, N_EXPERT, N_MOE);
    expect(bel.decodeHitRate).toBeGreaterThanOrEqual(opt.decodeHitRate);
  });

  it("localita' cross-token: W=1 con selezioni identiche => 100%", () => {
    const sel = [[0, 1, 2, 3], [4, 5, 6, 7]];
    const rows = [row(0, "d", sel, PRED8), row(0, "d", sel, PRED8), row(0, "d", sel, PRED8)];
    const loc = crossTokenLocality(rows, [1], N_EXPERT, N_MOE);
    expect(loc[0].fraction).toBe(1);
  });
});
