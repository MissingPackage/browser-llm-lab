import { describe, it, expect } from "vitest";
import { Lru, Lfru } from "../tools/oracle-moe/sim/policies.js";
import {
  simulate, usageCounts, topKeys, workingSet, skew, key,
  type TraceRow,
} from "../tools/oracle-moe/sim/simulate.js";

const N_EXPERT = 4;
const N_MOE = 2;

/** riga di traccia sintetica: top4 per layer + (opz.) top8 predetti */
function row(p: number, i: number, ph: "p" | "d", e: number[], pr?: number[]): TraceRow {
  return pr ? { p, i, ph, e, pr } : { p, i, ph, e };
}

describe("Lru", () => {
  it("hit sul riuso entro capacita', miss dopo eviction", () => {
    const c = new Lru(2);
    expect(c.access(1)).toBe(false);
    expect(c.access(2)).toBe(false);
    expect(c.access(1)).toBe(true);   // 1 e' il piu' recente
    expect(c.access(3)).toBe(false);  // evicta 2 (LRU)
    expect(c.has(2)).toBe(false);
    expect(c.access(1)).toBe(true);
    expect(c.size()).toBe(2);
  });

  it("non supera mai la capacita'", () => {
    const c = new Lru(3);
    for (let i = 0; i < 50; i++) c.access(i % 10);
    expect(c.size()).toBeLessThanOrEqual(3);
  });
});

describe("Lfru", () => {
  it("protegge il caldo dall'eviction (frequenza, non solo recency)", () => {
    const c = new Lfru({ capacity: 2 });
    for (let i = 0; i < 5; i++) c.access(1); // 1 diventa caldo
    c.access(2);
    c.access(3);                              // deve evictare 2, non 1
    expect(c.has(1)).toBe(true);
    expect(c.has(2)).toBe(false);
  });

  it("i pinnati non vengono mai evitti", () => {
    const c = new Lfru({ capacity: 2, pinned: [7] });
    for (let i = 10; i < 30; i++) c.access(i);
    expect(c.has(7)).toBe(true);
    expect(c.size()).toBeLessThanOrEqual(2);
  });

  it("il guard rifiuta la speculazione che sacrificherebbe un residente warm", () => {
    const c = new Lfru({ capacity: 1, prefetchGuard: true });
    c.access(1); c.access(1); c.access(1);   // warm: hits>=2, heat alto
    c.prefetch!(2);
    expect(c.has(1)).toBe(true);
    expect(c.has(2)).toBe(false);
    expect(c.stats.prefetchRejected).toBe(1);
  });

  it("senza guard la speculazione entra comunque", () => {
    const c = new Lfru({ capacity: 1, prefetchGuard: false });
    c.access(1); c.access(1); c.access(1);
    c.prefetch!(2);
    expect(c.has(2)).toBe(true);
  });

  it("il prefetch non conta come accesso (nessun hit fantasma)", () => {
    const c = new Lfru({ capacity: 4 });
    c.prefetch!(9);
    expect(c.has(9)).toBe(true);
    expect(c.access(9)).toBe(true); // il prefetch utile si vede come hit al primo uso vero
  });
});

describe("simulate", () => {
  const rows: TraceRow[] = [
    row(0, 0, "p", [0, 1, 2, 3, 0, 1, 2, 3]),
    row(0, 1, "d", [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0]),
    row(0, 2, "d", [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0]),
  ];

  it("con budget pieno l'hit-rate satura a 1 dopo il primo giro (invariante di spec)", () => {
    for (const policy of ["lru", "lfru", "lfru+pin", "lfru+pin+prefetch"] as const) {
      const r = simulate(rows, { budget: N_EXPERT * N_MOE, policy, nExpert: N_EXPERT, nMoe: N_MOE });
      // 8 accessi per riga; la prima riga sono tutti miss obbligati (cold)
      expect(r.hits).toBe(r.accesses - 8);
      expect(r.decodeHitRate).toBe(1);
    }
  });

  it("conta accessi e decode separatamente", () => {
    const r = simulate(rows, { budget: 8, policy: "lru", nExpert: N_EXPERT, nMoe: N_MOE });
    expect(r.accesses).toBe(3 * N_MOE * 4);
    expect(r.decodeAccesses).toBe(2 * N_MOE * 4);
  });

  it("il prefetch riscalda la cache: piu' hit di decode a budget stretto", () => {
    // traccia che alterna due gruppi di expert, con predizioni corrette
    const alt: TraceRow[] = [];
    for (let i = 0; i < 12; i++) {
      const a = i % 2 === 0 ? [0, 1, 2, 3] : [3, 2, 1, 0];
      const nxt = i % 2 === 0 ? [3, 2, 1, 0] : [0, 1, 2, 3];
      alt.push(row(0, i, "d", [...a, ...a], [...nxt, 0, 0, 0, 0, ...nxt, 0, 0, 0, 0]));
    }
    const base = simulate(alt, { budget: 4, policy: "lfru+pin", nExpert: N_EXPERT, nMoe: N_MOE });
    const pre = simulate(alt, { budget: 4, policy: "lfru+pin+prefetch", nExpert: N_EXPERT, nMoe: N_MOE, prefetchK: 4 });
    expect(pre.prefetched).toBeGreaterThan(0);
    expect(pre.decodeHits).toBeGreaterThanOrEqual(base.decodeHits);
  });

  it("e' deterministico", () => {
    const a = simulate(rows, { budget: 3, policy: "lfru", nExpert: N_EXPERT, nMoe: N_MOE });
    const b = simulate(rows, { budget: 3, policy: "lfru", nExpert: N_EXPERT, nMoe: N_MOE });
    expect(a).toEqual(b);
  });
});

describe("statistiche", () => {
  const rows: TraceRow[] = [
    row(0, 0, "p", [0, 0, 0, 1, 2, 2, 2, 3]),
    row(0, 1, "d", [0, 0, 0, 1, 2, 2, 2, 3]),
  ];

  it("usageCounts conta per (layer,expert) con chiave lineare", () => {
    const u = usageCounts(rows, N_EXPERT, N_MOE);
    expect(u.get(key(0, 0, N_EXPERT))).toBe(6); // 3 per riga x 2 righe, layer 0
    expect(u.get(key(1, 2, N_EXPERT))).toBe(6); // layer 1
    expect(u.get(key(0, 2, N_EXPERT))).toBeUndefined(); // expert 2 non usato nel layer 0
  });

  it("topKeys ordina per uso con tie-break deterministico", () => {
    const u = new Map([[5, 10], [3, 10], [9, 1]]);
    expect(topKeys(u, 2)).toEqual([3, 5]);
  });

  it("skew: cumulativa top-N monotona e a 1 con tutti gli expert", () => {
    const u = usageCounts(rows, N_EXPERT, N_MOE);
    const s = skew(u, N_EXPERT, N_MOE, [1, 2, 4]);
    expect(s.aggregate[0]).toBeLessThanOrEqual(s.aggregate[1]);
    expect(s.aggregate[2]).toBeCloseTo(1, 10);
    expect(s.perLayer).toHaveLength(N_MOE);
  });

  it("workingSet: mai piu' di 4 accessi distinti per finestra di 1 posizione", () => {
    const w = workingSet(rows, 1, N_EXPERT, N_MOE);
    expect(w.max).toBeLessThanOrEqual(N_MOE * 4);
    expect(w.window).toBe(1);
  });
});
