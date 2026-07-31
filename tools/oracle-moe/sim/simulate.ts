// Simulatore trace-driven della cache esperti (spec engine-fase-c1 §Policy simulate).
// Puro: prende record di traccia gia' parsati, non tocca il filesystem.

import { Lru, Lfru, type Cache } from "./policies.js";

export interface TraceRow {
  /** indice di prompt */
  p: number;
  /** posizione nel prompt */
  i: number;
  /** 'p' prefill | 'd' decode */
  ph: "p" | "d";
  /** top-4 veri, 4 id per layer MoE, in ordine di layer */
  e: number[];
  /** top-8 predetti per layer (solo decode) */
  pr?: number[];
}

export type PolicyName = "lru" | "lfru" | "lfru+pin" | "lfru+pin+prefetch";

export interface SimOptions {
  /** slot di cache (unita' = 1 expert) */
  budget: number;
  policy: PolicyName;
  nExpert: number;
  nMoe: number;
  /** quanti predetti prefetchare per layer (K del PILOT) */
  prefetchK?: number;
  /** chiavi pinnate (calcolate fuori: split temporale anti-leakage) */
  pinned?: number[];
  /** ogni quanti accessi dimezzare l'heat (default tier.h-like) */
  decayEvery?: number;
  /** quota del budget riservata al pin (default 0.5, da spec) */
  pinFraction?: number;
}

export interface SimResult {
  policy: PolicyName;
  budget: number;
  accesses: number;
  hits: number;
  hitRate: number;
  /** hit-rate ristretta alle posizioni di decode (il regime del paging) */
  decodeAccesses: number;
  decodeHits: number;
  decodeHitRate: number;
  prefetched: number;
  prefetchRejected: number;
  evictions: number;
}

export const key = (layer: number, id: number, nExpert: number) => layer * nExpert + id;

/**
 * Conta gli usi (layer,expert) su un sottoinsieme di righe: base del pin
 * appreso e delle statistiche di residenza.
 */
export function usageCounts(rows: TraceRow[], nExpert: number, nMoe: number): Map<number, number> {
  const m = new Map<number, number>();
  for (const r of rows) {
    for (let s = 0; s < nMoe; s++) {
      for (let j = 0; j < 4; j++) {
        const k = key(s, r.e[s * 4 + j], nExpert);
        m.set(k, (m.get(k) ?? 0) + 1);
      }
    }
  }
  return m;
}

/** Top-N chiavi per uso, con tie-break deterministico sulla chiave. */
export function topKeys(usage: Map<number, number>, n: number): number[] {
  return [...usage.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .slice(0, n)
    .map(([k]) => k);
}

export function simulate(rows: TraceRow[], opts: SimOptions): SimResult {
  const { budget, policy, nExpert, nMoe } = opts;
  const prefetchK = opts.prefetchK ?? 8;
  const usePin = policy === "lfru+pin" || policy === "lfru+pin+prefetch";
  const usePrefetch = policy === "lfru+pin+prefetch";
  const pinned = usePin ? (opts.pinned ?? []).slice(0, Math.floor(budget * (opts.pinFraction ?? 0.5))) : [];

  const cache: Cache =
    policy === "lru"
      ? new Lru(budget)
      : new Lfru({ capacity: budget, pinned, prefetchGuard: usePrefetch, decayEvery: opts.decayEvery });

  let accesses = 0, hits = 0, decodeAccesses = 0, decodeHits = 0;

  for (const r of rows) {
    for (let s = 0; s < nMoe; s++) {
      // prefetch dei predetti per QUESTO layer: nella realta' e' emesso al layer
      // precedente, un layer di anticipo — qui equivale a "prima dei suoi accessi".
      if (usePrefetch && r.pr && cache.prefetch) {
        for (let j = 0; j < prefetchK; j++) {
          cache.prefetch(key(s, r.pr[s * 8 + j], nExpert));
        }
      }
      for (let j = 0; j < 4; j++) {
        const hit = cache.access(key(s, r.e[s * 4 + j], nExpert));
        accesses++;
        if (hit) hits++;
        if (r.ph === "d") { decodeAccesses++; if (hit) decodeHits++; }
      }
    }
  }

  const st = cache instanceof Lfru ? cache.stats : { prefetched: 0, prefetchRejected: 0, evictions: 0 };
  return {
    policy, budget, accesses, hits,
    hitRate: accesses ? hits / accesses : 0,
    decodeAccesses, decodeHits,
    decodeHitRate: decodeAccesses ? decodeHits / decodeAccesses : 0,
    prefetched: st.prefetched, prefetchRejected: st.prefetchRejected, evictions: st.evictions,
  };
}

/** Expert unici (layer,expert) in finestre scorrevoli di W posizioni. */
export function workingSet(rows: TraceRow[], w: number, nExpert: number, nMoe: number): {
  window: number; mean: number; p95: number; max: number;
} {
  const sizes: number[] = [];
  for (let start = 0; start + w <= rows.length; start += Math.max(1, Math.floor(w / 4))) {
    const seen = new Set<number>();
    for (let r = start; r < start + w; r++) {
      const row = rows[r];
      for (let s = 0; s < nMoe; s++)
        for (let j = 0; j < 4; j++) seen.add(key(s, row.e[s * 4 + j], nExpert));
    }
    sizes.push(seen.size);
  }
  if (!sizes.length) return { window: w, mean: 0, p95: 0, max: 0 };
  sizes.sort((a, b) => a - b);
  return {
    window: w,
    mean: sizes.reduce((a, b) => a + b, 0) / sizes.length,
    p95: sizes[Math.min(sizes.length - 1, Math.floor(sizes.length * 0.95))],
    max: sizes[sizes.length - 1],
  };
}

/** Quota cumulativa di selezioni coperta dai top-N expert, per layer e aggregata. */
export function skew(usage: Map<number, number>, nExpert: number, nMoe: number, ns: number[]): {
  perLayer: { layer: number; cumulative: number[] }[]; aggregate: number[];
} {
  const perLayer: { layer: number; cumulative: number[] }[] = [];
  const aggTotals = new Array(ns.length).fill(0);
  let grand = 0;
  for (let s = 0; s < nMoe; s++) {
    const counts: number[] = [];
    let total = 0;
    for (let e = 0; e < nExpert; e++) {
      const c = usage.get(key(s, e, nExpert)) ?? 0;
      counts.push(c); total += c;
    }
    counts.sort((a, b) => b - a);
    const cumulative = ns.map((n) => {
      const sum = counts.slice(0, n).reduce((a, b) => a + b, 0);
      return total ? sum / total : 0;
    });
    ns.forEach((n, idx) => { aggTotals[idx] += counts.slice(0, n).reduce((a, b) => a + b, 0); });
    grand += total;
    perLayer.push({ layer: s, cumulative });
  }
  return { perLayer, aggregate: aggTotals.map((t) => (grand ? t / grand : 0)) };
}
