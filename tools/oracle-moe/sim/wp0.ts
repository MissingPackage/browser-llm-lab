// WP-0 — simulazione trace-driven della TASSA DI REPLAY del decode ottimistico
// (ruling PI 2026-08-06, docket item 18; studio 2026-08-06-waste-kimik3).
//
// Differenza dal simulatore C1 (simulate.ts): la semantica e' quella del
// regime a 1 submit/token. Dopo il submit la slotTable non si tocca piu',
// quindi TUTTO cio' che un token di decode causa (repair dei miss, prefetch
// LOOKA) diventa residente solo AL CONFINE col token successivo. Un miss non
// e' un ensure sincrono: e' un token sporco che si ripara con un replay dal
// primo layer sporco. Il prefill resta a semantica sincrona (il suo path
// puo' fare ensure, fase 5).
//
// Logica PURA come simulate.ts: nessun filesystem, nessun formato traccia.

import type { TraceRow } from "./simulate.js";
import { key } from "./simulate.js";

/** LRU O(1) sull'ordine di inserzione della Map (delete+set = move-to-back). */
export class LruFast {
  map = new Map<number, true>();
  capacity: number;
  constructor(capacity: number) { this.capacity = capacity; }
  has(k: number) { return this.map.has(k); }
  size() { return this.map.size; }
  /** move-to-back senza inserire se assente */
  touch(k: number) { if (this.map.has(k)) { this.map.delete(k); this.map.set(k, true); } }
  /** inserisce (o rinfresca) e ritorna il numero di eviction causate */
  insert(k: number): number {
    if (this.map.has(k)) { this.map.delete(k); this.map.set(k, true); return 0; }
    this.map.set(k, true);
    let ev = 0;
    while (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value as number;
      this.map.delete(oldest);
      ev++;
    }
    return ev;
  }
}

export interface Wp0Options {
  budget: number;
  nExpert: number;
  nMoe: number;
  /** 0 = niente prefetch; K>0 = primi K predetti LOOKA per layer, uniti al confine */
  prefetchK: number;
  /** posizioni di decode iniziali per prompt escluse dalle statistiche steady-state */
  warmupTokens?: number;
}

export interface Wp0TokenStats {
  decodeTokens: number;
  dirtyTokens: number;
  pDirty: number;
  meanMissPerToken: number;
  p95MissPerToken: number;
  maxMissPerToken: number;
  /** istogramma del PRIMO layer sporco (indice layer MoE 0..nMoe-1), solo token sporchi */
  firstMissLayerHist: number[];
  /** quota di token sporchi col primo miss nella seconda meta' dei layer (riparabili da una variante a 2 segmenti) */
  dirtyLateHalfFraction: number;
}

export interface Wp0Result {
  policy: string;
  budget: number;
  prefetchK: number;
  /** hit-rate per-selezione sul decode (continuita' con C1) */
  decodeAccesses: number;
  decodeHits: number;
  decodeHitRate: number;
  prefetchInserted: number;
  evictions: number;
  all: Wp0TokenStats;
  steady: Wp0TokenStats;
}

const emptyAcc = (nMoe: number) => ({
  tokens: 0, dirty: 0, missCounts: [] as number[], firstHist: new Array(nMoe).fill(0), lateDirty: 0,
});

function finish(acc: ReturnType<typeof emptyAcc>): Wp0TokenStats {
  const sorted = [...acc.missCounts].sort((a, b) => a - b);
  const mean = acc.missCounts.length ? acc.missCounts.reduce((a, b) => a + b, 0) / acc.missCounts.length : 0;
  const p95 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] : 0;
  return {
    decodeTokens: acc.tokens,
    dirtyTokens: acc.dirty,
    pDirty: acc.tokens ? acc.dirty / acc.tokens : 0,
    meanMissPerToken: mean,
    p95MissPerToken: p95,
    maxMissPerToken: sorted.length ? sorted[sorted.length - 1] : 0,
    firstMissLayerHist: acc.firstHist,
    dirtyLateHalfFraction: acc.dirty ? acc.lateDirty / acc.dirty : 0,
  };
}

/**
 * Regime ottimistico: prefill sincrono, decode a inserimento differito.
 * Il replay e' modellato come repair: i miss del token entrano in cache al
 * confine (il replay li ha fetchati), il costo si calcola fuori (cost model).
 */
export function simulateOptimistic(rows: TraceRow[], opts: Wp0Options): Wp0Result {
  const { budget, nExpert, nMoe, prefetchK } = opts;
  const warmup = opts.warmupTokens ?? 32;
  const cache = new LruFast(budget);
  let decodeAccesses = 0, decodeHits = 0, prefetchInserted = 0, evictions = 0;
  const all = emptyAcc(nMoe), steady = emptyAcc(nMoe);
  const decodePos = new Map<number, number>(); // prompt -> posizioni di decode viste

  for (const r of rows) {
    if (r.ph === "p") {
      // prefill: semantica sincrona (ensure per accesso), come C1
      for (let s = 0; s < nMoe; s++)
        for (let j = 0; j < 4; j++) evictions += cache.insert(key(s, r.e[s * 4 + j], nExpert));
      continue;
    }
    // decode: stato CONGELATO al submit — prima si valuta tutto, poi si applica
    const missSet = new Set<number>();
    let firstMissLayer = -1;
    for (let s = 0; s < nMoe; s++) {
      for (let j = 0; j < 4; j++) {
        const k = key(s, r.e[s * 4 + j], nExpert);
        decodeAccesses++;
        if (cache.has(k) && !missSet.has(k)) decodeHits++;
        else { missSet.add(k); if (firstMissLayer < 0) firstMissLayer = s; }
      }
    }
    // confine di token: touch dei residenti usati, repair dei miss, poi prefetch
    for (let s = 0; s < nMoe; s++)
      for (let j = 0; j < 4; j++) cache.touch(key(s, r.e[s * 4 + j], nExpert));
    for (const k of missSet) evictions += cache.insert(k);
    if (prefetchK > 0 && r.pr) {
      for (let s = 0; s < nMoe; s++) {
        for (let j = 0; j < prefetchK; j++) {
          const k = key(s, r.pr[s * 8 + j], nExpert);
          if (!cache.has(k)) { evictions += cache.insert(k); prefetchInserted++; }
        }
      }
    }
    // statistiche per-token
    const nMiss = missSet.size;
    const pos = decodePos.get(r.p) ?? 0;
    decodePos.set(r.p, pos + 1);
    for (const acc of pos >= warmup ? [all, steady] : [all]) {
      acc.tokens++;
      acc.missCounts.push(nMiss);
      if (nMiss > 0) {
        acc.dirty++;
        acc.firstHist[firstMissLayer]++;
        if (firstMissLayer >= nMoe / 2) acc.lateDirty++;
      }
    }
  }
  return {
    policy: prefetchK > 0 ? `opt+looka${prefetchK}` : "opt",
    budget, prefetchK,
    decodeAccesses, decodeHits,
    decodeHitRate: decodeAccesses ? decodeHits / decodeAccesses : 0,
    prefetchInserted, evictions,
    all: finish(all),
    steady: finish(steady),
  };
}

/**
 * Belady per-accesso (inserimento immediato): CEILING ASSOLUTO di ogni policy
 * (l'inserimento differito puo' solo peggiorare). Solo hit-rate per-selezione.
 */
export function simulateBelady(rows: TraceRow[], budget: number, nExpert: number, nMoe: number): {
  policy: "belady"; budget: number; decodeAccesses: number; decodeHits: number; decodeHitRate: number;
} {
  // stream di accessi + liste next-use per chiave
  const stream: number[] = [];
  const isDecode: boolean[] = [];
  for (const r of rows)
    for (let s = 0; s < nMoe; s++)
      for (let j = 0; j < 4; j++) { stream.push(key(s, r.e[s * 4 + j], nExpert)); isDecode.push(r.ph === "d"); }
  const nextUses = new Map<number, number[]>();
  for (let i = stream.length - 1; i >= 0; i--) {
    const arr = nextUses.get(stream[i]);
    if (arr) arr.push(i); else nextUses.set(stream[i], [i]);
  }
  // le liste sono in ordine decrescente: pop() da' il prossimo uso in ordine crescente
  const resident = new Set<number>();
  let decodeAccesses = 0, decodeHits = 0;
  for (let i = 0; i < stream.length; i++) {
    const k = stream[i];
    nextUses.get(k)!.pop();
    if (isDecode[i]) decodeAccesses++;
    if (resident.has(k)) { if (isDecode[i]) decodeHits++; continue; }
    if (resident.size >= budget) {
      let victim = -1, far = -1;
      for (const rk of resident) {
        const arr = nextUses.get(rk)!;
        const nu = arr.length ? arr[arr.length - 1] : Infinity;
        if (nu === Infinity) { victim = rk; break; }
        if (nu > far) { far = nu; victim = rk; }
      }
      resident.delete(victim);
    }
    resident.add(k);
  }
  return { policy: "belady", budget, decodeAccesses, decodeHits, decodeHitRate: decodeAccesses ? decodeHits / decodeAccesses : 0 };
}

/**
 * Localita' temporale cross-token del routing (decode, per prompt): quota
 * delle selezioni del token t presenti nell'unione delle selezioni degli
 * ultimi W token. E' il motivo per cui una LRU quasi-piena funziona, misurato.
 */
export function crossTokenLocality(rows: TraceRow[], windows: number[], nExpert: number, nMoe: number): {
  window: number; fraction: number;
}[] {
  const byPrompt = new Map<number, TraceRow[]>();
  for (const r of rows) {
    if (r.ph !== "d") continue;
    const arr = byPrompt.get(r.p) ?? [];
    arr.push(r);
    byPrompt.set(r.p, arr);
  }
  return windows.map((w) => {
    let inWindow = 0, total = 0;
    for (const seq of byPrompt.values()) {
      for (let t = w; t < seq.length; t++) {
        const seen = new Set<number>();
        for (let b = t - w; b < t; b++)
          for (let s = 0; s < nMoe; s++)
            for (let j = 0; j < 4; j++) seen.add(key(s, seq[b].e[s * 4 + j], nExpert));
        for (let s = 0; s < nMoe; s++)
          for (let j = 0; j < 4; j++) { total++; if (seen.has(key(s, seq[t].e[s * 4 + j], nExpert))) inWindow++; }
      }
    }
    return { window: w, fraction: total ? inWindow / total : 0 };
  });
}
