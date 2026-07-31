// Policy di cache degli esperti per il simulatore trace-driven (spec engine-fase-c1
// §Policy simulate). Logica PURA e testabile: in fase C3 questi stessi tipi girano
// nel worker del motore, quindi niente dipendenze da Node o dal formato traccia.
//
// Unita' di cache = 1 expert (layer, id) => slab singola (~5.3 MB q4 su GLM-4.7-Flash).
// Chiave lineare: key = layer * nExpert + id.

export interface Cache {
  /** Accesso reale: true se hit. Registra sempre l'uso. */
  access(key: number): boolean;
  /** Caricamento speculativo (prefetch): non conta come accesso, puo' essere rifiutato. */
  prefetch?(key: number): void;
  /** Numero di residenti (per invarianti nei test). */
  size(): number;
  has(key: number): boolean;
}

/** LRU puro: il pavimento del confronto. */
export class Lru implements Cache {
  private map = new Map<number, number>(); // key -> clock
  private clock = 0;
  constructor(private capacity: number) {}
  size() { return this.map.size; }
  has(k: number) { return this.map.has(k); }
  access(k: number): boolean {
    const hit = this.map.has(k);
    this.map.set(k, ++this.clock);
    if (this.map.size > this.capacity) this.evictOne();
    return hit;
  }
  private evictOne() {
    let victim = -1, best = Infinity;
    for (const [k, c] of this.map) if (c < best) { best = c; victim = k; }
    if (victim >= 0) this.map.delete(victim);
  }
}

interface Entry {
  heat: number;     // frequenza con decadimento
  recency: number;  // clock dell'ultimo accesso (troncato a 8 bit come tier.h)
  hits: number;     // accessi totali dalla residenza corrente (guard anti-eviction)
  pinned: boolean;
}

export interface LfruOptions {
  capacity: number;
  /** Chiavi pinnate (mai evitte). Devono stare nella capacity. */
  pinned?: Iterable<number>;
  /** Ogni quanti accessi dimezzare l'heat (tier.h: right-shift periodico). */
  decayEvery?: number;
  /** Guard: una speculazione non evicta un residente "warm" (>=2 accessi e piu' caldo). */
  prefetchGuard?: boolean;
}

/**
 * LFRU stile colibri `tier.h`: score = (heat << 8) | recency, con decadimento
 * periodico dell'heat e isteresi 25%+4 sull'eviction indotta da prefetch.
 */
export class Lfru implements Cache {
  private map = new Map<number, Entry>();
  private clock = 0;
  private sinceDecay = 0;
  private capacity: number;
  private decayEvery: number;
  private prefetchGuard: boolean;
  /** Contatori diagnostici (non entrano nell'hit-rate). */
  readonly stats = { prefetched: 0, prefetchRejected: 0, evictions: 0 };

  constructor(opts: LfruOptions) {
    this.capacity = opts.capacity;
    this.decayEvery = opts.decayEvery ?? 4096;
    this.prefetchGuard = opts.prefetchGuard ?? true;
    for (const k of opts.pinned ?? []) {
      if (this.map.size >= this.capacity) break;
      this.map.set(k, { heat: 1 << 16, recency: 255, hits: 0, pinned: true });
    }
  }

  size() { return this.map.size; }
  has(k: number) { return this.map.has(k); }

  private score(e: Entry): number {
    return e.heat * 256 + (e.recency & 0xff);
  }

  private touch(e: Entry) {
    e.heat += 1;
    e.recency = ++this.clock & 0xff;
    if (++this.sinceDecay >= this.decayEvery) {
      this.sinceDecay = 0;
      for (const v of this.map.values()) if (!v.pinned) v.heat >>= 1;
    }
  }

  access(k: number): boolean {
    const e = this.map.get(k);
    if (e) {
      e.hits++;
      this.touch(e);
      return true;
    }
    const fresh: Entry = { heat: 1, recency: ++this.clock & 0xff, hits: 1, pinned: false };
    this.insert(k, fresh, false);
    return false;
  }

  prefetch(k: number): void {
    if (this.map.has(k)) return;
    this.stats.prefetched++;
    const fresh: Entry = { heat: 1, recency: ++this.clock & 0xff, hits: 0, pinned: false };
    if (!this.insert(k, fresh, true)) this.stats.prefetchRejected++;
  }

  /** @returns false se l'inserimento e' stato rifiutato (solo per speculazioni). */
  private insert(k: number, e: Entry, speculative: boolean): boolean {
    if (this.map.size < this.capacity) { this.map.set(k, e); return true; }
    const victim = this.pickVictim();
    if (victim < 0) return false;
    if (speculative && this.prefetchGuard) {
      const v = this.map.get(victim)!;
      // isteresi colibri: un residente con >=2 accessi e score sensibilmente
      // piu' alto non viene sacrificato per una speculazione.
      if (v.hits >= 2 && this.score(v) > this.score(e) * 1.25 + 4) return false;
    }
    this.map.delete(victim);
    this.stats.evictions++;
    this.map.set(k, e);
    return true;
  }

  private pickVictim(): number {
    let victim = -1, best = Infinity;
    for (const [k, e] of this.map) {
      if (e.pinned) continue;
      const s = this.score(e);
      if (s < best) { best = s; victim = k; }
    }
    return victim;
  }
}
