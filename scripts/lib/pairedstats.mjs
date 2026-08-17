// LA STATISTICA DEL CONFRONTO FRA DUE QUANTIZZAZIONI, in un modulo puro.
//
// PERCHE' APPAIATO, e non "due medie a confronto". I due modelli vedono gli
// STESSI token nelle STESSE posizioni: la quantita' che interessa e' la
// differenza per token, non la differenza fra due medie. La varianza della
// differenza appaiata e' molto piu' bassa, perche' la difficolta' del token —
// che e' la sorgente di rumore dominante — si cancella.
//
// E' la stessa lezione che questo repo ha gia' pagato una volta, in un'altra
// forma: «un campione da 22 posizioni non distingue niente: ±1 colpo vale ±4,5
// punti. Prima di concludere da un conteggio di successi, guarda rango e
// log-prob del bersaglio — stessa informazione, varianza molto piu' bassa».
//
// PERCHE' IL BOOTSTRAP A BLOCCHI e non l'errore standard classico. I token di
// un testo NON sono indipendenti: sbagliare una parola rende piu' probabile
// sbagliare la successiva. Un SE calcolato come sigma/sqrt(n) su token
// autocorrelati e' ottimista di un fattore che non conosciamo. Il bootstrap a
// blocchi mobili ricampiona SEGMENTI contigui, quindi conserva la correlazione
// dentro il blocco e non pretende che non ci sia.
//
// DETERMINISMO: il generatore e' seminato e scritto qui (mulberry32, lo stesso
// della chat). Due esecuzioni sullo stesso artefatto danno lo stesso intervallo:
// un intervallo di confidenza che cambia a ogni run non e' un gate.

/** mulberry32: RNG riproducibile dal seed (identico a quello del sampler di chat). */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const mean = (xs) => {
  if (xs.length === 0) throw new Error("pairedstats: media di un campione vuoto");
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
};

/** nat → bit: le NLL di llama.cpp sono in nat, i bits/token si leggono meglio. */
export const NATS_PER_BIT = Math.LN2;
export const toBits = (nats) => nats / NATS_PER_BIT;

/**
 * Bootstrap a BLOCCHI MOBILI sulla media di `xs`.
 *
 * `blockLen` e' la scala a cui si presume che la correlazione si esaurisca: 512
 * token e' ~un paragrafo, e la scelta va DICHIARATA nell'artefatto perche' e'
 * l'unica assunzione libera di questa procedura.
 *
 * Ritorna la media osservata e i percentili [lo, hi] delle medie ricampionate.
 */
export function blockBootstrapCI(xs, { blockLen = 512, resamples = 2000, seed = 20260817, alpha = 0.05 } = {}) {
  const n = xs.length;
  if (n === 0) throw new Error("pairedstats: bootstrap di un campione vuoto");
  const L = Math.max(1, Math.min(blockLen, n));
  const nBlocks = Math.ceil(n / L);
  const nStarts = n - L + 1;
  const rand = rng(seed);
  const means = new Float64Array(resamples);
  for (let r = 0; r < resamples; r++) {
    let s = 0, taken = 0;
    for (let b = 0; b < nBlocks && taken < n; b++) {
      const start = Math.floor(rand() * nStarts);
      const end = Math.min(start + L, start + (n - taken));
      for (let i = start; i < end; i++) { s += xs[i]; taken++; }
    }
    means[r] = s / taken;
  }
  means.sort();
  const q = (p) => means[Math.min(resamples - 1, Math.max(0, Math.round(p * (resamples - 1))))];
  return {
    mean: mean(xs), lo: q(alpha / 2), hi: q(1 - alpha / 2),
    blockLen: L, resamples, seed, alpha,
  };
}

/**
 * Frazione di posizioni in cui B e' MIGLIORE di A (delta < 0 = meno sorpresa).
 * E' un test di segno: non assume nessuna distribuzione, e regge anche quando la
 * media e' trascinata da poche posizioni catastrofiche — che e' esattamente cio'
 * che una quantizzazione aggressiva produce.
 */
export function winRate(deltas, eps = 0) {
  let better = 0, worse = 0, tie = 0;
  for (const d of deltas) {
    if (d < -eps) better++;
    else if (d > eps) worse++;
    else tie++;
  }
  return { better, worse, tie, n: deltas.length, fracBetter: better / deltas.length };
}

/** Quantili campionari (per la CODA dei danni: la media non la mostra). */
export function quantiles(xs, ps = [0.5, 0.9, 0.99, 1]) {
  const s = Float64Array.from(xs);
  s.sort();
  const out = {};
  for (const p of ps) out[String(p)] = s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))];
  return out;
}

/**
 * Il confronto appaiato completo su due vettori di NLL per token.
 *
 * LANCIA se le lunghezze non combaciano: significa che i due modelli non hanno
 * visto gli stessi token, e allora nessuna delle statistiche qui sotto vuol
 * dire quello che sembra.
 */
export function pairedNll(nllA, nllB, opts = {}) {
  if (nllA.length !== nllB.length) {
    throw new Error(`pairedstats: ${nllA.length} vs ${nllB.length} posizioni — il confronto appaiato non e' definito`);
  }
  const deltas = new Float64Array(nllA.length);
  for (let i = 0; i < nllA.length; i++) deltas[i] = nllB[i] - nllA[i];
  const ci = blockBootstrapCI(deltas, opts);
  return {
    n: nllA.length,
    bitsA: toBits(mean(nllA)),
    bitsB: toBits(mean(nllB)),
    deltaBits: { mean: toBits(ci.mean), lo: toBits(ci.lo), hi: toBits(ci.hi) },
    blockLen: ci.blockLen, resamples: ci.resamples, seed: ci.seed,
    win: winRate(deltas),
    deltaTail: quantiles(deltas),
  };
}
