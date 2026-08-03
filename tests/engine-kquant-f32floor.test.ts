import { describe, it, expect } from "vitest";
import { dequantQ5_K, dequantQ6_K, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES } from "../src/engine/quant";
import { pairGemvSiluQ5KFastWgsl, gemvQ6KFastWgsl } from "../src/engine/kernels/wgsl";
import {
  KQUANT_FAST_WG, KQUANT_FAST_SUB, kquantFastXsLen,
  pairGemvSiluQ5KFastWorkgroupStorageBytes, gemvQ6KFastWorkgroupStorageBytes,
  KQUANT_FAST_Q5K_PAIR_REL_TOL, KQUANT_FAST_Q6K_REL_TOL,
} from "../src/engine/kquantfast";
import { GLM47_FLASH as G } from "../src/engine/shape";

// FAMIGLIA FAST K-QUANT: struttura del WGSL + pavimento aritmetico f32
// (C3a fase 4b it.13).
//
// Il file fa DUE lavori, e nessuno dei due è surrogabile con l'altro:
//
// 1. STRUTTURA — genera i kernel VERI e ne asserisce le costanti, i byte di
//    shared e l'uso dello stride paddato. Serve perché il pavimento qui sotto è
//    un'emulazione JS: da sola misurerebbe una copia privata, e una mutazione
//    nel WGSL la lascerebbe verde. Le due metà insieme legano il numero al
//    codice che lo produce (stesso pattern di engine-mlasplit.test.ts).
//
// 2. PAVIMENTO — emula in f32 l'aritmetica e il raggruppamento veri dei kernel
//    per stabilire sotto quale errore relativo NESSUN device può scendere. È la
//    derivazione delle tolleranze del ktest, che senza di essa sarebbero numeri
//    ammorbiditi a occhio.
//
// DA DOVE VIENE IL PAVIMENTO — due cause, misurate:
//   (a) FATTORIZZAZIONE. Il dequant Q5_K è w = d·sc·q − dmin·m; il kernel (come
//       il gemello lento) calcola d1·Σq·x − min1·Σx. I due addendi sono grandi
//       e il risultato piccolo: sui dati del ktest il numero di condizione
//       della riga arriva a ~10⁵. NON è colpa dei byte casuali: nasce dalla
//       media non nulla di q ∈ [0,31] contro una x a media zero, e c'è anche
//       sui pesi veri — forzare scale costanti PEGGIORA le cose, non le
//       migliora.
//   (b) CONTRAZIONE FMA. La spec WGSL PERMETTE di fondere a·b+c in una singola
//       operazione arrotondata, ma non lo impone. Due device conformi danno
//       quindi due risultati diversi, e il pavimento va preso come l'INVILUPPO
//       dei due modelli — non come il valore di uno solo. Misurato: per Q6_K la
//       contrazione MIGLIORA di ~5× (2,4e-4 → 4,7e-5) ed è ciò che decide se la
//       vecchia tolleranza 2e-4 regge; per il pair Q5_K PEGGIORA (3,4e-4 →
//       5,9e-4). Su somme che si cancellano l'FMA sposta il rounding, non lo
//       riduce, e la direzione dipende dai dati — motivo per cui la tolleranza
//       si tara sull'estremo peggiore di ciascun kernel, non su un modello solo.
//
// Cosa NON è la causa: l'`exp` del device. L'amplificazione di exp verso silu è
// ≤1 in modulo, quindi anche con 3 ULP di errore contribuisce ≲4e-7 — ordini di
// grandezza sotto il pavimento. Verificato qui sotto invece che supposto.

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
}
function randBytes(n: number, seed: number): Uint8Array {
  const r = lcg(seed);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.floor(r() * 256);
  return out;
}
function randF32(n: number, seed: number): Float32Array {
  const r = lcg(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = r() * 2 - 1;
  return out;
}
// stessa "scala f16 sana" dei ktest (esponente moderato: niente inf/denormali)
function fixScalesAt(src: Uint8Array, blockBytes: number, offs: number[]): void {
  for (let o = 0; o + blockBytes <= src.length; o += blockBytes) {
    for (const f of offs) src[o + f] = 0x2c | (src[o + f] & 0x03);
  }
}
const f = Math.fround;
function f16(lo: number, hi: number): number {
  const h = lo | (hi << 8);
  const s = h >> 15 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
function scaleMinK4(j: number, q: Uint8Array, qo: number): [number, number] {
  if (j < 4) return [q[qo + j] & 63, q[qo + j + 4] & 63];
  return [(q[qo + j + 4] & 0x0f) | ((q[qo + j - 4] >> 6) << 4),
          (q[qo + j + 4] >> 4) | ((q[qo + j] >> 6) << 4)];
}
function s8(v: number): number { return (v << 24) >> 24; }

// Riduzione ad albero su KQUANT_FAST_WG lane, identica al kernel
// (`red[t] += red[t + stride]`, stride 32→1). NON è un albero completo sui
// sottogruppi: i thread accumulano prima in serie (loop strided), e i due
// coincidono solo quando K/32 == 64.
function treeReduce(lane: Float64Array): number {
  const r = lane.slice();
  let stride = KQUANT_FAST_WG >> 1;
  while (stride > 0) {
    for (let i = 0; i < stride; i++) r[i] = f(r[i] + r[i + stride]);
    stride >>= 1;
  }
  return r[0];
}

// `fma` = true modella la contrazione massima concessa dalla spec: il prodotto
// resta esatto (f64 lo rappresenta senza perdita per questi operandi) e si
// arrotonda una volta sola. `fma` = false modella zero contrazione.
const mul = (a: number, b: number, fma: boolean): number => (fma ? a * b : f(a * b));

// Emulazione di pairGemvSiluQ5KFastWgsl per una riga: loop strided su
// sottogruppi da 32 pesi (sb = su>>3, g = su&7 ⇒ j = g>>1, metà = g&1, is = g),
// accumulo ((acc + d1·Σq·x) − min1·Σx) — l'associatività vera del WGSL — poi
// albero su 64 lane.
function gemvQ5KFastF32(src: Uint8Array, r: number, K: number, x: Float32Array, fma: boolean): number {
  const sbPerRow = K / 256, nSub = K / KQUANT_FAST_SUB;
  const lane = new Float64Array(KQUANT_FAST_WG);
  for (let t = 0; t < KQUANT_FAST_WG; t++) {
    let acc = 0;
    for (let su = t; su < nSub; su += KQUANT_FAST_WG) {
      const sb = su >> 3, g = su & 7, j = g >> 1, half = g & 1;
      const o = (r * sbPerRow + sb) * Q5_K_BLOCK_BYTES;
      const d = f16(src[o], src[o + 1]), dmin = f16(src[o + 2], src[o + 3]);
      const [sc, mn] = scaleMinK4(g, src, o + 4);
      const mask = (1 + half) << (2 * j);
      let dotq = 0, sx = 0;
      for (let l = 0; l < 32; l++) {
        const ql = src[o + 48 + j * 32 + l], qh = src[o + 16 + l];
        let q = half === 0 ? ql & 0xf : ql >> 4;
        if (qh & mask) q += 16;
        const xv = x[sb * 256 + j * 64 + half * 32 + l];
        dotq = f(dotq + mul(q, xv, fma));
        sx = f(sx + xv);
      }
      const d1 = f(d * sc), min1 = f(dmin * mn);
      acc = f(f(acc + mul(d1, dotq, fma)) - mul(min1, sx, fma));
    }
    lane[t] = acc;
  }
  return treeReduce(lane);
}

// Emulazione di gemvQ6KFastWgsl: sottogruppo g ⇒ (metà n = g>>2, quarto k =
// g&3); due accumulatori per i due indici di scala, poi acc + d·(sc0·a0 + sc1·a1).
function gemvQ6KFastF32(src: Uint8Array, r: number, K: number, x: Float32Array, fma: boolean): number {
  const sbPerRow = K / 256, nSub = K / KQUANT_FAST_SUB;
  const lane = new Float64Array(KQUANT_FAST_WG);
  for (let t = 0; t < KQUANT_FAST_WG; t++) {
    let acc = 0;
    for (let su = t; su < nSub; su += KQUANT_FAST_WG) {
      const sb = su >> 3, g = su & 7, n = g >> 2, k = g & 3;
      const o = (r * sbPerRow + sb) * Q6_K_BLOCK_BYTES;
      const d = f16(src[o + 208], src[o + 209]);
      const sc0 = s8(src[o + 192 + n * 8 + 2 * k]);
      const sc1 = s8(src[o + 192 + n * 8 + 2 * k + 1]);
      let a0 = 0, a1 = 0;
      for (let l = 0; l < 32; l++) {
        const ql = src[o + n * 64 + (k & 1) * 32 + l], qh = src[o + 128 + n * 32 + l];
        const lo = k < 2 ? ql & 0xf : ql >> 4;
        const q = (lo | (((qh >> (k * 2)) & 3) << 4)) - 32;
        const xv = x[sb * 256 + n * 128 + k * 32 + l];
        if (l < 16) a0 = f(a0 + mul(q, xv, fma)); else a1 = f(a1 + mul(q, xv, fma));
      }
      const inner = f(f(sc0 * a0) + mul(sc1, a1, fma));
      acc = f(acc + mul(d, inner, fma));
    }
    lane[t] = acc;
  }
  return treeReduce(lane);
}

const silu = (gv: number, uv: number): number => f(f(gv / f(1 + f(Math.exp(-gv)))) * uv);

// ---------------------------------------------------------------------------

describe("struttura dei kernel fast K-quant (scansione del WGSL generato)", () => {
  const pair = pairGemvSiluQ5KFastWgsl({ K: G.dModel, N: G.dFfnExpert });
  const q6down = gemvQ6KFastWgsl({ K: G.dFfnExpert, N: G.dModel });
  const q6head = gemvQ6KFastWgsl({ K: G.dModel, N: 1024 });
  const constOf = (src: string, name: string): number => {
    const m = src.match(new RegExp(`const ${name} = (\\d+)u;`));
    if (!m) throw new Error(`costante ${name} non trovata nel WGSL`);
    return Number(m[1]);
  };
  const wgBytes = (src: string): number => {
    let bytes = 0;
    for (const m of src.matchAll(/var<workgroup>\s+\w+:\s*array<f32,\s*(\d+)>/g)) bytes += 4 * Number(m[1]);
    for (const m of src.matchAll(/var<workgroup>\s+\w+:\s*f32\s*;/g)) { void m; bytes += 4; }
    return bytes;
  };

  it("le costanti del WGSL sono quelle del modulo di sizing", () => {
    for (const [src, K] of [[pair, G.dModel], [q6down, G.dFfnExpert], [q6head, G.dModel]] as const) {
      expect(constOf(src, "K")).toBe(K);
      expect(constOf(src, "SB_PER_ROW")).toBe(K / 256);
      expect(constOf(src, "N_SUB")).toBe(K / KQUANT_FAST_SUB);
      expect(src).toMatch(new RegExp(`@workgroup_size\\(${KQUANT_FAST_WG}\\)`));
    }
  });

  it("i byte di shared sono quelli che glmmodel confronta col limite del device", () => {
    expect(wgBytes(pair)).toBe(pairGemvSiluQ5KFastWorkgroupStorageBytes(G.dModel));
    expect(wgBytes(q6down)).toBe(gemvQ6KFastWorkgroupStorageBytes(G.dFfnExpert));
    expect(wgBytes(q6head)).toBe(gemvQ6KFastWorkgroupStorageBytes(G.dModel));
    // e CRESCONO con K: è la ragione per cui entrano nella guardia invece di
    // essere dati per scontati come lo split MLA (costante in ctxMax)
    expect(wgBytes(q6head)).toBeGreaterThan(wgBytes(q6down));
    expect(wgBytes(pair)).toBe(4 * (kquantFastXsLen(G.dModel) + 2 * KQUANT_FAST_WG));
  });

  it("lo stride paddato di x è nel kernel, non solo nel commento", () => {
    // senza `+ (e >> 5)` i 32 thread di un warp leggono tutti lo stesso banco
    for (const [src, K] of [[pair, G.dModel], [q6down, G.dFfnExpert], [q6head, G.dModel]] as const) {
      expect(src).toMatch(/xs\[i \+ \(i >> 5u\)\] = x\[i\]/);   // scrittura
      expect(src).toMatch(/xs\[e \+ \(e >> 5u\)\]/);            // lettura
      expect(src).toContain(`array<f32, ${kquantFastXsLen(K)}>`);
    }
  });
});

describe("pavimento f32: pair gemv Q5_K + silu·mul", () => {
  // stessi seed, stesse shape e stessa preparazione dei byte del caso ktest
  const K = G.dModel, N = G.dFfnExpert;
  const nBlocks = (K / 256) * N;
  const gSrc = randBytes(nBlocks * Q5_K_BLOCK_BYTES, 7701); fixScalesAt(gSrc, Q5_K_BLOCK_BYTES, [1, 3]);
  const uSrc = randBytes(nBlocks * Q5_K_BLOCK_BYTES, 7702); fixScalesAt(uSrc, Q5_K_BLOCK_BYTES, [1, 3]);
  const gw = new Float32Array(nBlocks * 256), uw = new Float32Array(nBlocks * 256);
  dequantQ5_K(gSrc, 0, nBlocks, gw);
  dequantQ5_K(uSrc, 0, nBlocks, uw);
  const x = randF32(K, 7703);

  let cond = 0, relNoFma = 0, relFma = 0, relExpPerturb = 0;
  for (let r = 0; r < N; r++) {
    let ag = 0, au = 0, absSum = 0;
    for (let i = 0; i < K; i++) {
      ag += gw[r * K + i] * x[i];
      au += uw[r * K + i] * x[i];
      absSum += Math.abs(gw[r * K + i] * x[i]);
    }
    cond = Math.max(cond, absSum / Math.max(Math.abs(ag), 1e-30));
    const ref = (ag / (1 + Math.exp(-ag))) * au;
    const den = Math.max(Math.abs(ref), 1e-6);
    for (const fma of [false, true]) {
      const gf = gemvQ5KFastF32(gSrc, r, K, x, fma), uf = gemvQ5KFastF32(uSrc, r, K, x, fma);
      const rel = Math.abs(silu(gf, uf) - ref) / den;
      if (fma) relFma = Math.max(relFma, rel); else relNoFma = Math.max(relNoFma, rel);
    }
    // sensibilità dell'uscita a un errore di 3 ULP sull'exp del device
    const e = Math.exp(-ag), eP = e * (1 + 3 * 2 ** -24);
    const sPert = (ag / (1 + eP)) * au;
    relExpPerturb = Math.max(relExpPerturb, Math.abs(sPert - ref) / den);
  }

  it("il condizionamento è ~10⁵, non ~10³", () => {
    expect(cond).toBeGreaterThan(1e4);
    expect(cond).toBeLessThan(1e7);
  });

  it("il pavimento sfonda i 2e-4 storici in ENTRAMBI i modelli di contrazione", () => {
    expect(relNoFma).toBeGreaterThan(2e-4);
    expect(relFma).toBeGreaterThan(2e-4);
    // ed entrambi stanno sotto la tolleranza scelta
    expect(relNoFma).toBeLessThan(KQUANT_FAST_Q5K_PAIR_REL_TOL);
    expect(relFma).toBeLessThan(KQUANT_FAST_Q5K_PAIR_REL_TOL);
  });

  it("l'exp del device NON è la causa: amplificazione ≤1 verso silu", () => {
    // 3 ULP su exp ⇒ ≲4e-7 sull'uscita, ordini di grandezza sotto il pavimento
    expect(relExpPerturb).toBeLessThan(1e-5);
    expect(relExpPerturb).toBeLessThan(relNoFma / 100);
  });

  it("il modello CON FMA riproduce la misura del device", () => {
    // Non "stesso ordine di grandezza": stesso NUMERO. Il ktest sul 4090
    // Laptop (Chrome/Dawn/Vulkan, 2026-08-03) misura maxRel 5,869e-4 e
    // l'emulazione a contrazione massima dà lo stesso valore a quattro cifre.
    // E' la prova che il pavimento è capito, non stimato: quel device fonde
    // ogni a·b+c, e il resto dell'aritmetica dell'emulazione (riduzione ad
    // albero, associatività ((acc + d1·dotq) − min1·sx)) coincide col kernel.
    expect(Math.abs(relFma - 5.869e-4) / 5.869e-4).toBeLessThan(0.01);
  });

  it("la tolleranza copre la misura del DEVICE con margine", () => {
    const DEVICE = 5.869e-4;
    expect(KQUANT_FAST_Q5K_PAIR_REL_TOL / DEVICE).toBeGreaterThan(1.5);
    // e resta ordini di grandezza sotto un bug strutturale: un sottogruppo su
    // 64 mal assegnato sposta di ~1/64 = 1,6e-2
    expect(KQUANT_FAST_Q5K_PAIR_REL_TOL).toBeLessThan(1 / 64 / 10);
  });
});

describe("pavimento f32: gemv Q6_K fast", () => {
  // il caso fragile è il down dello shexp: K=1536 ⇒ 48 sottogruppi su 64 lane
  const K = G.dFfnExpert, N = G.dModel;
  const nBlocks = (K / 256) * N;
  const src = randBytes(nBlocks * Q6_K_BLOCK_BYTES, 7800 + K + N);
  fixScalesAt(src, Q6_K_BLOCK_BYTES, [209]);
  const w = new Float32Array(nBlocks * 256);
  dequantQ6_K(src, 0, nBlocks, w);
  const x = randF32(K, 7810 + K);

  let relNoFma = 0, relFma = 0;
  for (let r = 0; r < N; r++) {
    let ref = 0;
    for (let i = 0; i < K; i++) ref += w[r * K + i] * x[i];
    const den = Math.max(Math.abs(ref), 1e-6);
    relNoFma = Math.max(relNoFma, Math.abs(gemvQ6KFastF32(src, r, K, x, false) - ref) / den);
    relFma = Math.max(relFma, Math.abs(gemvQ6KFastF32(src, r, K, x, true) - ref) / den);
  }

  it("la contrazione FMA vale un fattore ~5 su questo kernel", () => {
    // misurato: 2,413e-4 senza contrazione contro 4,711e-5 con. Al contrario
    // del Q5_K pair, dove la contrazione PEGGIORA il caso peggiore (3,4e-4 →
    // 5,9e-4): su somme che si cancellano l'FMA sposta il rounding, non lo
    // riduce, e la direzione dipende dai dati.
    expect(relFma).toBeLessThan(relNoFma / 3);
  });

  it("SENZA FMA il pavimento sfonda i 2e-4: la vecchia tolleranza era fragile", () => {
    // è il finding che motiva il cambio: a 2e-4 un device conforme che NON
    // fonde darebbe FAIL su un kernel corretto — il ktest passava solo perché
    // il device di sviluppo fonde
    expect(relNoFma).toBeGreaterThan(2e-4);
  });

  it("la tolleranza nuova sta ≥2× sopra il pavimento no-FMA (caso peggiore lecito)", () => {
    expect(KQUANT_FAST_Q6K_REL_TOL / relNoFma).toBeGreaterThan(2);
    expect(KQUANT_FAST_Q6K_REL_TOL).toBeLessThan(1 / 64 / 10);
  });
});
