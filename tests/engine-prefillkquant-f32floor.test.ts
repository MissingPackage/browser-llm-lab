import { describe, it, expect } from "vitest";
import { dequantQ5_K, f16ToF32, Q5_K_BLOCK_BYTES } from "../src/engine/quant";
import {
  prefillGemmQ5KSplitKIdotWgsl, prefillGemmQ5KSplitKWgsl,
  prefillQuantXQ8Wgsl, prefillSplitKCombineWgsl, prefillGemmSplitsFor,
} from "../src/engine/kernels/wgsl";
import {
  PREFILL_Q5K_KTEST_CASE,
  PREFILL_GEMM_Q5K_IDOT_REL_TOL, PREFILL_GEMM_Q5K_IDOT_ABS_TOL,
  PREFILL_GEMM_Q5K_F32_REL_TOL, PREFILL_GEMM_Q5K_F32_ABS_TOL,
} from "../src/engine/prefillkquant";

// PREFILL GEMM q5_K: struttura del WGSL + pavimento aritmetico f32 delle DUE
// vie (intera `dot4I8Packed` e fallback f32).
//
// Gemello di tests/engine-kquant-f32floor.test.ts, sull'altra riga: li' i GEMV
// fast della famiglia GLM, qui il moltiplicatore split-K del prefill. Stesso
// impianto a due meta', e nessuna delle due surroga l'altra:
//
// 1. STRUTTURA — genera i kernel VERI su `PREFILL_Q5K_KTEST_CASE` e ne asserisce
//    le costanti e gli array di workgroup. Serve perche' il pavimento qui sotto
//    e' un'emulazione JS: da sola misurerebbe una copia privata, e una mutazione
//    del WGSL la lascerebbe verde. Le due meta' insieme legano il numero al
//    codice che lo produce.
//
// 2. PAVIMENTO — emula in f32 l'aritmetica e il raggruppamento veri (per
//    superblocco, per gruppo, `d1*Sigma(q*x) - min1*Sigma(x)`, accumulo per
//    fetta e somma delle 4 fette come fa `prefillSplitKCombineWgsl`) e misura
//    sotto quale errore NESSUN device conforme puo' scendere. E' la derivazione
//    delle quattro tolleranze del ktest, che senza di essa sarebbero numeri
//    ammorbiditi a occhio.
//
// DUE MODELLI, NON UNO. La spec WGSL PERMETTE di contrarre `a*b + c` in una sola
// operazione arrotondata ma non lo impone: due device conformi danno due
// risultati diversi, e il pavimento e' l'INVILUPPO dei due (`floor = max(conFMA,
// senzaFMA)`), non il valore misurato su una macchina sola. E' la lezione gia'
// pagata sul Q6_K fast, dove la tolleranza a 2e-4 reggeva SOLO perche' il device
// di sviluppo fondeva.
//
// PERCHE' UN PAVIMENTO C'E' AFFATTO: il dequant Q5_K e' `w = d*sc*q - dmin*m`,
// mentre il kernel (per non dequantizzare nel ciclo interno) calcola
// `d1*Sigma(q*x) - min1*Sigma(x)`. I due addendi sono grandi e la loro
// differenza e' piccola: la fattorizzazione, non i byte casuali, e' cio' che
// produce l'errore. La META' (c) mostra che la tolleranza cosi' derivata resta
// comunque strettissima rispetto a un difetto strutturale.

const C = PREFILL_Q5K_KTEST_CASE;
const { K, N, M, splits } = C;
const SBPR = K / 256;          // superblocchi per riga di pesi
const PER = SBPR / splits;     // superblocchi per fetta
const BPR = K / 32;            // blocchi di attivazione per riga di chunk
const OPTS = { kind: "q5_K", K, N, M, splits } as const;

// ---------------------------------------------------------------------------
// Dati del caso: le stesse convenzioni del ktest, in una sede sola
// (`PREFILL_Q5K_KTEST_CASE`), cosi' pavimento e ktest non descrivono due
// esperimenti diversi.
// ---------------------------------------------------------------------------

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
// stessa "scala f16 sana" di testGemvC2/fixScalesAt (esponente moderato:
// niente inf, niente denormali)
function fixScalesAt(s: Uint8Array, blockBytes: number, offs: number[]): void {
  for (let o = 0; o + blockBytes <= s.length; o += blockBytes) {
    for (const f2 of offs) s[o + f2] = 0x2c | (s[o + f2] & 0x03);
  }
}
function scaleMinK4(j: number, q: Uint8Array, qo: number): [number, number] {
  if (j < 4) return [q[qo + j] & 63, q[qo + j + 4] & 63];
  return [(q[qo + j + 4] & 0x0f) | ((q[qo + j - 4] >> 6) << 4),
          (q[qo + j + 4] >> 4) | ((q[qo + j] >> 6) << 4)];
}
const f = Math.fround;
// `round` del WGSL: al piu' vicino, meta' esatta al PARI (non `Math.round`, che
// arrotonda le meta' verso +inf).
function roundTiesEven(v: number): number {
  const r = Math.round(v);   // sulle meta' esatte `Math.round` da' sempre ceil(v)
  return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

const nBlocks = SBPR * N;
const src = randBytes(nBlocks * Q5_K_BLOCK_BYTES, C.seedBlocks);
fixScalesAt(src, Q5_K_BLOCK_BYTES, [1, 3]);
const w = new Float32Array(nBlocks * 256);
dequantQ5_K(src, 0, nBlocks, w);
const x = randF32(M * K, C.seedX);

// Emulazione di `prefillQuantXQ8Wgsl`: amax/127 per blocco da 32, byte con
// segno. Il RIFERIMENTO del braccio idot si calcola su QUESTE attivazioni: il
// caso misura l'aritmetica del kernel, non la perdita del quantizzatore.
const xq = new Int8Array(M * K);
const xsc = new Float32Array(M * BPR);
const xsum = new Float32Array(M * BPR);   // f32(Sigma q) * sc, come `xsum` nel kernel
const xsx = new Float32Array(M * BPR);    // Sigma x in f32, come `sx` nella via f32
const xdq = new Float64Array(M * K);      // le attivazioni dopo il giro i8
for (let b = 0; b < M * BPR; b++) {
  let amax = 0;
  for (let i = 0; i < 32; i++) amax = Math.max(amax, Math.abs(x[b * 32 + i]));
  const sc = f(amax / 127);
  const inv = sc > 0 ? f(1 / sc) : 0;
  xsc[b] = sc;
  let sq = 0, sx = 0;
  for (let i = 0; i < 32; i++) {
    const q = Math.min(127, Math.max(-127, roundTiesEven(f(x[b * 32 + i] * inv))));
    xq[b * 32 + i] = q;
    xdq[b * 32 + i] = q * sc;
    sq += q;
    sx = f(sx + x[b * 32 + i]);
  }
  xsum[b] = f(sq * sc);
  xsx[b] = sx;
}

// Riferimenti f64 dal dequant esatto, nel layout d'uscita del kernel: y[m*N + r].
const refIdot = new Float64Array(M * N);
const refF32 = new Float64Array(M * N);
for (let r = 0; r < N; r++) {
  for (let m = 0; m < M; m++) {
    let ai = 0, af = 0;
    for (let i = 0; i < K; i++) {
      const wi = w[r * K + i];
      ai += wi * xdq[m * K + i];
      af += wi * x[m * K + i];
    }
    refIdot[m * N + r] = ai;
    refF32[m * N + r] = af;
  }
}

// ---------------------------------------------------------------------------
// Le due emulazioni f32. `fma` = true modella la contrazione MASSIMA concessa
// dalla spec (il prodotto resta esatto — f64 lo rappresenta senza perdita per
// questi operandi — e si arrotonda una volta sola); `fma` = false modella zero
// contrazione. `dropMin` e' la MUTAZIONE della meta' (c): omette il termine
// `- dmin*Sigma(x)`, cioe' proprio cio' che distingue un K-quant con minimo da
// un q4_0.
// ---------------------------------------------------------------------------

const term = (a: number, b: number, fma: boolean): number => (fma ? a * b : f(a * b));

// Somma delle fette: `prefillSplitKCombineWgsl`, in f32 e in ordine s = 0..S-1.
function combine(part: Float32Array): Float32Array {
  const y = new Float32Array(M);
  for (let m = 0; m < M; m++) {
    let v = 0;
    for (let s = 0; s < splits; s++) v = f(v + part[s * M + m]);
    y[m] = v;
  }
  return y;
}

const qlo = new Int32Array(32), qhi = new Int32Array(32);

// `prefillGemmQ5KSplitKIdotWgsl`, una riga di pesi r: per superblocco, per
// gruppo g (due sotto-blocchi is = 2g e is+1), prodotto scalare INTERO esatto e
// poi `acc + d1*i1*sc_x - min1*xsum + d2*i2*sc_x - min2*xsum` associato a
// sinistra, come lo scrive il WGSL.
function emulateIdotRow(r: number, fma: boolean, dropMin: boolean): Float32Array {
  const part = new Float32Array(splits * M);
  const acc = new Float64Array(M);
  for (let s = 0; s < splits; s++) {
    acc.fill(0);
    for (let sb = s * PER; sb < s * PER + PER; sb++) {
      const o = (r * SBPR + sb) * Q5_K_BLOCK_BYTES;
      const d = f16ToF32(src[o] | (src[o + 1] << 8));
      const dmin = f16ToF32(src[o + 2] | (src[o + 3] << 8));
      for (let g = 0; g < 4; g++) {
        const is = 2 * g;
        const [sc1, mn1] = scaleMinK4(is, src, o + 4);
        const [sc2, mn2] = scaleMinK4(is + 1, src, o + 4);
        const d1 = f(d * sc1), min1 = f(dmin * mn1);
        const d2 = f(d * sc2), min2 = f(dmin * mn2);
        for (let l = 0; l < 32; l++) {
          const ql = src[o + 48 + g * 32 + l], qh = src[o + 16 + l];
          qlo[l] = (ql & 0x0f) + (((qh >> is) & 1) << 4);
          qhi[l] = (ql >> 4) + (((qh >> (is + 1)) & 1) << 4);
        }
        for (let m = 0; m < M; m++) {
          const b1 = m * BPR + sb * 8 + is, b2 = b1 + 1;
          let i1 = 0, i2 = 0;
          for (let l = 0; l < 32; l++) {
            i1 += qlo[l] * xq[b1 * 32 + l];
            i2 += qhi[l] * xq[b2 * 32 + l];
          }
          let a = acc[m];
          a = f(a + term(f(d1 * i1), xsc[b1], fma));
          if (!dropMin) a = f(a - term(min1, xsum[b1], fma));
          a = f(a + term(f(d2 * i2), xsc[b2], fma));
          if (!dropMin) a = f(a - term(min2, xsum[b2], fma));
          acc[m] = a;
        }
      }
    }
    for (let m = 0; m < M; m++) part[s * M + m] = acc[m];
  }
  return combine(part);
}

const qf = new Float64Array(32);

// `prefillGemmQ5KSplitKWgsl`: UN sotto-blocco da 32 per volta, `qx` e `sx`
// accumulati in f32 nell'ordine del loop, poi `acc + dsc*qx - dmn*sx`.
function emulateF32Row(r: number, fma: boolean, dropMin: boolean): Float32Array {
  const part = new Float32Array(splits * M);
  const acc = new Float64Array(M);
  for (let s = 0; s < splits; s++) {
    acc.fill(0);
    for (let sb = s * PER; sb < s * PER + PER; sb++) {
      const o = (r * SBPR + sb) * Q5_K_BLOCK_BYTES;
      const d = f16ToF32(src[o] | (src[o + 1] << 8));
      const dmin = f16ToF32(src[o + 2] | (src[o + 3] << 8));
      for (let is = 0; is < 8; is++) {
        const g = is >> 1, hiHalf = (is & 1) === 1;
        const [sc, mn] = scaleMinK4(is, src, o + 4);
        const dsc = f(d * sc), dmn = f(dmin * mn);
        for (let l = 0; l < 32; l++) {
          const ql = src[o + 48 + g * 32 + l], qh = src[o + 16 + l];
          qf[l] = (hiHalf ? ql >> 4 : ql & 0x0f) + ((qh & (1 << is)) !== 0 ? 16 : 0);
        }
        for (let m = 0; m < M; m++) {
          const base = m * K + sb * 256 + is * 32;
          let qx = 0;
          for (let l = 0; l < 32; l++) qx = f(qx + term(qf[l], x[base + l], fma));
          let a = acc[m];
          a = f(a + term(dsc, qx, fma));
          if (!dropMin) a = f(a - term(dmn, xsx[m * BPR + sb * 8 + is], fma));
          acc[m] = a;
        }
      }
    }
    for (let m = 0; m < M; m++) part[s * M + m] = acc[m];
  }
  return combine(part);
}

// maxAbs/maxRel con la STESSA definizione di `compare` nel ktest worker
// (denominatore `max(|ref|, 1e-6)`): il numero che esce di qui e' il numero che
// il ktest confrontera' con la tolleranza, non un suo parente.
interface Err { abs: number; rel: number }
function measure(
  emul: (r: number, fma: boolean, dropMin: boolean) => Float32Array,
  ref: Float64Array, fma: boolean, dropMin: boolean,
): Err {
  let abs = 0, rel = 0;
  for (let r = 0; r < N; r++) {
    const y = emul(r, fma, dropMin);
    for (let m = 0; m < M; m++) {
      const e = ref[m * N + r], dd = Math.abs(y[m] - e);
      abs = Math.max(abs, dd);
      rel = Math.max(rel, dd / Math.max(Math.abs(e), 1e-6));
    }
  }
  return { abs, rel };
}

const idotNoFma = measure(emulateIdotRow, refIdot, false, false);
const idotFma = measure(emulateIdotRow, refIdot, true, false);
const idotMut = measure(emulateIdotRow, refIdot, true, true);
const f32NoFma = measure(emulateF32Row, refF32, false, false);
const f32Fma = measure(emulateF32Row, refF32, true, false);
const f32Mut = measure(emulateF32Row, refF32, true, true);

const floorIdotRel = Math.max(idotNoFma.rel, idotFma.rel);
const floorIdotAbs = Math.max(idotNoFma.abs, idotFma.abs);
const floorF32Rel = Math.max(f32NoFma.rel, f32Fma.rel);
const floorF32Abs = Math.max(f32NoFma.abs, f32Fma.abs);

const e = (v: number): string => v.toExponential(3);

// I DUE PAVIMENTI PER BRACCIO, STAMPATI: sono la derivazione che i commenti di
// prefillkquant.ts citano, e vanno letti anche quando il file e' verde (e' cosi'
// che ci si accorge che un pavimento si e' mosso prima che sfondi il 20x). A
// livello di modulo e non dentro un `it`, perche' il reporter non mostra lo
// stdout dei test che passano.
/* eslint-disable no-console */
console.log(`[idot] senzaFMA rel=${e(idotNoFma.rel)} abs=${e(idotNoFma.abs)} | ` +
  `conFMA rel=${e(idotFma.rel)} abs=${e(idotFma.abs)} | ` +
  `floor rel=${e(floorIdotRel)} abs=${e(floorIdotAbs)} | ` +
  `tol rel=${e(PREFILL_GEMM_Q5K_IDOT_REL_TOL)} (${(PREFILL_GEMM_Q5K_IDOT_REL_TOL / floorIdotRel).toFixed(1)}x) ` +
  `abs=${e(PREFILL_GEMM_Q5K_IDOT_ABS_TOL)} (${(PREFILL_GEMM_Q5K_IDOT_ABS_TOL / floorIdotAbs).toFixed(1)}x)`);
console.log(`[f32]  senzaFMA rel=${e(f32NoFma.rel)} abs=${e(f32NoFma.abs)} | ` +
  `conFMA rel=${e(f32Fma.rel)} abs=${e(f32Fma.abs)} | ` +
  `floor rel=${e(floorF32Rel)} abs=${e(floorF32Abs)} | ` +
  `tol rel=${e(PREFILL_GEMM_Q5K_F32_REL_TOL)} (${(PREFILL_GEMM_Q5K_F32_REL_TOL / floorF32Rel).toFixed(1)}x) ` +
  `abs=${e(PREFILL_GEMM_Q5K_F32_ABS_TOL)} (${(PREFILL_GEMM_Q5K_F32_ABS_TOL / floorF32Abs).toFixed(1)}x)`);
console.log(`[MUTATA senza -dmin*Sigma x] idot rel=${e(idotMut.rel)} abs=${e(idotMut.abs)} | ` +
  `f32 rel=${e(f32Mut.rel)} abs=${e(f32Mut.abs)}`);
/* eslint-enable no-console */

// ---------------------------------------------------------------------------

describe("prefill gemm q5_K: struttura del WGSL generato", () => {
  const idot = prefillGemmQ5KSplitKIdotWgsl(OPTS);
  const f32k = prefillGemmQ5KSplitKWgsl(OPTS);

  it("il caso del ktest e' la geometria che il piano sceglie da solo", () => {
    // se `prefillGemmSplitsFor` cambiasse idea sulle fette, il caso non
    // descriverebbe piu' la forma che gira in produzione
    expect(prefillGemmSplitsFor(K, N, "q5_K")).toBe(splits);
    expect(K % 256).toBe(0);
    // N%64 != 0 e' VOLUTO: e' cio' che esercita la guardia `r < N_ROWS`
    expect(N % 64).not.toBe(0);
    expect(idot).toContain(`const N_ROWS = ${N}u;`);
    expect(idot).toContain("if (r < N_ROWS) {");
    expect(f32k).toContain("if (r < N_ROWS) {");
  });

  it("le costanti dei due kernel sono quelle che l'emulazione presume", () => {
    for (const s of [idot, f32k]) {
      expect(s).toContain("const WORDS = 44u;");   // 176 B di superblocco q5_K
      expect(s).toContain("const SBPR = 16u;");    // K=4096 / 256
      expect(s).toContain("const PER = 4u;");      // 16 superblocchi / 4 fette
      expect(s).toContain(`const M_ROWS = ${M}u;`);
    }
    expect(SBPR).toBe(16);
    expect(PER).toBe(4);
  });

  it("gli array di workgroup sono quelli delle due vie", () => {
    // via intera: il superblocco INTERO di attivazioni i8 (M x 8 x 8 parole) +
    // le due riduzioni per (riga di chunk, sotto-blocco)
    expect(idot).toContain("var<workgroup> xs: array<u32, 1024>;");
    expect(idot).toContain("var<workgroup> xss: array<f32, 128>;");
    expect(idot).toContain("var<workgroup> xsum: array<f32, 128>;");
    // via f32: UN sotto-blocco da 32 per volta (M x 32 f32), non il superblocco
    expect(f32k).toContain("var<workgroup> xs: array<f32, 512>;");
    expect(f32k).not.toContain("array<u32,");
  });

  it("quantizzatore e combine emulati sono quelli veri", () => {
    // l'emulazione del quantizzatore presume `amax/127` su blocchi da 32 e
    // M*BPR blocchi; quella della combine, la somma delle 4 fette in f32
    expect(prefillQuantXQ8Wgsl({ K, M })).toContain(`const BLOCKS = ${M * BPR}u;`);
    expect(prefillQuantXQ8Wgsl({ K, M })).toContain("let sc = amax / 127.0;");
    expect(prefillSplitKCombineWgsl({ N, M, splits })).toContain(`const S = ${splits}u;`);
    expect(prefillSplitKCombineWgsl({ N, M, splits })).toContain(`const TOTAL = ${M * N}u;`);
    expect(prefillSplitKCombineWgsl({ N, M, splits }))
      .toContain("for (var s = 0u; s < S; s = s + 1u) { v = v + part[s * TOTAL + i]; }");
  });
});

describe("pavimento f32: prefill gemm q5_K via intera (dot4I8Packed)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    // nessuno dei due e' zero (l'emulazione fa aritmetica f32 davvero) e
    // nessuno dei due e' trascurabile rispetto all'altro: e' per questo che il
    // pavimento e' il LORO MASSIMO e non il valore di un modello solo
    for (const v of [idotNoFma.rel, idotFma.rel, idotNoFma.abs, idotFma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorIdotRel).toBe(Math.max(idotNoFma.rel, idotFma.rel));
    expect(floorIdotAbs).toBe(Math.max(idotNoFma.abs, idotFma.abs));
    // e nessuno dei due domina l'altro di un ordine di grandezza: sono lo stesso
    // fenomeno arrotondato in due modi leciti, non due regimi diversi
    expect(idotNoFma.rel / idotFma.rel).toBeLessThan(10);
    expect(idotFma.rel / idotNoFma.rel).toBeLessThan(10);
    expect(idotNoFma.abs / idotFma.abs).toBeLessThan(10);
    expect(idotFma.abs / idotNoFma.abs).toBeLessThan(10);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q5K_IDOT_REL_TOL).toBeGreaterThanOrEqual(floorIdotRel);
    expect(PREFILL_GEMM_Q5K_IDOT_REL_TOL).toBeLessThanOrEqual(20 * floorIdotRel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q5K_IDOT_ABS_TOL).toBeGreaterThanOrEqual(floorIdotAbs);
    expect(PREFILL_GEMM_Q5K_IDOT_ABS_TOL).toBeLessThanOrEqual(20 * floorIdotAbs);
  });
});

describe("pavimento f32: prefill gemm q5_K via f32 (fallback)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [f32NoFma.rel, f32Fma.rel, f32NoFma.abs, f32Fma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorF32Rel).toBe(Math.max(f32NoFma.rel, f32Fma.rel));
    expect(floorF32Abs).toBe(Math.max(f32NoFma.abs, f32Fma.abs));
    expect(f32NoFma.rel / f32Fma.rel).toBeLessThan(10);
    expect(f32Fma.rel / f32NoFma.rel).toBeLessThan(10);
    // SU QUESTA VIA I DUE MODELLI SI SCAMBIANO IL POSTO fra le due metriche: sul
    // REL vince il senza-FMA, sull'ABS il con-FMA. E' la prova concreta che «il
    // device di sviluppo fonde» non basta a tarare una tolleranza — servono
    // entrambi i modelli, e su entrambe le metriche.
    expect(f32NoFma.rel).toBeGreaterThan(f32Fma.rel);
    expect(f32Fma.abs).toBeGreaterThan(f32NoFma.abs);
    // e il fallback f32 sta SOPRA la via intera: il prodotto q*x qui e' in
    // virgola mobile per ogni elemento, li' e' un i32 esatto
    expect(floorF32Rel).toBeGreaterThan(floorIdotRel);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q5K_F32_REL_TOL).toBeGreaterThanOrEqual(floorF32Rel);
    expect(PREFILL_GEMM_Q5K_F32_REL_TOL).toBeLessThanOrEqual(20 * floorF32Rel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q5K_F32_ABS_TOL).toBeGreaterThanOrEqual(floorF32Abs);
    expect(PREFILL_GEMM_Q5K_F32_ABS_TOL).toBeLessThanOrEqual(20 * floorF32Abs);
  });
});

describe("discriminante: la tolleranza non copre un difetto strutturale", () => {
  // Il ktest passa se `maxAbs <= absTol OPPURE maxRel <= relTol` (la `compare`
  // del worker): perche' la mutazione sia CATTURATA deve sfondare ENTRAMBE le
  // tolleranze, ed e' questo che si asserisce.
  it("omettere -dmin*Sigma(x) sfonda le tolleranze della via intera di 10x", () => {
    expect(idotMut.rel / PREFILL_GEMM_Q5K_IDOT_REL_TOL).toBeGreaterThan(10);
    expect(idotMut.abs / PREFILL_GEMM_Q5K_IDOT_ABS_TOL).toBeGreaterThan(10);
  });

  it("omettere -dmin*Sigma(x) sfonda le tolleranze della via f32 di 10x", () => {
    expect(f32Mut.rel / PREFILL_GEMM_Q5K_F32_REL_TOL).toBeGreaterThan(10);
    expect(f32Mut.abs / PREFILL_GEMM_Q5K_F32_ABS_TOL).toBeGreaterThan(10);
  });
});
