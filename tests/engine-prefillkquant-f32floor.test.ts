import { describe, it, expect } from "vitest";
import { writeSync } from "node:fs";
import {
  dequantQ5_K, f16ToF32, Q5_K_BLOCK_BYTES,
  dequantQ4_1, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_WEIGHTS,
  dequantQ4_K, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_WEIGHTS,
  dequantQ6_K, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_WEIGHTS,
  dequantQ8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_WEIGHTS,
} from "../src/engine/quant";
import {
  prefillGemmQ5KSplitKIdotWgsl, prefillGemmQ5KSplitKWgsl,
  prefillGemmQ41SplitKIdotWgsl, prefillGemmQ41SplitKWgsl,
  prefillGemmQ4KSplitKIdotWgsl, prefillGemmQ4KSplitKWgsl,
  prefillGemmQ6KSplitKIdotWgsl, prefillGemmQ6KSplitKWgsl,
  prefillGemmQ80SplitKIdotWgsl, prefillGemmQ80SplitKWgsl,
  prefillQuantXQ8Wgsl, prefillSplitKCombineWgsl, prefillGemmSplitsFor,
} from "../src/engine/kernels/wgsl";
import {
  PREFILL_Q5K_KTEST_CASE,
  PREFILL_GEMM_Q5K_IDOT_REL_TOL, PREFILL_GEMM_Q5K_IDOT_ABS_TOL,
  PREFILL_GEMM_Q5K_F32_REL_TOL, PREFILL_GEMM_Q5K_F32_ABS_TOL,
  PREFILL_Q41_KTEST_CASE,
  PREFILL_GEMM_Q41_IDOT_REL_TOL, PREFILL_GEMM_Q41_IDOT_ABS_TOL,
  PREFILL_GEMM_Q41_F32_REL_TOL, PREFILL_GEMM_Q41_F32_ABS_TOL,
  PREFILL_Q4K_KTEST_CASE,
  PREFILL_GEMM_Q4K_IDOT_REL_TOL, PREFILL_GEMM_Q4K_IDOT_ABS_TOL,
  PREFILL_GEMM_Q4K_F32_REL_TOL, PREFILL_GEMM_Q4K_F32_ABS_TOL,
  PREFILL_Q6K_KTEST_CASE,
  PREFILL_GEMM_Q6K_IDOT_REL_TOL, PREFILL_GEMM_Q6K_IDOT_ABS_TOL,
  PREFILL_GEMM_Q6K_F32_REL_TOL, PREFILL_GEMM_Q6K_F32_ABS_TOL,
  PREFILL_Q80_KTEST_CASE,
  PREFILL_GEMM_Q80_IDOT_REL_TOL, PREFILL_GEMM_Q80_IDOT_ABS_TOL,
  PREFILL_GEMM_Q80_F32_REL_TOL, PREFILL_GEMM_Q80_F32_ABS_TOL,
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
// che ci si accorge che un pavimento si e' mosso prima che sfondi il 20x).
//
// SCRITTURA DIRETTA SU fd 1, non `console.log`. Vitest 4 sceglie da solo il
// reporter `agent` quando gira dentro una sessione di coding agent (`isAgent`
// di std-env: CLAUDECODE/AI_AGENT/...), e quel reporter e' `silent:
// "passed-only"` — cioe' butta via lo stdout dei file che PASSANO, che e'
// esattamente il caso in cui questi numeri servono. Spostare i log a livello di
// modulo invece che dentro un `it` non basta: l'intercettazione e' sulla
// `console`, non sul task. `writeSync(1, ...)` scavalca l'intercettazione e
// arriva allo stdout vero del processo, e la derivazione resta leggibile
// ovunque il gate giri.
const say = (line: string): void => { writeSync(1, `${line}\n`); };

say(`[idot] senzaFMA rel=${e(idotNoFma.rel)} abs=${e(idotNoFma.abs)} | ` +
  `conFMA rel=${e(idotFma.rel)} abs=${e(idotFma.abs)} | ` +
  `floor rel=${e(floorIdotRel)} abs=${e(floorIdotAbs)} | ` +
  `tol rel=${e(PREFILL_GEMM_Q5K_IDOT_REL_TOL)} (${(PREFILL_GEMM_Q5K_IDOT_REL_TOL / floorIdotRel).toFixed(1)}x) ` +
  `abs=${e(PREFILL_GEMM_Q5K_IDOT_ABS_TOL)} (${(PREFILL_GEMM_Q5K_IDOT_ABS_TOL / floorIdotAbs).toFixed(1)}x)`);
say(`[f32]  senzaFMA rel=${e(f32NoFma.rel)} abs=${e(f32NoFma.abs)} | ` +
  `conFMA rel=${e(f32Fma.rel)} abs=${e(f32Fma.abs)} | ` +
  `floor rel=${e(floorF32Rel)} abs=${e(floorF32Abs)} | ` +
  `tol rel=${e(PREFILL_GEMM_Q5K_F32_REL_TOL)} (${(PREFILL_GEMM_Q5K_F32_REL_TOL / floorF32Rel).toFixed(1)}x) ` +
  `abs=${e(PREFILL_GEMM_Q5K_F32_ABS_TOL)} (${(PREFILL_GEMM_Q5K_F32_ABS_TOL / floorF32Abs).toFixed(1)}x)`);
say(`[MUTATA senza -dmin*Sigma x] idot rel=${e(idotMut.rel)} abs=${e(idotMut.abs)} | ` +
  `f32 rel=${e(f32Mut.rel)} abs=${e(f32Mut.abs)}`);

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

// ===========================================================================
// PREFILL GEMM q4_1 — la riga 3 di engine-kquant, sullo stesso impianto a due
// meta' del q5_K qui sopra e NON come suo corollario.
//
// Il q4_1 sta in questo file e non in uno suo perche' l'ANTI-RICOPIATURA vuole
// i due pavimenti calcolati fianco a fianco: le quattro tolleranze q5_K sono
// gia' scritte e la tentazione, su un formato che condivide geometria di
// blocco e quantizzatore, e' di riusarle. Il blocco `pavimenti q4_1 !=
// pavimenti q5_K` piu' in basso confronta VARIABILI, non letterali: se un
// domani qualcuno ricopiasse i numeri dell'uno sull'altro, quello e' il posto
// in cui la cosa fa rumore.
//
// COSA CAMBIA DAVVERO rispetto al q5_K (e per cui il pavimento e' un altro
// numero, non lo stesso numero misurato due volte):
//   - il formato e' a BLOCCHI da 32, non a superblocchi da 256: K=9216 fa 288
//     blocchi per riga, che `prefillGemmSplitsFor` divide in 4 fette da PER=72
//     (contro i 16 superblocchi in 4 fette da 4 del q5_K). Dentro una fetta la
//     catena di somme f32 e' 144 addizioni arrotondate contro 64;
//   - l'aritmetica e' `w = d*q + m` con q in [0,15] e NESSUN offset -8, quindi
//     il terzo termine si SOMMA (`+ m*Sigma(x)`) invece di sottrarsi come il
//     `- dmin*Sigma(x)` del q5_K;
//   - le due scale d e m stanno in UNA parola (`unpack2x16float`), non in due
//     header di superblocco con le scale a 6 bit;
//   - le uscite hanno un ordine di grandezza diverso: |y| medio 1,93e2 qui
//     contro 5,37e3 sul caso q5_K. E' la ragione per cui i due pavimenti non si
//     confrontano a occhio — quello ASSOLUTO del q4_1 e' ~20x piu' BASSO,
//     quello RELATIVO ~39x piu' ALTO, e nessuno dei due si ricava dall'altro.
//
// La META' (c) — la mutazione che omette `m*Sigma(x)` — e' il discriminante
// dello STESSO difetto strutturale del q5_K letto sull'altro formato: senza
// quel termine il kernel calcola un q4_0 con i nibble non centrati, che e' il
// modo piu' facile di sbagliare questo port.
// ===========================================================================

const C41 = PREFILL_Q41_KTEST_CASE;
const K41 = C41.K, N41 = C41.N, M41 = C41.M, SPLITS41 = C41.splits;
const BPR41 = K41 / Q4_1_BLOCK_WEIGHTS;   // blocchi da 32 per riga di pesi
const PER41 = BPR41 / SPLITS41;           // blocchi per fetta
const OPTS41 = { kind: "q4_1", K: K41, N: N41, M: M41, splits: SPLITS41 } as const;

// Dati del caso, stesse convenzioni del q5_K: byte casuali + scale f16 sanate.
// Gli offset sanati sono [1, 3] come li' — ma li' erano `d` e `dmin` del
// superblocco da 176 B, qui sono `d` e `m` del blocco da 20 B: stessa
// preparazione, due layout diversi, ed e' `Q4_1_BLOCK_BYTES` a tenerli distinti.
const nBlocks41 = BPR41 * N41;
const src41 = randBytes(nBlocks41 * Q4_1_BLOCK_BYTES, C41.seedBlocks);
fixScalesAt(src41, Q4_1_BLOCK_BYTES, [1, 3]);
const w41 = new Float32Array(nBlocks41 * Q4_1_BLOCK_WEIGHTS);
dequantQ4_1(src41, 0, nBlocks41, w41);
const x41 = randF32(M41 * K41, C41.seedX);

// Emulazione di `prefillQuantXQ8Wgsl` — lo STESSO quantizzatore del q5_K e del
// q4_0 (blocchi da 32, amax/127): quel che cambia a valle e' come il
// moltiplicatore usa `xsum`, non come le attivazioni vengono preparate.
const xq41 = new Int8Array(M41 * K41);
const xsc41 = new Float32Array(M41 * BPR41);
const xsum41 = new Float32Array(M41 * BPR41);  // f32(Sigma q) * sc, come `xsum` nel kernel
const xdq41 = new Float64Array(M41 * K41);     // le attivazioni dopo il giro i8
// NIENTE `Sigma(x)` precalcolato qui, al contrario del q5_K: la via f32 del
// q4_1 accumula `sx` in REGISTRI dentro il ciclo (`dot(vec4<f32>(1.0), xa)`),
// perche' senza `dot4I8Packed` non c'e' niente da condividere fra le righe. Il
// termine e' lo stesso, l'ordine delle somme no — e l'ordine e' proprio cio'
// che il pavimento misura.
for (let b = 0; b < M41 * BPR41; b++) {
  let amax = 0;
  for (let i = 0; i < 32; i++) amax = Math.max(amax, Math.abs(x41[b * 32 + i]));
  const sc = f(amax / 127);
  const inv = sc > 0 ? f(1 / sc) : 0;
  xsc41[b] = sc;
  let sq = 0;
  for (let i = 0; i < 32; i++) {
    const q = Math.min(127, Math.max(-127, roundTiesEven(f(x41[b * 32 + i] * inv))));
    xq41[b * 32 + i] = q;
    xdq41[b * 32 + i] = q * sc;
    sq += q;
  }
  xsum41[b] = f(sq * sc);
}

// Riferimenti f64 dal dequant esatto (`dequantQ4_1`), nel layout d'uscita del
// kernel: y[m*N + r].
const refIdot41 = new Float64Array(M41 * N41);
const refF3241 = new Float64Array(M41 * N41);
for (let r = 0; r < N41; r++) {
  for (let m = 0; m < M41; m++) {
    let ai = 0, af = 0;
    for (let i = 0; i < K41; i++) {
      const wi = w41[r * K41 + i];
      ai += wi * xdq41[m * K41 + i];
      af += wi * x41[m * K41 + i];
    }
    refIdot41[m * N41 + r] = ai;
    refF3241[m * N41 + r] = af;
  }
}

// `prefillSplitKCombineWgsl` con le 4 fette del q4_1, in f32 e in ordine s.
function combine41(part: Float32Array): Float32Array {
  const y = new Float32Array(M41);
  for (let m = 0; m < M41; m++) {
    let v = 0;
    for (let s = 0; s < SPLITS41; s++) v = f(v + part[s * M41 + m]);
    y[m] = v;
  }
  return y;
}

// `dot()` del WGSL su vec4, modellato come somma sequenziale dei quattro
// termini nell'ordine lessicale delle componenti: `fma` = true lascia esatto il
// prodotto e arrotonda una volta sola, `fma` = false arrotonda anche il
// prodotto. E' la stessa coppia di modelli del q5_K, applicata dentro `dot`.
function dot4(
  a: Float64Array, ao: number, b: Float32Array | Float64Array, bo: number, fma: boolean,
): number {
  let v = 0;
  for (let i = 0; i < 4; i++) v = f(v + term(a[ao + i], b[bo + i], fma));
  return v;
}

const qn41 = new Int32Array(32);     // i 32 nibble del blocco, in [0,15]
const qf41 = new Float64Array(32);
const ones41 = new Float64Array([1, 1, 1, 1]);

// `prefillGemmQ41SplitKIdotWgsl`, una riga di pesi r: prodotto scalare INTERO
// esatto (i nibble non superano 15, le attivazioni sono i8) e poi
// `acc + d*f32(idot)*xss + m*xsum` associato a sinistra, come lo scrive il WGSL.
// Il passo a due blocchi per giro non cambia l'aritmetica — cambia solo quali
// blocchi stanno in memoria di gruppo — quindi qui i blocchi si scorrono uno
// per uno nello stesso ordine.
function emulateIdot41Row(r: number, fma: boolean, dropMin: boolean): Float32Array {
  const part = new Float32Array(SPLITS41 * M41);
  const acc = new Float64Array(M41);
  for (let s = 0; s < SPLITS41; s++) {
    acc.fill(0);
    for (let gb = s * PER41; gb < s * PER41 + PER41; gb++) {
      const o = (r * BPR41 + gb) * Q4_1_BLOCK_BYTES;
      const d = f16ToF32(src41[o] | (src41[o + 1] << 8));
      const mm = f16ToF32(src41[o + 2] | (src41[o + 3] << 8));
      // nibble basso -> elementi 0..15, nibble alto -> 16..31: la stessa
      // mappatura di `dequantQ4_1` e dei quartetti `lo`/`hi` del kernel
      for (let j = 0; j < 16; j++) {
        const by = src41[o + 4 + j];
        qn41[j] = by & 0x0f;
        qn41[16 + j] = by >> 4;
      }
      for (let m = 0; m < M41; m++) {
        const b = m * BPR41 + gb;
        let idot = 0;
        for (let l = 0; l < 32; l++) idot += qn41[l] * xq41[b * 32 + l];
        let a = acc[m];
        a = f(a + term(f(d * idot), xsc41[b], fma));
        if (!dropMin) a = f(a + term(mm, xsum41[b], fma));
        acc[m] = a;
      }
    }
    for (let m = 0; m < M41; m++) part[s * M41 + m] = acc[m];
  }
  return combine41(part);
}

// `prefillGemmQ41SplitKWgsl`: `qx` e `sx` accumulati in f32 a colpi di `dot()`
// su vec4 nell'ordine del loop (lo[wi] contro gli elementi 4wi..4wi+3, hi[wi]
// contro i 16+4wi..), poi `acc + d*qx + m*sx`.
function emulateF3241Row(r: number, fma: boolean, dropMin: boolean): Float32Array {
  const part = new Float32Array(SPLITS41 * M41);
  const acc = new Float64Array(M41);
  for (let s = 0; s < SPLITS41; s++) {
    acc.fill(0);
    for (let gb = s * PER41; gb < s * PER41 + PER41; gb++) {
      const o = (r * BPR41 + gb) * Q4_1_BLOCK_BYTES;
      const d = f16ToF32(src41[o] | (src41[o + 1] << 8));
      const mm = f16ToF32(src41[o + 2] | (src41[o + 3] << 8));
      for (let j = 0; j < 16; j++) {
        const by = src41[o + 4 + j];
        qf41[j] = by & 0x0f;
        qf41[16 + j] = by >> 4;
      }
      for (let m = 0; m < M41; m++) {
        const base = m * K41 + gb * 32;
        let qx = 0, sx = 0;
        for (let wi = 0; wi < 4; wi++) {
          qx = f(qx + dot4(qf41, wi * 4, x41, base + wi * 4, fma));
          qx = f(qx + dot4(qf41, 16 + wi * 4, x41, base + 16 + wi * 4, fma));
          sx = f(sx + dot4(ones41, 0, x41, base + wi * 4, fma));
          sx = f(sx + dot4(ones41, 0, x41, base + 16 + wi * 4, fma));
        }
        let a = acc[m];
        a = f(a + term(d, qx, fma));
        if (!dropMin) a = f(a + term(mm, sx, fma));
        acc[m] = a;
      }
    }
    for (let m = 0; m < M41; m++) part[s * M41 + m] = acc[m];
  }
  return combine41(part);
}

// stessa `compare` del ktest worker (denominatore `max(|ref|, 1e-6)`), sulle
// dimensioni del caso q4_1
function measure41(
  emul: (r: number, fma: boolean, dropMin: boolean) => Float32Array,
  ref: Float64Array, fma: boolean, dropMin: boolean,
): Err {
  let abs = 0, rel = 0;
  for (let r = 0; r < N41; r++) {
    const y = emul(r, fma, dropMin);
    for (let m = 0; m < M41; m++) {
      const ee = ref[m * N41 + r], dd = Math.abs(y[m] - ee);
      abs = Math.max(abs, dd);
      rel = Math.max(rel, dd / Math.max(Math.abs(ee), 1e-6));
    }
  }
  return { abs, rel };
}

const idot41NoFma = measure41(emulateIdot41Row, refIdot41, false, false);
const idot41Fma = measure41(emulateIdot41Row, refIdot41, true, false);
const idot41Mut = measure41(emulateIdot41Row, refIdot41, true, true);
const f3241NoFma = measure41(emulateF3241Row, refF3241, false, false);
const f3241Fma = measure41(emulateF3241Row, refF3241, true, false);
const f3241Mut = measure41(emulateF3241Row, refF3241, true, true);

const floorQ41IdotRel = Math.max(idot41NoFma.rel, idot41Fma.rel);
const floorQ41IdotAbs = Math.max(idot41NoFma.abs, idot41Fma.abs);
const floorQ41F32Rel = Math.max(f3241NoFma.rel, f3241Fma.rel);
const floorQ41F32Abs = Math.max(f3241NoFma.abs, f3241Fma.abs);

say(`[q41 idot] senzaFMA rel=${e(idot41NoFma.rel)} abs=${e(idot41NoFma.abs)} | ` +
  `conFMA rel=${e(idot41Fma.rel)} abs=${e(idot41Fma.abs)} | ` +
  `floor rel=${e(floorQ41IdotRel)} abs=${e(floorQ41IdotAbs)} | ` +
  `tol rel=${e(PREFILL_GEMM_Q41_IDOT_REL_TOL)} (${(PREFILL_GEMM_Q41_IDOT_REL_TOL / floorQ41IdotRel).toFixed(1)}x) ` +
  `abs=${e(PREFILL_GEMM_Q41_IDOT_ABS_TOL)} (${(PREFILL_GEMM_Q41_IDOT_ABS_TOL / floorQ41IdotAbs).toFixed(1)}x)`);
say(`[q41 f32]  senzaFMA rel=${e(f3241NoFma.rel)} abs=${e(f3241NoFma.abs)} | ` +
  `conFMA rel=${e(f3241Fma.rel)} abs=${e(f3241Fma.abs)} | ` +
  `floor rel=${e(floorQ41F32Rel)} abs=${e(floorQ41F32Abs)} | ` +
  `tol rel=${e(PREFILL_GEMM_Q41_F32_REL_TOL)} (${(PREFILL_GEMM_Q41_F32_REL_TOL / floorQ41F32Rel).toFixed(1)}x) ` +
  `abs=${e(PREFILL_GEMM_Q41_F32_ABS_TOL)} (${(PREFILL_GEMM_Q41_F32_ABS_TOL / floorQ41F32Abs).toFixed(1)}x)`);
say(`[q41 MUTATA senza +m*Sigma x] idot rel=${e(idot41Mut.rel)} abs=${e(idot41Mut.abs)} | ` +
  `f32 rel=${e(f3241Mut.rel)} abs=${e(f3241Mut.abs)}`);

// ---------------------------------------------------------------------------

describe("prefill gemm q4_1: struttura del WGSL generato", () => {
  const idot41 = prefillGemmQ41SplitKIdotWgsl(OPTS41);
  const f32k41 = prefillGemmQ41SplitKWgsl(OPTS41);

  it("il caso del ktest e' la geometria che il piano sceglie da solo", () => {
    // 288 blocchi da 32 si dividono in 4 fette da BK=2: e' il piano a dirlo,
    // non il caso a imporlo
    expect(prefillGemmSplitsFor(K41, N41, "q4_1")).toBe(SPLITS41);
    expect(prefillGemmSplitsFor(9216, 200, "q4_1")).toBe(4);
    expect(K41 % 64).toBe(0);
    // N%64 != 0 e' VOLUTO: e' cio' che esercita la guardia `r < N_ROWS`
    expect(N41 % 64).not.toBe(0);
    expect(200 % 64).not.toBe(0);
    expect(idot41).toContain("if (r < N_ROWS) {");
    expect(f32k41).toContain("if (r < N_ROWS) {");
  });

  it("le costanti dei due kernel sono quelle che l'emulazione presume", () => {
    for (const s of [idot41, f32k41]) {
      expect(s).toContain("const BPR = 288u;");    // K=9216 / 32
      expect(s).toContain("const PER = 72u;");     // 288 blocchi / 4 fette
      expect(s).toContain("const N_ROWS = 200u;");
      expect(s).toContain("const M_ROWS = 16u;");
    }
    expect(BPR41).toBe(288);
    expect(PER41).toBe(72);
    // niente `enable packed_4x8_integer_dot_product`: e' una language feature,
    // non un'estensione — scriverlo fa fallire la compilazione
    expect(idot41).not.toContain("enable packed_4x8_integer_dot_product");
    expect(f32k41).not.toContain("enable packed_4x8_integer_dot_product");
  });

  it("gli array di workgroup sono quelli delle due vie", () => {
    // via intera: due blocchi di attivazioni i8 (M x 16 parole) + le due
    // riduzioni per (riga di chunk, blocco), fra cui `xsum` = Sigma(x) del
    // termine costante che il q4_0 non ha
    expect(idot41).toContain("var<workgroup> xs: array<u32, 256>;");
    expect(idot41).toContain("var<workgroup> xss: array<f32, 32>;");
    expect(idot41).toContain("var<workgroup> xsum: array<f32, 32>;");
    // via f32: le stesse attivazioni lette dense, e Sigma(x) in registri
    expect(f32k41).toContain("var<workgroup> xs: array<vec4<f32>, 256>;");
    expect(f32k41).not.toContain("var<workgroup> xsum:");
  });

  it("quantizzatore e combine emulati sono quelli veri", () => {
    expect(prefillQuantXQ8Wgsl({ K: K41, M: M41 })).toContain(`const BLOCKS = ${M41 * BPR41}u;`);
    expect(prefillQuantXQ8Wgsl({ K: K41, M: M41 })).toContain("let sc = amax / 127.0;");
    expect(prefillSplitKCombineWgsl({ N: N41, M: M41, splits: SPLITS41 }))
      .toContain(`const S = ${SPLITS41}u;`);
    expect(prefillSplitKCombineWgsl({ N: N41, M: M41, splits: SPLITS41 }))
      .toContain(`const TOTAL = ${M41 * N41}u;`);
  });
});

describe("pavimento f32: prefill gemm q4_1 via intera (dot4I8Packed)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [idot41NoFma.rel, idot41Fma.rel, idot41NoFma.abs, idot41Fma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorQ41IdotRel).toBe(Math.max(idot41NoFma.rel, idot41Fma.rel));
    expect(floorQ41IdotAbs).toBe(Math.max(idot41NoFma.abs, idot41Fma.abs));
    expect(idot41NoFma.rel / idot41Fma.rel).toBeLessThan(10);
    expect(idot41Fma.rel / idot41NoFma.rel).toBeLessThan(10);
    expect(idot41NoFma.abs / idot41Fma.abs).toBeLessThan(10);
    expect(idot41Fma.abs / idot41NoFma.abs).toBeLessThan(10);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q41_IDOT_REL_TOL).toBeGreaterThanOrEqual(floorQ41IdotRel);
    expect(PREFILL_GEMM_Q41_IDOT_REL_TOL).toBeLessThanOrEqual(20 * floorQ41IdotRel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q41_IDOT_ABS_TOL).toBeGreaterThanOrEqual(floorQ41IdotAbs);
    expect(PREFILL_GEMM_Q41_IDOT_ABS_TOL).toBeLessThanOrEqual(20 * floorQ41IdotAbs);
  });
});

describe("pavimento f32: prefill gemm q4_1 via f32 (fallback)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [f3241NoFma.rel, f3241Fma.rel, f3241NoFma.abs, f3241Fma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorQ41F32Rel).toBe(Math.max(f3241NoFma.rel, f3241Fma.rel));
    expect(floorQ41F32Abs).toBe(Math.max(f3241NoFma.abs, f3241Fma.abs));
    expect(f3241NoFma.rel / f3241Fma.rel).toBeLessThan(10);
    expect(f3241Fma.rel / f3241NoFma.rel).toBeLessThan(10);
    expect(f3241NoFma.abs / f3241Fma.abs).toBeLessThan(10);
    expect(f3241Fma.abs / f3241NoFma.abs).toBeLessThan(10);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q41_F32_REL_TOL).toBeGreaterThanOrEqual(floorQ41F32Rel);
    expect(PREFILL_GEMM_Q41_F32_REL_TOL).toBeLessThanOrEqual(20 * floorQ41F32Rel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q41_F32_ABS_TOL).toBeGreaterThanOrEqual(floorQ41F32Abs);
    expect(PREFILL_GEMM_Q41_F32_ABS_TOL).toBeLessThanOrEqual(20 * floorQ41F32Abs);
  });
});

describe("anti-ricopiatura: i pavimenti q4_1 non sono quelli q5_K", () => {
  // Il confronto e' fra VARIABILI misurate nello stesso run, non fra letterali:
  // e' l'unico modo perche' resti vero anche se un domani i due casi cambiassero
  // shape. Se qualcuno tarasse le tolleranze q4_1 ricopiando i numeri q5_K, il
  // pavimento q4_1 resterebbe quello vero e il 20x lo direbbe — ma questo blocco
  // lo dice PRIMA, sul pavimento stesso.
  it("le due vie del q4_1 hanno pavimenti diversi dalle due vie del q5_K", () => {
    expect(floorQ41IdotRel).not.toBe(floorIdotRel);
    expect(floorQ41IdotAbs).not.toBe(floorIdotAbs);
    expect(floorQ41F32Rel).not.toBe(floorF32Rel);
    expect(floorQ41F32Abs).not.toBe(floorF32Abs);
  });

  it("i due casi sono due esperimenti diversi, non lo stesso scritto due volte", () => {
    // K diverso e quattro seed distinti: i due pavimenti non possono coincidere
    // per costruzione, e i due generatori non condividono un flusso
    expect(C41.K).not.toBe(C.K);
    expect(new Set([C41.seedBlocks, C41.seedX, C.seedBlocks, C.seedX]).size).toBe(4);
  });
});

describe("discriminante q4_1: la tolleranza non copre un difetto strutturale", () => {
  // La `compare` del ktest passa se `maxAbs <= absTol OPPURE maxRel <= relTol`:
  // perche' la mutazione sia CATTURATA deve sfondare ENTRAMBE le tolleranze.
  it("omettere +m*Sigma(x) sfonda le tolleranze della via intera di 10x", () => {
    expect(idot41Mut.rel / PREFILL_GEMM_Q41_IDOT_REL_TOL).toBeGreaterThan(10);
    expect(idot41Mut.abs / PREFILL_GEMM_Q41_IDOT_ABS_TOL).toBeGreaterThan(10);
  });

  it("omettere +m*Sigma(x) sfonda le tolleranze della via f32 di 10x", () => {
    expect(f3241Mut.rel / PREFILL_GEMM_Q41_F32_REL_TOL).toBeGreaterThan(10);
    expect(f3241Mut.abs / PREFILL_GEMM_Q41_F32_ABS_TOL).toBeGreaterThan(10);
  });
});

// ===========================================================================
// LE TRE FORME DELLA RIGA 4 — q4_K, q6_K, q8_0.
//
// Stesso impianto a due meta' dei due blocchi qui sopra, e stessa regola: il
// pavimento e' l'INVILUPPO dei due modelli di contrazione, la tolleranza si
// legge dal pavimento, e una mutazione dimostra che la banda cosi' derivata NON
// copre un difetto di formato.
//
// COSA CONDIVIDONO CON I DUE BLOCCHI PRECEDENTI, e cosa no. Condividono UNA
// cosa sola — la preparazione delle attivazioni (`quantizeActivations` qui
// sotto) — e la condividono perche' e' un FATTO del codice, non una comodita':
// `prefillQuantXQ8Wgsl` non ha un parametro `kind`, e' lo STESSO generatore per
// tutti e sei i moltiplicatori. Due copie divergenti di quella preparazione
// descriverebbero due quantizzatori dove ne gira uno solo. Tutto il resto —
// caso, emulazione per via, pavimenti, mutazione — resta scritto a mano per
// formato: e' quello che l'anti-ricopiatura protegge, ed e' quello che i
// blocchi `pavimenti X != pavimenti Y` confrontano fra VARIABILI misurate.
// ===========================================================================

/**
 * Le attivazioni come le prepara `prefillQuantXQ8Wgsl`: amax/127 per blocco da
 * 32, byte con segno, piu' le due riduzioni che i moltiplicatori interi leggono
 * da memoria di gruppo (`xsc` = scala del blocco, `xsum` = f32(Sigma q) * sc).
 *
 * `xdq` sono le attivazioni DOPO il giro a i8: e' su quelle che si calcola il
 * riferimento del braccio intero, perche' il caso deve misurare l'aritmetica
 * del kernel e non la perdita del quantizzatore — mescolarle renderebbe la
 * tolleranza illeggibile.
 */
function quantizeActivations(x: Float32Array, nb: number): {
  xq: Int8Array; xsc: Float32Array; xsum: Float32Array; xdq: Float64Array;
} {
  const xq = new Int8Array(nb * 32);
  const xsc = new Float32Array(nb);
  const xsum = new Float32Array(nb);
  const xdq = new Float64Array(nb * 32);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let i = 0; i < 32; i++) amax = Math.max(amax, Math.abs(x[b * 32 + i]));
    const sc = f(amax / 127);
    const inv = sc > 0 ? f(1 / sc) : 0;
    xsc[b] = sc;
    let sq = 0;
    for (let i = 0; i < 32; i++) {
      const q = Math.min(127, Math.max(-127, roundTiesEven(f(x[b * 32 + i] * inv))));
      xq[b * 32 + i] = q;
      xdq[b * 32 + i] = q * sc;
      sq += q;
    }
    xsum[b] = f(sq * sc);
  }
  return { xq, xsc, xsum, xdq };
}

/** I due riferimenti f64 nel layout d'uscita del kernel, y[m*N + r]. */
function refsOf(
  w: Float32Array, x: Float32Array, xdq: Float64Array, K_: number, N_: number, M_: number,
): { idot: Float64Array; f32: Float64Array } {
  const idot = new Float64Array(M_ * N_), f32 = new Float64Array(M_ * N_);
  for (let r = 0; r < N_; r++) {
    for (let m = 0; m < M_; m++) {
      let ai = 0, af = 0;
      for (let i = 0; i < K_; i++) {
        const wi = w[r * K_ + i];
        ai += wi * xdq[m * K_ + i];
        af += wi * x[m * K_ + i];
      }
      idot[m * N_ + r] = ai;
      f32[m * N_ + r] = af;
    }
  }
  return { idot, f32 };
}

/** `prefillSplitKCombineWgsl`: somma delle fette in f32, in ordine s = 0..S-1. */
function combineAt(part: Float32Array, M_: number, splits_: number): Float32Array {
  const y = new Float32Array(M_);
  for (let m = 0; m < M_; m++) {
    let v = 0;
    for (let s = 0; s < splits_; s++) v = f(v + part[s * M_ + m]);
    y[m] = v;
  }
  return y;
}

/** maxAbs/maxRel con la STESSA definizione di `compare` nel ktest worker. */
function measureAt(
  N_: number, M_: number,
  emul: (r: number, fma: boolean, mut: boolean) => Float32Array,
  ref: Float64Array, fma: boolean, mut: boolean,
): Err {
  let abs = 0, rel = 0;
  for (let r = 0; r < N_; r++) {
    const y = emul(r, fma, mut);
    for (let m = 0; m < M_; m++) {
      const ee = ref[m * N_ + r], dd = Math.abs(y[m] - ee);
      abs = Math.max(abs, dd);
      rel = Math.max(rel, dd / Math.max(Math.abs(ee), 1e-6));
    }
  }
  return { abs, rel };
}

/** byte -> int8 con segno (le scale del Q6_K, e i pesi del q8_0). */
const i8 = (b: number): number => (b << 24) >> 24;

/** la riga di derivazione stampata per un braccio, nel formato dei due blocchi sopra */
function sayFloor(
  tag: string, noFma: Err, withFma: Err, relTol: number, absTol: number,
): void {
  const fr = Math.max(noFma.rel, withFma.rel), fa = Math.max(noFma.abs, withFma.abs);
  say(`[${tag}] senzaFMA rel=${e(noFma.rel)} abs=${e(noFma.abs)} | ` +
    `conFMA rel=${e(withFma.rel)} abs=${e(withFma.abs)} | ` +
    `floor rel=${e(fr)} abs=${e(fa)} | ` +
    `tol rel=${e(relTol)} (${(relTol / fr).toFixed(1)}x) abs=${e(absTol)} (${(absTol / fa).toFixed(1)}x)`);
}

// ===========================================================================
// PREFILL GEMM q4_K — la prima delle tre forme della riga 4.
//
// COSA CAMBIA rispetto al q5_K, e per cui il pavimento e' un altro numero e non
// lo stesso misurato due volte:
//   - il superblocco e' da 144 B / 36 parole, non 176 B / 44: NON C'E' il piano
//     `qh` del quinto bit, e `qs` sta a byte 16 — dove il q5_K tiene il suo
//     `qh`. E' li' che un port distratto sbaglia, ed e' il primo controllo
//     strutturale qui sotto;
//   - i pesi hanno 4 bit invece di 5, cioe' q in [0,15] contro [0,31]: a parita'
//     di scala il prodotto scalare intero e' circa la meta', e le uscite sono
//     piu' piccole di conseguenza;
//   - K=2048 fa 8 superblocchi per riga in 4 fette da PER=2, contro i 16 in 4
//     fette da PER=4 del caso q5_K: dentro una fetta la catena di somme f32 e'
//     la META' in unita'.
// Header, `get_scale_min_k4` e l'aritmetica `d*sc*q - dmin*mn` sono invece gli
// stessi — ed e' proprio per questo che i due pavimenti vanno CONFRONTATI, non
// dedotti l'uno dall'altro.
//
// LA MUTAZIONE E' `- dmin*mn*Sigma(x)`, il termine costante del formato. E' la
// scelta giusta per il q4_K perche' e' l'UNICO punto in cui `dmin` e i minimi a
// 6 bit entrano nel risultato: un kernel che lo perde non e' un q4_K un po'
// sbagliato, e' un q4_0 con i nibble non centrati — cioe' il modo piu' facile di
// sbagliare questo port, e quello che nessun controllo di forma vede. La stessa
// mutazione discrimina sul q5_K perche' i due formati CONDIVIDONO quel termine:
// e' un fatto della famiglia, non una ricopiatura, e i due errori mutati sono
// due numeri diversi misurati qui accanto.
// ===========================================================================

const C4K = PREFILL_Q4K_KTEST_CASE;
const K4K = C4K.K, N4K = C4K.N, M4K = C4K.M, SPLITS4K = C4K.splits;
const SBPR4K = K4K / Q4_K_BLOCK_WEIGHTS;   // superblocchi da 256 per riga di pesi
const PER4K = SBPR4K / SPLITS4K;           // superblocchi per fetta
const BPR4K = K4K / 32;                    // blocchi di attivazione per riga di chunk
const OPTS4K = { kind: "q4_K", K: K4K, N: N4K, M: M4K, splits: SPLITS4K } as const;

const nBlocks4K = SBPR4K * N4K;
const src4K = randBytes(nBlocks4K * Q4_K_BLOCK_BYTES, C4K.seedBlocks);
fixScalesAt(src4K, Q4_K_BLOCK_BYTES, [1, 3]);   // d, dmin: stesso header del q5_K
const w4K = new Float32Array(nBlocks4K * Q4_K_BLOCK_WEIGHTS);
dequantQ4_K(src4K, 0, nBlocks4K, w4K);
const x4KA = randF32(M4K * K4K, C4K.seedX);
const A4K = quantizeActivations(x4KA, M4K * BPR4K);
const REF4K = refsOf(w4K, x4KA, A4K.xdq, K4K, N4K, M4K);

const q4kLo = new Int32Array(32), q4kHi = new Int32Array(32);

// `prefillGemmQ4KSplitKIdotWgsl`, una riga di pesi r: per superblocco, per
// gruppo g (due sotto-blocchi is = 2g e is+1), prodotto scalare INTERO esatto e
// poi `acc + d1*i1*sc_x - min1*xsum + d2*i2*sc_x - min2*xsum` associato a
// sinistra, come lo scrive il WGSL.
function emulateQ4KIdotRow(r: number, fma: boolean, dropMin: boolean): Float32Array {
  const part = new Float32Array(SPLITS4K * M4K);
  const acc = new Float64Array(M4K);
  for (let s = 0; s < SPLITS4K; s++) {
    acc.fill(0);
    for (let sb = s * PER4K; sb < s * PER4K + PER4K; sb++) {
      const o = (r * SBPR4K + sb) * Q4_K_BLOCK_BYTES;
      const d = f16ToF32(src4K[o] | (src4K[o + 1] << 8));
      const dmin = f16ToF32(src4K[o + 2] | (src4K[o + 3] << 8));
      for (let g = 0; g < 4; g++) {
        const is = 2 * g;
        const [sc1, mn1] = scaleMinK4(is, src4K, o + 4);
        const [sc2, mn2] = scaleMinK4(is + 1, src4K, o + 4);
        const d1 = f(d * sc1), min1 = f(dmin * mn1);
        const d2 = f(d * sc2), min2 = f(dmin * mn2);
        // `qs` a byte 16 (il q5_K lo tiene a 48, e a 16 ha il piano `qh`)
        for (let l = 0; l < 32; l++) {
          const ql = src4K[o + 16 + g * 32 + l];
          q4kLo[l] = ql & 0x0f;
          q4kHi[l] = ql >> 4;
        }
        for (let m = 0; m < M4K; m++) {
          const b1 = m * BPR4K + sb * 8 + is, b2 = b1 + 1;
          let i1 = 0, i2 = 0;
          for (let l = 0; l < 32; l++) {
            i1 += q4kLo[l] * A4K.xq[b1 * 32 + l];
            i2 += q4kHi[l] * A4K.xq[b2 * 32 + l];
          }
          let a = acc[m];
          a = f(a + term(f(d1 * i1), A4K.xsc[b1], fma));
          if (!dropMin) a = f(a - term(min1, A4K.xsum[b1], fma));
          a = f(a + term(f(d2 * i2), A4K.xsc[b2], fma));
          if (!dropMin) a = f(a - term(min2, A4K.xsum[b2], fma));
          acc[m] = a;
        }
      }
    }
    for (let m = 0; m < M4K; m++) part[s * M4K + m] = acc[m];
  }
  return combineAt(part, M4K, SPLITS4K);
}

const q4kF = new Float64Array(32);

// `prefillGemmQ4KSplitKWgsl`: UN sotto-blocco da 32 per volta, `qx` e `sx`
// accumulati in f32 NELLO STESSO ciclo (il q5_K prende invece Sigma(x) da una
// riduzione precalcolata), poi `acc + dsc*qx - dmn*sx`.
function emulateQ4KF32Row(r: number, fma: boolean, dropMin: boolean): Float32Array {
  const part = new Float32Array(SPLITS4K * M4K);
  const acc = new Float64Array(M4K);
  for (let s = 0; s < SPLITS4K; s++) {
    acc.fill(0);
    for (let sb = s * PER4K; sb < s * PER4K + PER4K; sb++) {
      const o = (r * SBPR4K + sb) * Q4_K_BLOCK_BYTES;
      const d = f16ToF32(src4K[o] | (src4K[o + 1] << 8));
      const dmin = f16ToF32(src4K[o + 2] | (src4K[o + 3] << 8));
      for (let is = 0; is < 8; is++) {
        const g = is >> 1, hiHalf = (is & 1) === 1;
        const [sc, mn] = scaleMinK4(is, src4K, o + 4);
        const dsc = f(d * sc), dmn = f(dmin * mn);
        for (let l = 0; l < 32; l++) {
          const ql = src4K[o + 16 + g * 32 + l];
          q4kF[l] = hiHalf ? ql >> 4 : ql & 0x0f;
        }
        for (let m = 0; m < M4K; m++) {
          const base = m * K4K + sb * 256 + is * 32;
          let qx = 0, sx = 0;
          for (let l = 0; l < 32; l++) {
            const xv = x4KA[base + l];
            qx = f(qx + term(q4kF[l], xv, fma));
            sx = f(sx + xv);
          }
          let a = acc[m];
          a = f(a + term(dsc, qx, fma));
          if (!dropMin) a = f(a - term(dmn, sx, fma));
          acc[m] = a;
        }
      }
    }
    for (let m = 0; m < M4K; m++) part[s * M4K + m] = acc[m];
  }
  return combineAt(part, M4K, SPLITS4K);
}

const idot4KNoFma = measureAt(N4K, M4K, emulateQ4KIdotRow, REF4K.idot, false, false);
const idot4KFma = measureAt(N4K, M4K, emulateQ4KIdotRow, REF4K.idot, true, false);
const idot4KMut = measureAt(N4K, M4K, emulateQ4KIdotRow, REF4K.idot, true, true);
const f324KNoFma = measureAt(N4K, M4K, emulateQ4KF32Row, REF4K.f32, false, false);
const f324KFma = measureAt(N4K, M4K, emulateQ4KF32Row, REF4K.f32, true, false);
const f324KMut = measureAt(N4K, M4K, emulateQ4KF32Row, REF4K.f32, true, true);

const floorQ4KIdotRel = Math.max(idot4KNoFma.rel, idot4KFma.rel);
const floorQ4KIdotAbs = Math.max(idot4KNoFma.abs, idot4KFma.abs);
const floorQ4KF32Rel = Math.max(f324KNoFma.rel, f324KFma.rel);
const floorQ4KF32Abs = Math.max(f324KNoFma.abs, f324KFma.abs);

sayFloor("q4K idot", idot4KNoFma, idot4KFma,
  PREFILL_GEMM_Q4K_IDOT_REL_TOL, PREFILL_GEMM_Q4K_IDOT_ABS_TOL);
sayFloor("q4K f32 ", f324KNoFma, f324KFma,
  PREFILL_GEMM_Q4K_F32_REL_TOL, PREFILL_GEMM_Q4K_F32_ABS_TOL);
say(`[q4K MUTATA senza -dmin*mn*Sigma x] idot rel=${e(idot4KMut.rel)} abs=${e(idot4KMut.abs)} | ` +
  `f32 rel=${e(f324KMut.rel)} abs=${e(f324KMut.abs)}`);

// media del modulo delle uscite: e' cio' che rende leggibile il confronto fra un
// pavimento ASSOLUTO e l'altro (due formati, due ordini di grandezza d'uscita)
const meanAbs = (v: Float64Array): number => {
  let s = 0;
  for (const x of v) s += Math.abs(x);
  return s / v.length;
};
say(`[q4K |y|] idot medio=${e(meanAbs(REF4K.idot))} f32 medio=${e(meanAbs(REF4K.f32))}`);

// ---------------------------------------------------------------------------

describe("prefill gemm q4_K: struttura del WGSL generato", () => {
  const idot4k = prefillGemmQ4KSplitKIdotWgsl(OPTS4K);
  const f32k4k = prefillGemmQ4KSplitKWgsl(OPTS4K);

  it("il caso del ktest e' la geometria che il piano sceglie da solo", () => {
    expect(prefillGemmSplitsFor(K4K, N4K, "q4_K")).toBe(SPLITS4K);
    expect(K4K % 256).toBe(0);
    // N%64 != 0 e' VOLUTO: e' cio' che esercita la guardia `r < N_ROWS`
    expect(N4K % 64).not.toBe(0);
    expect(idot4k).toContain(`const N_ROWS = ${N4K}u;`);
    expect(idot4k).toContain("if (r < N_ROWS) {");
    expect(f32k4k).toContain("if (r < N_ROWS) {");
  });

  it("le costanti dei due kernel sono quelle che l'emulazione presume", () => {
    for (const s of [idot4k, f32k4k]) {
      expect(s).toContain("const WORDS = 36u;");   // 144 B di superblocco q4_K
      expect(s).toContain("const SBPR = 8u;");     // K=2048 / 256
      expect(s).toContain("const PER = 2u;");      // 8 superblocchi / 4 fette
      expect(s).toContain(`const M_ROWS = ${M4K}u;`);
    }
    expect(SBPR4K).toBe(8);
    expect(PER4K).toBe(2);
    // niente `enable packed_4x8_integer_dot_product`: e' una language feature,
    // non un'estensione — scriverlo fa fallire la compilazione
    expect(idot4k).not.toContain("enable packed_4x8_integer_dot_product");
    expect(f32k4k).not.toContain("enable packed_4x8_integer_dot_product");
  });

  it("`qs` sta a byte 16, non a 48: e' il q5_K SENZA il piano del quinto bit", () => {
    // l'emulazione legge i nibble a `o + 16 + g*32 + l`; se il kernel li
    // leggesse dove li tiene il q5_K, il pavimento qui sotto misurerebbe una
    // copia privata e il ktest boccerebbe il kernel vero.
    expect(idot4k).toContain("let word = blocks[wb + 4u + g * 8u + ii];"); // parola 4 = byte 16
    expect(f32k4k).toContain("let ql = sbyte(wb, 16u + g * 32u + l);");
    // e non c'e' NIENTE del piano alto: nessun `qh`
    for (const s of [idot4k, f32k4k]) expect(s).not.toContain("qh");
  });

  it("gli array di workgroup sono quelli delle due vie", () => {
    // via intera: il superblocco INTERO di attivazioni i8 (M x 8 x 8 parole) +
    // le due riduzioni per (riga di chunk, sotto-blocco)
    expect(idot4k).toContain("var<workgroup> xs: array<u32, 1024>;");
    expect(idot4k).toContain("var<workgroup> xss: array<f32, 128>;");
    expect(idot4k).toContain("var<workgroup> xsum: array<f32, 128>;");
    // via f32: UN sotto-blocco da 32 per volta (M x 32 f32), non il superblocco
    expect(f32k4k).toContain("var<workgroup> xs: array<f32, 512>;");
    expect(f32k4k).not.toContain("array<u32,");
  });

  it("quantizzatore e combine emulati sono quelli veri", () => {
    expect(prefillQuantXQ8Wgsl({ K: K4K, M: M4K })).toContain(`const BLOCKS = ${M4K * BPR4K}u;`);
    expect(prefillQuantXQ8Wgsl({ K: K4K, M: M4K })).toContain("let sc = amax / 127.0;");
    expect(prefillSplitKCombineWgsl({ N: N4K, M: M4K, splits: SPLITS4K }))
      .toContain(`const S = ${SPLITS4K}u;`);
    expect(prefillSplitKCombineWgsl({ N: N4K, M: M4K, splits: SPLITS4K }))
      .toContain(`const TOTAL = ${M4K * N4K}u;`);
  });
});

describe("pavimento f32: prefill gemm q4_K via intera (dot4I8Packed)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [idot4KNoFma.rel, idot4KFma.rel, idot4KNoFma.abs, idot4KFma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorQ4KIdotRel).toBe(Math.max(idot4KNoFma.rel, idot4KFma.rel));
    expect(floorQ4KIdotAbs).toBe(Math.max(idot4KNoFma.abs, idot4KFma.abs));
    expect(idot4KNoFma.rel / idot4KFma.rel).toBeLessThan(10);
    expect(idot4KFma.rel / idot4KNoFma.rel).toBeLessThan(10);
    expect(idot4KNoFma.abs / idot4KFma.abs).toBeLessThan(10);
    expect(idot4KFma.abs / idot4KNoFma.abs).toBeLessThan(10);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q4K_IDOT_REL_TOL).toBeGreaterThanOrEqual(floorQ4KIdotRel);
    expect(PREFILL_GEMM_Q4K_IDOT_REL_TOL).toBeLessThanOrEqual(20 * floorQ4KIdotRel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q4K_IDOT_ABS_TOL).toBeGreaterThanOrEqual(floorQ4KIdotAbs);
    expect(PREFILL_GEMM_Q4K_IDOT_ABS_TOL).toBeLessThanOrEqual(20 * floorQ4KIdotAbs);
  });
});

describe("pavimento f32: prefill gemm q4_K via f32 (fallback)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [f324KNoFma.rel, f324KFma.rel, f324KNoFma.abs, f324KFma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorQ4KF32Rel).toBe(Math.max(f324KNoFma.rel, f324KFma.rel));
    expect(floorQ4KF32Abs).toBe(Math.max(f324KNoFma.abs, f324KFma.abs));
    expect(f324KNoFma.rel / f324KFma.rel).toBeLessThan(10);
    expect(f324KFma.rel / f324KNoFma.rel).toBeLessThan(10);
    expect(f324KNoFma.abs / f324KFma.abs).toBeLessThan(10);
    expect(f324KFma.abs / f324KNoFma.abs).toBeLessThan(10);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q4K_F32_REL_TOL).toBeGreaterThanOrEqual(floorQ4KF32Rel);
    expect(PREFILL_GEMM_Q4K_F32_REL_TOL).toBeLessThanOrEqual(20 * floorQ4KF32Rel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q4K_F32_ABS_TOL).toBeGreaterThanOrEqual(floorQ4KF32Abs);
    expect(PREFILL_GEMM_Q4K_F32_ABS_TOL).toBeLessThanOrEqual(20 * floorQ4KF32Abs);
  });
});

describe("discriminante q4_K: la tolleranza non copre un difetto strutturale", () => {
  // La `compare` del ktest passa se `maxAbs <= absTol OPPURE maxRel <= relTol`:
  // perche' la mutazione sia CATTURATA deve sfondare ENTRAMBE le tolleranze.
  it("omettere -dmin*mn*Sigma(x) sfonda le tolleranze della via intera di 10x", () => {
    expect(idot4KMut.rel / PREFILL_GEMM_Q4K_IDOT_REL_TOL).toBeGreaterThan(10);
    expect(idot4KMut.abs / PREFILL_GEMM_Q4K_IDOT_ABS_TOL).toBeGreaterThan(10);
  });

  it("omettere -dmin*mn*Sigma(x) sfonda le tolleranze della via f32 di 10x", () => {
    expect(f324KMut.rel / PREFILL_GEMM_Q4K_F32_REL_TOL).toBeGreaterThan(10);
    expect(f324KMut.abs / PREFILL_GEMM_Q4K_F32_ABS_TOL).toBeGreaterThan(10);
  });
});

describe("anti-ricopiatura: i pavimenti q4_K non sono quelli q5_K ne' quelli q4_1", () => {
  // Sul q4_K la tentazione e' piu' forte che altrove: header, `get_scale_min_k4`
  // e forma dell'aritmetica sono LETTERALMENTE quelli del q5_K, e le quattro
  // tolleranze del q5_K sono gia' scritte due schermate piu' su. Il confronto
  // qui e' fra VARIABILI misurate nello stesso run, non fra letterali.
  it("le due vie del q4_K hanno pavimenti diversi da quelle del q5_K e del q4_1", () => {
    expect(floorQ4KIdotRel).not.toBe(floorIdotRel);
    expect(floorQ4KIdotAbs).not.toBe(floorIdotAbs);
    expect(floorQ4KF32Rel).not.toBe(floorF32Rel);
    expect(floorQ4KF32Abs).not.toBe(floorF32Abs);
    expect(floorQ4KIdotRel).not.toBe(floorQ41IdotRel);
    expect(floorQ4KIdotAbs).not.toBe(floorQ41IdotAbs);
    expect(floorQ4KF32Rel).not.toBe(floorQ41F32Rel);
    expect(floorQ4KF32Abs).not.toBe(floorQ41F32Abs);
  });

  it("i tre casi sono tre esperimenti diversi, non lo stesso scritto tre volte", () => {
    expect(new Set([C.K, C41.K, C4K.K]).size).toBe(3);
    expect(new Set([
      C.seedBlocks, C.seedX, C41.seedBlocks, C41.seedX, C4K.seedBlocks, C4K.seedX,
    ]).size).toBe(6);
  });
});

// ===========================================================================
// PREFILL GEMM q6_K — la seconda delle tre forme della riga 4, e la piu' diversa
// di tutte e sei.
//
// COSA CAMBIA, e per cui nessun numero di questo blocco si ricava dagli altri:
//   - LE SCALE SONO PER SOTTO-BLOCCO DA 16, non da 32: 16 scale int8 per
//     superblocco. Le attivazioni pero' sono quantizzate per 32 — sempre lo
//     stesso `prefillQuantXQ8Wgsl` — quindi le due META' di un blocco
//     condividono la scala di x e hanno scale di peso diverse. E' per questo che
//     la via intera tiene `xh`, la somma delle attivazioni per MEZZO
//     sotto-blocco, dove le altre famiglie hanno una riduzione per blocco intero;
//   - NON C'E' UN MINIMO. La scala di peso e' int8 (con segno) e i pesi sono
//     CENTRATI SU -32: il termine costante non e' `dmin*mn` ma l'offset secco
//     `32`, lo stesso per ogni blocco di ogni riga di ogni tensore;
//   - le due vie portano quell'offset in DUE FORME ALGEBRICHE DIVERSE, ed e'
//     l'unico posto in cui i due bracci di un formato non calcolano la stessa
//     espressione. La via intera non puo' sottrarre 32 in forma impacchettata
//     (traboccherebbe fra i byte), quindi tiene q in [0,63] e fattorizza:
//     `Sigma (q-32)x = Sigma q*x - 32*Sigma x`. La via f32 sottrae 32 DAL PESO,
//     un elemento per volta;
//   - K=512 fa DUE superblocchi per riga in DUE fette da PER=1: la fetta piu'
//     corta che il piano possa produrre.
//
// LA MUTAZIONE E' L'OFFSET -32, e la ragione per cui e' questa: e' l'unico
// termine costante che il q6_K ha, ed e' cio' che rende i suoi pesi CENTRATI. Un
// kernel che lo perde legge sei bit senza segno come se fossero un peso — che e'
// esattamente cio' che il testo del kernel avverte di non fare, e cio' che
// nessun controllo di forma vede. Non e' la mutazione del q5_K/q4_K travestita:
// li' si toglie un prodotto di DUE scale lette dai byte del superblocco, qui una
// costante letterale che non sta in nessun byte. E soprattutto la si toglie in
// DUE POSTI DIVERSI — dal `- 32.0*xh[...]` fattorizzato della via intera e dal
// `- 32.0` per elemento della via f32 — quindi il discriminante prova che
// ENTRAMBE le strade algebriche portano l'offset, non solo che una lo porta.
// ===========================================================================

const C6K = PREFILL_Q6K_KTEST_CASE;
const K6K = C6K.K, N6K = C6K.N, M6K = C6K.M, SPLITS6K = C6K.splits;
const SBPR6K = K6K / Q6_K_BLOCK_WEIGHTS;   // superblocchi da 256 per riga di pesi
const PER6K = SBPR6K / SPLITS6K;           // superblocchi per fetta
const BPR6K = K6K / 32;                    // blocchi di attivazione per riga di chunk
const OPTS6K = { kind: "q6_K", K: K6K, N: N6K, M: M6K, splits: SPLITS6K } as const;

const nBlocks6K = SBPR6K * N6K;
const src6K = randBytes(nBlocks6K * Q6_K_BLOCK_BYTES, C6K.seedBlocks);
fixScalesAt(src6K, Q6_K_BLOCK_BYTES, [209]);   // `d` IN CODA: le 16 scale sono int8
const w6K = new Float32Array(nBlocks6K * Q6_K_BLOCK_WEIGHTS);
dequantQ6_K(src6K, 0, nBlocks6K, w6K);
const x6KA = randF32(M6K * K6K, C6K.seedX);
const A6K = quantizeActivations(x6KA, M6K * BPR6K);
const REF6K = refsOf(w6K, x6KA, A6K.xdq, K6K, N6K, M6K);

// `xh` del kernel: Sigma dei q i8 per MEZZA fetta di blocco (16 elementi), in
// intero esatto. E' la granularita' delle scale del Q6_K, e l'unica riduzione di
// questo file che non e' per blocco da 32.
const xh6K = new Float32Array(M6K * BPR6K * 2);
for (let b = 0; b < M6K * BPR6K; b++) {
  for (let half = 0; half < 2; half++) {
    let sq = 0;
    for (let i = 0; i < 16; i++) sq += A6K.xq[b * 32 + half * 16 + i];
    xh6K[b * 2 + half] = sq;
  }
}

const w6 = new Int32Array(32);   // i 32 pesi del sotto-blocco, SENZA segno in [0,63]

// `prefillGemmQ6KSplitKIdotWgsl`, una riga di pesi r: per superblocco, per meta'
// (n) e per quarto (c), prodotto scalare INTERO esatto sulle due meta' del
// blocco e poi `acc + d*sc_x*(scA*(iA - 32*xhA) + scB*(iB - 32*xhB))` come lo
// scrive il WGSL.
function emulateQ6KIdotRow(r: number, fma: boolean, dropOfs: boolean): Float32Array {
  const part = new Float32Array(SPLITS6K * M6K);
  const acc = new Float64Array(M6K);
  for (let s = 0; s < SPLITS6K; s++) {
    acc.fill(0);
    for (let sb = s * PER6K; sb < s * PER6K + PER6K; sb++) {
      const o = (r * SBPR6K + sb) * Q6_K_BLOCK_BYTES;
      const d = f16ToF32(src6K[o + 208] | (src6K[o + 209] << 8));
      for (let n = 0; n < 2; n++) {
        const scO = o + 192 + n * 8;
        for (let c = 0; c < 4; c++) {
          const blk = n * 4 + c;              // sotto-blocco da 32 nel superblocco
          const sh = 2 * c;                   // bit del piano alto per questo quarto
          const qlO = o + n * 64 + (c & 1) * 32;
          const qhO = o + 128 + n * 32;
          for (let l = 0; l < 32; l++) {
            const ql = src6K[qlO + l], qh = src6K[qhO + l];
            const nib = c >= 2 ? ql >> 4 : ql & 0x0f;
            w6[l] = nib | (((qh >> sh) & 3) << 4);
          }
          const scA = i8(src6K[scO + 2 * c]);
          const scB = i8(src6K[scO + 1 + 2 * c]);
          for (let m = 0; m < M6K; m++) {
            const b = m * BPR6K + sb * 8 + blk;
            let iA = 0, iB = 0;
            for (let l = 0; l < 16; l++) iA += w6[l] * A6K.xq[b * 32 + l];
            for (let l = 16; l < 32; l++) iB += w6[l] * A6K.xq[b * 32 + l];
            // l'offset -32 esce dal prodotto scalare, per CIASCUNA meta'
            const uA = dropOfs ? iA : f(iA - term(32, xh6K[b * 2], fma));
            const uB = dropOfs ? iB : f(iB - term(32, xh6K[b * 2 + 1], fma));
            const sum = f(term(scA, uA, fma) + term(scB, uB, fma));
            acc[m] = f(acc[m] + term(f(d * A6K.xsc[b]), sum, fma));
          }
        }
      }
    }
    for (let m = 0; m < M6K; m++) part[s * M6K + m] = acc[m];
  }
  return combineAt(part, M6K, SPLITS6K);
}

const q6f = new Float64Array(32);

// `prefillGemmQ6KSplitKWgsl`: UN sotto-blocco da 32 per volta, l'offset -32
// sottratto DAL PESO, due accumuli f32 separati per le due meta' (le due scale
// int8), poi `acc + d*(scA*a + scB*b)`.
function emulateQ6KF32Row(r: number, fma: boolean, dropOfs: boolean): Float32Array {
  const part = new Float32Array(SPLITS6K * M6K);
  const acc = new Float64Array(M6K);
  for (let s = 0; s < SPLITS6K; s++) {
    acc.fill(0);
    for (let sb = s * PER6K; sb < s * PER6K + PER6K; sb++) {
      const o = (r * SBPR6K + sb) * Q6_K_BLOCK_BYTES;
      const d = f16ToF32(src6K[o + 208] | (src6K[o + 209] << 8));
      for (let blk = 0; blk < 8; blk++) {
        const n = blk >> 2, c = blk & 3, sh = 2 * c;
        const qlO = o + n * 64 + (c & 1) * 32;
        const qhO = o + 128 + n * 32;
        const scO = o + 192 + n * 8;
        for (let l = 0; l < 32; l++) {
          const ql = src6K[qlO + l], qh = src6K[qhO + l];
          const nib = c >= 2 ? ql >> 4 : ql & 0x0f;
          q6f[l] = (nib | (((qh >> sh) & 3) << 4)) - (dropOfs ? 0 : 32);
        }
        const scA = i8(src6K[scO + 2 * c]);
        const scB = i8(src6K[scO + 1 + 2 * c]);
        for (let m = 0; m < M6K; m++) {
          const base = m * K6K + sb * 256 + blk * 32;
          let a1 = 0, b1 = 0;
          for (let l = 0; l < 16; l++) {
            a1 = f(a1 + term(q6f[l], x6KA[base + l], fma));
            b1 = f(b1 + term(q6f[l + 16], x6KA[base + l + 16], fma));
          }
          const inner = f(term(scA, a1, fma) + term(scB, b1, fma));
          acc[m] = f(acc[m] + term(d, inner, fma));
        }
      }
    }
    for (let m = 0; m < M6K; m++) part[s * M6K + m] = acc[m];
  }
  return combineAt(part, M6K, SPLITS6K);
}

const idot6KNoFma = measureAt(N6K, M6K, emulateQ6KIdotRow, REF6K.idot, false, false);
const idot6KFma = measureAt(N6K, M6K, emulateQ6KIdotRow, REF6K.idot, true, false);
const idot6KMut = measureAt(N6K, M6K, emulateQ6KIdotRow, REF6K.idot, true, true);
const f326KNoFma = measureAt(N6K, M6K, emulateQ6KF32Row, REF6K.f32, false, false);
const f326KFma = measureAt(N6K, M6K, emulateQ6KF32Row, REF6K.f32, true, false);
const f326KMut = measureAt(N6K, M6K, emulateQ6KF32Row, REF6K.f32, true, true);

const floorQ6KIdotRel = Math.max(idot6KNoFma.rel, idot6KFma.rel);
const floorQ6KIdotAbs = Math.max(idot6KNoFma.abs, idot6KFma.abs);
const floorQ6KF32Rel = Math.max(f326KNoFma.rel, f326KFma.rel);
const floorQ6KF32Abs = Math.max(f326KNoFma.abs, f326KFma.abs);

sayFloor("q6K idot", idot6KNoFma, idot6KFma,
  PREFILL_GEMM_Q6K_IDOT_REL_TOL, PREFILL_GEMM_Q6K_IDOT_ABS_TOL);
sayFloor("q6K f32 ", f326KNoFma, f326KFma,
  PREFILL_GEMM_Q6K_F32_REL_TOL, PREFILL_GEMM_Q6K_F32_ABS_TOL);
say(`[q6K MUTATA senza l'offset -32] idot rel=${e(idot6KMut.rel)} abs=${e(idot6KMut.abs)} | ` +
  `f32 rel=${e(f326KMut.rel)} abs=${e(f326KMut.abs)}`);

say(`[q6K |y|] idot medio=${e(meanAbs(REF6K.idot))} f32 medio=${e(meanAbs(REF6K.f32))}`);

// ---------------------------------------------------------------------------

describe("prefill gemm q6_K: struttura del WGSL generato", () => {
  const idot6k = prefillGemmQ6KSplitKIdotWgsl(OPTS6K);
  const f32k6k = prefillGemmQ6KSplitKWgsl(OPTS6K);

  it("il caso del ktest e' la geometria che il piano sceglie da solo", () => {
    // DUE fette, non quattro: e' il ramo di ripiego dei K-quant, ed e' il piano
    // a sceglierlo su K=512 — il caso non lo impone.
    expect(prefillGemmSplitsFor(K6K, N6K, "q6_K")).toBe(SPLITS6K);
    expect(SPLITS6K).toBe(2);
    expect(K6K % 256).toBe(0);
    expect(N6K % 64).not.toBe(0);
    expect(idot6k).toContain(`const N_ROWS = ${N6K}u;`);
    expect(idot6k).toContain("if (r < N_ROWS) {");
    expect(f32k6k).toContain("if (r < N_ROWS) {");
  });

  it("le costanti dei due kernel sono quelle che l'emulazione presume", () => {
    for (const s of [idot6k, f32k6k]) {
      expect(s).toContain("const WORDS = 53u;");   // 210 B + 2 di pad = 53 parole
      expect(s).toContain("const SBPR = 2u;");     // K=512 / 256
      expect(s).toContain("const PER = 1u;");      // 2 superblocchi / 2 fette
      expect(s).toContain(`const M_ROWS = ${M6K}u;`);
    }
    expect(SBPR6K).toBe(2);
    expect(PER6K).toBe(1);
    expect(idot6k).not.toContain("enable packed_4x8_integer_dot_product");
    expect(f32k6k).not.toContain("enable packed_4x8_integer_dot_product");
  });

  it("`d` sta IN CODA (parola 52) e le 16 scale sono int8, non f16", () => {
    // l'emulazione legge `d` a byte 208-209 e le scale con estensione di segno
    // da byte 192: se il kernel le leggesse altrimenti, il pavimento qui sotto
    // misurerebbe una copia privata.
    for (const s of [idot6k, f32k6k]) {
      expect(s).toContain("let d = unpack2x16float(blocks[wb + 52u]).x;");
      expect(s).toContain("return f32((i32(sbyte(base, i)) << 24u) >> 24u);");
      expect(s).toContain("let scO = 192u + n * 8u;");
    }
  });

  it("l'offset -32 sta in DUE FORME, una per via", () => {
    // e' il fatto che rende la mutazione qui sotto un discriminante di ENTRAMBE
    // le strade algebriche e non di una sola.
    expect(idot6k).toContain("32.0 * xh[(m * 8u + blk) * 2u]");   // fattorizzato
    expect(idot6k).not.toContain(") - 32.0;");
    expect(f32k6k).toContain("<< 4u)) - 32.0;");                  // per elemento
    expect(f32k6k).not.toContain("xh");
  });

  it("gli array di workgroup sono quelli delle due vie", () => {
    // via intera: il superblocco INTERO di attivazioni i8 + la scala per blocco
    // + `xh`, che e' per MEZZO sotto-blocco (M x 16) e non per blocco
    expect(idot6k).toContain("var<workgroup> xs: array<u32, 1024>;");
    expect(idot6k).toContain("var<workgroup> xss: array<f32, 128>;");
    expect(idot6k).toContain("var<workgroup> xh: array<f32, 256>;");
    expect(idot6k).not.toContain("var<workgroup> xsum:");
    // via f32: UN sotto-blocco da 32 per volta (M x 32 f32)
    expect(f32k6k).toContain("var<workgroup> xs: array<f32, 512>;");
    expect(f32k6k).not.toContain("array<u32,");
  });

  it("quantizzatore e combine emulati sono quelli veri", () => {
    expect(prefillQuantXQ8Wgsl({ K: K6K, M: M6K })).toContain(`const BLOCKS = ${M6K * BPR6K}u;`);
    expect(prefillQuantXQ8Wgsl({ K: K6K, M: M6K })).toContain("let sc = amax / 127.0;");
    expect(prefillSplitKCombineWgsl({ N: N6K, M: M6K, splits: SPLITS6K }))
      .toContain(`const S = ${SPLITS6K}u;`);
    expect(prefillSplitKCombineWgsl({ N: N6K, M: M6K, splits: SPLITS6K }))
      .toContain(`const TOTAL = ${M6K * N6K}u;`);
  });
});

describe("pavimento f32: prefill gemm q6_K via intera (dot4I8Packed)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [idot6KNoFma.rel, idot6KFma.rel, idot6KNoFma.abs, idot6KFma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorQ6KIdotRel).toBe(Math.max(idot6KNoFma.rel, idot6KFma.rel));
    expect(floorQ6KIdotAbs).toBe(Math.max(idot6KNoFma.abs, idot6KFma.abs));
    expect(idot6KNoFma.rel / idot6KFma.rel).toBeLessThan(10);
    expect(idot6KFma.rel / idot6KNoFma.rel).toBeLessThan(10);
    expect(idot6KNoFma.abs / idot6KFma.abs).toBeLessThan(10);
    expect(idot6KFma.abs / idot6KNoFma.abs).toBeLessThan(10);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q6K_IDOT_REL_TOL).toBeGreaterThanOrEqual(floorQ6KIdotRel);
    expect(PREFILL_GEMM_Q6K_IDOT_REL_TOL).toBeLessThanOrEqual(20 * floorQ6KIdotRel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q6K_IDOT_ABS_TOL).toBeGreaterThanOrEqual(floorQ6KIdotAbs);
    expect(PREFILL_GEMM_Q6K_IDOT_ABS_TOL).toBeLessThanOrEqual(20 * floorQ6KIdotAbs);
  });
});

describe("pavimento f32: prefill gemm q6_K via f32 (fallback)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [f326KNoFma.rel, f326KFma.rel, f326KNoFma.abs, f326KFma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorQ6KF32Rel).toBe(Math.max(f326KNoFma.rel, f326KFma.rel));
    expect(floorQ6KF32Abs).toBe(Math.max(f326KNoFma.abs, f326KFma.abs));
    expect(f326KNoFma.rel / f326KFma.rel).toBeLessThan(10);
    expect(f326KFma.rel / f326KNoFma.rel).toBeLessThan(10);
    expect(f326KNoFma.abs / f326KFma.abs).toBeLessThan(10);
    expect(f326KFma.abs / f326KNoFma.abs).toBeLessThan(10);
  });

  it("il pavimento RELATIVO della via f32 e' una CANCELLAZIONE, non un'anomalia", () => {
    // Il fatto misurato: sul braccio f32 il caso peggiore relativo (1,118e-2)
    // cade sull'uscita di modulo piu' PICCOLO dell'intera griglia — 4,395e-2
    // contro una media di 1,340e3, cinque ordini sotto — dove l'errore ASSOLUTO
    // vale 4,914e-4, cioe' dentro il pavimento assoluto di questa stessa via.
    // Il braccio intero non ci cade perche' il suo riferimento e' un altro (le
    // attivazioni dopo il giro a i8), e la sua uscita piu' piccola vale 7,469e-1.
    //
    // Da qui la forma del gate: le due metriche DIVERGONO sul relativo (piu' di
    // un ordine) e COINCIDONO sull'assoluto (entro il 20%). E' esattamente il
    // caso per cui la `compare` del ktest passa su `maxAbs <= absTol OPPURE
    // maxRel <= relTol`: su questo formato e' il ramo ASSOLUTO a portare il
    // giudizio, e la tolleranza relativa a 1,5e-1 non e' larga per compiacenza
    // ma perche' un solo denominatore vicino allo zero la fissa li'.
    expect(floorQ6KF32Rel / floorQ6KIdotRel).toBeGreaterThan(10);
    expect(floorQ6KF32Abs / floorQ6KIdotAbs).toBeLessThan(2);
    expect(floorQ6KIdotAbs / floorQ6KF32Abs).toBeLessThan(2);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q6K_F32_REL_TOL).toBeGreaterThanOrEqual(floorQ6KF32Rel);
    expect(PREFILL_GEMM_Q6K_F32_REL_TOL).toBeLessThanOrEqual(20 * floorQ6KF32Rel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q6K_F32_ABS_TOL).toBeGreaterThanOrEqual(floorQ6KF32Abs);
    expect(PREFILL_GEMM_Q6K_F32_ABS_TOL).toBeLessThanOrEqual(20 * floorQ6KF32Abs);
  });
});

describe("discriminante q6_K: la tolleranza non copre un difetto strutturale", () => {
  // La `compare` del ktest passa se `maxAbs <= absTol OPPURE maxRel <= relTol`:
  // perche' la mutazione sia CATTURATA deve sfondare ENTRAMBE le tolleranze.
  it("togliere l'offset -32 FATTORIZZATO sfonda le tolleranze della via intera di 10x", () => {
    expect(idot6KMut.rel / PREFILL_GEMM_Q6K_IDOT_REL_TOL).toBeGreaterThan(10);
    expect(idot6KMut.abs / PREFILL_GEMM_Q6K_IDOT_ABS_TOL).toBeGreaterThan(10);
  });

  it("togliere l'offset -32 DAL PESO sfonda le tolleranze della via f32 di 10x", () => {
    expect(f326KMut.rel / PREFILL_GEMM_Q6K_F32_REL_TOL).toBeGreaterThan(10);
    expect(f326KMut.abs / PREFILL_GEMM_Q6K_F32_ABS_TOL).toBeGreaterThan(10);
  });
});

describe("anti-ricopiatura: i pavimenti q6_K non sono quelli degli altri tre", () => {
  it("le due vie del q6_K hanno pavimenti diversi da q5_K, q4_1 e q4_K", () => {
    for (const [what, mine, others] of [
      ["idot rel", floorQ6KIdotRel, [floorIdotRel, floorQ41IdotRel, floorQ4KIdotRel]],
      ["idot abs", floorQ6KIdotAbs, [floorIdotAbs, floorQ41IdotAbs, floorQ4KIdotAbs]],
      ["f32 rel", floorQ6KF32Rel, [floorF32Rel, floorQ41F32Rel, floorQ4KF32Rel]],
      ["f32 abs", floorQ6KF32Abs, [floorF32Abs, floorQ41F32Abs, floorQ4KF32Abs]],
    ] as Array<[string, number, number[]]>) {
      for (const o of others) expect(mine, what).not.toBe(o);
    }
  });

  it("il caso q6_K non condivide ne' K ne' seed con gli altri", () => {
    expect(new Set([C.K, C41.K, C4K.K, C6K.K]).size).toBe(4);
    expect(new Set([
      C.seedBlocks, C.seedX, C41.seedBlocks, C41.seedX,
      C4K.seedBlocks, C4K.seedX, C6K.seedBlocks, C6K.seedX,
    ]).size).toBe(8);
  });
});

// ===========================================================================
// PREFILL GEMM q8_0 — la terza forma della riga 4, e la piu' semplice delle sei:
// i pesi SONO gia' i8.
//
// COSA CAMBIA:
//   - niente unpack e NIENTE TERMINE COSTANTE. Il risultato e' `d * Sigma(q*x)`
//     e basta: nessun minimo, nessun offset. La via intera e' `dot4I8Packed`
//     nudo su otto parole, e la scala del peso per la scala di x si applica una
//     volta per blocco;
//   - i pesi vivono in DUE buffer (`repackQ8_0` → `qs` + `scales`, con le scale
//     a DUE f16 per parola come il q4_0), non in uno solo come i tre K-quant;
//   - K=2048 fa 64 blocchi da 32 per riga in 4 fette da PER=16, e PER dev'essere
//     PARI perche' il ciclo avanza a due blocchi per giro.
//
// LA MUTAZIONE NON PUO' ESSERE «TOGLIERE UN TERMINE», e non e' una difficolta'
// da aggirare: e' l'informazione. Il q4_1 ha `+m*Sigma(x)`, i due K-quant con
// minimo hanno `-dmin*mn*Sigma(x)`, il q6_K ha l'offset -32; il q8_0 NON HA
// NIENTE da togliere, e una mutazione che togliesse la scala `d` sarebbe una
// tautologia (il risultato andrebbe a zero, e qualunque tolleranza lo
// prenderebbe).
//
// LA MUTAZIONE E' IL SEGNO DEI PESI: leggere gli otto byte del blocco come u8
// invece che come i8. E' la scelta giusta per QUESTO formato per tre ragioni
// misurabili sul testo del kernel:
//   1. e' l'unico dei sei in cui i BYTE DEI PESI sono con segno. Gli altri
//      cinque impacchettano nibble o sestine SENZA segno e portano la centratura
//      altrove (un minimo, un offset): qui il segno E' il formato;
//   2. e' a un carattere di distanza. La via intera chiama `dot4I8Packed`, e
//      `dot4U8Packed` esiste nella stessa language feature; la via f32 estende
//      il segno a mano con `i32(w << ((3u - by) * 8u)) >> 24u`, e la forma
//      sbagliata — `(w >> (by * 8u)) & 0xffu` — e' quella che si scrive per
//      istinto leggendo un byte;
//   3. non e' una perturbazione che si cancella. Il peso letto senza segno vale
//      `q + 256` per ogni byte negativo, quindi l'errore e' `256 * Sigma_{q<0}
//      x`: una somma a segni casuali, non un termine che si annulla in media.
// E la si applica ALLE DUE VIE, che qui — al contrario del q6_K — calcolano
// davvero la stessa espressione: il discriminante prova che nessuna delle due
// legge i pesi come se fossero senza segno.
// ===========================================================================

const C80 = PREFILL_Q80_KTEST_CASE;
const K80 = C80.K, N80 = C80.N, M80 = C80.M, SPLITS80 = C80.splits;
const BPR80 = K80 / Q8_0_BLOCK_WEIGHTS;   // blocchi da 32 per riga di pesi
const PER80 = BPR80 / SPLITS80;           // blocchi per fetta (PARI: il ciclo va a due)
const OPTS80 = { kind: "q8_0", K: K80, N: N80, M: M80, splits: SPLITS80 } as const;

const nBlocks80 = BPR80 * N80;
const src80 = randBytes(nBlocks80 * Q8_0_BLOCK_BYTES, C80.seedBlocks);
fixScalesAt(src80, Q8_0_BLOCK_BYTES, [1]);   // UNA scala f16 per blocco, come il q4_0
const w80 = new Float32Array(nBlocks80 * Q8_0_BLOCK_WEIGHTS);
dequantQ8_0(src80, 0, nBlocks80, w80);
const x80A = randF32(M80 * K80, C80.seedX);
const A80 = quantizeActivations(x80A, M80 * BPR80);
const REF80 = refsOf(w80, x80A, A80.xdq, K80, N80, M80);

const q80 = new Int32Array(32);

// `prefillGemmQ80SplitKIdotWgsl`, una riga di pesi r: prodotto scalare INTERO
// esatto su otto parole e poi `acc + sc*f32(idot)*xss` associato a sinistra,
// come lo scrive il WGSL. Il passo a due blocchi per giro non cambia
// l'aritmetica — cambia quali blocchi stanno in memoria di gruppo — quindi qui i
// blocchi si scorrono uno per uno nello stesso ordine.
// `unsigned` e' la MUTAZIONE: i byte del peso letti come u8 invece che come i8.
function emulateQ80IdotRow(r: number, fma: boolean, unsigned: boolean): Float32Array {
  const part = new Float32Array(SPLITS80 * M80);
  const acc = new Float64Array(M80);
  for (let s = 0; s < SPLITS80; s++) {
    acc.fill(0);
    for (let gb = s * PER80; gb < s * PER80 + PER80; gb++) {
      const o = (r * BPR80 + gb) * Q8_0_BLOCK_BYTES;
      const sc = f16ToF32(src80[o] | (src80[o + 1] << 8));
      for (let l = 0; l < 32; l++) {
        const by = src80[o + 2 + l];
        q80[l] = unsigned ? by : i8(by);
      }
      for (let m = 0; m < M80; m++) {
        const b = m * BPR80 + gb;
        let idot = 0;
        for (let l = 0; l < 32; l++) idot += q80[l] * A80.xq[b * 32 + l];
        acc[m] = f(acc[m] + term(f(sc * idot), A80.xsc[b], fma));
      }
    }
    for (let m = 0; m < M80; m++) part[s * M80 + m] = acc[m];
  }
  return combineAt(part, M80, SPLITS80);
}

const q80f = new Float64Array(32);

// `prefillGemmQ80SplitKWgsl`: i 32 pesi del blocco convertiti da i8 in registri,
// `qx` accumulato in f32 nell'ordine del loop, poi `acc + sc*qx`.
function emulateQ80F32Row(r: number, fma: boolean, unsigned: boolean): Float32Array {
  const part = new Float32Array(SPLITS80 * M80);
  const acc = new Float64Array(M80);
  for (let s = 0; s < SPLITS80; s++) {
    acc.fill(0);
    for (let gb = s * PER80; gb < s * PER80 + PER80; gb++) {
      const o = (r * BPR80 + gb) * Q8_0_BLOCK_BYTES;
      const sc = f16ToF32(src80[o] | (src80[o + 1] << 8));
      for (let l = 0; l < 32; l++) {
        const by = src80[o + 2 + l];
        q80f[l] = unsigned ? by : i8(by);
      }
      for (let m = 0; m < M80; m++) {
        const base = m * K80 + gb * 32;
        let qx = 0;
        for (let l = 0; l < 32; l++) qx = f(qx + term(q80f[l], x80A[base + l], fma));
        acc[m] = f(acc[m] + term(sc, qx, fma));
      }
    }
    for (let m = 0; m < M80; m++) part[s * M80 + m] = acc[m];
  }
  return combineAt(part, M80, SPLITS80);
}

const idot80NoFma = measureAt(N80, M80, emulateQ80IdotRow, REF80.idot, false, false);
const idot80Fma = measureAt(N80, M80, emulateQ80IdotRow, REF80.idot, true, false);
const idot80Mut = measureAt(N80, M80, emulateQ80IdotRow, REF80.idot, true, true);
const f3280NoFma = measureAt(N80, M80, emulateQ80F32Row, REF80.f32, false, false);
const f3280Fma = measureAt(N80, M80, emulateQ80F32Row, REF80.f32, true, false);
const f3280Mut = measureAt(N80, M80, emulateQ80F32Row, REF80.f32, true, true);

const floorQ80IdotRel = Math.max(idot80NoFma.rel, idot80Fma.rel);
const floorQ80IdotAbs = Math.max(idot80NoFma.abs, idot80Fma.abs);
const floorQ80F32Rel = Math.max(f3280NoFma.rel, f3280Fma.rel);
const floorQ80F32Abs = Math.max(f3280NoFma.abs, f3280Fma.abs);

sayFloor("q80 idot", idot80NoFma, idot80Fma,
  PREFILL_GEMM_Q80_IDOT_REL_TOL, PREFILL_GEMM_Q80_IDOT_ABS_TOL);
sayFloor("q80 f32 ", f3280NoFma, f3280Fma,
  PREFILL_GEMM_Q80_F32_REL_TOL, PREFILL_GEMM_Q80_F32_ABS_TOL);
say(`[q80 MUTATA pesi letti u8 invece di i8] idot rel=${e(idot80Mut.rel)} abs=${e(idot80Mut.abs)} | ` +
  `f32 rel=${e(f3280Mut.rel)} abs=${e(f3280Mut.abs)}`);
say(`[q80 |y|] idot medio=${e(meanAbs(REF80.idot))} f32 medio=${e(meanAbs(REF80.f32))}`);

// ---------------------------------------------------------------------------

describe("prefill gemm q8_0: struttura del WGSL generato", () => {
  const idot80 = prefillGemmQ80SplitKIdotWgsl(OPTS80);
  const f32k80 = prefillGemmQ80SplitKWgsl(OPTS80);

  it("il caso del ktest e' la geometria che il piano sceglie da solo", () => {
    expect(prefillGemmSplitsFor(K80, N80, "q8_0")).toBe(SPLITS80);
    expect(K80 % 64).toBe(0);
    // PER PARI: il ciclo avanza a due blocchi per giro, e una fetta dispari
    // leggerebbe un blocco della fetta successiva
    expect(PER80 % 2).toBe(0);
    expect(N80 % 64).not.toBe(0);
    expect(idot80).toContain(`const N_ROWS = ${N80}u;`);
    expect(idot80).toContain("if (r < N_ROWS) {");
    expect(f32k80).toContain("if (r < N_ROWS) {");
  });

  it("le costanti dei due kernel sono quelle che l'emulazione presume", () => {
    for (const s of [idot80, f32k80]) {
      expect(s).toContain("const BPR = 64u;");     // K=2048 / 32
      expect(s).toContain("const PER = 16u;");     // 64 blocchi / 4 fette
      expect(s).toContain(`const N_ROWS = ${N80}u;`);
      expect(s).toContain(`const M_ROWS = ${M80}u;`);
      // il passo a DUE blocchi per giro, che e' cio' che rende PER pari un
      // vincolo e non un'estetica
      expect(s).toContain("b0 = b0 + 2u;");
    }
    expect(BPR80).toBe(64);
    expect(PER80).toBe(16);
    expect(idot80).not.toContain("enable packed_4x8_integer_dot_product");
    expect(f32k80).not.toContain("enable packed_4x8_integer_dot_product");
  });

  it("DUE buffer di pesi e le scale a DUE f16 per parola, come il q4_0", () => {
    for (const s of [idot80, f32k80]) {
      expect(s).toContain("var<storage, read> qs: array<u32>;");
      expect(s).toContain("var<storage, read> scales: array<u32>;");
      // `scales[gb >> 1u][gb & 1u]`: due blocchi per parola. Sul q4_1 la stessa
      // riga sarebbe l'errore di formato (li' e' UNA parola per blocco).
      expect(s).toContain("let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];");
    }
  });

  it("i pesi si leggono CON SEGNO, e niente termine costante", () => {
    // il segno E' il formato: e' quello che la mutazione qui sotto toglie.
    expect(idot80).toContain("dot4I8Packed(w8[ii], xs[xo + ii])");
    expect(idot80).not.toContain("dot4U8Packed");
    expect(f32k80).toContain("q[ii * 4u + by] = f32((i32(w << ((3u - by) * 8u)) >> 24u));");
    // e nessuna riduzione delle attivazioni oltre la scala: il q8_0 non ha un
    // `xsum` ne' un `xh`, perche' non ha niente da moltiplicare per Sigma(x)
    for (const s of [idot80, f32k80]) {
      expect(s).not.toContain("var<workgroup> xsum:");
      expect(s).not.toContain("var<workgroup> xh:");
    }
  });

  it("gli array di workgroup sono quelli delle due vie", () => {
    // via intera: DUE blocchi di attivazioni i8 (M x 16 parole) + le due scale
    expect(idot80).toContain("var<workgroup> xs: array<u32, 256>;");
    expect(idot80).toContain("var<workgroup> xss: array<f32, 32>;");
    // via f32: DUE blocchi da 32 in virgola mobile (M x 64 f32), letti densi
    expect(f32k80).toContain("var<workgroup> xs: array<f32, 1024>;");
    expect(f32k80).not.toContain("array<u32,");
  });

  it("quantizzatore e combine emulati sono quelli veri", () => {
    expect(prefillQuantXQ8Wgsl({ K: K80, M: M80 })).toContain(`const BLOCKS = ${M80 * BPR80}u;`);
    expect(prefillQuantXQ8Wgsl({ K: K80, M: M80 })).toContain("let sc = amax / 127.0;");
    expect(prefillSplitKCombineWgsl({ N: N80, M: M80, splits: SPLITS80 }))
      .toContain(`const S = ${SPLITS80}u;`);
    expect(prefillSplitKCombineWgsl({ N: N80, M: M80, splits: SPLITS80 }))
      .toContain(`const TOTAL = ${M80 * N80}u;`);
  });
});

describe("pavimento f32: prefill gemm q8_0 via intera (dot4I8Packed)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [idot80NoFma.rel, idot80Fma.rel, idot80NoFma.abs, idot80Fma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorQ80IdotRel).toBe(Math.max(idot80NoFma.rel, idot80Fma.rel));
    expect(floorQ80IdotAbs).toBe(Math.max(idot80NoFma.abs, idot80Fma.abs));
    expect(idot80NoFma.rel / idot80Fma.rel).toBeLessThan(10);
    expect(idot80Fma.rel / idot80NoFma.rel).toBeLessThan(10);
    expect(idot80NoFma.abs / idot80Fma.abs).toBeLessThan(10);
    expect(idot80Fma.abs / idot80NoFma.abs).toBeLessThan(10);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q80_IDOT_REL_TOL).toBeGreaterThanOrEqual(floorQ80IdotRel);
    expect(PREFILL_GEMM_Q80_IDOT_REL_TOL).toBeLessThanOrEqual(20 * floorQ80IdotRel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q80_IDOT_ABS_TOL).toBeGreaterThanOrEqual(floorQ80IdotAbs);
    expect(PREFILL_GEMM_Q80_IDOT_ABS_TOL).toBeLessThanOrEqual(20 * floorQ80IdotAbs);
  });
});

describe("pavimento f32: prefill gemm q8_0 via f32 (fallback)", () => {
  it("i due modelli di contrazione sono entrambi reali e vicini", () => {
    for (const v of [f3280NoFma.rel, f3280Fma.rel, f3280NoFma.abs, f3280Fma.abs]) {
      expect(v).toBeGreaterThan(0);
    }
    expect(floorQ80F32Rel).toBe(Math.max(f3280NoFma.rel, f3280Fma.rel));
    expect(floorQ80F32Abs).toBe(Math.max(f3280NoFma.abs, f3280Fma.abs));
    expect(f3280NoFma.rel / f3280Fma.rel).toBeLessThan(10);
    expect(f3280Fma.rel / f3280NoFma.rel).toBeLessThan(10);
    expect(f3280NoFma.abs / f3280Fma.abs).toBeLessThan(10);
    expect(f3280Fma.abs / f3280NoFma.abs).toBeLessThan(10);
  });

  it("REL: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q80_F32_REL_TOL).toBeGreaterThanOrEqual(floorQ80F32Rel);
    expect(PREFILL_GEMM_Q80_F32_REL_TOL).toBeLessThanOrEqual(20 * floorQ80F32Rel);
  });

  it("ABS: pavimento <= tolleranza <= 20x pavimento", () => {
    expect(PREFILL_GEMM_Q80_F32_ABS_TOL).toBeGreaterThanOrEqual(floorQ80F32Abs);
    expect(PREFILL_GEMM_Q80_F32_ABS_TOL).toBeLessThanOrEqual(20 * floorQ80F32Abs);
  });
});

describe("discriminante q8_0: la tolleranza non copre un difetto strutturale", () => {
  // Il q8_0 NON HA un termine costante da togliere: la mutazione e' di natura
  // diversa da quelle degli altri quattro formati — i pesi letti come u8 invece
  // che come i8, cioe' `dot4U8Packed` al posto di `dot4I8Packed` sulla via
  // intera e un byte estratto senza estensione di segno su quella f32.
  it("leggere i pesi come u8 sfonda le tolleranze della via intera di 10x", () => {
    expect(idot80Mut.rel / PREFILL_GEMM_Q80_IDOT_REL_TOL).toBeGreaterThan(10);
    expect(idot80Mut.abs / PREFILL_GEMM_Q80_IDOT_ABS_TOL).toBeGreaterThan(10);
  });

  it("leggere i pesi come u8 sfonda le tolleranze della via f32 di 10x", () => {
    expect(f3280Mut.rel / PREFILL_GEMM_Q80_F32_REL_TOL).toBeGreaterThan(10);
    expect(f3280Mut.abs / PREFILL_GEMM_Q80_F32_ABS_TOL).toBeGreaterThan(10);
  });

  it("l'errore mutato e' dello stesso ORDINE dell'uscita: non si cancella", () => {
    // La ragione per cui questa mutazione discrimina, in una riga misurata: il
    // peso letto senza segno vale `q + 256` su ogni byte negativo, quindi
    // l'errore e' `256 * Sigma_{q<0} x` — una somma a segni casuali. Se invece
    // si cancellasse in media, l'errore assoluto mutato starebbe MOLTO sotto il
    // modulo tipico dell'uscita, e questo banco tollererebbe un kernel che legge
    // i pesi al contrario.
    const y = meanAbs(REF80.idot);
    expect(idot80Mut.abs / y).toBeGreaterThan(0.5);
    expect(f3280Mut.abs / meanAbs(REF80.f32)).toBeGreaterThan(0.5);
  });
});

describe("anti-ricopiatura: i pavimenti q8_0 non sono quelli degli altri quattro", () => {
  it("le due vie del q8_0 hanno pavimenti diversi da q5_K, q4_1, q4_K e q6_K", () => {
    for (const [what, mine, others] of [
      ["idot rel", floorQ80IdotRel, [floorIdotRel, floorQ41IdotRel, floorQ4KIdotRel, floorQ6KIdotRel]],
      ["idot abs", floorQ80IdotAbs, [floorIdotAbs, floorQ41IdotAbs, floorQ4KIdotAbs, floorQ6KIdotAbs]],
      ["f32 rel", floorQ80F32Rel, [floorF32Rel, floorQ41F32Rel, floorQ4KF32Rel, floorQ6KF32Rel]],
      ["f32 abs", floorQ80F32Abs, [floorF32Abs, floorQ41F32Abs, floorQ4KF32Abs, floorQ6KF32Abs]],
    ] as Array<[string, number, number[]]>) {
      for (const o of others) expect(mine, what).not.toBe(o);
    }
  });

  it("i cinque casi sono cinque esperimenti, e i dieci seed sono dieci", () => {
    // K si ripete (2048 sul q4_K e sul q8_0: sono due shape vere di due tensori
    // diversi del 35B), ma i seed no — ed e' il seed a rendere due esperimenti
    // indipendenti, non la shape.
    expect(new Set([
      C.seedBlocks, C.seedX, C41.seedBlocks, C41.seedX, C4K.seedBlocks, C4K.seedX,
      C6K.seedBlocks, C6K.seedX, C80.seedBlocks, C80.seedX,
    ]).size).toBe(10);
    expect(new Set([C.K, C41.K, C4K.K, C6K.K, C80.K]).size).toBe(4);
    expect(C80.K).toBe(C4K.K);   // due formati, due tensori, la stessa K vera
  });
});
