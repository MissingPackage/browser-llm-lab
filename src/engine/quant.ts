// Q4_0 — layout e dequant CPU di riferimento (fase A).
//
// Blocco Q4_0 (18 byte, 32 pesi): [scala f16 LE (2 B)] [16 B di nibbles].
// Peso j (0..31): nibble = j<16 ? low(byte[j]) : high(byte[j-16]); w = (nibble-8)*scala.
// La dequant in f32 è ESATTA (nessun arrotondamento oltre la scala f16): qualunque
// divergenza dal riferimento è un bug di layout, non rumore numerico — è la base
// dell'unit bit-exact richiesto dalla spec (§Soglie di conformance).
//
// Questo modulo è il riferimento CPU; il repack GPU-friendly (fase 4) e il kernel
// WGSL dovranno produrre ESATTAMENTE questi valori.

export const Q4_0_BLOCK_WEIGHTS = 32;
export const Q4_0_BLOCK_BYTES = 18;

// f16 (bits u16) → f32. Implementazione locale: niente dipendenza da shader-f16 né
// da Float16Array (copre anche denormali e inf/nan per completezza del riferimento).
export function f16ToF32(bits: number): number {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24; // denormale (o zero)
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

// Dequant di `nBlocks` blocchi Q4_0 consecutivi da `src` (offset in byte) in `dst`.
// Ritorna il numero di float scritti.
export function dequantQ4_0(
  src: Uint8Array, srcOffset: number, nBlocks: number, dst: Float32Array, dstOffset = 0,
): number {
  let o = srcOffset;
  let w = dstOffset;
  for (let b = 0; b < nBlocks; b++) {
    const scale = f16ToF32(src[o] | (src[o + 1] << 8));
    o += 2;
    for (let j = 0; j < 16; j++) {
      const byte = src[o + j];
      dst[w + j] = ((byte & 0x0f) - 8) * scale;       // low nibble → peso j
      dst[w + 16 + j] = ((byte >> 4) - 8) * scale;    // high nibble → peso j+16
    }
    o += 16;
    w += Q4_0_BLOCK_WEIGHTS;
  }
  return w - dstOffset;
}

// Dequant di una riga di un tensore Q4_0 (ne[0] = rowElems multiplo di 32).
export function dequantQ4_0Row(
  data: Uint8Array, tensorByteOffset: number, rowElems: number, row: number, dst: Float32Array,
): void {
  const blocksPerRow = rowElems / Q4_0_BLOCK_WEIGHTS;
  const rowBytes = blocksPerRow * Q4_0_BLOCK_BYTES;
  dequantQ4_0(data, tensorByteOffset + row * rowBytes, blocksPerRow, dst, 0);
}

// --- Repack GPU-friendly (spec §Formato pesi) ---
//
// I blocchi GGUF (18/34 B) non sono allineati a u32: si repacka una volta al load in
// due buffer per tensore: `qs` (nibbles/int8 come u32 LE, 4-allineati) e `scales`
// (bit f16 grezzi, due per u32 — in WGSL si legge con unpack2x16float, che NON
// richiede shader-f16). Il kernel deve produrre ESATTAMENTE i valori del reference.
export interface RepackedQuant {
  qs: Uint32Array; // Q4_0: 4 u32/blocco (16 B nibbles) · Q8_0: 8 u32/blocco (32 int8)
  scales: Uint32Array; // ⌈nBlocks/2⌉ u32: bit f16 del blocco b in half (b&1) di word (b>>1)
}

function packScale(scales: Uint32Array, b: number, f16bits: number): void {
  const w = b >> 1;
  scales[w] = (b & 1) ? (scales[w] | (f16bits << 16)) : (scales[w] | f16bits);
}

export function repackQ4_0(src: Uint8Array, srcOffset: number, nBlocks: number): RepackedQuant {
  const qs = new Uint32Array(nBlocks * 4);
  const scales = new Uint32Array(Math.ceil(nBlocks / 2));
  let o = srcOffset;
  for (let b = 0; b < nBlocks; b++) {
    packScale(scales, b, src[o] | (src[o + 1] << 8));
    o += 2;
    for (let w = 0; w < 4; w++) {
      qs[b * 4 + w] = src[o] | (src[o + 1] << 8) | (src[o + 2] << 16) | (src[o + 3] << 24);
      o += 4;
    }
  }
  return { qs, scales };
}

export function repackQ8_0(src: Uint8Array, srcOffset: number, nBlocks: number): RepackedQuant {
  const qs = new Uint32Array(nBlocks * 8);
  const scales = new Uint32Array(Math.ceil(nBlocks / 2));
  let o = srcOffset;
  for (let b = 0; b < nBlocks; b++) {
    packScale(scales, b, src[o] | (src[o + 1] << 8));
    o += 2;
    for (let w = 0; w < 8; w++) {
      qs[b * 8 + w] = src[o] | (src[o + 1] << 8) | (src[o + 2] << 16) | (src[o + 3] << 24);
      o += 4;
    }
  }
  return { qs, scales };
}

// Q4_1 (20 byte, 32 pesi): [d f16] [m f16] [16 B nibbles]. w = nibble*d + m
// (nibble NON centrato: 0..15). Layout GLM-4.7-Flash: ffn_down_exps blk.1-4 e
// ffn_down denso (spec C2 §1). Riferimento: dequantize_row_q4_1, ggml-quants.c
// (oracolo 5f55650).
export const Q4_1_BLOCK_WEIGHTS = 32;
export const Q4_1_BLOCK_BYTES = 20;

export function dequantQ4_1(
  src: Uint8Array, srcOffset: number, nBlocks: number, dst: Float32Array, dstOffset = 0,
): number {
  let o = srcOffset;
  let w = dstOffset;
  for (let b = 0; b < nBlocks; b++) {
    const d = f16ToF32(src[o] | (src[o + 1] << 8));
    const m = f16ToF32(src[o + 2] | (src[o + 3] << 8));
    o += 4;
    for (let j = 0; j < 16; j++) {
      const byte = src[o + j];
      dst[w + j] = (byte & 0x0f) * d + m;
      dst[w + 16 + j] = (byte >> 4) * d + m;
    }
    o += 16;
    w += Q4_1_BLOCK_WEIGHTS;
  }
  return w - dstOffset;
}

// --- K-quants (superblocco QK_K=256) — layout ggml-common.h, oracolo 5f55650 ---

export const QK_K = 256;

// get_scale_min_k4: 8 coppie (scala,min) a 6 bit impacchettate in 12 byte.
function scaleMinK4(j: number, q: Uint8Array, qo: number): [number, number] {
  if (j < 4) return [q[qo + j] & 63, q[qo + j + 4] & 63];
  return [
    (q[qo + j + 4] & 0x0f) | ((q[qo + j - 4] >> 6) << 4),
    (q[qo + j + 4] >> 4) | ((q[qo + j] >> 6) << 4),
  ];
}

// Q5_K (176 byte, 256 pesi): [d f16][dmin f16][scales 12 B][qh 32 B][qs 128 B].
// w = d*sc * (q5) - dmin*m, q5 = nibble | bit alto da qh. Riferimento:
// dequantize_row_q5_K. Layout GLM: ffn_gate_shexp/up_shexp.
export const Q5_K_BLOCK_WEIGHTS = QK_K;
export const Q5_K_BLOCK_BYTES = 176;

// Q4_K (144 byte, 256 pesi): [d f16][dmin f16][scales 12 B 6-bit][qs 128 B].
// w = d·sc·q4 − dmin·m — è Q5_K SENZA il piano qh. Riferimento:
// dequantize_row_q4_K. Layout q1: gli EXPERT del 35B-A3B UD-Q4_K_S
// (117/120 tensori expert; spec q1 §2-3, header dump 2026-08-10).
export const Q4_K_BLOCK_WEIGHTS = QK_K;
export const Q4_K_BLOCK_BYTES = 144;

export function dequantQ4_K(
  src: Uint8Array, srcOffset: number, nBlocks: number, dst: Float32Array, dstOffset = 0,
): number {
  let w = dstOffset;
  for (let b = 0; b < nBlocks; b++) {
    const o = srcOffset + b * Q4_K_BLOCK_BYTES;
    const d = f16ToF32(src[o] | (src[o + 1] << 8));
    const dmin = f16ToF32(src[o + 2] | (src[o + 3] << 8));
    const scalesO = o + 4;
    const qsO = o + 16;
    let is = 0;
    let ql = 0;
    for (let j = 0; j < QK_K; j += 64) {
      const [sc1, m1] = scaleMinK4(is, src, scalesO);
      const [sc2, m2] = scaleMinK4(is + 1, src, scalesO);
      const d1 = d * sc1, min1 = dmin * m1;
      const d2 = d * sc2, min2 = dmin * m2;
      for (let l = 0; l < 32; l++) dst[w++] = d1 * (src[qsO + ql + l] & 0x0f) - min1;
      for (let l = 0; l < 32; l++) dst[w++] = d2 * (src[qsO + ql + l] >> 4) - min2;
      ql += 32; is += 2;
    }
  }
  return w - dstOffset;
}

export function dequantQ5_K(
  src: Uint8Array, srcOffset: number, nBlocks: number, dst: Float32Array, dstOffset = 0,
): number {
  let w = dstOffset;
  for (let b = 0; b < nBlocks; b++) {
    const o = srcOffset + b * Q5_K_BLOCK_BYTES;
    const d = f16ToF32(src[o] | (src[o + 1] << 8));
    const dmin = f16ToF32(src[o + 2] | (src[o + 3] << 8));
    const scalesO = o + 4;
    const qhO = o + 16;
    const qsO = o + 48;
    let is = 0;
    let u1 = 1, u2 = 2;
    let ql = 0;
    for (let j = 0; j < QK_K; j += 64) {
      const [sc1, m1] = scaleMinK4(is, src, scalesO);
      const [sc2, m2] = scaleMinK4(is + 1, src, scalesO);
      const d1 = d * sc1, min1 = dmin * m1;
      const d2 = d * sc2, min2 = dmin * m2;
      for (let l = 0; l < 32; l++) {
        dst[w++] = d1 * ((src[qsO + ql + l] & 0x0f) + (src[qhO + l] & u1 ? 16 : 0)) - min1;
      }
      for (let l = 0; l < 32; l++) {
        dst[w++] = d2 * ((src[qsO + ql + l] >> 4) + (src[qhO + l] & u2 ? 16 : 0)) - min2;
      }
      ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
  }
  return w - dstOffset;
}

// Q6_K (210 byte, 256 pesi): [ql 128 B][qh 64 B][scales 16 int8][d f16 IN CODA].
// w = d * sc[i/16] * (q6 - 32). Riferimento: dequantize_row_q6_K. Layout GLM:
// ffn_down_shexp e output.weight.
export const Q6_K_BLOCK_WEIGHTS = QK_K;
export const Q6_K_BLOCK_BYTES = 210;

export function dequantQ6_K(
  src: Uint8Array, srcOffset: number, nBlocks: number, dst: Float32Array, dstOffset = 0,
): number {
  let w = dstOffset;
  for (let b = 0; b < nBlocks; b++) {
    const o = srcOffset + b * Q6_K_BLOCK_BYTES;
    const d = f16ToF32(src[o + 208] | (src[o + 209] << 8));
    let qlO = o;
    let qhO = o + 128;
    let scO = o + 192;
    for (let n = 0; n < QK_K; n += 128) {
      for (let l = 0; l < 32; l++) {
        const is = (l / 16) | 0;
        const q1 = ((src[qlO + l] & 0x0f) | (((src[qhO + l] >> 0) & 3) << 4)) - 32;
        const q2 = ((src[qlO + l + 32] & 0x0f) | (((src[qhO + l] >> 2) & 3) << 4)) - 32;
        const q3 = ((src[qlO + l] >> 4) | (((src[qhO + l] >> 4) & 3) << 4)) - 32;
        const q4 = ((src[qlO + l + 32] >> 4) | (((src[qhO + l] >> 6) & 3) << 4)) - 32;
        dst[w + l] = d * src8(src, scO + is) * q1;
        dst[w + l + 32] = d * src8(src, scO + is + 2) * q2;
        dst[w + l + 64] = d * src8(src, scO + is + 4) * q3;
        dst[w + l + 96] = d * src8(src, scO + is + 6) * q4;
      }
      w += 128; qlO += 64; qhO += 32; scO += 8;
    }
  }
  return w - dstOffset;
}

// byte → int8 con segno (le scale Q6_K sono int8).
function src8(src: Uint8Array, o: number): number {
  return (src[o] << 24) >> 24;
}

// Repack Q4_1: qs come Q4_0 (4 u32/blocco); "scales" = UN u32 per blocco con
// d nei 16 bit bassi e m negli alti (in WGSL: unpack2x16float → vec2(d, m)).
export function repackQ4_1(src: Uint8Array, srcOffset: number, nBlocks: number): RepackedQuant {
  const qs = new Uint32Array(nBlocks * 4);
  const scales = new Uint32Array(nBlocks);
  let o = srcOffset;
  for (let b = 0; b < nBlocks; b++) {
    scales[b] = (src[o] | (src[o + 1] << 8) | (src[o + 2] << 16) | (src[o + 3] << 24)) >>> 0;
    o += 4;
    for (let w = 0; w < 4; w++) {
      qs[b * 4 + w] = src[o] | (src[o + 1] << 8) | (src[o + 2] << 16) | (src[o + 3] << 24);
      o += 4;
    }
  }
  return { qs, scales };
}

// Repack K-quant: il superblocco resta nel layout GGUF, copiato in u32 LE con
// stride allineato a 4 byte (Q5_K: 176 B = 44 word esatte; Q6_K: 210 B → 53
// word, 2 B di pad). Il kernel indicizza byte dentro le word.
export function repackKQuant(
  src: Uint8Array, srcOffset: number, nBlocks: number, blockBytes: number,
): Uint32Array {
  const wordsPerBlock = Math.ceil(blockBytes / 4);
  const out = new Uint32Array(nBlocks * wordsPerBlock);
  for (let b = 0; b < nBlocks; b++) {
    const o = srcOffset + b * blockBytes;
    for (let j = 0; j < blockBytes; j++) {
      out[b * wordsPerBlock + (j >> 2)] |= src[o + j] << ((j & 3) * 8);
    }
  }
  return out;
}

// Q8_0 (34 byte, 32 pesi): [scala f16 LE (2 B)] [32 int8]. w = int8 * scala.
// Serve per output.weight del GGUF ufficiale (vedi gguf.ts). Anche questa è esatta
// in f32.
export const Q8_0_BLOCK_WEIGHTS = 32;
export const Q8_0_BLOCK_BYTES = 34;

export function dequantQ8_0(
  src: Uint8Array, srcOffset: number, nBlocks: number, dst: Float32Array, dstOffset = 0,
): number {
  let o = srcOffset;
  let w = dstOffset;
  for (let b = 0; b < nBlocks; b++) {
    const scale = f16ToF32(src[o] | (src[o + 1] << 8));
    o += 2;
    for (let j = 0; j < 32; j++) {
      dst[w + j] = ((src[o + j] << 24) >> 24) * scale; // int8 con segno
    }
    o += 32;
    w += Q8_0_BLOCK_WEIGHTS;
  }
  return w - dstOffset;
}

// --- Q3_K / Q2_K: dequant E quantizzatore (fase 4c slice A) ---
//
// Fin qui il repo sapeva solo LEGGERE pesi quantizzati. La 4c degrada gli expert
// più freddi Q4_0 → Q3_K, e senza pesi originali la catena è Q4_0 → f32 → Q3_K
// (errore COMPOSTO, design §2.2 caveat R8: le tabelle di perplexity pubbliche NON
// sono un proxy valido — si misura).
//
// Riferimento: ggml-quants.c del checkout oracolo (llama.cpp 5f55650), funzioni
// quantize_row_q3_K_ref / dequantize_row_q3_K / quantize_row_q2_K_ref /
// dequantize_row_q2_K. Senza imatrix `quantize_q3_K`/`quantize_q2_K` instradano
// sul path *_ref (make_q3_quants / make_qkx2_quants), che è quello riprodotto qui.
//
// FEDELTÀ NUMERICA: il C lavora in `float`. JS lavora in f64, quindi OGNI
// operazione intermedia è ri-arrotondata con Math.fround — è la condizione per il
// gate R1 (byte-identità con llama-quantize). Le somme sono arrotondate a ogni
// passo, e l'ordine di valutazione replica quello del C (associatività a sinistra).

const F32_SCRATCH = new Float32Array(1);
const U32_SCRATCH = new Uint32Array(F32_SCRATCH.buffer);

// f32 → bit f16, round-to-nearest-even con overflow a inf: stessa semantica di
// GGML_FP32_TO_FP16 (F16C vcvtps2ph / ggml_compute_fp32_to_fp16).
export function f32ToF16(value: number): number {
  F32_SCRATCH[0] = value;
  const x = U32_SCRATCH[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x0200 : 0); // inf / NaN quiet
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow → inf
  if (e <= 0) {
    if (e < -10) return sign; // sotto il minimo subnormale → ±0
    const m = mant | 0x800000; // mantissa con l'1 implicito
    const shift = 14 - e; // 14..24
    let h = m >>> shift;
    const rem = m & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1) !== 0)) h++;
    return sign | h;
  }
  let h = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1) !== 0)) h++; // il carry entra nell'esponente
  return sign | h;
}

// nearest_int di ggml: `float val = fval + 12582912.f` (= 1.5·2²³) più estrazione
// della mantissa = round-half-to-EVEN (non half-away-from-zero di Math.round).
//
// DOMINIO: esatta solo per |fval| ≤ 4 194 303 (2²²−1) — oltre, il trucco satura
// invece di arrotondare. È la stessa precondizione del C, dove è un `assert`
// compilato via in release. Qui non è raggiungibile dalla sorgente:
//  - l'input è SEMPRE un tensore Q4_0 dequantizzato, cioè x = (nibble−8)·s con
//    nibble−8 ∈ [−8, 7] e s scala f16 del blocco da 32;
//  - le chiamate con argomento `iscale·x` sono limitate per costruzione, perché
//    iscale = −nmax/max e |x| ≤ |max| dentro il sotto-blocco ⇒ |arg| ≤ nmax
//    (4 per Q3_K, 3 per Q2_K) e |arg| ≤ 32 per la quantizzazione delle scale;
//  - le due divisioni (x/d nella ri-quantizzazione, x·sl2/slx nel raffinamento)
//    sfonderebbero solo con un rapporto fra scale f16 dentro UN superblocco
//    > 5·10⁵, che questo modello non ha.
// Verificato empiricamente, non solo argomentato: byte-identità con ggml su
// 12 288 superblocchi reali di blk.5 più 64+8 di un secondo tensore. Una
// saturazione si manifesterebbe come divergenza di byte, non in silenzio.
function nearestInt(fval: number): number {
  return Math.fround(fval + 12582912) - 12582912;
}

const GROUP_MAX_EPS = 1e-15;

export const Q3_K_BLOCK_WEIGHTS = QK_K;
export const Q3_K_BLOCK_BYTES = 110; // hmask 32 | qs 64 | scales 12 | d f16
export const Q2_K_BLOCK_WEIGHTS = QK_K;
export const Q2_K_BLOCK_BYTES = 84; // scales 16 | qs 64 | d f16 | dmin f16

// Scala a 6 bit del sotto-blocco j (0..15) dai 12 byte impacchettati: 4 bit bassi
// nei primi 8 byte (due per byte) + 2 bit alti negli ultimi 4. Identica alla
// ricomposizione via `aux`/kmask di dequantize_row_q3_K, in forma scalare.
function q3kScale6(src: Uint8Array, so: number, j: number): number {
  const low = j < 8 ? (src[so + j] & 0x0f) : (src[so + j - 8] >> 4);
  const high = (src[so + 8 + (j & 3)] >> (2 * (j >> 2))) & 3;
  return (low | (high << 4)) - 32;
}

// Dequant di `nBlocks` superblocchi Q3_K. Riferimento: dequantize_row_q3_K.
export function dequantQ3_K(
  src: Uint8Array, srcOffset: number, nBlocks: number, dst: Float32Array, dstOffset = 0,
): number {
  let w = dstOffset;
  for (let b = 0; b < nBlocks; b++) {
    const o = srcOffset + b * Q3_K_BLOCK_BYTES;
    const hmO = o;
    const qsO = o + 32;
    const scO = o + 96;
    const dAll = f16ToF32(src[o + 108] | (src[o + 109] << 8));
    let m = 1;
    let is = 0;
    let q = qsO;
    for (let n = 0; n < QK_K; n += 128) {
      let shift = 0;
      for (let j = 0; j < 4; j++) {
        // fround: il C tiene dl e il prodotto in `float` — senza ri-arrotondare,
        // il doppio arrotondamento f64→f32 può divergere di 1 ulp dal riferimento.
        let dl = Math.fround(dAll * q3kScale6(src, scO, is++));
        for (let l = 0; l < 16; l++) {
          dst[w++] = Math.fround(dl * (((src[q + l] >> shift) & 3) - ((src[hmO + l] & m) ? 0 : 4)));
        }
        dl = Math.fround(dAll * q3kScale6(src, scO, is++));
        for (let l = 0; l < 16; l++) {
          dst[w++] = Math.fround(
            dl * (((src[q + l + 16] >> shift) & 3) - ((src[hmO + l + 16] & m) ? 0 : 4)));
        }
        shift += 2;
        m <<= 1;
      }
      q += 32;
    }
  }
  return w - dstOffset;
}

// Dequant di `nBlocks` superblocchi Q2_K. Riferimento: dequantize_row_q2_K.
export function dequantQ2_K(
  src: Uint8Array, srcOffset: number, nBlocks: number, dst: Float32Array, dstOffset = 0,
): number {
  let w = dstOffset;
  for (let b = 0; b < nBlocks; b++) {
    const o = srcOffset + b * Q2_K_BLOCK_BYTES;
    const qsO = o + 16;
    const d = f16ToF32(src[o + 80] | (src[o + 81] << 8));
    const dmin = f16ToF32(src[o + 82] | (src[o + 83] << 8));
    let is = 0;
    let q = qsO;
    for (let n = 0; n < QK_K; n += 128) {
      let shift = 0;
      for (let j = 0; j < 4; j++) {
        let sc = src[o + is++];
        let dl = Math.fround(d * (sc & 0x0f)), ml = Math.fround(dmin * (sc >> 4));
        for (let l = 0; l < 16; l++) {
          dst[w++] = Math.fround(Math.fround(dl * ((src[q + l] >> shift) & 3)) - ml);
        }
        sc = src[o + is++];
        dl = Math.fround(d * (sc & 0x0f)); ml = Math.fround(dmin * (sc >> 4));
        for (let l = 0; l < 16; l++) {
          dst[w++] = Math.fround(Math.fround(dl * ((src[q + l + 16] >> shift) & 3)) - ml);
        }
        shift += 2;
      }
      q += 32;
    }
  }
  return w - dstOffset;
}

// make_q3_quants(n, nmax, x, L, do_rmse=true): cerca la scala del sotto-blocco da
// 16 pesi con raffinamento greedy (max 5 passate, si ferma quando nessun livello
// cambia). L riceve livelli in [-nmax, nmax-1] + nmax. È la voce di costo dominante
// della quantizzazione Q3_K.
function makeQ3Quants(
  x: Float32Array, xo: number, n: number, nmax: number, L: Int8Array,
): number {
  let max = 0, amax = 0;
  for (let i = 0; i < n; i++) {
    const ax = Math.abs(x[xo + i]);
    if (ax > amax) { amax = ax; max = x[xo + i]; }
  }
  if (amax < GROUP_MAX_EPS) { L.fill(0, 0, n); return 0; }
  const iscale = Math.fround(-nmax / max);
  let sumlx = 0, suml2 = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[xo + i];
    let l = nearestInt(Math.fround(iscale * xi));
    l = Math.max(-nmax, Math.min(nmax - 1, l));
    L[i] = l;
    const w = Math.fround(xi * xi);
    sumlx = Math.fround(sumlx + Math.fround(Math.fround(w * xi) * l));
    suml2 = Math.fround(suml2 + Math.fround(Math.fround(w * l) * l));
  }
  for (let itry = 0; itry < 5; itry++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const xi = x[xo + i];
      const w = Math.fround(xi * xi);
      const wx = Math.fround(w * xi);
      let slx = Math.fround(sumlx - Math.fround(wx * L[i]));
      if (slx > 0) {
        let sl2 = Math.fround(suml2 - Math.fround(Math.fround(w * L[i]) * L[i]));
        let newL = nearestInt(Math.fround(Math.fround(xi * sl2) / slx));
        newL = Math.max(-nmax, Math.min(nmax - 1, newL));
        if (newL !== L[i]) {
          slx = Math.fround(slx + Math.fround(wx * newL));
          sl2 = Math.fround(sl2 + Math.fround(Math.fround(w * newL) * newL));
          if (sl2 > 0
            && Math.fround(Math.fround(slx * slx) * suml2)
             > Math.fround(Math.fround(sumlx * sumlx) * sl2)) {
            L[i] = newL; sumlx = slx; suml2 = sl2;
            changed++;
          }
        }
      }
    }
    if (!changed) break;
  }
  for (let i = 0; i < n; i++) L[i] += nmax;
  return suml2 > 0 ? Math.fround(sumlx / suml2) : 0;
}

// Quantizza `nBlocks` superblocchi (nBlocks*256 pesi f32) in Q3_K.
// Riferimento: quantize_row_q3_K_ref. Ritorna i byte scritti.
export function quantizeQ3_K(
  src: Float32Array, srcOffset: number, nBlocks: number, dst: Uint8Array, dstOffset = 0,
): number {
  const L = new Int8Array(QK_K);
  const Lsub = new Int8Array(16);
  const scales = new Float32Array(QK_K / 16);
  for (let b = 0; b < nBlocks; b++) {
    const xo = srcOffset + b * QK_K;
    const o = dstOffset + b * Q3_K_BLOCK_BYTES;
    dst.fill(0, o, o + Q3_K_BLOCK_BYTES);
    let maxScale = 0, amax = 0;
    for (let j = 0; j < 16; j++) {
      scales[j] = makeQ3Quants(src, xo + 16 * j, 16, 4, Lsub);
      for (let i = 0; i < 16; i++) L[16 * j + i] = Lsub[i];
      const s = Math.abs(scales[j]);
      if (s > amax) { amax = s; maxScale = scales[j]; }
    }
    const scO = o + 96;
    if (maxScale !== 0) {
      const iscale = Math.fround(-32 / maxScale);
      for (let j = 0; j < 16; j++) {
        let l = nearestInt(Math.fround(iscale * scales[j]));
        l = Math.max(-32, Math.min(31, l)) + 32;
        if (j < 8) dst[scO + j] |= l & 0x0f;
        else dst[scO + j - 8] |= (l & 0x0f) << 4;
        l >>= 4;
        dst[scO + 8 + (j & 3)] |= l << (2 * (j >> 2));
      }
      const dBits = f32ToF16(Math.fround(1 / iscale));
      dst[o + 108] = dBits & 0xff;
      dst[o + 109] = dBits >> 8;
    } else {
      dst[o + 108] = 0; dst[o + 109] = 0;
    }
    // ri-quantizzazione dei livelli con la scala EFFETTIVAMENTE codificata
    const dAll = f16ToF32(dst[o + 108] | (dst[o + 109] << 8));
    for (let j = 0; j < 16; j++) {
      const d = Math.fround(dAll * q3kScale6(dst, scO, j));
      if (d === 0) continue;
      for (let ii = 0; ii < 16; ii++) {
        let l = nearestInt(Math.fround(src[xo + 16 * j + ii] / d));
        l = Math.max(-4, Math.min(3, l));
        L[16 * j + ii] = l + 4;
      }
    }
    // bit alto: peso i → bit (i>>5) del byte hmask[i&31]
    let m = 0, hm = 1;
    for (let j = 0; j < QK_K; j++) {
      if (L[j] > 3) { dst[o + m] |= hm; L[j] -= 4; }
      if (++m === QK_K / 8) { m = 0; hm <<= 1; }
    }
    const qsO = o + 32;
    for (let j = 0; j < QK_K; j += 128) {
      for (let l = 0; l < 32; l++) {
        dst[qsO + (j >> 2) + l] =
          L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }
  }
  return nBlocks * Q3_K_BLOCK_BYTES;
}

// make_qkx2_quants(n, nmax, x, weights, L, &min, Laux, rmin, rdelta, nstep, use_mad):
// ricerca affine (scala + minimo) su nstep+1 candidati, criterio MAD pesato.
// Ritorna la scala e scrive `-min` in outMin[0].
function makeQkx2Quants(
  n: number, nmax: number, x: Float32Array, xo: number, weights: Float32Array,
  L: Uint8Array, Laux: Uint8Array, rmin: number, rdelta: number, nstep: number,
  outMin: Float32Array,
): number {
  let min = x[xo], max = x[xo];
  let sumW = weights[0];
  let sumX = Math.fround(sumW * x[xo]);
  for (let i = 1; i < n; i++) {
    const xi = x[xo + i];
    if (xi < min) min = xi;
    if (xi > max) max = xi;
    const w = weights[i];
    sumW = Math.fround(sumW + w);
    sumX = Math.fround(sumX + Math.fround(w * xi));
  }
  if (min > 0) min = 0;
  if (max === min) {
    L.fill(0, 0, n);
    outMin[0] = Math.fround(-min);
    return 0;
  }
  let iscale = Math.fround(nmax / Math.fround(max - min));
  let scale = Math.fround(1 / iscale);
  let bestError = 0;
  for (let i = 0; i < n; i++) {
    let l = nearestInt(Math.fround(iscale * Math.fround(x[xo + i] - min)));
    l = Math.max(0, Math.min(nmax, l));
    L[i] = l;
    const diff = Math.abs(Math.fround(Math.fround(Math.fround(scale * l) + min) - x[xo + i]));
    bestError = Math.fround(bestError + Math.fround(weights[i] * diff));
  }
  if (nstep < 1) { outMin[0] = Math.fround(-min); return scale; }
  for (let is = 0; is <= nstep; is++) {
    iscale = Math.fround(Math.fround(Math.fround(rmin + Math.fround(rdelta * is)) + nmax)
      / Math.fround(max - min));
    let sumL = 0, sumL2 = 0, sumXL = 0;
    for (let i = 0; i < n; i++) {
      let l = nearestInt(Math.fround(iscale * Math.fround(x[xo + i] - min)));
      l = Math.max(0, Math.min(nmax, l));
      Laux[i] = l;
      const w = weights[i];
      sumL = Math.fround(sumL + Math.fround(w * l));
      sumL2 = Math.fround(sumL2 + Math.fround(Math.fround(w * l) * l));
      sumXL = Math.fround(sumXL + Math.fround(Math.fround(w * l) * x[xo + i]));
    }
    const D = Math.fround(Math.fround(sumW * sumL2) - Math.fround(sumL * sumL));
    if (D > 0) {
      let thisScale = Math.fround(
        Math.fround(Math.fround(sumW * sumXL) - Math.fround(sumX * sumL)) / D);
      let thisMin = Math.fround(
        Math.fround(Math.fround(sumL2 * sumX) - Math.fround(sumL * sumXL)) / D);
      if (thisMin > 0) {
        thisMin = 0;
        thisScale = Math.fround(sumXL / sumL2);
      }
      let mad = 0;
      for (let i = 0; i < n; i++) {
        const diff = Math.abs(
          Math.fround(Math.fround(Math.fround(thisScale * Laux[i]) + thisMin) - x[xo + i]));
        mad = Math.fround(mad + Math.fround(weights[i] * diff));
      }
      if (mad < bestError) {
        for (let i = 0; i < n; i++) L[i] = Laux[i];
        bestError = mad;
        scale = thisScale;
        min = thisMin;
      }
    }
  }
  outMin[0] = Math.fround(-min);
  return scale;
}

// Quantizza `nBlocks` superblocchi in Q2_K. Riferimento: quantize_row_q2_K_ref
// (path senza imatrix: pesi = |x|, make_qkx2_quants con nstep=15 e MAD).
export function quantizeQ2_K(
  src: Float32Array, srcOffset: number, nBlocks: number, dst: Uint8Array, dstOffset = 0,
): number {
  const L = new Uint8Array(QK_K);
  const Laux = new Uint8Array(16);
  const Lsub = new Uint8Array(16);
  const weights = new Float32Array(16);
  const mins = new Float32Array(QK_K / 16);
  const scales = new Float32Array(QK_K / 16);
  const outMin = new Float32Array(1);
  const q4scale = 15;
  for (let b = 0; b < nBlocks; b++) {
    const xo = srcOffset + b * QK_K;
    const o = dstOffset + b * Q2_K_BLOCK_BYTES;
    dst.fill(0, o, o + Q2_K_BLOCK_BYTES);
    let maxScale = 0, maxMin = 0;
    for (let j = 0; j < 16; j++) {
      for (let l = 0; l < 16; l++) weights[l] = Math.abs(src[xo + 16 * j + l]);
      scales[j] = makeQkx2Quants(16, 3, src, xo + 16 * j, weights, Lsub, Laux, -0.5, 0.1, 15, outMin);
      mins[j] = outMin[0];
      for (let l = 0; l < 16; l++) L[16 * j + l] = Lsub[l];
      if (scales[j] > maxScale) maxScale = scales[j];
      if (mins[j] > maxMin) maxMin = mins[j];
    }
    if (maxScale > 0) {
      const iscale = Math.fround(q4scale / maxScale);
      for (let j = 0; j < 16; j++) dst[o + j] = nearestInt(Math.fround(iscale * scales[j]));
      const dBits = f32ToF16(Math.fround(maxScale / q4scale));
      dst[o + 80] = dBits & 0xff; dst[o + 81] = dBits >> 8;
    } else {
      for (let j = 0; j < 16; j++) dst[o + j] = 0;
      dst[o + 80] = 0; dst[o + 81] = 0;
    }
    if (maxMin > 0) {
      const iscale = Math.fround(q4scale / maxMin);
      for (let j = 0; j < 16; j++) dst[o + j] |= nearestInt(Math.fround(iscale * mins[j])) << 4;
      const mBits = f32ToF16(Math.fround(maxMin / q4scale));
      dst[o + 82] = mBits & 0xff; dst[o + 83] = mBits >> 8;
    } else {
      dst[o + 82] = 0; dst[o + 83] = 0;
    }
    const dSb = f16ToF32(dst[o + 80] | (dst[o + 81] << 8));
    const mSb = f16ToF32(dst[o + 82] | (dst[o + 83] << 8));
    for (let j = 0; j < 16; j++) {
      const d = Math.fround(dSb * (dst[o + j] & 0x0f));
      if (d === 0) continue;
      const dm = Math.fround(mSb * (dst[o + j] >> 4));
      for (let ii = 0; ii < 16; ii++) {
        let l = nearestInt(Math.fround(Math.fround(src[xo + 16 * j + ii] + dm) / d));
        l = Math.max(0, Math.min(3, l));
        L[16 * j + ii] = l;
      }
    }
    const qsO = o + 16;
    for (let j = 0; j < QK_K; j += 128) {
      for (let l = 0; l < 32; l++) {
        dst[qsO + (j >> 2) + l] =
          L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }
  }
  return nBlocks * Q2_K_BLOCK_BYTES;
}
