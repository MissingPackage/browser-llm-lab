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
