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
