// File slab degli expert: il repack all'IMPORT invece che nel path caldo
// (goal C3a fase 3, spec §2).
//
// PERCHÉ. `ExpertCache.ensure` chiamava `packExpertSlab` a ogni miss: legge i
// byte GGUF grezzi dei tre tensori di un expert e li ri-impacchetta nel layout
// [qs | scales] del motore. Costava **9,5 ms per miss × 4,47 miss/token =
// 41,4 ms/token** — il 19% del wall, su CPU, dentro il forward. Gli stessi byte
// si possono produrre UNA volta sola all'import.
//
// FORMATO. Header di 4 KiB + i 2.944 slab consecutivi, ordinati per
// (layer, expert) con layer da 1 a 46 (blk.0 è denso). Le due size-class sono
// contigue e in quest'ordine: prima i 256 slab down-Q4_1 (blk.1-4), poi i 2.688
// down-Q4_0 (blk.5-46) — così l'offset è un'aritmetica chiusa, senza tabella.
//
// INVALIDAZIONE. L'header porta magic, versione di layout e SHA-256 del GGUF
// sorgente: se uno dei tre non combacia il file si rigenera. Mai leggere byte
// di cui non si conosce la provenienza (postura ds4, la stessa dell'import).
import { GLM47_FLASH as G, GLM47_DOWN_EXPS_Q4_1_LAST } from "./shape";
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "./moe";

export const SLAB_MAGIC = "BLABSLAB";
/** Cambiare questo numero INVALIDA tutti i file slab esistenti. */
export const SLAB_LAYOUT_VERSION = 1;
export const SLAB_HEADER_BYTES = 4096;
export const SLAB_FILE_NAME = "GLM-4.7-Flash-Q4_0.slabs.bin";

/** Expert con down Q4_1 (blk.1..4): stanno per primi nel file. */
export const N_SLABS_Q4_1 = GLM47_DOWN_EXPS_Q4_1_LAST * G.nExpert;
/** Expert con down Q4_0 (blk.5..46). */
export const N_SLABS_Q4_0 = (G.nLayer - G.denseLead - GLM47_DOWN_EXPS_Q4_1_LAST) * G.nExpert;
export const N_SLABS = N_SLABS_Q4_1 + N_SLABS_Q4_0;

export const SLAB_DATA_BYTES =
  N_SLABS_Q4_1 * SLAB_DOWN_Q4_1.bytes + N_SLABS_Q4_0 * SLAB_DOWN_Q4_0.bytes;
export const SLAB_FILE_BYTES = SLAB_HEADER_BYTES + SLAB_DATA_BYTES;

/**
 * Indice sequenziale di (layer, expert) nel file. I layer MoE partono da
 * `denseLead`; i Q4_1 stanno davanti, quindi l'indice NON è `layer*64+expert`.
 */
export function slabIndex(layer: number, expert: number): number {
  if (layer < G.denseLead || layer >= G.nLayer) throw new Error(`slabfile: layer ${layer} non e' MoE`);
  if (expert < 0 || expert >= G.nExpert) throw new Error(`slabfile: expert ${expert} fuori range`);
  const isQ41 = layer <= GLM47_DOWN_EXPS_Q4_1_LAST;
  return isQ41
    ? (layer - G.denseLead) * G.nExpert + expert
    : N_SLABS_Q4_1 + (layer - GLM47_DOWN_EXPS_Q4_1_LAST - 1) * G.nExpert + expert;
}

/** Offset assoluto dello slab nel file (header incluso) e sua taglia. */
export function slabRange(layer: number, expert: number): { offset: number; bytes: number } {
  const i = slabIndex(layer, expert);
  return i < N_SLABS_Q4_1
    ? { offset: SLAB_HEADER_BYTES + i * SLAB_DOWN_Q4_1.bytes, bytes: SLAB_DOWN_Q4_1.bytes }
    : {
        offset: SLAB_HEADER_BYTES + N_SLABS_Q4_1 * SLAB_DOWN_Q4_1.bytes
          + (i - N_SLABS_Q4_1) * SLAB_DOWN_Q4_0.bytes,
        bytes: SLAB_DOWN_Q4_0.bytes,
      };
}

export interface SlabHeader {
  magic: string;
  layoutVersion: number;
  sourceSha256: string;
  nSlabsQ4_1: number;
  nSlabsQ4_0: number;
  slabBytesQ4_1: number;
  slabBytesQ4_0: number;
  dataBytes: number;
}

export function buildSlabHeader(sourceSha256: string): Uint8Array {
  if (!/^[0-9a-f]{64}$/.test(sourceSha256)) throw new Error("slabfile: SHA-256 sorgente non valido");
  const buf = new Uint8Array(SLAB_HEADER_BYTES);
  const dv = new DataView(buf.buffer);
  for (let i = 0; i < SLAB_MAGIC.length; i++) buf[i] = SLAB_MAGIC.charCodeAt(i);
  let o = 8;
  dv.setUint32(o, SLAB_LAYOUT_VERSION, true); o += 4;
  dv.setUint32(o, N_SLABS_Q4_1, true); o += 4;
  dv.setUint32(o, N_SLABS_Q4_0, true); o += 4;
  dv.setUint32(o, SLAB_DOWN_Q4_1.bytes, true); o += 4;
  dv.setUint32(o, SLAB_DOWN_Q4_0.bytes, true); o += 4;
  // dataBytes non entra in u32 (15,68 GB): due word, low/high
  dv.setUint32(o, SLAB_DATA_BYTES >>> 0, true); o += 4;
  dv.setUint32(o, Math.floor(SLAB_DATA_BYTES / 2 ** 32), true); o += 4;
  for (let i = 0; i < 32; i++) buf[o + i] = parseInt(sourceSha256.slice(i * 2, i * 2 + 2), 16);
  return buf;
}

/** null ⇒ header assente/illeggibile: il chiamante rigenera. */
export function parseSlabHeader(buf: Uint8Array): SlabHeader | null {
  if (buf.length < SLAB_HEADER_BYTES) return null;
  for (let i = 0; i < SLAB_MAGIC.length; i++) if (buf[i] !== SLAB_MAGIC.charCodeAt(i)) return null;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 8;
  const layoutVersion = dv.getUint32(o, true); o += 4;
  const nSlabsQ4_1 = dv.getUint32(o, true); o += 4;
  const nSlabsQ4_0 = dv.getUint32(o, true); o += 4;
  const slabBytesQ4_1 = dv.getUint32(o, true); o += 4;
  const slabBytesQ4_0 = dv.getUint32(o, true); o += 4;
  const lo = dv.getUint32(o, true); o += 4;
  const hi = dv.getUint32(o, true); o += 4;
  let sha = "";
  for (let i = 0; i < 32; i++) sha += buf[o + i].toString(16).padStart(2, "0");
  return {
    magic: SLAB_MAGIC, layoutVersion, sourceSha256: sha,
    nSlabsQ4_1, nSlabsQ4_0, slabBytesQ4_1, slabBytesQ4_0,
    dataBytes: hi * 2 ** 32 + lo,
  };
}

/**
 * Il file è utilizzabile? Ogni campo che non combacia è un motivo di
 * rigenerazione DICHIARATO, non un fallimento silenzioso.
 */
export function slabFileReason(
  header: SlabHeader | null, fileBytes: number, expectedSha256: string,
): string | null {
  if (!header) return "header assente o magic sbagliato";
  if (header.layoutVersion !== SLAB_LAYOUT_VERSION) {
    return `versione di layout ${header.layoutVersion} ≠ ${SLAB_LAYOUT_VERSION}`;
  }
  if (header.sourceSha256 !== expectedSha256) return "SHA-256 del GGUF sorgente diverso";
  if (header.nSlabsQ4_1 !== N_SLABS_Q4_1 || header.nSlabsQ4_0 !== N_SLABS_Q4_0) {
    return `conteggio slab ${header.nSlabsQ4_1}/${header.nSlabsQ4_0} ≠ ${N_SLABS_Q4_1}/${N_SLABS_Q4_0}`;
  }
  if (header.slabBytesQ4_1 !== SLAB_DOWN_Q4_1.bytes || header.slabBytesQ4_0 !== SLAB_DOWN_Q4_0.bytes) {
    return "taglia degli slab diversa dal layout corrente";
  }
  if (fileBytes !== SLAB_FILE_BYTES) return `file di ${fileBytes} B ≠ ${SLAB_FILE_BYTES} attesi`;
  return null; // valido
}
