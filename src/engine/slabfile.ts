// File slab degli expert: il repack all'IMPORT invece che nel path caldo
// (goal C3a fase 3, spec §2). PARAMETRICO SUL MODELLO da it.44-45.
//
// PERCHÉ. `ExpertCache.ensure` chiamava `packExpertSlab` a ogni miss: legge i
// byte GGUF grezzi dei tre tensori di un expert e li ri-impacchetta nel layout
// [qs | scales] del motore. Sul GLM costava **9,5 ms per miss × 4,47 miss/token
// = 41,4 ms/token**; sul 35B il conto è **7,11 s per sessione** (it.40). Gli
// stessi byte si possono produrre UNA volta sola.
//
// FORMATO. Header di 4 KiB + gli slab, raggruppati PER CLASSE e contigui dentro
// ogni classe — così l'offset resta aritmetica e non una tabella per slab.
// Quale layer appartiene a quale classe lo dice `slabgeom`, che lo deriva dal
// descrittore del modello.
//
// COSA E' CAMBIATO IN v2, e perché il numero di versione esiste. Fino a v1 la
// geometria era cablata sul GLM: DUE size-class CONTIGUE, con un `layer <=
// confine` a separarle. Sul 35B le classi si ALTERNANO (down q4_K sui layer
// 0-33, q6_K su 34, q4_K su 35-37, q6_K su 38-39, letto dall'header del GGUF),
// quindi quel confine non le descrive e avrebbe prodotto **offset validi e
// sbagliati**. L'header ora porta la LISTA delle classi invece di due coppie
// cablate.
//
// **Un file slab v1 esistente si rigenera**: `slabFileReason` lo dichiara come
// motivo, `ensureSlabs` lo riscrive su un temporaneo e rinomina. Costa una
// passata di repack (sul GLM ~5 s), una volta.
//
// INVALIDAZIONE. L'header porta magic, versione di layout, SHA-256 del GGUF
// sorgente e la geometria: se uno di questi non combacia il file si rigenera.
// Mai leggere byte di cui non si conosce la provenienza (postura ds4).
import { slabRangeOf, type SlabGeometry } from "./slabgeom";

export const SLAB_MAGIC = "BLABSLAB";
/**
 * Cambiare questo numero INVALIDA tutti i file slab esistenti.
 * v2 (it.45): header con la lista delle classi invece di due coppie cablate.
 */
export const SLAB_LAYOUT_VERSION = 2;
export const SLAB_HEADER_BYTES = 4096;

/** Quante classi l'header può descrivere. Oltre, il file non si scrive. */
export const SLAB_MAX_CLASSES = 16;

export interface SlabHeader {
  magic: string;
  layoutVersion: number;
  sourceSha256: string;
  /** una voce per size-class, NELL'ORDINE in cui stanno nel file */
  classes: Array<{ nSlabs: number; bytes: number }>;
  dataBytes: number;
}

/** Byte totali del file per una geometria: header + area dati. */
export const slabFileBytes = (g: SlabGeometry): number => SLAB_HEADER_BYTES + g.dataBytes;

/**
 * Offset ASSOLUTO dello slab nel file (header incluso) e sua taglia.
 * `slabgeom` lavora nell'area dati e non conosce l'header: la somma sta qui,
 * in un posto solo.
 */
export function slabFileRange(
  g: SlabGeometry, layer: number, expert: number,
): { offset: number; bytes: number } {
  const r = slabRangeOf(g, layer, expert);
  return { offset: SLAB_HEADER_BYTES + r.offset, bytes: r.bytes };
}

export function buildSlabHeader(g: SlabGeometry, sourceSha256: string): Uint8Array {
  if (!/^[0-9a-f]{64}$/.test(sourceSha256)) throw new Error("slabfile: SHA-256 sorgente non valido");
  if (g.classes.length > SLAB_MAX_CLASSES) {
    throw new Error(`slabfile: ${g.classes.length} size-class, il massimo e' ${SLAB_MAX_CLASSES}`);
  }
  const buf = new Uint8Array(SLAB_HEADER_BYTES);
  const dv = new DataView(buf.buffer);
  for (let i = 0; i < SLAB_MAGIC.length; i++) buf[i] = SLAB_MAGIC.charCodeAt(i);
  let o = 8;
  dv.setUint32(o, SLAB_LAYOUT_VERSION, true); o += 4;
  dv.setUint32(o, g.classes.length, true); o += 4;
  // dataBytes non entra in u32 (15,68 GB sul GLM): due word, low/high
  dv.setUint32(o, g.dataBytes >>> 0, true); o += 4;
  dv.setUint32(o, Math.floor(g.dataBytes / 2 ** 32), true); o += 4;
  for (let i = 0; i < 32; i++) buf[o + i] = parseInt(sourceSha256.slice(i * 2, i * 2 + 2), 16);
  o += 32;
  for (const c of g.classes) {
    dv.setUint32(o, c.nSlabs, true); o += 4;
    dv.setUint32(o, c.bytes, true); o += 4;
  }
  return buf;
}

/** null ⇒ header assente/illeggibile: il chiamante rigenera. */
export function parseSlabHeader(buf: Uint8Array): SlabHeader | null {
  if (buf.length < SLAB_HEADER_BYTES) return null;
  for (let i = 0; i < SLAB_MAGIC.length; i++) if (buf[i] !== SLAB_MAGIC.charCodeAt(i)) return null;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 8;
  const layoutVersion = dv.getUint32(o, true); o += 4;
  const nClasses = dv.getUint32(o, true); o += 4;
  const lo = dv.getUint32(o, true); o += 4;
  const hi = dv.getUint32(o, true); o += 4;
  let sha = "";
  for (let i = 0; i < 32; i++) sha += buf[o + i].toString(16).padStart(2, "0");
  o += 32;
  // un conteggio assurdo NON deve far leggere fuori dall'header: si rifiuta
  if (nClasses > SLAB_MAX_CLASSES || o + nClasses * 8 > SLAB_HEADER_BYTES) return null;
  const classes: Array<{ nSlabs: number; bytes: number }> = [];
  for (let i = 0; i < nClasses; i++) {
    const nSlabs = dv.getUint32(o, true); o += 4;
    const bytes = dv.getUint32(o, true); o += 4;
    classes.push({ nSlabs, bytes });
  }
  return {
    magic: SLAB_MAGIC, layoutVersion, sourceSha256: sha, classes,
    dataBytes: hi * 2 ** 32 + lo,
  };
}

/**
 * Il file è utilizzabile? Ogni campo che non combacia è un motivo di
 * rigenerazione DICHIARATO, non un fallimento silenzioso.
 */
export function slabFileReason(
  header: SlabHeader | null, fileBytes: number, expectedSha256: string, g: SlabGeometry,
): string | null {
  if (!header) return "header assente o magic sbagliato";
  if (header.layoutVersion !== SLAB_LAYOUT_VERSION) {
    return `versione di layout ${header.layoutVersion} ≠ ${SLAB_LAYOUT_VERSION}`;
  }
  if (header.sourceSha256 !== expectedSha256) return "SHA-256 del GGUF sorgente diverso";
  if (header.classes.length !== g.classes.length) {
    return `${header.classes.length} size-class ≠ ${g.classes.length} attese`;
  }
  for (let i = 0; i < g.classes.length; i++) {
    const h = header.classes[i], w = g.classes[i];
    if (h.nSlabs !== w.nSlabs || h.bytes !== w.bytes) {
      return `size-class ${i}: ${h.nSlabs}×${h.bytes} B ≠ ${w.nSlabs}×${w.bytes} attesi`;
    }
  }
  if (fileBytes !== slabFileBytes(g)) return `file di ${fileBytes} B ≠ ${slabFileBytes(g)} attesi`;
  return null; // valido
}
