// LA SORGENTE CHE LEGGE IL FILE SLAB, servita via HTTP Range.
//
// E' L'ULTIMO ANELLO della catena che i quattro pezzi precedenti hanno
// preparato (goal engine-velocita-decode):
//
//   l'interfaccia della cache  `{ raw, slab }`      residency.ts (it.42)
//   il formato del file        header v2 parametrico slabfile.ts  (it.44-45)
//   il descrittore             DERIVATO dall'header  slabgeom.ts  (it.46)
//   il convertitore offline    GGUF -> slab          scripts/q35-slab-build.mjs (it.47)
//   -----------------------------------------------------------------------
//   la sorgente che LEGGE      questo modulo
//
// COSA VALE, misurato sul turno vero del PI (it.40): `packExpertSlab` sparisce
// dal path caldo (**7,11 s per sessione**) e le richieste Range passano da
// **38.625 a 12.875**, perche' uno slab si legge in UN colpo invece di tre
// tensori separati.
//
// PERCHE' VIA HTTP E NON DA OPFS, come il GLM. La quota OPFS di questa origine
// e' **10,00 GiB** e `persist()` viene NEGATA (misurato in it.43): il file slab
// del 35B ne vuole 17,07. Il GLM ci sta e genera il suo file all'import
// (`glmsource.ts`, `ensureSlabs`); il 35B no, e prende l'altra meta' del ruling
// del PI — uno slab **gia' convertito**, servito come il GGUF.
//
// NON E' UN MODULO DEL 35B. Prende un `SlabModelDesc` e basta: la stessa
// sorgente serve qualunque famiglia MoE il cui file slab sia servito via Range,
// e il GLM potrebbe usarla il giorno in cui il suo non stesse piu' in OPFS. E'
// l'ordine che il PI ha dato — prima le forme globali, i modelli nuovi le
// ereditano da soli.
//
// LA POSTURA SULL'INVALIDAZIONE e' quella di `slabfile.ts`: ogni motivo per cui
// il file non e' utilizzabile viene DICHIARATO e restituito al chiamante, che
// torna ai byte grezzi. Mai leggere byte di cui non si conosce la provenienza —
// uno slab prodotto da un altro GGUF darebbe pesi validi e sbagliati, non un
// errore.
import { ggufRangeReader, httpFileBytes, type GgufRangeReader } from "./ggufrange";
import { SLAB_HEADER_BYTES, parseSlabHeader, slabFileRange, slabFileReason } from "./slabfile";
import { slabGeometry, type SlabGeometry, type SlabModelDesc } from "./slabgeom";

/**
 * Come si arriva ai byte del file slab. E' una dipendenza iniettata e non una
 * `fetch` cablata perche' i worker sanno DOVE stanno i modelli (prefisso
 * dell'URL) e i test non devono aprire una rete per esercitare la validazione.
 */
export interface SlabHttpDeps {
  /** byte totali del file, `null` se assente o non misurabile */
  size(fileName: string): Promise<number | null>;
  /** esattamente `len` byte da `off`, oppure LANCIA (mai una lettura corta muta) */
  read(fileName: string, off: number, len: number): Promise<Uint8Array>;
}

export interface SlabRangeSource {
  fileName: string;
  fileBytes: number;
  geom: SlabGeometry;
  /**
   * Lo slab GIA' impacchettato di (layer, expert), in UNA richiesta.
   * Asincrono per costruzione: `ExpertCache.ensure` e' sincrona, quindi il
   * chiamante `await`ta i soli miss e poi consegna i byte gia' in mano — la
   * stessa forma che il 35B usa oggi per i byte grezzi.
   */
  slab(layer: number, expert: number): Promise<Uint8Array>;
}

/** `src` non nullo ⇔ `reason` nullo: o il file e' utilizzabile, o si sa perche' no. */
export interface SlabOpenResult {
  src: SlabRangeSource | null;
  /** null = file valido e in uso; altrimenti il motivo del fallback ai byte grezzi */
  reason: string | null;
}

const why = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/**
 * Apre il file slab di un modello, se c'e' ed e' quello giusto.
 *
 * Non lancia MAI per un file assente o invalido: il chiamante ha sempre la via
 * dei byte grezzi, e un motore che si rifiutasse di partire perche' manca una
 * cache sarebbe peggio di uno piu' lento.
 */
export async function openSlabRangeSource(
  deps: SlabHttpDeps, desc: SlabModelDesc, expectedSha256: string,
): Promise<SlabOpenResult> {
  const geom = slabGeometry(desc);
  let size: number | null;
  try { size = await deps.size(desc.fileName); }
  catch (e) { return { src: null, reason: `taglia di ${desc.fileName} non leggibile: ${why(e)}` }; }
  if (size === null) return { src: null, reason: `file slab assente (${desc.fileName})` };

  let header;
  try { header = parseSlabHeader(await deps.read(desc.fileName, 0, SLAB_HEADER_BYTES)); }
  catch (e) { return { src: null, reason: `header di ${desc.fileName} illeggibile: ${why(e)}` }; }

  const reason = slabFileReason(header, size, expectedSha256, geom);
  if (reason) return { src: null, reason };

  return {
    reason: null,
    src: {
      fileName: desc.fileName,
      fileBytes: size,
      geom,
      slab(layer, expert) {
        const r = slabFileRange(geom, layer, expert);
        return deps.read(desc.fileName, r.offset, r.bytes);
      },
    },
  };
}

/**
 * Le deps HTTP, montate su un prefisso di URL. Sta qui e non nei worker perche'
 * i worker sono tre e la stessa funzione copiata tre volte e' il difetto che
 * `ggufrange.ts` esiste per aver gia' pagato una volta (cinque copie, e la
 * quinta aveva perso il controllo di lunghezza).
 *
 * Le letture passano dal lettore CONDIVISO, quindi finiscono nei contatori di
 * `ggufRangeStats` come tutte le altre: e' cosi' che si vede il 3->1 delle
 * richieste per miss senza aggiungere un contatore nuovo.
 */
export const httpSlabDeps = (baseUrl: string): SlabHttpDeps => {
  const readers = new Map<string, GgufRangeReader>();
  const readerOf = (fileName: string): GgufRangeReader => {
    let rd = readers.get(fileName);
    if (!rd) { rd = ggufRangeReader(() => baseUrl + fileName, `slab:${fileName}`); readers.set(fileName, rd); }
    return rd;
  };
  return {
    size: (fileName) => httpFileBytes(baseUrl + fileName),
    read: (fileName, off, len) => readerOf(fileName)(off, len),
  };
};
