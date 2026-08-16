// Test di `src/engine/slabsource.ts` — l'ultimo anello della catena dello slab
// pre-impacchettato (goal engine-velocita-decode, it.50).
//
// COSA ESERCITA: la decisione di usare o NON usare il file slab, che e' l'unica
// cosa che questo modulo fa oltre a un'aritmetica gia' coperta altrove
// (`slabgeom.test`, `slabfile.test`). Le deps sono iniettate: niente rete,
// niente file da 17 GiB, e la validazione si prova su un file finto di 2 KiB.
//
// PERCHE' OGNI CASO NEGATIVO CONTA. Uno slab prodotto da un ALTRO GGUF, o
// troncato, o di una versione di layout precedente, non darebbe un errore: darebbe
// **pesi validi e sbagliati**. E' la classe di difetto che questo progetto paga
// piu' cara, e la difesa e' che ogni motivo di rifiuto sia dichiarato invece che
// dedotto. Il fallback ai byte grezzi deve restare possibile SEMPRE: un motore
// che non parte perche' manca una cache sarebbe peggio di uno piu' lento.
import { describe, expect, it } from "vitest";
import { openSlabRangeSource, type SlabHttpDeps } from "../src/engine/slabsource";
import { buildSlabHeader, slabFileBytes, slabFileRange, SLAB_HEADER_BYTES } from "../src/engine/slabfile";
import { slabGeometry, type SlabModelDesc } from "../src/engine/slabgeom";
import { mkSlabLayout } from "../src/engine/moe";

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);

// Un modello finto minuscolo con DUE classi che si ALTERNANO — la forma del 35B
// (down q4_K sui layer 0-33, q6_K su 34, …), non quella del GLM a due corse
// contigue: se l'offset tornasse a un `layer <= confine` questi casi cadono.
// gli elementi non sono a caso: gli offset dello slab devono restare allineati
// a 256 B (`mkSlabLayout` lo pretende, ed e' un vincolo del binding GPU)
const LAY_A = mkSlabLayout("clsA", { kind: "q4_K", elems: 4096 }, { kind: "q4_K", elems: 4096 }, { kind: "q4_K", elems: 4096 });
const LAY_B = mkSlabLayout("clsB", { kind: "q4_K", elems: 4096 }, { kind: "q4_K", elems: 4096 }, { kind: "q6_K", elems: 32768 });
const DESC: SlabModelDesc = {
  fileName: "finto.slabs.bin",
  denseLead: 0,
  nLayer: 4,
  nExpert: 3,
  layoutOf: (l) => (l === 2 ? LAY_B : LAY_A),
};
const GEO = slabGeometry(DESC);

/** File slab in memoria: header vero + un'area dati riempita in modo riconoscibile. */
function mkFile(sha: string): Uint8Array {
  const buf = new Uint8Array(slabFileBytes(GEO));
  buf.set(buildSlabHeader(GEO, sha), 0);
  // ogni slab si firma col suo (layer, expert) al primo byte: cosi' un offset
  // sbagliato NON passa inosservato — e' il difetto che la geometria v2 esiste
  // per aver evitato
  for (let l = 0; l < DESC.nLayer; l++) {
    for (let e = 0; e < DESC.nExpert; e++) {
      const r = slabFileRange(GEO, l, e);
      buf[r.offset] = l * 16 + e + 1;
    }
  }
  return buf;
}

interface Rec { fileName: string; off: number; len: number }

function mkDeps(file: Uint8Array | null, rec: Rec[] = [], sizeOverride?: number): SlabHttpDeps {
  return {
    async size(fileName) {
      if (!file) return null;
      expect(fileName).toBe(DESC.fileName);
      return sizeOverride ?? file.length;
    },
    async read(fileName, off, len) {
      rec.push({ fileName, off, len });
      if (!file) throw new Error("file assente");
      if (off + len > file.length) throw new Error(`range corto ${file.length - off}/${len}`);
      return file.subarray(off, off + len);
    },
  };
}

describe("slabsource: la sorgente slab a Range accetta solo il file giusto", () => {
  it("file valido: apre, e legge ESATTAMENTE il range di (layer, expert)", async () => {
    const rec: Rec[] = [];
    const { src, reason } = await openSlabRangeSource(mkDeps(mkFile(SHA_A), rec), DESC, SHA_A);
    expect(reason).toBeNull();
    expect(src).not.toBeNull();
    expect(src!.fileName).toBe(DESC.fileName);
    expect(src!.fileBytes).toBe(slabFileBytes(GEO));
    // l'apertura legge SOLO l'header: aprire non deve costare byte di dati
    expect(rec).toEqual([{ fileName: DESC.fileName, off: 0, len: SLAB_HEADER_BYTES }]);

    for (const [l, e] of [[0, 0], [2, 1], [3, 2]] as const) {
      rec.length = 0;
      const slab = await src!.slab(l, e);
      const r = slabFileRange(GEO, l, e);
      expect(rec, "uno slab si legge in UNA richiesta, non tre").toEqual([
        { fileName: DESC.fileName, off: r.offset, len: r.bytes },
      ]);
      expect(slab.length).toBe(r.bytes);
      expect(slab[0], `lo slab letto per blk.${l}/${e} non e' il suo`).toBe(l * 16 + e + 1);
    }
  });

  it("la classe alternata: i due layout hanno taglie diverse e la lettura le rispetta", async () => {
    const { src } = await openSlabRangeSource(mkDeps(mkFile(SHA_A)), DESC, SHA_A);
    expect(LAY_A.bytes).not.toBe(LAY_B.bytes); // altrimenti il caso non prova niente
    expect((await src!.slab(2, 0)).length).toBe(LAY_B.bytes);
    expect((await src!.slab(1, 0)).length).toBe(LAY_A.bytes);
  });

  it("file assente: nessun errore, si dichiara e si torna ai byte grezzi", async () => {
    const { src, reason } = await openSlabRangeSource(mkDeps(null), DESC, SHA_A);
    expect(src).toBeNull();
    expect(reason).toContain("assente");
    expect(reason).toContain(DESC.fileName);
  });

  it("slab di un ALTRO GGUF: rifiutato sullo SHA dell'header", async () => {
    const { src, reason } = await openSlabRangeSource(mkDeps(mkFile(SHA_B)), DESC, SHA_A);
    expect(src).toBeNull();
    expect(reason).toBe("SHA-256 del GGUF sorgente diverso");
  });

  it("file troncato: la taglia non combacia e il file non si usa", async () => {
    const short = slabFileBytes(GEO) - 1;
    const { src, reason } = await openSlabRangeSource(mkDeps(mkFile(SHA_A), [], short), DESC, SHA_A);
    expect(src).toBeNull();
    expect(reason).toContain(`${short} B`);
  });

  it("geometria diversa (un expert in piu'): rifiutato sulle size-class", async () => {
    const altro: SlabModelDesc = { ...DESC, nExpert: DESC.nExpert + 1 };
    const { src, reason } = await openSlabRangeSource(mkDeps(mkFile(SHA_A)), altro, SHA_A);
    expect(src).toBeNull();
    expect(reason).toMatch(/size-class|B ≠/);
  });

  it("header illeggibile (byte a caso): rifiutato senza lanciare", async () => {
    const junk = new Uint8Array(slabFileBytes(GEO)).fill(7);
    const { src, reason } = await openSlabRangeSource(mkDeps(junk), DESC, SHA_A);
    expect(src).toBeNull();
    expect(reason).toBe("header assente o magic sbagliato");
  });

  it("un errore di trasporto sull'header NON propaga: diventa un motivo", async () => {
    const deps: SlabHttpDeps = {
      size: async () => slabFileBytes(GEO),
      read: async () => { throw new Error("Range non onorato (416)"); },
    };
    const { src, reason } = await openSlabRangeSource(deps, DESC, SHA_A);
    expect(src).toBeNull();
    expect(reason).toContain("416");
  });
});
