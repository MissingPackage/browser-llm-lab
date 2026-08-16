// La geometria parametrica riproduce ESATTAMENTE quella del GLM, e regge il
// 35B che quella non reggeva (goal engine-velocita-decode, it.44).
//
// IL CASO CHE CONTA E' IL PRIMO: `slabfile.ts` calcola gli offset del GLM con
// un'aritmetica chiusa su due corse contigue. Se il modulo parametrico non
// producesse gli STESSI offset, un file slab gia' su disco diventerebbe
// illeggibile — o peggio, leggibile e sbagliato. Qui si confrontano slab per
// slab, tutti e 2.944, contro l'implementazione storica.
//
// IL SECONDO CASO E' LA RAGIONE PER CUI IL MODULO ESISTE: sul 35B le classi si
// ALTERNANO (down q4_K sui layer 0-33, q6_K su 34, q4_K su 35-37, q6_K su
// 38-39), quindi «layer <= confine» non le descrive. Verificato sull'header del
// GGUF prima di scrivere una riga.
import { describe, expect, it } from "vitest";
import { slabGeometry, slabRangeOf, type SlabModelDesc } from "../src/engine/slabgeom";
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, mkSlabLayout, type SlabLayout } from "../src/engine/moe";
import { GLM47_FLASH as G, GLM47_DOWN_EXPS_Q4_1_LAST } from "../src/engine/shape";
import { SLAB_HEADER_BYTES } from "../src/engine/slabfile";

// L'ARITMETICA STORICA (layout v1), congelata qui come RIFERIMENTO.
//
// Prima di it.45 questa formula viveva in `slabfile.ts` e descriveva la
// disposizione degli slab del GLM sul disco. Ora `slabfile` e' parametrico e
// quella formula non esiste piu' — ma **la disposizione su disco e' la stessa**,
// e questo caso deve continuare a provarlo. Scriverla qui la rende quello che
// e': la SPECIFICA del layout del GLM, non un dettaglio di implementazione.
const N_SLABS_Q4_1_V1 = GLM47_DOWN_EXPS_Q4_1_LAST * G.nExpert;
const N_SLABS_Q4_0_V1 = (G.nLayer - G.denseLead - GLM47_DOWN_EXPS_Q4_1_LAST) * G.nExpert;
const N_SLABS = N_SLABS_Q4_1_V1 + N_SLABS_Q4_0_V1;
const SLAB_DATA_BYTES = N_SLABS_Q4_1_V1 * SLAB_DOWN_Q4_1.bytes + N_SLABS_Q4_0_V1 * SLAB_DOWN_Q4_0.bytes;
const slabRange = (layer: number, expert: number): { offset: number; bytes: number } => {
  const isQ41 = layer <= GLM47_DOWN_EXPS_Q4_1_LAST;
  const i = isQ41
    ? (layer - G.denseLead) * G.nExpert + expert
    : N_SLABS_Q4_1_V1 + (layer - GLM47_DOWN_EXPS_Q4_1_LAST - 1) * G.nExpert + expert;
  return i < N_SLABS_Q4_1_V1
    ? { offset: SLAB_HEADER_BYTES + i * SLAB_DOWN_Q4_1.bytes, bytes: SLAB_DOWN_Q4_1.bytes }
    : {
        offset: SLAB_HEADER_BYTES + N_SLABS_Q4_1_V1 * SLAB_DOWN_Q4_1.bytes
          + (i - N_SLABS_Q4_1_V1) * SLAB_DOWN_Q4_0.bytes,
        bytes: SLAB_DOWN_Q4_0.bytes,
      };
};

const GLM_DESC: SlabModelDesc = {
  fileName: "GLM-4.7-Flash-Q4_0.slabs.bin",
  denseLead: G.denseLead, nLayer: G.nLayer, nExpert: G.nExpert,
  layoutOf: (l) => (l <= GLM47_DOWN_EXPS_Q4_1_LAST ? SLAB_DOWN_Q4_1 : SLAB_DOWN_Q4_0),
};

describe("slabgeom: il GLM esce identico all'aritmetica storica (layout v1)", () => {
  const g = slabGeometry(GLM_DESC);

  it("stesso numero di slab e stessi byte di dati", () => {
    expect(g.nSlabs).toBe(N_SLABS);
    expect(g.dataBytes).toBe(SLAB_DATA_BYTES);
  });

  it("stesso offset e stessa taglia per TUTTI i 2.944 slab", () => {
    let checked = 0;
    for (let l = G.denseLead; l < G.nLayer; l++) {
      for (let e = 0; e < G.nExpert; e++) {
        const mine = slabRangeOf(g, l, e);
        const hist = slabRange(l, e);
        // il modulo parametrico non conosce l'header: il chiamante lo somma
        expect(mine.offset + SLAB_HEADER_BYTES, `layer ${l} expert ${e}`).toBe(hist.offset);
        expect(mine.bytes, `layer ${l} expert ${e}`).toBe(hist.bytes);
        checked++;
      }
    }
    expect(checked).toBe(N_SLABS);
  });

  it("le due classi restano nell'ordine del file: Q4_1 davanti", () => {
    expect(g.classes.map((c) => c.id)).toEqual([SLAB_DOWN_Q4_1.id, SLAB_DOWN_Q4_0.id]);
    expect(g.classes[0].base).toBe(0);
  });

  it("un layer denso e un expert fuori range LANCIANO, non restituiscono un offset", () => {
    expect(() => slabRangeOf(g, 0, 0)).toThrow(/non e' MoE/);
    expect(() => slabRangeOf(g, G.nLayer, 0)).toThrow(/non e' MoE/);
    expect(() => slabRangeOf(g, G.denseLead, G.nExpert)).toThrow(/fuori da/);
  });
});

describe("slabgeom: le classi ALTERNATE del 35B, che l'aritmetica del GLM non regge", () => {
  // le taglie non contano per questo caso: conta che le classi si alternino
  const A = mkSlabLayout("q4K", { kind: "q4_K", elems: 2048 * 512 },
    { kind: "q4_K", elems: 2048 * 512 }, { kind: "q4_K", elems: 512 * 2048 });
  const B = mkSlabLayout("q6K", { kind: "q4_K", elems: 2048 * 512 },
    { kind: "q4_K", elems: 2048 * 512 }, { kind: "q6_K", elems: 512 * 2048 });
  // la disposizione VERA, letta dall'header del GGUF in it.44
  const q6 = new Set([34, 38, 39]);
  const D: SlabModelDesc = {
    fileName: "q35.slabs.bin", denseLead: 0, nLayer: 40, nExpert: 256,
    layoutOf: (l) => (q6.has(l) ? B : A),
  };
  const g = slabGeometry(D);

  it("trova due classi e le conta bene, benche' si alternino", () => {
    expect(g.classes.map((c) => c.id)).toEqual(["q4K", "q6K"]);
    expect(g.classes[0].nSlabs).toBe(37 * 256);
    expect(g.classes[1].nSlabs).toBe(3 * 256);
  });

  it("ogni slab ha un offset DISTINTO e dentro l'area dati", () => {
    const seen = new Set<number>();
    for (let l = 0; l < 40; l++) {
      for (let e = 0; e < 256; e++) {
        const r = slabRangeOf(g, l, e);
        expect(seen.has(r.offset), `offset ripetuto a layer ${l} expert ${e}`).toBe(false);
        seen.add(r.offset);
        expect(r.offset + r.bytes).toBeLessThanOrEqual(g.dataBytes);
      }
    }
    expect(seen.size).toBe(40 * 256);
  });

  it("i layer q6_K NON sono contigui: e' il motivo per cui il confine del GLM non basta", () => {
    // con «layer <= confine» il 34 e il 38-39 finirebbero nella classe sbagliata
    expect(g.classOfLayer[33]).toBe(0);
    expect(g.classOfLayer[34]).toBe(1);
    expect(g.classOfLayer[35]).toBe(0);
    expect(g.classOfLayer[38]).toBe(1);
    // e i ranghi dentro la classe saltano i layer dell'altra
    expect(g.rankOfLayer[35]).toBe(34);
    expect(g.rankOfLayer[38]).toBe(1);
  });

  it("due layout con lo stesso id e taglie diverse LANCIANO", () => {
    const clash: SlabModelDesc = {
      ...D,
      layoutOf: (l) => (l === 0 ? A : ({ ...B, id: A.id } as SlabLayout)),
    };
    expect(() => slabGeometry(clash)).toThrow(/due taglie/);
  });
});

// ---------------------------------------------------------------------------
// IL DESCRITTORE DEL 35B, costruito dal GGUF VERO (it.46).
//
// I casi qui sopra usano una disposizione delle classi scritta a mano — vanno
// bene per provare l'aritmetica, non per provare che il modello sia quello.
// Questo legge l'header del file e verifica che la geometria che ne esce sia
// quella osservata: q6_K sui layer 34, 38, 39 e q4_K su tutti gli altri.
//
// Si salta se il GGUF non c'e' (CI, macchine pulite): un caso che dipende da un
// file da 20 GB non deve rompere una suite che gira ovunque.
// ---------------------------------------------------------------------------
import { existsSync, openSync, readSync, closeSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseGguf } from "../src/engine/gguf";
import { validateQwen35 } from "../src/engine/q35shape";
import { q35SlabDesc } from "../src/engine/q35expertstore";
import { slabFileBytes } from "../src/engine/slabfile";

const Q35_PATH = join(homedir(), ".cache/blab-models/q35/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf");
const SHA35 = "a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c";

describe.skipIf(!existsSync(Q35_PATH))("slabgeom: il descrittore del 35B viene dall'header, non da una lista", () => {
  const fd = openSync(Q35_PATH, "r");
  const b = Buffer.alloc(64 * 1024 * 1024);
  readSync(fd, b, 0, b.length, 0);
  closeSync(fd);
  const f = parseGguf(b.buffer.slice(b.byteOffset, b.byteOffset + b.length) as ArrayBuffer);
  const { shape, byName } = validateQwen35(f);
  const info = (n: string) => {
    const t = byName.get(n);
    if (!t) throw new Error(`tensore ${n} assente`);
    return t;
  };
  const g = slabGeometry(q35SlabDesc(shape, info, SHA35));

  it("trova DUE classi e le mette nell'ordine di prima apparizione", () => {
    expect(g.classes.map((c) => c.id)).toEqual(["q4k", "q6k"]);
  });

  it("i layer q6_K sono 34, 38 e 39 — LETTI dal file, non scritti qui", () => {
    const q6 = [...Array(shape.nLayer).keys()].filter((l) => g.classes[g.classOfLayer[l]].id === "q6k");
    expect(q6).toEqual([34, 38, 39]);
    // e sono ALTERNATI: fra il 34 e il 38 ci sono tre layer dell'altra classe,
    // cioe' esattamente cio' che l'aritmetica a confine del GLM non regge
    expect(g.classOfLayer[35]).toBe(g.classOfLayer[33]);
  });

  it("copre tutti i 40 x 256 expert, senza buchi e senza sovrapposizioni", () => {
    expect(g.nSlabs).toBe(shape.nLayer * (shape.nExpert as number));
    const rs = [];
    for (let l = 0; l < shape.nLayer; l++) {
      for (let e = 0; e < (shape.nExpert as number); e++) rs.push(slabRangeOf(g, l, e));
    }
    rs.sort((a, b) => a.offset - b.offset);
    expect(rs[0].offset).toBe(0);
    for (let i = 1; i < rs.length; i++) {
      expect(rs[i].offset, `slab ${i}`).toBe(rs[i - 1].offset + rs[i - 1].bytes);
    }
    expect(rs[rs.length - 1].offset + rs[rs.length - 1].bytes).toBe(g.dataBytes);
  });

  it("il file peserebbe ~17 GiB, cioe' il parco expert e non il modello intero", () => {
    const giB = slabFileBytes(g) / 2 ** 30;
    expect(giB).toBeGreaterThan(16.5);
    expect(giB).toBeLessThan(17.5);
  });

  it("il nome del file porta lo SHA: due quantizzazioni non si sovrascrivono", () => {
    const d = q35SlabDesc(shape, info, SHA35);
    expect(d.fileName).toContain(SHA35.slice(0, 16));
    expect(() => q35SlabDesc(shape, info, "non-uno-sha")).toThrow(/SHA-256/);
  });
});
