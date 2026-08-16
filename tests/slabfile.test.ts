// Formato del file slab (repack all'import, C3a fase 3) — aritmetica degli
// offset, header e regole di invalidazione. Il roundtrip vero
// (OPFS → slab → upload → readback coi byte del GGUF reale) sta nel ktest:
// serve browser + GPU.
import { describe, expect, it } from "vitest";
import { existsSync, openSync, readSync, closeSync, statSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseGguf } from "../src/engine/gguf";
import { GgufExpertIndex } from "../src/engine/expertstore";
import { packExpertSlab } from "../src/engine/moe";
import { GLM47_FLASH as G, GLM47_FLASH_SHA256, validateGlm47Flash } from "../src/engine/shape";
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "../src/engine/moe";
import { downIsQ4_1 } from "../src/engine/expertstore";
import {
  SLAB_HEADER_BYTES, SLAB_LAYOUT_VERSION, buildSlabHeader, parseSlabHeader,
  slabFileReason, slabFileRange, slabFileBytes,
} from "../src/engine/slabfile";
import { slabGeometry } from "../src/engine/slabgeom";
import { GLM_SLAB_DESC } from "../src/engine/glmsource";

const SHA = GLM47_FLASH_SHA256;

// La geometria del GLM, derivata dal suo descrittore (it.45): `slabfile.ts` non
// conosce piu' nessun modello, quindi i numeri che questi casi pinnavano come
// costanti del modulo ora si chiedono alla geometria. **Restano gli stessi
// numeri**, ed e' il punto: il refactor non doveva spostare un byte.
const GEO = slabGeometry(GLM_SLAB_DESC);
const N_SLABS = GEO.nSlabs;
const N_SLABS_Q4_1 = GEO.classes[0].nSlabs;
const N_SLABS_Q4_0 = GEO.classes[1].nSlabs;
const SLAB_DATA_BYTES = GEO.dataBytes;
const SLAB_FILE_BYTES = slabFileBytes(GEO);
const slabRange = (l: number, e: number): { offset: number; bytes: number } =>
  slabFileRange(GEO, l, e);
/** l'indice sequenziale non e' piu' un'API: si deriva dall'offset dentro la classe */
const slabIndex = (l: number, e: number): number => {
  if (l < 0 || l >= GEO.desc.nLayer) throw new Error(`slabfile: layer ${l} fuori dal modello`);
  const ci = GEO.classOfLayer[l];
  if (ci < 0) throw new Error(`slabfile: layer ${l} non e' MoE`);
  if (e < 0 || e >= GEO.desc.nExpert) throw new Error(`slabfile: expert ${e} fuori range`);
  let before = 0;
  for (let i = 0; i < ci; i++) before += GEO.classes[i].nSlabs;
  return before + GEO.rankOfLayer[l] * GEO.desc.nExpert + e;
};
const allExperts = function* (): Generator<[number, number]> {
  for (let l = G.denseLead; l < G.nLayer; l++) for (let e = 0; e < G.nExpert; e++) yield [l, e];
};

describe("indice e offset degli slab", () => {
  it("conteggi coerenti col parco (2.944 = 256 q4_1 + 2.688 q4_0)", () => {
    expect(N_SLABS_Q4_1).toBe(256);
    expect(N_SLABS_Q4_0).toBe(2688);
    expect(N_SLABS).toBe(2944);
    expect(SLAB_DATA_BYTES).toBe(15_678_308_352);
  });

  it("slabIndex e' una BIIEZIONE su tutti i 2.944 expert", () => {
    const seen = new Set<number>();
    for (const [l, e] of allExperts()) {
      const i = slabIndex(l, e);
      expect(i).toBeGreaterThanOrEqual(0);
      expect(i).toBeLessThan(N_SLABS);
      expect(seen.has(i), `collisione su blk.${l} expert ${e}`).toBe(false);
      seen.add(i);
    }
    expect(seen.size).toBe(N_SLABS);
  });

  it("gli slab NON si sovrappongono e coprono il file senza buchi", () => {
    const ranges = [...allExperts()].map(([l, e]) => slabRange(l, e)).sort((a, b) => a.offset - b.offset);
    expect(ranges[0].offset).toBe(SLAB_HEADER_BYTES);
    for (let i = 1; i < ranges.length; i++) {
      expect(ranges[i].offset, `slab ${i}`).toBe(ranges[i - 1].offset + ranges[i - 1].bytes);
    }
    const last = ranges[ranges.length - 1];
    expect(last.offset + last.bytes).toBe(SLAB_FILE_BYTES);
  });

  it("la taglia dello slab corrisponde SEMPRE alla size-class del layer", () => {
    for (const [l, e] of allExperts()) {
      const want = downIsQ4_1(l) ? SLAB_DOWN_Q4_1.bytes : SLAB_DOWN_Q4_0.bytes;
      expect(slabRange(l, e).bytes, `blk.${l}`).toBe(want);
    }
  });

  it("i q4_1 stanno DAVANTI ai q4_0 (e' cio' che rende l'offset aritmetico)", () => {
    const lastQ41 = slabRange(4, G.nExpert - 1);
    const firstQ40 = slabRange(5, 0);
    expect(lastQ41.offset + lastQ41.bytes).toBe(firstQ40.offset);
    expect(lastQ41.bytes).toBe(SLAB_DOWN_Q4_1.bytes);
    expect(firstQ40.bytes).toBe(SLAB_DOWN_Q4_0.bytes);
  });

  it("layer denso ed expert fuori range → throw (validazione hard)", () => {
    expect(() => slabIndex(0, 0)).toThrow();
    expect(() => slabIndex(G.nLayer, 0)).toThrow();
    expect(() => slabIndex(5, G.nExpert)).toThrow();
    expect(() => slabIndex(5, -1)).toThrow();
  });

  it("gli offset restano allineati a 256 (requisito dei bind range)", () => {
    for (const [l, e] of allExperts()) expect(slabRange(l, e).offset % 256).toBe(0);
  });
});

describe("header e invalidazione", () => {
  it("roundtrip build → parse", () => {
    const h = parseSlabHeader(buildSlabHeader(GEO, SHA))!;
    expect(h.layoutVersion).toBe(SLAB_LAYOUT_VERSION);
    expect(h.sourceSha256).toBe(SHA);
    // v2: la LISTA delle classi invece di due coppie cablate — ma i numeri del
    // GLM restano gli stessi, ed e' il punto del refactor
    expect(h.classes.map((c) => c.nSlabs)).toEqual([N_SLABS_Q4_1, N_SLABS_Q4_0]);
    expect(h.classes.map((c) => c.bytes)).toEqual([SLAB_DOWN_Q4_1.bytes, SLAB_DOWN_Q4_0.bytes]);
    expect(h.dataBytes).toBe(SLAB_DATA_BYTES); // > 2^32: low/high word
  });

  it("un header valido e la taglia giusta ⇒ nessuna rigenerazione", () => {
    expect(slabFileReason(parseSlabHeader(buildSlabHeader(GEO, SHA)), SLAB_FILE_BYTES, SHA, GEO)).toBeNull();
  });

  it("ogni motivo di rigenerazione e' DICHIARATO, non silenzioso", () => {
    const h = parseSlabHeader(buildSlabHeader(GEO, SHA))!;
    const cls = (i: number, patch: Partial<{ nSlabs: number; bytes: number }>) =>
      ({ ...h, classes: h.classes.map((c, j) => (j === i ? { ...c, ...patch } : c)) });
    expect(slabFileReason(null, SLAB_FILE_BYTES, SHA, GEO)).toMatch(/magic/);
    expect(slabFileReason({ ...h, layoutVersion: 99 }, SLAB_FILE_BYTES, SHA, GEO)).toMatch(/versione di layout/);
    expect(slabFileReason(h, SLAB_FILE_BYTES, "0".repeat(64), GEO)).toMatch(/SHA-256/);
    expect(slabFileReason(cls(1, { nSlabs: 1 }), SLAB_FILE_BYTES, SHA, GEO)).toMatch(/size-class/);
    expect(slabFileReason(cls(1, { bytes: 1 }), SLAB_FILE_BYTES, SHA, GEO)).toMatch(/size-class/);
    // una classe in meno: il caso che v1 non poteva nemmeno esprimere
    expect(slabFileReason({ ...h, classes: h.classes.slice(0, 1) }, SLAB_FILE_BYTES, SHA, GEO))
      .toMatch(/size-class/);
    expect(slabFileReason(h, SLAB_FILE_BYTES - 1, SHA, GEO)).toMatch(/file di/);
  });

  it("magic sbagliato ⇒ parse null (non si legge mai roba di ignota provenienza)", () => {
    const buf = buildSlabHeader(GEO, SHA);
    buf[3] = 0x00;
    expect(parseSlabHeader(buf)).toBeNull();
  });

  it("un file troncato a meta' non passa per valido", () => {
    const h = parseSlabHeader(buildSlabHeader(GEO, SHA));
    expect(slabFileReason(h, Math.floor(SLAB_FILE_BYTES / 2), SHA, GEO)).not.toBeNull();
  });

  it("SHA sorgente non valido ⇒ throw alla costruzione dell'header", () => {
    expect(() => buildSlabHeader(GEO, "non-uno-sha")).toThrow();
  });
});


// ---------------------------------------------------------------------------
// Verifica del FILE generato, non solo del formato: gli slab su disco devono
// essere byte-identici a `packExpertSlab` sui byte grezzi del GGUF.
// I test unitari sopra provano l'aritmetica degli offset, e
// `engine-residency.test.ts` prova che il percorso rapido carica in VRAM gli
// stessi byte del percorso lento — ma nessuno dei due guarda il FILE. Questo sì.
// Si salta se il GGUF o il profilo E2E non ci sono (CI, macchine pulite).
// ---------------------------------------------------------------------------
const GGUF_PATH = join(homedir(), ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf");
const FS_DIR = join(homedir(), ".cache/blab-glmroute-profile/Default/File System/000/t/00");
const findSlabFile = (): string | null => {
  if (!existsSync(FS_DIR)) return null;
  for (const f of readdirSync(FS_DIR)) {
    const p = join(FS_DIR, f);
    try { if (statSync(p).size === SLAB_FILE_BYTES) return p; } catch { /* ignora */ }
  }
  return null;
};
const slabPath = findSlabFile();

/**
 * Il file su disco e' del formato che sappiamo leggere?
 *
 * Da it.45 il layout e' v2 (lista delle classi nell'header invece di due coppie
 * cablate) e `parseSlabHeader` **rifiuta** un v1 — che e' il comportamento
 * giusto: leggere byte con la geometria sbagliata darebbe slab validi e
 * sbagliati. Un file v1 sul disco non e' un fallimento di questo caso, e' un
 * file che `ensureSlabs` rigenerera' al prossimo load dichiarandone il motivo.
 *
 * Lo si SALTA dicendo perche', invece di fallire (rumore su un fatto atteso) o
 * di passare in silenzio (che nasconderebbe un file davvero corrotto).
 */
const slabHeaderReadable = (): boolean => {
  if (!slabPath) return false;
  const fd = openSync(slabPath, "r");
  try {
    const b = Buffer.alloc(SLAB_HEADER_BYTES);
    readSync(fd, b, 0, SLAB_HEADER_BYTES, 0);
    const h = parseSlabHeader(new Uint8Array(b.buffer, b.byteOffset, SLAB_HEADER_BYTES));
    if (h === null) {
      console.log("[slabfile] file slab su disco non leggibile col layout v"
        + `${SLAB_LAYOUT_VERSION} (verosimilmente v1): il caso sul FILE si salta,`
        + " `ensureSlabs` lo rigenerera' al prossimo load");
      return false;
    }
    return true;
  } finally { closeSync(fd); }
};

describe.skipIf(!existsSync(GGUF_PATH) || !slabPath || !slabHeaderReadable())("file slab generato vs GGUF", () => {
  it("gli slab su disco sono identici a packExpertSlab(byte GGUF)", () => {
    const gfd = openSync(GGUF_PATH, "r");
    const sfd = openSync(slabPath!, "r");
    try {
      const hdr = Buffer.alloc(64 * 1024 * 1024);
      readSync(gfd, hdr, 0, hdr.length, 0);
      const f = parseGguf(hdr.buffer.slice(hdr.byteOffset, hdr.byteOffset + hdr.length) as ArrayBuffer);
      const idx = new GgufExpertIndex(validateGlm47Flash(f), f.dataOffset);
      const read = (fd: number, off: number, n: number): Uint8Array => {
        const b = Buffer.alloc(n);
        readSync(fd, b, 0, n, off);
        return new Uint8Array(b.buffer, b.byteOffset, n);
      };
      // header: provenienza dichiarata
      expect(parseSlabHeader(read(sfd, 0, SLAB_HEADER_BYTES))!.sourceSha256).toBe(GLM47_FLASH_SHA256);
      // campione: estremi e centro di entrambe le size-class
      for (const [l, e] of [[1, 0], [4, 63], [2, 31], [5, 0], [46, 63], [20, 17]] as Array<[number, number]>) {
        const r = idx.ranges(l, e);
        const atteso = packExpertSlab(
          read(gfd, r.gate.offset, r.gate.bytes), read(gfd, r.up.offset, r.up.bytes),
          read(gfd, r.down.offset, r.down.bytes), downIsQ4_1(l) ? SLAB_DOWN_Q4_1 : SLAB_DOWN_Q4_0);
        const sr = slabRange(l, e);
        const got = read(sfd, sr.offset, sr.bytes);
        expect(got.length, `blk.${l}/${e} taglia`).toBe(atteso.length);
        let diff = -1;
        for (let j = 0; j < atteso.length; j++) if (atteso[j] !== got[j]) { diff = j; break; }
        expect(diff, `blk.${l} expert ${e}: primo byte diverso`).toBe(-1);
      }
    } finally { closeSync(gfd); closeSync(sfd); }
  }, 120_000);
});
