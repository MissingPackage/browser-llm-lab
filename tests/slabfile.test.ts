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
  N_SLABS, N_SLABS_Q4_0, N_SLABS_Q4_1, SLAB_DATA_BYTES, SLAB_FILE_BYTES,
  SLAB_HEADER_BYTES, SLAB_LAYOUT_VERSION, buildSlabHeader, parseSlabHeader,
  slabFileReason, slabIndex, slabRange,
} from "../src/engine/slabfile";

const SHA = GLM47_FLASH_SHA256;
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
    const h = parseSlabHeader(buildSlabHeader(SHA))!;
    expect(h.layoutVersion).toBe(SLAB_LAYOUT_VERSION);
    expect(h.sourceSha256).toBe(SHA);
    expect(h.nSlabsQ4_1).toBe(N_SLABS_Q4_1);
    expect(h.nSlabsQ4_0).toBe(N_SLABS_Q4_0);
    expect(h.dataBytes).toBe(SLAB_DATA_BYTES); // > 2^32: low/high word
  });

  it("un header valido e la taglia giusta ⇒ nessuna rigenerazione", () => {
    expect(slabFileReason(parseSlabHeader(buildSlabHeader(SHA)), SLAB_FILE_BYTES, SHA)).toBeNull();
  });

  it("ogni motivo di rigenerazione e' DICHIARATO, non silenzioso", () => {
    const h = parseSlabHeader(buildSlabHeader(SHA))!;
    expect(slabFileReason(null, SLAB_FILE_BYTES, SHA)).toMatch(/magic/);
    expect(slabFileReason({ ...h, layoutVersion: 99 }, SLAB_FILE_BYTES, SHA)).toMatch(/versione di layout/);
    expect(slabFileReason(h, SLAB_FILE_BYTES, "0".repeat(64))).toMatch(/SHA-256/);
    expect(slabFileReason({ ...h, nSlabsQ4_0: 1 }, SLAB_FILE_BYTES, SHA)).toMatch(/conteggio slab/);
    expect(slabFileReason({ ...h, slabBytesQ4_0: 1 }, SLAB_FILE_BYTES, SHA)).toMatch(/taglia/);
    expect(slabFileReason(h, SLAB_FILE_BYTES - 1, SHA)).toMatch(/file di/);
  });

  it("magic sbagliato ⇒ parse null (non si legge mai roba di ignota provenienza)", () => {
    const buf = buildSlabHeader(SHA);
    buf[3] = 0x00;
    expect(parseSlabHeader(buf)).toBeNull();
  });

  it("un file troncato a meta' non passa per valido", () => {
    const h = parseSlabHeader(buildSlabHeader(SHA));
    expect(slabFileReason(h, Math.floor(SLAB_FILE_BYTES / 2), SHA)).not.toBeNull();
  });

  it("SHA sorgente non valido ⇒ throw alla costruzione dell'header", () => {
    expect(() => buildSlabHeader("non-uno-sha")).toThrow();
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

describe.skipIf(!existsSync(GGUF_PATH) || !slabPath)("file slab generato vs GGUF", () => {
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
