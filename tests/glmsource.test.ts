// Test di `src/engine/glmsource.ts` (C3a fase 4d, em.6: "glmsource sotto
// test"). Store in memoria iniettati via GlmSourceDeps: si esercita la logica
// vera di apertura (size-match ⇒ import saltato, SHA canonico all'import,
// byte-check hard), la validazione slab (header valido ⇒ nessuna rigenerazione)
// e l'aritmetica delle letture (range, size-class dei buffer down, slabRange).
// FUORI da qui, dichiarato: il loop di RIGENERAZIONE slab (repack di 2.944
// expert = ~15 GB di I/O — si esercita in produzione a ogni rebuild, e la sua
// decisione `slabFileReason` è già coperta da slabfile.test.ts) e OPFS stesso.
import { describe, expect, it } from "vitest";
import { GlmOpfsSource, GLM47_GGUF_BYTES, type GlmSourceDeps, type SlabStoreLike } from "../src/engine/glmsource";
import { parseGguf, tensorByteSize, type GgufTensorInfo } from "../src/engine/gguf";
import { GLM47_FLASH as G, GLM47_FLASH_SHA256, validateGlm47Flash } from "../src/engine/shape";
import { EXPERT_GATE_UP_BYTES, EXPERT_DOWN_Q4_0_BYTES, EXPERT_DOWN_Q4_1_BYTES, downIsQ4_1 } from "../src/engine/expertstore";
import { buildSlabHeader, slabFileRange, slabFileBytes } from "../src/engine/slabfile";
import { slabGeometry } from "../src/engine/slabgeom";
import { GLM_SLAB_DESC } from "../src/engine/glmsource";

// la geometria del GLM viene dal suo descrittore (it.45): `slabfile` non
// conosce piu' nessun modello, e il nome del file e' dato del modello
const GEO = slabGeometry(GLM_SLAB_DESC);
const SLAB_FILE_NAME = GLM_SLAB_DESC.fileName;
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "../src/engine/moe";
import { glmFixture } from "./helpers/glm-gguf-fixture";

const GGUF_NAME = "GLM-4.7-Flash-Q4_0.gguf";

interface ReadRec { offset: number; bytes: number }

/** Store in memoria: serve `content` da offset 0, zeri oltre; registra le letture. */
class MemStore implements SlabStoreLike {
  reads: ReadRec[] = [];
  imports: Array<{ url: string; sha: string }> = [];
  private content: Uint8Array;
  sizeVal: number;
  private importResult?: { bytes: number; ms: number };
  constructor(content: Uint8Array, sizeVal: number, importResult?: { bytes: number; ms: number }) {
    this.content = content;
    this.sizeVal = sizeVal;
    this.importResult = importResult;
  }
  size(): number { return this.sizeVal; }
  read(offset: number, bytes: number, dst?: Uint8Array): Uint8Array {
    this.reads.push({ offset, bytes });
    const out = dst && dst.length >= bytes ? dst.subarray(0, bytes) : new Uint8Array(bytes);
    out.fill(0);
    if (offset < this.content.length) {
      const n = Math.min(bytes, this.content.length - offset);
      out.set(this.content.subarray(offset, offset + n));
    }
    return out;
  }
  write(data: Uint8Array, _at: number): number { return data.length; }
  truncate(): void { /* no-op */ }
  flush(): void { /* no-op */ }
  close(): void { /* no-op */ }
  async importFromUrl(url: string, sha: string): Promise<{ bytes: number; ms: number }> {
    this.imports.push({ url, sha });
    if (!this.importResult) throw new Error("import non previsto in questo scenario");
    // post-import lo store "ha" il file: il size-check di open() e' gia' passato
    this.sizeVal = GLM47_GGUF_BYTES;
    return this.importResult;
  }
}

const fixtureBytes = new Uint8Array(glmFixture());
// l'ultimo slab del file: i q4_1 stanno davanti, quindi chiude un layer q4_0
const SLAB_FILE_BYTES = slabFileBytes(GEO);

function world(opts: { ggufSize?: number; importResult?: { bytes: number; ms: number } } = {}) {
  const gguf = new MemStore(fixtureBytes, opts.ggufSize ?? GLM47_GGUF_BYTES, opts.importResult);
  const slabs = new MemStore(buildSlabHeader(GEO, GLM47_FLASH_SHA256), SLAB_FILE_BYTES);
  const moves: Array<{ tmp: string; final: string }> = [];
  const deps: GlmSourceDeps = {
    async openStore(name) {
      if (name === GGUF_NAME) return gguf;
      if (name === SLAB_FILE_NAME) return slabs;
      throw new Error(`store inatteso: ${name}`);
    },
    async moveFile(tmp, final) { moves.push({ tmp, final }); },
  };
  return { gguf, slabs, moves, deps };
}

describe("GlmOpfsSource.openWith (store in memoria)", () => {
  it("size-match: import SALTATO, slab valido non rigenerato", async () => {
    const w = world();
    const src = await GlmOpfsSource.openWith(w.deps, "/models/x.gguf");
    expect(w.gguf.imports).toEqual([]);
    expect(src.importMs).toBe(0);
    expect(src.hasSlabs).toBe(true);
    expect(src.slabRebuildReason).toBeNull();
    expect(src.slabBuildMs).toBe(0);
    expect(w.moves).toEqual([]); // nessuna rigenerazione = nessun rename
  });

  it("size mismatch: import con lo SHA canonico, ms riportati", async () => {
    const w = world({ ggufSize: 123, importResult: { bytes: GLM47_GGUF_BYTES, ms: 4321 } });
    const src = await GlmOpfsSource.openWith(w.deps, "/models/x.gguf");
    expect(w.gguf.imports).toEqual([{ url: "/models/x.gguf", sha: GLM47_FLASH_SHA256 }]);
    expect(src.importMs).toBe(4321);
  });

  it("import coi byte sbagliati: THROW (check hard, non warning)", async () => {
    const w = world({ ggufSize: 123, importResult: { bytes: GLM47_GGUF_BYTES - 1, ms: 1 } });
    await expect(GlmOpfsSource.openWith(w.deps, "/x")).rejects.toThrow(/import .* ≠/);
  });
});

describe("letture (range e size-class)", () => {
  const dataOffset = parseGguf(glmFixture()).dataOffset;
  const byName = validateGlm47Flash(parseGguf(glmFixture()));

  async function openSrc() {
    const w = world();
    const src = await GlmOpfsSource.openWith(w.deps, "/x");
    w.gguf.reads.length = 0; // scarta le letture di header/validazione
    w.slabs.reads.length = 0;
    return { w, src };
  }

  it("nonExpert: legge dataOffset+offset per tensorByteSize; tensore assente = throw", async () => {
    const { w, src } = await openSrc();
    const info = byName.get("token_embd.weight") as GgufTensorInfo;
    const out = src.nonExpert("token_embd.weight");
    expect(w.gguf.reads).toEqual([{ offset: dataOffset + info.offset, bytes: tensorByteSize(info) }]);
    expect(out.length).toBe(tensorByteSize(info));
    expect(() => src.nonExpert("blk.99.non_esiste")).toThrow(/assente/);
  });

  it("expert: gate/up a taglia fissa, down nella size-class del layer", async () => {
    const { src } = await openSrc();
    const l41 = G.denseLead; // primo layer MoE: down q4_1
    const l40 = G.nLayer - 1; // ultimo: down q4_0
    expect(downIsQ4_1(l41)).toBe(true);
    expect(downIsQ4_1(l40)).toBe(false);
    const a = src.expert(l41, 0);
    expect(a.gate.length).toBe(EXPERT_GATE_UP_BYTES);
    expect(a.up.length).toBe(EXPERT_GATE_UP_BYTES);
    expect(a.down.length).toBe(EXPERT_DOWN_Q4_1_BYTES);
    expect(src.expert(l40, 63).down.length).toBe(EXPERT_DOWN_Q4_0_BYTES);
  });

  it("expertSlab: legge ESATTAMENTE slabFileRange(GEO, l,e) dal file slab, buffer della classe giusta", async () => {
    const { w, src } = await openSrc();
    const l = G.nLayer - 1, e = 7;
    const r = slabFileRange(GEO, l, e);
    const out = src.expertSlab(l, e);
    expect(w.slabs.reads).toEqual([{ offset: r.offset, bytes: r.bytes }]);
    expect(out.length).toBe(r.bytes);
    expect(r.bytes).toBe(downIsQ4_1(l) ? SLAB_DOWN_Q4_1.bytes : SLAB_DOWN_Q4_0.bytes);
    // la lettura NON tocca il GGUF: lo slab e' gia' impacchettato
    expect(w.gguf.reads).toEqual([]);
  });

  it("header slab valido ma taglia sbagliata ⇒ la validazione la vede (reason ≠ null)", async () => {
    // Non si esercita il loop di rigenerazione (dichiarato in testa al file):
    // si verifica che una taglia monca ARRIVI alla decisione di rigenerare.
    const w = world();
    w.slabs.sizeVal = SLAB_FILE_BYTES - 1;
    // il regen loop partirebbe: lo si intercetta facendo fallire il tmp store
    w.deps.openStore = async (name) => {
      if (name === GGUF_NAME) return w.gguf;
      if (name === SLAB_FILE_NAME) return w.slabs;
      throw new Error(`REGEN-TRIGGERED:${name}`);
    };
    await expect(GlmOpfsSource.openWith(w.deps, "/x")).rejects.toThrow(/REGEN-TRIGGERED:.*\.tmp/);
  });
});
