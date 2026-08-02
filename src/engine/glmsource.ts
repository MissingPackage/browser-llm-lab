// GlmWeightSource di PRODUZIONE (goal C2 fase 5 slice 3b): il GGUF vive in
// OPFS (spec §5) — import one-shot con SHA-256 streaming, header parsato da
// OPFS, non-expert per nome, expert per range via GgufExpertIndex. Solo
// worker dedicato (SyncAccessHandle).
import { parseGguf, tensorByteSize, type GgufTensorInfo } from "./gguf";
import { validateGlm47Flash, GLM47_FLASH_SHA256 } from "./shape";
import { ExpertOpfsStore, GgufExpertIndex, downIsQ4_1, EXPERT_GATE_UP_BYTES, EXPERT_DOWN_Q4_0_BYTES, EXPERT_DOWN_Q4_1_BYTES } from "./expertstore";
import { GLM47_FLASH as G } from "./shape";
import { packExpertSlab, SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "./moe";
import {
  SLAB_FILE_NAME, SLAB_HEADER_BYTES, N_SLABS, buildSlabHeader, parseSlabHeader,
  slabFileReason, slabRange,
} from "./slabfile";
import type { GlmWeightSource } from "./glmmodel";
import type { ExpertRawBytes } from "./residency";

// Byte su disco del GGUF canonico (verifier C1 it.1; il model_size di
// llama-bench è la sola sezione dati).
export const GLM47_GGUF_BYTES = 17_216_676_192;
const HEADER_BYTES = 64 * 1024 * 1024;
const OPFS_NAME = "GLM-4.7-Flash-Q4_0.gguf";

export class GlmOpfsSource implements GlmWeightSource {
  private store: ExpertOpfsStore;
  private byName: Map<string, GgufTensorInfo>;
  private dataOffset: number;
  private idx: GgufExpertIndex;
  importMs = 0; // 0 = file già in OPFS (import saltato su size-match: l'hash
                // è stato verificato all'import originale, OPFS è locale)
  // buffer riusati per i range expert: packExpertSlab copia subito, niente aliasing
  private gateBuf = new Uint8Array(EXPERT_GATE_UP_BYTES);
  private upBuf = new Uint8Array(EXPERT_GATE_UP_BYTES);
  private downBuf40 = new Uint8Array(EXPERT_DOWN_Q4_0_BYTES);
  private downBuf41 = new Uint8Array(EXPERT_DOWN_Q4_1_BYTES);
  // file slab (repack all'import): null se non generabile
  private slabs: ExpertOpfsStore | null = null;
  private slabBuf40 = new Uint8Array(SLAB_DOWN_Q4_0.bytes);
  private slabBuf41 = new Uint8Array(SLAB_DOWN_Q4_1.bytes);
  /** ms spesi a generare il file slab (0 = era già valido). */
  slabBuildMs = 0;
  /** motivo della rigenerazione, per il report (null = nessuna). */
  slabRebuildReason: string | null = null;

  private constructor(store: ExpertOpfsStore, byName: Map<string, GgufTensorInfo>, dataOffset: number) {
    this.store = store;
    this.byName = byName;
    this.dataOffset = dataOffset;
    this.idx = new GgufExpertIndex(byName, dataOffset);
  }

  static async open(url: string, onProgress?: (msg: string) => void): Promise<GlmOpfsSource> {
    const store = await ExpertOpfsStore.open(OPFS_NAME);
    let importMs = 0;
    if (store.size() !== GLM47_GGUF_BYTES) {
      onProgress?.(`import OPFS ${OPFS_NAME} (17.2 GB, SHA streaming)…`);
      const r = await store.importFromUrl(url, GLM47_FLASH_SHA256);
      if (r.bytes !== GLM47_GGUF_BYTES) throw new Error(`glmsource: import ${r.bytes} B ≠ ${GLM47_GGUF_BYTES}`);
      importMs = r.ms;
      onProgress?.(`import ok in ${(r.ms / 1000).toFixed(0)} s`);
    }
    const header = store.read(0, HEADER_BYTES);
    const f = parseGguf(header.buffer.slice(header.byteOffset, header.byteOffset + HEADER_BYTES) as ArrayBuffer);
    const byName = validateGlm47Flash(f); // validazione hard (844 tensori, tipi, dims)
    const src = new GlmOpfsSource(store, byName, f.dataOffset);
    src.importMs = importMs;
    const s = await GlmOpfsSource.ensureSlabs(store, src.idx, onProgress);
    src.slabs = s.slabs;
    src.slabBuildMs = s.ms;
    src.slabRebuildReason = s.reason;
    return src;
  }

  /**
   * Slab GIÀ impacchettato (repack all'import, C3a fase 3). Presente solo se
   * il file slab è stato generato e validato; `ExpertCache` lo usa per saltare
   * `packExpertSlab` nel path caldo — i 41,4 ms/token che il goal deve togliere.
   */
  expertSlab(layer: number, expert: number): Uint8Array {
    if (!this.slabs) throw new Error("glmsource: file slab non disponibile");
    const r = slabRange(layer, expert);
    const dst = r.bytes === SLAB_DOWN_Q4_1.bytes ? this.slabBuf41 : this.slabBuf40;
    return this.slabs.read(r.offset, r.bytes, dst);
  }

  get hasSlabs(): boolean {
    return this.slabs !== null;
  }

  /**
   * Genera il file slab se manca o non è valido. Scrive su un file TEMPORANEO
   * e rinomina alla fine: un'interruzione non lascia mai un file valido a metà.
   * Ritorna il motivo della rigenerazione (null = era già buono).
   */
  private static async ensureSlabs(
    store: ExpertOpfsStore, idx: GgufExpertIndex, onProgress?: (msg: string) => void,
  ): Promise<{ slabs: ExpertOpfsStore; reason: string | null; ms: number }> {
    const t0 = performance.now();
    let slabs = await ExpertOpfsStore.open(SLAB_FILE_NAME);
    const size = slabs.size();
    const header = size >= SLAB_HEADER_BYTES ? parseSlabHeader(slabs.read(0, SLAB_HEADER_BYTES)) : null;
    const reason = slabFileReason(header, size, GLM47_FLASH_SHA256);
    if (!reason) return { slabs, reason: null, ms: 0 };

    onProgress?.(`file slab da rigenerare (${reason}) — repack di ${N_SLABS} expert…`);
    slabs.close();
    const tmpName = `${SLAB_FILE_NAME}.tmp`;
    const tmp = await ExpertOpfsStore.open(tmpName);
    tmp.truncate(0);
    tmp.write(buildSlabHeader(GLM47_FLASH_SHA256), 0);
    const gate = new Uint8Array(EXPERT_GATE_UP_BYTES);
    const up = new Uint8Array(EXPERT_GATE_UP_BYTES);
    const down40 = new Uint8Array(EXPERT_DOWN_Q4_0_BYTES);
    const down41 = new Uint8Array(EXPERT_DOWN_Q4_1_BYTES);
    let done = 0;
    for (let l = G.denseLead; l < G.nLayer; l++) {
      const layout = downIsQ4_1(l) ? SLAB_DOWN_Q4_1 : SLAB_DOWN_Q4_0;
      const dbuf = downIsQ4_1(l) ? down41 : down40;
      for (let e = 0; e < G.nExpert; e++) {
        const r = idx.ranges(l, e);
        const slab = packExpertSlab(
          store.read(r.gate.offset, r.gate.bytes, gate),
          store.read(r.up.offset, r.up.bytes, up),
          store.read(r.down.offset, r.down.bytes, dbuf),
          layout,
        );
        tmp.write(slab, slabRange(l, e).offset);
        if (++done % 256 === 0) onProgress?.(`repack ${done}/${N_SLABS} expert…`);
      }
    }
    tmp.flush();
    tmp.close();
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle("models", { create: true });
    await dir.removeEntry(SLAB_FILE_NAME).catch(() => { /* non esisteva */ });
    // move() è l'unico rename atomico disponibile in OPFS
    const th = await dir.getFileHandle(tmpName);
    await (th as FileSystemFileHandle & { move(name: string): Promise<void> }).move(SLAB_FILE_NAME);
    slabs = await ExpertOpfsStore.open(SLAB_FILE_NAME);
    const ms = performance.now() - t0;
    onProgress?.(`file slab pronto in ${(ms / 1000).toFixed(0)} s`);
    return { slabs, reason, ms };
  }

  nonExpert(name: string): Uint8Array {
    const info = this.byName.get(name);
    if (!info) throw new Error(`glmsource: tensore ${name} assente`);
    return this.store.read(this.dataOffset + info.offset, tensorByteSize(info));
  }

  expert(layer: number, e: number): ExpertRawBytes {
    const r = this.idx.ranges(layer, e);
    const down = downIsQ4_1(layer) ? this.downBuf41 : this.downBuf40;
    return {
      gate: this.store.read(r.gate.offset, r.gate.bytes, this.gateBuf),
      up: this.store.read(r.up.offset, r.up.bytes, this.upBuf),
      down: this.store.read(r.down.offset, r.down.bytes, down),
    };
  }

  close(): void {
    this.slabs?.close();
    this.store.close();
  }
}
