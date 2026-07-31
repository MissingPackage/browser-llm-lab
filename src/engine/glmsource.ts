// GlmWeightSource di PRODUZIONE (goal C2 fase 5 slice 3b): il GGUF vive in
// OPFS (spec §5) — import one-shot con SHA-256 streaming, header parsato da
// OPFS, non-expert per nome, expert per range via GgufExpertIndex. Solo
// worker dedicato (SyncAccessHandle).
import { parseGguf, tensorByteSize, type GgufTensorInfo } from "./gguf";
import { validateGlm47Flash, GLM47_FLASH_SHA256 } from "./shape";
import { ExpertOpfsStore, GgufExpertIndex, downIsQ4_1, EXPERT_GATE_UP_BYTES, EXPERT_DOWN_Q4_0_BYTES, EXPERT_DOWN_Q4_1_BYTES } from "./expertstore";
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
    return src;
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
    this.store.close();
  }
}
