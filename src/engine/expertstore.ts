// Sorgente byte degli expert (goal C2 fase 5, spec §5): il GGUF vive in OPFS
// (copiato UNA volta al primo load, SHA-256 verificato in streaming) e i miss
// expert leggono range da lì via FileSystemSyncAccessHandle (solo worker
// dedicato — stessa posture di kvstore.ts). Questo NON è il paging di C3:
// nessuna policy su OPFS, nessun tier, nessun prefetch — OPFS è solo DOVE sta
// il file in un browser. Il path di load resta però compatibile col prefetch
// di C3 (ruling docket C1 item 4): read(offset, len) è già random-access.
import type { GgufTensorInfo } from "./gguf";
import { GLM47_FLASH as G, GLM47_DOWN_EXPS_Q4_1_LAST } from "./shape";

// ---------------------------------------------------------------------------
// SHA-256 incrementale (streaming): crypto.subtle.digest vuole l'intero buffer
// in memoria — impraticabile sui 17.2 GB del GGUF. Implementazione standard
// FIPS 180-4, verificata contro crypto.subtle nei test node. Il costo (~1 min/GB
// in JS puro) si paga UNA volta, all'import; il numero va in telemetria.
// ---------------------------------------------------------------------------

const K256 = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

export class Sha256Stream {
  private h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  private block = new Uint8Array(64);
  private blockLen = 0;
  private totalBytes = 0;
  private w = new Uint32Array(64);

  update(data: Uint8Array): void {
    this.totalBytes += data.length;
    let off = 0;
    if (this.blockLen > 0) {
      const take = Math.min(64 - this.blockLen, data.length);
      this.block.set(data.subarray(0, take), this.blockLen);
      this.blockLen += take;
      off = take;
      if (this.blockLen === 64) { this.compress(this.block, 0); this.blockLen = 0; }
    }
    while (off + 64 <= data.length) { this.compress(data, off); off += 64; }
    if (off < data.length) {
      this.block.set(data.subarray(off), 0);
      this.blockLen = data.length - off;
    }
  }

  hex(): string {
    // padding su una COPIA dello stato: hex() non muta lo stream
    const h = this.h.slice();
    const tail = new Uint8Array(128);
    tail.set(this.block.subarray(0, this.blockLen), 0);
    tail[this.blockLen] = 0x80;
    const padLen = this.blockLen + 1 <= 56 ? 64 : 128;
    const bits = this.totalBytes * 8; // < 2^53: esatto in double (il GGUF è 17.2e9 B)
    const dv = new DataView(tail.buffer);
    dv.setUint32(padLen - 8, Math.floor(bits / 2 ** 32), false);
    dv.setUint32(padLen - 4, bits >>> 0, false);
    const saved = this.h;
    this.h = h;
    this.compress(tail, 0);
    if (padLen === 128) this.compress(tail, 64);
    const out = [...this.h].map((v) => (v >>> 0).toString(16).padStart(8, "0")).join("");
    this.h = saved;
    return out;
  }

  private compress(buf: Uint8Array, off: number): void {
    const w = this.w, h = this.h;
    for (let i = 0; i < 16; i++) {
      const o = off + i * 4;
      w[i] = (buf[o] << 24) | (buf[o + 1] << 16) | (buf[o + 2] << 8) | buf[o + 3];
    }
    for (let i = 16; i < 64; i++) {
      const a = w[i - 15], b = w[i - 2];
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + K256[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      hh = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    h[0] = (h[0] + a) >>> 0; h[1] = (h[1] + b) >>> 0; h[2] = (h[2] + c) >>> 0; h[3] = (h[3] + d) >>> 0;
    h[4] = (h[4] + e) >>> 0; h[5] = (h[5] + f) >>> 0; h[6] = (h[6] + g) >>> 0; h[7] = (h[7] + hh) >>> 0;
  }
}

// ---------------------------------------------------------------------------
// Indice expert → range di byte nel file. I tensori expert sono 3D
// [ne0, ne1, nExpert] row-major: l'expert e è una fetta CONTIGUA di
// tensorByteSize/nExpert byte. Funzione pura: usabile in node (test sul file
// reale) e nel worker (header parsato da OPFS).
// ---------------------------------------------------------------------------

export interface ExpertRange { offset: number; bytes: number } // offset ASSOLUTO nel file

export interface ExpertRanges { gate: ExpertRange; up: ExpertRange; down: ExpertRange }

// 98.304 blocchi per tensore-expert (= (2048/32)·1536 per gate/up e
// (1536/32)·2048 per down); byte: Q4_0 18/blocco, Q4_1 20/blocco.
const EXPERT_BLOCKS = (G.dModel / 32) * G.dFfnExpert;
export const EXPERT_GATE_UP_BYTES = EXPERT_BLOCKS * 18;   // 1.769.472
export const EXPERT_DOWN_Q4_0_BYTES = EXPERT_BLOCKS * 18; // 1.769.472
export const EXPERT_DOWN_Q4_1_BYTES = EXPERT_BLOCKS * 20; // 1.966.080

export function downIsQ4_1(layer: number): boolean {
  return layer >= 1 && layer <= GLM47_DOWN_EXPS_Q4_1_LAST;
}

export class GgufExpertIndex {
  // per layer MoE: offset assoluto di inizio dei tre tensori exps
  private base: Array<{ gate: number; up: number; down: number } | null>;

  // byName = output di validateGlm47Flash (validazione hard GIÀ passata).
  constructor(byName: Map<string, GgufTensorInfo>, dataOffset: number) {
    this.base = Array.from({ length: G.nLayer }, (_, l) => {
      if (l < G.denseLead) return null;
      const t = (n: string): number => {
        const info = byName.get(`blk.${l}.${n}.weight`);
        if (!info) throw new Error(`expert index: blk.${l}.${n} assente`);
        return dataOffset + info.offset;
      };
      return { gate: t("ffn_gate_exps"), up: t("ffn_up_exps"), down: t("ffn_down_exps") };
    });
  }

  ranges(layer: number, expert: number): ExpertRanges {
    const b = this.base[layer];
    if (!b) throw new Error(`expert index: blk.${layer} è denso`);
    if (expert < 0 || expert >= G.nExpert) throw new Error(`expert index: expert ${expert} fuori range`);
    const downBytes = downIsQ4_1(layer) ? EXPERT_DOWN_Q4_1_BYTES : EXPERT_DOWN_Q4_0_BYTES;
    return {
      gate: { offset: b.gate + expert * EXPERT_GATE_UP_BYTES, bytes: EXPERT_GATE_UP_BYTES },
      up: { offset: b.up + expert * EXPERT_GATE_UP_BYTES, bytes: EXPERT_GATE_UP_BYTES },
      down: { offset: b.down + expert * downBytes, bytes: downBytes },
    };
  }
}

// ---------------------------------------------------------------------------
// Store OPFS (solo worker dedicato).
// ---------------------------------------------------------------------------

export interface OpfsImportResult { bytes: number; sha256: string; ms: number }

const IMPORT_CHUNK = 32 * 1024 * 1024;

export class ExpertOpfsStore {
  private handle: FileSystemSyncAccessHandle;

  private constructor(handle: FileSystemSyncAccessHandle) {
    this.handle = handle;
  }

  static async open(fileName: string): Promise<ExpertOpfsStore> {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle("models", { create: true });
    const fh = await dir.getFileHandle(fileName, { create: true });
    return new ExpertOpfsStore(await fh.createSyncAccessHandle());
  }

  size(): number {
    return this.handle.getSize();
  }

  // Copia streaming URL → OPFS con hash incrementale. expectedSha256 vuoto ⇒
  // nessuna verifica (fixture sintetiche nei test); mismatch ⇒ file troncato a
  // 0 e throw (validazione hard, postura ds4).
  async importFromUrl(url: string, expectedSha256: string): Promise<OpfsImportResult> {
    const t0 = performance.now();
    const r = await fetch(url);
    if (!r.ok || !r.body) throw new Error(`opfs import: fetch ${url} → ${r.status}`);
    const sha = new Sha256Stream();
    const h = this.handle;
    h.truncate(0);
    let at = 0;
    const reader = r.body.getReader();
    // riaccorpa i chunk di rete in write da ~32 MB (banda write OPFS misurata 2.2 GB/s)
    let pending: Uint8Array[] = [];
    let pendingBytes = 0;
    const flush = () => {
      for (const c of pending) { h.write(c, { at }); sha.update(c); at += c.length; }
      pending = []; pendingBytes = 0;
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      pending.push(value);
      pendingBytes += value.length;
      if (pendingBytes >= IMPORT_CHUNK) flush();
    }
    flush();
    h.flush();
    const digest = sha.hex();
    if (expectedSha256 && digest !== expectedSha256) {
      h.truncate(0);
      h.flush();
      throw new Error(`opfs import: SHA-256 mismatch (${digest} ≠ ${expectedSha256})`);
    }
    return { bytes: at, sha256: digest, ms: performance.now() - t0 };
  }

  // Read sincrona di un range (miss expert nel forward: lo stallo è dichiarato
  // e misurato — è il numero che C3 deve battere).
  read(offset: number, bytes: number, dst?: Uint8Array): Uint8Array {
    const out = dst ?? new Uint8Array(bytes);
    if (out.length !== bytes) throw new Error("opfs read: dst di taglia sbagliata");
    const got = this.handle.read(out, { at: offset });
    if (got !== bytes) throw new Error(`opfs read: ${got}/${bytes} B a offset ${offset}`);
    return out;
  }

  close(): void {
    this.handle.close();
  }
}
