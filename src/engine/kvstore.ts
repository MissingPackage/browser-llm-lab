// Prefix-cache OPFS (spec B1 §Formato prefix-cache + §Quota/eviction).
//
// Questo modulo ha due metà NETTE:
//  1) codec dell'envelope BKV1 + chiave + politica LRU: PURI, testabili in CI senza
//     OPFS/GPU (convenzione fase A);
//  2) I/O OPFS via FileSystemSyncAccessHandle: SOLO worker dedicato.
//
// Envelope binario little-endian (validazione hard, mismatch ⇒ throw — postura ds4:
// la cache non deve mai né brickare il motore né degradare in silenzio):
//   [0]  magic "BKV1" (u32)        [4]  version envelope (u32)
//   [8]  lastUsedMs (f64, offset FISSO per l'update in place)
//   [16] hitCount (u32)            [20] metaBytes (u32, multiplo di 4)
//   [24] meta JSON (padded a 4 B)  [24+metaBytes] tokens u32[tokenCount]
//   [...] payload f32: per layer, K[tokenCount×kvDim] poi V[tokenCount×kvDim]
// Niente logits nel payload (ruling spec c: si ricalcola 1 forward al restore).

export const KV_MAGIC = 0x31564b42; // "BKV1" LE
export const KV_ENVELOPE_VERSION = 1;
export const KV_LAYOUT_VERSION = 1; // layout della KV del motore (righe per posizione)
export const KV_HEADER_BYTES = 24;
export const KV_LASTUSED_OFFSET = 8;
export const KV_HITCOUNT_OFFSET = 16;
export const KV_BUDGET_DEFAULT = 512 * 1024 * 1024; // ruling spec f

export interface KvMeta {
  modelSha256: string;
  layoutVersion: number;
  nLayer: number;
  kvDim: number;
  ctxMax: number;
  tokenCount: number;
  createdAt: number;
}

export interface KvExpect {
  modelSha256: string;
  layoutVersion: number;
  nLayer: number;
  kvDim: number;
  ctxMax: number;
}

export interface KvCheckpoint {
  meta: KvMeta;
  tokens: Uint32Array;
  payload: Float32Array; // view sull'envelope (per layer: K poi V)
  lastUsedMs: number;
  hitCount: number;
}

const enc = new TextEncoder();
const dec = new TextDecoder();

export function payloadLength(meta: Pick<KvMeta, "nLayer" | "kvDim" | "tokenCount">): number {
  return meta.nLayer * 2 * meta.tokenCount * meta.kvDim;
}

export function encodeKv(meta: KvMeta, tokens: Uint32Array, payload: Float32Array, lastUsedMs: number, hitCount = 0): ArrayBuffer {
  if (tokens.length !== meta.tokenCount) throw new Error(`encodeKv: tokens ${tokens.length} !== tokenCount ${meta.tokenCount}`);
  if (payload.length !== payloadLength(meta)) throw new Error(`encodeKv: payload ${payload.length} !== atteso ${payloadLength(meta)}`);
  if (meta.tokenCount < 1 || meta.tokenCount > meta.ctxMax) throw new Error(`encodeKv: tokenCount fuori range (${meta.tokenCount})`);
  const metaRaw = enc.encode(JSON.stringify(meta));
  const metaBytes = Math.ceil(metaRaw.length / 4) * 4; // pad a 4 B: tokens/payload allineati
  const total = KV_HEADER_BYTES + metaBytes + tokens.length * 4 + payload.length * 4;
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  dv.setUint32(0, KV_MAGIC, true);
  dv.setUint32(4, KV_ENVELOPE_VERSION, true);
  dv.setFloat64(KV_LASTUSED_OFFSET, lastUsedMs, true);
  dv.setUint32(KV_HITCOUNT_OFFSET, hitCount, true);
  dv.setUint32(20, metaBytes, true);
  new Uint8Array(buf, KV_HEADER_BYTES, metaRaw.length).set(metaRaw);
  new Uint8Array(buf, KV_HEADER_BYTES + metaRaw.length, metaBytes - metaRaw.length).fill(0x20); // spazi: JSON.parse li ignora
  new Uint32Array(buf, KV_HEADER_BYTES + metaBytes, tokens.length).set(tokens);
  new Float32Array(buf, KV_HEADER_BYTES + metaBytes + tokens.length * 4, payload.length).set(payload);
  return buf;
}

export function decodeKv(buf: ArrayBuffer, expect: KvExpect): KvCheckpoint {
  if (buf.byteLength < KV_HEADER_BYTES) throw new Error("kvcache: envelope troncato (header)");
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== KV_MAGIC) throw new Error("kvcache: magic mismatch");
  const ver = dv.getUint32(4, true);
  if (ver !== KV_ENVELOPE_VERSION) throw new Error(`kvcache: version envelope ${ver} !== ${KV_ENVELOPE_VERSION}`);
  const lastUsedMs = dv.getFloat64(KV_LASTUSED_OFFSET, true);
  const hitCount = dv.getUint32(KV_HITCOUNT_OFFSET, true);
  const metaBytes = dv.getUint32(20, true);
  if (metaBytes % 4 !== 0 || KV_HEADER_BYTES + metaBytes > buf.byteLength) throw new Error("kvcache: metaBytes non valido");
  let meta: KvMeta;
  try {
    meta = JSON.parse(dec.decode(new Uint8Array(buf, KV_HEADER_BYTES, metaBytes))) as KvMeta;
  } catch {
    throw new Error("kvcache: meta JSON corrotto");
  }
  for (const k of ["modelSha256", "layoutVersion", "nLayer", "kvDim", "ctxMax"] as const) {
    if (meta[k] !== expect[k]) throw new Error(`kvcache: meta.${k} = ${String(meta[k])} !== atteso ${String(expect[k])}`);
  }
  if (!Number.isInteger(meta.tokenCount) || meta.tokenCount < 1 || meta.tokenCount > meta.ctxMax) {
    throw new Error(`kvcache: tokenCount non valido (${meta.tokenCount})`);
  }
  const plen = payloadLength(meta);
  const expected = KV_HEADER_BYTES + metaBytes + meta.tokenCount * 4 + plen * 4;
  if (buf.byteLength !== expected) throw new Error(`kvcache: dimensione ${buf.byteLength} !== attesa ${expected}`);
  return {
    meta,
    tokens: new Uint32Array(buf, KV_HEADER_BYTES + metaBytes, meta.tokenCount),
    payload: new Float32Array(buf, KV_HEADER_BYTES + metaBytes + meta.tokenCount * 4, plen),
    lastUsedMs,
    hitCount,
  };
}

// Chiave (ruling spec a: token-id, non testo): SHA-256 di
// layoutVersion (u32 LE) ‖ sha256(GGUF) (32 byte raw) ‖ tokenIds (u32 LE).
export async function kvKey(layoutVersion: number, modelSha256Hex: string, tokenIds: number[] | Uint32Array): Promise<string> {
  if (!/^[0-9a-f]{64}$/.test(modelSha256Hex)) throw new Error("kvKey: modelSha256 non è hex sha-256");
  const toks = tokenIds instanceof Uint32Array ? tokenIds : Uint32Array.from(tokenIds);
  const buf = new Uint8Array(4 + 32 + toks.length * 4);
  new DataView(buf.buffer).setUint32(0, layoutVersion, true);
  for (let i = 0; i < 32; i++) buf[4 + i] = parseInt(modelSha256Hex.slice(i * 2, i * 2 + 2), 16);
  buf.set(new Uint8Array(toks.buffer, toks.byteOffset, toks.byteLength), 36);
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// File: /kvcache/<primi-32-hex>.kvc (spec §Formato)
export function kvFileName(keyHex: string): string {
  if (!/^[0-9a-f]{64}$/.test(keyHex)) throw new Error("kvFileName: chiave non valida");
  return `${keyHex.slice(0, 32)}.kvc`;
}

// Eviction LRU semplice su lastUsedMs (ruling spec d: niente scoring a densità in
// v1). Pura: dato l'indice, ritorna i nomi da rimuovere perché l'entry in arrivo
// stia nel budget; incoming > budget ⇒ throw (non evacuare tutto per niente).
export interface KvStoreEntry { name: string; size: number; lastUsedMs: number }

export function pickEvictions(entries: KvStoreEntry[], budgetBytes: number, incomingBytes: number): string[] {
  if (incomingBytes > budgetBytes) throw new Error(`kvcache: entry ${incomingBytes} B oltre il budget ${budgetBytes} B`);
  let total = entries.reduce((a, e) => a + e.size, 0);
  if (total + incomingBytes <= budgetBytes) return [];
  const out: string[] = [];
  for (const e of [...entries].sort((a, b) => a.lastUsedMs - b.lastUsedMs)) {
    out.push(e.name);
    total -= e.size;
    if (total + incomingBytes <= budgetBytes) break;
  }
  return out;
}

// ------------------------- I/O OPFS (solo worker dedicato) -------------------------
// SyncAccessHandle: banda misurata nel repo (results/opfs-bench/) write 2.2 GB/s,
// read warm 7.5-11.7 GB/s ⇒ ~10 ms per gli 11.5 MB di un checkpoint da 469 token.

export class KvStoreOpfs {
  dir: FileSystemDirectoryHandle;
  budgetBytes: number;

  constructor(dir: FileSystemDirectoryHandle, budgetBytes = KV_BUDGET_DEFAULT) {
    this.dir = dir;
    this.budgetBytes = budgetBytes;
  }

  static async open(dirName = "kvcache", budgetBytes = KV_BUDGET_DEFAULT): Promise<KvStoreOpfs> {
    const root = await navigator.storage.getDirectory();
    return new KvStoreOpfs(await root.getDirectoryHandle(dirName, { create: true }), budgetBytes);
  }

  async list(): Promise<KvStoreEntry[]> {
    const out: KvStoreEntry[] = [];
    const it = (this.dir as unknown as { entries(): AsyncIterableIterator<[string, FileSystemHandle]> }).entries();
    for await (const [name, h] of it) {
      if (!name.endsWith(".kvc") || h.kind !== "file") continue;
      const f = await (h as FileSystemFileHandle).getFile();
      if (f.size < KV_HEADER_BYTES) { out.push({ name, size: f.size, lastUsedMs: 0 }); continue; }
      const head = new DataView(await f.slice(0, KV_HEADER_BYTES).arrayBuffer());
      out.push({ name, size: f.size, lastUsedMs: head.getFloat64(KV_LASTUSED_OFFSET, true) });
    }
    return out;
  }

  private async writeFile(name: string, bytes: ArrayBuffer): Promise<void> {
    const fh = await this.dir.getFileHandle(name, { create: true });
    const h = await fh.createSyncAccessHandle();
    try {
      h.truncate(0);
      h.write(new Uint8Array(bytes), { at: 0 });
      h.flush();
    } finally {
      h.close();
    }
  }

  // Save con eviction proattiva a budget + retry SINGOLO su QuotaExceededError del
  // filesystem (spec §Quota: evict LRU e retry una volta, poi throw).
  async save(name: string, bytes: ArrayBuffer): Promise<{ evicted: string[] }> {
    const entries = (await this.list()).filter((e) => e.name !== name);
    const evicted = pickEvictions(entries, this.budgetBytes, bytes.byteLength);
    for (const n of evicted) await this.remove(n);
    try {
      await this.writeFile(name, bytes);
    } catch (e) {
      if ((e as DOMException)?.name !== "QuotaExceededError") throw e;
      const left = (await this.list()).filter((x) => x.name !== name).sort((a, b) => a.lastUsedMs - b.lastUsedMs);
      if (!left.length) throw e;
      await this.remove(left[0].name);
      evicted.push(left[0].name);
      await this.writeFile(name, bytes); // se rifallisce: throw naturale (contratto)
    }
    return { evicted };
  }

  // Load + touch LRU in place (lastUsedMs/hitCount a offset fisso, spec §Formato).
  async load(name: string): Promise<ArrayBuffer> {
    const fh = await this.dir.getFileHandle(name); // NotFoundError ⇒ miss, gestito dal chiamante
    const h = await fh.createSyncAccessHandle();
    try {
      const size = h.getSize();
      const buf = new ArrayBuffer(size);
      h.read(new Uint8Array(buf), { at: 0 });
      const dv = new DataView(buf);
      const now = Date.now();
      const hits = size >= KV_HEADER_BYTES ? dv.getUint32(KV_HITCOUNT_OFFSET, true) + 1 : 1;
      const patch = new DataView(new ArrayBuffer(12));
      patch.setFloat64(0, now, true);
      patch.setUint32(8, hits, true);
      h.write(new Uint8Array(patch.buffer), { at: KV_LASTUSED_OFFSET });
      h.flush();
      if (size >= KV_HEADER_BYTES) {
        dv.setFloat64(KV_LASTUSED_OFFSET, now, true);
        dv.setUint32(KV_HITCOUNT_OFFSET, hits, true);
      }
      return buf;
    } finally {
      h.close();
    }
  }

  async remove(name: string): Promise<void> {
    await this.dir.removeEntry(name);
  }
}
