import { describe, it, expect } from "vitest";
import {
  encodeKv, decodeKv, kvKey, kvFileName, pickEvictions, payloadLength,
  KV_HEADER_BYTES, KV_LASTUSED_OFFSET, KV_HITCOUNT_OFFSET, KV_MAGIC,
  type KvMeta, type KvExpect,
} from "../src/engine/kvstore";

// Unit CI (zero OPFS/GPU) sul codec envelope BKV1, la chiave e la politica LRU —
// la metà pura di kvstore.ts (spec B1 §Formato prefix-cache / §Quota-eviction).

const SHA = "a".repeat(64);
const meta = (over: Partial<KvMeta> = {}): KvMeta => ({
  modelSha256: SHA, layoutVersion: 1, nLayer: 2, kvDim: 4, ctxMax: 16,
  tokenCount: 3, createdAt: 1753858800000, ...over,
});
const expect_: KvExpect = { modelSha256: SHA, layoutVersion: 1, nLayer: 2, kvDim: 4, ctxMax: 16 };
const mkPayload = (m: KvMeta) => Float32Array.from({ length: payloadLength(m) }, (_, i) => i / 7);
const mkTokens = (m: KvMeta) => Uint32Array.from({ length: m.tokenCount }, (_, i) => 100 + i);

describe("envelope BKV1 — roundtrip e layout", () => {
  it("encode→decode preserva meta, tokens, payload, lastUsed/hitCount", () => {
    const m = meta();
    const buf = encodeKv(m, mkTokens(m), mkPayload(m), 123456.5, 7);
    const ck = decodeKv(buf, expect_);
    expect(ck.meta).toEqual(m);
    expect([...ck.tokens]).toEqual([100, 101, 102]);
    expect([...ck.payload]).toEqual([...mkPayload(m)]);
    expect(ck.lastUsedMs).toBe(123456.5);
    expect(ck.hitCount).toBe(7);
  });

  it("header a offset fissi: magic/metaBytes allineati, lastUsed/hitCount patchabili in place", () => {
    const m = meta();
    const buf = encodeKv(m, mkTokens(m), mkPayload(m), 1, 0);
    const dv = new DataView(buf);
    expect(dv.getUint32(0, true)).toBe(KV_MAGIC);
    expect(dv.getUint32(20, true) % 4).toBe(0); // metaBytes padded ⇒ payload allineato
    // update in place come farebbe il touch LRU dello store
    dv.setFloat64(KV_LASTUSED_OFFSET, 999.25, true);
    dv.setUint32(KV_HITCOUNT_OFFSET, 42, true);
    const ck = decodeKv(buf, expect_);
    expect(ck.lastUsedMs).toBe(999.25);
    expect(ck.hitCount).toBe(42);
    expect([...ck.payload]).toEqual([...mkPayload(m)]); // il patch non tocca il payload
  });

  it("dimensione: 469 token full-shape ≈ 11.5 MB (sanity della stima di spec)", () => {
    const m = meta({ nLayer: 24, kvDim: 128, ctxMax: 1024, tokenCount: 469 });
    expect(payloadLength(m) * 4).toBe(24 * 2 * 469 * 128 * 4); // ~11.5 MB
  });
});

describe("envelope BKV1 — validazione hard (mismatch ⇒ throw)", () => {
  const good = () => {
    const m = meta();
    return encodeKv(m, mkTokens(m), mkPayload(m), 1, 0);
  };

  it("magic/version corrotti", () => {
    const b1 = good(); new DataView(b1).setUint32(0, 0xdeadbeef, true);
    expect(() => decodeKv(b1, expect_)).toThrow(/magic/);
    const b2 = good(); new DataView(b2).setUint32(4, 99, true);
    expect(() => decodeKv(b2, expect_)).toThrow(/version/);
  });

  it("meta divergente dall'atteso: modelSha256, layoutVersion, nLayer, kvDim, ctxMax", () => {
    const b = good();
    expect(() => decodeKv(b, { ...expect_, modelSha256: "b".repeat(64) })).toThrow(/modelSha256/);
    expect(() => decodeKv(b, { ...expect_, layoutVersion: 2 })).toThrow(/layoutVersion/);
    expect(() => decodeKv(b, { ...expect_, nLayer: 3 })).toThrow(/nLayer/);
    expect(() => decodeKv(b, { ...expect_, kvDim: 8 })).toThrow(/kvDim/);
    expect(() => decodeKv(b, { ...expect_, ctxMax: 32 })).toThrow(/ctxMax/);
  });

  it("troncature e meta corrotto", () => {
    const b = good();
    expect(() => decodeKv(b.slice(0, 10), expect_)).toThrow(/troncato/);
    expect(() => decodeKv(b.slice(0, b.byteLength - 4), expect_)).toThrow(/dimensione/);
    const b2 = good();
    new Uint8Array(b2, KV_HEADER_BYTES, 4).fill(0xff); // sporca il JSON
    expect(() => decodeKv(b2, expect_)).toThrow(/meta JSON/);
    const b3 = good();
    new DataView(b3).setUint32(20, b3.byteLength, true); // metaBytes oltre il file
    expect(() => decodeKv(b3, expect_)).toThrow(/metaBytes/);
  });

  it("encode: incoerenze tokens/payload/tokenCount ⇒ throw", () => {
    const m = meta();
    expect(() => encodeKv(m, new Uint32Array(2), mkPayload(m), 1)).toThrow(/tokens/);
    expect(() => encodeKv(m, mkTokens(m), new Float32Array(5), 1)).toThrow(/payload/);
    expect(() => encodeKv(meta({ tokenCount: 0 }), new Uint32Array(0), new Float32Array(0), 1)).toThrow();
    expect(() => encodeKv(meta({ tokenCount: 17 }), new Uint32Array(17), new Float32Array(payloadLength(meta({ tokenCount: 17 }))), 1)).toThrow(/fuori range/);
  });
});

describe("chiave prefix-cache (token-id, ruling spec a)", () => {
  it("deterministica; cambia con tokens, modello e layoutVersion", async () => {
    const k1 = await kvKey(1, SHA, [1, 2, 3]);
    expect(k1).toMatch(/^[0-9a-f]{64}$/);
    expect(await kvKey(1, SHA, [1, 2, 3])).toBe(k1);
    expect(await kvKey(1, SHA, [1, 2, 4])).not.toBe(k1);
    expect(await kvKey(1, SHA, [1, 2])).not.toBe(k1);
    expect(await kvKey(2, SHA, [1, 2, 3])).not.toBe(k1);
    expect(await kvKey(1, "b".repeat(64), [1, 2, 3])).not.toBe(k1);
  });

  it("filename = primi 32 hex + .kvc; input non validi ⇒ throw", async () => {
    const k = await kvKey(1, SHA, [5]);
    expect(kvFileName(k)).toBe(`${k.slice(0, 32)}.kvc`);
    expect(() => kvFileName("xyz")).toThrow();
    await expect(kvKey(1, "nothex", [1])).rejects.toThrow();
  });
});

describe("eviction LRU (ruling spec d: semplice, su lastUsedMs)", () => {
  const E = (name: string, size: number, lastUsedMs: number) => ({ name, size, lastUsedMs });

  it("niente eviction se l'entry ci sta", () => {
    expect(pickEvictions([E("a", 100, 1)], 1000, 200)).toEqual([]);
  });

  it("evict in ordine LRU finché l'entry in arrivo ci sta", () => {
    const entries = [E("nuovo", 400, 30), E("vecchio", 400, 10), E("medio", 400, 20)];
    expect(pickEvictions(entries, 1000, 300)).toEqual(["vecchio", "medio"]);
    expect(pickEvictions(entries, 1000, 100)).toEqual(["vecchio"]);
  });

  it("incoming oltre il budget ⇒ throw (mai evacuare tutto per niente)", () => {
    expect(() => pickEvictions([], 100, 101)).toThrow(/budget/);
  });

  it("budget esatto: nessuna eviction superflua", () => {
    expect(pickEvictions([E("a", 500, 1)], 1000, 500)).toEqual([]);
    expect(pickEvictions([E("a", 501, 1)], 1000, 500)).toEqual(["a"]);
  });
});
