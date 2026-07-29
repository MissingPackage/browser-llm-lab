import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { parseGguf, tensorByteSize, GGML_TYPE, GGUF_KV_TYPE } from "../src/engine/gguf";
import {
  f16ToF32, dequantQ4_0, dequantQ8_0, Q4_0_BLOCK_BYTES,
} from "../src/engine/quant";
import { validateQwen25_05B, QWEN25_05B } from "../src/engine/shape";

// --- builder di fixture GGUF v3 sintetiche (solo header: la sezione dati può
// --- restare vuota, il parser non la legge) ---

type KvVal =
  | { t: "u32"; v: number } | { t: "f32"; v: number } | { t: "str"; v: string }
  | { t: "bool"; v: boolean } | { t: "u64"; v: number } | { t: "arr_str"; v: string[] };

function buildGguf(
  kv: Record<string, KvVal>,
  tensors: { name: string; dims: number[]; type: number; offset: number }[],
): ArrayBuffer {
  const bytes: number[] = [];
  const enc = new TextEncoder();
  const pushU32 = (x: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, x, true); bytes.push(...b); };
  const pushU64 = (x: number) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(x), true); bytes.push(...b); };
  const pushF32 = (x: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, x, true); bytes.push(...b); };
  const pushStr = (s: string) => { const u = enc.encode(s); pushU64(u.length); bytes.push(...u); };

  pushU32(0x46554747); // "GGUF"
  pushU32(3);
  pushU64(tensors.length);
  pushU64(Object.keys(kv).length);
  for (const [key, val] of Object.entries(kv)) {
    pushStr(key);
    if (val.t === "u32") { pushU32(GGUF_KV_TYPE.UINT32); pushU32(val.v); }
    else if (val.t === "f32") { pushU32(GGUF_KV_TYPE.FLOAT32); pushF32(val.v); }
    else if (val.t === "str") { pushU32(GGUF_KV_TYPE.STRING); pushStr(val.v); }
    else if (val.t === "bool") { pushU32(GGUF_KV_TYPE.BOOL); bytes.push(val.v ? 1 : 0); }
    else if (val.t === "u64") { pushU32(GGUF_KV_TYPE.UINT64); pushU64(val.v); }
    else { pushU32(GGUF_KV_TYPE.ARRAY); pushU32(GGUF_KV_TYPE.STRING); pushU64(val.v.length); for (const s of val.v) pushStr(s); }
  }
  for (const t of tensors) {
    pushStr(t.name);
    pushU32(t.dims.length);
    for (const d of t.dims) pushU64(d);
    pushU32(t.type);
    pushU64(t.offset);
  }
  return new Uint8Array(bytes).buffer;
}

// KV + inventario completo Qwen2.5-0.5B (dims/tipi come nel file reale, dati vuoti)
function qwenFixture(mutate?: (ts: { name: string; dims: number[]; type: number; offset: number }[]) => void): ArrayBuffer {
  const S = QWEN25_05B;
  const ts: { name: string; dims: number[]; type: number; offset: number }[] = [
    { name: "token_embd.weight", dims: [S.dModel, S.vocab], type: GGML_TYPE.Q4_0, offset: 0 },
    { name: "output.weight", dims: [S.dModel, S.vocab], type: GGML_TYPE.Q8_0, offset: 0 },
    { name: "output_norm.weight", dims: [S.dModel], type: GGML_TYPE.F32, offset: 0 },
  ];
  for (let l = 0; l < S.nLayer; l++) {
    const kv = S.nKvHead * S.headDim;
    ts.push(
      { name: `blk.${l}.attn_norm.weight`, dims: [S.dModel], type: GGML_TYPE.F32, offset: 0 },
      { name: `blk.${l}.attn_q.weight`, dims: [S.dModel, S.dModel], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.attn_q.bias`, dims: [S.dModel], type: GGML_TYPE.F32, offset: 0 },
      { name: `blk.${l}.attn_k.weight`, dims: [S.dModel, kv], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.attn_k.bias`, dims: [kv], type: GGML_TYPE.F32, offset: 0 },
      { name: `blk.${l}.attn_v.weight`, dims: [S.dModel, kv], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.attn_v.bias`, dims: [kv], type: GGML_TYPE.F32, offset: 0 },
      { name: `blk.${l}.attn_output.weight`, dims: [S.dModel, S.dModel], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.ffn_norm.weight`, dims: [S.dModel], type: GGML_TYPE.F32, offset: 0 },
      { name: `blk.${l}.ffn_gate.weight`, dims: [S.dModel, S.dFfn], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.ffn_up.weight`, dims: [S.dModel, S.dFfn], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.ffn_down.weight`, dims: [S.dFfn, S.dModel], type: GGML_TYPE.Q4_0, offset: 0 },
    );
  }
  mutate?.(ts);
  return buildGguf({
    "general.architecture": { t: "str", v: "qwen2" },
    "qwen2.block_count": { t: "u32", v: S.nLayer },
    "qwen2.embedding_length": { t: "u32", v: S.dModel },
    "qwen2.feed_forward_length": { t: "u32", v: S.dFfn },
    "qwen2.attention.head_count": { t: "u32", v: S.nHead },
    "qwen2.attention.head_count_kv": { t: "u32", v: S.nKvHead },
    "qwen2.rope.freq_base": { t: "f32", v: S.ropeFreqBase },
  }, ts);
}

describe("gguf parser", () => {
  it("parse di una fixture minima: metadata tipati + tensor info + alignment", () => {
    const buf = buildGguf(
      {
        "general.architecture": { t: "str", v: "qwen2" },
        "x.count": { t: "u32", v: 24 },
        "x.flag": { t: "bool", v: true },
        "x.big": { t: "u64", v: 2 ** 40 },
        "x.eps": { t: "f32", v: 1e-6 },
        "x.arr": { t: "arr_str", v: ["a", "bb"] },
      },
      [{ name: "t0.weight", dims: [32, 4], type: GGML_TYPE.Q4_0, offset: 0 }],
    );
    const f = parseGguf(buf);
    expect(f.version).toBe(3);
    expect(f.metadata["general.architecture"]).toBe("qwen2");
    expect(f.metadata["x.count"]).toBe(24);
    expect(f.metadata["x.flag"]).toBe(true);
    expect(f.metadata["x.big"]).toBe(2 ** 40);
    expect(f.metadata["x.eps"]).toBeCloseTo(1e-6, 12);
    expect(f.metadata["x.arr"]).toEqual(["a", "bb"]);
    expect(f.tensors).toHaveLength(1);
    expect(f.tensors[0]).toEqual({ name: "t0.weight", dims: [32, 4], type: GGML_TYPE.Q4_0, offset: 0 });
    expect(f.dataOffset % 32).toBe(0);
  });

  it("magic/version sbagliati -> throw", () => {
    expect(() => parseGguf(new ArrayBuffer(16))).toThrow(/magic/);
    const buf = buildGguf({}, []);
    new DataView(buf).setUint32(4, 2, true); // version 2
    expect(() => parseGguf(buf)).toThrow(/version/);
  });

  it("tensorByteSize: F32/F16/Q4_0/Q8_0 e rifiuti", () => {
    expect(tensorByteSize({ name: "a", dims: [896], type: GGML_TYPE.F32, offset: 0 })).toBe(3584);
    expect(tensorByteSize({ name: "b", dims: [64, 2], type: GGML_TYPE.F16, offset: 0 })).toBe(256);
    expect(tensorByteSize({ name: "c", dims: [32, 2], type: GGML_TYPE.Q4_0, offset: 0 })).toBe(36);
    expect(tensorByteSize({ name: "d", dims: [32], type: GGML_TYPE.Q8_0, offset: 0 })).toBe(34);
    expect(() => tensorByteSize({ name: "e", dims: [33], type: GGML_TYPE.Q4_0, offset: 0 })).toThrow(/multiplo/);
    expect(() => tensorByteSize({ name: "f", dims: [32], type: 99, offset: 0 })).toThrow(/non supportato/);
  });
});

describe("quant reference", () => {
  it("f16ToF32: valori noti", () => {
    expect(f16ToF32(0x3c00)).toBe(1);
    expect(f16ToF32(0xc000)).toBe(-2);
    expect(f16ToF32(0x3800)).toBe(0.5);
    expect(f16ToF32(0x0000)).toBe(0);
    expect(f16ToF32(0x0001)).toBe(2 ** -24); // denormale minimo
    expect(f16ToF32(0x7c00)).toBe(Infinity);
  });

  it("dequantQ4_0: blocco noto, low/high nibble al posto giusto", () => {
    // scala 1.0 (0x3C00), byte j = j | ((31-j)<<4): peso j = j-8, peso 16+j = (31-j)-8
    const src = new Uint8Array(Q4_0_BLOCK_BYTES);
    src[0] = 0x00; src[1] = 0x3c;
    for (let j = 0; j < 16; j++) src[2 + j] = j | ((15 - j) << 4);
    const dst = new Float32Array(32);
    expect(dequantQ4_0(src, 0, 1, dst)).toBe(32);
    for (let j = 0; j < 16; j++) {
      expect(dst[j]).toBe(j - 8);
      expect(dst[16 + j]).toBe(15 - j - 8);
    }
  });

  it("dequantQ4_0: la scala moltiplica", () => {
    const src = new Uint8Array(Q4_0_BLOCK_BYTES);
    src[0] = 0x00; src[1] = 0x38; // 0.5
    src[2] = 0x0f; // low=15 -> 7, high=0 -> -8
    const dst = new Float32Array(32);
    dequantQ4_0(src, 0, 1, dst);
    expect(dst[0]).toBe(3.5);
    expect(dst[16]).toBe(-4);
  });

  it("dequantQ8_0: int8 con segno", () => {
    const src = new Uint8Array(34);
    src[0] = 0x00; src[1] = 0x3c; // scala 1.0
    src[2] = 0x7f;       // 127
    src[3] = 0x80;       // -128
    src[4] = 0xff;       // -1
    const dst = new Float32Array(32);
    expect(dequantQ8_0(src, 0, 1, dst)).toBe(32);
    expect(dst[0]).toBe(127);
    expect(dst[1]).toBe(-128);
    expect(dst[2]).toBe(-1);
    expect(dst[3]).toBe(0);
  });
});

describe("shape Qwen2.5-0.5B", () => {
  it("fixture completa (291 tensori) -> valida", () => {
    const byName = validateQwen25_05B(parseGguf(qwenFixture()));
    expect(byName.size).toBe(291);
  });

  it("tensore mancante / tipo sbagliato / dims sbagliate -> throw", () => {
    expect(() => validateQwen25_05B(parseGguf(qwenFixture((ts) => ts.pop()))))
      .toThrow(/mancante|tensori/);
    expect(() => validateQwen25_05B(parseGguf(qwenFixture((ts) => { ts[1].type = GGML_TYPE.Q4_0; }))))
      .toThrow(/tipo/);
    expect(() => validateQwen25_05B(parseGguf(qwenFixture((ts) => { ts[0].dims = [896, 1]; }))))
      .toThrow(/dims/);
  });
});

const REAL = `${homedir()}/.cache/blab-models/qwen2.5-0.5b-instruct-q4_0.gguf`;
describe.skipIf(!existsSync(REAL))("GGUF reale (skip in CI)", () => {
  it("parse + validazione shape del file ufficiale", () => {
    const raw = readFileSync(REAL);
    const f = parseGguf(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    const byName = validateQwen25_05B(f);
    expect(byName.size).toBe(291);
    // bounds: ogni tensore sta dentro il file
    for (const t of f.tensors) {
      expect(f.dataOffset + t.offset + tensorByteSize(t)).toBeLessThanOrEqual(raw.byteLength);
    }
  });
});

describe("repack GPU-friendly", () => {
  // Decodifica CPU che simula ESATTAMENTE ciò che farà il WGSL (unpack nibble da u32,
  // unpack2x16float della scala): deve coincidere col dequant reference.
  it("repackQ4_0: la decodifica stile-WGSL coincide col reference", async () => {
    const { repackQ4_0, dequantQ4_0, f16ToF32, Q4_0_BLOCK_BYTES } = await import("../src/engine/quant");
    const nBlocks = 5;
    const src = new Uint8Array(nBlocks * Q4_0_BLOCK_BYTES);
    let seed = 7;
    const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) % 256;
    for (let i = 0; i < src.length; i++) src[i] = rnd();
    // scale f16 plausibili (esponente moderato) per evitare inf
    for (let b = 0; b < nBlocks; b++) { src[b * 18] = rnd(); src[b * 18 + 1] = 0x2c | (b & 3); }
    const ref = new Float32Array(nBlocks * 32);
    dequantQ4_0(src, 0, nBlocks, ref);
    const { qs, scales } = repackQ4_0(src, 0, nBlocks);
    for (let b = 0; b < nBlocks; b++) {
      const sBits = (scales[b >> 1] >>> ((b & 1) * 16)) & 0xffff;
      const scale = f16ToF32(sBits);
      for (let j = 0; j < 32; j++) {
        const byte = (qs[b * 4 + ((j % 16) >> 2)] >>> ((j % 16 & 3) * 8)) & 0xff;
        const nib = j < 16 ? (byte & 0x0f) : (byte >> 4);
        expect((nib - 8) * scale, `blocco ${b} peso ${j}`).toBe(ref[b * 32 + j]);
      }
    }
  });

  it("repackQ8_0: idem", async () => {
    const { repackQ8_0, dequantQ8_0, f16ToF32, Q8_0_BLOCK_BYTES } = await import("../src/engine/quant");
    const nBlocks = 3;
    const src = new Uint8Array(nBlocks * Q8_0_BLOCK_BYTES);
    let seed = 42;
    const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) % 256;
    for (let i = 0; i < src.length; i++) src[i] = rnd();
    for (let b = 0; b < nBlocks; b++) { src[b * 34 + 1] = 0x30; }
    const ref = new Float32Array(nBlocks * 32);
    dequantQ8_0(src, 0, nBlocks, ref);
    const { qs, scales } = repackQ8_0(src, 0, nBlocks);
    for (let b = 0; b < nBlocks; b++) {
      const scale = f16ToF32((scales[b >> 1] >>> ((b & 1) * 16)) & 0xffff);
      for (let j = 0; j < 32; j++) {
        const byte = (qs[b * 8 + (j >> 2)] >>> ((j & 3) * 8)) & 0xff;
        expect(((byte << 24) >> 24) * scale, `blocco ${b} peso ${j}`).toBe(ref[b * 32 + j]);
      }
    }
  });
});
