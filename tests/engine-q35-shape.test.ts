import { describe, it, expect } from "vitest";
import { closeSync, existsSync, openSync, readSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseGguf, tensorByteSize } from "../src/engine/gguf";
import { q35IsFullAttn, validateQwen35 } from "../src/engine/q35shape";

// Load test di fase 2 (q1): i 3 GGUF pinnati si aprono, la shape derivata dai
// metadata combacia con l'inventario tensori COMPLETO (dims calcolate, tipi
// nell'allow-list, zero tensori inattesi) e i valori chiave coincidono con
// l'header dump committato (results/engine/q35-header-dump-2026-08-10.json).
// Gira solo col modello scaricato (CI salta): pattern engine-cpuref-glm.
const DIR = join(homedir(), ".cache/blab-models/q35");
const FILES = {
  "Qwen3.5-4B": "Qwen3.5-4B-Q4_0.gguf",
  "Qwen3.5-9B": "Qwen3.5-9B-Q4_0.gguf",
  "Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
} as const;
const HEADER_BYTES = 64 * 1024 * 1024; // pattern glmsource: header+kv+tensor info nei primi 64 MB

function parseHeader(path: string) {
  const fd = openSync(path, "r");
  const buf = Buffer.alloc(HEADER_BYTES);
  const n = readSync(fd, buf, 0, HEADER_BYTES, 0);
  closeSync(fd);
  return parseGguf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + n) as ArrayBuffer);
}

const have = existsSync(join(DIR, FILES["Qwen3.5-4B"]));

describe.skipIf(!have)("q35 reader: shape derivata + inventario completo", () => {
  it("4B: denso tied, 32 layer (8 full), 426 tensori", () => {
    const f = parseHeader(join(DIR, FILES["Qwen3.5-4B"]));
    const { shape } = validateQwen35(f);
    expect(shape.arch).toBe("qwen35");
    expect(shape.nLayer).toBe(32);
    expect(shape.nFull).toBe(8);
    expect(shape.dModel).toBe(2560);
    expect(shape.nHead).toBe(16);
    expect(shape.nKvHead).toBe(4);
    expect(shape.headDim).toBe(256);
    expect(shape.dFfn).toBe(9216);
    expect(shape.vocab).toBe(248320);
    expect(shape.tiedEmbeddings).toBe(true);
    expect(shape.linKHead).toBe(16);
    expect(shape.linVHead).toBe(32);
    expect(shape.linHeadDim).toBe(128);
    expect(shape.linConvK).toBe(4);
    expect(f.tensors.length).toBe(426);
    // pattern ibrido: full a 3,7,…,31 (0-based)
    expect(q35IsFullAttn(shape, 3)).toBe(true);
    expect(q35IsFullAttn(shape, 4)).toBe(false);
    expect(q35IsFullAttn(shape, 31)).toBe(true);
  });

  it("9B: denso NON tied, 427 tensori", () => {
    const f = parseHeader(join(DIR, FILES["Qwen3.5-9B"]));
    const { shape } = validateQwen35(f);
    expect(shape.arch).toBe("qwen35");
    expect(shape.nLayer).toBe(32);
    expect(shape.dModel).toBe(4096);
    expect(shape.dFfn).toBe(12288);
    expect(shape.tiedEmbeddings).toBe(false);
    expect(f.tensors.length).toBe(427);
  });

  it("35B: MoE 256 top-8, 40 layer (10 full), 733 tensori, KV 40 960 B/token", () => {
    const f = parseHeader(join(DIR, FILES["Qwen3.6-35B-A3B"]));
    const { shape, byName } = validateQwen35(f);
    expect(shape.arch).toBe("qwen35moe");
    expect(shape.nLayer).toBe(40);
    expect(shape.nFull).toBe(10);
    expect(shape.dModel).toBe(2048);
    expect(shape.nKvHead).toBe(2);
    expect(shape.nExpert).toBe(256);
    expect(shape.nExpertUsed).toBe(8);
    expect(shape.dFfnExpert).toBe(512);
    expect(shape.tiedEmbeddings).toBe(false);
    expect(f.tensors.length).toBe(733);
    // KV solo sui layer full: nFull × nKvHead × headDim × 2 (K+V) × 4 B (f32)
    expect(shape.nFull * shape.nKvHead * shape.headDim * 2 * 4).toBe(40960);
    // il parco expert: 3 tensori stacked per layer, byte del dump (Q4_K+Q6_K)
    let expertBytes = 0;
    for (const [name, t] of byName) {
      if (/_exps\./.test(name)) expertBytes += tensorByteSize(t);
    }
    expect(expertBytes).toBe(17666408448 + 660602880);
  });

  it("shape rifiuta architetture estranee", () => {
    const f = parseHeader(join(DIR, FILES["Qwen3.5-4B"]));
    const rotto = { ...f, metadata: { ...f.metadata, "general.architecture": "qwen2" } };
    expect(() => validateQwen35(rotto)).toThrow(/non è qwen35/);
  });
});
