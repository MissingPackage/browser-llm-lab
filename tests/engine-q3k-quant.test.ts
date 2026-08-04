// Q3_K/Q2_K: quantizzatore e dequant contro i vettori dell'ORACOLO (fase 4c slice A).
//
// La fixture glm-q3k-q2k-blocks.json è pesi VERI (blk.5.ffn_gate_exps expert 0,
// dequantizzati da Q4_0) più i byte quantizzati e i float dequantizzati prodotti
// da tools/oracle-moe/kqref, che linka la libggml del checkout oracolo
// (llama.cpp 5f55650) e chiama quantize_q3_K / quantize_q2_K /
// dequantize_row_q3_K / dequantize_row_q2_K. Rigenerabile con
// `Q3K_ROUNDTRIP=1 npx vitest run tests/analysis-q3k-roundtrip.test.ts`, che sul
// tensore INTERO verifica anche l'identità byte con `llama-quantize
// --allow-requantize --pure` (gate R1 del design §10).
//
// Qui il gate è bit-exact: il quantizzatore è deterministico e la fixture è il
// riferimento pinnato. Qualunque divergenza è un bug, non rumore numerico.
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  quantizeQ3_K, dequantQ3_K, quantizeQ2_K, dequantQ2_K, f32ToF16, f16ToF32,
  Q3_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES, Q3_K_BLOCK_WEIGHTS, Q2_K_BLOCK_WEIGHTS, QK_K,
} from "../src/engine/quant";

interface FixtureCase {
  name: string; tensor: string; expert: number; blocks: number;
  srcF32: string; q3k: string; q3kDequant: string; q2k: string; q2kDequant: string;
}
const fx = JSON.parse(
  readFileSync(join(process.cwd(), "tests/fixtures/glm-q3k-q2k-blocks.json"), "utf8"),
) as { cases: FixtureCase[] };

const bytes = (b64: string): Uint8Array => new Uint8Array(Buffer.from(b64, "base64"));
const floats = (b64: string): Float32Array => {
  const b = Buffer.from(b64, "base64");
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};

describe("Q3_K / Q2_K (fase 4c)", () => {
  it("costanti di formato coerenti con ggml-common.h", () => {
    expect(Q3_K_BLOCK_WEIGHTS).toBe(QK_K);
    expect(Q2_K_BLOCK_WEIGHTS).toBe(QK_K);
    expect(Q3_K_BLOCK_BYTES).toBe(32 + 64 + 12 + 2); // hmask | qs | scales | d
    expect(Q2_K_BLOCK_BYTES).toBe(16 + 64 + 2 + 2);  // scales | qs | d | dmin
    expect((Q3_K_BLOCK_BYTES * 8) / QK_K).toBe(3.4375); // bpw dichiarati dal design §2.3
    expect((Q2_K_BLOCK_BYTES * 8) / QK_K).toBe(2.625);
  });

  it("f32ToF16 è round-to-nearest-even e chiude il giro con f16ToF32", () => {
    // valori esatti in f16
    for (const v of [0, 1, -1, 0.5, 2048, -65504, 6.103515625e-5]) {
      expect(f16ToF32(f32ToF16(v))).toBe(v);
    }
    expect(f32ToF16(0)).toBe(0x0000);
    expect(f32ToF16(-0)).toBe(0x8000);
    expect(f32ToF16(1)).toBe(0x3c00);
    expect(f32ToF16(-2)).toBe(0xc000);
    expect(f32ToF16(65504)).toBe(0x7bff);     // max normale
    expect(f32ToF16(65536)).toBe(0x7c00);     // overflow → inf
    expect(f32ToF16(6e-8)).toBe(0x0001);      // subnormale minimo
    expect(f32ToF16(1e-10)).toBe(0x0000);     // underflow → 0
    expect(f32ToF16(Infinity)).toBe(0x7c00);
    // ties-to-even: 1 + 2^-11 sta a metà fra 0x3c00 e 0x3c01 → vince il pari
    expect(f32ToF16(1 + 2 ** -11)).toBe(0x3c00);
    expect(f32ToF16(1 + 3 * 2 ** -11)).toBe(0x3c02);
  });

  it("la fixture copre più di un tensore (una sola fonte era cieca)", () => {
    expect(fx.cases.length).toBeGreaterThanOrEqual(2);
    expect(new Set(fx.cases.map((c) => c.tensor)).size).toBeGreaterThanOrEqual(2);
    expect(Math.max(...fx.cases.map((c) => c.blocks))).toBeGreaterThanOrEqual(64);
  });

  for (const c of fx.cases) {
    it(`[${c.name}] quantizeQ3_K riproduce byte per byte quantize_q3_K dell'oracolo`, () => {
      const x = floats(c.srcF32);
      const ref = bytes(c.q3k);
      expect(x.length).toBe(c.blocks * QK_K);
      expect(ref.length).toBe(c.blocks * Q3_K_BLOCK_BYTES);
      const out = new Uint8Array(ref.length);
      expect(quantizeQ3_K(x, 0, c.blocks, out, 0)).toBe(ref.length);
      expect(Array.from(out)).toEqual(Array.from(ref));
    });

    it(`[${c.name}] dequantQ3_K riproduce dequantize_row_q3_K bit per bit`, () => {
      const ref = floats(c.q3kDequant);
      const out = new Float32Array(ref.length);
      expect(dequantQ3_K(bytes(c.q3k), 0, c.blocks, out, 0)).toBe(ref.length);
      expect(Array.from(out)).toEqual(Array.from(ref));
    });

    it(`[${c.name}] quantizeQ2_K riproduce byte per byte quantize_q2_K dell'oracolo`, () => {
      const x = floats(c.srcF32);
      const ref = bytes(c.q2k);
      expect(ref.length).toBe(c.blocks * Q2_K_BLOCK_BYTES);
      const out = new Uint8Array(ref.length);
      expect(quantizeQ2_K(x, 0, c.blocks, out, 0)).toBe(ref.length);
      expect(Array.from(out)).toEqual(Array.from(ref));
    });

    it(`[${c.name}] dequantQ2_K riproduce dequantize_row_q2_K bit per bit`, () => {
      const ref = floats(c.q2kDequant);
      const out = new Float32Array(ref.length);
      expect(dequantQ2_K(bytes(c.q2k), 0, c.blocks, out, 0)).toBe(ref.length);
      expect(Array.from(out)).toEqual(Array.from(ref));
    });
  }

  it("offset non-zero: quantize e dequant rispettano srcOffset/dstOffset", () => {
    const c = fx.cases[0];
    const x = floats(c.srcF32);
    const full = new Uint8Array(c.blocks * Q3_K_BLOCK_BYTES);
    quantizeQ3_K(x, 0, c.blocks, full, 0);
    // ri-quantizza solo i blocchi 2..3 partendo da metà buffer
    const part = new Uint8Array(4 * Q3_K_BLOCK_BYTES);
    quantizeQ3_K(x, 2 * QK_K, 2, part, 2 * Q3_K_BLOCK_BYTES);
    expect(Array.from(part.subarray(2 * Q3_K_BLOCK_BYTES, 4 * Q3_K_BLOCK_BYTES)))
      .toEqual(Array.from(full.subarray(2 * Q3_K_BLOCK_BYTES, 4 * Q3_K_BLOCK_BYTES)));
    const dq = new Float32Array(4 * QK_K);
    dequantQ3_K(full, 2 * Q3_K_BLOCK_BYTES, 2, dq, 2 * QK_K);
    const refAll = floats(c.q3kDequant);
    expect(Array.from(dq.subarray(2 * QK_K, 4 * QK_K)))
      .toEqual(Array.from(refAll.subarray(2 * QK_K, 4 * QK_K)));
  });

  it("blocco di soli zeri: scala nulla, nessun NaN", () => {
    const zeros = new Float32Array(QK_K);
    const q3 = new Uint8Array(Q3_K_BLOCK_BYTES);
    quantizeQ3_K(zeros, 0, 1, q3, 0);
    const d3 = new Float32Array(QK_K);
    dequantQ3_K(q3, 0, 1, d3);
    expect(d3.every((v) => v === 0)).toBe(true);
    const q2 = new Uint8Array(Q2_K_BLOCK_BYTES);
    quantizeQ2_K(zeros, 0, 1, q2, 0);
    const d2 = new Float32Array(QK_K);
    dequantQ2_K(q2, 0, 1, d2);
    expect(d2.every((v) => v === 0)).toBe(true);
  });
});
