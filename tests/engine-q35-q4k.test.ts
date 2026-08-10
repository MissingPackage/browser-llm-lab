import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { dequantQ4_K, Q4_K_BLOCK_BYTES } from "../src/engine/quant";

// dequantQ4_K vs ANCORA ESTERNA (q1 fase 7): gguf-py dequantize sugli stessi
// byte REALI (primi 4 superblocchi di blk.0.ffn_gate_exps del 35B UD-Q4_K_S,
// incapsulati nel fixture: il test NON richiede il modello scaricato).
// Identità attesa al rounding f32: stessa aritmetica d·sc·q − dmin·m.
describe("dequantQ4_K vs gguf-py (byte reali 35B)", () => {
  it("1024 valori identici al riferimento esterno", () => {
    const fx = JSON.parse(readFileSync(join(process.cwd(), "tests/fixtures/q35-q4k-dequant-ref.json"), "utf8")) as {
      rawBase64: string; values: number[]; superblocks: number;
    };
    const raw = Uint8Array.from(Buffer.from(fx.rawBase64, "base64"));
    expect(raw.length).toBe(fx.superblocks * Q4_K_BLOCK_BYTES);
    const out = new Float32Array(fx.superblocks * 256);
    dequantQ4_K(raw, 0, fx.superblocks, out);
    expect(fx.values.length).toBe(out.length);
    for (let i = 0; i < out.length; i++) {
      expect(Math.abs(out[i] - fx.values[i]), `i=${i}`).toBeLessThan(1e-7);
    }
  });
});
