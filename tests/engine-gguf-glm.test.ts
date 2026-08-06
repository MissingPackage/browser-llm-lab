// Fase C2 fase 3: parser+shape GLM-4.7-Flash (deepseek2) e dequant Q4_1/Q5_K/Q6_K.
//
// Tre livelli di verifica:
//  1. fixture GGUF sintetica (inventario 844 tensori, dati vuoti) → validateGlm47Flash
//     passa, e OGNI mutazione (tipo, dims, tensore mancante, conteggio, metadata) → throw;
//  2. dequant contro fixture da byte REALI del GGUF, valori attesi calcolati da
//     gguf-py (implementazione indipendente): match esatto f32;
//  3. (solo se il GGUF da 17 GB è sul box) parse dell'header reale + validazione
//     completa + spot-check bounds — il "load headless" del done-when di fase.
import { describe, it, expect } from "vitest";
import { existsSync, openSync, readSync, closeSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { readFileSync } from "node:fs";
import { parseGguf, tensorByteSize, GGML_TYPE, type GgufTensorInfo } from "../src/engine/gguf";
import {
  dequantQ4_0, dequantQ8_0, dequantQ4_1, dequantQ5_K, dequantQ6_K,
  Q4_1_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
} from "../src/engine/quant";
import {
  validateGlm47Flash, GLM47_FLASH, GLM47_FLASH_SHA256,
} from "../src/engine/shape";

const G = GLM47_FLASH;
const GGUF_PATH = join(homedir(), ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf");
const GGUF_SIZE = 17216676192; // byte su disco (verifier C1 it.1; il model_size
// 17207200256 di llama-bench è la sola sezione dati, senza header)

// --- builder fixture (stesso formato di engine-gguf.test.ts, qui con f64/bool) ---

import { glmFixture } from "./helpers/glm-gguf-fixture";

describe("shape GLM-4.7-Flash (deepseek2)", () => {
  it("fixture completa: validazione passa, 844 tensori", () => {
    const f = parseGguf(glmFixture());
    expect(f.tensors).toHaveLength(844);
    const byName = validateGlm47Flash(f);
    expect(byName.size).toBe(844);
  });

  it("SHA-256 canonico pinnato", () => {
    expect(GLM47_FLASH_SHA256).toBe("d0bbdfcde6e323ebf90a8b9e95da57100e972be1ec6f0bfa0fad0feaa426557e");
  });

  it("down_exps: Q4_0 al posto di Q4_1 su blk.1 -> throw", () => {
    const f = parseGguf(glmFixture((ts) => {
      const t = ts.find((x) => x.name === "blk.1.ffn_down_exps.weight");
      if (t) t.type = GGML_TYPE.Q4_0;
    }));
    expect(() => validateGlm47Flash(f)).toThrow(/ffn_down_exps/);
  });

  it("down_exps: Q4_1 al posto di Q4_0 su blk.5 -> throw", () => {
    const f = parseGguf(glmFixture((ts) => {
      const t = ts.find((x) => x.name === "blk.5.ffn_down_exps.weight");
      if (t) t.type = GGML_TYPE.Q4_1;
    }));
    expect(() => validateGlm47Flash(f)).toThrow(/ffn_down_exps/);
  });

  it("tensore mancante -> throw", () => {
    const f = parseGguf(glmFixture((ts) => {
      const i = ts.findIndex((x) => x.name === "blk.7.exp_probs_b.bias");
      ts.splice(i, 1);
    }));
    expect(() => validateGlm47Flash(f)).toThrow(/exp_probs_b/);
  });

  it("dims sbagliate (kv_lora) -> throw", () => {
    const f = parseGguf(glmFixture((ts) => {
      const t = ts.find((x) => x.name === "blk.3.attn_kv_a_norm.weight");
      if (t) t.dims = [G.kvLora + 1];
    }));
    expect(() => validateGlm47Flash(f)).toThrow(/attn_kv_a_norm/);
  });

  it("tensore in piu -> throw sul conteggio", () => {
    const f = parseGguf(glmFixture((ts) => {
      ts.push({ name: "blk.0.nextn.embed_tokens.weight", dims: [G.dModel], type: GGML_TYPE.F32, offset: 0 });
    }));
    expect(() => validateGlm47Flash(f)).toThrow(/845 tensori/);
  });

  it("gating func diversa da sigmoid -> throw", () => {
    const f = parseGguf(glmFixture(undefined, (kv) => {
      kv["deepseek2.expert_gating_func"] = { t: "u32", v: 1 };
    }));
    expect(() => validateGlm47Flash(f)).toThrow(/expert_gating_func/);
  });

  it("arch sbagliata (glm4moe) -> throw", () => {
    const f = parseGguf(glmFixture(undefined, (kv) => {
      kv["general.architecture"] = { t: "str", v: "glm4moe" };
    }));
    expect(() => validateGlm47Flash(f)).toThrow(/architecture/);
  });
});

describe("tensorByteSize per i tipi GLM", () => {
  const mk = (type: number, dims: number[]): GgufTensorInfo => ({ name: "t", dims, type, offset: 0 });
  it("Q4_1: 20 B/blocco; Q5_K: 176; Q6_K: 210", () => {
    expect(tensorByteSize(mk(GGML_TYPE.Q4_1, [32]))).toBe(Q4_1_BLOCK_BYTES);
    expect(tensorByteSize(mk(GGML_TYPE.Q5_K, [256]))).toBe(Q5_K_BLOCK_BYTES);
    expect(tensorByteSize(mk(GGML_TYPE.Q6_K, [256]))).toBe(Q6_K_BLOCK_BYTES);
  });
  it("byte per expert: 5.308.416 (down Q4_0) e 5.505.024 (down Q4_1)", () => {
    const gate = tensorByteSize(mk(GGML_TYPE.Q4_0, [G.dModel, G.dFfnExpert]));
    const down40 = tensorByteSize(mk(GGML_TYPE.Q4_0, [G.dFfnExpert, G.dModel]));
    const down41 = tensorByteSize(mk(GGML_TYPE.Q4_1, [G.dFfnExpert, G.dModel]));
    expect(2 * gate + down40).toBe(5308416);
    expect(2 * gate + down41).toBe(5505024);
  });
  it("ne[0] non multiplo del blocco -> throw", () => {
    expect(() => tensorByteSize(mk(GGML_TYPE.Q4_1, [33]))).toThrow(/multiplo di 32/);
    expect(() => tensorByteSize(mk(GGML_TYPE.Q5_K, [128]))).toThrow(/multiplo di 256/);
    expect(() => tensorByteSize(mk(GGML_TYPE.Q6_K, [255, 2]))).toThrow(/multiplo di 256/);
  });
});

describe("dequant vs gguf-py su byte reali del GGUF", () => {
  const fx = JSON.parse(readFileSync(join(__dirname, "fixtures/glm-quant-blocks.json"), "utf8")) as {
    cases: { tensor: string; ggmlType: number; blockWeights: number; nBlocks: number; bytesB64: string; expected: number[] }[];
  };
  const dequantBy: Record<number, (src: Uint8Array, o: number, n: number, dst: Float32Array) => number> = {
    [GGML_TYPE.Q4_0]: dequantQ4_0,
    [GGML_TYPE.Q4_1]: dequantQ4_1,
    [GGML_TYPE.Q8_0]: dequantQ8_0,
    [GGML_TYPE.Q5_K]: dequantQ5_K,
    [GGML_TYPE.Q6_K]: dequantQ6_K,
  };
  for (const c of fx.cases) {
    it(`${c.tensor} (tipo ${c.ggmlType}): match f32 esatto`, () => {
      const src = Uint8Array.from(atob(c.bytesB64), (ch) => ch.charCodeAt(0));
      const dst = new Float32Array(c.nBlocks * c.blockWeights);
      const n = dequantBy[c.ggmlType](src, 0, c.nBlocks, dst);
      expect(n).toBe(dst.length);
      for (let i = 0; i < dst.length; i++) {
        // gguf-py calcola in f32: il confronto è a ULP-zero (Math.fround)
        expect(dst[i], `${c.tensor}[${i}]`).toBe(Math.fround(c.expected[i]));
      }
    });
  }
});

// Load "headless" del file reale: solo header (64 MiB bastano: dataOffset ~5 MiB),
// validazione completa contro la shape, bounds dei tensori dentro il file.
describe.skipIf(!existsSync(GGUF_PATH))("GGUF GLM reale (17 GB, solo su box con il file)", () => {
  it("header parse + validateGlm47Flash + bounds", () => {
    const HEADER_BYTES = 64 * 1024 * 1024;
    const fd = openSync(GGUF_PATH, "r");
    const buf = Buffer.alloc(HEADER_BYTES);
    readSync(fd, buf, 0, HEADER_BYTES, 0);
    closeSync(fd);
    const f = parseGguf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + HEADER_BYTES));
    const byName = validateGlm47Flash(f);
    expect(byName.size).toBe(844);
    const fileSize = statSync(GGUF_PATH).size;
    expect(fileSize).toBe(GGUF_SIZE);
    // ogni tensore sta nei bounds del file
    for (const t of f.tensors) {
      expect(f.dataOffset + t.offset + tensorByteSize(t), t.name).toBeLessThanOrEqual(fileSize);
    }
    // il parco routed misurato da C1: 42×64 expert "leggeri" + 4×64 "pesanti"
    const heavy = f.tensors.filter((t) => t.name.endsWith("ffn_down_exps.weight") && t.type === GGML_TYPE.Q4_1);
    expect(heavy).toHaveLength(4);
  });
});
