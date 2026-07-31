// ANALISI (non gate, gira solo con ROUTE_ANALYSIS=1): discriminatore per il
// mancato gate routing di it.9 — il replay GPU dà set-match decode 85.85%
// (report routing-smoke-p4) con firma da drift (mismatch a 1 elemento, ~2%
// già al layer 1, degrado liscio con la profondità). Qui il full-model f64
// ESATTO (cpuref, naive, zero GPU) replica le prime N posizioni del prompt 4
// e confronta il SUO routing con la traccia oracolo sugli stessi (pos, layer):
//   cpuref≈GPU (~96% sul subset) ⇒ il motore è fedele all'aritmetica esatta e
//     il disaccordo è dell'oracolo (attivazioni quantizzate q8 sui near-tie);
//   cpuref≈100% ⇒ il bug è nel motore.
// Layer-streaming: un layer f64 alla volta in RAM (16 pos teacher-forced).
import { describe, it, expect } from "vitest";
import { existsSync, openSync, readSync, closeSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseGguf, tensorByteSize, GGML_TYPE, type GgufTensorInfo } from "../src/engine/gguf";
import { validateGlm47Flash, GLM47_FLASH as G } from "../src/engine/shape";
import { GgufExpertIndex, downIsQ4_1 } from "../src/engine/expertstore";
import {
  GlmDenseLayerRefF64, GlmMoeLayerRefF64, type GlmMoeExpertWeights,
} from "../src/engine/cpuref";
import { dequantQ4_0, dequantQ4_1, dequantQ8_0, dequantQ5_K, dequantQ6_K } from "../src/engine/quant";

const GGUF = join(homedir(), ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf");
const TRACE = join(process.cwd(), "public/models/glm-route-trace.json");
const NPOS = Number(process.env.ROUTE_ANALYSIS_NPOS ?? 16);
const PROMPT = Number(process.env.ROUTE_ANALYSIS_PROMPT ?? 4);

describe.skipIf(!process.env.ROUTE_ANALYSIS || !existsSync(GGUF) || !existsSync(TRACE))("routing cpuref-f64 vs traccia (analisi it.9)", () => {
  it(`prompt ${PROMPT}, prime ${NPOS} posizioni, 46 layer MoE`, () => {
    const fd = openSync(GGUF, "r");
    const HEADER = 64 * 1024 * 1024;
    const hbuf = Buffer.alloc(HEADER);
    readSync(fd, hbuf, 0, HEADER, 0);
    const f = parseGguf(hbuf.buffer.slice(hbuf.byteOffset, hbuf.byteOffset + HEADER));
    const byName = validateGlm47Flash(f);
    const idx = new GgufExpertIndex(byName, f.dataOffset);
    const readRange = (off: number, len: number): Uint8Array => {
      const b = Buffer.alloc(len);
      readSync(fd, b, 0, len, off);
      return new Uint8Array(b.buffer, b.byteOffset, len);
    };
    const deq = (raw: Uint8Array, type: number): Float32Array => {
      if (type === GGML_TYPE.F32) return new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
      const [bb, per, fn] =
        type === GGML_TYPE.Q4_0 ? [18, 32, dequantQ4_0] :
        type === GGML_TYPE.Q4_1 ? [20, 32, dequantQ4_1] :
        type === GGML_TYPE.Q8_0 ? [34, 32, dequantQ8_0] :
        type === GGML_TYPE.Q5_K ? [176, 256, dequantQ5_K] : [210, 256, dequantQ6_K];
      const nB = raw.length / (bb as number);
      const out = new Float32Array(nB * (per as number));
      (fn as typeof dequantQ4_0)(raw, 0, nB, out);
      return out;
    };
    const tensor = (name: string): Float32Array => {
      const info = byName.get(name) as GgufTensorInfo;
      return deq(readRange(f.dataOffset + info.offset, tensorByteSize(info)), info.type);
    };

    const trace = JSON.parse(readFileSync(TRACE, "utf8")) as {
      rows: Array<{ p: number; i: number; tok: number; ph: string; e: number[] }>;
    };
    const rows = trace.rows.filter((r) => r.p === PROMPT).sort((a, b) => a.i - b.i).slice(0, NPOS);
    expect(rows.length).toBe(NPOS);

    // hidden iniziali: righe embd dei token (Q4_0, dequant esatta)
    const embdInfo = byName.get("token_embd.weight")!;
    const rowBytes = (G.dModel / 32) * 18;
    let hiddens: Float64Array[] = rows.map((r) => {
      const raw = readRange(f.dataOffset + embdInfo.offset + r.tok * rowBytes, rowBytes);
      return Float64Array.from(deq(raw, GGML_TYPE.Q4_0));
    });

    const attnW = (l: number) => ({
      attnNorm: tensor(`blk.${l}.attn_norm.weight`), wQA: tensor(`blk.${l}.attn_q_a.weight`),
      qANorm: tensor(`blk.${l}.attn_q_a_norm.weight`), wQB: tensor(`blk.${l}.attn_q_b.weight`),
      wKvA: tensor(`blk.${l}.attn_kv_a_mqa.weight`), kvANorm: tensor(`blk.${l}.attn_kv_a_norm.weight`),
      wKB: tensor(`blk.${l}.attn_k_b.weight`), wVB: tensor(`blk.${l}.attn_v_b.weight`),
      wO: tensor(`blk.${l}.attn_output.weight`),
    });

    // layer 0 denso
    {
      const ref = new GlmDenseLayerRefF64({
        ...attnW(0), ffnNorm: tensor("blk.0.ffn_norm.weight"),
        wGate: tensor("blk.0.ffn_gate.weight"), wUp: tensor("blk.0.ffn_up.weight"),
        wDown: tensor("blk.0.ffn_down.weight"),
      });
      hiddens = hiddens.map((h) => ref.forward(h));
    }

    let total = 0, match = 0;
    const perLayer: Array<{ layer: number; total: number; match: number }> = [];
    const samples: Array<{ i: number; layer: number; got: number[]; want: number[] }> = [];
    for (let l = 1; l < G.nLayer; l++) {
      const expCache = new Map<number, GlmMoeExpertWeights>();
      const ref = new GlmMoeLayerRefF64(attnW(l), tensor(`blk.${l}.ffn_norm.weight`), {
        routerW: tensor(`blk.${l}.ffn_gate_inp.weight`),
        routerBias: tensor(`blk.${l}.exp_probs_b.bias`),
        expert: (e: number): GlmMoeExpertWeights => {
          let got = expCache.get(e);
          if (!got) {
            const r = idx.ranges(l, e);
            got = {
              gate: deq(readRange(r.gate.offset, r.gate.bytes), GGML_TYPE.Q4_0),
              up: deq(readRange(r.up.offset, r.up.bytes), GGML_TYPE.Q4_0),
              down: deq(readRange(r.down.offset, r.down.bytes), downIsQ4_1(l) ? GGML_TYPE.Q4_1 : GGML_TYPE.Q4_0),
            };
            expCache.set(e, got);
          }
          return got;
        },
        gateShexp: tensor(`blk.${l}.ffn_gate_shexp.weight`),
        upShexp: tensor(`blk.${l}.ffn_up_shexp.weight`),
        downShexp: tensor(`blk.${l}.ffn_down_shexp.weight`),
      });
      const c = { layer: l, total: 0, match: 0 };
      hiddens = hiddens.map((h, k) => {
        const out = ref.forward(h);
        const got = Array.from(ref.lastRouting!.experts);
        const want = rows[k].e.slice((l - 1) * 4, (l - 1) * 4 + 4);
        const ok = want.every((e) => got.includes(e));
        c.total++; total++;
        if (ok) { c.match++; match++; }
        else if (samples.length < 40) samples.push({ i: rows[k].i, layer: l, got, want });
        return out;
      });
      perLayer.push(c);
      // eslint-disable-next-line no-console
      console.log(`L${l}: ${c.match}/${c.total}  (cum ${(100 * match / total).toFixed(2)}%)`);
    }
    closeSync(fd);

    const report = {
      kind: "routing-cpuref-analysis", prompt: PROMPT, nPos: NPOS,
      total, match, pct: 100 * match / total, perLayer, samples,
    };
    writeFileSync(join(process.cwd(), "results/engine/routing-cpuref-analysis-2026-07-31.json"), JSON.stringify(report, null, 1));
    // eslint-disable-next-line no-console
    console.log(`TOTALE cpuref-f64 vs traccia: ${match}/${total} = ${(100 * match / total).toFixed(2)}%`);
    expect(total).toBe(NPOS * (G.nLayer - 1));
  }, 45 * 60_000);
});
