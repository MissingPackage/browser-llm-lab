// ANALISI gate (i) di spec §7 (gira solo con CONF_ANALYSIS=1): full-model
// cpuref f64 in formulazione ABSORBED (la naive è impraticabile a ctx>1k;
// identità naive↔absorbed provata a <1e-8 — suite engine-cpuref-glm — e
// ri-asserita qui dal self-check sulle prime 2 posizioni) sul replay
// teacher-forced dei golden. Output: argmax cpuref per ogni posizione golden
// → confronto col motore (gate (i): agreement ≥99%) assemblato a valle.
// Layer-streaming: un layer f64 alla volta (RAM), head Q6_K solo alle
// posizioni golden.
import { describe, it, expect } from "vitest";
import { existsSync, openSync, readSync, closeSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseGguf, tensorByteSize, GGML_TYPE, type GgufTensorInfo } from "../src/engine/gguf";
import { validateGlm47Flash, GLM47_FLASH as G } from "../src/engine/shape";
import { GgufExpertIndex, downIsQ4_1 } from "../src/engine/expertstore";
import {
  GlmMlaAttnRefF64, GlmMlaAttnAbsorbedRefF64, glmMoeFfnRefF64, rmsnormF64, matvecF64,
  type GlmMoeExpertWeights,
} from "../src/engine/cpuref";
import { dequantQ4_0, dequantQ4_1, dequantQ8_0, dequantQ5_K, dequantQ6_K } from "../src/engine/quant";

const GGUF = join(homedir(), ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf");
const GOLDEN = join(process.cwd(), "results/engine/golden/glm47flash/golden-glm47flash-q4_0-2026-07-31.json");
const PROMPT = Number(process.env.CONF_ANALYSIS_PROMPT ?? 7);
const MAXGEN = Number(process.env.CONF_ANALYSIS_MAXGEN ?? 128);

describe.skipIf(!process.env.CONF_ANALYSIS || !existsSync(GGUF))("logits cpuref-f64 absorbed full-model (gate (i) fase 6)", () => {
  it(`prompt ${PROMPT}, ${MAXGEN} posizioni golden`, () => {
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

    const golden = JSON.parse(readFileSync(GOLDEN, "utf8")) as {
      prompts: Array<{ id: string; promptTokens: number[]; generated: number[]; positions: Array<{ argmax: number }> }>;
    };
    const pr = golden.prompts[PROMPT];
    const nGen = Math.min(pr.generated.length, MAXGEN);
    const seq = [...pr.promptTokens, ...pr.generated.slice(0, Math.max(0, nGen - 1))];
    const n = pr.promptTokens.length;

    const embdInfo = byName.get("token_embd.weight")!;
    const rowBytes = (G.dModel / 32) * 18;
    let hiddens: Float64Array[] = seq.map((tok) =>
      Float64Array.from(deq(readRange(f.dataOffset + embdInfo.offset + tok * rowBytes, rowBytes), GGML_TYPE.Q4_0)));

    const attnW = (l: number) => ({
      attnNorm: tensor(`blk.${l}.attn_norm.weight`), wQA: tensor(`blk.${l}.attn_q_a.weight`),
      qANorm: tensor(`blk.${l}.attn_q_a_norm.weight`), wQB: tensor(`blk.${l}.attn_q_b.weight`),
      wKvA: tensor(`blk.${l}.attn_kv_a_mqa.weight`), kvANorm: tensor(`blk.${l}.attn_kv_a_norm.weight`),
      wKB: tensor(`blk.${l}.attn_k_b.weight`), wVB: tensor(`blk.${l}.attn_v_b.weight`),
      wO: tensor(`blk.${l}.attn_output.weight`),
    });

    const t0 = Date.now();
    for (let l = 0; l < G.nLayer; l++) {
      const aw = attnW(l);
      const attn = new GlmMlaAttnAbsorbedRefF64(aw);
      // self-check identità naive↔absorbed sulle prime 2 posizioni del layer
      const naive = new GlmMlaAttnRefF64(aw);
      const ffnNorm = tensor(`blk.${l}.ffn_norm.weight`);
      let dense: { g: Float32Array; u: Float32Array; d: Float32Array } | null = null;
      let moeW: Parameters<typeof glmMoeFfnRefF64>[1] | null = null;
      if (l < G.denseLead) {
        dense = { g: tensor(`blk.${l}.ffn_gate.weight`), u: tensor(`blk.${l}.ffn_up.weight`), d: tensor(`blk.${l}.ffn_down.weight`) };
      } else {
        const cache = new Map<number, GlmMoeExpertWeights>();
        moeW = {
          routerW: tensor(`blk.${l}.ffn_gate_inp.weight`), routerBias: tensor(`blk.${l}.exp_probs_b.bias`),
          expert: (e: number): GlmMoeExpertWeights => {
            let got = cache.get(e);
            if (!got) {
              const r = idx.ranges(l, e);
              got = {
                gate: deq(readRange(r.gate.offset, r.gate.bytes), GGML_TYPE.Q4_0),
                up: deq(readRange(r.up.offset, r.up.bytes), GGML_TYPE.Q4_0),
                down: deq(readRange(r.down.offset, r.down.bytes), downIsQ4_1(l) ? GGML_TYPE.Q4_1 : GGML_TYPE.Q4_0),
              };
              cache.set(e, got);
            }
            return got;
          },
          gateShexp: tensor(`blk.${l}.ffn_gate_shexp.weight`),
          upShexp: tensor(`blk.${l}.ffn_up_shexp.weight`),
          downShexp: tensor(`blk.${l}.ffn_down_shexp.weight`),
        };
      }
      hiddens = hiddens.map((h, k) => {
        const x = Float64Array.from(h);
        if (k < 2) {
          const xa = Float64Array.from(h);
          naive.attend(xa);
          attn.attend(x);
          let d = 0;
          for (let i = 0; i < G.dModel; i++) d = Math.max(d, Math.abs(x[i] - xa[i]));
          if (d > 1e-8) throw new Error(`self-check naive/absorbed L${l} pos${k}: ${d}`);
        } else {
          attn.attend(x);
        }
        const fn = rmsnormF64(x, ffnNorm, G.rmsEps);
        if (dense) {
          const g = matvecF64(dense.g, 0, fn, G.dFfnDense);
          const u = matvecF64(dense.u, 0, fn, G.dFfnDense);
          for (let i = 0; i < G.dFfnDense; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
          const dn = matvecF64(dense.d, 0, g, G.dModel);
          for (let i = 0; i < G.dModel; i++) x[i] += dn[i];
        } else {
          const r = glmMoeFfnRefF64(fn, moeW!);
          for (let i = 0; i < G.dModel; i++) x[i] += r.out[i];
        }
        return x;
      });
      // eslint-disable-next-line no-console
      console.log(`L${l} done (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
    }

    // head SOLO alle posizioni golden
    const outNorm = tensor("output_norm.weight");
    const outW = tensor("output.weight"); // Q6_K dequant: 317M f32 (~1.27 GB)
    const out: Array<{ k: number; argmax: number; top5: Array<[number, number]>; goldArgmax: number }> = [];
    for (let k = 0; k < nGen; k++) {
      const h = hiddens[n - 1 + k];
      const fn = rmsnormF64(h, outNorm, G.rmsEps);
      let best = 0;
      const logits = new Float64Array(G.vocab);
      for (let r = 0; r < G.vocab; r++) {
        const base = r * G.dModel;
        let acc = 0;
        for (let i = 0; i < G.dModel; i++) acc += outW[base + i] * fn[i];
        logits[r] = acc;
        if (acc > logits[best]) best = r;
      }
      const idxs = Array.from({ length: G.vocab }, (_, i) => i).sort((a, b) => logits[b] - logits[a]).slice(0, 5);
      out.push({ k, argmax: best, top5: idxs.map((i) => [i, logits[i]] as [number, number]), goldArgmax: pr.positions[k].argmax });
      // eslint-disable-next-line no-console
      if (k % 16 === 0) console.log(`head ${k}/${nGen} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
    }
    closeSync(fd);
    const agreeGold = out.filter((o) => o.argmax === o.goldArgmax).length;
    writeFileSync(join(process.cwd(), `results/engine/logits-cpuref-p${PROMPT}-2026-08-01.json`),
      JSON.stringify({ kind: "logits-cpuref-analysis", prompt: PROMPT, nGen, positions: out, agreeGolden: agreeGold }, null, 1));
    // eslint-disable-next-line no-console
    console.log(`CPUREF DONE: ${nGen} pos, argmax==golden su ${agreeGold}/${nGen}`);
    expect(out.length).toBe(nGen);
  }, 6 * 60 * 60_000);
});
