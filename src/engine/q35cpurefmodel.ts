// cpuref del MODELLO denso Qwen 3.5 (4B/9B) — fase 4 q1, spec §5 punto 2.
// SOLO test/conformance, mai in produzione: è l'anello di differential
// testing fra oracolo llama.cpp e kernel WGSL (pattern cpuref.ts / GLM).
//
// Attivazioni in f64; pesi f32 dequantizzati per-LAYER in streaming (il 4B
// intero in f32 sarebbe ~17 GB: qui vive un layer alla volta, ~300 MB max) —
// tranne la HEAD (embedding tied Q6_K, ~2.5 GB f32) dequantizzata una volta.
//
// Semantica full-attention dalla fonte llama.cpp b10333 (qwen35.cpp
// build_layer_attn + ggml-cpu ops.cpp, letti 2026-08-10):
// - attn_q produce [2·nHead·256]: per head, [q(256) | gate(256)] (viste a
//   stride 512; il gate NON riceve né norm né rope: sigmoid a valle);
// - QK-norm RMS per head (256) con attn_{q,k}_norm PRIMA del rope;
// - mrope text-only: sezioni [11,11,10,0] con posizioni tutte uguali ⇒
//   COLLASSA a NEOX su n_dims=64: coppie (j, j+32) per j<32,
//   theta_j = pos / freq_base^(j/32), canali 64..255 passthrough
//   (ggml_mrope_cache_init: le theta t/h/w/e partono uguali e scalano in
//   lockstep ⇒ il sector non cambia theta; rotate_pairs con offset 32);
// - GQA standard: q-head h usa kv-head h / (nHead/nKvHead), scala 1/√256;
// - out = concat(attn·V) · sigmoid(gate) → wo;
// - ffn denso SwiGLU: down(silu(gate(x))·up(x)); residui pre/post norm come
//   nel graph (attn_norm → attn → +res; post_attention_norm → ffn → +res).
// I layer linear riusano Q35DeltaNetRef (stessa semantica ktestata 75/75).
import { GGML_TYPE, parseGguf, tensorByteSize, type GgufFile, type GgufTensorInfo } from "./gguf";
import { dequantQ4_0, dequantQ4_1, dequantQ5_K, dequantQ6_K, dequantQ8_0 } from "./quant";
import { Q35DeltaNetRef } from "./q35cpuref";
import { q35IsFullAttn, validateQwen35, type Q35Shape } from "./q35shape";

const silu = (x: number): number => x / (1 + Math.exp(-x));
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

function rmsnormF64(x: Float64Array, w: Float32Array, eps: number): Float64Array {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const scale = 1 / Math.sqrt(ss / x.length + eps);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale * w[i];
  return out;
}

function matVecF64(w: Float32Array, x: Float64Array, rows: number): Float64Array {
  const k = x.length;
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * k;
    for (let i = 0; i < k; i++) acc += w[base + i] * x[i];
    out[r] = acc;
  }
  return out;
}

/** rope text-only (mrope collassato): in place su vec[?·stride], primi 64 canali. */
function ropeText(vec: Float64Array, nVec: number, stride: number, nDims: number, pos: number, freqBase: number): void {
  const half = nDims / 2;
  for (let h = 0; h < nVec; h++) {
    const base = h * stride;
    for (let j = 0; j < half; j++) {
      const theta = pos / freqBase ** (j / half);
      const c = Math.cos(theta), s = Math.sin(theta);
      const a = vec[base + j], b = vec[base + j + half];
      vec[base + j] = a * c - b * s;
      vec[base + j + half] = a * s + b * c;
    }
  }
}

export class Q35CpuRefModel {
  readonly shape: Q35Shape;
  private f: GgufFile;
  private byName: Map<string, GgufTensorInfo>;
  private bytes: Uint8Array;
  private buf: ArrayBuffer;
  private headF32: Float32Array | null = null;

  constructor(buf: ArrayBuffer) {
    this.buf = buf;
    this.bytes = new Uint8Array(buf);
    this.f = parseGguf(buf);
    const v = validateQwen35(this.f);
    this.shape = v.shape;
    this.byName = v.byName;
    if (this.shape.arch !== "qwen35") throw new Error("q35cpurefmodel: solo densi (qwen35)");
  }

  /** Dequantizza un tensore intero in f32 (streaming per-layer: il chiamante non lo trattiene). */
  dequant(name: string): Float32Array {
    const t = this.byName.get(name);
    if (!t) throw new Error(`q35cpurefmodel: tensore ${name} assente`);
    const elems = t.dims.reduce((a, b) => a * b, 1);
    const dst = new Float32Array(elems);
    const off = this.f.dataOffset + t.offset;
    switch (t.type) {
      case GGML_TYPE.F32: dst.set(new Float32Array(this.buf, off, elems)); break;
      case GGML_TYPE.Q4_0: dequantQ4_0(this.bytes, off, elems / 32, dst); break;
      case GGML_TYPE.Q4_1: dequantQ4_1(this.bytes, off, elems / 32, dst); break;
      case GGML_TYPE.Q8_0: dequantQ8_0(this.bytes, off, elems / 32, dst); break;
      case GGML_TYPE.Q5_K: dequantQ5_K(this.bytes, off, elems / 256, dst); break;
      case GGML_TYPE.Q6_K: dequantQ6_K(this.bytes, off, elems / 256, dst); break;
      default: throw new Error(`q35cpurefmodel: tipo ${t.type} non gestito (${name}, ${tensorByteSize(t)} B)`);
    }
    return dst;
  }

  /** Riga dell'embedding (tied): dequant della sola riga `token`. */
  embedRow(token: number): Float64Array {
    const t = this.byName.get("token_embd.weight");
    if (!t) throw new Error("q35cpurefmodel: token_embd assente");
    const d = this.shape.dModel;
    const out = new Float32Array(d);
    const off = this.f.dataOffset + t.offset;
    if (t.type === GGML_TYPE.Q6_K) {
      const rowBlocks = d / 256;
      dequantQ6_K(this.bytes, off + token * rowBlocks * 210, rowBlocks, out);
    } else if (t.type === GGML_TYPE.Q4_0) {
      dequantQ4_0(this.bytes, off + token * (d / 32) * 18, d / 32, out);
    } else throw new Error(`q35cpurefmodel: embd tipo ${t.type} non gestito`);
    return Float64Array.from(out);
  }

  private head(): Float32Array {
    if (!this.headF32) this.headF32 = this.dequant("token_embd.weight"); // tied (4B)
    return this.headF32;
  }

  /**
   * Forward TEACHER-FORCED su tutti i token; ritorna per ogni posizione
   * l'argmax dei logits (predizione del token successivo). Streaming
   * per-layer: attivazioni [T][d] in RAM, pesi un layer alla volta.
   */
  forward(tokens: number[], onLayer?: (l: number) => void): { argmax: Int32Array; lastLogits: Float32Array } {
    const S = this.shape;
    const T = tokens.length;
    const d = S.dModel;
    let hidden: Float64Array[] = tokens.map((tk) => this.embedRow(tk));

    for (let l = 0; l < S.nLayer; l++) {
      onLayer?.(l);
      const b = `blk.${l}.`;
      const attnNorm = this.dequant(`${b}attn_norm.weight`);
      const postNorm = this.dequant(`${b}post_attention_norm.weight`);
      const attnOut: Float64Array[] = [];

      if (q35IsFullAttn(S, l)) {
        const wq = this.dequant(`${b}attn_q.weight`);
        const wk = this.dequant(`${b}attn_k.weight`);
        const wv = this.dequant(`${b}attn_v.weight`);
        const wo = this.dequant(`${b}attn_output.weight`);
        const qNormW = this.dequant(`${b}attn_q_norm.weight`);
        const kNormW = this.dequant(`${b}attn_k_norm.weight`);
        const hd = S.headDim;
        const group = S.nHead / S.nKvHead;
        const scale = 1 / Math.sqrt(hd);
        // K/V di tutte le posizioni (teacher-forced, causale)
        const ks: Float64Array[] = [], vs: Float64Array[] = [];
        const qs: Float64Array[] = [], gates: Float64Array[] = [];
        for (let t = 0; t < T; t++) {
          const xn = rmsnormF64(hidden[t], attnNorm, S.rmsEps);
          const qFull = matVecF64(wq, xn, 2 * S.nHead * hd);
          const k = matVecF64(wk, xn, S.nKvHead * hd);
          const v = matVecF64(wv, xn, S.nKvHead * hd);
          // split q|gate per head (stride 2·hd), QK-norm per head, poi rope
          const q = new Float64Array(S.nHead * hd);
          const gate = new Float64Array(S.nHead * hd);
          for (let h = 0; h < S.nHead; h++) {
            for (let i = 0; i < hd; i++) {
              q[h * hd + i] = qFull[h * 2 * hd + i];
              gate[h * hd + i] = qFull[h * 2 * hd + hd + i];
            }
            const qh = q.subarray(h * hd, (h + 1) * hd);
            qh.set(rmsnormF64(qh as Float64Array, qNormW, S.rmsEps));
          }
          for (let h = 0; h < S.nKvHead; h++) {
            const kh = k.subarray(h * hd, (h + 1) * hd);
            kh.set(rmsnormF64(kh as Float64Array, kNormW, S.rmsEps));
          }
          ropeText(q, S.nHead, hd, S.ropeDims, t, S.ropeFreqBase);
          ropeText(k, S.nKvHead, hd, S.ropeDims, t, S.ropeFreqBase);
          qs.push(q); gates.push(gate); ks.push(k); vs.push(v);
        }
        for (let t = 0; t < T; t++) {
          const o = new Float64Array(S.nHead * hd);
          for (let h = 0; h < S.nHead; h++) {
            const kvH = Math.floor(h / group) * hd;
            // softmax causale online (f64)
            let m = -Infinity;
            const sc = new Float64Array(t + 1);
            for (let p = 0; p <= t; p++) {
              let dot = 0;
              for (let i = 0; i < hd; i++) dot += qs[t][h * hd + i] * ks[p][kvH + i];
              sc[p] = dot * scale;
              if (sc[p] > m) m = sc[p];
            }
            let sum = 0;
            for (let p = 0; p <= t; p++) { sc[p] = Math.exp(sc[p] - m); sum += sc[p]; }
            for (let p = 0; p <= t; p++) {
              const w = sc[p] / sum;
              for (let i = 0; i < hd; i++) o[h * hd + i] += w * vs[p][kvH + i];
            }
            for (let i = 0; i < hd; i++) o[h * hd + i] *= sigmoid(gates[t][h * hd + i]);
          }
          attnOut.push(matVecF64(wo, o, d));
        }
      } else {
        const ref = new Q35DeltaNetRef(
          { d, nK: S.linKHead, nV: S.linVHead, hd: S.linHeadDim, convK: S.linConvK, eps: S.rmsEps },
          {
            wqkv: this.dequant(`${b}attn_qkv.weight`),
            wgate: this.dequant(`${b}attn_gate.weight`),
            conv: this.dequant(`${b}ssm_conv1d.weight`),
            wbeta: this.dequant(`${b}ssm_beta.weight`),
            walpha: this.dequant(`${b}ssm_alpha.weight`),
            dtBias: this.dequant(`${b}ssm_dt.bias`),
            a: this.dequant(`${b}ssm_a`),
            ssmNorm: this.dequant(`${b}ssm_norm.weight`),
            wout: this.dequant(`${b}ssm_out.weight`),
          },
        );
        for (let t = 0; t < T; t++) attnOut.push(ref.step(rmsnormF64(hidden[t], attnNorm, S.rmsEps)));
      }

      const wg = this.dequant(`${b}ffn_gate.weight`);
      const wu = this.dequant(`${b}ffn_up.weight`);
      const wd = this.dequant(`${b}ffn_down.weight`);
      const dFfn = S.dFfn as number;
      for (let t = 0; t < T; t++) {
        const afterAttn = new Float64Array(d);
        for (let i = 0; i < d; i++) afterAttn[i] = hidden[t][i] + attnOut[t][i];
        const xn = rmsnormF64(afterAttn, postNorm, S.rmsEps);
        const gt = matVecF64(wg, xn, dFfn);
        const up = matVecF64(wu, xn, dFfn);
        for (let i = 0; i < dFfn; i++) gt[i] = silu(gt[i]) * up[i];
        const dn = matVecF64(wd, gt, d);
        for (let i = 0; i < d; i++) afterAttn[i] += dn[i];
        hidden[t] = afterAttn;
      }
    }

    const outNorm = this.dequant("output_norm.weight");
    const head = this.head();
    const argmax = new Int32Array(T);
    let lastLogits = new Float32Array(0);
    for (let t = 0; t < T; t++) {
      const xn = rmsnormF64(hidden[t], outNorm, S.rmsEps);
      let best = -Infinity, bi = -1;
      const logits = new Float32Array(S.vocab);
      for (let r = 0; r < S.vocab; r++) {
        let acc = 0;
        const base = r * d;
        for (let i = 0; i < d; i++) acc += head[base + i] * xn[i];
        logits[r] = acc;
        if (acc > best) { best = acc; bi = r; }
      }
      argmax[t] = bi;
      if (t === T - 1) lastLogits = logits;
    }
    return { argmax, lastLogits };
  }
}
