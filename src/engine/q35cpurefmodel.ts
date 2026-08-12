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
import { dequantQ4_0, dequantQ4_1, dequantQ4_K, dequantQ5_K, dequantQ6_K, dequantQ8_0 } from "./quant";
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

/**
 * FFN MoE qwen35moe (f64) — semantica dalla FONTE llama.cpp b10333
 * (qwen35moe.cpp build_layer_ffn + llama-graph.cpp build_moe_ffn, letti
 * 2026-08-10), NON da GLM (che usa sigmoid+bias+scale 1.8):
 *   logits = Wrouter·x (F32, [256]); probs = softmax(logits); top-8 per
 *   probs (nessun bias); weights = probs[sel] / clamp(Σ, 6.103515625e-5)
 *   (norm_w=true; il clamp è ESATTO da build_moe_ffn); w_scale non letto
 *   dall'arch ⇒ 0 ⇒ NESSUNO scale; out = Σ w_e · SwiGLU_e(x) (pesatura
 *   DOPO il down); shared: SwiGLU_sh(x) · sigmoid(w_gate_sh·x) (gate
 *   SCALARE) sommato a out.
 * `experts.get(e)` fornisce i pesi dell'expert e on-demand (il chiamante
 * decide dequant/caching: 256 expert f32 interi = 3.2 GB/layer).
 */
export interface Q35MoeLayerWeights {
  router: Float32Array; // [nExpert, d] F32
  sharedGate: Float32Array; // [d] F32 (ffn_gate_inp_shexp)
  shGate: Float32Array; // [dE, d]... convenzione ggml: [d in, dE out] → righe dE
  shUp: Float32Array;
  shDown: Float32Array; // [dE in, d out] → righe d
  expert: (e: number) => { gate: Float32Array; up: Float32Array; down: Float32Array };
}

export function q35MoeFfnRefF64(
  x: Float64Array, w: Q35MoeLayerWeights, nExpert: number, topK: number, dFfnExpert: number,
): { out: Float64Array; selected: number[]; weights: number[] } {
  const d = x.length;
  const logits = matVecF64(w.router, x, nExpert);
  // softmax f64 su tutti gli expert
  let m = -Infinity;
  for (let e = 0; e < nExpert; e++) if (logits[e] > m) m = logits[e];
  const probs = new Float64Array(nExpert);
  let sum = 0;
  for (let e = 0; e < nExpert; e++) { probs[e] = Math.exp(logits[e] - m); sum += probs[e]; }
  for (let e = 0; e < nExpert; e++) probs[e] /= sum;
  // top-K per probs (argsort stabile come ggml_argsort_top_k)
  const idx = Array.from({ length: nExpert }, (_, e) => e).sort((a, b) => probs[b] - probs[a] || a - b);
  const selected = idx.slice(0, topK);
  let wSum = 0;
  for (const e of selected) wSum += probs[e];
  const wClamp = Math.max(wSum, 6.103515625e-5); // clamp ESATTO di build_moe_ffn
  const weights = selected.map((e) => probs[e] / wClamp);
  const out = new Float64Array(d);
  for (let k = 0; k < topK; k++) {
    const ex = w.expert(selected[k]);
    const g = matVecF64(ex.gate, x, dFfnExpert);
    const u = matVecF64(ex.up, x, dFfnExpert);
    for (let i = 0; i < dFfnExpert; i++) g[i] = silu(g[i]) * u[i];
    const dn = matVecF64(ex.down, g, d);
    const wk = weights[k];
    for (let i = 0; i < d; i++) out[i] += wk * dn[i];
  }
  // shared expert: SwiGLU · sigmoid(gate scalare)
  const sg = matVecF64(w.shGate, x, dFfnExpert);
  const su = matVecF64(w.shUp, x, dFfnExpert);
  for (let i = 0; i < dFfnExpert; i++) sg[i] = silu(sg[i]) * su[i];
  const sd = matVecF64(w.shDown, sg, d);
  let gateRaw = 0;
  for (let i = 0; i < d; i++) gateRaw += w.sharedGate[i] * x[i];
  const gateS = sigmoid(gateRaw);
  for (let i = 0; i < d; i++) out[i] += gateS * sd[i];
  return { out, selected, weights };
}

/**
 * Sorgente byte LAZY (it.15): il 35B (20.9 GB) non sta in RAM col resto —
 * slice(off,len) restituisce una COPIA con byteOffset 0 (allineamento F32).
 * Implementazioni: wrap di ArrayBuffer (4B/9B) o pread da fd (tests/helpers).
 */
export interface Q35ByteSource {
  size: number;
  slice(off: number, len: number): Uint8Array;
}

export class Q35CpuRefModel {
  readonly shape: Q35Shape;
  private f: GgufFile;
  private byName: Map<string, GgufTensorInfo>;
  private src: Q35ByteSource;
  private headF32: Float32Array | null = null;

  constructor(input: ArrayBuffer | Q35ByteSource) {
    if (input instanceof ArrayBuffer) {
      const buf = input;
      this.src = { size: buf.byteLength, slice: (off, len) => new Uint8Array(buf.slice(off, off + len)) };
    } else {
      this.src = input;
    }
    const headerLen = Math.min(this.src.size, 64 * 1024 * 1024);
    const header = this.src.slice(0, headerLen);
    this.f = parseGguf(header.buffer as ArrayBuffer);
    const v = validateQwen35(this.f);
    this.shape = v.shape;
    this.byName = v.byName;
  }

  /** Dequantizza un tensore intero in f32 (streaming per-layer: il chiamante non lo trattiene). */
  dequant(name: string): Float32Array {
    const t = this.byName.get(name);
    if (!t) throw new Error(`q35cpurefmodel: tensore ${name} assente`);
    const elems = t.dims.reduce((a, b) => a * b, 1);
    const dst = new Float32Array(elems);
    const raw = this.src.slice(this.f.dataOffset + t.offset, tensorByteSize(t));
    switch (t.type) {
      case GGML_TYPE.F32: dst.set(new Float32Array(raw.buffer, 0, elems)); break;
      case GGML_TYPE.Q4_0: dequantQ4_0(raw, 0, elems / 32, dst); break;
      case GGML_TYPE.Q4_1: dequantQ4_1(raw, 0, elems / 32, dst); break;
      case GGML_TYPE.Q8_0: dequantQ8_0(raw, 0, elems / 32, dst); break;
      case GGML_TYPE.Q4_K: dequantQ4_K(raw, 0, elems / 256, dst); break;
      case GGML_TYPE.Q5_K: dequantQ5_K(raw, 0, elems / 256, dst); break;
      case GGML_TYPE.Q6_K: dequantQ6_K(raw, 0, elems / 256, dst); break;
      default: throw new Error(`q35cpurefmodel: tipo ${t.type} non gestito (${name}, ${tensorByteSize(t)} B)`);
    }
    return dst;
  }

  /** Slice di UN expert dai tensori stacked [d|dE, dE|d, nE] (it.15, lazy). */
  dequantExpert(name: string, e: number): Float32Array {
    const t = this.byName.get(name);
    if (!t) throw new Error(`q35cpurefmodel: tensore ${name} assente`);
    const elemsPer = t.dims[0] * t.dims[1];
    const dst = new Float32Array(elemsPer);
    let blockW: number, blockB: number, fn: (s: Uint8Array, o: number, n: number, d: Float32Array) => number;
    switch (t.type) {
      case GGML_TYPE.Q4_K: blockW = 256; blockB = 144; fn = dequantQ4_K; break;
      case GGML_TYPE.Q6_K: blockW = 256; blockB = 210; fn = dequantQ6_K; break;
      case GGML_TYPE.Q8_0: blockW = 32; blockB = 34; fn = dequantQ8_0; break;
      default: throw new Error(`q35cpurefmodel: expert tipo ${t.type} non gestito (${name})`);
    }
    const bytesPer = (elemsPer / blockW) * blockB;
    const raw = this.src.slice(this.f.dataOffset + t.offset + e * bytesPer, bytesPer);
    fn(raw, 0, elemsPer / blockW, dst);
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
      const raw = this.src.slice(off + token * (d / 256) * 210, (d / 256) * 210);
      dequantQ6_K(raw, 0, d / 256, out);
    } else if (t.type === GGML_TYPE.Q4_0) {
      const raw = this.src.slice(off + token * (d / 32) * 18, (d / 32) * 18);
      dequantQ4_0(raw, 0, d / 32, out);
    } else if (t.type === GGML_TYPE.Q8_0) {
      const raw = this.src.slice(off + token * (d / 32) * 34, (d / 32) * 34);
      dequantQ8_0(raw, 0, d / 32, out);
    } else throw new Error(`q35cpurefmodel: embd tipo ${t.type} non gestito`);
    return Float64Array.from(out);
  }

  private head(): Float32Array {
    if (!this.headF32) {
      this.headF32 = this.dequant(this.shape.tiedEmbeddings ? "token_embd.weight" : "output.weight");
    }
    return this.headF32;
  }

  /**
   * Forward TEACHER-FORCED su tutti i token; ritorna per ogni posizione
   * l'argmax dei logits (predizione del token successivo). Streaming
   * per-layer: attivazioni [T][d] in RAM, pesi un layer alla volta.
   */
  forward(tokens: number[], onLayer?: (l: number) => void, onFinalHidden?: (h: Float64Array[]) => void): { argmax: Int32Array; lastLogits: Float32Array } {
    const S = this.shape;
    const T = tokens.length;
    const d = S.dModel;
    let hidden: Float64Array[] = tokens.map((tk) => this.embedRow(tk));

    for (let l = 0; l < S.nLayer; l++) {
      onLayer?.(l);
      const b = `blk.${l}.`;
      const postNorm = this.dequant(`${b}post_attention_norm.weight`);
      const attnOut = this.attnLayerRef(l, hidden);
      if (S.arch === "qwen35") {
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
      } else {
        // MoE (qwen35moe): expert dequant LAZY con cache per-layer (rilasciata
        // a fine layer: 256 expert f32 = ~3.2 GB transitori nel caso peggiore)
        const nE = S.nExpert as number;
        const dE = S.dFfnExpert as number;
        const moeW: Q35MoeLayerWeights = {
          router: this.dequant(`${b}ffn_gate_inp.weight`),
          sharedGate: this.dequant(`${b}ffn_gate_inp_shexp.weight`),
          shGate: this.dequant(`${b}ffn_gate_shexp.weight`),
          shUp: this.dequant(`${b}ffn_up_shexp.weight`),
          shDown: this.dequant(`${b}ffn_down_shexp.weight`),
          expert: (() => {
            const cache = new Map<number, { gate: Float32Array; up: Float32Array; down: Float32Array }>();
            return (e: number) => {
              let hit = cache.get(e);
              if (!hit) {
                hit = {
                  gate: this.dequantExpert(`${b}ffn_gate_exps.weight`, e),
                  up: this.dequantExpert(`${b}ffn_up_exps.weight`, e),
                  down: this.dequantExpert(`${b}ffn_down_exps.weight`, e),
                };
                cache.set(e, hit);
              }
              return hit;
            };
          })(),
        };
        for (let t = 0; t < T; t++) {
          const afterAttn = new Float64Array(d);
          for (let i = 0; i < d; i++) afterAttn[i] = hidden[t][i] + attnOut[t][i];
          const xn = rmsnormF64(afterAttn, postNorm, S.rmsEps);
          const { out } = q35MoeFfnRefF64(xn, moeW, nE, S.nExpertUsed as number, dE);
          for (let i = 0; i < d; i++) afterAttn[i] += out[i];
          hidden[t] = afterAttn;
        }
      }
    }

    onFinalHidden?.(hidden);
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

  /**
   * TESTA MTP (NextN) in f64 — riferimento della fase 7.
   *
   * Data la posizione i, la testa vede l'hidden FINALE del modello a i (prima
   * di `output_norm`) e l'embedding del token i+1 — cioe' quello che il modello
   * ha appena predetto — e predice il token **i+2**. E' un token di draft
   * gratuito per ogni token vero.
   *
   * ORDINE DELLA CONCATENAZIONE: `eh_proj` e' [2*d, d] e i due ingressi sono
   * l'embedding e l'hidden, ma QUALE dei due venga prima nel file non e'
   * documentato nei metadata e NON lo indovino. `embFirst` lo rende un
   * parametro, e il test lo risolve per misura: l'ordine giusto predice il
   * token i+2 molto sopra il caso, quello sbagliato no. La misura che serve
   * comunque (accept-rate) e' anche il discriminante.
   *
   * PERCHE' ESISTE QUESTO RIFERIMENTO, e non si va diritti alla GPU: il gate
   * dello spec-dec ("token accettati == token del greedy") e' INSENSIBILE alla
   * qualita' della testa, perche' la verifica scarta i draft sbagliati. Una
   * testa implementata male passerebbe il gate e si manifesterebbe come
   * accept-rate basso, cioe' come un risultato negativo SULL'MTP invece che
   * come un bug nostro. Questo numero, misurato prima di scrivere una riga di
   * WGSL, e' la prova indipendente che manca al gate.
   */
  mtpDraftRef(tokens: number[], embFirst: boolean, hiddenIn?: Float64Array[]): Int32Array {
    const S = this.shape;
    if (S.mtpLayers < 1) throw new Error("q35cpuref: il file non porta la testa MTP (mtpLayers 0)");
    const d = S.dModel;
    const T = tokens.length;
    // `hiddenIn` evita di rifare il forward quando si provano i due ordini di
    // concatenazione sullo STESSO campione: il modello e' la parte cara.
    let hidden: Float64Array[] = hiddenIn ?? [];
    if (!hiddenIn) this.forward(tokens, undefined, (h) => { hidden = h.map((v) => Float64Array.from(v)); });

    const b = `blk.${S.nLayer}.`;
    const enorm = this.dequant(`${b}nextn.enorm.weight`);
    const hnorm = this.dequant(`${b}nextn.hnorm.weight`);
    const ehProj = this.dequant(`${b}nextn.eh_proj.weight`);
    const shNorm = this.dequant(`${b}nextn.shared_head_norm.weight`);

    // h'_i = eh_proj([norm(emb(t_{i+1})) ; norm(hidden_i)]) — l'ultima posizione
    // non ha un t_{i+1} noto, quindi la testa produce T-1 draft.
    const hp: Float64Array[] = [];
    for (let i = 0; i + 1 < T; i++) {
      const e = rmsnormF64(this.embedRow(tokens[i + 1]), enorm, S.rmsEps);
      const hh = rmsnormF64(hidden[i], hnorm, S.rmsEps);
      const cat = new Float64Array(2 * d);
      cat.set(embFirst ? e : hh, 0);
      cat.set(embFirst ? hh : e, d);
      hp.push(matVecF64(ehProj, cat, d));
    }

    // Il blocco della testa e' un layer normale: attn (FULL, forzato) + ffn.
    const attnOut = this.attnLayerRef(S.nLayer, hp, true);
    const postNorm = this.dequant(`${b}post_attention_norm.weight`);
    const wg = this.dequant(`${b}ffn_gate.weight`);
    const wu = this.dequant(`${b}ffn_up.weight`);
    const wd = this.dequant(`${b}ffn_down.weight`);
    const dFfn = S.dFfn as number;
    const head = this.head();
    const pred = new Int32Array(hp.length);
    for (let t = 0; t < hp.length; t++) {
      const x = new Float64Array(d);
      for (let i = 0; i < d; i++) x[i] = hp[t][i] + attnOut[t][i];
      const xn = rmsnormF64(x, postNorm, S.rmsEps);
      const gt = matVecF64(wg, xn, dFfn);
      const up = matVecF64(wu, xn, dFfn);
      for (let i = 0; i < dFfn; i++) gt[i] = silu(gt[i]) * up[i];
      const dn = matVecF64(wd, gt, d);
      for (let i = 0; i < d; i++) x[i] += dn[i];
      // lm_head CONDIVISA col modello, preceduta dalla norma della testa
      const hn = rmsnormF64(x, shNorm, S.rmsEps);
      let best = -Infinity, bi = -1;
      for (let r = 0; r < S.vocab; r++) {
        let acc = 0;
        const base = r * d;
        for (let i = 0; i < d; i++) acc += head[base + i] * hn[i];
        if (acc > best) { best = acc; bi = r; }
      }
      pred[t] = bi;
    }
    return pred;
  }

  /**
   * Ramo ATTENTION del layer l (pre-residual), teacher-forced su tutte le
   * posizioni: attn_norm + (full-attn | DeltaNet). Fonte unica anche per i
   * fixture del ktest GPU (fase 4 slice 2): il riferimento del layer è
   * QUESTO, non una copia.
   */
  attnLayerRef(l: number, hidden: Float64Array[], forceFull?: boolean): Float64Array[] {
    const S = this.shape;
    const T = hidden.length;
    const d = S.dModel;
    const b = `blk.${l}.`;
    const attnNorm = this.dequant(`${b}attn_norm.weight`);
    const attnOut: Float64Array[] = [];

    // `forceFull` esiste per la TESTA MTP: sta a blk.<nLayer> e la regola
    // dell'interval la direbbe deltanet (sul 4B 32 % 4 === 0), mentre il file
    // porta attn_q/k/v. Assente = comportamento di prima, bit per bit.
    if (forceFull ?? q35IsFullAttn(S, l)) {
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
    return attnOut;
  }
}
