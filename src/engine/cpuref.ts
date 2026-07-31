// Forward CPU di riferimento per Qwen2.5-0.5B — SOLO test/debug, mai nel percorso
// di produzione (il motore è WebGPU; questo è l'anello di differential testing tra
// oracolo llama.cpp e kernel WGSL: se il CPU-ref concorda con l'oracolo, la nostra
// comprensione del modello è giusta e ogni divergenza GPU è un bug di kernel).
//
// f32 puro, dequant dal riferimento quant.ts, zero dipendenze. Lento per design
// (~1-2 s/token): va usato su pochi token, non sul corpus intero.
import { GGML_TYPE, parseGguf, tensorByteSize } from "./gguf";
import { dequantQ4_0, dequantQ8_0 } from "./quant";
import { GLM47_FLASH as G, QWEN25_05B as S, validateQwen25_05B } from "./shape";

interface Tensors { [name: string]: Float32Array }

// Dequantizza l'intero modello in f32 (~2.5 GB: accettabile su una dev box, è il
// prezzo del riferimento semplice).
export function loadCpuRef(buf: ArrayBuffer): Tensors {
  const f = parseGguf(buf);
  const byName = validateQwen25_05B(f);
  const bytes = new Uint8Array(buf);
  const out: Tensors = {};
  for (const [name, t] of byName) {
    const elems = t.dims.reduce((a, b) => a * b, 1);
    const dst = new Float32Array(elems);
    const off = f.dataOffset + t.offset;
    if (t.type === GGML_TYPE.F32) {
      dst.set(new Float32Array(buf, off, elems));
    } else if (t.type === GGML_TYPE.Q4_0) {
      dequantQ4_0(bytes, off, elems / 32, dst);
    } else if (t.type === GGML_TYPE.Q8_0) {
      dequantQ8_0(bytes, off, elems / 32, dst);
    } else {
      throw new Error(`cpuref: tipo ${t.type} inatteso per ${name} (${tensorByteSize(t)} B)`);
    }
    out[name] = dst;
  }
  return out;
}

function rmsnorm(x: Float32Array, w: Float32Array, eps: number): Float32Array {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const scale = 1 / Math.sqrt(ss / x.length + eps);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale * w[i];
  return out;
}

// mul_mat ggml: W ha righe da `k` elementi (ne[0]=k), out[i] = W[i,:]·x (+ bias).
function matvec(w: Float32Array, x: Float32Array, rows: number, bias?: Float32Array): Float32Array {
  const k = x.length;
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const base = r * k;
    let acc = 0;
    for (let i = 0; i < k; i++) acc += w[base + i] * x[i];
    out[r] = bias ? acc + bias[r] : acc;
  }
  return out;
}

// RoPE NEOX (qwen2, llama.cpp rope type neox): coppie (j, j+half) dentro ogni head.
function ropeNeox(v: Float32Array, nHead: number, pos: number): void {
  const hd = S.headDim;
  const half = hd / 2;
  for (let h = 0; h < nHead; h++) {
    const base = h * hd;
    for (let j = 0; j < half; j++) {
      const theta = pos / S.ropeFreqBase ** (j / half);
      const cos = Math.cos(theta), sin = Math.sin(theta);
      const a = v[base + j], b = v[base + j + half];
      v[base + j] = a * cos - b * sin;
      v[base + j + half] = a * sin + b * cos;
    }
  }
}

export class CpuRefSession {
  private t: Tensors;
  private kCache: Float32Array[]; // per layer: [ctx, nKvHead*headDim]
  private vCache: Float32Array[];
  private nPast = 0;
  private ctxMax: number;

  constructor(t: Tensors, ctxMax = 512) {
    this.t = t;
    this.ctxMax = ctxMax;
    const kvDim = S.nKvHead * S.headDim;
    this.kCache = Array.from({ length: S.nLayer }, () => new Float32Array(ctxMax * kvDim));
    this.vCache = Array.from({ length: S.nLayer }, () => new Float32Array(ctxMax * kvDim));
  }

  // Un forward di un token alla posizione corrente; ritorna i logits f32 completi.
  forward(token: number): Float32Array {
    if (this.nPast >= this.ctxMax) throw new Error("cpuref: contesto pieno");
    const t = this.t;
    const kvDim = S.nKvHead * S.headDim;
    const groups = S.nHead / S.nKvHead;
    let x = t["token_embd.weight"].slice(token * S.dModel, (token + 1) * S.dModel);

    for (let l = 0; l < S.nLayer; l++) {
      const p = (n: string) => t[`blk.${l}.${n}`];
      // attention
      const hn = rmsnorm(x, p("attn_norm.weight"), S.rmsEps);
      const q = matvec(p("attn_q.weight"), hn, S.dModel, p("attn_q.bias"));
      const k = matvec(p("attn_k.weight"), hn, kvDim, p("attn_k.bias"));
      const v = matvec(p("attn_v.weight"), hn, kvDim, p("attn_v.bias"));
      ropeNeox(q, S.nHead, this.nPast);
      ropeNeox(k, S.nKvHead, this.nPast);
      this.kCache[l].set(k, this.nPast * kvDim);
      this.vCache[l].set(v, this.nPast * kvDim);

      const attnOut = new Float32Array(S.dModel);
      const scale = 1 / Math.sqrt(S.headDim);
      for (let h = 0; h < S.nHead; h++) {
        const kvHead = Math.floor(h / groups);
        const qOff = h * S.headDim;
        const scores = new Float32Array(this.nPast + 1);
        for (let pos = 0; pos <= this.nPast; pos++) {
          const kOff = pos * kvDim + kvHead * S.headDim;
          let acc = 0;
          for (let i = 0; i < S.headDim; i++) acc += q[qOff + i] * this.kCache[l][kOff + i];
          scores[pos] = acc * scale;
        }
        // softmax f32
        let max = -Infinity;
        for (const s of scores) if (s > max) max = s;
        let sum = 0;
        for (let pos = 0; pos < scores.length; pos++) { scores[pos] = Math.exp(scores[pos] - max); sum += scores[pos]; }
        for (let pos = 0; pos < scores.length; pos++) scores[pos] /= sum;
        for (let pos = 0; pos <= this.nPast; pos++) {
          const vOff = pos * kvDim + kvHead * S.headDim;
          const w = scores[pos];
          for (let i = 0; i < S.headDim; i++) attnOut[qOff + i] += w * this.vCache[l][vOff + i];
        }
      }
      const o = matvec(p("attn_output.weight"), attnOut, S.dModel);
      for (let i = 0; i < S.dModel; i++) x[i] += o[i];

      // ffn
      const fn = rmsnorm(x, p("ffn_norm.weight"), S.rmsEps);
      const gate = matvec(p("ffn_gate.weight"), fn, S.dFfn);
      const up = matvec(p("ffn_up.weight"), fn, S.dFfn);
      for (let i = 0; i < S.dFfn; i++) {
        const g = gate[i];
        gate[i] = (g / (1 + Math.exp(-g))) * up[i]; // silu(gate)*up
      }
      const down = matvec(p("ffn_down.weight"), gate, S.dModel);
      for (let i = 0; i < S.dModel; i++) x[i] += down[i];
    }

    const final = rmsnorm(x, t["output_norm.weight"], S.rmsEps);
    this.nPast++;
    return matvec(t["output.weight"], final, S.vocab);
  }
}

export function argmax(v: Float32Array): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

// --- GLM-4.7-Flash (goal C2 fase 4): riferimento f64 del layer denso (blk.0) ---
//
// MLA in formulazione NAIVE (decompressione K/V per head via wk_b/wv_b
// trasposti), deliberatamente DIVERSA dall'absorbed del motore GPU: se le due
// formulazioni concordano sui pesi reali, l'algebra dell'assorbimento e
// l'indicizzazione dei kernel sono giuste (gate cpuref del doppio gate §7).
// Semantica verificata nell'oracolo (it.4): rope NORM su q[192..256) per head
// e su kv[512..576) PRIMA della kv_a_norm (che tocca solo [0..512)); cache =
// [c_kv normata | k_pe ruotato]; scale 1/sqrt(256); V = c_kv normata.

export interface GlmDenseLayerWeights {
  attnNorm: Float32Array; // [2048]
  wQA: Float32Array;      // [768 righe × 2048]
  qANorm: Float32Array;   // [768]
  wQB: Float32Array;      // [5120 righe × 768]
  wKvA: Float32Array;     // [576 righe × 2048]
  kvANorm: Float32Array;  // [512]
  wKB: Float32Array;      // [20 head][512 righe × 192]
  wVB: Float32Array;      // [20 head][256 righe × 512]
  wO: Float32Array;       // [2048 righe × 5120]
  ffnNorm: Float32Array;  // [2048]
  wGate: Float32Array;    // [10240 righe × 2048]
  wUp: Float32Array;      // [10240 righe × 2048]
  wDown: Float32Array;    // [2048 righe × 10240]
}

function rmsnormF64(x: Float64Array, w: ArrayLike<number>, eps: number): Float64Array {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const scale = 1 / Math.sqrt(ss / x.length + eps);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale * w[i];
  return out;
}

function matvecF64(w: ArrayLike<number>, wOff: number, x: Float64Array, rows: number): Float64Array {
  const k = x.length;
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    const base = wOff + r * k;
    let acc = 0;
    for (let i = 0; i < k; i++) acc += w[base + i] * x[i];
    out[r] = acc;
  }
  return out;
}

// RoPE tipo NORM (coppie consecutive) sul segmento [offset, offset+dims) di ogni
// vettore: theta_i = pos · base^(−2i/dims) — riscontro llama-model.cpp/ops.cpp it.4.
function ropeNormF64(v: Float64Array, nVec: number, stride: number, offset: number, dims: number, freqBase: number, pos: number): void {
  for (let h = 0; h < nVec; h++) {
    for (let i = 0; i < dims / 2; i++) {
      const theta = pos * freqBase ** (-(2 * i) / dims);
      const c = Math.cos(theta), s = Math.sin(theta);
      const b = h * stride + offset + 2 * i;
      const a0 = v[b], a1 = v[b + 1];
      v[b] = a0 * c - a1 * s;
      v[b + 1] = a0 * s + a1 * c;
    }
  }
}

// --- GLM-4.7-Flash (goal C2 fase 5): riferimento f64 del blocco MoE-FFN ---
//
// Input = hidden POST ffn_norm; output = moe_out + ffn_shexp (il residuo
// ffn_inp resta fuori, come in deepseek2.cpp r.387-412). Router e selezione
// implementati QUI in modo indipendente da moe.ts (selezione sort-based vs
// scan-based: la concordanza dei due percorsi è parte del gate, pattern
// naive/absorbed di fase 4). Semantica dall'oracolo (it.6): sigmoid dei
// logits; bias exp_probs_b SOLO nella selezione; pesi = sigmoid senza bias
// normalizzati (denominatore clampato a 6.103515625e-5) × 1.8; pesatura DOPO
// il down; shexp calcolato sullo STESSO input post-norm e sommato a moe_out.

export interface GlmMoeExpertWeights {
  gate: ArrayLike<number>; // [1536 righe × 2048]
  up: ArrayLike<number>;   // [1536 righe × 2048]
  down: ArrayLike<number>; // [2048 righe × 1536]
}

export interface GlmMoeFfnWeights {
  routerW: ArrayLike<number>;    // [64 righe × 2048] f32
  routerBias: ArrayLike<number>; // exp_probs_b [64]
  expert: (e: number) => GlmMoeExpertWeights; // accesso lazy (i test ne materializzano 4)
  gateShexp: ArrayLike<number>;  // [1536 righe × 2048]
  upShexp: ArrayLike<number>;
  downShexp: ArrayLike<number>;  // [2048 righe × 1536]
}

export interface GlmMoeFfnRefOut {
  out: Float64Array;     // [2048] moe_out + shexp
  experts: Int32Array;   // top-4 selezionati (ordine decrescente di score biased)
  weights: Float64Array; // pesi di mixing allineati (già ×1.8)
}

function ffnChainF64(
  gate: ArrayLike<number>, up: ArrayLike<number>, down: ArrayLike<number>,
  x: Float64Array, dFfn: number, dModel: number,
): Float64Array {
  const g = matvecF64(gate, 0, x, dFfn);
  const u = matvecF64(up, 0, x, dFfn);
  for (let i = 0; i < dFfn; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
  return matvecF64(down, 0, g, dModel);
}

export function glmMoeFfnRefF64(fnIn: ArrayLike<number>, w: GlmMoeFfnWeights): GlmMoeFfnRefOut {
  const fn = Float64Array.from(fnIn as ArrayLike<number>);
  const logits = matvecF64(w.routerW, 0, fn, G.nExpert);
  const probs = new Float64Array(G.nExpert);
  for (let i = 0; i < G.nExpert; i++) probs[i] = 1 / (1 + Math.exp(-logits[i]));
  // selezione sort-based: score biased decrescente, pareggi → indice minore
  const order = Array.from({ length: G.nExpert }, (_, i) => i).sort((a, b) => {
    const sa = probs[a] + Number(w.routerBias[a]), sb = probs[b] + Number(w.routerBias[b]);
    return sb !== sa ? sb - sa : a - b;
  });
  const experts = Int32Array.from(order.slice(0, G.nExpertUsed));
  let sum = 0;
  for (const e of experts) sum += probs[e];
  const denom = Math.max(sum, 6.103515625e-5);
  const weights = Float64Array.from(experts, (e) => (probs[e] / denom) * G.weightsScale);

  const out = ffnChainF64(w.gateShexp, w.upShexp, w.downShexp, fn, G.dFfnExpert, G.dModel);
  for (let k = 0; k < experts.length; k++) {
    const ex = w.expert(experts[k]);
    const d = ffnChainF64(ex.gate, ex.up, ex.down, fn, G.dFfnExpert, G.dModel);
    for (let i = 0; i < G.dModel; i++) out[i] += weights[k] * d[i];
  }
  return { out, experts, weights };
}

export class GlmDenseLayerRefF64 {
  private w: GlmDenseLayerWeights;
  private cKv: Float64Array[] = []; // per pos: [512] normata
  private kPe: Float64Array[] = []; // per pos: [64] ruotata
  constructor(w: GlmDenseLayerWeights) { this.w = w; }

  // Forward di un token alla prossima posizione (decode); ritorna l'hidden
  // post-layer (2048). L'input non viene mutato.
  forward(xIn: ArrayLike<number>): Float64Array {
    const w = this.w;
    const pos = this.cKv.length;
    const HL = G.qkNope + G.ropeDims; // 256
    const x = Float64Array.from(xIn as ArrayLike<number>);

    // attention MLA naive
    const hn = rmsnormF64(x, w.attnNorm, G.rmsEps);
    const qa = matvecF64(w.wQA, 0, hn, G.qLora);
    const qan = rmsnormF64(qa, w.qANorm, G.rmsEps);
    const q = matvecF64(w.wQB, 0, qan, G.nHead * HL);
    ropeNormF64(q, G.nHead, HL, G.qkNope, G.ropeDims, G.ropeFreqBase, pos);
    const kv = matvecF64(w.wKvA, 0, hn, G.keyLen);
    ropeNormF64(kv, 1, G.keyLen, G.kvLora, G.ropeDims, G.ropeFreqBase, pos);
    const cKv = rmsnormF64(kv.subarray(0, G.kvLora) as Float64Array, w.kvANorm, G.rmsEps);
    this.cKv.push(cKv);
    this.kPe.push(Float64Array.from(kv.subarray(G.kvLora, G.keyLen)));

    const scale = 1 / Math.sqrt(G.headLenMla);
    const attnCat = new Float64Array(G.nHead * G.headLenMla);
    for (let h = 0; h < G.nHead; h++) {
      const qOff = h * HL;
      const scores = new Float64Array(pos + 1);
      for (let p = 0; p <= pos; p++) {
        // k_nope(h,p) = wk_b(h)ᵀ · c_kv(p) — accesso TRASPOSTO (naive)
        const ck = this.cKv[p];
        let acc = 0;
        for (let i = 0; i < G.qkNope; i++) {
          let kn = 0;
          const base = h * G.kvLora * G.qkNope + i;
          for (let r = 0; r < G.kvLora; r++) kn += w.wKB[base + r * G.qkNope] * ck[r];
          acc += q[qOff + i] * kn;
        }
        const kp = this.kPe[p];
        for (let j = 0; j < G.ropeDims; j++) acc += q[qOff + G.qkNope + j] * kp[j];
        scores[p] = acc * scale;
      }
      let m = -Infinity;
      for (const s of scores) m = Math.max(m, s);
      let sum = 0;
      for (let p = 0; p <= pos; p++) { scores[p] = Math.exp(scores[p] - m); sum += scores[p]; }
      // v(h,p) = wv_b(h) · c_kv(p): decompressione per POSIZIONE (naive)
      for (let p = 0; p <= pos; p++) {
        const wp = scores[p] / sum;
        const ck = this.cKv[p];
        for (let mI = 0; mI < G.headLenMla; mI++) {
          const base = h * G.headLenMla * G.kvLora + mI * G.kvLora;
          let acc = 0;
          for (let r = 0; r < G.kvLora; r++) acc += w.wVB[base + r] * ck[r];
          attnCat[h * G.headLenMla + mI] += wp * acc;
        }
      }
    }
    const o = matvecF64(w.wO, 0, attnCat, G.dModel);
    for (let i = 0; i < G.dModel; i++) x[i] += o[i];

    // ffn denso (blk.0)
    const fn = rmsnormF64(x, w.ffnNorm, G.rmsEps);
    const gate = matvecF64(w.wGate, 0, fn, G.dFfnDense);
    const up = matvecF64(w.wUp, 0, fn, G.dFfnDense);
    for (let i = 0; i < G.dFfnDense; i++) gate[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * up[i];
    const down = matvecF64(w.wDown, 0, gate, G.dModel);
    for (let i = 0; i < G.dModel; i++) x[i] += down[i];
    return x;
  }
}
