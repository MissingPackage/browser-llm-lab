// Forward CPU di riferimento per Qwen2.5-0.5B — SOLO test/debug, mai nel percorso
// di produzione (il motore è WebGPU; questo è l'anello di differential testing tra
// oracolo llama.cpp e kernel WGSL: se il CPU-ref concorda con l'oracolo, la nostra
// comprensione del modello è giusta e ogni divergenza GPU è un bug di kernel).
//
// f32 puro, dequant dal riferimento quant.ts, zero dipendenze. Lento per design
// (~1-2 s/token): va usato su pochi token, non sul corpus intero.
import { GGML_TYPE, parseGguf, tensorByteSize } from "./gguf";
import { dequantQ4_0, dequantQ8_0 } from "./quant";
import { QWEN25_05B as S, validateQwen25_05B } from "./shape";

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
