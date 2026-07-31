// Identità algebrica del layer GLM (goal C2 fase 4): il cpuref f64 NAIVE
// (decompressione K/V per posizione, cpuref.ts) deve coincidere a rounding f64
// con la formulazione ABSORBED (quella del motore GPU), qui implementata da
// zero — rms/rope/matvec locali, accesso DIRETTO alle righe di wk_b/wv_b.
// Un errore di trasposizione o di ordine rope/norm in una delle due rompe
// l'identità; il gate coi pesi reali è il ktest (glm-layer0-conformance).
import { describe, expect, it } from "vitest";
import { GlmDenseLayerRefF64 } from "../src/engine/cpuref";
import { GLM47_FLASH as G } from "../src/engine/shape";

const HL = G.qkNope + G.ropeDims; // 256

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
}

function randW(n: number, seed: number, scale: number): Float32Array {
  const r = lcg(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (r() * 2 - 1) * scale;
  return out;
}

// --- implementazione absorbed INDIPENDENTE (solo per questo test) ---

function rms(x: Float64Array, w: Float32Array, eps: number): Float64Array {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const s = 1 / Math.sqrt(ss / x.length + eps);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * s * w[i];
  return out;
}

function matvec(w: Float32Array, x: Float64Array, rows: number): Float64Array {
  const k = x.length;
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    for (let i = 0; i < k; i++) acc += w[r * k + i] * x[i];
    out[r] = acc;
  }
  return out;
}

function rope(v: Float64Array, nVec: number, stride: number, off: number, dims: number, base: number, pos: number): void {
  for (let h = 0; h < nVec; h++) {
    for (let i = 0; i < dims / 2; i++) {
      const th = pos * base ** (-(2 * i) / dims);
      const c = Math.cos(th), s = Math.sin(th);
      const b = h * stride + off + 2 * i;
      const a0 = v[b], a1 = v[b + 1];
      v[b] = a0 * c - a1 * s;
      v[b + 1] = a0 * s + a1 * c;
    }
  }
}

interface W {
  attnNorm: Float32Array; qANorm: Float32Array; kvANorm: Float32Array; ffnNorm: Float32Array;
  wQA: Float32Array; wQB: Float32Array; wKvA: Float32Array; wKB: Float32Array; wVB: Float32Array;
  wO: Float32Array; wGate: Float32Array; wUp: Float32Array; wDown: Float32Array;
}

class AbsorbedRef {
  private w: W;
  private cKv: Float64Array[] = [];
  private kPe: Float64Array[] = [];
  constructor(w: W) { this.w = w; }

  forward(xIn: Float32Array): Float64Array {
    const w = this.w;
    const pos = this.cKv.length;
    const x = Float64Array.from(xIn);
    const hn = rms(x, w.attnNorm, G.rmsEps);
    const qan = rms(matvec(w.wQA, hn, G.qLora), w.qANorm, G.rmsEps);
    const q = matvec(w.wQB, qan, G.nHead * HL);
    rope(q, G.nHead, HL, G.qkNope, G.ropeDims, G.ropeFreqBase, pos);
    const kv = matvec(w.wKvA, hn, G.keyLen);
    rope(kv, 1, G.keyLen, G.kvLora, G.ropeDims, G.ropeFreqBase, pos);
    this.cKv.push(rms(kv.subarray(0, G.kvLora), w.kvANorm, G.rmsEps));
    this.kPe.push(Float64Array.from(kv.subarray(G.kvLora)));

    const scale = 1 / Math.sqrt(G.headLenMla);
    const cat = new Float64Array(G.nHead * G.headLenMla);
    for (let h = 0; h < G.nHead; h++) {
      // assorbimento: q_ckv = wk_b(h) · q_nope — righe DIRETTE [512×192]
      const qCkv = new Float64Array(G.kvLora);
      for (let r = 0; r < G.kvLora; r++) {
        let acc = 0;
        const base = h * G.kvLora * G.qkNope + r * G.qkNope;
        for (let i = 0; i < G.qkNope; i++) acc += w.wKB[base + i] * q[h * HL + i];
        qCkv[r] = acc;
      }
      const scores = new Float64Array(pos + 1);
      for (let p = 0; p <= pos; p++) {
        let acc = 0;
        for (let r = 0; r < G.kvLora; r++) acc += qCkv[r] * this.cKv[p][r];
        for (let j = 0; j < G.ropeDims; j++) acc += q[h * HL + G.qkNope + j] * this.kPe[p][j];
        scores[p] = acc * scale;
      }
      let m = -Infinity;
      for (const s of scores) m = Math.max(m, s);
      let sum = 0;
      for (let p = 0; p <= pos; p++) { scores[p] = Math.exp(scores[p] - m); sum += scores[p]; }
      // attn in spazio c_kv, poi wv_b(h) — righe DIRETTE [256×512]
      const attnCkv = new Float64Array(G.kvLora);
      for (let p = 0; p <= pos; p++) {
        const wp = scores[p] / sum;
        for (let r = 0; r < G.kvLora; r++) attnCkv[r] += wp * this.cKv[p][r];
      }
      for (let mI = 0; mI < G.headLenMla; mI++) {
        let acc = 0;
        const base = h * G.headLenMla * G.kvLora + mI * G.kvLora;
        for (let r = 0; r < G.kvLora; r++) acc += w.wVB[base + r] * attnCkv[r];
        cat[h * G.headLenMla + mI] = acc;
      }
    }
    const o = matvec(w.wO, cat, G.dModel);
    for (let i = 0; i < G.dModel; i++) x[i] += o[i];
    const fn = rms(x, w.ffnNorm, G.rmsEps);
    const gate = matvec(w.wGate, fn, G.dFfnDense);
    const up = matvec(w.wUp, fn, G.dFfnDense);
    for (let i = 0; i < G.dFfnDense; i++) gate[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * up[i];
    const down = matvec(w.wDown, gate, G.dModel);
    for (let i = 0; i < G.dModel; i++) x[i] += down[i];
    return x;
  }
}

describe("cpuref GLM layer denso — identità naive/absorbed", () => {
  it("le due formulazioni MLA coincidono a rounding f64 su 2 posizioni", () => {
    const w: W = {
      attnNorm: randW(G.dModel, 1, 1), qANorm: randW(G.qLora, 2, 1),
      kvANorm: randW(G.kvLora, 3, 1), ffnNorm: randW(G.dModel, 4, 1),
      wQA: randW(G.qLora * G.dModel, 5, 0.05),
      wQB: randW(G.nHead * HL * G.qLora, 6, 0.05),
      wKvA: randW(G.keyLen * G.dModel, 7, 0.05),
      wKB: randW(G.nHead * G.kvLora * G.qkNope, 8, 0.05),
      wVB: randW(G.nHead * G.headLenMla * G.kvLora, 9, 0.05),
      wO: randW(G.dModel * G.nHead * G.headLenMla, 10, 0.02),
      wGate: randW(G.dFfnDense * G.dModel, 11, 0.02),
      wUp: randW(G.dFfnDense * G.dModel, 12, 0.02),
      wDown: randW(G.dModel * G.dFfnDense, 13, 0.02),
    };
    const naive = new GlmDenseLayerRefF64(w);
    const absorbed = new AbsorbedRef(w);
    for (let pos = 0; pos < 2; pos++) {
      const x = randW(G.dModel, 100 + pos, 0.5);
      const a = naive.forward(x);
      const b = absorbed.forward(x);
      let maxAbs = 0;
      for (let i = 0; i < G.dModel; i++) maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
      expect(maxAbs).toBeLessThan(1e-8);
    }
  });
});
