// cpuref-f64 della catena linear-attention Qwen 3.5/3.6 (Gated DeltaNet) —
// fase 3 q1, spec §4: il riferimento matematico PRIMA del kernel WGSL.
//
// Semantica presa dalla FONTE llama.cpp @ b10333 (la build dell'oracolo), non
// da memoria — file letti 2026-08-10:
//   src/models/qwen35.cpp           (build_layer_attn_linear, build_qkvz)
//   src/models/delta-net-base.cpp   (build_delta_net_autoregressive: la
//                                    ricorrenza per-token; il chunked è
//                                    l'ottimizzazione della stessa matematica)
//   ggml/src/ggml-cpu/ops.cpp       (ssm_conv: dot scorrevole k=4 su
//                                    concat(stato[3], x); l2_norm: scale =
//                                    1/max(√Σx², eps) — eps È UN FLOOR)
//   ggml/src/ggml-impl.h            (softplus: x>20 ? x : log(1+eˣ))
//
// Catena per layer linear (input x = post attn_norm, per token):
//   qkv = Wqkv·x               layout [q(nK·hd) | k(nK·hd) | v(nV·hd)]
//   z   = Wgate·x              [nV·hd]      (gate della norm finale)
//   β_h = sigmoid((Wβ·x)_h)    per v-head
//   g_h = a_h · softplus((Wα·x)_h + dtBias_h)   (a_h = −exp(A_log), dal file)
//   conv: per canale c, out_c = Σ_{i<4} w_c[i]·hist_c[i]  poi SiLU
//         (hist = ultimi 4 valori di qkv, stato = 3 precedenti)
//   split q,k,v dal conv-out; L2-norm PER HEAD su q,k (floor eps); v no
//   broadcast k-head→v-head: head h usa k-head (h mod nK)  (ggml_repeat tila)
//   q ← q/√hd
//   ricorrenza per v-head h, stato S[i,j] (i=key, j=value), ORDINE ESATTO:
//     S ← S·exp(g_h)                       (decay PRIMA della lettura)
//     sk[j] = Σ_i S[i,j]·k[i]              (lettura con S già decaduto)
//     d[j]  = β_h·(v[j] − sk[j])           (delta rule)
//     S[i,j] += k[i]·d[j]                  (update)
//     o[j]  = Σ_i S[i,j]·q[i]              (output con S aggiornato)
//   out_h = RMSnorm(o_h; ssmNorm, eps) · silu(z_h)   (gated norm per head)
//   y = Wout·concat(out)       [d]
//
// Tutto f64; i pesi arrivano f32 (dequantizzati a monte). Gli intermedi
// dell'ULTIMO step restano esposti (tap.*) per il confronto puntuale del
// ktest WGSL (fase 3) e la localizzazione per-passo delle divergenze.

export interface Q35DeltaNetDims {
  d: number; // dModel
  nK: number; // K-head (ssm.group_count, 16)
  nV: number; // V-head (ssm.time_step_rank, 32)
  hd: number; // head dim (ssm.state_size, 128)
  convK: number; // 4
  eps: number; // rms_eps (1e-6): usato da l2norm (floor) e da RMSnorm
}

export interface Q35DeltaNetWeights {
  // convenzione ggml mul_mat: y[o] = Σ_i W[o·nIn + i]·x[i] (righe = output)
  wqkv: Float32Array; // [(2nK+nV)·hd, d]
  wgate: Float32Array; // [nV·hd, d]
  conv: Float32Array; // [(2nK+nV)·hd, convK] — riga per canale
  wbeta: Float32Array; // [nV, d]
  walpha: Float32Array; // [nV, d]
  dtBias: Float32Array; // [nV]
  a: Float32Array; // [nV] — già −exp(A_log) nel GGUF
  ssmNorm: Float32Array; // [hd]
  wout: Float32Array; // [d, nV·hd]
}

export function softplusGgml(x: number): number {
  return x > 20 ? x : Math.log(1 + Math.exp(x));
}

const silu = (x: number): number => x / (1 + Math.exp(-x));
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

function matVec(w: Float32Array, x: Float64Array, nOut: number, nIn: number): Float64Array {
  if (w.length !== nOut * nIn) throw new Error(`q35cpuref: matVec ${w.length} ≠ ${nOut}×${nIn}`);
  const y = new Float64Array(nOut);
  for (let o = 0; o < nOut; o++) {
    let s = 0;
    const base = o * nIn;
    for (let i = 0; i < nIn; i++) s += w[base + i] * x[i];
    y[o] = s;
  }
  return y;
}

/**
 * Il CUORE della ricorrenza (un token, un v-head), esportato puro per i test
 * di proprietà e per il ktest WGSL. Muta S in place. Ritorna o[hd].
 * q va già scalato (1/√hd) e k già L2-normato dal chiamante.
 */
export function deltaNetStepCore(
  S: Float64Array, // [hd·hd], S[i·hd+j], i=key j=value
  q: Float64Array,
  k: Float64Array,
  v: Float64Array,
  g: number,
  beta: number,
  hd: number,
): Float64Array {
  const decay = Math.exp(g);
  for (let i = 0; i < hd * hd; i++) S[i] *= decay;
  const sk = new Float64Array(hd);
  for (let i = 0; i < hd; i++) {
    const ki = k[i];
    if (ki === 0) continue;
    const row = i * hd;
    for (let j = 0; j < hd; j++) sk[j] += S[row + j] * ki;
  }
  const dlt = new Float64Array(hd);
  for (let j = 0; j < hd; j++) dlt[j] = beta * (v[j] - sk[j]);
  for (let i = 0; i < hd; i++) {
    const ki = k[i];
    if (ki === 0) continue;
    const row = i * hd;
    for (let j = 0; j < hd; j++) S[row + j] += ki * dlt[j];
  }
  const o = new Float64Array(hd);
  for (let i = 0; i < hd; i++) {
    const qi = q[i];
    if (qi === 0) continue;
    const row = i * hd;
    for (let j = 0; j < hd; j++) o[j] += S[row + j] * qi;
  }
  return o;
}

/** Intermedi dell'ultimo step, per confronto puntuale nel ktest. */
export interface Q35DeltaNetTap {
  qkvPreConv: Float64Array;
  convOut: Float64Array; // post SiLU
  qNorm: Float64Array; // [nK·hd] post l2norm, PRE scala 1/√hd
  kNorm: Float64Array; // [nK·hd]
  v: Float64Array; // [nV·hd]
  g: Float64Array; // [nV]
  beta: Float64Array; // [nV]
  oCore: Float64Array; // [nV·hd] output ricorrenza pre gated-norm
  gated: Float64Array; // [nV·hd] post RMSnorm·silu(z)
}

export class Q35DeltaNetRef {
  private D: Q35DeltaNetDims;
  private W: Q35DeltaNetWeights;
  private qkvDim: number;
  private keyDim: number;
  /** stato conv: ultimi convK−1 vettori qkv (pre-conv), più recente in coda */
  convState: Float64Array;
  /** stato ricorrente per v-head: [nV][hd·hd] */
  S: Float64Array;
  tap: Q35DeltaNetTap | null = null;

  constructor(dims: Q35DeltaNetDims, weights: Q35DeltaNetWeights) {
    this.D = dims;
    this.W = weights;
    this.keyDim = dims.nK * dims.hd;
    this.qkvDim = 2 * this.keyDim + dims.nV * dims.hd;
    this.convState = new Float64Array((dims.convK - 1) * this.qkvDim);
    this.S = new Float64Array(dims.nV * dims.hd * dims.hd);
    if (weights.conv.length !== this.qkvDim * dims.convK) throw new Error("q35cpuref: conv len");
    if (weights.a.length !== dims.nV || weights.dtBias.length !== dims.nV) throw new Error("q35cpuref: a/dtBias len");
    if (weights.ssmNorm.length !== dims.hd) throw new Error("q35cpuref: ssmNorm len");
  }

  /** Un token. x = riga post attn_norm [d]. Ritorna il contributo linear-attn [d]. */
  step(x: Float64Array): Float64Array {
    const { d, nK, nV, hd, convK, eps } = this.D;
    const inner = nV * hd;
    const qkv = matVec(this.W.wqkv, x, this.qkvDim, d);
    const z = matVec(this.W.wgate, x, inner, d);
    const betaRaw = matVec(this.W.wbeta, x, nV, d);
    const alphaRaw = matVec(this.W.walpha, x, nV, d);
    const beta = new Float64Array(nV);
    const g = new Float64Array(nV);
    for (let h = 0; h < nV; h++) {
      beta[h] = sigmoid(betaRaw[h]);
      g[h] = this.W.a[h] * softplusGgml(alphaRaw[h] + this.W.dtBias[h]);
    }

    // conv causale k=convK: storia = [convState | qkv], poi shift dello stato
    const K1 = convK - 1;
    const convOut = new Float64Array(this.qkvDim);
    for (let c = 0; c < this.qkvDim; c++) {
      let s = 0;
      for (let i = 0; i < convK; i++) {
        const src = i < K1 ? this.convState[i * this.qkvDim + c] : qkv[c];
        s += this.W.conv[c * convK + i] * src;
      }
      convOut[c] = silu(s);
    }
    // shift: scarta il più vecchio, appendi qkv corrente
    this.convState.copyWithin(0, this.qkvDim);
    this.convState.set(qkv, K1 > 0 ? (K1 - 1) * this.qkvDim : 0);

    // split + L2-norm per head su q,k (floor eps), v intatto
    const qN = new Float64Array(this.keyDim);
    const kN = new Float64Array(this.keyDim);
    const l2 = (dst: Float64Array, srcOff: number, headOff: number): void => {
      let ss = 0;
      for (let i = 0; i < hd; i++) {
        const v0 = convOut[srcOff + headOff + i];
        ss += v0 * v0;
      }
      const scale = 1 / Math.max(Math.sqrt(ss), eps);
      for (let i = 0; i < hd; i++) dst[headOff + i] = convOut[srcOff + headOff + i] * scale;
    };
    for (let h = 0; h < nK; h++) {
      l2(qN, 0, h * hd);
      l2(kN, this.keyDim, h * hd);
    }
    const v = convOut.slice(2 * this.keyDim, 2 * this.keyDim + inner);

    // ricorrenza per v-head; k-head = h mod nK (ggml_repeat tila), q/√hd
    const scaleQ = 1 / Math.sqrt(hd);
    const oCore = new Float64Array(inner);
    const qh = new Float64Array(hd);
    const kh = new Float64Array(hd);
    const vh = new Float64Array(hd);
    for (let h = 0; h < nV; h++) {
      const kHead = (h % nK) * hd;
      for (let i = 0; i < hd; i++) {
        qh[i] = qN[kHead + i] * scaleQ;
        kh[i] = kN[kHead + i];
        vh[i] = v[h * hd + i];
      }
      const o = deltaNetStepCore(this.S.subarray(h * hd * hd, (h + 1) * hd * hd), qh, kh, vh, g[h], beta[h], hd);
      oCore.set(o, h * hd);
    }

    // gated norm per head: RMSnorm(o; ssmNorm) · silu(z)
    const gated = new Float64Array(inner);
    for (let h = 0; h < nV; h++) {
      let ss = 0;
      for (let i = 0; i < hd; i++) {
        const o = oCore[h * hd + i];
        ss += o * o;
      }
      const inv = 1 / Math.sqrt(ss / hd + eps);
      for (let i = 0; i < hd; i++) {
        const idx = h * hd + i;
        gated[idx] = oCore[idx] * inv * this.W.ssmNorm[i] * silu(z[idx]);
      }
    }

    this.tap = { qkvPreConv: qkv, convOut, qNorm: qN, kNorm: kN, v, g, beta, oCore, gated };
    return matVec(this.W.wout, gated, d, inner);
  }
}
