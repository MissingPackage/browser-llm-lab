// Campione sintetico DETERMINISTICO della catena DeltaNet (fase 3 q1) —
// bersaglio CONDIVISO di: test anti-drift node (tests/engine-q35-cpuref-
// deltanet.test.ts via tests/helpers/), generatore fixture e ktest WGSL
// (ktest.worker.ts importa da qui: src non importa da tests/).
// Dimensioni ridotte ma con la STESSA struttura del 4B (nV = 2·nK ⇒ il
// broadcast k-head h mod nK è esercitato).
import { Q35DeltaNetRef, type Q35DeltaNetDims, type Q35DeltaNetWeights } from "./q35cpuref";

// LCG deterministico: niente Math.random, campione riproducibile bit-a-bit.
export function sampleLcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000 - 0.5;
  };
}

export const SAMPLE_DIMS: Q35DeltaNetDims = { d: 32, nK: 4, nV: 8, hd: 16, convK: 4, eps: 1e-6 };
export const SAMPLE_T = 12;

export function sampleWeights(): Q35DeltaNetWeights {
  const D = SAMPLE_DIMS;
  const r = sampleLcg(20260810);
  const arr = (n: number, scale: number): Float32Array => Float32Array.from({ length: n }, () => r() * scale);
  const qkvDim = (2 * D.nK + D.nV) * D.hd;
  const inner = D.nV * D.hd;
  return {
    wqkv: arr(qkvDim * D.d, 0.5),
    wgate: arr(inner * D.d, 0.5),
    conv: arr(qkvDim * D.convK, 0.8),
    wbeta: arr(D.nV * D.d, 0.5),
    walpha: arr(D.nV * D.d, 0.5),
    dtBias: arr(D.nV, 1),
    a: Float32Array.from({ length: D.nV }, () => -Math.exp(r() * 2)), // −exp(A_log): negativo come nel file
    ssmNorm: arr(D.hd, 1).map((x) => 1 + x * 0.1) as Float32Array,
    wout: arr(D.d * inner, 0.5),
  };
}

export function sampleInputs(): Float64Array[] {
  const r = sampleLcg(42);
  return Array.from({ length: SAMPLE_T }, () => Float64Array.from({ length: SAMPLE_DIMS.d }, () => r() * 2));
}

/** Esegue la catena cpuref-f64 sul campione. Ritorna gli output per token. */
export function runSample(): { outputs: number[][]; ref: Q35DeltaNetRef } {
  const ref = new Q35DeltaNetRef(SAMPLE_DIMS, sampleWeights());
  const outputs = sampleInputs().map((x) => Array.from(ref.step(x)));
  return { outputs, ref };
}
