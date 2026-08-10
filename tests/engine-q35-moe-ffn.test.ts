import { describe, it, expect } from "vitest";
import { q35MoeFfnRefF64, type Q35MoeLayerWeights } from "../src/engine/q35cpurefmodel";

// Proprietà del FFN MoE qwen35moe (q1 fase 7, semantica dalla fonte b10333):
// softmax→top-K→norm(sum-1 con clamp), pesatura DOPO il down, shared con gate
// sigmoid SCALARE. Test in dimensioni piccole con identità verificabili a mano.
const D = 8, NE = 6, DE = 4;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000 - 0.5;
  };
}
const arr = (n: number, r: () => number, scale = 1): Float32Array => Float32Array.from({ length: n }, () => r() * scale);

function synth(seed: number): Q35MoeLayerWeights {
  const r = lcg(seed);
  const experts = Array.from({ length: NE }, () => ({
    gate: arr(DE * D, r), up: arr(DE * D, r), down: arr(D * DE, r),
  }));
  return {
    router: arr(NE * D, r, 2),
    sharedGate: arr(D, r, 2),
    shGate: arr(DE * D, r), shUp: arr(DE * D, r), shDown: arr(D * DE, r),
    expert: (e) => experts[e],
  };
}

const silu = (v: number): number => v / (1 + Math.exp(-v));
const x0 = Float64Array.from({ length: D }, (_, i) => Math.sin(i + 1));

describe("q35MoeFfnRefF64: proprietà dalla fonte", () => {
  it("pesi selezionati: somma 1 (norm con clamp), ordinati per prob decrescente", () => {
    const { weights, selected } = q35MoeFfnRefF64(x0, synth(1), NE, 3, DE);
    expect(selected.length).toBe(3);
    const s = weights.reduce((a, b) => a + b, 0);
    expect(Math.abs(s - 1)).toBeLessThan(1e-12);
    for (let i = 1; i < weights.length; i++) expect(weights[i]).toBeLessThanOrEqual(weights[i - 1] + 1e-15);
  });

  it("topK = nExpert ⇒ mixture densa: out == Σ softmax_e·SwiGLU_e + shared (calcolo indipendente)", () => {
    const w = synth(2);
    const { out } = q35MoeFfnRefF64(x0, w, NE, NE, DE);
    // ricalcolo INDIPENDENTE (formulazione densa, niente top-k/norm: con
    // tutti gli expert la norm divide per Σ=1)
    const logits = new Float64Array(NE);
    for (let e = 0; e < NE; e++) for (let i = 0; i < D; i++) logits[e] += w.router[e * D + i] * x0[i];
    const mx = Math.max(...logits);
    const p = Array.from(logits, (l) => Math.exp(l - mx));
    const ps = p.reduce((a, b) => a + b, 0);
    const ref = new Float64Array(D);
    for (let e = 0; e < NE; e++) {
      const ex = w.expert(e);
      const g = new Float64Array(DE), u = new Float64Array(DE);
      for (let j = 0; j < DE; j++) for (let i = 0; i < D; i++) { g[j] += ex.gate[j * D + i] * x0[i]; u[j] += ex.up[j * D + i] * x0[i]; }
      for (let j = 0; j < DE; j++) g[j] = silu(g[j]) * u[j];
      for (let i = 0; i < D; i++) { let acc = 0; for (let j = 0; j < DE; j++) acc += ex.down[i * DE + j] * g[j]; ref[i] += (p[e] / ps) * acc; }
    }
    // shared
    const sg = new Float64Array(DE), su = new Float64Array(DE);
    for (let j = 0; j < DE; j++) for (let i = 0; i < D; i++) { sg[j] += w.shGate[j * D + i] * x0[i]; su[j] += w.shUp[j * D + i] * x0[i]; }
    for (let j = 0; j < DE; j++) sg[j] = silu(sg[j]) * su[j];
    let gr = 0;
    for (let i = 0; i < D; i++) gr += w.sharedGate[i] * x0[i];
    const gs = 1 / (1 + Math.exp(-gr));
    for (let i = 0; i < D; i++) { let acc = 0; for (let j = 0; j < DE; j++) acc += w.shDown[i * DE + j] * sg[j]; ref[i] += gs * acc; }
    for (let i = 0; i < D; i++) expect(Math.abs(out[i] - ref[i]), `i=${i}`).toBeLessThan(1e-12);
  });

  it("il gate del shared expert è SCALARE e sigmoide: con router azzerato e expert nulli resta solo shared·σ(g·x)", () => {
    const w = synth(3);
    const zero = new Float32Array(DE * D);
    const zeroD = new Float32Array(D * DE);
    const wz: Q35MoeLayerWeights = { ...w, expert: () => ({ gate: zero, up: zero, down: zeroD }) };
    const { out } = q35MoeFfnRefF64(x0, wz, NE, 2, DE);
    // ricalcola solo shared
    const sg = new Float64Array(DE), su = new Float64Array(DE);
    for (let j = 0; j < DE; j++) for (let i = 0; i < D; i++) { sg[j] += w.shGate[j * D + i] * x0[i]; su[j] += w.shUp[j * D + i] * x0[i]; }
    for (let j = 0; j < DE; j++) sg[j] = silu(sg[j]) * su[j];
    let gr = 0;
    for (let i = 0; i < D; i++) gr += w.sharedGate[i] * x0[i];
    const gs = 1 / (1 + Math.exp(-gr));
    for (let i = 0; i < D; i++) {
      let acc = 0;
      for (let j = 0; j < DE; j++) acc += w.shDown[i * DE + j] * sg[j];
      expect(Math.abs(out[i] - gs * acc)).toBeLessThan(1e-12);
    }
  });
});
