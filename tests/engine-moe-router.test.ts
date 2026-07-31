// Router MoE (goal C2 fase 5): semantica replicata da build_moe_ffn (oracolo
// 5f55650) — sigmoid, bias exp_probs_b SOLO nella selezione, pesi normalizzati
// con clamp del denominatore, ×1.8, tie-break indice minore. Più il layout
// slab expert (packExpertSlab) contro i repack di riferimento.
import { describe, expect, it } from "vitest";
import {
  routerSelect, packExpertSlab, SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, WEIGHTS_SUM_CLAMP_MIN,
} from "../src/engine/moe";
import { repackQ4_0, repackQ4_1 } from "../src/engine/quant";
import { GLM47_FLASH as G } from "../src/engine/shape";

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

describe("routerSelect (replica build_moe_ffn)", () => {
  it("tie-break: a parità di score vince l'indice minore", () => {
    const logits = new Float32Array(G.nExpert); // tutti 0 ⇒ probs tutti 0.5
    const bias = new Float32Array(G.nExpert);
    const { experts, weights } = routerSelect(logits, bias);
    expect(Array.from(experts)).toEqual([0, 1, 2, 3]);
    // pesi uniformi: 0.5/2.0 × 1.8
    for (const w of weights) expect(w).toBeCloseTo(0.45, 12);
  });

  it("il bias decide la selezione ma NON entra nei pesi di mixing", () => {
    const logits = new Float32Array(G.nExpert).fill(-5);
    logits.set([2, 1.5, 1.0, 0.5, 0]);
    const bias = new Float32Array(G.nExpert);
    bias[4] = 0.4; // σ(0)+0.4 = 0.9 > σ(2) = 0.8808 ⇒ expert 4 primo
    const { experts, weights } = routerSelect(logits, bias);
    expect(Array.from(experts)).toEqual([4, 0, 1, 2]);
    const probs = [sigmoid(0), sigmoid(2), sigmoid(1.5), sigmoid(1)];
    const sum = probs.reduce((a, b) => a + b, 0);
    for (let k = 0; k < 4; k++) {
      expect(weights[k]).toBeCloseTo((probs[k] / sum) * G.weightsScale, 10);
    }
  });

  it("somma dei pesi = weightsScale (1.8) nel regime normale", () => {
    const r = (s: number) => () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
    const rand = r(42);
    for (let trial = 0; trial < 50; trial++) {
      const logits = Float32Array.from({ length: G.nExpert }, () => (rand() * 2 - 1) * 3);
      const bias = Float32Array.from({ length: G.nExpert }, () => (rand() * 2 - 1) * 0.5);
      const { weights } = routerSelect(logits, bias);
      const sum = weights.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(G.weightsScale, 9);
    }
  });

  it("denominatore clampato a 6.103515625e-5 (niente divisione per ~0)", () => {
    const logits = new Float32Array(G.nExpert).fill(-20); // probs ≈ 2e-9
    const bias = new Float32Array(G.nExpert);
    const { experts, weights } = routerSelect(logits, bias);
    expect(Array.from(experts)).toEqual([0, 1, 2, 3]);
    const p = sigmoid(-20);
    for (const w of weights) {
      expect(Number.isFinite(w)).toBe(true);
      expect(w).toBeCloseTo((p / WEIGHTS_SUM_CLAMP_MIN) * G.weightsScale, 12);
    }
  });

  it("concorda con una selezione sort-based indipendente su input casuali", () => {
    const r = (s: number) => () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
    const rand = r(1337);
    for (let trial = 0; trial < 200; trial++) {
      const logits = Float32Array.from({ length: G.nExpert }, () => (rand() * 2 - 1) * 4);
      const bias = Float32Array.from({ length: G.nExpert }, () => (rand() * 2 - 1) * 0.6);
      const sel = routerSelect(logits, bias);
      const scored = Array.from({ length: G.nExpert }, (_, i) => ({ i, s: sigmoid(logits[i]) + bias[i] }))
        .sort((a, b) => (b.s !== a.s ? b.s - a.s : a.i - b.i));
      expect(Array.from(sel.experts)).toEqual(scored.slice(0, 4).map((e) => e.i));
    }
  });
});

describe("packExpertSlab (layout slab per-slot)", () => {
  const blocks = (G.dModel / 32) * G.dFfnExpert; // 98.304
  const randBytes = (n: number, seed: number) => {
    let s = seed >>> 0;
    return Uint8Array.from({ length: n }, () => {
      s = (s * 1103515245 + 12345) >>> 0;
      return s & 0xff;
    });
  };

  it("taglie esatte delle due size-class (spec §1) e offset allineati a 256", () => {
    expect(SLAB_DOWN_Q4_0.bytes).toBe(5_308_416);
    expect(SLAB_DOWN_Q4_1.bytes).toBe(5_505_024);
    for (const l of [SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1]) {
      for (const off of [l.gateQs, l.gateScales, l.upQs, l.upScales, l.downQs, l.downScales]) {
        expect(off % 256).toBe(0);
      }
    }
  });

  // niente toEqual sui typed array multi-MB (il deep-diff di vitest va in
  // timeout): confronto byte-per-byte, riporta l'indice del primo mismatch
  const firstMismatch = (a: Uint8Array, b: Uint8Array): number => {
    if (a.length !== b.length) return -2;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return i;
    return -1;
  };

  it("i segmenti dello slab coincidono col repack di riferimento", () => {
    const gate = randBytes(blocks * 18, 1);
    const up = randBytes(blocks * 18, 2);
    const down = randBytes(blocks * 20, 3);
    const slab = packExpertSlab(gate, up, down, SLAB_DOWN_Q4_1);
    expect(slab.length).toBe(SLAB_DOWN_Q4_1.bytes);
    const u8 = (a: Uint32Array) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const rGate = repackQ4_0(gate, 0, blocks);
    const rUp = repackQ4_0(up, 0, blocks);
    const rDown = repackQ4_1(down, 0, blocks);
    const l = SLAB_DOWN_Q4_1;
    const segs: Array<[string, number, number, Uint8Array]> = [
      ["gateQs", l.gateQs, l.qsBytes, u8(rGate.qs)],
      ["gateScales", l.gateScales, l.gateScalesBytes, u8(rGate.scales)],
      ["upQs", l.upQs, l.qsBytes, u8(rUp.qs)],
      ["upScales", l.upScales, l.gateScalesBytes, u8(rUp.scales)],
      ["downQs", l.downQs, l.qsBytes, u8(rDown.qs)],
      ["downScales", l.downScales, l.downScalesBytes, u8(rDown.scales)],
    ];
    for (const [name, off, size, ref] of segs) {
      expect(`${name}:${firstMismatch(slab.subarray(off, off + size), ref)}`).toBe(`${name}:-1`);
    }
  });

  it("rifiuta byte di taglia sbagliata (validazione hard)", () => {
    const ok = randBytes(blocks * 18, 4);
    expect(() => packExpertSlab(ok, ok, randBytes(blocks * 18, 5), SLAB_DOWN_Q4_1)).toThrow();
    expect(() => packExpertSlab(ok.subarray(1), ok, randBytes(blocks * 20, 6), SLAB_DOWN_Q4_1)).toThrow();
  });
});
