import { describe, it, expect } from "vitest";
import {
  MLA_CHUNK_P, MLA_TILE_W, MLA_TILE_STRIDE, mlaSMax, mlaPartialsLen, mlaActiveParts,
  mlaPartRange, mlaSplitWorkgroupStorageBytes, mlaSplitPartWorkgroupStorageBytes,
  MLA_SPLIT_REDUCE_WORKGROUP_STORAGE_BYTES,
} from "../src/engine/mlasplit";
import { mlaAttnSplitPartWgsl, mlaAttnSplitReduceWgsl } from "../src/engine/kernels/wgsl";
import { GLM47_FLASH as G } from "../src/engine/shape";

// Unit CPU-side sul piano dell'attention MLA split (C3a fase 4c) — convenzione
// CI-senza-GPU di attnsplit: geometria delle partizioni + il legame fra le
// costanti del modulo e il WGSL VERO che ne discende. La matematica dei kernel
// la verificano i ktest (mla-attn-split*, contro riferimento f64 e contro il
// monolitico).

const KERNEL_OPTS = {
  nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax: 525,
  scale: 1 / Math.sqrt(G.headLenMla), chunk: MLA_CHUNK_P,
};

describe("geometria partizioni MLA", () => {
  it("sMax e partials per la shape del motore (20 head, kvLora 512)", () => {
    expect(mlaSMax(512)).toBe(32);
    expect(mlaSMax(525)).toBe(33); // ctxMax non multiplo del chunk: si arrotonda su
    expect(mlaPartialsLen(G.nHead, G.kvLora, 525)).toBe(20 * 33 * 514);
  });

  it("partizioni attive: n=pos+1 coperto esattamente, senza buchi né overlap", () => {
    for (const pos of [0, 1, 7, 15, 16, 40, 200, 524]) {
      const n = pos + 1;
      const nParts = mlaActiveParts(pos);
      expect(nParts).toBe(Math.ceil(n / MLA_CHUNK_P));
      let next = 0;
      for (let p = 0; p < nParts; p++) {
        const { begin, end } = mlaPartRange(p, n);
        expect(begin).toBe(next);
        expect(end).toBeGreaterThan(begin); // ogni partizione attiva è non-vuota
        next = end;
      }
      expect(next).toBe(n);
    }
  });

  it("le partizioni oltre le attive escono subito (begin >= n)", () => {
    for (const pos of [0, 15, 16, 200]) {
      const n = pos + 1;
      for (let p = mlaActiveParts(pos); p < mlaSMax(525); p++) {
        expect(mlaPartRange(p, n).begin).toBeGreaterThanOrEqual(n);
      }
    }
  });
});

describe("il modulo e il WGSL che ne discende non si scollano", () => {
  const part = mlaAttnSplitPartWgsl(KERNEL_OPTS);
  const reduce = mlaAttnSplitReduceWgsl({ nHead: G.nHead, kvLora: G.kvLora, ctxMax: 525, chunk: MLA_CHUNK_P });
  const constOf = (src: string, name: string): number => {
    const m = src.match(new RegExp(`const ${name} = (\\d+)u;`));
    if (!m) throw new Error(`costante ${name} non trovata nel WGSL`);
    return Number(m[1]);
  };

  // somma dei var<workgroup> dichiarati da un kernel generato: array<f32, N> e
  // scalari f32. È la ri-derivazione che tiene insieme mlasplit.ts e il WGSL —
  // se qualcuno cambia un tile in WGSL senza toccare il modulo, i test cadono.
  const wgBytes = (src: string): number => {
    let bytes = 0;
    for (const m of src.matchAll(/var<workgroup>\s+\w+:\s*array<f32,\s*(\d+)>/g)) bytes += 4 * Number(m[1]);
    for (const m of src.matchAll(/var<workgroup>\s+\w+:\s*f32\s*;/g)) { void m; bytes += 4; }
    return bytes;
  };

  it("workgroup storage: le formule sono la somma dei var<workgroup> dei DUE kernel", () => {
    expect(wgBytes(part)).toBe(mlaSplitPartWorkgroupStorageBytes(G.nHead));
    expect(wgBytes(part)).toBe(10_800);
    expect(wgBytes(reduce)).toBe(MLA_SPLIT_REDUCE_WORKGROUP_STORAGE_BYTES);
    expect(wgBytes(reduce)).toBe(8); // solo M e L: nessun array [sMax] di pesi
    // il fabbisogno dichiarato al device è il massimo dei due (domina il part)
    expect(mlaSplitWorkgroupStorageBytes(G.nHead)).toBe(wgBytes(part));
  });

  it("il fabbisogno di ENTRAMBI i pass è costante in ctxMax", () => {
    // è questo che toglie il vincolo sul contesto che aveva il monolitico
    // (scores[ctxMax] in shared). Verificarlo sul solo pass 1 sarebbe vacuo: il
    // pass 2 indicizza le partizioni, ed è lì che un array [sMax] tornerebbe a
    // crescere col contesto senza che nessuno se ne accorga.
    for (const ctxMax of [512, 525, 65_536, 200_000]) {
      expect(wgBytes(mlaAttnSplitPartWgsl({ ...KERNEL_OPTS, ctxMax })))
        .toBe(mlaSplitPartWorkgroupStorageBytes(G.nHead));
      expect(wgBytes(mlaAttnSplitReduceWgsl({ nHead: G.nHead, kvLora: G.kvLora, ctxMax, chunk: MLA_CHUNK_P })))
        .toBe(MLA_SPLIT_REDUCE_WORKGROUP_STORAGE_BYTES);
    }
    // e il massimo sta sotto il default di spec WebGPU (16 KiB) a ogni contesto:
    // nessun limite da negoziare, a differenza del monolitico
    expect(mlaSplitWorkgroupStorageBytes(G.nHead)).toBeLessThan(16_384);
  });

  it("le costanti del kernel sono quelle del modulo", () => {
    expect(constOf(part, "CHUNK")).toBe(MLA_CHUNK_P);
    expect(constOf(part, "TW")).toBe(MLA_TILE_W);
    expect(constOf(part, "TWP")).toBe(MLA_TILE_STRIDE);
    expect(constOf(part, "S_MAX")).toBe(mlaSMax(525));
    expect(constOf(part, "PART_STRIDE")).toBe(G.kvLora + 2);
    expect(constOf(reduce, "CHUNK")).toBe(MLA_CHUNK_P);
    expect(constOf(reduce, "S_MAX")).toBe(mlaSMax(525));
    expect(constOf(reduce, "PART_STRIDE")).toBe(G.kvLora + 2);
  });

  it("il buffer partials copre l'indice massimo che il kernel scrive", () => {
    // il kernel indirizza (h*S_MAX + part)*PART_STRIDE + kvLora + 1
    const sMax = mlaSMax(525);
    const last = ((G.nHead - 1) * sMax + (sMax - 1)) * (G.kvLora + 2) + G.kvLora + 1;
    expect(last).toBeLessThan(mlaPartialsLen(G.nHead, G.kvLora, 525));
    expect(last).toBe(mlaPartialsLen(G.nHead, G.kvLora, 525) - 1);
  });

  it("il pass 1 esce PRIMA di ogni barrier (return uniforme per workgroup)", () => {
    // WGSL: un workgroupBarrier raggiunto solo da parte del workgroup è UB.
    // Il return delle partizioni oltre il contesto deve precedere il primo.
    const iReturn = part.indexOf("if (begin >= n) { return; }");
    const iBarrier = part.indexOf("workgroupBarrier()");
    expect(iReturn).toBeGreaterThan(0);
    expect(iReturn).toBeLessThan(iBarrier);
    expect(part.lastIndexOf("return;")).toBe(part.indexOf("return;")); // un solo return
  });

  it("il chunk è scelto per occupancy: al contesto del bench riempie la GPU", () => {
    // 33 workgroup a pos 524 contro i 20 del monolitico (una head per wg): è
    // la ragione per cui il chunk NON è 64 (sarebbero 9).
    expect(mlaActiveParts(524)).toBe(33);
    expect(mlaActiveParts(524)).toBeGreaterThan(G.nHead);
  });
});
