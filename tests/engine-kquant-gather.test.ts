// Modo GATHER dei GEMV K-quant (goal engine-velocita-decode, riga 2).
// Test SENZA GPU sul testo WGSL generato.
//
// LA TESI CHE QUESTO TEST DIFENDE, ed e' la ragione per cui la riga 2 e'
// abbordabile: **il gather non tocca l'aritmetica.** Cambia da dove arriva `x`
// (riga raccolta dal buffer `gather` invece della riga zero) e dove va `y`
// (slot `(m*nUsed + kslot)` invece della riga) — nient'altro. L'unpack dei
// superblocchi, l'offset, il termine `Σx` e la riduzione sono le stesse righe
// verificate su GPU vera dai casi ktest di `engine-kquant`.
//
// Se e' vero, i casi ktest del gather devono validare l'INDIRIZZAMENTO e non
// ri-validare l'aritmetica — cioe' la riga 2 costa un cablaggio, non una
// riscrittura numerica. Il test lo rende una proprieta' verificata a ogni
// commit invece di un'affermazione nel journal.
//
// COME: si NORMALIZZA il sorgente gather riportandolo alla forma plain
// (si tolgono preambolo e offset, si riscrive la testa e la scrittura di y) e
// si pretende **uguaglianza esatta** col plain. Un'asserzione per campione
// ("contiene ancora `nibLo`") lascerebbe passare una modifica dentro il corpo;
// l'uguaglianza dopo normalizzazione no.
import { describe, expect, it } from "vitest";
import { gemvQ4KWgsl, gemvQ6KWgsl } from "../src/engine/kernels/wgsl";

// shape vere del 35B (header dump 2026-08-10): dModel 2048, dFfnExpert 512
const GATE_UP = { K: 2048, N: 512 };
const DOWN = { K: 512, N: 2048 };
const N_USED = 8; // il 35B e' top-8; il GLM top-4

/** riporta il sorgente gather alla forma plain, punto di variazione per punto */
function unGather(src: string, o: { K: number; N: number; nUsed: number }): string {
  return src
    // testa: `gather` sparisce e `y` torna al binding 2
    .replace("@group(0) @binding(2) var<storage, read> gather: array<u32>;\n", "")
    .replace(
      "@group(0) @binding(3) var<storage, read_write> y: array<f32>;",
      "@group(0) @binding(2) var<storage, read_write> y: array<f32>;")
    // preambolo del gather
    .replace(`  let gk = gather[wid.z];\n  let mRow = gk & 0xffffu;\n  let kslot = gk >> 16u;\n  let xR = mRow * ${o.K}u;\n`, "")
    // offset di riga sulle letture di x
    .replace(/x\[xR \+ /g, "x[")
    // scrittura: dallo slot alla riga
    .replace(`y[(mRow * ${o.nUsed}u + kslot) * ${o.N}u + r]`, "y[r]");
}

const FAMS = [
  { name: "q4_K", gen: gemvQ4KWgsl },
  { name: "q6_K", gen: gemvQ6KWgsl },
] as const;

describe("GEMV K-quant — il modo gather non tocca l'aritmetica", () => {
  for (const { name, gen } of FAMS) {
    for (const [shape, dims] of [["gate/up", GATE_UP], ["down", DOWN]] as const) {
      it(`[1] ${name} ${shape}: normalizzato, il gather E' il plain`, () => {
        const plain = gen({ ...dims });
        const gath = gen({ ...dims, gather: { nUsed: N_USED } });
        expect(gath).not.toBe(plain); // altrimenti il modo non fa niente
        expect(unGather(gath, { ...dims, nUsed: N_USED })).toBe(plain);
      });
    }

    it(`[2] ${name}: senza gather il testo e' quello di prima, byte per byte`, () => {
      // non-regressione del path in produzione: chi non passa `gather` deve
      // ottenere ESATTAMENTE il kernel di ieri
      expect(gen({ ...GATE_UP, gather: undefined })).toBe(gen({ ...GATE_UP }));
    });

    it(`[3] ${name}: lo stride degli slot segue nUsed`, () => {
      for (const nUsed of [4, 8]) {
        const src = gen({ ...GATE_UP, gather: { nUsed } });
        expect(src).toContain(`(mRow * ${nUsed}u + kslot) * ${GATE_UP.N}u + r`);
      }
    });

    it(`[4] ${name}: le combinazioni che non esistono lanciano, con la ragione`, () => {
      const arena = { nBuf: 2, slabWords: 1024, slabsPerBuf: 8, tensorWords: 128 };
      // gather + arena: nel gather l'expert lo fissa il dispatch, non la Sel
      expect(() => gen({ ...GATE_UP, arena, gather: { nUsed: 8 } })).toThrow(/gather e arena/);
      // gather + batch: si contenderebbero wid.z
      expect(() => gen({ ...GATE_UP, batch: true, gather: { nUsed: 8 } })).toThrow(/gather e batch/);
      // gather + accum: il contratto a slot vuole la scrittura NON pesata
      expect(() => gen({ ...GATE_UP, arena, accum: true, gather: { nUsed: 8 } })).toThrow(/gather e (arena|accum)/);
      for (const bad of [0, -1, 2.5]) {
        expect(() => gen({ ...GATE_UP, gather: { nUsed: bad } })).toThrow(/nUsed/);
      }
    });

    it(`[5] ${name}: la scrittura del gather NON e' pesata`, () => {
      // il contratto a slot esiste per la bit-identita': se qui comparisse un
      // `sel.w *`, la somma dei k passerebbe da moeCombine al kernel e
      // l'ordine delle somme f32 non sarebbe piu' quello del decode
      const src = gen({ ...DOWN, gather: { nUsed: N_USED } });
      const write = src.split("\n").find((l) => l.includes("kslot) *"));
      expect(write, "riga di scrittura dello slot non trovata").toBeDefined();
      expect(write!).not.toContain("sel.w");
      expect(write!).toContain("= partial[0];");
    });
  }

  it("[6] i binding del modo gather sono quelli dichiarati: [0 blocks, 1 x, 2 gather, 3 y]", () => {
    // l'ordine e' congelato nel commento della funzione: chi costruisce il bind
    // group lo legge da li'. Se cambia qui e non li', il bind group punta al
    // buffer sbagliato e il kernel legge pesi come indici.
    const src = gemvQ4KWgsl({ ...GATE_UP, gather: { nUsed: N_USED } });
    const binds = [...src.matchAll(/@binding\((\d+)\) var<storage, (read|read_write)> (\w+)/g)]
      .map((m) => `${m[1]}:${m[3]}`);
    expect(binds.slice(0, 4)).toEqual(["0:blocks", "1:x", "2:gather", "3:y"]);
  });
});
