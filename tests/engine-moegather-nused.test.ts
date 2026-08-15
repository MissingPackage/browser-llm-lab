// La famiglia a gather MoE e' parametrica sul top-K (goal engine-velocita-decode,
// riga 2). Test SENZA GPU: si genera il WGSL e si controlla lo STRIDE degli slot.
//
// PERCHE'. I tre kernel della famiglia indirizzano lo stesso array di slot:
// la riga `m` occupa `nUsed` posizioni consecutive e `kslot` sceglie quale.
// Lo stride era scritto `4u` A MANO in tre punti (GLM e' top-4) mentre
// `moeCombineWgsl` era gia' parametrico su `nUsed`. Il 35B e' top-8.
//
// IL MODO IN CUI QUESTO SI ROMPE E' SILENZIOSO. Un disaccordo di stride fra i
// tre kernel non lancia e non fa fallire la compilazione: legge e scrive slot
// sbagliati, e produce numeri plausibili. Con top-4 e stride 8 la combine
// leggerebbe gli slot delle righe successive — nessun accesso fuori range,
// nessun NaN, solo il modello sbagliato. E' esattamente la classe di difetto
// che i gate numerici a tolleranza lasciano passare.
//
// COSA NON PROVA: che i kernel calcolino il valore giusto. L'aritmetica ha i
// suoi casi ktest su GPU vera. Qui si prova l'INDIRIZZAMENTO, che e' la parte
// che la parametrizzazione puo' rompere.
import { describe, expect, it } from "vitest";
import {
  pairGemvSiluGatherWgsl, gemvDownSlotsWgsl, moeCombineWgsl,
} from "../src/engine/kernels/wgsl";

/** occorrenze non sovrapposte di `needle` in `hay` */
const count = (hay: string, needle: string): number => hay.split(needle).length - 1;

// shape GLM (top-4) e 35B (top-8), dai due modelli veri
const GLM = { K: 2048, N: 1408, nUsed: 4 };
const Q35 = { K: 2048, N: 512, nUsed: 8 };

describe("gather MoE — lo stride degli slot segue nUsed", () => {
  it("[1] il default resta 4: il GLM non cambia una riga di WGSL", () => {
    // Il default e' la garanzia di non-regressione per la famiglia gia' in
    // produzione: chi non passa `nUsed` deve ottenere IL TESTO DI PRIMA.
    const pairD = pairGemvSiluGatherWgsl({ K: GLM.K, N: GLM.N });
    const pair4 = pairGemvSiluGatherWgsl({ K: GLM.K, N: GLM.N, nUsed: 4 });
    expect(pairD).toBe(pair4);
    const downD = gemvDownSlotsWgsl({ kind: "q4_0", K: GLM.N, N: GLM.K });
    const down4 = gemvDownSlotsWgsl({ kind: "q4_0", K: GLM.N, N: GLM.K, nUsed: 4 });
    expect(downD).toBe(down4);
    // e lo stride scritto e' davvero 4, non un altro numero che coincide
    expect(pairD).toContain("m * 4u + kslot");
    expect(downD).toContain("m * 4u + kslot");
  });

  it("[2] a nUsed = 8 sparisce OGNI stride 4 dai due kernel a gather", () => {
    const pair = pairGemvSiluGatherWgsl({ K: Q35.K, N: Q35.N, nUsed: 8 });
    const down = gemvDownSlotsWgsl({ kind: "q4_0", K: Q35.N, N: Q35.K, nUsed: 8 });
    for (const [name, src] of [["pair", pair], ["down", down]] as const) {
      expect(count(src, "m * 8u + kslot"), `${name}: stride 8 assente`).toBeGreaterThan(0);
      expect(count(src, "m * 4u + kslot"), `${name}: e' rimasto uno stride 4 cablato`).toBe(0);
    }
    // il pair ne ha uno (scrittura hSlots), il down due (lettura hSlots +
    // scrittura ySlots): se il conteggio cambia, un sito nuovo e' comparso e
    // va parametrizzato anche lui invece di restare indietro in silenzio
    expect(count(pair, "m * 8u + kslot")).toBe(1);
    expect(count(down, "m * 8u + kslot")).toBe(2);
  });

  it("[3] i TRE kernel concordano sullo stride — il disaccordo e' silenzioso", () => {
    for (const nUsed of [4, 8]) {
      const pair = pairGemvSiluGatherWgsl({ K: Q35.K, N: Q35.N, nUsed });
      const down = gemvDownSlotsWgsl({ kind: "q4_0", K: Q35.N, N: Q35.K, nUsed });
      const comb = moeCombineWgsl({ D: Q35.K, nUsed });
      expect(pair).toContain(`m * ${nUsed}u + kslot`);
      expect(down).toContain(`m * ${nUsed}u + kslot`);
      // la combine usa la costante N_USED al posto del letterale
      expect(comb).toContain(`const N_USED = ${nUsed}u;`);
      expect(comb).toContain("m * N_USED + k");
    }
  });

  it("[4] il default della combine e quello dei gather sono lo STESSO numero", () => {
    // Tre default scritti in tre punti diversi possono divergere con una
    // modifica innocente. Qui si legano: se qualcuno cambia il default di uno
    // solo, questo test cade.
    const comb = moeCombineWgsl({ D: 64 });
    const pair = pairGemvSiluGatherWgsl({ K: 64, N: 64 });
    const m = /const N_USED = (\d+)u;/.exec(comb);
    expect(m, "N_USED non trovato nella combine").not.toBeNull();
    expect(pair).toContain(`m * ${m![1]}u + kslot`);
  });

  it("[5] un nUsed assurdo lancia invece di generare WGSL storto", () => {
    for (const bad of [0, -1, 2.5, NaN]) {
      expect(() => pairGemvSiluGatherWgsl({ K: 64, N: 64, nUsed: bad })).toThrow(/nUsed/);
      expect(() => gemvDownSlotsWgsl({ kind: "q4_0", K: 64, N: 64, nUsed: bad })).toThrow(/nUsed/);
    }
  });
});
