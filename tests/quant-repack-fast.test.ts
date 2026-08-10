// Il repack K-quant veloce dà gli STESSI BYTE di quello scalare (goal fase-D
// fase 2). La forma scalare — `out[j>>2] |= src[j] << (j&3)*8` — è la
// DEFINIZIONE del risultato; qui è riscritta come oracolo indipendente e
// confrontata con l'implementazione, che su little-endian fa una memcpy.
//
// Perché il test conta: la memcpy è corretta solo se la macchina è LE e solo
// se il padding a parola resta a ZERO. Sbagliare la seconda cosa non
// darebbe un errore, darebbe pesi Q6_K leggermente diversi — il tipo di bug
// che si vede solo come qualità che cala.
import { describe, expect, it } from "vitest";
import { repackKQuant } from "../src/engine/quant";

const scalare = (src: Uint8Array, off: number, nBlocks: number, bb: number): Uint32Array => {
  const wpb = Math.ceil(bb / 4);
  const out = new Uint32Array(nBlocks * wpb);
  for (let b = 0; b < nBlocks; b++) {
    const o = off + b * bb;
    for (let j = 0; j < bb; j++) out[b * wpb + (j >> 2)] |= src[o + j] << ((j & 3) * 8);
  }
  return out;
};

const synth = (n: number, s0: number): Uint8Array => {
  const o = new Uint8Array(n); let s = s0 >>> 0;
  for (let i = 0; i < n; i++) { s = (s * 1664525 + 1013904223) >>> 0; o[i] = s >>> 24; }
  return o;
};

describe("repackKQuant veloce == scalare", () => {
  // 144 = Q4_K e 176 = Q5_K (multipli di 4, nessun padding); 210 = Q6_K
  // (padding a 212: due byte a zero per superblocco)
  it.each([144, 176, 210])("blockBytes %i: parola per parola", (bb) => {
      const nB = 37; // primo: sbaglia chi assume blocchi allineati a potenze di 2
      const src = synth(nB * bb + 13, bb);
      for (const off of [0, 13]) {
        const atteso = scalare(src, off, nB, bb);
        const ottenuto = repackKQuant(src, off, nB, bb);
        expect(ottenuto.length).toBe(atteso.length);
        let primo = -1;
        for (let i = 0; i < atteso.length; i++) if (atteso[i] !== ottenuto[i]) { primo = i; break; }
        expect(primo, `prima parola diversa: indice ${primo} (offset src ${off})`).toBe(-1);
      }
    });

  it("Q6_K: i due byte di padding di OGNI superblocco sono zero", () => {
    const nB = 9;
    const src = synth(nB * 210, 7).fill(0xff); // sorgente tutta a 1: il padding deve restare 0
    const out = repackKQuant(src, 0, nB, 210);
    const bytes = new Uint8Array(out.buffer);
    for (let b = 0; b < nB; b++) {
      expect(bytes[b * 212 + 210], `superblocco ${b}, byte 210`).toBe(0);
      expect(bytes[b * 212 + 211], `superblocco ${b}, byte 211`).toBe(0);
    }
    expect(bytes[0]).toBe(0xff); // ...e i dati veri ci sono
  });
});
