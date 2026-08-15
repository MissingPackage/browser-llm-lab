// Modo KFAN dei GEMV K-quant (goal engine-velocita-decode, riga 2c).
// Test SENZA GPU sul testo WGSL generato.
//
// COSA E' IL KFAN. Nel decode il modello sceglie 8 expert per token e oggi il
// motore li dispaccia UNO PER UNO: (8 expert x 4 operazioni) + 1 somma, per 40
// layer = **1.320 dispatch per token**, misurati in 40,753 ms — ~31 us l'uno,
// **13,9 GB/s effettivi** sui 566 MB di pesi letti, contro gli 0,98 ms che
// quegli stessi byte costerebbero a 576 GB/s. Il decode e' 41x sopra il
// pavimento di banda perche' e' dispatch/occupancy-bound, non perche' manchi
// banda. Il kfan mette il k su `wid.z`: **1.320 -> 200 dispatch**.
//
// LE DUE PROPRIETA' CHE QUESTO TEST DIFENDE:
//
// 1. **L'aritmetica non cambia.** Come per il gather (it.4), si NORMALIZZA il
//    sorgente kfan riportandolo alla forma accumulante e si pretende
//    uguaglianza ESATTA. Le uniche differenze ammesse sono tre righe, ed e' il
//    test a elencarle: se ne compare una quarta, cade.
//
// 2. **Il miss scrive ZERO, non esce.** E' la differenza che si paga cara e in
//    silenzio. Il ramo d'arena normale fa `if (!ok) { return; }`, giusto quando
//    il contributo si ACCUMULA (un expert mancante somma zero). Con gli slot
//    uscire lascerebbe lo slot col valore del TOKEN PRECEDENTE, e `moeCombine`
//    lo sommerebbe: nessun crash, nessun NaN, il modello sbagliato solo sui
//    token che hanno avuto un miss. Il kfan scrive `select(0.0, partial[0], ok)`.
import { describe, expect, it } from "vitest";
import { gemvQ4KWgsl, gemvQ6KWgsl } from "../src/engine/kernels/wgsl";

const ARENA = { nBuf: 2, slabWords: 1024, slabsPerBuf: 8, tensorWords: 128 };
// shape vere del 35B: gate/up [2048,512], down [512,2048]; top-8
const GATE_UP = { K: 2048, N: 512 };
const DOWN = { K: 512, N: 2048 };
const N_USED = 8;

const FAMS = [
  { name: "q4_K", gen: gemvQ4KWgsl },
  { name: "q6_K", gen: gemvQ6KWgsl },
] as const;

/** riporta il sorgente kfan alla forma accumulante, differenza per differenza */
function unKfan(src: string, N: number): string {
  return src
    // (a) l'indice di Sel: da «il k su wid.z» al k fissato dall'offset dinamico
    .replace("let sel = selBuf[moeIdx.selIdx + wid.z];", "let sel = selBuf[moeIdx.selIdx];")
    // (b) l'uscita anticipata sul miss, che il kfan non puo' avere
    .replace(
      `  gBase = base + ${ARENA.tensorWords}u;\n`,
      `  gBase = base + ${ARENA.tensorWords}u;\n  if (!ok) { return; }\n`)
    // (c) la coda: e' la differenza SEMANTICA voluta (slot non pesato contro
    //     accumulo pesato), non un semplice ri-indirizzamento
    .replace(
      `y[wid.z * ${N}u + r] = select(0.0, partial[0], ok);`,
      "y[r] = y[r] + sel.w * partial[0];");
}

describe("GEMV K-quant — il modo kfan non tocca l'aritmetica", () => {
  for (const { name, gen } of FAMS) {
    for (const [shape, dims] of [["gate/up", GATE_UP], ["down", DOWN]] as const) {
      it(`[1] ${name} ${shape}: normalizzato, il kfan E' il kernel accumulante`, () => {
        const acc = gen({ ...dims, arena: ARENA, accum: true });
        const kfan = gen({ ...dims, arena: ARENA, kfan: { nUsed: N_USED } });
        expect(kfan).not.toBe(acc);
        // se resta una sola riga di differenza oltre le tre dichiarate, cade
        expect(unKfan(kfan, dims.N)).toBe(acc);
      });
    }

    it(`[2] ${name}: senza kfan il testo e' quello di prima, byte per byte`, () => {
      // non-regressione del decode in produzione
      expect(gen({ ...GATE_UP, arena: ARENA, accum: true, kfan: undefined }))
        .toBe(gen({ ...GATE_UP, arena: ARENA, accum: true }));
      expect(gen({ ...GATE_UP })).toBe(gen({ ...GATE_UP, kfan: undefined }));
    });

    it(`[3] ${name}: sul MISS lo slot riceve zero, e il kernel NON esce`, () => {
      const kfan = gen({ ...DOWN, arena: ARENA, kfan: { nUsed: N_USED } });
      // l'uscita anticipata lascerebbe lo slot col valore del token precedente
      expect(kfan).not.toContain("if (!ok) { return; }");
      // e il contributo nullo e' scritto, non omesso
      expect(kfan).toContain("select(0.0, partial[0], ok)");
      // `ok` resta definito: e' il degrado DICHIARATO, non un indirizzo a caso
      expect(kfan).toContain("let ok = sel.slot != 0xffffffffu;");
    });

    it(`[4] ${name}: il k viene da wid.z, e le Sel del layer sono contigue`, () => {
      const kfan = gen({ ...GATE_UP, arena: ARENA, kfan: { nUsed: N_USED } });
      // `selIdx = (row*nMoeLayer + m)*topK + k` (q35gpumodel.ts:1377): le topK
      // entry di un (riga, layer) sono adiacenti, quindi + wid.z le percorre
      expect(kfan).toContain("selBuf[moeIdx.selIdx + wid.z]");
      // la scrittura va nello slot wid.z, che e' il layout che moeCombine legge
      expect(kfan).toContain(`y[wid.z * ${GATE_UP.N}u + r]`);
    });

    it(`[5] ${name}: la scrittura del kfan NON e' pesata`, () => {
      // se pesasse, la somma dei k passerebbe dal moeCombine al kernel e
      // l'ordine delle somme f32 non sarebbe piu' quello del decode: salta la
      // bit-identita', che qui e' pretesa esatta e non a tolleranza
      const kfan = gen({ ...DOWN, arena: ARENA, kfan: { nUsed: N_USED } });
      const write = kfan.split("\n").find((l) => l.includes("wid.z *") && l.includes("select("));
      expect(write, "riga di scrittura dello slot non trovata").toBeDefined();
      expect(write!).not.toContain("sel.w");
    });

    it(`[6] ${name}: le combinazioni che non esistono lanciano, con la ragione`, () => {
      // kfan senza arena: non c'e' nessuna Sel da indicizzare
      expect(() => gen({ ...GATE_UP, kfan: { nUsed: 8 } })).toThrow(/kfan esige il regime d'arena/);
      // kfan + batch: si contenderebbero wid.z
      expect(() => gen({ ...GATE_UP, arena: ARENA, batch: true, kfan: { nUsed: 8 } }))
        .toThrow(/batch e arena|kfan e batch/);
      // kfan + accum: 8 k nello stesso dispatch che accumulano su y[r] corrono
      expect(() => gen({ ...GATE_UP, arena: ARENA, accum: true, kfan: { nUsed: 8 } }))
        .toThrow(/kfan e accum/);
      // kfan + gather: righe l'uno, k l'altro, stesso wid.z
      expect(() => gen({ ...GATE_UP, gather: { nUsed: 4 }, kfan: { nUsed: 8 } }))
        .toThrow(/gather e kfan/);
      for (const bad of [0, -1, 2.5]) {
        expect(() => gen({ ...GATE_UP, arena: ARENA, kfan: { nUsed: bad } })).toThrow(/nUsed/);
      }
    });
  }

  it("[7] kfan e gather restano due modi distinti, non due nomi dello stesso", () => {
    // il gather e' PLAIN (l'expert lo fissa il dispatch) e mette le RIGHE su
    // wid.z; il kfan e' d'ARENA (l'expert lo risolve Sel) e mette i K. Se un
    // giorno collassassero, questo test lo dice invece di lasciarlo scoprire a
    // un bind group che punta al buffer sbagliato.
    const gath = gemvQ4KWgsl({ ...GATE_UP, gather: { nUsed: 8 } });
    const kfan = gemvQ4KWgsl({ ...GATE_UP, arena: ARENA, kfan: { nUsed: 8 } });
    expect(gath).toContain("var<storage, read> gather: array<u32>;");
    expect(kfan).not.toContain("gather");
    expect(kfan).toContain("selBuf");
    expect(gath).not.toContain("selBuf");
  });
});

// ————————————————————————————————————————————————————————————————————————
// La combine che il kfan consuma (goal engine-velocita-decode, riga 2c).
//
// Il kfan scrive gli slot; qualcuno deve sommarli. Sul 35B i pesi di mixing
// stanno GIA' in VRAM dentro la `Sel` che il motore scrive per ogni layer:
// farli ricopiare in un `wBuf` a parte costerebbe una `writeBuffer` per layer
// nel ciclo caldo — 40 per token — per dati che sono gia' sul device.
// La variante `weightsFromSel` li legge da li'.
import { moeCombineWgsl } from "../src/engine/kernels/wgsl";

describe("moeCombine — la variante che legge i pesi dalla Sel", () => {
  it("[8] l'unica differenza aritmetica e' DA DOVE viene il peso", () => {
    const wbuf = moeCombineWgsl({ D: 2048, nUsed: 8 });
    const sel = moeCombineWgsl({ D: 2048, nUsed: 8, weightsFromSel: true });
    // normalizzata, la variante sel E' quella a wBuf
    const norm = sel
      .replace("struct Sel { id: u32, slot: u32, w: f32, flags: u32 }\nstruct MoeIdx { selIdx: u32, tableBase: u32, moeLayer: u32, pad: u32 }\n@group(0) @binding(3) var<storage, read> selBuf: array<Sel>;\n@group(0) @binding(4) var<uniform> moeIdx: MoeIdx;",
        "@group(0) @binding(3) var<storage, read> wBuf: array<f32>;")
      .replace("selBuf[moeIdx.selIdx + k].w", "wBuf[m * N_USED + k]");
    expect(norm).toBe(wbuf);
  });

  it("[9] il default e' invariato: il GLM continua a usare il wBuf esplicito", () => {
    expect(moeCombineWgsl({ D: 64, nUsed: 4 })).toBe(moeCombineWgsl({ D: 64, nUsed: 4, weightsFromSel: false }));
    expect(moeCombineWgsl({ D: 64, nUsed: 4 })).toContain("wBuf");
    expect(moeCombineWgsl({ D: 64, nUsed: 4 })).not.toContain("selBuf");
  });

  it("[10] l'ORDINE delle somme resta k crescente — e' il gate di bit-identita'", () => {
    // il decode di oggi (encodeExperts, `for k2` ascendente) somma i k in
    // quest'ordine: se la combine lo cambiasse, i near-tie dell'argmax
    // potrebbero flippare e la bit-identita' salterebbe
    for (const sel of [false, true]) {
      const src = moeCombineWgsl({ D: 2048, nUsed: 8, weightsFromSel: sel });
      expect(src).toContain("for (var k = 0u; k < N_USED; k = k + 1u)");
      expect(src).toContain("var t = sM[m * D + i];"); // si parte dallo shexp
    }
  });

  it("[11] la variante sel indicizza i topK CONTIGUI di un (riga, layer)", () => {
    const sel = moeCombineWgsl({ D: 2048, nUsed: 8, weightsFromSel: true });
    // stesso indirizzo che il kfan usa per la Sel: selIdx + k
    expect(sel).toContain("selBuf[moeIdx.selIdx + k].w");
    // ed e' M=1 per costruzione: chi dispaccia usa y = 1 (documentato)
    expect(sel).toContain("@group(0) @binding(4) var<uniform> moeIdx: MoeIdx;");
  });
});
