import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { q35KQuantGemvWgsl, q35KQuantKindOfGgml } from "../src/engine/q35gpumodel";
import {
  gemvQ2KWgsl, gemvQ3KWgsl, gemvQ4KWgsl, gemvQ5KWgsl, gemvQ6KWgsl,
  KQUANT_GEMV_DESC, type KArenaOpts, type KQuantGemvKind,
} from "../src/engine/kernels/wgsl";
import { GGML_TYPE, tensorByteSize, type GgmlTypeId } from "../src/engine/gguf";
import type { QuantKind } from "../src/engine/moe";

// L'INSTRADAMENTO DEI K-QUANT NEL RAMO EXPERT (spec 2026-08-17 §4 T3).
//
// Prima di questo task la scelta del kernel viveva in CATENE DI TERNARI
// ripetute in tre punti di `q35gpumodel` (`t.type === Q4_K ? gemvQ4KWgsl : …`,
// una per il gemv statico, una per il gemello batch, una per la pipeline
// d'arena). Aggiungere Q2_K e Q3_K avrebbe voluto dire allungare TRE catene:
// tre posti che tengono una decisione sola, cioe' la forma che questo repo ha
// gia' pagato (it.7).
//
// Qui si pinna il rimedio, e in DUE metà, perche' la decisione ha due capi:
//   - `q35KQuantKindOfGgml`: dal tipo GGUF letto nell'header al kind del
//     formato. E' il capo che il brief nomina per esteso — «il kind passato e'
//     quello VERO del tensore, mai un cast a un altro formato»;
//   - `q35KQuantGemvWgsl`: dal kind al testo del kernel, identico a quello dei
//     generatori congelati di `wgsl.ts`.
// Il primo capo NON si prova col secondo: kernel, `blockBytes` e kind del piano
// discendono TUTTI da quella lettura, quindi una mappa trasposta li sposta
// insieme e nessuna identita' interna se ne accorge. La prova sta nel caso
// [mappa], che confronta la taglia del superblocco con `tensorByteSize` —
// l'altra implementazione dello stesso fatto, quella che legge il file.
//
// Tutto CPU-side: si confrontano stringhe e numeri, nessun device WebGPU.

const SRC_PATH = join(__dirname, "..", "src/engine/q35gpumodel.ts");
const SRC = readFileSync(SRC_PATH, "utf8");

/** la shape vera di gate/up di un expert del 35B */
const GATE_UP = { K: 2048, N: 512 };
/** la shape vera del down (K e N scambiati) */
const DOWN = { K: 512, N: 2048 };
/** un'arena qualunque ma PLAUSIBILE: il testo emesso dipende dai suoi numeri */
const ARENA: KArenaOpts = { nBuf: 2, slabWords: 1024, slabsPerBuf: 8, tensorWords: 128 };

const GEN: Record<KQuantGemvKind, (o: { K: number; N: number; batch?: boolean }) => string> = {
  q2_K: gemvQ2KWgsl, q3_K: gemvQ3KWgsl, q4_K: gemvQ4KWgsl,
  q5_K: gemvQ5KWgsl, q6_K: gemvQ6KWgsl,
};
const KINDS = ["q2_K", "q3_K", "q4_K", "q5_K", "q6_K"] as const;

describe("q35KQuantGemvWgsl: un solo selettore per i cinque K-quant", () => {
  it("(a) per ogni formato emette il testo del generatore congelato, carattere per carattere", () => {
    for (const k of KINDS) {
      expect(q35KQuantGemvWgsl(k, GATE_UP), `${k}: il selettore diverge dal generatore`)
        .toBe(GEN[k](GATE_UP));
    }
  });

  it("(a-bis) i cinque testi sono DIVERSI fra loro: il selettore non collassa i formati", () => {
    // senza questo caso, un selettore che ritorna sempre il q4_K passerebbe
    // (a) per il q4_K e nessuno leggerebbe gli altri quattro fallimenti
    const seen = new Set(KINDS.map((k) => q35KQuantGemvWgsl(k, GATE_UP)));
    expect(seen.size).toBe(KINDS.length);
  });

  it("(b) stessa identita' sulla variante batch, per tutti e cinque", () => {
    for (const k of KINDS) {
      expect(q35KQuantGemvWgsl(k, { ...GATE_UP, batch: true }), `${k} batch`)
        .toBe(GEN[k]({ ...GATE_UP, batch: true }));
    }
  });

  it("(b-bis) q2_K e q3_K in regime d'arena, col down che ACCUMULA", () => {
    // e' la forma esatta della pipeline expert: gate/up in arena semplice, il
    // down in arena con `accum` (il peso arriva da `Sel`, niente axpy dopo)
    for (const [k, gen] of [["q2_K", gemvQ2KWgsl], ["q3_K", gemvQ3KWgsl]] as const) {
      expect(q35KQuantGemvWgsl(k, { ...GATE_UP, arena: ARENA }), `${k} arena gate/up`)
        .toBe(gen({ ...GATE_UP, arena: ARENA }));
      expect(q35KQuantGemvWgsl(k, { ...DOWN, arena: ARENA, accum: true }), `${k} arena accum down`)
        .toBe(gen({ ...DOWN, arena: ARENA, accum: true }));
    }
  });

  it("(c) un formato non K-quant LANCIA, e l'errore nomina il formato rifiutato", () => {
    // Non basta «lancia»: un selettore che lancia sempre passerebbe. Si
    // pretende (1) che il messaggio nomini il kind arrivato — senza, chi legge
    // il crash non sa quale tensore ha sbagliato ramo — e (2) che i cinque
    // K-quant nello STESSO giro non lancino (il caso (a) usa solo la forma
    // base; qui si copre anche l'arena, dove un rifiuto di troppo spegnerebbe
    // gli expert invece dei pesi statici).
    for (const bad of ["q4_0", "q4_1", "q8_0"] as QuantKind[]) {
      expect(() => q35KQuantGemvWgsl(bad, GATE_UP), `${bad} doveva lanciare`)
        .toThrow(new RegExp(bad));
    }
    for (const k of KINDS.filter((x) => x !== "q5_K")) {
      expect(() => q35KQuantGemvWgsl(k, { ...GATE_UP, arena: ARENA }), `${k} arena non deve lanciare`)
        .not.toThrow();
    }
  });

  it("(c-bis) il q5_K rifiuta i modi che la sua firma CONGELATA non ha", () => {
    // `gemvQ5KWgsl` e' `{K, N, batch}` e basta: `KQUANT_GEMV_DESC.q5_K.modes`
    // vale `batchOnly`. Passargli un'arena sarebbe un no-op SILENZIOSO — il
    // kernel emesso leggerebbe `blocks` invece dell'arena, cioe' un binding che
    // in regime d'arena non esiste. Meglio l'eccezione al momento di emettere.
    expect(KQUANT_GEMV_DESC.q5_K.modes, "se il q5_K guadagna i modi pieni, questo caso va riscritto")
      .toBe("batchOnly");
    expect(() => q35KQuantGemvWgsl("q5_K", { ...GATE_UP, arena: ARENA })).toThrow(/q5_K/);
    expect(() => q35KQuantGemvWgsl("q5_K", { ...DOWN, accum: true })).toThrow(/q5_K/);
  });
});

describe("[mappa] il kind e' quello VERO del tensore letto dall'header", () => {
  /** le cinque coppie, scritte a mano: sono il contratto, non una derivazione */
  const COPPIE: [GgmlTypeId, KQuantGemvKind][] = [
    [GGML_TYPE.Q2_K, "q2_K"], [GGML_TYPE.Q3_K, "q3_K"], [GGML_TYPE.Q4_K, "q4_K"],
    [GGML_TYPE.Q5_K, "q5_K"], [GGML_TYPE.Q6_K, "q6_K"],
  ];

  it("ogni tipo GGUF K-quant si mappa sul PROPRIO kind", () => {
    for (const [type, kind] of COPPIE) {
      expect(q35KQuantKindOfGgml(type), `GGML_TYPE ${type}`).toBe(kind);
    }
  });

  it("i tipi che non sono K-quant danno `undefined`, non un kind a caso", () => {
    // `undefined` e' il segnale che manda `loadW` sui rami legacy: se qui
    // tornasse un kind, un q8_0 verrebbe letto a superblocchi da 256
    for (const t of [GGML_TYPE.F32, GGML_TYPE.F16, GGML_TYPE.Q4_0, GGML_TYPE.Q4_1, GGML_TYPE.Q8_0]) {
      expect(q35KQuantKindOfGgml(t), `GGML_TYPE ${t}`).toBeUndefined();
    }
  });

  it("il superblocco del kind mappato coincide con quello che il LETTORE usa per quel tipo", () => {
    // LA PROVA CHE UCCIDE LA MAPPA TRASPOSTA. Kernel, `blockBytes` e kind del
    // piano nascono tutti dalla stessa lettura: scambiare due righe della mappa
    // li sposta INSIEME, e nessun confronto interno al selettore se ne accorge.
    // `tensorByteSize` e' l'altra implementazione dello stesso fatto — quella
    // che decide quanti byte leggere dal file — e non passa dalla mappa. Se le
    // due divergono, il motore legge un superblocco da 84 B col passo di uno da
    // 110: WGSL che compila, nessuna eccezione WebGPU, logit sbagliati.
    for (const [type] of COPPIE) {
      const kind = q35KQuantKindOfGgml(type)!;
      const unaRiga = tensorByteSize({ name: "t", dims: [256, 1], type, offset: 0 });
      expect(KQUANT_GEMV_DESC[kind].blockBytes, `GGML_TYPE ${type} -> ${kind}`).toBe(unaRiga);
    }
  });

  it("i cinque `blockBytes` sono 84/110/144/176/210, e li usa il caricatore", () => {
    const atteso: Record<KQuantGemvKind, number> = {
      q2_K: 84, q3_K: 110, q4_K: 144, q5_K: 176, q6_K: 210,
    };
    for (const k of KINDS) expect(KQUANT_GEMV_DESC[k].blockBytes, k).toBe(atteso[k]);
    // il repack del tensore statico NON ri-deriva la taglia a mano: la legge
    // dal descrittore, cosi' formato del kernel e passo del buffer non possono
    // separarsi
    expect(SRC, "blockBytes va derivato da KQUANT_GEMV_DESC, non da un ternario")
      .toMatch(/KQUANT_GEMV_DESC\[[A-Za-z0-9_.]+\]\.blockBytes/);
  });
});

describe("(d) la scelta del kernel avviene in UN posto solo", () => {
  it("nessuna catena di ternari sui generatori nel sorgente", () => {
    expect(SRC).not.toContain("GGML_TYPE.Q4_K ? gemvQ4KWgsl");
    expect(SRC, "ternario su GGML_TYPE che sceglie un gemv").not.toMatch(/GGML_TYPE\.\w+\s*\?\s*gemvQ/);
    expect(SRC, "ternario che sceglie un gemv K-quant").not.toMatch(/\?\s*gemvQ[2-6]KWgsl\s*\(/);
    expect(SRC, "ternario sul kind che sceglie un gemv").not.toMatch(/kind\s*===\s*["']q[2-6]_K["']\s*\?/);
  });

  it("ogni generatore K-quant e' invocato al massimo una volta, dentro il selettore", () => {
    for (const k of KINDS) {
      const name = `gemvQ${k[1]}KWgsl(`;
      const n = SRC.split(name).length - 1;
      expect(n, `${name} invocato ${n} volte: la scelta deve stare nel selettore`)
        .toBeLessThanOrEqual(1);
    }
    expect(SRC).toContain("export function q35KQuantGemvWgsl");
  });

  it("i siti expert chiedono il kernel al selettore, col kind del LAYOUT dello slab", () => {
    // gate/up/down non nominano piu' un formato: lo prendono da `L.<t>.kind`,
    // che e' il layout dello slab, cioe' il tipo vero dei tre tensori. Sul GGUF
    // `bartowski Q2_K` una stessa classe ha gate/up Q2_K e down Q3_K (o Q4_K):
    // un letterale qui li appiattirebbe sullo stesso kernel.
    for (const t of ["gate", "up", "down"]) {
      expect(SRC, `pipeline expert ${t}: kind dal layout`)
        .toMatch(new RegExp(`q35KQuantGemvWgsl\\(L\\.${t}\\.kind,`));
    }
  });
});
