import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { globSync } from "node:fs";
import {
  mkSlabLayout, packExpertSlab, routerSelect, ROUTER_GLM47, ROUTER_QWEN35MOE,
  SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1,
} from "../src/engine/moe";

// GATE STRUTTURALE del goal engine-fase-d (ruling PI 2026-08-10, direction
// §7-ter): il codice si UNIFORMA — una meccanica, UNA implementazione. Una
// famiglia nuova è una CONFIGURAZIONE, mai un path parallelo.
//
// Pattern di `tests/gpudevice.test.ts`, che già vieta per costruzione un
// secondo sito di `requestDevice`: qui la scansione del sorgente vieta un
// secondo router, un secondo layout di slab, una seconda arena/LRU.

const SRC = globSync("src/engine/**/*.ts").filter((f) => !f.endsWith(".d.ts"));
const read = (f: string): string => readFileSync(f, "utf8");

// RATCHET DELLE DUPLICAZIONI: questa lista può solo ACCORCIARSI. Il test è
// verde sullo stato noto, rosso se una duplicazione COMPARE, e rosso anche
// se una sparisce senza aggiornare la lista — così la parità non si può né
// perdere né dichiarare a voce. A fase 1 completa: liste vuote (tranne le
// eccezioni dichiarate).
const DUP_NOTE = {
  // ECCEZIONE DICHIARATA E PERMANENTE: il cpuref è un riferimento
  // INDIPENDENTE per il differential testing — se usasse lo stesso codice
  // del motore, un bug del router sarebbe invisibile al confronto. Resta.
  router: ["src/engine/q35cpurefmodel.ts", "src/engine/q35gpumodel.ts"],
  slab: ["src/engine/q35gpumodel.ts"],
  arena: ["src/engine/q35gpumodel.ts"],
};

describe("una meccanica, una implementazione (gate strutturale fase-D)", () => {
  it("il ROUTER vive solo in moe.ts: nessun altro modulo ricalcola gating+top-K", () => {
    const offenders: string[] = [];
    for (const f of SRC) {
      if (f.endsWith("moe.ts")) continue;
      const s = read(f);
      // firma di un router riscritto in casa: softmax/sigmoid sui logit +
      // selezione top-K nello stesso file
      const hasGating = /Math\.exp\(-?logits?\[/.test(s) || /probs\[e\] = Math\.exp\(/.test(s);
      const hasTopK = /\.slice\(0, *(topK|nUsed|nExpertUsed)\)/.test(s) || /taken\[best\] = 1/.test(s);
      if (hasGating && hasTopK) offenders.push(f);
    }
    expect(offenders.sort(), "ratchet router: la lista può solo accorciarsi (usa routerSelect); il cpuref è eccezione dichiarata").toEqual(DUP_NOTE.router);
  });

  it("il LAYOUT SLAB vive solo in moe.ts: nessun altro modulo calcola offset di slot a mano", () => {
    const offenders: string[] = [];
    for (const f of SRC) {
      if (f.endsWith("moe.ts")) continue;
      const s = read(f);
      // firma di un layout riscritto: byte per slot sommati a mano dai
      // tensori dell'expert (gate+up+down) nello stesso file
      if (/slotBytes\s*[:=][^;]*gate[A-Za-z]*\s*\+/.test(s) || /2 \* gateRp \+ down/.test(s)) {
        offenders.push(f);
      }
    }
    expect(offenders.sort(), "ratchet layout slab: la lista può solo accorciarsi (usa mkSlabLayout)").toEqual(DUP_NOTE.slab);
  });

  it("l'ARENA+LRU vive solo in residency.ts: nessun altro modulo tiene una propria slot table", () => {
    const offenders: string[] = [];
    for (const f of SRC) {
      if (f.endsWith("residency.ts")) continue;
      const s = read(f);
      // firma di una cache di slot riscritta: mappa chiave→slot + evizione LRU
      const hasByKey = /byKey\s*[:.]/.test(s) || /new Map<number, number>\(\)/.test(s);
      const hasLru = /lru\.(delete|set|entries)/.test(s);
      if (hasByKey && hasLru) offenders.push(f);
    }
    expect(offenders.sort(), "ratchet arena/LRU: la lista può solo accorciarsi (usa ExpertCache di residency.ts)").toEqual(DUP_NOTE.arena);
  });
});

describe("moe parametrico: GLM e qwen35moe sono configurazioni", () => {
  it("le classi GLM derivate dal builder generico hanno i byte storici", () => {
    // 5.308.416 e 5.505.024: le due size-class misurate sul file (spec C2 §1)
    expect(SLAB_DOWN_Q4_0.bytes).toBe(5308416);
    expect(SLAB_DOWN_Q4_1.bytes).toBe(5505024);
    expect(SLAB_DOWN_Q4_0.gateQs).toBe(0);
    expect(SLAB_DOWN_Q4_0.qsBytes).toBe(1572864);
    expect(SLAB_DOWN_Q4_1.downScalesBytes).toBe(393216);
  });

  it("lo stesso builder produce le classi qwen35moe (K-quant, segmento unico)", () => {
    const E = 2048 * 512; // dModel × dFfnExpert del 35B-A3B
    const q4k = mkSlabLayout("q35-down-q4_K", { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E });
    const q6k = mkSlabLayout("q35-down-q6_K", { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E }, { kind: "q6_K", elems: E });
    expect(q4k.bytes).toBe(3 * 589824);          // 1.769.472 — il numero misurato in q1
    expect(q6k.bytes).toBe(2 * 589824 + 868352); // 2.048.000 — con il padding 210→212
    // K-quant: nessun segmento scale separato
    expect(q4k.gate.scales).toBeNull();
    expect(q6k.down.scales).toBeNull();
    // offset allineati (il builder lo asserisce, qui lo si vede)
    expect(q6k.down.data % 256).toBe(0);
  });

  it("router: GLM (sigmoid+bias+scale) e qwen35moe (softmax puro) dallo stesso codice", () => {
    const logits = [0.5, -1, 2, 0.1, 3, -0.2];
    const bias = [0, 0, 0, 5, 0, 0]; // spinge l'expert 3 nella SELEZIONE
    const glm = routerSelect(logits, bias, { ...ROUTER_GLM47, nUsed: 2 });
    // il bias entra solo nella selezione: 3 è scelto, ma il suo peso resta
    // quello del probs NON biased
    expect(Array.from(glm.experts)).toEqual([3, 4]);
    const sig = (x: number): number => 1 / (1 + Math.exp(-x));
    const s3 = sig(0.1), s4 = sig(3);
    expect(glm.weights[0]).toBeCloseTo((s3 / (s3 + s4)) * 1.8, 12);

    const qwen = routerSelect(logits, null, { ...ROUTER_QWEN35MOE, nUsed: 2 });
    // niente bias: vincono i due logit più alti; pesi = softmax rinormalizzato,
    // nessuno scale
    expect(Array.from(qwen.experts)).toEqual([4, 2]);
    expect(qwen.weights[0] + qwen.weights[1]).toBeCloseTo(1, 12);
    const e4 = Math.exp(3 - 3), e2 = Math.exp(2 - 3);
    expect(qwen.weights[0]).toBeCloseTo(e4 / (e4 + e2), 12);
  });

  it("packExpertSlab è parametrico e verifica la taglia dei byte sorgente", () => {
    const E = 256 * 128; // taglia che tiene gli offset 256-allineati
    const l = mkSlabLayout("t", { kind: "q4_0", elems: E }, { kind: "q4_0", elems: E }, { kind: "q4_K", elems: E });
    const ok4_0 = new Uint8Array((E / 32) * 18);
    const ok4_K = new Uint8Array((E / 256) * 144);
    expect(() => packExpertSlab(ok4_0, ok4_0, ok4_K, l)).not.toThrow();
    // taglia sbagliata ⇒ throw con il nome del tensore
    expect(() => packExpertSlab(ok4_0, ok4_0, new Uint8Array(10), l)).toThrow(/down .* attesi/);
    expect(() => packExpertSlab(new Uint8Array(10), ok4_0, ok4_K, l)).toThrow(/gate .* attesi/);
    // LIMITE NOTO del controllo di taglia (trovato da questo test): q4_0 e
    // q4_K hanno gli STESSI byte/peso (0.5625) — 18 B/32 pesi = 144 B/256
    // pesi. Byte di un q4_0 passati dove il layout vuole q4_K NON sono
    // distinguibili dalla lunghezza: il tipo lo detta il layout, che a sua
    // volta viene dal GGUF (q35shape/shape lo validano contro il file).
    expect((E / 32) * 18).toBe((E / 256) * 144);
  });
});
