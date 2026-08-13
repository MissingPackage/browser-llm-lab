import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  planPrefill, causalLen, PREFILL_M, PREFILL_M_DENSE05B, PREFILL_SUBMIT_TOKENS,
} from "../src/engine/prefillplan";
import { QWEN_WORKGROUP_STORAGE_BYTES } from "../src/engine/gpulimits";
import { createKvLen } from "../src/engine/kvlen";

// Unit CPU-side sul piano di chunking (spec B1 §Forward multi-token) —
// convenzione CI-senza-GPU della fase A: la geometria del piano (copertura,
// posizioni, maschera causale, granularità di submit) si verifica qui; la
// matematica dei kernel la verifica la conformance (gate doppio).
//
// Le asserzioni sulla geometria sono scritte IN FUNZIONE di PREFILL_M, non del
// suo valore: alzare ancora la convenzione non deve costringere a riscrivere il
// file (l'unico vincolo residuo, dichiarato dov'e' usato, e' che M divida i 64
// token del boundary di submit).

describe("planPrefill — copertura e continuità", () => {
  it("469 token (prompt del bench): copre tutto, chunk pieni + ultimo parziale", () => {
    const plan = planPrefill(469, 0);
    // ORACOLO INDIPENDENTE dal loop di planPrefill: il piano è il copri-minimo di
    // 469 righe con chunk di al più M — cioè M·len basta e M·(len−1) NON basta.
    // (Il `Math.ceil(469/M)` sarebbe la formula dell'implementazione ri-scritta.)
    expect(PREFILL_M * plan.length).toBeGreaterThanOrEqual(469);
    expect(PREFILL_M * (plan.length - 1)).toBeLessThan(469);
    expect(plan.slice(0, -1).every((c) => c.rows === PREFILL_M)).toBe(true);
    // resto della divisione, non il 5 di M=8: con M=16 è ancora 5, con M=32 è 21
    expect(plan[plan.length - 1].rows).toBe(469 % PREFILL_M || PREFILL_M);
    // copertura senza buchi né overlap
    let next = 0;
    for (const c of plan) {
      expect(c.start).toBe(next);
      next = c.start + c.rows;
    }
    expect(next).toBe(469);
  });

  it("posBase = posStart + start per ogni chunk", () => {
    const plan = planPrefill(20, 100);
    for (const c of plan) expect(c.posBase).toBe(100 + c.start);
  });

  it("rows sempre in [1, M]; solo l'ultimo chunk può essere parziale", () => {
    for (const n of [1, 7, 8, 9, 16, 63, 64, 65, 469]) {
      const plan = planPrefill(n, 0);
      for (const c of plan) {
        expect(c.rows).toBeGreaterThanOrEqual(1);
        expect(c.rows).toBeLessThanOrEqual(PREFILL_M);
      }
      expect(plan.slice(0, -1).every((c) => c.rows === PREFILL_M)).toBe(true);
      expect(plan.reduce((a, c) => a + c.rows, 0)).toBe(n);
    }
  });

  it("casi minimi: n=1 e n=M", () => {
    expect(planPrefill(1, 0)).toEqual([{ start: 0, rows: 1, posBase: 0, submitAfter: true }]);
    const one = planPrefill(PREFILL_M, 5);
    expect(one).toEqual([{ start: 0, rows: PREFILL_M, posBase: 5, submitAfter: true }]);
  });
});

describe("planPrefill — granularità di submit", () => {
  it("submit ogni 64 token processati (knee della sim), sempre sull'ultimo", () => {
    const plan = planPrefill(469, 0);
    const submitsAt = plan
      .filter((c) => c.submitAfter)
      .map((c) => c.start + c.rows);
    // PRECONDIZIONE del caso concreto qui sotto: M divide i 64 del knee, quindi
    // un boundary di submit cade sempre a fine chunk. Se un domani M non dividesse
    // 64 (es. 24), il piano NON sbaglia — sono questi attesi a non valere più, e
    // questa riga lo dice invece di lasciare un diff misterioso.
    expect(PREFILL_SUBMIT_TOKENS % PREFILL_M).toBe(0);
    // boundary sui multipli di 64 + chiusura sul totale
    expect(submitsAt).toEqual([64, 128, 192, 256, 320, 384, 448, 469]);
  });

  it("il conteggio è in token, non in chunk (ultimo chunk parziale non sfasa)", () => {
    // submitEvery=4, M=3: chunk da 3+1 ⇒ submit dopo 4 token (secondo chunk)
    const plan = planPrefill(8, 0, 3, 4);
    expect(plan.map((c) => [c.rows, c.submitAfter])).toEqual([
      [3, false], [3, false], [2, true], // 3,6,8: multipli di 4 solo a fine (8)
    ]);
    const plan2 = planPrefill(9, 0, 2, 4);
    expect(plan2.map((c) => [c.rows, c.submitAfter])).toEqual([
      [2, false], [2, true], [2, false], [2, true], [1, true],
    ]);
  });

  it("n sotto la soglia di submit: un solo submit finale", () => {
    const plan = planPrefill(30, 0);
    expect(plan.filter((c) => c.submitAfter).length).toBe(1);
    expect(plan[plan.length - 1].submitAfter).toBe(true);
  });

  it("l'ultimo chunk ha SEMPRE submitAfter (il piano non lascia lavoro pendente)", () => {
    for (const n of [1, 8, 63, 64, 65, 128, 469, 1000]) {
      const plan = planPrefill(n, 0);
      expect(plan[plan.length - 1].submitAfter).toBe(true);
    }
  });
});

describe("causalLen — maschera causale intra-chunk", () => {
  it("la riga i del chunk vede posBase+i+1 posizioni (riga i vede KV [0, posBase+i])", () => {
    expect(causalLen(0, 0)).toBe(1);   // primo token in assoluto: vede solo sé
    expect(causalLen(0, 7)).toBe(8);
    expect(causalLen(64, 0)).toBe(65);
    expect(causalLen(64, 7)).toBe(72);
  });

  it("coerenza col piano: l'ultima riga dell'ultimo chunk vede tutto il prefisso", () => {
    const n = 469;
    const plan = planPrefill(n, 0);
    const last = plan[plan.length - 1];
    expect(causalLen(last.posBase, last.rows - 1)).toBe(n);
  });

  it("argomenti negativi: throw", () => {
    expect(() => causalLen(-1, 0)).toThrow();
    expect(() => causalLen(0, -1)).toThrow();
  });
});

describe("planPrefill — validazione hard (postura ds4)", () => {
  it("input non validi: throw, mai piano vuoto silenzioso", () => {
    expect(() => planPrefill(0, 0)).toThrow();
    expect(() => planPrefill(-1, 0)).toThrow();
    expect(() => planPrefill(1.5, 0)).toThrow();
    expect(() => planPrefill(8, -1)).toThrow();
    expect(() => planPrefill(8, 0, 0)).toThrow();
    expect(() => planPrefill(8, 0, 8, 0)).toThrow();
  });

  it("costanti del piano coerenti con la spec (M ≥ 16, submit ~64)", () => {
    // M=8 e' degenere per l'obiettivo (PHASES C7-2, ratificato dal PI): la
    // convenzione del piano e' M ≥ 16. Il 64 resta il knee misurato.
    expect(PREFILL_M).toBeGreaterThanOrEqual(16);
    expect(PREFILL_SUBMIT_TOKENS).toBe(64);
  });
});

// -------------------------------------------------------------------------
// L'ECCEZIONE del path denso Qwen2.5-0.5B, dimostrata invece che commentata.
// -------------------------------------------------------------------------
describe("PREFILL_M_DENSE05B — il PIN del path denso, col suo conto", () => {
  // rmsPairGemmSiluChunkFast, K = dModel 896 (il consumatore MASSIMO del path
  // Qwen fuso — gpulimits.ts §QWEN_WORKGROUP_STORAGE_BYTES): il workgroup storage
  // e' 4·K·mMax (tile delle attivazioni) + 256·mMax (partial) + 16·mMax (gRes)
  // byte — verificato contro il WGSL vero (wgsl.ts, xs4/partial/gRes). LINEARE in M.
  const wgStorage = (m: number) => 4 * 896 * m + 256 * m + 16 * m;
  // Cap OPERATIVO di oggi: il device chiede min(adapter, QWEN_WORKGROUP_STORAGE_BYTES)
  // — gpulimits.ts, e tests/gpudevice.test.ts lo verifica sul limite concesso.
  const DEVICE_WG_CAP = QWEN_WORKGROUP_STORAGE_BYTES;
  // Cap storico «richiesto a mano» prima della negoziazione (gpulimits.ts §1): piu'
  // largo di quello operativo, e la dimostrazione regge anche contro di lui.
  const LEGACY_WG_CAP = 32_768;

  it("il conto a M=8 sta sotto il cap — ed e' esattamente quello negoziato", () => {
    expect(PREFILL_M_DENSE05B).toBe(8);
    expect(wgStorage(PREFILL_M_DENSE05B)).toBe(30_848);
    // il pin non e' un numero a sé: e' il limite che il device concede davvero
    expect(wgStorage(PREFILL_M_DENSE05B)).toBe(DEVICE_WG_CAP);
    expect(wgStorage(PREFILL_M_DENSE05B)).toBeLessThanOrEqual(LEGACY_WG_CAP);
  });

  it("lo stesso conto alla convenzione PREFILL_M sfonda il cap: ecco PERCHE' il pin", () => {
    // attesi DERIVATI (linearità in M), non pinnati: il test dimostra l'invariante
    // «wgStorage(PREFILL_M) > cap», che resta vero — e più vero — se M sale ancora.
    expect(wgStorage(PREFILL_M)).toBe((PREFILL_M / PREFILL_M_DENSE05B) * wgStorage(PREFILL_M_DENSE05B));
    expect(wgStorage(PREFILL_M)).toBeGreaterThan(DEVICE_WG_CAP);
    expect(wgStorage(PREFILL_M)).toBeGreaterThan(LEGACY_WG_CAP);
    // e non di un pelo: già a M=16 (il valore di oggi) sono 61 696 B, quasi il doppio
    expect(wgStorage(16)).toBe(61_696);
    expect(wgStorage(16) - LEGACY_WG_CAP).toBe(28_928);
  });

  it("gpuforward.ts NON dipende piu' da PREFILL_M (alzare la convenzione non trascina il denso)", () => {
    const src = readFileSync(join(__dirname, "../src/engine/gpuforward.ts"), "utf8");
    expect(src).toMatch(/PREFILL_M_DENSE05B/);
    expect(src).not.toMatch(/PREFILL_M\b(?!_DENSE05B)/);
  });

  it("i report di gate (engine.worker.ts) dichiarano il knob ESEGUITO, non la convenzione", () => {
    // il campo prefill.mMax dei JSON engine-conformance / bench descrive createEngine
    // di gpuforward.ts, che gira a PREFILL_M_DENSE05B: se qui rientrasse PREFILL_M il
    // gate mentirebbe (knob dichiarato ≠ knob eseguito).
    const src = readFileSync(join(__dirname, "../src/engine/engine.worker.ts"), "utf8");
    expect(src).toMatch(/mMax: PREFILL_M_DENSE05B/);
    expect(src).not.toMatch(/PREFILL_M\b(?!_DENSE05B)/);
  });
});

describe("kvLen — semantica crop (spec B1 §Crop, contratto hard)", () => {
  it("decode sequenziale: assertNext(pos)+advance() avanzano il pointer", () => {
    const kv = createKvLen(16);
    for (let pos = 0; pos < 5; pos++) {
      kv.assertNext(pos);
      kv.advance();
    }
    expect(kv.len).toBe(5);
  });

  it("pos !== kvLen: throw (niente posizioni libere)", () => {
    const kv = createKvLen(16);
    kv.assertNext(0); kv.advance();
    expect(() => kv.assertNext(0)).toThrow(/pos 0 !== kvLen 1/);
    expect(() => kv.assertNext(2)).toThrow(/pos 2 !== kvLen 1/);
  });

  it("prefill a chunk: assertNext(posStart, n) + advance(n) — il piano avanza di M", () => {
    const kv = createKvLen(1024);
    kv.assertNext(0, 469);
    kv.advance(469);
    expect(kv.len).toBe(469);
    expect(() => kv.assertNext(0, 8)).toThrow(); // prefill da pos già consumate: crop prima
  });

  it("capacità: assertNext oltre ctxMax ⇒ contesto pieno", () => {
    const kv = createKvLen(8);
    kv.assertNext(0, 8); kv.advance(8);
    expect(() => kv.assertNext(8)).toThrow(/contesto pieno/);
    const kv2 = createKvLen(8);
    expect(() => kv2.assertNext(0, 9)).toThrow(/contesto pieno/);
  });

  it("crop(P): riporta il pointer indietro; il decode riparte ESATTAMENTE da P", () => {
    const kv = createKvLen(64);
    kv.advance(48); // prefisso
    kv.advance(16); // generazione
    kv.crop(48);
    expect(kv.len).toBe(48);
    kv.assertNext(48); kv.advance();
    expect(kv.len).toBe(49);
  });

  it("crop in avanti, negativo o non-intero: throw (validazione hard)", () => {
    const kv = createKvLen(64);
    kv.advance(10);
    expect(() => kv.crop(11)).toThrow(/crop non valido/);
    expect(() => kv.crop(-1)).toThrow(/crop non valido/);
    expect(() => kv.crop(2.5)).toThrow(/crop non valido/);
    kv.crop(10); // no-op legale (toLen === len)
    expect(kv.len).toBe(10);
  });

  it("reset() ≡ crop(0)", () => {
    const kv = createKvLen(64);
    kv.advance(20);
    kv.reset();
    expect(kv.len).toBe(0);
    kv.assertNext(0); kv.advance();
    expect(kv.len).toBe(1);
  });

  it("ctxMax non valido alla creazione: throw", () => {
    expect(() => createKvLen(0)).toThrow();
    expect(() => createKvLen(-1)).toThrow();
    expect(() => createKvLen(1.5)).toThrow();
  });
});
