import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { globSync } from "node:fs";
import {
  mkSlabLayout, packExpertSlab, routerSelect, ROUTER_GLM47, ROUTER_QWEN35MOE,
  SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, WEIGHTS_SUM_CLAMP_MIN,
} from "../src/engine/moe";
import {
  MOE_CFG_GLM47, MIN_SLOTS, PARK_Q4_0, PARK_Q4_1, SLOT_TABLE_ENTRIES,
  expertKey, expertKeyFor, minSlotsOf, moeParkOf, slotTableEntriesOf, expertSlots, arenaNeeds,
  type MoeModelConfig,
} from "../src/engine/residency";

// GATE STRUTTURALE del goal engine-fase-d (ruling PI 2026-08-10, direction
// §7-ter): il codice si UNIFORMA — una meccanica, UNA implementazione. Una
// famiglia nuova è una CONFIGURAZIONE, mai un path parallelo.
//
// PATTERN: `tests/gpudevice.test.ts` — non si verifica sito per sito (lista
// che marcisce), si VIETA l'esistenza di siti fuori dal punto unico
// intercettando qualcosa che NON SI PUÒ EVITARE, con allowlist motivata.
//
// La prima versione (it.1) usava firme TESTUALI ricalcate sul testo degli
// offender: il verifier di it.2 ha dimostrato che una copia con spaziatura
// diversa sfuggiva. Questa versione (it.4) usa due INVARIANTI:
//
//  A. ARENA/SLAB — per mettere un expert in VRAM servono per forza (i) i NOMI
//     GGUF dei tensori expert (`ffn_{gate,up,down}_exps`, convenzione di
//     llama.cpp valida per OGNI famiglia MoE) e (ii) la creazione di buffer
//     GPU. Chi fa entrambe DEVE passare dalla meccanica (moe.ts/residency.ts).
//
//  B. ROUTER — un router MoE fedele DEVE applicare il clamp del denominatore
//     di `build_moe_ffn` (6.103515625e-5, minimo f16 normale). Chi scrive quel
//     letterale invece di importare `WEIGHTS_SUM_CLAMP_MIN` riscrive il router.
//
// Il secondo `describe` mette alla prova IL GATE STESSO su offender sintetici:
// senza, un predicato può marcire in un no-op senza che nessuno se ne accorga.

const SRC = globSync("src/engine/**/*.ts").filter((f) => !f.endsWith(".d.ts"));
const read = (f: string): string => readFileSync(f, "utf8");

/** NOMI dei tensori expert nel GGUF: inevitabili per chiunque li carichi */
const EXPERT_TENSORS = /ffn_(gate|up|down)_exps/;
/** creazione di memoria GPU: inevitabile per chiunque li porti in VRAM */
const GPU_ALLOC = /createBuffer\s*\(/;
/** il clamp di build_moe_ffn: inevitabile per un router MoE fedele */
const CLAMP_LITERAL = /6\.103515625e-5/;
/** l'import della meccanica unica */
const IMPORTS_MECHANISM = /from "\.{1,2}\/(moe|residency)"/;

// ALLOWLIST CON RAZIONALE (pattern gpudevice.test): ogni riga dice PERCHÉ quel
// file non passa dalla meccanica. Aggiungere qui senza motivo = rifare a mano
// la deriva che questo gate esiste per impedire.
const ARENA_ALLOWED: Record<string, string> = {
  "src/engine/q35gpumodel.ts":
    "DEBITO NOTO (docket fase-D item 4): arena+LRU proprie, da migrare a ExpertCache — la riga sparisce con la migrazione",
  "src/engine/ktest/ktest.worker.ts":
    "harness dei kernel: impacchetta expert a mano PER CONFRONTARE la meccanica con un riferimento indipendente",
};
const ROUTER_ALLOWED: Record<string, string> = {
  "src/engine/moe.ts": "il punto unico stesso: qui la costante WEIGHTS_SUM_CLAMP_MIN è definita",
  "src/engine/cpuref.ts": "CPUREF GLM: riferimento indipendente del differential testing (categoria dichiarata, docket item 3)",
  "src/engine/q35cpurefmodel.ts": "CPUREF Qwen: stessa categoria dichiarata, indipendenza voluta",
  "src/engine/q35gpumodel.ts": "DEBITO NOTO (docket fase-D item 4): router proprio, da sostituire con routerSelect",
  "src/engine/ktest/ktest.worker.ts": "harness dei kernel: verifica il valore del clamp contro la costante importata",
};

describe("una meccanica, una implementazione — INVARIANTI (gate strutturale fase-D)", () => {
  it("A. chi porta EXPERT in VRAM passa dalla meccanica (o è in allowlist motivata)", () => {
    const offenders: string[] = [];
    for (const f of SRC) {
      const s2 = read(f);
      if (!EXPERT_TENSORS.test(s2) || !GPU_ALLOC.test(s2)) continue; // non fa slab
      if (IMPORTS_MECHANISM.test(s2)) continue;                      // passa dalla meccanica
      if (f in ARENA_ALLOWED) continue;                              // eccezione motivata
      offenders.push(f);
    }
    expect(offenders.sort(), "chi nomina i tensori expert E crea buffer GPU deve importare moe.ts/residency.ts").toEqual([]);
  });

  it("B. il CLAMP del router vive solo in moe.ts (gli altri importano la costante)", () => {
    const offenders: string[] = [];
    for (const f of SRC) {
      if (!CLAMP_LITERAL.test(read(f))) continue;
      if (f in ROUTER_ALLOWED) continue;
      offenders.push(f);
    }
    expect(offenders.sort(), "usa WEIGHTS_SUM_CLAMP_MIN: un router fedele non può evitare quel clamp").toEqual([]);
    expect(WEIGHTS_SUM_CLAMP_MIN).toBe(6.103515625e-5);
  });

  it("l'allowlist è un DEBITO tracciato, non un parcheggio", () => {
    for (const [f, why] of Object.entries({ ...ARENA_ALLOWED, ...ROUTER_ALLOWED })) {
      expect(why.length, `${f}: razionale troppo corto`).toBeGreaterThan(30);
    }
    // le voci "DEBITO NOTO" devono sparire: la fase 1 non è chiusa finché ci sono
    const debiti = [...new Set(Object.entries({ ...ARENA_ALLOWED, ...ROUTER_ALLOWED })
      .filter(([, why]) => why.startsWith("DEBITO NOTO")).map(([f]) => f))];
    expect(debiti).toEqual(["src/engine/q35gpumodel.ts"]);
  });
});

describe("il gate mette alla prova SE STESSO (anti-marciume)", () => {
  // Un predicato che non becca più niente è indistinguibile da un gate verde.
  const arenaHit = (s2: string): boolean =>
    EXPERT_TENSORS.test(s2) && GPU_ALLOC.test(s2) && !IMPORTS_MECHANISM.test(s2);

  it("becca un'arena scritta a mano, comunque la si formatti", () => {
    expect(arenaHit('const t = "blk.0.ffn_gate_exps.weight"; device.createBuffer({size: 1});')).toBe(true);
    // spaziatura diversa: la variante che sfuggiva alle firme testuali di it.1
    expect(arenaHit("const n='ffn_down_exps'; dev . createBuffer ( {size:1} )")).toBe(true);
    // nomi di variabile completamente diversi
    expect(arenaHit("const foo = `blk.${i}.ffn_up_exps.weight`; gpu.createBuffer({});")).toBe(true);
  });

  it("NON becca chi passa dalla meccanica, né chi non fa slab", () => {
    expect(arenaHit('import { packExpertSlab } from "./moe";\nconst t="ffn_gate_exps"; device.createBuffer({});')).toBe(false);
    expect(arenaHit('import { ExpertCache } from "../residency";\nconst t="ffn_up_exps"; d.createBuffer({});')).toBe(false);
    expect(arenaHit('const t = "ffn_gate_exps.weight"; // solo validazione shape, niente VRAM')).toBe(false);
    expect(arenaHit("device.createBuffer({ size: 16 }); // buffer qualsiasi, niente expert")).toBe(false);
  });

  it("il predicato del router becca il letterale comunque scritto", () => {
    expect(CLAMP_LITERAL.test("Math.max(sum, 6.103515625e-5)")).toBe(true);
    expect(CLAMP_LITERAL.test("const MIN=6.103515625e-5;")).toBe(true);
    expect(CLAMP_LITERAL.test("import { WEIGHTS_SUM_CLAMP_MIN } from './moe';")).toBe(false);
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

describe("residency parametrica: GLM è una configurazione (fase-D fase 1)", () => {
  it("i valori storici GLM sono DERIVATI dalla config, non cablati", () => {
    const park = moeParkOf(MOE_CFG_GLM47);
    expect(park.q4_0).toBe(PARK_Q4_0);   // 2.688
    expect(park.q4_1).toBe(PARK_Q4_1);   // 256
    expect(park.q4_0 + park.q4_1).toBe(2944); // il parco misurato in C1
    expect(slotTableEntriesOf(MOE_CFG_GLM47)).toBe(SLOT_TABLE_ENTRIES);
    expect(minSlotsOf(MOE_CFG_GLM47)).toEqual(MIN_SLOTS);
    expect(expertKeyFor(MOE_CFG_GLM47, 7, 3)).toBe(expertKey(7, 3));
  });

  it("una config qwen35moe produce parco/chiavi/minimi coerenti dallo STESSO codice", () => {
    // 35B-A3B: 40 layer tutti MoE, 256 expert, top-8, due classi dal file
    const E = 2048 * 512;
    const q4k = mkSlabLayout("q4_K", { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E });
    const q6k = mkSlabLayout("q6_K", { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E }, { kind: "q6_K", elems: E });
    const DOWN_Q6K_LAYERS = new Set([34, 38, 39]); // i 3 layer down-Q6_K del file
    const cfg: MoeModelConfig = {
      id: "qwen3.6-35b-a3b", nLayer: 40, denseLead: 0, nExpert: 256, nExpertUsed: 8,
      classes: ["q4_K", "q6_K"],
      classOf: (l) => (DOWN_Q6K_LAYERS.has(l) ? "q6_K" : "q4_K"),
      layout: (c) => (c === "q6_K" ? q6k : q4k),
    };
    const park = moeParkOf(cfg);
    expect(park.q4_K).toBe(37 * 256);
    expect(park.q6_K).toBe(3 * 256);
    expect(park.q4_K + park.q6_K).toBe(10240); // il parco misurato in q1
    expect(slotTableEntriesOf(cfg)).toBe(10240);
    expect(minSlotsOf(cfg)).toEqual({ q4_K: 8 * 37, q6_K: 8 * 3 });
    // chiavi: nessuna collisione fra (layer, expert) distinti
    expect(expertKeyFor(cfg, 1, 0)).toBe(256);
    expect(expertKeyFor(cfg, 0, 255)).toBe(255);
  });
});

describe("il MOTORE della residenza è cfg-driven (fase-D fase 1 slice C)", () => {
  const E = 2048 * 512;
  const q4k = mkSlabLayout("q4_K", { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E });
  const q6k = mkSlabLayout("q6_K", { kind: "q4_K", elems: E }, { kind: "q4_K", elems: E }, { kind: "q6_K", elems: E });
  const DOWN_Q6K = new Set([34, 38, 39]);
  const qwen: MoeModelConfig = {
    id: "qwen3.6-35b-a3b", nLayer: 40, denseLead: 0, nExpert: 256, nExpertUsed: 8,
    classes: ["q4_K", "q6_K"],
    classOf: (l) => (DOWN_Q6K.has(l) ? "q6_K" : "q4_K"),
    layout: (c) => (c === "q6_K" ? q6k : q4k),
  };

  it("expertSlots ripartisce sul parco DELLA CONFIG (q35 non passa più da slotsOverride)", () => {
    const budget = 12 * 2 ** 30;
    const s = expertSlots({ budgetBytes: budget, cfg: qwen });
    // proporzionale al parco: 37/40 dei byte alla classe q4_K, 3/40 alla q6_K
    const park = moeParkOf(qwen);
    expect(s.q4_K).toBe(Math.min(Math.floor((budget * park.q4_K / 10240) / q4k.bytes), park.q4_K));
    expect(s.q6_K).toBe(Math.min(Math.floor((budget * park.q6_K / 10240) / q6k.bytes), park.q6_K));
    expect(s.q4_K).toBeGreaterThan(0);
    expect(s.q6_K).toBeGreaterThan(0);
    // e la ripartizione GLM resta quella storica (nessuna regressione)
    const g = expertSlots({ budgetBytes: budget });
    expect(g.q4_0 + g.q4_1).toBeGreaterThan(2000);
  });

  it("arenaNeeds dimensiona sulle classi della config, non su quelle GLM", () => {
    const needs = arenaNeeds({
      budgetBytes: 12 * 2 ** 30, cfg: qwen,
      maxBufferBytes: 2 * 2 ** 30, maxBindingBytes: 2 * 2 ** 30,
    });
    expect(needs.arenaBuffers).toBeGreaterThanOrEqual(1);
    // la finestra deve reggere almeno uno slab della classe più grande
    expect(needs.arenaWindowBytes).toBeGreaterThanOrEqual(q6k.bytes);
  });

  it("slotsOverride con chiavi di un'altra config FALLISCE (niente cache a zero classi)", () => {
    expect(() => expertSlots({ budgetBytes: 1, slotsOverride: { q4_0: 10, q4_1: 10 }, cfg: qwen }))
      .toThrow(/senza la classe "q4_K"/);
  });

  it("i campi compat legacy FALLISCONO su un layout K-quant invece di mentire", () => {
    // trappola trovata dal verifier it.2: prima restituivano un offset finto
    expect(() => q4k.gateScales).toThrow(/non esiste sul formato q4_K/);
    expect(() => q6k.downScales).toThrow(/non esiste sul formato q6_K/);
    // sui legacy restano leciti e corretti
    expect(SLAB_DOWN_Q4_0.gateScales).toBe(1572864);
    // ma NON esplodono su spread/serializzazione (getter non enumerabili)
    expect(() => JSON.stringify(q4k)).not.toThrow();
    expect(() => ({ ...q6k })).not.toThrow();
  });
});
