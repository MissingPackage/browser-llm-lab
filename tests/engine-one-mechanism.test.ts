import { describe, it, expect } from "vitest";
import { globSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
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
// §7-ter): il codice si UNIFORMA — una meccanica, UNA implementazione.
//
// PATTERN: `tests/gpudevice.test.ts`. Non si verifica sito per sito (lista che
// marcisce): si intercetta UN ATTO SINGOLO E INEVITABILE e si pretende che
// avvenga solo nei siti DICHIARATI, con allowlist motivata.
//
// CHE COSA GARANTISCE QUESTO FILE, ONESTAMENTE (it.6, dopo tre bocciature):
// NON garantisce "una meccanica, una implementazione" — un test che scansiona
// il sorgente NON PUÒ, perché la differenza fra una duplicazione e una seconda
// famiglia legittima è SEMANTICA. Il verifier l'ha dimostrato: un router Qwen
// (softmax puro, niente clamp, niente nomi di tensori, niente VRAM) è
// invisibile a qualunque impronta testuale — ed è codice che deve esistere.
// L'INVARIANTE VERO è altrove: `tests/types/slotref-brand.ts` + il marchio di
// conio di `SlotRef` in residency.ts, che il COMPILATORE sorveglia — un'arena
// parallela non può fabbricare il riferimento che i kernel esigono.
// Questo file è un RATCHET su impronte NOTE: alza il costo di una deriva
// distratta e tiene il debito dichiarato (le voci DEBITO NOTO). Non è una
// prova, ed è sbagliato usarlo come tale.
//
// STORIA DI QUESTO GATE (tenuta qui perché è la lezione, non un dettaglio):
//  - it.1: firme TESTUALI ricalcate sul testo degli offender. Il verifier
//    dimostrò che una copia con spaziatura diversa sfuggiva.
//  - it.4 prima versione: CONGIUNZIONE di due indizi nello stesso file (nomi
//    dei tensori expert AND createBuffer). Il verifier la ruppe ESEGUENDO tre
//    evasioni in buona fede: (1) nome costruito per parti `ffn_${p}_exps`;
//    (2) nomi in un modulo e buffer in un altro — cioè la struttura che il
//    repo GIÀ usa (expertstore nomina, residency alloca); (3) importare
//    QUALSIASI cosa da moe/residency esentava dall'invariante.
//  - it.5 (questa): due invarianti SEPARATI, ognuno su un atto singolo, senza
//    esenzioni per import. Chi alloca memoria GPU va dichiarato; chi nomina i
//    tensori expert va dichiarato. Nessuna congiunzione da spezzare.

// Scansione ancorata a __dirname e su TUTTO src/ (non solo engine): a it.5 il
// glob era relativo alla cwd e limitato a src/engine — il verifier è passato
// da `src/microbench/` e da un file `.mts` senza far scattare niente.
const SRC = globSync(join(__dirname, "../src/**/*.{ts,mts,cts,tsx}"))
  .filter((f) => !f.endsWith(".d.ts"))
  .map((f) => relative(join(__dirname, ".."), f))
  .sort();
const read = (f: string): string => readFileSync(f, "utf8");

// --- I PREDICATI: definiti UNA VOLTA e usati sia dalle asserzioni sia dai
// test anti-marciume (il verifier ha mostrato che ricomporli nei test
// sintetici lascia svuotare il ciclo dell'asserzione senza accorgersene).
/** allocazione di memoria GPU: l'unica porta, come `requestDevice` */
export const allocatesGpu = (src: string): boolean => /\bcreateBuffer\b/.test(src);
/** NOMI dei tensori expert nel GGUF, anche costruiti per parti */
export const namesExpertTensors = (src: string): boolean => /ffn_[A-Za-z0-9_${}.[\]]*_exps/.test(src);
/** il clamp di `build_moe_ffn`: un router GLM-fedele non può evitarlo */
export const hardcodesRouterClamp = (src: string): boolean => /6\.103515625e-5/.test(src);

// --- ALLOWLIST CON RAZIONALE (pattern gpudevice.test): ogni riga dice PERCHÉ.
// Aggiungere qui senza motivo = rifare a mano la deriva che il gate impedisce.
const ALLOC_ALLOWED: Record<string, string> = {
  "src/engine/residency.ts": "il punto unico dell'arena expert: è QUI che gli slab diventano VRAM",
  "src/engine/glmmodel.ts": "orchestratore GLM: scratch, KV e uniform del forward (non slab expert, che chiede a ExpertCache)",
  "src/engine/glmforward.ts": "forward per-layer GLM (harness dei ktest): scratch e pesi non-expert",
  "src/engine/gpuforward.ts": "forward denso Qwen2.5: pesi e scratch di un modello SENZA expert",
  "src/engine/engine.worker.ts": "worker di bring-up: buffer di diagnostica",
  "src/engine/ktest/ktest.worker.ts": "harness dei kernel: alloca a mano per confrontare la meccanica con riferimenti indipendenti",
  "src/engine/glmbench/glmbench.worker.ts": "harness di bench: buffer di telemetria e staging",
  "src/microbench/runner.ts": "microbench dei kernel: alloca buffer di prova, nessun expert e nessuna residenza",
  "src/engine/q35gpumodel.ts": "DEBITO NOTO (docket fase-D item 4): arena expert propria, da migrare a ExpertCache — la riga sparisce con la migrazione",
};
const EXPERT_NAME_ALLOWED: Record<string, string> = {
  "src/engine/shape.ts": "validazione hard dell'inventario GLM contro il GGUF: nomina i tensori, non li porta in VRAM",
  "src/engine/q35shape.ts": "stessa cosa per la famiglia Qwen 3.5/3.6",
  "src/engine/expertstore.ts": "reader OPFS dei byte grezzi: nomina i tensori per leggerli dal file, nessuna GPU",
  "src/engine/quant.ts": "solo commenti che citano i tensori come esempio di layout",
  "src/engine/cpuref.ts": "CPUREF GLM: riferimento indipendente del differential testing (categoria dichiarata, docket item 3)",
  "src/engine/q35cpurefmodel.ts": "CPUREF Qwen: stessa categoria dichiarata, indipendenza voluta",
  "src/engine/ktest/ktest.worker.ts": "harness dei kernel: costruisce casi con pesi expert reali",
  "src/engine/q35gpumodel.ts": "DEBITO NOTO (docket fase-D item 4): legge e impacchetta expert per conto proprio",
};
const ROUTER_CLAMP_ALLOWED: Record<string, string> = {
  "src/engine/moe.ts": "il punto unico stesso: qui la costante WEIGHTS_SUM_CLAMP_MIN è definita",
  "src/engine/cpuref.ts": "CPUREF GLM: riferimento indipendente del differential testing (categoria dichiarata)",
  "src/engine/q35cpurefmodel.ts": "CPUREF Qwen: stessa categoria dichiarata, indipendenza voluta",
  "src/engine/q35gpumodel.ts": "DEBITO NOTO (docket fase-D item 4): router proprio, da sostituire con routerSelect",
  "src/engine/ktest/ktest.worker.ts": "harness dei kernel: verifica il valore del clamp contro la costante importata",
};

const offendersOf = (hit: (s: string) => boolean, allowed: Record<string, string>): string[] =>
  SRC.filter((f) => hit(read(f)) && !(f in allowed)).sort();

describe("ratchet sulle impronte note (NON una prova: v. il commento in testa)", () => {
  it("la scansione non è vuota (senza questo guard, un gate rotto è verde)", () => {
    // il guard che gpudevice.test ha e che a it.4 mancava
    expect(SRC.length).toBeGreaterThan(30);
  });

  it("A. la memoria GPU si alloca SOLO dai siti dichiarati", () => {
    expect(offendersOf(allocatesGpu, ALLOC_ALLOWED),
      "createBuffer fuori dai siti dichiarati: gli slab expert passano da ExpertCache (residency.ts)").toEqual([]);
  });

  it("B. i NOMI dei tensori expert vivono SOLO nei siti dichiarati", () => {
    expect(offendersOf(namesExpertTensors, EXPERT_NAME_ALLOWED),
      "chi nomina ffn_*_exps deve essere un reader/validatore/cpuref dichiarato, o passare dalla meccanica").toEqual([]);
  });

  it("C. il CLAMP del router vive solo in moe.ts (gli altri importano la costante)", () => {
    expect(offendersOf(hardcodesRouterClamp, ROUTER_CLAMP_ALLOWED),
      "usa WEIGHTS_SUM_CLAMP_MIN invece del letterale").toEqual([]);
    expect(WEIGHTS_SUM_CLAMP_MIN).toBe(6.103515625e-5);
  });

  it("l'allowlist è un DEBITO tracciato, non un parcheggio", () => {
    const all = { ...ALLOC_ALLOWED, ...EXPERT_NAME_ALLOWED, ...ROUTER_CLAMP_ALLOWED };
    for (const [f, why] of Object.entries(all)) {
      expect(why.length, `${f}: razionale troppo corto`).toBeGreaterThan(30);
    }
    // le voci "DEBITO NOTO" devono sparire: la fase 1 non chiude finché ci sono
    const debiti = [...new Set([ALLOC_ALLOWED, EXPERT_NAME_ALLOWED, ROUTER_CLAMP_ALLOWED]
      .flatMap((m) => Object.entries(m).filter(([, w]) => w.startsWith("DEBITO NOTO")).map(([f]) => f)))];
    expect(debiti).toEqual(["src/engine/q35gpumodel.ts"]);
  });
});

describe("il gate mette alla prova SE STESSO (anti-marciume)", () => {
  // I predicati sono gli STESSI oggetti usati dalle asserzioni (via
  // `offendersOf`): mutarli qui rompe là, e svuotare là rompe qui.
  it("A becca l'allocazione GPU comunque scritta", () => {
    expect(allocatesGpu("dev.createBuffer({size:1})")).toBe(true);
    expect(allocatesGpu("device . createBuffer ( { size: 1 } )")).toBe(true);
    expect(allocatesGpu("const mk = (d) => d.createBuffer(desc);")).toBe(true);
    expect(allocatesGpu("// niente GPU qui")).toBe(false);
  });

  it("B becca i nomi expert anche COSTRUITI PER PARTI (l'evasione del verifier it.4)", () => {
    expect(namesExpertTensors('"blk.0.ffn_gate_exps.weight"')).toBe(true);
    expect(namesExpertTensors("`blk.${l}.ffn_${p}_exps.weight`")).toBe(true);   // evasione 1
    expect(namesExpertTensors("const n = `ffn_${kind}_exps`;")).toBe(true);
    expect(namesExpertTensors('"ffn_gate.weight" // denso, non expert')).toBe(false);
  });

  it("C becca il letterale del clamp comunque scritto", () => {
    expect(hardcodesRouterClamp("Math.max(sum, 6.103515625e-5)")).toBe(true);
    expect(hardcodesRouterClamp("const MIN=6.103515625e-5;")).toBe(true);
    expect(hardcodesRouterClamp("import { WEIGHTS_SUM_CLAMP_MIN } from './moe';")).toBe(false);
  });

  it("nessuna esenzione per import: importare dalla meccanica NON è un lasciapassare", () => {
    // evasione 3 del verifier it.4: bastava importare la costante per sparire
    const evasione = 'import { WEIGHTS_SUM_CLAMP_MIN } from "./moe";\nconst t="ffn_gate_exps"; d.createBuffer({});';
    expect(allocatesGpu(evasione)).toBe(true);
    expect(namesExpertTensors(evasione)).toBe(true);
  });

  it("offendersOf non può essere svuotato in silenzio (buco M2 del verifier it.4)", () => {
    // il ciclo delle asserzioni è QUESTA funzione: se qualcuno la rendesse
    // vacua (`return []`), i test sintetici sui predicati resterebbero verdi.
    // Qui la si esercita direttamente con un predicato che becca tutto e una
    // allowlist vuota: deve elencare l'intero src/engine.
    expect(offendersOf(() => true, {}).length).toBe(SRC.length);
    expect(offendersOf(() => false, {})).toEqual([]);
    // e con una allowlist che copre tutto, zero offender
    const tutti = Object.fromEntries(SRC.map((f) => [f, "razionale sintetico lungo abbastanza per il test"]));
    expect(offendersOf(() => true, tutti)).toEqual([]);
  });

  it("nessuna congiunzione da spezzare: i due invarianti sono INDIPENDENTI", () => {
    // evasione 2 del verifier it.4: nomi in un file, buffer in un altro
    const fileA = 'const t = "blk.0.ffn_down_exps.weight"; // nessuna GPU qui';
    const fileB = "dev.createBuffer({ size: n }); // nessun nome qui";
    expect(namesExpertTensors(fileA)).toBe(true);   // A scatta sul primo
    expect(allocatesGpu(fileB)).toBe(true);         // B scatta sul secondo
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
