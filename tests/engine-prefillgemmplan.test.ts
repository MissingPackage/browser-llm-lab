import { describe, expect, it } from "vitest";
import {
  PREFILL_IDOT_FEATURE, PREFILL_M1_LEGACY, prefillGemmCapsFor, planPrefillGemm,
  prefillGemmScratchFor, prefillPlanDispatches,
  type PrefillSite,
} from "../src/engine/prefillgemmplan";
import {
  prefillGemmSplitsFor, prefillPartialFloats, prefillGemmWorkgroupStorageBytes,
  prefillQuantXQ8Wgsl, PREFILL_SPLITS_MEASURED, PREFILL_SPLITS_UNSPLIT,
} from "../src/engine/kernels/wgsl";
import { dispatchWeightBytes, type PrefillDispatch, type PrefillQuantKind } from "../src/engine/prefillbytes";

// PIANO del GEMM di prefill: quale via prende ogni sito, con quante fette, con
// quanto scratch. Modulo PURO — nessuna GPU, nessun WGSL generato — come
// `gemvcaps.ts` e `prefillbytes.ts`: gira in CI su qualunque runner.
//
// La tesi che questi test sorvegliano e' UNA: il piano NON ridecide niente che
// sia gia' deciso accanto al kernel. Fette, parziali, workgroup storage e il
// predicato di ammissibilita' (quali kind, e per ognuno la sua unita' di taglio
// — 64 sul q4_0, 256 sul q5_K) arrivano da `kernels/wgsl.ts`, e i confronti qui
// sotto sono SEMPRE contro la funzione importata, mai contro una costante
// ricopiata — e' il bug di it.7 (due posti che decidevano le
// righe-per-workgroup, e le decidevano diverse) preso alla radice.
//
// La copertura per CALL-SITE (l'ex `[6f]`, che scansionava q35gpumodel.ts) NON
// vive piu' qui: si e' SPOSTATA in tests/engine-prefillwiring-q5k.test.ts, dove
// sta accanto al cablaggio che censisce. Questo file resta puro — nessun
// `readFileSync` su un modulo del motore.

/** M del prefill in produzione (PREFILL_M = 16, riga 2 dell'ondata TTFT). */
const M = 16;

/** Le due shape MISURATE in riga 1 (qkv e down del 4B a chunk 16). */
const SHAPES = [
  { K: 2560, N: 9216 },
  { K: 9216, N: 2560 },
] as const;

// ---------------------------------------------------------------------------
// (1) LA VIA
// ---------------------------------------------------------------------------
describe("[1] la via: idot dove si puo', f32 come fallback dichiarato, legacy col perche'", () => {
  it("q4_0 con K%64==0: idot quando la feature c'e', f32 quando non c'e'", () => {
    for (const { K, N } of SHAPES) {
      const yes = planPrefillGemm({ kind: "q4_0", K, N, M, idot: true });
      const no = planPrefillGemm({ kind: "q4_0", K, N, M, idot: false });
      expect(yes.via, `K${K}xN${N}`).toBe("idot");
      expect(no.via, `K${K}xN${N}`).toBe("f32");
      // il `reason` non e' mai vuoto su NESSUNA via (postura di GemvCaps.why)
      expect(yes.reason.length).toBeGreaterThan(0);
      expect(no.reason.length).toBeGreaterThan(0);
      // il fallback si DICHIARA tale: chi legge la telemetria deve vedere che
      // la via f32 e' un ripiego, non una scelta migliore
      expect(no.reason).toContain(PREFILL_IDOT_FEATURE);
      expect(no.reason).toContain("FALLBACK DICHIARATO");
    }
  });

  it("q5_K con K%256==0: idot quando la feature c'e', f32 quando non c'e'", () => {
    // La riga 2 di engine-kquant: `blk.*.ssm_out` del 4B (K=4096 = 16
    // superblocchi, N=2560) esce dalla lista delle eccezioni ed entra nella via
    // veloce. Le fette sono 4 — contate in SUPERBLOCCHI, non in blocchi da 32.
    const yes = planPrefillGemm({ kind: "q5_K", K: 4096, N: 2560, M, idot: true });
    const no = planPrefillGemm({ kind: "q5_K", K: 4096, N: 2560, M, idot: false });
    expect(yes.via).toBe("idot");
    expect(no.via).toBe("f32");
    expect(yes.splits).toBe(4);
    expect(no.splits).toBe(4);
    expect(yes.splits).toBe(prefillGemmSplitsFor(4096, 2560, "q5_K"));
    expect(yes.partialFloats).toBe(4 * M * 2560);
    expect(no.partialFloats).toBe(4 * M * 2560);
    // il kind REALE arriva al kernel: se il piano continuasse a passare "q4_0"
    // i numeri sarebbero quelli dei blocchi da 32 (K/32 = 128 -> 4 fette da
    // BK=2, e 1.152 B di storage invece di 5.120)
    expect(yes.wgStorageBytes).toBe(5120);
    expect(no.wgStorageBytes).toBe(2048);
    expect(yes.wgStorageBytes).toBe(
      prefillGemmWorkgroupStorageBytes({ kind: "q5_K", K: 4096, N: 2560, M, splits: 4 }, "idot"));
    expect(no.wgStorageBytes).toBe(
      prefillGemmWorkgroupStorageBytes({ kind: "q5_K", K: 4096, N: 2560, M, splits: 4 }, "f32"));
    // la via intera Q5_K riusa `prefillQuantXQ8Wgsl` TALE E QUALE (sotto-blocchi
    // da 32, correzione C0-2): stesse attivazioni, nessun secondo quantizzatore
    const blocks = M * (4096 / 32);
    expect(yes.xqU32).toBe(blocks * 8);
    expect(yes.xscF32).toBe(blocks);
    expect(no.xqU32).toBe(0);
    expect(no.xscF32).toBe(0);
    expect(no.reason).toContain("FALLBACK DICHIARATO");
    for (const r of [yes, no]) expect(r.reason).toContain("q5_K");
  });

  it("kind fuori dalle vie veloci: legacy, con una ragione >= 40 caratteri che NOMINA il kind", () => {
    const kinds: PrefillQuantKind[] = ["q8_0", "q4_K", "q6_K", "f32"];
    for (const kind of kinds) {
      for (const idot of [true, false]) {
        const r = planPrefillGemm({ kind, K: 2560, N: 9216, M, idot });
        expect(r.via, `${kind} idot=${idot}`).toBe("legacy");
        expect(r.reason.length, `${kind}: "${r.reason}"`).toBeGreaterThanOrEqual(40);
        expect(r.reason, `${kind}`).toContain(kind);
      }
    }
    // IL q4_1 NON E' PIU' QUI, ed e' la riga 3 di `engine-kquant` ad averlo
    // tolto: i 4 `ffn_down` dei layer 0-3 hanno la loro forma multi-riga,
    // misurata 22,57x a M=16. Questa asserzione diceva `legacy` e adesso dice
    // il contrario — cambiata deliberatamente, non aggirata.
    const q41 = planPrefillGemm({ kind: "q4_1", K: 9216, N: 2560, M, idot: true });
    expect(q41.via).toBe("idot");
    expect(q41.reason).toContain("q4_1");
    expect(q41.reason.length).toBeGreaterThanOrEqual(40);
    // e senza la feature intera resta la sua f32, non la legacy
    expect(planPrefillGemm({ kind: "q4_1", K: 9216, N: 2560, M, idot: false }).via).toBe("f32");
  });

  it("K%64!=0: legacy, con una ragione che NOMINA K", () => {
    for (const K of [2592, 896 + 32, 100]) {
      const r = planPrefillGemm({ kind: "q4_0", K, N: 2560, M, idot: true });
      expect(r.via, `K=${K}`).toBe("legacy");
      expect(r.reason, `K=${K}`).toContain(String(K));
      expect(r.reason, `K=${K}`).toContain("64");
    }
  });

  it("il rifiuto NON e' un secondo predicato: e' la voce del kernel, riportata", () => {
    // Il piano non ricopia ne' `kind !== q4_0` ne' `K % 64`: sonda il contorno
    // del kernel (`prefillGemmCheck`, dentro `prefillPartialFloats`) e riporta
    // il suo messaggio. Qui si verifica proprio quello — la ragione CONTIENE
    // alla lettera cio' che il kernel dice quando gli si chiede la shape.
    // COSA E' CAMBIATO IN QUESTA LISTA, E PERCHE' (riga 4 di engine-kquant):
    // c'erano `q4_K` e `q6_K` a K=4096. Da riga 4 quei due formati HANNO il
    // loro kernel, quindi `prefillPartialFloats` non solleva piu' su quella
    // shape e il caso non provava piu' niente — pretendere il contrario
    // avrebbe misurato il passato. Al loro posto c'e' `f32`, che un
    // moltiplicatore multi-riga non ce l'ha e non l'avra': il caso che questo
    // test vuole e' «il kernel rifiuta, il piano riporta», non «il piano non
    // instrada» (quello e' il flag `wired`, e sta in
    // tests/engine-prefillgemmplan-notwired.test.ts).
    for (const bad of [
      { kind: "f32" as PrefillQuantKind, K: 4096 },
      { kind: "q4_0" as PrefillQuantKind, K: 2592 },
      // Q5_K con K non multiplo di 256: il formato ora e' AMMESSO, e' la
      // geometria a non tornare (un superblocco non si taglia a meta').
      { kind: "q5_K" as PrefillQuantKind, K: 2688 },
    ]) {
      let kernelMsg = "";
      try {
        prefillPartialFloats({
          kind: bad.kind as "q4_0", K: bad.K, N: 2560, M, splits: PREFILL_SPLITS_UNSPLIT,
        });
      } catch (e) {
        kernelMsg = (e as Error).message;
      }
      expect(kernelMsg.length, `${bad.kind} K=${bad.K}: il kernel deve rifiutare`).toBeGreaterThan(0);
      const r = planPrefillGemm({ kind: bad.kind, K: bad.K, N: 2560, M, idot: true });
      expect(r.via).toBe("legacy");
      expect(r.reason).toContain(kernelMsg);
    }
    // ...e l'ordine dei rifiuti e' quello del kernel: prima il formato, poi la
    // geometria. Su un kind NON INSTRADATO (q4_K, q6_K) con K storto, il
    // messaggio deve dire il KIND — che e' la ragione strutturale e quella su
    // cui si puo' agire — e non il K, che e' la seconda. Da riga 4 il rifiuto
    // di questi due arriva dal FLAG DI CABLAGGIO invece che dal contorno del
    // kernel, e l'ordine e' lo stesso per la stessa ragione: anche con K buono
    // resterebbero legacy, quindi parlare del K sarebbe una risposta vera e
    // inutile.
    for (const kind of ["q4_K", "q6_K"] as const) {
      const both = planPrefillGemm({ kind, K: 2592, N: 2560, M, idot: true });
      expect(both.reason, kind).toContain(kind);
      expect(both.reason, kind).not.toContain("2592");
    }
    // Il rifiuto GEOMETRICO del q5_K, invece, nomina K e 256: il formato e'
    // buono, e' la shape che non si divide in superblocchi.
    const geom = planPrefillGemm({ kind: "q5_K", K: 2592, N: 2560, M, idot: true });
    expect(geom.via).toBe("legacy");
    expect(geom.reason).toContain("2592");
    expect(geom.reason).toContain("256");
  });

  it("sulla via legacy non si prenota nulla: fette e scratch a zero", () => {
    // Un parziale o uno storage NON nullo su una via che non accende la
    // pipeline veloce gonfierebbe lo scratch CONDIVISO (prefillGemmScratchFor
    // prende il max) per un buffer che nessun dispatch legge.
    for (const r of [
      planPrefillGemm({ kind: "q4_K", K: 4096, N: 2560, M, idot: true }),
      planPrefillGemm({ kind: "q6_K", K: 4096, N: 2560, M, idot: true }),
      // e anche a M=1, dove la via veloce non si prende su NESSUN kind
      planPrefillGemm({ kind: "q5_K", K: 4096, N: 2560, M: 1, idot: true }),
      planPrefillGemm({ kind: "q4_0", K: 2560, N: 9216, M: 1, idot: true }),
    ]) {
      expect(r.splits).toBe(0);
      expect(r.partialFloats).toBe(0);
      expect(r.wgStorageBytes).toBe(0);
      expect(r.xqU32).toBe(0);
      expect(r.xscF32).toBe(0);
    }
  });

  // -------------------------------------------------------------------------
  // [1f] M=1 — la clausola 7 del done-when della riga 2, e la sua ONESTA'.
  //
  // La clausola dice: a M=1 si resta legacy su OGNI kind. Il test la sorveglia,
  // MA sorveglia anche che la ragione non spacci il 0,56x per una legge
  // generale — perche' non lo e': al banco a M=1 la forma multi-riga e' PIU'
  // VELOCE della legacy su q4_0/idot (8,64x) e su q5_K/idot (3,47x), e piu'
  // lenta soltanto su q5_K/f32 (0,56x). La clausola resta perche' e' di
  // contratto e conservativa; la stringa deve dirlo, non nasconderlo.
  // -------------------------------------------------------------------------
  it("[1f] M=1 ⇒ legacy su OGNI kind, con la misura di banco ATTRIBUITA alla sua cella", () => {
    const kinds: PrefillQuantKind[] = ["q4_0", "q4_1", "q8_0", "q4_K", "q5_K", "q6_K", "f32"];
    for (const kind of kinds) {
      for (const idot of [true, false]) {
        // K scelto ammissibile su ENTRAMBE le scale (4096 = 128 blocchi da 32 =
        // 16 superblocchi da 256): se il piano instradasse, instraderebbe qui.
        const r = planPrefillGemm({ kind, K: 4096, N: 2560, M: 1, idot });
        expect(r.via, `${kind} idot=${idot}`).toBe("legacy");
        expect(r.reason, `${kind} idot=${idot}`).toContain("M=1");
        expect(r.reason, `${kind} idot=${idot}`).toContain("0,56x");
        // la cella e' NOMINATA, e le celle contrarie pure: una ragione che
        // dicesse solo "0,56x" sarebbe una misura falsa in telemetria
        expect(r.reason, `${kind}`).toContain("q5_K sulla via f32");
        expect(r.reason, `${kind}`).toContain("3,47x");
        expect(r.reason, `${kind}`).toContain("8,64x");
        expect(r.reason, `${kind}`).toContain("CONTRATTO");
      }
    }
    // e a M=16, con lo STESSO kind e lo STESSO K, la via veloce si prende: il
    // rifiuto e' di M, non della shape
    expect(planPrefillGemm({ kind: "q5_K", K: 4096, N: 2560, M: 16, idot: true }).via).toBe("idot");
    expect(planPrefillGemm({ kind: "q4_0", K: 4096, N: 2560, M: 16, idot: true }).via).toBe("idot");
  });

  it("[1f] la cella pinnata nella costante e' quella del JSON, e il rapporto torna", () => {
    // Il 0,56x non e' un numero scritto a mano dentro una stringa: viene da
    // `PREFILL_M1_LEGACY.measured`, e qui si verifica che il rapporto dichiarato
    // sia davvero legacy/veloce di quella cella
    // (kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json, q5_K/base-batch-z
    // 0,08288 ms vs q5_K/splitk-f32 0,148992 ms).
    const m = PREFILL_M1_LEGACY.measured;
    expect(m.kind).toBe("q5_K");
    expect(m.via).toBe("f32");
    expect(m.legacyMs / m.fastMs).toBeCloseTo(0.5563, 4);
    expect(m.ratio).toBe("0,56x");
    expect(PREFILL_M1_LEGACY.M).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// (2) I NUMERI ARRIVANO DAL KERNEL, NON DA UNA COPIA
// ---------------------------------------------------------------------------
describe("[2] splits / partialFloats / wgStorageBytes IDENTICI alle funzioni di wgsl.ts", () => {
  for (const { K, N } of SHAPES) {
    it(`K${K}xN${N} a M=${M}: il piano non ridecide nessuno dei tre numeri`, () => {
      const splits = prefillGemmSplitsFor(K, N);
      const opts = { kind: "q4_0" as const, K, N, M, splits };
      for (const via of ["idot", "f32"] as const) {
        const r = planPrefillGemm({ kind: "q4_0", K, N, M, idot: via === "idot" });
        expect(r.via).toBe(via);
        expect(r.splits).toBe(splits);
        expect(r.partialFloats).toBe(prefillPartialFloats(opts));
        expect(r.wgStorageBytes).toBe(prefillGemmWorkgroupStorageBytes(opts, via));
      }
    });
  }

  it("le fette sono le 4 MISURATE su entrambe le shape di riga 1", () => {
    // confronto contro la costante ESPORTATA dal kernel, non contro il numero 4
    for (const { K, N } of SHAPES) {
      expect(planPrefillGemm({ kind: "q4_0", K, N, M, idot: true }).splits)
        .toBe(PREFILL_SPLITS_MEASURED);
    }
  });

  it("xq/xsc della via intera = il layout del kernel di quantizzazione", () => {
    // CONTROLLO INCROCIATO sul TESTO WGSL: `prefillQuantXQ8Wgsl` dichiara
    // `const BLOCKS = <M*bpr>u`, scrive 8 u32 per blocco (`xq[b * 8u + i]`) e
    // UNA scala per blocco (`xsc[b]`). Se il piano prenotasse meno di cosi',
    // l'ultimo blocco scriverebbe fuori dal buffer.
    for (const { K, N } of SHAPES) {
      const blocks = M * (K / 32);
      expect(prefillQuantXQ8Wgsl({ K, M })).toContain(`const BLOCKS = ${blocks}u;`);
      const r = planPrefillGemm({ kind: "q4_0", K, N, M, idot: true });
      expect(r.xqU32).toBe(blocks * 8);
      expect(r.xscF32).toBe(blocks);
      // via f32: nessuna attivazione quantizzata, quindi ZERO (non "piccolo")
      const f = planPrefillGemm({ kind: "q4_0", K, N, M, idot: false });
      expect(f.xqU32).toBe(0);
      expect(f.xscF32).toBe(0);
    }
  });
});

// ---------------------------------------------------------------------------
// (3) WORKGROUP STORAGE
// ---------------------------------------------------------------------------
describe("[3] workgroup storage: 1.152 B via intera, 4.096 B via f32, e il contesto non entra", () => {
  it(`a M=${M} sono 1.152 e 4.096 byte`, () => {
    const a = planPrefillGemm({ kind: "q4_0", K: 2560, N: 9216, M, idot: true });
    const b = planPrefillGemm({ kind: "q4_0", K: 2560, N: 9216, M, idot: false });
    expect(a.wgStorageBytes).toBe(1152);
    expect(b.wgStorageBytes).toBe(4096);
  });

  it("COSTANTE in ctxMax: il contesto non compare nella firma ne' nel numero", () => {
    // Lo storage del moltiplicatore dipende SOLO da M (M*16 u32 + M*2 f32 sulla
    // via intera): ne' K, ne' N, ne' il contesto lo muovono. E' la ragione per
    // cui la riga 4 (portabilita') ottiene il tetto gratis: un prompt piu'
    // lungo non alza mai questo numero. `ctxMax` non e' un argomento di
    // `planPrefillGemm` — e' proprio questo il punto.
    const KN = [
      { K: 2560, N: 9216 }, { K: 9216, N: 2560 },
      { K: 4096, N: 2560 }, { K: 896, N: 4864 },
    ];
    for (const { K, N } of KN) {
      expect(planPrefillGemm({ kind: "q4_0", K, N, M, idot: true }).wgStorageBytes,
        `K${K}xN${N}`).toBe(1152);
      expect(planPrefillGemm({ kind: "q4_0", K, N, M, idot: false }).wgStorageBytes,
        `K${K}xN${N}`).toBe(4096);
    }
    // solo M lo muove, e linearmente
    expect(planPrefillGemm({ kind: "q4_0", K: 2560, N: 9216, M: 8, idot: true }).wgStorageBytes).toBe(576);
  });
});

// ---------------------------------------------------------------------------
// (4) LA SHAPE CHE NON DIVIDE
// ---------------------------------------------------------------------------
describe("[4] K=896 (0.5B): le fette ripiegano a 1, senza throw", () => {
  it("28 blocchi non stanno in 4 fette da BK=2 ⇒ nessuno split-K, non un errore", () => {
    const K = 896, N = 4864;
    expect(K % 64).toBe(0);
    expect(() => planPrefillGemm({ kind: "q4_0", K, N, M, idot: true })).not.toThrow();
    const r = planPrefillGemm({ kind: "q4_0", K, N, M, idot: true });
    expect(r.via).toBe("idot");
    expect(r.splits).toBe(PREFILL_SPLITS_UNSPLIT);
    expect(r.splits).toBe(prefillGemmSplitsFor(K, N));
    expect(r.partialFloats).toBe(prefillPartialFloats({ kind: "q4_0", K, N, M, splits: r.splits }));
    expect(r.partialFloats).toBe(1 * M * N);
  });
});

// ---------------------------------------------------------------------------
// (5) CAPS
// ---------------------------------------------------------------------------
describe("[5] prefillGemmCapsFor: la feature decide, e il perche' non e' mai vuoto", () => {
  const src = (has: string[]): { wgslLanguageFeatures: { has(f: string): boolean } } =>
    ({ wgslLanguageFeatures: { has: (f: string) => has.includes(f) } });

  it("feature presente ⇒ idot true", () => {
    const c = prefillGemmCapsFor(src([PREFILL_IDOT_FEATURE]));
    expect(c.idot).toBe(true);
    expect(c.why.length).toBeGreaterThan(0);
    expect(c.why).toContain(PREFILL_IDOT_FEATURE);
  });

  it("feature assente ⇒ idot false con why non vuoto", () => {
    const c = prefillGemmCapsFor(src(["readonly_and_readwrite_storage_textures"]));
    expect(c.idot).toBe(false);
    expect(c.why.length).toBeGreaterThan(0);
    expect(c.why).toContain(PREFILL_IDOT_FEATURE);
  });

  it("sorgente SENZA wgslLanguageFeatures ⇒ false (non sapere ≠ avere)", () => {
    const c = prefillGemmCapsFor({});
    expect(c.idot).toBe(false);
    expect(c.why.length).toBeGreaterThan(0);
  });

  it("il nome della feature e' quello della language feature, non di un'estensione", () => {
    // it.5 e' costata una run intera: `enable packed_4x8_integer_dot_product;`
    // fa fallire la compilazione con «expected extension». La feature si legge
    // da `navigator.gpu.wgslLanguageFeatures`, non da `device.features`.
    expect(PREFILL_IDOT_FEATURE).toBe("packed_4x8_integer_dot_product");
  });
});

// ---------------------------------------------------------------------------
// (6) ACCETTAZIONE 2 DEL GOAL — in aritmetica pura, sui siti del 4B
// ---------------------------------------------------------------------------
//
// PROVENIENZA DELLA LISTA (non e' inventata e non e' dedotta dalla shape):
// header GGUF di `/home/neuromancer/.cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf`
// (426 tensori), riletto tensore per tensore il 2026-08-13; i totali per
// famiglia sono quelli gia' pinnati in
// `results/engine/q35-header-dump-2026-08-10.json` (`typeHistogram`), e il test
// [6b] li riverifica — n E byte — contro questa lista, cosi' la lista non puo'
// scivolare in silenzio. La struttura (32 layer, full quando l%4==3) e' quella
// che l'header dichiara: `attn_q/k/v/output` compaiono esattamente sui layer
// 3,7,...,31 e `attn_qkv/gate` + `ssm_*` sugli altri 24.
//
// Dimensioni GGUF = [K, N] (ne[0] = ingresso). Nessun sito per-layer e' q6_K —
// il q6_K del 4B sta solo in `token_embd`, che non e' un GEMM di prefill —
// quindi su tutti i kind di questa lista i byte device coincidono con quelli
// del file e il confronto con l'istogramma e' lecito.
const Q35_4B = { nLayer: 32, fullInterval: 4 } as const;

/** Siti FULL-attention (8 layer: l%4==3) — tutti Q4_0. */
const SITES_FULL: readonly { site: string; kind: PrefillQuantKind; K: number; N: number }[] = [
  { site: "attn_q", kind: "q4_0", K: 2560, N: 8192 },   // gate fuso: 2*nHead*headDim
  { site: "attn_k", kind: "q4_0", K: 2560, N: 1024 },   // nKvHead*headDim
  { site: "attn_v", kind: "q4_0", K: 2560, N: 1024 },
  { site: "attn_output", kind: "q4_0", K: 4096, N: 2560 },
];

/** Siti LINEAR-attention / Gated DeltaNet (24 layer). */
const SITES_LINEAR: readonly { site: string; kind: PrefillQuantKind; K: number; N: number }[] = [
  { site: "attn_qkv", kind: "q4_0", K: 2560, N: 8192 },  // (2*nK+nV)*hd
  { site: "attn_gate", kind: "q4_0", K: 2560, N: 4096 }, // nV*hd
  { site: "ssm_alpha", kind: "q8_0", K: 2560, N: 32 },
  { site: "ssm_beta", kind: "q8_0", K: 2560, N: 32 },
  { site: "ssm_out", kind: "q5_K", K: 4096, N: 2560 },   // Q5_K nel file, non Q4_0
];

/** FFN densa su OGNI layer; `ffn_down` e' Q4_1 sui layer 0..3, Q4_0 altrove. */
const FFN_Q41_LAYERS = [0, 1, 2, 3] as const;

function sites4B(): PrefillSite[] {
  const out: PrefillSite[] = [];
  for (let l = 0; l < Q35_4B.nLayer; l++) {
    const full = l % Q35_4B.fullInterval === Q35_4B.fullInterval - 1;
    for (const s of full ? SITES_FULL : SITES_LINEAR) out.push({ ...s, site: `blk.${l}.${s.site}` });
    out.push({ site: `blk.${l}.ffn_gate`, kind: "q4_0", K: 2560, N: 9216 });
    out.push({ site: `blk.${l}.ffn_up`, kind: "q4_0", K: 2560, N: 9216 });
    out.push({
      site: `blk.${l}.ffn_down`,
      kind: (FFN_Q41_LAYERS as readonly number[]).includes(l) ? "q4_1" : "q4_0",
      K: 9216, N: 2560,
    });
  }
  return out;
}

/**
 * `typeHistogram` di results/engine/q35-header-dump-2026-08-10.json, entry
 * Qwen3.5-4B-Q4_0.gguf, limitato alle famiglie di PESI LINEARI per layer (le
 * F32 di norm/conv e il `token_embd` Q6_K non sono GEMM di prefill).
 */
const HIST_4B = {
  attnQ4_0: { n: 80, bytes: 589_824_000 },
  ffnQ4_0: { n: 92, bytes: 1_220_935_680 },
  ffnQ4_1: { n: 4, bytes: 58_982_400 },
  ssmQ8_0: { n: 48, bytes: 4_177_920 },
  ssmQ5_K: { n: 24, bytes: 173_015_040 },
} as const;

const sum = (ds: PrefillDispatch[]): number => ds.reduce((a, d) => a + dispatchWeightBytes(d), 0);

/** Byte di UNA passata sui pesi (M=1): il "peso del tensore" visto dal kernel. */
const onePass = (ss: PrefillSite[]): number =>
  sum(ss.map((s) => ({ form: "multirow" as const, kind: s.kind, K: s.K, N: s.N, M: 1 })));

describe("[6] ACCETTAZIONE 2: il traffico pesi del prefill del 4B, prima e dopo", () => {
  it("[6a] la lista pinnata ha la forma dell'header (248 siti lineari per layer)", () => {
    const s = sites4B();
    expect(s.length).toBe(8 * 4 + 24 * 5 + 32 * 3);   // 32 + 120 + 96 = 248
    expect(s.filter((x) => x.site.endsWith("ssm_out")).length).toBe(24);
    expect(s.filter((x) => x.kind === "q4_1").length).toBe(4);
    expect(s.filter((x) => x.kind === "q8_0").length).toBe(48);
  });

  it("[6b] n E byte per famiglia sono quelli del typeHistogram gia' pinnato", () => {
    // Qui la stessa somma esce dalla lista dei SITI: se un sito cambia shape,
    // kind o conteggio, questi dieci numeri non tornano piu'.
    const isAttn = (s: PrefillSite): boolean => s.site.includes(".attn_");
    const isFfn = (s: PrefillSite): boolean => s.site.includes(".ffn_");
    const isSsm = (s: PrefillSite): boolean => s.site.includes(".ssm_");
    const fam: [string, (s: PrefillSite) => boolean, { n: number; bytes: number }][] = [
      ["attn:Q4_0", (s) => isAttn(s) && s.kind === "q4_0", HIST_4B.attnQ4_0],
      ["ffn:Q4_0", (s) => isFfn(s) && s.kind === "q4_0", HIST_4B.ffnQ4_0],
      ["ffn:Q4_1", (s) => isFfn(s) && s.kind === "q4_1", HIST_4B.ffnQ4_1],
      ["ssm:Q8_0", (s) => isSsm(s) && s.kind === "q8_0", HIST_4B.ssmQ8_0],
      ["ssm:Q5_K", (s) => isSsm(s) && s.kind === "q5_K", HIST_4B.ssmQ5_K],
    ];
    for (const [name, pred, want] of fam) {
      const got = sites4B().filter(pred);
      expect(got.length, name).toBe(want.n);
      expect(onePass(got), name).toBe(want.bytes);
    }
  });

  it("[6c] il rapporto legacy/piano a M=16 — misurato, stampato, e pinnato dove puo' fallire", () => {
    const sites = sites4B();
    const { dispatches, legacy, exceptions } = prefillPlanDispatches({ sites, M, idot: true });
    expect(dispatches.length).toBe(sites.length);
    expect(legacy.length).toBe(sites.length);
    // il controfattuale e' la STESSA sequenza di shape, contata a form legacy
    for (let i = 0; i < sites.length; i++) {
      expect(legacy[i].form).toBe("legacy");
      expect(legacy[i].K).toBe(sites[i].K);
      expect(legacy[i].N).toBe(sites[i].N);
      expect(legacy[i].kind).toBe(sites[i].kind);
      expect(legacy[i].M).toBe(M);
    }

    const before = sum(legacy);
    const after = sum(dispatches);
    const ratio = before / after;

    const covered = sites.filter((_, i) => dispatches[i].form === "multirow");
    const excepted = sites.filter((_, i) => dispatches[i].form === "legacy");

    console.log(
      `[ACCETTAZIONE 2] Qwen3.5-4B, M=${M}: legacy ${(before / 1e9).toFixed(3)} GB → piano ` +
      `${(after / 1e9).toFixed(3)} GB = ${ratio.toFixed(4)}x sull'inventario per-layer INTERO ` +
      `(barra della riga 2 di engine-kquant: >= 10,9x; teorico massimo M=${M}); ` +
      `copertura ${covered.length}/${sites.length} siti = ` +
      `${(100 * onePass(covered) / onePass(sites)).toFixed(3)}% dei byte; ` +
      `${exceptions.length} eccezioni = ${(100 * onePass(excepted) / onePass(sites)).toFixed(3)}% dei byte`);

    // Le 24 `ssm_out` Q5_K sono la riga 2 di engine-kquant: erano il 37,9% del
    // tempo del prefill sul fallback legacy, e ora stanno fra i `multirow`.
    const ssmOut = sites.map((s, i) => ({ s, d: dispatches[i] })).filter((x) => x.s.site.endsWith("ssm_out"));
    expect(ssmOut.length).toBe(24);
    for (const { s, d } of ssmOut) {
      expect(s.kind, s.site).toBe("q5_K");
      expect(d.form, s.site).toBe("multirow");
    }
    // ...e nessuna di loro compare piu' fra le eccezioni
    expect(exceptions.some((e) => e.site.endsWith("ssm_out"))).toBe(false);

    // -----------------------------------------------------------------------
    // IL GATE, e cosa lo fa fallire.
    //
    // NON e' `covRatio >= 8`: il rapporto sui soli dispatch multirow vale M per
    // COSTRUZIONE di `dispatchWeightBytes` (legacy = M·N·perRow, multirow =
    // N·perRow), quindi resterebbe 16 anche se la copertura crollasse da 172
    // siti a uno solo. Un'asserzione che non puo' fallire non e' un gate.
    //
    // Il gate vero e' l'identita' fra il piano ENUMERATO (248 dispatch usciti
    // da `prefillPlanDispatches`) e la forma chiusa costruita sui BYTE PINNATI
    // dell'header: F = byte coperti dalla via veloce, L = byte rimasti legacy.
    //   prima = M·(F+L)   ogni sito riletto M volte
    //   dopo  = F + M·L   i coperti UNA volta, le eccezioni ancora M
    // Se un sito coperto scivolasse a legacy, `after` crescerebbe e queste due
    // uguaglianze fallirebbero; se un'eccezione sparisse dalla lista, anche.
    // -----------------------------------------------------------------------
    // RIGA 3: il q4_1 passa da L a F. Restano legacy i soli 48 siti Q8_0
    // `ssm_alpha`/`ssm_beta` — 0,204% dei byte, N=32 righe di uscita contro le
    // 64 per workgroup della forma split-K: mezzo workgroup per dispatch,
    // ESCLUSI COI NUMERI e non dimenticati (v. GOAL.md).
    const F = HIST_4B.attnQ4_0.bytes + HIST_4B.ffnQ4_0.bytes + HIST_4B.ssmQ5_K.bytes
      + HIST_4B.ffnQ4_1.bytes;   // via veloce
    const L = HIST_4B.ssmQ8_0.bytes;
    expect(F).toBe(2_042_757_120);
    expect(L).toBe(4_177_920);
    expect(before).toBe(M * (F + L));
    expect(after).toBe(F + M * L);
    expect(covered.length).toBe(
      HIST_4B.attnQ4_0.n + HIST_4B.ffnQ4_0.n + HIST_4B.ssmQ5_K.n + HIST_4B.ffnQ4_1.n);   // 200
    expect(covered.length).toBe(200);
    expect(excepted.length).toBe(HIST_4B.ssmQ8_0.n);   // 48
    expect(excepted.length).toBe(48);
    expect(exceptions.length).toBe(excepted.length);
    expect(onePass(covered)).toBe(F);
    expect(onePass(excepted)).toBe(L);
    for (const e of exceptions) expect(e.reason.length).toBeGreaterThanOrEqual(40);

    // RESTA UN SOLO KIND LEGACY, e non e' un residuo dimenticato: i 48 siti
    // Q8_0 `ssm_alpha`/`ssm_beta` sono lo 0,204% dei byte con N=32 righe di
    // uscita, cioe' mezzo workgroup per dispatch sulla forma split-K. Sono
    // ESCLUSI COI NUMERI dal contratto. Il Q5_K se n'e' andato con la riga 2,
    // il Q4_1 con la riga 3.
    expect([...new Set(exceptions.map((e) => e.kind))].sort()).toEqual(["q8_0"]);

    // e il numero che ne esce, pinnato stretto perche' si legga nel goal.
    // La barra della riga 3 e' >= 15,5x sull'inventario per-layer INTERO
    // (era >= 10,9 dopo la riga 2, e 5,8593 prima del goal).
    expect(ratio).toBeGreaterThanOrEqual(15.5);
    expect(ratio).toBeCloseTo(15.5247, 4);
  });

  // [6d] NON C'E' PIU', ed e' la sua stessa ragione ad averlo tolto. Registrava
  // la DISTANZA fra il testo originale del contratto (>= 8) e la barra del
  // ruling PI del 2026-08-13 (>= 5,5), e diceva a chiare lettere: «se un giorno
  // la copertura sale, questo test fallisce e va cancellato con la sua
  // ragione». La copertura e' salita — le 24 `ssm_out` Q5_K sono passate alla
  // via veloce e il rapporto e' 10,94x, cioe' oltre l'8 che [6d] asseriva
  // essere irraggiungibile. Cancellarlo e' eseguire la sua istruzione; tenerlo
  // sarebbe stato un `expect(ratio).toBeLessThan(8)` rosso su un miglioramento.
  // L'identita' `covRatio === M` che portava con se' era esplicitamente una
  // definizione e non una misura, e resta sorvegliata da [6c].

  it("[6e] senza la feature intera il traffico non cambia: la via f32 muove gli stessi byte", () => {
    // Il fallback dichiarato costa in TEMPO (1,745x), non in byte: la forma
    // resta multi-riga, quindi il rapporto e' identico. Se un giorno divergesse
    // vorrebbe dire che il fallback non e' piu' la stessa forma.
    const sites = sites4B();
    const a = prefillPlanDispatches({ sites, M, idot: true });
    const b = prefillPlanDispatches({ sites, M, idot: false });
    expect(sum(b.dispatches)).toBe(sum(a.dispatches));
    expect(b.exceptions.length).toBe(a.exceptions.length);
  });
});

// ---------------------------------------------------------------------------
// (7) SCRATCH CONDIVISO
// ---------------------------------------------------------------------------
describe("[7] prefillGemmScratchFor: UN solo set di buffer, quindi il MAX", () => {
  it("il max sulle shape, non la somma", () => {
    const sites: PrefillSite[] = [
      { site: "ffn_gate", kind: "q4_0", K: 2560, N: 9216 },
      { site: "ffn_down", kind: "q4_0", K: 9216, N: 2560 },
      { site: "attn_gate", kind: "q4_0", K: 2560, N: 4096 },
    ];
    const got = prefillGemmScratchFor({ sites, M, idot: true });
    const each = sites.map((s) => planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M, idot: true }));
    const maxPart = Math.max(...each.map((r) => r.partialFloats));
    const sumPart = each.reduce((a, r) => a + r.partialFloats, 0);
    expect(got.partialFloats).toBe(maxPart);
    expect(got.partialFloats).toBeLessThan(sumPart);
    expect(got.xqU32).toBe(Math.max(...each.map((r) => r.xqU32)));
    expect(got.xscF32).toBe(Math.max(...each.map((r) => r.xscF32)));
    // il piu' grande dei tre parziali e' quello di ffn_gate: 4 fette · 16 · 9216
    expect(got.partialFloats).toBe(PREFILL_SPLITS_MEASURED * M * 9216);
    // xq/xsc scalano con K, non con N: il massimo e' K=9216
    expect(got.xqU32).toBe(M * (9216 / 32) * 8);
    expect(got.xscF32).toBe(M * (9216 / 32));
  });

  it("le eccezioni non prenotano nulla e non alzano il max", () => {
    const fast: PrefillSite[] = [{ site: "ffn_gate", kind: "q4_0", K: 2560, N: 9216 }];
    // Le eccezioni di OGGI, dopo la riga 3: ne' il Q5_K ne' il Q4_1 sono piu'
    // fra loro. Restano i K-quant del 35B (misurati e NON cablati) e il Q8_0.
    // Il q4_1 e' stato tolto da questa lista quando ha smesso di essere
    // un'eccezione: lasciarlo avrebbe reso il test verde misurando il passato.
    const mixed: PrefillSite[] = [
      ...fast,
      { site: "moe_down", kind: "q4_K", K: 4096, N: 2560 },
      { site: "ssm_alpha", kind: "q8_0", K: 2560, N: 32 },
    ];
    expect(prefillGemmScratchFor({ sites: mixed, M, idot: true }))
      .toEqual(prefillGemmScratchFor({ sites: fast, M, idot: true }));
  });

  it("sull'intero 4B lo scratch e' quello della shape peggiore, non di 248 siti", () => {
    const got = prefillGemmScratchFor({ sites: sites4B(), M, idot: true });
    expect(got.partialFloats).toBe(PREFILL_SPLITS_MEASURED * M * 9216);   // 589.824 f32 = 2,36 MB
    expect(got.xqU32).toBe(M * (9216 / 32) * 8);
    expect(got.xscF32).toBe(M * (9216 / 32));
    // via f32: niente attivazioni quantizzate, ma gli stessi parziali
    const f = prefillGemmScratchFor({ sites: sites4B(), M, idot: false });
    expect(f.partialFloats).toBe(got.partialFloats);
    expect(f.xqU32).toBe(0);
    expect(f.xscF32).toBe(0);
  });
});
