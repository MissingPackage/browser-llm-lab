// IL PIANO NON INSTRADA LE TRE FORME DELLA RIGA 4 (q4_K, q6_K, q8_0).
//
// La tesi di questo file e' UNA, ed e' il contrario di quella degli altri test
// di port: che un kernel esista, sia portato dal banco byte per byte e sia
// misurato NON deve bastare a farlo eseguire in produzione. Le tre famiglie del
// 35B nascono qui — la riga 4 del goal engine-kquant le consegna al goal
// successivo — e finche' nessuno le cabla il 4B deve emettere ESATTAMENTE i
// dispatch di ieri.
//
// PERCHE' SERVE UN TEST E NON BASTA IL BUON SENSO. `prefillGemmCheck`
// (kernels/wgsl.ts) controlla kind, K e fette: NON GUARDA N. Il 4B ha 48 siti
// q8_0 (`ssm_alpha`/`ssm_beta`) con K=2560 e N=32, cioe' una shape che il
// contorno del kernel accetta senza battere ciglio — e con N=32 su una forma che
// produce 64 righe di uscita per workgroup, ogni dispatch userebbe MEZZO
// workgroup. Quei siti sono lo 0,204% dei byte del prefill ed erano gia'
// «esclusi coi numeri» dal contratto della riga 3. Senza il flag `wired` il
// solo fatto di aver portato il kernel q8_0 li avrebbe instradati, cambiando in
// silenzio cio' che il 4B esegue: questo file e' il sensore di quel cambio.
//
// COME FALLISCE (ed e' il modo in cui deve fallire): chi mette `wired: true` su
// uno dei tre senza fare il resto del lavoro — misura fresca, ktest, cablaggio
// del sito, guardia sul formato dove si EMETTE — trova rosso qui, e non su un
// numero astratto: sulla shape concreta dei 48 siti del 4B.
import { describe, expect, it } from "vitest";
import { readFileSync, globSync } from "node:fs";
import { join, relative } from "node:path";
import {
  planPrefillGemm, prefillPlanDispatches, type PrefillSite,
} from "../src/engine/prefillgemmplan";
import {
  PREFILL_GEMM_KINDS, PREFILL_GEMM_WIRED_KINDS, PREFILL_SPLITS_UNSPLIT,
  prefillGemmWiring, prefillPartialFloats,
  prefillGemmQ4KSplitKIdotWgsl, prefillGemmQ6KSplitKIdotWgsl, prefillGemmQ80SplitKIdotWgsl,
  type PrefillGemmKind,
} from "../src/engine/kernels/wgsl";
import type { PrefillQuantKind } from "../src/engine/prefillbytes";

/** M del prefill in produzione (PREFILL_M = 16). */
const M = 16;

/** I tre formati che la riga 4 porta SENZA cablare. */
const NOT_WIRED = ["q4_K", "q6_K", "q8_0"] as const;

/**
 * Per ognuno, una shape che il CONTORNO DEL KERNEL ACCETTA. E' la condizione
 * che rende questo file un test e non una tautologia: se la shape fosse
 * rifiutata dalla geometria, "resta legacy" sarebbe vero per un'altra ragione e
 * il flag non verrebbe esercitato affatto. Sono le shape misurate in fase 0
 * (`KQUANT_SHAPES` in src/microbench/ttKQuant.ts).
 */
const ACCEPTED: { kind: PrefillGemmKind; K: number; N: number; what: string }[] = [
  { kind: "q4_K", K: 2048, N: 512, what: "35B expert gate/up (117 tensori = 17,67 GB)" },
  { kind: "q4_K", K: 512, N: 2048, what: "35B expert down (DUE superblocchi per riga)" },
  { kind: "q6_K", K: 512, N: 2048, what: "35B expert down di 3 layer" },
  { kind: "q8_0", K: 2048, N: 4096, what: "35B attn q-proj (100 tensori = 1,09 GB)" },
];

/**
 * IL CASO CONCRETO che il contratto nomina: `ssm_alpha`/`ssm_beta` del 4B.
 * 48 siti, q8_0, K=2560 (80 blocchi da 32) e N=32 — mezzo workgroup per
 * dispatch sulla forma split-K a 64 righe di uscita.
 */
const SSM_Q80 = { kind: "q8_0" as const, K: 2560, N: 32 };

describe("[w1] il flag: sei kernel, tre instradati, e ogni non-cablato dice perche'", () => {
  it("PREFILL_GEMM_WIRED_KINDS e' un SOTTOINSIEME PROPRIO dell'elenco dei kernel", () => {
    expect([...PREFILL_GEMM_KINDS]).toEqual(["q4_0", "q5_K", "q4_1", "q4_K", "q6_K", "q8_0"]);
    expect([...PREFILL_GEMM_WIRED_KINDS]).toEqual(["q4_0", "q5_K", "q4_1"]);
    // e i non-cablati sono esattamente i tre della riga 4
    const notWired = PREFILL_GEMM_KINDS.filter((k) => !prefillGemmWiring(k).wired);
    expect([...notWired]).toEqual([...NOT_WIRED]);
  });

  it("ogni ragione di cablaggio e' una ragione (>= 60 caratteri), su TUTTI i kind", () => {
    // Su un `wired: false` la ragione e' quella che finisce in telemetria
    // quando un sito resta legacy; su un `wired: true` e' la misura che
    // giustifica il cablaggio. Un booleano nudo non e' diagnosticabile.
    for (const k of PREFILL_GEMM_KINDS) {
      const w = prefillGemmWiring(k);
      expect(w.why.length, `${k}: ragione troppo corta ("${w.why}")`).toBeGreaterThanOrEqual(60);
    }
  });

  it("la ragione del q8_0 NOMINA il fatto che lo esclude: N=32 sul 4B", () => {
    // Non e' una preferenza di gusto: e' il numero del contratto. Se un giorno
    // la ragione diventasse generica, si perderebbe l'unica cosa che permette
    // di ridiscutere l'esclusione con i dati in mano.
    const why = prefillGemmWiring("q8_0").why;
    expect(why).toContain("N=32");
    expect(why).toContain("0,204%");
  });

  it("le ragioni di q4_K e q6_K dicono che il 4B non ne ha e che il cablaggio e' il goal dopo", () => {
    for (const k of ["q4_K", "q6_K"] as const) {
      expect(prefillGemmWiring(k).why, k).toContain("NON CABLATO");
      expect(prefillGemmWiring(k).why, k).toContain("35B");
    }
  });
});

describe("[w2] il piano NON instrada i tre kind nuovi, e la shape sarebbe accettabile", () => {
  for (const { kind, K, N, what } of ACCEPTED) {
    it(`${kind} K${K}xN${N} (${what}): il KERNEL accetta, il PIANO resta legacy`, () => {
      // (a) IL SENSORE: il contorno del kernel non ha niente da ridire su
      //     questa shape. Senza questa riga il test non distinguerebbe "non
      //     cablato" da "geometria sbagliata".
      expect(() => prefillPartialFloats({ kind, K, N, M, splits: PREFILL_SPLITS_UNSPLIT }))
        .not.toThrow();
      // (b) IL FATTO: il piano rende legacy comunque, su ENTRAMBE le vie.
      for (const idot of [true, false]) {
        const r = planPrefillGemm({ kind, K, N, M, idot });
        expect(r.via, `${kind} idot=${idot}: ${r.reason}`).toBe("legacy");
        // (c) LA RAGIONE: dice il kind, dice che e' il cablaggio a mancare, e
        //     NON accusa la geometria (che qui e' buona).
        expect(r.reason, `${kind}`).toContain(kind);
        expect(r.reason, `${kind}`).toContain("cablat");
        expect(r.reason.length, `${kind}`).toBeGreaterThanOrEqual(40);
        expect(r.reason, `${kind}: la ragione non deve accusare la geometria`)
          .not.toContain("non accetta questa shape");
        // (d) NIENTE SCRATCH: una via legacy che prenotasse buffer gonfierebbe
        //     il massimo condiviso per dispatch che nessuno emette.
        expect(r.splits).toBe(0);
        expect(r.partialFloats).toBe(0);
        expect(r.wgStorageBytes).toBe(0);
        expect(r.xqU32).toBe(0);
        expect(r.xscF32).toBe(0);
      }
    });
  }

  it("il sensore vale: sulle STESSE identiche condizioni un kind cablato viene instradato", () => {
    // Se il piano rendesse legacy per un motivo qualunque (un bug in
    // `planPrefillGemm`, un M sbagliato), tutti i casi qui sopra sarebbero verdi
    // e vuoti. Questa riga lo esclude: stessa funzione, stesso M, un kind
    // `wired` con una shape accettabile.
    expect(planPrefillGemm({ kind: "q5_K", K: 512, N: 2048, M, idot: true }).via).toBe("idot");
    expect(planPrefillGemm({ kind: "q4_0", K: 2048, N: 4096, M, idot: true }).via).toBe("idot");
    expect(planPrefillGemm({ kind: "q4_1", K: 2048, N: 4096, M, idot: false }).via).toBe("f32");
  });
});

describe("[w3] il caso concreto: i 48 siti q8_0 del 4B restano legacy", () => {
  it("ssm_alpha/ssm_beta (q8_0 K=2560 N=32): il kernel accetta la shape, N compreso", () => {
    // IL PUNTO DI TUTTO IL MECCANISMO, in una riga: `prefillGemmCheck` non
    // guarda N. Su questa shape non solleva — e la forma split-K produce 64
    // righe di uscita per workgroup contro le 32 che servono.
    //
    // AGGIORNAMENTO (goal engine-velocita-decode, riga 2d): resta vero che il
    // CONTORNO DEL KERNEL non guarda N, ed e' giusto cosi' — il kernel a N=32
    // e' corretto, e le query di dimensionamento vanno interrogabili su
    // qualunque shape. Cio' che e' cambiato e' che ora il PIANO guarda N
    // (`kernelVerdict`, `PREFILL_GEMM_ROWS_PER_WG`), quindi questi 48 siti sono
    // esclusi DUE volte: dal flag di famiglia e dalla shape. La seconda e'
    // quella che li terra' fuori quando il q8_0 verra' cablato per il 35B.
    // Il predicato ha i suoi casi in `tests/engine-prefillgemm-nmin.test.ts`.
    expect(() => prefillPartialFloats({ ...SSM_Q80, M, splits: PREFILL_SPLITS_UNSPLIT }))
      .not.toThrow();
    expect(SSM_Q80.N).toBeLessThan(64);
  });

  it("...e il piano li lascia comunque sulla legacy, con la sua ragione", () => {
    for (const idot of [true, false]) {
      const r = planPrefillGemm({ ...SSM_Q80, M, idot });
      expect(r.via, `idot=${idot}: ${r.reason}`).toBe("legacy");
      expect(r.reason).toContain("q8_0");
    }
  });

  it("sui 48 siti veri il piano emette 48 dispatch legacy, non uno di meno", () => {
    // La forma in cui il difetto si vedrebbe davvero: non un kind astratto, ma
    // i siti del 4B come li enumera il piano.
    const sites: PrefillSite[] = [];
    for (let l = 0; l < 32; l++) {
      if (l % 4 === 3) continue;                       // i layer full-attention non hanno ssm_*
      for (const s of ["ssm_alpha", "ssm_beta"]) {
        sites.push({ site: `blk.${l}.${s}`, kind: SSM_Q80.kind, K: SSM_Q80.K, N: SSM_Q80.N });
      }
    }
    expect(sites.length, "24 layer DeltaNet x 2 siti").toBe(48);
    const { dispatches, exceptions } = prefillPlanDispatches({ sites, M, idot: true });
    expect(dispatches.every((d) => d.form === "legacy")).toBe(true);
    expect(exceptions.length).toBe(48);
    for (const e of exceptions) {
      expect(e.kind).toBe("q8_0");
      expect(e.reason.length).toBeGreaterThanOrEqual(40);
    }
  });
});

// ---------------------------------------------------------------------------
// [w4] GATE STRUTTURALE: il flag si consuma in UN POSTO SOLO.
//
// Stessa postura di tests/gpudevice.test.ts e engine-one-mechanism.test.ts: non
// si elencano i siti buoni (lista che marcisce), si intercetta l'ATTO — leggere
// il cablaggio — e si pretende che avvenga solo dove e' dichiarato. Un secondo
// posto che decide chi e' instradato e' la stessa classe di difetto di it.7
// (due sedi che stabilivano le righe-per-workgroup, e le stabilivano diverse).
// ---------------------------------------------------------------------------
describe("[w4] `prefillGemmWiring` si legge solo nel piano", () => {
  const ROOT = join(__dirname, "..");
  const SRC = globSync(join(ROOT, "src/**/*.{ts,mts,cts,tsx}"))
    .filter((f) => !f.endsWith(".d.ts"))
    .map((f) => relative(ROOT, f))
    .sort();

  /** dove e' LECITO leggere il flag, e perche' */
  const ALLOWED: Record<string, string> = {
    "src/engine/kernels/wgsl.ts": "e' la sede del flag: qui `prefillGemmWiring` viene DEFINITA",
    "src/engine/prefillgemmplan.ts": "l'unico consumatore previsto: e' il posto dove si decide la via di un sito",
  };

  it("la scansione non e' vuota (senza questo guard un gate rotto e' verde)", () => {
    expect(SRC.length).toBeGreaterThan(30);
  });

  it("nessun secondo predicato di cablaggio fuori dal piano", () => {
    const hits = SRC.filter((f) => /prefillGemmWiring|PREFILL_GEMM_WIRED_KINDS/
      .test(readFileSync(join(ROOT, f), "utf8")));
    expect(hits.filter((f) => !(f in ALLOWED)),
      "chi vuole sapere se un formato e' instradato lo chiede al piano, non al flag").toEqual([]);
    // e i due siti dichiarati ci sono davvero: un'allowlist che copre file
    // inesistenti non prova niente
    expect(hits.sort()).toEqual(Object.keys(ALLOWED).sort());
  });

  it("il wiring del 4B non nomina i tre formati non cablati", () => {
    // La guardia doppia del cablaggio (`route.via !== "legacy" && kk === "…"`)
    // e' cio' che ha intercettato un caso vero in riga 3. Qui si verifica il
    // suo presupposto: in q35gpumodel.ts non esiste un ramo che emetta uno dei
    // tre kernel nuovi. Se un giorno ci sara', sara' perche' qualcuno ha
    // cablato — e allora questo test va cambiato con la sua ragione, non
    // aggirato.
    const model = readFileSync(join(ROOT, "src/engine/q35gpumodel.ts"), "utf8");
    for (const gen of ["prefillGemmQ4KSplitK", "prefillGemmQ6KSplitK", "prefillGemmQ80SplitK"]) {
      expect(model, `${gen} non deve comparire nel motore finche' non e' cablato`)
        .not.toContain(gen);
    }
  });

  it("i tre kernel esistono davvero (non e' 'non instradato' perche' non c'e')", () => {
    // L'altra meta' della tesi. Il testo riga-per-riga contro il banco sta in
    // tests/engine-prefillgemm.test.ts; qui basta che generino.
    expect(prefillGemmQ4KSplitKIdotWgsl({ kind: "q4_K", K: 2048, N: 512, M, splits: 4 }))
      .toContain("dot4I8Packed");
    expect(prefillGemmQ6KSplitKIdotWgsl({ kind: "q6_K", K: 512, N: 2048, M, splits: 2 }))
      .toContain("dot4I8Packed");
    expect(prefillGemmQ80SplitKIdotWgsl({ kind: "q8_0", K: 2048, N: 4096, M, splits: 4 }))
      .toContain("dot4I8Packed");
  });
});

describe("[w5] M=1 e i kind non cablati non si confondono", () => {
  it("a M=16 il rifiuto e' del CABLAGGIO, a M=1 e' dell'M — e le ragioni lo dicono", () => {
    // Due clausole diverse che producono lo stesso `legacy`: se le ragioni si
    // confondessero, in telemetria non si distinguerebbe piu' un formato non
    // cablato da un chunk di una riga sola.
    const kinds: PrefillQuantKind[] = [...NOT_WIRED];
    for (const kind of kinds) {
      const wiring = planPrefillGemm({ kind, K: 2048, N: 4096, M: 16, idot: true });
      expect(wiring.reason).toContain("cablat");
      expect(wiring.reason).not.toContain("M=1");
      const m1 = planPrefillGemm({ kind, K: 2048, N: 4096, M: 1, idot: true });
      expect(m1.reason).toContain("M=1");
      expect(m1.reason).not.toContain("wired");
    }
  });
});
