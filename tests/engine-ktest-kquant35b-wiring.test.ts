// Cablaggio dei TRE banchi della riga 4 del ktest: q4_K, q6_K, q8_0.
//
// Test STATICO, senza GPU: stessa postura di `tests/engine-ktest-q5k-wiring.test.ts`
// e `tests/engine-ktest-q41-wiring.test.ts` — si SCANSIONA il sorgente
// dell'orchestratore invece di fidarsi di una descrizione a parole. Qui si prova
// il CABLAGGIO dei tre banchi nuovi, non la loro aritmetica: che i sei casi
// esistano e siano registrati una volta sola, che ogni braccio chiami i suoi
// generatori con le liste di binding congelate e con le GRIGLIE degli helper,
// che le opzioni portino il `kind` del formato, e che shape/seed/tolleranze
// arrivino da `prefillkquant` invece di essere ricopiati a mano qui dentro.
//
// UN FILE PER TRE FORMATI, e non e' la stessa scelta del floor test. Li' i
// blocchi restano scritti a mano perche' cio' che si confronta sono NUMERI —
// pavimenti che si somigliano abbastanza da tentare la ricopiatura, e un blocco
// anti-ricopiatura che li mette uno accanto all'altro. Qui invece cio' che si
// congela e' una LISTA DI NOMI per formato, e tre liste in tabella si leggono
// meglio di tre file identici a meno di una lettera: la differenza che conta —
// il q8_0 ha DUE buffer di pesi e quindi cinque/quattro binding dove i due
// K-quant ne hanno quattro/tre — sta scritta in colonna, dove si vede.
//
// LE TRE FORME NON SONO CABLATE IN PRODUZIONE (`wired: false` in
// `PREFILL_GEMM_SPEC`, sorvegliato da tests/engine-prefillgemmplan-notwired.test.ts):
// questo file NON le cabla e non prova che qualcosa ci passi. Prova che il
// BANCO le esercita, che e' l'unico modo perche' una forma non instradata sia
// comunque verificata sul device.
//
// Il verdetto NUMERICO non e' di questo test: lo da' `node
// tools/harness/engine-ktest.mjs`, che vuole un vite server e un Chrome vero.
// Un ktest dichiarato verde senza averlo eseguito e' il difetto piu' grave
// registrato in questo repo: questo file non lo simula, verifica solo che i
// banchi siano agganciati dove il driver li trovera'.
//
// Il file si LEGGE come testo. Importarlo tirerebbe dentro i tipi WebGPU e un
// modulo worker (che in vitest node non si carica). Il testo non si scansiona
// grezzo: commenti, stringhe e template WGSL vengono prima BIANCATI
// (`blankNonCode` in `tests/helpers/source-scan.ts`, dove lo scanner e' stato
// FATTORIZZATO dopo esserne diventato la quarta copia). Senza quel passo un
// commento che nomina `prefillSplitKCombineWgsl(` sposta i conteggi, e una lista
// `[blocks, xq, part, xsc]` citata in un commento diventa la lista che il test
// misura al posto di quella vera.
import { describe, expect, it } from "vitest";
import {
  raw, code, uncommented, lineOf, hitsOf, closeOf, callsTo, topLevelItems,
  importsFrom, has, helperRange, helperBody, literalHits,
} from "./helpers/source-scan";

const KTEST = "src/engine/ktest/ktest.worker.ts";
const DRIVER = "tools/harness/engine-ktest.mjs";

/**
 * Un banco della riga 4, con tutto cio' che di lui e' CONGELATO.
 *
 * Le liste di binding sono quelle che dichiarano i generatori in
 * kernels/wgsl.ts, e sono la ragione per cui questa tabella esiste: un banco
 * che ereditasse le liste del gemello sbagliato compilerebbe e poi legherebbe
 * le scale dove il kernel aspetta le attivazioni.
 */
interface Bank {
  /** come si chiama il formato nel `kind` delle opzioni */
  kind: string;
  /** la funzione del banco nel worker */
  fn: string;
  /** il caso in prefillkquant.ts da cui escono shape e seed */
  caseConst: string;
  /** i due nomi di caso che il driver vedra' */
  idotCase: string;
  f32Case: string;
  /** i due moltiplicatori, gia' in kernels/wgsl.ts */
  idotGen: string;
  f32Gen: string;
  /** le due liste di binding congelate */
  idotBindings: string[];
  f32Bindings: string[];
  /** le due coppie di tolleranze */
  idotTols: string[];
  f32Tols: string[];
  /**
   * Da dove escono i pesi sul device. I K-quant restano in UN buffer
   * (`repackKQuant`, il superblocco GGUF copiato in parole); il q8_0 si spezza
   * in due (`repackQ8_0` → qs + scales) come il q4_0 e il q4_1. Se un banco
   * ri-impaccasse i byte a modo suo, proverebbe un kernel che in produzione non
   * gira — e le liste di binding sopra sarebbero soddisfatte da variabili
   * qualsiasi che si chiamano come i buffer veri.
   */
  repack: string;
  /** i campi che `repack` deve destrutturare, o null se rende un array solo */
  repackFields: string | null;
}

const BANKS: Bank[] = [
  {
    kind: "q4_K",
    fn: "testPrefillGemmQ4KMultiRow",
    caseConst: "PREFILL_Q4K_KTEST_CASE",
    idotCase: "prefill-gemm-q4k-multirow-idot",
    f32Case: "prefill-gemm-q4k-multirow-f32",
    idotGen: "prefillGemmQ4KSplitKIdotWgsl",
    f32Gen: "prefillGemmQ4KSplitKWgsl",
    idotBindings: ["blocks", "xq", "part", "xsc"],
    f32Bindings: ["blocks", "x", "part"],
    idotTols: ["PREFILL_GEMM_Q4K_IDOT_REL_TOL", "PREFILL_GEMM_Q4K_IDOT_ABS_TOL"],
    f32Tols: ["PREFILL_GEMM_Q4K_F32_REL_TOL", "PREFILL_GEMM_Q4K_F32_ABS_TOL"],
    repack: "repackKQuant",
    repackFields: null,
  },
  {
    kind: "q6_K",
    fn: "testPrefillGemmQ6KMultiRow",
    caseConst: "PREFILL_Q6K_KTEST_CASE",
    idotCase: "prefill-gemm-q6k-multirow-idot",
    f32Case: "prefill-gemm-q6k-multirow-f32",
    idotGen: "prefillGemmQ6KSplitKIdotWgsl",
    f32Gen: "prefillGemmQ6KSplitKWgsl",
    // stesse liste del q4_K: e' un K-quant, quindi UN buffer di pesi
    idotBindings: ["blocks", "xq", "part", "xsc"],
    f32Bindings: ["blocks", "x", "part"],
    idotTols: ["PREFILL_GEMM_Q6K_IDOT_REL_TOL", "PREFILL_GEMM_Q6K_IDOT_ABS_TOL"],
    f32Tols: ["PREFILL_GEMM_Q6K_F32_REL_TOL", "PREFILL_GEMM_Q6K_F32_ABS_TOL"],
    repack: "repackKQuant",
    repackFields: null,
  },
  {
    kind: "q8_0",
    fn: "testPrefillGemmQ80MultiRow",
    caseConst: "PREFILL_Q80_KTEST_CASE",
    idotCase: "prefill-gemm-q80-multirow-idot",
    f32Case: "prefill-gemm-q80-multirow-f32",
    idotGen: "prefillGemmQ80SplitKIdotWgsl",
    f32Gen: "prefillGemmQ80SplitKWgsl",
    // LA RIGA CHE GIUSTIFICA LA TABELLA: il q8_0 ha DUE buffer di pesi, quindi
    // cinque binding sulla via intera e quattro sulla f32, dove i due K-quant
    // qui sopra ne hanno quattro e tre. Un banco che ereditasse le liste del
    // vicino compilerebbe e poi legherebbe le scale dove il kernel aspetta le
    // attivazioni.
    idotBindings: ["qs", "scales", "xq", "part", "xsc"],
    f32Bindings: ["qs", "scales", "x", "part"],
    idotTols: ["PREFILL_GEMM_Q80_IDOT_REL_TOL", "PREFILL_GEMM_Q80_IDOT_ABS_TOL"],
    f32Tols: ["PREFILL_GEMM_Q80_F32_REL_TOL", "PREFILL_GEMM_Q80_F32_ABS_TOL"],
    repack: "repackQ8_0",
    repackFields: "qs, scales",
  },
];

/**
 * Quanti `results.push(` c'erano nel ktest PRIMA della riga 4: 73 (i 72 di
 * prima del q4_1 piu' il suo). I tre banchi nuovi ne AGGIUNGONO uno ciascuno —
 * il conteggio puo' solo crescere, e se un push sparisse mentre ne entra uno
 * nuovo il totale resterebbe e la soglia del driver mentirebbe.
 */
const PUSHES_BEFORE_RIGA4 = 73;

/** corpo del banco, o un messaggio che dice DOVE manca */
function bankBody(src: string, b: Bank): string {
  const body = helperBody(src, b.fn);
  expect(body === null ? `${KTEST}: il banco ${b.kind} manca — ${b.fn} non e' definito` : "ok").toBe("ok");
  return body!;
}

/** le due `compare(` del banco, ognuna col suo testo di argomenti */
function compareCalls(src: string, b: Bank): string[] {
  const body = bankBody(src, b);
  return hitsOf(body, "compare", "\\(").map((at) => {
    const open = body.indexOf("(", at);
    return body.slice(open, closeOf(body, open) + 1);
  });
}

describe.each(BANKS.map((b) => [b.kind, b] as const))(
  "banco %s della riga 4: cablaggio nel ktest", (_kind, b) => {
    describe("(a) i due casi esistono, e ciascuno una volta sola", () => {
      for (const name of [b.idotCase, b.f32Case]) {
        it(`"${name}" e' un nome di caso, dichiarato una volta`, () => {
          const src = uncommented(KTEST); // i nomi sono stringhe: qui si tengono
          const where = literalHits(src, name).map((i) => `${KTEST}:${lineOf(src, i)}`);
          expect(where.length, `caso "${name}" a ${where.join(", ") || "NESSUNA riga"}`).toBe(1);
        });
      }
    });

    describe("(b) il banco e' definito una volta e registrato una volta", () => {
      it(`${b.fn}( compare 2 volte: definizione + registrazione`, () => {
        const src = code(KTEST);
        const where = hitsOf(src, b.fn, "\\(").map((i) => `${KTEST}:${lineOf(src, i)}`);
        expect(where.length, `banco ${b.kind}: ${b.fn}( a ${where.join(", ") || "NESSUNA riga"}`).toBe(2);
      });

      it(`la firma congelata e' async ${b.fn}(g: Gpu): Promise<KResult[]>`, () => {
        // il tipo di ritorno e' un ARRAY: e' quello che rende necessario lo
        // spread alla registrazione, ed e' l'unica forma in cui i due bracci
        // contano come due casi del driver invece che come uno.
        const src = code(KTEST);
        const sig = new RegExp(`async\\s+function\\s+${b.fn}\\s*\\(\\s*g\\s*:\\s*Gpu\\s*\\)\\s*:\\s*Promise\\s*<\\s*KResult\\[\\]\\s*>`);
        expect(sig.test(src), `banco ${b.kind}: firma di ${b.fn} non trovata in ${KTEST}`).toBe(true);
      });

      it("una delle due e' la definizione, l'altra e' dentro results.push NEL CORPO di main()", () => {
        const src = code(KTEST);
        const at = (i: number): string => `${KTEST}:${lineOf(src, i)}`;
        // il corpo di main() si prende per INDICI: "registrato" e' una domanda
        // di POSIZIONE, non di forma della riga. Un `results.push(...await FN(g))`
        // identico ma scritto in una funzione che nessuno chiama supererebbe
        // qualunque controllo testuale, e il driver non eseguirebbe MAI i due
        // casi nuovi (ktest verde, banco mai misurato).
        const mainAt = helperRange(src, "main");
        expect(mainAt === null ? `${KTEST}: main() non trovata — registrazione non verificabile` : "ok").toBe("ok");
        const hits = hitsOf(src, b.fn, "\\(");
        const def = hits.filter((i) => /function\s+$/.test(src.slice(Math.max(0, i - 32), i)));
        // lo SPREAD e' la firma del banco a due risultati: `results.push(await …)`
        // compilerebbe pure, e spingerebbe un array dentro la lista dei KResult.
        const reg = hits.filter((i) => /results\s*\.\s*push\s*\(\s*\.\.\.\s*await\s+$/.test(src.slice(Math.max(0, i - 96), i)));
        const inMain = reg.filter((i) => i > mainAt!.from && i < mainAt!.to);
        expect(def.length, `banco ${b.kind}: definizione di ${b.fn} a ${def.map(at).join(", ") || "NESSUNA riga"}`).toBe(1);
        expect(reg.length, `banco ${b.kind}: results.push(...await ${b.fn}(...)) a ${reg.map(at).join(", ") || "NESSUNA riga"}`).toBe(1);
        const body = mainAt === null ? "?" : `${lineOf(src, mainAt.from)}-${lineOf(src, mainAt.to)}`;
        expect(inMain.length, `banco ${b.kind}: registrazione FUORI dal corpo di main() (${KTEST}:${body}) — push a ${reg.map(at).join(", ") || "NESSUNA riga"}: il driver non eseguirebbe i due casi`).toBe(1);
      });
    });

    describe("(c) ogni braccio chiama i suoi generatori con i binding congelati", () => {
      // Si contano NEL CORPO del banco, non nel file: quantizzatore e combine
      // sono gli STESSI generatori per tutti e sei i banchi di prefill, e un
      // conteggio a livello di file misurerebbe gli altri cinque.
      const EXPECT: Array<{ gen: string; times: number; bindings: string[] }> = [
        { gen: "prefillQuantXQ8Wgsl", times: 1, bindings: ["x", "xq", "xsc"] },
        { gen: b.idotGen, times: 1, bindings: b.idotBindings },
        { gen: b.f32Gen, times: 1, bindings: b.f32Bindings },
        { gen: "prefillSplitKCombineWgsl", times: 2, bindings: ["part", "y"] },
      ];
      for (const e of EXPECT) {
        it(`${e.gen}( ${e.times} volta/e nel banco ${b.kind}, binding [${e.bindings.join(", ")}]`, () => {
          const body = bankBody(code(KTEST), b);
          const calls = callsTo(body, e.gen);
          const all = calls.map((c) => `${b.fn}+${c.line}`);
          expect(calls.length, `banco ${b.kind}: ${e.gen}( a ${all.join(", ") || "NESSUNA riga"}`).toBe(e.times);
          const bad = calls.flatMap((c) => {
            const where = `banco ${b.kind} (${b.fn}+${c.line})`;
            if (c.bindings === null) {
              return [`${where} lista di binding non inline dopo ${e.gen}(...): cablaggio non verificabile`];
            }
            if (c.bindings.join(", ") !== e.bindings.join(", ")) {
              return [`${where} lega [${c.bindings.join(", ")}] invece di [${e.bindings.join(", ")}]`];
            }
            return [];
          });
          expect(bad.join("\n")).toBe("");
        });
      }

      it("i due moltiplicatori sono importati da ../kernels/wgsl", () => {
        const imported = importsFrom(uncommented(KTEST), "kernels/wgsl");
        const missing = [b.idotGen, b.f32Gen].filter((n) => !has(imported, n));
        expect(missing.join(", "), `banco ${b.kind}: da kernels/wgsl mancano ${missing.join(", ")}`).toBe("");
      });

      it(`i pesi sul device escono da ${b.repack}, non da un impacchettamento locale`, () => {
        const body = bankBody(code(KTEST), b);
        if (b.repackFields === null) {
          // K-quant: UN buffer, il superblocco GGUF copiato in parole
          expect(hitsOf(body, b.repack, "\\(").length,
            `banco ${b.kind}: ${b.repack}( nel corpo di ${b.fn}`).toBe(1);
          return;
        }
        const m = new RegExp(`const\\s*\\{([^}]*)\\}\\s*=\\s*${b.repack}\\s*\\(`).exec(body);
        expect(m === null ? `banco ${b.kind}: manca const { ${b.repackFields} } = ${b.repack}(...) in ${b.fn}` : "ok").toBe("ok");
        // si guardano le CHIAVI, non i nomi locali: sono i nomi dei BUFFER (le
        // liste di binding sopra li congelano li'), quindi i byte grezzi
        // arrivano per forza con un alias — `const { qs: qsSrc, ... }`.
        const fields = m![1].split(",").map((f) => f.split(":")[0].trim()).filter((f) => f.length > 0);
        expect(fields.join(", "), `banco ${b.kind}: ${b.repack} destruttura [${fields.join(", ")}]`).toBe(b.repackFields);
      });
    });

    describe("(c2) le griglie sono quelle degli helper, e le opzioni portano il kind", () => {
      // Il difetto che questo blocco esiste per prendere (gia' pagato sul
      // gemello Q5_K): una griglia che NON esiste, e opzioni senza `kind`.
      // Nessun conteggio di generatori se ne accorge, perche' il difetto sta
      // negli argomenti — e un import rotto ferma TUTTO il worker, non solo i
      // due casi nuovi.
      const GRIDS: Array<[string, number]> = [
        ["prefillGemmGrid", 2], ["prefillQuantXGrid", 1], ["prefillCombineGrid", 2],
      ];

      it.each(GRIDS)("%s( compare %i volta/e nel banco, e viene importato da ../kernels/wgsl", (gr, want) => {
        const body = bankBody(code(KTEST), b);
        expect(hitsOf(body, gr, "\\(").length, `banco ${b.kind}: ${gr}( nel corpo di ${b.fn}`).toBe(want);
        const imported = importsFrom(uncommented(KTEST), "kernels/wgsl");
        expect(has(imported, gr), `${gr} dev'essere importato da ../kernels/wgsl`).toBe(true);
      });

      it("nessuna griglia ri-derivata a mano dentro il banco", () => {
        // una `Math.ceil((M*N)/64)` scritta qui sarebbe una seconda copia della
        // geometria di un kernel che vive in un altro file.
        expect(bankBody(code(KTEST), b), `banco ${b.kind}: griglia ricalcolata a mano in ${b.fn}`)
          .not.toMatch(/Math\.ceil\s*\(/);
      });

      it("i due moltiplicatori e la loro griglia ricevono LE STESSE opzioni", () => {
        const body = bankBody(code(KTEST), b);
        const bad: string[] = [];
        const names = [b.idotGen, b.f32Gen, "prefillGemmGrid"];
        const args = names.flatMap((n) => callsTo(body, n).map((c) => ({ n, a: c.args })));
        for (const { n, a } of args) {
          if (a.length !== 1 || !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(a[0])) {
            bad.push(`banco ${b.kind}: ${n}(${a.join(", ")}): atteso UN identificatore di opzioni`);
          }
        }
        const ids = [...new Set(args.map((x) => x.a[0]))];
        expect(bad.join("\n")).toBe("");
        expect(ids.length, `banco ${b.kind}: opzioni distinte fra ${names.join("/")}: ${ids.join(", ")}`).toBe(1);
      });

      it(`le opzioni sono un PrefillGemmOpts con kind "${b.kind}"`, () => {
        // `kind` sceglie l'unita' di taglio di K: senza, `prefillGemmCheck`
        // rifiuta la shape al primo dispatch e il tipo non compila. Col `kind`
        // di un altro formato compilerebbe e misurerebbe il passo sbagliato —
        // ed e' un rischio VERO qui, perche' il q4_K e il q6_K condividono
        // l'unita' (superblocco da 256) con il q5_K gia' presente nel file.
        const body = bankBody(uncommented(KTEST), b); // il valore di `kind` e' una stringa
        const decl = /const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*PrefillGemmOpts\s*=\s*\{/.exec(body);
        expect(decl === null ? `banco ${b.kind}: in ${b.fn} manca una const tipizzata PrefillGemmOpts` : "ok").toBe("ok");
        const obj = body.slice(body.indexOf("{", decl!.index + decl![0].length - 1));
        const fields = topLevelItems(obj, 0).map((fi) => fi.replace(/\s+/g, ""));
        expect(fields.join(", "), `banco ${b.kind}: campi delle opzioni = ${fields.join(", ")}`)
          .toBe(`kind:"${b.kind}", K, N, M, splits`);
        const imported = importsFrom(uncommented(KTEST), "kernels/wgsl");
        expect(has(imported, "PrefillGemmOpts"), "PrefillGemmOpts dev'essere importato da ../kernels/wgsl").toBe(true);
      });
    });

    describe("(d) shape, seed e tolleranze arrivano da prefillkquant", () => {
      const TOLS = [...b.idotTols, ...b.f32Tols];

      it(`il ktest importa ${b.caseConst} e le quattro tolleranze da ../prefillkquant`, () => {
        const src = uncommented(KTEST); // gli import sono stringhe: qui non si biancano
        const imported = importsFrom(src, "prefillkquant");
        const missing = [b.caseConst, ...TOLS].filter((n) => !has(imported, n));
        expect(missing.join(", "), `banco ${b.kind}: import da prefillkquant = "${imported}" — mancano: ${missing.join(", ")}`).toBe("");
      });

      it(`shape e seed NON sono ricopiati: escono da ${b.caseConst}`, () => {
        const body = bankBody(code(KTEST), b);
        const m = new RegExp(`const\\s*\\{([^}]*)\\}\\s*=\\s*${b.caseConst}`).exec(body);
        expect(m === null ? `banco ${b.kind}: ${b.fn} deve destrutturare ${b.caseConst}` : "ok").toBe("ok");
        const names = m![1].split(",").map((fi) => fi.trim());
        const missing = ["K", "N", "M", "splits"].filter((n) => !names.includes(n));
        expect(missing.join(", "), `banco ${b.kind}: dalla destrutturazione di ${b.caseConst} mancano ${missing.join(", ")}`).toBe("");
      });

      it("nelle due compare( non c'e' nessun letterale numerico", () => {
        const calls = compareCalls(code(KTEST), b);
        expect(calls.length, `banco ${b.kind}: compare( nel corpo di ${b.fn}: ${calls.length}`).toBe(2);
        // una tolleranza scritta a mano qui dentro sarebbe una seconda verita'
        // accanto a quella di prefillkquant: il numero si importa, non si
        // ricopia. I letterali si cercano fuori dagli identificatori
        // (`Float32Array` e `PREFILL_GEMM_Q4K_...` hanno cifre, ma non sono numeri).
        const bad = calls.flatMap((args) => {
          const lit = args.match(/(?<![A-Za-z0-9_$])\d[\d._eE+-]*/g);
          return lit ? [`banco ${b.kind}: compare(...) con letterale/i ${lit.join(", ")}: ${args.replace(/\s+/g, " ")}`] : [];
        });
        expect(bad.join("\n")).toBe("");
      });

      it("ogni braccio riceve LA SUA coppia di tolleranze, una volta sola", () => {
        // importarle tutte e quattro e poi passare due volte la stessa coppia
        // giudicherebbe una via col pavimento dell'altra.
        const body = bankBody(code(KTEST), b);
        const bad = TOLS.flatMap((t) => {
          const n = hitsOf(body, t).length;
          return n === 1 ? [] : [`banco ${b.kind}: ${t} usata ${n} volte nel banco (attesa 1)`];
        });
        expect(bad.join("\n")).toBe("");
        // il braccio si riconosce dal NOME DEL CASO che la compare porta come
        // primo argomento: qui si legge il sorgente con le stringhe intatte,
        // cosi' non serve congelare anche i nomi delle variabili di lettura.
        const calls = compareCalls(uncommented(KTEST), b);
        for (const [name, tols] of [[b.idotCase, b.idotTols], [b.f32Case, b.f32Tols]] as Array<[string, string[]]>) {
          const arm = calls.filter((a) => literalHits(a, name).length === 1);
          expect(arm.length, `banco ${b.kind}: compare( del caso "${name}"`).toBe(1);
          const missing = tols.filter((t) => !has(arm[0], t));
          expect(missing.join(", "), `banco ${b.kind}: al caso "${name}" mancano ${missing.join(", ")} — ${arm[0].replace(/\s+/g, " ")}`).toBe("");
        }
      });
    });
  });

describe("(e) i banchi della riga 4 si aggiungono senza abbassare la soglia del driver", () => {
  // `tools/harness/engine-ktest.mjs` NON e' posseduto da questo task: si legge
  // SOLA LETTURA per verificare che il default resti 90, cioe' che i sei casi
  // nuovi non siano stati fatti entrare abbassando la barra. Che 90 sia SOTTO il
  // numero di casi registrati e' una proprieta' della RUN del driver (i banchi
  // registrano piu' casi di quante siano le chiamate statiche: sweep e cicli), e
  // questo test statico non la simula.
  it("KTEST_MIN_PASS ha ancora default 90", () => {
    const m = /KTEST_MIN_PASS\s*\?\?\s*(\d+)/.exec(raw(DRIVER));
    expect(m === null ? `${DRIVER}: default di KTEST_MIN_PASS non trovato` : "ok").toBe("ok");
    expect(Number(m![1]), `${DRIVER}: KTEST_MIN_PASS default = ${m![1]}`).toBe(90);
  });

  it("i tre banchi si AGGIUNGONO: nessun results.push rimosso", () => {
    const src = code(KTEST);
    const pushes = hitsOf(src, "results", "\\.push\\s*\\(").length;
    const want = PUSHES_BEFORE_RIGA4 + BANKS.length;
    expect(pushes, `riga 4: results.push( nel ktest = ${pushes}, atteso >= ${want} (i ${PUSHES_BEFORE_RIGA4} di prima piu' i ${BANKS.length} banchi nuovi)`)
      .toBeGreaterThanOrEqual(want);
  });
});

describe("(f) i tre banchi sono TRE, e non tre volte lo stesso", () => {
  // Il difetto che questo blocco prende: tre banchi copiati che finiscono a
  // condividere un caso, un generatore o una tolleranza. Le liste della tabella
  // sopra sono cio' che il resto del file congela — qui si verifica che siano
  // DISTINTE fra loro, altrimenti congelerebbero la stessa cosa tre volte.
  it("nomi di caso, funzioni, generatori, casi e tolleranze: tutti distinti", () => {
    const all = <T>(pick: (b: Bank) => T[]): T[] => BANKS.flatMap(pick);
    for (const [what, list] of [
      ["nomi di caso", all((b) => [b.idotCase, b.f32Case])],
      ["funzioni di banco", all((b) => [b.fn])],
      ["generatori", all((b) => [b.idotGen, b.f32Gen])],
      ["costanti di caso", all((b) => [b.caseConst])],
      ["tolleranze", all((b) => [...b.idotTols, ...b.f32Tols])],
      ["kind", all((b) => [b.kind])],
    ] as Array<[string, string[]]>) {
      expect(new Set(list).size, `${what}: ${list.join(", ")}`).toBe(list.length);
    }
  });

  it("i tre casi di prefillkquant hanno seed tutti diversi, anche fra formati", () => {
    // due casi che condividono un seed condividono un flusso di byte, e allora
    // i loro pavimenti possono coincidere per costruzione — che e' esattamente
    // cio' che il floor test confronta.
    const src = raw("src/engine/prefillkquant.ts");
    const seeds = [...src.matchAll(/seedBlocks:\s*(\d+),\s*seedX:\s*(\d+)/g)]
      .flatMap((m) => [m[1], m[2]]);
    expect(seeds.length, `seed trovati in prefillkquant.ts: ${seeds.join(", ")}`)
      .toBe(2 * (BANKS.length + 2)); // + i casi q5_K e q4_1 delle righe 2 e 3
    expect(new Set(seeds).size, `seed duplicati: ${seeds.join(", ")}`).toBe(seeds.length);
  });
});
