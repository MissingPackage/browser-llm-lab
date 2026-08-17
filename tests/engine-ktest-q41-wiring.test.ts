// Cablaggio del banco Q4_1 multi-riga del ktest (goal K-quant, task ktest-q41).
//
// FASE ROSSA. Questo file e' il contratto ESEGUIBILE del banco q4_1 e si scrive
// PRIMA che il banco esista: finche' `testPrefillGemmQ41MultiRow` non e' in
// `src/engine/ktest/ktest.worker.ts`, la suite qui sotto e' ROSSA PER
// COSTRUZIONE, e ogni fallimento nomina il pezzo che manca (funzione, casi,
// tolleranze, binding, registrazione in `main()`).
//
// Test STATICO, senza GPU: stessa postura di
// `tests/engine-ktest-q5k-wiring.test.ts` — si SCANSIONA il sorgente
// dell'orchestratore invece di fidarsi di una descrizione a parole. Qui si prova
// il CABLAGGIO del banco nuovo, non la sua aritmetica: che i due casi esistano e
// siano registrati una volta sola, che ogni braccio chiami i suoi generatori con
// le liste di binding congelate e con le GRIGLIE degli helper, che le opzioni
// portino il `kind` del formato, e che shape/seed/tolleranze arrivino da
// `prefillkquant` invece di essere ricopiati a mano qui dentro.
//
// COSA CAMBIA RISPETTO AL GEMELLO Q5_K, e non e' un dettaglio di forma: il q4_1
// ha DUE buffer di pesi, non uno. `repackQ4_1` spezza il formato in `qs` +
// `scales` (d nei 16 bit bassi, m negli alti di UNA parola per blocco), mentre
// il superblocco Q5_K viaggia in un buffer solo. Da qui le due liste di binding
// congelate — idot `[qs, scales, xq, part, xsc]` a CINQUE elementi e f32
// `[qs, scales, x, part]` a QUATTRO — che sono le stesse che dichiarano
// `prefillGemmQ41SplitKIdotWgsl` / `prefillGemmQ41SplitKWgsl` in kernels/wgsl.ts.
// Un banco che ereditasse le liste del Q5_K compilerebbe e poi legherebbe le
// scale dove il kernel aspetta le attivazioni.
//
// SECONDA DIFFERENZA, di metodo: i generatori CONDIVISI (`prefillQuantXQ8Wgsl`,
// `prefillSplitKCombineWgsl`) e le tre griglie si contano NEL CORPO del banco
// nuovo, non nel file intero come faceva il gemello quando era l'unico banco di
// prefill K-quant. Il banco Q5_K esiste gia' e usa gli stessi generatori: un
// conteggio a livello di file misurerebbe LUI, e questo test deve poter fallire
// solo per il q4_1 (e restare indifferente a quello che il q5_K fa o smette di
// fare).
//
// E IL GEMELLO Q5_K FA LO STESSO. Quando questo file e' stato scritto, il
// gemello contava ancora quei generatori condivisi A LIVELLO DI FILE
// (`callsTo(code(KTEST), …)`), e qui stava annotato il debito: un conteggio
// cosi' sarebbe tornato ROSSO a ogni banco nuovo per una ragione che non ha
// niente a che vedere con cio' che il test difende. Il debito e' stato PAGATO —
// `tests/engine-ktest-q5k-wiring.test.ts` blocco (c) conta dentro
// `bankBody(code(KTEST))`, col suo commento accanto — e con i tre banchi della
// riga 4 (q4_K, q6_K, q8_0) la cosa si vede: sei banchi usano gli stessi due
// generatori condivisi, e nessuno dei cinque file di cablaggio se ne accorge.
// La regola che ne resta, per chi aggiunge il settimo: i generatori condivisi si
// contano SEMPRE nel corpo del proprio banco.
//
// Il verdetto NUMERICO non e' di questo test: lo da' `node
// .harness/tools/engine-ktest.mjs`, che vuole un vite server e un Chrome vero.
// Un ktest dichiarato verde senza averlo eseguito e' il difetto piu' grave
// registrato in questo repo: questo file non lo simula, verifica solo che il
// banco sia agganciato dove il driver lo trovera'.
//
// Il file si LEGGE come testo. Importarlo tirerebbe dentro i tipi WebGPU e un
// modulo worker (che in vitest node non si carica). Il testo non si scansiona
// grezzo: commenti, stringhe e template WGSL vengono prima BIANCATI
// (`blankNonCode` in `tests/helpers/source-scan.ts`, dove lo scanner e' stato
// FATTORIZZATO dopo esserne diventato la quarta copia). Senza quel passo un
// commento che nomina `prefillSplitKCombineWgsl(` sposta i conteggi, e una lista
// `[qs, scales, xq, part, xsc]` citata in un commento diventa la lista che il
// test misura al posto di quella vera.
import { describe, expect, it } from "vitest";
import {
  raw, code, uncommented, lineOf, hitsOf, closeOf, callsTo, topLevelItems,
  importsFrom, has, helperRange, helperBody, literalHits,
} from "./helpers/source-scan";

const KTEST = "src/engine/ktest/ktest.worker.ts";
const DRIVER = ".harness/tools/engine-ktest.mjs";

const IDOT = "prefill-gemm-q41-multirow-idot";
const F32 = "prefill-gemm-q41-multirow-f32";
const FN = "testPrefillGemmQ41MultiRow";
const CASE = "PREFILL_Q41_KTEST_CASE";
const IDOT_TOLS = ["PREFILL_GEMM_Q41_IDOT_REL_TOL", "PREFILL_GEMM_Q41_IDOT_ABS_TOL"];
const F32_TOLS = ["PREFILL_GEMM_Q41_F32_REL_TOL", "PREFILL_GEMM_Q41_F32_ABS_TOL"];
const TOLS = [...IDOT_TOLS, ...F32_TOLS];
// i due moltiplicatori del formato: gia' in kernels/wgsl.ts, e il banco deve
// usare QUELLI (non ri-generare il WGSL a modo suo).
const IDOT_GEN = "prefillGemmQ41SplitKIdotWgsl";
const F32_GEN = "prefillGemmQ41SplitKWgsl";
// il ktest attuale registra 72 banchi: il q4_1 ne AGGIUNGE uno.
const PUSHES_BEFORE_Q41 = 72;

/** corpo del banco q4_1, o un messaggio che dice DOVE manca */
function bankBody(src: string): string {
  const body = helperBody(src, FN);
  expect(body === null ? `${KTEST}: il banco q4_1 manca — ${FN} non e' definito` : "ok").toBe("ok");
  return body!;
}

/** le due `compare(` del banco, ognuna col suo testo di argomenti */
function compareCalls(src: string): string[] {
  const body = bankBody(src);
  return hitsOf(body, "compare", "\\(").map((at) => {
    const open = body.indexOf("(", at);
    return body.slice(open, closeOf(body, open) + 1);
  });
}

describe("(a) i due casi q4_1 esistono, e ciascuno una volta sola", () => {
  for (const name of [IDOT, F32]) {
    it(`"${name}" e' un nome di caso, dichiarato una volta`, () => {
      const src = uncommented(KTEST); // i nomi sono stringhe: qui si tengono
      const where = literalHits(src, name).map((i) => `${KTEST}:${lineOf(src, i)}`);
      expect(where.length, `caso q4_1 "${name}" a ${where.join(", ") || "NESSUNA riga"}`).toBe(1);
    });
  }
});

describe("(b) il banco q4_1 e' definito una volta e registrato una volta", () => {
  it(`${FN}( compare 2 volte: definizione + registrazione`, () => {
    const src = code(KTEST);
    const where = hitsOf(src, FN, "\\(").map((i) => `${KTEST}:${lineOf(src, i)}`);
    expect(where.length, `banco q4_1: ${FN}( a ${where.join(", ") || "NESSUNA riga"}`).toBe(2);
  });

  it(`la firma congelata e' async ${FN}(g: Gpu): Promise<KResult[]>`, () => {
    // il tipo di ritorno e' un ARRAY: e' quello che rende necessario lo spread
    // alla registrazione, ed e' l'unica forma in cui i due bracci contano come
    // due casi del driver invece che come uno.
    const src = code(KTEST);
    const sig = new RegExp(`async\\s+function\\s+${FN}\\s*\\(\\s*g\\s*:\\s*Gpu\\s*\\)\\s*:\\s*Promise\\s*<\\s*KResult\\[\\]\\s*>`);
    expect(sig.test(src), `banco q4_1: firma di ${FN} non trovata in ${KTEST}`).toBe(true);
  });

  it(`una delle due e' la definizione, l'altra e' dentro results.push NEL CORPO di main()`, () => {
    const src = code(KTEST);
    const at = (i: number): string => `${KTEST}:${lineOf(src, i)}`;
    // il corpo di main() si prende per INDICI: "registrato" e' una domanda di
    // POSIZIONE, non di forma della riga. Un `results.push(...await FN(g))`
    // identico ma scritto in una funzione che nessuno chiama supererebbe
    // qualunque controllo testuale, e il driver non eseguirebbe MAI i due casi
    // nuovi (ktest verde, banco q4_1 mai misurato).
    const mainAt = helperRange(src, "main");
    expect(mainAt === null ? `${KTEST}: main() non trovata — registrazione non verificabile` : "ok").toBe("ok");
    const hits = hitsOf(src, FN, "\\(");
    const def = hits.filter((i) => /function\s+$/.test(src.slice(Math.max(0, i - 32), i)));
    // lo SPREAD e' la firma del banco a due risultati: `results.push(await …)`
    // compilerebbe pure, e spingerebbe un array dentro la lista dei KResult.
    // Si guarda il testo che PRECEDE la chiamata (non la riga), cosi' una
    // registrazione spezzata su due righe resta valida.
    const reg = hits.filter((i) => /results\s*\.\s*push\s*\(\s*\.\.\.\s*await\s+$/.test(src.slice(Math.max(0, i - 96), i)));
    const inMain = reg.filter((i) => i > mainAt!.from && i < mainAt!.to);
    expect(def.length, `banco q4_1: definizione di ${FN} a ${def.map(at).join(", ") || "NESSUNA riga"}`).toBe(1);
    expect(reg.length, `banco q4_1: results.push(...await ${FN}(...)) a ${reg.map(at).join(", ") || "NESSUNA riga"}`).toBe(1);
    const body = mainAt === null ? "?" : `${lineOf(src, mainAt.from)}-${lineOf(src, mainAt.to)}`;
    expect(inMain.length, `banco q4_1: registrazione FUORI dal corpo di main() (${KTEST}:${body}) — push a ${reg.map(at).join(", ") || "NESSUNA riga"}: il driver non eseguirebbe i due casi q4_1`).toBe(1);
  });
});

describe("(c) ogni braccio q4_1 chiama i suoi generatori con i binding congelati", () => {
  // Le liste sono quelle dei kernel q4_1, e sono DIVERSE da quelle del Q5_K
  // perche' `repackQ4_1` produce DUE buffer di pesi (qs + scales) dove il
  // superblocco Q5_K ne produce uno:
  //   idot [qs, scales, xq, part, xsc]  — 5 binding
  //   f32  [qs, scales, x, part]        — 4 binding
  // Quantizzazione e combine restano quelle del q4_0: [x, xq, xsc], [part, y].
  // Si contano NEL CORPO del banco (non nel file), perche' quantizzatore e
  // combine li usa anche il banco Q5_K gia' presente.
  const EXPECT: Array<{ gen: string; times: number; bindings: string[] }> = [
    { gen: "prefillQuantXQ8Wgsl", times: 1, bindings: ["x", "xq", "xsc"] },
    { gen: IDOT_GEN, times: 1, bindings: ["qs", "scales", "xq", "part", "xsc"] },
    { gen: F32_GEN, times: 1, bindings: ["qs", "scales", "x", "part"] },
    { gen: "prefillSplitKCombineWgsl", times: 2, bindings: ["part", "y"] },
  ];
  for (const e of EXPECT) {
    it(`${e.gen}( ${e.times} volta/e nel banco q4_1, binding [${e.bindings.join(", ")}]`, () => {
      const body = bankBody(code(KTEST));
      const calls = callsTo(body, e.gen);
      const all = calls.map((c) => `${FN}+${c.line}`);
      expect(calls.length, `banco q4_1: ${e.gen}( a ${all.join(", ") || "NESSUNA riga"}`).toBe(e.times);
      const bad = calls.flatMap((c) => {
        const where = `banco q4_1 (${FN}+${c.line})`;
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

  it("i due moltiplicatori q4_1 sono importati da ../kernels/wgsl", () => {
    const imported = importsFrom(uncommented(KTEST), "kernels/wgsl");
    const missing = [IDOT_GEN, F32_GEN].filter((n) => !has(imported, n));
    expect(missing.join(", "), `banco q4_1: da kernels/wgsl mancano ${missing.join(", ")}`).toBe("");
  });

  it("i due buffer dei pesi escono da repackQ4_1, non da un impacchettamento locale", () => {
    // il q4_1 e' l'unico formato di questa famiglia che si spezza in due: se il
    // banco ri-impaccasse i byte a modo suo, proverebbe un kernel che in
    // produzione non gira (e le due liste di binding sopra sarebbero soddisfatte
    // da variabili qualsiasi che si chiamano qs/scales).
    const body = bankBody(code(KTEST));
    const m = /const\s*\{([^}]*)\}\s*=\s*repackQ4_1\s*\(/.exec(body);
    expect(m === null ? `banco q4_1: manca const { qs, scales } = repackQ4_1(...) in ${FN}` : "ok").toBe("ok");
    // si guardano le CHIAVI, non i nomi locali: `qs` e `scales` sono i nomi dei
    // due BUFFER (le liste di binding sopra li congelano li'), quindi i byte
    // grezzi arrivano per forza con un alias — `const { qs: qsSrc, ... }`.
    const fields = m![1].split(",").map((f) => f.split(":")[0].trim()).filter((f) => f.length > 0);
    expect(fields.join(", "), `banco q4_1: repackQ4_1 destruttura [${fields.join(", ")}]`).toBe("qs, scales");
  });
});

describe("(c2) le griglie del banco q4_1 sono quelle degli helper, e le opzioni portano il kind", () => {
  // Il difetto che questo blocco esiste per prendere (gia' pagato sul gemello
  // Q5_K): una `prefillGemmQ41Grid` che NON esiste, e opzioni senza `kind`.
  // Nessun conteggio di generatori se ne accorge, perche' il difetto sta negli
  // argomenti — e un import rotto ferma TUTTO il worker, non solo i due casi
  // nuovi.
  const GRIDS: Array<[string, number]> = [
    ["prefillGemmGrid", 2], ["prefillQuantXGrid", 1], ["prefillCombineGrid", 2],
  ];

  it.each(GRIDS)("%s( compare %i volta/e nel banco q4_1, e viene importato da ../kernels/wgsl", (g, want) => {
    const body = bankBody(code(KTEST));
    expect(hitsOf(body, g, "\\(").length, `banco q4_1: ${g}( nel corpo di ${FN}`).toBe(want);
    const imported = importsFrom(uncommented(KTEST), "kernels/wgsl");
    expect(has(imported, g), `${g} dev'essere importato da ../kernels/wgsl`).toBe(true);
  });

  it("nessuna griglia ri-derivata a mano dentro il banco q4_1", () => {
    // una `Math.ceil((M*N)/64)` scritta qui sarebbe una seconda copia della
    // geometria di un kernel che vive in un altro file.
    expect(bankBody(code(KTEST)), `banco q4_1: griglia ricalcolata a mano in ${FN}`).not.toMatch(/Math\.ceil\s*\(/);
  });

  it("i due moltiplicatori q4_1 e la loro griglia ricevono LE STESSE opzioni", () => {
    const body = bankBody(code(KTEST));
    const bad: string[] = [];
    const names = [IDOT_GEN, F32_GEN, "prefillGemmGrid"];
    const args = names.flatMap((n) => callsTo(body, n).map((c) => ({ n, a: c.args })));
    for (const { n, a } of args) {
      if (a.length !== 1 || !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(a[0])) {
        bad.push(`banco q4_1: ${n}(${a.join(", ")}): atteso UN identificatore di opzioni`);
      }
    }
    const ids = [...new Set(args.map((x) => x.a[0]))];
    expect(bad.join("\n")).toBe("");
    expect(ids.length, `banco q4_1: opzioni distinte fra ${names.join("/")}: ${ids.join(", ")}`).toBe(1);
  });

  it("le opzioni del banco q4_1 sono un PrefillGemmOpts con kind \"q4_1\"", () => {
    // `kind` sceglie il passo del blocco: senza, `prefillGemmCheck` rifiuta la
    // shape al primo dispatch, e il tipo non compila. Con `kind: "q5_K"` copiato
    // dal gemello, compilerebbe e misurerebbe il passo sbagliato.
    const body = bankBody(uncommented(KTEST)); // il valore di `kind` e' una stringa
    const decl = /const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*PrefillGemmOpts\s*=\s*\{/.exec(body);
    expect(decl === null ? `banco q4_1: in ${FN} manca una const tipizzata PrefillGemmOpts` : "ok").toBe("ok");
    const obj = body.slice(body.indexOf("{", decl!.index + decl![0].length - 1));
    const fields = topLevelItems(obj, 0).map((f) => f.replace(/\s+/g, ""));
    expect(fields.join(", "), `banco q4_1: campi delle opzioni = ${fields.join(", ")}`)
      .toBe('kind:"q4_1", K, N, M, splits');
    const imported = importsFrom(uncommented(KTEST), "kernels/wgsl");
    expect(has(imported, "PrefillGemmOpts"), "PrefillGemmOpts dev'essere importato da ../kernels/wgsl").toBe(true);
  });
});

describe("(d) shape, seed e tolleranze del q4_1 arrivano da prefillkquant", () => {
  it(`il ktest importa ${CASE} e le quattro tolleranze q4_1 da ../prefillkquant`, () => {
    const src = uncommented(KTEST); // gli import sono stringhe: qui non si biancano
    const imported = importsFrom(src, "prefillkquant");
    const missing = [CASE, ...TOLS].filter((n) => !has(imported, n));
    expect(missing.join(", "), `banco q4_1: import da prefillkquant = "${imported}" — mancano: ${missing.join(", ")}`).toBe("");
  });

  it(`shape e seed del caso q4_1 NON sono ricopiati: escono da ${CASE}`, () => {
    const body = bankBody(code(KTEST));
    const m = new RegExp(`const\\s*\\{([^}]*)\\}\\s*=\\s*${CASE}`).exec(body);
    expect(m === null ? `banco q4_1: ${FN} deve destrutturare ${CASE}` : "ok").toBe("ok");
    const names = m![1].split(",").map((f) => f.trim());
    const missing = ["K", "N", "M", "splits"].filter((n) => !names.includes(n));
    expect(missing.join(", "), `banco q4_1: dalla destrutturazione di ${CASE} mancano ${missing.join(", ")}`).toBe("");
  });

  it("nelle due compare( del banco q4_1 non c'e' nessun letterale numerico", () => {
    const calls = compareCalls(code(KTEST));
    expect(calls.length, `banco q4_1: compare( nel corpo di ${FN}: ${calls.length}`).toBe(2);
    // una tolleranza scritta a mano qui dentro sarebbe una seconda verita'
    // accanto a quella di prefillkquant: il numero si importa, non si ricopia.
    // I letterali si cercano fuori dagli identificatori (`Float32Array` e
    // `PREFILL_GEMM_Q41_...` hanno cifre, ma non sono numeri).
    const bad = calls.flatMap((args) => {
      const lit = args.match(/(?<![A-Za-z0-9_$])\d[\d._eE+-]*/g);
      return lit ? [`banco q4_1: compare(...) con letterale/i ${lit.join(", ")}: ${args.replace(/\s+/g, " ")}`] : [];
    });
    expect(bad.join("\n")).toBe("");
  });

  it("ogni braccio q4_1 riceve LA SUA coppia di tolleranze, una volta sola", () => {
    // importarle tutte e quattro e poi passare due volte la stessa coppia
    // giudicherebbe una via col pavimento dell'altra.
    const body = bankBody(code(KTEST));
    const bad = TOLS.flatMap((t) => {
      const n = hitsOf(body, t).length;
      return n === 1 ? [] : [`banco q4_1: ${t} usata ${n} volte nel banco (attesa 1)`];
    });
    expect(bad.join("\n")).toBe("");
    // il braccio si riconosce dal NOME DEL CASO che la compare porta come primo
    // argomento: qui si legge il sorgente con le stringhe intatte, cosi' non
    // serve congelare anche i nomi delle variabili di lettura.
    const calls = compareCalls(uncommented(KTEST));
    for (const [name, tols] of [[IDOT, IDOT_TOLS], [F32, F32_TOLS]] as Array<[string, string[]]>) {
      const arm = calls.filter((a) => literalHits(a, name).length === 1);
      expect(arm.length, `banco q4_1: compare( del caso "${name}"`).toBe(1);
      const missing = tols.filter((t) => !has(arm[0], t));
      expect(missing.join(", "), `banco q4_1: al caso "${name}" mancano ${missing.join(", ")} — ${arm[0].replace(/\s+/g, " ")}`).toBe("");
    }
  });
});

describe("(e) i casi q4_1 si aggiungono senza abbassare la soglia del driver", () => {
  // `.harness/tools/engine-ktest.mjs` NON e' posseduto da questo task: si legge
  // SOLA LETTURA per verificare che il default resti 90, cioe' che i due casi
  // nuovi non siano stati fatti entrare abbassando la barra. Che 90 sia SOTTO il
  // numero di casi registrati e' una proprieta' della RUN del driver (i banchi
  // registrano piu' casi di quante siano le chiamate statiche: sweep e cicli), e
  // questo test statico non la simula.
  it("KTEST_MIN_PASS ha ancora default 90", () => {
    const m = /KTEST_MIN_PASS\s*\?\?\s*(\d+)/.exec(raw(DRIVER));
    expect(m === null ? `${DRIVER}: default di KTEST_MIN_PASS non trovato` : "ok").toBe("ok");
    expect(Number(m![1]), `${DRIVER}: KTEST_MIN_PASS default = ${m![1]}`).toBe(90);
  });

  it("il banco q4_1 si AGGIUNGE: nessun results.push rimosso", () => {
    // il conteggio dei banchi puo' solo crescere: se un push sparisce mentre ne
    // entra uno nuovo, il totale resta e la soglia mente.
    const src = code(KTEST);
    const pushes = hitsOf(src, "results", "\\.push\\s*\\(").length;
    expect(pushes, `banco q4_1: results.push( nel ktest = ${pushes}, atteso >= ${PUSHES_BEFORE_Q41 + 1} (i 72 di prima + quello del q4_1)`)
      .toBeGreaterThanOrEqual(PUSHES_BEFORE_Q41 + 1);
  });
});
