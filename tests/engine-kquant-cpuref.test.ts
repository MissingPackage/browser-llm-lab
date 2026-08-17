// Conformance per formato contro il riferimento CPU: i banchi ktest di Q2_K e
// Q3_K (spec 2026-08-17-q2k-q3k-kernels, T5).
//
// Test STATICO, senza GPU: stessa postura di `tests/engine-ktest-q5k-wiring.test.ts`
// e `tests/engine-ktest-kquant35b-wiring.test.ts` — si SCANSIONA il sorgente
// dell'orchestratore invece di fidarsi di una descrizione a parole. Il verdetto
// NUMERICO non e' di questo file: lo da' `node .harness/tools/engine-ktest.mjs`,
// che vuole un vite server e un Chrome vero, e che questa build NON esegue (la
// GPU e' una sola e la contesa fra agenti e' un difetto gia' pagato da questo
// progetto). Qui si prova solo che i quattro banchi esistano, che siano
// REGISTRATI dove il driver li trovera', che confrontino col dequant di
// `quant.ts` e che il pavimento sia quello dei K-quant gia' esistenti.
//
// PERCHE' LA REGISTRAZIONE E' META' DEL TEST. Un banco scritto e non
// registrato non viene MAI eseguito: compila, si legge bene in review, e non
// misura niente. E' il difetto che questo repo ha gia' pagato una volta
// («un ktest dichiarato verde senza averlo eseguito»), quindi il conteggio dei
// `results.push(` e' pinnato e deve crescere ESATTAMENTE di quattro: se un
// push sparisse mentre ne entrano quattro, il totale mentirebbe e la soglia
// del driver con lui.
//
// PERCHE' LA SOGLIA E' UN'ASSERZIONE E NON UNA NOTA. Un formato nuovo che non
// passa a 2e-4/1e-3 e' un kernel da correggere, non una tolleranza da
// allargare: il modo piu' economico di far diventare verde un banco rotto e'
// scrivergli accanto un numero piu' grande. Qui si pinna che il verdetto dei
// sei formati C2 esca da UN SOLO `compare(`, con LA STESSA coppia di prima.
//
// Il file si LEGGE come testo. Importarlo tirerebbe dentro i tipi WebGPU e un
// modulo worker (che in vitest node non si carica). Il testo non si scansiona
// grezzo: commenti, stringhe e template WGSL vengono prima BIANCATI
// (`blankNonCode` in `tests/helpers/source-scan.ts`). Senza quel passo un
// commento che nomina `testGemvC2(` sposta i conteggi.
import { describe, expect, it } from "vitest";
import {
  code, uncommented, codeNoImports, hitsOf, callsTo,
  importsFrom, has, helperRange, helperBody, numLiterals,
} from "./helpers/source-scan";

const KTEST = "src/engine/ktest/ktest.worker.ts";

/** il banco C2 e' UNO SOLO, parametrico sul formato: i due nuovi lo estendono */
const FN = "testGemvC2";
/** il descrittore per formato che il banco consuma: una riga per formato */
const TABLE = "C2_FMT";

/**
 * Le SHAPE degli expert del 35B, che sono il motivo per cui questi due formati
 * entrano nel motore: gate/up leggono K=2048 e scrivono N=512, il down fa il
 * contrario. Non sono taglie campione — sono quelle su cui il kernel girera'.
 */
const KINDS = ["q2_K", "q3_K"];
const SHAPES: Array<[number, number]> = [[2048, 512], [512, 2048]];
const BANKS = KINDS.flatMap((kind) => SHAPES.map(([K, N]) => ({ kind, K, N })));

/**
 * Quanti `results.push(` c'erano nel ktest PRIMA di questi quattro banchi.
 * Pinnato: i banchi nuovi devono AGGIUNGERE, non sostituire. Il conteggio si
 * misura sul sorgente biancato, cosi' un `results.push(` citato in un commento
 * non lo sposta.
 */
const PUSHES_BEFORE_Q2K_Q3K = 76;

/**
 * La coppia di tolleranze dei casi K-quant gia' esistenti (`relTol`, `absTol`
 * nell'ordine di `compare`). E' il pavimento che q4_K/q5_K/q6_K passano oggi:
 * i due formati nuovi lo ereditano perche' passano dallo STESSO `compare(`,
 * non perche' qualcuno abbia ricopiato due numeri.
 */
const KQUANT_C2_TOL = ["2e-4", "1e-3"];

/** la firma normalizzata di una chiamata: `g, "q2_K", 2048, 512` */
const sigOf = (args: string[]): string => args.map((a) => a.replace(/\s+/g, "")).join(", ");
/** la firma attesa di un banco */
const wantSig = (b: { kind: string; K: number; N: number }): string =>
  ["g", `"${b.kind}"`, String(b.K), String(b.N)].join(", ");

/** corpo di un helper, o un fallimento che dice DOVE manca */
function bodyOf(src: string, name: string): string {
  const body = helperBody(src, name);
  expect(body === null ? `${KTEST}: ${name} non e' definito` : "ok").toBe("ok");
  return body!;
}

describe("(a) i quattro banchi q2_K/q3_K esistono, sulle shape degli expert del 35B", () => {
  for (const b of BANKS) {
    it(`${FN}(g, "${b.kind}", ${b.K}, ${b.N}) esiste una volta sola`, () => {
      // gli argomenti si leggono sul sorgente coi soli COMMENTI biancati: il
      // formato e' una stringa, e il biancamento delle stringhe la farebbe
      // sparire proprio dall'argomento che qui distingue un banco dall'altro.
      const src = uncommented(KTEST);
      const calls = callsTo(src, FN);
      const found = calls.filter((c) => sigOf(c.args) === wantSig(b));
      const all = calls.map((c) => `${KTEST}:${c.line} ${FN}(${c.args.join(", ")})`);
      expect(found.length, `atteso UN ${FN}(${wantSig(b)}); chiamate viste:\n${all.join("\n")}`).toBe(1);
    });
  }

  it("per ogni formato le due shape sono INVERSE l'una dell'altra, non la stessa due volte", () => {
    // un copia-incolla che lascia la stessa shape due volte darebbe quattro
    // registrazioni e due coperture: il gate e' che ogni formato porti una
    // coppia (K,N) E la sua inversa (N,K) — che e' il rapporto vero fra il
    // gate/up e il down di un expert, e si LEGGE dal sorgente invece di essere
    // ricopiato qui accanto.
    const calls = callsTo(uncommented(KTEST), FN);
    for (const kind of KINDS) {
      const shapes = calls
        .map((c) => c.args.map((a) => a.replace(/\s+/g, "")))
        .filter((a) => a[1] === `"${kind}"`)
        .map((a) => [a[2], a[3]] as [string, string]);
      const seen = shapes.map(([K, N]) => `${K}x${N}`);
      expect(new Set(seen).size, `${kind}: shape distinte = ${seen.join(", ") || "NESSUNA"}`).toBe(2);
      const inverted = shapes.some(([K, N]) => shapes.some(([K2, N2]) => K2 === N && N2 === K));
      expect(inverted, `${kind}: le due shape non sono l'una l'inversa dell'altra — ${seen.join(", ")}`).toBe(true);
    }
  });
});

describe("(b) i quattro banchi sono REGISTRATI nell'elenco che il ktest esegue", () => {
  for (const b of BANKS) {
    it(`${b.kind} ${b.K}x${b.N}: results.push(await ${FN}(...)) dentro il corpo di main()`, () => {
      const src = uncommented(KTEST);
      const call = callsTo(src, FN).filter((c) => sigOf(c.args) === wantSig(b))[0];
      expect(call === undefined ? `${KTEST}: manca ${FN}(${wantSig(b)})` : "ok").toBe("ok");
      // la registrazione si riconosce dal testo che PRECEDE la chiamata, non
      // dalla riga: `results.push(\n  await testGemvC2(...)` e' legittimo.
      const before = src.slice(Math.max(0, call.at - 96), call.at);
      expect(/results\s*\.\s*push\s*\(\s*await\s+$/.test(before),
        `${KTEST}:${call.line} — ${FN}(${wantSig(b)}) non e' dentro un results.push(await ...): il driver non lo eseguirebbe`).toBe(true);
      const main = helperRange(src, "main");
      expect(main === null ? `${KTEST}: main() non trovato` : "ok").toBe("ok");
      expect(call.at > main!.from && call.at < main!.to,
        `${KTEST}:${call.line} — registrazione FUORI dal corpo di main(): compila e non viene mai eseguita`).toBe(true);
    });
  }

  it(`i results.push( passano da ${PUSHES_BEFORE_Q2K_Q3K} a ${PUSHES_BEFORE_Q2K_Q3K + BANKS.length}: i banchi si AGGIUNGONO`, () => {
    // uguaglianza e non `>=`: quattro banchi nuovi e un push sparito darebbero
    // 79 — e la soglia del driver (KTEST_MIN_PASS) resterebbe soddisfatta da un
    // insieme di casi diverso da quello che si e' dichiarato.
    const pushes = hitsOf(code(KTEST), "results", "\\.push\\s*\\(").length;
    expect(pushes, `results.push( nel ktest = ${pushes}, atteso ${PUSHES_BEFORE_Q2K_Q3K} + ${BANKS.length}`)
      .toBe(PUSHES_BEFORE_Q2K_Q3K + BANKS.length);
  });
});

describe("(c) il confronto e' col dequant di quant.ts, e col gemv di wgsl.ts", () => {
  const NEEDED: Array<[string, string[]]> = [
    // il riferimento CPU: `dequantQ2_K`/`dequantQ3_K` sono gia' verificati
    // byte-identici a llama-quantize, quindi il banco non ha una seconda
    // verita' da mantenere. I byte per superblocco vengono da li' per lo
    // stesso motivo: 84 e 110 scritti a mano qui sarebbero una copia.
    ["../quant", ["dequantQ2_K", "dequantQ3_K", "Q2_K_BLOCK_BYTES", "Q3_K_BLOCK_BYTES"]],
    ["../kernels/wgsl", ["gemvQ2KWgsl", "gemvQ3KWgsl"]],
  ];

  for (const [mod, idents] of NEEDED) {
    it(`il ktest importa ${idents.join(", ")} da ${mod}`, () => {
      const imported = importsFrom(uncommented(KTEST), mod);
      const missing = idents.filter((n) => !has(imported, n));
      expect(missing.join(", "), `import da ${mod} = "${imported}" — mancano: ${missing.join(", ")}`).toBe("");
    });

    it(`${idents.join(", ")}: importati E usati (un import morto non prova niente)`, () => {
      const body = codeNoImports(KTEST);
      const bad = idents.filter((n) => hitsOf(body, n).length === 0);
      expect(bad.join(", "), `identificatori importati e mai usati: ${bad.join(", ")}`).toBe("");
    });
  }

  it(`${FN} riempie il riferimento con un dequant, non con un'aritmetica sua`, () => {
    // il banco non deve ri-derivare i pesi: se lo facesse, un errore di layout
    // finirebbe in ENTRAMBI i lati del confronto e il banco resterebbe verde.
    const body = bodyOf(code(KTEST), FN);
    expect(/(?<![A-Za-z0-9_$])(dequant[A-Za-z0-9_]*|[A-Za-z0-9_$.]*\.dequant)\s*\(/.test(body),
      `nel corpo di ${FN} non compare nessuna chiamata a un dequant`).toBe(true);
  });

  it(`nel descrittore ${TABLE} ogni fatto dei due formati nuovi sta in UNA riga sola`, () => {
    // Ambito: il DESCRITTORE del banco C2, non il worker intero — altrove
    // (prefill, MoE, expert store) gli stessi nomi possono comparire quante
    // volte serve, ed e' cosi' che il q5_K e il q6_K vivono oggi. Qui invece
    // due righe che nominano `Q2_K_BLOCK_BYTES` vorrebbero dire due
    // descrizioni dello stesso formato: la seconda puo' divergere in silenzio,
    // ed e' il modo in cui un banco finisce per misurare un formato diverso da
    // quello che dichiara.
    const table = bodyOf(codeNoImports(KTEST), TABLE);
    const bad = ["Q2_K_BLOCK_BYTES", "Q3_K_BLOCK_BYTES", "dequantQ2_K", "dequantQ3_K", "gemvQ2KWgsl", "gemvQ3KWgsl"]
      .flatMap((n) => {
        const uses = hitsOf(table, n).length;
        return uses === 1 ? [] : [`${n}: ${uses} usi in ${TABLE} (atteso 1)`];
      });
    expect(bad.join("\n")).toBe("");
  });
});

describe("(d) la soglia e' quella dei K-quant esistenti: nessun pavimento nuovo", () => {
  it(`${FN} ha UN SOLO compare(, con le tolleranze ${KQUANT_C2_TOL.join(", ")}`, () => {
    const body = bodyOf(code(KTEST), FN);
    const calls = callsTo(body, "compare");
    expect(calls.length, `compare( nel corpo di ${FN}: ${calls.length} (atteso 1: i sei formati condividono il verdetto)`).toBe(1);
    // gli ultimi due argomenti di `compare(kernel, got, ref, relTol, absTol)`
    const tol = calls[0].args.slice(-2).map((a) => a.replace(/\s+/g, ""));
    expect(tol.join(", "), `compare(${calls[0].args.join(", ")})`).toBe(KQUANT_C2_TOL.join(", "));
  });

  it("nessuna tolleranza nuova per q2_K/q3_K nel worker", () => {
    // ne' una costante dedicata (`..._Q2K_..._TOL`), ne' una coppia di numeri
    // sulle righe di registrazione: i quattro banchi passano dal `compare` di
    // sopra, e il solo modo di ammorbidirli sarebbe cambiarlo per tutti e sei.
    const src = code(KTEST);
    const consts = [...src.matchAll(/(?<![A-Za-z0-9_$])[A-Z][A-Z0-9_]*Q[23]K[A-Z0-9_]*TOL(?![A-Za-z0-9_$])/g)].map((m) => m[0]);
    expect(consts.join(", "), `costanti di tolleranza dedicate ai formati nuovi: ${consts.join(", ")}`).toBe("");

    const bad = callsTo(uncommented(KTEST), FN)
      .filter((c) => /"q[23]_K"/.test(c.args.join(",")))
      .flatMap((c) => {
        const extra = c.args.slice(4);
        const lits = numLiterals(c.args.slice(2).join(",")).filter((n) => !["2048", "512"].includes(n));
        return extra.length > 0 || lits.length > 0
          ? [`${KTEST}:${c.line} ${FN}(${c.args.join(", ")}): argomenti oltre shape`]
          : [];
      });
    expect(bad.join("\n")).toBe("");
  });
});
