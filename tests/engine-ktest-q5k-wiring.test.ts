// Cablaggio del banco Q5_K multi-riga del ktest (goal K-quant, task ktest-q5k).
//
// Test STATICO, senza GPU: stessa postura di `tests/engine-q35attnwiring.test.ts`
// — si SCANSIONA il sorgente dell'orchestratore invece di fidarsi di una
// descrizione a parole. Qui si prova il CABLAGGIO del banco nuovo, non la sua
// aritmetica: che i due casi esistano e siano registrati una volta sola, che
// ogni braccio chiami i suoi generatori con le liste di binding congelate e con
// le GRIGLIE degli helper, che le opzioni portino il `kind` del formato, e che
// shape/seed/tolleranze arrivino da `prefillkquant` invece di essere ricopiati
// a mano qui dentro.
//
// Il verdetto NUMERICO non e' di questo test: lo da' `node
// .harness/tools/engine-ktest.mjs`, che vuole un vite server e un Chrome vero.
// Un ktest dichiarato verde senza averlo eseguito e' il difetto piu' grave
// registrato in questo repo: questo file non lo simula, verifica solo che il
// banco sia agganciato dove il driver lo trovera'.
//
// PERCHE' GRIGLIA E `kind` SONO ASSERITI QUI. La prima stesura di questo file
// contava i quattro generatori di shader e nient'altro: restava verde su un
// sorgente che NON COMPILAVA (griglia importata con un nome inesistente,
// `PrefillGemmOpts` senza il campo `kind` — che a runtime fa scattare il
// rifiuto di `prefillGemmCheck` al primo dispatch). Un test di cablaggio che
// non vede l'unico argomento da cui dipende il passo del superblocco misura il
// contorno del difetto, non il difetto.
//
// Il file si LEGGE come testo. Importarlo tirerebbe dentro i tipi WebGPU e un
// modulo worker (che in vitest node non si carica). Il testo non si scansiona
// grezzo: commenti, stringhe e template WGSL vengono prima BIANCATI
// (§ blankNonCode, helper COPIATO da engine-q35attnwiring.test.ts e non
// importato, perche' quel file e' un test e non un modulo di libreria). Senza
// quel passo un commento che nomina `prefillSplitKCombineWgsl(` sposta i
// conteggi, e una lista `[blocks, xq, part, xsc]` citata in un commento diventa
// la lista che il test misura al posto di quella vera.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const KTEST = "src/engine/ktest/ktest.worker.ts";
const DRIVER = ".harness/tools/engine-ktest.mjs";
// come engine-q35attnwiring.test.ts: percorso ancorato al file di test, non
// alla directory da cui si lancia vitest.
const ROOT = join(__dirname, "..");
const raw = (rel: string): string => readFileSync(join(ROOT, rel), "utf8");

const IDOT = "prefill-gemm-q5k-multirow-idot";
const F32 = "prefill-gemm-q5k-multirow-f32";
const FN = "testPrefillGemmQ5KMultiRow";
const TOLS = [
  "PREFILL_GEMM_Q5K_IDOT_REL_TOL", "PREFILL_GEMM_Q5K_IDOT_ABS_TOL",
  "PREFILL_GEMM_Q5K_F32_REL_TOL", "PREFILL_GEMM_Q5K_F32_ABS_TOL",
];

function blankNonCode(src: string, keepStrings: boolean): string {
  const out = src.split("");
  const blank = (i: number): void => { if (i < out.length && out[i] !== "\n") out[i] = " "; };
  const blankTo = (from: number, to: number): void => { for (let i = from; i < to; i++) blank(i); };
  // dopo uno di questi caratteri una `/` apre un regex, non una divisione
  const VALUE_POS = /^$|[(,=:[!&|?{};+\-*%~^<>]/;
  let prev = "";

  const skipQuoted = (open: number, quote: string): number => {
    if (!keepStrings) blank(open);
    for (let i = open + 1; i < src.length; i++) {
      const c = src[i];
      if (c === "\\") { if (!keepStrings) blankTo(i, i + 2); i++; continue; }
      if (c === "\n") return i; // stringa non chiusa: non mangiarsi il resto del file
      if (!keepStrings) blank(i);
      if (c === quote) return i + 1;
    }
    return src.length;
  };
  const skipRegex = (open: number): number => {
    let cls = false;
    for (let i = open + 1; i < src.length; i++) {
      const c = src[i];
      if (c === "\\") { i++; continue; }
      if (c === "\n") return open + 1; // non era un regex dopotutto: era una divisione
      if (c === "[") cls = true;
      else if (c === "]") cls = false;
      else if (c === "/" && !cls) { blankTo(open, i + 1); return i + 1; }
    }
    return open + 1;
  };
  /** consuma codice fino a fine file, o alla `}` che chiude un `${...}` */
  const code = (start: number, inTemplate: boolean): number => {
    let i = start, brace = 0;
    while (i < src.length) {
      const c = src[i], d = src[i + 1] ?? "";
      if (inTemplate && c === "}" && brace === 0) return i;
      if (c === "/" && d === "/") { while (i < src.length && src[i] !== "\n") blank(i++); continue; }
      if (c === "/" && d === "*") {
        const e = src.indexOf("*/", i + 2), stop = e < 0 ? src.length : e + 2;
        blankTo(i, stop); i = stop; continue;
      }
      if (c === "/" && VALUE_POS.test(prev)) { i = skipRegex(i); prev = "/"; continue; }
      if (c === '"' || c === "'") { i = skipQuoted(i, c); prev = '"'; continue; }
      if (c === "`") { i = skipTemplate(i); prev = "`"; continue; }
      if (c === "{") brace++;
      else if (c === "}") brace--;
      if (!/\s/.test(c)) prev = c;
      i++;
    }
    return i;
  };
  const skipTemplate = (open: number): number => {
    if (!keepStrings) blank(open);
    let i = open + 1;
    while (i < src.length) {
      const c = src[i];
      if (c === "\\") { if (!keepStrings) blankTo(i, i + 2); i += 2; continue; }
      if (c === "`") { if (!keepStrings) blank(i); return i + 1; }
      if (c === "$" && src[i + 1] === "{") {
        if (!keepStrings) blankTo(i, i + 2);
        const close = code(i + 2, true);
        if (!keepStrings) blank(close);
        i = close + 1; continue;
      }
      if (!keepStrings) blank(i);
      i++;
    }
    return i;
  };
  code(0, false);
  return out.join("");
}

const memo = new Map<string, string>();
/** sorgente con commenti/stringhe/template biancati: quello che si conta */
const code = (rel: string): string => {
  const key = `code:${rel}`;
  if (!memo.has(key)) memo.set(key, blankNonCode(raw(rel), false));
  return memo.get(key)!;
};
/** sorgente coi soli commenti biancati: serve agli import e ai literal, che sono stringhe */
const uncommented = (rel: string): string => {
  const key = `unc:${rel}`;
  if (!memo.has(key)) memo.set(key, blankNonCode(raw(rel), true));
  return memo.get(key)!;
};

const lineOf = (src: string, idx: number): number => src.slice(0, idx).split("\n").length;
/** occorrenze di un identificatore (non di una sottostringa qualsiasi) */
const hitsOf = (src: string, ident: string, suffix = ""): number[] =>
  [...src.matchAll(new RegExp(`(?<![A-Za-z0-9_$])${ident}\\s*${suffix}`, "g"))].map((m) => m.index!);

const CLOSE: Record<string, string> = { "(": ")", "[": "]", "{": "}" };

/** indice del delimitatore che chiude il gruppo aperto a `open`, o -1 */
function closeOf(src: string, open: number): number {
  const stack: string[] = [];
  for (let i = open; i < src.length; i++) {
    const c = src[i];
    if (c === "(" || c === "[" || c === "{") stack.push(CLOSE[c]);
    else if (c === ")" || c === "]" || c === "}") {
      if (stack.pop() !== c) return -1;
      if (stack.length === 0) return i;
    }
  }
  return -1;
}

/** elementi al livello superiore di un gruppo `[...]` / `(...)` (virgole a profondita' 0) */
function topLevelItems(src: string, open: number): string[] {
  const end = closeOf(src, open);
  if (end < 0) return [];
  const items: string[] = [];
  let depth = 0, cur = "";
  for (const c of src.slice(open + 1, end)) {
    if (c === "(" || c === "[" || c === "{") depth++;
    else if (c === ")" || c === "]" || c === "}") depth--;
    if (c === "," && depth === 0) { items.push(cur.trim()); cur = ""; continue; }
    cur += c;
  }
  if (cur.trim().length > 0) items.push(cur.trim());
  return items;
}

interface Call { at: number; line: number; args: string[]; bindings: string[] | null }

/**
 * Chiamate a `name(`, ognuna coi SUOI argomenti e con la SUA lista di binding.
 * La lista dev'essere INLINE subito dopo la chiamata (solo spazi e virgole in
 * mezzo): se e' passata per variabile, `bindings` resta null e il test lo DICE,
 * invece di agganciare il primo `[` che trova piu' avanti — che sarebbe un
 * array scorrelato.
 */
function callsTo(src: string, name: string): Call[] {
  return hitsOf(src, name, "\\(").map((at) => {
    const open = src.indexOf("(", at);
    const end = closeOf(src, open);
    let j = end + 1;
    while (end > 0 && j < src.length && /[\s,]/.test(src[j])) j++;
    return {
      at,
      line: lineOf(src, at),
      args: topLevelItems(src, open),
      bindings: end > 0 && src[j] === "[" ? topLevelItems(src, j) : null,
    };
  });
}

/** contenuto di ogni `import { ... } from "<src>"` che viene da `suffix` */
const importsFrom = (src: string, suffix: string): string =>
  [...src.matchAll(/import\s+(?:type\s+)?\{([^}]*)\}\s*from\s*["']([^"']+)["']/g)]
    .filter((m) => m[2].endsWith(suffix)).map((m) => m[1].replace(/\s+/g, " ").trim()).join(" ; ");

const has = (list: string, ident: string): boolean =>
  new RegExp(`(?<![A-Za-z0-9_$])${ident}(?![A-Za-z0-9_$])`).test(list);

/**
 * Definizione di un helper, sia nella forma `function f(` sia in quella
 * `const f = (`. Ritorna il corpo `{...}`: il primo gruppo graffo a profondita'
 * di parentesi 0 dopo il nome (cosi' parametri destrutturati e tipi inline non
 * lo confondono).
 */
function helperBody(src: string, name: string): string | null {
  const def = new RegExp(`(?:function\\s+${name}\\s*\\(|const\\s+${name}\\s*=)`).exec(src);
  if (!def) return null;
  let depth = 0;
  for (let i = def.index; i < src.length; i++) {
    const c = src[i];
    if (c === "(" || c === "[") depth++;
    else if (c === ")" || c === "]") depth--;
    else if (c === "{" && depth === 0) {
      const end = closeOf(src, i);
      return end < 0 ? null : src.slice(i, end + 1);
    }
  }
  return null;
}

/** corpo del banco nuovo, o un messaggio che dice DOVE manca */
function bankBody(src: string): string {
  const body = helperBody(src, FN);
  expect(body === null ? `${KTEST}: ${FN} non e' definito` : "ok").toBe("ok");
  return body!;
}

/** quante volte una stringa compare come LETTERALE (commenti gia' biancati) */
const literalHits = (src: string, s: string): number[] =>
  [...src.matchAll(new RegExp(`["'\`]${s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}["'\`]`, "g"))].map((m) => m.index!);

describe("(a) i due casi nuovi esistono, e ciascuno una volta sola", () => {
  for (const name of [IDOT, F32]) {
    it(`"${name}" e' un nome di caso, dichiarato una volta`, () => {
      const src = uncommented(KTEST); // i nomi sono stringhe: qui si tengono
      const where = literalHits(src, name).map((i) => `${KTEST}:${lineOf(src, i)}`);
      expect(where.length, `"${name}" a ${where.join(", ") || "NESSUNA riga"}`).toBe(1);
    });
  }
});

describe("(b) il banco e' definito una volta e registrato una volta", () => {
  it(`${FN}( compare 2 volte: definizione + registrazione`, () => {
    const src = code(KTEST);
    const where = hitsOf(src, FN, "\\(").map((i) => `${KTEST}:${lineOf(src, i)}`);
    expect(where.length, `${FN}( a ${where.join(", ") || "NESSUNA riga"}`).toBe(2);
  });

  it(`una delle due e' la definizione, l'altra e' dentro results.push`, () => {
    const src = code(KTEST);
    const lines = src.split("\n");
    const hits = hitsOf(src, FN, "\\(").map((i) => lineOf(src, i));
    const def = hits.filter((n) => new RegExp(`function\\s+${FN}\\s*\\(`).test(lines[n - 1]));
    // lo SPREAD e' la firma del banco a due risultati: `results.push(await …)`
    // compilerebbe pure, e spingerebbe un array dentro la lista dei KResult.
    const reg = hits.filter((n) => new RegExp(`results\\.push\\(\\s*\\.\\.\\.\\s*await\\s+${FN}\\s*\\(`).test(lines[n - 1]));
    expect(def.length, `definizione di ${FN} a riga ${def.join(", ") || "NESSUNA"}`).toBe(1);
    expect(reg.length, `results.push(...await ${FN}(...)) a riga ${reg.join(", ") || "NESSUNA"}`).toBe(1);
  });
});

describe("(c) ogni braccio chiama i suoi generatori con i binding congelati", () => {
  // Le liste sono quelle dei kernel: idot [blocks, xq, part, xsc] (le
  // attivazioni quantizzate e le loro scale sono DUE buffer distinti), f32
  // [blocks, x, part] (x grezza), combine [part, y]. La quantizzazione segue
  // l'ordine di binding del suo generatore: [x, xq, xsc].
  const EXPECT: Array<{ gen: string; times: number; bindings: string[] }> = [
    { gen: "prefillQuantXQ8Wgsl", times: 1, bindings: ["x", "xq", "xsc"] },
    { gen: "prefillGemmQ5KSplitKIdotWgsl", times: 1, bindings: ["blocks", "xq", "part", "xsc"] },
    { gen: "prefillGemmQ5KSplitKWgsl", times: 1, bindings: ["blocks", "x", "part"] },
    { gen: "prefillSplitKCombineWgsl", times: 2, bindings: ["part", "y"] },
  ];
  for (const e of EXPECT) {
    it(`${e.gen}( ${e.times} volta/e, binding [${e.bindings.join(", ")}]`, () => {
      // NEL CORPO DEL BANCO, non nel file: da riga 3 esiste anche il banco
      // q4_1, che usa gli STESSI generatori condivisi (`prefillQuantXQ8Wgsl` e
      // `prefillSplitKCombineWgsl`). Contarli su tutto il file misurava
      // "quanti banchi esistono", non "come e' cablato questo" — e sarebbe
      // tornato rosso a ogni banco nuovo, per una ragione che non ha niente a
      // che vedere con cio' che il test difende.
      const calls = callsTo(bankBody(code(KTEST)), e.gen);
      const all = calls.map((c) => `${KTEST}:${c.line}`);
      expect(calls.length, `${e.gen}( a ${all.join(", ") || "NESSUNA riga"}`).toBe(e.times);
      const bad = calls.flatMap((c) => {
        const where = `${KTEST}:${c.line}`;
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
});

describe("(c2) le griglie sono quelle degli helper, e le opzioni portano il kind", () => {
  // Il difetto che questo blocco esiste per prendere: la prima stesura importava
  // una `prefillGemmQ5KGrid` che NON esiste (l'helper vero e' `prefillGemmGrid`,
  // lo stesso che usa `q35gpumodel`) e costruiva le opzioni senza `kind`. Nessun
  // conteggio di generatori se ne accorge, perche' il difetto sta negli
  // argomenti — e un import rotto ferma TUTTO il worker, non solo i due casi
  // nuovi.
  // `prefillGemmGrid` e `prefillCombineGrid` DUE volte — una per braccio, che
  // sono due catene complete e non due varianti dello stesso dispatch;
  // `prefillQuantXGrid` una sola, perche' le attivazioni si quantizzano solo
  // per la via intera.
  const GRIDS: Array<[string, number]> = [
    ["prefillGemmGrid", 2], ["prefillQuantXGrid", 1], ["prefillCombineGrid", 2],
  ];

  it.each(GRIDS)("%s( compare %i volta/e nel banco, e viene importato da ../kernels/wgsl", (g, want) => {
    const body = bankBody(code(KTEST));
    expect(hitsOf(body, g, "\\(").length, `${g}( nel corpo di ${FN}`).toBe(want);
    const imported = importsFrom(uncommented(KTEST), "kernels/wgsl");
    expect(has(imported, g), `${g} dev'essere importato da ../kernels/wgsl`).toBe(true);
  });

  it("nessuna griglia ri-derivata a mano dentro il banco", () => {
    // una `Math.ceil((M*N)/64)` scritta qui sarebbe una seconda copia della
    // geometria di un kernel che vive in un altro file.
    expect(bankBody(code(KTEST)), `griglia ricalcolata a mano in ${FN}`).not.toMatch(/Math\.ceil\s*\(/);
  });

  it("i due moltiplicatori e la loro griglia ricevono LE STESSE opzioni", () => {
    const body = bankBody(code(KTEST));
    const bad: string[] = [];
    const names = ["prefillGemmQ5KSplitKIdotWgsl", "prefillGemmQ5KSplitKWgsl", "prefillGemmGrid"];
    const args = names.flatMap((n) => callsTo(body, n).map((c) => ({ n, a: c.args })));
    for (const { n, a } of args) {
      if (a.length !== 1 || !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(a[0])) {
        bad.push(`${n}(${a.join(", ")}): atteso UN identificatore di opzioni`);
      }
    }
    const ids = [...new Set(args.map((x) => x.a[0]))];
    expect(bad.join("\n")).toBe("");
    expect(ids.length, `opzioni distinte fra ${names.join("/")}: ${ids.join(", ")}`).toBe(1);
  });

  it("le opzioni sono un PrefillGemmOpts con kind \"q5_K\"", () => {
    // `kind` sceglie il passo del superblocco: senza, `prefillGemmCheck` rifiuta
    // la shape al primo dispatch, e il tipo non compila.
    const body = bankBody(uncommented(KTEST)); // il valore di `kind` e' una stringa
    const decl = /const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*PrefillGemmOpts\s*=\s*\{/.exec(body);
    expect(decl === null ? `${KTEST}: in ${FN} manca una const tipizzata PrefillGemmOpts` : "ok").toBe("ok");
    const obj = body.slice(body.indexOf("{", decl!.index + decl![0].length - 1));
    const fields = topLevelItems(obj, 0).map((f) => f.replace(/\s+/g, ""));
    expect(fields, `campi delle opzioni: ${fields.join(", ")}`).toContain('kind:"q5_K"');
    const imported = importsFrom(uncommented(KTEST), "kernels/wgsl");
    expect(has(imported, "PrefillGemmOpts"), "PrefillGemmOpts dev'essere importato da ../kernels/wgsl").toBe(true);
  });
});

describe("(d) shape, seed e tolleranze arrivano da prefillkquant", () => {
  it("il ktest importa il caso e le quattro tolleranze da ../prefillkquant", () => {
    const src = uncommented(KTEST); // gli import sono stringhe: qui non si biancano
    const imported = importsFrom(src, "prefillkquant");
    const missing = ["PREFILL_Q5K_KTEST_CASE", ...TOLS].filter((n) => !has(imported, n));
    expect(missing.join(", "), `import da prefillkquant = "${imported}" — mancano: ${missing.join(", ")}`).toBe("");
  });

  it("shape e seed del caso NON sono ricopiati: escono dalla destrutturazione", () => {
    const body = bankBody(code(KTEST));
    expect(body, `${FN} deve destrutturare PREFILL_Q5K_KTEST_CASE`)
      .toMatch(/const\s*\{[^}]*\}\s*=\s*PREFILL_Q5K_KTEST_CASE/);
  });

  it("nelle due compare( del banco nuovo non c'e' nessun letterale numerico", () => {
    const body = bankBody(code(KTEST));
    const calls = hitsOf(body, "compare", "\\(");
    expect(calls.length, `compare( nel corpo di ${FN}: ${calls.length}`).toBe(2);
    // una tolleranza scritta a mano qui dentro sarebbe una seconda verita'
    // accanto a quella di prefillkquant: il numero si importa, non si ricopia.
    // I letterali si cercano fuori dagli identificatori (`Float32Array` e
    // `PREFILL_GEMM_Q5K_...` hanno cifre, ma non sono numeri).
    const bad = calls.flatMap((at) => {
      const open = body.indexOf("(", at);
      const args = body.slice(open, closeOf(body, open) + 1);
      const lit = args.match(/(?<![A-Za-z0-9_$])\d[\d._eE+-]*/g);
      return lit ? [`compare(...) con letterale/i ${lit.join(", ")}: ${args.replace(/\s+/g, " ")}`] : [];
    });
    expect(bad.join("\n")).toBe("");
  });

  it("ogni braccio riceve LA SUA coppia di tolleranze, una volta sola", () => {
    // importarle tutte e quattro e poi passare due volte la stessa coppia
    // giudicherebbe una via col pavimento dell'altra.
    const body = bankBody(code(KTEST));
    const bad = TOLS.flatMap((t) => {
      const n = hitsOf(body, t).length;
      return n === 1 ? [] : [`${t} usata ${n} volte nel banco (attesa 1)`];
    });
    expect(bad.join("\n")).toBe("");
    const calls = hitsOf(body, "compare", "\\(").map((at) => {
      const open = body.indexOf("(", at);
      return body.slice(open, closeOf(body, open) + 1);
    });
    // il braccio si riconosce dai buffer letti: `gotIdot` / `gotF32` sono
    // argomenti, non stringhe, quindi sopravvivono al biancamento.
    const idot = calls.filter((a) => /Idot/.test(a));
    const f32 = calls.filter((a) => /F32/.test(a));
    expect(idot.length, "compare( del braccio intero").toBe(1);
    expect(f32.length, "compare( del braccio f32").toBe(1);
    expect(has(idot[0], "PREFILL_GEMM_Q5K_IDOT_REL_TOL") && has(idot[0], "PREFILL_GEMM_Q5K_IDOT_ABS_TOL"),
      `tolleranze del braccio intero: ${idot[0].replace(/\s+/g, " ")}`).toBe(true);
    expect(has(f32[0], "PREFILL_GEMM_Q5K_F32_REL_TOL") && has(f32[0], "PREFILL_GEMM_Q5K_F32_ABS_TOL"),
      `tolleranze del braccio f32: ${f32[0].replace(/\s+/g, " ")}`).toBe(true);
  });
});

describe("(e) la soglia del driver non e' stata abbassata", () => {
  // `.harness/tools/engine-ktest.mjs` NON e' posseduto da questo task: qui si
  // verifica solo che il default resti 90, cioe' che i due casi nuovi non
  // siano stati fatti entrare abbassando la barra. Che 90 sia SOTTO il numero
  // di casi registrati e' una proprieta' della RUN del driver (i banchi
  // registrano piu' casi di quante siano le chiamate statiche: sweep e cicli),
  // e questo test statico non la simula.
  it("KTEST_MIN_PASS ha ancora default 90", () => {
    const m = /KTEST_MIN_PASS\s*\?\?\s*(\d+)/.exec(raw(DRIVER));
    expect(m === null ? `${DRIVER}: default di KTEST_MIN_PASS non trovato` : "ok").toBe("ok");
    expect(Number(m![1]), `${DRIVER}: KTEST_MIN_PASS default = ${m![1]}`).toBe(90);
  });

  it("i due casi nuovi si AGGIUNGONO: nessun results.push rimosso", () => {
    // il conteggio dei banchi puo' solo crescere: se un push sparisce mentre
    // ne entra uno nuovo, il totale resta e la soglia mente.
    const src = code(KTEST);
    const pushes = hitsOf(src, "results", "\\.push\\s*\\(").length;
    expect(pushes, `results.push( nel ktest: ${pushes}`).toBeGreaterThanOrEqual(72);
  });
});
