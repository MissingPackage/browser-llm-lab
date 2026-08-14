// CABLAGGIO della via veloce Q5_K nel prefill (goal engine-kquant, riga 2).
//
// PERCHE' QUESTO FILE ESISTE, in una riga: il kernel q5_K e il piano che lo
// instrada esistono gia' e sono testati (engine-prefillgemm, engine-
// prefillgemmplan), ma finche' NESSUNO li chiama il motore emette esattamente
// gli stessi dispatch di ieri. E' il difetto di it.14 in forma nuova — allora
// `prefillgemmplan.ts` era completo e nessuno lo importava — e nessun test di
// aritmetica se ne accorge, perche' l'aritmetica e' giusta: e' il CABLAGGIO che
// manca.
//
// L'ALTRA META' DEL PUNTO: il consumatore vero del q5_K NON passa da `gemvB`.
// `ssm_out` (Q5_K, 24 layer del 4B, 173 MB) e' caricato dal ramo K-quant di
// `loadW`, che ha il suo `pushB` e non tocca `gemvB` nemmeno per sbaglio —
// `gemvB` prende `{qs, scales}`, il ramo K-quant ha `{blocks}`. Cablare solo
// `gemvB` non cambierebbe UNA riga del comportamento reale, e passerebbe
// qualunque test scritto su `gemvB`. Per questo qui si asserisce DOVE stanno le
// chiamate, non solo quante sono.
//
// POSTURA (tests/engine-q35attnwiring.test.ts): il file si LEGGE come testo —
// importarlo tirerebbe dentro i tipi WebGPU — e il testo si scansiona BIANCATO.
// Commenti, stringhe e template WGSL vengono sostituiti da spazi prima di
// contare: in q35gpumodel.ts ci sono commenti che nominano `planPrefillGemm`,
// `gemvB` e le liste di binding, e senza quel passo un commento diventa la
// chiamata che il test misura — falso verde su un cablaggio rotto, falso rosso
// su uno corretto. L'helper si COPIA e non si importa: engine-q35attnwiring non
// e' un modulo di libreria e non e' fra i file che questo task possiede.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { planPrefillGemm } from "../src/engine/prefillgemmplan";
import { PREFILL_SPLITS_MEASURED } from "../src/engine/kernels/wgsl";
import type { PrefillQuantKind } from "../src/engine/prefillbytes";

const MODEL = "src/engine/q35gpumodel.ts";
// ancorato al file di test, non alla directory da cui si lancia vitest
const ROOT = join(__dirname, "..");
const raw = (rel: string): string => readFileSync(join(ROOT, rel), "utf8");

/**
 * Sostituisce con spazi (le righe si conservano) tutto cio' che non e' codice:
 * commenti di riga e di blocco, letterali regex e — se `keepStrings` e' falso —
 * stringhe e template literal. Le interpolazioni `${...}` restano codice,
 * perche' codice sono. I template vanno biancati anche solo per contare le
 * parentesi: i kernel WGSL vivono li' dentro e le loro graffe non sono JS.
 *
 * Limite noto: un letterale regex in posizione di valore dopo una PAROLA
 * CHIAVE (`return /x/.test(s)`) viene letto come divisione. Nel file scansionato
 * i soli regex stanno dopo `(`, che e' riconosciuto.
 */
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
  const code0 = (start: number, inTemplate: boolean): number => {
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
        const close = code0(i + 2, true);
        if (!keepStrings) blank(close);
        i = close + 1; continue;
      }
      if (!keepStrings) blank(i);
      i++;
    }
    return i;
  };
  code0(0, false);
  return out.join("");
}

const memo = new Map<string, string>();
/** sorgente con commenti/stringhe/template biancati: quello che si conta */
const code = (rel: string): string => {
  const key = `code:${rel}`;
  if (!memo.has(key)) memo.set(key, blankNonCode(raw(rel), false));
  return memo.get(key)!;
};
/** sorgente coi soli commenti biancati: serve agli import, che sono stringhe */
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

interface Call { at: number; line: number; batch: boolean; bindings: string[] | null }

/**
 * Chiamate a `name(`, ognuna con la SUA lista di binding. La lista deve essere
 * INLINE subito dopo la chiamata (solo spazi e virgole in mezzo): se e' passata
 * per variabile, `bindings` resta null e il test lo DICE, invece di agganciare
 * il primo `[` che trova piu' avanti — che sarebbe un array scorrelato.
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
      batch: end > 0 && /batch\s*:\s*true/.test(src.slice(open, end)),
      bindings: end > 0 && src[j] === "[" ? topLevelItems(src, j) : null,
    };
  });
}

/**
 * Corpo di un helper `function f(` o `const f = (…) => {`. Ritorna il primo
 * gruppo graffo a profondita' di parentesi 0 dopo il nome, cosi' i parametri
 * destrutturati e i tipi inline fra parentesi non lo confondono.
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

/**
 * Corpo `{…}` dell'`if (…)` che contiene `needle`. `helperBody` non serve qui:
 * `loadW` e' dichiarata `const loadW = async (…): Promise<{…}> => {`, e il tipo
 * di ritorno porta una graffa a profondita' di parentesi 0 — cioe' `helperBody`
 * restituirebbe l'ANNOTAZIONE DI TIPO al posto del corpo. Il ramo si prende dal
 * suo `if`, che e' anche cio' che il contratto nomina.
 */
function ifBlockContaining(src: string, needle: string): { body: string; at: number } | null {
  const at = src.indexOf(needle);
  if (at < 0) return null;
  const iff = src.lastIndexOf("if (", at);
  if (iff < 0) return null;
  const cond = closeOf(src, src.indexOf("(", iff));
  if (cond < 0) return null;
  const brace = src.indexOf("{", cond);
  const end = brace < 0 ? -1 : closeOf(src, brace);
  return end < 0 ? null : { body: src.slice(brace, end + 1), at: iff };
}

/** Il ramo K-quant di `loadW`: quello che carica Q4_K/Q5_K/Q6_K. */
function kquantBranch(src: string): { body: string; at: number } {
  const b = ifBlockContaining(src, "GGML_TYPE.Q4_K");
  expect(b, "ramo K-quant di loadW non trovato in q35gpumodel.ts").not.toBeNull();
  return b!;
}

/**
 * Coppie `chiave -> testo del valore` al livello superiore dell'oggetto che si
 * apre a `brace`. Lo shorthand (`{ splits }`) si mappa su se stesso, cosi' non
 * sparisce dal confronto.
 */
function objAt(src: string, brace: number): Map<string, string> {
  const m = new Map<string, string>();
  for (const item of topLevelItems(src, brace)) {
    const i = item.indexOf(":");
    if (i < 0) m.set(item.trim(), item.trim());
    else m.set(item.slice(0, i).trim(), item.slice(i + 1).trim());
  }
  return m;
}

/**
 * L'oggetto passato come PRIMO argomento a `name(`. Null se l'argomento non e'
 * un letterale oggetto: il test lo dice, invece di confrontare il vuoto col
 * vuoto e dichiararsi verde.
 */
function argObjOf(src: string, name: string): Map<string, string> | null {
  const c = callsTo(src, name)[0];
  if (!c) return null;
  const open = src.indexOf("(", c.at);
  const end = closeOf(src, open);
  const brace = src.indexOf("{", open);
  return brace < 0 || end < 0 || brace > end ? null : objAt(src, brace);
}

/** L'oggetto di una `const <nome> = { … }`. */
function constObjOf(src: string, ident: string): Map<string, string> | null {
  const d = new RegExp(`const\\s+${ident}\\s*(?::[^=;]*)?=`).exec(src);
  if (!d) return null;
  const brace = src.indexOf("{", d.index);
  return brace < 0 ? null : objAt(src, brace);
}

/** La condizione dell'`if (…)` che segue l'indice `from`. */
function condAfter(src: string, from: number): string | null {
  const iff = src.indexOf("if (", from);
  if (iff < 0) return null;
  const open = src.indexOf("(", iff);
  const end = closeOf(src, open);
  return end < 0 ? null : src.slice(open + 1, end);
}

/**
 * I DUE SITI che chiedono la rotta, con quello che ciascuno chiama K e N e col
 * formato del kernel che emette. Le asserzioni sotto girano su entrambi: la
 * regola «cio' che chiedi al piano e' cio' che dici al kernel» e' una sola, e
 * un sito che la rompe si vede senza doverlo aver previsto.
 */
const SITES = [
  {
    name: "gemvB",
    body: (src: string) => helperBody(src, "gemvB"),
    K: "w.k", N: "w.n",
    /** il kernel emesso qui e' `prefillGemmQ4SplitK*Wgsl` ⇒ il formato e' q4_0 */
    fmt: "q4_0",
    kernel: "prefillGemmQ4SplitKIdotWgsl",
  },
  {
    name: "ramo K-quant di loadW",
    body: (src: string) => kquantBranch(src).body,
    K: "k", N: "n",
    /** il kernel emesso qui e' `prefillGemmQ5KSplitK*Wgsl` ⇒ il formato e' q5_K */
    fmt: "q5_K",
    kernel: "prefillGemmQ5KSplitKIdotWgsl",
  },
] as const;

// ---------------------------------------------------------------------------
// (a) L'IMBUTO: la rotta si chiede al piano, e da DUE posti — non da uno, non
//     da tre.
//
// Il censimento di it.17 diceva UNO (`gemvB`) e si fermava li'; ma `ssm_out`
// non passa da `gemvB`, quindi quel "uno" descriveva una copertura che sui
// K-quant era zero. Dopo il cablaggio i posti che decidono sono due, e sono
// nominati: chi ne aggiunge un terzo sta ri-derivando la rotta da qualche altra
// parte, che e' esattamente la forma del difetto di it.7.
// ---------------------------------------------------------------------------
describe("[a] `planPrefillGemm` sta in DUE posti: `gemvB` e il ramo K-quant di `loadW`", () => {
  it("q35gpumodel importa il piano e lo chiama esattamente due volte", () => {
    const src = code(MODEL);
    expect(uncommented(MODEL), "q35gpumodel deve importare il piano").toMatch(
      /import\s*\{[^}]*planPrefillGemm[^}]*\}\s*from\s*["'`]\.\/prefillgemmplan["'`]/);
    const calls = hitsOf(src, "planPrefillGemm", "\\(");
    const where = calls.map((i) => `${MODEL}:${lineOf(src, i)}`);
    expect(calls.length, `planPrefillGemm( a ${where.join(", ") || "NESSUNA riga"}`).toBe(2);
  });

  it("una chiamata e' dentro `gemvB`, l'altra dentro il ramo K-quant di `loadW`", () => {
    const src = code(MODEL);
    const gemvB = helperBody(src, "gemvB");
    expect(gemvB, "gemvB non trovata").not.toBeNull();
    const kq = kquantBranch(src);
    const where = hitsOf(src, "planPrefillGemm", "\\(").map((i) => `${MODEL}:${lineOf(src, i)}`);
    expect(hitsOf(gemvB!, "planPrefillGemm", "\\(").length,
      `dentro gemvB; tutte le chiamate sono a ${where.join(", ")}`).toBe(1);
    expect(hitsOf(kq.body, "planPrefillGemm", "\\(").length,
      `dentro il ramo K-quant (riga ${lineOf(src, kq.at)}); tutte le chiamate sono a ${where.join(", ")}`)
      .toBe(1);
  });

  // -------------------------------------------------------------------------
  // GLI ARGOMENTI, non solo la presenza della chiamata.
  //
  // Il difetto che questa asserzione chiude e' stato TROVATO per mutazione, non
  // immaginato: cambiare `M: M_MAX` in `M: 1` fa tornare `legacy` da OGNI
  // chiamata (la clausola `PREFILL_M1_LEGACY` scatta prima di qualunque domanda
  // al kernel), quindi la via veloce diventa codice morto e il motore emette
  // esattamente i dispatch di ieri — con la suite intera verde, perche' un test
  // che conta le chiamate vede la chiamata. Scambiare `K: k, N: n` in `K: n,
  // N: k` passava allo stesso modo. E' it.14 di nuovo: il cablaggio c'e' come
  // TESTO e non come COMPORTAMENTO.
  // -------------------------------------------------------------------------
  it.each(SITES)("$name: gli argomenti della rotta sono K, N, M e idot del sito", (site) => {
    const body = site.body(code(MODEL));
    expect(body, `${site.name}: corpo non trovato`).not.toBeNull();
    const args = argObjOf(body!, "planPrefillGemm");
    expect(args, `${site.name}: l'argomento del piano dev'essere un letterale oggetto`)
      .not.toBeNull();
    expect([...args!.keys()].sort(), `${site.name}: campi passati al piano`)
      .toEqual(["K", "M", "N", "idot", "kind"]);
    // Le DUE dimensioni del sito, nell'ordine giusto: scambiarle instrada una
    // shape che non e' quella che si moltiplica, e il piano risponde su quella.
    expect(args!.get("K"), `${site.name}: K`).toBe(site.K);
    expect(args!.get("N"), `${site.name}: N`).toBe(site.N);
    // L'M del piano a chunk, non un numero: a M=1 il piano rende legacy per
    // clausola e la via veloce non si accende mai.
    expect(args!.get("M"), `${site.name}: M`).toBe("M_MAX");
    // La capacita' SONDATA, non `true`: su un device senza la language feature
    // un `true` scritto qui emetterebbe il kernel intero che non compila.
    expect(args!.get("idot"), `${site.name}: idot`).toBe("prefillIdot");
    // `kind` e' un IDENTIFICATORE. Sul sorgente biancato una stringa e' fatta
    // di spazi: un `kind: "q4_0"` qui sarebbe vuoto, e l'asserzione lo esclude
    // senza dover distinguere quale letterale fosse.
    expect(args!.get("kind"), `${site.name}: il kind dev'essere una variabile, non un letterale`)
      .toMatch(/^[A-Za-z_$][A-Za-z0-9_$]*$/);
  });

  it.each(SITES)("$name: cio' che si chiede al piano e' cio' che si dice al kernel", (site) => {
    // La regola in una riga: gli `opts` del kernel veloce NON sono un secondo
    // insieme di numeri. Se `K`/`N`/`M`/`kind` del kernel possono divergere da
    // quelli con cui si e' chiesta la rotta, allora il piano risponde su una
    // shape e il kernel ne moltiplica un'altra — e nessuno se ne accorge,
    // perche' entrambe le forme sono valide di per se'.
    const body = site.body(code(MODEL))!;
    const plan = argObjOf(body, "planPrefillGemm")!;
    const kernelArg = callsTo(body, site.kernel)[0];
    expect(kernelArg, `${site.name}: ${site.kernel}( non trovata`).toBeTruthy();
    const open = body.indexOf("(", kernelArg.at);
    const ident = body.slice(open + 1, closeOf(body, open)).trim();
    expect(ident, `${site.name}: il kernel dev'essere emesso con gli opts, non con un oggetto al volo`)
      .toMatch(/^[A-Za-z_$][A-Za-z0-9_$]*$/);
    const opts = constObjOf(body, ident);
    expect(opts, `${site.name}: dichiarazione di \`${ident}\` non trovata`).not.toBeNull();
    for (const f of ["kind", "K", "N", "M"] as const) {
      expect(opts!.get(f), `${site.name}: opts.${f} diverso da quello chiesto al piano`)
        .toBe(plan.get(f));
    }
    // e le fette vengono dalla rotta, non da un numero riscritto qui
    expect(opts!.get("splits"), `${site.name}: splits`).toMatch(/^[A-Za-z_$][A-Za-z0-9_$]*\.splits$/);
  });

  it.each(SITES)("$name: la via veloce si apre su DUE condizioni — rotta e formato", (site) => {
    // Il commento non basta. Che «una rotta non-legacy qui SIGNIFICA q5_K»
    // dipende da `PREFILL_GEMM_KINDS`, che vive in kernels/wgsl.ts: il giorno
    // che il q4_K entra in quell'elenco, questo ramo emetterebbe il kernel del
    // q5_K su superblocchi da 144 B letti col passo da 176 — logit sbagliati,
    // nessuna eccezione, nessun errore di validazione WebGPU, e questo file
    // ancora verde perche' conta chiamate e non formati. La garanzia sta dove
    // si EMETTE.
    //
    // Sorgente con i commenti biancati e le stringhe INTATTE: il letterale del
    // formato va letto, ma un commento che lo nomina non deve valere come
    // condizione (in questo file ce ne sono, ed e' il tranello dichiarato in
    // testa).
    const body = site.body(uncommented(MODEL))!;
    const plan = callsTo(body, "planPrefillGemm")[0];
    const cond = condAfter(body, plan.at);
    expect(cond, `${site.name}: nessun \`if\` dopo la chiamata al piano`).not.toBeNull();
    const kind = argObjOf(site.body(code(MODEL))!, "planPrefillGemm")!.get("kind")!;
    expect(cond!, `${site.name}: la condizione deve escludere la rotta legacy`)
      .toMatch(/!==\s*["']legacy["']/);
    expect(cond!, `${site.name}: la condizione deve verificare che il formato sia ${site.fmt}`)
      .toMatch(new RegExp(`${kind}\\s*===\\s*["']${site.fmt}["']`));
  });

  it("il kind del ramo K-quant nomina TUTTI E TRE i formati, non solo quello veloce", () => {
    // La variabile che si passa al piano dev'essere il tipo REALE del tensore:
    // se coprisse il solo q5_K, il q4_K e il q6_K arriverebbero al piano sotto
    // falso nome e il piano risponderebbe sulla geometria sbagliata (e' la
    // bugia che il piano stesso si e' appena tolto, `o.kind as "q4_0"`).
    const kqCode = kquantBranch(code(MODEL)).body;
    const ident = argObjOf(kqCode, "planPrefillGemm")!.get("kind")!;
    // sorgente coi commenti biancati: il ternario e' codice, i commenti che
    // nominano gli stessi letterali no
    const kqUnc = kquantBranch(uncommented(MODEL)).body;
    const decl = new RegExp(`const\\s+${ident}\\s*(?::[^=;]*)?=([^;]*);`).exec(kqUnc);
    expect(decl, `dichiarazione di \`${ident}\` non trovata nel ramo K-quant`).not.toBeNull();
    for (const lit of ["q4_K", "q5_K", "q6_K"]) {
      expect(decl![1], `\`${ident}\` deve nominare ${lit}`).toContain(lit);
    }
  });
});

// ---------------------------------------------------------------------------
// (b) LE EMISSIONI del ramo K-quant, con le liste di binding CONGELATE.
//
// L'ordine dei binding non e' stile: e' il contratto `@group(0) @binding(i)`
// dei kernel. Uno scambio fra `prefillXq` e `prefillXsc` non fa fallire nessuna
// validazione WebGPU (sono entrambi storage buffer) e produce numeri sbagliati
// in silenzio — la stessa classe di guasto che ha fatto nascere [g] in
// engine-prefillgemm.test.ts, ma sul lato del CHIAMANTE.
// ---------------------------------------------------------------------------
describe("[b] il ramo K-quant emette la via veloce q5_K, e tiene il gemv come legacy", () => {
  const FROZEN: [string, string[]][] = [
    ["prefillQuantXQ8Wgsl", ["src", "prefillXq", "prefillXsc"]],
    ["prefillGemmQ5KSplitKIdotWgsl", ["w.blocks", "prefillXq", "prefillPart", "prefillXsc"]],
    ["prefillGemmQ5KSplitKWgsl", ["w.blocks", "src", "prefillPart"]],
    ["prefillSplitKCombineWgsl", ["prefillPart", "dst"]],
  ];

  it.each(FROZEN)("%s: chiamata UNA volta, con i binding nell'ordine congelato", (name, want) => {
    const kq = kquantBranch(code(MODEL));
    const calls = callsTo(kq.body, name);
    expect(calls.length, `${name}( nel ramo K-quant`).toBe(1);
    expect(calls[0].bindings, `${name}: la lista di binding dev'essere INLINE dopo la chiamata`)
      .not.toBeNull();
    expect(calls[0].bindings).toEqual(want);
  });

  it("le griglie NON si calcolano a mano: `prefillGemmGrid`/`prefillCombineGrid`/`prefillQuantXGrid`", () => {
    const src = code(MODEL);
    const kq = kquantBranch(src);
    // `prefillGemmGrid` DUE volte: una per via (idot e f32), che sono due
    // dispatch alternativi con la stessa griglia ma opzioni distinte. Le altre
    // due una volta sola: quantizzazione e combine non si sdoppiano.
    for (const [g, want] of [["prefillGemmGrid", 2], ["prefillCombineGrid", 1],
      ["prefillQuantXGrid", 1]] as const) {
      expect(hitsOf(kq.body, g, "\\(").length, `${g}( nel ramo K-quant`).toBe(want);
      expect(uncommented(MODEL), `${g} dev'essere importato`).toMatch(
        new RegExp(`import\\s*\\{[^}]*${g}[^}]*\\}\\s*from\\s*["'\`]\\./kernels/wgsl["'\`]`, "s"));
    }
    // e nessuna griglia ricostruita a mano dentro il ramo
    expect(kq.body, "griglia ri-derivata a mano nel ramo K-quant").not.toMatch(/Math\.ceil\s*\(/);
  });

  it.each(SITES)("$name: NESSUNA griglia della via veloce e' ricalcolata a mano", (site) => {
    // Il cablaggio nuovo prendeva le griglie dagli helper mentre `gemvB`, nello
    // STESSO file, riscriveva a mano la stessa aritmetica trenta righe piu'
    // sotto: `[Math.ceil((M_MAX * (w.k / 32)) / 64), 1, 1]` e `[Math.ceil((M_MAX
    // * w.n) / 64), 1, 1]`. Davano gli stessi numeri — ed e' esattamente cio'
    // che rende il difetto invisibile: sono una SECONDA copia della geometria
    // di un kernel che sta in un altro file, e la copia non porta con se' il
    // rifiuto di `prefillQuantXGrid` (K non multiplo di 64). La clausola (d) del
    // done-when vieta il `% 64 === 0` scritto a mano; questa vieta la griglia,
    // che e' la stessa soglia in forma di divisione.
    const body = site.body(code(MODEL))!;
    expect(body, `${site.name}: griglia ri-derivata a mano`).not.toMatch(/Math\.ceil\s*\(/);
    for (const g of ["prefillQuantXGrid", "prefillCombineGrid"]) {
      expect(hitsOf(body, g, "\\(").length, `${site.name}: ${g}(`).toBe(1);
    }
  });

  it("il ternario `gemvQ4KWgsl/gemvQ5KWgsl/gemvQ6KWgsl({batch:true})` resta come ramo legacy", () => {
    const kq = kquantBranch(code(MODEL));
    for (const n of ["gemvQ4KWgsl", "gemvQ5KWgsl", "gemvQ6KWgsl"]) {
      const batched = callsTo(kq.body, n).filter((c) => c.batch);
      expect(batched.length, `${n}({… batch: true}) — il fallback legacy non si butta`).toBe(1);
    }
  });

  it("il path di DECODE non e' toccato: `push` col gemv K-quant scalare resta", () => {
    const kq = kquantBranch(code(MODEL));
    for (const n of ["gemvQ4KWgsl", "gemvQ5KWgsl", "gemvQ6KWgsl"]) {
      const scalar = callsTo(kq.body, n).filter((c) => !c.batch);
      expect(scalar.length, `${n}({…}) scalare (decode)`).toBe(1);
    }
    expect(hitsOf(kq.body, "gemvGrid", "\\(").length,
      "la griglia del gemv di decode resta `gemvGrid`").toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// (c) LA WORKLIST delle emissioni batch FUORI dall'imbuto del piano.
// ---------------------------------------------------------------------------
describe("[c] worklist: cosa emette ancora `batch: true` senza chiedere la rotta", () => {
  it("sono due `gemvF32Wgsl`, e sono quelle due", () => {
    const src = code(MODEL);
    // ESCLUSIONI, una per una:
    //  attnDecodeWgsl  non e' un GEMM, ha la sua riga ed e' gia' in streaming;
    //  gemvQuantWgsl   e' il fallback DENTRO `gemvB`, cioe' la via legacy
    //                  scelta DAL PIANO — coperto, non eccezione;
    //  gemvQ4K/Q5K/Q6K DAL CABLAGGIO IN POI sono il fallback dentro un ramo che
    //                  la rotta la chiede: stessa posizione di `gemvQuantWgsl`.
    //                  Prima di questo task erano eccezioni vere ed erano nella
    //                  worklist; questa e' la riga che cambia.
    const EXCLUDED = new Set(["attnDecodeWgsl", "gemvQuantWgsl",
      "gemvQ4KWgsl", "gemvQ5KWgsl", "gemvQ6KWgsl"]);
    const batchSites = [...src.matchAll(/(\w+Wgsl)\(\{[^}]*batch:\s*true/g)]
      .map((m) => ({ name: m[1], line: lineOf(src, m.index!) }));
    const worklist = batchSites.filter((s) => !EXCLUDED.has(s.name));
    expect(worklist.map((s) => s.name).sort(),
      `worklist a ${worklist.map((s) => `${MODEL}:${s.line}`).join(", ")}`)
      .toEqual(["gemvF32Wgsl", "gemvF32Wgsl"]);
  });

  it("i tre gemv K-quant batch stanno DENTRO il ramo che chiede la rotta", () => {
    // L'esclusione qui sopra non e' una dispensa: vale solo perche' quelle
    // emissioni sono nel ramo che ha appena chiesto `planPrefillGemm`. Se
    // qualcuno le sposta fuori, l'esclusione diventa una copertura finta —
    // questo test e' cio' che lo impedisce.
    const src = code(MODEL);
    const kq = kquantBranch(src);
    for (const n of ["gemvQ4KWgsl", "gemvQ5KWgsl", "gemvQ6KWgsl"]) {
      const all = callsTo(src, n).filter((c) => c.batch);
      const inside = callsTo(kq.body, n).filter((c) => c.batch);
      expect(all.length, `${n} batch nel file`).toBe(inside.length);
    }
  });
});

// ---------------------------------------------------------------------------
// (d) Nessuna condizione della via veloce ri-derivata a mano.
// ---------------------------------------------------------------------------
describe("[d] nessun call-site ri-deriva la condizione della via veloce", () => {
  it("`% 64 === 0` non compare in q35gpumodel.ts", () => {
    // La firma del difetto di it.14: `kind === "q4_0" && k % 64 === 0` scritto
    // a mano invece di chiesto al piano. Sul q5_K sarebbe pure la soglia
    // SBAGLIATA (l'unita' e' il superblocco da 256), quindi una seconda soglia
    // che diverge in silenzio dal kernel non e' un'ipotesi: e' il caso normale.
    expect(code(MODEL), "condizione della via veloce ri-derivata a mano fuori dal piano")
      .not.toMatch(/%\s*64\s*===\s*0/);
  });
});

// ---------------------------------------------------------------------------
// (e) CLAUSOLA (d) — I BUFFER, ri-verificati perche' il consumatore e' cambiato.
//
// `prefillPart`, `prefillXq` e `prefillXsc` sono UNO SOLO per tutto il modello e
// sono dimensionati su `prefillMaxN`. Finche' l'unico consumatore era `gemvB`,
// il massimo si leggeva sulle shape che passavano di li'; ora ci passa anche il
// ramo K-quant, cioe' `ssm_out` (q5_K, K=4096, N=2560). La domanda non e'
// retorica: un buffer corto darebbe scritture fuori range che la validazione
// WebGPU NON vede, perche' il binding e' l'intero buffer.
//
// ESITO MISURATO QUI SOTTO: il massimo NON cambia. `ssm_out` sta sotto ogni
// tetto — N=2560 contro 9216, K=4096 contro 9216 — e i tre massimi restano
// realizzati dalle stesse shape q4_0 di prima. La clausola resta dichiarata,
// non abolita: e' il test che lo ridice a ogni run, e stampa QUALE shape
// realizza ciascun massimo.
// ---------------------------------------------------------------------------
describe("[e] CLAUSOLA (d): i buffer condivisi reggono anche il consumatore nuovo", () => {
  const M = 16;

  /**
   * Dimensioni del 4B (header GGUF, gia' pinnate in engine-q35-shape.test.ts):
   * dModel 2560, nHead 16, headDim 256, nKvHead 4, linKHead 16, linVHead 32,
   * linHeadDim 128, dFfn 9216. Nessun expert: dE = nE = 0.
   */
  const S4B = {
    dModel: 2560, nHead: 16, headDim: 256, nKvHead: 4,
    linKHead: 16, linVHead: 32, linHeadDim: 128, dFfn: 9216,
    dFfnExpert: 0, nExpert: 0,
  } as const;

  /** La STESSA espressione di q35gpumodel.ts (riga ~635), non un numero copiato. */
  const prefillMaxN = Math.max(
    S4B.dModel,                                                        // d
    (2 * S4B.linKHead + S4B.linVHead) * S4B.linHeadDim,                // qkvDim
    S4B.linVHead * S4B.linHeadDim,                                     // inner
    2 * (S4B.nHead * S4B.headDim),                                     // 2*qDim
    S4B.nKvHead * S4B.headDim,                                         // kvDim
    S4B.linVHead, S4B.dFfn, S4B.dFfnExpert, S4B.nExpert,
  );

  /**
   * Le shape del 4B che il piano INSTRADA a M=16 (le altre — q8_0 2560x32 e
   * q4_1 9216x2560 — restano legacy e non toccano questi buffer).
   */
  const ROUTED: { site: string; kind: PrefillQuantKind; K: number; N: number }[] = [
    { site: "attn_q / attn_qkv", kind: "q4_0", K: 2560, N: 8192 },
    { site: "attn_k / attn_v", kind: "q4_0", K: 2560, N: 1024 },
    { site: "attn_output", kind: "q4_0", K: 4096, N: 2560 },
    { site: "attn_gate", kind: "q4_0", K: 2560, N: 4096 },
    { site: "ffn_gate / ffn_up", kind: "q4_0", K: 2560, N: 9216 },
    { site: "ffn_down", kind: "q4_0", K: 9216, N: 2560 },
    { site: "ssm_out", kind: "q5_K", K: 4096, N: 2560 },
  ];

  it("prefillMaxN del 4B e' 9216, e viene dalla FFN", () => {
    expect(prefillMaxN).toBe(9216);
    expect(prefillMaxN).toBe(S4B.dFfn);
  });

  it("tutte e sette le shape sono davvero instradate (se no il resto non prova niente)", () => {
    for (const s of ROUTED) {
      const r = planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M, idot: true });
      expect(r.via, `${s.site} (${s.kind} ${s.K}x${s.N}): ${r.reason}`).not.toBe("legacy");
    }
    // e il q5_K passa anche senza la language feature, sulla via f32
    const f = planPrefillGemm({ kind: "q5_K", K: 4096, N: 2560, M, idot: false });
    expect(f.via, f.reason).toBe("f32");
  });

  it("`prefillPart` regge: partialFloats <= PREFILL_SPLITS_MEASURED*M*prefillMaxN", () => {
    const cap = PREFILL_SPLITS_MEASURED * M * prefillMaxN;
    expect(cap).toBe(4 * 16 * 9216);
    let top = { site: "", v: -1 };
    for (const s of ROUTED) {
      for (const idot of [true, false]) {
        const r = planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M, idot });
        expect(r.partialFloats, `${s.site} idot=${idot}: ${r.partialFloats} > ${cap}`)
          .toBeLessThanOrEqual(cap);
        if (r.partialFloats > top.v) top = { site: `${s.site} (${s.kind} ${s.K}x${s.N})`, v: r.partialFloats };
      }
    }
    // DICHIARAZIONE: chi realizza il massimo. Non e' `ssm_out`.
    expect(top.site).toBe("ffn_gate / ffn_up (q4_0 2560x9216)");
    expect(top.v).toBe(cap);          // il tetto e' RAGGIUNTO, non abbondante
  });

  it("`prefillXq` regge: xqU32 <= M*prefillMaxN/4", () => {
    const cap = M * prefillMaxN / 4;
    expect(cap).toBe(36864);
    let top = { site: "", v: -1 };
    for (const s of ROUTED) {
      const r = planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M, idot: true });
      expect(r.xqU32, `${s.site}: ${r.xqU32} > ${cap}`).toBeLessThanOrEqual(cap);
      if (r.xqU32 > top.v) top = { site: `${s.site} (${s.kind} ${s.K}x${s.N})`, v: r.xqU32 };
    }
    expect(top.site).toBe("ffn_down (q4_0 9216x2560)");
    expect(top.v).toBe(cap);
    // `ssm_out` — il consumatore NUOVO — sta a meno della meta' del tetto.
    const ssm = planPrefillGemm({ kind: "q5_K", K: 4096, N: 2560, M, idot: true });
    expect(ssm.xqU32).toBe(16 * (4096 / 32) * 8);
    expect(ssm.xqU32 * 2).toBeLessThan(cap);
  });

  it("`prefillXsc` regge: xscF32 <= M*prefillMaxN/32", () => {
    const cap = M * prefillMaxN / 32;
    expect(cap).toBe(4608);
    let top = { site: "", v: -1 };
    for (const s of ROUTED) {
      const r = planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M, idot: true });
      expect(r.xscF32, `${s.site}: ${r.xscF32} > ${cap}`).toBeLessThanOrEqual(cap);
      if (r.xscF32 > top.v) top = { site: `${s.site} (${s.kind} ${s.K}x${s.N})`, v: r.xscF32 };
    }
    expect(top.site).toBe("ffn_down (q4_0 9216x2560)");
    expect(top.v).toBe(cap);
  });

  it("il commento dei buffer DICHIARA che il consumatore non e' piu' solo `gemvB`", () => {
    // La clausola (d) chiede che il ri-esame sia scritto dove i buffer si
    // allocano, non solo qui: un commento che dice ancora "le `n` che `gemvB`
    // puo' ricevere" e' una mappa vecchia del territorio.
    const src = raw(MODEL);
    const at = src.indexOf("const prefillMaxN");
    expect(at, "prefillMaxN non trovato").toBeGreaterThan(0);
    const doc = src.slice(Math.max(0, at - 2000), at);
    expect(doc, "il commento di prefillPart deve nominare il ramo K-quant / ssm_out")
      .toMatch(/ssm_out|K-quant|kquant/);
  });
});

// ---------------------------------------------------------------------------
// [6f] IL CENSIMENTO PER CALL-SITE — ricostituito e aggiornato.
//
// Viveva in tests/engine-prefillgemmplan.test.ts ed e' stato tolto da li' col
// task che ha aperto il q5_K al piano (il suo contenuto parlava di
// q35gpumodel.ts, non del piano). Torna qui, dove sta il file che censisce.
//
// Censimento del 2026-08-14, dopo il cablaggio del q5_K:
//
//   COPERTI     i GEMM quantizzati passano da DUE imbuti che chiedono la rotta
//               a `planPrefillGemm`: `gemvB` (q4_0/q4_1/q8_0, dalle due
//               fabbriche di peso e dai tre dispatch del blocco MoE) e il ramo
//               K-quant di `loadW` (q4_K/q5_K/q6_K — sul 4B: `ssm_out`).
//   ECCEZIONI   2 siti, stessa ragione strutturale: pesi F32, che non sono un
//               GEMM quantizzato e non hanno un kernel di prefill.
//                 gemvF32Wgsl  ramo F32 di `loadW`
//                 gemvF32Wgsl  scalare dello shared expert
//   INSTRADAMENTO  coperto != veloce. `q8_0` e `q4_1` chiedono la rotta e il
//               piano risponde "legacy": la via veloce non ha la loro misura.
//               Sono due cose diverse, ed e' la distinzione che il censimento
//               sbagliato di it.17 aveva perso.
//
// NON e' un'eccezione l'attenzione (`attnDecodeWgsl`): non e' un GEMM, ha la
// sua riga ed e' gia' in streaming.
// ---------------------------------------------------------------------------
describe("[6f] CLAUSOLA (d): la copertura per CALL-SITE, con worklist ed eccezioni motivate", () => {
  const M = 16;

  it("i kind che restano legacy sul 4B sono DUE, e q8_0 e' quello che si dimentica", () => {
    // ATTENZIONE ALLA DISTINZIONE, che questo test esiste per tenere ferma:
    // `q8_0` NON e' un'eccezione di COPERTURA — passa dall'imbuto e chiede la
    // rotta, come si deve. E' il PIANO a rispondergli "legacy". Coperto dalla
    // convenzione, legacy per instradamento.
    const ALL: { kind: PrefillQuantKind; K: number; N: number }[] = [
      { kind: "q4_0", K: 2560, N: 8192 }, { kind: "q4_0", K: 2560, N: 1024 },
      { kind: "q4_0", K: 4096, N: 2560 }, { kind: "q4_0", K: 2560, N: 4096 },
      { kind: "q4_0", K: 2560, N: 9216 }, { kind: "q4_0", K: 9216, N: 2560 },
      { kind: "q8_0", K: 2560, N: 32 }, { kind: "q4_1", K: 9216, N: 2560 },
      { kind: "q5_K", K: 4096, N: 2560 },
    ];
    const legacy = ALL.map((s) => ({ s, r: planPrefillGemm({ ...s, M, idot: true }) }))
      .filter((x) => x.r.via === "legacy");
    expect([...new Set(legacy.map((x) => x.s.kind))].sort(),
      `i kind che restano legacy: ${legacy.map((x) => `${x.s.kind} ${x.s.K}x${x.s.N}`).join(", ")}`)
      .toEqual(["q4_1", "q8_0"]);
    for (const x of legacy) expect(x.r.reason.length, `${x.s.kind}`).toBeGreaterThanOrEqual(40);
  });

  it("il q5_K NON e' piu' fra i legacy: e' la riga che questo task chiude", () => {
    const r = planPrefillGemm({ kind: "q5_K", K: 4096, N: 2560, M, idot: true });
    expect(r.via, r.reason).toBe("idot");
    expect(r.splits, "4096/256 = 16 superblocchi, divisibili in 4 fette").toBe(4);
  });
});
