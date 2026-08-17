// Lo scanner di sorgente dei test STATICI: legge un file del motore come TESTO e
// risponde a domande di forma («questa chiamata esiste?», «con che binding?»,
// «dentro il corpo di main()?»).
//
// PERCHE' ESISTE UN MODULO INVECE DI UNA COPIA PER FILE. Serve ai test che
// SCANSIONANO `src/engine/ktest/ktest.worker.ts`: quel file non si puo'
// importare in vitest node (tira dentro i tipi WebGPU e un modulo worker), e
// l'unico modo di provare un cablaggio senza GPU e' leggerlo. Le prime stesure
// hanno ricopiato queste ~190 righe in ogni file di cablaggio, e alla quarta
// copia il ruling di riuso di questo progetto e' esplicito: la forma buona
// diventa la radice e i siti vecchi migrano. La copia non era gratis — era gia'
// DIVERGITA': il gemello piu' vecchio (`engine-ktest-q5k-wiring.test.ts`) aveva
// un `helperBody` che rendeva il solo testo, e per la domanda «la registrazione
// e' DENTRO main()?» (che e' di POSIZIONE, non di forma della riga) serviva
// l'`helperRange` scritto solo nelle copie successive. Qui la forma buona e' una
// sola, e chi la migliora la migliora per tutti.
//
// Precedente: `tests/helpers/glm-gguf-fixture.ts`, estratta da un test per lo
// stesso motivo. Un file sotto `tests/helpers/` non e' un test — non ha `it`,
// non asserisce: le asserzioni restano nei file che possiedono il contratto,
// perche' il MESSAGGIO di fallimento («banco q4_1: ...») e' parte del contratto
// e non si puo' generalizzare.
import { readFileSync } from "node:fs";
import { join } from "node:path";

// la radice del repo: `tests/helpers/` sta due livelli sotto. Ancorata al file,
// non alla directory da cui si lancia vitest.
const ROOT = join(__dirname, "..", "..");

/**
 * Bianca cio' che NON e' codice, conservando le POSIZIONI carattere per
 * carattere (uno spazio al posto di ogni carattere tolto, gli a-capo intatti):
 * cosi' gli indici del testo biancato e quelli del sorgente vero coincidono, e
 * `lineOf` continua a dire la riga giusta.
 *
 * Senza questo passo un commento che nomina `prefillSplitKCombineWgsl(` sposta i
 * conteggi, e una lista `[blocks, xq, part, xsc]` CITATA in un commento diventa
 * la lista che il test misura al posto di quella vera.
 *
 * `keepStrings`: le stringhe si tengono quando cio' che si cerca E' una stringa
 * (gli import, i nomi di caso, un `kind: "q5_K"`); si biancano quando si contano
 * identificatori, perche' un template WGSL puo' contenerne a decine.
 */
export function blankNonCode(src: string, keepStrings: boolean): string {
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
  const scanCode = (start: number, inTemplate: boolean): number => {
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
        const close = scanCode(i + 2, true);
        if (!keepStrings) blank(close);
        i = close + 1; continue;
      }
      if (!keepStrings) blank(i);
      i++;
    }
    return i;
  };
  scanCode(0, false);
  return out.join("");
}

/** sorgente grezzo di un file, per percorso relativo alla radice del repo */
export const raw = (rel: string): string => readFileSync(join(ROOT, rel), "utf8");

const memo = new Map<string, string>();
/** sorgente con commenti/stringhe/template biancati: quello su cui si CONTA */
export const code = (rel: string): string => {
  const key = `code:${rel}`;
  if (!memo.has(key)) memo.set(key, blankNonCode(raw(rel), false));
  return memo.get(key)!;
};
/** sorgente coi soli commenti biancati: serve agli import e ai literal, che sono stringhe */
export const uncommented = (rel: string): string => {
  const key = `unc:${rel}`;
  if (!memo.has(key)) memo.set(key, blankNonCode(raw(rel), true));
  return memo.get(key)!;
};

/**
 * Il sorgente biancato SENZA le righe di `import`: e' li' che si contano gli
 * USI. Gli span degli import si cercano sul sorgente con le stringhe INTATTE (il
 * modulo e' una stringa, e biancata la clausola `from "..."` non si riconosce
 * piu'), ma si cancellano da quello biancato: il biancamento conserva le
 * POSIZIONI carattere per carattere, quindi i due combaciano.
 */
export const codeNoImports = (rel: string): string => {
  const key = `noimp:${rel}`;
  if (!memo.has(key)) {
    const out = code(rel).split("");
    for (const m of uncommented(rel).matchAll(/import\s[\s\S]*?from\s*["'][^"']+["']/g)) {
      for (let i = m.index!; i < m.index! + m[0].length; i++) if (out[i] !== "\n") out[i] = " ";
    }
    memo.set(key, out.join(""));
  }
  return memo.get(key)!;
};

/** riga 1-based dell'indice `idx` */
export const lineOf = (src: string, idx: number): number => src.slice(0, idx).split("\n").length;

/** occorrenze di un IDENTIFICATORE (non di una sottostringa qualsiasi) */
export const hitsOf = (src: string, ident: string, suffix = ""): number[] =>
  [...src.matchAll(new RegExp(`(?<![A-Za-z0-9_$])${ident}\\s*${suffix}`, "g"))].map((m) => m.index!);

/** quante volte una stringa compare come LETTERALE (commenti gia' biancati) */
export const literalHits = (src: string, s: string): number[] =>
  [...src.matchAll(new RegExp(`["'\`]${s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}["'\`]`, "g"))].map((m) => m.index!);

/** i letterali NUMERICI di un frammento (`Float32Array` ha cifre ma non e' un numero) */
export const numLiterals = (s: string): string[] => s.match(/(?<![A-Za-z0-9_$.])\d[\d.eE+-]*/g) ?? [];

const CLOSE: Record<string, string> = { "(": ")", "[": "]", "{": "}" };

/** indice del delimitatore che chiude il gruppo aperto a `open`, o -1 */
export function closeOf(src: string, open: number): number {
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
export function topLevelItems(src: string, open: number): string[] {
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

export interface Call { at: number; line: number; args: string[]; bindings: string[] | null }

/**
 * Chiamate a `name(`, ognuna coi SUOI argomenti e con la SUA lista di binding.
 * La lista dev'essere INLINE subito dopo la chiamata (solo spazi e virgole in
 * mezzo): se e' passata per variabile, `bindings` resta null e il chiamante lo
 * DICE, invece di agganciare il primo `[` che trova piu' avanti — che sarebbe un
 * array scorrelato.
 */
export function callsTo(src: string, name: string): Call[] {
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
export const importsFrom = (src: string, suffix: string): string =>
  [...src.matchAll(/import\s+(?:type\s+)?\{([^}]*)\}\s*from\s*["']([^"']+)["']/g)]
    .filter((m) => m[2].endsWith(suffix)).map((m) => m[1].replace(/\s+/g, " ").trim()).join(" ; ");

/** `ident` compare nella lista come identificatore intero */
export const has = (list: string, ident: string): boolean =>
  new RegExp(`(?<![A-Za-z0-9_$])${ident}(?![A-Za-z0-9_$])`).test(list);

/**
 * Estremi del corpo di un helper, sia nella forma `function f(` sia in quella
 * `const f = (`: il primo gruppo graffo a profondita' di parentesi 0 dopo il
 * nome (cosi' parametri destrutturati e tipi inline non lo confondono).
 *
 * Rende gli INDICI, non solo il testo, perche' «dentro main()» e' una domanda di
 * POSIZIONE: un `results.push(...await FN(g))` identico ma scritto in una
 * funzione che nessuno chiama supererebbe qualunque controllo testuale, e il
 * driver non eseguirebbe MAI il banco.
 */
export function helperRange(src: string, name: string): { from: number; to: number } | null {
  const def = new RegExp(`(?:function\\s+${name}\\s*\\(|const\\s+${name}\\s*=)`).exec(src);
  if (!def) return null;
  let depth = 0;
  for (let i = def.index; i < src.length; i++) {
    const c = src[i];
    if (c === "(" || c === "[") depth++;
    else if (c === ")" || c === "]") depth--;
    else if (c === "{" && depth === 0) {
      const end = closeOf(src, i);
      return end < 0 ? null : { from: i, to: end };
    }
  }
  return null;
}

/** corpo `{...}` dell'helper, o null se non e' definito */
export function helperBody(src: string, name: string): string | null {
  const r = helperRange(src, name);
  return r === null ? null : src.slice(r.from, r.to + 1);
}
