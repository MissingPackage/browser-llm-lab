// COMPOSIZIONE del segmento `gemm:ffn-down-q41` (goal engine-ttft, riga 4).
//
// PERCHE' QUESTO FILE ESISTE, in una riga: la riga 5 vorra' LEGGERE il
// checkpoint `gemm:ffn-down-q41` come «il costo dei quattro tensori Q4_1», e
// quel numero vale solo se il segmento contiene esattamente i dispatch che si
// crede contenga. Un `pbCat` e' una variabile che TIMBRA tutto cio' che passa
// da `pushB`/`gemvB`/`pushBRows1` finche' qualcuno non la riassegna: non ha
// confini dichiarati, non ha un tipo, e nessun test di aritmetica si accorge se
// un dispatch in piu' entra nel segmento — i logit restano identici, cambia
// solo la BOLLETTA di chi legge il checkpoint. E' esattamente il difetto che la
// riga 5 di engine-ttft ha misurato al contrario (la ricorrenza DeltaNet
// nascosta dentro `deltanet:gates`): li' il termine grosso stava in una
// categoria che non era la sua, e la proiezione ne usciva sbagliata pur essendo
// l'aritmetica giusta. Qui si inchioda il confine PRIMA che la misura lo usi.
//
// IL CONFINE NON E' LA GRAFFA. `pbCat` e' un `let` di funzione: il timbro NON
// muore alla `}` del ramo `if (PB)` in cui viene scritto. Resta vivo sulla coda
// del ramo `!isMoe`, sulla coda del corpo del ciclo sui layer, oltre il ciclo, e
// — girando — sulla TESTA dell'iterazione successiva, fino al primo `pbCat =`
// che incontra. Un test che guardasse solo dentro la graffa direbbe verde a un
// sorgente che nel segmento ci ha infilato quattro dispatch per token. Percio'
// qui la REGIONE VIVA e' costruita per intero (cinque tratti, sotto) e si
// pretende che oltre il primo sia VUOTA.
//
// POSTURA (tests/engine-prefillwiring-q5k.test.ts): il sorgente si LEGGE come
// testo — importarlo tirerebbe dentro i tipi WebGPU — e il testo si scansiona
// BIANCATO. Non e' cerimonia: alla riga 1113 di q35gpumodel.ts c'e' un commento
// che NOMINA `gemm:ffn-down`, e senza biancamento «la categoria compare in un
// solo sito» diventa falso su un file corretto (e, specularmente, un commento
// che nomini `gemm:ffn-down-q41` renderebbe verde un sorgente che la categoria
// non la assegna piu'). L'helper `blankNonCode` si COPIA da
// engine-prefillwiring-q5k: quel file non e' un modulo di libreria e non e' fra
// i file che questo task possiede.
//
// NIENTE ASSERZIONI A OCCHIO: la lista dei dispatch del segmento non e'
// dedotta ne' contata a mano, e' ESTRATTA dal sorgente biancato fra
// l'assegnazione e i confini della regione viva, e confrontata con una lista
// scritta per esteso qui sotto. Se un dispatch entra o esce dal segmento, il
// confronto cade.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const MODEL = "src/engine/q35gpumodel.ts";
const DUMP = "results/engine/q35-header-dump-2026-08-10.json";
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
 *
 * COPIATO da tests/engine-prefillwiring-q5k.test.ts (vedi intestazione): la
 * lunghezza dell'output e' pari a quella dell'input, quindi gli offset di una
 * versione biancata valgono anche nell'altra e nel sorgente grezzo.
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
/**
 * Sorgente con commenti/stringhe/template biancati. E' la vista su cui si fa
 * TUTTO il lavoro strutturale — parentesi, graffe, chiamate — perche' e' l'unica
 * in cui una graffa e' una graffa: i template WGSL ne portano di proprie.
 */
const code = (rel: string): string => {
  const key = `code:${rel}`;
  if (!memo.has(key)) memo.set(key, blankNonCode(raw(rel), false));
  return memo.get(key)!;
};
/**
 * Sorgente coi soli commenti biancati: serve alle CATEGORIE, che sono stringhe.
 * E' la vista in cui `"gemm:ffn-down-q41"` esiste ancora ma il commento che lo
 * nominasse no. Si usa per LEGGERE valori, mai per contare delimitatori.
 */
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

/** indice della `{` del blocco piu' interno che contiene `at`, o -1 */
function blockOpenOf(src: string, at: number): number {
  const stack: number[] = [];
  for (let i = 0; i < at; i++) {
    const c = src[i];
    if (c === "{") stack.push(i);
    else if (c === "}") stack.pop();
  }
  return stack.length ? stack[stack.length - 1] : -1;
}

const norm = (s: string): string => s.replace(/\s+/g, " ").trim();

// ---------------------------------------------------------------- estrazione

interface Assign { at: number; end: number; line: number; rhs: string }

/**
 * Ogni assegnazione a `pbCat`, col testo del valore assegnato (fino al `;` a
 * profondita' 0). Le POSIZIONI si trovano sulla vista biancata (`c`), il VALORE
 * si legge sulla vista con le stringhe (`s`): la stringa E' il valore che si
 * vuole leggere, ma non deve mai partecipare al conteggio delle parentesi.
 */
function pbCatAssigns(c: string, s: string): Assign[] {
  return [...c.matchAll(/(?<![A-Za-z0-9_$])pbCat\s*=(?!=)/g)].map((m) => {
    const at = m.index!;
    const from = at + m[0].length;
    let i = from, depth = 0;
    for (; i < c.length; i++) {
      const ch = c[i];
      if (ch === "(" || ch === "[" || ch === "{") depth++;
      else if (ch === ")" || ch === "]" || ch === "}") depth--;
      else if (ch === ";" && depth === 0) break;
    }
    return { at, end: i, line: lineOf(c, at), rhs: norm(s.slice(from, i)) };
  });
}

interface Block { at: number; line: number; head: string; open: number; end: number }

/** I `<kw> (…) { … }` del file: testa normalizzata ed estensione del corpo. */
function headedBlocks(src: string, kw: string): Block[] {
  const out: Block[] = [];
  for (const at of hitsOf(src, kw, "\\(")) {
    const p = src.indexOf("(", at);
    const pe = closeOf(src, p);
    if (pe < 0) continue;
    let j = pe + 1;
    while (j < src.length && /\s/.test(src[j])) j++;
    if (src[j] !== "{") continue; // senza graffe non delimita un ramo
    const end = closeOf(src, j);
    if (end < 0) continue;
    out.push({ at, line: lineOf(src, at), head: norm(src.slice(p + 1, pe)), open: j, end });
  }
  return out;
}

/** I rami `if` che contengono `at`, dal piu' interno al piu' esterno. */
const enclosingIfs = (src: string, at: number): Block[] =>
  headedBlocks(src, "if").filter((b) => b.open < at && at < b.end).sort((x, y) => y.open - x.open);

/** Il blocco `else { … }` che segue la `}` a `blockEnd`, se c'e'. */
function elseBlockOf(src: string, blockEnd: number): { open: number; end: number } | null {
  let j = blockEnd + 1;
  while (j < src.length && /\s/.test(src[j])) j++;
  if (src.slice(j, j + 4) !== "else") return null;
  j += 4;
  while (j < src.length && /\s/.test(src[j])) j++;
  if (src[j] !== "{") return null;
  const end = closeOf(src, j);
  return end < 0 ? null : { open: j, end };
}

/**
 * I siti che ACCODANO un dispatch nel piano batch, in ordine di testo.
 *
 * L'insieme e' `gemvB` / `pushB` / `pushBRows1`, con o senza ricevitore
 * (`wg.pushB(…)` e' un dispatch quanto `pushB(…)`): sono le tre funzioni che
 * fanno `PB.steps.push({… cat: pbCat})`. `pushBRows1` sta nell'elenco anche se
 * oggi nel segmento non compare — e' proprio il caso «un dispatch entra nel
 * segmento» che questo test deve poter vedere, e limitarsi a `pushB(` non lo
 * vedrebbe (`pushBRows1(` non e' `pushB(`).
 *
 * Gira sulla vista `code`, dove le stringhe sono biancate: una categoria che
 * contenesse `pushB(` non puo' fingersi una chiamata.
 */
const EMITTERS = /(?<![A-Za-z0-9_$])((?:[A-Za-z_$][A-Za-z0-9_$]*\.)?(?:gemvB|pushB|pushBRows1))\s*\(/g;

interface Dispatch { at: number; line: number; text: string }

function dispatchesIn(src: string, from: number, to: number): Dispatch[] {
  const out: Dispatch[] = [];
  for (const m of src.slice(from, to).matchAll(EMITTERS)) {
    const at = from + m.index!;
    const open = src.indexOf("(", at + m[1].length);
    const end = closeOf(src, open);
    out.push({ at, line: lineOf(src, at), text: norm(src.slice(at, end < 0 ? src.length : end + 1)) });
  }
  return out;
}

/** etichetta leggibile in caso di fallimento: «1124: pushB(...)» */
const label = (c: Dispatch): string => `${c.line}: ${c.text}`;

// ------------------------------------------------------------- attese scritte

/**
 * LA COMPOSIZIONE ATTESA DEL SEGMENTO, per esteso. Due dispatch, in quest'ordine:
 * il GEMM di `ffn_down` (il tensore Q4_1) e il residuo che lo somma a `x`.
 *
 * IL SECONDO NON E' UN GEMM ed e' timbrato lo stesso: `addInPlaceWgsl` e' un
 * add elementwise. Sta qui perche' il test dice cosa il segmento CONTIENE, non
 * cosa dovrebbe contenere — se la riga 5 vorra' leggere `gemm:ffn-down-q41`
 * come «il solo costo del GEMM Q4_1», questo e' il termine da sottrarre o da
 * spostare, ed e' una decisione del PI, non di questo file (che non tocca il
 * sorgente).
 */
const SEGMENTO_ATTESO = [
  "gemvB(wd, PB.gateF, PB.attnY)",
  "pushB(addInPlaceWgsl(d, true), [PB.x, PB.attnY], [Math.ceil(d / 64), M_MAX, 1])",
];

const CAT_Q41 = "gemm:ffn-down-q41";
const CAT_BASE = "gemm:ffn-down";
/** `gemm:ffn-down` NON seguito da un suffisso: distingue la base dalla `-q41` */
const CAT_BASE_RE = /gemm:ffn-down(?![-\w])/g;

const src = code(MODEL);
const srcS = uncommented(MODEL);
const assigns = pbCatAssigns(src, srcS);
const q41 = assigns.filter((a) => /wd\.kind\s*===\s*"q4_1"/.test(a.rhs));

/**
 * L'unica assegnazione condizionata al kind Q4_1. Passa da qui e non da
 * `q41[0]` perche' un sorgente che la categoria non la assegna piu' deve
 * fallire DICENDOLO, non con un `undefined.at` a valle: e' lo stato in cui
 * questo file e' nato rosso (il sorgente prima della riga 4 non aveva ne' la
 * ternaria ne' la categoria).
 */
function soloQ41(): Assign {
  expect(
    q41.map((a) => `${a.line}: pbCat = ${a.rhs}`),
    "attesa UNA assegnazione a pbCat condizionata a `wd.kind === \"q4_1\"`",
  ).toHaveLength(1);
  return q41[0];
}

/** Il ciclo sui layer: e' il perimetro entro cui il timbro «gira». */
function loopBlock(): Block {
  const fors = headedBlocks(src, "for").filter((b) => /S\.nLayer/.test(b.head));
  expect(fors.map((b) => `${b.line}: for (${b.head})`), "atteso UN ciclo sui layer").toHaveLength(1);
  return fors[0];
}

/**
 * LA REGIONE VIVA del timbro Q4_1: i tratti di sorgente che, a runtime, un
 * dispatch attraversa con `cat === "gemm:ffn-down-q41"` addosso. Non e' un
 * intervallo solo, perche' `pbCat` non ha scope: e' la coda del ramo che lo
 * scrive, piu' tutto cio' che viene dopo, piu' — girando il ciclo — la testa
 * dell'iterazione successiva fino al primo `pbCat =` che la resetta.
 *
 * Il ramo `else` (MoE) NON e' nella lista, e la ragione si asserisce invece di
 * assumerla (vedi il test sul giro): e' irraggiungibile nella stessa iterazione
 * (i due rami si escludono) e nelle successive lo si tocca solo DOPO la prima
 * assegnazione del corpo del ciclo, che riscrive il timbro.
 */
interface Regione { nome: string; from: number; to: number }

function regioniVive(A: Assign): { segmento: Regione; code: Regione[] } {
  const rami = enclosingIfs(src, A.at);
  const pb = rami[0], dense = rami[1];
  expect([pb?.head, dense?.head], "i due rami piu' interni").toEqual(["PB", "!isMoe"]);
  const els = elseBlockOf(src, dense.end);
  expect(els, "il ramo `!isMoe` non ha un `else { … }`: la regione viva va ricalcolata").not.toBeNull();
  const loop = loopBlock();
  expect(loop.open, "il ramo dense non sta nel ciclo sui layer").toBeLessThan(dense.at);
  expect(loop.end).toBeGreaterThan(els!.end);

  // fine dello scope di `pbCat`: la `}` del blocco in cui e' dichiarato
  const decl = hitsOf(src, "let", "pbCat");
  expect(decl, "attesa UNA dichiarazione `let pbCat`").toHaveLength(1);
  const scopeEnd = closeOf(src, blockOpenOf(src, decl[0]));
  expect(scopeEnd).toBeGreaterThan(loop.end);

  const primo = assigns.find((a) => a.at > loop.open && a.at < loop.end)!;

  return {
    segmento: { nome: "coda di if (PB): IL SEGMENTO", from: A.end, to: pb.end },
    code: [
      { nome: "coda di if (!isMoe), dopo la graffa di if (PB)", from: pb.end, to: dense.end },
      { nome: "coda del corpo del ciclo, dopo il ramo dense/MoE", from: els!.end, to: loop.end },
      { nome: "testa dell'iterazione successiva, fino al primo pbCat =", from: loop.open, to: primo.at },
      { nome: "dopo il ciclo, fino a fine scope di pbCat", from: loop.end, to: scopeEnd },
    ],
  };
}

describe("pbCat: composizione del segmento gemm:ffn-down-q41", () => {
  it("un solo sito assegna la categoria Q4_1, e i commenti non contano", () => {
    // il biancamento e' load-bearing, non igiene: nel grezzo `gemm:ffn-down`
    // compare due volte (riga 1113, in un commento; riga 1122, nel codice), nel
    // biancato una sola. Se questo confronto si invertisse, tutte le conte di
    // sotto starebbero misurando prosa.
    expect(raw(MODEL).match(CAT_BASE_RE)?.length ?? 0).toBeGreaterThan(srcS.match(CAT_BASE_RE)?.length ?? 0);

    const A = soloQ41();
    expect(A.rhs).toContain(CAT_Q41);
    // esattamente UN sito di codice nomina la categoria, e ha l'offset di quel
    // sito: non «una volta da qualche parte».
    const siti = [...srcS.matchAll(new RegExp(CAT_Q41, "g"))].map((m) => m.index!);
    expect(siti).toHaveLength(1);
    expect(siti[0]).toBeGreaterThan(A.at);
    expect(siti[0]).toBeLessThan(A.end);
  });

  it("l'assegnazione sta in if (!isMoe) / if (PB), sul wd caricato da ffn_down.weight", () => {
    const A = soloQ41();
    const rami = enclosingIfs(src, A.at).map((b) => b.head);
    // i due rami piu' interni, nell'ordine: il contratto nomina questi.
    expect(rami.slice(0, 2)).toEqual(["PB", "!isMoe"]);
    // e sta al livello del ramo, non dentro un blocco annidato che potrebbe
    // non essere percorso: la graffa piu' interna che la contiene e' quella di
    // `if (PB)`.
    expect(blockOpenOf(src, A.at)).toBe(enclosingIfs(src, A.at)[0].open);

    // `wd` e' il tensore di ffn_down, e la sua dichiarazione sta nello stesso
    // ramo dense: senza questo, «wd» sarebbe un nome qualunque.
    const decl = /const\s+wd\s*=\s*await\s+[A-Za-z0-9_$]+\(\s*`\$\{b\}ffn_down\.weight`\s*\)/.exec(srcS);
    expect(decl, "dichiarazione di wd da `${b}ffn_down.weight` non trovata").not.toBeNull();
    expect(hitsOf(src, "const\\s+wd", "=")).toHaveLength(1);
    const dense = enclosingIfs(src, A.at).find((b) => b.head === "!isMoe")!;
    expect(decl!.index).toBeGreaterThan(dense.open);
    expect(decl!.index).toBeLessThan(dense.end);
  });

  it("e' l'ULTIMA assegnazione a pbCat prima del gemvB(wd, ...)", () => {
    const A = soloQ41();
    const gemvWd = dispatchesIn(src, 0, src.length).filter((c) => /^gemvB\(\s*wd\s*,/.test(c.text));
    expect(gemvWd).toHaveLength(1);
    const prima = assigns.filter((a) => a.at < gemvWd[0].at);
    expect(prima.length).toBeGreaterThan(0);
    expect(prima[prima.length - 1].at).toBe(A.at);
    // e il timbro precede davvero il dispatch che deve timbrare
    expect(A.end).toBeLessThan(gemvWd[0].at);
  });

  it("il segmento contiene ESATTAMENTE i dispatch attesi, in ordine", () => {
    const A = soloQ41();
    const { segmento } = regioniVive(A);
    const seg = dispatchesIn(src, segmento.from, segmento.to);
    expect(seg.map((c) => c.text)).toEqual(SEGMENTO_ATTESO);
    // il segmento e' contiguo al timbro: nessun dispatch fra `;` e il primo
    expect(seg[0].at).toBeGreaterThan(A.end);
  });

  it("il timbro non sopravvive: fuori dal ramo la regione viva e' VUOTA", () => {
    // IL CUORE DEL TEST, e la ragione per cui il confine non e' la graffa.
    // `pbCat` resta scritto finche' non lo si riscrive: dopo la `}` di
    // `if (PB)` il timbro Q4_1 e' ancora addosso a chiunque emetta, fino alla
    // fine del ramo dense, fino alla fine del corpo del ciclo, e oltre il ciclo
    // fino a fine funzione. Ognuno di quei tratti deve essere sgombro, o il
    // checkpoint della riga 5 sta pagando dispatch che non sono i quattro
    // tensori Q4_1.
    //
    // Non c'e' una lista attesa qui perche' l'attesa e' il VUOTO: qualunque
    // `pushB`/`gemvB`/`pushBRows1` che compaia in questi tratti finisce nel
    // segmento a runtime, e va o spostato o preceduto da un nuovo `pbCat =`.
    const A = soloQ41();
    const fuori = regioniVive(A).code.filter((r) => r.nome.startsWith("coda") || r.nome.startsWith("dopo"));
    expect(fuori.length).toBe(3);
    for (const r of fuori) {
      expect(dispatchesIn(src, r.from, r.to).map(label), `dispatch nella regione viva «${r.nome}»`).toEqual([]);
    }
    // e nessuno riassegna pbCat dopo la Q4_1: e' proprio per questo che la
    // regione viva si allunga fino a fine scope invece di fermarsi prima.
    expect(assigns.filter((a) => a.at > A.at).map((a) => a.line)).toEqual([]);
  });

  it("il GIRO del ciclo: la testa dell'iterazione successiva e' sgombra e resetta il timbro", () => {
    // L'ALTRA META' DEL CONFINE. Il timbro esce dall'iterazione N e rientra
    // nell'iterazione N+1: tutto cio' che sta fra la `{` del ciclo e il primo
    // `pbCat =` viene fatturato al segmento Q4_1 dei layer 0-3. Deve essere
    // sgombro, e il reset deve essere INCONDIZIONATO — se il primo `pbCat =`
    // stesse dentro un `if`, il giro potrebbe saltarlo e portare il timbro
    // dentro il ramo MoE, che di dispatch ne ha nove.
    const A = soloQ41();
    const loop = loopBlock();
    const giro = regioniVive(A).code.find((r) => r.nome.startsWith("testa"))!;
    expect(dispatchesIn(src, giro.from, giro.to).map(label), "dispatch prima del reset del timbro").toEqual([]);

    const primo = assigns.find((a) => a.at > loop.open && a.at < loop.end)!;
    expect(primo.rhs).toBe('"norm:attn"');
    // incondizionato: la graffa piu' interna che lo contiene e' quella del ciclo
    expect(blockOpenOf(src, primo.at)).toBe(loop.open);

    // ED ECCO PERCHE' IL RAMO `else` (MoE) NON STA NELLA REGIONE VIVA pur
    // essendo pieno di dispatch: non e' raggiungibile dall'assegnazione Q4_1
    // nella stessa iterazione (i due rami si escludono), e nelle successive lo
    // si raggiunge solo passando dal reset qui sopra.
    const dense = enclosingIfs(src, A.at).find((b) => b.head === "!isMoe")!;
    const els = elseBlockOf(src, dense.end)!;
    expect(els.open).toBeGreaterThan(primo.at);
    // il ramo MoE dispatch ne ha, e il test lo sa: non e' «vuoto quindi ok»
    expect(dispatchesIn(src, els.open, els.end).length).toBeGreaterThan(0);
  });

  it("nessun altro sito assegna la categoria Q4_1, e l'else resta gemm:ffn-down", () => {
    const A = soloQ41();
    // la ternaria, letta per pezzi dal sorgente biancato
    const t = /^(.*?)\s*\?\s*"([^"]*)"\s*:\s*"([^"]*)"$/.exec(A.rhs);
    expect(t, `RHS non e' una ternaria fra due categorie: ${A.rhs}`).not.toBeNull();
    expect(norm(t![1])).toBe('wd.kind === "q4_1"');
    expect(t![2]).toBe(CAT_Q41);
    expect(t![3]).toBe(CAT_BASE);

    // nessun'altra assegnazione a pbCat, ovunque, nomina la categoria Q4_1
    expect(assigns.filter((a) => a.rhs.includes(CAT_Q41)).map((a) => a.line)).toEqual([A.line]);
    // ne' la nomina qualunque altro pezzo di codice (import, mappe, default)
    expect(srcS.split(CAT_Q41).length - 1).toBe(1);
  });

  it("i quattro tensori Q4_1 del 4B sono i quattro siti che il segmento timbra", () => {
    // ANCORAGGIO ALL'ARTEFATTO. La categoria esiste perche' sul 4B ci sono
    // QUATTRO tensori Q4_1, non uno. Sono `blk.{0,1,2,3}.ffn_down.weight`
    // (verificato leggendo i nomi dal GGUF: gli altri 92 tensori della classe
    // `ffn/shexp` sono q4_0, e 92 + 4 = 96 = 32 layer × gate/up/down) — cioe'
    // esattamente il tensore che questo ramo carica in `wd`, nel ramo `!isMoe`
    // dove sta il timbro. Il legame 4 tensori ↔ 4 timbrate e' quindi: UN sito di
    // codice, attraversato una volta per layer denso, che stampa la categoria
    // Q4_1 sui primi 4 layer e `gemm:ffn-down` sui restanti 28. E' la quota che
    // fino a ieri stava DEDOTTA dentro `gemm:ffn-down` insieme ai q4_0.
    //
    // L'artefatto e' il testimone che il numero e' 4 e non 1: se una revisione
    // del modello ne portasse 5, questa asserzione cade e la lettura della riga
    // 5 va rifatta prima, non dopo.
    const dump = JSON.parse(raw(DUMP)) as Array<{ file: string; typeHistogram: Record<string, { n: number; bytes: number }> }>;
    expect(dump[0].file).toContain("Qwen3.5-4B-Q4_0.gguf");
    expect(dump[0].typeHistogram["ffn/shexp:Q4_1"].n).toBe(4);
    // un solo sito di codice li timbra tutti e quattro: il conto dei tensori
    // non e' il conto dei siti, ed e' la ragione per cui la riga 5 leggera' UN
    // checkpoint e non quattro.
    expect(q41).toHaveLength(1);
  });
});
