// Cablaggio dell'attention decode a split (goal engine-kernel-decode, T2).
//
// Test STATICO, senza GPU: stessa postura di `tests/gpulimits.test.ts`, che
// ri-deriva i limiti SCANSIONANDO il WGSL vero invece di fidarsi di un numero
// scritto a mano. Qui si scansionano i due ORCHESTRATORI (`q35gpumodel.ts` e
// `ktest/ktest.worker.ts`) e si asserisce il CABLAGGIO, non l'aritmetica:
// l'aritmetica dello split ha gia' i suoi test (engine-q35attnsplit,
// engine-q35attndecode), ma nessuno di quelli si accorge se il pass 1 resta
// appeso a una lista di binding a 4 buffer — cioe' se il kernel a split viene
// generato e poi chiamato con la firma del kernel monolitico di ieri.
//
// I file si LEGGONO come testo. Importarli tirerebbe dentro i tipi WebGPU e un
// modulo worker (che in vitest node non si carica): non serve, perche' cio' che
// va provato e' quali chiamate esistono e con quanti buffer, non cosa fanno.
//
// Il testo non si scansiona grezzo: commenti, stringhe e template WGSL vengono
// prima BIANCATI (§ blankNonCode). Senza quel passo un commento che nomina
// `attnDecodeWgsl(` sposta i conteggi congelati, e — peggio — un commento che
// cita una lista `[q, kCache, vCache, attnO]` accanto alla chiamata diventa la
// lista che il test misura al posto di quella vera, in entrambi i versi:
// falso verde su un cablaggio rotto, falso rosso su uno corretto. In questi due
// file ci sono gia' 17 righe di commento che contengono una parentesi quadra.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const MODEL = "src/engine/q35gpumodel.ts";
const KTEST = "src/engine/ktest/ktest.worker.ts";
// come gpulimits.test.ts: percorso ancorato al file di test, non alla directory
// da cui si lancia vitest (con process.cwd() un `cd tests` fa ENOENT).
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
 * CHIAVE (`return /x/.test(s)`) viene letto come divisione. Nei due file
 * scansionati i soli regex stanno dopo `(`, che e' riconosciuto.
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

/** contenuto di ogni `import { ... } from "<src>"` che viene da `suffix` */
const importsFrom = (src: string, suffix: string): string =>
  [...src.matchAll(/import\s+(?:type\s+)?\{([^}]*)\}\s*from\s*["']([^"']+)["']/g)]
    .filter((m) => m[2].endsWith(suffix)).map((m) => m[1].replace(/\s+/g, " ").trim()).join(" ; ");

/**
 * Definizione di un helper, sia nella forma `function f(` sia in quella
 * `const f = (` — q35gpumodel definisce `push`/`pushB` come const-arrow, e il
 * cablaggio non deve dipendere da quale delle due forme sceglie chi scrive.
 * Ritorna il corpo `{...}`: il primo gruppo graffo a profondita' di parentesi 0
 * dopo il nome (cosi' parametri destrutturati e tipi inline non lo confondono).
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

describe("(a) q35gpumodel: lo split e' cablato UNA volta e usato DUE", () => {
  it("pushAttnDecodeSplit: 1 definizione + 2 chiamate (piano token, testa MTP)", () => {
    const src = code(MODEL);
    // definizione e usi insieme: `f(` per la chiamata e per `function f(`,
    // `f =` per la forma const-arrow.
    const where = hitsOf(src, "pushAttnDecodeSplit", "[(=]").map((i) => `${MODEL}:${lineOf(src, i)}`);
    expect(where.length, `pushAttnDecodeSplit a ${where.join(", ") || "NESSUNA riga"}`).toBe(3);
  });

  it("il corpo dell'helper emette pass 1 E combine", () => {
    const body = helperBody(code(MODEL), "pushAttnDecodeSplit");
    expect(body === null ? `${MODEL}: pushAttnDecodeSplit non e' definito` : "ok").toBe("ok");
    const missing = ["attnDecodeWgsl(", "attnDecodeCombineWgsl("].filter((n) => !body!.includes(n));
    expect(missing.join(", "), `il corpo di pushAttnDecodeSplit non chiama: ${missing.join(", ")}`).toBe("");
  });

  it("attnDecodeWgsl( resta 2 volte (helper + ramo batch), attnDecodeRefWgsl 0", () => {
    // 2 e non 3: i due call-site non-batch passano dall'helper, il ramo
    // `batch: true` del piano a chunk resta monolitico per costruzione.
    const src = code(MODEL);
    const w = hitsOf(src, "attnDecodeWgsl", "\\(").map((i) => `${MODEL}:${lineOf(src, i)}`);
    expect(w.length, `attnDecodeWgsl( a ${w.join(", ") || "NESSUNA riga"}`).toBe(2);
    const r = hitsOf(src, "attnDecodeRefWgsl").map((i) => `${MODEL}:${lineOf(src, i)}`);
    // il kernel di riferimento sta nel ktest, non in produzione
    expect(r.length, `attnDecodeRefWgsl a ${r.join(", ")}`).toBe(0);
  });
});

describe("(b) ktest.worker: i banchi coprono split, batch e riferimento", () => {
  it("attnDecodeWgsl( 4 volte, di cui esattamente 1 con batch: true", () => {
    const src = code(KTEST);
    const calls = callsTo(src, "attnDecodeWgsl");
    const all = calls.map((c) => `${KTEST}:${c.line}`);
    expect(calls.length, `attnDecodeWgsl( a ${all.join(", ") || "NESSUNA riga"}`).toBe(4);
    const batch = calls.filter((c) => c.batch).map((c) => `${KTEST}:${c.line}`);
    expect(batch.length, `batch: true a ${batch.join(", ") || "NESSUN call-site"}`).toBe(1);
  });

  it("attnDecodeCombineWgsl( 3 volte (una per pass 1 non-batch) e attnDecodeRefWgsl( 1", () => {
    const src = code(KTEST);
    const comb = hitsOf(src, "attnDecodeCombineWgsl", "\\(").map((i) => `${KTEST}:${lineOf(src, i)}`);
    expect(comb.length, `attnDecodeCombineWgsl( a ${comb.join(", ") || "NESSUNA riga"}`).toBe(3);
    const ref = hitsOf(src, "attnDecodeRefWgsl", "\\(").map((i) => `${KTEST}:${lineOf(src, i)}`);
    expect(ref.length, `attnDecodeRefWgsl( a ${ref.join(", ") || "NESSUNA riga"}`).toBe(1);
  });
});

describe("(c) nessun call-site non-batch a 4 buffer", () => {
  // Il difetto che questo test esiste per prendere: il pass 1 dello split
  // scrive in DUE buffer parziali (partOut, partMS) al posto dell'unico
  // `attnO` di ieri. Una chiamata rimasta a [q, kCache, vCache, attnO] compila
  // e gira — e legge come output un buffer che il kernel non riempie piu'.
  //
  // La lista conta ELEMENTI, non buffer: `[q, kC, vC, o, uni]` (4 buffer + una
  // uniform, la forma monolitica di ktest.worker:2958 oggi) ne ha gia' 5. Per
  // questo al conteggio si aggiunge l'accoppiamento: ogni pass 1 non-batch deve
  // avere un `attnDecodeCombineWgsl(` prima del pass 1 non-batch successivo.
  // Cinque elementi senza combine a valle non sono uno split.
  for (const rel of [MODEL, KTEST]) {
    it(`${rel}: ogni chiamata non-batch lega 5 elementi e ha il suo combine`, () => {
      const src = code(rel);
      const pass1 = callsTo(src, "attnDecodeWgsl").filter((c) => !c.batch);
      const combines = callsTo(src, "attnDecodeCombineWgsl").map((c) => c.at);
      expect(pass1.length, `${rel}: nessuna chiamata non-batch trovata`).toBeGreaterThan(0);
      const bad = pass1.flatMap((c, k) => {
        const where = `${rel}:${c.line}`;
        if (c.bindings === null) {
          return [`${where} lista di binding non inline dopo attnDecodeWgsl(...): cablaggio non verificabile`];
        }
        if (c.bindings.length !== 5) {
          return [`${where} lega ${c.bindings.length} elementi [${c.bindings.join(", ")}]`];
        }
        const next = pass1[k + 1]?.at ?? src.length;
        if (!combines.some((i) => i > c.at && i < next)) {
          return [`${where} nessun attnDecodeCombineWgsl( tra questo pass 1 e il successivo`];
        }
        return [];
      });
      expect(bad.join("\n")).toBe("");
    });
  }
});

describe("(d) i due file importano il piano dello split e il combine", () => {
  for (const rel of [MODEL, KTEST]) {
    it(`${rel}: q35attnsplit + attnDecodeCombineWgsl`, () => {
      const src = uncommented(rel); // gli import sono stringhe: qui non si biancano
      const split = importsFrom(src, "q35attnsplit");
      expect(
        /q35AttnPartialsFloats|q35AttnSplitPlan/.test(split) ? "ok" : `${rel}: import da q35attnsplit = "${split}"`,
      ).toBe("ok");
      const kern = importsFrom(src, "kernels/wgsl");
      expect(
        kern.includes("attnDecodeCombineWgsl") ? "ok" : `${rel}: import da kernels/wgsl = "${kern}"`,
      ).toBe("ok");
    });
  }
});
