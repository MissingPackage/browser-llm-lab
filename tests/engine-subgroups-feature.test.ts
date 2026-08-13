// La feature `subgroups` e' CONCESSA sui due device che contano (goal
// gemv-subgroups, T3).
//
// Test STATICO, senza GPU: stessa postura di `tests/engine-q35attnwiring.test.ts`
// — i worker si LEGGONO come testo invece di importarli, perche' importarli
// tirerebbe dentro i tipi WebGPU e un modulo worker (che in vitest node non si
// carica), e cio' che va provato e' COSA viene chiesto al device, non cosa fa
// il worker dopo.
//
// PERCHE' ESISTE. `createEngineDevice` filtra: `requiredFeatures =
// optionalFeatures.filter(f => adapter.features.has(f))`. Se `subgroups` non
// sta nell'`optionalFeatures` del chiamante, l'adapter puo' pure esporla —
// il device non la avra' mai, `gemvCapsFor(device)` dira' `sg: false` e il
// kernel a subgroup non entrera' MAI in gioco. Il difetto sarebbe muto: niente
// errore, niente shader rotto, solo il path lento per sempre. Il rovescio: chi
// aggiunge `subgroups` puo' SOSTITUIRE l'array e perdere `timestamp-query`,
// e allora a diventare muta e' la telemetria (`has()` falso, gpuBusy null).
// Il test guarda tutte e due le direzioni.
//
// I due file sono quelli su cui si misura e si gioca: q35conf.worker e' il
// device del bench `scripts/q35-bench-run.mjs` (ms/token), chat.worker e'
// quello della chat. Le ALTRE assenze (engine.worker/gpuforward sul path
// Qwen2.5, i worker GLM) sono eccezioni dichiarate, non dimenticanze: le
// censisce T6-coverage-census, non questo test.
//
// Il testo non si scansiona grezzo: i commenti vengono prima BIANCATI (le
// stringhe NO — qui sono proprio i nomi delle feature il dato da leggere).
// Senza quel passo il commento di q35conf che nomina `timestamp-query` due
// righe sopra la chiamata basterebbe a far passare il test anche con l'array
// svuotato.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const Q35CONF = "src/engine/q35conf/q35conf.worker.ts";
const CHAT = "src/engine/chat/chat.worker.ts";
const GPUDEVICE = "src/engine/gpudevice.ts";
// come engine-q35attnwiring.test.ts: percorso ancorato al file di test, non
// alla directory da cui si lancia vitest.
const ROOT = join(__dirname, "..");
const raw = (rel: string): string => readFileSync(join(ROOT, rel), "utf8");

/**
 * Sostituisce con spazi (le righe si conservano) tutto cio' che non e' codice:
 * commenti di riga e di blocco, letterali regex e — se `keepStrings` e' falso —
 * stringhe e template literal. Copia di quella di engine-q35attnwiring.test.ts:
 * i due test sono deliberatamente indipendenti (un helper condiviso in
 * `tests/` diventerebbe un modulo che va tenuto in piedi da solo).
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
/**
 * Sorgente coi soli COMMENTI biancati: le stringhe restano, perche' i nomi
 * delle feature ("subgroups", "timestamp-query") sono stringhe — biancarle
 * cancellerebbe proprio il dato che il test misura.
 */
const uncommented = (rel: string): string => {
  if (!memo.has(rel)) memo.set(rel, blankNonCode(raw(rel), true));
  return memo.get(rel)!;
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

interface DeviceCall { line: number; features: string[] | null }

/**
 * Le chiamate a `createEngineDevice(`, ognuna con le feature opzionali che
 * chiede. L'array deve stare INLINE dentro l'oggetto di opzioni: se e' passato
 * per variabile `features` resta null e il test lo DICE, invece di agganciare
 * il primo `[` che trova (che sarebbe un array scorrelato). Ogni elemento e'
 * normalizzato togliendo il cast (`"subgroups" as GPUFeatureName`, la forma che
 * serve perche' il tipo `GPUFeatureName` non elenca subgroups) e le virgolette.
 */
function deviceCalls(src: string): DeviceCall[] {
  return hitsOf(src, "createEngineDevice", "\\(").map((at) => {
    const open = src.indexOf("(", at);
    const end = closeOf(src, open);
    const line = lineOf(src, at);
    if (end < 0) return { line, features: null };
    const prop = [...src.slice(open, end).matchAll(/(?<![A-Za-z0-9_$])optionalFeatures\s*:/g)][0];
    if (!prop) return { line, features: null };
    const bracket = src.indexOf("[", open + prop.index! + prop[0].length);
    if (bracket < 0 || bracket > end) return { line, features: null };
    return {
      line,
      features: topLevelItems(src, bracket)
        .map((s) => s.replace(/\s+as\s+[A-Za-z0-9_$.]+\s*$/, "").trim().replace(/^["'`]|["'`]$/g, "")),
    };
  });
}

describe("subgroups e' fra le feature opzionali del device del bench e della chat", () => {
  for (const rel of [Q35CONF, CHAT]) {
    it(`${rel}: optionalFeatures contiene subgroups e timestamp-query`, () => {
      const calls = deviceCalls(uncommented(rel));
      // un solo device per worker: se ne comparisse un secondo, il test va
      // riscritto guardandoli tutti invece di cadere in un falso verde.
      expect(calls.length, `${rel}: createEngineDevice( a ${calls.map((c) => c.line).join(", ") || "NESSUNA riga"}`).toBe(1);
      const c = calls[0];
      expect(
        c.features === null
          ? `${rel}:${c.line} createEngineDevice senza optionalFeatures inline: feature non verificabili`
          : "ok",
      ).toBe("ok");
      const missing = ["subgroups", "timestamp-query"].filter((f) => !c.features!.includes(f));
      expect(
        missing.join(", "),
        `${rel}:${c.line} optionalFeatures = [${c.features!.join(", ")}] — mancano: ${missing.join(", ")}`,
      ).toBe("");
    });
  }
});

describe("gpudevice: il filtro sull'adapter e' ancora quello", () => {
  // T3 non tocca gpudevice.ts, e questa asserzione lo mette per iscritto: il
  // valore di aggiungere `subgroups` all'array sta TUTTO nel filtro. Se domani
  // qualcuno chiedesse le optionalFeatures senza filtrarle, su un adapter che
  // non le espone `requestDevice` rigetterebbe — e i due worker qui sopra,
  // appena modificati, sarebbero i primi a non partire piu'.
  it("createEngineDevice filtra optionalFeatures con adapter.features.has(f)", () => {
    const src = uncommented(GPUDEVICE);
    expect(
      src.includes("adapter.features.has(f)") ? "ok" : `${GPUDEVICE}: filtro adapter.features.has(f) assente`,
    ).toBe("ok");
  });
});
