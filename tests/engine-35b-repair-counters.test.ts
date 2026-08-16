// Scomposizione del repair del 35B (goal engine-35b-residency, riga 1).
//
// PERCHE' QUESTO TEST ESISTE. Sul turno del 2026-08-15
// (`results/chat/chat-35b-2026-08-15T16-25-12.json`) il 43% del tempo di parete
// del 35B non aveva un nome: stava dentro `repairMs` ma fuori da
// `packMs`+`uploadMs`, e `readMs` di `residency.ts` non lo vedeva perche' misura
// `readRaw` DENTRO `ensure` mentre il 35B fa l'I/O PRIMA (l'avvertenza e' a
// `residency.ts:250-257`, ed era li' da prima: nessuno l'aveva incrociata con
// un turno lungo). Un contatore DICHIARATO E MAI INCREMENTATO, o incrementato
// e mai serializzato, riprodurrebbe esattamente quel buco — con l'aggravante di
// sembrare risolto. Il test pretende la catena intera: dichiarato, scritto,
// propagato ai due worker che lo serializzano.
//
// STATICO, senza GPU: stessa postura di `engine-q35attnwiring.test.ts`, si
// legge il sorgente come testo. Qui pero' bastano i COMMENTI biancati, non
// anche stringhe e template: gli identificatori cercati sono chiavi d'oggetto e
// accessi a proprieta', e in questi tre file non compaiono dentro letterali.
// Servono biancati i commenti perche' si': questo goal ne ha scritti parecchi
// che nominano i contatori, e senza il passo il test si darebbe verde da solo
// leggendo la propria documentazione.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const MODEL = "src/engine/q35gpumodel.ts";
const CHAT = "src/engine/chat/chat.worker.ts";
const CONF = "src/engine/q35conf/q35conf.worker.ts";
// ancorato al file di test, non a process.cwd(): con `cd tests` sarebbe ENOENT
const ROOT = join(__dirname, "..");

/** commenti di riga e di blocco sostituiti da spazi (le righe si conservano) */
function stripComments(src: string): string {
  const out = src.split("");
  let i = 0;
  while (i < src.length) {
    const c = src[i], d = src[i + 1] ?? "";
    if (c === "/" && d === "/") {
      while (i < src.length && src[i] !== "\n") { out[i] = " "; i++; }
      continue;
    }
    if (c === "/" && d === "*") {
      const e = src.indexOf("*/", i + 2), stop = e < 0 ? src.length : e + 2;
      for (let j = i; j < stop; j++) if (out[j] !== "\n") out[j] = " ";
      i = stop; continue;
    }
    i++;
  }
  return out.join("");
}

const memo = new Map<string, string>();
const code = (rel: string): string => {
  if (!memo.has(rel)) memo.set(rel, stripComments(readFileSync(join(ROOT, rel), "utf8")));
  return memo.get(rel)!;
};

/** i contatori che la riga 1 aggiunge, con il verso in cui vanno scritti */
const NEW_COUNTERS = [
  "fetchRepairMs", "fetchRepairCalls", "fetchRepairBytes",
  // `fetchPrepBytes` e' arrivato dopo (riga 2b, it.31) e la sua assenza aveva
  // un costo: senza, i due regimi di fetch si confrontavano in ms PER CHIAMATA
  // di taglia ignota, e il 2,1x che apre quella riga poggiava sull'assunzione
  // che `prep` e `repair` leggessero gli stessi byte. Il repair il suo
  // contatore ce l'aveva dal primo giorno; il prep no, e nessuno se n'era
  // accorto perche' la lista qui sotto lo rispecchiava.
  "fetchPrepMs", "fetchPrepCalls", "fetchPrepBytes",
  "replayPassMs", "flushMs",
] as const;

describe("35B — la scomposizione del repair e' dichiarata, scritta e propagata", () => {
  it("[1] ogni contatore nuovo e' nel tipo di ritorno di perf() E nell'accumulatore", () => {
    const src = code(MODEL);
    // il tipo: dentro la firma `perf(): { ... }`
    const sig = src.indexOf("perf(): {");
    expect(sig, "firma di perf() non trovata").toBeGreaterThan(-1);
    const sigEnd = src.indexOf("moeStats", sig);
    const typeBlock = src.slice(sig, sigEnd > sig ? sigEnd : sig + 4000);
    // l'accumulatore: `const perfAcc = { ... }`
    const acc = src.indexOf("const perfAcc = {");
    expect(acc, "accumulatore perfAcc non trovato").toBeGreaterThan(-1);
    const accBlock = src.slice(acc, src.indexOf("};", acc));
    for (const k of NEW_COUNTERS) {
      expect(typeBlock, `${k} manca dal tipo di ritorno di perf()`).toContain(k);
      expect(accBlock, `${k} manca dall'inizializzatore di perfAcc`).toContain(k);
    }
  });

  it("[2] ogni contatore nuovo viene INCREMENTATO almeno una volta", () => {
    const src = code(MODEL);
    for (const k of NEW_COUNTERS) {
      const written = new RegExp(`perfAcc\\.${k}\\s*\\+=`).test(src);
      expect(written, `perfAcc.${k} e' dichiarato ma non lo scrive nessuno`).toBe(true);
    }
  });

  it("[3] le DUE await della fetch sono entrambe misurate, e separatamente (C0-3)", () => {
    const src = code(MODEL);
    // i siti: `await Promise.all(...readExpert...)`. Sono due, e restano due:
    // il repair (miss scoperti a fine pass) e prepLayer (miss noti prima del
    // layer). Un terzo sito non misurato sarebbe un buco nuovo.
    const sites = src.split("\n").filter((l) => l.includes("Promise.all") && l.includes("readExpert"));
    expect(sites.length, "i siti di fetch degli expert non sono piu' due").toBe(2);
    // e i due contatori sono distinti: sommarli renderebbe illeggibile quale
    // dei due regimi paga, che e' la ragione per cui sono due nomi e non uno
    expect(src).toContain("perfAcc.fetchRepairMs +=");
    expect(src).toContain("perfAcc.fetchPrepMs +=");
    expect(/perfAcc\.fetchRepairMs\s*\+=/.test(src) && /perfAcc\.fetchPrepMs\s*\+=/.test(src)).toBe(true);
  });

  it("[4] tTail sta PRIMA del loop di repair — il fatto che il commento di perf() dichiara", () => {
    const src = code(MODEL);
    // Il commento di `perf()` diceva che `tailCpuMs` e' «la contabilita' di fine
    // token». E' falso: `tTail` e' preso prima del `while`, quindi tailCpu
    // include repair e replay (84% del token sul 35B, non un residuo). Il
    // commento e' stato corretto il 2026-08-15; questo test impedisce che la
    // struttura cambi lasciando indietro la correzione — o viceversa.
    const tTail = src.indexOf("const tTail = performance.now()");
    const loop = src.indexOf("while (cur.missCount > 0)");
    const close = src.indexOf("perfAcc.tailCpuMs +=");
    expect(tTail, "tTail non trovato").toBeGreaterThan(-1);
    expect(loop, "loop di repair non trovato").toBeGreaterThan(-1);
    expect(tTail, "tTail non e' piu' prima del loop di repair: il commento di perf() va rifatto")
      .toBeLessThan(loop);
    expect(close, "la chiusura di tailCpuMs non e' piu' dopo il loop").toBeGreaterThan(loop);
  });

  it("[5] i contatori arrivano fino ai due worker che li serializzano", () => {
    // Un contatore che si ferma nel motore non finisce in nessun artefatto, e
    // l'artefatto e' l'unica cosa che il done-when puo' leggere.
    const chat = code(CHAT), conf = code(CONF);
    for (const k of NEW_COUNTERS) {
      expect(chat, `${k} non arriva al JSON della chat`).toContain(k);
      expect(conf, `${k} non arriva al report di q35conf`).toContain(k);
    }
  });

  it("[6] q35conf calcola la quota NOMINATA di tailCpuMs, che e' la clausola del done-when", () => {
    const conf = code(CONF);
    // `namedFrac` >= 0,95 e' il done-when della riga 1. Va calcolato
    // NELL'ARTEFATTO, non a mano dopo: un done-when che si verifica con una
    // calcolatrice non e' meccanico.
    expect(conf).toContain("namedFrac");
    expect(conf).toContain("accountingMs");
    // e la contabilita' vera si ottiene per DIFFERENZA, mai per assunzione
    expect(conf.replace(/\s+/g, " "))
      .toContain("r.tailCpuMs - r.repairMs - r.replayPassMs");
  });
});
