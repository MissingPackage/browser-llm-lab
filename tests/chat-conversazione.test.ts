// LE DUE COPIE DELLA CONVERSAZIONE DEVONO COINCIDERE.
//
// La conversazione di prova (10 turni di follow-up, ruling del PI 2026-08-17)
// vive in DUE posti perche' i due runner parlano linguaggi diversi:
// `scripts/chat-smoke.mjs` pilota il motore nel browser, `scripts/chat-llamacpp.py`
// il riferimento su CPU. Non e' una duplicazione da fattorizzare — non esiste un
// modulo che JS e Python importino entrambi senza aggiungere una dipendenza — ma
// e' una duplicazione da SORVEGLIARE: se le due liste divergono, i due bracci
// rispondono a domande diverse e il confronto di qualita' misura il prompt
// invece del modello.
//
// Lo stesso vale per il RENDERING del prompt: il runner Python dichiara di
// copiarlo dal worker, e qui si verifica che le stringhe siano quelle.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

const read = (p: string): string => readFileSync(new URL(`../${p}`, import.meta.url), "utf8");

/** Estrae le domande da un array literal delimitato da `apri`/`chiudi`. */
function domande(src: string, apri: string): string[] {
  const i = src.indexOf(apri);
  if (i < 0) throw new Error(`blocco "${apri}" non trovato`);
  const j = src.indexOf("];", i);
  const blocco = src.slice(i, j);
  return [...blocco.matchAll(/^\s*"((?:[^"\\]|\\.)*)",\s*$/gm)].map((m) => m[1]);
}

describe("la conversazione di prova e' la STESSA nei due runner", () => {
  const js = domande(read("scripts/chat-smoke.mjs"), "const CONVERSAZIONE = [");
  const py = domande(read("scripts/chat-llamacpp.py"), "CONVERSAZIONE = [");

  it("sono dieci turni in entrambi", () => {
    expect(js.length, "chat-smoke.mjs").toBe(10);
    expect(py.length, "chat-llamacpp.py").toBe(10);
  });

  it("le domande coincidono una per una", () => {
    expect(py).toEqual(js);
  });

  it("il primo turno e' quello dei bracci storici del 2026-08-17", () => {
    // se cambia, i confronti con le misure precedenti smettono di reggere e va
    // detto invece che scoperto
    expect(js[0]).toBe("Che relazione c'e' tra entropia dell'informazione e compressione?");
  });

  it("il rendering del riferimento e' quello del worker, non il template Jinja", () => {
    const py2 = read("scripts/chat-llamacpp.py");
    expect(py2).toContain('SYS = "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n"');
    expect(py2).toContain('USER = "<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n"');
    // e il worker deve ancora usare quelle stesse marche
    const w = read("src/engine/chat/chat.worker.ts");
    expect(w).toContain("<|im_start|>");
    expect(w).toContain("<|im_end|>");
  });
});
