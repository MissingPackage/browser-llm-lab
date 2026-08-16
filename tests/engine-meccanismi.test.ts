// La mappa dei meccanismi non mente (docs/architettura/MECCANISMI.md).
//
// PERCHE' ESISTE. Il 2026-08-16, in una notte sola, ho "scoperto" tre volte
// qualcosa che era gia' costruito e non adottato: il pre-pack degli slab, la
// policy `tier`, e le due leve del decode spente nella chat. Il PI l'ha detto
// meglio di me: non e' che misuriamo poco, e' che mi perdo il software che
// abbiamo scritto.
//
// La mappa serve a rendere visibili le CELLE VUOTE — un grep dice cosa c'e',
// non cosa manca. Ma una mappa che marcisce e' peggio di nessuna mappa: la si
// consulta e ci si fida. Questi casi la tengono onesta confrontandola col
// SORGENTE, non con se stessa.
//
// COSA VERIFICANO E COSA NO. Verificano le affermazioni STRUTTURALI («chi usa
// cosa», «qual e' il default»), che sono quelle che mi hanno fatto sbagliare.
// NON verificano i numeri misurati: quelli stanno negli artefatti in results/
// e hanno il loro journal. Un test che pinnasse anche i ms diventerebbe una
// seconda verita' da aggiornare a ogni run.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

const read = (p: string): string => readFileSync(new URL(`../${p}`, import.meta.url), "utf8");
const MAPPA = read("docs/architettura/MECCANISMI.md");

/** il sorgente senza commenti: le affermazioni si provano sul CODICE */
const code = (p: string): string =>
  read(p).replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");

describe("la mappa dei meccanismi combacia col sorgente", () => {
  it("l'interfaccia dello slab pre-impacchettato vive in residency.ts", () => {
    expect(MAPPA).toContain("`residency.ts:110`");
    expect(code("src/engine/residency.ts")).toMatch(/slab\?:\s*\(/);
  });

  it("il GLM adotta { raw, slab } e il 35B NO — e la mappa lo dice", () => {
    const glm = code("src/engine/glmmodel.ts");
    const q35 = code("src/engine/q35gpumodel.ts");
    const glmAdotta = /slab:\s*\(/.test(glm);
    const q35Adotta = /slab:\s*\(/.test(q35);
    expect(glmAdotta, "glmmodel deve passare { raw, slab }").toBe(true);
    // IL GIORNO CHE q35 LO ADOTTA, questo caso fallisce e la riga va aggiornata:
    // e' il punto — la cella vuota deve smettere di essere vuota anche nella mappa
    expect(q35Adotta, "se q35gpumodel ora adotta lo slab, aggiorna MECCANISMI.md §1").toBe(false);
    expect(MAPPA).toContain("non adottato");
  });

  it("le due leve del decode nascono SPENTE e la chat le accende", () => {
    const q35 = code("src/engine/q35gpumodel.ts");
    expect(q35).toMatch(/let kfanEnabled = false/);
    expect(q35).toMatch(/let splitkEnabled = false/);
    const chat = code("src/engine/chat/chat.worker.ts");
    expect(chat).toMatch(/setKfan\(true\)/);
    expect(chat).toMatch(/setSplitk\(true\)/);
    expect(MAPPA).toContain("nascono spente");
  });

  it("la policy di residenza ha default `lru`, come dichiarato", () => {
    const res = code("src/engine/residency.ts");
    expect(res).toMatch(/policy\?:\s*"lru"\s*\|\s*"tier"/);
    expect(code("src/engine/q35gpumodel.ts")).toMatch(/expertPolicy\s*\?\?\s*"lru"/);
    expect(MAPPA).toContain("**`lru`**");
  });

  it("la via intera nel decode e' cablata e spenta, col nome che la mappa cita", () => {
    expect(code("src/engine/q35gpumodel.ts")).toMatch(/DEC_SPLITK_IDOT\s*=\s*false/);
    expect(MAPPA).toContain("DEC_SPLITK_IDOT=false");
  });

  it("il lettore Range e' UNO e la mappa dice che tutti e cinque lo usano", () => {
    expect(MAPPA).toContain("tutti e 5 i chiamanti");
    // lo stesso invariante che `engine-ggufrange.test.ts` prova contando i siti:
    // qui basta che il modulo sia quello nominato
    expect(code("src/engine/ggufrange.ts")).toMatch(/ggufRangeReader/);
  });

  it("il regime di lettura e' dichiarato dal glmbench e NON da q35conf", () => {
    expect(/readRegime/.test(code("src/engine/glmbench/glmbench.worker.ts"))).toBe(true);
    // cella vuota dichiarata: se q35conf lo aggiunge, la mappa va aggiornata
    expect(/readRegime/.test(code("src/engine/q35conf/q35conf.worker.ts")),
      "se q35conf ora dichiara il regime, aggiorna MECCANISMI.md §4").toBe(false);
    expect(MAPPA).toContain("q35conf non lo dichiara");
  });

  it("il prefetch lookahead NON e' implementato nel motore: solo misurato", () => {
    // se un giorno comparisse in un modello, la riga «non implementato» mente
    for (const p of ["src/engine/q35gpumodel.ts", "src/engine/glmmodel.ts"]) {
      expect(/looka/i.test(code(p)), `${p} sembra implementare il lookahead: aggiorna la mappa`).toBe(false);
    }
    expect(MAPPA).toContain("non implementato");
  });

  it("ogni strumento di misura elencato esiste davvero", () => {
    const runner = code("scripts/q35-conf-run.mjs");
    for (const flag of ["io-probe", "gpu-time", "logit-probe", "kfan", "splitk"]) {
      expect(runner, `--${flag} elencato nella mappa`).toContain(flag);
    }
    expect(code("scripts/chat-smoke.mjs")).toContain("policy");
    expect(code("scripts/q35-slab-build.mjs")).toContain("dry-run");
  });
});
