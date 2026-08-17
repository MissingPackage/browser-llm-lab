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
import { KQUANT_GEMV_DESC, prefillGemmWiring } from "../src/engine/kernels/wgsl";

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

  it("ENTRAMBE le famiglie adottano { raw, slab }, per due strade diverse — e la mappa dice quali", () => {
    // La cella vuota di it.49 e' stata riempita in it.50: il 35B legge lo slab
    // gia' impacchettato dal file servito via Range, il GLM dal suo file in
    // OPFS generato all'import. Due sorgenti, UNA interfaccia — ed e' quello
    // che questo caso sorveglia: se qualcuno riscrive il pack nel path caldo
    // del 35B, o ricopia la lettura invece di passare dalla sorgente, qui si
    // vede.
    const glm = code("src/engine/glmmodel.ts");
    const q35 = code("src/engine/q35gpumodel.ts");
    expect(/slab:\s*\(/.test(glm), "glmmodel deve passare { raw, slab }").toBe(true);
    expect(q35, "q35gpumodel deve aprire la sorgente slab").toContain("openSlabRangeSource");
    expect(q35, "q35gpumodel deve consegnare lo slab alla cache con slabInHand").toContain("slabInHand");
    // il fallback dev'essere DICHIARATO nell'artefatto, non silenzioso
    expect(q35, "il motivo del fallback deve finire in moeStats").toContain("slabSource");
    expect(MAPPA).toContain("`slabsource.ts`");
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
      "se q35conf ora dichiara il regime, aggiorna MECCANISMI.md §5").toBe(false);
    expect(MAPPA).toContain("q35conf non lo dichiara");
  });

  it("il prefetch lookahead NON e' implementato nel motore: solo misurato", () => {
    // se un giorno comparisse in un modello, la riga «non implementato» mente
    for (const p of ["src/engine/q35gpumodel.ts", "src/engine/glmmodel.ts"]) {
      expect(/looka/i.test(code(p)), `${p} sembra implementare il lookahead: aggiorna la mappa`).toBe(false);
    }
    expect(MAPPA).toContain("non implementato");
  });

  it("i gemv K-quant escono da UN generatore, e la mappa lo dice col nome giusto", () => {
    // La riga §4 afferma «tutti e 5 i formati escono da UN generatore». Se
    // qualcuno riscrivesse un gemv a mano, la mappa mentirebbe: qui si prova
    // che i cinque wrapper delegano davvero e che i descrittori sono cinque.
    expect(MAPPA).toContain("`wgsl.ts:gemvKQuantWgsl`");
    expect(Object.keys(KQUANT_GEMV_DESC).sort())
      .toEqual(["q2_K", "q3_K", "q4_K", "q5_K", "q6_K"]);
    const w = code("src/engine/kernels/wgsl.ts");
    for (const fmt of ["q2_K", "q3_K", "q4_K", "q5_K", "q6_K"]) {
      expect(w, `il wrapper ${fmt} deve delegare al nucleo`)
        .toContain(`gemvKQuantWgsl(KQUANT_GEMV_DESC.${fmt}, opts)`);
    }
  });

  it("Q2_K/Q3_K: instradati nel decode, ancora MAI ESEGUITI SU GPU", () => {
    // La cella si e' mossa di mezzo passo il 2026-08-17 (spec K-quant, T3): il
    // ramo expert di q35gpumodel ADESSO li sceglie — e la mappa lo dice — ma
    // nessun device li ha girati. La riga sotto e' quella che resta vera, ed e'
    // quella che conta: finche' non c'e' un ktest e una misura, il guadagno e'
    // un candidato.
    expect(MAPPA).toContain("MAI ESEGUITI SU GPU");
    for (const k of ["q2_K", "q3_K"] as const) {
      expect(prefillGemmWiring(k).wired, `${k} risulta cablato nel prefill: aggiorna MECCANISMI.md §3/§4`)
        .toBe(false);
    }
    // il DECODE invece li instrada: il selettore unico li ha fra i cinque, e la
    // mappa non deve piu' dire il contrario. Se qualcuno lo smontasse, la mappa
    // mentirebbe nell'altro verso — quindi si prova la presenza, non l'assenza.
    const q35 = code("src/engine/q35gpumodel.ts");
    for (const g of ["gemvQ2KWgsl", "gemvQ3KWgsl"]) {
      expect(q35, `${g} non piu' instradato: aggiorna MECCANISMI.md §4`).toContain(g);
    }
    expect(MAPPA, "il selettore unico dei gemv K-quant deve stare in §4")
      .toContain("q35KQuantGemvWgsl");
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
