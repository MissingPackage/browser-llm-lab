// Il fixture del ktest non e' lo scratch dei runner di bench
// (goal engine-velocita-decode, it.19).
//
// COSA E' SUCCESSO. `public/models/q35/golden-full.json` aveva UN nome, DUE
// scrittori (`q35-conf-run.mjs`, `q35-bench-run.mjs`, che ci copiano il golden
// della loro run — qualunque modello, qualunque `--golden-kind`) e DUE lettori
// (`q35conf.worker.ts`, che pero' ne verifica lo SHA, e `ktest.worker.ts`, che
// non verificava niente). Una run `--model 35b --golden-kind smoke` ci lasciava
// 39 token; il caso `q35-mtp-draft-4b` misurava allora l'accept-rate della
// testa MTP su 37 confronti e lo confrontava col riferimento preso su 62.
//
// Il modo in cui si presentava e' la parte che vale: **un FAIL con un numero
// plausibile** (12/37 = 32,4% contro 50%), indistinguibile da una regressione
// del modello, in un gate di merge. E' costata mezz'ora di GPU per essere
// esclusa — due esecuzioni intere del parco kernel — prima che qualcuno
// leggesse la lunghezza della finestra.
//
// Il journal di `engine-fase-d` it.53 aveva gia' incontrato lo stesso sintomo
// («13/38 = 34,2% — un numero giusto su una finestra sbagliata») e l'aveva
// risolto puntando il file giusto, senza togliere la collisione. Questo test
// toglie la possibilita' che torni.
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

const src = (p: string) => readFileSync(new URL(`../${p}`, import.meta.url), "utf8");

/** Il fixture stabile del ktest: nessun runner deve scriverlo. */
const KTEST_FIXTURE = "/models/q35/golden-q35-4b-full.json";
/** Lo scratch per-run dei bench: il nome deve dire che e' volatile. */
const RUN_SCRATCH = "golden-run.json";

const RUNNERS = ["scripts/q35-conf-run.mjs", "scripts/q35-bench-run.mjs"];

describe("golden: il fixture del ktest e lo scratch dei bench sono file diversi", () => {
  it("il ktest legge il SUO fixture, non lo scratch delle run", () => {
    const k = src("src/engine/ktest/ktest.worker.ts");
    expect(k).toContain(`fetch("${KTEST_FIXTURE}")`);
    // le asserzioni negative guardano la FETCH e non il testo: i commenti
    // nominano il vecchio percorso apposta, ed e' giusto che lo facciano
    expect(k, "il ktest non deve leggere lo scratch dei bench")
      .not.toContain('fetch("/models/q35/golden-run.json")');
    expect(k).not.toContain('fetch("/models/q35/golden-full.json")');
  });

  for (const r of RUNNERS) {
    it(`${r} copia nello scratch e NON nel fixture del ktest`, () => {
      const s = src(r);
      expect(s).toContain(`copyFileSync(golden, join(ROOT, "public/models/q35/${RUN_SCRATCH}"))`);
      expect(s, "un runner di bench che scrive il fixture del ktest ricrea la collisione")
        .not.toContain("golden-q35-4b-full.json");
      expect(s).not.toContain('join(ROOT, "public/models/q35/golden-full.json")');
    });
  }

  it("il worker della conformance legge lo scratch, che e' cio' che il suo runner ha appena scritto", () => {
    const w = src("src/engine/q35conf/q35conf.worker.ts");
    expect(w).toContain(`fetch("/models/q35/${RUN_SCRATCH}")`);
    // la difesa che il conf worker aveva gia' e che il ktest non aveva: lo SHA
    expect(w).toContain("golden.modelSha256 !== M.sha");
  });

  it("il runner del ktest provvede il fixture da solo, dal repo", () => {
    const t = src(".harness/tools/engine-ktest.mjs");
    expect(t).toContain("results/engine/golden/q35/golden-q35-4b-full-2026-08-10.json");
    expect(t).toContain("public/models/q35/golden-q35-4b-full.json");
    expect(t).toMatch(/copyFileSync\(GOLDEN_SRC, GOLDEN_DST\)/);
  });

  it("il caso MTP verifica la finestra invece di assumerla", () => {
    const k = src("src/engine/ktest/ktest.worker.ts");
    // senza questo controllo il test confronta un accept-rate misurato su una
    // finestra qualsiasi con un riferimento preso su 64 token
    expect(k).toMatch(/if \(tokens\.length !== W\)/);
    expect(k).toContain("FINESTRA SBAGLIATA");
  });

  it("il golden 4B full nel repo copre davvero la finestra da 64 del riferimento", () => {
    const g = JSON.parse(src("results/engine/golden/q35/golden-q35-4b-full-2026-08-10.json")) as {
      prompts: { promptTokens: number[]; generated: number[] }[];
    };
    const p = g.prompts[0];
    expect(p.promptTokens.length + p.generated.length).toBeGreaterThanOrEqual(64);
  });
});
