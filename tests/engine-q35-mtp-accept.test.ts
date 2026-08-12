import { describe, it, expect } from "vitest";
import { existsSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { Q35CpuRefModel, type Q35ByteSource } from "../src/engine/q35cpurefmodel";

// ACCEPT-RATE INTRINSECO della testa MTP, misurato in f64 PRIMA di scrivere una
// riga di WGSL (fase 7, it.49).
//
// Perche' prima e non dopo: il gate dello spec-dec ("token accettati == token
// del greedy") e' insensibile alla qualita' della testa — la verifica scarta i
// draft sbagliati — quindi una testa implementata male passerebbe il gate e si
// vedrebbe come accept-rate basso, cioe' come un risultato negativo SULL'MTP
// invece che come un bug nostro. Questo numero e' la prova indipendente.
//
// E risolve un'ambiguita' che non e' documentata nei metadata: `eh_proj` e'
// [2*d, d] e concatena embedding e hidden, ma in QUALE ordine non si sa. Si
// misurano entrambi sullo stesso hidden: l'ordine giusto predice il token i+2
// molto sopra il caso, quello sbagliato no.
//
// LENTO per costruzione (forward f64 + una passata sul vocabolario da 248 320
// per posizione): gated su Q35_MTP=1, fuori dalla suite di default.
const MODEL = join(homedir(), ".cache/blab-models/q35/Qwen3.5-4B-MTP-Q4_0.gguf");
const GOLDEN = join(process.cwd(), "results/engine/golden/q35/golden-q35-4b-full-2026-08-10.json");
const run = process.env.Q35_MTP === "1" && existsSync(MODEL) && existsSync(GOLDEN);

/** Sorgente a byte su file: il GGUF sta su disco, non in un ArrayBuffer da 2,6 GB. */
function fileSource(path: string): Q35ByteSource {
  const fd = openSync(path, "r");
  return {
    size: statSync(path).size,
    slice(off: number, len: number): Uint8Array {
      const b = Buffer.alloc(len);
      readSync(fd, b, 0, len, off);
      return new Uint8Array(b);
    },
  };
}

describe.skipIf(!run)("accept-rate intrinseco della testa MTP (4B, f64)", () => {
  it("predice il token i+2 molto sopra il caso, e l'ordine di eh_proj si decide qui", () => {
    const golden = JSON.parse(readFileSync(GOLDEN, "utf8"));
    const p0 = golden.prompts[0];
    // Un campione corto: ogni posizione costa un giro sul vocabolario in f64.
    const tokens: number[] = [...p0.promptTokens, ...p0.generated].slice(0, 24);

    const m = new Q35CpuRefModel(fileSource(MODEL));
    let hidden: Float64Array[] = [];
    const { argmax } = m.forward(tokens, undefined, (h) => { hidden = h.map((v) => Float64Array.from(v)); });

    // DUE bersagli diversi, e quello che conta e' il secondo:
    //  - `corpus`: la testa indovina il token VERO del testo. E' una misura di
    //    qualita' del modello, non di accept-rate.
    //  - `model`: la testa concorda con cio' che il modello stesso produrrebbe
    //    greedy — `argmax[i+1]` e' la predizione del modello per il token i+2.
    //    QUESTO e' l'accept-rate: in spec-dec il draft si accetta se coincide
    //    col greedy del target, non col testo vero. Confonderli sottostima la
    //    testa di quanto il modello stesso sbaglia sul corpus (circa meta').
    const score = (embFirst: boolean): { corpus: number; model: number; hitM: number; tot: number } => {
      const pred = m.mtpDraftRef(tokens, embFirst, hidden);
      let hitC = 0, hitM = 0, tot = 0;
      for (let i = 0; i + 2 < tokens.length; i++) {
        tot++;
        if (pred[i] === tokens[i + 2]) hitC++;
        if (pred[i] === argmax[i + 1]) hitM++;
      }
      return { corpus: (100 * hitC) / tot, model: (100 * hitM) / tot, hitM, tot };
    };

    const embFirst = score(true);
    const hidFirst = score(false);
    // eslint-disable-next-line no-console
    console.log(`[mtp] [emb;hidden]: accept-rate vs modello ${embFirst.hitM}/${embFirst.tot} = ${embFirst.model.toFixed(1)}% ` +
      `(vs corpus ${embFirst.corpus.toFixed(1)}%)  |  [hidden;emb]: ${hidFirst.model.toFixed(1)}% (corpus ${hidFirst.corpus.toFixed(1)}%)`);

    const best = Math.max(embFirst.model, hidFirst.model);
    // Il caso e' 1/248 320 ≈ 0%. Una testa CORRETTA su testo reale sta molto
    // sopra: la soglia a 30% e' volutamente bassa — serve a distinguere
    // "funziona" da "non funziona", non a fissare un ratchet di qualita'.
    expect(best).toBeGreaterThan(30);
    // I due ordini non possono essere ugualmente buoni: se lo fossero, la
    // misura non avrebbe deciso niente e l'ordine resterebbe ignoto.
    expect(Math.abs(embFirst.model - hidFirst.model)).toBeGreaterThan(10);
  }, 30 * 60_000);
});
