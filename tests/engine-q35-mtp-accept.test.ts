import { describe, it, expect } from "vitest";
import { existsSync, openSync, readFileSync, readSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { Q35CpuRefModel, type Q35ByteSource, type Q35MtpProbe } from "../src/engine/q35cpurefmodel";

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
    // Ogni posizione costa un giro sul vocabolario in f64, quindi la finestra e'
    // corta per forza — ma 24 token (22 posizioni) NON discriminano: ±1 hit vale
    // ±4,5 punti, e it.51 ha visto l'ablazione dell'attenzione cambiare 17
    // predizioni su 23 lasciando il conteggio identico. 64 e' il compromesso
    // (~6 min); `Q35_MTP_WINDOW` la muove senza toccare il file.
    const WINDOW = Number(process.env.Q35_MTP_WINDOW ?? 64);
    const tokens: number[] = [...p0.promptTokens, ...p0.generated].slice(0, WINDOW);

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
    //
    // RANGO E LOG-PROB, non solo il top-1 (it.51). Il conteggio degli hit
    // quantizza: su poche decine di posizioni non distingue una testa che lavora
    // da una menomata. Il rango del bersaglio nei logits della testa (0 = top-1)
    // e la sua log-prob misurano la stessa cosa in continuo e costano zero — i
    // logits li stiamo gia' calcolando tutti per fare l'argmax.
    const score = (embFirst: boolean, dbg?: Q35MtpProbe) => {
      const ranks: number[] = [], lps: number[] = [], onTraj: boolean[] = [];
      const probe: Q35MtpProbe = {
        ...dbg,
        onLogits: (t, lg) => {
          if (t + 2 >= tokens.length) return; // l'ultima posizione non ha bersaglio
          const target = argmax[t + 1];
          let max = -Infinity;
          for (let r = 0; r < lg.length; r++) if (lg[r] > max) max = lg[r];
          let sum = 0, rank = 0;
          for (let r = 0; r < lg.length; r++) { sum += Math.exp(lg[r] - max); if (lg[r] > lg[target]) rank++; }
          ranks.push(rank);
          lps.push(lg[target] - max - Math.log(sum));
          // "on-trajectory": il greedy del modello coincide col corpus, cioe' la
          // sequenza somiglia a quella che il modello genererebbe da solo — che
          // e' l'UNICO regime in cui lo spec-dec gira davvero.
          onTraj.push(argmax[t + 1] === tokens[t + 2]);
        },
      };
      const pred = m.mtpDraftRef(tokens, embFirst, hidden, probe);
      let hitC = 0, hitM = 0, tot = 0, hitOn = 0, totOn = 0;
      for (let i = 0; i + 2 < tokens.length; i++) {
        tot++;
        if (pred[i] === tokens[i + 2]) hitC++;
        if (pred[i] === argmax[i + 1]) hitM++;
        if (argmax[i + 1] === tokens[i + 2]) { totOn++; if (pred[i] === argmax[i + 1]) hitOn++; }
      }
      const med = (a: number[]) => (a.length ? [...a].sort((x, y) => x - y)[a.length >> 1] : NaN);
      const mean = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
      return {
        corpus: (100 * hitC) / tot, model: (100 * hitM) / tot, hitM, tot, pred,
        onTraj: { pct: (100 * hitOn) / (totOn || 1), hit: hitOn, tot: totOn },
        rankMean: mean(ranks), rankMed: med(ranks), rankTop1: ranks.filter((r) => r === 0).length,
        rankTop8: ranks.filter((r) => r < 8).length, lpMean: mean(lps),
      };
    };

    // ABLAZIONE DELL'ATTENZIONE (it.51). Il blocco della testa e' l'unico pezzo
    // del modello che nessun golden copre: e' il controllo che il gate dello
    // spec-dec non da'. it.51 con finestra 24: accept-rate IDENTICO (7/22) ma 17
    // predizioni su 23 diverse e ‖attn‖ ≈ 0,97·‖h'‖ — l'attenzione lavora, il
    // conteggio non se ne accorge. Con rango e log-prob si vede il segno.
    let stats = { attnRel: 0, ffnRel: 0 };
    const embFirst = score(true, { onStats: (s) => { stats = s; } });
    const hidFirst = score(false);
    const noAttn = score(true, { ablateAttn: true });
    let diffAbl = 0;
    for (let i = 0; i < embFirst.pred.length; i++) if (embFirst.pred[i] !== noAttn.pred[i]) diffAbl++;
    const dump = (s: ReturnType<typeof score>) => ({
      accept: s.model, corpus: s.corpus, hit: `${s.hitM}/${s.tot}`, onTraj: s.onTraj,
      rankMean: s.rankMean, rankMed: s.rankMed, top1: s.rankTop1, top8: s.rankTop8, lpMean: s.lpMean,
    });
    writeFileSync("/tmp/mtp-probe.json", JSON.stringify({
      window: tokens.length, embFirst: dump(embFirst), hidFirst: dump(hidFirst),
      ablateAttn: { ...dump(noAttn), predDiverse: diffAbl, su: embFirst.pred.length },
      magnitudini: stats,
    }, null, 2));

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
