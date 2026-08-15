import { describe, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { weightBytesPerRow } from "../src/engine/prefillbytes";
import { planPrefillGemm } from "../src/engine/prefillgemmplan";
import { q35PrefillBandaByCat, q35PrefillSites4B } from "../src/engine/q35prefillsites";

// I BYTE DEL CHECKPOINT VENGONO DA `prefillbytes.ts`, NON DA UNA COSTANTE.
//
// Clausola del goal engine-kquant: «I BYTE DEL SEGMENTO SI CHIEDONO AL METER,
// non si ricopiano». Fino al 2026-08-15 `scripts/build-ttft-checkpoint.mjs`
// portava due letterali (173_015_040 e 589_824_000) e due `forma` scritte a
// mano, ed ERANO GIA' SBAGLIATI: `gemm:deltanet-out` diceva `legacy` col suo
// `* M` dopo che la riga 2 del goal l'aveva portato a multi-riga (16x i byte
// veri), e `gemm:qkv` contava l'INTERA famiglia attn Q4_0 — 80 tensori — su un
// segmento che ne cronometra 24 (5x).
//
// Questo file e' il gate che rende quel ritorno rumoroso. Tre livelli, e servono
// tutti e tre:
//   (a) STRUTTURALE sul sorgente: nessun letterale di byte, nessun `* M` scritto
//       accanto a una forma. Precedente in casa: `tests/engine-prefillbytes.test.ts`
//       (d), che legge `prefillbytes.ts` come testo per tenerlo puro.
//   (b) DERIVAZIONE: i byte del modulo di inventario ricalcolati QUI con
//       `weightBytesPerRow`, cioe' con la funzione di `prefillbytes.ts` e non con
//       l'aggregatore che si sta verificando.
//   (c) END-TO-END: lo script eseguito davvero su artefatti sintetici, e il JSON
//       che produce confrontato con (b). E' l'unico livello che dimostra che il
//       CHECKPOINT — non il modulo — pubblica quei byte.
// (a) da solo passerebbe su uno script che deriva i byte sbagliati; (b) da solo
// passerebbe su uno script che il modulo non lo chiama nemmeno.

const RADICE = join(__dirname, "..");
const SCRIPT = join(RADICE, "scripts/build-ttft-checkpoint.mjs");
const M = 16;

/** M del checkpoint: PREFILL_M in produzione. Il fixture ne usa lo stesso. */
const raw = (rel: string): string => readFileSync(join(RADICE, rel), "utf8");

/**
 * Sorgente coi COMMENTI biancati (le stringhe restano). E' la vista giusta per
 * cercare NUMERI: un letterale nascosto dentro un template resta visibile,
 * mentre la testata del builder — che i vecchi letterali li NOMINA per
 * spiegarli — non conta come una ricaduta.
 */
function senzaCommenti(src: string): string {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, (m) => " ".repeat(m.length))
    .replace(/(^|[^:])\/\/[^\n]*/g, (m, p1: string) => p1 + " ".repeat(m.length - p1.length));
}

/**
 * Sorgente con commenti E stringhe biancati: la vista in cui un `* M` e' un
 * moltiplicatore e non una parola dentro una nota. Semplificata rispetto a
 * `engine-prefillpbcat-q41.test.ts` (che deve contare graffe dentro template
 * WGSL): qui basta togliere il testo.
 */
function soloCodice(src: string): string {
  let out = senzaCommenti(src);
  out = out.replace(/`(?:\\.|[^`\\])*`/g, (m) => " ".repeat(m.length));
  out = out.replace(/"(?:\\.|[^"\\])*"/g, (m) => " ".repeat(m.length));
  out = out.replace(/'(?:\\.|[^'\\])*'/g, (m) => " ".repeat(m.length));
  return out;
}

/**
 * I byte di un segmento RICALCOLATI QUI, dalla funzione di `prefillbytes.ts` e
 * dalla via del piano. Non passa da `dispatchWeightBytes` ne'
 * da `q35PrefillBandaByCat`: e' la seconda opinione, non la stessa somma.
 */
function byteAttesiPerChunk(cat: string): number {
  let tot = 0;
  for (const s of q35PrefillSites4B()) {
    if (s.cat !== cat) continue;
    const legacy = planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M, idot: true }).via === "legacy";
    tot += (legacy ? M : 1) * s.N * weightBytesPerRow(s.kind, s.K);
  }
  return tot;
}

// ---------------------------------------------------------------------------
// (a) IL SORGENTE: niente letterali, niente `* M` a mano
// ---------------------------------------------------------------------------
describe("(a) il builder del checkpoint non contiene i byte, li chiede", () => {
  const script = raw("scripts/build-ttft-checkpoint.mjs");
  const inventario = raw("src/engine/q35prefillsites.ts");

  it("la sorgente e' stata letta davvero (guard: un path sbagliato sarebbe verde)", () => {
    expect(script).toContain("q35-ttft-kernel-checkpoint");
    expect(inventario).toContain("q35PrefillBandaByCat");
    expect(script.length).toBeGreaterThan(2000);
  });

  it("nessun letterale grande nel codice: i byte non si scrivono, si contano", () => {
    // >= 7 cifre (con o senza `_`): i byte di un segmento del 4B stanno tutti
    // sopra questa soglia (il piu' piccolo, gemm:ffn-down-q41, e' 58.982.400),
    // mentre i numeri legittimi del builder — baseline 87618 ms, barra 21905 ms,
    // picco 9,26 TFLOP/s — stanno tutti sotto. `4e9`/`1e9` sono in notazione
    // esponenziale e non sono cifre ricopiate: sono la definizione di 2·P·T e la
    // conversione in GB.
    for (const [nome, src] of [["builder", script], ["inventario", inventario]] as const) {
      const grandi = [...senzaCommenti(src).matchAll(/(?<![.\w])\d[\d_]{6,}(?![\w.])/g)].map((m) => m[0]);
      expect(grandi, `letterali di byte in ${nome}`).toEqual([]);
    }
  });

  it("i due letterali della versione sbagliata non tornano, in nessuna forma", () => {
    // 173_015_040 (24 ssm_out Q5_K) e 589_824_000 (l'INTERA famiglia attn Q4_0,
    // attribuita a un segmento che ne cronometra 24).
    const codice = soloCodice(script) + soloCodice(inventario);
    for (const n of ["173015040", "173_015_040", "589824000", "589_824_000", "117964800"]) {
      expect(codice.includes(n), `letterale ${n} rimesso nel codice`).toBe(false);
    }
  });

  it("nessun `* M` scritto a mano: il moltiplicatore sta dentro la forma", () => {
    // E' la meta' del bug che i letterali da soli non spiegano: `byteTotali =
    // 173_015_040 * M * nChunk` modellava «i pesi si rileggono M volte per
    // chunk», cioe' la forma legacy, e con essa e' rimasto indietro. Ora M entra
    // solo dentro `dispatchWeightBytes` (legacy = M·N·bytesPerRow).
    const codice = soloCodice(script);
    expect([...codice.matchAll(/\*\s*M(?![A-Za-z0-9_$])/g)].map((m) => m[0]), "`* M` nel builder").toEqual([]);
  });

  it("il builder importa la derivazione, e non ha una seconda aritmetica", () => {
    const codice = soloCodice(script);
    expect(codice).toContain("q35PrefillBandaByCat");
    expect(codice).toContain("importaTs(");
    // il path del modulo e' una stringa, quindi si cerca nella vista che le
    // stringhe le tiene
    expect(senzaCommenti(script)).toContain("src/engine/q35prefillsites.ts");
    // la forma NON e' una stringa nel builder: le uniche `forma:` che scrive
    // vengono dal piano. (La vista `soloCodice` ha le stringhe biancate, quindi
    // qui un `forma: "legacy"` si vedrebbe come `forma:` seguito da spazi.)
    expect(senzaCommenti(script)).not.toMatch(/forma:\s*["'`]/);
  });

  it("l'inventario dei siti NON vive piu' in un file di test", () => {
    // era in tests/engine-prefillgemmplan.test.ts: uno script non puo'
    // importarlo, ed e' per questo che i numeri erano stati ricopiati.
    const test = raw("tests/engine-prefillgemmplan.test.ts");
    expect(test).toContain("q35PrefillSites4B");
    // le shape dei siti del 4B stanno in UN posto solo: se tornassero nel test,
    // tornerebbe anche la possibilita' di ricopiarle altrove
    expect(senzaCommenti(test).includes("attn_qkv")).toBe(false);
    expect(senzaCommenti(test).includes("ssm_out")).toBe(true);   // le ASSERZIONI su ssm_out restano
    expect(inventario).toContain("attn_qkv");
  });
});

// ---------------------------------------------------------------------------
// (b) LA DERIVAZIONE: byte da prefillbytes, forma dal piano
// ---------------------------------------------------------------------------
describe("(b) byte e forma derivati: la seconda opinione torna", () => {
  const banda = q35PrefillBandaByCat({ M, idot: true });

  it("copre i 248 siti dell'inventario, nessuno perso per strada", () => {
    expect(banda.reduce((a, x) => a + x.siti, 0)).toBe(q35PrefillSites4B().length);
    expect(q35PrefillSites4B().length).toBe(248);
  });

  it("ogni segmento: i byte per chunk sono quelli di weightBytesPerRow", () => {
    for (const x of banda) {
      expect(x.bytePerChunk, x.cat).toBe(byteAttesiPerChunk(x.cat));
      expect(x.bytePerChunk, x.cat).toBeGreaterThan(0);
    }
  });

  it("gemm:deltanet-out e' MULTIROW, e i suoi byte non hanno piu' il fattore M", () => {
    // LA REGRESSIONE CHE QUESTO FILE ESISTE PER FERMARE. Le 24 `ssm_out` Q5_K
    // sono passate alla forma multi-riga con la riga 2 di engine-kquant: il
    // checkpoint le dichiarava ancora `legacy`, e pubblicava 16x i byte.
    const d = banda.find((x) => x.cat === "gemm:deltanet-out")!;
    expect(d.forma).toBe("multirow");
    expect(d.siti).toBe(24);
    expect(d.tensori).toContain("ssm_out");
    expect(d.bytePerChunk).toBe(d.bytePerPassata);          // M NON entra
    expect(d.bytePerChunk).toBe(24 * 2560 * weightBytesPerRow("q5_K", 4096));
  });

  it("gemm:qkv conta i 24 tensori del SEGMENTO, non gli 80 della famiglia", () => {
    const q = banda.find((x) => x.cat === "gemm:qkv")!;
    expect(q.siti).toBe(24);                                 // attn_q/k/v su 8 layer full
    expect(q.bytePerPassata).toBe(
      8 * (8192 + 1024 + 1024) * weightBytesPerRow("q4_0", 2560));
    // gli altri 56 tensori attn stanno in ALTRI due segmenti, e la somma dei
    // tre e' la famiglia intera dell'header (589.824.000 B)
    const attn = ["gemm:qkv", "gemm:attn-out", "deltanet:gemm"]
      .map((c) => banda.find((x) => x.cat === c)!.bytePerPassata)
      .reduce((a, x) => a + x, 0);
    const q8 = 48 * 32 * weightBytesPerRow("q8_0", 2560);     // ssm_alpha/beta, non attn
    expect(attn - q8).toBe(589_824_000);
  });

  it("deltanet:gemm e' MISTA, e solo i suoi 48 siti legacy pagano M", () => {
    const d = banda.find((x) => x.cat === "deltanet:gemm")!;
    expect(d.forma).toBe("mista");
    expect(d.formaPerSito).toEqual({ multirow: 48, legacy: 48 });
    const legacy = 48 * 32 * weightBytesPerRow("q8_0", 2560);
    expect(d.bytePerChunk).toBe(d.bytePerPassata - legacy + M * legacy);
  });

  it("la forma non dipende dalla language feature intera", () => {
    // `idot` e `f32` sono la STESSA forma multi-riga: se i byte cambiassero col
    // device, il checkpoint pubblicherebbe la banda di chi l'ha girato.
    const senza = q35PrefillBandaByCat({ M, idot: false });
    expect(senza.map((x) => [x.cat, x.forma, x.bytePerChunk]))
      .toEqual(banda.map((x) => [x.cat, x.forma, x.bytePerChunk]));
  });

  it("ogni categoria dell'inventario e' una categoria che il motore timbra", () => {
    // ANCORAGGIO AL MOTORE: l'attribuzione sito -> segmento e' dichiarata in
    // `q35prefillsites.ts`, ma il timbro lo mette `q35gpumodel.ts` con
    // `pbCat = "..."`. Se una categoria sparisse o cambiasse nome li', qui si
    // vede — invece di vedersi in un checkpoint con un segmento non attribuito.
    const src = raw("src/engine/q35gpumodel.ts");
    const timbrate = new Set(
      [...src.matchAll(/pbCat\s*=\s*([^;]+);/g)]
        .flatMap((m) => [...m[1].matchAll(/"([^"]+)"/g)].map((q) => q[1])));
    expect(timbrate.size).toBeGreaterThan(5);
    for (const x of banda) expect([...timbrate], `categoria "${x.cat}"`).toContain(x.cat);
  });
});

// ---------------------------------------------------------------------------
// (c) END-TO-END: lo script, eseguito, su artefatti sintetici
// ---------------------------------------------------------------------------
describe("(c) il checkpoint prodotto porta i byte derivati", () => {
  const dir = mkdtempSync(join(tmpdir(), "ttft-ck-"));
  const banda = q35PrefillBandaByCat({ M, idot: true });
  const TOKENS = 1024;
  const nChunk = TOKENS / M;

  /**
   * Artefatto di segmenti SINTETICO, non uno pinnato in `results/`: i tempi qui
   * non devono essere veri, devono essere NOTI. I dispatch per categoria seguono
   * la struttura che il motore emette davvero (3 per sito multi-riga —
   * quantizzazione, moltiplicazione, combine — e 1 per sito legacy), perche' il
   * builder li usa per rifiutare un artefatto piu' vecchio del piano.
   */
  const byCat: Record<string, unknown> = {};
  const gpu: Record<string, unknown> = {};
  for (const x of banda) {
    byCat[x.cat] = {
      dispatches: 3 * x.formaPerSito.multirow + x.formaPerSito.legacy,
      workgroups: 64 * x.siti, wgMin: 32, wgMax: 640,
    };
    gpu[x.cat] = { ms: 6.4 * x.siti, passes: 64 };
  }
  byCat["norm:attn"] = { dispatches: 32, workgroups: 512, wgMin: 16, wgMax: 16 };
  gpu["norm:attn"] = { ms: 3.2, passes: 64 };

  const segPath = join(dir, "segmenti.json");
  const benchPath = join(dir, "bench.json");
  const ratchetPath = join(dir, "ratchet.json");
  const outPath = join(dir, "checkpoint.json");

  writeFileSync(segPath, JSON.stringify({
    schemaVersion: 1, kind: "q35-prefillchunk-4b", date: "2026-08-15",
    model: "Qwen3.5-4B-Q4_0.gguf", chunkM: M, chunks: nChunk, tokens: TOKENS,
    declared: "fixture", plan: { dispatches: 1, workgroupsTotal: 1, sm: 128, byCat },
    gpuTimeByCat: { chunks: nChunk, overflow: 0, byCat: gpu },
    msPerChunk: { chunked: 35.18 },
  }));
  writeFileSync(benchPath, JSON.stringify({
    schemaVersion: 1, kind: "q35-bench-4b-fullresident", date: "2026-08-15",
    model: "Qwen3.5-4B-Q4_0.gguf", modelSha256: "0".repeat(64), declared: "fixture",
    prompt: { idx: 0, tokens: TOKENS + 1 },
    prefill: { tokens: TOKENS, ms: 4000, tokS: 256 },
    decode: { n: 64, msPerTokenP50: 20, tokS: 48, firstMs: 20 },
    hostState: { declared: "fixture" },
  }));
  writeFileSync(ratchetPath, JSON.stringify({
    date: "2026-08-15", ktest: "111 PASS / 0 FAIL", vitest: "998 passed | 10 skipped",
    tsc: "pulito", top1SequenzialeVsOracolo: "1012/1024 = 98,828%",
    top1PrefillAChunkVsOracolo: "1012/1024 = 98,828%",
    sequenzeGenerateIdentiche: "8/8 prompt", decodeSoglia: 45.5,
    // Il contratto che si sta chiudendo: baseline e barra NON stanno piu' nel
    // builder (ci stavano quelle di engine-ttft, e su una run di engine-kquant
    // pubblicavano la discesa del goal sbagliato).
    contratto: {
      goal: "engine-kquant", baselineWarmMs: 32127,
      barraContrattoMs: 22500, barraNiceToHaveMs: 18000,
    },
  }));

  const esegui = (args: string[]): { code: number; err: string } => {
    try {
      execFileSync(process.execPath, [SCRIPT, ...args], { encoding: "utf8", stdio: "pipe" });
      return { code: 0, err: "" };
    } catch (e) {
      const x = e as { status?: number; stderr?: string };
      return { code: x.status ?? -1, err: String(x.stderr ?? "") };
    }
  };

  it("gira, e ogni riga di banda ha i byte di weightBytesPerRow", () => {
    const r = esegui([benchPath, segPath, outPath, "--ratchet", ratchetPath]);
    expect(r.err.slice(0, 4000), "il builder e' uscito con errore").toBe("");
    expect(r.code).toBe(0);

    const ck = JSON.parse(readFileSync(outPath, "utf8")) as {
      banda: { cat: string; forma: string; bytePerChunk: number; byteTotali: number; gbs: number; msTotale: number }[];
      bandaNonAttribuita: { cat: string }[];
      chunks: number;
    };
    expect(ck.chunks).toBe(nChunk);
    expect(ck.banda.length).toBe(banda.length);
    for (const x of ck.banda) {
      expect(x.bytePerChunk, x.cat).toBe(byteAttesiPerChunk(x.cat));
      expect(x.byteTotali, x.cat).toBe(byteAttesiPerChunk(x.cat) * nChunk);
      expect(x.gbs, x.cat).toBeCloseTo(x.byteTotali / 1e9 / (x.msTotale / 1000), 1);
    }
    // il segmento che il vecchio builder sbagliava, letto dal JSON PRODOTTO
    const dn = ck.banda.find((x) => x.cat === "gemm:deltanet-out")!;
    expect(dn.forma).toBe("multirow");
    expect(dn.bytePerChunk).toBe(173_015_040);      // NON 173_015_040 * 16
    const qkv = ck.banda.find((x) => x.cat === "gemm:qkv")!;
    expect(qkv.bytePerChunk).toBe(117_964_800);     // NON 589_824_000
    // i segmenti senza pesi di GEMM sono elencati, non taciuti
    expect(ck.bandaNonAttribuita.map((x) => x.cat)).toEqual(["norm:attn"]);
  });

  it("rifiuta un artefatto piu' VECCHIO del piano invece di pubblicarlo", () => {
    // E' il caso reale del 2026-08-14: `gemm:deltanet-out` misurato ancora
    // legacy (un dispatch per sito) e derivato multi-riga. Un checkpoint uscito
    // di qui avrebbe pubblicato i byte di un piano che quella run non aveva.
    const vecchio = join(dir, "segmenti-vecchi.json");
    const j = JSON.parse(readFileSync(segPath, "utf8")) as { plan: { byCat: Record<string, { dispatches: number }> } };
    j.plan.byCat["gemm:deltanet-out"].dispatches = 24;
    writeFileSync(vecchio, JSON.stringify(j));
    const r = esegui([benchPath, vecchio, outPath, "--ratchet", ratchetPath]);
    expect(r.code).not.toBe(0);
    expect(r.err).toContain("gemm:deltanet-out");
    expect(r.err).toContain("PRECEDENTE al piano di questo albero");
  });

  it("rifiuta una categoria che l'artefatto non conosce", () => {
    const monco = join(dir, "segmenti-monchi.json");
    const j = JSON.parse(readFileSync(segPath, "utf8")) as {
      plan: { byCat: Record<string, unknown> }; gpuTimeByCat: { byCat: Record<string, unknown> };
    };
    delete j.plan.byCat["gemm:ffn-down-q41"];
    delete j.gpuTimeByCat.byCat["gemm:ffn-down-q41"];
    writeFileSync(monco, JSON.stringify(j));
    const r = esegui([benchPath, monco, outPath, "--ratchet", ratchetPath]);
    expect(r.code).not.toBe(0);
    expect(r.err).toContain("gemm:ffn-down-q41");
  });

  it("senza --ratchet non scrive niente: il ratchet non ha default", () => {
    const r = esegui([benchPath, segPath, outPath]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("--ratchet");
  });

  it("un ratchet incompleto e' un errore, non un campo mancante nel JSON", () => {
    const rotto = join(dir, "ratchet-rotto.json");
    writeFileSync(rotto, JSON.stringify({ date: "2026-08-15", tsc: "pulito", decodeSoglia: 45.5 }));
    const r = esegui([benchPath, segPath, outPath, "--ratchet", rotto]);
    expect(r.code).not.toBe(0);
    expect(r.err).toContain("ktest");
  });

  // BASELINE E BARRA SONO DEL CONTRATTO, NON DEL BUILDER. Stavano incise nel
  // sorgente (87.618 e 21.905, i numeri di engine-ttft) e sulla prima run di
  // engine-kquant — baseline 32.127, barra 22.500 — avrebbero pubblicato una
  // discesa di 5,108x invece di 1,873x: due contratti mescolati in un campo che
  // si legge come un risultato. Questo test e' cio' che impedisce di rimetterli.
  it("senza `contratto` il checkpoint non si scrive, e la baseline non ha default", () => {
    const senza = join(dir, "ratchet-senza-contratto.json");
    const completo = JSON.parse(readFileSync(ratchetPath, "utf8")) as Record<string, unknown>;
    delete completo.contratto;
    writeFileSync(senza, JSON.stringify(completo));
    const r = esegui([benchPath, segPath, outPath, "--ratchet", senza]);
    expect(r.code).not.toBe(0);
    expect(r.err).toContain("contratto");

    // E il sorgente non deve piu' contenere i due numeri del goal precedente.
    const src = readFileSync(SCRIPT, "utf8");
    const codice = src.split("\n").filter((l) => !l.trim().startsWith("//")).join("\n");
    expect(codice, "87618 e' tornato nel codice del builder").not.toMatch(/\b87618\b/);
    expect(codice, "21905 e' tornato nel codice del builder").not.toMatch(/\b21905\b/);
  });
});
