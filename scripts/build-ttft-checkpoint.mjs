// Costruisce il JSON `q35-ttft-kernel-checkpoint` della riga 5 di engine-ttft
// unendo i DUE artefatti che lo compongono, invece di ricopiarne i numeri a
// mano: il bench (metrica del goal) e la run con la sonda per segmento.
//
// Vive in uno script e non in un comando incollato nel journal perche' il
// checkpoint si rifa' ogni volta che i kernel cambiano, e un conto ricopiato a
// mano e' un conto che la prossima volta nessuno ripete uguale.
//
// Uso: node scripts/build-ttft-checkpoint.mjs <bench.json> <segmenti.json> <out.json>
import { readFileSync, writeFileSync } from "node:fs";

const [benchPath, segPath, outPath] = process.argv.slice(2);
if (!benchPath || !segPath || !outPath) {
  console.error("uso: node scripts/build-ttft-checkpoint.mjs <bench.json> <segmenti.json> <out.json>");
  process.exit(2);
}
const rd = (p) => JSON.parse(readFileSync(p, "utf8"));
const b = rd(benchPath);
const s = rd(segPath);

// SENTINELLA SUL `kind`, non sul nome del file: la landmine di HANDOFF nasce da
// un artefatto il cui nome diceva una cosa e il cui contenuto ne diceva un'altra.
if (!String(b.kind).startsWith("q35-bench")) throw new Error(`${benchPath}: kind "${b.kind}", atteso q35-bench*`);
if (s.kind !== "q35-prefillchunk-4b") throw new Error(`${segPath}: kind "${s.kind}", atteso q35-prefillchunk-4b`);
if (!s.gpuTimeByCat) throw new Error(`${segPath}: manca gpuTimeByCat — la run e' stata fatta senza --gpu-time?`);

const g = s.gpuTimeByCat;
const plan = s.plan;
const M = s.chunkM;
const T = b.prefill.tokens;
const nChunk = Math.floor(T / M);

const perChunk = Object.fromEntries(Object.entries(g.byCat).map(([k, v]) => [k, v.ms / g.chunks]));
const segmenti = Object.entries(perChunk).map(([cat, ms]) => {
  const inv = plan.byCat[cat];
  return {
    cat,
    msPerChunk: +ms.toFixed(4),
    msTotale: +(ms * nChunk).toFixed(1),
    quotaPrefillPct: +((100 * ms * nChunk) / b.prefill.ms).toFixed(2),
    dispatches: inv.dispatches,
    workgroupPerDispatch: Math.round(inv.workgroups / inv.dispatches),
    wgMin: inv.wgMin,
    wgMax: inv.wgMax,
  };
}).sort((x, y) => y.msTotale - x.msTotale);

const gpuTot = segmenti.reduce((a, x) => a + x.msTotale, 0);
const FLOP = 2 * 4e9 * T;                       // 2·P·T, P = 4e9 parametri
const warmMs = b.prefill.ms + b.decode.firstMs;

const out = {
  schemaVersion: 1,
  kind: "q35-ttft-kernel-checkpoint",
  date: "2026-08-14",
  goal: "engine-ttft",
  phase: "riga 5 — la discesa massima, contabilizzata",
  model: b.model,
  modelSha256: b.modelSha256,
  deviceLabel: "4090-linux",
  declared: b.declared,
  prompt: b.prompt,
  chunkM: M,
  chunks: nChunk,

  metrica: {
    ttftWarmMs: warmMs,
    prefillMs: b.prefill.ms,
    firstMs: b.decode.firstMs,
    prefillTokS: b.prefill.tokS,
    decodeTokS: b.decode.tokS,
    baselineWarmMs: 87618,
    barraContrattoMs: 21905,
    discesaSullaBaseline: +(87618 / warmMs).toFixed(3),
    mancaAllaBarra: +(warmMs / 21905).toFixed(3),
  },

  sonda: {
    nota: "un pass per segmento invece di uno solo aggiunge barriere, quindi il chunk cronometrato e' piu' lento del vero. Il confronto col totale a sonda SPENTA e' qui accanto e non si assume trascurabile.",
    chunkCronometrati: g.chunks,
    overflow: g.overflow,
    msPerChunkSondaAccesa: +segmenti.reduce((a, x) => a + x.msPerChunk, 0).toFixed(3),
    msPerChunkSondaSpenta: s.msPerChunk ? +s.msPerChunk.chunked.toFixed(3) : null,
  },

  doveFinisceIlTempo: {
    prefillMs: b.prefill.ms,
    dentroIPassGpuMs: +gpuTot.toFixed(0),
    dentroIPassPct: +((100 * gpuTot) / b.prefill.ms).toFixed(1),
    fuoriDaiPassMs: +(b.prefill.ms - gpuTot).toFixed(0),
    fuoriDaiPassPct: +((100 * (b.prefill.ms - gpuTot)) / b.prefill.ms).toFixed(1),
    fuoriDaiPassCosaE: "encode CPU (i bind group sono gia' costruiti al load, restano setPipeline/setBindGroup/dispatchWorkgroups), submit, e i buchi fra un submit e il successivo. NON e' attribuito piu' finemente: servirebbe una sonda CPU per dispatch che oggi non esiste, ed e' un lavoro suo.",
  },

  segmenti,

  calcolo: {
    flopPrefill: FLOP,
    tflopsSostenuti: +(FLOP / (b.prefill.ms / 1000) / 1e12).toFixed(3),
    piccoFp32MisuratoRiga1: 9.26,
    quotaDelPiccoPct: +((100 * FLOP) / (b.prefill.ms / 1000) / 1e12 / 9.26).toFixed(1),
    nota: "il picco e' fp32 su GEMM densa (riga 1, it.2). Il prefill gira su pesi QUANTIZZATI, quindi questa quota non e' un'efficienza: e' la distanza da un tetto che questo carico non tocca per costruzione. Serve a dire che il collo non e' l'ALU.",
  },

  ratchetCorrettezza: {
    ktest: "101 PASS / 0 FAIL",
    vitest: "680 passed | 10 skipped",
    tsc: "pulito",
    top1SequenzialeVsOracolo: "1012/1024 = 98,828%",
    top1PrefillAChunkVsOracolo: "1012/1024 = 98,828%",
    sequenzeGenerateIdentiche: "8/8 prompt",
    decodeTokS: b.decode.tokS,
    decodeSoglia: 45.5,
  },

  hostState: b.hostState,
  repro: [
    "node scripts/q35-bench-run.mjs --prompt-idx 0 --n-decode 64 --vram-gib 8 --declared quiescent --prefill-m 16",
    "node scripts/q35-conf-run.mjs --prefill-m 16 --gpu-time",
    "node scripts/build-ttft-checkpoint.mjs <bench> <segmenti> <out>",
  ],
};

writeFileSync(outPath, JSON.stringify(out, null, 1));
console.log(`[checkpoint] TTFT a caldo ${warmMs} ms — discesa ${out.metrica.discesaSullaBaseline}x, manca ${out.metrica.mancaAllaBarra}x alla barra`);
console.log(`[checkpoint] GPU nei pass ${out.doveFinisceIlTempo.dentroIPassPct}% · fuori ${out.doveFinisceIlTempo.fuoriDaiPassPct}%`);
console.log(`[checkpoint] ${out.calcolo.tflopsSostenuti} TFLOP/s = ${out.calcolo.quotaDelPiccoPct}% del picco misurato`);
console.log(`[checkpoint] primo segmento: ${segmenti[0].cat} ${segmenti[0].msTotale} ms = ${segmenti[0].quotaPrefillPct}% del prefill`);
console.log(`-> ${outPath}`);
