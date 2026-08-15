// Costruisce il JSON `q35-ttft-kernel-checkpoint` della riga 5 di engine-ttft
// unendo i DUE artefatti che lo compongono, invece di ricopiarne i numeri a
// mano: il bench (metrica del goal) e la run con la sonda per segmento.
//
// Vive in uno script e non in un comando incollato nel journal perche' il
// checkpoint si rifa' ogni volta che i kernel cambiano, e un conto ricopiato a
// mano e' un conto che la prossima volta nessuno ripete uguale.
//
// I BYTE PER SEGMENTO NON SI RICOPIANO (clausola del goal engine-kquant). Fino
// al 2026-08-15 il blocco `banda` portava due letterali (173_015_040 e
// 589_824_000) e due `forma` scritte a mano, ed era gia' SBAGLIATO in due modi:
//   - `gemm:deltanet-out` diceva `forma: "legacy"` col suo `* M` — vero quando
//     fu scritto, falso da quando la riga 2 di engine-kquant ha portato le 24
//     `ssm_out` Q5_K alla forma multi-riga: un checkpoint prodotto oggi avrebbe
//     pubblicato 16 volte i byte veri;
//   - `gemm:qkv` contava 589_824_000 byte, che sono l'INTERA famiglia attn Q4_0
//     (80 tensori, cioe' anche `attn_output` e i 48 `attn_qkv`/`attn_gate` che
//     stanno in ALTRI due segmenti). Quel segmento ne cronometra 24, per
//     117_964_800 byte: i 738 GB/s pubblicati il 2026-08-14 erano 5x.
// Ora i byte li conta `dispatchWeightBytes` (src/engine/prefillbytes.ts) e la
// forma la decide `planPrefillGemm` (src/engine/prefillgemmplan.ts), applicati
// all'inventario dei siti (src/engine/q35prefillsites.ts). Il moltiplicatore M
// non e' piu' scritto da nessuna parte: e' dentro la definizione di `legacy`.
//
// Uso: node scripts/build-ttft-checkpoint.mjs <bench.json> <segmenti.json> <out.json> --ratchet <ratchet.json>
import { readFileSync, writeFileSync } from "node:fs";
import { importaTs } from "./lib/tsimport.mjs";

const argv = process.argv.slice(2);
const flag = (name) => {
  const i = argv.indexOf(name);
  return i < 0 ? null : argv[i + 1] ?? null;
};
const [benchPath, segPath, outPath] = argv.filter((a, i) => !a.startsWith("--") && !String(argv[i - 1]).startsWith("--"));
const ratchetPath = flag("--ratchet");
if (!benchPath || !segPath || !outPath) {
  console.error("uso: node scripts/build-ttft-checkpoint.mjs <bench.json> <segmenti.json> <out.json> --ratchet <ratchet.json>");
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

// ---------------------------------------------------------------------------
// BANDA EFFICACE PER SEGMENTO — la voce del done-when che chiede i GB/s.
// Byte e forma DERIVATI, mai ricopiati (v. la testata del file).
// ---------------------------------------------------------------------------
const { q35PrefillBandaByCat } = await importaTs("src/engine/q35prefillsites.ts");

// LA FORMA NON DIPENDE DALLA LANGUAGE FEATURE, e non lo si assume: `idot` e
// `f32` sono la STESSA forma multi-riga (cambia il ciclo interno, non il riuso),
// quindi i byte devono coincidere. Se un giorno divergessero, il checkpoint
// starebbe pubblicando i byte del device sbagliato e qui si ferma.
const derivato = q35PrefillBandaByCat({ M, idot: true });
for (const d of q35PrefillBandaByCat({ M, idot: false })) {
  const con = derivato.find((x) => x.cat === d.cat);
  if (!con || con.bytePerChunk !== d.bytePerChunk || con.forma !== d.forma) {
    throw new Error(`la forma di "${d.cat}" dipende da idot (${con?.forma} vs ${d.forma}): `
      + "i byte pubblicati dipenderebbero dal device che ha girato, non dal piano");
  }
}

// COERENZA FRA IL PIANO DERIVATO E QUELLO MISURATO. E' il controllo che
// mancava: il 14 agosto il checkpoint ha unito una `forma` scritta a mano con i
// millisecondi di una run che quella forma non l'aveva. La sonda registra i
// DISPATCH per segmento, e le due forme ne emettono un numero diverso — la via
// veloce ne aggiunge (quantizzazione delle attivazioni + combine dei parziali),
// la legacy e' un dispatch per sito. Il minimo che si puo' affermare senza
// ricopiare la struttura interna del motore e' quindi: >= 2 per sito
// multi-riga, >= 1 per sito legacy.
// COSA PRENDE: una categoria derivata multi-riga che nell'artefatto era ancora
// tutta legacy (esattamente il caso `gemm:deltanet-out` del 2026-08-14: 24
// dispatch su 24 siti). COSA NON PRENDE: una divergenza su pochi siti dentro
// una categoria mista. Detto, non nascosto.
const banda = [];
for (const d of derivato) {
  const inv = plan.byCat[d.cat];
  const seg = segmenti.find((x) => x.cat === d.cat);
  if (!inv || !seg) {
    throw new Error(`il segmento "${d.cat}" esiste nel piano derivato ma non nell'artefatto `
      + `${segPath}: la run e' stata prodotta da un motore che non aveva questa categoria — `
      + "piano derivato e piano misurato non sono lo stesso piano, e la banda non e' attribuibile");
  }
  const minDispatch = 2 * d.formaPerSito.multirow + d.formaPerSito.legacy;
  if (inv.dispatches < minDispatch) {
    throw new Error(`"${d.cat}": l'artefatto ha ${inv.dispatches} dispatch su ${d.siti} siti, `
      + `ma il piano derivato ne vuole almeno ${minDispatch} (${d.formaPerSito.multirow} multi-riga, `
      + `${d.formaPerSito.legacy} legacy). La run e' PRECEDENTE al piano di questo albero: `
      + "pubblicare i byte di oggi sui millisecondi di ieri e' il modo di sbagliare che questo controllo esiste per fermare");
  }
  const byteTotali = d.bytePerChunk * nChunk;
  banda.push({
    cat: d.cat,
    forma: d.forma,
    formaPerSito: d.formaPerSito,
    tensori: d.tensori,
    bytePerPassata: d.bytePerPassata,
    bytePerChunk: d.bytePerChunk,
    byteTotali,
    dispatchesMisurati: inv.dispatches,
    msTotale: seg.msTotale,
    gbs: +(byteTotali / 1e9 / (seg.msTotale / 1000)).toFixed(1),
  });
}
banda.sort((x, y) => y.msTotale - x.msTotale);

// I segmenti che restano fuori, con la ragione: un elenco vuoto direbbe che
// tutto e' attribuito, e non e' cosi'.
const bandaNonAttribuita = Object.keys(plan.byCat)
  .filter((cat) => !derivato.some((d) => d.cat === cat))
  .map((cat) => ({ cat, perche: "nessun peso di GEMM in questo segmento (norm, attivazioni, ricorrenza, attenzione): i byte che attraversa non sono pesi e non stanno nell'inventario dei siti" }));

// ---------------------------------------------------------------------------
// RATCHET DI CORRETTEZZA — dei numeri, non delle stringhe di tre iterazioni fa.
//
// Prima stava scritto qui dentro: «101 PASS / 0 FAIL», «680 passed | 10
// skipped». Il 2026-08-15 i valori veri erano 111 PASS e 998 passed: la stessa
// malattia dei byte ricopiati, con l'aggravante che questi NON sono derivabili —
// il ktest vuole una GPU e vitest vuole essere eseguito. Non esiste un artefatto
// da leggere (il driver ktest non scrive JSON, vitest non ha un reporter su
// file configurato in questo repo), quindi la scelta non e' fra "derivare" e
// "ricopiare": e' fra ricopiare DENTRO IL CODICE, dove il valore sopravvive a
// chi l'ha misurato, e ricopiare in un FILE DATATO passato alla run, che
// invecchia in modo visibile.
//
// Forma scelta: `--ratchet <file.json>` OBBLIGATORIO, senza default. Il campo
// non ha piu' un valore di comodo che compare da solo in un artefatto: o si
// dichiara con la sua data, o il checkpoint non si scrive. E la data si
// confronta con quella della run: un ratchet piu' vecchio del bench che
// certifica esce nel JSON marcato tale, invece di passare per fresco.
const RATCHET_CAMPI = [
  "ktest", "vitest", "tsc",
  "top1SequenzialeVsOracolo", "top1PrefillAChunkVsOracolo", "sequenzeGenerateIdentiche",
];
if (!ratchetPath) {
  console.error("manca --ratchet <file.json>. Il ratchet di correttezza non ha default: i valori\n"
    + "vanno dalla sessione di gate che li ha misurati. Formato:\n"
    + JSON.stringify({
      date: "AAAA-MM-GG",
      ktest: "111 PASS / 0 FAIL",
      vitest: "998 passed | 10 skipped",
      tsc: "pulito",
      top1SequenzialeVsOracolo: "1012/1024 = 98,828%",
      top1PrefillAChunkVsOracolo: "1012/1024 = 98,828%",
      sequenzeGenerateIdentiche: "8/8 prompt",
      decodeSoglia: 45.5,
    }, null, 1));
  process.exit(2);
}
const r = rd(ratchetPath);
for (const k of [...RATCHET_CAMPI, "date"]) {
  if (typeof r[k] !== "string" || r[k].trim() === "") {
    throw new Error(`${ratchetPath}: campo "${k}" mancante o vuoto — il ratchet non ha default`);
  }
}
if (typeof r.decodeSoglia !== "number") throw new Error(`${ratchetPath}: "decodeSoglia" deve essere un numero`);
// IL CONTRATTO CHE SI STA CHIUDENDO, obbligatorio e senza default: baseline e
// barra sono la cosa che cambia da un goal al successivo, ed erano incise nel
// builder. Un checkpoint che le eredita da chi l'ha scritto pubblica la discesa
// di un altro goal.
if (!r.contratto || typeof r.contratto !== "object") {
  throw new Error(`${ratchetPath}: manca "contratto" — servono { goal, baselineWarmMs, barraContrattoMs } `
    + "(e barraNiceToHaveMs se il contratto ne ha una). Senza, la discesa sarebbe misurata su una baseline arbitraria");
}
for (const k of ["baselineWarmMs", "barraContrattoMs"]) {
  if (typeof r.contratto[k] !== "number") throw new Error(`${ratchetPath}: "contratto.${k}" deve essere un numero`);
}
if (typeof r.contratto.goal !== "string" || !r.contratto.goal.trim()) {
  throw new Error(`${ratchetPath}: "contratto.goal" mancante — il checkpoint deve dire QUALE contratto sta chiudendo`);
}
const benchDate = b.date ?? s.date ?? null;
const ratchetStantio = benchDate !== null && r.date < benchDate;
if (ratchetStantio) {
  console.warn(`[checkpoint] ATTENZIONE: ratchet del ${r.date}, run del ${benchDate}: `
    + "certifica un albero precedente alla misura. Il flag finisce nel JSON.");
}

const out = {
  schemaVersion: 2,
  kind: "q35-ttft-kernel-checkpoint",
  // LA DATA E' QUELLA DELLA RUN, non una stringa nel builder: un checkpoint
  // rifatto oggi con la data di ieri e' la stessa malattia dei byte ricopiati.
  date: b.date ?? s.date ?? new Date().toISOString().slice(0, 10),
  // IL GOAL LO DICE IL RATCHET, non una stringa nel builder. Fino al 2026-08-15
  // qui c'era `"engine-ttft"` inciso, mentre `metrica.goal` (piu' sotto) leggeva
  // gia' il contratto vero: lo stesso file si auto-attribuiva a DUE goal diversi.
  // E' il residuo esatto della malattia curata nel blocco accanto — la baseline
  // e la barra di engine-ttft incise nel builder, che avrebbero pubblicato una
  // discesa di 5,108x mescolando due contratti.
  goal: r.contratto.goal,
  phase: r.contratto.fase ?? "riga di chiusura — la discesa, contabilizzata",
  model: b.model,
  modelSha256: b.modelSha256,
  deviceLabel: "4090-linux",
  declared: b.declared,
  prompt: b.prompt,
  chunkM: M,
  chunks: nChunk,

  metrica: {
    ttftWarmMs: warmMs,
    // SCOMPOSTO ACCANTO ALL'AGGREGATO, che e' una clausola del done-when e non
    // un vezzo: `ttftAggregatoMs` include il LOAD, `ttftWarmMs` no. Sono due
    // numeri diversi (26.601 contro 17.153 su questa run) e pubblicarne uno
    // solo e' il modo in cui un TTFT "a caldo" diventa un TTFT a freddo senza
    // che nessuno se ne accorga.
    ttftAggregatoMs: b.ttftMs ?? null,
    loadMs: b.loadMs ?? null,
    prefillMs: b.prefill.ms,
    firstMs: b.decode.firstMs,
    prefillTokS: b.prefill.tokS,
    decodeTokS: b.decode.tokS,
    // LA BASELINE E LA BARRA VENGONO DAL CONTRATTO CHE SI STA CHIUDENDO, e non
    // sono scritte qui. Stavano incise nel builder — 87.618 e 21.905, cioe' i
    // numeri del goal engine-ttft — e il primo checkpoint di engine-kquant, che
    // ha una baseline diversa (32.127) e una barra diversa (22.500), avrebbe
    // pubblicato una discesa di 5,108x mescolando due contratti. E' la stessa
    // malattia dei byte ricopiati, un blocco piu' in la'.
    goal: r.contratto.goal,
    baselineWarmMs: r.contratto.baselineWarmMs,
    barraContrattoMs: r.contratto.barraContrattoMs,
    barraNiceToHaveMs: r.contratto.barraNiceToHaveMs ?? null,
    discesaSullaBaseline: +(r.contratto.baselineWarmMs / warmMs).toFixed(3),
    // < 1 vuol dire SOTTO la barra. Il nome diceva "manca" anche quando la
    // barra era gia' passata: qui il verdetto e' esplicito e non si legge
    // all'incontrario.
    quotaDellaBarra: +(warmMs / r.contratto.barraContrattoMs).toFixed(3),
    barraPassata: warmMs < r.contratto.barraContrattoMs,
    niceToHavePassato: r.contratto.barraNiceToHaveMs != null
      ? warmMs < r.contratto.barraNiceToHaveMs : null,
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

  bandaNota: "BYTE E FORMA DERIVATI, non ricopiati: i byte li conta dispatchWeightBytes (src/engine/prefillbytes.ts), la forma la decide planPrefillGemm (src/engine/prefillgemmplan.ts), l'inventario dei 248 siti e' src/engine/q35prefillsites.ts. Il moltiplicatore M vive solo dentro la definizione di `legacy` (M·N·bytesPerRow contro N·bytesPerRow): non c'e' un `* M` scritto accanto a una forma scritta. `bytePerPassata` e' il peso dei tensori del segmento (M non entra), `bytePerChunk` e' cio' che il segmento fa attraversare la memoria in un chunk. ATTENZIONE ALLA LETTURA: un GB/s sopra la banda DRAM del device non e' traffico verso la memoria — e' la cache che serve i pesi fra un chunk e il successivo; e un GB/s molto basso su una forma multi-riga dice che quel segmento non e' limitato dai byte dei pesi, non che la memoria vada piano. Schema 2 (era 1): nello schema 1 `banda` copriva due categorie con byte ricopiati, e quelli di `gemm:qkv` erano l'intera famiglia attn Q4_0 invece dei 24 tensori del segmento.",
  banda,
  bandaNonAttribuita,

  calcolo: {
    flopPrefill: FLOP,
    tflopsSostenuti: +(FLOP / (b.prefill.ms / 1000) / 1e12).toFixed(3),
    piccoFp32MisuratoRiga1: 9.26,
    quotaDelPiccoPct: +((100 * FLOP) / (b.prefill.ms / 1000) / 1e12 / 9.26).toFixed(1),
    nota: "il picco e' fp32 su GEMM densa (riga 1, it.2). Il prefill gira su pesi QUANTIZZATI, quindi questa quota non e' un'efficienza: e' la distanza da un tetto che questo carico non tocca per costruzione. Serve a dire che il collo non e' l'ALU.",
  },

  ratchetCorrettezza: {
    ...Object.fromEntries(RATCHET_CAMPI.map((k) => [k, r[k]])),
    // decodeTokS non si dichiara: e' misurato, e sta nel bench di questa run
    decodeTokS: b.decode.tokS,
    decodeSoglia: r.decodeSoglia,
    fonte: ratchetPath,
    dataRatchet: r.date,
    dataRun: benchDate,
    ratchetPiuVecchioDellaRun: ratchetStantio,
  },

  hostState: b.hostState,
  repro: [
    "node scripts/q35-bench-run.mjs --prompt-idx 0 --n-decode 64 --vram-gib 8 --declared quiescent --prefill-m 16",
    "node scripts/q35-conf-run.mjs --prefill-m 16 --gpu-time",
    "node scripts/build-ttft-checkpoint.mjs <bench> <segmenti> <out> --ratchet <ratchet.json>",
  ],
};

writeFileSync(outPath, JSON.stringify(out, null, 1));
console.log(`[checkpoint] ${out.metrica.goal}: TTFT a caldo ${warmMs} ms — ${out.metrica.discesaSullaBaseline}x sulla baseline ${out.metrica.baselineWarmMs}`);
console.log(`[checkpoint] barra ${out.metrica.barraContrattoMs}: ${out.metrica.barraPassata ? "PASSATA" : "NON passata"}`
  + (out.metrica.barraNiceToHaveMs != null
    ? ` · nice-to-have ${out.metrica.barraNiceToHaveMs}: ${out.metrica.niceToHavePassato ? "PASSATO" : "NON passato"}` : ""));
console.log(`[checkpoint] GPU nei pass ${out.doveFinisceIlTempo.dentroIPassPct}% · fuori ${out.doveFinisceIlTempo.fuoriDaiPassPct}%`);
console.log(`[checkpoint] ${out.calcolo.tflopsSostenuti} TFLOP/s = ${out.calcolo.quotaDelPiccoPct}% del picco misurato`);
console.log(`[checkpoint] primo segmento: ${segmenti[0].cat} ${segmenti[0].msTotale} ms = ${segmenti[0].quotaPrefillPct}% del prefill`);
for (const x of banda) {
  console.log(`[banda] ${x.cat.padEnd(20)} ${x.forma.padEnd(9)} ${x.tensori} — `
    + `${(x.byteTotali / 1e9).toFixed(1)} GB / ${x.msTotale} ms = ${x.gbs} GB/s`);
}
console.log(`[banda] non attribuiti (nessun peso di GEMM): ${bandaNonAttribuita.map((x) => x.cat).join(", ")}`);
console.log(`-> ${outPath}`);
