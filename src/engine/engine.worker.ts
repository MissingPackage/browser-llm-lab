// Worker del motore — fase A. Modalità conformance: corpus in token-id, teacher
// forcing sul golden (confronto per-posizione pulito), campionamento periodico dei
// logit per il Δ sui top-32 golden (riportato, non gated).
import { createEngine, CTX_MAX, type EngineHandle } from "./gpuforward";
import { PREFILL_M, PREFILL_SUBMIT_TOKENS } from "./prefillplan";
import { parseGguf } from "./gguf";
import { dequantQ4_0Row } from "./quant";
import { QWEN25_05B } from "./shape";
import { encodeKv, decodeKv, kvKey, kvFileName, KvStoreOpfs, KV_LAYOUT_VERSION, type KvMeta } from "./kvstore";

interface GoldenPos { argmax: number; top: [number, number][] }
interface Golden {
  prompts: { id: string; promptTokens: number[]; positions: GoldenPos[] }[];
}

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const progress = (text: string, frac: number) => post({ type: "progress", text, frac });

async function fetchBuf(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}

async function runConformance(modelUrl: string, goldenUrl: string, sampleEvery: number, telemetryGpu = false): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const golden = JSON.parse(new TextDecoder().decode(await fetchBuf(goldenUrl))) as Golden;
  // taps=[11]: esercita il contratto tap (spec §Tap) dentro il run di conformance —
  // non cambia la matematica, aggiunge una copy per token. Check strutturale sotto.
  // telemetryGpu: opzionale (fase B1) — dopo il fix mapAsync-post-submit il liv.2
  // deve lasciare la conformance invariata anche ATTIVO.
  const engine: EngineHandle = await createEngine(gguf, progress, { taps: [11], telemetryGpu });
  post({ type: "meta", dispatchesPerToken: engine.dispatchesPerToken });
  let tapCheck: { layer: number; len: number; nonZero: boolean } | null = null;

  let agree = 0, total = 0, maxDlogit = 0;
  const perPrompt: { id: string; agree: number; total: number; mismatches: { pos: number; got: number; gold: number }[] }[] = [];
  const t0 = performance.now();
  let forwards = 0;

  for (const p of golden.prompts) {
    let pos = 0;
    engine.reset(); // contratto kvLen (fase 4): il riavvio da pos 0 è esplicito
    // prefill CHUNKED M≤8 (fase B1: la conformance gira col percorso M>1 attivo —
    // stesse posizioni confrontate, cambia solo come si riempie la KV)
    if (p.promptTokens.length > 1) {
      await engine.prefillChunked(p.promptTokens.slice(0, -1), 0);
      pos = p.promptTokens.length - 1;
    }
    let prev = p.promptTokens[p.promptTokens.length - 1];
    const row = { id: p.id, agree: 0, total: 0, got: [] as number[], mismatches: [] as { pos: number; got: number; gold: number }[] };
    for (let i = 0; i < p.positions.length; i++) {
      const got = await engine.forwardToken(prev, pos++);
      forwards = pos;
      const gold = p.positions[i];
      row.got.push(got);
      row.total++; total++;
      if (got === gold.argmax) { row.agree++; agree++; }
      else if (row.mismatches.length < 20) row.mismatches.push({ pos: i, got, gold: gold.argmax });
      if (i % sampleEvery === 0) {
        const logits = await engine.readLogits();
        for (const [tid, glogit] of gold.top) {
          maxDlogit = Math.max(maxDlogit, Math.abs(logits[tid] - glogit));
        }
      }
      prev = gold.argmax; // teacher forcing sul golden
      if (!tapCheck) {
        const tap = await engine.readTap(11);
        tapCheck = { layer: 11, len: tap.length, nonZero: tap.some((v) => v !== 0) };
      }
      if (i % 16 === 0) progress(`${p.id}: ${i}/${p.positions.length} (agree ${agree}/${total})`, total / 512);
    }
    perPrompt.push(row);
  }
  const wallMs = performance.now() - t0;
  post({
    type: "done",
    report: {
      schemaVersion: 1, kind: "engine-conformance",
      top1Pct: (agree / total) * 100, agree, total, maxDlogitSampled: maxDlogit, tapCheck,
      perPrompt, dispatchesPerToken: engine.dispatchesPerToken,
      wallMs, msPerForward: wallMs / Math.max(1, forwards * golden.prompts.length),
      // stato dei knob del run nel JSON (nota verifier fase 2: gate auto-evidenti)
      telemetryGpu,
      prefill: { path: "chunked", mMax: PREFILL_M, submitTokens: PREFILL_SUBMIT_TOKENS },
    },
  });
}

// Bench decode: protocollo del repo (warmup scartato + N repliche, stesso prompt,
// greedy self-feeding). Il decode rate è misurato dal primo all'ultimo token generato.
async function runBench(modelUrl: string, promptUrl: string, genTokens: number, replicates: number): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const promptFix = JSON.parse(new TextDecoder().decode(await fetchBuf(promptUrl))) as { promptId: string; tokens: number[] };
  const engine = await createEngine(gguf, progress);
  // fase 6: il prefill delle repliche è il percorso CHUNKED M≤8 (il gate di spec
  // §Soglie confronta prefillMs con la baseline sequenziale same-day, sotto)
  const runOnce = async (label: string): Promise<{ decodeToksPerSec: number; prefillMs: number; msPerTokenDecode: number }> => {
    engine.reset(); // contratto kvLen (fase 4)
    const t0 = performance.now();
    await engine.prefillChunked(promptFix.tokens.slice(0, -1), 0);
    const tPrefill = performance.now();
    let pos = promptFix.tokens.length - 1;
    let prev = promptFix.tokens[promptFix.tokens.length - 1];
    let tFirst = 0;
    for (let i = 0; i < genTokens; i++) {
      prev = await engine.forwardToken(prev, pos++);
      if (i === 0) tFirst = performance.now();
      if (i % 32 === 0) progress(`${label}: ${i}/${genTokens}`, i / genTokens);
    }
    const tEnd = performance.now();
    return {
      decodeToksPerSec: ((genTokens - 1) / (tEnd - tFirst)) * 1000,
      prefillMs: tPrefill - t0,
      msPerTokenDecode: (tEnd - tFirst) / (genTokens - 1),
    };
  };
  // baseline sequenziale SAME-DAY (spec §Soglie: prefillMs.mean ≤ 1/3 di questa):
  // stesso prefisso, forwardToken con await per token come il bench di fase A
  const seqPrefillOnce = async (): Promise<number> => {
    engine.reset();
    let pos = 0;
    const t0 = performance.now();
    for (let i = 0; i < promptFix.tokens.length - 1; i++) await engine.forwardToken(promptFix.tokens[i], pos++);
    return performance.now() - t0;
  };
  const stats = (rs: { decodeToksPerSec: number }[]) => {
    const rates = rs.map((r) => r.decodeToksPerSec);
    const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
    const stdev = Math.sqrt(rates.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, rates.length - 1));
    return { mean, stdev };
  };
  const warmup = await runOnce("warmup (scartato)");
  // A/B overhead telemetria (spec §Telemetria: ~zero da spenta): N repliche ON, N OFF.
  engine.setTelemetry(true);
  const repsOn = [];
  for (let r = 0; r < replicates; r++) repsOn.push(await runOnce(`replica ON ${r + 1}/${replicates}`));
  const telemetry = await engine.getTelemetry();
  engine.setTelemetry(false);
  const repsOff = [];
  for (let r = 0; r < replicates; r++) repsOff.push(await runOnce(`replica OFF ${r + 1}/${replicates}`));
  const on = stats(repsOn);
  const off = stats(repsOff);
  const overheadPct = ((1 / on.mean - 1 / off.mean) / (1 / off.mean)) * 100;
  progress("baseline seq same-day…", 0.9);
  const seqReps: number[] = [];
  for (let r = 0; r < replicates; r++) seqReps.push(await seqPrefillOnce());
  const prefillVals = repsOff.map((x) => x.prefillMs);
  const prefillMean = prefillVals.reduce((a, b) => a + b, 0) / prefillVals.length;
  const seqMean = seqReps.reduce((a, b) => a + b, 0) / seqReps.length;
  post({
    type: "done",
    report: {
      schemaVersion: 3, kind: "engine-bench", promptId: promptFix.promptId,
      promptTokens: promptFix.tokens.length, genTokens, replicates,
      warmup, reps: repsOff, decodeToksPerSec: off, // headline = telemetria spenta
      telemetryOn: { reps: repsOn, decodeToksPerSec: on },
      telemetry, telemetryOverheadPct: overheadPct,
      dispatchesPerToken: engine.dispatchesPerToken,
      // gate prefill (spec §Soglie 1, ruling Pareto): chunked vs seq same-day
      prefill: {
        path: "chunked", mMax: PREFILL_M, submitTokens: PREFILL_SUBMIT_TOKENS,
        prefillMs: { mean: prefillMean, reps: prefillVals },
        seqBaselineMs: { mean: seqMean, reps: seqReps },
        speedupVsSeq: seqMean / prefillMean,
        thresholdMs: seqMean / 3,
        gatePass: prefillMean <= seqMean / 3,
      },
      quant: "Q4_0 (gguf)", note: "confronto cross-quant con WebLLM q4f32_1 MLC: dichiarato",
    },
  });
}

// Prefix-cache OPFS (fase 5). Due modalità complementari, orchestrate dallo script
// scripts/prefix-cache.mjs con DUE page load (⇒ il restore avviene in un worker
// NUOVO, sessione fredda, come da done-when):
//  - pcsave: prefill del prefisso, save del checkpoint su OPFS, poi continuazione
//    greedy ininterrotta (il riferimento);
//  - pcrestore: lookup per chiave, decode hard-validato, writeKv, continuazione
//    greedy (deve essere token-identica) + misura re-prefill dello stesso prefisso.
const PC_CONT = 64;
const KV_DIM_SHAPE = QWEN25_05B.nKvHead * QWEN25_05B.headDim;

const sha256Hex = async (buf: ArrayBuffer): Promise<string> =>
  [...new Uint8Array(await crypto.subtle.digest("SHA-256", buf))].map((b) => b.toString(16).padStart(2, "0")).join("");

const genGreedy = async (engine: EngineHandle, fromPos: number, prev: number, n: number): Promise<number[]> => {
  const out: number[] = [];
  let pos = fromPos, p = prev;
  for (let i = 0; i < n; i++) { p = await engine.forwardToken(p, pos++); out.push(p); }
  return out;
};

async function runPcSave(modelUrl: string, promptUrl: string): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const modelSha256 = await sha256Hex(gguf);
  const promptFix = JSON.parse(new TextDecoder().decode(await fetchBuf(promptUrl))) as { promptId: string; tokens: number[] };
  const engine = await createEngine(gguf, progress, { telemetry: false });
  const prefix = promptFix.tokens.slice(0, -1);
  const t0 = performance.now();
  await engine.prefillChunked(prefix, 0);
  const prefillMs = performance.now() - t0;
  progress("save checkpoint…", 0.5);
  const payload = await engine.readKv();
  const meta: KvMeta = {
    modelSha256, layoutVersion: KV_LAYOUT_VERSION, nLayer: QWEN25_05B.nLayer,
    kvDim: KV_DIM_SHAPE, ctxMax: CTX_MAX, tokenCount: prefix.length, createdAt: Date.now(),
  };
  const bytes = encodeKv(meta, Uint32Array.from(prefix), payload, Date.now(), 0);
  const key = await kvKey(KV_LAYOUT_VERSION, modelSha256, prefix);
  const store = await KvStoreOpfs.open();
  const tS = performance.now();
  const { evicted } = await store.save(kvFileName(key), bytes);
  const saveMs = performance.now() - tS;
  progress("continuazione ininterrotta…", 0.7);
  const contTokens = await genGreedy(engine, prefix.length, promptFix.tokens[promptFix.tokens.length - 1], PC_CONT);
  post({
    type: "done",
    report: {
      schemaVersion: 1, kind: "engine-pc-save", promptId: promptFix.promptId,
      key, file: kvFileName(key), prefixTokens: prefix.length, checkpointBytes: bytes.byteLength,
      prefillMs, saveMs, evicted, contTokens,
    },
  });
}

async function runPcRestore(modelUrl: string, promptUrl: string): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const modelSha256 = await sha256Hex(gguf);
  const promptFix = JSON.parse(new TextDecoder().decode(await fetchBuf(promptUrl))) as { promptId: string; tokens: number[] };
  const engine = await createEngine(gguf, progress, { telemetry: false });
  const prefix = promptFix.tokens.slice(0, -1);
  const key = await kvKey(KV_LAYOUT_VERSION, modelSha256, prefix);
  const store = await KvStoreOpfs.open();
  progress("restore da OPFS…", 0.4);
  const t0 = performance.now();
  const buf = await store.load(kvFileName(key)); // miss ⇒ NotFoundError: il chiamante fa fallback a prefill pieno
  const tLoad = performance.now();
  const ck = decodeKv(buf, { modelSha256, layoutVersion: KV_LAYOUT_VERSION, nLayer: QWEN25_05B.nLayer, kvDim: KV_DIM_SHAPE, ctxMax: CTX_MAX });
  // lookup v1 = match esatto full-prefix: la chiave lo implica, i token lo PROVANO
  if (ck.tokens.length !== prefix.length || prefix.some((t, i) => ck.tokens[i] !== t)) throw new Error("kvcache: tokens mismatch");
  const tDecode = performance.now();
  engine.writeKv(ck.payload, ck.meta.tokenCount);
  const tWrite = performance.now();
  await engine.device.queue.onSubmittedWorkDone(); // il restore include l'upload GPU (misura onesta)
  const restoreMs = performance.now() - t0;
  const restoreBreakdown = { opfsLoadMs: tLoad - t0, decodeMs: tDecode - tLoad, writeKvMs: tWrite - tDecode, gpuSyncMs: restoreMs - (tWrite - t0) };
  progress("continuazione da restore…", 0.6);
  const contTokens = await genGreedy(engine, prefix.length, promptFix.tokens[promptFix.tokens.length - 1], PC_CONT);
  progress("re-prefill di confronto…", 0.8);
  engine.reset();
  const t1 = performance.now();
  await engine.prefillChunked(prefix, 0);
  const reprefillMs = performance.now() - t1;
  post({
    type: "done",
    report: {
      schemaVersion: 1, kind: "engine-pc-restore", promptId: promptFix.promptId,
      key, prefixTokens: prefix.length, hitCount: ck.hitCount,
      restoreMs, restoreBreakdown, reprefillMs, contTokens,
    },
  });
}

// Prova meccanica rollback KV (fase 4, done-when di PHASES): "genera N, crop a P,
// rigenera" vs run fresco dallo stesso prefisso — sequenze token IDENTICHE (greedy
// deterministico). Tre check: crop al prefisso, crop a metà generazione (pos
// arbitraria ammessa dal contratto), reset+re-prefill.
async function runRollback(modelUrl: string, promptUrl: string): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const promptFix = JSON.parse(new TextDecoder().decode(await fetchBuf(promptUrl))) as { promptId: string; tokens: number[] };
  const engine = await createEngine(gguf, progress, { telemetry: false });
  const PREFIX = 48, GEN = 64, MID = 16;
  const prefix = promptFix.tokens.slice(0, PREFIX);
  const gen = async (fromPos: number, prev: number, n: number): Promise<number[]> => {
    const out: number[] = [];
    let pos = fromPos, p = prev;
    for (let i = 0; i < n; i++) { p = await engine.forwardToken(p, pos++); out.push(p); }
    return out;
  };
  const cmp = (name: string, a: number[], b: number[]) => {
    const at = a.findIndex((v, i) => v !== b[i]);
    return { name, match: a.length === b.length && at === -1, divergeAt: at === -1 ? null : at };
  };
  // run A: prefill del prefisso (kvLen = PREFIX-1), genera GEN token greedy
  await engine.prefillChunked(prefix.slice(0, -1), 0);
  const seqA = await gen(PREFIX - 1, prefix[PREFIX - 1], GEN);
  const kvAfterGen = engine.kvLen; // atteso PREFIX-1+GEN
  // check 1: crop a P = PREFIX-1, rigenera dallo stesso token
  progress("crop al prefisso…", 0.4);
  engine.crop(PREFIX - 1);
  const seqB = await gen(PREFIX - 1, prefix[PREFIX - 1], GEN);
  // check 2: crop a metà generazione (P = PREFIX-1+MID), rigenera la coda
  progress("crop a metà generazione…", 0.6);
  engine.crop(PREFIX - 1 + MID);
  const seqTail = await gen(PREFIX - 1 + MID, seqA[MID - 1], GEN - MID);
  // check 3: run fresco — reset + re-prefill dello stesso prefisso
  progress("run fresco…", 0.8);
  engine.reset();
  await engine.prefillChunked(prefix.slice(0, -1), 0);
  const seqC = await gen(PREFIX - 1, prefix[PREFIX - 1], GEN);
  const checks = [
    cmp("crop-al-prefisso-rigenera", seqB, seqA),
    cmp("crop-a-meta-generazione", seqTail, seqA.slice(MID)),
    cmp("run-fresco-stesso-prefisso", seqC, seqA),
  ];
  const kvLenOk = kvAfterGen === PREFIX - 1 + GEN;
  post({
    type: "done",
    report: {
      schemaVersion: 1, kind: "engine-kv-rollback", promptId: promptFix.promptId,
      prefixTokens: PREFIX, genTokens: GEN, cropMid: MID,
      kvLenAfterGen: kvAfterGen, kvLenOk, checks,
      pass: kvLenOk && checks.every((c) => c.match),
    },
  });
}

// Diag parità prefill chunked (fase 3, TEMPORANEA finché il percorso M>1 non è
// stabile): confronta i logits dell'ultima posizione fra prefill sequenziale e
// chunked sugli stessi token, bisecando per lunghezza. Discrimina: L=1 ⇒ bug
// GEMM/rms/rope; L∈[2,8] ⇒ maschera causale intra-chunk / append; L≥9 ⇒ cross-chunk.
async function runPrefillDiag(modelUrl: string, promptUrl: string): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const promptFix = JSON.parse(new TextDecoder().decode(await fetchBuf(promptUrl))) as { promptId: string; tokens: number[] };
  const TAPS = [0, 11, 23]; // bisezione per stadio: post-layer 0 / metà / ultimo
  const engine = await createEngine(gguf, progress, { telemetry: false, taps: TAPS });
  // lunghezze scelte per coprire OGNI rows dell'ultimo chunk (1..8) anche in
  // regime multi-chunk: il buco di copertura di fase 6 (rows 2-7 dopo chunk pieni)
  const lens = [1, 3, 8, 9, 12, 15, 16, 20, 36, 64, 65, 68, 128];
  const maxAbsDiff = (a: Float32Array, b: Float32Array) => {
    let d = 0;
    for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
    return d;
  };
  // riproduzione ESATTA dello scenario conformance, PRIMA di ogni altro lavoro sul
  // motore (variabile: il prefill chunked come PRIMA operazione, motore freddo):
  // per ogni prompt golden, chunk PRIMA di seq, decode teacher-forced vs golden.
  const golden = JSON.parse(new TextDecoder().decode(await fetchBuf("/results/engine/golden/golden-qwen25-05b-q4_0.json"))) as
    { prompts: { id: string; promptTokens: number[]; positions: { argmax: number }[] }[] };
  const goldenCases: { id: string; len: number; agreeChunk: number; agreeChunk2: number; agreeSeq: number; total: number; chunkEqSeq: boolean; chunk2EqSeq: boolean; badRows: string[] }[] = [];
  for (const p of golden.prompts) {
    const toks = p.promptTokens;
    const DEC = Math.min(64, p.positions.length);
    const teacherDecode = async (): Promise<number[]> => {
      const ids: number[] = [];
      let pos = toks.length - 1;
      let prev = toks[toks.length - 1];
      for (let i = 0; i < DEC; i++) { ids.push(await engine.forwardToken(prev, pos++)); prev = p.positions[i].argmax; }
      return ids;
    };
    engine.reset();
    await engine.prefillChunked(toks.slice(0, -1), 0);
    const kvChunk = await engine.readKv(); // dump cache POST-prefill, PRIMA del decode
    const chunkIds = await teacherDecode();
    // secondo giro chunk IDENTICO: se questo è corretto e il primo no, la falla è
    // una race read-before-write mascherata dallo stato lasciato dal giro prima
    engine.reset();
    await engine.prefillChunked(toks.slice(0, -1), 0);
    const chunk2Ids = await teacherDecode();
    engine.reset();
    let pos = 0;
    for (let i = 0; i < toks.length - 1; i++) await engine.forwardToken(toks[i], pos++);
    const kvSeq = await engine.readKv();
    const seqIds = await teacherDecode();
    // confronto cache riga-per-riga: layout per layer [K righe | V righe], n righe
    const nRows = toks.length - 1;
    const KVD = KV_DIM_SHAPE;
    const badRows: string[] = [];
    for (let l = 0; l < QWEN25_05B.nLayer && badRows.length < 12; l++) {
      for (const half of [0, 1]) {
        for (let rr = 0; rr < nRows; rr++) {
          const off = (l * 2 + half) * nRows * KVD + rr * KVD;
          let d = 0;
          for (let i = 0; i < KVD; i++) d = Math.max(d, Math.abs(kvChunk[off + i] - kvSeq[off + i]));
          if (d > 1e-4) { badRows.push(`L${l}${half ? "V" : "K"}r${rr}:${d.toFixed(3)}`); if (badRows.length >= 12) break; }
        }
        if (badRows.length >= 12) break;
      }
    }
    goldenCases.push({
      id: p.id, len: toks.length, total: DEC,
      agreeChunk: chunkIds.filter((v, i) => v === p.positions[i].argmax).length,
      agreeChunk2: chunk2Ids.filter((v, i) => v === p.positions[i].argmax).length,
      agreeSeq: seqIds.filter((v, i) => v === p.positions[i].argmax).length,
      chunkEqSeq: chunkIds.every((v, i) => v === seqIds[i]),
      chunk2EqSeq: chunk2Ids.every((v, i) => v === seqIds[i]),
      badRows,
    });
  }
  const results: { L: number; argmaxMatch: boolean; refArgmax: number; gotArgmax: number; maxDlogit: number; tapDiff: Record<number, number> }[] = [];
  for (const L of lens) {
    progress(`diag L=${L}…`, results.length / lens.length);
    const toks = promptFix.tokens.slice(0, L);
    let pos = 0;
    let refId = -1;
    engine.reset(); // contratto kvLen (fase 4)
    for (const t of toks) refId = await engine.forwardToken(t, pos++);
    const refLogits = await engine.readLogits();
    const refTaps = new Map<number, Float32Array>();
    for (const l of TAPS) refTaps.set(l, await engine.readTap(l));
    engine.reset();
    const gotId = await engine.prefillChunked(toks, 0);
    const logits = await engine.readLogits();
    const tapDiff: Record<number, number> = {};
    for (const l of TAPS) tapDiff[l] = maxAbsDiff(await engine.readTap(l), refTaps.get(l)!);
    results.push({ L, argmaxMatch: gotId === refId, refArgmax: refId, gotArgmax: gotId, maxDlogit: maxAbsDiff(logits, refLogits), tapDiff });
  }
  post({
    type: "done",
    report: { schemaVersion: 1, kind: "engine-prefill-diag", promptId: promptFix.promptId, lens: results, goldenCases },
  });
}

// Test kernel GEMM chunk in ISOLAMENTO (fase 3, TEMPORANEO): pesi reali repackati,
// input deterministico, un dispatch, confronto vs dequant CPU — separa "kernel
// sbagliato" da "piano/cablaggio sbagliato".
async function runKernelDiag(modelUrl: string): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const f = parseGguf(gguf);
  const bytes = new Uint8Array(gguf);
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no webgpu");
  const device = await adapter.requestDevice({
    requiredLimits: { maxComputeWorkgroupStorageSize: Math.min(adapter.limits.maxComputeWorkgroupStorageSize, 32768) },
  });
  device.addEventListener("uncapturederror", (e) => {
    throw new Error(`GPU error: ${(e as GPUUncapturedErrorEvent).error.message.slice(0, 200)}`);
  });
  const M = 8;
  // LCG deterministico per l'input
  let seed = 42;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  const cases: { name: string; tensor: string; K: number; N: number }[] = [
    { name: "down(K=4864)", tensor: "blk.0.ffn_down.weight", K: 4864, N: 896 },
    { name: "o(K=896)", tensor: "blk.0.attn_output.weight", K: 896, N: 896 },
  ];
  const results: { name: string; maxErrPerRow: number[] }[] = [];
  const { gemmResidChunkFastWgsl: gen } = await import("./kernels/wgsl");
  const { repackQ4_0: rp } = await import("./quant");
  for (const cs of cases) {
    const info = f.tensors.find((t) => t.name === cs.tensor)!;
    const nBlocks = cs.K * cs.N / 32;
    const rpk = rp(bytes, f.dataOffset + info.offset, nBlocks);
    const mk = (data: Uint32Array | Float32Array, usage = GPUBufferUsage.STORAGE) => {
      const b = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, data as BufferSource);
      return b;
    };
    const xArr = new Float32Array(M * cs.K);
    for (let i = 0; i < xArr.length; i++) xArr[i] = rnd();
    const qsB = mk(rpk.qs); const scB = mk(rpk.scales); const xB = mk(xArr);
    const yB = device.createBuffer({ size: M * cs.N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const cB = mk(new Uint32Array([0, M, 0, 0]), GPUBufferUsage.UNIFORM);
    const pipe = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: gen({ K: cs.K, N: cs.N, mMax: M }) }), entryPoint: "main" },
    });
    const bind = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [qsB, scB, xB, yB, cB].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(cs.N / 4), 1);
    pass.end();
    const st = device.createBuffer({ size: M * cs.N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(yB, 0, st, 0, M * cs.N * 4);
    device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const y = new Float32Array(st.getMappedRange().slice(0));
    st.unmap();
    // riferimento CPU (y parte da zero: residual ≡ output puro)
    const wRow = new Float32Array(cs.K);
    const maxErrPerRow = new Array(M).fill(0);
    for (let r = 0; r < cs.N; r++) {
      dequantQ4_0Row(bytes, f.dataOffset + info.offset, cs.K, r, wRow);
      for (let m = 0; m < M; m++) {
        let acc = 0;
        for (let i = 0; i < cs.K; i++) acc += wRow[i] * xArr[m * cs.K + i];
        maxErrPerRow[m] = Math.max(maxErrPerRow[m], Math.abs(y[m * cs.N + r] - acc));
      }
    }
    results.push({ name: cs.name, maxErrPerRow });
    progress(`${cs.name} fatto`, 0.5);
  }
  device.destroy();
  post({ type: "done", report: { schemaVersion: 1, kind: "engine-kernel-diag", cases: results } });
}

// Diagnosi known-issue telemetria liv.2 (fase B1, timeboxed): matrice A/B a una
// variabile. Sintomi fase A: timestamp AZZERATI + corruzione del compute quando il
// liv.2 è attivo (conformance 98.05%→93.4%). Ipotesi: H1 = timestampWrites nel pass
// descriptor; H2 = resolve/copy nello stesso encoder del forward; H3 = ring lungo.
// Detector: 64 token greedy deterministici + logits finali vs baseline off.
async function runTsqDiag(modelUrl: string, promptUrl: string): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const promptFix = JSON.parse(new TextDecoder().decode(await fetchBuf(promptUrl))) as { promptId: string; tokens: number[] };
  const PREFILL = 32, DECODE = 64;
  const prompt = promptFix.tokens.slice(0, PREFILL);

  type DiagOpts = { telemetryGpu?: boolean; tsqDiag?: { ring?: number; resolveEnc?: "same" | "own" | "none" } };
  const variants: { label: string; opts: DiagOpts; hyp: string }[] = [
    { label: "off", opts: {}, hyp: "baseline" },
    { label: "on-ring64-same", opts: { telemetryGpu: true }, hyp: "comportamento fase A (sintomo atteso)" },
    { label: "on-ring1-same", opts: { telemetryGpu: true, tsqDiag: { ring: 1 } }, hyp: "H3: ring lungo" },
    { label: "on-ring1-own", opts: { telemetryGpu: true, tsqDiag: { ring: 1, resolveEnc: "own" } }, hyp: "H2: resolve nello stesso encoder" },
    { label: "on-writes-only", opts: { telemetryGpu: true, tsqDiag: { resolveEnc: "none" } }, hyp: "H1: timestampWrites da solo" },
  ];

  let refIds: number[] | null = null;
  let refLogits: Float32Array | null = null;
  const results: { label: string; hyp: string; idsMatch: boolean | null; divergeAt: number | null; maxDlogit: number | null; gpuMsPerToken: number | null; timestampNote: string }[] = [];
  for (const v of variants) {
    progress(`variante ${v.label}…`, results.length / variants.length);
    const engine = await createEngine(gguf, progress, v.opts);
    let pos = 0;
    for (let i = 0; i < prompt.length - 1; i++) await engine.forwardToken(prompt[i], pos++);
    let prev = prompt[prompt.length - 1];
    const ids: number[] = [];
    for (let i = 0; i < DECODE; i++) { prev = await engine.forwardToken(prev, pos++); ids.push(prev); }
    const logits = await engine.readLogits();
    const tel = await engine.getTelemetry();
    engine.destroy();
    if (v.label === "off") { refIds = ids; refLogits = logits; }
    const divergeAt = refIds ? ids.findIndex((id, i) => id !== refIds![i]) : null;
    let maxD: number | null = null;
    if (refLogits) { maxD = 0; for (let i = 0; i < logits.length; i++) maxD = Math.max(maxD, Math.abs(logits[i] - refLogits[i])); }
    results.push({
      label: v.label, hyp: v.hyp,
      idsMatch: refIds ? divergeAt === -1 : null,
      divergeAt: divergeAt === -1 ? null : divergeAt,
      maxDlogit: maxD, gpuMsPerToken: tel.gpuMsPerToken, timestampNote: tel.timestampNote,
    });
  }
  post({
    type: "done",
    report: {
      schemaVersion: 1, kind: "engine-tsq-diag", promptId: promptFix.promptId,
      prefillTokens: PREFILL, decodeTokens: DECODE, variants: results,
    },
  });
}

self.onmessage = (e: MessageEvent) => {
  const m = e.data as {
    type: string; modelUrl: string; goldenUrl?: string; promptUrl?: string;
    sampleEvery?: number; genTokens?: number; replicates?: number;
  };
  const fail = (err: unknown) =>
    post({ type: "error", message: err instanceof Error ? `${err.message}\n${err.stack}` : String(err) });
  if (m.type === "conformance") {
    runConformance(m.modelUrl, m.goldenUrl!, m.sampleEvery ?? 16, (m as { telemetryGpu?: boolean }).telemetryGpu ?? false).catch(fail);
  } else if (m.type === "bench") {
    runBench(m.modelUrl, m.promptUrl!, m.genTokens ?? 256, m.replicates ?? 3).catch(fail);
  } else if (m.type === "tsqdiag") {
    runTsqDiag(m.modelUrl, m.promptUrl!).catch(fail);
  } else if (m.type === "prefilldiag") {
    runPrefillDiag(m.modelUrl, m.promptUrl!).catch(fail);
  } else if (m.type === "kerneldiag") {
    runKernelDiag(m.modelUrl).catch(fail);
  } else if (m.type === "rollback") {
    runRollback(m.modelUrl, m.promptUrl!).catch(fail);
  } else if (m.type === "pcsave") {
    runPcSave(m.modelUrl, m.promptUrl!).catch(fail);
  } else if (m.type === "pcrestore") {
    runPcRestore(m.modelUrl, m.promptUrl!).catch(fail);
  }
};
