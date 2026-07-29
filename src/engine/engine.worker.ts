// Worker del motore — fase A. Modalità conformance: corpus in token-id, teacher
// forcing sul golden (confronto per-posizione pulito), campionamento periodico dei
// logit per il Δ sui top-32 golden (riportato, non gated).
import { createEngine, type EngineHandle } from "./gpuforward";

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
    // prefill sequenziale (riparte da pos 0: la cache si sovrascrive per posizione)
    for (let i = 0; i < p.promptTokens.length - 1; i++) await engine.forwardToken(p.promptTokens[i], pos++);
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
  const runOnce = async (label: string): Promise<{ decodeToksPerSec: number; prefillMs: number; msPerTokenDecode: number }> => {
    let pos = 0;
    const t0 = performance.now();
    for (let i = 0; i < promptFix.tokens.length - 1; i++) await engine.forwardToken(promptFix.tokens[i], pos++);
    const tPrefill = performance.now();
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
  post({
    type: "done",
    report: {
      schemaVersion: 2, kind: "engine-bench", promptId: promptFix.promptId,
      promptTokens: promptFix.tokens.length, genTokens, replicates,
      warmup, reps: repsOff, decodeToksPerSec: off, // headline = telemetria spenta
      telemetryOn: { reps: repsOn, decodeToksPerSec: on },
      telemetry, telemetryOverheadPct: overheadPct,
      dispatchesPerToken: engine.dispatchesPerToken,
      quant: "Q4_0 (gguf)", note: "confronto cross-quant con WebLLM q4f32_1 MLC: dichiarato",
    },
  });
}

// Simulazione fase B1 (per fissare la soglia prefill della spec, criterio Pareto):
// misura il floor "de-sync only" — prefill batched sul piano M=1 attuale, senza
// readback per token, a granularità di submit variabile. Il percorso vero M≤8 (GEMM,
// lm_head solo sull'ultima posizione) dovrà battere questo floor, altrimenti non
// aggiunge nulla al semplice batching dei submit.
async function runPrefillSim(modelUrl: string, promptUrl: string, replicates: number): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const promptFix = JSON.parse(new TextDecoder().decode(await fetchBuf(promptUrl))) as { promptId: string; tokens: number[] };
  const engine = await createEngine(gguf, progress, { telemetry: false });
  const prefillTokens = promptFix.tokens.slice(0, -1); // stesso perimetro del bench
  const CHECK_DECODE = 8;

  // baseline: prefill sequenziale con await per token (identico al bench fase A)
  const seqOnce = async (): Promise<{ ms: number; decoded: number[] }> => {
    let pos = 0;
    const t0 = performance.now();
    for (const t of prefillTokens) await engine.forwardToken(t, pos++);
    const ms = performance.now() - t0;
    let prev = promptFix.tokens[promptFix.tokens.length - 1];
    const decoded: number[] = [];
    for (let i = 0; i < CHECK_DECODE; i++) { prev = await engine.forwardToken(prev, pos++); decoded.push(prev); }
    return { ms, decoded };
  };
  const batchedOnce = async (tokensPerSubmit: number, skipHead = false): Promise<{ ms: number; decoded: number[] }> => {
    const t0 = performance.now();
    await engine.prefillBatched(prefillTokens, 0, tokensPerSubmit, skipHead);
    const ms = performance.now() - t0;
    let pos = prefillTokens.length;
    let prev = promptFix.tokens[promptFix.tokens.length - 1];
    const decoded: number[] = [];
    for (let i = 0; i < CHECK_DECODE; i++) { prev = await engine.forwardToken(prev, pos++); decoded.push(prev); }
    return { ms, decoded };
  };

  const variants: { label: string; run: () => Promise<{ ms: number; decoded: number[] }> }[] = [
    { label: "seq-await", run: seqOnce },
    { label: "batched-1", run: () => batchedOnce(1) },
    { label: "batched-8", run: () => batchedOnce(8) },
    { label: "batched-64", run: () => batchedOnce(64) },
    { label: `batched-all`, run: () => batchedOnce(prefillTokens.length) },
    { label: "batched-64-nohead", run: () => batchedOnce(64, true) },
  ];
  const ref = await seqOnce(); // warmup + sequenza di riferimento per il check
  const results: { label: string; prefillMs: number[]; msPerToken: number; decodeMatch: boolean }[] = [];
  for (const v of variants) {
    const times: number[] = [];
    let match = true;
    for (let r = 0; r < replicates; r++) {
      progress(`${v.label}: replica ${r + 1}/${replicates}`, 0.5);
      const out = await v.run();
      times.push(out.ms);
      match &&= out.decoded.every((id, i) => id === ref.decoded[i]);
    }
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    results.push({ label: v.label, prefillMs: times, msPerToken: mean / prefillTokens.length, decodeMatch: match });
  }
  post({
    type: "done",
    report: {
      schemaVersion: 1, kind: "engine-prefill-sim", promptId: promptFix.promptId,
      prefillTokens: prefillTokens.length, replicates, checkDecodeTokens: CHECK_DECODE,
      dispatchesPerToken: engine.dispatchesPerToken,
      note: "floor de-sync-only sul piano M=1 fuso (lm_head+argmax ancora per ogni posizione, come la baseline); il percorso M<=8 GEMM deve battere questo",
      variants: results,
    },
  });
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
  } else if (m.type === "prefillsim") {
    runPrefillSim(m.modelUrl, m.promptUrl!, m.replicates ?? 3).catch(fail);
  } else if (m.type === "tsqdiag") {
    runTsqDiag(m.modelUrl, m.promptUrl!).catch(fail);
  }
};
