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

async function runConformance(modelUrl: string, goldenUrl: string, sampleEvery: number): Promise<void> {
  progress("fetch modello…", 0);
  const gguf = await fetchBuf(modelUrl);
  const golden = JSON.parse(new TextDecoder().decode(await fetchBuf(goldenUrl))) as Golden;
  // taps=[11]: esercita il contratto tap (spec §Tap) dentro il run di conformance —
  // non cambia la matematica, aggiunge una copy per token. Check strutturale sotto.
  const engine: EngineHandle = await createEngine(gguf, progress, { taps: [11] });
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
    const row = { id: p.id, agree: 0, total: 0, mismatches: [] as { pos: number; got: number; gold: number }[] };
    for (let i = 0; i < p.positions.length; i++) {
      const got = await engine.forwardToken(prev, pos++);
      forwards = pos;
      const gold = p.positions[i];
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
  const warmup = await runOnce("warmup (scartato)");
  const reps = [];
  for (let r = 0; r < replicates; r++) reps.push(await runOnce(`replica ${r + 1}/${replicates}`));
  const rates = reps.map((r) => r.decodeToksPerSec);
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
  const stdev = Math.sqrt(rates.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, rates.length - 1));
  post({
    type: "done",
    report: {
      schemaVersion: 1, kind: "engine-bench", promptId: promptFix.promptId,
      promptTokens: promptFix.tokens.length, genTokens, replicates,
      warmup, reps, decodeToksPerSec: { mean, stdev },
      dispatchesPerToken: engine.dispatchesPerToken,
      quant: "Q4_0 (gguf)", note: "confronto cross-quant con WebLLM q4f32_1 MLC: dichiarato",
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
    runConformance(m.modelUrl, m.goldenUrl!, m.sampleEvery ?? 16).catch(fail);
  } else if (m.type === "bench") {
    runBench(m.modelUrl, m.promptUrl!, m.genTokens ?? 256, m.replicates ?? 3).catch(fail);
  }
};
