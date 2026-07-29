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

self.onmessage = (e: MessageEvent) => {
  const m = e.data as { type: string; modelUrl: string; goldenUrl: string; sampleEvery?: number };
  if (m.type === "conformance") {
    runConformance(m.modelUrl, m.goldenUrl, m.sampleEvery ?? 16).catch((err) =>
      post({ type: "error", message: err instanceof Error ? `${err.message}\n${err.stack}` : String(err) }),
    );
  }
};
