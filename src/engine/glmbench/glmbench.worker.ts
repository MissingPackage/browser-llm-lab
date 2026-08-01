// Bench GLM full-model (goal C2 fase 6 slice 2, spec §8): protocollo B2 —
// warmup scartato + N repliche sullo stesso prompt, finestre dichiarate, ctx
// del corpus golden. Misura prefill (posizioni del prompt) e decode (greedy
// dal logit del motore, nessun teacher forcing: è il percorso di produzione)
// con i GATE hard del contratto: decode ≥13.43 tok/s, prefill ≥56.58 tok/s
// (floor = oracolo CPU C1, llama-bench 5f55650).
// Telemetria per fase (input C3): dispatch/token, sync router/token, hit-rate
// residenza, stallo ms/token scomposto (read OPFS / pack CPU / upload), slot
// occupati, eviction.
import { GLM47_FLASH as G, GLM47_FLASH_SHA256 } from "../shape";
import { dequantQ4_0Row } from "../quant";
import { createGlmModel } from "../glmmodel";
import { GlmOpfsSource } from "../glmsource";
import type { ExpertCacheStats } from "../residency";

interface GoldenPrompt { id: string; promptTokens: number[]; generated: number[] }
interface Golden { modelSha256: string; prompts: GoldenPrompt[] }
interface Cfg { prompt: number; nGen: number; replicates: number; budgetGiB: number }

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const progress = (msg: string) => post({ type: "progress", msg });

// Delta dei contatori cumulativi della cache (niente reset: i contatori sono
// del componente, il bench ne prende le differenze per fase).
interface PhaseResidency {
  hits: number; misses: number; evictions: number; hitRate: number | null;
  readMs: number; packMs: number; uploadMs: number; stallMs: number;
  bytesRead: number; bytesUploaded: number;
}
const residencyDelta = (a: ExpertCacheStats, b: ExpertCacheStats): PhaseResidency => {
  const hits = b.hits - a.hits, misses = b.misses - a.misses;
  const readMs = b.readMs - a.readMs, packMs = b.packMs - a.packMs, uploadMs = b.uploadMs - a.uploadMs;
  return {
    hits, misses, evictions: b.evictions - a.evictions,
    hitRate: hits + misses > 0 ? hits / (hits + misses) : null,
    readMs, packMs, uploadMs, stallMs: readMs + packMs + uploadMs,
    bytesRead: b.bytesRead - a.bytesRead, bytesUploaded: b.bytesUploaded - a.bytesUploaded,
  };
};

// Per-token del decode: wall, stallo residenza e miss del SINGOLO token —
// permette l'attribuzione senza strumentare il motore (i token a zero miss
// danno il costo puro GPU+sync, il resto la pendenza per miss).
interface TokenRec { ms: number; stallMs: number; misses: number }

interface PhaseResult {
  tokens: number; ms: number; toksPerSec: number; msPerToken: number;
  msPerTokenP50: number; residency: PhaseResidency;
}
interface RunResult { prefill: PhaseResult; decode: PhaseResult; decodeTokens: TokenRec[]; generated: number[] }

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const stats = (v: number[]) => {
  const mean = v.reduce((a, b) => a + b, 0) / Math.max(1, v.length);
  const varr = v.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, v.length - 1);
  return { median: median(v), mean, stdev: v.length > 1 ? Math.sqrt(varr) : 0, reps: v };
};

async function main(cfg: Cfg): Promise<void> {
  const t0 = performance.now();
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("niente adapter WebGPU");
  const lim = adapter.limits;
  const maxBuf = Math.min(lim.maxBufferSize, 2 * (1 << 30));
  const maxBind = Math.min(lim.maxStorageBufferBindingSize, 2 * (1 << 30));
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: maxBuf, maxStorageBufferBindingSize: maxBind,
      maxComputeWorkgroupStorageSize: Math.min(lim.maxComputeWorkgroupStorageSize, 32768),
    },
  });

  const source = await GlmOpfsSource.open("/models/GLM-4.7-Flash-Q4_0.gguf", progress);
  const golden = (await (await fetch("/models/glm-conf-golden.json")).json()) as Golden;
  if (golden.modelSha256 !== GLM47_FLASH_SHA256) throw new Error("golden: SHA GGUF diverso dal canonico");
  const pr = golden.prompts[cfg.prompt];
  if (!pr) throw new Error(`prompt ${cfg.prompt} assente nel corpus golden`);

  const promptTokens = pr.promptTokens;
  const nPrompt = promptTokens.length;
  const ctxMax = nPrompt + cfg.nGen;

  progress("carico token_embd…");
  const embd = source.nonExpert("token_embd.weight");
  const xRow = new Float32Array(G.dModel);
  const embed = (tok: number): Float32Array => {
    dequantQ4_0Row(embd, 0, G.dModel, tok, xRow);
    return xRow;
  };

  progress(`modello 47 layer + head, ctxMax ${ctxMax}, slab ${cfg.budgetGiB} GiB…`);
  const tBuild0 = performance.now();
  const model = createGlmModel(device, source, {
    ctxMax, head: true,
    cache: {
      budgetBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
      maxBindingBytes: maxBind, maxBufferBytes: maxBuf, timing: true,
    },
  });
  const buildMs = performance.now() - tBuild0;

  const argmax = (lg: Float32Array): number => {
    let a = 0;
    for (let t = 1; t < lg.length; t++) if (lg[t] > lg[a]) a = t;
    return a;
  };

  // Una replica completa: prefill (posizioni 0..nPrompt-1, logits solo
  // sull'ultima) + decode greedy di nGen token. La KV riparte da pos 0 a ogni
  // replica: l'attention legge 0..pos, le entry oltre pos sono irrilevanti.
  const runOnce = async (label: string): Promise<RunResult> => {
    const rPre0 = model.cacheStats();
    const tPre0 = performance.now();
    let last = 0;
    const preTok: number[] = [];
    for (let i = 0; i < nPrompt; i++) {
      const tTok = performance.now();
      const wantLogits = i === nPrompt - 1;
      const r = await model.forward(embed(promptTokens[i]), i, wantLogits);
      preTok.push(performance.now() - tTok);
      if (wantLogits) last = argmax(r.logits!);
      if (i % 32 === 0) {
        post({ type: "tick", msg: `${label}: prefill ${i}/${nPrompt} (${(1000 * (i + 1) / (performance.now() - tPre0)).toFixed(2)} tok/s)` });
      }
    }
    const preMs = performance.now() - tPre0;
    const rPre1 = model.cacheStats();

    const tDec0 = performance.now();
    const decTok: number[] = [];
    const decodeTokens: TokenRec[] = [];
    const generated: number[] = [];
    for (let k = 0; k < cfg.nGen; k++) {
      generated.push(last);
      const sBefore = model.cacheStats();
      const tTok = performance.now();
      const r = await model.forward(embed(last), nPrompt + k, true);
      const ms = performance.now() - tTok;
      const d = residencyDelta(sBefore, model.cacheStats());
      decTok.push(ms);
      decodeTokens.push({ ms, stallMs: d.stallMs, misses: d.misses });
      last = argmax(r.logits!);
      if (k % 8 === 0) {
        post({ type: "tick", msg: `${label}: decode ${k}/${cfg.nGen} (${(1000 * (k + 1) / (performance.now() - tDec0)).toFixed(2)} tok/s)` });
      }
    }
    const decMs = performance.now() - tDec0;
    const rDec1 = model.cacheStats();

    const phase = (tokens: number, ms: number, per: number[], res: PhaseResidency): PhaseResult => ({
      tokens, ms, toksPerSec: (tokens / ms) * 1000, msPerToken: ms / tokens,
      msPerTokenP50: median(per), residency: res,
    });
    const out: RunResult = {
      prefill: phase(nPrompt, preMs, preTok, residencyDelta(rPre0, rPre1)),
      decode: phase(cfg.nGen, decMs, decTok, residencyDelta(rPre1, rDec1)),
      decodeTokens, generated,
    };
    post({
      type: "progress",
      msg: `${label}: prefill ${out.prefill.toksPerSec.toFixed(2)} tok/s — decode ${out.decode.toksPerSec.toFixed(2)} tok/s ` +
        `(hit ${(100 * (out.decode.residency.hitRate ?? 0)).toFixed(1)}%, stallo ${(out.decode.residency.stallMs / cfg.nGen).toFixed(1)} ms/token)`,
    });
    return out;
  };

  const warmup = await runOnce("warmup (scartato)");
  const reps: RunResult[] = [];
  for (let r = 0; r < cfg.replicates; r++) reps.push(await runOnce(`replica ${r + 1}/${cfg.replicates}`));

  const decodeTps = stats(reps.map((r) => r.decode.toksPerSec));
  const prefillTps = stats(reps.map((r) => r.prefill.toksPerSec));
  const GATE_DECODE = 13.43, GATE_PREFILL = 56.58;
  const st = model.cacheStats();
  const nMoe = G.nLayer - G.denseLead;

  // aggregati di telemetria sulle repliche (mediane), input C3
  const decRes = reps.map((r) => r.decode.residency);
  const tele = {
    dispatchesPerToken: model.dispatchesPerToken,
    syncsPerToken: nMoe + 1, // 46 readback router (selezione su CPU) + 1 logits/hidden
    decode: {
      hitRate: median(decRes.map((r) => r.hitRate ?? 0)),
      missesPerToken: median(decRes.map((r) => r.misses / cfg.nGen)),
      stallMsPerToken: median(decRes.map((r) => r.stallMs / cfg.nGen)),
      readMsPerToken: median(decRes.map((r) => r.readMs / cfg.nGen)),
      packMsPerToken: median(decRes.map((r) => r.packMs / cfg.nGen)),
      uploadMsPerToken: median(decRes.map((r) => r.uploadMs / cfg.nGen)),
      msPerToken: median(reps.map((r) => r.decode.msPerToken)),
      // quota di wall non spiegata dallo stallo residenza = GPU + sync + CPU
      residuoMsPerToken: median(reps.map((r, i) => r.decode.msPerToken - decRes[i].stallMs / cfg.nGen)),
      evictionsPerToken: median(decRes.map((r) => r.evictions / cfg.nGen)),
    },
    occupancy: { occupied: st.occupied, slots: st.slots },
    // attribuzione senza strumentare il motore: i token a ZERO miss danno il
    // costo puro (1.816 dispatch + 47 sync CPU↔GPU); la pendenza per miss dà
    // il costo marginale della residenza (retta ai minimi quadrati su tutti i
    // token di decode delle repliche).
    attribution: (() => {
      const all = reps.flatMap((r) => r.decodeTokens);
      const zero = all.filter((t) => t.misses === 0).map((t) => t.ms);
      const n = all.length;
      const mx = all.reduce((a, t) => a + t.misses, 0) / n;
      const my = all.reduce((a, t) => a + t.ms, 0) / n;
      const sxy = all.reduce((a, t) => a + (t.misses - mx) * (t.ms - my), 0);
      const sxx = all.reduce((a, t) => a + (t.misses - mx) ** 2, 0);
      const slope = sxx > 0 ? sxy / sxx : null;
      const withMiss = all.filter((t) => t.misses > 0);
      return {
        tokens: n, zeroMissTokens: zero.length,
        msPerTokenZeroMiss: zero.length ? median(zero) : null,
        msPerTokenNetStall: median(all.map((t) => t.ms - t.stallMs)),
        fitInterceptMs: slope === null ? null : my - slope * mx,
        fitSlopeMsPerMiss: slope,
        missCostMsMedian: withMiss.length ? median(withMiss.map((t) => t.stallMs / t.misses)) : null,
      };
    })(),
  };

  model.destroy();
  source.close();

  post({
    type: "done",
    report: {
      kind: "glm-bench", schemaVersion: 1, date: new Date().toISOString(),
      ggufSha256: GLM47_FLASH_SHA256,
      config: {
        promptIdx: cfg.prompt, promptId: pr.id, promptTokens: nPrompt, nGen: cfg.nGen,
        replicates: cfg.replicates, budgetGiB: cfg.budgetGiB, ctxMax,
        prefillPath: "decode-only (nessun percorso batch M>1 nel motore GLM)",
        decodePath: "single-step greedy (argmax dai logits del motore)",
        rateWindow: "intera fase (prefill: nPrompt posizioni; decode: nGen token)",
        protocol: "B2: warmup scartato + N repliche, mediana come headline",
      },
      gates: {
        decodeGateToksPerSec: GATE_DECODE, decodeMedian: decodeTps.median,
        decodePass: decodeTps.median >= GATE_DECODE,
        prefillGateToksPerSec: GATE_PREFILL, prefillMedian: prefillTps.median,
        prefillPass: prefillTps.median >= GATE_PREFILL,
        floorSource: "results/engine/moe-oracle/llama-bench-glm47flash-q4_0-2026-07-30.json (llama.cpp 5f55650, CPU i9-14900HX 16 thread, n_prompt 512 / n_gen 64)",
      },
      decodeToksPerSec: decodeTps, prefillToksPerSec: prefillTps,
      telemetry: tele,
      warmup: { prefill: warmup.prefill, decode: warmup.decode },
      reps,
      buildMs, wallMs: performance.now() - t0,
      importMs: source.importMs,
    },
  });
}

self.onmessage = (ev: MessageEvent) => {
  void main(ev.data as Cfg).catch((e) => post({ type: "error", message: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) }));
};
