// Conformance logits full-model GLM (goal C2 fase 6, spec §7 emendata):
// replay teacher-forced dei golden di fase 1 (prompt + 128 greedy dell'oracolo)
// nel motore con output head; per ogni posizione golden: top-1 vs golden
// (gate ii ≥97%), KL media sui top-32 e max|Δ| logit (secondarie), e dump di
// argmax+top-32 del motore per il gate (i) vs cpuref-f64 (assemblato dal
// runner con l'output dell'analisi node).
import { GLM47_FLASH as G, GLM47_FLASH_SHA256 } from "../shape";
import { dequantQ4_0Row } from "../quant";
import { createGlmModel } from "../glmmodel";
import { GlmOpfsSource } from "../glmsource";
import { negotiateLimits, slabBufferCap, grantedLimits } from "../gpulimits";
import { arenaNeeds } from "../residency";

interface GoldenPos { argmax: number; top: Array<[number, number]> }
interface GoldenPrompt { id: string; promptTokens: number[]; generated: number[]; positions: GoldenPos[] }
interface Golden {
  modelSha256: string; oracle: { commit: string }; corpusHash: string; topK: number;
  prompts: GoldenPrompt[];
}
interface Cfg { prompts?: number[]; maxGen?: number; budgetGiB: number }

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const progress = (msg: string) => post({ type: "progress", msg });

async function main(cfg: Cfg): Promise<void> {
  const t0 = performance.now();
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("niente adapter WebGPU");
  const source = await GlmOpfsSource.open("/models/GLM-4.7-Flash-Q4_0.gguf", progress);
  const golden = (await (await fetch("/models/glm-conf-golden.json")).json()) as Golden;
  if (golden.modelSha256 !== GLM47_FLASH_SHA256) throw new Error("golden: SHA GGUF diverso dal canonico");

  const prompts = golden.prompts.filter((_, i) => !cfg.prompts || cfg.prompts.includes(i));
  const maxGen = cfg.maxGen ?? Infinity;
  const ctxMax = Math.max(...prompts.map((p) => p.promptTokens.length + Math.min(p.generated.length, maxGen)));

  // Device creato DOPO ctxMax: i limiti sono DERIVATI dai consumatori (C3a it.6)
  const device = await adapter.requestDevice({
    requiredLimits: negotiateLimits(adapter, {
      ctxMax, head: { vocab: G.vocab, dModel: G.dModel },
      slabClassBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
      // arena expert (C3a fase 4 strato 1): binding d'arena e finestra
      ...arenaNeeds({
        budgetBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
        maxBufferBytes: adapter.limits.maxBufferSize,
        maxBindingBytes: adapter.limits.maxStorageBufferBindingSize,
      }),
    }),
  });
  const { maxBindingBytes: maxBind, maxBufferBytes: maxBuf } = slabBufferCap(device);

  progress("carico token_embd…");
  const embd = source.nonExpert("token_embd.weight");
  const xRow = new Float32Array(G.dModel);
  progress(`modello 47 layer + head, ctxMax ${ctxMax}…`);
  const model = createGlmModel(device, source, {
    ctxMax, head: true,
    cache: { budgetBytes: Math.floor(cfg.budgetGiB * (1 << 30)), maxBindingBytes: maxBind, maxBufferBytes: maxBuf, timing: true },
  });

  let top1Ok = 0, top1Tot = 0, klSum = 0, maxDl = 0;
  const perPrompt: Array<{ id: string; golden: number; top1: number; klMean: number; maxDl: number }> = [];
  const positions: Array<{ p: number; k: number; gold: number; got: number; kl: number; dl: number; gotTop: Array<[number, number]> }> = [];
  const tReplay0 = performance.now();
  let doneRows = 0, totalRows = prompts.reduce((s, p) => s + p.promptTokens.length + Math.min(p.generated.length, maxGen) - 1, 0);

  for (const [pi, pr] of prompts.entries()) {
    const nGen = Math.min(pr.generated.length, maxGen);
    const seq = [...pr.promptTokens, ...pr.generated.slice(0, Math.max(0, nGen - 1))];
    const n = pr.promptTokens.length;
    let pTop1 = 0, pKl = 0, pMaxDl = 0;
    for (let i = 0; i < seq.length; i++) {
      dequantQ4_0Row(embd, 0, G.dModel, seq[i], xRow);
      const wantLogits = i >= n - 1;
      const r = await model.forward(xRow, i, wantLogits);
      doneRows++;
      if (wantLogits) {
        const k = i - (n - 1);
        const gp = pr.positions[k];
        const lg = r.logits!;
        let arg = 0;
        for (let t = 1; t < lg.length; t++) if (lg[t] > lg[arg]) arg = t;
        top1Tot++;
        if (arg === gp.argmax) { top1Ok++; pTop1++; }
        // KL sui top-32 del golden (softmax ristrette agli stessi 32 id) + max|Δ| logit
        const ids = gp.top.map((t) => t[0]);
        const gl = gp.top.map((t) => t[1]);
        const ol = ids.map((id) => lg[id]);
        const sm = (v: number[]): number[] => {
          const m = Math.max(...v);
          const e = v.map((x) => Math.exp(x - m));
          const s = e.reduce((a, b) => a + b, 0);
          return e.map((x) => x / s);
        };
        const pg = sm(gl), po = sm(ol);
        let kl = 0;
        for (let j = 0; j < ids.length; j++) kl += pg[j] * Math.log(pg[j] / Math.max(po[j], 1e-12));
        let dl = 0;
        for (let j = 0; j < ids.length; j++) dl = Math.max(dl, Math.abs(ol[j] - gl[j]));
        klSum += kl; pKl += kl;
        maxDl = Math.max(maxDl, dl); pMaxDl = Math.max(pMaxDl, dl);
        // top-32 del motore per il gate (i) vs cpuref (dump)
        const idx = Array.from(lg.keys());
        idx.sort((a, b) => lg[b] - lg[a]);
        positions.push({
          p: pi, k, gold: gp.argmax, got: arg, kl, dl,
          gotTop: idx.slice(0, golden.topK).map((t) => [t, lg[t]] as [number, number]),
        });
      }
      if (doneRows % 25 === 0) {
        const rate = doneRows / ((performance.now() - tReplay0) / 1000);
        const st = model.cacheStats();
        post({
          type: "tick",
          msg: `${doneRows}/${totalRows} pos (${rate.toFixed(1)}/s, prompt ${pi}) — top1 ${top1Tot ? (100 * top1Ok / top1Tot).toFixed(2) : "—"}% ` +
            `su ${top1Tot} — hit ${(100 * st.hits / Math.max(1, st.hits + st.misses)).toFixed(1)}%`,
        });
      }
    }
    perPrompt.push({ id: pr.id, golden: nGen, top1: pTop1 / Math.max(1, nGen), klMean: pKl / Math.max(1, nGen), maxDl: pMaxDl });
  }

  const st = model.cacheStats();
  model.destroy();
  source.close();
  const top1Pct = top1Tot ? 100 * top1Ok / top1Tot : null;
  post({
    type: "done",
    report: {
      kind: "glm-logits-conformance", schemaVersion: 1, date: new Date().toISOString(),
      ggufSha256: GLM47_FLASH_SHA256, oracle: golden.oracle, corpusHash: golden.corpusHash,
      config: { prompts: cfg.prompts ?? null, maxGen: cfg.maxGen ?? null, budgetGiB: cfg.budgetGiB, ctxMax },
      gateGolden: { threshold: 97, top1Ok, top1Tot, pct: top1Pct, pass: top1Tot > 0 && top1Pct! >= 97 },
      secondary: { klMeanTop32: top1Tot ? klSum / top1Tot : null, maxAbsDeltaLogit: maxDl },
      perPrompt, positions,
      residency: { ...st, hitRate: st.hits + st.misses > 0 ? st.hits / (st.hits + st.misses) : null, importMs: source.importMs },
      dispatchesPerTokenPlanned: model.dispatchesPerTokenPlanned, // DERIVATO dal piano, non contato (C3a it.3)
    deviceLimits: grantedLimits(device), // limiti CONCESSI: senza, un confronto fra run non e' falsificabile
      wallMs: performance.now() - t0, replayMs: performance.now() - tReplay0,
    },
  });
}

self.onmessage = (ev: MessageEvent) => {
  void main(ev.data as Cfg).catch((e) => post({ type: "error", message: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) }));
};
