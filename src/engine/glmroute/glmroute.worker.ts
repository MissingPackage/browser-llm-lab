// Conformance routing GLM (goal C2 fase 5 slice 3b, spec §7): replay
// teacher-forced dei prompt della traccia C1 nel motore (createGlmModel a 47
// layer, pesi reali da OPFS) e set-match dei top-4 per (posizione, layer)
// contro la traccia oracolo. GATE: match ≥99% sulle posizioni DECODE — sotto
// soglia ci si ferma e si debugga il router (niente fase 6).
//
// Il replay usa i token della traccia (non i propri argmax): la conformance
// dei logits è il gate di fase 6; qui si misura il routing a parità di input.
import { GLM47_FLASH as G, GLM47_FLASH_SHA256 } from "../shape";
import { dequantQ4_0Row } from "../quant";
import { createGlmModel } from "../glmmodel";
import { GlmOpfsSource } from "../glmsource";
import { negotiateLimits, slabBufferCap, grantedLimits } from "../gpulimits";

interface TraceRow { p: number; i: number; tok: number; ph: "p" | "d"; e: number[] }
interface TraceFile {
  header: { ggufSha256: string; llamaCppCommit: string; corpusHash: string; nMoe: number };
  rows: TraceRow[];
}
interface Cfg { cap?: number; prompts?: number[]; budgetGiB: number }

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const progress = (msg: string) => post({ type: "progress", msg });

interface LayerCounter { total: number; match: number }

async function main(cfg: Cfg): Promise<void> {
  const t0 = performance.now();
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("niente adapter WebGPU");

  const source = await GlmOpfsSource.open("/models/GLM-4.7-Flash-Q4_0.gguf", progress);
  const trace = (await (await fetch("/models/glm-route-trace.json")).json()) as TraceFile;
  if (trace.header.ggufSha256 !== GLM47_FLASH_SHA256) throw new Error("trace: SHA GGUF diverso dal canonico");

  let rows = trace.rows;
  if (cfg.prompts) rows = rows.filter((r) => cfg.prompts!.includes(r.p));
  const byPrompt = new Map<number, TraceRow[]>();
  for (const r of rows) {
    let a = byPrompt.get(r.p);
    if (!a) { a = []; byPrompt.set(r.p, a); }
    a.push(r);
  }
  for (const a of byPrompt.values()) a.sort((x, y) => x.i - y.i);
  if (cfg.cap) for (const [p, a] of byPrompt) byPrompt.set(p, a.slice(0, cfg.cap));
  const totalRows = [...byPrompt.values()].reduce((s, a) => s + a.length, 0);
  const ctxMax = Math.max(...[...byPrompt.values()].map((a) => a.length));

  // Device creato DOPO ctxMax: mlaAttnDecode tiene scores[ctxMax] in workgroup
  // memory (4*ctxMax+256 B) e il corpus arriva a 6688 pos — il requisito si
  // DERIVA dal contesto invece di chiedere il massimo (C3a it.6).
  const device = await adapter.requestDevice({
    requiredLimits: negotiateLimits(adapter, {
      ctxMax, head: { vocab: G.vocab, dModel: G.dModel },
      slabClassBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
    }),
  });
  const { maxBindingBytes: maxBind, maxBufferBytes: maxBuf } = slabBufferCap(device);
  progress(`adapter ${adapter.info?.vendor ?? "?"} — maxBuffer ${(maxBuf / 2 ** 30).toFixed(1)} GiB`);

  // embedding: l'intero token_embd in RAM (178 MB), dequant Q4_0 per riga
  progress("carico token_embd…");
  const embd = source.nonExpert("token_embd.weight");
  const xRow = new Float32Array(G.dModel);

  progress(`costruisco il modello (47 layer, ctxMax ${ctxMax})…`);
  const model = createGlmModel(device, source, {
    ctxMax,
    cache: {
      budgetBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
      maxBindingBytes: maxBind, maxBufferBytes: maxBuf, timing: true,
    },
  });
  const slots = model.cacheStats().slots;
  progress(`cache expert: ${slots.q4_0}+${slots.q4_1} slot (${(((slots.q4_0 * 5308416 + slots.q4_1 * 5505024) / 2 ** 30)).toFixed(1)} GiB)`);

  const perLayerDecode: LayerCounter[] = Array.from({ length: G.nLayer }, () => ({ total: 0, match: 0 }));
  const phase = { p: { total: 0, match: 0 }, d: { total: 0, match: 0 } };
  const samples: Array<{ p: number; i: number; layer: number; got: number[]; want: number[] }> = [];
  let doneRows = 0;
  const tReplay0 = performance.now();

  for (const [p, a] of byPrompt) {
    for (const row of a) {
      dequantQ4_0Row(embd, 0, G.dModel, row.tok, xRow);
      const { routing } = await model.forward(xRow, row.i);
      if (routing.length !== G.nLayer - G.denseLead) throw new Error(`routing ${routing.length} layer`);
      for (const r of routing) {
        const moeIdx = r.layer - G.denseLead;
        const want = row.e.slice(moeIdx * 4, moeIdx * 4 + 4);
        const got = Array.from(r.experts);
        const ok = got.length === 4 && want.every((e) => got.includes(e));
        phase[row.ph].total++;
        if (ok) phase[row.ph].match++;
        if (row.ph === "d") {
          perLayerDecode[r.layer].total++;
          if (ok) perLayerDecode[r.layer].match++;
        }
        if (!ok && samples.length < 30) samples.push({ p, i: row.i, layer: r.layer, got, want });
      }
      doneRows++;
      if (doneRows % 25 === 0) {
        const st = model.cacheStats();
        const dm = phase.d.total ? (100 * phase.d.match / phase.d.total).toFixed(3) : "—";
        const pm = phase.p.total ? (100 * phase.p.match / phase.p.total).toFixed(3) : "—";
        const rate = doneRows / ((performance.now() - tReplay0) / 1000);
        post({
          type: "tick",
          msg: `${doneRows}/${totalRows} pos (${rate.toFixed(1)}/s, prompt ${p}) — match p ${pm}% d ${dm}% — ` +
            `hit ${(100 * st.hits / Math.max(1, st.hits + st.misses)).toFixed(1)}% (${st.misses} miss, ${(st.bytesRead / 2 ** 30).toFixed(1)} GiB letti)`,
        });
      }
    }
  }

  const st = model.cacheStats();
  model.destroy();
  source.close();
  const report = {
    kind: "glm-routing-conformance",
    schemaVersion: 1,
    date: new Date().toISOString(),
    ggufSha256: GLM47_FLASH_SHA256,
    oracle: { llamaCppCommit: trace.header.llamaCppCommit, corpusHash: trace.header.corpusHash },
    config: { cap: cfg.cap ?? null, prompts: cfg.prompts ?? null, budgetGiB: cfg.budgetGiB, ctxMax },
    positions: totalRows,
    setMatch: {
      prefill: { ...phase.p, pct: phase.p.total ? 100 * phase.p.match / phase.p.total : null },
      decode: { ...phase.d, pct: phase.d.total ? 100 * phase.d.match / phase.d.total : null },
    },
    gate: { threshold: 99, pass: phase.d.total > 0 && 100 * phase.d.match / phase.d.total >= 99 },
    perLayerDecode: perLayerDecode
      .map((c, l) => ({ layer: l, ...c, pct: c.total ? 100 * c.match / c.total : null }))
      .filter((c) => c.total > 0),
    mismatchSamples: samples,
    residency: {
      ...st,
      hitRate: st.hits + st.misses > 0 ? st.hits / (st.hits + st.misses) : null,
      slots,
      importMs: source.importMs,
    },
    dispatchesPerTokenPlanned: model.dispatchesPerTokenPlanned, // DERIVATO dal piano, non contato (C3a it.3)
    deviceLimits: grantedLimits(device), // limiti CONCESSI: senza, un confronto fra run non e' falsificabile
    wallMs: performance.now() - t0,
    replayMs: performance.now() - tReplay0,
  };
  post({ type: "done", report });
}

self.onmessage = (ev: MessageEvent) => {
  void main(ev.data as Cfg).catch((e) => post({ type: "error", message: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) }));
};
