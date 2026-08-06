// Harness identità prefillChunk (C3a fase 5 fetta b): STESSO modello, due run
// dalla posizione 0 — (A) prefill sequenziale forward-per-token, (B)
// prefillChunk a chunk di M — poi la STESSA generazione greedy. Le righe KV si
// sovrascrivono per posizione e l'attention legge solo n = pos+1 (il "crop
// implicito" del path Qwen): le due run sono indipendenti per costruzione.
//
// Il gate (journal it.32): hidden e logits del CONFINE bit-uguali + id
// generati identici + routing identico su ogni (posizione, layer) del prompt.
// Una riga KV corrotta propagherebbe su tutti i logit successivi: il
// decode-continuation copre tutte le righe del prefill.
import { GLM47_FLASH as G, GLM47_FLASH_SHA256 } from "../shape";
import { dequantQ4_0Row } from "../quant";
import { createGlmModel, type GlmRouting } from "../glmmodel";
import { GLM_PREFILL_M } from "../glmprefillplan";
import { GlmOpfsSource } from "../glmsource";
import { slabBufferCap, grantedLimits } from "../gpulimits";
import { createEngineDevice } from "../gpudevice";
import { arenaNeeds } from "../residency";

interface Golden { modelSha256: string; prompts: Array<{ id: string; promptTokens: number[]; generated: number[] }> }
interface Cfg { prompt: number; nGen: number; budgetGiB: number }

type Out =
  | { type: "progress" | "tick"; msg: string }
  | { type: "done"; report: unknown }
  | { type: "error"; message: string };
const post = (m: Out) => (self as unknown as Worker).postMessage(m);
const progress = (msg: string) => post({ type: "progress", msg });

const argmax = (v: Float32Array): number => {
  let a = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[a]) a = i;
  return a;
};

async function main(cfg: Cfg): Promise<void> {
  const t0 = performance.now();
  const source = await GlmOpfsSource.open("/models/GLM-4.7-Flash-Q4_0.gguf", progress);
  const golden = (await (await fetch("/models/glm-conf-golden.json")).json()) as Golden;
  if (golden.modelSha256 !== GLM47_FLASH_SHA256) throw new Error("golden: SHA GGUF diverso dal canonico");
  const pr = golden.prompts[cfg.prompt];
  if (!pr) throw new Error(`prompt ${cfg.prompt} assente nel golden`);
  const prompt = pr.promptTokens;
  const ctxMax = prompt.length + cfg.nGen;

  const { device } = await createEngineDevice({
    label: "glmprefill",
    needs: (adapter) => ({
      ctxMax, head: { vocab: G.vocab, dModel: G.dModel },
      slabClassBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
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
  const embed = (tok: number): Float32Array => {
    dequantQ4_0Row(embd, 0, G.dModel, tok, xRow);
    return xRow;
  };
  progress(`modello 47 layer + head, ctxMax ${ctxMax}…`);
  const model = createGlmModel(device, source, {
    ctxMax, head: true,
    cache: { budgetBytes: Math.floor(cfg.budgetGiB * (1 << 30)), maxBindingBytes: maxBind, maxBufferBytes: maxBuf, timing: true },
  });

  // generazione greedy condivisa: parte dai logits del confine
  const generate = async (boundaryLogits: Float32Array): Promise<number[]> => {
    const ids: number[] = [];
    let prev = argmax(boundaryLogits);
    ids.push(prev);
    for (let j = 1; j < cfg.nGen; j++) {
      const r = await model.forward(embed(prev), prompt.length + j - 1, true);
      prev = argmax(r.logits!);
      ids.push(prev);
    }
    return ids;
  };

  // ---- RUN A: prefill SEQUENZIALE (riferimento) ----
  progress(`run A: prefill sequenziale (${prompt.length} pos)…`);
  const routingA: GlmRouting[][] = [];
  let hiddenA: Float32Array | null = null;
  let logitsA: Float32Array | null = null;
  const tA0 = performance.now();
  for (let i = 0; i < prompt.length; i++) {
    const last = i === prompt.length - 1;
    const r = await model.forward(embed(prompt[i]), i, last);
    routingA.push(r.routing);
    if (last) { hiddenA = r.hidden; logitsA = r.logits!; }
    if (i % 32 === 0) post({ type: "tick", msg: `A: ${i}/${prompt.length}` });
  }
  const prefillMsA = performance.now() - tA0;
  const idsA = await generate(logitsA!);

  // ---- RUN B: prefillChunk (M=16, ultimo chunk parziale) ----
  progress(`run B: prefillChunk (M=${GLM_PREFILL_M})…`);
  const routingB: GlmRouting[][] = [];
  let hiddenB: Float32Array | null = null;
  let logitsB: Float32Array | null = null;
  const tB0 = performance.now();
  for (let base = 0; base < prompt.length; base += GLM_PREFILL_M) {
    const m = Math.min(GLM_PREFILL_M, prompt.length - base);
    const rows = new Float32Array(m * G.dModel);
    for (let r = 0; r < m; r++) rows.set(embed(prompt[base + r]), r * G.dModel);
    const last = base + m >= prompt.length;
    const res = await model.prefillChunk(rows, base, last);
    for (const rr of res.routing) routingB.push(rr);
    if (last) { hiddenB = res.hidden; logitsB = res.logits!; }
    post({ type: "tick", msg: `B: ${Math.min(base + m, prompt.length)}/${prompt.length}` });
  }
  const prefillMsB = performance.now() - tB0;
  const idsB = await generate(logitsB!);

  // ---- confronto ----
  const bitEq = (a: Float32Array, b: Float32Array): { eq: boolean; maxAbs: number; nDiff: number } => {
    let maxAbs = 0, nDiff = 0;
    for (let i = 0; i < a.length; i++) {
      if (!Object.is(a[i], b[i])) nDiff++;
      maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
    }
    return { eq: nDiff === 0, maxAbs, nDiff };
  };
  const hiddenCmp = bitEq(hiddenA!, hiddenB!);
  const logitsCmp = bitEq(logitsA!, logitsB!);
  let routingDiff = 0, routingTot = 0;
  for (let p = 0; p < prompt.length; p++) {
    for (let l = 0; l < routingA[p].length; l++) {
      routingTot++;
      const ea = routingA[p][l].experts, eb = routingB[p][l].experts;
      for (let k = 0; k < ea.length; k++) if (ea[k] !== eb[k]) { routingDiff++; break; }
    }
  }
  const idsEq = idsA.length === idsB.length && idsA.every((v, i) => v === idsB[i]);
  const argmaxEq = argmax(logitsA!) === argmax(logitsB!);
  const pass = hiddenCmp.eq && logitsCmp.eq && idsEq && routingDiff === 0;

  model.destroy();
  source.close();
  post({
    type: "done",
    report: {
      kind: "glm-prefill-identity", schemaVersion: 1, date: new Date().toISOString(),
      ggufSha256: GLM47_FLASH_SHA256,
      config: { prompt: cfg.prompt, promptId: pr.id, promptTokens: prompt.length, nGen: cfg.nGen, chunkM: GLM_PREFILL_M, budgetGiB: cfg.budgetGiB, ctxMax },
      gate: {
        pass,
        hiddenBitEqual: hiddenCmp.eq, hiddenMaxAbs: hiddenCmp.maxAbs, hiddenDiffs: hiddenCmp.nDiff,
        logitsBitEqual: logitsCmp.eq, logitsMaxAbs: logitsCmp.maxAbs, logitsDiffs: logitsCmp.nDiff,
        boundaryArgmaxEqual: argmaxEq,
        generatedIdsEqual: idsEq, idsA, idsB,
        routingMismatch: routingDiff, routingTot,
      },
      prefillMs: { sequential: prefillMsA, chunked: prefillMsB, speedup: prefillMsA / prefillMsB },
      deviceLimits: grantedLimits(device),
      wallMs: performance.now() - t0, importMs: source.importMs,
    },
  });
}

self.onmessage = (ev: MessageEvent) => {
  void main(ev.data as Cfg).catch((e) => post({ type: "error", message: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) }));
};
