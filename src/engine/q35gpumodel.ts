// Orchestratore GPU full-model per Qwen 3.5 DENSO (4B/9B) — q1 fase 4
// slice 3. Assemblatore "correttezza prima" (pattern del bring-up
// gpuforward): decode teacher-forced, piano di step PRECOSTRUITO una volta
// (pipeline cache + bind group statici su scratch condivisi), un submit per
// token, readback dei logits e argmax su CPU. Full residency by design (il
// 4B Q4 sta in VRAM: è il regime del tier 16 GB; il paging resta al MoE).
//
// Ogni pezzo è GIÀ provato: kernel ktestati (79/79) e assembly dei due tipi
// di layer con pesi reali == cpuref a L2rel ~1e-7 (it.7). Qui c'è SOLO
// l'orchestrazione: loop 32 layer + embed row + ffn + head + residui.
import {
  addInPlaceWgsl, ARGMAX_CHUNK, argmaxStage1Wgsl, argmaxStage2Wgsl, attnDecodeWgsl, axpyWgsl,
  gemvF32Wgsl, gemvGrid, gemvQ4KWgsl, gemvQ5KWgsl,
  gemvQ6KWgsl, gemvQuantWgsl, kvAppendWgsl, rmsnormWgsl, ropeNeoxWgsl, sigmoidMulWgsl,
  siluMulWgsl, stridedCopyWgsl, routerTopKWgsl, axpySelWgsl,
  SEL_BYTES, MOE_IDX_BYTES, MOE_IDX_STRIDE, type KArenaOpts,
} from "./kernels/wgsl";
import { expertArenaBindings } from "./gpulimits";
import type { SlabTensorLayout } from "./moe";
import { deltaNetConvWgsl, deltaNetCoreWgsl, deltaNetGatesWgsl } from "./kernels/deltanet";
import { GGML_TYPE, tensorByteSize, type GgufTensorInfo } from "./gguf";
import { dequantQ4_0, dequantQ6_K, dequantQ8_0, repackKQuant, repackQ4_0, repackQ4_1, repackQ8_0, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES } from "./quant";
import { q35IsFullAttn, type Q35Shape } from "./q35shape";
import { ROUTER_QWEN35MOE, routerSelect, WEIGHTS_SUM_CLAMP_MIN, type RouterConfig } from "./moe";
import {
  ExpertCache, moeParkOf, type ExpertClass, type ExpertRawBytes, type SlotRef,
} from "./residency";
import { q35ExpertReader, q35MoeConfig } from "./q35expertstore";

export interface Q35RawReader {
  shape: Q35Shape;
  info(name: string): GgufTensorInfo;
  /** byte GREZZI del tensore (il chiamante può scartarli dopo l'upload) */
  read(name: string): Promise<Uint8Array>;
  /** sotto-range di un tensore (it.17: slab expert on-miss) — OBBLIGATORIO per MoE */
  readRange?(name: string, off: number, len: number): Promise<Uint8Array>;
}

interface Step { pipe: GPUComputePipeline; bind: GPUBindGroup; wgs: [number, number, number] }

/**
 * Esito del confronto ROUTER GPU (in ombra) vs selezione CPU, sui layer VERI
 * (goal fase-D fase 3b, fetta 3b). it.13 ha già gateato lo stesso kernel su 64
 * estrazioni sintetiche 256x8: qui la domanda è diversa e la sintetica non la
 * poteva porre — se la DISTRIBUZIONE vera dei logits del 35B produce margini
 * sotto la risoluzione dell'f32, la GPU (f32) e la CPU (f64) sceglierebbero
 * expert diversi. Per questo si riporta anche il margine minimo osservato: un
 * conteggio di flip a zero senza sapere quanto era stretto il caso peggiore
 * non dice se il gate ha visto il caso difficile.
 */
export interface Q35RouterShadowStats {
  /** confronti = layer MoE x token */
  comparisons: number;
  /** selezioni confrontate = comparisons x topK */
  picks: number;
  /** l'INSIEME dei top-K differisce: è il difetto che conta */
  setFlips: number;
  /** stesso insieme, ordine diverso (i pesi seguono l'expert, ma l'ordine
   *  decide quale slot finisce in quale entry di Sel) */
  orderFlips: number;
  /** errore relativo massimo sui pesi di mixing (CPU f64 come riferimento) */
  maxWeightRelErr: number;
  /** margine minimo sui logit fra l'ultimo preso e il primo scartato: la
   *  softmax è monotona, quindi è questo il numero che decide l'ordinamento */
  minLogitMargin: number;
  /** slot risolti dalla GPU diversi da quelli che la CPU ha poi usato, ESCLUSI
   *  i miss: l'ombra gira PRIMA degli `ensure` del suo layer, quindi un expert
   *  che sta per essere caricato si risolve MISS ed è corretto così */
  slotMismatch: number;
  /** miss visti dalla GPU (flag di Sel) e dalla CPU (`isResident`) */
  missGpu: number; missCpu: number; missDisagree: number;
}

export interface Q35GpuModel {
  /**
   * Un token teacher-forced; ritorna l'argmax dei logits. Con read=false
   * NON copia/mappa i logits (prefill sequenziale: gli step si accodano
   * senza sync; tornare -1) — la conformance legge solo dove serve.
   */
  step(token: number, pos: number, read?: boolean): Promise<number>;
  /**
   * K token TEACHER-FORCED in UN SOLO submit, con l'argmax calcolato su GPU
   * (goal fase-D fase 3). `step` aspetta il readback dei logits a ogni token
   * — 604 KB e, soprattutto, un round-trip in cui CPU e GPU si aspettano a
   * vicenda: misurato 51,5 ms/token contro 35,8 senza sync sul 4B. Qui gli
   * step si accodano tutti, ogni step scrive il proprio argmax in un buffer,
   * e si legge UNA volta sola K·4 byte.
   *
   * Teacher-forced: i token di ingresso sono noti in anticipo (conformance,
   * golden). La generazione libera ha bisogno del feedback su GPU (embed
   * gather), che è un passo a parte.
   *
   * `null` sui modelli MoE: la selezione degli expert passa dalla CPU a ogni
   * layer, quindi il batch non è possibile finché il routing non è su GPU.
   */
  decodeBatch: ((tokens: ArrayLike<number>, posStart: number) => Promise<number[]>) | null;
  /** azzera gli stati ricorrenti (conv + S) di tutti i layer linear: nuovo prompt. */
  resetState(): void;
  dispatchesPerToken: number;
  /** stats MoE (null sui densi): arena, routing esatto, residenza. */
  moeStats: (() => {
    hits: number; misses: number; uploadedBytes: number;
    /** scomposizione del costo per MISS (fase D fase 2): dove vanno i ms */
    readMs: number; packMs: number; uploadMs: number;
    routing: Record<string, number>; nSlots: Record<string, number>;
    parkSlots: Record<string, number>; slotBytes: Record<string, number>;
  }) | null;
  /**
   * Scomposizione del costo per token (fase D fase 3): dove vanno i ms del
   * decode. `embedMs` = dequant CPU della riga di embedding + writeBuffer;
   * `readbackMs` = attesa del readback dei logits (604 KB sul 4B);
   * `argmaxMs` = il max su `vocab` float in CPU. Tutti e tre spariscono col
   * pattern decodeBatch — misurarli PRIMA e' come si decide se conviene.
   *
   * `submits` e `readbacks` (fetta 3c) sono i CONTATORI su cui la fase 3b si
   * chiude: `submits` conta le `queue.submit`, `readbacks` i punti di attesa
   * GPU→CPU (piu' mapAsync concorrenti risolte insieme valgono UNO: e' un
   * round-trip solo). Si contano su ENTRAMBI i path, altrimenti il "prima" del
   * done-when sarebbe una stima.
   */
  perf(): { tokens: number; embedMs: number; readbackMs: number; argmaxMs: number; submits: number; readbacks: number };
  /**
   * Accende/spegne il path a submit unico a caldo (solo se il modello e' stato
   * costruito con `select: "optimistic"`; altrimenti LANCIA — non esiste il
   * cablaggio). E' quello che permette al gate di misurare le due passate sulla
   * STESSA cache: passata sync a freddo, passata ottimistica a caldo.
   */
  setOptimistic(on: boolean): void;
  /** DEBUG (it.17): dopo step(), hidden x a valle del layer indicato (solo MoE). */
  readTap(layer: number): Promise<Float32Array>;
  /** Esito dell'OMBRA del router GPU; `null` se non è stata accesa. */
  routerShadowStats: (() => Q35RouterShadowStats) | null;
  destroy(): void;
}

export interface Q35GpuModelOpts {
  /**
   * Accende il router+resolve su GPU in OMBRA (fase 3b, fetta 3b): gira in
   * coda al GEMV del router, nello stesso submit, e scrive una regione
   * PARALLELA di `Sel` che nessun kernel expert legge. La selezione di
   * produzione resta quella della CPU: qui si misura la fedeltà sui layer
   * veri, non si cambia il risultato. Costa un dispatch e una copia da 128 B
   * per layer, e si spegne di default.
   */
  routerShadow?: boolean;
  /**
   * Path di selezione degli expert (fase 3b, fetta 3c). PORT del `select` di
   * `glmmodel`, con gli stessi nomi:
   *
   * - `"cpu"` (default): il path di oggi — un submit e un readback dei logits
   *   del router PER LAYER, la CPU sceglie, `ensure`, e riempie `Sel`.
   * - `"optimistic"`: il token intero in UN submit. La `Sel` di produzione la
   *   scrive il router su GPU, che risolve expert→slot dalla `slotTable` e
   *   marca `dirtyB` quando trova un MISS; nessuno sulla CPU vede la selezione
   *   mentre il token gira, e il routing si ricostruisce dalla `Sel` letta in
   *   coda.
   *
   * OPT-IN per costruzione: finché la residenza non è raggiunta il path sync
   * costa meno (it.16: a cache fredda 39 token su 39 sono sporchi), e la
   * SOGLIA d'ingresso è materia della fase 5 — qui c'è il meccanismo, non la
   * policy. Costruire con `"optimistic"` costruisce ENTRAMBI i path e accende
   * il secondo; `setOptimistic(false)` torna al primo a caldo (è così che il
   * gate misura le due passate sulla stessa cache).
   */
  select?: "cpu" | "optimistic";
}

export async function createQ35GpuModel(
  device: GPUDevice, r: Q35RawReader, ctxMax = 64, arenaBudgetBytes = 12 * (1 << 30),
  opts: Q35GpuModelOpts = {},
): Promise<Q35GpuModel> {
  const S = r.shape;
  const isMoe = S.arch === "qwen35moe";
  if (isMoe && !r.readRange) throw new Error("q35gpumodel: MoE richiede reader.readRange");
  const d = S.dModel;
  const hd = S.headDim;
  const qDim = S.nHead * hd;
  const kvDim = S.nKvHead * hd;
  const qkvDim = (2 * S.linKHead + S.linVHead) * S.linHeadDim;
  const inner = S.linVHead * S.linHeadDim;
  const dFfn = (S.dFfn ?? 0) as number;

  const pipes = new Map<string, GPUComputePipeline>();
  const pipe = (code: string): GPUComputePipeline => {
    let p = pipes.get(code);
    if (!p) {
      p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
      pipes.set(code, p);
    }
    return p;
  };
  const sbuf = (data: Float32Array | Uint32Array): GPUBuffer => {
    const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(b, 0, data as BufferSource);
    return b;
  };
  const empty = (bytes: number): GPUBuffer =>
    device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });

  // pesi quantizzati → upload (i raw si scartano subito: streaming)
  const q40 = async (name: string): Promise<{ qs: GPUBuffer; scales: GPUBuffer; k: number; n: number; kind: "q4_0" | "q4_1" }> => {
    const t = r.info(name);
    const raw = await r.read(name);
    const nBlocks = (t.dims[0] / 32) * t.dims[1];
    const kind = t.type === GGML_TYPE.Q4_1 ? "q4_1" : "q4_0";
    if (t.type !== GGML_TYPE.Q4_0 && t.type !== GGML_TYPE.Q4_1) throw new Error(`q35gpumodel: ${name} tipo ${t.type}, atteso q4_0/q4_1`);
    const { qs, scales } = kind === "q4_1" ? repackQ4_1(raw, 0, nBlocks) : repackQ4_0(raw, 0, nBlocks);
    return { qs: sbuf(qs), scales: sbuf(scales), k: t.dims[0], n: t.dims[1], kind };
  };
  const q80 = async (name: string): Promise<{ qs: GPUBuffer; scales: GPUBuffer; k: number; n: number }> => {
    const t = r.info(name);
    const raw = await r.read(name);
    const { qs, scales } = repackQ8_0(raw, 0, (t.dims[0] / 32) * t.dims[1]);
    return { qs: sbuf(qs), scales: sbuf(scales), k: t.dims[0], n: t.dims[1] };
  };
  const kquant = async (name: string, blockBytes: number): Promise<{ blocks: GPUBuffer; k: number; n: number }> => {
    const t = r.info(name);
    const raw = await r.read(name);
    return { blocks: sbuf(repackKQuant(raw, 0, (t.dims[0] / 256) * t.dims[1], blockBytes)), k: t.dims[0], n: t.dims[1] };
  };
  const f32buf = async (name: string): Promise<GPUBuffer> => {
    const raw = await r.read(name);
    return sbuf(new Float32Array(raw.slice().buffer, 0, raw.byteLength / 4));
  };
  /** Loader GENERICO type-driven (it.17): gemv col kernel giusto per il tipo
   *  REALE del tensore (35B: attn/ssm_out/shexp Q8_0, alpha/beta F32). */
  const loadW = async (name: string): Promise<{ n: number; k: number; push: (src: GPUBuffer, dst: GPUBuffer) => void }> => {
    const t = r.info(name);
    const [k, n] = [t.dims[0], t.dims[1]];
    if (t.type === GGML_TYPE.F32) {
      const w = await f32buf(name);
      return { n, k, push: (src, dst) => push(gemvF32Wgsl({ K: k, N: n }), [w, src, dst], gemvGrid(n)) };
    }
    if (t.type === GGML_TYPE.Q4_0 || t.type === GGML_TYPE.Q4_1) {
      const w = await q40(name);
      return { n, k, push: (src, dst) => gemv(w, src, dst) };
    }
    if (t.type === GGML_TYPE.Q8_0) {
      const w = await q80(name);
      return { n, k, push: (src, dst) => gemv(w, src, dst, "q8_0") };
    }
    if (t.type === GGML_TYPE.Q4_K || t.type === GGML_TYPE.Q5_K || t.type === GGML_TYPE.Q6_K) {
      const blockBytes = t.type === GGML_TYPE.Q4_K ? 144 : t.type === GGML_TYPE.Q5_K ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;
      const w = await kquant(name, blockBytes);
      const code = t.type === GGML_TYPE.Q4_K ? gemvQ4KWgsl({ K: k, N: n }) : t.type === GGML_TYPE.Q5_K ? gemvQ5KWgsl({ K: k, N: n }) : gemvQ6KWgsl({ K: k, N: n });
      return { n, k, push: (src, dst) => push(code, [w.blocks, src, dst], gemvGrid(n)) };
    }
    throw new Error(`q35gpumodel: loadW tipo ${t.type} non gestito (${name})`);
  };

  // uniform (pos, nPast) — uno solo, aggiornato per token
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // scratch condivisi fra i layer (i bind group restano statici)
  const x = empty(d * 4);
  const xn = empty(d * 4);
  const attnY = empty(d * 4);
  const qkv = empty(qkvDim * 4), z = empty(inner * 4);
  const bRaw = empty(S.linVHead * 4), aRaw = empty(S.linVHead * 4), bSig = empty(S.linVHead * 4), gVal = empty(S.linVHead * 4);
  const convOut = empty(qkvDim * 4), gated = empty(inner * 4);
  const qFull = empty(2 * qDim * 4), kCur = empty(kvDim * 4), vCur = empty(kvDim * 4);
  const qB = empty(qDim * 4), gateB = empty(qDim * 4), qN = empty(qDim * 4), kN = empty(kvDim * 4);
  const attnO = empty(qDim * 4);
  const gateF = empty(dFfn ? dFfn * 4 : 16), upF = empty(dFfn ? dFfn * 4 : 16);
  // --- MoE (it.17): scratch dedicati (i dinamici non toccano quelli shexp) ---
  const nE = S.nExpert ?? 0, topK = S.nExpertUsed ?? 0, dE = S.dFfnExpert ?? 0;
  const routerLogits = empty(Math.max(nE, 4) * 4);
  const routerStaging = device.createBuffer({ size: Math.max(nE, 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const moeAcc = empty(d * 4);
  const gateS = empty(Math.max(dE, 4) * 4), upS = empty(Math.max(dE, 4) * 4), dnS = empty(d * 4), shScalar = empty(16);
  const gateE = empty(Math.max(dE, 4) * 4), upE = empty(Math.max(dE, 4) * 4), dnE = empty(d * 4);
  /** confini dei segmenti statici: segmento i = steps[cuts[i-1]..cuts[i]) */
  const cuts: number[] = [];
  /**
   * Layer ASSOLUTO dell'i-esimo segmento MoE. Nella famiglia Qwen 3.5/3.6 gli
   * expert ci sono da blk.0 (`denseLead: 0`) e i due indici coincidono, ma la
   * chiave della slotTable e la classe dello slab si leggono col layer
   * assoluto: tenerlo esplicito costa un array di 40 interi ed evita che un
   * modello della famiglia con dei layer densi in testa indirizzi la classe
   * sbagliata SENZA fallire.
   */
  const moeLayerAbs: number[] = [];

  // embedding: raw Q6_K tenuto CPU-side per il gather della riga; head = stesso
  // tensore (tied) o output.weight, su GPU
  // embd: Q6_K (4B) o Q4_0 (9B — che tiene invece la HEAD in Q6_K: inverso
  // del 4B, scoperto dal file in it.11)
  const embdInfo = r.info("token_embd.weight");
  if (embdInfo.type !== GGML_TYPE.Q6_K && embdInfo.type !== GGML_TYPE.Q4_0 && embdInfo.type !== GGML_TYPE.Q8_0) {
    throw new Error(`q35gpumodel: embd tipo ${embdInfo.type}, atteso Q6_K/Q4_0/Q8_0`);
  }
  const embdRaw = await r.read("token_embd.weight");
  // head: Q6_K (4B tied) o Q4_0 (9B non-tied, output.weight) — it.11
  const headName = S.tiedEmbeddings ? "token_embd.weight" : "output.weight";
  const headT = r.info(headName);
  let headStep: (src: GPUBuffer, dst: GPUBuffer) => void;
  if (headT.type === GGML_TYPE.Q6_K) {
    const head = await kquant(headName, Q6_K_BLOCK_BYTES);
    headStep = (src, dst) => push(gemvQ6KWgsl({ K: head.k, N: head.n }), [head.blocks, src, dst], gemvGrid(head.n));
  } else if (headT.type === GGML_TYPE.Q4_0) {
    const head = await q40(headName);
    headStep = (src, dst) => gemv(head, src, dst);
  } else {
    throw new Error(`q35gpumodel: head ${headName} tipo ${headT.type}, atteso Q6_K/Q4_0`);
  }
  const outNorm = await f32buf("output_norm.weight");
  const logits = empty(S.vocab * 4);
  const staging = device.createBuffer({ size: S.vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const steps: Step[] = [];
  const stateBufs: { buf: GPUBuffer; bytes: number }[] = [];
  const push = (code: string, bufs: GPUBuffer[], wgs: number | [number, number], withUni = false): void => {
    const p = pipe(code);
    const entries: GPUBindGroupEntry[] = bufs.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    if (withUni) entries.push({ binding: bufs.length, resource: { buffer: uni } });
    const bind = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries });
    steps.push({ pipe: p, bind, wgs: typeof wgs === "number" ? [wgs, 1, 1] : [wgs[0], wgs[1], 1] });
  };
  const gemv = (w: { qs: GPUBuffer; scales: GPUBuffer; k: number; n: number; kind?: "q4_0" | "q4_1" | "q8_0" }, src: GPUBuffer, dst: GPUBuffer, kind?: "q4_0" | "q4_1" | "q8_0"): void => {
    const kk = kind ?? w.kind ?? "q4_0";
    push(gemvQuantWgsl({ kind: kk, K: w.k, N: w.n, hasBias: false }), [w.qs, w.scales, src, dst], gemvGrid(w.n));
  };

  for (let l = 0; l < S.nLayer; l++) {
    const b = `blk.${l}.`;
    const attnNorm = await f32buf(`${b}attn_norm.weight`);
    const postNorm = await f32buf(`${b}post_attention_norm.weight`);
    push(rmsnormWgsl(d, S.rmsEps), [x, attnNorm, xn], 1);

    if (q35IsFullAttn(S, l)) {
      const wq = await loadW(`${b}attn_q.weight`);
      const wk = await loadW(`${b}attn_k.weight`);
      const wv = await loadW(`${b}attn_v.weight`);
      const wo = await loadW(`${b}attn_output.weight`);
      const qNormW = await f32buf(`${b}attn_q_norm.weight`);
      const kNormW = await f32buf(`${b}attn_k_norm.weight`);
      const kCache = empty(ctxMax * kvDim * 4);
      const vCache = empty(ctxMax * kvDim * 4);
      wq.push(xn, qFull);
      wk.push(xn, kCur);
      wv.push(xn, vCur);
      push(stridedCopyWgsl({ nVec: S.nHead, len: hd, srcStride: 2 * hd, srcOffset: 0, dstStride: hd, dstOffset: 0 }), [qFull, qB], Math.ceil(qDim / 64));
      push(stridedCopyWgsl({ nVec: S.nHead, len: hd, srcStride: 2 * hd, srcOffset: hd, dstStride: hd, dstOffset: 0 }), [qFull, gateB], Math.ceil(qDim / 64));
      push(rmsnormWgsl(hd, S.rmsEps, true), [qB, qNormW, qN], S.nHead);
      push(rmsnormWgsl(hd, S.rmsEps, true), [kCur, kNormW, kN], S.nKvHead);
      push(ropeNeoxWgsl(S.nHead, hd, S.ropeFreqBase, S.ropeDims), [qN], Math.ceil((S.nHead * S.ropeDims / 2) / 64), true);
      push(ropeNeoxWgsl(S.nKvHead, hd, S.ropeFreqBase, S.ropeDims), [kN], Math.ceil((S.nKvHead * S.ropeDims / 2) / 64), true);
      push(kvAppendWgsl(kvDim), [kN, kCache], Math.ceil(kvDim / 64), true);
      push(kvAppendWgsl(kvDim), [vCur, vCache], Math.ceil(kvDim / 64), true);
      push(attnDecodeWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: hd, ctxMax }), [qN, kCache, vCache, attnO], S.nHead, true);
      push(sigmoidMulWgsl(qDim), [attnO, gateB], Math.ceil(qDim / 64));
      wo.push(attnO, attnY);
    } else {
      const wqkv = await loadW(`${b}attn_qkv.weight`);
      const wz = await loadW(`${b}attn_gate.weight`);
      const wb = await loadW(`${b}ssm_beta.weight`);
      const wa = await loadW(`${b}ssm_alpha.weight`);
      const convW = await f32buf(`${b}ssm_conv1d.weight`);
      const aBuf = await f32buf(`${b}ssm_a`);
      const dtBuf = await f32buf(`${b}ssm_dt.bias`);
      const nrmBuf = await f32buf(`${b}ssm_norm.weight`);
      const wOut = await loadW(`${b}ssm_out.weight`);
      const convSt = sbuf(new Float32Array((S.linConvK - 1) * qkvDim));
      const stateS = sbuf(new Float32Array(S.linVHead * S.linHeadDim * S.linHeadDim));
      stateBufs.push({ buf: convSt, bytes: (S.linConvK - 1) * qkvDim * 4 });
      stateBufs.push({ buf: stateS, bytes: S.linVHead * S.linHeadDim * S.linHeadDim * 4 });
      wqkv.push(xn, qkv);
      wz.push(xn, z);
      wb.push(xn, bRaw);
      wa.push(xn, aRaw);
      push(deltaNetGatesWgsl(S.linVHead), [bRaw, aRaw, aBuf, dtBuf, bSig, gVal], 1);
      push(deltaNetConvWgsl(qkvDim, S.linConvK), [convSt, qkv, convW, convOut], Math.ceil(qkvDim / 64));
      push(deltaNetCoreWgsl({ hd: S.linHeadDim, nK: S.linKHead, nV: S.linVHead, eps: S.rmsEps }), [convOut, stateS, bSig, gVal, z, nrmBuf, gated], S.linVHead);
      wOut.push(gated, attnY);
    }
    push(addInPlaceWgsl(d), [x, attnY], Math.ceil(d / 64));

    if (!isMoe) {
      const wg = await q40(`${b}ffn_gate.weight`);
      const wu = await q40(`${b}ffn_up.weight`);
      const wd = await q40(`${b}ffn_down.weight`);
      push(rmsnormWgsl(d, S.rmsEps), [x, postNorm, xn], 1);
      gemv(wg, xn, gateF);
      gemv(wu, xn, upF);
      push(siluMulWgsl(dFfn), [gateF, upF], Math.ceil(dFfn / 64));
      gemv(wd, gateF, attnY);
      push(addInPlaceWgsl(d), [x, attnY], Math.ceil(d / 64));
    } else {
      // MoE (it.17): il segmento STATICO del layer chiude con shexp (statico:
      // non dipende dalla selezione, accumula in moeAcc col gate sigmoid su
      // GPU) + router → routerLogits. La parte per-expert è DINAMICA (readback
      // della selezione su CPU, correttezza-prima: 1 sync/layer DICHIARATO).
      push(rmsnormWgsl(d, S.rmsEps), [x, postNorm, xn], 1);
      const shGate = await loadW(`${b}ffn_gate_shexp.weight`);
      const shUp = await loadW(`${b}ffn_up_shexp.weight`);
      const shDown = await loadW(`${b}ffn_down_shexp.weight`);
      const shScalarW = await f32buf(`${b}ffn_gate_inp_shexp.weight`);
      shGate.push(xn, gateS);
      shUp.push(xn, upS);
      push(siluMulWgsl(dE), [gateS, upS], Math.ceil(dE / 64));
      shDown.push(gateS, dnS);
      push(gemvF32Wgsl({ K: d, N: 1 }), [shScalarW, xn, shScalar], 1);
      push(axpyWgsl(d, true), [moeAcc, dnS, shScalar], Math.ceil(d / 64));
      const router = await loadW(`${b}ffn_gate_inp.weight`);
      router.push(xn, routerLogits);
      cuts.push(steps.length);
      moeLayerAbs.push(l);
    }
  }

  push(rmsnormWgsl(d, S.rmsEps), [x, outNorm, xn], 1);
  headStep(xn, logits);

  const embdKind = embdInfo.type === GGML_TYPE.Q6_K ? "q6k" : embdInfo.type === GGML_TYPE.Q8_0 ? "q80" : "q40";
  const rowBlocks = embdKind === "q6k" ? d / 256 : d / 32;
  const rowBytes = embdKind === "q6k" ? (d / 256) * 210 : (d / 32) * (embdKind === "q80" ? 34 : 18);
  const embRow = new Float32Array(d);
  const dispatchesPerToken = steps.length;
// ====== residenza expert: LA MECCANICA UNICA (goal fase-D fase 1, it.7) ======
// L'arena a chunk + LRU + bind group scritte a mano qui in q1 it.16 sono
// SPARITE: questa è la stessa `ExpertCache` che serve GLM, guidata da una
// `MoeModelConfig` costruita dai metadata del GGUF. Il layout degli slab che
// ne esce coincide BYTE PER BYTE con quello che questo file calcolava a mano
// (test `q35-slab-parity`, CPU-only), quindi la migrazione non sposta un byte
// in VRAM: cambia il proprietario del meccanismo — e con lui arrivano a Qwen
// le ottimizzazioni che GLM aveva e questo path no (ruling direction §7-ter).
  interface MoeRt {
    nSlots: Record<string, number>; parkSlots: Record<string, number>; slotBytes: Record<string, number>;
    routing: Map<number, number>;
    stats(): { hits: number; misses: number; uploadedBytes: number; readMs: number; packMs: number; uploadMs: number };
    runLayer(l: number, logitsF32: Float32Array): Promise<void>;
    /**
     * OMBRA (fase 3b, fetta 3b): accoda il router+resolve su GPU nello STESSO
     * encoder del segmento statico, subito dopo il GEMV che ha scritto i
     * logits, e copia la `Sel` d'ombra nella staging. `null` se l'ombra è
     * spenta — il path di produzione non cambia di una riga.
     */
    shadowEncode: ((enc: GPUCommandEncoder, m: number) => void) | null;
    /** Confronta l'ombra col la selezione CPU del layer appena eseguito. */
    shadowCompare: (() => Promise<void>) | null;
    shadowStats: (() => Q35RouterShadowStats) | null;
    /**
     * IL TOKEN INTERO IN UN SUBMIT (fetta 3c). `null` se il modello non è stato
     * costruito con `select: "optimistic"`. Ritorna i logits se `read`, e fa da
     * sé tutta la contabilità di fine token (routing, hit, miss): nel path
     * sync quella contabilità la fa `runLayer`, qui non c'è nessun momento in
     * cui la CPU veda la selezione prima della fine.
     */
    runTokenOptimistic: ((read: boolean) => Promise<Float32Array | null>) | null;
    destroy(): void;
  }
  let moe: MoeRt | null = null;
  if (isMoe) {
    // i NOMI dei tensori expert e i loro byte stanno in `q35expertstore`, il
    // gemello di `expertstore` (GLM): qui non si nomina nessun tensore expert
    // e non si calcola nessun offset di slab.
    const cfg = q35MoeConfig(S, (n) => r.info(n));
    const readExpert = q35ExpertReader(S, (n) => r.info(n), r.readRange!.bind(r));
    // I limiti sono quelli NEGOZIATI col device (il conf worker chiede
    // slabClassBytes = 2 GiB): usarli invece di una costante evita il buffer
    // monolitico che in it.17 falliva in silenzio oltre il cap dell'adapter.
    const cache = new ExpertCache(device, {
      budgetBytes: arenaBudgetBytes,
      maxBindingBytes: device.limits.maxStorageBufferBindingSize,
      maxBufferBytes: device.limits.maxBufferSize,
      cfg,
      // telemetria liv.1 SEMPRE accesa su q35: sono 4 performance.now() per
      // MISS (non per token, non sugli hit) contro una lettura Range da ~1,7 MB
      // — non misurabile. In cambio ogni run di conformance porta con sé la
      // scomposizione read/pack/upload, che è il numero su cui la fase 2
      // decide.
      timing: true,
      // REGIME D'ARENA (fase 3b, fetta 3a). I buffer di classe si bindano
      // INTERI e l'indirizzo dello slab lo ricava il KERNEL dallo slot che
      // legge in `Sel`. È la precondizione per togliere il readback per layer:
      // finché è la CPU a calcolare i sotto-range, deve conoscere la selezione
      // PRIMA di poter accodare il layer, e quindi deve leggere i logits del
      // router. Non cambia né quanti slot ci sono né quanta VRAM costano
      // (`expertSlots` non guarda il regime): cambia la taglia dei buffer di
      // classe, e con essa quanti sono.
      arena: true,
      // La tabella expertKey→slot in VRAM non la legge ancora nessuno in
      // questa fetta (il router è su CPU). Si accende qui perché è lo stesso
      // `ensure` a tenerla aggiornata: in fetta 3b il resolve su GPU la trova
      // popolata dal primo token invece che dal secondo.
      slotTable: true,
    });
    const routerCfg: RouterConfig = { ...ROUTER_QWEN35MOE, nUsed: topK };
    const shadowOn = opts.routerShadow === true;
    if (shadowOn && opts.select === "optimistic") {
      throw new Error(
        "q35gpumodel: routerShadow e select \"optimistic\" insieme non hanno senso — l'ombra " +
        "confronta il router GPU con la selezione CPU, e in optimistic la selezione CPU non esiste");
    }
    const nMoeLayer = cuts.length;
    const nSel = nMoeLayer * topK;
    // `MoeIdx` si binda a dynamic offset: la spaziatura delle entry deve essere
    // un multiplo dell'allineamento CONCESSO, non del 256 di spec.
    if (MOE_IDX_STRIDE % device.limits.minUniformBufferOffsetAlignment !== 0) {
      throw new Error(
        `q35 MoE arena: stride ${MOE_IDX_STRIDE} non multiplo di ` +
        `minUniformBufferOffsetAlignment ${device.limits.minUniformBufferOffsetAlignment}`);
    }
    // `Sel`: una entry per (layer MoE, k) — {id, slot, w, flags}. In questa
    // fetta la riempie la CPU dopo gli `ensure`, esattamente come faceva con i
    // sotto-range; in fetta 3c ci scriverà il router su GPU e la CPU non
    // toccherà più questo buffer nel path caldo. È dimensionata per tutto il
    // token (40×8 entry, 5 120 B) e non per un layer: il layout non deve
    // cambiare quando i 40 submit diventano uno.
    // In modo OMBRA il buffer RADDOPPIA (design §4 di GLM): [0, nSel) è la
    // produzione — quella che i kernel expert leggono e che riempie la CPU —,
    // [nSel, 2·nSel) è l'ombra dove scrive il resolve GPU. Due regioni dello
    // stesso buffer e non due buffer: lo stesso kernel serve la fetta 3c
    // cambiando solo la entry di uniform che lo indirizza.
    const nSelTot = shadowOn ? 2 * nSel : nSel;
    const selBuf = device.createBuffer({
      size: nSelTot * SEL_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const selScratch = new ArrayBuffer(topK * SEL_BYTES);
    const selU32 = new Uint32Array(selScratch);
    const selF32 = new Float32Array(selScratch);
    // `MoeIdx` è STATICA: contenuto noto al load, una scrittura sola. Il dynamic
    // offset sceglie la entry, la entry dice quale `Sel` leggere. `tableBase` è
    // la base del layer nella slotTable e la userà il resolve GPU (fetta 3b).
    // In ombra servono nMoeLayer entry IN PIÙ, una per layer, che indirizzano
    // il router: stesso `tableBase`, ma `selIdx` già spostato nell'ombra. È
    // così che il kernel di resolve resta identico fra ombra e produzione —
    // cambia la entry, non il WGSL.
    const nIdx = nSel + (shadowOn ? nMoeLayer : 0);
    const moeIdxUni = device.createBuffer({
      size: nIdx * MOE_IDX_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    {
      const u = new Uint32Array(nIdx * (MOE_IDX_STRIDE / 4));
      for (let m = 0; m < nMoeLayer; m++) {
        const tableBase = moeLayerAbs[m] * cfg.nExpert;
        for (let k = 0; k < topK; k++) {
          const selIdx = m * topK + k;
          const w = selIdx * (MOE_IDX_STRIDE / 4);
          u[w] = selIdx; u[w + 1] = tableBase; u[w + 2] = m; u[w + 3] = 0;
        }
        if (shadowOn) {
          const w = (nSel + m) * (MOE_IDX_STRIDE / 4);
          u[w] = nSel + m * topK; u[w + 1] = tableBase; u[w + 2] = m; u[w + 3] = 0;
        }
      }
      device.queue.writeBuffer(moeIdxUni, 0, u as unknown as BufferSource);
    }
    // i kernel si scelgono dal LAYOUT della classe, non da un'assunzione sul
    // formato: se un GGUF della famiglia arrivasse con gate/up diversi, qui
    // si ferma con un messaggio invece di dequantizzare byte sbagliati.
    const gk = cfg.layout(cfg.classes[0]).gate.kind;
    if (gk !== "q4_K") throw new Error(`q35 MoE: nessun kernel gemv per gate/up ${gk}`);
    /**
     * La catena expert di UNA classe in regime d'arena (PORT da `glmmodel`).
     * Il bind group layout è ESPLICITO e non `"auto"`: `hasDynamicOffset` non
     * si esprime con layout auto, ed è l'offset dinamico a scegliere la entry
     * di `Sel`. Il risultato è che i bind group sono TRE per classe — gate, up,
     * down — invece di tre per slot: l'indirizzo non è più nel bind group.
     */
    const mkExpertClass = (cls: ExpertClass) => {
      const geo = cache.arenaGeometry(cls);
      const need = expertArenaBindings(geo.nBuf);
      if (need > device.limits.maxStorageBuffersPerShaderStage) {
        throw new Error(
          `q35 MoE arena: la classe ${cls} ha ${geo.nBuf} buffer d'arena ⇒ ${need} storage binding, ` +
          `il device ne concede ${device.limits.maxStorageBuffersPerShaderStage} — negoziare ` +
          "maxStorageBuffersPerShaderStage con arenaBuffers (gpulimits/arenaNeeds)");
      }
      const L = cfg.layout(cls);
      const dk = L.down.kind;
      if (dk !== "q4_K" && dk !== "q6_K") throw new Error(`q35 MoE: nessun kernel gemv per down ${dk}`);
      const kar = (t: SlabTensorLayout): KArenaOpts => ({
        nBuf: geo.nBuf, slabWords: geo.slabWords, slabsPerBuf: geo.slabsPerBuf, tensorWords: t.data / 4,
      });
      const bgl = device.createBindGroupLayout({
        entries: [
          ...Array.from({ length: geo.nBuf }, (_, j) => ({
            binding: j, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const },
          })),
          { binding: geo.nBuf, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
          { binding: geo.nBuf + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } },
          { binding: geo.nBuf + 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
          {
            binding: geo.nBuf + 3, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: MOE_IDX_BYTES },
          },
        ],
      });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const mkPipe = (code: string): GPUComputePipeline => device.createComputePipeline({
        layout, compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
      const bg = (src: GPUBuffer, dst: GPUBuffer): GPUBindGroup => device.createBindGroup({
        layout: bgl,
        entries: [
          ...cache.arenaBuffers(cls).map((b, j) => ({ binding: j, resource: { buffer: b } })),
          { binding: geo.nBuf, resource: { buffer: src } },
          { binding: geo.nBuf + 1, resource: { buffer: dst } },
          { binding: geo.nBuf + 2, resource: { buffer: selBuf } },
          // `size` ESPLICITA: con hasDynamicOffset la validazione pretende
          // offset+dynamicOffset+size <= buffer.size, e senza `size` il binding
          // varrebbe l'intero buffer ⇒ qualunque offset dinamico > 0 sarebbe
          // illegale (glmmodel:1007 documenta la stessa trappola).
          { binding: geo.nBuf + 3, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
        ],
      });
      return {
        nBuf: geo.nBuf,
        pGate: mkPipe(gemvQ4KWgsl({ K: d, N: dE, arena: kar(L.gate) })),
        pUp: mkPipe(gemvQ4KWgsl({ K: d, N: dE, arena: kar(L.up) })),
        pDown: mkPipe(dk === "q6_K"
          ? gemvQ6KWgsl({ K: dE, N: d, arena: kar(L.down) })
          : gemvQ4KWgsl({ K: dE, N: d, arena: kar(L.down) })),
        bgGate: bg(xn, gateE), bgUp: bg(xn, upE), bgDown: bg(gateE, dnE),
      };
    };
    const expertCls: Record<ExpertClass, ReturnType<typeof mkExpertClass>> = {};
    for (const c of cfg.classes) expertCls[c] = mkExpertClass(c);
    const pSilu = pipe(siluMulWgsl(dE));
    const pAxpy = pipe(axpyWgsl(d));
    const pAdd = pipe(addInPlaceWgsl(d));
    const wBufs = Array.from({ length: topK }, () => device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
    const mkBg = (p2: GPUComputePipeline, entries: (GPUBuffer | GPUBufferBinding)[]): GPUBindGroup =>
      device.createBindGroup({
        layout: p2.getBindGroupLayout(0),
        entries: entries.map((e, i) => ({ binding: i, resource: (e as GPUBufferBinding).buffer ? (e as GPUBufferBinding) : { buffer: e as GPUBuffer } })),
      });
    const bgSilu = mkBg(pSilu, [gateE, upE]);
    const bgAxpy = wBufs.map((w) => mkBg(pAxpy, [moeAcc, dnE, w]));
    const bgAddRes = mkBg(pAdd, [x, moeAcc]);
    // ---- OMBRA: router + resolve su GPU accanto alla selezione CPU (fetta 3b) ----
    // Il kernel è quello di it.13, in configurazione qwen35moe (softmax). Il
    // binding `bias` resta dichiarato anche qui e ci si lega un buffer di ZERI:
    // `probs[i] + 0.0` è esatto ed è letteralmente ciò che `routerSelect` fa
    // quando la config non usa il bias — così il layout non dipende dalla
    // famiglia (scelta motivata in it.13).
    const shadow = shadowOn ? ((): {
      encode: (enc: GPUCommandEncoder, m: number) => void;
      compare: () => Promise<void>;
      stats: () => Q35RouterShadowStats;
      note: (sel: { experts: ArrayLike<number>; weights: ArrayLike<number> }, slots: SlotRef[], missing: Set<number>, logits: Float32Array) => void;
    } => {
      const bgl = device.createBindGroupLayout({
        entries: [
          ...[0, 1].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
          ...[2, 3, 4].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
          {
            binding: 6, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: MOE_IDX_BYTES },
          },
        ],
      });
      const code = routerTopKWgsl({
        nExpert: nE, nUsed: topK, weightsScale: routerCfg.weightsScale,
        clampMin: WEIGHTS_SUM_CLAMP_MIN, gating: routerCfg.gating,
        resolve: { nExpert: nE, nUsed: topK },
      });
      const pipeR = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
      const zeroBias = sbuf(new Float32Array(nE));
      const ids = empty(topK * 4), wts = empty(topK * 4);
      const bind = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: routerLogits } },
          { binding: 1, resource: { buffer: zeroBias } },
          { binding: 2, resource: { buffer: ids } },
          { binding: 3, resource: { buffer: wts } },
          { binding: 4, resource: { buffer: selBuf } },
          { binding: 5, resource: { buffer: cache.slotTableBuffer() } },
          { binding: 6, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
        ],
      });
      const selStaging = device.createBuffer({
        size: topK * SEL_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const st: Q35RouterShadowStats = {
        comparisons: 0, picks: 0, setFlips: 0, orderFlips: 0, maxWeightRelErr: 0,
        minLogitMargin: Infinity, slotMismatch: 0, missGpu: 0, missCpu: 0, missDisagree: 0,
      };
      // ultima selezione CPU del layer, messa da parte per il confronto
      let cpuIds: number[] = [], cpuW: number[] = [], cpuSlots: number[] = [], cpuMiss: boolean[] = [];
      let cpuMargin = Infinity;
      return {
        encode: (enc, m) => {
          const p2 = enc.beginComputePass();
          p2.setPipeline(pipeR);
          p2.setBindGroup(0, bind, [(nSel + m) * MOE_IDX_STRIDE]);
          p2.dispatchWorkgroups(1);
          p2.end();
          enc.copyBufferToBuffer(selBuf, (nSel + m * topK) * SEL_BYTES, selStaging, 0, topK * SEL_BYTES);
        },
        note: (sel, slots, missing, lg) => {
          cpuIds = Array.from(sel.experts as ArrayLike<number>).slice(0, topK);
          cpuW = Array.from(sel.weights as ArrayLike<number>).slice(0, topK);
          cpuSlots = slots.map((s) => s.idx);
          cpuMiss = cpuIds.map((e) => missing.has(e));
          // margine sui LOGIT fra l'ultimo preso e il primo scartato: la
          // softmax è monotona, quindi l'ordinamento lo decide questo numero.
          // È la misura di "quanto era difficile il caso", senza la quale uno
          // zero flip non dice se il gate ha visto il caso stretto.
          const taken = new Set(cpuIds);
          let lastIn = Infinity, firstOut = -Infinity;
          for (let i = 0; i < nE; i++) {
            if (taken.has(i)) { if (lg[i] < lastIn) lastIn = lg[i]; }
            else if (lg[i] > firstOut) firstOut = lg[i];
          }
          cpuMargin = lastIn - firstOut;
        },
        compare: async () => {
          await selStaging.mapAsync(GPUMapMode.READ);
          const raw = selStaging.getMappedRange().slice(0);
          selStaging.unmap();
          const u = new Uint32Array(raw), fl = new Float32Array(raw);
          st.comparisons++;
          st.picks += topK;
          if (cpuMargin < st.minLogitMargin) st.minLogitMargin = cpuMargin;
          const gpuIds: number[] = [];
          for (let k = 0; k < topK; k++) gpuIds.push(u[k * 4]);
          const same = new Set(gpuIds);
          let setEq = same.size === cpuIds.length;
          if (setEq) for (const e of cpuIds) if (!same.has(e)) { setEq = false; break; }
          if (!setEq) st.setFlips++;
          else if (gpuIds.some((e, k) => e !== cpuIds[k])) st.orderFlips++;
          for (let k = 0; k < topK; k++) if (cpuMiss[k]) st.missCpu++;
          // Slot e miss si appaiano per EXPERT, non per posizione: con un flip
          // d'ordine la posizione k porterebbe expert diversi e il confronto
          // sarebbe un falso allarme (o peggio, un falso via libera).
          const cpuSlotOf = new Map<number, number>();
          const cpuMissOf = new Map<number, boolean>();
          const cpuWOf = new Map<number, number>();
          cpuIds.forEach((e, i) => { cpuSlotOf.set(e, cpuSlots[i]); cpuMissOf.set(e, cpuMiss[i]); cpuWOf.set(e, cpuW[i]); });
          for (let k = 0; k < topK; k++) {
            const e = gpuIds[k], slotGpu = u[k * 4 + 1], missFlag = (u[k * 4 + 3] & 1) === 1;
            if (missFlag) st.missGpu++;
            if (!cpuSlotOf.has(e)) continue; // expert non scelto dalla CPU: è già un setFlip
            const wGpu = fl[k * 4 + 2], wCpu = cpuWOf.get(e)!;
            if (Math.abs(wCpu) > 1e-12) {
              const rel = Math.abs(wGpu - wCpu) / Math.abs(wCpu);
              if (rel > st.maxWeightRelErr) st.maxWeightRelErr = rel;
            }
            // L'ombra gira PRIMA degli `ensure` del layer: un expert che la CPU
            // sta per caricare si risolve MISS, ed è la residenza parziale vista
            // da GPU, non un errore. Il disaccordo è l'altro verso: la GPU dice
            // MISS su un expert che era GIÀ residente, o dà uno slot diverso da
            // quello che la CPU ha poi usato per un expert residente.
            if (missFlag !== cpuMissOf.get(e)) st.missDisagree++;
            else if (!missFlag && slotGpu !== cpuSlotOf.get(e)) st.slotMismatch++;
          }
        },
        stats: () => ({ ...st }),
      };
    })() : null;
    // ---- PATH A SUBMIT UNICO (fetta 3c): la `Sel` di produzione la scrive la GPU ----
    // Cambia UN pezzo rispetto alla fetta 3a: chi riempie `Sel`. I kernel
    // expert sono gli stessi, il bind group è lo stesso, l'offset dinamico è lo
    // stesso — l'indirezione era stata costruita apposta perché questo passo
    // costasse una entry di uniform e non una riscrittura (design §1).
    const opt = opts.select === "optimistic" ? ((): {
      dirtyB: GPUBuffer; dirtyStaging: GPUBuffer; selStaging: GPUBuffer;
      pipeR: GPUComputePipeline; bindR: GPUBindGroup;
      pAxpySel: GPUComputePipeline; bgAxpySel: GPUBindGroup;
      destroy(): void;
    } => {
      const dirtyB = device.createBuffer({
        size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const dirtyStaging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const selStaging = device.createBuffer({
        size: nSel * SEL_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      // Router+resolve di PRODUZIONE: lo stesso kernel dell'ombra con in più il
      // binding 7 (`dirtyB`) — [0] = primo layer MoE con miss (atomicMin), [1] =
      // conteggio (atomicAdd). È l'unica differenza di WGSL fra ombra e
      // produzione; la differenza di CABLAGGIO è la entry di `MoeIdx` che
      // l'offset dinamico seleziona, e quella di produzione è la (layer, k=0),
      // il cui `selIdx` punta alla regione che i kernel expert leggono.
      const bgl = device.createBindGroupLayout({
        entries: [
          ...[0, 1].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
          ...[2, 3, 4].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
          {
            binding: 6, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: MOE_IDX_BYTES },
          },
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } },
        ],
      });
      const pipeR = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: {
          module: device.createShaderModule({
            code: routerTopKWgsl({
              nExpert: nE, nUsed: topK, weightsScale: routerCfg.weightsScale,
              clampMin: WEIGHTS_SUM_CLAMP_MIN, gating: routerCfg.gating,
              resolve: { nExpert: nE, nUsed: topK, dirty: true },
            }),
          }),
          entryPoint: "main",
        },
      });
      const zeroBias = sbuf(new Float32Array(nE));
      const ids = empty(topK * 4), wts = empty(topK * 4);
      const bindR = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: routerLogits } },
          { binding: 1, resource: { buffer: zeroBias } },
          { binding: 2, resource: { buffer: ids } },
          { binding: 3, resource: { buffer: wts } },
          { binding: 4, resource: { buffer: selBuf } },
          { binding: 5, resource: { buffer: cache.slotTableBuffer() } },
          { binding: 6, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
          { binding: 7, resource: { buffer: dirtyB } },
        ],
      });
      // Il peso di mixing ora NASCE su GPU (campo `w` di `Sel`): l'axpy che
      // legge da un buffer riempito dalla CPU non ha più nessuno che lo riempia.
      const bglA = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } },
          ...[1, 2].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
          {
            binding: 3, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: MOE_IDX_BYTES },
          },
        ],
      });
      const pAxpySel = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bglA] }),
        compute: { module: device.createShaderModule({ code: axpySelWgsl(d) }), entryPoint: "main" },
      });
      const bgAxpySel = device.createBindGroup({
        layout: bglA,
        entries: [
          { binding: 0, resource: { buffer: moeAcc } },
          { binding: 1, resource: { buffer: dnE } },
          { binding: 2, resource: { buffer: selBuf } },
          { binding: 3, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
        ],
      });
      return {
        dirtyB, dirtyStaging, selStaging, pipeR, bindR, pAxpySel, bgAxpySel,
        destroy: () => { dirtyB.destroy(); dirtyStaging.destroy(); selStaging.destroy(); },
      };
    })() : null;
    /** sentinel di `dirtyB`: [0] è un atomicMin sul layer MoE, [1] un atomicAdd */
    const dirtyInit = new Uint32Array([0xffffffff, 0, 0, 0]);
    const routing = new Map<number, number>();
    const parkSlots = moeParkOf(cfg);
    const slotBytes: Record<string, number> = {};
    for (const c of cfg.classes) slotBytes[c] = cfg.layout(c).bytes;
    moe = {
      nSlots: cache.stats().slots, parkSlots, slotBytes, routing,
      stats: () => {
      const st = cache.stats();
      return {
        hits: st.hits, misses: st.misses, uploadedBytes: st.bytesUploaded,
        readMs: st.readMs, packMs: st.packMs, uploadMs: st.uploadMs,
      };
    },
      destroy: () => { opt?.destroy(); cache.destroy(); },
      shadowEncode: shadow ? (enc, m) => shadow.encode(enc, m) : null,
      shadowCompare: shadow ? () => shadow.compare() : null,
      shadowStats: shadow ? () => shadow.stats() : null,
      runTokenOptimistic: opt ? async (read: boolean): Promise<Float32Array | null> => {
        // reset di `dirtyB` PER TOKEN: `queue.writeBuffer` è ordinata prima
        // dell'intero submit che segue, quindi arriva prima di ogni resolve.
        device.queue.writeBuffer(opt.dirtyB, 0, dirtyInit as unknown as BufferSource);
        // Error scope come CONTRATTO di ogni encode nuovo (landmine B1): una
        // pipeline invalida fa droppare il submit IN SILENZIO e i readback
        // tornano stale-ma-plausibili.
        device.pushErrorScope("validation");
        device.pushErrorScope("out-of-memory");
        const enc = device.createCommandEncoder();
        const gg2 = gemvGrid(d);
        for (let m = 0; m < nMoeLayer; m++) {
          // `moeAcc` si azzera DENTRO l'encoder. La `queue.writeBuffer` del path
          // sync qui non funzionerebbe: è ordinata PRIMA dell'intero submit,
          // quindi i 40 layer vedrebbero un solo azzeramento — l'ultimo.
          enc.clearBuffer(moeAcc, 0, d * 4);
          const pass = enc.beginComputePass();
          for (let i = m === 0 ? 0 : cuts[m - 1]; i < cuts[m]; i++) {
            const st = steps[i];
            pass.setPipeline(st.pipe);
            pass.setBindGroup(0, st.bind);
            pass.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
          }
          // router+resolve: legge i logits che il GEMV appena accodato ha
          // scritto e pubblica `Sel` per gli 8 dispatch che seguono, nello
          // stesso pass. L'entry di `MoeIdx` è la (layer, k=0): il suo `selIdx`
          // è la base della regione di PRODUZIONE del layer.
          pass.setPipeline(opt.pipeR);
          pass.setBindGroup(0, opt.bindR, [m * topK * MOE_IDX_STRIDE]);
          pass.dispatchWorkgroups(1);
          const E = expertCls[cfg.classOf(moeLayerAbs[m])];
          for (let k2 = 0; k2 < topK; k2++) {
            const dyn = (m * topK + k2) * MOE_IDX_STRIDE;
            pass.setPipeline(E.pGate); pass.setBindGroup(0, E.bgGate, [dyn]); pass.dispatchWorkgroups(dE, 1, 1);
            pass.setPipeline(E.pUp); pass.setBindGroup(0, E.bgUp, [dyn]); pass.dispatchWorkgroups(dE, 1, 1);
            pass.setPipeline(pSilu); pass.setBindGroup(0, bgSilu); pass.dispatchWorkgroups(Math.ceil(dE / 64), 1, 1);
            pass.setPipeline(E.pDown); pass.setBindGroup(0, E.bgDown, [dyn]); pass.dispatchWorkgroups(gg2[0], gg2[1], 1);
            pass.setPipeline(opt.pAxpySel); pass.setBindGroup(0, opt.bgAxpySel, [dyn]); pass.dispatchWorkgroups(Math.ceil(d / 64), 1, 1);
          }
          pass.setPipeline(pAdd); pass.setBindGroup(0, bgAddRes); pass.dispatchWorkgroups(Math.ceil(d / 64), 1, 1);
          pass.end();
        }
        const tail = enc.beginComputePass();
        for (let i = cuts[nMoeLayer - 1]; i < steps.length; i++) {
          const st = steps[i];
          tail.setPipeline(st.pipe);
          tail.setBindGroup(0, st.bind);
          tail.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
        }
        tail.end();
        if (read) enc.copyBufferToBuffer(logits, 0, staging, 0, S.vocab * 4);
        // `Sel` INTERA in coda: è il solo momento in cui le regioni dei 40 layer
        // sono complete, ed è insieme la decisione del router e ciò che i kernel
        // hanno letto davvero.
        enc.copyBufferToBuffer(selBuf, 0, opt.selStaging, 0, nSel * SEL_BYTES);
        enc.copyBufferToBuffer(opt.dirtyB, 0, opt.dirtyStaging, 0, 16);
        device.queue.submit([enc.finish()]);
        perfAcc.submits++;
        // I1: da qui al readback la slotTable è INTOCCABILE — il resolve l'ha
        // già letta e gli slot che ha pubblicato in `Sel` devono restare quelli.
        cache.setInFlight(true);
        const errOom = await device.popErrorScope();
        const errVal = await device.popErrorScope();
        if (errOom ?? errVal) throw new Error(`q35 optimistic error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
        // Le mapAsync partono INSIEME (stesso submit alle spalle): un
        // round-trip host solo, ed è per questo che vale UN readback.
        const tRb = performance.now();
        const maps: Promise<undefined>[] = [
          opt.selStaging.mapAsync(GPUMapMode.READ),
          opt.dirtyStaging.mapAsync(GPUMapMode.READ),
        ];
        if (read) maps.push(staging.mapAsync(GPUMapMode.READ));
        await Promise.all(maps);
        perfAcc.readbacks++;
        perfAcc.readbackMs += performance.now() - tRb;
        cache.setInFlight(false); // confine di token: la tabella torna toccabile
        const su = new Uint32Array(opt.selStaging.getMappedRange().slice(0));
        opt.selStaging.unmap();
        const du = new Uint32Array(opt.dirtyStaging.getMappedRange().slice(0));
        opt.dirtyStaging.unmap();
        let lg: Float32Array | null = null;
        if (read) {
          lg = new Float32Array(staging.getMappedRange().slice(0, S.vocab * 4));
          staging.unmap();
        }
        // I MISS si derivano dalla `Sel` e si incrociano col flag aggregato che
        // ha scritto il kernel: due strade indipendenti verso lo stesso fatto.
        // Una divergenza è un bug strutturale (il resolve ha scritto `Sel` e
        // `dirtyB` in disaccordo), mai un dato da interpretare.
        let missCount = 0, firstDirty = -1;
        for (let m = 0; m < nMoeLayer; m++) {
          for (let k2 = 0; k2 < topK; k2++) {
            if ((su[(m * topK + k2) * 4 + 3] & 1) !== 0) {
              missCount++;
              if (firstDirty < 0) firstDirty = m;
            }
          }
        }
        const kernelFirst = du[0] === 0xffffffff ? -1 : du[0];
        if (du[1] !== missCount || kernelFirst !== firstDirty) {
          throw new Error(
            `q35 optimistic: dirtyB (${du[1]} miss, primo layer MoE ${kernelFirst}) != derivazione ` +
            `da Sel (${missCount} miss, primo layer MoE ${firstDirty}) — il resolve ha scritto Sel e ` +
            "dirtyB in disaccordo: bug strutturale, token non interpretabile");
        }
        if (missCount > 0) {
          // DEGRADO DEFINITO, non silenzioso: gli expert mancanti non hanno
          // partecipato (contributo zero, `axpySelWgsl`), quindi il token è
          // SBAGLIATO e si dice invece di restituire numeri plausibili.
          // Il repair+replay che rende utilizzabile questo caso è la fetta
          // successiva; qui il meccanismo esiste e il regime in cui conviene
          // (residenza raggiunta) è quello in cui `missCount` è 0 per
          // costruzione — misurato in it.16: 0 token sporchi su 39 a caldo.
          throw new Error(
            `q35 optimistic: ${missCount} MISS nella Sel di questo token (primo layer MoE ` +
            `${firstDirty}) — token INVALIDO. Il path a submit unico esige la residenza già ` +
            "raggiunta: senza repair+replay un expert mancante non partecipa e il risultato non " +
            "è quello del path sync. Scaldare la cache col path sync (setOptimistic(false)).");
        }
        // Contabilità di fine token, dalla `Sel` letta in coda: routing (il
        // numero che il gate confronta), hit e LRU. Nel path sync la fa
        // `ensure` una selezione alla volta; qui la CPU vede tutto insieme e
        // dopo, ma deve vedere ESATTAMENTE le stesse cose.
        for (let m = 0; m < nMoeLayer; m++) {
          const l = moeLayerAbs[m];
          const experts = new Uint32Array(topK);
          for (let k2 = 0; k2 < topK; k2++) {
            const e = su[(m * topK + k2) * 4];
            experts[k2] = e;
            const key = cache.keyOf(l, e);
            routing.set(key, (routing.get(key) ?? 0) + 1);
            cache.noteResidentHit(l, e);
          }
          cache.noteSelection(l, experts);
        }
        return lg;
      } : null,
      async runLayer(m: number, logitsF32: Float32Array): Promise<void> {
        const l = moeLayerAbs[m];
        // Selezione: IL router unico, in configurazione qwen35moe (softmax,
        // niente bias, niente scale). Stessa matematica del cpuref, che prima
        // era ricopiata qui a mano.
        const sel = routerSelect(logitsF32, null, routerCfg);
        // L'I/O sta FUORI dalla cache: `ensure` è sincrona (GLM legge da
        // memoria), il 35B no. Guardo chi manca, `await`to solo quelli, e poi
        // consegno i byte già in mano. Sugli hit non si legge niente.
        const raw = new Map<number, ExpertRawBytes>();
        const missing: number[] = [];
        for (const e of sel.experts) if (!cache.isResident(l, e)) missing.push(e);
        if (missing.length > 0) {
          const got = await Promise.all(missing.map((e) => readExpert(l, e)));
          missing.forEach((e, i) => raw.set(e, got[i]));
        }
        // i top-K del token devono coesistere: nessuno di loro può essere
        // vittima di eviction per far posto agli altri.
        const pinned = new Set<number>();
        for (const e of sel.experts) pinned.add(cache.keyOf(l, e));
        const slots: SlotRef[] = [];
        for (const e of sel.experts) {
          const key = cache.keyOf(l, e);
          routing.set(key, (routing.get(key) ?? 0) + 1);
          slots.push(cache.ensure(l, e, (_ly, ex) => raw.get(ex)!, pinned).slot);
        }
        cache.noteSelection(l, sel.experts);
        // l'ombra confronta contro CIÒ CHE LA CPU HA DECISO, e i miss vanno
        // presi PRIMA degli `ensure`: dopo, `isResident` direbbe sempre sì.
        shadow?.note(sel, slots, new Set(missing), logitsF32);
        // `Sel` del layer. L'indirizzo dell'expert è `slot.idx` — l'indice
        // GLOBALE dello slot nella classe — e da qui in poi è il KERNEL a
        // ricavarne (binding, base): la CPU non calcola più sotto-range. Lo
        // `SlotRef` resta il solo modo di ottenerlo (marchio di conio,
        // `residency.ts`): l'indirizzo cambia rappresentazione, non provenienza.
        for (let k2 = 0; k2 < topK; k2++) {
          selU32[k2 * 4] = sel.experts[k2];
          selU32[k2 * 4 + 1] = slots[k2].idx;
          selF32[k2 * 4 + 2] = sel.weights[k2];
          selU32[k2 * 4 + 3] = 0;
        }
        device.queue.writeBuffer(selBuf, m * topK * SEL_BYTES, selScratch);
        // La tabella si pubblica DOPO gli `ensure` del layer: le writeBuffer
        // degli slab sono già in coda, quindi la tabella arriva dopo il dato che
        // indirizza (R5 del design d'arena — l'ordine inverso pubblicherebbe uno
        // slot ancora vuoto). Qui non la legge nessuno: serve dalla fetta 3b.
        cache.flushSlotTable();
        for (let k2 = 0; k2 < topK; k2++) {
          device.queue.writeBuffer(wBufs[k2], 0, new Float32Array([sel.weights[k2], 0, 0, 0]));
        }
        const E = expertCls[cfg.classOf(l)];
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        const disp = (p2: GPUComputePipeline, b2: GPUBindGroup, wg: [number, number, number], dyn?: number): void => {
          pass.setPipeline(p2);
          if (dyn === undefined) pass.setBindGroup(0, b2);
          else pass.setBindGroup(0, b2, [dyn]);
          pass.dispatchWorkgroups(wg[0], wg[1], wg[2]);
        };
        const gg2 = gemvGrid(d);
        for (let k2 = 0; k2 < topK; k2++) {
          // l'offset dinamico sceglie la entry (layer, k) di MoeIdx, che punta
          // alla Sel di quell'expert: è l'unico parametro per-expert rimasto.
          const dyn = (m * topK + k2) * MOE_IDX_STRIDE;
          disp(E.pGate, E.bgGate, [dE, 1, 1], dyn);
          disp(E.pUp, E.bgUp, [dE, 1, 1], dyn);
          disp(pSilu, bgSilu, [Math.ceil(dE / 64), 1, 1]);
          disp(E.pDown, E.bgDown, [gg2[0], gg2[1], 1], dyn);
          disp(pAxpy, bgAxpy[k2], [Math.ceil(d / 64), 1, 1]);
        }
        disp(pAdd, bgAddRes, [Math.ceil(d / 64), 1, 1]); // x += moeAcc (shexp + expert)
        pass.end();
        device.queue.submit([enc.finish()]);
        perfAcc.submits++;
      },
    };
  }
  const perfAcc = { tokens: 0, embedMs: 0, readbackMs: 0, argmaxMs: 0, submits: 0, readbacks: 0 };
  /** path attivo: `true` solo se costruito con `select: "optimistic"` (fetta 3c) */
  let optimisticOn = opts.select === "optimistic";
  const zeroAcc = new Float32Array(d);
  const tapStaging = device.createBuffer({ size: d * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let tapLayer = -1;
  let tapWanted = -1;
  let tapValue: Float32Array | null = null;

  // --- argmax su GPU + batch teacher-forced (fase D fase 3) ---
  const N_PARTIALS = Math.ceil(S.vocab / ARGMAX_CHUNK);
  const amaxPartMax = empty(N_PARTIALS * 4);
  const amaxPartIdx = empty(N_PARTIALS * 4);
  const amaxOut = empty(16);
  const pAmax1 = pipe(argmaxStage1Wgsl(S.vocab));
  const pAmax2 = pipe(argmaxStage2Wgsl(N_PARTIALS));
  const bgAmax1 = device.createBindGroup({
    layout: pAmax1.getBindGroupLayout(0),
    entries: [logits, amaxPartMax, amaxPartIdx].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  const bgAmax2 = device.createBindGroup({
    layout: pAmax2.getBindGroupLayout(0),
    entries: [amaxPartMax, amaxPartIdx, amaxOut].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  const BATCH_MAX = 32;
  // Le righe di embedding e gli uniform dei K step NON si possono scrivere con
  // writeBuffer dentro l'encoder: `queue.writeBuffer` e' ordinata PRIMA del
  // submit, quindi tutti gli step vedrebbero l'ultimo valore scritto. Si
  // impacchettano in un buffer solo e si copiano nello scratch all'inizio di
  // ogni step, DENTRO l'encoder (stesso schema di `dbSlots` in gpuforward).
  const embBatch = empty(BATCH_MAX * d * 4);
  const uniBatch = device.createBuffer({ size: BATCH_MAX * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const idsBatch = empty(BATCH_MAX * 4);
  const stagingIds = device.createBuffer({ size: BATCH_MAX * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const embRowsCpu = new Float32Array(BATCH_MAX * d);
  const uniCpu = new Uint32Array(BATCH_MAX * 4);
  const dequantRow = (token: number, dst: Float32Array, at: number): void => {
    const sub = dst.subarray(at, at + d);
    if (embdKind === "q6k") dequantQ6_K(embdRaw, token * rowBytes, rowBlocks, sub);
    else if (embdKind === "q80") dequantQ8_0(embdRaw, token * rowBytes, rowBlocks, sub);
    else dequantQ4_0(embdRaw, token * rowBytes, rowBlocks, sub);
  };

  const runSeg = (from: number, to: number, tail?: (enc: GPUCommandEncoder) => void): void => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = from; i < to; i++) {
      const st = steps[i];
      pass.setPipeline(st.pipe);
      pass.setBindGroup(0, st.bind);
      pass.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
    }
    pass.end();
    tail?.(enc);
    device.queue.submit([enc.finish()]);
    perfAcc.submits++;
  };
  const readStaging = async (b: GPUBuffer, bytes: number): Promise<Float32Array> => {
    perfAcc.readbacks++;
    await b.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(b.getMappedRange().slice(0, bytes));
    b.unmap();
    return out;
  };

  return {
    dispatchesPerToken,
    perf: () => ({ ...perfAcc }),
    decodeBatch: moe ? null : async (tokens: ArrayLike<number>, posStart: number): Promise<number[]> => {
      const k = tokens.length;
      if (k < 1 || k > BATCH_MAX) throw new Error(`q35 decodeBatch: k=${k} fuori da [1, ${BATCH_MAX}]`);
      const tEmb = performance.now();
      for (let i = 0; i < k; i++) {
        dequantRow(tokens[i] as number, embRowsCpu, i * d);
        uniCpu[i * 4] = posStart + i;
        uniCpu[i * 4 + 1] = posStart + i;
      }
      device.queue.writeBuffer(embBatch, 0, embRowsCpu, 0, k * d);
      device.queue.writeBuffer(uniBatch, 0, uniCpu, 0, k * 4);
      perfAcc.embedMs += performance.now() - tEmb;
      perfAcc.tokens += k;
      // Error scope come CONTRATTO di ogni percorso di encode nuovo (landmine
      // B1: una pipeline invalida fa droppare i submit IN SILENZIO, e i
      // readback tornano valori stale ma plausibili).
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      const enc = device.createCommandEncoder();
      for (let i = 0; i < k; i++) {
        enc.copyBufferToBuffer(embBatch, i * d * 4, x, 0, d * 4);
        enc.copyBufferToBuffer(uniBatch, i * 16, uni, 0, 16);
        const pass = enc.beginComputePass();
        for (const st of steps) {
          pass.setPipeline(st.pipe);
          pass.setBindGroup(0, st.bind);
          pass.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
        }
        pass.setPipeline(pAmax1); pass.setBindGroup(0, bgAmax1); pass.dispatchWorkgroups(N_PARTIALS);
        pass.setPipeline(pAmax2); pass.setBindGroup(0, bgAmax2); pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(amaxOut, 0, idsBatch, i * 4, 4);
      }
      enc.copyBufferToBuffer(idsBatch, 0, stagingIds, 0, k * 4);
      device.queue.submit([enc.finish()]);
      const errOom = await device.popErrorScope();
      const errVal = await device.popErrorScope();
      if (errOom ?? errVal) throw new Error(`q35 decodeBatch error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
      await stagingIds.mapAsync(GPUMapMode.READ);
      const ids = [...new Uint32Array(stagingIds.getMappedRange()).subarray(0, k)];
      stagingIds.unmap();
      return ids;
    },
    async step(token: number, pos: number, read = true): Promise<number> {
      const tEmb = performance.now();
      if (embdKind === "q6k") dequantQ6_K(embdRaw, token * rowBytes, rowBlocks, embRow);
      else if (embdKind === "q80") dequantQ8_0(embdRaw, token * rowBytes, rowBlocks, embRow);
      else dequantQ4_0(embdRaw, token * rowBytes, rowBlocks, embRow);
      device.queue.writeBuffer(x, 0, embRow);
      perfAcc.embedMs += performance.now() - tEmb;
      perfAcc.tokens++;
      device.queue.writeBuffer(uni, 0, new Uint32Array([pos, pos, 0, 0]));
      if (moe && optimisticOn) {
        // UN submit per il token intero: nessun readback del router, la
        // selezione la fa e la risolve la GPU. I logits arrivano dalla stessa
        // attesa che porta `Sel` e `dirtyB` (fetta 3c).
        const lgOpt = await moe.runTokenOptimistic!(read);
        if (!read || !lgOpt) return -1;
        const tAmOpt = performance.now();
        let bestO = -Infinity, biO = -1;
        for (let i = 0; i < S.vocab; i++) if (lgOpt[i] > bestO) { bestO = lgOpt[i]; biO = i; }
        perfAcc.argmaxMs += performance.now() - tAmOpt;
        return biO;
      }
      if (!moe) {
        runSeg(0, steps.length, read ? (enc) => enc.copyBufferToBuffer(logits, 0, staging, 0, S.vocab * 4) : undefined);
      } else {
        tapWanted = tapLayer;
        // MoE segmentato (correttezza-prima, 1 readback router/layer DICHIARATO):
        // per layer: zero moeAcc → segmento statico (attn+shexp+router) → read
        // logits router → selezione CPU + ensure + dispatch dinamici.
        let from = 0;
        for (let li = 0; li < cuts.length; li++) {
          device.queue.writeBuffer(moeAcc, 0, zeroAcc);
          runSeg(from, cuts[li], (enc) => {
            // OMBRA (fetta 3b): il router GPU gira nello STESSO submit del
            // segmento statico, in coda al GEMV che ha appena scritto i logits.
            // Scrive la regione d'ombra di Sel, che nessun kernel expert legge.
            moe.shadowEncode?.(enc, li);
            enc.copyBufferToBuffer(routerLogits, 0, routerStaging, 0, nE * 4);
            if (tapWanted === -100 - li) enc.copyBufferToBuffer(x, 0, tapStaging, 0, d * 4);
          });
          if (tapWanted === -100 - li) tapValue = await readStaging(tapStaging, d * 4);
          const lg = await readStaging(routerStaging, nE * 4);
          await moe.runLayer(li, lg);
          // dopo runLayer: il confronto vuole la selezione CPU e i suoi slot
          await moe.shadowCompare?.();
          if (tapWanted === li) {
            const enc2 = device.createCommandEncoder();
            enc2.copyBufferToBuffer(x, 0, tapStaging, 0, d * 4);
            device.queue.submit([enc2.finish()]);
            tapValue = await readStaging(tapStaging, d * 4);
          }
          from = cuts[li];
        }
        runSeg(from, steps.length, read ? (enc) => enc.copyBufferToBuffer(logits, 0, staging, 0, S.vocab * 4) : undefined);
      }
      if (!read) return -1;
      const tRb = performance.now();
      const lg = await readStaging(staging, S.vocab * 4);
      const tAm = performance.now();
      perfAcc.readbackMs += tAm - tRb;
      let best = -Infinity, bi = -1;
      for (let i = 0; i < S.vocab; i++) if (lg[i] > best) { best = lg[i]; bi = i; }
      perfAcc.argmaxMs += performance.now() - tAm;
      return bi;
    },
    async readTap(layer: number): Promise<Float32Array> {
      tapLayer = layer;
      const v = tapValue;
      tapValue = null;
      return v ?? new Float32Array(0);
    },
    setOptimistic(on: boolean): void {
      if (on && !moe?.runTokenOptimistic) {
        throw new Error('q35gpumodel: setOptimistic(true) su un modello costruito senza select "optimistic"');
      }
      optimisticOn = on;
    },
    routerShadowStats: moe?.shadowStats ? () => moe!.shadowStats!() : null,
    moeStats: moe
      ? () => ({
          ...moe!.stats(),
          routing: Object.fromEntries([...moe!.routing.entries()].map(([k2, v2]) => [String(k2), v2])),
          nSlots: moe!.nSlots, parkSlots: moe!.parkSlots, slotBytes: moe!.slotBytes,
        })
      : null,
    resetState(): void {
      for (const s of stateBufs) device.queue.writeBuffer(s.buf, 0, new Float32Array(s.bytes / 4));
    },
    destroy(): void {
      for (const [, p] of pipes) void p;
      moe?.destroy(); // l'arena expert è VRAM vera: senza questa il modello la teneva
    },
  };
}

/** byte attesi di un tensore (comodo per i reader Range). */
export function q35TensorBytes(t: GgufTensorInfo): number {
  return tensorByteSize(t);
}
