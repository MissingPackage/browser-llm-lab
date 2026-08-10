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
  siluMulWgsl, stridedCopyWgsl,
} from "./kernels/wgsl";
import { deltaNetConvWgsl, deltaNetCoreWgsl, deltaNetGatesWgsl } from "./kernels/deltanet";
import { GGML_TYPE, tensorByteSize, type GgufTensorInfo } from "./gguf";
import { dequantQ4_0, dequantQ6_K, dequantQ8_0, repackKQuant, repackQ4_0, repackQ4_1, repackQ8_0, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES } from "./quant";
import { q35IsFullAttn, type Q35Shape } from "./q35shape";
import { ROUTER_QWEN35MOE, routerSelect, type RouterConfig } from "./moe";
import {
  ExpertCache, moeParkOf, slotTensorRanges, type ExpertRawBytes, type SlotRef,
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
   */
  perf(): { tokens: number; embedMs: number; readbackMs: number; argmaxMs: number };
  /** DEBUG (it.17): dopo step(), hidden x a valle del layer indicato (solo MoE). */
  readTap(layer: number): Promise<Float32Array>;
  destroy(): void;
}

export async function createQ35GpuModel(device: GPUDevice, r: Q35RawReader, ctxMax = 64, arenaBudgetBytes = 12 * (1 << 30)): Promise<Q35GpuModel> {
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
    });
    const routerCfg: RouterConfig = { ...ROUTER_QWEN35MOE, nUsed: topK };
    // i kernel si scelgono dal LAYOUT della classe, non da un'assunzione sul
    // formato: se un GGUF della famiglia arrivasse con gate/up diversi, qui
    // si ferma con un messaggio invece di dequantizzare byte sbagliati.
    const gk = cfg.layout(cfg.classes[0]).gate.kind;
    if (gk !== "q4_K") throw new Error(`q35 MoE: nessun kernel gemv per gate/up ${gk}`);
    const pGate = pipe(gemvQ4KWgsl({ K: d, N: dE }));
    const pSilu = pipe(siluMulWgsl(dE));
    const pDown: Record<string, GPUComputePipeline> = {};
    for (const c of cfg.classes) {
      const dk = cfg.layout(c).down.kind;
      if (dk !== "q4_K" && dk !== "q6_K") throw new Error(`q35 MoE: nessun kernel gemv per down ${dk}`);
      pDown[c] = pipe(dk === "q6_K" ? gemvQ6KWgsl({ K: dE, N: d }) : gemvQ4KWgsl({ K: dE, N: d }));
    }
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
    // I bind group dipendono dallo SLOT (buffer + offset), non dall'expert che
    // ci abita: si costruiscono una volta per slot e non si invalidano mai —
    // l'eviction cambia i byte, non l'indirizzo.
    const bgBySlot = new Map<string, GPUBindGroup[]>();
    const slotBgs = (s: SlotRef): GPUBindGroup[] => {
      const k = `${s.cls}:${s.idx}`;
      let b2 = bgBySlot.get(k);
      if (!b2) {
        const rg = slotTensorRanges(s);
        b2 = [mkBg(pGate, [rg.gate, xn, gateE]), mkBg(pGate, [rg.up, xn, upE]), mkBg(pDown[s.cls], [rg.down, gateE, dnE])];
        bgBySlot.set(k, b2);
      }
      return b2;
    };
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
      destroy: () => cache.destroy(),
      async runLayer(l: number, logitsF32: Float32Array): Promise<void> {
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
        for (let k2 = 0; k2 < topK; k2++) {
          device.queue.writeBuffer(wBufs[k2], 0, new Float32Array([sel.weights[k2], 0, 0, 0]));
        }
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        const disp = (p2: GPUComputePipeline, b2: GPUBindGroup, wg: [number, number, number]): void => {
          pass.setPipeline(p2);
          pass.setBindGroup(0, b2);
          pass.dispatchWorkgroups(wg[0], wg[1], wg[2]);
        };
        const gg2 = gemvGrid(d);
        for (let k2 = 0; k2 < topK; k2++) {
          const [bGate, bUp, bDown] = slotBgs(slots[k2]);
          disp(pGate, bGate, [dE, 1, 1]);
          disp(pGate, bUp, [dE, 1, 1]);
          disp(pSilu, bgSilu, [Math.ceil(dE / 64), 1, 1]);
          disp(pDown[slots[k2].cls], bDown, [gg2[0], gg2[1], 1]);
          disp(pAxpy, bgAxpy[k2], [Math.ceil(d / 64), 1, 1]);
        }
        disp(pAdd, bgAddRes, [Math.ceil(d / 64), 1, 1]); // x += moeAcc (shexp + expert)
        pass.end();
        device.queue.submit([enc.finish()]);
      },
    };
  }
  const perfAcc = { tokens: 0, embedMs: 0, readbackMs: 0, argmaxMs: 0 };
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
  };
  const readStaging = async (b: GPUBuffer, bytes: number): Promise<Float32Array> => {
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
            enc.copyBufferToBuffer(routerLogits, 0, routerStaging, 0, nE * 4);
            if (tapWanted === -100 - li) enc.copyBufferToBuffer(x, 0, tapStaging, 0, d * 4);
          });
          if (tapWanted === -100 - li) tapValue = await readStaging(tapStaging, d * 4);
          const lg = await readStaging(routerStaging, nE * 4);
          await moe.runLayer(li, lg);
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
