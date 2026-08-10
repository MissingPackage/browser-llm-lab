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
  addInPlaceWgsl, attnDecodeWgsl, axpyWgsl, gemvF32Wgsl, gemvGrid, gemvQ4KWgsl, gemvQ5KWgsl,
  gemvQ6KWgsl, gemvQuantWgsl, kvAppendWgsl, rmsnormWgsl, ropeNeoxWgsl, sigmoidMulWgsl,
  siluMulWgsl, stridedCopyWgsl,
} from "./kernels/wgsl";
import { deltaNetConvWgsl, deltaNetCoreWgsl, deltaNetGatesWgsl } from "./kernels/deltanet";
import { GGML_TYPE, tensorByteSize, type GgufTensorInfo } from "./gguf";
import { dequantQ4_0, dequantQ6_K, dequantQ8_0, repackKQuant, repackQ4_0, repackQ4_1, repackQ8_0, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES } from "./quant";
import { q35IsFullAttn, type Q35Shape } from "./q35shape";

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
  /** azzera gli stati ricorrenti (conv + S) di tutti i layer linear: nuovo prompt. */
  resetState(): void;
  dispatchesPerToken: number;
  /** stats MoE (null sui densi): arena, routing esatto, residenza. */
  moeStats: (() => { hits: number; misses: number; uploadedBytes: number; routing: Record<string, number>; nSlots: Record<string, number>; parkSlots: Record<string, number>; slotBytes: Record<string, number> }) | null;
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
// ============ arena MoE + LRU + esecuzione dinamica (it.17) ============
  interface MoeRt {
    stats: { hits: number; misses: number; uploadedBytes: number; routing: Map<number, number> };
    nSlots: Record<string, number>; parkSlots: Record<string, number>; slotBytes: Record<string, number>;
    runLayer(l: number, logitsF32: Float32Array): Promise<void>;
  }
  let moe: MoeRt | null = null;
  if (isMoe) {
    const readRange = r.readRange!.bind(r);
    const classOf = (l: number): "q4k" | "q6k" => (r.info(`blk.${l}.ffn_down_exps.weight`).type === GGML_TYPE.Q6_K ? "q6k" : "q4k");
    const gT = r.info("blk.0.ffn_gate_exps.weight");
    const gateRp = ((gT.dims[0] * gT.dims[1]) / 256) * 144; // gate/up: Q4_K su tutti i layer (header dump)
    const downRawPer = (l: number): number => {
      const t = r.info(`blk.${l}.ffn_down_exps.weight`);
      return ((t.dims[0] * t.dims[1]) / 256) * (t.type === GGML_TYPE.Q6_K ? 210 : 144);
    };
    const downRpPer = (l: number): number => {
      const t = r.info(`blk.${l}.ffn_down_exps.weight`);
      return ((t.dims[0] * t.dims[1]) / 256) * (t.type === GGML_TYPE.Q6_K ? 212 : 144);
    };
    const slotBytes: Record<string, number> = { q4k: 0, q6k: 0 };
    const parkSlots: Record<string, number> = { q4k: 0, q6k: 0 };
    for (let l = 0; l < S.nLayer; l++) {
      const c = classOf(l);
      parkSlots[c] += nE;
      if (!slotBytes[c]) slotBytes[c] = 2 * gateRp + downRpPer(l);
    }
    const totPark = parkSlots.q4k * slotBytes.q4k + parkSlots.q6k * slotBytes.q6k;
    const frac = Math.min(1, arenaBudgetBytes / totPark);
    const nSlots: Record<string, number> = {
      q4k: Math.max(topK * 2, Math.floor(parkSlots.q4k * frac)),
      q6k: Math.max(topK * 2, Math.floor(parkSlots.q6k * frac)),
    };
    // Arena A CHUNK (it.17 root-cause: un buffer monolitico da ~11 GB sfora
    // maxBufferSize — l'adapter stesso cappa a ~4 GB): chunk ≤ 2 GiB, slot →
    // (chunk, offset locale). I needs del device devono chiedere
    // slabClassBytes ≥ CHUNK (il conf worker lo fa).
    const CHUNK = 2 * (1 << 30);
    const slotsPerChunk: Record<string, number> = {
      q4k: Math.max(1, Math.floor(CHUNK / slotBytes.q4k)),
      q6k: Math.max(1, Math.floor(CHUNK / slotBytes.q6k)),
    };
    const mkChunks = (cls: "q4k" | "q6k"): GPUBuffer[] => {
      const n = nSlots[cls];
      const per = slotsPerChunk[cls];
      const out: GPUBuffer[] = [];
      for (let i = 0; i < Math.ceil(n / per); i++) {
        out.push(empty(Math.min(per, n - i * per) * slotBytes[cls]));
      }
      return out;
    };
    const arenaChunks: Record<string, GPUBuffer[]> = { q4k: mkChunks("q4k"), q6k: mkChunks("q6k") };
    const slotLoc = (cls: "q4k" | "q6k", slot: number): { buf: GPUBuffer; base: number } => {
      const per = slotsPerChunk[cls];
      return { buf: arenaChunks[cls][Math.floor(slot / per)], base: (slot % per) * slotBytes[cls] };
    };
    const pGate = pipe(gemvQ4KWgsl({ K: d, N: dE }));
    const pSilu = pipe(siluMulWgsl(dE));
    const pDown: Record<string, GPUComputePipeline> = {
      q4k: pipe(gemvQ4KWgsl({ K: dE, N: d })),
      q6k: pipe(gemvQ6KWgsl({ K: dE, N: d })),
    };
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
    interface SlotState { byKey: Map<number, number>; lru: Map<number, number>; bgs: (GPUBindGroup[] | null)[] }
    const slots: Record<string, SlotState> = {
      q4k: { byKey: new Map(), lru: new Map(), bgs: Array(nSlots.q4k).fill(null) },
      q6k: { byKey: new Map(), lru: new Map(), bgs: Array(nSlots.q6k).fill(null) },
    };
    const stats = { hits: 0, misses: 0, uploadedBytes: 0, routing: new Map<number, number>() };
    const rpU8 = (raw: Uint8Array, bb: number): Uint8Array => {
      const w = repackKQuant(raw, 0, raw.length / bb, bb);
      return new Uint8Array(w.buffer, 0, w.byteLength);
    };
    const ensure = async (l: number, e: number): Promise<{ cls: "q4k" | "q6k"; slot: number }> => {
      const cls = classOf(l);
      const st = slots[cls];
      const key = l * nE + e;
      let slot = st.byKey.get(key);
      if (slot !== undefined) {
        stats.hits++;
        st.lru.delete(key);
        st.lru.set(key, slot);
        return { cls, slot };
      }
      stats.misses++;
      if (st.byKey.size < nSlots[cls]) slot = st.byKey.size;
      else {
        const [oldKey, oldSlot] = st.lru.entries().next().value as [number, number];
        st.lru.delete(oldKey);
        st.byKey.delete(oldKey);
        slot = oldSlot;
      }
      const dT = r.info(`blk.${l}.ffn_down_exps.weight`);
      const gRawPer = ((gT.dims[0] * gT.dims[1]) / 256) * 144;
      const [gRaw, uRaw, dRaw] = await Promise.all([
        readRange(`blk.${l}.ffn_gate_exps.weight`, e * gRawPer, gRawPer),
        readRange(`blk.${l}.ffn_up_exps.weight`, e * gRawPer, gRawPer),
        readRange(`blk.${l}.ffn_down_exps.weight`, e * downRawPer(l), downRawPer(l)),
      ]);
      const { buf: chunkBuf, base } = slotLoc(cls, slot);
      const dp = rpU8(dRaw, dT.type === GGML_TYPE.Q6_K ? 210 : 144);
      device.queue.writeBuffer(chunkBuf, base, rpU8(gRaw, 144));
      device.queue.writeBuffer(chunkBuf, base + gateRp, rpU8(uRaw, 144));
      device.queue.writeBuffer(chunkBuf, base + 2 * gateRp, dp);
      stats.uploadedBytes += 2 * gateRp + dp.byteLength;
      st.byKey.set(key, slot);
      st.lru.set(key, slot);
      st.bgs[slot] = null;
      return { cls, slot };
    };
    const slotBgs = (cls: "q4k" | "q6k", slot: number): GPUBindGroup[] => {
      const st = slots[cls];
      let b2 = st.bgs[slot];
      if (!b2) {
        const { buf: chunkBuf, base } = slotLoc(cls, slot);
        b2 = [
          mkBg(pGate, [{ buffer: chunkBuf, offset: base, size: gateRp }, xn, gateE]),
          mkBg(pGate, [{ buffer: chunkBuf, offset: base + gateRp, size: gateRp }, xn, upE]),
          mkBg(pDown[cls], [{ buffer: chunkBuf, offset: base + 2 * gateRp, size: slotBytes[cls] - 2 * gateRp }, gateE, dnE]),
        ];
        st.bgs[slot] = b2;
      }
      return b2;
    };
    moe = {
      stats, nSlots, parkSlots, slotBytes,
      async runLayer(l: number, logitsF32: Float32Array): Promise<void> {
        // selezione CPU: STESSA matematica del cpuref (softmax→topK→norm clamp)
        let mx = -Infinity;
        for (let e = 0; e < nE; e++) if (logitsF32[e] > mx) mx = logitsF32[e];
        const probs = new Float64Array(nE);
        let sm = 0;
        for (let e = 0; e < nE; e++) { probs[e] = Math.exp(logitsF32[e] - mx); sm += probs[e]; }
        for (let e = 0; e < nE; e++) probs[e] /= sm;
        const sel = Array.from({ length: nE }, (_, e2) => e2).sort((a2, b2) => probs[b2] - probs[a2] || a2 - b2).slice(0, topK);
        const wSum = Math.max(sel.reduce((a2, e2) => a2 + probs[e2], 0), 6.103515625e-5);
        const placed: { cls: "q4k" | "q6k"; slot: number }[] = [];
        for (const e2 of sel) {
          stats.routing.set(l * nE + e2, (stats.routing.get(l * nE + e2) ?? 0) + 1);
          placed.push(await ensure(l, e2));
        }
        sel.forEach((e2, k2) => device.queue.writeBuffer(wBufs[k2], 0, new Float32Array([probs[e2] / wSum, 0, 0, 0])));
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        const disp = (p2: GPUComputePipeline, b2: GPUBindGroup, wg: [number, number, number]): void => {
          pass.setPipeline(p2);
          pass.setBindGroup(0, b2);
          pass.dispatchWorkgroups(wg[0], wg[1], wg[2]);
        };
        const gg2 = gemvGrid(d);
        for (let k2 = 0; k2 < topK; k2++) {
          const [bGate, bUp, bDown] = slotBgs(placed[k2].cls, placed[k2].slot);
          disp(pGate, bGate, [dE, 1, 1]);
          disp(pGate, bUp, [dE, 1, 1]);
          disp(pSilu, bgSilu, [Math.ceil(dE / 64), 1, 1]);
          disp(pDown[placed[k2].cls], bDown, [gg2[0], gg2[1], 1]);
          disp(pAxpy, bgAxpy[k2], [Math.ceil(d / 64), 1, 1]);
        }
        disp(pAdd, bgAddRes, [Math.ceil(d / 64), 1, 1]); // x += moeAcc (shexp + expert)
        pass.end();
        device.queue.submit([enc.finish()]);
      },
    };
  }
  const zeroAcc = new Float32Array(d);
  const tapStaging = device.createBuffer({ size: d * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let tapLayer = -1;
  let tapWanted = -1;
  let tapValue: Float32Array | null = null;

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
    async step(token: number, pos: number, read = true): Promise<number> {
      if (embdKind === "q6k") dequantQ6_K(embdRaw, token * rowBytes, rowBlocks, embRow);
      else if (embdKind === "q80") dequantQ8_0(embdRaw, token * rowBytes, rowBlocks, embRow);
      else dequantQ4_0(embdRaw, token * rowBytes, rowBlocks, embRow);
      device.queue.writeBuffer(x, 0, embRow);
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
      const lg = await readStaging(staging, S.vocab * 4);
      let best = -Infinity, bi = -1;
      for (let i = 0; i < S.vocab; i++) if (lg[i] > best) { best = lg[i]; bi = i; }
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
          hits: moe!.stats.hits, misses: moe!.stats.misses, uploadedBytes: moe!.stats.uploadedBytes,
          routing: Object.fromEntries([...moe!.stats.routing.entries()].map(([k2, v2]) => [String(k2), v2])),
          nSlots: moe!.nSlots, parkSlots: moe!.parkSlots, slotBytes: moe!.slotBytes,
        })
      : null,
    resetState(): void {
      for (const s of stateBufs) device.queue.writeBuffer(s.buf, 0, new Float32Array(s.bytes / 4));
    },
    destroy(): void {
      for (const [, p] of pipes) void p;
    },
  };
}

/** byte attesi di un tensore (comodo per i reader Range). */
export function q35TensorBytes(t: GgufTensorInfo): number {
  return tensorByteSize(t);
}
