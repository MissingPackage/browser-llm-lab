// Load GPU + forward decode di Qwen2.5-0.5B — assemblatore "correttezza prima".
//
// Orchestrazione deliberatamente semplice: op non fuse (~340 dispatch/token), MA con
// bind group precostruiti al load e un solo submit per token — nel NOSTRO motore i
// buffer hanno indirizzi statici, quindi L1/L2 sono gratis by construction (è la tesi:
// ciò che in WebLLM richiede rework, qui è il default). La fusione (L3) arriva col
// piano statico vero. Questo modulo è il gate di parità (conformance) e lo scaffolding
// di debug dei kernel fusi.
import { parseGguf, GGML_TYPE, type GgufTensorInfo } from "./gguf";
import { repackQ4_0, repackQ8_0, dequantQ4_0Row } from "./quant";
import { QWEN25_05B as S, validateQwen25_05B } from "./shape";
import {
  gemvQuantWgsl, rmsnormWgsl, ropeNeoxWgsl, kvAppendWgsl, attnDecodeWgsl,
  siluMulWgsl, addInPlaceWgsl, argmaxStage1Wgsl, argmaxStage2Wgsl, ARGMAX_CHUNK,
  gemvGrid, attnFusedWgsl,
  rmsGemvQ8FastWgsl, rmsPairGemvSiluFastWgsl, gemvResidualFastWgsl, rmsGemvQ4FastBiasWgsl,
} from "./kernels/wgsl";
import type { RepackedQuant } from "./quant";

export const CTX_MAX = 1024; // bench: prompt ~469 + warmup/gen 256 (scores in shared: 4 KB, ok)
const KV_DIM = S.nKvHead * S.headDim;
const N_PARTIALS = Math.ceil(S.vocab / ARGMAX_CHUNK);

interface QuantBufs { qs: GPUBuffer; scales: GPUBuffer }

export interface EngineHandle {
  device: GPUDevice;
  forwardToken(token: number, pos: number): Promise<number>; // ritorna argmax id
  readLogits(): Promise<Float32Array>;
  // Tap hidden-states (contratto DeepSpec, spec §Tap): hidden post-layer dei layer
  // richiesti in opts.taps, copiati a ogni forward. Zero step emessi se taps=[].
  readTap(layer: number): Promise<Float32Array>;
  reset(): void;
  dispatchesPerToken: number;
  destroy(): void;
}

export async function createEngine(
  gguf: ArrayBuffer,
  onProgress: (text: string, frac: number) => void,
  opts: { taps?: number[]; fused?: boolean } = {},
): Promise<EngineHandle> {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU non disponibile");
  // requiredLimits ESPLICITI (landmine: il default a 128 MiB non basta per l'lm_head
  // Q8_0 da ~145 MB e produrrebbe garbage/errore).
  const lim = adapter.limits;
  const need = 256 * 1024 * 1024;
  if (lim.maxStorageBufferBindingSize < need) throw new Error("adapter: maxStorageBufferBindingSize insufficiente");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: need,
      maxBufferSize: need,
      // il kernel down-proj tiene xin (4864 f32) in shared: 19.7 KB > default 16 KB.
      // 32 KB è nel range di tutti i device target (S22: 32768, probe fase-1b).
      maxComputeWorkgroupStorageSize: Math.min(lim.maxComputeWorkgroupStorageSize, 32768),
    },
  });
  // Gli errori di validazione WebGPU sono silenziosi by default e trasformano ogni
  // submit in un no-op (visto in fase di bring-up: grid > 65535 ⇒ top-1 0.2% muto).
  // Qui urlano.
  device.addEventListener("uncapturederror", (e) => {
    const msg = (e as GPUUncapturedErrorEvent).error.message;
    console.error("[engine][gpu-error]", msg.slice(0, 400));
    throw new Error(`GPU error: ${msg.slice(0, 200)}`);
  });

  const f = parseGguf(gguf);
  const byName = validateQwen25_05B(f);
  const bytes = new Uint8Array(gguf);

  const upload = (data: Uint32Array | Float32Array, usage = GPUBufferUsage.STORAGE): GPUBuffer => {
    const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data as BufferSource);
    return b;
  };
  const storage = (bytesLen: number, extra = 0): GPUBuffer =>
    device.createBuffer({ size: Math.max(16, bytesLen), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra });

  const quantBufs = (t: GgufTensorInfo): QuantBufs => {
    const nBlocks = t.dims.reduce((a, b) => a * b, 1) / 32;
    const off = f.dataOffset + t.offset;
    const rp = t.type === GGML_TYPE.Q4_0 ? repackQ4_0(bytes, off, nBlocks) : repackQ8_0(bytes, off, nBlocks);
    return { qs: upload(rp.qs), scales: upload(rp.scales) };
  };
  const f32Buf = (t: GgufTensorInfo): GPUBuffer => {
    const elems = t.dims.reduce((a, b) => a * b, 1);
    return upload(new Float32Array(gguf, f.dataOffset + t.offset, elems));
  };

  // Concat QKV al repack (piano fuso): righe q|k|v in un solo qs/scales — i
  // conteggi blocchi per tensore sono pari, quindi le scale impacchettate a coppie
  // si concatenano senza ricodifica.
  const repackConcat = (ts: GgufTensorInfo[]): { qs: Uint32Array; scales: Uint32Array } => {
    const parts: RepackedQuant[] = ts.map((t) => {
      const nBlocks = t.dims.reduce((a, b) => a * b, 1) / 32;
      if (nBlocks % 2 !== 0) throw new Error(`repackConcat: blocchi dispari in ${t.name}`);
      return repackQ4_0(bytes, f.dataOffset + t.offset, nBlocks);
    });
    const qs = new Uint32Array(parts.reduce((a, r) => a + r.qs.length, 0));
    const scales = new Uint32Array(parts.reduce((a, r) => a + r.scales.length, 0));
    let qo = 0, so = 0;
    for (const r of parts) { qs.set(r.qs, qo); scales.set(r.scales, so); qo += r.qs.length; so += r.scales.length; }
    return { qs, scales };
  };

  // --- pesi su GPU ---
  const T = (name: string) => byName.get(name)!;
  onProgress("upload pesi…", 0);
  const layers = [];
  for (let l = 0; l < S.nLayer; l++) {
    const p = (n: string) => T(`blk.${l}.${n}`);
    const qkvConcat = repackConcat([p("attn_q.weight"), p("attn_k.weight"), p("attn_v.weight")]);
    const biasCat = new Float32Array(S.dModel + 2 * KV_DIM);
    biasCat.set(new Float32Array(gguf, f.dataOffset + p("attn_q.bias").offset, S.dModel), 0);
    biasCat.set(new Float32Array(gguf, f.dataOffset + p("attn_k.bias").offset, KV_DIM), S.dModel);
    biasCat.set(new Float32Array(gguf, f.dataOffset + p("attn_v.bias").offset, KV_DIM), S.dModel + KV_DIM);
    layers.push({
      qkv: { qs: upload(qkvConcat.qs), scales: upload(qkvConcat.scales) },
      qkvBias: upload(biasCat),
      attnNorm: f32Buf(p("attn_norm.weight")),
      wq: quantBufs(p("attn_q.weight")), bq: f32Buf(p("attn_q.bias")),
      wk: quantBufs(p("attn_k.weight")), bk: f32Buf(p("attn_k.bias")),
      wv: quantBufs(p("attn_v.weight")), bv: f32Buf(p("attn_v.bias")),
      wo: quantBufs(p("attn_output.weight")),
      ffnNorm: f32Buf(p("ffn_norm.weight")),
      wGate: quantBufs(p("ffn_gate.weight")),
      wUp: quantBufs(p("ffn_up.weight")),
      wDown: quantBufs(p("ffn_down.weight")),
      kCache: storage(CTX_MAX * KV_DIM * 4),
      vCache: storage(CTX_MAX * KV_DIM * 4),
    });
    onProgress(`upload pesi… layer ${l + 1}/${S.nLayer}`, (l + 1) / (S.nLayer + 1));
  }
  const outNorm = f32Buf(T("output_norm.weight"));
  const wOut = quantBufs(T("output.weight"));
  // token_embd resta CPU-side: una riga dequantizzata per token via writeBuffer
  const embdInfo = T("token_embd.weight");
  const embdOff = f.dataOffset + embdInfo.offset;
  const embdRow = new Float32Array(S.dModel);
  onProgress("compilazione pipeline…", 0.96);

  // --- attivazioni ---
  const x = storage(S.dModel * 4);
  const hn = storage(S.dModel * 4);
  const qB = storage(S.dModel * 4);
  const qkvB = storage((S.dModel + 2 * KV_DIM) * 4);
  const kB = storage(KV_DIM * 4);
  const vB = storage(KV_DIM * 4);
  const attnOut = storage(S.dModel * 4);
  const tmpD = storage(S.dModel * 4);       // out di o-proj / ffn-down
  const gateB = storage(S.dFfn * 4);
  const upB = storage(S.dFfn * 4);
  const logits = storage(S.vocab * 4);
  const pmax = storage(N_PARTIALS * 4);
  const pidx = storage(N_PARTIALS * 4);
  const amaxOut = storage(16);
  const P = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const stagingId = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  // --- pipeline (15 varianti) + bind group precostruiti ---
  const mkPipe = (code: string) =>
    device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const pipes = {
    rms: mkPipe(rmsnormWgsl(S.dModel, S.rmsEps)),
    gemvQDD_b: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: S.dModel, N: S.dModel, hasBias: true })),
    gemvQDKV_b: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: S.dModel, N: KV_DIM, hasBias: true })),
    gemvQDD: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: S.dModel, N: S.dModel, hasBias: false })),
    gemvQDF: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: S.dModel, N: S.dFfn, hasBias: false })),
    gemvQFD: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: S.dFfn, N: S.dModel, hasBias: false })),
    gemvOut: mkPipe(gemvQuantWgsl({ kind: "q8_0", K: S.dModel, N: S.vocab, hasBias: false })),
    ropeQ: mkPipe(ropeNeoxWgsl(S.nHead, S.headDim, S.ropeFreqBase)),
    ropeK: mkPipe(ropeNeoxWgsl(S.nKvHead, S.headDim, S.ropeFreqBase)),
    kvApp: mkPipe(kvAppendWgsl(KV_DIM)),
    attn: mkPipe(attnDecodeWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: S.headDim, ctxMax: CTX_MAX })),
    silu: mkPipe(siluMulWgsl(S.dFfn)),
    add: mkPipe(addInPlaceWgsl(S.dModel)),
    am1: mkPipe(argmaxStage1Wgsl(S.vocab)),
    am2: mkPipe(argmaxStage2Wgsl(N_PARTIALS)),
    // fusi (L3)
    rmsQkv: mkPipe(rmsGemvQ4FastBiasWgsl({ K: S.dModel, N: S.dModel + 2 * KV_DIM, eps: S.rmsEps })),
    attnF: mkPipe(attnFusedWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: S.headDim, ctxMax: CTX_MAX, freqBase: S.ropeFreqBase })),
    oResid: mkPipe(gemvResidualFastWgsl({ K: S.dModel, N: S.dModel })),
    pairSilu: mkPipe(rmsPairGemvSiluFastWgsl({ K: S.dModel, N: S.dFfn, eps: S.rmsEps })),
    downResid: mkPipe(gemvResidualFastWgsl({ K: S.dFfn, N: S.dModel })),
    rmsLmHead: mkPipe(rmsGemvQ8FastWgsl({ K: S.dModel, N: S.vocab, eps: S.rmsEps })),
  };
  const bg = (pipe: GPUComputePipeline, bufs: GPUBuffer[], uni?: GPUBuffer) =>
    device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        ...bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
        ...(uni ? [{ binding: bufs.length, resource: { buffer: uni } }] : []),
      ],
    });

  type Step =
    | { kind: "compute"; pipe: GPUComputePipeline; bind: GPUBindGroup; wgs: [number, number] }
    | { kind: "copy"; src: GPUBuffer; dst: GPUBuffer; bytes: number };
  const steps: Step[] = [];
  const push = (pipe: GPUComputePipeline, bufs: GPUBuffer[], wgs: number | [number, number], uni?: GPUBuffer) =>
    steps.push({ kind: "compute", pipe, bind: bg(pipe, bufs, uni), wgs: typeof wgs === "number" ? [wgs, 1] : wgs });
  // Tap hidden-states: buffer dedicato per layer richiesto; la copy vive fra i
  // dispatch (chiude e riapre il pass), emessa SOLO se il tap e' attivo.
  const tapBufs = new Map<number, GPUBuffer>();
  for (const l of opts.taps ?? []) {
    if (l < 0 || l >= S.nLayer) throw new Error(`tap fuori range: ${l}`);
    tapBufs.set(l, storage(S.dModel * 4));
  }

  const fused = opts.fused ?? true;
  for (const L of layers) {
    if (fused) {
      push(pipes.rmsQkv, [L.qkv.qs, L.qkv.scales, x, L.attnNorm, qkvB, L.qkvBias], gemvGrid((S.dModel + 2 * KV_DIM) / 4));
      push(pipes.attnF, [qkvB, L.kCache, L.vCache, attnOut], S.nHead, P);
      push(pipes.oResid, [L.wo.qs, L.wo.scales, attnOut, x], gemvGrid(S.dModel / 4));
      push(pipes.pairSilu, [L.wGate.qs, L.wGate.scales, L.wUp.qs, L.wUp.scales, x, L.ffnNorm, gateB], gemvGrid(S.dFfn / 4));
      push(pipes.downResid, [L.wDown.qs, L.wDown.scales, gateB, x], gemvGrid(S.dModel / 4));
      const tapDstF = tapBufs.get(layers.indexOf(L));
      if (tapDstF) steps.push({ kind: "copy", src: x, dst: tapDstF, bytes: S.dModel * 4 });
      continue;
    }
    push(pipes.rms, [x, L.attnNorm, hn], 1);
    push(pipes.gemvQDD_b, [L.wq.qs, L.wq.scales, hn, qB, L.bq], gemvGrid(S.dModel));
    push(pipes.gemvQDKV_b, [L.wk.qs, L.wk.scales, hn, kB, L.bk], gemvGrid(KV_DIM));
    push(pipes.gemvQDKV_b, [L.wv.qs, L.wv.scales, hn, vB, L.bv], gemvGrid(KV_DIM));
    push(pipes.ropeQ, [qB], Math.ceil((S.nHead * S.headDim / 2) / 64), P);
    push(pipes.ropeK, [kB], Math.ceil((S.nKvHead * S.headDim / 2) / 64), P);
    push(pipes.kvApp, [kB, L.kCache], Math.ceil(KV_DIM / 64), P);
    push(pipes.kvApp, [vB, L.vCache], Math.ceil(KV_DIM / 64), P);
    push(pipes.attn, [qB, L.kCache, L.vCache, attnOut], S.nHead, P);
    push(pipes.gemvQDD, [L.wo.qs, L.wo.scales, attnOut, tmpD], gemvGrid(S.dModel));
    push(pipes.add, [x, tmpD], Math.ceil(S.dModel / 64));
    push(pipes.rms, [x, L.ffnNorm, hn], 1);
    push(pipes.gemvQDF, [L.wGate.qs, L.wGate.scales, hn, gateB], gemvGrid(S.dFfn));
    push(pipes.gemvQDF, [L.wUp.qs, L.wUp.scales, hn, upB], gemvGrid(S.dFfn));
    push(pipes.silu, [gateB, upB], Math.ceil(S.dFfn / 64));
    push(pipes.gemvQFD, [L.wDown.qs, L.wDown.scales, gateB, tmpD], gemvGrid(S.dModel));
    push(pipes.add, [x, tmpD], Math.ceil(S.dModel / 64));
    const tapDst = tapBufs.get(layers.indexOf(L));
    if (tapDst) steps.push({ kind: "copy", src: x, dst: tapDst, bytes: S.dModel * 4 });
  }
  if (fused) {
    push(pipes.rmsLmHead, [wOut.qs, wOut.scales, x, outNorm, logits], gemvGrid(S.vocab / 4));
  } else {
    push(pipes.rms, [x, outNorm, hn], 1);
    push(pipes.gemvOut, [wOut.qs, wOut.scales, hn, logits], gemvGrid(S.vocab));
  }
  push(pipes.am1, [logits, pmax, pidx], N_PARTIALS);
  push(pipes.am2, [pmax, pidx, amaxOut], 1);
  onProgress("pronto", 1);

  return {
    device,
    dispatchesPerToken: steps.filter((st) => st.kind === "compute").length,
    reset() { /* le cache si sovrascrivono per posizione: basta ripartire da pos 0 */ },
    async forwardToken(token: number, pos: number): Promise<number> {
      if (pos >= CTX_MAX) throw new Error("contesto pieno");
      dequantQ4_0Row(bytes, embdOff, S.dModel, token, embdRow);
      device.queue.writeBuffer(x, 0, embdRow as unknown as BufferSource);
      device.queue.writeBuffer(P, 0, new Uint32Array([pos, pos, 0, 0]));
      const enc = device.createCommandEncoder();
      let pass = enc.beginComputePass();
      for (const s of steps) {
        if (s.kind === "copy") {
          pass.end();
          enc.copyBufferToBuffer(s.src, 0, s.dst, 0, s.bytes);
          pass = enc.beginComputePass();
          continue;
        }
        pass.setPipeline(s.pipe);
        pass.setBindGroup(0, s.bind);
        pass.dispatchWorkgroups(s.wgs[0], s.wgs[1]);
      }
      pass.end();
      enc.copyBufferToBuffer(amaxOut, 0, stagingId, 0, 4);
      device.queue.submit([enc.finish()]);
      await stagingId.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(stagingId.getMappedRange())[0];
      stagingId.unmap();
      return id;
    },
    async readTap(layer: number): Promise<Float32Array> {
      const src = tapBufs.get(layer);
      if (!src) throw new Error(`tap non attivo per layer ${layer}`);
      const staging = device.createBuffer({ size: S.dModel * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, S.dModel * 4);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.destroy();
      return out;
    },
    async readLogits(): Promise<Float32Array> {
      const staging = device.createBuffer({ size: S.vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(logits, 0, staging, 0, S.vocab * 4);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.destroy();
      return out;
    },
    destroy() { device.destroy(); },
  };
}
