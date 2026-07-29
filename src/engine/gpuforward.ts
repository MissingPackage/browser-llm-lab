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
  gemvGrid,
} from "./kernels/wgsl";

export const CTX_MAX = 512;
const KV_DIM = S.nKvHead * S.headDim;
const N_PARTIALS = Math.ceil(S.vocab / ARGMAX_CHUNK);

interface QuantBufs { qs: GPUBuffer; scales: GPUBuffer }

export interface EngineHandle {
  device: GPUDevice;
  forwardToken(token: number, pos: number): Promise<number>; // ritorna argmax id
  readLogits(): Promise<Float32Array>;
  reset(): void;
  dispatchesPerToken: number;
  destroy(): void;
}

export async function createEngine(
  gguf: ArrayBuffer,
  onProgress: (text: string, frac: number) => void,
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

  // --- pesi su GPU ---
  const T = (name: string) => byName.get(name)!;
  onProgress("upload pesi…", 0);
  const layers = [];
  for (let l = 0; l < S.nLayer; l++) {
    const p = (n: string) => T(`blk.${l}.${n}`);
    layers.push({
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
  };
  const bg = (pipe: GPUComputePipeline, bufs: GPUBuffer[], uni?: GPUBuffer) =>
    device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        ...bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
        ...(uni ? [{ binding: bufs.length, resource: { buffer: uni } }] : []),
      ],
    });

  type Step = { pipe: GPUComputePipeline; bind: GPUBindGroup; wgs: [number, number] };
  const steps: Step[] = [];
  const push = (pipe: GPUComputePipeline, bufs: GPUBuffer[], wgs: number | [number, number], uni?: GPUBuffer) =>
    steps.push({ pipe, bind: bg(pipe, bufs, uni), wgs: typeof wgs === "number" ? [wgs, 1] : wgs });

  for (const L of layers) {
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
  }
  push(pipes.rms, [x, outNorm, hn], 1);
  push(pipes.gemvOut, [wOut.qs, wOut.scales, hn, logits], gemvGrid(S.vocab));
  push(pipes.am1, [logits, pmax, pidx], N_PARTIALS);
  push(pipes.am2, [pmax, pidx, amaxOut], 1);
  onProgress("pronto", 1);

  return {
    device,
    dispatchesPerToken: steps.length,
    reset() { /* le cache si sovrascrivono per posizione: basta ripartire da pos 0 */ },
    async forwardToken(token: number, pos: number): Promise<number> {
      if (pos >= CTX_MAX) throw new Error("contesto pieno");
      dequantQ4_0Row(bytes, embdOff, S.dModel, token, embdRow);
      device.queue.writeBuffer(x, 0, embdRow as unknown as BufferSource);
      device.queue.writeBuffer(P, 0, new Uint32Array([pos, pos, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const s of steps) {
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
