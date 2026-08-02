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
  rmsGemvQ8FastWgsl, rmsPairGemvSiluFastWgsl, gemvResidualFastWgsl, rmsGemvQ4FastBiasWgsl,
  ropeChunkWgsl, kvAppendChunkWgsl, attnPrefillChunkWgsl,
  rmsGemmQkvChunkFastWgsl, rmsPairGemmSiluChunkFastWgsl, gemmResidChunkFastWgsl,
  attnSplitPartWgsl, attnSplitReduceWgsl, embedGatherQ4Wgsl,
} from "./kernels/wgsl";
import { planPrefill, PREFILL_M } from "./prefillplan";
import { ATTN_CHUNK_P, attnSMax, attnPartialsLen } from "./attnsplit";
import { negotiateLimits } from "./gpulimits";
import { planDecodeBatch, DECODE_K_MAX, DECODE_SLOT_STRIDE } from "./decodebatch";
import { createKvLen } from "./kvlen";
import type { RepackedQuant } from "./quant";

export const CTX_MAX = 1024; // bench: prompt ~469 + warmup/gen 256 (scores in shared: 4 KB, ok)
const KV_DIM = S.nKvHead * S.headDim;
const N_PARTIALS = Math.ceil(S.vocab / ARGMAX_CHUNK);

interface QuantBufs { qs: GPUBuffer; scales: GPUBuffer }

export interface EngineTelemetry {
  forwards: number;
  encodeCpuMsPerToken: number;
  gpuMsPerToken: number | null; // null se timestamp-query assente o spenta
  timestampNote: string;
}

export interface EngineHandle {
  device: GPUDevice;
  // Contratto hard (spec B1 §Crop): pos === kvLen o throw — il length pointer è
  // l'unica verità sulla posizione; chi usava pos libere si rompe by design
  // (usare crop/reset). kvLen++ a forward riuscito. Ritorna l'argmax id.
  forwardToken(token: number, pos: number): Promise<number>;
  // Rollback zero-GPU: kvLen = toLen (toLen ≤ kvLen o throw). Le righe oltre kvLen
  // sono garbage mai letto: si sovrascrivono al prossimo forward per posizione.
  crop(toLen: number): void;
  readonly kvLen: number;
  // Prefix-cache (spec §Formato): payload per-layer [K righe [0,kvLen) | V righe]
  // nel layout ESATTO dell'envelope BKV1 — readKv per il save (copy GPU→staging→CPU),
  // writeKv per il restore (writeBuffer per layer + pointer impostato).
  readKv(): Promise<Float32Array>;
  writeKv(payload: Float32Array, tokenCount: number): void;
  // Decode multi-step K≤8 (spec B2 §Decode loop multi-step): K forward in un solo
  // submit con feedback del token on-GPU (l'argmax dello step i è l'input
  // dell'embedding gather dello step i+1 — token_embd vive su GPU) e UNA mapAsync
  // per batch. pos parte da posStart === kvLen (contratto hard); kvLen += k a
  // batch riuscito. Ritorna i k argmax id (greedy). EOS mid-batch: il chiamante
  // taglia con crop() (semantica trimAtEos, esatta per il rollback B1).
  // K=1 NON sostituisce forwardToken: quello resta l'oracolo di identità.
  decodeBatch(prevToken: number, posStart: number, k: number): Promise<number[]>;
  // Prefill multi-token M≤8 (spec B1 §Forward multi-token): piano a chunk con GEMM
  // small-batch, zero readback durante il prefill (embedding dequantizzate CPU-side
  // in un buffer unico, copy per chunk nell'encoder), submit ogni ~64 token,
  // lm_head+argmax SOLO sull'ultima posizione. Ritorna l'argmax id dell'ultima
  // posizione (come farebbe forwardToken sequenziale sull'ultimo token).
  prefillChunked(tokens: number[], posStart: number): Promise<number>;
  readLogits(): Promise<Float32Array>;
  // Telemetria nativa (spec §Telemetria): livello 1 = encode CPU (2 letture clock/
  // token); livello 2 = GPU ms via timestamp-query (ring, resolve lazy ogni 64 token,
  // MAI bloccante nel loop). Zero-overhead da spenta: nessun timer, nessun
  // timestampWrites nel pass descriptor.
  setTelemetry(on: boolean): void;
  getTelemetry(): Promise<EngineTelemetry>;
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
  opts: {
    taps?: number[]; fused?: boolean; telemetry?: boolean; telemetryGpu?: boolean;
  } = {},
): Promise<EngineHandle> {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU non disponibile");
  // Limiti DERIVATI dai consumatori (C3a it.8): prima qui c'era
  // `const need = 256 * 1024 * 1024`, l'ultima costante difensiva senza
  // consumatore dichiarato rimasta nel repo — lo stesso difetto che
  // gpulimits.ts documenta come gia' commesso due volte. I due consumatori
  // veri del path Qwen sono: `wOut.qs` (output.weight Q8_0 ripacchettata,
  // 136.134.656 B — il default di spec da 128 MiB NON basta, era la landmine)
  // e `rmsPairGemmSiluChunkFast` (30.848 B di workgroup storage; il commento
  // storico citava 19,7 KB del down-proj, che NON e' il massimo).
  const hasTsq = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: hasTsq ? (["timestamp-query"] as GPUFeatureName[]) : [],
    requiredLimits: negotiateLimits(adapter, {
      ctxMax: CTX_MAX,
      mlaAttention: false,                 // Qwen non ha l'attention MLA
      kvBytesPerLayer: CTX_MAX * KV_DIM * 4,
      extraBindings: [{
        bytes: Math.ceil((S.vocab * S.dModel) / 32) * 32, // qs Q8_0 = 1 B/peso
        consumer: "Qwen output.weight Q8_0 qs (lm_head)",
      }],
    }),
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
  interface LayerBufs {
    qkv: QuantBufs; qkvBias: GPUBuffer; attnNorm: GPUBuffer;
    wq: QuantBufs; bq: GPUBuffer; wk: QuantBufs; bk: GPUBuffer; wv: QuantBufs; bv: GPUBuffer;
    wo: QuantBufs; ffnNorm: GPUBuffer; wGate: QuantBufs; wUp: QuantBufs; wDown: QuantBufs;
    kCache: GPUBuffer; vCache: GPUBuffer;
  }
  const layers: LayerBufs[] = [];
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
  // token_embd: CPU-side per forwardToken/prefill (una riga dequantizzata via
  // writeBuffer, invariato) E repackato su GPU (~68 MB q4) per il feedback
  // on-GPU del decode multi-step (spec B2, ruling 2026-07-30 decisione c)
  const embdInfo = T("token_embd.weight");
  const embdOff = f.dataOffset + embdInfo.offset;
  const embdRow = new Float32Array(S.dModel);
  onProgress("repack token_embd per GPU…", 0.94);
  const wEmbd = quantBufs(embdInfo);
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
  // attention split (spec B2): partial {out|m|l} per (head, partizione) — griglia
  // fissa (nHead, S_MAX), il piano statico resta immutabile
  const attnPartials = storage(attnPartialsLen(S.nHead, S.headDim, CTX_MAX) * 4);
  const pmax = storage(N_PARTIALS * 4);
  const pidx = storage(N_PARTIALS * 4);
  const amaxOut = storage(16);
  const P = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const stagingId = device.createBuffer({ label: "staging-argmax-id", size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  // --- telemetria ---
  let telemetryOn = opts.telemetry ?? true;
  let tForwards = 0;
  let tEncodeCpuMs = 0;
  const TSQ_RING = 64; // token per batch di resolve (lazy, mai bloccante)
  // Livello 2 dietro opt-in (feature di misura: zero-overhead da spenta by design).
  // Il known-issue fase A (timestamp azzerati + corruzione compute) e' stato
  // root-causato e RISOLTO in fase B1 (2026-07-29): era mapAsync chiamata prima del
  // submit — vedi armTsq e docs/engine/tsq-diag-2026-07-29.md. Col liv.2 attivo la
  // conformance resta PASS (gate doppio) e i timestamp sono reali (~2.2 ms/token GPU).
  const wantGpuTs = hasTsq && opts.telemetryGpu === true;
  const querySet = wantGpuTs ? device.createQuerySet({ type: "timestamp", count: TSQ_RING * 2 }) : null;
  const tsqResolve = wantGpuTs
    ? device.createBuffer({ label: "tsq-resolve", size: TSQ_RING * 2 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null;
  const pendingTsq: Promise<number[]>[] = [];
  let tsqIdx = 0;
  // Root-cause del known-issue fase A (diagnosi 2026-07-29, tsq-diag): mapAsync era
  // chiamata DENTRO flushTsq, cioè prima del submit che contiene la copy verso lo
  // staging — Dawn valida "buffer used in submit while mapped" e DROPPA l'intero
  // command buffer (⇒ token saltato = corruzione; copy mai eseguita = timestamp a
  // zero). Fix: flushTsq encoda soltanto; armTsq (la mapAsync) parte DOPO il submit.
  const flushTsq = (enc: GPUCommandEncoder, count: number): GPUBuffer | null => {
    if (!querySet || !tsqResolve) return null;
    enc.resolveQuerySet(querySet, 0, count * 2, tsqResolve, 0);
    const staging = device.createBuffer({ label: "tsq-staging", size: count * 2 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(tsqResolve, 0, staging, 0, count * 2 * 8);
    return staging;
  };
  const armTsq = (staging: GPUBuffer, count: number): void => {
    pendingTsq.push(
      (async () => {
        await staging.mapAsync(GPUMapMode.READ);
        const ts = new BigUint64Array(staging.getMappedRange().slice(0));
        staging.destroy();
        const out: number[] = [];
        for (let i = 0; i < count; i++) out.push(Number(ts[i * 2 + 1] - ts[i * 2]) / 1e6);
        return out;
      })(),
    );
  };

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
    // attention split (spec B2, sostituisce attnFusedWgsl nel piano di produzione:
    // quello resta nel sorgente per microbench/debug — il fuso a 14 wg scala
    // linearmente col contesto, lo split è ~piatto: attn-bench-4090-2026-07-30)
    attnPart: mkPipe(attnSplitPartWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: S.headDim, ctxMax: CTX_MAX, freqBase: S.ropeFreqBase, chunkP: ATTN_CHUNK_P })),
    attnReduce: mkPipe(attnSplitReduceWgsl({ nHead: S.nHead, headDim: S.headDim, ctxMax: CTX_MAX, chunkP: ATTN_CHUNK_P })),
    oResid: mkPipe(gemvResidualFastWgsl({ K: S.dModel, N: S.dModel })),
    pairSilu: mkPipe(rmsPairGemvSiluFastWgsl({ K: S.dModel, N: S.dFfn, eps: S.rmsEps })),
    downResid: mkPipe(gemvResidualFastWgsl({ K: S.dFfn, N: S.dModel })),
    rmsLmHead: mkPipe(rmsGemvQ8FastWgsl({ K: S.dModel, N: S.vocab, eps: S.rmsEps })),
    embedGather: mkPipe(embedGatherQ4Wgsl({ K: S.dModel })),
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
      push(pipes.attnPart, [qkvB, L.kCache, L.vCache, attnPartials], [S.nHead, attnSMax(CTX_MAX)], P);
      push(pipes.attnReduce, [attnPartials, attnOut], S.nHead, P);
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

  // --- piano prefill M≤8 (compilato a parte, spec B1 §Forward multi-token) ---
  // Stessi step logici del piano fuso decode ma per chunk di M righe: GEMM
  // small-batch (riga di peso riusata per M attivazioni), attention con maschera
  // causale intra-chunk, KV append di M righe in un dispatch, RoPE con pos
  // per-riga. lm_head/argmax NON sono nel piano: girano una volta sola a fine
  // prefill riusando la coda del piano decode (tailSteps).
  const M_MAX = PREFILL_M;
  const QKV_DIM = S.dModel + 2 * KV_DIM;
  const xM = storage(M_MAX * S.dModel * 4);
  const qkvM = storage(M_MAX * QKV_DIM * 4);
  const attnOutM = storage(M_MAX * S.dModel * 4);
  const gateM = storage(M_MAX * S.dFfn * 4);
  const C = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // slot uniform per chunk (stride 256, pattern della sim di fase 1): scritti in
  // blocco al via del prefill, copiati in C dentro l'encoder prima di ogni chunk
  const pcSlots = device.createBuffer({
    size: Math.ceil(CTX_MAX / M_MAX) * 256,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const pipesPre = {
    rmsQkvC: mkPipe(rmsGemmQkvChunkFastWgsl({ K: S.dModel, N: QKV_DIM, eps: S.rmsEps, mMax: M_MAX })),
    ropeC: mkPipe(ropeChunkWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: S.headDim, freqBase: S.ropeFreqBase, mMax: M_MAX })),
    kvAppC: mkPipe(kvAppendChunkWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: S.headDim, mMax: M_MAX })),
    attnC: mkPipe(attnPrefillChunkWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: S.headDim, ctxMax: CTX_MAX, mMax: M_MAX })),
    oResidC: mkPipe(gemmResidChunkFastWgsl({ K: S.dModel, N: S.dModel, mMax: M_MAX })),
    pairSiluC: mkPipe(rmsPairGemmSiluChunkFastWgsl({ K: S.dModel, N: S.dFfn, eps: S.rmsEps, mMax: M_MAX })),
    downResidC: mkPipe(gemmResidChunkFastWgsl({ K: S.dFfn, N: S.dModel, mMax: M_MAX })),
  };
  // La copy del tap (ultima riga del chunk, contratto spec §Tap) dipende da rows:
  // si risolve a encode-time, qui resta un marker con il buffer di destinazione.
  type PreStep = Step | { kind: "tapLast"; dst: GPUBuffer };
  const preSteps: PreStep[] = [];
  const pushPre = (pipe: GPUComputePipeline, bufs: GPUBuffer[], wgs: number | [number, number]) =>
    preSteps.push({ kind: "compute", pipe, bind: bg(pipe, bufs, C), wgs: typeof wgs === "number" ? [wgs, 1] : wgs });
  for (const L of layers) {
    pushPre(pipesPre.rmsQkvC, [L.qkv.qs, L.qkv.scales, xM, L.attnNorm, qkvM, L.qkvBias], gemvGrid(QKV_DIM / 4));
    pushPre(pipesPre.ropeC, [qkvM], [Math.ceil(((S.nHead + S.nKvHead) * S.headDim / 2) / 64), M_MAX]);
    pushPre(pipesPre.kvAppC, [qkvM, L.kCache, L.vCache], [Math.ceil(KV_DIM / 64), M_MAX]);
    pushPre(pipesPre.attnC, [qkvM, L.kCache, L.vCache, attnOutM], [S.nHead, M_MAX]);
    pushPre(pipesPre.oResidC, [L.wo.qs, L.wo.scales, attnOutM, xM], gemvGrid(S.dModel / 4));
    pushPre(pipesPre.pairSiluC, [L.wGate.qs, L.wGate.scales, L.wUp.qs, L.wUp.scales, xM, L.ffnNorm, gateM], gemvGrid(S.dFfn / 4));
    pushPre(pipesPre.downResidC, [L.wDown.qs, L.wDown.scales, gateM, xM], gemvGrid(S.dModel / 4));
    const tapDstP = tapBufs.get(layers.indexOf(L));
    if (tapDstP) preSteps.push({ kind: "tapLast", dst: tapDstP });
  }
  // coda del piano decode (lm_head [+rms] + argmax): riusata a fine prefill
  const tailSteps = steps.slice(steps.length - (fused ? 3 : 4));
  // --- decode multi-step (spec B2): bind group dell'embed gather + slot P + staging ids ---
  const bgEmbed = bg(pipes.embedGather, [wEmbd.qs, wEmbd.scales, amaxOut, x]);
  const dbSlots = device.createBuffer({
    size: DECODE_K_MAX * DECODE_SLOT_STRIDE,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const idsBatch = storage(DECODE_K_MAX * 4); // COPY_SRC/COPY_DST inclusi da storage()
  const stagingIds = device.createBuffer({
    label: "staging-batch-ids", size: DECODE_K_MAX * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  // length pointer della KV (spec §Crop): unica verità sulla posizione
  const kv = createKvLen(CTX_MAX);
  onProgress("pronto", 1);

  return {
    device,
    dispatchesPerToken: steps.filter((st) => st.kind === "compute").length,
    reset() { kv.reset(); }, // ≡ crop(0): le righe si sovrascrivono per posizione
    crop(toLen: number) { kv.crop(toLen); }, // zero lavoro GPU (spec §Crop)
    get kvLen() { return kv.len; },
    async forwardToken(token: number, pos: number): Promise<number> {
      kv.assertNext(pos); // pos === kvLen o throw (+ capacità)
      dequantQ4_0Row(bytes, embdOff, S.dModel, token, embdRow);
      device.queue.writeBuffer(x, 0, embdRow as unknown as BufferSource);
      device.queue.writeBuffer(P, 0, new Uint32Array([pos, pos, 0, 0]));
      const tEnc0 = telemetryOn ? performance.now() : 0;
      const enc = device.createCommandEncoder();
      const passDesc: GPUComputePassDescriptor = telemetryOn && querySet
        ? { timestampWrites: { querySet, beginningOfPassWriteIndex: tsqIdx * 2, endOfPassWriteIndex: tsqIdx * 2 + 1 } }
        : {};
      let pass = enc.beginComputePass(passDesc);
      for (const s of steps) {
        if (s.kind === "copy") {
          pass.end();
          enc.copyBufferToBuffer(s.src, 0, s.dst, 0, s.bytes);
          pass = enc.beginComputePass(); // i pass successivi del token non ri-scrivono timestamp
          continue;
        }
        pass.setPipeline(s.pipe);
        pass.setBindGroup(0, s.bind);
        pass.dispatchWorkgroups(s.wgs[0], s.wgs[1]);
      }
      pass.end();
      enc.copyBufferToBuffer(amaxOut, 0, stagingId, 0, 4);
      let tsqStaging: GPUBuffer | null = null;
      if (telemetryOn && querySet) {
        tsqIdx++;
        if (tsqIdx === TSQ_RING) {
          tsqStaging = flushTsq(enc, TSQ_RING);
          tsqIdx = 0;
        }
      }
      device.queue.submit([enc.finish()]);
      if (tsqStaging) armTsq(tsqStaging, TSQ_RING); // mapAsync SOLO dopo il submit
      if (telemetryOn) { tEncodeCpuMs += performance.now() - tEnc0; tForwards++; }
      await stagingId.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(stagingId.getMappedRange())[0];
      stagingId.unmap();
      kv.advance(); // a forward riuscito
      return id;
    },
    async decodeBatch(prevToken: number, posStart: number, k: number): Promise<number[]> {
      kv.assertNext(posStart, k); // posStart === kvLen o throw (+ capacità)
      const plan = planDecodeBatch(posStart, k, CTX_MAX);
      // seed del feedback: amaxOut = prevToken (lo step 0 legge il seme, gli step
      // successivi l'argmax scritto dallo step precedente — stesso buffer)
      device.queue.writeBuffer(amaxOut, 0, new Uint32Array([prevToken, 0, 0, 0]));
      const slots = new Uint32Array(k * (DECODE_SLOT_STRIDE / 4));
      plan.forEach((s, i) => { slots[i * 64] = s.pos; slots[i * 64 + 1] = s.pos; });
      device.queue.writeBuffer(dbSlots, 0, slots as unknown as BufferSource);
      // Error scope come CONTRATTO di ogni percorso di encode nuovo (landmine B1:
      // pipeline invalida ⇒ submit droppati muti e readback stale plausibili)
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      const tEnc0 = telemetryOn ? performance.now() : 0;
      const enc = device.createCommandEncoder();
      let tsqStaging: GPUBuffer | null = null;
      for (let i = 0; i < k; i++) {
        enc.copyBufferToBuffer(dbSlots, plan[i].slotOffset, P, 0, 16);
        const passDesc: GPUComputePassDescriptor = telemetryOn && querySet
          ? { timestampWrites: { querySet, beginningOfPassWriteIndex: tsqIdx * 2, endOfPassWriteIndex: tsqIdx * 2 + 1 } }
          : {};
        let pass = enc.beginComputePass(passDesc);
        pass.setPipeline(pipes.embedGather);
        pass.setBindGroup(0, bgEmbed);
        pass.dispatchWorkgroups(Math.ceil(S.dModel / 64));
        for (const s of steps) {
          if (s.kind === "copy") {
            pass.end();
            enc.copyBufferToBuffer(s.src, 0, s.dst, 0, s.bytes);
            pass = enc.beginComputePass(); // i pass riaperti dello step non ri-scrivono timestamp
            continue;
          }
          pass.setPipeline(s.pipe);
          pass.setBindGroup(0, s.bind);
          pass.dispatchWorkgroups(s.wgs[0], s.wgs[1]);
        }
        pass.end();
        enc.copyBufferToBuffer(amaxOut, 0, idsBatch, i * 4, 4);
        if (telemetryOn && querySet) {
          tsqIdx++;
          if (tsqIdx === TSQ_RING) {
            tsqStaging = flushTsq(enc, TSQ_RING); // al più un flush per batch (k ≤ 8 < ring)
            tsqIdx = 0;
          }
        }
      }
      enc.copyBufferToBuffer(idsBatch, 0, stagingIds, 0, k * 4);
      device.queue.submit([enc.finish()]);
      if (tsqStaging) armTsq(tsqStaging, TSQ_RING); // mapAsync SOLO dopo il submit
      if (telemetryOn) { tEncodeCpuMs += performance.now() - tEnc0; tForwards += k; }
      const errOom = await device.popErrorScope();
      const errVal = await device.popErrorScope();
      if (errOom || errVal) throw new Error(`decodeBatch error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
      await stagingIds.mapAsync(GPUMapMode.READ);
      const ids = [...new Uint32Array(stagingIds.getMappedRange()).subarray(0, k)];
      stagingIds.unmap();
      kv.advance(k); // a batch riuscito
      return ids;
    },
    async prefillChunked(tokens: number[], posStart: number): Promise<number> {
      const n = tokens.length;
      if (n < 1) throw new Error("prefill vuoto");
      kv.assertNext(posStart, n); // posStart === kvLen o throw (+ capacità)
      const plan = planPrefill(n, posStart);
      // Error scope come CONTRATTO del prefill (lezione fase 6): una pipeline
      // invalida fa droppare a Dawn ogni submit IN SILENZIO e i readback restituiscono
      // dati stale del run precedente — qui il throw è sincrono e attribuibile.
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      // zero readback durante il prefill (l'input è il prompt, noto): embedding di
      // tutto il prompt dequantizzate CPU-side in un buffer unico, copy per chunk
      // nell'encoder; una sola sync a fine prefill (la mapAsync dell'argmax finale).
      const emb = new Float32Array(n * S.dModel);
      for (let i = 0; i < n; i++)
        dequantQ4_0Row(bytes, embdOff, S.dModel, tokens[i], emb.subarray(i * S.dModel, (i + 1) * S.dModel));
      const embBuf = storage(emb.byteLength);
      device.queue.writeBuffer(embBuf, 0, emb as unknown as BufferSource);
      const slots = new Uint32Array(plan.length * 64); // stride 256 B (copy safe)
      plan.forEach((c, ci) => { slots[ci * 64] = c.posBase; slots[ci * 64 + 1] = c.rows; });
      device.queue.writeBuffer(pcSlots, 0, slots as unknown as BufferSource);

      let enc = device.createCommandEncoder();
      for (let ci = 0; ci < plan.length; ci++) {
        const c = plan[ci];
        enc.copyBufferToBuffer(embBuf, c.start * S.dModel * 4, xM, 0, c.rows * S.dModel * 4);
        enc.copyBufferToBuffer(pcSlots, ci * 256, C, 0, 16);
        let pass = enc.beginComputePass();
        for (const s of preSteps) {
          if (s.kind !== "compute") { // tapLast: hidden dell'ULTIMO token del chunk
            pass.end();
            enc.copyBufferToBuffer(xM, (c.rows - 1) * S.dModel * 4, s.dst, 0, S.dModel * 4);
            pass = enc.beginComputePass();
            continue;
          }
          pass.setPipeline(s.pipe);
          pass.setBindGroup(0, s.bind);
          pass.dispatchWorkgroups(s.wgs[0], s.wgs[1]);
        }
        pass.end();
        if (c.submitAfter && ci < plan.length - 1) {
          device.queue.submit([enc.finish()]);
          enc = device.createCommandEncoder();
        }
      }
      // lm_head + argmax SOLO sull'ultima posizione dell'ultimo chunk: si copia
      // l'ultima riga di xM nell'x del decode e si riusa la coda del piano fuso.
      const last = plan[plan.length - 1];
      enc.copyBufferToBuffer(xM, (last.rows - 1) * S.dModel * 4, x, 0, S.dModel * 4);
      let pass = enc.beginComputePass();
      for (const s of tailSteps) {
        if (s.kind !== "compute") continue; // la coda non contiene copy
        pass.setPipeline(s.pipe);
        pass.setBindGroup(0, s.bind);
        pass.dispatchWorkgroups(s.wgs[0], s.wgs[1]);
      }
      pass.end();
      enc.copyBufferToBuffer(amaxOut, 0, stagingId, 0, 4);
      device.queue.submit([enc.finish()]);
      const errOom = await device.popErrorScope();
      const errVal = await device.popErrorScope();
      if (errOom || errVal) throw new Error(`prefill error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
      await stagingId.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(stagingId.getMappedRange())[0];
      stagingId.unmap();
      embBuf.destroy();
      kv.advance(n); // il piano prefill avanza il pointer dell'intero chunk-set
      return id;
    },
    async readKv(): Promise<Float32Array> {
      const n = kv.len;
      if (n < 1) throw new Error("readKv: cache vuota");
      const rowBytes = n * KV_DIM * 4;
      const staging = device.createBuffer({
        size: S.nLayer * 2 * rowBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      layers.forEach((L, l) => {
        enc.copyBufferToBuffer(L.kCache, 0, staging, (l * 2) * rowBytes, rowBytes);
        enc.copyBufferToBuffer(L.vCache, 0, staging, (l * 2 + 1) * rowBytes, rowBytes);
      });
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.destroy();
      return out;
    },
    writeKv(payload: Float32Array, tokenCount: number): void {
      if (!Number.isInteger(tokenCount) || tokenCount < 1 || tokenCount > CTX_MAX) {
        throw new Error(`writeKv: tokenCount non valido (${tokenCount})`);
      }
      const rowElems = tokenCount * KV_DIM;
      if (payload.length !== S.nLayer * 2 * rowElems) {
        throw new Error(`writeKv: payload ${payload.length} !== atteso ${S.nLayer * 2 * rowElems}`);
      }
      layers.forEach((L, l) => {
        device.queue.writeBuffer(L.kCache, 0, payload.subarray((l * 2) * rowElems, (l * 2 + 1) * rowElems) as unknown as BufferSource);
        device.queue.writeBuffer(L.vCache, 0, payload.subarray((l * 2 + 1) * rowElems, (l * 2 + 2) * rowElems) as unknown as BufferSource);
      });
      kv.reset();
      kv.advance(tokenCount); // il pointer riparte dal checkpoint restaurato
    },
    setTelemetry(on: boolean): void {
      telemetryOn = on;
      tsqIdx = 0; // il ring riparte: i timestamp parziali del batch corrente si scartano
    },
    async getTelemetry() {
      const batches = await Promise.all(pendingTsq);
      pendingTsq.length = 0;
      // v>0 scarta i pass sotto il quanto di Chrome (~100us) e gli eventuali reset
      // del counter GPU (delta negativi, documentati) — non più gli "zeri" del
      // known-issue fase A, risolto (vedi armTsq).
      const gpuMs = batches.flat().filter((v) => v > 0);
      const mean = gpuMs.length ? gpuMs.reduce((a, b) => a + b, 0) / gpuMs.length : null;
      return {
        forwards: tForwards,
        encodeCpuMsPerToken: tForwards ? tEncodeCpuMs / tForwards : 0,
        gpuMsPerToken: mean,
        timestampNote: querySet
          ? "quantizzazione Chrome ~100us, media su batch (ring)"
          : "livello 2 opt-in (telemetryGpu) — spento: zero-overhead by design",
      };
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
