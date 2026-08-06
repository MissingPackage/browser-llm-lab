// Kernel-test del motore: ogni kernel WGSL contro un riferimento JS calcolato QUI,
// su dati pseudo-casuali seeded (LCG: riproducibile). Non serve il modello: valida
// indicizzazione, riduzioni e dequant dei kernel prima di qualunque orchestratore.
//
// Tolleranze: i kernel accumulano f32 in ordine diverso dal riferimento (f64 in JS)
// ⇒ confronto a tolleranza relativa+assoluta, per kernel. Le parti intere (nibble,
// int8, indici argmax) devono essere esatte.
import {
  gemvQuantWgsl, gemvF32Wgsl, gemvQ5KWgsl, gemvQ6KWgsl, rmsnormWgsl, ropeNeoxWgsl, kvAppendWgsl,
  attnDecodeWgsl, siluMulWgsl, addInPlaceWgsl, argmaxStage1Wgsl, argmaxStage2Wgsl,
  ARGMAX_CHUNK, ropeMlaNormWgsl, gemvQ8HeadsWgsl, mlaAttnDecodeWgsl, stridedCopyWgsl,
  routerTopKWgsl, pairGemvSiluFastWgsl, gemvAccumFastWgsl, gemvGrid, type ArenaOpts,
  pairGemvSiluGatherWgsl, gemvDownSlotsWgsl, moeCombineWgsl,
  mlaAttnSplitPartWgsl, mlaAttnSplitReduceWgsl,
  pairGemvSiluQ5KFastWgsl, gemvQ6KFastWgsl,
} from "../kernels/wgsl";
import { MLA_CHUNK_P, mlaSMax, mlaPartialsLen } from "../mlasplit";
import { KQUANT_FAST_Q5K_PAIR_REL_TOL, KQUANT_FAST_Q6K_REL_TOL } from "../kquantfast";
import { createGlmLayer0 } from "../glmforward";
import {
  GlmDenseLayerRefF64, GlmMoeLayerRefF64, glmMoeFfnRefF64, type GlmMoeExpertWeights,
} from "../cpuref";
import { createGlmModel, type GlmWeightSource } from "../glmmodel";
import {
  routerSelect, packExpertSlab, SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, WEIGHTS_SUM_CLAMP_MIN,
} from "../moe";
import { ExpertOpfsStore } from "../expertstore";
import { ExpertCache, arenaNeeds, expertKey, modelExpertPark, type ExpertRawBytes } from "../residency";
import {
  repackQ4_0, repackQ8_0, repackQ4_1, repackKQuant,
  dequantQ4_0, dequantQ8_0, dequantQ4_1, dequantQ5_K, dequantQ6_K,
  Q4_1_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
} from "../quant";
import { QWEN25_05B as S, GLM47_FLASH as G } from "../shape";
import { createEngineDevice } from "../gpudevice";
import { planMoeChunk } from "../glmprefillplan";

interface KResult {
  kernel: string; pass: boolean; maxAbs: number; maxRel: number; note?: string;
  /**
   * Misure NUMERICHE del caso, per i confronti fra due esecuzioni (la pagina le
   * ignora: mostra kernel/esito/max/note). Servono a un gate che nessun singolo
   * caso puo' dare — "questa variante non ha cambiato NIENTE dell'esito" si
   * verifica solo mettendo due run uno accanto all'altro.
   */
  metrics?: Record<string, number>;
}

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
}

function randF32(n: number, seed: number, scale = 1): Float32Array {
  const r = lcg(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (r() * 2 - 1) * scale;
  return out;
}

function randBytes(n: number, seed: number): Uint8Array {
  const r = lcg(seed);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.floor(r() * 256);
  return out;
}

// scala f16 "sana" nei byte di blocco (esponente moderato: niente inf/denormali)
function fixScales(src: Uint8Array, blockBytes: number): void {
  for (let o = 0; o + blockBytes <= src.length; o += blockBytes) {
    src[o + 1] = 0x2c | (src[o + 1] & 0x03);
  }
}

function compare(kernel: string, got: Float32Array, ref: Float32Array, relTol: number, absTol: number): KResult {
  let maxAbs = 0, maxRel = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(got[i] - ref[i]);
    maxAbs = Math.max(maxAbs, d);
    const denom = Math.max(Math.abs(ref[i]), 1e-6);
    maxRel = Math.max(maxRel, d / denom);
  }
  return { kernel, pass: got.length === ref.length && (maxAbs <= absTol || maxRel <= relTol), maxAbs, maxRel };
}

// Riferimento f64 dell'attention MLA absorbed: score a 576 su tutta la cache,
// softmax, out = Σ softmax(p)·c_kv[p, j] sulle prime 512 componenti. È il
// riferimento CONDIVISO fra il kernel monolitico e lo split (fase 4c): i due
// devono cadere sulla stessa risposta, non ognuno sulla propria.
function mlaAttnRefF64(q: Float32Array, cache: Float32Array, nPast: number, scale: number): Float32Array {
  const W = G.kvLora + G.ropeDims;
  const ref = new Float32Array(G.nHead * G.kvLora);
  for (let h = 0; h < G.nHead; h++) {
    const sc = new Float64Array(nPast + 1);
    for (let p = 0; p <= nPast; p++) {
      let acc = 0;
      for (let i = 0; i < W; i++) acc += q[h * W + i] * cache[p * W + i];
      sc[p] = acc * scale;
    }
    let m = -Infinity;
    for (const v of sc) m = Math.max(m, v);
    let sum = 0;
    for (let p = 0; p <= nPast; p++) { sc[p] = Math.exp(sc[p] - m); sum += sc[p]; }
    for (let j = 0; j < G.kvLora; j++) {
      let acc = 0;
      for (let p = 0; p <= nPast; p++) acc += (sc[p] / sum) * cache[p * W + j];
      ref[h * G.kvLora + j] = acc;
    }
  }
  return ref;
}

// Arena expert nel ktest (C3a fase 4 strato 1): la finestra si TAGLIA a 3 slab
// apposta. A finestra piena una classe da pochi slot starebbe in un buffer solo
// (nBuf = 1) e lo switch di `ld4` non avrebbe archi: il caso che conta — slot in
// un buffer diverso dal primo — non verrebbe mai eseguito. Con 3 slab per buffer
// e 7 slot la classe q4_1 del mini-modello ha nBuf = 3.
const KTEST_ARENA_WINDOW = 3 * SLAB_DOWN_Q4_1.bytes;
const KTEST_SLOTS = { q4_0: 4, q4_1: 7 };
// Residenza TOTALE per costruzione (C3a fase 4 slice C): il mini-modello ha un
// solo layer MoE (blk.1, classe q4_1) e quindi un parco di G.nExpert = 64 expert.
// Con 64 slot la cache non puo' evincere nemmeno volendo, che e' la precondizione
// del modo `select:"gpu"`. La finestra si allarga a 22 slab apposta per tenere
// nBuf = ceil(64/22) = 3 come negli altri casi d'arena: cosi' lo switch di `ld4`
// ha gli stessi archi e la variabile in esame resta UNA (chi riempie Sel).
// A finestra 3 slab servirebbero 22 buffer, cioe' 25 storage binding: oltre
// ARENA_BUFFERS_MAX e oltre quanto il device concede.
const KTEST_ARENA_WINDOW_GPU = 22 * SLAB_DOWN_Q4_1.bytes;
const KTEST_SLOTS_GPU = { q4_0: 4, q4_1: 64 };

/** Le due configurazioni d'arena del ktest, con i limiti negoziati sul massimo. */
const ktestArena = (gpu: boolean) => ({
  slotsOverride: gpu ? KTEST_SLOTS_GPU : KTEST_SLOTS,
  window: gpu ? KTEST_ARENA_WINDOW_GPU : KTEST_ARENA_WINDOW,
});
function ktestArenaNeeds(): { arenaBuffers: number; arenaWindowBytes: number } {
  const of = (gpu: boolean) => {
    const a = ktestArena(gpu);
    return arenaNeeds({
      budgetBytes: 0, slotsOverride: a.slotsOverride,
      maxBufferBytes: a.window, maxBindingBytes: a.window,
    });
  };
  const a = of(false), b = of(true);
  return {
    arenaBuffers: Math.max(a.arenaBuffers, b.arenaBuffers),
    arenaWindowBytes: Math.max(a.arenaWindowBytes, b.arenaWindowBytes),
  };
}

class Gpu {
  device!: GPUDevice;
  async init(): Promise<string> {
    // ktest gira mini-modelli sintetici (ctxMax <= 64, vocab ridotto) ma binda
    // i pesi DENSI VERI di blk.0: il tensore piu' grande e' ffn_gate/up q4_0
    // [2048 -> 10240] = 10.485.760 B di qs. E' quello a determinare il
    // requisito, dichiarato come consumatore invece che come cap inventato.
    const { adapter, device } = await createEngineDevice({
      label: "ktest",
      needs: {
        ctxMax: 64,
        extraBindings: [{ bytes: 10_485_760, consumer: "ktest: blk.0 ffn_gate/up q4_0 qs (pesi reali)" }],
        // i due requisiti dell'arena, calcolati dalla stessa aritmetica della
        // cache (residency.arenaNeeds) e non ricopiati qui. Le configurazioni
        // sono DUE (residenza parziale e residenza totale dello slice C) e il
        // device e' uno: si chiede il massimo dei due requisiti, come farebbe
        // `limitsFor` fra due need sullo stesso limite.
        ...ktestArenaNeeds(),
      },
    });
    this.device = device;
    const info = adapter.info;
    return `${info?.vendor ?? "?"} ${info?.architecture ?? ""}`;
  }
  buf(data: Float32Array | Uint32Array, usage = GPUBufferUsage.STORAGE): GPUBuffer {
    const b = this.device.createBuffer({
      size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(b, 0, data as BufferSource);
    return b;
  }
  empty(bytes: number): GPUBuffer {
    return this.device.createBuffer({
      size: Math.max(16, bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }
  // binding con {offset, size}: sotto-range di un buffer di classe slab (MoE C2)
  // workgroups: 1D (storico) o [x,y,z] — i kernel gather della fase 5 usano
  // wid.z come indice di riga raccolta
  async run(code: string, bindings: Array<GPUBuffer | { buffer: GPUBuffer; offset: number; size: number }>, workgroups: number | [number, number, number], uniform?: GPUBuffer): Promise<void> {
    const module = this.device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error");
    if (errs.length) throw new Error(`WGSL: ${errs[0].message} @${errs[0].lineNum}`);
    const pipeline = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const entries: GPUBindGroupEntry[] = bindings.map((b, i) => ({ binding: i, resource: "buffer" in b ? b : { buffer: b } }));
    if (uniform) entries.push({ binding: bindings.length, resource: { buffer: uniform } });
    const bg = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    if (typeof workgroups === "number") pass.dispatchWorkgroups(workgroups);
    else pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
  async read(b: GPUBuffer, bytes: number, srcOffset = 0): Promise<ArrayBuffer> {
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(b, srcOffset, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0);
    staging.destroy();
    return out;
  }
  uniform(pos: number, nPast: number): GPUBuffer {
    const b = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, new Uint32Array([pos, nPast, 0, 0]));
    return b;
  }
}

async function testGemv(g: Gpu, kind: "q4_0" | "q8_0", K: number, N: number, hasBias: boolean): Promise<KResult> {
  const blockBytes = kind === "q4_0" ? 18 : 34;
  const nBlocks = (K / 32) * N;
  const src = randBytes(nBlocks * blockBytes, 1234 + K + N);
  fixScales(src, blockBytes);
  const { qs, scales } = kind === "q4_0" ? repackQ4_0(src, 0, nBlocks) : repackQ8_0(src, 0, nBlocks);
  const w = new Float32Array(nBlocks * 32);
  (kind === "q4_0" ? dequantQ4_0 : dequantQ8_0)(src, 0, nBlocks, w);
  const x = randF32(K, 99);
  const bias = randF32(N, 7);
  const ref = new Float32Array(N);
  for (let r = 0; r < N; r++) {
    let acc = 0;
    for (let i = 0; i < K; i++) acc += w[r * K + i] * x[i];
    ref[r] = hasBias ? acc + bias[r] : acc;
  }
  const y = g.empty(N * 4);
  const bufs = [g.buf(qs), g.buf(scales), g.buf(x), y];
  if (hasBias) bufs.push(g.buf(bias));
  await g.run(gemvQuantWgsl({ kind, K, N, hasBias }), bufs, N);
  const got = new Float32Array(await g.read(y, N * 4));
  return compare(`gemv-${kind}-${K}x${N}${hasBias ? "-bias" : ""}`, got, ref, 2e-4, 1e-3);
}

// come fixScales, ma per gli scale f16 dei formati C2 (offset multipli nel blocco)
function fixScalesAt(src: Uint8Array, blockBytes: number, hiByteOffsets: number[]): void {
  for (let o = 0; o + blockBytes <= src.length; o += blockBytes) {
    for (const off of hiByteOffsets) src[o + off] = 0x2c | (src[o + off] & 0x03);
  }
}

// GEMV dei formati GLM (goal C2 fase 4): kernel vs dequant CPU di riferimento
// (a loro volta validate su byte reali del GGUF contro gguf-py, it.2).
async function testGemvC2(g: Gpu, kind: "q4_1" | "q5_K" | "q6_K", K: number, N: number): Promise<KResult> {
  const blockWeights = kind === "q4_1" ? 32 : 256;
  const blockBytes = kind === "q4_1" ? Q4_1_BLOCK_BYTES : kind === "q5_K" ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;
  const nBlocks = (K / blockWeights) * N;
  const src = randBytes(nBlocks * blockBytes, 4321 + K + N);
  if (kind === "q4_1") fixScalesAt(src, blockBytes, [1, 3]);       // d, m
  else if (kind === "q5_K") fixScalesAt(src, blockBytes, [1, 3]);  // d, dmin
  else fixScalesAt(src, blockBytes, [209]);                        // d in coda
  const w = new Float32Array(nBlocks * blockWeights);
  (kind === "q4_1" ? dequantQ4_1 : kind === "q5_K" ? dequantQ5_K : dequantQ6_K)(src, 0, nBlocks, w);
  const x = randF32(K, 77 + K);
  const ref = new Float32Array(N);
  for (let r = 0; r < N; r++) {
    let acc = 0;
    for (let i = 0; i < K; i++) acc += w[r * K + i] * x[i];
    ref[r] = acc;
  }
  const y = g.empty(N * 4);
  if (kind === "q4_1") {
    const { qs, scales } = repackQ4_1(src, 0, nBlocks);
    await g.run(gemvQuantWgsl({ kind, K, N, hasBias: false }), [g.buf(qs), g.buf(scales), g.buf(x), y], N);
  } else {
    const blocks = repackKQuant(src, 0, nBlocks, blockBytes);
    const code = kind === "q5_K" ? gemvQ5KWgsl({ K, N }) : gemvQ6KWgsl({ K, N });
    await g.run(code, [g.buf(blocks), g.buf(x), y], N);
  }
  const got = new Float32Array(await g.read(y, N * 4));
  return compare(`gemv-${kind}-${K}x${N}`, got, ref, 2e-4, 1e-3);
}

// riferimento JS del rope NORM (coppie consecutive) sul segmento [offset, offset+dims)
function ropeNormRef(v: Float32Array, nVec: number, stride: number, offset: number, dims: number, freqBase: number, pos: number): void {
  for (let h = 0; h < nVec; h++) {
    for (let i = 0; i < dims / 2; i++) {
      const theta = pos * freqBase ** (-(2 * i) / dims);
      const c = Math.cos(theta), s = Math.sin(theta);
      const b = h * stride + offset + 2 * i;
      const a0 = v[b], a1 = v[b + 1];
      v[b] = a0 * c - a1 * s;
      v[b + 1] = a0 * s + a1 * c;
    }
  }
}

// Conformance layer-level (C2 fase 4 slice 3): layer 0 GLM completo su GPU
// (MLA absorbed, glmforward) vs cpuref f64 in formulazione NAIVE, sui PESI
// REALI di blk.0 estratti dal GGUF (fixture gen-glm-layer-fixture.py) e sugli
// embedding dei primi 16 token del corpus golden p0 — replay decode con cache.
// Gate: L2rel ≤ 1e-3 e max|Δ| ≤ 5e-2 sull'hidden post-layer (2048 × 16 pos).
async function testGlmLayer0Real(g: Gpu): Promise<KResult> {
  const metaRes = await fetch("/models/glm-layer0/meta.json");
  if (!metaRes.ok) {
    return {
      kernel: "glm-layer0-conformance", pass: false, maxAbs: NaN, maxRel: NaN,
      note: "fixture assente: uv run scripts/gen-glm-layer-fixture.py",
    };
  }
  const meta = await metaRes.json() as {
    tokens: number[]; embdRowBytes: number;
    tensors: Record<string, { offset: number; bytes: number; ggmlType: number }>;
  };
  const bin = new Uint8Array(await (await fetch("/models/glm-layer0/fixture.bin")).arrayBuffer());
  const seg = (n: string): Uint8Array => {
    const t = meta.tensors[n];
    return bin.subarray(t.offset, t.offset + t.bytes);
  };
  const f32 = (n: string): Float32Array => {
    const t = meta.tensors[n];
    return new Float32Array(bin.buffer, bin.byteOffset + t.offset, t.bytes / 4);
  };
  const deq = (n: string, kind: "q4_0" | "q8_0" | "q4_1"): Float32Array => {
    const s = seg(n);
    const bb = kind === "q4_0" ? 18 : kind === "q8_0" ? 34 : Q4_1_BLOCK_BYTES;
    const nB = s.length / bb;
    const out = new Float32Array(nB * 32);
    (kind === "q4_0" ? dequantQ4_0 : kind === "q8_0" ? dequantQ8_0 : dequantQ4_1)(s, 0, nB, out);
    return out;
  };
  const norms = {
    attnNorm: f32("attnNorm"), qANorm: f32("qANorm"),
    kvANorm: f32("kvANorm"), ffnNorm: f32("ffnNorm"),
  };
  const ref = new GlmDenseLayerRefF64({
    ...norms,
    wQA: deq("wQA", "q4_0"), wQB: deq("wQB", "q4_0"), wKvA: deq("wKvA", "q8_0"),
    wKB: deq("wKB", "q8_0"), wVB: deq("wVB", "q8_0"), wO: deq("wO", "q4_0"),
    wGate: deq("wGate", "q4_0"), wUp: deq("wUp", "q4_0"), wDown: deq("wDown", "q4_1"),
  });
  const layer = createGlmLayer0(g.device, {
    ...norms,
    wQA: seg("wQA"), wQB: seg("wQB"), wKvA: seg("wKvA"), wKB: seg("wKB"),
    wVB: seg("wVB"), wO: seg("wO"), wGate: seg("wGate"), wUp: seg("wUp"), wDown: seg("wDown"),
  }, 64);
  const embd = seg("embdRows");
  let maxAbs = 0, maxRel = 0, l2e = 0, l2r = 0;
  for (let p = 0; p < meta.tokens.length; p++) {
    const x = new Float32Array(G.dModel);
    dequantQ4_0(embd, p * meta.embdRowBytes, G.dModel / 32, x);
    const refOut = ref.forward(x);
    const gpuOut = await layer.forward(x, p);
    for (let i = 0; i < G.dModel; i++) {
      const d = Math.abs(gpuOut[i] - refOut[i]);
      maxAbs = Math.max(maxAbs, d);
      maxRel = Math.max(maxRel, d / Math.max(Math.abs(refOut[i]), 1e-6));
      l2e += d * d;
      l2r += refOut[i] * refOut[i];
    }
  }
  layer.destroy();
  const l2 = Math.sqrt(l2e / l2r);
  return {
    kernel: "glm-layer0-conformance", pass: l2 <= 1e-3 && maxAbs <= 5e-2, maxAbs, maxRel,
    note: `pesi reali blk.0, ${meta.tokens.length} pos decode, L2rel=${l2.toExponential(2)}, ${layer.dispatchesPerToken} dispatch`,
  };
}

// GEMV f32 (router MoE, C2 fase 5): ffn_gate_inp [64×2048] F32, ref f64.
async function testGemvF32(g: Gpu): Promise<KResult> {
  const K = G.dModel, N = G.nExpert;
  const w = randF32(K * N, 8801);
  const x = randF32(K, 8802);
  const ref = new Float32Array(N);
  for (let r = 0; r < N; r++) {
    let acc = 0;
    for (let i = 0; i < K; i++) acc += w[r * K + i] * x[i];
    ref[r] = acc;
  }
  const y = g.empty(N * 4);
  await g.run(gemvF32Wgsl({ K, N }), [g.buf(w), g.buf(x), y], N);
  return compare("gemv-f32-router", new Float32Array(await g.read(y, N * 4)), ref, 2e-4, 1e-3);
}

// GEMV con accumulo pesato (down per-expert, C2 fase 5): y += s·W·x.
async function testGemvAccum(g: Gpu, kind: "q4_0" | "q4_1"): Promise<KResult> {
  const K = G.dFfnExpert, N = G.dModel;
  const blockBytes = kind === "q4_0" ? 18 : Q4_1_BLOCK_BYTES;
  const nBlocks = (K / 32) * N;
  const src = randBytes(nBlocks * blockBytes, 8810 + blockBytes);
  if (kind === "q4_0") fixScales(src, blockBytes);
  else fixScalesAt(src, blockBytes, [1, 3]);
  const w = new Float32Array(nBlocks * 32);
  (kind === "q4_0" ? dequantQ4_0 : dequantQ4_1)(src, 0, nBlocks, w);
  const x = randF32(K, 8812);
  const y0 = randF32(N, 8813);
  const s = 0.6789;
  const ref = new Float32Array(N);
  for (let r = 0; r < N; r++) {
    let acc = 0;
    for (let i = 0; i < K; i++) acc += w[r * K + i] * x[i];
    ref[r] = y0[r] + s * acc;
  }
  const { qs, scales } = (kind === "q4_0" ? repackQ4_0 : repackQ4_1)(src, 0, nBlocks);
  const y = g.buf(y0);
  await g.run(gemvQuantWgsl({ kind, K, N, hasBias: false, scaledAccum: true }),
    [g.buf(qs), g.buf(scales), g.buf(x), y, g.buf(new Float32Array([s]))], N);
  return compare(`gemv-${kind}-accum`, new Float32Array(await g.read(y, N * 4)), ref, 2e-4, 1e-3);
}

// Blocco MoE-FFN completo (C2 fase 5 slice 1), dims reali GLM: router GEMV f32
// su GPU → selezione top-4 su CPU (routerSelect, replica build_moe_ffn) → shexp
// Q5_K/Q6_K → 4 catene expert dagli SLAB impacchettati (bind group con offset
// per-slot, come farà la residenza) con down ad accumulo pesato — contro il
// riferimento f64 glmMoeFfnRefF64 (router e selezione indipendenti). Due
// size-class: down Q4_0 (blk.5-46) e Q4_1 (blk.1-4).
async function testMoeFfnBlock(g: Gpu, downKind: "q4_0" | "q4_1"): Promise<KResult> {
  const name = `moe-ffn-block-down${downKind}`;
  const layout = downKind === "q4_0" ? SLAB_DOWN_Q4_0 : SLAB_DOWN_Q4_1;
  const seed = downKind === "q4_0" ? 9000 : 9100;
  const exBlocks = (G.dModel / 32) * G.dFfnExpert;
  const downBlockBytes = downKind === "q4_0" ? 18 : Q4_1_BLOCK_BYTES;

  const fn = randF32(G.dModel, seed + 1, 0.5);
  const routerW = randF32(G.nExpert * G.dModel, seed + 2, 0.05);
  const bias = randF32(G.nExpert, seed + 3, 0.5);

  // shexp: Q5_K gate/up [2048→1536], Q6_K down [1536→2048]
  const sbShexp = (G.dModel / 256) * G.dFfnExpert; // = (dFfnExpert/256)*dModel
  const gateShexpRaw = randBytes(sbShexp * Q5_K_BLOCK_BYTES, seed + 4);
  const upShexpRaw = randBytes(sbShexp * Q5_K_BLOCK_BYTES, seed + 5);
  const downShexpRaw = randBytes(sbShexp * Q6_K_BLOCK_BYTES, seed + 6);
  fixScalesAt(gateShexpRaw, Q5_K_BLOCK_BYTES, [1, 3]);
  fixScalesAt(upShexpRaw, Q5_K_BLOCK_BYTES, [1, 3]);
  fixScalesAt(downShexpRaw, Q6_K_BLOCK_BYTES, [209]);

  // pesi expert sintetici SOLO per i 4 che il riferimento seleziona (lazy)
  const rawByExpert = new Map<number, { gate: Uint8Array; up: Uint8Array; down: Uint8Array }>();
  const rawExpert = (e: number) => {
    let r = rawByExpert.get(e);
    if (!r) {
      r = {
        gate: randBytes(exBlocks * 18, seed + 10 + 7 * e),
        up: randBytes(exBlocks * 18, seed + 11 + 7 * e),
        down: randBytes(exBlocks * downBlockBytes, seed + 12 + 7 * e),
      };
      fixScales(r.gate, 18);
      fixScales(r.up, 18);
      if (downKind === "q4_0") fixScales(r.down, 18);
      else fixScalesAt(r.down, Q4_1_BLOCK_BYTES, [1, 3]);
      rawByExpert.set(e, r);
    }
    return r;
  };
  const deq = (raw: Uint8Array, kind: "q4_0" | "q4_1" | "q5_K" | "q6_K"): Float32Array => {
    const perBlock = kind === "q4_0" || kind === "q4_1" ? 32 : 256;
    const bb = kind === "q4_0" ? 18 : kind === "q4_1" ? Q4_1_BLOCK_BYTES : kind === "q5_K" ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;
    const nB = raw.length / bb;
    const out = new Float32Array(nB * perBlock);
    (kind === "q4_0" ? dequantQ4_0 : kind === "q4_1" ? dequantQ4_1 : kind === "q5_K" ? dequantQ5_K : dequantQ6_K)(raw, 0, nB, out);
    return out;
  };
  const ref = glmMoeFfnRefF64(fn, {
    routerW, routerBias: bias,
    expert: (e: number): GlmMoeExpertWeights => {
      const r = rawExpert(e);
      return { gate: deq(r.gate, "q4_0"), up: deq(r.up, "q4_0"), down: deq(r.down, downKind) };
    },
    gateShexp: deq(gateShexpRaw, "q5_K"), upShexp: deq(upShexpRaw, "q5_K"), downShexp: deq(downShexpRaw, "q6_K"),
  });

  // --- GPU ---
  const fnBuf = g.buf(fn);
  const logitsBuf = g.empty(G.nExpert * 4);
  await g.run(gemvF32Wgsl({ K: G.dModel, N: G.nExpert }), [g.buf(routerW), fnBuf, logitsBuf], G.nExpert);
  const logits = new Float32Array(await g.read(logitsBuf, G.nExpert * 4));
  const sel = routerSelect(logits, bias);
  const refSet = new Set(Array.from(ref.experts));
  if (sel.experts.length !== 4 || !Array.from(sel.experts).every((e) => refSet.has(e))) {
    return {
      kernel: name, pass: false, maxAbs: NaN, maxRel: NaN,
      note: `selezione GPU {${Array.from(sel.experts)}} ≠ ref {${Array.from(ref.experts)}}`,
    };
  }

  const gateB = g.empty(G.dFfnExpert * 4);
  const upB = g.empty(G.dFfnExpert * 4);
  const moeOut = g.empty(G.dModel * 4);
  // shexp scrive moeOut (poi i 4 expert accumulano sopra: cur = moe_out + shexp)
  await g.run(gemvQ5KWgsl({ K: G.dModel, N: G.dFfnExpert }), [g.buf(repackKQuant(gateShexpRaw, 0, sbShexp, Q5_K_BLOCK_BYTES)), fnBuf, gateB], G.dFfnExpert);
  await g.run(gemvQ5KWgsl({ K: G.dModel, N: G.dFfnExpert }), [g.buf(repackKQuant(upShexpRaw, 0, sbShexp, Q5_K_BLOCK_BYTES)), fnBuf, upB], G.dFfnExpert);
  await g.run(siluMulWgsl(G.dFfnExpert), [gateB, upB], Math.ceil(G.dFfnExpert / 64));
  await g.run(gemvQ6KWgsl({ K: G.dFfnExpert, N: G.dModel }), [g.buf(repackKQuant(downShexpRaw, 0, sbShexp, Q6_K_BLOCK_BYTES)), gateB, moeOut], G.dModel);

  // buffer di classe con 4 slot; ogni expert selezionato in uno slot diverso
  const slabBuf = g.empty(4 * layout.bytes);
  for (let k = 0; k < 4; k++) {
    const r = rawExpert(sel.experts[k]);
    g.device.queue.writeBuffer(slabBuf, k * layout.bytes, packExpertSlab(r.gate, r.up, r.down, layout) as unknown as BufferSource);
  }
  for (let k = 0; k < 4; k++) {
    const base = k * layout.bytes;
    const bind = (off: number, size: number) => ({ buffer: slabBuf, offset: base + off, size });
    await g.run(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.dFfnExpert, hasBias: false }),
      [bind(layout.gateQs, layout.qsBytes), bind(layout.gateScales, layout.gateScalesBytes), fnBuf, gateB], G.dFfnExpert);
    await g.run(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.dFfnExpert, hasBias: false }),
      [bind(layout.upQs, layout.qsBytes), bind(layout.upScales, layout.gateScalesBytes), fnBuf, upB], G.dFfnExpert);
    await g.run(siluMulWgsl(G.dFfnExpert), [gateB, upB], Math.ceil(G.dFfnExpert / 64));
    await g.run(gemvQuantWgsl({ kind: downKind, K: G.dFfnExpert, N: G.dModel, hasBias: false, scaledAccum: true }),
      [bind(layout.downQs, layout.qsBytes), bind(layout.downScales, layout.downScalesBytes), gateB, moeOut,
        g.buf(new Float32Array([sel.weights[k]]))], G.dModel);
  }
  const got = new Float32Array(await g.read(moeOut, G.dModel * 4));
  const refOut = Float32Array.from(ref.out);
  const res = compare(name, got, refOut, 5e-4, 1e-3);
  res.note = `experts=[${Array.from(sel.experts)}] Σw=${sel.weights.reduce((a, b) => a + b, 0).toFixed(6)}`;
  return res;
}

// Residenza minima (C2 fase 5 slice 2): roundtrip completo su hardware vero —
// import streaming in OPFS con SHA-256 incrementale verificato vs crypto.subtle,
// read SyncAccessHandle su miss, packExpertSlab, writeBuffer allo slot, readback
// GPU byte-ESATTO, LRU con eviction e riuso slot, telemetria. Le classi LRU e
// l'aritmetica del riparto sono già unit-testate in node (engine-residency);
// qui si valida il plumbing browser+GPU che node non può coprire.
async function testResidencyOpfs(g: Gpu): Promise<KResult> {
  const name = "residency-opfs-roundtrip";
  const GU = 1_769_472; // gate/up e down q4_0
  const D41 = 1_966_080;
  // file sintetico: 5 expert classe q4_0 (layer 5) + 2 classe q4_1 (layer 2)
  const plan: Array<{ layer: number; expert: number }> = [
    ...[0, 1, 2, 3, 9].map((expert) => ({ layer: 5, expert })),
    ...[0, 1].map((expert) => ({ layer: 2, expert })),
  ];
  const offsets = new Map<number, { gate: number; up: number; down: number; downBytes: number }>();
  const parts: Uint8Array[] = [];
  let at = 0;
  for (const [i, p] of plan.entries()) {
    const downBytes = p.layer <= 4 ? D41 : GU;
    const gate = randBytes(GU, 7000 + i * 3);
    const up = randBytes(GU, 7001 + i * 3);
    const down = randBytes(downBytes, 7002 + i * 3);
    offsets.set(expertKey(p.layer, p.expert), { gate: at, up: at + GU, down: at + 2 * GU, downBytes });
    parts.push(gate, up, down);
    at += 2 * GU + downBytes;
  }
  const file = new Uint8Array(at);
  { let o = 0; for (const s of parts) { file.set(s, o); o += s.length; } }
  const expectedSha = [...new Uint8Array(await crypto.subtle.digest("SHA-256", file as unknown as BufferSource))]
    .map((b) => b.toString(16).padStart(2, "0")).join("");

  const store = await ExpertOpfsStore.open("ktest-residency.bin");
  const cleanup = async () => {
    store.close();
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle("models");
    await dir.removeEntry("ktest-residency.bin");
  };
  try {
    const url = URL.createObjectURL(new Blob([file as unknown as BlobPart]));
    const imp = await store.importFromUrl(url, expectedSha); // sha sbagliato ⇒ throw
    URL.revokeObjectURL(url);
    if (imp.bytes !== at || store.size() !== at) {
      return { kernel: name, pass: false, maxAbs: NaN, maxRel: NaN, note: `import ${imp.bytes}/${at} B` };
    }
    // hash sbagliato ⇒ throw + truncate (validazione hard)
    let hardFail = false;
    const url2 = URL.createObjectURL(new Blob([file as unknown as BlobPart]));
    try { await store.importFromUrl(url2, "0".repeat(64)); } catch { hardFail = true; }
    URL.revokeObjectURL(url2);
    if (!hardFail || store.size() !== 0) {
      return { kernel: name, pass: false, maxAbs: NaN, maxRel: NaN, note: "mismatch SHA non rifiutato/troncato" };
    }
    const url3 = URL.createObjectURL(new Blob([file as unknown as BlobPart]));
    await store.importFromUrl(url3, expectedSha); // re-import per il resto del test
    URL.revokeObjectURL(url3);

    const rawOf = (layer: number, expert: number): ExpertRawBytes => {
      const o = offsets.get(expertKey(layer, expert))!;
      return {
        gate: store.read(o.gate, GU), up: store.read(o.up, GU), down: store.read(o.down, o.downBytes),
      };
    };
    const cache = new ExpertCache(g.device, {
      budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 },
      maxBindingBytes: 1 << 30, maxBufferBytes: 1 << 30, timing: true,
    });
    const mismatches: string[] = [];
    const checkSlot = async (layer: number, expert: number, slot: { buffer: GPUBuffer; offset: number; layout: { bytes: number } }) => {
      const o = offsets.get(expertKey(layer, expert))!;
      const want = packExpertSlab(
        file.subarray(o.gate, o.gate + GU), file.subarray(o.up, o.up + GU),
        file.subarray(o.down, o.down + o.downBytes),
        layer <= 4 ? SLAB_DOWN_Q4_1 : SLAB_DOWN_Q4_0);
      const got = new Uint8Array(await g.read(slot.buffer, slot.layout.bytes, slot.offset));
      for (let i = 0; i < want.length; i++) {
        if (got[i] !== want[i]) { mismatches.push(`L${layer}e${expert}@${i}`); return; }
      }
    };

    // 4 miss (riempie q4_0), hit con touch, eviction del LRU, riuso slot
    const s0 = cache.ensure(5, 0, rawOf);
    for (const e of [1, 2, 3]) cache.ensure(5, e, rawOf);
    const h0 = cache.ensure(5, 0, rawOf);              // hit + touch (LRU → e1)
    const s9 = cache.ensure(5, 9, rawOf);              // evince e1, riusa il suo slot
    const r1 = cache.ensure(5, 1, rawOf);              // e1 rientra (miss)
    const q0 = cache.ensure(2, 0, rawOf);              // classe q4_1 separata
    const q1 = cache.ensure(2, 1, rawOf);
    await checkSlot(5, 0, s0.slot);
    await checkSlot(5, 9, s9.slot);
    await checkSlot(5, 1, r1.slot);
    await checkSlot(2, 0, q0.slot);
    await checkSlot(2, 1, q1.slot);
    const st = cache.stats();
    cache.destroy();
    const okStats =
      h0.hit && !s9.hit && !r1.hit &&
      st.hits === 1 && st.misses === 8 && st.evictions === 2 &&
      st.occupied.q4_0 === 4 && st.occupied.q4_1 === 2 &&
      st.bytesUploaded === 6 * SLAB_DOWN_Q4_0.bytes + 2 * SLAB_DOWN_Q4_1.bytes &&
      st.readMs > 0 && st.packMs > 0 && st.uploadMs >= 0;
    return {
      kernel: name, pass: mismatches.length === 0 && okStats,
      maxAbs: mismatches.length, maxRel: 0,
      note: `import ${(at / 1e6).toFixed(1)} MB sha-ok in ${imp.ms.toFixed(0)} ms; ` +
        `h${st.hits}/m${st.misses}/ev${st.evictions}; read ${st.readMs.toFixed(1)} pack ${st.packMs.toFixed(1)} up ${st.uploadMs.toFixed(1)} ms` +
        (mismatches.length ? `; MISMATCH ${mismatches.join(",")}` : "") + (okStats ? "" : "; STATS KO"),
    };
  } finally {
    await cleanup();
  }
}

// Forward multi-layer (C2 fase 5 slice 3): mini-modello 2 layer (blk.0 denso +
// blk.1 MoE, classe down Q4_1) con pesi SINTETICI attraverso il path di
// PRODUZIONE (createGlmModel: sync router per layer, ensure con pinned, bind
// group per-slot cached, shexp K-quant, scaledAccum) vs catena di riferimento
// f64 (GlmDenseLayerRefF64 + GlmMoeLayerRefF64: attention naive + selezione
// sort-based indipendenti). Confronta hidden post-2-layer, insiemi top-4 e
// pesi di mixing su 6 posizioni decode con cache expert volutamente stretta
// (eviction e re-fetch inclusi nel percorso).
/**
 * Le uscite di UN run del mini-modello, posizione per posizione. Servono al modo
 * gpu (slice C): l'identita' col run cpu si verifica sui VALORI, non sulle
 * metriche aggregate — due run possono avere lo stesso L2rel verso il
 * riferimento f64 e nondimeno differire fra loro.
 */
interface ModelTrace {
  hidden: Float32Array[]; logits: Float32Array[];
  experts: number[][]; weights: number[][];
}

async function testGlmModel2Layer(
  g: Gpu, select: "cpu" | "shadow" | "gpu" = "cpu",
  io?: { record?: ModelTrace; against?: ModelTrace },
): Promise<KResult> {
  const shadow = select === "shadow";
  const gpuSel = select === "gpu";
  const name = gpuSel ? "glm-model-2layer-gpu" : shadow ? "glm-model-2layer-shadow" : "glm-model-2layer";
  const exBlocks = (G.dModel / 32) * G.dFfnExpert;
  const b32 = (n: number) => (n / 32);
  const sizes: Record<string, [number, "f32" | "q4_0" | "q4_1" | "q8_0" | "q5_K" | "q6_K"]> = {
    "attn_norm.weight": [G.dModel * 4, "f32"],
    "attn_q_a.weight": [b32(G.dModel * G.qLora) * 18, "q4_0"],
    "attn_q_a_norm.weight": [G.qLora * 4, "f32"],
    "attn_q_b.weight": [b32(G.qLora * G.nHead * 256) * 18, "q4_0"],
    "attn_kv_a_mqa.weight": [b32(G.dModel * G.keyLen) * 34, "q8_0"],
    "attn_kv_a_norm.weight": [G.kvLora * 4, "f32"],
    "attn_k_b.weight": [b32(G.qkNope * G.kvLora * G.nHead) * 34, "q8_0"],
    "attn_v_b.weight": [b32(G.kvLora * G.headLenMla * G.nHead) * 34, "q8_0"],
    "attn_output.weight": [b32(G.nHead * G.headLenMla * G.dModel) * 18, "q4_0"],
    "ffn_norm.weight": [G.dModel * 4, "f32"],
    "ffn_gate.weight": [b32(G.dModel * G.dFfnDense) * 18, "q4_0"],
    "ffn_up.weight": [b32(G.dModel * G.dFfnDense) * 18, "q4_0"],
    "ffn_down.weight": [b32(G.dFfnDense * G.dModel) * 20, "q4_1"],
    "ffn_gate_inp.weight": [G.nExpert * G.dModel * 4, "f32"],
    "exp_probs_b.bias": [G.nExpert * 4, "f32"],
    "ffn_gate_shexp.weight": [(G.dModel * G.dFfnExpert / 256) * Q5_K_BLOCK_BYTES, "q5_K"],
    "ffn_up_shexp.weight": [(G.dModel * G.dFfnExpert / 256) * Q5_K_BLOCK_BYTES, "q5_K"],
    "ffn_down_shexp.weight": [(G.dFfnExpert * G.dModel / 256) * Q6_K_BLOCK_BYTES, "q6_K"],
    // head sintetica a vocab RIDOTTA (2048) — la matematica gemvQ6K è la stessa
    "output_norm.weight": [G.dModel * 4, "f32"],
    "output.weight": [(G.dModel * 2048 / 256) * Q6_K_BLOCK_BYTES, "q6_K"],
  };
  const VOCAB_T = 2048;
  const seedOf = (s: string): number => {
    let h = 5381;
    for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0;
    return h;
  };
  const genBytes = (kind: string, bytes: number, seed: number): Uint8Array => {
    if (kind === "f32") {
      const n = bytes / 4;
      const scale = 0.05; // router/bias/norme: ampiezze sane
      const f = randF32(n, seed, scale);
      return new Uint8Array(f.buffer, 0, bytes);
    }
    const src = randBytes(bytes, seed);
    if (kind === "q4_0" || kind === "q8_0") fixScales(src, kind === "q4_0" ? 18 : 34);
    else if (kind === "q4_1") fixScalesAt(src, Q4_1_BLOCK_BYTES, [1, 3]);
    else if (kind === "q5_K") fixScalesAt(src, Q5_K_BLOCK_BYTES, [1, 3]);
    else fixScalesAt(src, Q6_K_BLOCK_BYTES, [209]);
    return src;
  };
  const byName = new Map<string, Uint8Array>();
  const nonExpert = (full: string): Uint8Array => {
    let got = byName.get(full);
    if (!got) {
      const short = full.replace(/^blk\.\d+\./, "");
      const spec = sizes[short];
      if (!spec) throw new Error(`mock source: ${full}?`);
      // le norme a 1±piccolo (rms stabile), il resto come da spec
      if (short.endsWith("norm.weight")) {
        const n = spec[0] / 4;
        const f = randF32(n, seedOf(full), 0.1);
        for (let i = 0; i < n; i++) f[i] += 1;
        got = new Uint8Array(f.buffer, 0, spec[0]);
      } else {
        got = genBytes(spec[1], spec[0], seedOf(full));
      }
      byName.set(full, got);
    }
    return got;
  };
  const expertRaw = new Map<number, { gate: Uint8Array; up: Uint8Array; down: Uint8Array }>();
  const expert = (layer: number, e: number) => {
    const key = layer * 64 + e;
    let got = expertRaw.get(key);
    if (!got) {
      got = {
        gate: genBytes("q4_0", exBlocks * 18, 40_000 + key * 3),
        up: genBytes("q4_0", exBlocks * 18, 40_001 + key * 3),
        down: genBytes("q4_1", exBlocks * Q4_1_BLOCK_BYTES, 40_002 + key * 3), // blk.1 ⇒ classe Q4_1
      };
      expertRaw.set(key, got);
    }
    return got;
  };
  const srcMock: GlmWeightSource = { nonExpert, expert };

  // ---- riferimento f64 (dequant degli stessi byte) ----
  const deqBy = (raw: Uint8Array, kind: string): Float32Array => {
    if (kind === "f32") return new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
    const bb = kind === "q4_0" ? 18 : kind === "q8_0" ? 34 : kind === "q4_1" ? Q4_1_BLOCK_BYTES : kind === "q5_K" ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;
    const per = kind === "q5_K" || kind === "q6_K" ? 256 : 32;
    const nB = raw.length / bb;
    const out = new Float32Array(nB * per);
    (kind === "q4_0" ? dequantQ4_0 : kind === "q8_0" ? dequantQ8_0 : kind === "q4_1" ? dequantQ4_1 : kind === "q5_K" ? dequantQ5_K : dequantQ6_K)(raw, 0, nB, out);
    return out;
  };
  const dq = (l: number, short: string): Float32Array => deqBy(nonExpert(`blk.${l}.${short}`), sizes[short.replace(/^blk\.\d+\./, "")][1]);
  const attnW = (l: number) => ({
    attnNorm: dq(l, "attn_norm.weight"), wQA: dq(l, "attn_q_a.weight"), qANorm: dq(l, "attn_q_a_norm.weight"),
    wQB: dq(l, "attn_q_b.weight"), wKvA: dq(l, "attn_kv_a_mqa.weight"), kvANorm: dq(l, "attn_kv_a_norm.weight"),
    wKB: dq(l, "attn_k_b.weight"), wVB: dq(l, "attn_v_b.weight"), wO: dq(l, "attn_output.weight"),
  });
  const ref0 = new GlmDenseLayerRefF64({
    ...attnW(0), ffnNorm: dq(0, "ffn_norm.weight"),
    wGate: dq(0, "ffn_gate.weight"), wUp: dq(0, "ffn_up.weight"), wDown: dq(0, "ffn_down.weight"),
  });
  const ref1 = new GlmMoeLayerRefF64(attnW(1), dq(1, "ffn_norm.weight"), {
    routerW: dq(1, "ffn_gate_inp.weight"), routerBias: dq(1, "exp_probs_b.bias"),
    expert: (e: number): GlmMoeExpertWeights => {
      const r = expert(1, e);
      return { gate: deqBy(r.gate, "q4_0"), up: deqBy(r.up, "q4_0"), down: deqBy(r.down, "q4_1") };
    },
    gateShexp: dq(1, "ffn_gate_shexp.weight"), upShexp: dq(1, "ffn_up_shexp.weight"), downShexp: dq(1, "ffn_down_shexp.weight"),
  });

  // Finestra d'arena tagliata a 3 slab (KTEST_ARENA_WINDOW): con 7 slot la
  // classe q4_1 — l'unica che blk.1 usa — sta su 3 buffer, quindi lo switch di
  // `ld4` ha archi veri e gli slot cadono anche fuori dal primo buffer. A
  // finestra piena questo test avrebbe nBuf = 1 e non proverebbe niente.
  // In modo gpu la geometria cambia (64 slot = residenza totale, finestra 22
  // slab) ma nBuf resta 3: cambia CHI riempie Sel, non come si indirizza.
  const arenaCfg = ktestArena(gpuSel);
  const model = createGlmModel(g.device, srcMock, {
    nLayer: 2, ctxMax: 16, head: true, vocab: VOCAB_T, select,
    cache: {
      budgetBytes: 0, slotsOverride: arenaCfg.slotsOverride,
      maxBindingBytes: arenaCfg.window, maxBufferBytes: arenaCfg.window, timing: true,
    },
  });
  // Registrazione dei contatori CPU accesa (gpu=false: senza timestamp-query
  // gpuBusy resta null, e qui servono solo submit e routerSyncs). Senza questa
  // riga restavano a zero e l'invariante del design §6 — "in shadow i contatori
  // non cambiano" — non era testata da nessuno: un submit aggiunto nel ramo
  // shadow sarebbe passato sotto ogni gate.
  model.setTelemetry(true, false);
  // ref f64 dell'head: rms(output_norm) + matvec Q6_K dequant
  const outNormF = deqBy(nonExpert("output_norm.weight"), "f32");
  const outWF = deqBy(nonExpert("output.weight"), "q6_K");
  const headRefF64 = (h: Float64Array): Float64Array => {
    let ss = 0;
    for (let i = 0; i < G.dModel; i++) ss += h[i] * h[i];
    const sc = 1 / Math.sqrt(ss / G.dModel + G.rmsEps);
    const out = new Float64Array(VOCAB_T);
    for (let r = 0; r < VOCAB_T; r++) {
      let acc = 0;
      const base = r * G.dModel;
      for (let i = 0; i < G.dModel; i++) acc += outWF[base + i] * (h[i] * sc * outNormF[i]);
      out[r] = acc;
    }
    return out;
  };
  const NPOS = 6;
  let l2e = 0, l2r = 0, maxAbs = 0, maxRel = 0, wMaxRel = 0, logitMaxRel = 0;
  let argmaxOk = 0;
  const problems: string[] = [];
  // --- accumulatori del solo modo shadow (router GPU + resolve in ombra) ---
  let gpuSetOk = 0, gpuOrderOk = 0, gpuWMaxRel = 0, gpuResolved = 0, gpuMiss = 0, vramChecked = 0;
  // --- accumulatori del solo modo gpu: identita' VALORE PER VALORE col run cpu ---
  let cpuBitEq = 0, cpuBitTot = 0, cpuMaxRel = 0, cpuLogitBitEq = 0, cpuLogitMaxRel = 0, cpuWMaxRel = 0;
  let cpuArgmaxEq = 0;
  // --- osservatore INDIPENDENTE dei submit (solo modo gpu) ---
  // `telemetry().submits` e' un contatore che il forward incrementa da se': se
  // qualcuno aggiungesse un submit intermedio SENZA incrementarlo, il gate
  // "1 submit per token" resterebbe verde su un contatore che mente. Qui si conta
  // la cosa vera — le chiamate a `queue.submit` — e poi si chiede che i due
  // numeri coincidano. Il wrap e' locale al caso e si rimuove nel finally: e' il
  // device condiviso di tutti i ktest.
  const queue = g.device.queue as GPUQueue & { submit: GPUQueue["submit"] };
  const submitVero = queue.submit.bind(queue);
  let submitOsservati = 0;
  if (gpuSel) {
    queue.submit = (buffers: Iterable<GPUCommandBuffer>) => { submitOsservati++; return submitVero(buffers); };
  }
  // Il preload della residenza totale e' gia' avvenuto (in createGlmModel) e i
  // suoi submit non ci sono: `writeBuffer` non passa da qui. Il conteggio parte
  // pulito dal primo forward.
  for (let p = 0; p < NPOS; p++) {
    const xIn = randF32(G.dModel, 60_000 + p, 0.5);
    const refH = ref1.forward(ref0.forward(xIn));
    const refR = ref1.lastRouting!;
    const got = await model.forward(xIn, p, true);
    if (got.routing.length !== 1) { problems.push(`pos ${p}: ${got.routing.length} routing`); break; }
    // registrazione PRIMA di qualunque `continue`: se una posizione uscisse dal
    // giro senza essere registrata, gli indici della traccia non sarebbero piu'
    // quelli delle posizioni e il confronto gpu-vs-cpu confronterebbe token diversi
    if (io?.record) {
      io.record.hidden.push(got.hidden.slice());
      io.record.logits.push(got.logits!.slice());
      io.record.experts.push(Array.from(got.routing[0].experts));
      io.record.weights.push(Array.from(got.routing[0].weights));
    }
    const gotSet = new Set(Array.from(got.routing[0].experts));
    if (gotSet.size !== 4 || !Array.from(refR.experts).every((e) => gotSet.has(e))) {
      problems.push(`pos ${p}: top4 {${Array.from(got.routing[0].experts)}} ≠ ref {${Array.from(refR.experts)}}`);
      continue;
    }
    for (let k = 0; k < 4; k++) {
      const e = got.routing[0].experts[k];
      const kRef = Array.from(refR.experts).indexOf(e);
      const d = Math.abs(got.routing[0].weights[k] - refR.weights[kRef]);
      wMaxRel = Math.max(wMaxRel, d / Math.max(Math.abs(refR.weights[kRef]), 1e-9));
    }
    // --- shadow: cosa AVREBBE scelto il router GPU, e come ha risolto gli slot ---
    // NIENTE `continue` qui dentro: le posizioni contano tutte nel denominatore
    // di L2rel e dell'argmax, e un ramo shadow che ne saltasse una lascerebbe la
    // nota a dire "6 pos" su 5 misurate.
    if (shadow) {
      const cpu = Array.from(got.routing[0].experts);
      const gpu = got.routing[0].gpu;
      const vram = got.routing[0].vram;
      if (!gpu || !vram) {
        problems.push(`pos ${p}: modo shadow senza le due regioni di Sel`);
      } else {
        const ids = Array.from(gpu.experts);
        if (new Set(ids).size === 4 && ids.every((e) => cpu.includes(e))) {
          gpuSetOk++;
          if (ids.every((e, k) => e === cpu[k])) gpuOrderOk++;
          for (let k = 0; k < 4; k++) {
            const want = got.routing[0].weights[cpu.indexOf(ids[k])];
            gpuWMaxRel = Math.max(gpuWMaxRel, Math.abs(gpu.weights[k] - want) / Math.max(Math.abs(want), 1e-9));
          }
        } else {
          problems.push(`pos ${p}: top4 GPU {${ids}} != CPU {${cpu}}`);
        }
        // Il router GPU gira PRIMA degli `ensure` del suo layer: alla posizione 0
        // la tabella e' ancora tutta MISS per costruzione, ed e' l'unico modo di
        // vedere che il resolve la legge davvero (se leggesse zeri darebbe slot 0).
        const bySlot = new Map<number, number>();
        for (let k = 0; k < 4; k++) {
          const miss = (gpu.flags[k] & 1) !== 0;
          if (miss !== (gpu.slots[k] === 0xffffffff)) {
            problems.push(`pos ${p} k${k}: flags ${gpu.flags[k]} incoerente con slot ${gpu.slots[k]}`);
          }
          if (miss) { gpuMiss++; continue; }
          gpuResolved++;
          if (gpu.slots[k] >= KTEST_SLOTS.q4_1) problems.push(`pos ${p} k${k}: slot ${gpu.slots[k]} fuori da [0,${KTEST_SLOTS.q4_1})`);
          const prev = bySlot.get(gpu.slots[k]);
          if (prev !== undefined) problems.push(`pos ${p}: expert ${prev} e ${ids[k]} sullo stesso slot ${gpu.slots[k]}`);
          bySlot.set(gpu.slots[k], ids[k]);
        }
        if (p === 0 && gpu.flags.some((f) => (f & 1) === 0)) {
          problems.push("pos 0: uno slot risolto prima del primo ensure (tabella non azzerata?)");
        }
        // --- Sel di PRODUZIONE riletta dalla VRAM (R5, lato che decide) ---
        // E' quello che i kernel expert hanno letto davvero: se la writeBuffer
        // della CPU finisse nel layer sbagliato, arrivasse in ritardo o venisse
        // sovrascritta, qui i quattro id non sarebbero quelli di routerSelect.
        // I pesi si confrontano dopo `Math.fround`: in Sel stanno in f32, la CPU
        // li decide in f64 — la troncatura e' attesa, un valore diverso no.
        for (let k = 0; k < 4; k++) {
          if (vram.experts[k] !== cpu[k]) {
            problems.push(`pos ${p} k${k}: Sel in VRAM ha id ${vram.experts[k]}, la CPU aveva scelto ${cpu[k]}`);
          }
          if (vram.weights[k] !== Math.fround(got.routing[0].weights[k])) {
            problems.push(`pos ${p} k${k}: peso in VRAM ${vram.weights[k]} != f32(${got.routing[0].weights[k]})`);
          }
          if (vram.flags[k] !== 0) problems.push(`pos ${p} k${k}: flags ${vram.flags[k]} nella Sel di produzione`);
          if (vram.slots[k] >= KTEST_SLOTS.q4_1) {
            problems.push(`pos ${p} k${k}: slot di produzione ${vram.slots[k]} fuori da [0,${KTEST_SLOTS.q4_1})`);
          }
          vramChecked++;
        }
        if (new Set(Array.from(vram.slots)).size !== 4) {
          problems.push(`pos ${p}: la Sel di produzione ripete uno slot (${Array.from(vram.slots)})`);
        }
      }
    }
    // --- gpu (slice C): comanda il router GPU, e la Sel di produzione E' la
    // decisione. Due cose da verificare che nessun altro caso vede: che la
    // residenza totale abbia retto (zero flag MISS, slot distinti e nel range) e
    // che il risultato sia quello del run cpu sugli stessi input.
    if (gpuSel) {
      const vram = got.routing[0].vram;
      if (!vram) {
        problems.push(`pos ${p}: modo gpu senza la Sel riletta dalla VRAM`);
      } else {
        for (let k = 0; k < 4; k++) {
          if ((vram.flags[k] & 1) !== 0) {
            problems.push(`pos ${p} k${k}: flag MISS nella Sel di produzione (residenza totale violata)`);
          }
          if (vram.slots[k] >= KTEST_SLOTS_GPU.q4_1) {
            problems.push(`pos ${p} k${k}: slot ${vram.slots[k]} fuori da [0,${KTEST_SLOTS_GPU.q4_1})`);
          }
          // Il preload carica gli expert in ordine 0..63 e la free list e' LIFO
          // discendente (residency: `nSlots-1-i`, `pop()`) ⇒ l'expert `e` finisce
          // nello slot `e`. Non e' un dettaglio decorativo: e' l'unico modo di
          // vedere che il resolve legge la RIGA GIUSTA della slotTable e non una
          // qualunque — con id e slot scorrelati, un tableBase sbagliato darebbe
          // comunque slot legali e L2rel plausibile.
          if (vram.slots[k] !== vram.experts[k]) {
            problems.push(`pos ${p} k${k}: expert ${vram.experts[k]} risolto allo slot ${vram.slots[k]} (atteso ${vram.experts[k]}: preload in ordine)`);
          }
          vramChecked++;
        }
        if (new Set(Array.from(vram.slots)).size !== 4) {
          problems.push(`pos ${p}: la Sel di produzione ripete uno slot (${Array.from(vram.slots)})`);
        }
      }
      const ref = io?.against;
      if (!ref) {
        problems.push("modo gpu senza la traccia del run cpu: l'identita' non e' verificabile");
      } else {
        const rh = ref.hidden[p], rl = ref.logits[p], rE = ref.experts[p], rW = ref.weights[p];
        for (let i = 0; i < G.dModel; i++) {
          if (Object.is(got.hidden[i], rh[i])) cpuBitEq++;
          cpuMaxRel = Math.max(cpuMaxRel, Math.abs(got.hidden[i] - rh[i]) / Math.max(Math.abs(rh[i]), 1e-6));
        }
        cpuBitTot += G.dModel;
        let aGpu = 0, aCpu = 0;
        for (let r = 0; r < VOCAB_T; r++) {
          if (Object.is(got.logits![r], rl[r])) cpuLogitBitEq++;
          cpuLogitMaxRel = Math.max(cpuLogitMaxRel, Math.abs(got.logits![r] - rl[r]) / Math.max(Math.abs(rl[r]), 1e-3));
          if (got.logits![r] > got.logits![aGpu]) aGpu = r;
          if (rl[r] > rl[aCpu]) aCpu = r;
        }
        // ONESTA' SU QUESTA RIGA: l'argmax dei due run coincide gia' per
        // costruzione — entrambi sono gated a `argmax === refArg` contro lo stesso
        // riferimento f64, quindi l'uguaglianza fra loro e' IMPLICATA e il potere
        // discriminante e' zero. Resta perche' rende esplicito nel report cio' che
        // altrove e' solo deducibile, non perche' aggiunga copertura: i portatori
        // veri dell'identita' sono `cpuMaxRel` (hidden), gli id per-k qui sotto e
        // `cpuWMaxRel` (pesi).
        if (aGpu === aCpu) cpuArgmaxEq++;
        else problems.push(`pos ${p}: argmax gpu ${aGpu} != cpu ${aCpu}`);
        // Gli id devono essere IDENTICI, per k e nell'ordine (il router GPU
        // replica routerSelect, ordinamento incluso); i pesi no — la CPU li decide
        // in f64 e la GPU in f32, ed e' il narrowing gia' misurato nello slice B
        // (wMaxRel 4.43e-7 sul corpus).
        // R8: questo gate tratta OGNI differenza di id come difetto, anche un flip
        // near-tie legittimo. A 6 posizioni × 1 layer MoE il rate misurato in it.16
        // (13 flip su 1 438 604 selezioni = 9e-6) non scatta mai — l'attesa e'
        // ~5e-5 flip nell'intero caso. Chi allarga il caso (piu' posizioni, piu'
        // layer, corpus vero) deve aspettarsi che prima o poi scatti, e allora la
        // domanda non e' "e' rotto" ma "e' un near-tie": lo dice glmroute.
        for (let k = 0; k < 4; k++) {
          if (got.routing[0].experts[k] !== rE[k]) {
            problems.push(`pos ${p} k${k}: expert gpu ${got.routing[0].experts[k]} != cpu ${rE[k]}`);
          }
          cpuWMaxRel = Math.max(cpuWMaxRel, Math.abs(got.routing[0].weights[k] - rW[k]) / Math.max(Math.abs(rW[k]), 1e-9));
        }
      }
    }
    for (let i = 0; i < G.dModel; i++) {
      const d = Math.abs(got.hidden[i] - refH[i]);
      maxAbs = Math.max(maxAbs, d);
      maxRel = Math.max(maxRel, d / Math.max(Math.abs(refH[i]), 1e-6));
      l2e += d * d;
      l2r += refH[i] * refH[i];
    }
    // head: logits vs ref f64 + argmax
    const refL = headRefF64(refH);
    let refArg = 0;
    for (let r = 1; r < VOCAB_T; r++) if (refL[r] > refL[refArg]) refArg = r;
    let gotArg = 0;
    const gl = got.logits!;
    for (let r = 1; r < VOCAB_T; r++) if (gl[r] > gl[gotArg]) gotArg = r;
    if (gotArg === refArg) argmaxOk++;
    for (let r = 0; r < VOCAB_T; r++) {
      logitMaxRel = Math.max(logitMaxRel, Math.abs(gl[r] - refL[r]) / Math.max(Math.abs(refL[r]), 1e-3));
    }
  }
  // Ripristino del `submit` vero, e SOLO dove era stato sostituito: negli altri
  // modi il device condiviso non si tocca affatto. Non serve un `finally`: se un
  // forward alza, il runner ktest interrompe l'intera esecuzione (nessun caso
  // successivo gira), quindi il wrap non puo' sopravvivere al caso — e comunque
  // delega sempre al vero.
  if (gpuSel) queue.submit = submitVero;
  const st = model.cacheStats();
  // Il gate 2 dello slice A chiede nBuf >= 3: se la finestra o gli slot
  // cambiassero, questo test tornerebbe a girare a buffer singolo senza dirlo.
  const nBufTest = arenaNeeds({
    budgetBytes: 0, slotsOverride: arenaCfg.slotsOverride,
    maxBufferBytes: arenaCfg.window, maxBindingBytes: arenaCfg.window,
  }).arenaBuffers;
  if (nBufTest < 3) problems.push(`arena a nBuf=${nBufTest}: lo switch di ld4 non ha archi`);
  // Dispatch: due asserzioni DISTINTE, non una tautologia. Prima di C3a il gate
  // era `dispatchesPerToken === 61`, cioe' la formula confrontata con se stessa:
  // passava sempre, e non avrebbe mai visto un pass WGSL aggiunto o tolto.
  // Ora: (a) il piano vale quello che ci aspettiamo; (b) il runtime EMETTE
  // davvero quel numero piu' i 2 della testa (rms + lm_head), che il piano non
  // contiene. E' (b) a rendere il gate un test.
  const tel = await model.telemetry();
  const measuredPerToken = tel.dispatches / Math.max(1, tel.forwards);
  // Mini-modello: 2 layer (1 denso blk.0 + 1 MoE). Scritto come somma dei
  // termini, non come numero: quando una fase cambia la catena, qui si vede
  // QUALE termine e' cambiato. Fase 4b: la catena expert e' passata da 4
  // dispatch (gate, up, silu, down) a 2 (gate+up+silu fusi, down) ⇒ il termine
  // MoE scende da 23 a 15 e il totale da 61 a 53. Fase 4c: l'attention split
  // spezza il kernel MLA in part+reduce ⇒ il termine attn sale da 16 a 17 per
  // layer e il totale da 53 a 55. It.13: lo shexp fonde gate+up+silu Q5_K ⇒ il
  // suo termine scende da 4 a 2 e il totale da 55 a 53. Slice B: in modo shadow
  // il router+resolve su GPU aggiunge 1 dispatch per layer MoE (53 → 54); in
  // modo "cpu" il termine NON c'e', ed e' il gate a dirlo. Slice C: il modo gpu
  // ha lo STESSO termine dello shadow (54) — l'interruttore sposta i confini dei
  // submit e toglie i readback, non i dispatch.
  const ROUTER_GPU = shadow || gpuSel ? 1 : 0;
  const PLANNED = 17 * 2 + 6 * 1 + (2 + ROUTER_GPU + 2 + G.nExpertUsed * 2 + 1) * 1, HEAD = 2;
  const dispatchOk = model.dispatchesPerTokenPlanned === PLANNED && measuredPerToken === PLANNED + HEAD;
  if (!dispatchOk) {
    problems.push(`dispatch: piano ${model.dispatchesPerTokenPlanned} (atteso ${PLANNED}), ` +
      `misurati ${measuredPerToken}/token (atteso ${PLANNED + HEAD} = piano + testa)`);
  }
  const planned = model.dispatchesPerTokenPlanned;
  model.destroy();
  const l2 = Math.sqrt(l2e / Math.max(l2r, 1e-30));
  // In shadow si aggiungono tre condizioni: l'insieme del router GPU coincide
  // con quello della CPU su OGNI posizione, i pesi stanno nella tolleranza f32
  // di it.9, e il resolve ha risolto almeno uno slot davvero (senza questo, un
  // resolve che marcasse tutto MISS passerebbe le prime due).
  // (piu' il cross-check sulla Sel di produzione: 4 entry per posizione, tutte
  // controllate, altrimenti la rilettura non ha guardato niente)
  const shadowOk = !shadow || (gpuSetOk === NPOS && gpuWMaxRel <= 1e-5 && gpuResolved > 0 && gpuMiss > 0
    && vramChecked === NPOS * G.nExpertUsed);
  const submitsPerToken = tel.submits / Math.max(1, tel.forwards);
  const syncsPerToken = tel.routerSyncs / Math.max(1, tel.forwards);
  // Il gate dello slice C, nei termini del design §4:
  //  - `routerSyncs === 0` e `submits === 1` per token: e' L'INTERRUTTORE. Se
  //    uno dei due non ci sta, il modo gpu sta ancora facendo il sync per layer.
  //  - `selMiss === 0`: nessun expert saltato (R6).
  //  - zero evizioni: la residenza totale non e' una speranza, e' misurata.
  //  - residenza totale COMPLETA: `misses` == parco del modello e occupancy per
  //    classe == parco per classe. Le sole 24 entry di Sel osservate nei forward
  //    dicono che 24 risoluzioni sono andate a segno, non che i 64 expert siano
  //    tutti dentro: il preload va verificato per intero, o "totale" e' una parola.
  //  - un osservatore INDIPENDENTE del submit (wrap di queue.submit) coerente col
  //    contatore di telemetria: il gate non deve poggiare su chi si auto-dichiara.
  //  - identita' col run cpu. Fra i due run cambia UNA cosa sola: il peso di
  //    mixing, calcolato in f32 dal router GPU invece che in f64 da routerSelect
  //    (it.9: i due tengono a 1e-6; slice B su corpus vero: 4.43e-7). Il peso
  //    entra LINEARE nella somma expert ⇒ su `hidden` la perturbazione relativa e'
  //    al piu' quella sul peso. Le soglie sono derivate dalla MISURA, non dal
  //    limite teorico: pesi 1.27e-7 ⇒ gate 1e-6 (~8×), hidden 2.50e-7 ⇒ gate 1e-6
  //    (~4×, ed e' anche la soglia R2 del design per la riassociazione benigna).
  //    Gli id degli expert devono coincidere ESATTAMENTE.
  //  - sui LOGIT la stessa soglia sarebbe sbagliata, e la prima stesura di questo
  //    gate lo era: la testa (rms + GEMV su 2048 termini con cancellazione, e il
  //    pavimento 1e-3 al denominatore) AMPLIFICA di ~1e3 — con hidden a 2.5e-7 i
  //    logit stanno a 2.56e-4, cioe' la stessa distanza a cui il run cpu sta gia'
  //    dal riferimento f64 (1.91e-4). Serviva una soglia INDIPENDENTE dai gate
  //    esistenti (quella per disuguaglianza triangolare, 2·5e-3, e' implicata da
  //    loro: non discrimina niente), quindi si prende la banda della misura:
  //    2.56e-4 ⇒ gate 1e-3, margine ~4× come per le altre due.
  // Il parco atteso arriva dalla STESSA funzione che la precondizione usa per
  // decidere se il modo gpu puo' partire: se sbagliasse, sbaglierebbe in due
  // punti nello stesso modo — ma allora il caso negativo (7 slot per 64 expert)
  // non troverebbe piu' il suo messaggio.
  const parco = modelExpertPark(2);
  const preloadOk = !gpuSel || (
    st.misses === parco.q4_0 + parco.q4_1
    && st.occupied.q4_0 === parco.q4_0 && st.occupied.q4_1 === parco.q4_1);
  if (gpuSel && !preloadOk) {
    problems.push(`preload incompleto: m${st.misses} occupied q4_0=${st.occupied.q4_0} q4_1=${st.occupied.q4_1} ` +
      `(atteso m${parco.q4_0 + parco.q4_1}, q4_0=${parco.q4_0}, q4_1=${parco.q4_1})`);
  }
  const submitOk = !gpuSel || (submitOsservati === NPOS && submitOsservati === tel.submits);
  if (gpuSel && !submitOk) {
    problems.push(`submit osservati ${submitOsservati} in ${NPOS} forward (attesi ${NPOS}), ` +
      `telemetria ne dichiara ${tel.submits}`);
  }
  const gpuOk = !gpuSel || (
    syncsPerToken === 0 && submitsPerToken === 1 && tel.selMiss === 0 && st.evictions === 0
    && vramChecked === NPOS * G.nExpertUsed && cpuArgmaxEq === NPOS && preloadOk && submitOk
    && cpuMaxRel <= 1e-6 && cpuLogitMaxRel <= 1e-3 && cpuWMaxRel <= 1e-6);
  const pass = problems.length === 0 && l2 <= 1e-3 && wMaxRel <= 1e-3 && st.misses > 0
    && dispatchOk && argmaxOk === NPOS && logitMaxRel <= 5e-3 && shadowOk && gpuOk;
  return {
    kernel: name, pass, maxAbs, maxRel,
    note: `${NPOS} pos, L2rel=${l2.toExponential(2)}, wMaxRel=${wMaxRel.toExponential(2)}, ` +
      `argmax ${argmaxOk}/${NPOS}, logitMaxRel=${logitMaxRel.toExponential(2)}, ` +
      `h${st.hits}/m${st.misses}/ev${st.evictions}, arena nBuf=${nBufTest}, ` +
      `${measuredPerToken} dispatch/token misurati ` +
      `(piano ${planned} + testa ${HEAD})` +
      `, ${submitsPerToken} submit / ${syncsPerToken} sync per token` +
      (shadow
        ? `; router GPU: set ${gpuSetOk}/${NPOS}, ordine ${gpuOrderOk}/${NPOS}, ` +
          `wMaxRel=${gpuWMaxRel.toExponential(2)}, slot risolti ${gpuResolved}, miss ${gpuMiss}, ` +
          `Sel di produzione riletta ${vramChecked}/${NPOS * G.nExpertUsed}`
        : "") +
      (gpuSel
        ? `; selMiss=${tel.selMiss}, Sel di produzione riletta ${vramChecked}/${NPOS * G.nExpertUsed}, ` +
          `preload ${st.occupied.q4_0 + st.occupied.q4_1}/${parco.q4_0 + parco.q4_1} expert, ` +
          `submit osservati ${submitOsservati}/${NPOS} (telemetria ${tel.submits}); ` +
          // I portatori dell'identita' sono questi tre — hidden, id, pesi: sono le
          // uniche misure che il caso cpu non fa gia'. `argmax` e i logit sono
          // riportati per leggibilita', ma il primo e' implicato dai gate vs f64.
          `vs run cpu: hidden maxRel=${cpuMaxRel.toExponential(2)} (bit-identici ${cpuBitEq}/${cpuBitTot}), ` +
          `id expert identici per-k, pesi maxRel=${cpuWMaxRel.toExponential(2)}; ` +
          `logits maxRel=${cpuLogitMaxRel.toExponential(2)} (bit-identici ${cpuLogitBitEq}/${NPOS * VOCAB_T}), ` +
          `argmax ${cpuArgmaxEq}/${NPOS}`
        : "") +
      (problems.length ? `; ${problems.join("; ")}` : ""),
    // le grandezze che il confronto cpu-vs-shadow deve trovare IDENTICHE
    metrics: {
      l2, wMaxRel, maxAbs, maxRel, logitMaxRel, argmaxOk,
      hits: st.hits, misses: st.misses, evictions: st.evictions,
      // §6 del design: submit e routerSyncs NON cambiano fra A e B — sono i due
      // numeri che dicono se il sync per layer e' ancora dov'era. In C cambiano
      // entrambi, ed e' il punto: 1 submit e 0 sync per token.
      submits: submitsPerToken, routerSyncs: syncsPerToken, selMiss: tel.selMiss,
      dispatchPerToken: measuredPerToken, planned,
    },
  };
}

// Lo slice B dichiara che in ombra la CPU decide ANCORA tutto: il forward non
// cambia di un bit. Non e' verificabile dentro un caso solo — si verifica
// mettendo i due run uno accanto all'altro e chiedendo che ogni misura
// coincida ESATTAMENTE (non entro tolleranza: identica), tranne il conteggio
// dei dispatch, che deve differire di 1 per layer MoE e non di piu'.
// `submits` e `routerSyncs` sono nella lista per il design §6: in B il sync per
// layer resta esattamente dov'era, e un submit in piu' nel ramo shadow — che
// nessun'altra asserzione vedrebbe — cade qui.
function testShadowInvariance(cpu: KResult, shadow: KResult, nMoeLayer: number): KResult {
  const a = cpu.metrics!, b = shadow.metrics!;
  const problems: string[] = [];
  for (const k of [
    "l2", "wMaxRel", "maxAbs", "maxRel", "logitMaxRel", "argmaxOk",
    // `selMiss` e' nella lista per R6: in shadow la Sel di PRODUZIONE la scrive
    // ancora la CPU dopo gli ensure, quindi resta a 0 come in cpu — un resolve
    // che finisse per sbaglio nella regione di produzione lo si vedrebbe qui.
    "hits", "misses", "evictions", "submits", "routerSyncs", "selMiss",
  ]) {
    if (!Object.is(a[k], b[k])) problems.push(`${k}: cpu ${a[k]} != shadow ${b[k]}`);
  }
  // e i due contatori devono essere quelli VERI del mini-modello, non zero:
  // un `setTelemetry` dimenticato renderebbe il confronto sopra una tautologia
  if (!(a.submits > 0 && a.routerSyncs > 0)) {
    problems.push(`contatori spenti (submits ${a.submits}, routerSyncs ${a.routerSyncs}): il confronto non prova niente`);
  }
  const dDispatch = b.dispatchPerToken - a.dispatchPerToken;
  if (dDispatch !== nMoeLayer) problems.push(`dispatch/token: +${dDispatch}, atteso +${nMoeLayer}`);
  if (b.planned - a.planned !== nMoeLayer) problems.push(`piano: +${b.planned - a.planned}, atteso +${nMoeLayer}`);
  return {
    kernel: "glm-model-shadow-invariance",
    pass: problems.length === 0,
    maxAbs: Math.abs(b.l2 - a.l2), maxRel: 0,
    note: `L2rel ${a.l2.toExponential(2)} in entrambi, ${a.dispatchPerToken} → ${b.dispatchPerToken} dispatch/token, `
      + `submit ${a.submits} e sync ${a.routerSyncs} per token invariati`
      + (problems.length ? `; ${problems.join("; ")}` : ""),
  };
}

// Il modo gpu a residenza PARZIALE (C3a fase 4 slice C): deve rifiutarsi di
// partire, con un errore che dice anche PERCHE'. E' l'altra meta' del gate — un
// interruttore che accetta la residenza parziale non e' un interruttore, e'
// corruzione silenziosa (R6): senza sync la CPU non pinna piu' niente, e uno slot
// risolto puo' essere evinto DOPO la risoluzione nello stesso token.
// La sorgente pesi ALZA a ogni lettura: cosi' il caso prova anche che la
// precondizione scatta prima di toccare un solo byte di modello (se un giorno si
// spostasse dopo, il messaggio sarebbe quello del mock e il test cadrebbe).
async function testGlmGpuPartialResidency(g: Gpu): Promise<KResult> {
  const name = "glm-model-gpu-residenza-parziale";
  const srcNever: GlmWeightSource = {
    nonExpert: (n: string) => { throw new Error(`mock: precondizione tardiva, letto ${n}`); },
    expert: () => { throw new Error("mock: precondizione tardiva, letto un expert"); },
  };
  let msg = "";
  let modello: ReturnType<typeof createGlmModel> | null = null;
  try {
    modello = createGlmModel(g.device, srcNever, {
      nLayer: 2, ctxMax: 16, select: "gpu",
      cache: {
        budgetBytes: 0, slotsOverride: KTEST_SLOTS, // 7 slot q4_1 per 64 expert
        maxBindingBytes: KTEST_ARENA_WINDOW, maxBufferBytes: KTEST_ARENA_WINDOW,
      },
    });
  } catch (e) {
    msg = e instanceof Error ? e.message : String(e);
  }
  modello?.destroy(); // non deve succedere: se succede, niente VRAM appesa
  // Frammenti ASSERITI, non "contiene qualcosa": il numero vero (7 slot per 64),
  // la ragione (residenza totale) e il seam che la review ha segnalato.
  const want = [
    "residenza totale",
    `classe q4_1: ${KTEST_SLOTS.q4_1} slot per ${G.nExpert} expert`,
    "evinto DOPO la risoluzione",
  ];
  const mancanti = want.filter((w) => !msg.includes(w));
  const pass = modello === null && mancanti.length === 0;
  return {
    kernel: name, pass, maxAbs: 0, maxRel: 0,
    note: modello === null
      ? `errore atteso: "${msg.slice(0, 160)}…"` + (mancanti.length ? `; MANCANO ${mancanti.join(" | ")}` : "")
      : "il modo gpu e' partito a residenza parziale (7 slot per 64 expert)",
  };
}

// Famiglia fusa portata su GLM (C3a fase 4b). Stessi riferimenti f64 dei
// gemelli generici che sostituiscono: se il risultato coincide entro le
// tolleranze gia' in uso per i gemv, la struttura nuova non ha cambiato la
// matematica — ha cambiato solo come legge la memoria.
async function testPairGemvSiluFast(g: Gpu): Promise<KResult> {
  const K = G.dModel, N = G.dFfnExpert;
  const nBlocks = (K / 32) * N;
  const gSrc = randBytes(nBlocks * 18, 5501); fixScales(gSrc, 18);
  const uSrc = randBytes(nBlocks * 18, 5502); fixScales(uSrc, 18);
  const gw = new Float32Array(nBlocks * 32), uw = new Float32Array(nBlocks * 32);
  dequantQ4_0(gSrc, 0, nBlocks, gw);
  dequantQ4_0(uSrc, 0, nBlocks, uw);
  const x = randF32(K, 5503);
  const ref = new Float32Array(N);
  for (let r = 0; r < N; r++) {
    let ag = 0, au = 0;
    for (let i = 0; i < K; i++) { ag += gw[r * K + i] * x[i]; au += uw[r * K + i] * x[i]; }
    ref[r] = (ag / (1 + Math.exp(-ag))) * au;
  }
  const gp = repackQ4_0(gSrc, 0, nBlocks), up = repackQ4_0(uSrc, 0, nBlocks);
  const out = g.empty(N * 4);
  await g.run(pairGemvSiluFastWgsl({ K, N }),
    [g.buf(gp.qs), g.buf(gp.scales), g.buf(up.qs), g.buf(up.scales), g.buf(x), out],
    gemvGrid(N / 4)[0]);
  return compare("pair-gemv-silu-fast-exp", new Float32Array(await g.read(out, N * 4)), ref, 2e-4, 1e-3);
}

// Famiglia fast portata sui K-quant (C3a fase 4b it.13). Stessi riferimenti
// dequant CPU e stesse tolleranze dei casi gemv-q5_K/gemv-q6_K: la struttura
// nuova (8 thread per superblocco, word in registri, x in shared) deve dare la
// stessa risposta dei gemelli lenti — cambia come legge la memoria, non la
// matematica.
async function testPairGemvSiluQ5KFast(g: Gpu): Promise<KResult> {
  const K = G.dModel, N = G.dFfnExpert;
  const nBlocks = (K / 256) * N;
  const gSrc = randBytes(nBlocks * Q5_K_BLOCK_BYTES, 7701); fixScalesAt(gSrc, Q5_K_BLOCK_BYTES, [1, 3]);
  const uSrc = randBytes(nBlocks * Q5_K_BLOCK_BYTES, 7702); fixScalesAt(uSrc, Q5_K_BLOCK_BYTES, [1, 3]);
  const gw = new Float32Array(nBlocks * 256), uw = new Float32Array(nBlocks * 256);
  dequantQ5_K(gSrc, 0, nBlocks, gw);
  dequantQ5_K(uSrc, 0, nBlocks, uw);
  const x = randF32(K, 7703);
  const ref = new Float32Array(N);
  for (let r = 0; r < N; r++) {
    let ag = 0, au = 0;
    for (let i = 0; i < K; i++) { ag += gw[r * K + i] * x[i]; au += uw[r * K + i] * x[i]; }
    ref[r] = (ag / (1 + Math.exp(-ag))) * au;
  }
  const out = g.empty(N * 4);
  await g.run(pairGemvSiluQ5KFastWgsl({ K, N }), [
    g.buf(repackKQuant(gSrc, 0, nBlocks, Q5_K_BLOCK_BYTES)),
    g.buf(repackKQuant(uSrc, 0, nBlocks, Q5_K_BLOCK_BYTES)),
    g.buf(x), out,
  ], gemvGrid(N)[0]);
  // Tolleranza IMPORTATA da kquantfast.ts, non ricopiata: la sua derivazione
  // (pavimento f32 della forma fattorizzata + contrazione FMA) e' misurata in
  // tests/engine-kquant-f32floor.test.ts, che importa la stessa costante.
  return compare("pair-gemv-silu-q5k-fast-shexp", new Float32Array(await g.read(out, N * 4)), ref,
    KQUANT_FAST_Q5K_PAIR_REL_TOL, 1e-3);
}

async function testGemvQ6KFast(g: Gpu, K: number, N: number): Promise<KResult> {
  const nBlocks = (K / 256) * N;
  const src = randBytes(nBlocks * Q6_K_BLOCK_BYTES, 7800 + K + N);
  fixScalesAt(src, Q6_K_BLOCK_BYTES, [209]);
  const w = new Float32Array(nBlocks * 256);
  dequantQ6_K(src, 0, nBlocks, w);
  const x = randF32(K, 7810 + K);
  const ref = new Float32Array(N);
  for (let r = 0; r < N; r++) {
    let acc = 0;
    for (let i = 0; i < K; i++) acc += w[r * K + i] * x[i];
    ref[r] = acc;
  }
  const y = g.empty(N * 4);
  await g.run(gemvQ6KFastWgsl({ K, N }),
    [g.buf(repackKQuant(src, 0, nBlocks, Q6_K_BLOCK_BYTES)), g.buf(x), y], gemvGrid(N)[0]);
  // relTol dal modulo di sizing e non i 2e-4 storici: a 2e-4 questo kernel
  // passa SOLO se il device contrae a·b+c in FMA — cosa che la spec WGSL
  // permette ma non impone. Il pavimento senza contrazione e' ~2,4e-4
  // (misurato in tests/engine-kquant-f32floor.test.ts): un driver conforme che
  // non fonde darebbe FAIL su un kernel corretto.
  return compare(`gemv-q6k-fast-${K}x${N}`, new Float32Array(await g.read(y, N * 4)), ref,
    KQUANT_FAST_Q6K_REL_TOL, 1e-3);
}

async function testGemvAccumFast(g: Gpu, kind: "q4_0" | "q4_1"): Promise<KResult> {
  const K = G.dFfnExpert, N = G.dModel;
  const blockBytes = kind === "q4_0" ? 18 : Q4_1_BLOCK_BYTES;
  const nBlocks = (K / 32) * N;
  const src = randBytes(nBlocks * blockBytes, 6610 + blockBytes);
  if (kind === "q4_0") fixScales(src, blockBytes);
  else fixScalesAt(src, blockBytes, [1, 3]);
  const w = new Float32Array(nBlocks * 32);
  (kind === "q4_0" ? dequantQ4_0 : dequantQ4_1)(src, 0, nBlocks, w);
  const x = randF32(K, 6612);
  const y0 = randF32(N, 6613);
  const s = 0.6789;
  const ref = new Float32Array(N);
  for (let r = 0; r < N; r++) {
    let acc = 0;
    for (let i = 0; i < K; i++) acc += w[r * K + i] * x[i];
    ref[r] = y0[r] + s * acc;
  }
  const { qs, scales } = (kind === "q4_0" ? repackQ4_0 : repackQ4_1)(src, 0, nBlocks);
  const y = g.buf(y0);
  await g.run(gemvAccumFastWgsl({ kind, K, N }),
    [g.buf(qs), g.buf(scales), g.buf(x), y, g.buf(new Float32Array([s]))], gemvGrid(N / 4)[0]);
  return compare(`gemv-${kind}-accum-fast`, new Float32Array(await g.read(y, N * 4)), ref, 2e-4, 1e-3);
}

// ---- arena expert vs binding a sotto-range (C3a fase 4 strato 1, gate 1) ----
// LO STESSO slab, letto nei due regimi: bindato a sotto-range dai kernel a
// binding fisso (quello che gira in produzione fino a it.14) e dentro un'arena
// da 3 slab, al TERZO offset del SECONDO buffer — cosi' l'aritmetica dello slot
// deve produrre insieme l'arco giusto dello switch e una base diversa da zero.
// Il gate e' l'identita' BIT-A-BIT delle uscite f32: cambia da dove i byte
// arrivano, non che cosa il kernel ci fa sopra. Fallback dichiarato nel design
// (R2): se la contrazione FMA cade in modo diverso fra i due moduli, il gate si
// declassa a maxRel <= 1e-6 — e il note lo DICE, non lo nasconde.
const ARENA_R2_FALLBACK_REL = 1e-6;

// shexp del chunk (fase 5, it.28): le varianti batch dei kernel K-quant contro
// il per-riga — stesso modulo, stesso corpo, cambia solo l'indicizzazione di
// riga ⇒ l'atteso e' BIT-IDENTICO senza fallback.
async function testShexpBatchVsPerRow(g: Gpu): Promise<KResult> {
  const name = "shexp-batch-vs-per-row";
  const M = 3;
  const K = G.dModel, N = G.dFfnExpert; // gate/up: K→N; down: N→K
  const sbGU = (K / 256) * N, sbDn = (N / 256) * K;
  const gSrc = randBytes(sbGU * Q5_K_BLOCK_BYTES, 28_001); fixScalesAt(gSrc, Q5_K_BLOCK_BYTES, [1, 3]);
  const uSrc = randBytes(sbGU * Q5_K_BLOCK_BYTES, 28_002); fixScalesAt(uSrc, Q5_K_BLOCK_BYTES, [1, 3]);
  const dSrc = randBytes(sbDn * Q6_K_BLOCK_BYTES, 28_003); fixScalesAt(dSrc, Q6_K_BLOCK_BYTES, [209]);
  const gB = g.buf(repackKQuant(gSrc, 0, sbGU, Q5_K_BLOCK_BYTES));
  const uB = g.buf(repackKQuant(uSrc, 0, sbGU, Q5_K_BLOCK_BYTES));
  const dB = g.buf(repackKQuant(dSrc, 0, sbDn, Q6_K_BLOCK_BYTES));
  const xs = Array.from({ length: M }, (_, m) => randF32(K, 28_010 + m, 0.5));

  // per-riga (kernel di produzione, it.13)
  const sRef: Float32Array[] = [];
  for (let m = 0; m < M; m++) {
    const gate = g.empty(N * 4);
    await g.run(pairGemvSiluQ5KFastWgsl({ K, N }), [gB, uB, g.buf(xs[m]), gate], gemvGrid(N)[0]);
    const y = g.empty(K * 4);
    await g.run(gemvQ6KFastWgsl({ K: N, N: K }), [dB, gate, y], gemvGrid(K)[0]);
    sRef.push(new Float32Array(await g.read(y, K * 4)));
  }

  // batch su M righe (wid.z = riga)
  const xM = g.empty(M * K * 4);
  for (let m = 0; m < M; m++) g.device.queue.writeBuffer(xM, m * K * 4, xs[m] as unknown as BufferSource);
  const gateM = g.empty(M * N * 4);
  const gGU = gemvGrid(N), gDn = gemvGrid(K);
  await g.run(pairGemvSiluQ5KFastWgsl({ K, N, batch: true }), [gB, uB, xM, gateM], [gGU[0], gGU[1], M]);
  const yM = g.empty(M * K * 4);
  await g.run(gemvQ6KFastWgsl({ K: N, N: K, batch: true }), [dB, gateM, yM], [gDn[0], gDn[1], M]);
  const got = new Float32Array(await g.read(yM, M * K * 4));

  let maxAbs = 0, bitIdentical = true;
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < K; i++) {
      const a = got[m * K + i], b = sRef[m][i];
      if (!Object.is(a, b)) bitIdentical = false;
      maxAbs = Math.max(maxAbs, Math.abs(a - b));
    }
  }
  return {
    kernel: name, pass: bitIdentical, maxAbs, maxRel: maxAbs,
    note: bitIdentical ? `BIT-IDENTICO su ${M} righe (stesso modulo, indicizzazione di riga)` : "divergenza inattesa: stesso modulo, va investigata subito",
  };
}

// ============ PREFILL BATCHED M>1 (C3a fase 5, it.27) ============
// Il gate della fase ridotto ai kernel: la catena batched (gather su unione +
// slot y[m][k] + combine k-order) contro la catena DECODE vera (pairGemvSilu +
// gemvAccum in ordine k + addInPlace), stesse selezioni, stessi pesi sintetici
// a geometria reale. L'atteso e' l'identita' BIT-A-BIT (il piano e' costruito
// per questo — glmprefillplan); fallback dichiarato come R2: se la contrazione
// FMA cade in modo diverso fra i moduli, il gate si declassa a maxRel ≤ 1e-6
// e il note lo dice.
async function testPrefillMoeBatchedChain(g: Gpu): Promise<KResult> {
  const name = "prefill-moe-batched-vs-decode-chain";
  const dev = g.device;
  const layout = SLAB_DOWN_Q4_0;
  const K = G.dModel, N = G.dFfnExpert; // gate/up: K→N; down: N→K
  const M = 4, NE = 6;                  // 4 righe su un pool di 6 expert ⇒ molteplicita' reale
  const seed = 27_100;
  const exBlocks = (K / 32) * N;

  // NE expert sintetici nello stesso formato slab del caso arena
  const one: GPUBuffer[] = [];
  for (let e = 0; e < NE; e++) {
    const gateRaw = randBytes(exBlocks * 18, seed + e * 10); fixScales(gateRaw, 18);
    const upRaw = randBytes(exBlocks * 18, seed + e * 10 + 1); fixScales(upRaw, 18);
    const downRaw = randBytes(exBlocks * 18, seed + e * 10 + 2); fixScales(downRaw, 18);
    const slab = packExpertSlab(gateRaw, upRaw, downRaw, layout);
    const b = g.empty(layout.bytes);
    dev.queue.writeBuffer(b, 0, slab as unknown as BufferSource);
    one.push(b);
  }
  const sub = (e: number, off: number, size: number) => ({ buffer: one[e], offset: off, size });
  const gu4 = (e: number): Array<{ buffer: GPUBuffer; offset: number; size: number }> => [
    sub(e, layout.gateQs, layout.qsBytes), sub(e, layout.gateScales, layout.gateScalesBytes),
    sub(e, layout.upQs, layout.qsBytes), sub(e, layout.upScales, layout.gateScalesBytes),
  ];
  const dn2 = (e: number): Array<{ buffer: GPUBuffer; offset: number; size: number }> => [
    sub(e, layout.downQs, layout.qsBytes), sub(e, layout.downScales, layout.downScalesBytes),
  ];

  // selezioni deterministiche con condivisione fra righe (unione < 4M)
  const wsRaw = randF32(M * 4, seed + 90, 1.8);
  const sels = Array.from({ length: M }, (_, m) => ({
    experts: Int32Array.from([m % NE, (m + 1) % NE, (m + 2) % NE, (m + 3) % NE]),
    weights: Float64Array.from(wsRaw.subarray(m * 4, m * 4 + 4)),
  }));
  const plan = planMoeChunk(sels, 16);
  const xs = Array.from({ length: M }, (_, m) => randF32(K, seed + 50 + m, 0.5));
  const sms = Array.from({ length: M }, (_, m) => randF32(K, seed + 70 + m, 0.5)); // shexp finto

  // --- riferimento: catena DECODE per riga, k crescente, poi addInPlace ---
  const xRef: Float32Array[] = [];
  for (let m = 0; m < M; m++) {
    const xB = g.buf(xs[m]);
    const moeOut = g.buf(sms[m]); // moeOut parte dallo shexp (catena di glmmodel)
    for (let k = 0; k < 4; k++) {
      const e = sels[m].experts[k];
      const gu = g.empty(N * 4);
      await g.run(pairGemvSiluFastWgsl({ K, N }), [...gu4(e), xB, gu], gemvGrid(N / 4)[0]);
      const wB = g.buf(new Float32Array([plan.weights[m * 4 + k]]));
      await g.run(gemvAccumFastWgsl({ kind: "q4_0", K: N, N: K }), [...dn2(e), gu, moeOut, wB], gemvGrid(K / 4)[0]);
    }
    await g.run(addInPlaceWgsl(K), [xB, moeOut], Math.ceil(K / 64));
    xRef.push(new Float32Array(await g.read(xB, K * 4)));
  }

  // --- batched: per expert dell'unione (ordine di unione), poi combine ---
  const xMB = g.empty(M * K * 4);
  const xMInit = new Float32Array(M * K);
  for (let m = 0; m < M; m++) xMInit.set(xs[m], m * K);
  dev.queue.writeBuffer(xMB, 0, xMInit as unknown as BufferSource);
  const sMB = g.empty(M * K * 4);
  const sMInit = new Float32Array(M * K);
  for (let m = 0; m < M; m++) sMInit.set(sms[m], m * K);
  dev.queue.writeBuffer(sMB, 0, sMInit as unknown as BufferSource);
  const hSlots = g.empty(M * 4 * N * 4);
  const ySlots = g.empty(M * 4 * K * 4);
  const wBuf = g.buf(plan.weights);
  const gN = gemvGrid(N / 4), gK = gemvGrid(K / 4);
  for (const b of plan.experts) {
    const gatherB = g.buf(Uint32Array.from(b.rows, (row: number, i: number) => row | (b.slots[i] << 16)));
    await g.run(pairGemvSiluGatherWgsl({ K, N }),
      [...gu4(b.expert), xMB, gatherB, hSlots], [gN[0], gN[1], b.rows.length]);
    await g.run(gemvDownSlotsWgsl({ kind: "q4_0", K: N, N: K }),
      [...dn2(b.expert), hSlots, gatherB, ySlots], [gK[0], gK[1], b.rows.length]);
  }
  await g.run(moeCombineWgsl({ D: K }), [xMB, sMB, ySlots, wBuf], [Math.ceil(K / 64), M, 1]);
  const got = new Float32Array(await g.read(xMB, M * K * 4));

  // --- confronto: bit-identita', con fallback R2 dichiarato ---
  let maxAbs = 0, maxRel = 0, bitIdentical = true;
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < K; i++) {
      const a = got[m * K + i], b = xRef[m][i];
      if (!Object.is(a, b)) bitIdentical = false;
      const d = Math.abs(a - b);
      maxAbs = Math.max(maxAbs, d);
      maxRel = Math.max(maxRel, d / Math.max(1e-12, Math.abs(b)));
    }
  }
  const pass = bitIdentical || maxRel <= ARENA_R2_FALLBACK_REL;
  return {
    kernel: name, pass, maxAbs, maxRel,
    note: bitIdentical
      ? `BIT-IDENTICO: M=${M}, unione ${plan.experts.length} expert su ${M * 4} selezioni (molteplicita' ${(M * 4 / plan.experts.length).toFixed(2)})`
      : `NON bit-identico (fallback R2 ${pass ? "entro" : "OLTRE"} ${ARENA_R2_FALLBACK_REL}): contrazione FMA diversa fra moduli — da investigare prima del wiring`,
  };
}

async function testExpertArenaVsSlotRange(g: Gpu, cls: "q4_0" | "q4_1"): Promise<KResult[]> {
  const name = `expert-arena-vs-slotrange-${cls}`;
  const dev = g.device;
  const layout = cls === "q4_0" ? SLAB_DOWN_Q4_0 : SLAB_DOWN_Q4_1;
  const K = G.dModel, N = G.dFfnExpert;
  const exBlocks = (G.dModel / 32) * G.dFfnExpert;
  const downBB = cls === "q4_0" ? 18 : Q4_1_BLOCK_BYTES;
  const seed = cls === "q4_0" ? 12_100 : 12_200;

  const gateRaw = randBytes(exBlocks * 18, seed); fixScales(gateRaw, 18);
  const upRaw = randBytes(exBlocks * 18, seed + 1); fixScales(upRaw, 18);
  const downRaw = randBytes(exBlocks * downBB, seed + 2);
  if (cls === "q4_0") fixScales(downRaw, 18); else fixScalesAt(downRaw, Q4_1_BLOCK_BYTES, [1, 3]);
  const slab = packExpertSlab(gateRaw, upRaw, downRaw, layout);
  const x = randF32(K, seed + 3, 0.5);
  const y0 = randF32(G.dModel, seed + 4, 0.5);
  const wMix = new Float32Array([0.6789]); // stesso f32 nei due regimi

  const SLABS_PER_BUF = 3, NBUF = 2, SLOT = 5, SEL_IDX = 3, NSEL = 4;
  const MISS_IDX = 1, MISS_SLOT = 0xffffffff;
  const arena: ArenaOpts = {
    nBuf: NBUF, slabWords: layout.bytes / 4, slabsPerBuf: SLABS_PER_BUF,
    qsWords: layout.downQs / 4, scalesWords: layout.downScales / 4,
    gateQsWords: layout.gateQs / 4, gateScWords: layout.gateScales / 4,
    upQsWords: layout.upQs / 4, upScWords: layout.upScales / 4,
  };

  // --- regime a sotto-range: un buffer da uno slab, sei binding {offset,size} ---
  const one = g.empty(layout.bytes);
  dev.queue.writeBuffer(one, 0, slab as unknown as BufferSource);
  const sub = (off: number, size: number) => ({ buffer: one, offset: off, size });
  const xRefB = g.buf(x), xArenaB = g.buf(x), wMixB = g.buf(wMix);
  const guRef = g.empty(N * 4);
  await g.run(pairGemvSiluFastWgsl({ K, N }), [
    sub(layout.gateQs, layout.qsBytes), sub(layout.gateScales, layout.gateScalesBytes),
    sub(layout.upQs, layout.qsBytes), sub(layout.upScales, layout.gateScalesBytes),
    xRefB, guRef,
  ], gemvGrid(N / 4)[0]);
  const yRef = g.buf(y0);
  await g.run(gemvAccumFastWgsl({ kind: cls, K: G.dFfnExpert, N: G.dModel }), [
    sub(layout.downQs, layout.qsBytes), sub(layout.downScales, layout.downScalesBytes),
    guRef, yRef, wMixB,
  ], gemvGrid(G.dModel / 4)[0]);

  // --- regime arena: 2 buffer da 3 slab, lo slab nello slot 5 (buf 1, base 2) ---
  const bufs = [g.empty(SLABS_PER_BUF * layout.bytes), g.empty(SLABS_PER_BUF * layout.bytes)];
  dev.queue.writeBuffer(bufs[1], 2 * layout.bytes, slab as unknown as BufferSource);
  const selBytes = new ArrayBuffer(NSEL * 16);
  const selU32 = new Uint32Array(selBytes), selF32 = new Float32Array(selBytes);
  selU32[SEL_IDX * 4] = 42;          // id expert (non letto dai kernel)
  selU32[SEL_IDX * 4 + 1] = SLOT;
  selF32[SEL_IDX * 4 + 2] = wMix[0];
  selU32[SEL_IDX * 4 + 3] = 0;
  // entry di MISS: slot sentinella, peso NON nullo (se il kernel scrivesse lo
  // stesso, un peso a zero lo nasconderebbe) e flags con il bit 0 alzato
  selU32[MISS_IDX * 4] = 99;
  selU32[MISS_IDX * 4 + 1] = MISS_SLOT;
  selF32[MISS_IDX * 4 + 2] = 0.5;
  selU32[MISS_IDX * 4 + 3] = 1;
  const selBuf = g.buf(new Uint32Array(selBytes));
  const uniData = new Uint32Array(NSEL * 64);
  for (let i = 0; i < NSEL; i++) uniData[i * 64] = i; // MoeIdx.selIdx
  const uni = dev.createBuffer({ size: uniData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(uni, 0, uniData as unknown as BufferSource);

  const runArena = async (code: string, xB: GPUBuffer, outB: GPUBuffer, wg: number, selIdx = SEL_IDX): Promise<void> => {
    const module = dev.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error");
    if (errs.length) throw new Error(`WGSL arena: ${errs[0].message} @${errs[0].lineNum}`);
    const st = (type: GPUBufferBindingType, binding: number): GPUBindGroupLayoutEntry =>
      ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const bgl = dev.createBindGroupLayout({
      entries: [
        ...bufs.map((_, j) => st("read-only-storage", j)),
        st("read-only-storage", NBUF), st("storage", NBUF + 1), st("read-only-storage", NBUF + 2),
        {
          binding: NBUF + 3, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
        },
      ],
    });
    const pipeline = dev.createComputePipeline({
      layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module, entryPoint: "main" },
    });
    const bg = dev.createBindGroup({
      layout: bgl,
      entries: [
        ...bufs.map((b, j) => ({ binding: j, resource: { buffer: b } })),
        { binding: NBUF, resource: { buffer: xB } },
        { binding: NBUF + 1, resource: { buffer: outB } },
        { binding: NBUF + 2, resource: { buffer: selBuf } },
        { binding: NBUF + 3, resource: { buffer: uni, offset: 0, size: 16 } },
      ],
    });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg, [selIdx * 256]);
    pass.dispatchWorkgroups(wg);
    pass.end();
    dev.queue.submit([enc.finish()]);
  };

  const guArena = g.empty(N * 4);
  await runArena(pairGemvSiluFastWgsl({ K, N, arena }), xArenaB, guArena, gemvGrid(N / 4)[0]);
  const yArena = g.buf(y0);
  await runArena(gemvAccumFastWgsl({ kind: cls, K: G.dFfnExpert, N: G.dModel, arena }),
    guArena, yArena, gemvGrid(G.dModel / 4)[0]);

  // --- MISS: slot sentinella ⇒ contributo NULLO, non un indirizzo a caso ---
  // E' la rete su cui si appoggeranno gli slice B e C (flags/selMiss): il
  // degrado a residenza parziale dev'essere DEFINITO. I due buffer di uscita
  // sono pre-riempiti con valori noti e devono restare tali, bit per bit.
  const guMiss0 = randF32(N, seed + 5, 0.5);
  const yMiss0 = randF32(G.dModel, seed + 6, 0.5);
  const guMiss = g.buf(guMiss0), yMiss = g.buf(yMiss0);
  await runArena(pairGemvSiluFastWgsl({ K, N, arena }), xArenaB, guMiss, gemvGrid(N / 4)[0], MISS_IDX);
  await runArena(gemvAccumFastWgsl({ kind: cls, K: G.dFfnExpert, N: G.dModel, arena }),
    guMiss, yMiss, gemvGrid(G.dModel / 4)[0], MISS_IDX);

  // --- confronto BIT-A-BIT (e, se cade, di quanto) ---
  const bits = async (a: GPUBuffer, b: GPUBuffer, n: number) => {
    const A = new Uint32Array(await g.read(a, n * 4)), B = new Uint32Array(await g.read(b, n * 4));
    const fA = new Float32Array(A.buffer), fB = new Float32Array(B.buffer);
    let diff = 0, maxRel = 0, maxAbs = 0;
    for (let i = 0; i < n; i++) {
      if (A[i] !== B[i]) diff++;
      const d = Math.abs(fA[i] - fB[i]);
      maxAbs = Math.max(maxAbs, d);
      maxRel = Math.max(maxRel, d / Math.max(Math.abs(fA[i]), 1e-6));
    }
    return { diff, maxRel, maxAbs };
  };
  const gu = await bits(guRef, guArena, N);
  const yy = await bits(yRef, yArena, G.dModel);
  // il MISS si confronta con i valori PRE-ESISTENTI, non con l'altro regime
  const guM = new Uint32Array(await g.read(guMiss, N * 4));
  const yM = new Uint32Array(await g.read(yMiss, G.dModel * 4));
  const guM0 = new Uint32Array(guMiss0.buffer.slice(0)), yM0 = new Uint32Array(yMiss0.buffer.slice(0));
  let missDiff = 0;
  for (let i = 0; i < N; i++) if (guM[i] !== guM0[i]) missDiff++;
  for (let i = 0; i < G.dModel; i++) if (yM[i] !== yM0[i]) missDiff++;
  for (const b of [one, xRefB, xArenaB, wMixB, guRef, yRef, guArena, yArena, guMiss, yMiss, selBuf, uni, ...bufs]) {
    b.destroy();
  }

  const diff = gu.diff + yy.diff;
  const maxRel = Math.max(gu.maxRel, yy.maxRel);
  const maxAbs = Math.max(gu.maxAbs, yy.maxAbs);
  const note = diff === 0
    ? `BIT-A-BIT identico su ${N} + ${G.dModel} f32 (slot ${SLOT} = buffer 1 offset 2, selIdx ${SEL_IDX})`
    : `NON bit-identico: ${diff} f32 diversi (gate/up ${gu.diff}, down ${yy.diff}), `
      + `maxRel=${maxRel.toExponential(2)} — fallback R2 del design `
      + `(soglia ${ARENA_R2_FALLBACK_REL.toExponential(0)})`;
  return [
    { kernel: name, pass: diff === 0 || maxRel <= ARENA_R2_FALLBACK_REL, maxAbs, maxRel, note },
    {
      kernel: `expert-arena-miss-${cls}`, pass: missDiff === 0, maxAbs: missDiff, maxRel: 0,
      note: missDiff === 0
        ? `slot 0xffffffff (w=0.5, flags=1): gate/up e down non toccano l'uscita, bit per bit`
        : `slot di MISS: ${missDiff} f32 SCRITTI — il degrado non e' definito`,
    },
  ];
}

// Router top-k su GPU vs `routerSelect` (CPU, f64) — C3a fase 4 strato 1.
// Il kernel calcola in f32 e la CPU in f64: NON e' una replica bit-identica, e
// questo caso serve a misurare DOVE diverge invece di assumere che non lo faccia.
//
// Cosa e' gate e cosa e' misura:
// - gate: l'INSIEME dei 4 expert deve coincidere su ogni estrazione in cui la
//   separazione fra il 4o e il 5o punteggio e' >= SEP_GATE. Sotto quella soglia
//   f32 non ha abbastanza bit per decidere, e un flip li' e' fisica, non un bug.
// - gate: i pesi di mixing entro tolleranza relativa.
// - misura riportata in `note`: la separazione piu' piccola a cui l'insieme ha
//   comunque retto, e quante estrazioni sono cadute sotto la soglia.
// L'ORDINE dentro i 4 non e' gate: id e peso viaggiano appaiati (slots[k] con
// wExp[k]), quindi una permutazione non cambia la matematica del layer. Se
// diverge lo si riporta lo stesso, perche' cambierebbe il confronto posizionale
// con la routing conformance.
const ROUTER_SEP_GATE = 1e-5;

async function testRouterTopK(g: Gpu, draws: number): Promise<KResult> {
  const code = routerTopKWgsl({
    nExpert: G.nExpert, nUsed: G.nExpertUsed,
    weightsScale: G.weightsScale, clampMin: WEIGHTS_SUM_CLAMP_MIN,
  });
  const nU = G.nExpertUsed;
  let maxRelW = 0, maxAbsW = 0;
  let setFlips = 0, orderFlips = 0, belowGate = 0;
  let minSepHeld = Infinity, maxSepFlipped = 0;

  for (let d = 0; d < draws; d++) {
    const logits = randF32(G.nExpert, 9001 + d * 7, 4);
    const bias = randF32(G.nExpert, 4201 + d * 13, 0.1);
    const ref = routerSelect(logits, bias);

    // separazione fra ultimo selezionato e primo escluso, in f64: e' la
    // grandezza che decide se f32 puo' sbagliare insieme
    const sel = Array.from(logits, (v, i) => 1 / (1 + Math.exp(-v)) + bias[i]);
    const ranked = sel.slice().sort((a, b) => b - a);
    const sep = ranked[nU - 1] - ranked[nU];

    const lb = g.buf(logits), bb = g.buf(bias);
    const idsB = g.empty(nU * 4), wtsB = g.empty(nU * 4);
    await g.run(code, [lb, bb, idsB, wtsB], 1);
    const ids = new Uint32Array(await g.read(idsB, nU * 4));
    const wts = new Float32Array(await g.read(wtsB, nU * 4));
    for (const b of [lb, bb, idsB, wtsB]) b.destroy();

    const sameSet = new Set(ids).size === nU
      && Array.from(ids).every((e) => Array.from(ref.experts).includes(e));
    const sameOrder = Array.from(ids).every((e, k) => e === ref.experts[k]);
    if (sep < ROUTER_SEP_GATE) belowGate++;
    if (!sameSet) {
      setFlips += sep >= ROUTER_SEP_GATE ? 1 : 0;
      maxSepFlipped = Math.max(maxSepFlipped, sep);
    } else {
      minSepHeld = Math.min(minSepHeld, sep);
    }
    if (!sameOrder) orderFlips++;

    // i pesi si confrontano solo quando l'insieme coincide: su insiemi diversi
    // sarebbe un confronto fra quantita' diverse, non un errore numerico
    if (sameSet) {
      for (let k = 0; k < nU; k++) {
        const want = ref.weights[Array.from(ref.experts).indexOf(ids[k])];
        const d1 = Math.abs(wts[k] - want);
        maxAbsW = Math.max(maxAbsW, d1);
        maxRelW = Math.max(maxRelW, d1 / Math.max(Math.abs(want), 1e-6));
      }
    }
  }

  const note = `draws=${draws} setFlips(sep>=${ROUTER_SEP_GATE})=${setFlips} `
    + `orderFlips=${orderFlips} sottoSoglia=${belowGate} `
    + `minSepRetto=${minSepHeld === Infinity ? "-" : minSepHeld.toExponential(2)}`
    + (maxSepFlipped > 0 ? ` maxSepFlippato=${maxSepFlipped.toExponential(2)}` : "");
  return {
    kernel: "router-top4-gpu-vs-cpu",
    pass: setFlips === 0 && maxRelW <= 1e-5,
    maxAbs: maxAbsW, maxRel: maxRelW, note,
  };
}

// Near-tie COSTRUITO. Il caso random sopra non esercita mai il gate: con 64
// expert e punteggi sparsi la separazione fra 4o e 5o non scende sotto ~3e-5 da
// sola. Qui la separazione si impone — si sposta il bias del primo escluso
// finche' sel[5o] = sel[4o] - eps — e si scende con eps finche' f32 cede.
// Serve a rispondere con un numero alla domanda "quanto e' vicino il baratro",
// invece di dichiarare una soglia che nessuna estrazione ha mai visitato.
async function testRouterNearTie(g: Gpu, epsList: number[], bases: number): Promise<KResult> {
  const code = routerTopKWgsl({
    nExpert: G.nExpert, nUsed: G.nExpertUsed,
    weightsScale: G.weightsScale, clampMin: WEIGHTS_SUM_CLAMP_MIN,
  });
  const nU = G.nExpertUsed;
  const heldAt: number[] = [], flippedAt: number[] = [];

  for (const eps of epsList) {
    let held = true;
    for (let b = 0; b < bases; b++) {
      const logits = randF32(G.nExpert, 7717 + b * 31, 4);
      const bias = randF32(G.nExpert, 3313 + b * 17, 0.1);
      const probs = Array.from(logits, (v) => 1 / (1 + Math.exp(-v)));
      const sel = probs.map((p, i) => p + bias[i]);
      // rank per punteggio: il 4o selezionato e il 5o (primo escluso)
      const order = sel.map((v, i) => [v, i] as const).sort((x, y) => y[0] - x[0]);
      const last = order[nU - 1][1], first = order[nU][1];
      // il bias entra additivamente in sel ⇒ la separazione si impone esatta
      bias[first] = sel[last] - eps - probs[first];

      const ref = routerSelect(logits, bias);
      const lb = g.buf(logits), bb = g.buf(bias);
      const idsB = g.empty(nU * 4), wtsB = g.empty(nU * 4);
      await g.run(code, [lb, bb, idsB, wtsB], 1);
      const ids = new Uint32Array(await g.read(idsB, nU * 4));
      for (const x of [lb, bb, idsB, wtsB]) x.destroy();

      const sameSet = new Set(ids).size === nU
        && Array.from(ids).every((e) => Array.from(ref.experts).includes(e));
      if (!sameSet) { held = false; break; }
    }
    (held ? heldAt : flippedAt).push(eps);
  }

  const minHeld = heldAt.length ? Math.min(...heldAt) : Infinity;
  const maxFlipped = flippedAt.length ? Math.max(...flippedAt) : 0;
  return {
    kernel: "router-top4-near-tie",
    pass: minHeld <= ROUTER_SEP_GATE,   // il gate dichiarato deve reggere DAVVERO
    maxAbs: 0, maxRel: 0,
    note: `bases=${bases} tenuto fino a eps=${minHeld === Infinity ? "-" : minHeld.toExponential(0)}`
      + ` primo flip a eps=${maxFlipped ? maxFlipped.toExponential(0) : "nessuno"}`
      + ` (gate dichiarato ${ROUTER_SEP_GATE.toExponential(0)})`,
  };
}

// Resolve + slotTable (C3a fase 4 slice B) allo stato puro: il router GPU
// scrive Sel leggendo la tabella che ExpertCache mantiene, e la tabella e'
// quella VERA — gli slot li dichiara `ensure`, non il test. E' l'unico punto in
// cui si puo' verificare l'indirizzo risolto CONTRO un valore noto: dentro il
// modello gli slot li sceglie la LRU e nessuno di fuori li vede.
// Tre cose, in ordine: (a) chi e' residente si risolve al suo slot; (b) chi non
// lo e' esce MISS con flags bit 0; (c) dopo l'eviction la tabella torna MISS —
// e' il rischio R5 del design, l'unico che si vede solo qui.
async function testRouterResolveSlotTable(g: Gpu): Promise<KResult> {
  const nU = G.nExpertUsed, LAYER = 1; // blk.1-4: classe q4_1 (7 slot nel ktest)
  const code = routerTopKWgsl({
    nExpert: G.nExpert, nUsed: nU,
    weightsScale: G.weightsScale, clampMin: WEIGHTS_SUM_CLAMP_MIN,
    resolve: { nExpert: G.nExpert, nUsed: nU },
  });
  const cache = new ExpertCache(g.device, {
    budgetBytes: 0, slotsOverride: KTEST_SLOTS,
    maxBindingBytes: KTEST_ARENA_WINDOW, maxBufferBytes: KTEST_ARENA_WINDOW,
    arena: true, slotTable: true,
  });
  const zero = new Uint8Array(SLAB_DOWN_Q4_1.bytes); // il contenuto non conta: conta lo slot
  const reader = {
    raw: (): ExpertRawBytes => { throw new Error("resolve-slottable: percorso raw non previsto"); },
    slab: () => zero,
  };

  const logits = randF32(G.nExpert, 8811, 4);
  const bias = randF32(G.nExpert, 8822, 0.1);
  const ref = routerSelect(logits, bias);
  const sel4 = Array.from(ref.experts);
  // residenti: il 1o e il 3o dei quattro selezionati, piu' un expert ESTRANEO
  // alla selezione (se il resolve sbagliasse base della tabella, il suo slot
  // finirebbe addosso a qualcun altro e il confronto se ne accorgerebbe)
  const resident = new Map<number, number>();
  for (const e of [sel4[0], sel4[2], (sel4[0] + 7) % G.nExpert]) {
    if (resident.has(e)) continue;
    resident.set(e, cache.ensure(LAYER, e, reader).slot.idx);
  }
  cache.flushSlotTable();

  // Sel a 8 entry, scritte a partire da selIdx=4: le prime 4 sono veleno e
  // devono restare tali (il resolve scrive DOVE dice l'uniform, non a offset 0)
  const NSEL = 8, SEL_BASE = 4, POISON = 0xdeadbeef;
  const poison = new Uint32Array(NSEL * 4).fill(POISON);
  const selB = g.buf(poison);
  const uniOf = (selIdx: number): GPUBuffer => {
    const b = g.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    g.device.queue.writeBuffer(b, 0, new Uint32Array([selIdx, LAYER * G.nExpert, 0, 0]));
    return b;
  };
  const idsB = g.empty(nU * 4), wtsB = g.empty(nU * 4);
  const table = cache.slotTableBuffer();
  const runResolve = async (): Promise<{ id: number; slot: number; w: number; flags: number }[]> => {
    await g.run(code, [g.buf(logits), g.buf(bias), idsB, wtsB, selB, table], 1, uniOf(SEL_BASE));
    const raw = await g.read(selB, NSEL * 16);
    const u = new Uint32Array(raw), f = new Float32Array(raw);
    return Array.from({ length: nU }, (_, k) => {
      const o = (SEL_BASE + k) * 4;
      return { id: u[o], slot: u[o + 1], w: f[o + 2], flags: u[o + 3] };
    });
  };

  const problems: string[] = [];
  let maxRelW = 0, maxAbsW = 0;
  const got = await runResolve();
  // (0) le entry fuori dal layer non sono state toccate
  {
    const u = new Uint32Array(await g.read(selB, SEL_BASE * 16));
    if (!u.every((v) => v === POISON)) problems.push("il resolve ha scritto fuori dalle entry del layer");
  }
  for (let k = 0; k < nU; k++) {
    if (got[k].id !== sel4[k]) problems.push(`k${k}: id ${got[k].id} != CPU ${sel4[k]}`);
    const want = resident.has(sel4[k]) ? resident.get(sel4[k])! : 0xffffffff;
    if (got[k].slot !== want) problems.push(`k${k}: slot ${got[k].slot} != ${want} (expert ${sel4[k]})`);
    const wantFlag = resident.has(sel4[k]) ? 0 : 1;
    if (got[k].flags !== wantFlag) problems.push(`k${k}: flags ${got[k].flags} != ${wantFlag}`);
    const d = Math.abs(got[k].w - ref.weights[k]);
    maxAbsW = Math.max(maxAbsW, d);
    maxRelW = Math.max(maxRelW, d / Math.max(Math.abs(ref.weights[k]), 1e-9));
  }
  // (c) eviction: 7 expert nuovi riempiono tutti gli slot della classe ⇒ i
  // residenti di prima sono usciti, e la tabella deve dirlo
  for (let i = 0; i < KTEST_SLOTS.q4_1; i++) cache.ensure(LAYER, 32 + i, reader);
  cache.flushSlotTable();
  const after = await runResolve();
  for (let k = 0; k < nU; k++) {
    if (after[k].slot !== 0xffffffff || after[k].flags !== 1) {
      problems.push(`post-eviction k${k}: slot ${after[k].slot} flags ${after[k].flags} (atteso MISS)`);
    }
  }
  cache.destroy();
  for (const b of [selB, idsB, wtsB]) b.destroy();
  return {
    kernel: "router-resolve-slottable",
    pass: problems.length === 0 && maxRelW <= 1e-5,
    maxAbs: maxAbsW, maxRel: maxRelW,
    note: `residenti ${[...resident.entries()].map(([e, s]) => `e${e}→s${s}`).join(",")}, `
      + `top4 CPU {${sel4}}, wMaxRel=${maxRelW.toExponential(2)}, `
      + `MISS post-eviction ${after.filter((s) => s.flags === 1).length}/${nU}`
      + (problems.length ? `; ${problems.join("; ")}` : ""),
  };
}

async function main(): Promise<void> {
  const g = new Gpu();
  const adapterDesc = await g.init();
  post({ type: "adapter", desc: adapterDesc });
  const results: KResult[] = [];
  const D = S.dModel, HD = S.headDim, KV = S.nKvHead * HD;

  try {
    // GEMV: taglie reali del modello (q/o proj, k/v proj, ffn, lm_head ridotto)
    results.push(await testGemv(g, "q4_0", D, D, true));       // attn_q
    results.push(await testGemv(g, "q4_0", D, KV, true));      // attn_k/v
    results.push(await testGemv(g, "q4_0", D, S.dFfn, false)); // ffn gate/up
    results.push(await testGemv(g, "q4_0", S.dFfn, D, false)); // ffn down
    results.push(await testGemv(g, "q8_0", D, 2048, false));   // lm_head (N ridotto: stessa matematica)

    // GEMV formati GLM (C2 fase 4), taglie reali del modello-tesi
    results.push(await testGemvC2(g, "q4_1", G.dFfnExpert, G.dModel));  // expert down blk.1-4
    results.push(await testGemvC2(g, "q4_1", G.dFfnDense, 256));        // dense down (righe ridotte)
    results.push(await testGemvC2(g, "q5_K", G.dModel, G.dFfnExpert));  // gate/up shexp
    results.push(await testGemvC2(g, "q6_K", G.dFfnExpert, G.dModel));  // down shexp
    results.push(await testGemvC2(g, "q6_K", G.dModel, 1024));          // output head (N ridotto)

    // --- MoE (C2 fase 5 slice 1): router, accumulo pesato, blocco completo ---
    results.push(await testGemvF32(g));
    results.push(await testGemvAccum(g, "q4_0"));
    results.push(await testGemvAccum(g, "q4_1"));
    results.push(await testMoeFfnBlock(g, "q4_0"));
    results.push(await testMoeFfnBlock(g, "q4_1"));

    // --- prefill batched M>1 (C3a fase 5): catena su unione vs catena decode ---
    results.push(await testPrefillMoeBatchedChain(g));
    results.push(await testShexpBatchVsPerRow(g));

    // --- router top-4 su GPU (C3a fase 4 strato 1): fedelta' f32 vs CPU f64 ---
    results.push(await testRouterTopK(g, 64));
    results.push(await testRouterNearTie(g, [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], 8));
    // slice B: la coda di resolve scrive Sel leggendo la slotTable vera
    results.push(await testRouterResolveSlotTable(g));

    // --- famiglia fusa portata su GLM (C3a fase 4b) ---
    results.push(await testPairGemvSiluFast(g));
    results.push(await testGemvAccumFast(g, "q4_0"));
    results.push(await testGemvAccumFast(g, "q4_1"));

    // --- arena expert (C3a fase 4 strato 1, slice A): stesso slab, due regimi,
    //     piu' la semantica del MISS ---
    results.push(...await testExpertArenaVsSlotRange(g, "q4_0"));
    results.push(...await testExpertArenaVsSlotRange(g, "q4_1"));

    // --- famiglia fast sui K-quant (C3a fase 4b it.13): shexp + head ---
    results.push(await testPairGemvSiluQ5KFast(g));
    results.push(await testGemvQ6KFast(g, G.dFfnExpert, G.dModel)); // down shexp, K=1536 (6 superblocchi)
    results.push(await testGemvQ6KFast(g, G.dModel, 1024));         // output head, K=2048 (N ridotto)

    // --- kernel MLA absorbed (C2 fase 4 slice 2), dims reali GLM ---
    const HL = G.qkNope + G.ropeDims; // 256: head len di q
    const W576 = G.kvLora + G.ropeDims;

    { // rope NORM su q (20 head, offset 192) e su kv_cmpr_pe (1 vettore 576, offset 512)
      const pos = 129;
      const q = randF32(G.nHead * HL, 61);
      const refQ = q.slice();
      ropeNormRef(refQ, G.nHead, HL, G.qkNope, G.ropeDims, G.ropeFreqBase, pos);
      const qBuf = g.buf(q);
      await g.run(ropeMlaNormWgsl({ nVec: G.nHead, stride: HL, offset: G.qkNope, ropeDims: G.ropeDims, freqBase: G.ropeFreqBase }),
        [qBuf], Math.ceil((G.nHead * G.ropeDims / 2) / 64), g.uniform(pos, 0));
      results.push(compare("rope-mla-norm-q", new Float32Array(await g.read(qBuf, q.byteLength)), refQ, 5e-4, 1e-4));

      const kv = randF32(W576, 62);
      const refKv = kv.slice();
      ropeNormRef(refKv, 1, W576, G.kvLora, G.ropeDims, G.ropeFreqBase, pos);
      const kvBuf = g.buf(kv);
      await g.run(ropeMlaNormWgsl({ nVec: 1, stride: W576, offset: G.kvLora, ropeDims: G.ropeDims, freqBase: G.ropeFreqBase }),
        [kvBuf], Math.ceil((G.ropeDims / 2) / 64), g.uniform(pos, 0));
      results.push(compare("rope-mla-norm-kpe", new Float32Array(await g.read(kvBuf, kv.byteLength)), refKv, 5e-4, 1e-4));
    }

    { // gemvQ8Heads: assorbimento wk_b [192,512,20] e uscita wv_b [512,256,20]
      for (const [name, K, rows, xStride, xOffset] of [
        ["absorb-kb", G.qkNope, G.kvLora, HL, 0],
        ["vout-vb", G.kvLora, G.headLenMla, G.kvLora, 0],
      ] as const) {
        const nBlocks = (K / 32) * rows * G.nHead;
        const src = randBytes(nBlocks * 34, 555 + K);
        fixScales(src, 34);
        const { qs, scales } = repackQ8_0(src, 0, nBlocks);
        const w = new Float32Array(nBlocks * 32);
        dequantQ8_0(src, 0, nBlocks, w);
        const x = randF32(G.nHead * xStride + K, 91 + K);
        const ref = new Float32Array(rows * G.nHead);
        for (let r = 0; r < rows * G.nHead; r++) {
          const head = Math.floor(r / rows);
          let acc = 0;
          for (let i = 0; i < K; i++) acc += w[r * K + i] * x[head * xStride + xOffset + i];
          ref[r] = acc;
        }
        const y = g.empty(rows * G.nHead * 4);
        await g.run(gemvQ8HeadsWgsl({ K, rowsPerHead: rows, nHead: G.nHead, xStride, xOffset }),
          [g.buf(qs), g.buf(scales), g.buf(x), y], rows * G.nHead);
        results.push(compare(`gemv-q8-heads-${name}`, new Float32Array(await g.read(y, rows * G.nHead * 4)), ref, 2e-4, 1e-3));
      }
    }

    { // attention decode MLA su cache 576 (nPast=40), scale = 1/sqrt(256)
      const nPast = 40, ctxMax = 512;
      const scale = 1 / Math.sqrt(G.headLenMla);
      const q = randF32(G.nHead * W576, 71, 0.5);
      const cache = randF32(ctxMax * W576, 72, 0.5);
      const ref = mlaAttnRefF64(q, cache, nPast, scale);
      const out = g.empty(G.nHead * G.kvLora * 4);
      await g.run(mlaAttnDecodeWgsl({ nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax, scale }),
        [g.buf(q), g.buf(cache), out, ], G.nHead, g.uniform(nPast, nPast));
      results.push(compare("mla-attn-decode", new Float32Array(await g.read(out, G.nHead * G.kvLora * 4)), ref, 5e-4, 1e-4));
    }

    { // attention MLA SPLIT sul contesto (fase 4c): part+reduce vs LO STESSO
      // riferimento f64 del monolitico, sui casi di bordo del chunking
      const ctxMax = 512;
      const scale = 1 / Math.sqrt(G.headLenMla);
      const q = randF32(G.nHead * W576, 71, 0.5);
      const cache = randF32(ctxMax * W576, 72, 0.5);
      const partCode = mlaAttnSplitPartWgsl({
        nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax, scale, chunk: MLA_CHUNK_P,
      });
      const reduceCode = mlaAttnSplitReduceWgsl({ nHead: G.nHead, kvLora: G.kvLora, ctxMax, chunk: MLA_CHUNK_P });
      const qBuf = g.buf(q), cBuf = g.buf(cache);
      const runSplit = async (nPast: number): Promise<Float32Array> => {
        // partials pre-sporcato: le partizioni oltre il contesto NON devono
        // essere lette dal pass 2 (se lo fossero, questi NaN lo direbbero)
        const dirty = new Float32Array(mlaPartialsLen(G.nHead, G.kvLora, ctxMax)).fill(NaN);
        const partials = g.buf(dirty);
        const out = g.empty(G.nHead * G.kvLora * 4);
        const u = g.uniform(nPast, nPast);
        await g.run(partCode, [qBuf, cBuf, partials], mlaSMax(ctxMax), u);
        await g.run(reduceCode, [partials, out], G.nHead, u);
        const got = new Float32Array(await g.read(out, G.nHead * G.kvLora * 4));
        partials.destroy(); // 1,3 MB per invocazione: 6 invocazioni = 8 MB di VRAM
        out.destroy();
        return got;
      };
      // nPast < CHUNK; bordo esatto (CHUNK-1 e CHUNK); a cavallo di piu' chunk;
      // contesto lungo (32 partizioni attive)
      for (const nPast of [7, MLA_CHUNK_P - 1, MLA_CHUNK_P, 40, 200]) {
        const ref = mlaAttnRefF64(q, cache, nPast, scale);
        results.push(compare(`mla-attn-split-p${nPast}`, await runSplit(nPast), ref, 5e-4, 1e-4));
      }
      // identita' split vs monolitico sugli stessi input: entrambi f32, NON
      // bit-identici (ordine delle somme diverso) ⇒ tolleranza stretta
      const nPast = 200;
      const mono = g.empty(G.nHead * G.kvLora * 4);
      await g.run(mlaAttnDecodeWgsl({ nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax, scale }),
        [qBuf, cBuf, mono], G.nHead, g.uniform(nPast, nPast));
      results.push(compare("mla-attn-split-vs-mono", await runSplit(nPast),
        new Float32Array(await g.read(mono, G.nHead * G.kvLora * 4)), 1e-4, 1e-5));
    }

    { // MLA split BATCH (fase 5, it.29): M righe con passato CRESCENTE
      // (causalita' intra-chunk: n = base+m+1) vs il per-riga — stesso corpo,
      // cambia solo l'indicizzazione ⇒ atteso BIT-IDENTICO, incluso il caso
      // in cui righe diverse attivano un numero DIVERSO di partizioni.
      const ctxMax = 512, M = 3, basePast = MLA_CHUNK_P - 2; // riga 0 sotto il bordo, riga 2 sopra
      const scale = 1 / Math.sqrt(G.headLenMla);
      const cache = randF32(ctxMax * W576, 91, 0.5);
      const qRows = Array.from({ length: M }, (_, m) => randF32(G.nHead * W576, 92 + m, 0.5));
      const partCode = mlaAttnSplitPartWgsl({ nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax, scale, chunk: MLA_CHUNK_P });
      const reduceCode = mlaAttnSplitReduceWgsl({ nHead: G.nHead, kvLora: G.kvLora, ctxMax, chunk: MLA_CHUNK_P });
      const cBuf = g.buf(cache);
      const perRow: Float32Array[] = [];
      for (let m = 0; m < M; m++) {
        const partials = g.buf(new Float32Array(mlaPartialsLen(G.nHead, G.kvLora, ctxMax)).fill(NaN));
        const out = g.empty(G.nHead * G.kvLora * 4);
        const u = g.uniform(basePast + m, basePast + m);
        await g.run(partCode, [g.buf(qRows[m]), cBuf, partials], mlaSMax(ctxMax), u);
        await g.run(reduceCode, [partials, out], G.nHead, u);
        perRow.push(new Float32Array(await g.read(out, G.nHead * G.kvLora * 4)));
        partials.destroy(); out.destroy();
      }
      const qM = g.empty(M * G.nHead * W576 * 4);
      for (let m = 0; m < M; m++) g.device.queue.writeBuffer(qM, m * G.nHead * W576 * 4, qRows[m] as unknown as BufferSource);
      const rowPast = g.buf(Uint32Array.from({ length: M }, (_, m) => basePast + m));
      const partialsM = g.buf(new Float32Array(M * mlaPartialsLen(G.nHead, G.kvLora, ctxMax)).fill(NaN));
      const outM = g.empty(M * G.nHead * G.kvLora * 4);
      await g.run(mlaAttnSplitPartWgsl({ nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax, scale, chunk: MLA_CHUNK_P, batch: true }),
        [qM, cBuf, partialsM, rowPast], [mlaSMax(ctxMax), 1, M]);
      await g.run(mlaAttnSplitReduceWgsl({ nHead: G.nHead, kvLora: G.kvLora, ctxMax, chunk: MLA_CHUNK_P, batch: true }),
        [partialsM, outM, rowPast], [G.nHead, 1, M]);
      const gotM = new Float32Array(await g.read(outM, M * G.nHead * G.kvLora * 4));
      let maxAbs = 0, bitIdentical = true;
      for (let m = 0; m < M; m++) {
        for (let i = 0; i < G.nHead * G.kvLora; i++) {
          const a = gotM[m * G.nHead * G.kvLora + i], b = perRow[m][i];
          if (!Object.is(a, b)) bitIdentical = false;
          maxAbs = Math.max(maxAbs, Math.abs(a - b));
        }
      }
      partialsM.destroy(); outM.destroy();
      results.push({
        kernel: "mla-split-batch-vs-per-row", pass: bitIdentical, maxAbs, maxRel: maxAbs,
        note: bitIdentical
          ? `BIT-IDENTICO su ${M} righe a passato crescente (${basePast}..${basePast + M - 1}, bordo di partizione attraversato)`
          : "divergenza inattesa: stesso corpo, indicizzazione da investigare",
      });
    }

    { // strided copy (assemblaggio q576 = [q_ckv | q_rope] per head): esatto
      const src = randF32(G.nHead * HL, 81);
      const dst = randF32(G.nHead * W576, 82); // pre-riempito: la copy non deve toccare il resto
      const ref = dst.slice();
      for (let h = 0; h < G.nHead; h++) {
        for (let j = 0; j < G.ropeDims; j++) {
          ref[h * W576 + G.kvLora + j] = src[h * HL + G.qkNope + j];
        }
      }
      const dstBuf = g.buf(dst);
      await g.run(
        stridedCopyWgsl({ nVec: G.nHead, len: G.ropeDims, srcStride: HL, srcOffset: G.qkNope, dstStride: W576, dstOffset: G.kvLora }),
        [g.buf(src), dstBuf], Math.ceil((G.nHead * G.ropeDims) / 64));
      results.push(compare("strided-copy-qrope", new Float32Array(await g.read(dstBuf, dst.byteLength)), ref, 0, 0));
    }

    { // rmsnorm
      const x = randF32(D, 5), w = randF32(D, 6);
      let ss = 0;
      for (let i = 0; i < D; i++) ss += x[i] * x[i];
      const rms = 1 / Math.sqrt(ss / D + S.rmsEps);
      const ref = new Float32Array(D);
      for (let i = 0; i < D; i++) ref[i] = x[i] * rms * w[i];
      const out = g.empty(D * 4);
      await g.run(rmsnormWgsl(D, S.rmsEps), [g.buf(x), g.buf(w), out], 1);
      results.push(compare("rmsnorm", new Float32Array(await g.read(out, D * 4)), ref, 1e-4, 1e-5));
    }

    { // rope neox (q a 14 head) — in-place
      const pos = 37;
      const v = randF32(S.nHead * HD, 11);
      const ref = v.slice();
      const half = HD / 2;
      for (let h = 0; h < S.nHead; h++) {
        for (let j = 0; j < half; j++) {
          const theta = pos / S.ropeFreqBase ** (j / half);
          const c = Math.cos(theta), s = Math.sin(theta);
          const a = ref[h * HD + j], b = ref[h * HD + j + half];
          ref[h * HD + j] = a * c - b * s;
          ref[h * HD + j + half] = a * s + b * c;
        }
      }
      const buf = g.buf(v);
      await g.run(ropeNeoxWgsl(S.nHead, HD, S.ropeFreqBase), [buf], Math.ceil((S.nHead * half) / 64), g.uniform(pos, 0));
      results.push(compare("rope-neox", new Float32Array(await g.read(buf, v.byteLength)), ref, 5e-4, 1e-4));
    }

    { // kv append + attention decode (nPast = 40)
      const nPast = 40;
      const ctxMax = 512;
      const kCacheData = randF32(ctxMax * KV, 21, 0.5);
      const vCacheData = randF32(ctxMax * KV, 22, 0.5);
      const q = randF32(S.nHead * HD, 23);
      const kCur = randF32(KV, 24), vCur = randF32(KV, 25);
      // append ref
      kCacheData.set(kCur, nPast * KV);
      vCacheData.set(vCur, nPast * KV);
      // attn ref
      const ref = new Float32Array(S.nHead * HD);
      const groups = S.nHead / S.nKvHead;
      for (let h = 0; h < S.nHead; h++) {
        const kvHead = Math.floor(h / groups);
        const scores = new Float32Array(nPast + 1);
        for (let p = 0; p <= nPast; p++) {
          let acc = 0;
          for (let i = 0; i < HD; i++) acc += q[h * HD + i] * kCacheData[p * KV + kvHead * HD + i];
          scores[p] = acc / Math.sqrt(HD);
        }
        let m = -Infinity;
        for (const s of scores) m = Math.max(m, s);
        let sum = 0;
        for (let p = 0; p <= nPast; p++) { scores[p] = Math.exp(scores[p] - m); sum += scores[p]; }
        for (let p = 0; p <= nPast; p++) scores[p] /= sum;
        for (let i = 0; i < HD; i++) {
          let acc = 0;
          for (let p = 0; p <= nPast; p++) acc += scores[p] * vCacheData[p * KV + kvHead * HD + i];
          ref[h * HD + i] = acc;
        }
      }
      // gpu: cache SENZA la riga corrente, poi kvAppend, poi attn
      const kBuf = g.buf(kCacheData.slice().fill(0, nPast * KV, (nPast + 1) * KV));
      const vBuf = g.buf(vCacheData.slice().fill(0, nPast * KV, (nPast + 1) * KV));
      const u = g.uniform(nPast, nPast);
      await g.run(kvAppendWgsl(KV), [g.buf(kCur), kBuf], Math.ceil(KV / 64), u);
      await g.run(kvAppendWgsl(KV), [g.buf(vCur), vBuf], Math.ceil(KV / 64), u);
      const out = g.empty(S.nHead * HD * 4);
      await g.run(attnDecodeWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: HD, ctxMax }),
        [g.buf(q), kBuf, vBuf, out], S.nHead, u);
      results.push(compare("kv-append+attn-decode", new Float32Array(await g.read(out, S.nHead * HD * 4)), ref, 5e-4, 1e-4));
    }

    { // silu-mul + add
      const gate = randF32(S.dFfn, 31), up = randF32(S.dFfn, 32);
      const refSilu = new Float32Array(S.dFfn);
      for (let i = 0; i < S.dFfn; i++) refSilu[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * up[i];
      const gBuf = g.buf(gate);
      await g.run(siluMulWgsl(S.dFfn), [gBuf, g.buf(up)], Math.ceil(S.dFfn / 64));
      results.push(compare("silu-mul", new Float32Array(await g.read(gBuf, S.dFfn * 4)), refSilu, 1e-4, 1e-5));

      const x = randF32(D, 41), y = randF32(D, 42);
      const refAdd = new Float32Array(D);
      for (let i = 0; i < D; i++) refAdd[i] = x[i] + y[i];
      const xBuf = g.buf(x);
      await g.run(addInPlaceWgsl(D), [xBuf, g.buf(y)], Math.ceil(D / 64));
      results.push(compare("add-inplace", new Float32Array(await g.read(xBuf, D * 4)), refAdd, 0, 0));
    }

    { // argmax due stadi su vocab reale
      const N = S.vocab;
      const x = randF32(N, 51, 10);
      x[137713] = 99; // massimo noto
      let refIdx = 0;
      for (let i = 1; i < N; i++) if (x[i] > x[refIdx]) refIdx = i;
      const nPartials = Math.ceil(N / ARGMAX_CHUNK);
      const pmax = g.empty(nPartials * 4), pidx = g.empty(nPartials * 4), out = g.empty(16);
      await g.run(argmaxStage1Wgsl(N), [g.buf(x), pmax, pidx], nPartials);
      await g.run(argmaxStage2Wgsl(nPartials), [pmax, pidx, out], 1);
      const got = new Uint32Array(await g.read(out, 4))[0];
      results.push({ kernel: "argmax-2stage", pass: got === refIdx, maxAbs: Math.abs(got - refIdx), maxRel: 0, note: `got=${got} ref=${refIdx}` });
    }

    // residenza minima (fase 5 slice 2): OPFS + cache VRAM su hardware vero
    results.push(await testResidencyOpfs(g));

    // forward multi-layer (fase 5 slice 3): path di produzione vs ref f64.
    // Due esecuzioni: quella di produzione ("cpu") e quella con il router GPU in
    // ombra ("shadow", slice B), piu' il confronto fra le due — e' quest'ultimo
    // a dire che l'ombra non ha spostato niente.
    const cpuTrace: ModelTrace = { hidden: [], logits: [], experts: [], weights: [] };
    const mCpu = await testGlmModel2Layer(g, "cpu", { record: cpuTrace });
    const mShadow = await testGlmModel2Layer(g, "shadow");
    results.push(mCpu, mShadow, testShadowInvariance(mCpu, mShadow, 1));
    // slice C: l'interruttore. Il modo gpu gira a residenza TOTALE (64 slot per
    // 64 expert) e si confronta con le uscite VERE del run cpu, non con le sue
    // metriche; poi il caso negativo, che e' l'altra meta' del gate.
    results.push(await testGlmModel2Layer(g, "gpu", { against: cpuTrace }));
    results.push(await testGlmGpuPartialResidency(g));

    // conformance layer-level con pesi reali (fase 4 slice 3) — il test lungo in coda
    results.push(await testGlmLayer0Real(g));
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e), results });
    return;
  }
  post({ type: "done", results });
}

void main();
