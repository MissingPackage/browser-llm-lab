// Kernel-test del motore: ogni kernel WGSL contro un riferimento JS calcolato QUI,
// su dati pseudo-casuali seeded (LCG: riproducibile). Non serve il modello: valida
// indicizzazione, riduzioni e dequant dei kernel prima di qualunque orchestratore.
//
// Tolleranze: i kernel accumulano f32 in ordine diverso dal riferimento (f64 in JS)
// ⇒ confronto a tolleranza relativa+assoluta, per kernel. Le parti intere (nibble,
// int8, indici argmax) devono essere esatte.
import {
  gemvQuantWgsl, gemvF32Wgsl, gemvQ5KWgsl, gemvQ6KWgsl, rmsnormWgsl, ropeNeoxWgsl, kvAppendWgsl,
  attnDecodeWgsl, attnDecodeRefWgsl, attnDecodeCombineWgsl,
  siluMulWgsl, addInPlaceWgsl, argmaxStage1Wgsl, argmaxStage2Wgsl, type KArenaOpts,
  ARGMAX_CHUNK, ropeMlaNormWgsl, gemvQ8HeadsWgsl, mlaAttnDecodeWgsl, stridedCopyWgsl,
  routerTopKWgsl, pairGemvSiluFastWgsl, gemvAccumFastWgsl, gemvGrid, type ArenaOpts,
  pairGemvSiluGatherWgsl, gemvDownSlotsWgsl, moeCombineWgsl,
  mlaAttnSplitPartWgsl, mlaAttnSplitReduceWgsl,
  pairGemvSiluQ5KFastWgsl, gemvQ6KFastWgsl,
} from "../kernels/wgsl";
import { MLA_CHUNK_P, mlaSMax, mlaPartialsLen } from "../mlasplit";
import { q35AttnSplitPlan, q35AttnPartialsFloats } from "../q35attnsplit";
import { KQUANT_FAST_Q5K_PAIR_REL_TOL, KQUANT_FAST_Q6K_REL_TOL } from "../kquantfast";
import { createGlmLayer0 } from "../glmforward";
import {
  GlmDenseLayerRefF64, GlmMoeLayerRefF64, glmMoeFfnRefF64, type GlmMoeExpertWeights,
} from "../cpuref";
import { createGlmModel, type GlmWeightSource } from "../glmmodel";
import {
  routerSelect, ROUTER_GLM47, ROUTER_QWEN35MOE, packExpertSlab, SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, WEIGHTS_SUM_CLAMP_MIN,
} from "../moe";
import { ExpertOpfsStore } from "../expertstore";
import { ExpertCache, arenaNeeds, expertKey, modelExpertPark, type ExpertRawBytes } from "../residency";
import {
  repackQ4_0, repackQ8_0, repackQ4_1, repackKQuant,
  dequantQ4_0, dequantQ8_0, dequantQ4_1, dequantQ4_K, dequantQ5_K, dequantQ6_K,
  Q4_1_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
} from "../quant";
import { QWEN25_05B as S, GLM47_FLASH as G } from "../shape";
import { createEngineDevice } from "../gpudevice";
import { planMoeChunk } from "../moeprefillplan";
import { deltaNetConvWgsl, deltaNetGatesWgsl, deltaNetCoreWgsl } from "../kernels/deltanet";
import { axpyWgsl, gemvQ4KWgsl, sigmoidMulWgsl } from "../kernels/wgsl";
import { q35MoeFfnRefF64, type Q35MoeLayerWeights } from "../q35cpurefmodel";
import { createQ35GpuModel, q35TensorBytes } from "../q35gpumodel";
import { validateQwen35 } from "../q35shape";
import { parseGguf, type GgufTensorInfo } from "../gguf";
import { deltaNetStepCore, softplusGgml, Q35DeltaNetRef } from "../q35cpuref";
import { SAMPLE_DIMS, SAMPLE_T, sampleWeights, sampleInputs } from "../q35sample";

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
        extraBindings: [
          { bytes: 10_485_760, consumer: "ktest: blk.0 ffn_gate/up q4_0 qs (pesi reali)" },
          // q1 fase 4 slice 3: head Q6_K del 4B (vocab 248320 × d 2560, repack
          // ~527 MB) bindata intera dal gemv della testa nel full-model test
          { bytes: 530_000_000, consumer: "ktest: q35 4B head Q6_K (full-model)" },
        ],
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

// --- DeltaNet q35 (q1 fase 3): ogni kernel vs cpuref-f64 (q35cpuref) ---

async function testDeltaNetConv(g: Gpu, C: number): Promise<KResult> {
  const convK = 4, K1 = convK - 1;
  const state0 = randF32(K1 * C, 811, 0.8);
  const x = randF32(C, 812, 0.8);
  const w = randF32(C * convK, 813, 0.8);
  const refOut = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let i = 0; i < convK; i++) s += w[c * convK + i] * (i < K1 ? state0[i * C + c] : x[c]);
    refOut[c] = s / (1 + Math.exp(-s));
  }
  const refState = new Float32Array(K1 * C);
  for (let c = 0; c < C; c++) {
    for (let i = 0; i + 1 < K1; i++) refState[i * C + c] = state0[(i + 1) * C + c];
    refState[(K1 - 1) * C + c] = x[c];
  }
  const stBuf = g.buf(state0), out = g.empty(C * 4);
  await g.run(deltaNetConvWgsl(C, convK), [stBuf, g.buf(x), g.buf(w), out], Math.ceil(C / 64));
  const r1 = compare(`deltanet-conv-${C}`, new Float32Array(await g.read(out, C * 4)), refOut, 5e-4, 1e-5);
  const r2 = compare(`deltanet-conv-${C}-state`, new Float32Array(await g.read(stBuf, K1 * C * 4)), refState, 0, 0);
  return { kernel: r1.kernel, pass: r1.pass && r2.pass, maxAbs: Math.max(r1.maxAbs, r2.maxAbs), maxRel: Math.max(r1.maxRel, r2.maxRel), note: `out+shift stato (shift esatto: ${r2.pass})` };
}

async function testDeltaNetGates(g: Gpu): Promise<KResult> {
  const nV = 32;
  const betaRaw = randF32(nV, 821, 3), alphaRaw = randF32(nV, 822, 12), dt = randF32(nV, 823, 2);
  const a = Float32Array.from(randF32(nV, 824, 1), (v) => -Math.exp(v));
  const refB = new Float32Array(nV), refG = new Float32Array(nV);
  for (let h = 0; h < nV; h++) {
    refB[h] = 1 / (1 + Math.exp(-betaRaw[h]));
    refG[h] = a[h] * softplusGgml(alphaRaw[h] + dt[h]);
  }
  const bB = g.empty(nV * 4), bG = g.empty(nV * 4);
  await g.run(deltaNetGatesWgsl(nV), [g.buf(betaRaw), g.buf(alphaRaw), g.buf(a), g.buf(dt), bB, bG], 1);
  const r1 = compare("deltanet-gates-beta", new Float32Array(await g.read(bB, nV * 4)), refB, 1e-5, 1e-6);
  const r2 = compare("deltanet-gates-g", new Float32Array(await g.read(bG, nV * 4)), refG, 1e-4, 1e-5);
  return { kernel: "deltanet-gates", pass: r1.pass && r2.pass, maxAbs: Math.max(r1.maxAbs, r2.maxAbs), maxRel: Math.max(r1.maxRel, r2.maxRel) };
}

async function testDeltaNetCore(g: Gpu, hd: number, nK: number, nV: number): Promise<KResult> {
  const eps = 1e-6, keyDim = nK * hd, inner = nV * hd;
  const convOut = randF32(2 * keyDim + inner, 831, 0.9);
  const S0 = randF32(nV * hd * hd, 832, 0.3);
  const beta = Float32Array.from(randF32(nV, 833, 3), (v) => 1 / (1 + Math.exp(-v)));
  const gg = Float32Array.from(randF32(nV, 834, 1), (v) => -Math.exp(v)); // decay < 1
  const z = randF32(inner, 835, 2);
  const normW = Float32Array.from(randF32(hd, 836, 0.2), (v) => 1 + v);
  // ref f64: l2norm(q,k) → core → gated norm (stessa catena del cpuref.step)
  const refOut = new Float32Array(inner);
  const refS = new Float32Array(nV * hd * hd);
  for (let h = 0; h < nV; h++) {
    const kh = (h % nK) * hd;
    const l2 = (off: number): Float64Array => {
      let ss = 0;
      for (let i = 0; i < hd; i++) ss += convOut[off + i] ** 2;
      const sc = 1 / Math.max(Math.sqrt(ss), eps);
      return Float64Array.from({ length: hd }, (_, i) => convOut[off + i] * sc);
    };
    const q = l2(kh).map((v) => v / Math.sqrt(hd)) as Float64Array;
    const k = l2(keyDim + kh);
    const v = Float64Array.from({ length: hd }, (_, i) => convOut[2 * keyDim + h * hd + i]);
    const Sh = Float64Array.from(S0.subarray(h * hd * hd, (h + 1) * hd * hd));
    const o = deltaNetStepCore(Sh, q, k, v, gg[h], beta[h], hd);
    let ss = 0;
    for (let i = 0; i < hd; i++) ss += o[i] * o[i];
    const inv = 1 / Math.sqrt(ss / hd + eps);
    for (let i = 0; i < hd; i++) {
      const zj = z[h * hd + i];
      refOut[h * hd + i] = o[i] * inv * normW[i] * (zj / (1 + Math.exp(-zj)));
    }
    refS.set(Float32Array.from(Sh), h * hd * hd);
  }
  const sBuf = g.buf(S0), out = g.empty(inner * 4);
  await g.run(deltaNetCoreWgsl({ hd, nK, nV, eps }), [g.buf(convOut), sBuf, g.buf(beta), g.buf(gg), g.buf(z), g.buf(normW), out], nV);
  const r1 = compare(`deltanet-core-hd${hd}`, new Float32Array(await g.read(out, inner * 4)), refOut, 5e-4, 1e-4);
  const r2 = compare(`deltanet-core-hd${hd}-S`, new Float32Array(await g.read(sBuf, nV * hd * hd * 4)), refS, 5e-4, 1e-4);
  return { kernel: r1.kernel, pass: r1.pass && r2.pass, maxAbs: Math.max(r1.maxAbs, r2.maxAbs), maxRel: Math.max(r1.maxRel, r2.maxRel), note: "out + stato post-update" };
}

async function testDeltaNetChain(g: Gpu): Promise<KResult> {
  // catena INTERA sul campione pinnato (proiezioni gemv-f32 + conv + gates +
  // core + wout), stato persistente su GPU per T=12 token; riferimento =
  // cpuref-f64 sugli STESSI input arrotondati a f32 (isola l'errore kernel).
  const D = SAMPLE_DIMS;
  const W = sampleWeights();
  const keyDim = D.nK * D.hd, inner = D.nV * D.hd, qkvDim = 2 * keyDim + inner;
  const ref = new Q35DeltaNetRef(D, W);
  const wqkv = g.buf(W.wqkv), wgate = g.buf(W.wgate), wbeta = g.buf(W.wbeta), walpha = g.buf(W.walpha);
  const wout = g.buf(W.wout), aBuf = g.buf(W.a), dtBuf = g.buf(W.dtBias), nrm = g.buf(W.ssmNorm);
  const convW = g.buf(W.conv);
  const convSt = g.buf(new Float32Array((D.convK - 1) * qkvDim));
  const sBuf = g.buf(new Float32Array(D.nV * D.hd * D.hd));
  const qkv = g.empty(qkvDim * 4), z = g.empty(inner * 4), bRaw = g.empty(D.nV * 4), aRaw = g.empty(D.nV * 4);
  const bSig = g.empty(D.nV * 4), gVal = g.empty(D.nV * 4), convOut = g.empty(qkvDim * 4);
  const gated = g.empty(inner * 4), y = g.empty(D.d * 4);
  let maxAbs = 0, maxRel = 0, pass = true;
  for (const x64 of sampleInputs()) {
    const x32 = Float32Array.from(x64);
    const refY = Float32Array.from(ref.step(Float64Array.from(x32)));
    const xB = g.buf(x32);
    await g.run(gemvF32Wgsl({ K: D.d, N: qkvDim }), [wqkv, xB, qkv], qkvDim);
    await g.run(gemvF32Wgsl({ K: D.d, N: inner }), [wgate, xB, z], inner);
    await g.run(gemvF32Wgsl({ K: D.d, N: D.nV }), [wbeta, xB, bRaw], D.nV);
    await g.run(gemvF32Wgsl({ K: D.d, N: D.nV }), [walpha, xB, aRaw], D.nV);
    await g.run(deltaNetGatesWgsl(D.nV), [bRaw, aRaw, aBuf, dtBuf, bSig, gVal], 1);
    await g.run(deltaNetConvWgsl(qkvDim, D.convK), [convSt, qkv, convW, convOut], Math.ceil(qkvDim / 64));
    await g.run(deltaNetCoreWgsl({ hd: D.hd, nK: D.nK, nV: D.nV, eps: D.eps }), [convOut, sBuf, bSig, gVal, z, nrm, gated], D.nV);
    await g.run(gemvF32Wgsl({ K: inner, N: D.d }), [wout, gated, y], D.d);
    const got = new Float32Array(await g.read(y, D.d * 4));
    const r = compare("chain-step", got, refY, 2e-3, 5e-4);
    maxAbs = Math.max(maxAbs, r.maxAbs);
    maxRel = Math.max(maxRel, r.maxRel);
    pass = pass && r.pass;
  }
  return { kernel: "deltanet-chain-T12", pass, maxAbs, maxRel, note: `campione pinnato ${SAMPLE_T} token, stato persistente su GPU` };
}

// --- q35 fase 4 slice 2: micro-kernel nuovi + assembly layer con pesi REALI ---

async function testRopePartial(g: Gpu): Promise<KResult> {
  // rope NEOX parziale (64 su 256, qwen35): rotazione solo sui primi 64
  // canali di ogni head, il resto DEVE restare invariato bit-a-bit.
  const nHead = 16, hd = 256, dims = 64, pos = 7;
  const v = randF32(nHead * hd, 911);
  const ref = v.slice();
  const half = dims / 2;
  for (let h = 0; h < nHead; h++) {
    for (let j = 0; j < half; j++) {
      const theta = pos / 1e7 ** (j / half);
      const c = Math.cos(theta), s = Math.sin(theta);
      const a = ref[h * hd + j], b = ref[h * hd + j + half];
      ref[h * hd + j] = a * c - b * s;
      ref[h * hd + j + half] = a * s + b * c;
    }
  }
  const buf = g.buf(v);
  await g.run(ropeNeoxWgsl(nHead, hd, 1e7, dims), [buf], Math.ceil((nHead * half) / 64), g.uniform(pos, 0));
  const got = new Float32Array(await g.read(buf, v.byteLength));
  // canali non ruotati: identita' esatta
  let untouched = true;
  for (let h = 0; h < nHead; h++) {
    for (let i = dims; i < hd; i++) if (got[h * hd + i] !== v[h * hd + i]) untouched = false;
  }
  const r = compare("rope-neox-partial-64of256", got, ref, 5e-4, 1e-4);
  return { ...r, pass: r.pass && untouched, note: `canali ≥64 invariati: ${untouched}` };
}

async function testSigmoidMul(g: Gpu): Promise<KResult> {
  const D = 4096;
  const x = randF32(D, 921), gg = randF32(D, 922, 3);
  const ref = new Float32Array(D);
  for (let i = 0; i < D; i++) ref[i] = x[i] / (1 + Math.exp(-gg[i]));
  const buf = g.buf(x);
  await g.run(sigmoidMulWgsl(D), [buf, g.buf(gg)], Math.ceil(D / 64));
  return compare("sigmoid-mul", new Float32Array(await g.read(buf, D * 4)), ref, 1e-5, 1e-6);
}

/** Assembly GPU dei layer attention qwen35 con pesi REALI del 4B vs cpuref (fixture). */
async function testQ35AttnLayersReal(g: Gpu): Promise<KResult[]> {
  const metaRes = await fetch("/models/q35-attn/meta.json");
  if (!metaRes.ok) {
    return [{ kernel: "q35-attn-real", pass: false, maxAbs: NaN, maxRel: NaN, note: "fixture assente: npx vite-node scripts/q35-attn-fixture-gen.mjs" }];
  }
  interface TensorEntry { suffix: string; type: number; dims: number[]; offset: number; bytes: number }
  const meta = (await metaRes.json()) as {
    dims: { d: number; nHead: number; nKvHead: number; headDim: number; ropeDims: number; freqBase: number; rmsEps: number; nK: number; nV: number; hd: number; convK: number };
    T: number;
    layers: { l: number; kind: string; tensors: TensorEntry[] }[];
    inputs: { offset: number; bytes: number };
    expected: { l: number; offset: number; bytes: number }[];
  };
  const bin = new Uint8Array(await (await fetch("/models/q35-attn/fixture.bin")).arrayBuffer());
  const D = meta.dims;
  const f32c = (off: number, bytes: number): Float32Array => new Float32Array(bin.slice(off, off + bytes).buffer);
  const results: KResult[] = [];

  for (const L of meta.layers) {
    const tn = new Map(L.tensors.map((t) => [t.suffix, t]));
    const raw = (s: string): TensorEntry => {
      const t = tn.get(s);
      if (!t) throw new Error(`fixture: ${s} assente`);
      return t;
    };
    const q40 = (s: string): { qs: GPUBuffer; scales: GPUBuffer; n: number; k: number } => {
      const t = raw(s);
      const nBlocks = (t.dims[0] / 32) * t.dims[1];
      const { qs, scales } = repackQ4_0(bin, t.offset, nBlocks);
      return { qs: g.buf(qs), scales: g.buf(scales), n: t.dims[1], k: t.dims[0] };
    };
    const exp = meta.expected.find((e) => e.l === L.l);
    if (!exp) throw new Error("fixture: expected assente");
    const expected = f32c(exp.offset, exp.bytes);
    const attnNorm = g.buf(f32c(raw("attn_norm.weight").offset, raw("attn_norm.weight").bytes));
    let l2n = 0, l2d = 0, maxAbs = 0, maxRel = 0;

    if (L.kind === "linear") {
      const qkvW = q40("attn_qkv.weight");
      const zW = q40("attn_gate.weight");
      const bT = raw("ssm_beta.weight"), aT = raw("ssm_alpha.weight");
      const bRk = repackQ8_0(bin, bT.offset, (bT.dims[0] / 32) * bT.dims[1]);
      const aRk = repackQ8_0(bin, aT.offset, (aT.dims[0] / 32) * aT.dims[1]);
      const outT = raw("ssm_out.weight");
      const outBlocks = repackKQuant(bin, outT.offset, (outT.dims[0] / 256) * outT.dims[1], Q5_K_BLOCK_BYTES);
      const convW = g.buf(f32c(raw("ssm_conv1d.weight").offset, raw("ssm_conv1d.weight").bytes));
      const aBuf = g.buf(f32c(raw("ssm_a").offset, raw("ssm_a").bytes));
      const dtBuf = g.buf(f32c(raw("ssm_dt.bias").offset, raw("ssm_dt.bias").bytes));
      const nrmBuf = g.buf(f32c(raw("ssm_norm.weight").offset, raw("ssm_norm.weight").bytes));
      const qkvDim = (2 * D.nK + D.nV) * D.hd;
      const inner = D.nV * D.hd;
      const convSt = g.buf(new Float32Array((D.convK - 1) * qkvDim));
      const S = g.buf(new Float32Array(D.nV * D.hd * D.hd));
      const xn = g.empty(D.d * 4), qkv = g.empty(qkvDim * 4), z = g.empty(inner * 4);
      const bRaw = g.empty(D.nV * 4), aRaw = g.empty(D.nV * 4), bSig = g.empty(D.nV * 4), gVal = g.empty(D.nV * 4);
      const convOut = g.empty(qkvDim * 4), gated = g.empty(inner * 4), y = g.empty(D.d * 4);
      for (let t = 0; t < meta.T; t++) {
        const x = g.buf(f32c(meta.inputs.offset + t * D.d * 4, D.d * 4));
        await g.run(rmsnormWgsl(D.d, D.rmsEps), [x, attnNorm, xn], 1);
        await g.run(gemvQuantWgsl({ kind: "q4_0", K: qkvW.k, N: qkvW.n, hasBias: false }), [qkvW.qs, qkvW.scales, xn, qkv], qkvW.n);
        await g.run(gemvQuantWgsl({ kind: "q4_0", K: zW.k, N: zW.n, hasBias: false }), [zW.qs, zW.scales, xn, z], zW.n);
        await g.run(gemvQuantWgsl({ kind: "q8_0", K: bT.dims[0], N: bT.dims[1], hasBias: false }), [g.buf(bRk.qs), g.buf(bRk.scales), xn, bRaw], bT.dims[1]);
        await g.run(gemvQuantWgsl({ kind: "q8_0", K: aT.dims[0], N: aT.dims[1], hasBias: false }), [g.buf(aRk.qs), g.buf(aRk.scales), xn, aRaw], aT.dims[1]);
        await g.run(deltaNetGatesWgsl(D.nV), [bRaw, aRaw, aBuf, dtBuf, bSig, gVal], 1);
        await g.run(deltaNetConvWgsl(qkvDim, D.convK), [convSt, qkv, convW, convOut], Math.ceil(qkvDim / 64));
        await g.run(deltaNetCoreWgsl({ hd: D.hd, nK: D.nK, nV: D.nV, eps: D.rmsEps }), [convOut, S, bSig, gVal, z, nrmBuf, gated], D.nV);
        await g.run(gemvQ5KWgsl({ K: inner, N: outT.dims[1] }), [g.buf(outBlocks), gated, y], outT.dims[1]);
        const got = new Float32Array(await g.read(y, D.d * 4));
        for (let i = 0; i < D.d; i++) {
          const e = expected[t * D.d + i], dif = Math.abs(got[i] - e);
          l2n += dif * dif; l2d += e * e;
          if (dif > maxAbs) maxAbs = dif;
          if (Math.abs(e) > 1e-4) maxRel = Math.max(maxRel, dif / Math.abs(e));
        }
      }
    } else {
      const wq = q40("attn_q.weight"), wk = q40("attn_k.weight"), wv = q40("attn_v.weight"), wo = q40("attn_output.weight");
      const qNormW = g.buf(f32c(raw("attn_q_norm.weight").offset, raw("attn_q_norm.weight").bytes));
      const kNormW = g.buf(f32c(raw("attn_k_norm.weight").offset, raw("attn_k_norm.weight").bytes));
      const hd = D.headDim, qDim = D.nHead * hd, kvDim = D.nKvHead * hd;
      const kCache = g.buf(new Float32Array(meta.T * kvDim));
      const vCache = g.buf(new Float32Array(meta.T * kvDim));
      const xn = g.empty(D.d * 4), qFull = g.empty(2 * qDim * 4), kBuf = g.empty(kvDim * 4), vBuf = g.empty(kvDim * 4);
      const qB = g.empty(qDim * 4), gateB = g.empty(qDim * 4), qN = g.empty(qDim * 4), kN = g.empty(kvDim * 4);
      const attnO = g.empty(qDim * 4), y = g.empty(D.d * 4);
      // parziali dello split del decode: la griglia e' fissa in ctxMax, quindi
      // i buffer si allocano UNA volta e si riusano fra le posizioni t (i
      // dispatch sono sequenziali, ogni pass 1 riscrive per intero cio' che il
      // pass 2 ha gia' letto).
      const aSplits = q35AttnSplitPlan(meta.T).splits;
      const aPart = q35AttnPartialsFloats({ nHead: D.nHead, headDim: hd, ctxMax: meta.T });
      const partOut = g.empty(aPart.out * 4), partMS = g.empty(aPart.ms * 4);
      for (let t = 0; t < meta.T; t++) {
        const u = g.uniform(t, t);
        const x = g.buf(f32c(meta.inputs.offset + t * D.d * 4, D.d * 4));
        await g.run(rmsnormWgsl(D.d, D.rmsEps), [x, attnNorm, xn], 1);
        await g.run(gemvQuantWgsl({ kind: "q4_0", K: wq.k, N: wq.n, hasBias: false }), [wq.qs, wq.scales, xn, qFull], wq.n);
        await g.run(gemvQuantWgsl({ kind: "q4_0", K: wk.k, N: wk.n, hasBias: false }), [wk.qs, wk.scales, xn, kBuf], wk.n);
        await g.run(gemvQuantWgsl({ kind: "q4_0", K: wv.k, N: wv.n, hasBias: false }), [wv.qs, wv.scales, xn, vBuf], wv.n);
        await g.run(stridedCopyWgsl({ nVec: D.nHead, len: hd, srcStride: 2 * hd, srcOffset: 0, dstStride: hd, dstOffset: 0 }), [qFull, qB], Math.ceil(qDim / 64));
        await g.run(stridedCopyWgsl({ nVec: D.nHead, len: hd, srcStride: 2 * hd, srcOffset: hd, dstStride: hd, dstOffset: 0 }), [qFull, gateB], Math.ceil(qDim / 64));
        await g.run(rmsnormWgsl(hd, D.rmsEps, true), [qB, qNormW, qN], D.nHead);
        await g.run(rmsnormWgsl(hd, D.rmsEps, true), [kBuf, kNormW, kN], D.nKvHead);
        await g.run(ropeNeoxWgsl(D.nHead, hd, D.freqBase, D.ropeDims), [qN], Math.ceil((D.nHead * D.ropeDims / 2) / 64), u);
        await g.run(ropeNeoxWgsl(D.nKvHead, hd, D.freqBase, D.ropeDims), [kN], Math.ceil((D.nKvHead * D.ropeDims / 2) / 64), u);
        await g.run(kvAppendWgsl(kvDim), [kN, kCache], Math.ceil(kvDim / 64), u);
        await g.run(kvAppendWgsl(kvDim), [vBuf, vCache], Math.ceil(kvDim / 64), u);
        await g.run(attnDecodeWgsl({ nHead: D.nHead, nKvHead: D.nKvHead, headDim: hd, ctxMax: meta.T }),
          [qN, kCache, vCache, partOut, partMS], [D.nHead, aSplits, 1], u);
        await g.run(attnDecodeCombineWgsl({ nHead: D.nHead, headDim: hd, ctxMax: meta.T }), [partOut, partMS, attnO], D.nHead);
        await g.run(sigmoidMulWgsl(qDim), [attnO, gateB], Math.ceil(qDim / 64));
        await g.run(gemvQuantWgsl({ kind: "q4_0", K: wo.k, N: wo.n, hasBias: false }), [wo.qs, wo.scales, attnO, y], wo.n);
        const got = new Float32Array(await g.read(y, D.d * 4));
        for (let i = 0; i < D.d; i++) {
          const e = expected[t * D.d + i], dif = Math.abs(got[i] - e);
          l2n += dif * dif; l2d += e * e;
          if (dif > maxAbs) maxAbs = dif;
          if (Math.abs(e) > 1e-4) maxRel = Math.max(maxRel, dif / Math.abs(e));
        }
      }
    }
    const l2 = Math.sqrt(l2n / Math.max(l2d, 1e-12));
    results.push({
      kernel: `q35-attn-${L.kind}-real-blk${L.l}`,
      pass: l2 <= 1e-3 && maxAbs <= 5e-2,
      maxAbs, maxRel,
      note: `pesi reali 4B, T=${meta.T}, L2rel=${l2.toExponential(2)}`,
      metrics: { l2rel: l2 },
    });
  }
  return results;
}

/**
 * TESTA MTP (NextN) su GPU contro il cpuref f64 (fase-D fase 7, it.52).
 *
 * Il blocco e' un layer full-attention normale preceduto da `eh_proj`, quindi
 * NON introduce un solo kernel nuovo: introduce solo CABLAGGIO — due norme su
 * ingressi di scala diversa, una concatenazione [2d] e una GEMV q8_0 al posto
 * della q4_0 di tutti gli altri pesi. Ed e' esattamente il cablaggio che qui si
 * copre: se enorm e hnorm finissero scambiate, o le due meta' della
 * concatenazione invertite, il modello pieno non se ne accorgerebbe (la testa
 * non e' sul suo percorso) e lo spec-dec lo mostrerebbe solo come accept-rate
 * basso — cioe' come un risultato negativo sull'MTP invece che come un bug
 * nostro. La lm_head resta FUORI: e' condivisa e gia' coperta dal full-model.
 */
async function testQ35MtpHeadReal(g: Gpu): Promise<KResult> {
  const metaRes = await fetch("/models/q35-mtp/meta.json");
  if (!metaRes.ok) {
    return { kernel: "q35-mtp-head-real", pass: false, maxAbs: NaN, maxRel: NaN, note: "fixture assente: npx vite-node scripts/q35-mtp-fixture-gen.mjs" };
  }
  interface TensorEntry { suffix: string; type: number; dims: number[]; offset: number; bytes: number }
  const meta = (await metaRes.json()) as {
    dims: { d: number; nHead: number; nKvHead: number; headDim: number; ropeDims: number; freqBase: number; rmsEps: number; dFfn: number };
    T: number; blk: number; tensors: TensorEntry[];
    emb: { offset: number; bytes: number };
    hidden: { offset: number; bytes: number };
    expected: { offset: number; bytes: number };
  };
  const bin = new Uint8Array(await (await fetch("/models/q35-mtp/fixture.bin")).arrayBuffer());
  const D = meta.dims;
  const d = D.d, hd = D.headDim, qDim = D.nHead * hd, kvDim = D.nKvHead * hd, dFfn = D.dFfn;
  const f32c = (off: number, bytes: number): Float32Array => new Float32Array(bin.slice(off, off + bytes).buffer);
  const tn = new Map(meta.tensors.map((t) => [t.suffix, t]));
  const raw = (s: string): TensorEntry => {
    const t = tn.get(s);
    if (!t) throw new Error(`fixture mtp: ${s} assente`);
    return t;
  };
  const f32w = (s: string): GPUBuffer => g.buf(f32c(raw(s).offset, raw(s).bytes));
  const q40 = (s: string): { qs: GPUBuffer; scales: GPUBuffer; n: number; k: number } => {
    const t = raw(s);
    const { qs, scales } = repackQ4_0(bin, t.offset, (t.dims[0] / 32) * t.dims[1]);
    return { qs: g.buf(qs), scales: g.buf(scales), n: t.dims[1], k: t.dims[0] };
  };
  const q80 = (s: string): { qs: GPUBuffer; scales: GPUBuffer; n: number; k: number } => {
    const t = raw(s);
    const { qs, scales } = repackQ8_0(bin, t.offset, (t.dims[0] / 32) * t.dims[1]);
    return { qs: g.buf(qs), scales: g.buf(scales), n: t.dims[1], k: t.dims[0] };
  };

  const enorm = f32w("nextn.enorm.weight"), hnorm = f32w("nextn.hnorm.weight");
  const shNorm = f32w("nextn.shared_head_norm.weight"), attnNorm = f32w("attn_norm.weight");
  const postNorm = f32w("post_attention_norm.weight");
  const qNormW = f32w("attn_q_norm.weight"), kNormW = f32w("attn_k_norm.weight");
  const eh = q80("nextn.eh_proj.weight");
  const wq = q40("attn_q.weight"), wk = q40("attn_k.weight"), wv = q40("attn_v.weight"), wo = q40("attn_output.weight");
  const wg = q40("ffn_gate.weight"), wu = q40("ffn_up.weight"), wd = q40("ffn_down.weight");

  const kCache = g.buf(new Float32Array(meta.T * kvDim)), vCache = g.buf(new Float32Array(meta.T * kvDim));
  const eN = g.empty(d * 4), hN = g.empty(d * 4), cat = g.empty(2 * d * 4), hp = g.empty(d * 4);
  const xn = g.empty(d * 4), qFull = g.empty(2 * qDim * 4), kBuf = g.empty(kvDim * 4), vBuf = g.empty(kvDim * 4);
  const qB = g.empty(qDim * 4), gateB = g.empty(qDim * 4), qN = g.empty(qDim * 4), kN = g.empty(kvDim * 4);
  const attnO = g.empty(qDim * 4), y = g.empty(d * 4), gateF = g.empty(dFfn * 4), upF = g.empty(dFfn * 4);
  const out = g.empty(d * 4);
  // parziali dello split del decode, allocati una volta e riusati fra le t
  const aSplits = q35AttnSplitPlan(meta.T).splits;
  const aPart = q35AttnPartialsFloats({ nHead: D.nHead, headDim: hd, ctxMax: meta.T });
  const partOut = g.empty(aPart.out * 4), partMS = g.empty(aPart.ms * 4);
  const expected = f32c(meta.expected.offset, meta.expected.bytes);
  const gq = (w: { qs: GPUBuffer; scales: GPUBuffer; n: number; k: number }, kind: "q4_0" | "q8_0", x: GPUBuffer, o: GPUBuffer): Promise<void> =>
    g.run(gemvQuantWgsl({ kind, K: w.k, N: w.n, hasBias: false }), [w.qs, w.scales, x, o], w.n);

  let l2n = 0, l2d = 0, maxAbs = 0, maxRel = 0;
  for (let t = 0; t < meta.T; t++) {
    const u = g.uniform(t, t);
    const embT = g.buf(f32c(meta.emb.offset + t * d * 4, d * 4));
    const hidT = g.buf(f32c(meta.hidden.offset + t * d * 4, d * 4));
    // h' = eh_proj([norm_e(emb) ; norm_h(hidden)]) — l'ordine e' quello deciso
    // per misura in it.49 e confermato su vLLM in it.51: embedding PRIMO.
    await g.run(rmsnormWgsl(d, D.rmsEps), [embT, enorm, eN], 1);
    await g.run(rmsnormWgsl(d, D.rmsEps), [hidT, hnorm, hN], 1);
    await g.run(stridedCopyWgsl({ nVec: 1, len: d, srcStride: d, srcOffset: 0, dstStride: 2 * d, dstOffset: 0 }), [eN, cat], Math.ceil(d / 64));
    await g.run(stridedCopyWgsl({ nVec: 1, len: d, srcStride: d, srcOffset: 0, dstStride: 2 * d, dstOffset: d }), [hN, cat], Math.ceil(d / 64));
    await gq(eh, "q8_0", cat, hp);
    // ramo attention: identico a un layer full del modello, KV cache SUA
    await g.run(rmsnormWgsl(d, D.rmsEps), [hp, attnNorm, xn], 1);
    await gq(wq, "q4_0", xn, qFull);
    await gq(wk, "q4_0", xn, kBuf);
    await gq(wv, "q4_0", xn, vBuf);
    await g.run(stridedCopyWgsl({ nVec: D.nHead, len: hd, srcStride: 2 * hd, srcOffset: 0, dstStride: hd, dstOffset: 0 }), [qFull, qB], Math.ceil(qDim / 64));
    await g.run(stridedCopyWgsl({ nVec: D.nHead, len: hd, srcStride: 2 * hd, srcOffset: hd, dstStride: hd, dstOffset: 0 }), [qFull, gateB], Math.ceil(qDim / 64));
    await g.run(rmsnormWgsl(hd, D.rmsEps, true), [qB, qNormW, qN], D.nHead);
    await g.run(rmsnormWgsl(hd, D.rmsEps, true), [kBuf, kNormW, kN], D.nKvHead);
    await g.run(ropeNeoxWgsl(D.nHead, hd, D.freqBase, D.ropeDims), [qN], Math.ceil((D.nHead * D.ropeDims / 2) / 64), u);
    await g.run(ropeNeoxWgsl(D.nKvHead, hd, D.freqBase, D.ropeDims), [kN], Math.ceil((D.nKvHead * D.ropeDims / 2) / 64), u);
    await g.run(kvAppendWgsl(kvDim), [kN, kCache], Math.ceil(kvDim / 64), u);
    await g.run(kvAppendWgsl(kvDim), [vBuf, vCache], Math.ceil(kvDim / 64), u);
    await g.run(attnDecodeWgsl({ nHead: D.nHead, nKvHead: D.nKvHead, headDim: hd, ctxMax: meta.T }),
      [qN, kCache, vCache, partOut, partMS], [D.nHead, aSplits, 1], u);
    await g.run(attnDecodeCombineWgsl({ nHead: D.nHead, headDim: hd, ctxMax: meta.T }), [partOut, partMS, attnO], D.nHead);
    await g.run(sigmoidMulWgsl(qDim), [attnO, gateB], Math.ceil(qDim / 64));
    await gq(wo, "q4_0", attnO, y);
    await g.run(addInPlaceWgsl(d), [hp, y], Math.ceil(d / 64));
    // ffn denso + norma della testa (la lm_head condivisa resta fuori)
    await g.run(rmsnormWgsl(d, D.rmsEps), [hp, postNorm, xn], 1);
    await gq(wg, "q4_0", xn, gateF);
    await gq(wu, "q4_0", xn, upF);
    await g.run(siluMulWgsl(dFfn), [gateF, upF], Math.ceil(dFfn / 64));
    await gq(wd, "q4_0", gateF, y);
    await g.run(addInPlaceWgsl(d), [hp, y], Math.ceil(d / 64));
    await g.run(rmsnormWgsl(d, D.rmsEps), [hp, shNorm, out], 1);
    const got = new Float32Array(await g.read(out, d * 4));
    for (let i = 0; i < d; i++) {
      const e = expected[t * d + i], dif = Math.abs(got[i] - e);
      l2n += dif * dif; l2d += e * e;
      if (dif > maxAbs) maxAbs = dif;
      if (Math.abs(e) > 1e-4) maxRel = Math.max(maxRel, dif / Math.abs(e));
    }
  }
  const l2 = Math.sqrt(l2n / Math.max(l2d, 1e-12));
  return {
    kernel: `q35-mtp-head-real-blk${meta.blk}`,
    pass: l2 <= 1e-3 && maxAbs <= 5e-2,
    maxAbs, maxRel,
    note: `pesi reali 4B-MTP, T=${meta.T}, L2rel=${l2.toExponential(2)}`,
    metrics: { l2rel: l2 },
  };
}

/**
 * ACCEPT-RATE della testa MTP sul path VERO (fase-D fase 7, it.53).
 *
 * it.52 ha pinnato la matematica del blocco contro il cpuref (L2rel 2,6e-7) su
 * pesi caricati a mano da un fixture. Questo test copre cio' che il fixture NON
 * puo' coprire: i pesi caricati dal loader del modello, la cache KV della testa
 * che attraversa le posizioni, e il draft che esce dalla lm_head CONDIVISA col
 * modello. Il numero che produce e' quello che la riga 7 chiede — accept-rate
 * per modello su micro-campione — misurato dove servira' davvero.
 *
 * BERSAGLIO: il greedy del MODELLO, non il corpus. In spec-dec il draft si
 * accetta se coincide con cio' che il target avrebbe prodotto, e confonderlo
 * col testo vero sottostima la testa di quanto il modello stesso sbaglia
 * (it.49). Riferimento CPU sulla stessa finestra: 31/62 = 50,0% (it.51).
 */
async function testQ35MtpDraft4B(g: Gpu): Promise<KResult[]> {
  // golden-FULL e non lo smoke: lo smoke ha 40 token in tutto, e il confronto
  // col riferimento CPU vuole la STESSA finestra (i primi 64 token del prompt 0
  // — identici nei due golden, e' lo stesso corpus).
  const goldenRes = await fetch("/models/q35/golden-full.json");
  const headRes = await fetch("/models/Qwen3.5-4B-MTP-Q4_0.gguf", { headers: { Range: "bytes=0-15" } });
  if (!goldenRes.ok || headRes.status !== 206) {
    return [{ kernel: "q35-mtp-draft-4b", pass: false, maxAbs: NaN, maxRel: NaN, note: "manca golden-full.json o il symlink del GGUF MTP in public/models" }];
  }
  const golden = (await goldenRes.json()) as { prompts: { promptTokens: number[]; generated: number[] }[] };
  const URL_GGUF = "/models/Qwen3.5-4B-MTP-Q4_0.gguf";
  const range = async (off: number, len: number): Promise<Uint8Array> => {
    const rr = await fetch(URL_GGUF, { headers: { Range: `bytes=${off}-${off + len - 1}` } });
    if (rr.status !== 206) throw new Error(`q35-mtp-draft: Range non onorato (${rr.status})`);
    const ab = await rr.arrayBuffer();
    if (ab.byteLength !== len) throw new Error(`q35-mtp-draft: Range corto ${ab.byteLength}/${len}`);
    return new Uint8Array(ab);
  };
  const t0 = performance.now();
  const header = await range(0, 64 * 1024 * 1024);
  const f = parseGguf(header.buffer.slice(header.byteOffset, header.byteOffset + header.byteLength) as ArrayBuffer);
  const { shape, byName } = validateQwen35(f);
  const info = (name: string): GgufTensorInfo => {
    const t = byName.get(name);
    if (!t) throw new Error(`q35-mtp-draft: tensore ${name} assente`);
    return t;
  };
  const W = 64;
  const model = await createQ35GpuModel(g.device, {
    shape, info, read: (name) => range(f.dataOffset + info(name).offset, q35TensorBytes(info(name))),
    // `prefillM: 2` non e' il prefill: e' il piano a 2 righe che la verifica
    // speculativa usa per leggere i pesi UNA volta sola (it.55).
  }, W + 8, 12 * (1 << 30), { mtp: true, prefillM: 2 });
  if (!model.mtpDraft) {
    model.destroy();
    return [{ kernel: "q35-mtp-draft-4b", pass: false, maxAbs: NaN, maxRel: NaN, note: "il modello non ha costruito la testa (mtpLayers 0?)" }];
  }
  const loadS = ((performance.now() - t0) / 1000).toFixed(1);
  const p = golden.prompts[0];
  const tokens = [...p.promptTokens, ...p.generated].slice(0, W);
  const t1 = performance.now();
  const am: number[] = [], draft: number[] = [];
  for (let t = 0; t + 1 < tokens.length; t++) {
    am.push(await model.step(tokens[t], t));
    // SUBITO dopo lo step: la testa legge `x`, che il token dopo sovrascrive.
    draft.push(await model.mtpDraft(tokens[t + 1]));
  }
  am.push(await model.step(tokens[tokens.length - 1], tokens.length - 1));
  const perf = model.perf();
  const drafts = perf.mtpDrafts, msPerDraft = perf.mtpMs / Math.max(1, drafts);
  const runS = ((performance.now() - t1) / 1000).toFixed(1);

  let hit = 0, tot = 0;
  for (let i = 0; i + 2 < tokens.length; i++) { tot++; if (draft[i] === am[i + 1]) hit++; }
  const acc = (100 * hit) / tot;
  // Il riferimento CPU f64 sulla STESSA finestra e' 50,0% (it.51). La soglia a
  // 40 e' larga di proposito: fra CPU e GPU cambiano precisione (f64 vs f32) e
  // hidden (il full-model diverge dall'oracolo su ~1% delle posizioni), quindi
  // pretendere l'uguaglianza sarebbe pretendere il rumore. Sotto 40 non c'e'
  // rumore che tenga: e' cablaggio rotto.
  const rAccept: KResult = {
    kernel: "q35-mtp-draft-4b",
    pass: acc >= 40,
    maxAbs: NaN, maxRel: NaN,
    note: `accept-rate GPU ${hit}/${tot} = ${acc.toFixed(1)}% (cpuref f64 stessa finestra: 50,0%) — ${drafts} draft a ${msPerDraft.toFixed(2)} ms (load ${loadS}s, run ${runS}s)`,
    metrics: { acceptPct: acc, msPerDraft },
  };

  // ---- GATE SECCO della riga 7: generare con draft+verify deve dare gli
  // STESSI token della generazione sequenziale. Non "quasi": identici, uno per
  // uno. E' l'unico gate che distingue una speculazione corretta da una che
  // accetta draft sbagliati — e l'unico che si accorge di uno stato ricorrente
  // non riparato, che altrimenti produce testo plausibile e diverso.
  const P = 24, N = 16;
  const prefill = async (): Promise<number> => {
    model.resetState();
    let a = -1;
    for (let i = 0; i < P; i++) a = await model.step(tokens[i], i);
    return a;
  };
  // Il cronometro parte DOPO il prefill in entrambi i bracci: 24 posizioni di
  // prefill dentro un ms/token su 16 token valgono ~50 ms a token e
  // renderebbero i due numeri incomparabili con qualunque altra misura.
  let nx = await prefill();
  const t2 = performance.now();
  const seqRef: number[] = [];
  for (let k = 0, q = P; k < N; k++, q++) { seqRef.push(nx); nx = await model.step(nx, q); }
  const seqMs = (performance.now() - t2) / N;

  nx = await prefill();
  const t3 = performance.now();
  const seqSpec: number[] = [];
  let q = P, passes = 0, acceptedPasses = 0;
  while (seqSpec.length < N) {
    // h_{q-1} e' in `x`; il draft e' l'ipotesi su t_{q+1}
    const dr = await model.mtpDraft!(nx);
    const [b0, b1] = await model.specVerify!(nx, dr, q);
    passes++;
    seqSpec.push(nx);
    if (dr === b0) { acceptedPasses++; seqSpec.push(b0); nx = b1; q += 2; }
    else { model.specRollback!(); nx = b0; q += 1; }
  }
  seqSpec.length = N;
  const specMs = (performance.now() - t3) / N;

  // --- stesso ciclo, ma sul piano a 2 righe: i pesi si leggono una volta ---
  nx = await prefill();
  const t4 = performance.now();
  const seqSpecB: number[] = [];
  let qB = P, passesB = 0, acceptedB = 0;
  while (seqSpecB.length < N) {
    const dr = await model.mtpDraft!(nx);
    const [b0, b1] = await model.specVerifyBatched!(nx, dr, qB);
    passesB++;
    seqSpecB.push(nx);
    const ok = dr === b0;
    model.specCommit!(ok);
    if (ok) { acceptedB++; seqSpecB.push(b0); nx = b1; qB += 2; }
    else { nx = b0; qB += 1; }
  }
  seqSpecB.length = N;
  const specBMs = (performance.now() - t4) / N;

  // ---- DOVE VANNO I 60 ms DELLA PASSATA (it.56) ----
  // Non si ottimizza a naso: si separa il CORPO del modello dalla CODA
  // (norma finale + lm_head + argmax + readback). `step(read=false)` esegue
  // esattamente il corpo — la coda e' tagliata dal `headCut` della fase 4 — e
  // una fence sulla coda della queue rende il tempo confrontabile con quello
  // del token intero, che aspetta il readback per costruzione.
  const fence = (): Promise<undefined> => g.device.queue.onSubmittedWorkDone();
  const K = 12;
  await prefill();
  await fence();
  const t5 = performance.now();
  for (let k = 0, q5 = P; k < K; k++, q5++) await model.step(tokens[q5 % tokens.length], q5, false);
  await fence();
  const bodyMs = (performance.now() - t5) / K;
  await prefill();
  await fence();
  const t6 = performance.now();
  for (let k = 0, q6 = P; k < K; k++, q6++) await model.step(tokens[q6 % tokens.length], q6, true);
  await fence();
  const fullMs = (performance.now() - t6) / K;
  const pf = model.perf();
  model.destroy();

  const diffAt = (a: number[]): number => {
    for (let i = 0; i < N; i++) if (a[i] !== seqRef[i]) return i;
    return -1;
  };
  const firstDiff = diffAt(seqSpec), firstDiffB = diffAt(seqSpecB);
  const accPass = (100 * acceptedPasses) / Math.max(1, passes);
  const accPassB = (100 * acceptedB) / Math.max(1, passesB);
  const rInv: KResult = {
    kernel: "q35-mtp-specdec-invariance",
    pass: firstDiff < 0 && firstDiffB < 0,
    maxAbs: NaN, maxRel: NaN,
    note: firstDiff >= 0 || firstDiffB >= 0
      ? `DIVERGE: per-riga al token ${firstDiff}, batch al token ${firstDiffB} (riferimento sequenziale)`
      : `${N}/${N} token IDENTICI al sequenziale su ENTRAMBI i path · per-riga ${passes} passate ${acceptedPasses} accettate (${accPass.toFixed(0)}%) · batch ${passesB} passate ${acceptedB} accettate (${accPassB.toFixed(0)}%), ${pf.specRejects} rollback · ms/token: sequenziale ${seqMs.toFixed(1)} · spec per-riga ${specMs.toFixed(1)} · spec batch ${specBMs.toFixed(1)}`,
    metrics: { acceptedPct: accPassB, msPerTokenSpec: specMs, msPerTokenSpecBatched: specBMs, msPerTokenSeq: seqMs },
  };
  // La passata di verifica batch, ricostruita: corpo a 2 righe + DUE code.
  // Il modello atteso e' `corpo2 + 2*coda`; quanto il corpo a 2 righe costa
  // davvero si ottiene per differenza, ed e' il numero che dice se il batch
  // sta amortizzando la lettura dei pesi o no.
  const tailMs = fullMs - bodyMs;
  const passMs = specBMs * N / Math.max(1, passesB);
  const body2Ms = passMs - 2 * tailMs - msPerDraft;
  const rCost: KResult = {
    kernel: "q35-mtp-costo-verifica",
    pass: bodyMs > 0 && tailMs > 0,
    maxAbs: NaN, maxRel: NaN,
    note: `token intero ${fullMs.toFixed(1)} = corpo ${bodyMs.toFixed(1)} + coda ${tailMs.toFixed(1)} (norma+lm_head+argmax+readback) · passata di verifica ${passMs.toFixed(1)} = corpo2 ${body2Ms.toFixed(1)} + 2 code ${(2 * tailMs).toFixed(1)} + draft ${msPerDraft.toFixed(1)} · corpo2/corpo = ${(body2Ms / bodyMs).toFixed(2)}x`,
    metrics: { bodyMs, tailMs, passMs, body2Ms, body2Ratio: body2Ms / bodyMs },
  };
  return [rAccept, rInv, rCost];
}

/** Full-model GPU 4B teacher-forced: argmax GPU == ORACOLO sul golden smoke (fase 4 slice 3). */
async function testQ35Model4B(g: Gpu): Promise<KResult> {
  const goldenRes = await fetch("/models/q35/golden-smoke.json");
  const ggufHead = await fetch("/models/Qwen3.5-4B-Q4_0.gguf", { method: "HEAD" });
  if (!goldenRes.ok || !ggufHead.ok) {
    return { kernel: "q35-model-4b-argmax", pass: false, maxAbs: NaN, maxRel: NaN, note: "manca golden-smoke.json o symlink GGUF in public/models (vedi journal it.8)" };
  }
  const golden = (await goldenRes.json()) as {
    prompts: { promptTokens: number[]; generated: number[]; positions: { argmax: number }[] }[];
  };
  const t0 = performance.now();
  // Niente file intero in RAM: 2.58 GB sfonda sia arrayBuffer() (cap 2 GiB
  // Chromium) sia l'allocazione di un ArrayBuffer singolo. Range per-tensore
  // (vite risponde 206, verificato): header 64 MB, poi un GET per tensore,
  // upload e scarto — la stessa postura streaming del loader cpuref.
  const URL_GGUF = "/models/Qwen3.5-4B-Q4_0.gguf";
  const range = async (off: number, len: number): Promise<Uint8Array> => {
    const rr = await fetch(URL_GGUF, { headers: { Range: `bytes=${off}-${off + len - 1}` } });
    if (rr.status !== 206) throw new Error(`q35-model: Range non onorato (${rr.status})`);
    const ab = await rr.arrayBuffer();
    if (ab.byteLength !== len) throw new Error(`q35-model: Range corto ${ab.byteLength}/${len}`);
    return new Uint8Array(ab);
  };
  const header = await range(0, 64 * 1024 * 1024);
  const f = parseGguf(header.buffer.slice(header.byteOffset, header.byteOffset + header.byteLength) as ArrayBuffer);
  const { shape, byName } = validateQwen35(f);
  const model = await createQ35GpuModel(g.device, {
    shape,
    info: (name) => {
      const t = byName.get(name);
      if (!t) throw new Error(`q35-model: tensore ${name} assente`);
      return t;
    },
    read: (name) => {
      const t = byName.get(name);
      if (!t) throw new Error(`q35-model: tensore ${name} assente`);
      return range(f.dataOffset + t.offset, q35TensorBytes(t));
    },
  }, 64);
  const loadS = ((performance.now() - t0) / 1000).toFixed(1);
  const p = golden.prompts[0];
  const P = p.promptTokens.length;
  const tokens = [...p.promptTokens, ...p.generated.slice(0, -1)];
  const t1 = performance.now();
  let ok = 0;
  const detail: string[] = [];
  for (let t = 0; t < tokens.length; t++) {
    const am = await model.step(tokens[t], t);
    const gi = t - (P - 1);
    if (gi >= 0 && gi < p.positions.length) {
      const want = p.positions[gi].argmax;
      if (am === want) ok++;
      else detail.push(`pos gen ${gi}: gpu ${am} vs oracolo ${want}`);
    }
  }
  const decodeS = ((performance.now() - t1) / 1000).toFixed(1);

  // ---- MICRO-BENCH (fase D fase 3), riscritto dopo il FAIL del verifier su
  // it.10. L'errore era di METODO, non di codice: il braccio "con sync" era la
  // PRIMA passata dopo il load (a freddo) e quello "batch" la terza (a caldo),
  // quindi ~8 ms/token di warm-up finivano tutti nel braccio lento e il delta
  // usciva gonfiato di ~5x. Qui: una passata di WARM-UP scartata, poi i
  // bracci INTERLEAVATI e ripetuti, e si riporta la MEDIANA con la
  // dispersione — non un campione singolo.
  const passSync = async (): Promise<number> => {
    model.resetState();
    const t = performance.now();
    for (let i2 = 0; i2 < tokens.length; i2++) await model.step(tokens[i2], i2);
    return (performance.now() - t) / tokens.length;
  };
  const passBatch = async (): Promise<{ ms: number; ids: number[] }> => {
    model.resetState();
    const ids: number[] = [];
    const t = performance.now();
    for (let off = 0; off < tokens.length; off += 32) {
      ids.push(...await model.decodeBatch!(tokens.slice(off, off + 32), off));
    }
    return { ms: (performance.now() - t) / tokens.length, ids };
  };
  const passNoSync = async (): Promise<number> => {
    model.resetState();
    const t = performance.now();
    for (let i2 = 0; i2 < tokens.length; i2++) await model.step(tokens[i2], i2, false);
    await model.step(tokens[tokens.length - 1], tokens.length, true);
    return (performance.now() - t) / (tokens.length + 1);
  };
  await passSync(); // WARM-UP scartato: la prima passata dopo il load non e' comparabile

  const REP = 3;
  const msSync: number[] = [], msBatch: number[] = [], msNoSync: number[] = [];
  let batchIds: number[] = [];
  for (let r = 0; r < REP; r++) {
    msSync.push(await passSync());
    const b = await passBatch();
    msBatch.push(b.ms); batchIds = b.ids;
    msNoSync.push(await passNoSync());
  }
  const med = (a: number[]): number => [...a].sort((p2, q2) => p2 - q2)[a.length >> 1];
  const disp = (a: number[]): string => `${Math.min(...a).toFixed(1)}-${Math.max(...a).toFixed(1)}`;
  const medSync = med(msSync), medBatch = med(msBatch), medNoSync = med(msNoSync);

  // GATE SECCO: gli argmax del batch devono essere IDENTICI a quelli del path
  // a readback. Sono interi: o sono uguali o non lo sono.
  const seqIds: number[] = [];
  model.resetState();
  for (let t = 0; t < tokens.length; t++) seqIds.push(await model.step(tokens[t], t));
  let diff = -1;
  for (let i2 = 0; i2 < tokens.length; i2++) if (seqIds[i2] !== batchIds[i2]) { diff = i2; break; }
  model.resetState();
  const pf = model.perf();

  return {
    kernel: "q35-model-4b-argmax",
    pass: ok === p.positions.length && diff === -1,
    maxAbs: 0, maxRel: 0,
    note: `argmax GPU == oracolo ${ok}/${p.positions.length} (load ${loadS}s, ${tokens.length} pos in ${decodeS}s, ${model.dispatchesPerToken} dispatch/token) — MICRO-BENCH a caldo A CONTESTO ${tokens.length} POSIZIONI (non confrontabile con un bench su prompt vero senza correggere per il contesto: ~10,4 us/posizione, it.59), ${REP} ripetizioni interleavate (mediana, [min-max]): sync ${medSync.toFixed(1)} [${disp(msSync)}] · batch ${medBatch.toFixed(1)} [${disp(msBatch)}] · senza-sync ${medNoSync.toFixed(1)} [${disp(msNoSync)}] ⇒ delta batch ${(medBatch - medSync).toFixed(1)} ms/token (${((medBatch / medSync - 1) * 100).toFixed(1)}%); argmax ${diff === -1 ? "IDENTICO" : `DIVERSO alla posizione ${diff} (seq ${seqIds[diff]} vs batch ${batchIds[diff]})`} su ${tokens.length} token${detail.length ? " — " + detail.join("; ") : ""}`,
    metrics: {
      okPositions: ok, dispatchesPerToken: model.dispatchesPerToken,
      msTokenSyncMediana: +medSync.toFixed(2), msTokenBatchMediana: +medBatch.toFixed(2),
      msTokenSenzaSyncMediana: +medNoSync.toFixed(2),
      msTokenSyncSpread: +(Math.max(...msSync) - Math.min(...msSync)).toFixed(2),
      msTokenBatchSpread: +(Math.max(...msBatch) - Math.min(...msBatch)).toFixed(2),
      deltaBatchMsToken: +(medBatch - medSync).toFixed(2),
      msTokenEmbedCpu: +(pf.embedMs / pf.tokens).toFixed(3),
      argmaxDiffBatch: diff, // -1 = identico
    },
  };
}

/**
 * Blocco MoE 35B su GPU con pesi REALI vs cpuref (q1 fase 7 slice 3a):
 * router F32 → top-8 (CPU, correttezza-prima) → per-expert gemv Q4_K a
 * OFFSET nel buffer di classe (il seme dell'arena parametrica) → axpy
 * pesato → shared expert Q8_0 con gate sigmoid su GPU. Un layer con down
 * Q4_K e uno con down Q6_K (il mix UD). Pesi via fetch Range dal GGUF.
 */
async function testQ35MoeBlockReal(g: Gpu): Promise<KResult[]> {
  const URL35 = "/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf";
  const head = await fetch(URL35, { method: "HEAD" });
  if (!head.ok) {
    return [{ kernel: "q35-moe-block-real", pass: false, maxAbs: NaN, maxRel: NaN, note: "symlink 35B assente in public/models" }];
  }
  const range = async (off: number, len: number): Promise<Uint8Array> => {
    const rr = await fetch(URL35, { headers: { Range: `bytes=${off}-${off + len - 1}` } });
    if (rr.status !== 206) throw new Error(`q35-moe: Range ${rr.status}`);
    return new Uint8Array(await rr.arrayBuffer());
  };
  const hdr = await range(0, 64 * 1024 * 1024);
  const f = parseGguf(hdr.buffer.slice(hdr.byteOffset, hdr.byteOffset + hdr.byteLength) as ArrayBuffer);
  const { shape, byName } = validateQwen35(f);
  const nE = shape.nExpert as number, topK = shape.nExpertUsed as number, dE = shape.dFfnExpert as number;
  const d = shape.dModel;
  // un layer down-Q4_K e uno down-Q6_K (mix UD enumerato dal file)
  const downType = (l: number): number => byName.get(`blk.${l}.ffn_down_exps.weight`)!.type;
  const layers: number[] = [];
  for (let l = 0; l < shape.nLayer && layers.length < 2; l++) {
    const t = downType(l);
    if (layers.length === 0 && t === GGML_TYPE_Q4K) layers.push(l);
    else if (layers.length === 1 && t === GGML_TYPE_Q6K) layers.push(l);
  }
  if (layers.length < 2) for (let l = 0; l < shape.nLayer && layers.length < 2; l++) if (!layers.includes(l)) layers.push(l);

  const results: KResult[] = [];
  for (const L of layers) {
    const b = `blk.${L}.`;
    const info = (s: string) => byName.get(`${b}${s}`)!;
    const tBytes = (s: string) => q35TensorBytes(info(s));
    const fetchT = (s: string) => range(f.dataOffset + info(s).offset, tBytes(s));
    const routerRaw = await fetchT("ffn_gate_inp.weight");
    const router = new Float32Array(routerRaw.slice().buffer);
    const shGateV = new Float32Array((await fetchT("ffn_gate_inp_shexp.weight")).slice().buffer);
    const shTens = { gate: await fetchT("ffn_gate_shexp.weight"), up: await fetchT("ffn_up_shexp.weight"), down: await fetchT("ffn_down_shexp.weight") };
    // input sintetico realistico + selezione con la STESSA matematica del cpuref
    const x64 = Float64Array.from(randF32(d, 1600 + L, 0.5));
    const logits = new Float64Array(nE);
    for (let e = 0; e < nE; e++) {
      let acc = 0;
      for (let i = 0; i < d; i++) acc += router[e * d + i] * x64[i];
      logits[e] = acc;
    }
    const mx = Math.max(...logits);
    const probs = Array.from(logits, (v) => Math.exp(v - mx));
    const ps = probs.reduce((a, c) => a + c, 0);
    const sel = Array.from({ length: nE }, (_, e) => e).sort((a2, b2) => probs[b2] - probs[a2] || a2 - b2).slice(0, topK);
    const wSum = Math.max(sel.reduce((a2, e) => a2 + probs[e], 0) / ps, 6.103515625e-5);
    const weights = sel.map((e) => probs[e] / ps / wSum);
    // pesi expert selezionati via Range (slice per-expert dai tensori stacked)
    const expBytes = (s: string) => {
      const t = info(s);
      const elemsPer = t.dims[0] * t.dims[1];
      const bb = t.type === GGML_TYPE_Q4K ? 144 : 210;
      // rp = byte REPACKED (repackKQuant padda i superblocchi a word: 210→212)
      return { t, elemsPer, bb, per: (elemsPer / 256) * bb, rp: (elemsPer / 256) * (bb === 210 ? 212 : bb) };
    };
    const gU = expBytes("ffn_gate_exps.weight"), uU = expBytes("ffn_up_exps.weight"), dU = expBytes("ffn_down_exps.weight");
    const expRaw = new Map<number, { gate: Uint8Array; up: Uint8Array; down: Uint8Array }>();
    for (const e of sel) {
      expRaw.set(e, {
        gate: await range(f.dataOffset + gU.t.offset + e * gU.per, gU.per),
        up: await range(f.dataOffset + uU.t.offset + e * uU.per, uU.per),
        down: await range(f.dataOffset + dU.t.offset + e * dU.per, dU.per),
      });
    }
    // cpuref f64 (stessa fonte unica del modello)
    const refW: Q35MoeLayerWeights = {
      router, sharedGate: shGateV,
      shGate: deqQ80(shTens.gate), shUp: deqQ80(shTens.up), shDown: deqQ80(shTens.down),
      expert: (e) => {
        const r = expRaw.get(e)!;
        const dq = (raw: Uint8Array, spec: typeof gU): Float32Array => {
          const out = new Float32Array(spec.elemsPer);
          (spec.bb === 144 ? dequantQ4_K : dequantQ6_K)(raw, 0, spec.elemsPer / 256, out);
          return out;
        };
        return { gate: dq(r.gate, gU), up: dq(r.up, uU), down: dq(r.down, dU) };
      },
    };
    const ref = q35MoeFfnRefF64(x64, refW, nE, topK, dE);

    // --- GPU: arena di classe con OFFSET binding (il seme del paging q35) ---
    const slotBytes = gU.rp + uU.rp + dU.rp; // layout REPACKED (Q6_K paddato)
    const arena = g.empty(slotBytes * topK);
    sel.forEach((e, k) => {
      const r = expRaw.get(e)!;
      const packed = new Uint8Array(slotBytes);
      packed.set(repackKQuantU8(r.gate, gU.bb), 0);
      packed.set(repackKQuantU8(r.up, uU.bb), gU.rp);
      packed.set(repackKQuantU8(r.down, dU.bb), gU.rp + uU.rp);
      g.device.queue.writeBuffer(arena, k * slotBytes, packed);
    });
    const xB = g.buf(Float32Array.from(x64));
    const out = g.buf(new Float32Array(d));
    const gateT = g.empty(dE * 4), upT = g.empty(dE * 4), dnT = g.empty(d * 4);
    const gemvGate = gemvQ4KWgsl({ K: d, N: dE });
    const gemvDown = dU.bb === 144 ? gemvQ4KWgsl({ K: dE, N: d }) : gemvQ6KWgsl({ K: dE, N: d });
    // SINGLE-PASS come il modello (it.17): un encoder, un pass, scratch riusati
    // fra gli 8 expert, pesi in wBufs dedicati — replica ESATTA del runtime.
    const mkP = (code: string) => g.device.createComputePipeline({ layout: "auto", compute: { module: g.device.createShaderModule({ code }), entryPoint: "main" } });
    const pG = mkP(gemvGate), pS = mkP(siluMulWgsl(dE)), pD = mkP(gemvDown), pA = mkP(axpyWgsl(d));
    const wBufsK = weights.map((w) => {
      const bwk = g.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      g.device.queue.writeBuffer(bwk, 0, new Float32Array([w, 0, 0, 0]));
      return bwk;
    });
    const mb = (pp: GPUComputePipeline, es: (GPUBuffer | GPUBufferBinding)[]) => g.device.createBindGroup({
      layout: pp.getBindGroupLayout(0),
      entries: es.map((e, i) => ({ binding: i, resource: (e as GPUBufferBinding).buffer ? (e as GPUBufferBinding) : { buffer: e as GPUBuffer } })),
    });
    {
      const enc = g.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      const dsp = (pp: GPUComputePipeline, bgx: GPUBindGroup, wx: number, wy = 1) => { pass.setPipeline(pp); pass.setBindGroup(0, bgx); pass.dispatchWorkgroups(wx, wy, 1); };
      const bgS = mb(pS, [gateT, upT]);
      for (let k = 0; k < topK; k++) {
        const base = k * slotBytes;
        dsp(pG, mb(pG, [{ buffer: arena, offset: base, size: gU.rp }, xB, gateT]), dE);
        dsp(pG, mb(pG, [{ buffer: arena, offset: base + gU.rp, size: uU.rp }, xB, upT]), dE);
        dsp(pS, bgS, Math.ceil(dE / 64));
        const gg3 = gemvGrid(d);
        dsp(pD, mb(pD, [{ buffer: arena, offset: base + gU.rp + uU.rp, size: dU.rp }, gateT, dnT]), gg3[0], gg3[1]);
        dsp(pA, mb(pA, [out, dnT, wBufsK[k]]), Math.ceil(d / 64));
      }
      pass.end();
      g.device.queue.submit([enc.finish()]);
    }

    // --- ARENA vera (fase D fase 3b): stesso buffer bindato INTERO, lo slot
    // arriva da `Sel` e il kernel ricava da solo (binding, base). E' il regime
    // che toglie alla CPU il calcolo degli indirizzi — e quindi il readback
    // per layer. Qui si prova che produce gli STESSI BIT del binding a
    // sotto-range: un offset sbagliato darebbe un risultato plausibile ma
    // diverso, che e' il modo peggiore di sbagliare.
    const outArena = g.buf(new Float32Array(d));
    {
      const kar = (tensorWords: number): KArenaOpts => ({
        nBuf: 1, slabWords: slotBytes / 4, slabsPerBuf: topK, tensorWords,
      });
      const pGA = mkP(gemvQ4KWgsl({ K: d, N: dE, arena: kar(0) }));
      const pUA = mkP(gemvQ4KWgsl({ K: d, N: dE, arena: kar(gU.rp / 4) }));
      const dnArena = { K: dE, N: d, arena: kar((gU.rp + uU.rp) / 4) };
      const pDA = mkP(dU.bb === 144 ? gemvQ4KWgsl(dnArena) : gemvQ6KWgsl(dnArena));
      // Sel: una entry per k — {id, slot, w, flags}. Qui la riempie la CPU,
      // come nella slice A di GLM; il router su GPU la scrivera' al suo posto.
      const selData = new ArrayBuffer(topK * 16);
      const selU = new Uint32Array(selData), selF = new Float32Array(selData);
      sel.forEach((e, k) => { selU[k * 4] = e; selU[k * 4 + 1] = k; selF[k * 4 + 2] = weights[k]; });
      const selBuf = g.device.createBuffer({ size: topK * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      g.device.queue.writeBuffer(selBuf, 0, new Uint8Array(selData));
      // moeIdx: un sotto-range per k (l'offset uniform vuole 256 byte di
      // allineamento). NON si puo' riscrivere lo stesso buffer in ciclo:
      // queue.writeBuffer e' ordinata prima del submit e tutti gli step
      // vedrebbero l'ultimo valore.
      const idxBuf = g.device.createBuffer({ size: topK * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      for (let k = 0; k < topK; k++) g.device.queue.writeBuffer(idxBuf, k * 256, new Uint32Array([k, 0, L, 0]));
      const gateA = g.empty(dE * 4), upA = g.empty(dE * 4), dnA = g.empty(d * 4);
      const enc = g.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      const dsp = (pp: GPUComputePipeline, bgx: GPUBindGroup, wx: number, wy = 1) => { pass.setPipeline(pp); pass.setBindGroup(0, bgx); pass.dispatchWorkgroups(wx, wy, 1); };
      const bgSA = mb(pS, [gateA, upA]);
      for (let k = 0; k < topK; k++) {
        const idx: GPUBufferBinding = { buffer: idxBuf, offset: k * 256, size: 16 };
        dsp(pGA, mb(pGA, [arena, xB, gateA, selBuf, idx]), dE);
        dsp(pUA, mb(pUA, [arena, xB, upA, selBuf, idx]), dE);
        dsp(pS, bgSA, Math.ceil(dE / 64));
        const gg4 = gemvGrid(d);
        dsp(pDA, mb(pDA, [arena, gateA, dnA, selBuf, idx]), gg4[0], gg4[1]);
        dsp(pA, mb(pA, [outArena, dnA, wBufsK[k]]), Math.ceil(d / 64));
      }
      pass.end();
      g.device.queue.submit([enc.finish()]);
    }
    // Confronto PRIMA che il shared expert si accumuli su `out`: qui i due
    // buffer contengono solo il contributo degli expert, calcolato nei due
    // regimi di indirizzamento.
    const subRangeF = new Float32Array(await g.read(out, d * 4));
    const arenaF = new Float32Array(await g.read(outArena, d * 4));
    let arenaDiff = -1;
    for (let i = 0; i < d; i++) {
      if (!Object.is(subRangeF[i], arenaF[i])) { arenaDiff = i; break; }
    }

    // shared expert Q8_0 + gate sigmoid su GPU
    const shg = repackQ8_0(shTens.gate, 0, (d / 32) * dE);
    const shu = repackQ8_0(shTens.up, 0, (d / 32) * dE);
    const shd = repackQ8_0(shTens.down, 0, (dE / 32) * d);
    await g.run(gemvQuantWgsl({ kind: "q8_0", K: d, N: dE, hasBias: false }), [g.buf(shg.qs), g.buf(shg.scales), xB, gateT], dE);
    await g.run(gemvQuantWgsl({ kind: "q8_0", K: d, N: dE, hasBias: false }), [g.buf(shu.qs), g.buf(shu.scales), xB, upT], dE);
    await g.run(siluMulWgsl(dE), [gateT, upT], Math.ceil(dE / 64));
    await g.run(gemvQuantWgsl({ kind: "q8_0", K: dE, N: d, hasBias: false }), [g.buf(shd.qs), g.buf(shd.scales), gateT, dnT], [...gemvGrid(d), 1] as [number, number, number]);
    const gScal = g.empty(4);
    await g.run(gemvF32Wgsl({ K: d, N: 1 }), [g.buf(shGateV), xB, gScal], 1);
    await g.run(axpyWgsl(d, true), [out, dnT, gScal], Math.ceil(d / 64));

    const got = new Float32Array(await g.read(out, d * 4));
    const refF = Float32Array.from(ref.out);
    let l2n = 0, l2d = 0, maxAbs = 0, maxRel = 0;
    for (let i = 0; i < d; i++) {
      const dif = Math.abs(got[i] - refF[i]);
      l2n += dif * dif; l2d += refF[i] * refF[i];
      if (dif > maxAbs) maxAbs = dif;
      if (Math.abs(refF[i]) > 1e-4) maxRel = Math.max(maxRel, dif / Math.abs(refF[i]));
    }
    const l2 = Math.sqrt(l2n / Math.max(l2d, 1e-12));
    results.push({
      kernel: `q35-arena-vs-slotrange-blk${L}-${dU.bb === 144 ? "q4k" : "q6k"}down`,
      pass: arenaDiff === -1,
      maxAbs: 0, maxRel: 0,
      note: arenaDiff === -1
        ? `BIT-A-BIT identico su ${d} f32 (arena bindata intera, slot da Sel) vs binding a sotto-range`
        : `DIVERSO all'indice ${arenaDiff}: sotto-range ${subRangeF[arenaDiff]} vs arena ${arenaF[arenaDiff]}`,
    });
    results.push({
      kernel: `q35-moe-block-real-blk${L}-${dU.bb === 144 ? "q4k" : "q6k"}down`,
      pass: l2 <= 1e-3 && maxAbs <= 5e-2,
      maxAbs, maxRel,
      note: `pesi reali 35B, top-${topK} di ${nE}, arena offset-binding, L2rel=${l2.toExponential(2)}`,
      metrics: { l2rel: l2 },
    });
  }
  return results;
}

const GGML_TYPE_Q4K = 12, GGML_TYPE_Q6K = 14;
const deqQ80 = (raw: Uint8Array): Float32Array => {
  const out = new Float32Array((raw.length / 34) * 32);
  dequantQ8_0(raw, 0, raw.length / 34, out);
  return out;
};
/** repackKQuant ritorna Uint32Array: vista U8 per il packing nello slot. */
const repackKQuantU8 = (raw: Uint8Array, blockBytes: number): Uint8Array => {
  const w = repackKQuant(raw, 0, raw.length / blockBytes, blockBytes);
  return new Uint8Array(w.buffer, w.byteOffset, w.byteLength);
};

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
async function testGemvC2(g: Gpu, kind: "q4_1" | "q4_K" | "q5_K" | "q6_K", K: number, N: number): Promise<KResult> {
  const blockWeights = kind === "q4_1" ? 32 : 256;
  const blockBytes = kind === "q4_1" ? Q4_1_BLOCK_BYTES : kind === "q4_K" ? Q4_K_BLOCK_BYTES : kind === "q5_K" ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;
  const nBlocks = (K / blockWeights) * N;
  const src = randBytes(nBlocks * blockBytes, 4321 + K + N);
  if (kind === "q4_1") fixScalesAt(src, blockBytes, [1, 3]);       // d, m
  else if (kind === "q4_K" || kind === "q5_K") fixScalesAt(src, blockBytes, [1, 3]);  // d, dmin
  else fixScalesAt(src, blockBytes, [209]);                        // d in coda
  const w = new Float32Array(nBlocks * blockWeights);
  (kind === "q4_1" ? dequantQ4_1 : kind === "q4_K" ? dequantQ4_K : kind === "q5_K" ? dequantQ5_K : dequantQ6_K)(src, 0, nBlocks, w);
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
    const code = kind === "q4_K" ? gemvQ4KWgsl({ K, N }) : kind === "q5_K" ? gemvQ5KWgsl({ K, N }) : gemvQ6KWgsl({ K, N });
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
  const sel = routerSelect(logits, bias, ROUTER_GLM47);
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

/**
 * Kit del mini-modello 2 layer: sorgente sintetica deterministica (stessi seed
 * ⇒ stessi byte a ogni chiamata) + costruttori dei riferimenti f64. Estratto
 * da testGlmModel2Layer (C3b fase 2) perche' i casi del decode ottimistico
 * hanno bisogno DELLO STESSO modello con riferimenti propri — in particolare
 * `mkRef1({zeroExperts})`, il riferimento del degrado DEFINITO: un expert
 * marcato MISS contribuisce ZERO (guardia `ok` dei kernel d'arena), e il
 * riferimento lo modella azzerandone i pesi (gate 0 ⇒ silu(0)·up = 0 ⇒
 * down·0 = 0), a router INVARIATO (i pesi del router sono non-expert).
 */
function mkMiniModelKit() {
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
  const mkRef0 = () => new GlmDenseLayerRefF64({
    ...attnW(0), ffnNorm: dq(0, "ffn_norm.weight"),
    wGate: dq(0, "ffn_gate.weight"), wUp: dq(0, "ffn_up.weight"), wDown: dq(0, "ffn_down.weight"),
  });
  const mkRef1 = (zeroExperts?: Set<number>) => new GlmMoeLayerRefF64(attnW(1), dq(1, "ffn_norm.weight"), {
    routerW: dq(1, "ffn_gate_inp.weight"), routerBias: dq(1, "exp_probs_b.bias"),
    expert: (e: number): GlmMoeExpertWeights => {
      const r = expert(1, e);
      const w = { gate: deqBy(r.gate, "q4_0"), up: deqBy(r.up, "q4_0"), down: deqBy(r.down, "q4_1") };
      // il degrado DEFINITO del MISS: contributo zero, selezione invariata
      if (zeroExperts?.has(e)) {
        return { gate: new Float32Array(w.gate.length), up: new Float32Array(w.up.length), down: new Float32Array(w.down.length) };
      }
      return w;
    },
    gateShexp: dq(1, "ffn_gate_shexp.weight"), upShexp: dq(1, "ffn_up_shexp.weight"), downShexp: dq(1, "ffn_down_shexp.weight"),
  });
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
  return { srcMock, VOCAB_T, mkRef0, mkRef1, headRefF64 };
}

async function testGlmModel2Layer(
  g: Gpu, select: "cpu" | "shadow" | "gpu" = "cpu",
  io?: { record?: ModelTrace; against?: ModelTrace },
): Promise<KResult> {
  const shadow = select === "shadow";
  const gpuSel = select === "gpu";
  const name = gpuSel ? "glm-model-2layer-gpu" : shadow ? "glm-model-2layer-shadow" : "glm-model-2layer";
  const kit = mkMiniModelKit();
  const { srcMock, VOCAB_T, headRefF64 } = kit;
  const ref0 = kit.mkRef0();
  const ref1 = kit.mkRef1();

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
  // (ref f64 dell'head: ora nel kit — headRefF64)
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

// ============ C3B FASE 2 — decode ottimistico: flag di miss + checkpoint ====
// Spec: docs/superpowers/specs/2026-08-07-engine-fase-c3b-decode-ottimistico.md
// Qui si misura il MECCANISMO, non il repair (fase 3): (a) il resolve marca
// ESATTAMENTE i miss forzati e mai altro (0 falsi positivi); (b) il degrado e'
// quello DEFINITO (contributo zero, selezione invariata) — confronto col
// riferimento f64 a expert azzerati; (c) il checkpoint hidden e' bit-identico
// fra "gpu" e "optimistic" e ancorato ai contenuti (riga 0 = xIn del token).

async function testGlmOptimisticForcedMiss(g: Gpu): Promise<KResult> {
  const name = "glm-optimistic-forced-miss";
  const problems: string[] = [];
  const arenaCfg = ktestArena(true); // 64 slot: la precondizione 0.88 passa, i MISS li inietta l'harness
  // repair SPENTO (solo harness): questo caso misura il flag e il degrado
  // definito, che il repair della fase 3 altrimenti ripara prima del ritorno.
  // Il repair acceso lo esercita glm-optimistic-identita.
  const mkModel = () => createGlmModel(g.device, mkMiniModelKit().srcMock, {
    nLayer: 2, ctxMax: 16, head: true, vocab: 2048, select: "optimistic", optimisticRepair: false,
    cache: {
      budgetBytes: 0, slotsOverride: arenaCfg.slotsOverride,
      maxBindingBytes: arenaCfg.window, maxBufferBytes: arenaCfg.window, timing: true,
    },
  });
  const x0 = randF32(G.dModel, 61_000, 0.5);
  // riferimento pieno: selezione + hidden del token PULITO
  const kitF = mkMiniModelKit();
  const refF0 = kitF.mkRef0(), refF1 = kitF.mkRef1();
  const refHF = refF1.forward(refF0.forward(x0));
  const refSel = refF1.lastRouting!;

  // ---- scenario 1: run PULITA — dirty presente, vuoto, zero falsi positivi ----
  const m1 = mkModel();
  m1.setTelemetry(true, false);
  const r1 = await m1.forward(x0, 0, true);
  const tel1 = await m1.telemetry();
  const planned1 = m1.dispatchesPerTokenPlanned;
  m1.destroy();
  if (!r1.dirty) problems.push("run pulita senza campo dirty (modo optimistic deve riportarlo sempre)");
  else if (r1.dirty.missCount !== 0 || r1.dirty.firstDirtyLayer !== -1 || r1.dirty.misses.length !== 0) {
    problems.push(`falsi positivi: missCount=${r1.dirty.missCount} first=${r1.dirty.firstDirtyLayer} misses=${r1.dirty.misses.length}`);
  }
  // identita' del token pulito col riferimento pieno (stessi gate del caso modello)
  let l2e = 0, l2r = 0;
  for (let i = 0; i < G.dModel; i++) { const d = r1.hidden[i] - refHF[i]; l2e += d * d; l2r += refHF[i] * refHF[i]; }
  const l2Clean = Math.sqrt(l2e / Math.max(l2r, 1e-30));
  if (l2Clean > 1e-3) problems.push(`run pulita: L2rel=${l2Clean.toExponential(2)} > 1e-3 vs ref pieno`);
  // l'interruttore vale anche qui: 1 submit / 0 sync per token
  if (tel1.submits !== 1 || tel1.routerSyncs !== 0) {
    problems.push(`contatori: ${tel1.submits} submit / ${tel1.routerSyncs} sync (attesi 1/0)`);
  }
  if (tel1.dispatches !== planned1 + 2) {
    problems.push(`dispatch ${tel1.dispatches} != piano ${planned1} + testa 2`);
  }

  // ---- scenario 2: MISS FORZATI su 2 dei 4 selezionati + 1 controllo non selezionato ----
  const selIds = Array.from(refSel.experts);
  const c1 = selIds[1], c2 = selIds[3];
  let ctrl = 0; while (selIds.includes(ctrl)) ctrl++; // il controllo NON deve comparire nei miss
  const m2 = mkModel();
  m2.setTelemetry(true, false);
  m2.debugMarkMiss([{ layer: 1, id: c1 }, { layer: 1, id: c2 }, { layer: 1, id: ctrl }]);
  const r2 = await m2.forward(x0, 0, true);
  m2.destroy();
  // selezione INVARIATA dal marking (il router non guarda la slotTable)
  const gotIds = Array.from(r2.routing[0].experts);
  if (!gotIds.every((e, k) => e === selIds[k])) {
    problems.push(`selezione cambiata dal marking: {${gotIds}} != {${selIds}}`);
  }
  // il flag: esattamente i due forzati, sui k giusti, controllo assente
  const wantMiss = [{ layer: 1, id: c1, k: 1 }, { layer: 1, id: c2, k: 3 }];
  const gotMiss = r2.dirty?.misses ?? [];
  const missOk = r2.dirty !== undefined && r2.dirty.missCount === 2 && r2.dirty.firstDirtyLayer === 1
    && gotMiss.length === 2 && wantMiss.every((w, i) =>
      gotMiss[i].layer === w.layer && gotMiss[i].id === w.id && gotMiss[i].k === w.k);
  if (!missOk) {
    problems.push(`miss attesi ${JSON.stringify(wantMiss)}, riportati ${JSON.stringify(gotMiss)} ` +
      `(missCount=${r2.dirty?.missCount}, first=${r2.dirty?.firstDirtyLayer})`);
  }
  // i flag nella Sel di produzione riletta: bit 0 esattamente dove deve
  const vram = r2.routing[0].vram;
  if (!vram) problems.push("Sel di produzione non riletta");
  else {
    for (let k = 0; k < G.nExpertUsed; k++) {
      const wantFlag = vram.experts[k] === c1 || vram.experts[k] === c2 ? 1 : 0;
      if ((vram.flags[k] & 1) !== wantFlag) {
        problems.push(`k${k}: flag ${vram.flags[k]} per expert ${vram.experts[k]} (atteso bit0=${wantFlag})`);
      }
    }
  }
  // il degrado DEFINITO: hidden e logits del token sporco == riferimento f64
  // con i DUE expert azzerati (contributo zero, guardia `ok` dei kernel)
  const kitZ = mkMiniModelKit();
  const refZ0 = kitZ.mkRef0(), refZ1 = kitZ.mkRef1(new Set([c1, c2]));
  const refHZ = refZ1.forward(refZ0.forward(x0));
  const refLZ = kitZ.headRefF64(refHZ);
  let l2ze = 0, l2zr = 0, logitZMaxRel = 0;
  for (let i = 0; i < G.dModel; i++) { const d = r2.hidden[i] - refHZ[i]; l2ze += d * d; l2zr += refHZ[i] * refHZ[i]; }
  const l2Dirty = Math.sqrt(l2ze / Math.max(l2zr, 1e-30));
  if (l2Dirty > 1e-3) problems.push(`token sporco: L2rel=${l2Dirty.toExponential(2)} > 1e-3 vs ref a expert azzerati`);
  let aGot = 0, aRef = 0;
  for (let r = 0; r < 2048; r++) {
    logitZMaxRel = Math.max(logitZMaxRel, Math.abs(r2.logits![r] - refLZ[r]) / Math.max(Math.abs(refLZ[r]), 1e-3));
    if (r2.logits![r] > r2.logits![aGot]) aGot = r;
    if (refLZ[r] > refLZ[aRef]) aRef = r;
  }
  if (aGot !== aRef || logitZMaxRel > 5e-3) {
    problems.push(`head del token sporco: argmax ${aGot} vs ${aRef}, logitMaxRel=${logitZMaxRel.toExponential(2)}`);
  }
  return {
    kernel: name, pass: problems.length === 0, maxAbs: 0, maxRel: Math.max(l2Clean, l2Dirty),
    note: `pulito: L2rel=${l2Clean.toExponential(2)}, 1 submit/0 sync, dispatch ok; ` +
      `forzati {${c1},${c2}} (+controllo ${ctrl}): missCount=${r2.dirty?.missCount}, ` +
      `first=${r2.dirty?.firstDirtyLayer}, degrado vs ref-azzerato L2rel=${l2Dirty.toExponential(2)}, ` +
      `logitMaxRel=${logitZMaxRel.toExponential(2)}` +
      (problems.length ? `; ${problems.join("; ")}` : ""),
  };
}

async function testGlmCheckpointHidden(g: Gpu): Promise<KResult> {
  const name = "glm-hidden-checkpoint";
  const problems: string[] = [];
  const arenaCfg = ktestArena(true);
  const mk = (select: "gpu" | "optimistic") => createGlmModel(g.device, mkMiniModelKit().srcMock, {
    nLayer: 2, ctxMax: 16, head: true, vocab: 2048, select, checkpointHidden: true,
    cache: {
      budgetBytes: 0, slotsOverride: arenaCfg.slotsOverride,
      maxBindingBytes: arenaCfg.window, maxBufferBytes: arenaCfg.window, timing: true,
    },
  });
  const mG = mk("gpu"), mO = mk("optimistic");
  mG.setTelemetry(true, false);
  const NPOS = 3;
  const xs = Array.from({ length: NPOS }, (_, p) => randF32(G.dModel, 62_000 + p, 0.5));
  for (let p = 0; p < NPOS; p++) {
    await mG.forward(xs[p], p, true);
    await mO.forward(xs[p], p, true);
  }
  const ckptG = await mG.readHiddenCkpt();
  const ckptO = await mO.readHiddenCkpt();
  const telG = await mG.telemetry();
  const plannedG = mG.dispatchesPerTokenPlanned;
  mG.destroy(); mO.destroy();
  if (ckptG.length !== 2 * G.dModel) problems.push(`checkpoint di ${ckptG.length} f32 (attesi ${2 * G.dModel})`);
  // (1) bit-identita' fra i due modi: stessa sequenza, stessi kernel, stesse copy
  let bitEq = 0;
  for (let i = 0; i < ckptG.length; i++) if (Object.is(ckptG[i], ckptO[i])) bitEq++;
  if (bitEq !== ckptG.length) problems.push(`gpu vs optimistic: ${bitEq}/${ckptG.length} bit-identici`);
  // (2) ancora di contenuto, riga 0: l'input del layer 0 all'ULTIMO token e'
  // xIn cosi' com'e' (writeBuffer di x, copy prima di ogni dispatch del token)
  let row0Eq = 0;
  const xLast = xs[NPOS - 1];
  for (let i = 0; i < G.dModel; i++) if (Object.is(ckptG[i], xLast[i])) row0Eq++;
  if (row0Eq !== G.dModel) problems.push(`riga 0: ${row0Eq}/${G.dModel} bit-identici a xIn dell'ultimo token`);
  // (3) ancora di contenuto, riga 1: l'input del layer 1 all'ultimo token vs
  // riferimento f64 (ref0 STATEFUL: si fanno passare le stesse 3 posizioni)
  const kit = mkMiniModelKit();
  const ref0 = kit.mkRef0(), ref1 = kit.mkRef1();
  let h1inLast: Float64Array | null = null;
  for (let p = 0; p < NPOS; p++) {
    const h1 = ref0.forward(xs[p]);
    ref1.forward(h1); // tiene coerente la KV del layer 1 (non serve l'uscita)
    if (p === NPOS - 1) h1inLast = h1;
  }
  let l2e = 0, l2r = 0;
  for (let i = 0; i < G.dModel; i++) {
    const d = ckptG[G.dModel + i] - h1inLast![i];
    l2e += d * d; l2r += h1inLast![i] * h1inLast![i];
  }
  const l2Row1 = Math.sqrt(l2e / Math.max(l2r, 1e-30));
  if (l2Row1 > 1e-3) problems.push(`riga 1: L2rel=${l2Row1.toExponential(2)} > 1e-3 vs input f64 del layer 1`);
  // (4) le copy non sono dispatch: il conteggio non si muove
  if (telG.dispatches / NPOS !== plannedG + 2) {
    problems.push(`dispatch/token ${telG.dispatches / NPOS} != piano ${plannedG} + testa 2 (le copy contano?)`);
  }
  return {
    kernel: name, pass: problems.length === 0, maxAbs: 0, maxRel: l2Row1,
    note: `${NPOS} pos: gpu==optimistic ${bitEq}/${ckptG.length} bit, riga0==xIn ${row0Eq}/${G.dModel} bit, ` +
      `riga1 vs f64 L2rel=${l2Row1.toExponential(2)}, dispatch/token invariati` +
      (problems.length ? `; ${problems.join("; ")}` : ""),
  };
}

// Il test d'IDENTITA' del contratto (fase 3): decode ottimistico vs sincrono
// sulla STESSA sequenza, argmax identico su tutte le posizioni — e qui il
// claim del GOAL e' piu' forte: "qualita' BIT-invariata", quindi il gate e'
// la bit-uguaglianza di hidden e logits, non solo l'argmax. Ai token forzati
// il replay parte dal primo layer sporco (startLayer > 0: rientro dal
// checkpoint esercitato davvero) e l'esito deve restare bit-identico.
async function testGlmOptimisticIdentity(g: Gpu): Promise<KResult> {
  const name = "glm-optimistic-identita";
  const problems: string[] = [];
  const arenaCfg = ktestArena(true);
  const mkOpts = (select: "gpu" | "optimistic") => ({
    nLayer: 2, ctxMax: 64, head: true, vocab: 2048, select,
    cache: {
      budgetBytes: 0, slotsOverride: arenaCfg.slotsOverride,
      maxBindingBytes: arenaCfg.window, maxBufferBytes: arenaCfg.window, timing: true,
    },
  });
  const mSync = createGlmModel(g.device, mkMiniModelKit().srcMock, mkOpts("gpu"));
  const mOpt = createGlmModel(g.device, mkMiniModelKit().srcMock, mkOpts("optimistic"));
  mOpt.setTelemetry(true, false);
  const NPOS = 64;                       // il [ASSUMED >= 64 token] del contratto
  const FORCE = new Set([5, 20, 40]);    // token con miss forzato ⇒ replay
  let bitH = 0, bitHTot = 0, bitL = 0, bitLTot = 0, argmaxEq = 0, replayed = 0;
  for (let p = 0; p < NPOS; p++) {
    const xp = randF32(G.dModel, 63_000 + p, 0.5);
    const rs = await mSync.forward(xp, p, true);
    if (FORCE.has(p)) {
      // si evincono 2 dei 4 che il SINCRONO ha appena selezionato: per identita'
      // l'ottimistico selezionera' gli stessi ⇒ miss garantito, replay garantito
      const ids = Array.from(rs.routing[0].experts);
      mOpt.debugMarkMiss([{ layer: 1, id: ids[0] }, { layer: 1, id: ids[2] }]);
    }
    const ro = await mOpt.forward(xp, p, true);
    if (FORCE.has(p)) {
      if (!ro.dirty || ro.dirty.missCount === 0 || ro.dirty.repaired !== true) {
        problems.push(`pos ${p}: replay atteso, dirty=${JSON.stringify(ro.dirty)}`);
      } else replayed++;
    } else if (ro.dirty && ro.dirty.missCount !== 0) {
      problems.push(`pos ${p}: miss inatteso (${ro.dirty.missCount})`);
    }
    // routing identico (id per-k, ogni layer MoE) — il replay ridecide uguale (I4)
    for (const [mi, r] of ro.routing.entries()) {
      const sExp = rs.routing[mi].experts;
      for (let k = 0; k < G.nExpertUsed; k++) {
        if (r.experts[k] !== sExp[k]) problems.push(`pos ${p} m${mi} k${k}: expert ${r.experts[k]} != ${sExp[k]}`);
      }
    }
    for (let i = 0; i < G.dModel; i++) { if (Object.is(ro.hidden[i], rs.hidden[i])) bitH++; }
    bitHTot += G.dModel;
    let aO = 0, aS = 0;
    for (let r = 0; r < 2048; r++) {
      if (Object.is(ro.logits![r], rs.logits![r])) bitL++;
      if (ro.logits![r] > ro.logits![aO]) aO = r;
      if (rs.logits![r] > rs.logits![aS]) aS = r;
    }
    bitLTot += 2048;
    if (aO === aS) argmaxEq++;
    else problems.push(`pos ${p}: argmax ${aO} != ${aS}`);
  }
  const tel = await mOpt.telemetry();
  mSync.destroy(); mOpt.destroy();
  // contabilita' del gate strutturale: 1 submit sui puliti, 2 sui rigiocati,
  // zero sync router SEMPRE; forwards conta i token, non i giri.
  if (tel.forwards !== NPOS || tel.dirtyTokens !== FORCE.size || tel.replays !== FORCE.size
    || tel.submits !== NPOS + FORCE.size || tel.routerSyncs !== 0) {
    problems.push(`contatori: forwards=${tel.forwards} dirty=${tel.dirtyTokens} replays=${tel.replays} ` +
      `submits=${tel.submits} (attesi ${NPOS}/${FORCE.size}/${FORCE.size}/${NPOS + FORCE.size}), sync=${tel.routerSyncs}`);
  }
  const pass = problems.length === 0 && bitH === bitHTot && bitL === bitLTot
    && argmaxEq === NPOS && replayed === FORCE.size;
  return {
    kernel: name, pass, maxAbs: 0, maxRel: 0,
    note: `${NPOS} pos (${FORCE.size} con replay): hidden bit ${bitH}/${bitHTot}, ` +
      `logits bit ${bitL}/${bitLTot}, argmax ${argmaxEq}/${NPOS}, ` +
      `submits ${tel.submits} = ${NPOS}+${tel.replays} replay, sync/token 0, repairMs=${tel.repairMs.toFixed(1)}` +
      (problems.length ? `; ${problems.slice(0, 6).join("; ")}` : ""),
  };
}

async function testGlmOptimisticPrecondition(g: Gpu): Promise<KResult> {
  const name = "glm-optimistic-precondizione";
  const srcNever: GlmWeightSource = {
    nonExpert: (n: string) => { throw new Error(`mock: precondizione tardiva, letto ${n}`); },
    expert: () => { throw new Error("mock: precondizione tardiva, letto un expert"); },
  };
  let msg = "";
  let modello: ReturnType<typeof createGlmModel> | null = null;
  try {
    modello = createGlmModel(g.device, srcNever, {
      nLayer: 2, ctxMax: 16, select: "optimistic",
      cache: {
        budgetBytes: 0, slotsOverride: KTEST_SLOTS, // 7 slot q4_1 per 64 expert = 11% < 88%
        maxBindingBytes: KTEST_ARENA_WINDOW, maxBufferBytes: KTEST_ARENA_WINDOW,
      },
    });
  } catch (e) {
    msg = e instanceof Error ? e.message : String(e);
  }
  modello?.destroy();
  // Precondizione emendata in it.4 (docket c3b item 6): sul TOTALE, soglia
  // 0.8 — il mini-modello ha parco 64 (solo q4_1) e 7 slot utili = 10.9%.
  const want = [
    "residenza NEAR-TOTAL",
    `${KTEST_SLOTS.q4_1} slot per ${G.nExpert} expert = ${(100 * KTEST_SLOTS.q4_1 / G.nExpert).toFixed(1)}% < 0.8`,
    "C3c",
  ];
  const mancanti = want.filter((w) => !msg.includes(w));
  const pass = modello === null && mancanti.length === 0;
  return {
    kernel: name, pass, maxAbs: 0, maxRel: 0,
    note: modello === null
      ? `errore atteso: "${msg.slice(0, 160)}…"` + (mancanti.length ? `; MANCANO ${mancanti.join(" | ")}` : "")
      : "il modo optimistic e' partito sotto la precondizione (7 slot per 64 expert)",
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

// Sweep delle varianti batch MECCANICHE dei kernel densi (fase 5, it.30):
// ognuna vs il per-riga sullo stesso dato — stesso corpo, cambia solo
// l'indicizzazione di riga ⇒ atteso BIT-IDENTICO per tutte, senza fallback.
async function testDenseBatchSweep(g: Gpu): Promise<KResult[]> {
  const M = 3;
  const out: KResult[] = [];
  const bitCmp = (name: string, got: Float32Array, refs: Float32Array[], perLen: number): KResult => {
    let maxAbs = 0, ok = true;
    for (let m = 0; m < M; m++) {
      for (let i = 0; i < perLen; i++) {
        const a = got[m * perLen + i], b = refs[m][i];
        if (!Object.is(a, b)) ok = false;
        maxAbs = Math.max(maxAbs, Math.abs(a - b));
      }
    }
    return { kernel: name, pass: ok, maxAbs, maxRel: maxAbs, note: ok ? `BIT-IDENTICO su ${M} righe` : "divergenza inattesa" };
  };
  const rowsBuf = (rows: Float32Array[]): GPUBuffer => {
    const b = g.empty(M * rows[0].length * 4);
    rows.forEach((r, m) => g.device.queue.writeBuffer(b, m * r.length * 4, r as unknown as BufferSource));
    return b;
  };

  { // gemvQuant nei tre kind (QA/QB/O = q4_0, KvA = q8_0, dense down = q4_1)
    for (const kind of ["q4_0", "q4_1", "q8_0"] as const) {
      const K = 256, N = 128;
      const blockBytes = kind === "q4_0" ? 18 : kind === "q4_1" ? Q4_1_BLOCK_BYTES : 34;
      const nBlocks = (K / 32) * N;
      const src = randBytes(nBlocks * blockBytes, 30_100 + blockBytes);
      if (kind === "q4_0") fixScales(src, blockBytes);
      else if (kind === "q4_1") fixScalesAt(src, blockBytes, [1, 3]);
      else fixScales(src, blockBytes);
      const rp = (kind === "q4_0" ? repackQ4_0 : kind === "q4_1" ? repackQ4_1 : repackQ8_0)(src, 0, nBlocks);
      const qsB = g.buf(rp.qs), scB = g.buf(rp.scales);
      const xs = Array.from({ length: M }, (_, m) => randF32(K, 30_110 + m, 0.5));
      const refs: Float32Array[] = [];
      for (let m = 0; m < M; m++) {
        const y = g.empty(N * 4);
        await g.run(gemvQuantWgsl({ kind, K, N, hasBias: false }), [qsB, scB, g.buf(xs[m]), y], gemvGrid(N)[0]);
        refs.push(new Float32Array(await g.read(y, N * 4)));
      }
      const yM = g.empty(M * N * 4);
      await g.run(gemvQuantWgsl({ kind, K, N, hasBias: false, batch: true }),
        [qsB, scB, rowsBuf(xs), yM], [gemvGrid(N)[0], gemvGrid(N)[1], M]);
      out.push(bitCmp(`dense-batch-gemv-${kind}`, new Float32Array(await g.read(yM, M * N * 4)), refs, N));
    }
  }
  { // gemvF32 (router GEMM): l'ultimo generatore del corredo
    const K = 128, N = 64;
    const wN = randF32(K * N, 30_700, 0.5);
    const wB = g.buf(wN);
    const xs = Array.from({ length: M }, (_, m) => randF32(K, 30_710 + m, 0.5));
    const refs: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const y = g.empty(N * 4);
      await g.run(gemvF32Wgsl({ K, N }), [wB, g.buf(xs[m]), y], N);
      refs.push(new Float32Array(await g.read(y, N * 4)));
    }
    const yM = g.empty(M * N * 4);
    await g.run(gemvF32Wgsl({ K, N, batch: true }), [wB, rowsBuf(xs), yM], [N, 1, M]);
    out.push(bitCmp('dense-batch-gemv-f32', new Float32Array(await g.read(yM, M * N * 4)), refs, N));
  }
  { // gemvQ8Heads (absorbKb/voutVb): x per head DENTRO la riga
    const K = 64, rowsPerHead = 8, nHead = 4, xStride = 80, xOffset = 16;
    const N = rowsPerHead * nHead;
    const nBlocks = (K / 32) * N;
    const src = randBytes(nBlocks * 34, 30_200); fixScales(src, 34);
    const rp = repackQ8_0(src, 0, nBlocks);
    const qsB = g.buf(rp.qs), scB = g.buf(rp.scales);
    const xs = Array.from({ length: M }, (_, m) => randF32(nHead * xStride, 30_210 + m, 0.5));
    const refs: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const y = g.empty(N * 4);
      await g.run(gemvQ8HeadsWgsl({ K, rowsPerHead, nHead, xStride, xOffset }), [qsB, scB, g.buf(xs[m]), y], N);
      refs.push(new Float32Array(await g.read(y, N * 4)));
    }
    const yM = g.empty(M * N * 4);
    await g.run(gemvQ8HeadsWgsl({ K, rowsPerHead, nHead, xStride, xOffset, batch: true }),
      [qsB, scB, rowsBuf(xs), yM], [N, 1, M]);
    out.push(bitCmp("dense-batch-gemv-q8heads", new Float32Array(await g.read(yM, M * N * 4)), refs, N));
  }
  { // rmsnorm (rmsD/rmsQA/rmsKvA)
    const D = 512;
    const wN = randF32(D, 30_300, 1);
    const wB = g.buf(wN);
    const xs = Array.from({ length: M }, (_, m) => randF32(D, 30_310 + m, 0.5));
    const refs: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const o = g.empty(D * 4);
      await g.run(rmsnormWgsl(D, 1e-5), [g.buf(xs[m]), wB, o], 1);
      refs.push(new Float32Array(await g.read(o, D * 4)));
    }
    const oM = g.empty(M * D * 4);
    await g.run(rmsnormWgsl(D, 1e-5, true), [rowsBuf(xs), wB, oM], M);
    out.push(bitCmp("dense-batch-rmsnorm", new Float32Array(await g.read(oM, M * D * 4)), refs, D));
  }
  { // ropeMlaNorm (ropeQ/ropeKPe): posizioni per riga CRESCENTI
    const nVec = 4, stride = 80, offset = 16, ropeDims = 32;
    const len = nVec * stride;
    const basePos = 37;
    const vs = Array.from({ length: M }, (_, m) => randF32(len, 30_410 + m, 0.5));
    const refs: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const v = g.buf(vs[m]);
      await g.run(ropeMlaNormWgsl({ nVec, stride, offset, ropeDims, freqBase: 10000 }),
        [v], Math.ceil((nVec * ropeDims / 2) / 64), g.uniform(basePos + m, basePos + m));
      refs.push(new Float32Array(await g.read(v, len * 4)));
    }
    const vM = rowsBuf(vs);
    const rowPos = g.buf(Uint32Array.from({ length: M }, (_, m) => basePos + m));
    await g.run(ropeMlaNormWgsl({ nVec, stride, offset, ropeDims, freqBase: 10000, batch: true }),
      [vM, rowPos], [Math.ceil((nVec * ropeDims / 2) / 64), M, 1]);
    out.push(bitCmp("dense-batch-rope-mla", new Float32Array(await g.read(vM, M * len * 4)), refs, len));
  }
  { // kvAppend: M righe a posizioni consecutive nella STESSA cache
    const W = 96, ctx = 32, basePos = 7;
    const rows = Array.from({ length: M }, (_, m) => randF32(W, 30_510 + m, 0.5));
    const cacheRef = g.buf(randF32(ctx * W, 30_500, 0.5));
    const pre = new Float32Array(await g.read(cacheRef, ctx * W * 4));
    for (let m = 0; m < M; m++) {
      await g.run(kvAppendWgsl(W), [g.buf(rows[m]), cacheRef], Math.ceil(W / 64), g.uniform(basePos + m, basePos + m));
    }
    const refAll = new Float32Array(await g.read(cacheRef, ctx * W * 4));
    const cacheB = g.buf(pre);
    const rowPos = g.buf(Uint32Array.from({ length: M }, (_, m) => basePos + m));
    await g.run(kvAppendWgsl(W, true), [rowsBuf(rows), cacheB, rowPos], [Math.ceil(W / 64), M, 1]);
    const gotAll = new Float32Array(await g.read(cacheB, ctx * W * 4));
    let ok = true, maxAbs = 0;
    for (let i = 0; i < ctx * W; i++) {
      if (!Object.is(gotAll[i], refAll[i])) ok = false;
      maxAbs = Math.max(maxAbs, Math.abs(gotAll[i] - refAll[i]));
    }
    out.push({ kernel: "dense-batch-kvappend", pass: ok, maxAbs, maxRel: maxAbs, note: ok ? `BIT-IDENTICA l'intera cache (righe ${basePos}..${basePos + M - 1} scritte, il resto intatto)` : "divergenza inattesa" });
  }
  { // stridedCopy (copyCkv/copyQRope): dst pre-riempito, solo la finestra si muove
    const nVec = 4, len = 16, srcStride = 80, srcOffset = 48, dstStride = 96, dstOffset = 64;
    const sLen = nVec * srcStride, dLen = nVec * dstStride;
    const srcs = Array.from({ length: M }, (_, m) => randF32(sLen, 30_610 + m, 0.5));
    const dst0 = Array.from({ length: M }, (_, m) => randF32(dLen, 30_620 + m, 0.5));
    const refs: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const d = g.buf(dst0[m]);
      await g.run(stridedCopyWgsl({ nVec, len, srcStride, srcOffset, dstStride, dstOffset }),
        [g.buf(srcs[m]), d], Math.ceil((nVec * len) / 64));
      refs.push(new Float32Array(await g.read(d, dLen * 4)));
    }
    const dM = rowsBuf(dst0);
    await g.run(stridedCopyWgsl({ nVec, len, srcStride, srcOffset, dstStride, dstOffset, batch: true }),
      [rowsBuf(srcs), dM], [Math.ceil((nVec * len) / 64), M, 1]);
    out.push(bitCmp("dense-batch-strided-copy", new Float32Array(await g.read(dM, M * dLen * 4)), refs, dLen));
  }
  // ---- kernel a M righe che mancavano al path q35 (fase 4, it.25) ----
  { // ropeNeox: posizioni per riga CRESCENTI (e' il caso del chunk di prefill)
    const nHead = 3, headDim = 32, ropeDims = 16, base = 10000;
    const len = nHead * headDim;
    const basePos = 41;
    const xs = Array.from({ length: M }, (_, m) => randF32(len, 31_100 + m, 0.7));
    const refs: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const v = g.buf(xs[m]);
      const uni = g.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      g.device.queue.writeBuffer(uni, 0, new Uint32Array([basePos + m, basePos + m, 0, 0]) as unknown as BufferSource);
      await g.run(ropeNeoxWgsl(nHead, headDim, base, ropeDims), [v, uni], Math.ceil((nHead * ropeDims / 2) / 64));
      refs.push(new Float32Array(await g.read(v, len * 4)));
    }
    const vM = rowsBuf(xs);
    const posB = g.empty(M * 4);
    g.device.queue.writeBuffer(posB, 0, Uint32Array.from({ length: M }, (_, m) => basePos + m) as unknown as BufferSource);
    await g.run(ropeNeoxWgsl(nHead, headDim, base, ropeDims, true), [vM, posB], [Math.ceil((nHead * ropeDims / 2) / 64), M, 1]);
    out.push(bitCmp("dense-batch-rope-neox", new Float32Array(await g.read(vM, M * len * 4)), refs, len));
  }
  { // elementwise: siluMul, sigmoidMul, addInPlace
    const D = 96;
    const mk = (seed: number) => Array.from({ length: M }, (_, m) => randF32(D, seed + m, 0.9));
    for (const which of ["siluMul", "sigmoidMul", "addInPlace"] as const) {
      const as = mk(31_200), bs = mk(31_300);
      const code = (batch?: boolean): string => which === "siluMul"
        ? siluMulWgsl(D, batch) : which === "sigmoidMul" ? sigmoidMulWgsl(D, batch) : addInPlaceWgsl(D, batch);
      const refs: Float32Array[] = [];
      for (let m = 0; m < M; m++) {
        const a = g.buf(as[m]);
        await g.run(code(), [a, g.buf(bs[m])], Math.ceil(D / 64));
        refs.push(new Float32Array(await g.read(a, D * 4)));
      }
      const aM = rowsBuf(as), bM = rowsBuf(bs);
      await g.run(code(true), [aM, bM], [Math.ceil(D / 64), M, 1]);
      out.push(bitCmp(`dense-batch-${which}`, new Float32Array(await g.read(aM, M * D * 4)), refs, D));
    }
  }
  { // deltaNetGates: row-parallel (conv e core NO: sono la ricorrenza)
    const nV = 40;
    const betas = Array.from({ length: M }, (_, m) => randF32(nV, 31_400 + m, 1.2));
    const alphas = Array.from({ length: M }, (_, m) => randF32(nV, 31_450 + m, 1.2));
    const aCoef = g.buf(randF32(nV, 31_500, 0.5));
    const dtB = g.buf(randF32(nV, 31_510, 0.3));
    const refs: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const bo = g.empty(nV * 4), go = g.empty(nV * 4);
      await g.run(deltaNetGatesWgsl(nV), [g.buf(betas[m]), g.buf(alphas[m]), aCoef, dtB, bo, go], Math.ceil(nV / 64));
      const b0 = new Float32Array(await g.read(bo, nV * 4));
      const g0 = new Float32Array(await g.read(go, nV * 4));
      const cat = new Float32Array(2 * nV);
      cat.set(b0, 0); cat.set(g0, nV);
      refs.push(cat);
    }
    const boM = g.empty(M * nV * 4), goM = g.empty(M * nV * 4);
    await g.run(deltaNetGatesWgsl(nV, true), [rowsBuf(betas), rowsBuf(alphas), aCoef, dtB, boM, goM], [Math.ceil(nV / 64), M, 1]);
    const bAll = new Float32Array(await g.read(boM, M * nV * 4));
    const gAll = new Float32Array(await g.read(goM, M * nV * 4));
    const got = new Float32Array(M * 2 * nV);
    for (let m = 0; m < M; m++) {
      got.set(bAll.subarray(m * nV, (m + 1) * nV), m * 2 * nV);
      got.set(gAll.subarray(m * nV, (m + 1) * nV), m * 2 * nV + nV);
    }
    out.push(bitCmp("dense-batch-deltanet-gates", got, refs, 2 * nV));
  }
  { // attnDecode a chunk (fase 4, it.26): righe con nPast CRESCENTE, cioe' il
    // caso vero del prefill — la riga m vede se stessa e le righe precedenti
    // dello stesso chunk, e NON quelle dopo. La cache si riempie PRIMA, come
    // nell'encoder.
    //
    // NON PIU' BIT A BIT, E LA TOLLERANZA E' DICHIARATA PRIMA (riga 3 (d) di
    // PHASES del goal engine-ttft). Il riferimento resta il template LEGACY
    // per-riga (`attnDecodeRefWgsl`), che e' l'unico riferimento indipendente
    // che questo banco abbia; ma il ramo `batch` non ne e' piu' il gemello
    // strutturale. Dal task T1-kernel-batch-streaming e' softmax in STREAMING:
    // il massimo e la somma si riscalano online a ogni tile di 64 posizioni, e
    // il prodotto scalare passa da 32 somme scalari a 8 `dot()` su vec4. Sono
    // due riassociazioni della stessa somma, quindi la differenza dai bit del
    // legacy NON e' un errore: e' la definizione della forma nuova. Pretendere
    // `Object.is` qui sarebbe un gate rosso a prescindere dalla correttezza.
    //
    // La banda NON e' un numero scelto a occhio: i due kernel sono stati
    // simulati in f32 fedele (ogni operazione via `Math.fround`, 64 thread in
    // lock-step) sugli STESSI due casi qui sotto, e la divergenza misurata e'
    //   un tile   (n = 10..12):  max assoluto 2,98e-8, max relativo 4,26e-6
    //   cinque tile (n = 301..303): max assoluto 1,68e-8, max relativo 3,95e-5
    // La banda relativa sta 2,5x sopra il caso peggiore misurato, quella
    // assoluta ~600x: stretta abbastanza da vedere un indice sbagliato o un tile
    // perso, larga abbastanza da assorbire la fma della GPU vera (la simulazione
    // e' su CPU e non la modella). `compare` passa se UNA delle due bande regge,
    // e quella assoluta e' li' per le componenti vicine a zero, dove il relativo
    // esplode senza significare niente.
    const ATTN_CHUNK_REL_TOL = 1e-4, ATTN_CHUNK_ABS_TOL = 1e-5;
    // DUE casi, e il secondo non e' un lusso. Il tile e' da 64 posizioni: col
    // solo caso corto (n = 10..12) il ciclo fa UN giro, `rs` vale 0 per
    // costruzione (il massimo parte da -3e38) e il rescale online fra tile — che
    // e' TUTTO il codice nuovo della riscrittura — non verrebbe eseguito mai. Il
    // primo comportamento non banale compare a n >= 65.
    const CASES = [
      { ctxMax: 64, basePast: 9, tag: "" },              // n = 10..12: UN tile
      { ctxMax: 320, basePast: 300, tag: "-multitile" }, // n = 301..303: 5 tile
    ];
    const nHead = 4, nKvHead = 2, headDim = 32;
    const kvDim = nKvHead * headDim, qDim = nHead * headDim;
    for (const C of CASES) {
      const { ctxMax, basePast } = C; // la riga m guarda 0..basePast+m
      const kC = g.buf(randF32(ctxMax * kvDim, 31_600, 0.4));
      const vC = g.buf(randF32(ctxMax * kvDim, 31_610, 0.4));
      const qs = Array.from({ length: M }, (_, m) => randF32(qDim, 31_620 + m, 0.6));
      const ref = new Float32Array(M * qDim);
      for (let m = 0; m < M; m++) {
        const o = g.empty(qDim * 4);
        const uni = g.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        g.device.queue.writeBuffer(uni, 0, new Uint32Array([basePast + m, basePast + m, 0, 0]) as unknown as BufferSource);
        await g.run(attnDecodeRefWgsl({ nHead, nKvHead, headDim, ctxMax }), [g.buf(qs[m]), kC, vC, o, uni], nHead);
        ref.set(new Float32Array(await g.read(o, qDim * 4)), m * qDim);
      }
      const oM = g.empty(M * qDim * 4);
      const pastB = g.empty(M * 4);
      g.device.queue.writeBuffer(pastB, 0, Uint32Array.from({ length: M }, (_, m) => basePast + m) as unknown as BufferSource);
      await g.run(attnDecodeWgsl({ nHead, nKvHead, headDim, ctxMax, batch: true }),
        [rowsBuf(qs), kC, vC, oM, pastB], [nHead, M, 1]);
      const got = new Float32Array(await g.read(oM, M * qDim * 4));
      const r = compare(`dense-batch-attn-chunk${C.tag}`, got, ref, ATTN_CHUNK_REL_TOL, ATTN_CHUNK_ABS_TOL);
      const tiles = Math.ceil((basePast + M) / 64);
      out.push({
        ...r,
        note: `streaming vs riferimento per-riga legacy su ${M} righe, n = ${basePast + 1}..${basePast + M} => ${tiles} tile da 64 (rescale online ${tiles > 1 ? "ATTRAVERSATO" : "non attraversato"}). NON bit-identico per costruzione; banda dichiarata rel ${ATTN_CHUNK_REL_TOL} / abs ${ATTN_CHUNK_ABS_TOL}`,
      });
    }
  }
  { // deltanet conv+core con la RIGA da uniform (fase 4, it.30): la ricorrenza
    // non si batcha, quindi restano M dispatch IN ORDINE — qui si prova che
    // indicizzare la riga da uniform da' gli stessi bit del kernel per-riga
    // eseguito M volte, STATO COMPRESO (la riga m legge cio' che ha lasciato
    // la m−1: se l'indicizzazione fosse sbagliata, la catena divergerebbe al
    // secondo passo).
    const hd = 16, nK = 2, nV = 4, convK = 4, eps = 1e-6;
    const qkvDim = (2 * nK + nV) * hd;
    const cvLen = 2 * nK * hd + nV * hd;
    const xs = Array.from({ length: M }, (_, m) => randF32(qkvDim, 31_700 + m, 0.5));
    const wConv = g.buf(randF32(qkvDim * convK, 31_710, 0.3));
    const betas = Array.from({ length: M }, (_, m) => randF32(nV, 31_720 + m, 0.8));
    const gs = Array.from({ length: M }, (_, m) => randF32(nV, 31_730 + m, -0.2));
    const zs = Array.from({ length: M }, (_, m) => randF32(nV * hd, 31_740 + m, 0.5));
    const normW = g.buf(randF32(hd, 31_750, 1));
    const st0 = randF32((convK - 1) * qkvDim, 31_760, 0.2);
    const S0 = randF32(nV * hd * hd, 31_770, 0.1);
    // riferimento: kernel per-riga, M volte, sullo STESSO stato che evolve
    const stR = g.buf(st0), SR = g.buf(S0);
    const refs: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const cv = g.empty(cvLen * 4), ov = g.empty(nV * hd * 4);
      await g.run(deltaNetConvWgsl(qkvDim, convK), [stR, g.buf(xs[m]), wConv, cv], Math.ceil(qkvDim / 64));
      await g.run(deltaNetCoreWgsl({ hd, nK, nV, eps }),
        [cv, SR, g.buf(betas[m]), g.buf(gs[m]), g.buf(zs[m]), normW, ov], nV);
      refs.push(new Float32Array(await g.read(ov, nV * hd * 4)));
    }
    // batch: stessi input a righe, riga da uniform, stato che evolve UNA volta
    const stB = g.buf(st0), SB = g.buf(S0);
    const cvM = g.empty(M * cvLen * 4), ovM = g.empty(M * nV * hd * 4);
    const rowUni = Array.from({ length: M }, (_, m) => {
      const b = g.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      g.device.queue.writeBuffer(b, 0, new Uint32Array([m, 0, 0, 0]) as unknown as BufferSource);
      return b;
    });
    const xM = rowsBuf(xs), bM = rowsBuf(betas), gM = rowsBuf(gs), zM = rowsBuf(zs);
    for (let m = 0; m < M; m++) {
      await g.run(deltaNetConvWgsl(qkvDim, convK, true),
        [stB, xM, wConv, cvM], Math.ceil(qkvDim / 64), rowUni[m]);
      await g.run(deltaNetCoreWgsl({ hd, nK, nV, eps }, true),
        [cvM, SB, bM, gM, zM, normW, ovM], nV, rowUni[m]);
    }
    out.push(bitCmp("dense-rows-deltanet-recurrence",
      new Float32Array(await g.read(ovM, M * nV * hd * 4)), refs, nV * hd));
  }
  { // APPIATTIMENTO (fase 4, it.31): i kernel PER-VETTORE (rmsnorm per head,
    // stridedCopy per head) non hanno bisogno di un modo "M righe". I buffer
    // sono row-major con passo di riga = nVec*len, quindi il vettore (riga,
    // head) sta all'indice riga*nVec + head: basta dispatchare nVec*M vettori
    // invece di nVec. Qui si prova che l'identita' regge alla geometria VERA
    // di q35 (rmsnorm per head su qB, stridedCopy che estrae q e gate da qFull).
    const nHead = 4, headDim = 32;
    const qDim = nHead * headDim;
    const wN = g.buf(randF32(headDim, 31_800, 1));
    const qs = Array.from({ length: M }, (_, m) => randF32(qDim, 31_810 + m, 0.7));
    const refsN: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const o = g.empty(qDim * 4);
      await g.run(rmsnormWgsl(headDim, 1e-6, true), [g.buf(qs[m]), wN, o], nHead);
      refsN.push(new Float32Array(await g.read(o, qDim * 4)));
    }
    const oN = g.empty(M * qDim * 4);
    await g.run(rmsnormWgsl(headDim, 1e-6, true), [rowsBuf(qs), wN, oN], nHead * M);
    out.push(bitCmp("dense-flat-rmsnorm-per-head", new Float32Array(await g.read(oN, M * qDim * 4)), refsN, qDim));

    // stridedCopy: da qFull [2*qDim per riga] estrae q (offset 0) e gate (hd)
    const qFulls = Array.from({ length: M }, (_, m) => randF32(2 * qDim, 31_850 + m, 0.6));
    const sc = (nVec: number) => stridedCopyWgsl({
      nVec, len: headDim, srcStride: 2 * headDim, srcOffset: headDim, dstStride: headDim, dstOffset: 0,
    });
    const refsC: Float32Array[] = [];
    for (let m = 0; m < M; m++) {
      const o = g.empty(qDim * 4);
      await g.run(sc(nHead), [g.buf(qFulls[m]), o], Math.ceil(qDim / 64));
      refsC.push(new Float32Array(await g.read(o, qDim * 4)));
    }
    const oC = g.empty(M * qDim * 4);
    await g.run(sc(nHead * M), [rowsBuf(qFulls), oC], Math.ceil((M * qDim) / 64));
    out.push(bitCmp("dense-flat-stridedcopy", new Float32Array(await g.read(oC, M * qDim * 4)), refsC, qDim));
  }
  return out;
}

// ============ PREFILL BATCHED M>1 (C3a fase 5, it.27) ============
// Il gate della fase ridotto ai kernel: la catena batched (gather su unione +
// slot y[m][k] + combine k-order) contro la catena DECODE vera (pairGemvSilu +
// gemvAccum in ordine k + addInPlace), stesse selezioni, stessi pesi sintetici
// a geometria reale. L'atteso e' l'identita' BIT-A-BIT (il piano e' costruito
// per questo — moeprefillplan); fallback dichiarato come R2: se la contrazione
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
    const ref = routerSelect(logits, bias, ROUTER_GLM47);

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


// Router+resolve QWEN su GPU vs `routerSelect` (CPU, f64) — fase D fase 3b,
// fetta 2. Stessa metodologia del caso GLM qui sopra, con due differenze che
// contano:
//  - gating SOFTMAX senza bias (`ROUTER_QWEN35MOE`), 256 expert, top-8;
//  - la selezione softmax e' MONOTONA nei logit, quindi la separazione che
//    decide se f32 puo' sbagliare e' quella fra l'8o e il 9o LOGIT, non fra
//    due punteggi derivati. Il kernel pero' confronta le probs, e li' l'exp
//    in f32 puo' ancora invertire due valori vicinissimi: il gate resta a
//    separazione, come su GLM.
// La coda di RESOLVE si prova nello stesso caso: una slotTable finta mappa i
// selezionati su slot noti e uno di loro su MISS, e si pretende che `Sel`
// riporti slot e flag esatti — e' l'unico punto in cui l'indirizzo dell'expert
// smette di passare dalla CPU.
async function testRouterQwenGpuVsCpu(g: Gpu, draws: number): Promise<KResult> {
  const NE = 256, NU = 8;
  const code = routerTopKWgsl({
    nExpert: NE, nUsed: NU, weightsScale: 1, clampMin: WEIGHTS_SUM_CLAMP_MIN,
    gating: "softmax", resolve: { nExpert: NE, nUsed: NU },
  });
  const cfg = { ...ROUTER_QWEN35MOE, nUsed: NU };
  const zeros = new Float32Array(NE); // bias assente: `probs + 0.0` e' esatto
  let maxRelW = 0, maxAbsW = 0, setFlips = 0, orderFlips = 0, belowGate = 0;
  let minSepHeld = Infinity, maxSepFlipped = 0, resolveBad = 0;

  for (let dr = 0; dr < draws; dr++) {
    const logits = randF32(NE, 7717 + dr * 11, 4);
    const ref = routerSelect(logits, null, cfg);
    // separazione fra ultimo selezionato e primo escluso, in probabilita' f64
    const mx = Math.max(...logits);
    const ex = Array.from(logits, (v) => Math.exp(v - mx));
    const zs = ex.reduce((a, b) => a + b, 0);
    const probs = ex.map((v) => v / zs);
    const ranked = probs.slice().sort((a, b) => b - a);
    const sep = ranked[NU - 1] - ranked[NU];

    // slotTable finta: i selezionati su slot noti, uno su MISS
    const table = new Uint32Array(NE).fill(0xffffffff);
    const attesoSlot = new Map<number, number>();
    Array.from(ref.experts).forEach((e, k) => {
      if (k === 3) return;               // il quarto resta MISS di proposito
      table[e] = 100 + k;
      attesoSlot.set(e, 100 + k);
    });

    const lb = g.buf(logits), bb = g.buf(zeros);
    const idsB = g.empty(NU * 4), wtsB = g.empty(NU * 4);
    const selB = g.empty(NU * 16), tabB = g.buf(table);
    const idxB = g.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    g.device.queue.writeBuffer(idxB, 0, new Uint32Array([0, 0, 0, 0]));
    await g.run(code, [lb, bb, idsB, wtsB, selB, tabB, idxB], 1);
    const ids = new Uint32Array(await g.read(idsB, NU * 4));
    const wts = new Float32Array(await g.read(wtsB, NU * 4));
    const selRaw = await g.read(selB, NU * 16);
    const selU = new Uint32Array(selRaw), selF = new Float32Array(selRaw);
    for (const b of [lb, bb, idsB, wtsB, selB, tabB, idxB]) b.destroy();

    const sameSet = new Set(ids).size === NU
      && Array.from(ids).every((e) => Array.from(ref.experts).includes(e));
    const sameOrder = Array.from(ids).every((e, k) => e === ref.experts[k]);
    if (sep < ROUTER_SEP_GATE) belowGate++;
    if (!sameSet) { setFlips += sep >= ROUTER_SEP_GATE ? 1 : 0; maxSepFlipped = Math.max(maxSepFlipped, sep); }
    else { minSepHeld = Math.min(minSepHeld, sep); }
    if (!sameOrder) orderFlips++;

    // RESOLVE: slot e flag devono essere ESATTI (sono interi, non numerica)
    for (let k = 0; k < NU; k++) {
      const e = selU[k * 4];
      const atteso = attesoSlot.get(e) ?? 0xffffffff;
      const flagAtteso = atteso === 0xffffffff ? 1 : 0;
      if (selU[k * 4 + 1] !== atteso || selU[k * 4 + 3] !== flagAtteso || e !== ids[k]) resolveBad++;
    }

    if (sameSet) {
      for (let k = 0; k < NU; k++) {
        const want = ref.weights[Array.from(ref.experts).indexOf(ids[k])];
        const d1 = Math.abs(wts[k] - want);
        maxAbsW = Math.max(maxAbsW, d1);
        maxRelW = Math.max(maxRelW, d1 / Math.max(Math.abs(want), 1e-6));
        // il peso deve arrivare anche dentro Sel, dove lo leggono i kernel
        if (Math.abs(selF[k * 4 + 2] - wts[k]) > 0) resolveBad++;
      }
    }
  }
  const note = `softmax 256x8, draws=${draws} setFlips(sep>=${ROUTER_SEP_GATE})=${setFlips} `
    + `orderFlips=${orderFlips} sottoSoglia=${belowGate} resolveErrati=${resolveBad} `
    + `minSepRetto=${minSepHeld === Infinity ? "-" : minSepHeld.toExponential(2)}`
    + (maxSepFlipped > 0 ? ` maxSepFlippato=${maxSepFlipped.toExponential(2)}` : "");
  return {
    kernel: "q35-router-resolve-gpu-vs-cpu",
    pass: setFlips === 0 && resolveBad === 0 && maxRelW <= 1e-5,
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

      const ref = routerSelect(logits, bias, ROUTER_GLM47);
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
  const ref = routerSelect(logits, bias, ROUTER_GLM47);
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
    results.push(await testGemvC2(g, "q4_K", 2048, 512));               // expert 35B gate/up (q1 fase 7)
    results.push(await testGemvC2(g, "q4_K", 512, 2048));               // expert 35B down (q1 fase 7)
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
    results.push(...await testDenseBatchSweep(g));

    // --- router top-4 su GPU (C3a fase 4 strato 1): fedelta' f32 vs CPU f64 ---
    results.push(await testRouterTopK(g, 64));
    results.push(await testRouterQwenGpuVsCpu(g, 64));
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

    // --- DeltaNet q35 (q1 fase 3): conv, gates, core, catena sul campione ---
    results.push(await testDeltaNetConv(g, 256));   // dims campione
    results.push(await testDeltaNetConv(g, 8192));  // dims reali 4B
    results.push(await testDeltaNetGates(g));
    results.push(await testDeltaNetCore(g, 16, 4, 8));    // dims campione
    results.push(await testDeltaNetCore(g, 128, 16, 32)); // dims reali 4B/9B/35B
    results.push(await testDeltaNetChain(g));

    // --- q35 fase 4 slice 2: micro-kernel + assembly layer con pesi REALI 4B ---
    results.push(await testRopePartial(g));
    results.push(await testSigmoidMul(g));
    results.push(...await testQ35AttnLayersReal(g));
    results.push(await testQ35MtpHeadReal(g)); // testa MTP: blocco su GPU == cpuref (fase 7 it.52)
    results.push(...await testQ35MtpDraft4B(g)); // testa MTP nel modello vero: accept-rate + gate secco (fase 7 it.53-54)
    results.push(await testQ35Model4B(g)); // full-model: argmax GPU == oracolo (slice 3)
    results.push(...await testQ35MoeBlockReal(g)); // MoE 35B block reale (fase 7 slice 3a)

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
      const aPart = q35AttnPartialsFloats({ nHead: S.nHead, headDim: HD, ctxMax });
      const partOut = g.empty(aPart.out * 4), partMS = g.empty(aPart.ms * 4);
      await g.run(attnDecodeWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: HD, ctxMax }),
        [g.buf(q), kBuf, vBuf, partOut, partMS], [S.nHead, q35AttnSplitPlan(ctxMax).splits, 1], u);
      await g.run(attnDecodeCombineWgsl({ nHead: S.nHead, headDim: HD, ctxMax }), [partOut, partMS, out], S.nHead);
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

    // decode ottimistico (C3b fase 2-3): flag di miss, degrado definito,
    // checkpoint, identita' repair+replay, precondizione
    results.push(await testGlmOptimisticForcedMiss(g));
    results.push(await testGlmCheckpointHidden(g));
    results.push(await testGlmOptimisticIdentity(g));
    results.push(await testGlmOptimisticPrecondition(g));

    // conformance layer-level con pesi reali (fase 4 slice 3) — il test lungo in coda
    results.push(await testGlmLayer0Real(g));
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e), results });
    return;
  }
  post({ type: "done", results });
}

void main();
