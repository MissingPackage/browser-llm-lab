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
} from "../kernels/wgsl";
import { createGlmLayer0 } from "../glmforward";
import { GlmDenseLayerRefF64, glmMoeFfnRefF64, type GlmMoeExpertWeights } from "../cpuref";
import { routerSelect, packExpertSlab, SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "../moe";
import { ExpertOpfsStore } from "../expertstore";
import { ExpertCache, expertKey, type ExpertRawBytes } from "../residency";
import {
  repackQ4_0, repackQ8_0, repackQ4_1, repackKQuant,
  dequantQ4_0, dequantQ8_0, dequantQ4_1, dequantQ5_K, dequantQ6_K,
  Q4_1_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
} from "../quant";
import { QWEN25_05B as S, GLM47_FLASH as G } from "../shape";

interface KResult { kernel: string; pass: boolean; maxAbs: number; maxRel: number; note?: string }

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

class Gpu {
  device!: GPUDevice;
  async init(): Promise<string> {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("niente adapter WebGPU");
    const lim = adapter.limits;
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(lim.maxStorageBufferBindingSize, 1 << 30),
        maxBufferSize: Math.min(lim.maxBufferSize, 1 << 30),
      },
    });
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
  async run(code: string, bindings: Array<GPUBuffer | { buffer: GPUBuffer; offset: number; size: number }>, workgroups: number, uniform?: GPUBuffer): Promise<void> {
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
    pass.dispatchWorkgroups(workgroups);
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
      const ref = new Float32Array(G.nHead * G.kvLora);
      for (let h = 0; h < G.nHead; h++) {
        const sc = new Float64Array(nPast + 1);
        for (let p = 0; p <= nPast; p++) {
          let acc = 0;
          for (let i = 0; i < W576; i++) acc += q[h * W576 + i] * cache[p * W576 + i];
          sc[p] = acc * scale;
        }
        let m = -Infinity;
        for (const v of sc) m = Math.max(m, v);
        let sum = 0;
        for (let p = 0; p <= nPast; p++) { sc[p] = Math.exp(sc[p] - m); sum += sc[p]; }
        for (let j = 0; j < G.kvLora; j++) {
          let acc = 0;
          for (let p = 0; p <= nPast; p++) acc += (sc[p] / sum) * cache[p * W576 + j];
          ref[h * G.kvLora + j] = acc;
        }
      }
      const out = g.empty(G.nHead * G.kvLora * 4);
      await g.run(mlaAttnDecodeWgsl({ nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax, scale }),
        [g.buf(q), g.buf(cache), out, ], G.nHead, g.uniform(nPast, nPast));
      results.push(compare("mla-attn-decode", new Float32Array(await g.read(out, G.nHead * G.kvLora * 4)), ref, 5e-4, 1e-4));
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

    // conformance layer-level con pesi reali (fase 4 slice 3) — il test lungo in coda
    results.push(await testGlmLayer0Real(g));
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e), results });
    return;
  }
  post({ type: "done", results });
}

void main();
