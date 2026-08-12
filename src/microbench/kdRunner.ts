// Runner della FASE 0 di engine-kernel-decode: micro-bench isolato delle
// varianti candidate dei due kernel caldi del decode. Zero run di modello.
//
// DISEGNO (pre-registrato in docs/deep-dive/kernel-decode-fase0-prereg-2026-08-13.md):
//  - una cella per (kernel, variante, forma);
//  - baseline e varianti compilate dallo STESSO harness, con gli STESSI buffer;
//  - esecuzione INTERLEAVATA (round-robin variante-per-ripetizione): la deriva
//    DVFS non deve coincidere con l'ordine delle varianti;
//  - >= 3 warm-up scartati, >= 10 campioni; si riporta p50 + dispersione;
//  - gate di lavoro: checksum finito, != 0, ed entro 1e-3 RELATIVO dal checksum
//    della forma attuale sulla stessa cella. Fuori tolleranza => cella skipped.
//
// La FORMA ATTUALE dei due kernel non e' ricopiata: si importa da
// src/engine/kernels/wgsl.ts. Nessun file di src/engine/** viene modificato.

import { attnDecodeWgsl, gemvQuantWgsl } from "../engine/kernels/wgsl";
import { createEngineDevice } from "../engine/gpudevice";
import { grantedLimits } from "../engine/gpulimits";
import { probeWebGPU } from "../probe";
import {
  MICROBENCH_SCHEMA_VERSION,
  type FeatureProbeResult,
  type KernelDecodeCell,
  type KernelDecodeProbe,
  type KernelDecodeRunFile,
  type KernelDecodeSkipped,
  type SampleStats,
  type TimingSource,
} from "./mbSchema";
import {
  ATTN_SHAPE, attnCombineWgsl, attnStreamWgsl, attnVec4Wgsl, splitChunkLen,
} from "./kdAttn";
import {
  DOT4I8_PROBE_WGSL, F16_PROBE_WGSL, SUBGROUP_PROBE_WGSL, gemvGrid,
  gemvVec4Rows2SgWgsl, gemvVec4Rows4Wgsl, gemvVec4SubgroupWgsl, gemvVec4Wgsl,
} from "./kdGemv";

export const WARMUP_SAMPLES = 3;
export const MEASURED_SAMPLES = 10;
export const GEMV_OPS_PER_SAMPLE = 16;
export const ATTN_SPLITS = 16;
/**
 * Layer di KV allocati per le celle "coldkv". La KV di UN layer a n=6333 pesa
 * 51,9 MB e la L2 della Lovelace ne tiene 72: misurata da sola, la cella misura
 * la L2. Ruotando il layer a ogni dispatch (8 x 51,9 = 415 MB) la cache viene
 * sfrattata fra un accesso e il successivo — che e' quello che succede nel
 * motore vero, dove ogni layer legge la SUA KV una volta per token.
 * AGGIUNTA POST-HOC, dichiarata: la pre-registrazione non prevedeva questo asse.
 */
export const KV_LAYERS = 8;
/** Forme reali del 4B (spec di fase 0). lm_head = l'unica non cache-resident. */
export const GEMV_SHAPES: Array<{ K: number; N: number; label: string }> = [
  { K: 2560, N: 2560, label: "qkv/o-proj" },
  { K: 2560, N: 9216, label: "gate/up-proj" },
  { K: 9216, N: 2560, label: "down-proj" },
  { K: 2560, N: 248320, label: "lm_head" },
];

// --------------------------------------------------------------------------
// statistica
// --------------------------------------------------------------------------
export function stats(samples: number[]): SampleStats {
  if (samples.length === 0) throw new Error("stats: nessun campione");
  const s = [...samples].sort((a, b) => a - b);
  const q = (f: number): number => s[Math.min(s.length - 1, Math.floor(f * (s.length - 1) + 0.5))];
  return { p50: q(0.5), min: s[0], max: s[s.length - 1], iqr: q(0.75) - q(0.25), n: s.length, samples };
}

// --------------------------------------------------------------------------
// dati
// --------------------------------------------------------------------------
/** mulberry32: dati riproducibili, nessun Math.random (stesso PRNG di runner.ts). */
function fillRandomU32(out: Uint32Array, seed: number): void {
  let a = seed >>> 0;
  for (let i = 0; i < out.length; i++) {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    out[i] = (t ^ (t >>> 14)) >>> 0;
  }
}

function fillRandomF32(out: Float32Array, seed: number): void {
  const u = new Uint32Array(out.length);
  fillRandomU32(u, seed);
  for (let i = 0; i < out.length; i++) out[i] = (u[i] / 4294967296) * 2 - 1;
}

function f32ToF16Bits(v: number): number {
  const f32 = new Float32Array(1); const u32 = new Uint32Array(f32.buffer);
  f32[0] = v; const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = (x >>> 13) & 0x3ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

// --------------------------------------------------------------------------
// meccanica di misura
// --------------------------------------------------------------------------
/**
 * `bindGroups` e' una LISTA perche' le celle "coldkv" ruotano il layer KV a ogni
 * dispatch (v. KV_LAYERS): con un solo layer da 51,9 MB il working set sta tutto
 * nella L2 da 72 MB della Lovelace e la cella misura la cache, non la VRAM —
 * esattamente il difetto che la pre-registrazione aveva dichiarato per il GEMV e
 * NON per l'attenzione.
 */
interface Dispatch { pipeline: GPUComputePipeline; bindGroups: GPUBindGroup[]; gx: number; gy: number }

interface Timer {
  querySet: GPUQuerySet | null;
  queryBuf: GPUBuffer | null;
  readBuf: GPUBuffer | null;
  useTs: boolean;
}

async function sampleOnce(
  device: GPUDevice, timer: Timer, ops: Dispatch[], opsPerSample: number, rot: { i: number },
): Promise<{ gpuMs: number | null; cpuMs: number }> {
  const enc = device.createCommandEncoder();
  const desc: GPUComputePassDescriptor = {};
  if (timer.useTs && timer.querySet) {
    desc.timestampWrites = { querySet: timer.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
  }
  const pass = enc.beginComputePass(desc);
  for (let o = 0; o < opsPerSample; o++) {
    for (const d of ops) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bindGroups[rot.i % d.bindGroups.length]);
      pass.dispatchWorkgroups(d.gx, d.gy);
    }
    rot.i++;
  }
  pass.end();
  if (timer.useTs && timer.querySet && timer.queryBuf && timer.readBuf) {
    enc.resolveQuerySet(timer.querySet, 0, 2, timer.queryBuf, 0);
    enc.copyBufferToBuffer(timer.queryBuf, 0, timer.readBuf, 0, 16);
  }
  const t0 = performance.now();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const cpuMs = (performance.now() - t0) / opsPerSample;
  let gpuMs: number | null = null;
  if (timer.useTs && timer.readBuf) {
    await timer.readBuf.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(timer.readBuf.getMappedRange().slice(0));
    timer.readBuf.unmap();
    const d = Number(ts[1] - ts[0]) / 1e6;
    if (d > 0) gpuMs = d / opsPerSample;
  }
  return { gpuMs, cpuMs };
}

interface VariantSpec {
  id: string;
  ops: Dispatch[];
  opsPerSample: number;
  /** true = la misura in batch, riportata a parte (il grade usa il singolo) */
  batched: boolean;
  /** stato di rotazione del layer KV, condiviso fra i campioni della variante */
  rot: { i: number };
}

interface Measured { gpu: number[]; cpu: number[] }

/** Esecuzione INTERLEAVATA: round-robin variante-per-ripetizione. */
async function measureInterleaved(
  device: GPUDevice, timer: Timer, variants: VariantSpec[], onProgress: (s: string) => void,
): Promise<Map<string, Measured>> {
  const out = new Map<string, Measured>();
  for (const v of variants) out.set(v.id, { gpu: [], cpu: [] });
  for (let rep = 0; rep < WARMUP_SAMPLES + MEASURED_SAMPLES; rep++) {
    const measuring = rep >= WARMUP_SAMPLES;
    onProgress(`rep ${rep + 1}/${WARMUP_SAMPLES + MEASURED_SAMPLES}${measuring ? "" : " (warm-up scartato)"}`);
    for (const v of variants) {
      const r = await sampleOnce(device, timer, v.ops, v.opsPerSample, v.rot);
      if (!measuring) continue;
      const m = out.get(v.id)!;
      if (r.gpuMs !== null) m.gpu.push(r.gpuMs);
      m.cpu.push(r.cpuMs);
    }
  }
  return out;
}

async function readChecksum(
  device: GPUDevice, buf: GPUBuffer, floats: number,
): Promise<{ sum: number; abs: number }> {
  const read = device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, read, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const y = new Float32Array(read.getMappedRange().slice(0));
  read.unmap(); read.destroy();
  let sum = 0; let abs = 0;
  for (let i = 0; i < y.length; i++) { sum += y[i]; abs += Math.abs(y[i]); }
  return { sum, abs };
}

async function runOnce(device: GPUDevice, ops: Dispatch[]): Promise<void> {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  for (const d of ops) {
    pass.setPipeline(d.pipeline); pass.setBindGroup(0, d.bindGroups[0]); pass.dispatchWorkgroups(d.gx, d.gy);
  }
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function compile(
  device: GPUDevice, code: string, label: string,
): Promise<{ pipeline: GPUComputePipeline | null; error: string | null }> {
  device.pushErrorScope("validation");
  try {
    const module = device.createShaderModule({ code, label });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error");
    if (errs.length) {
      await device.popErrorScope();
      return { pipeline: null, error: errs.map((e) => `${e.lineNum}: ${e.message}`).join(" | ").slice(0, 400) };
    }
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const err = await device.popErrorScope();
    if (err) return { pipeline: null, error: err.message.slice(0, 400) };
    return { pipeline, error: null };
  } catch (e) {
    await device.popErrorScope().catch(() => null);
    return { pipeline: null, error: e instanceof Error ? e.message.slice(0, 400) : String(e) };
  }
}

// --------------------------------------------------------------------------
// P1: sonda delle feature — presente/assente DICHIARATO MISURANDO
// --------------------------------------------------------------------------
async function probeFeature(
  device: GPUDevice, code: string, expect: (out: Float32Array) => boolean, outFloats: number,
): Promise<{ compiles: boolean; correct: boolean | null; note: string; out: number[] }> {
  const { pipeline, error } = await compile(device, code, "feature-probe");
  if (!pipeline) return { compiles: false, correct: null, note: error ?? "compilazione fallita", out: [] };
  const buf = device.createBuffer({ size: outFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
  await runOnce(device, [{ pipeline, bindGroups: [bg], gx: 1, gy: 1 }]);
  const read = device.createBuffer({ size: outFloats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, read, 0, outFloats * 4);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const v = new Float32Array(read.getMappedRange().slice(0));
  read.unmap(); read.destroy(); buf.destroy();
  const ok = expect(v);
  return { compiles: true, correct: ok, note: `out=[${[...v].join(", ")}]`, out: [...v] };
}

// --------------------------------------------------------------------------
// il run
// --------------------------------------------------------------------------
export interface KdRunOpts {
  deviceLabel: string;
  hostState: string;
  onProgress: (s: string) => void;
}

export async function runKernelDecodeBench(o: KdRunOpts): Promise<KernelDecodeRunFile> {
  const { onProgress } = o;
  const probe = await probeWebGPU(navigator.gpu, navigator as { userAgent: string; deviceMemory?: number });
  if (!probe.webgpu) throw new Error("WebGPU non disponibile");

  const biggestGemv = GEMV_SHAPES.reduce((a, b) => (b.K * b.N > a.K * a.N ? b : a));
  const { adapter, device, has } = await createEngineDevice({
    label: "kd-microbench",
    optionalFeatures: ["timestamp-query", "shader-f16", "subgroups" as GPUFeatureName,
      "chromium-experimental-subgroups" as GPUFeatureName,
      "chromium-experimental-subgroup-matrix" as GPUFeatureName],
    needs: {
      ctxMax: ATTN_SHAPE.ctxMax,
      mlaAttention: false,
      kvBytesPerLayer: ATTN_SHAPE.ctxMax * ATTN_SHAPE.nKvHead * ATTN_SHAPE.headDim * 4,
      extraBindings: [{
        bytes: (biggestGemv.K * biggestGemv.N) / 2,
        consumer: `kd-microbench: qs q4_0 della forma piu' grande (${biggestGemv.label} K${biggestGemv.K}xN${biggestGemv.N})`,
      }],
    },
  });

  const cells: KernelDecodeCell[] = [];
  const skipped: KernelDecodeSkipped[] = [];
  const useTs = has("timestamp-query");
  const timingSource: TimingSource = useTs ? "timestamp-query" : "cpu";
  const timer: Timer = { useTs, querySet: null, queryBuf: null, readBuf: null };
  if (useTs) {
    timer.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    timer.queryBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    timer.readBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  // ---------------- P1: sonda -----------------------------------------------
  onProgress("sonda delle feature (P1)…");
  const wgslFeat = [...(navigator.gpu.wgslLanguageFeatures ?? [])].sort();
  const adapterFeatures = [...adapter.features].sort();
  const deviceFeatures = [...device.features].sort();
  const features: Record<string, FeatureProbeResult> = {};
  const sgExposed = adapterFeatures.includes("subgroups") || adapterFeatures.includes("chromium-experimental-subgroups");
  const sgGranted = deviceFeatures.includes("subgroups") || deviceFeatures.includes("chromium-experimental-subgroups");
  let subgroupSizeObserved: number | null = null;
  {
    const r = await probeFeature(device, SUBGROUP_PROBE_WGSL, (v) => Math.abs(v[0] - 2016) < 1e-3, 4);
    if (r.out.length > 1 && r.out[1] > 0) subgroupSizeObserved = r.out[1];
    features["subgroups"] = {
      exposed: sgExposed, granted: sgGranted, compiles: r.compiles, correct: r.correct,
      note: `subgroupAdd(0..63) atteso 2016 — ${r.note}`,
    };
  }
  {
    const exposed = wgslFeat.includes("packed_4x8_integer_dot_product");
    const r = await probeFeature(device, DOT4I8_PROBE_WGSL, (v) => Math.abs(v[0] - 70) < 1e-3, 1);
    features["packed_4x8_integer_dot_product"] = {
      exposed, granted: null, compiles: r.compiles, correct: r.correct,
      note: `dot4I8Packed((1,2,3,4),(5,6,7,8)) atteso 70 — ${r.note}`,
    };
  }
  {
    const exposed = adapterFeatures.includes("shader-f16");
    const r = await probeFeature(device, F16_PROBE_WGSL, (v) => Math.abs(v[0] - 3) < 1e-3, 1);
    features["shader-f16"] = {
      exposed, granted: deviceFeatures.includes("shader-f16"), compiles: r.compiles, correct: r.correct,
      note: `1.5h*2.0h atteso 3 — ${r.note}`,
    };
  }
  features["chromium-experimental-subgroup-matrix"] = {
    exposed: adapterFeatures.includes("chromium-experimental-subgroup-matrix"),
    // CHIESTA fra le optionalFeatures: `granted` e' una misura, non un'assenza
    // di richiesta. Le config u8/f32 le enumera adapter.info.subgroupMatrixConfigs.
    granted: deviceFeatures.includes("chromium-experimental-subgroup-matrix"),
    compiles: null, correct: null,
    note: `subgroupMatrixConfigs: ${JSON.stringify(
      [...((adapter as unknown as { info?: { subgroupMatrixConfigs?: Iterable<Record<string, unknown>> } }).info?.subgroupMatrixConfigs ?? [])]
        .map((c) => ({ component: c.componentType, result: c.resultComponentType, M: c.M, N: c.N, K: c.K })),
    )}`,
  };
  const info = (adapter as unknown as { info?: Record<string, unknown> }).info;
  const kdProbe: KernelDecodeProbe = {
    adapterFeatures, deviceFeatures, wgslLanguageFeatures: wgslFeat,
    adapterInfo: info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : {},
    grantedLimits: grantedLimits(device),
    subgroupSizeObserved,
    features,
  };

  // ---------------- ATTENZIONE ---------------------------------------------
  onProgress("attenzione: allocazione buffer…");
  const { nHead, nKvHead, headDim, ctxMax, n } = ATTN_SHAPE;
  const kvDim = nKvHead * headDim;
  const layerBytes = ctxMax * kvDim * 4; // 26 214 400 B, multiplo di 256 (allineamento offset)
  const qBuf = device.createBuffer({ size: nHead * headDim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const kBuf = device.createBuffer({ size: layerBytes * KV_LAYERS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const vBuf = device.createBuffer({ size: layerBytes * KV_LAYERS, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outBuf = device.createBuffer({ size: nHead * headDim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const parBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const partOut = device.createBuffer({ size: nHead * ATTN_SPLITS * headDim * 4, usage: GPUBufferUsage.STORAGE });
  const partMS = device.createBuffer({ size: nHead * ATTN_SPLITS * 2 * 4, usage: GPUBufferUsage.STORAGE });
  {
    const qd = new Float32Array(nHead * headDim); fillRandomF32(qd, 7);
    device.queue.writeBuffer(qBuf, 0, qd);
    // TUTTI i layer portano gli STESSI dati: cambia l'indirizzo, non il
    // contenuto — cosi' il checksum di una cella coldkv resta confrontabile con
    // quello della base, e la banda misurata non ne risente.
    const kd = new Float32Array(ctxMax * kvDim); fillRandomF32(kd, 1000);
    const vd = new Float32Array(ctxMax * kvDim); fillRandomF32(vd, 90000);
    for (let l = 0; l < KV_LAYERS; l++) {
      device.queue.writeBuffer(kBuf, l * layerBytes, kd);
      device.queue.writeBuffer(vBuf, l * layerBytes, vd);
    }
    device.queue.writeBuffer(parBuf, 0, new Uint32Array([n - 1, n - 1, 0, 0]));
    await device.queue.onSubmittedWorkDone();
  }
  const kvEntries = (l: number): GPUBindGroupEntry[] => [
    { binding: 1, resource: { buffer: kBuf, offset: l * layerBytes, size: layerBytes } },
    { binding: 2, resource: { buffer: vBuf, offset: l * layerBytes, size: layerBytes } },
  ];

  const attnSources: Array<{ id: string; code: string; combine: number | null; gx: number; gy: number; heads: number; ctx: string }> = [
    { id: "base", code: attnDecodeWgsl({ nHead, nKvHead, headDim, ctxMax }), combine: null, gx: nHead, gy: 1, heads: 1,
      ctx: "forma attuale importata da src/engine/kernels/wgsl.ts: 1 workgroup (64 thread) per head, scores[ctxMax] in workgroup memory, ciclo scalare su headDim, kvHead = h/4" },
    { id: "vec4", code: attnVec4Wgsl(), combine: null, gx: nHead, gy: 1, heads: 1,
      ctx: "forma attuale + letture vec4<f32> su headDim; scores[ctxMax] e 1 wg/head invariati" },
    { id: "stream", code: attnStreamWgsl({ headsPerWg: 1, splits: 1 }), combine: null, gx: nHead, gy: 1, heads: 1,
      ctx: "softmax in streaming a tile di 64 posizioni (niente scores[ctxMax]) + vec4; 1 wg per head, nessuna fusione GQA, nessuno split" },
    { id: "gqa-stream", code: attnStreamWgsl({ headsPerWg: 4, splits: 1 }), combine: null, gx: nKvHead, gy: 1, heads: 4,
      ctx: "streaming + vec4 + un workgroup per gruppo GQA: la riga KV si legge UNA volta per le 4 head invece di 4" },
    { id: "split", code: attnStreamWgsl({ headsPerWg: 1, splits: ATTN_SPLITS }), combine: ATTN_SPLITS, gx: nHead, gy: ATTN_SPLITS, heads: 1,
      ctx: `streaming + vec4 + split del contesto su ${ATTN_SPLITS} workgroup per head (chunk ${splitChunkLen(ATTN_SPLITS)}) + kernel di combinazione log-sum-exp` },
    { id: "split-gqa", code: attnStreamWgsl({ headsPerWg: 4, splits: ATTN_SPLITS }), combine: ATTN_SPLITS, gx: nKvHead, gy: ATTN_SPLITS, heads: 4,
      ctx: `streaming + vec4 + fusione GQA + split del contesto su ${ATTN_SPLITS} workgroup per gruppo + combinazione log-sum-exp` },
  ];

  const attnVariants: VariantSpec[] = [];
  const attnMeta = new Map<string, { ops: Dispatch[]; heads: number; ctx: string; dispatches: number }>();
  const combinePipes = new Map<number, GPUComputePipeline>();
  for (const s of attnSources) {
    onProgress(`attenzione: compilo ${s.id}…`);
    const { pipeline, error } = await compile(device, s.code, `attn-${s.id}`);
    if (!pipeline) {
      skipped.push({ kernel: "attn-decode", variant: s.id, shape: { nHead, nKvHead, headDim, n }, reason: `compilazione: ${error}` });
      continue;
    }
    const mainBgs = Array.from({ length: KV_LAYERS }, (_, l) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: s.combine === null
        ? [
          { binding: 0, resource: { buffer: qBuf } }, ...kvEntries(l),
          { binding: 3, resource: { buffer: outBuf } }, { binding: 4, resource: { buffer: parBuf } },
        ]
        : [
          { binding: 0, resource: { buffer: qBuf } }, ...kvEntries(l),
          { binding: 3, resource: { buffer: partOut } }, { binding: 4, resource: { buffer: parBuf } },
          { binding: 5, resource: { buffer: partMS } },
        ],
    }));
    // hot = sempre il layer 0 (KV in L2); cold = rotazione su KV_LAYERS layer
    const build = (bgs: GPUBindGroup[]): Dispatch[] => [{ pipeline, bindGroups: bgs, gx: s.gx, gy: s.gy }];
    const hotOps = build([mainBgs[0]]);
    const coldOps = build(mainBgs);
    if (s.combine !== null) {
      let cp = combinePipes.get(s.combine);
      if (!cp) {
        const c = await compile(device, attnCombineWgsl(s.combine), "attn-combine");
        if (!c.pipeline) {
          skipped.push({ kernel: "attn-decode", variant: s.id, shape: { nHead, nKvHead, headDim, n }, reason: `combine: ${c.error}` });
          continue;
        }
        cp = c.pipeline; combinePipes.set(s.combine, cp);
      }
      const cbg = device.createBindGroup({
        layout: cp.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: partOut } }, { binding: 1, resource: { buffer: partMS } },
          { binding: 2, resource: { buffer: outBuf } },
        ],
      });
      const comb: Dispatch = { pipeline: cp, bindGroups: [cbg], gx: nHead, gy: 1 };
      hotOps.push(comb); coldOps.push(comb);
    }
    attnMeta.set(s.id, { ops: hotOps, heads: s.heads, ctx: s.ctx, dispatches: hotOps.length });
    attnMeta.set(`${s.id}-coldkv`, {
      ops: coldOps, heads: s.heads, dispatches: coldOps.length,
      ctx: `${s.ctx} — KV su ${KV_LAYERS} layer a rotazione (working set ${((layerBytes * 2 * KV_LAYERS) / 2 ** 20).toFixed(0)} MiB, fuori dalla L2): AGGIUNTA POST-HOC, non pre-registrata`,
    });
    attnVariants.push({ id: s.id, ops: hotOps, opsPerSample: 1, batched: false, rot: { i: 0 } });
    attnVariants.push({ id: `${s.id}#batch16`, ops: hotOps, opsPerSample: 16, batched: true, rot: { i: 0 } });
    attnVariants.push({ id: `${s.id}-coldkv`, ops: coldOps, opsPerSample: 1, batched: false, rot: { i: 0 } });
    attnVariants.push({ id: `${s.id}-coldkv#batch16`, ops: coldOps, opsPerSample: 16, batched: true, rot: { i: 0 } });
  }

  onProgress("attenzione: misura interleavata…");
  const attnMeasured = await measureInterleaved(device, timer, attnVariants, (m) => onProgress(`attenzione ${m}`));

  // checksum per variante (l'output e' condiviso: si rilegge subito dopo il run)
  const attnCk = new Map<string, { sum: number; abs: number }>();
  for (const [id, meta] of attnMeta) {
    await runOnce(device, meta.ops);
    attnCk.set(id, await readChecksum(device, outBuf, nHead * headDim));
  }
  const attnBaseCk = attnCk.get("base");

  const bytesUniqueAttn = n * nKvHead * headDim * 2 * 4; // 51,9 MB a n=6333
  for (const [id, meta] of attnMeta) {
    const single = attnMeasured.get(id)!;
    const batch = attnMeasured.get(`${id}#batch16`)!;
    const ck = attnCk.get(id)!;
    const relDiff = attnBaseCk && attnBaseCk.sum !== 0 ? Math.abs(ck.sum - attnBaseCk.sum) / Math.abs(attnBaseCk.sum) : null;
    const shape = { nHead, nKvHead, headDim, n, ctxMax };
    if (!Number.isFinite(ck.sum) || ck.sum === 0) {
      skipped.push({ kernel: "attn-decode", variant: id, shape, reason: `checksum sospetto (${ck.sum})` });
      continue;
    }
    if (relDiff !== null && relDiff > 1e-3) {
      skipped.push({ kernel: "attn-decode", variant: id, shape, reason: `checksum fuori tolleranza 1e-3: relDiff ${relDiff.toExponential(3)} (base ${attnBaseCk!.sum}, variante ${ck.sum})` });
      continue;
    }
    const msSrc = single.gpu.length > 0 ? single.gpu : single.cpu;
    const st = stats(msSrc);
    // traffico EMESSO: 4x l'unico nella forma attuale (ridondanza GQA), 1x se il
    // workgroup tratta il gruppo intero
    const emitted = bytesUniqueAttn * (meta.heads === 4 ? 1 : nHead / nKvHead);
    cells.push({
      kernel: "attn-decode", variant: id, shape, context: meta.ctx,
      dispatchesPerOp: meta.dispatches, opsPerSample: 1, warmupDiscarded: WARMUP_SAMPLES,
      timingSource: single.gpu.length > 0 ? timingSource : "cpu",
      msPerOp: st, cpuMsPerOp: stats(single.cpu),
      msPerOpBatched: stats(batch.gpu.length > 0 ? batch.gpu : batch.cpu),
      bytesUnique: bytesUniqueAttn, bytesEmitted: emitted,
      effectiveGBps: bytesUniqueAttn / 1e6 / st.p50,
      emittedGBps: emitted / 1e6 / st.p50,
      weightsPerSecond: null,
      checksum: ck.sum, checksumAbs: ck.abs, checksumRelDiff: relDiff,
      notes: `un layer per dispatch; nPast=${n - 1}. p50 su ${st.n} campioni interleavati.`,
    });
  }
  for (const b of [qBuf, kBuf, vBuf, outBuf, parBuf, partOut, partMS]) b.destroy();

  // ---------------- GEMV q4_0 ----------------------------------------------
  const sgOk = features["subgroups"].compiles === true && features["subgroups"].correct === true;
  for (const shape of GEMV_SHAPES) {
    const { K, N, label } = shape;
    onProgress(`gemv ${label} K${K}xN${N}: allocazione…`);
    const blocks = (N * K) / 32;
    const qsBuf = device.createBuffer({ size: (N * K) / 2, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    fillRandomU32(new Uint32Array(qsBuf.getMappedRange()), 4242);
    qsBuf.unmap();
    const scBuf = device.createBuffer({ size: blocks * 2, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    {
      const u = new Uint32Array(scBuf.getMappedRange());
      const r = new Uint32Array(u.length); fillRandomU32(r, 99);
      for (let i = 0; i < u.length; i++) {
        const a = f32ToF16Bits(0.01 + ((r[i] & 0xffff) / 65536) * 0.02);
        const b = f32ToF16Bits(0.01 + ((r[i] >>> 16) / 65536) * 0.02);
        u[i] = a | (b << 16);
      }
      scBuf.unmap();
    }
    const xBuf = device.createBuffer({ size: K * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    fillRandomF32(new Float32Array(xBuf.getMappedRange()), 1337);
    xBuf.unmap();
    const yBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const srcs: Array<{ id: string; code: string; rowsPerWg: number; ctx: string; gate?: string }> = [
      { id: "base", code: gemvQuantWgsl({ kind: "q4_0", K, N, hasBias: false }), rowsPerWg: 1,
        ctx: "forma attuale importata da src/engine/kernels/wgsl.ts: 64 thread per riga, load u32 scalari, nibble byte per byte, riduzione ad albero" },
      { id: "vec4", code: gemvVec4Wgsl(K, N), rowsPerWg: 1,
        ctx: "una load vec4<u32> (16 B) per blocco + dot() su vec4<f32> di x; 1 riga per workgroup, riduzione ad albero" },
      { id: "vec4-sg", code: gemvVec4SubgroupWgsl(K, N), rowsPerWg: 1,
        ctx: "come vec4, riduzione via subgroupAdd", gate: sgOk ? undefined : "subgroups assente o non corretto sulla sonda P1" },
      { id: "vec4-rows4", code: gemvVec4Rows4Wgsl(K, N), rowsPerWg: 4,
        ctx: "vec4 + 4 righe per workgroup (16 lane per riga): x riletta una volta per 4 righe invece di 4" },
      { id: "vec4-rows2-sg", code: gemvVec4Rows2SgWgsl(K, N), rowsPerWg: 2,
        ctx: "vec4 + 2 righe per workgroup, una riga per subgroup: la riduzione di riga e' un solo subgroupAdd",
        gate: sgOk && subgroupSizeObserved === 32 ? undefined : `richiede subgroups e subgroupSize 32 (osservato: ${subgroupSizeObserved})` },
    ];

    const variants: VariantSpec[] = [];
    const meta = new Map<string, { ops: Dispatch[]; ctx: string }>();
    for (const s of srcs) {
      if (s.gate) {
        skipped.push({ kernel: "gemv-q4_0", variant: s.id, shape: { K, N }, reason: s.gate });
        continue;
      }
      const { pipeline, error } = await compile(device, s.code, `gemv-${s.id}`);
      if (!pipeline) {
        skipped.push({ kernel: "gemv-q4_0", variant: s.id, shape: { K, N }, reason: `compilazione: ${error}` });
        continue;
      }
      const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: qsBuf } }, { binding: 1, resource: { buffer: scBuf } },
          { binding: 2, resource: { buffer: xBuf } }, { binding: 3, resource: { buffer: yBuf } },
        ],
      });
      const [gx, gy] = gemvGrid(Math.ceil(N / s.rowsPerWg));
      const ops: Dispatch[] = [{ pipeline, bindGroups: [bg], gx, gy }];
      meta.set(s.id, { ops, ctx: s.ctx });
      variants.push({ id: s.id, ops, opsPerSample: GEMV_OPS_PER_SAMPLE, batched: false, rot: { i: 0 } });
    }

    onProgress(`gemv ${label}: misura interleavata…`);
    const measured = await measureInterleaved(device, timer, variants, (m) => onProgress(`gemv ${label} ${m}`));
    const cks = new Map<string, { sum: number; abs: number }>();
    for (const [id, m] of meta) {
      await runOnce(device, m.ops);
      cks.set(id, await readChecksum(device, yBuf, N));
    }
    const baseCk = cks.get("base");
    // formula pre-registrata (nota: il termine 0,5625 include gia' le scale, il
    // termine successivo le riconta — si riporta com'e' stata pre-registrata e
    // si riporta anche la variante senza doppio conteggio)
    const bytesUnique = N * K * 0.5625 + (N * K / 32) * 2 + K * 4;
    const bytesStrict = (N * K) / 2 + (N * K / 32) * 2 + K * 4;
    for (const [id, m] of meta) {
      const mm = measured.get(id)!;
      const ck = cks.get(id)!;
      const relDiff = baseCk && baseCk.sum !== 0 ? Math.abs(ck.sum - baseCk.sum) / Math.abs(baseCk.sum) : null;
      if (!Number.isFinite(ck.sum) || ck.sum === 0) {
        skipped.push({ kernel: "gemv-q4_0", variant: id, shape: { K, N }, reason: `checksum sospetto (${ck.sum})` });
        continue;
      }
      if (relDiff !== null && relDiff > 1e-3) {
        skipped.push({ kernel: "gemv-q4_0", variant: id, shape: { K, N }, reason: `checksum fuori tolleranza 1e-3: relDiff ${relDiff.toExponential(3)}` });
        continue;
      }
      const st = stats(mm.gpu.length > 0 ? mm.gpu : mm.cpu);
      cells.push({
        kernel: "gemv-q4_0", variant: id, shape: { K, N }, context: `${label} — ${m.ctx}`,
        dispatchesPerOp: 1, opsPerSample: GEMV_OPS_PER_SAMPLE, warmupDiscarded: WARMUP_SAMPLES,
        timingSource: mm.gpu.length > 0 ? timingSource : "cpu",
        msPerOp: st, cpuMsPerOp: stats(mm.cpu), msPerOpBatched: null,
        bytesUnique, bytesEmitted: bytesUnique,
        effectiveGBps: bytesUnique / 1e6 / st.p50,
        emittedGBps: bytesStrict / 1e6 / st.p50,
        weightsPerSecond: (N * K) / (st.p50 / 1000),
        checksum: ck.sum, checksumAbs: ck.abs, checksumRelDiff: relDiff,
        notes: `${GEMV_OPS_PER_SAMPLE} dispatch identici per campione (hazard WAW su y), tempo per dispatch = delta/${GEMV_OPS_PER_SAMPLE}. bytesUnique = formula pre-registrata; emittedGBps usa la formula senza doppio conteggio delle scale (${bytesStrict} B).`,
      });
    }
    for (const b of [qsBuf, scBuf, xBuf, yBuf]) b.destroy();
  }

  timer.querySet?.destroy(); timer.queryBuf?.destroy(); timer.readBuf?.destroy();

  return {
    schemaVersion: MICROBENCH_SCHEMA_VERSION,
    kind: "microbench-kernel-decode",
    goal: "engine-kernel-decode fase 0",
    prereg: "docs/deep-dive/kernel-decode-fase0-prereg-2026-08-13.md",
    deviceLabel: o.deviceLabel,
    hostState: { declared: o.hostState },
    ts: new Date().toISOString(),
    probe, kdProbe, cells, skipped,
  };
}
