// Runner della RIGA 1 di engine-ttft: sonde e varianti del PREFILL.
// Pre-registrazione: docs/deep-dive/ttft-riga1-prereg-2026-08-13.md
//
// Zero run di modello: solo micro-bench isolati. Il banco e' quello della fase 0
// del goal precedente (kdRunner.ts) — statistica, esecuzione INTERLEAVATA,
// timer, gate di checksum e compilazione si IMPORTANO da li', non si riscrivono.
// Qui vivono solo le tre sonde nuove:
//   (a) picco di calcolo fp32 su GEMM densa (ttGemm.gemmDenseF32Wgsl);
//   (b) moltiplicatore multi-riga a M = 1, 8, 16, 32 — forma attuale importata
//       da src/engine/kernels/wgsl.ts contro pesi-in-registri / pesi-in-shared /
//       split-K;
//   (c) attenzione a chunk del prefill a ctx 388 e 6333 — legacy importato
//       contro la famiglia streaming.
// La sonda (d) — spazzata del tetto negoziabile — vive nel DRIVER: richiede
// device distinti con `requiredLimits` espliciti, e src/ ha un punto unico di
// creazione device che negozia (tests/gpudevice.test.ts lo impone).

// `attnDecodeLegacyBatchWgsl` e non `attnDecodeWgsl({ batch: true })`: dal task
// T1-kernel-batch-streaming quest'ultimo emette la forma in STREAMING, cioe' la
// CANDIDATA. Usarlo qui farebbe misurare alla sonda streaming contro streaming.
import { attnDecodeLegacyBatchWgsl, gemvQuantWgsl, gemvGrid } from "../engine/kernels/wgsl";
import { createEngineDevice } from "../engine/gpudevice";
import { grantedLimits } from "../engine/gpulimits";
import { probeWebGPU } from "../probe";
import {
  MICROBENCH_SCHEMA_VERSION,
  type FeatureProbeResult,
  type KernelDecodeProbe,
  type KernelDecodeSkipped,
  type TimingSource,
  type TtftCell,
  type TtftRunFile,
} from "./mbSchema";
import {
  WARMUP_SAMPLES, GEMV_OPS_PER_SAMPLE,
  compile, fillRandomF32, fillRandomU32, f32ToF16Bits,
  measureInterleaved, readChecksum, runOnce, stats,
  type Dispatch, type Timer, type VariantSpec,
} from "./kdRunner";
import { DOT4I8_PROBE_WGSL, F16_PROBE_WGSL, SUBGROUP_PROBE_WGSL } from "./kdGemv";
import {
  gemmDenseF32Wgsl, gemmDenseGrid, gemmQ4MultiRowRegsWgsl, gemmQ4MultiRowRegsGrid,
  gemmQ4MultiRowSharedWgsl, gemmQ4MultiRowSharedGrid, gemmQ4MultiRowSplitKWgsl,
  gemmQ4MultiRowSplitKIdotWgsl,
  splitKCombineWgsl, workgroupStorageBytes, quantXQ8Wgsl,
} from "./ttGemm";
import { TT_ATTN_SHAPE, attnPrefillGrid, attnPrefillStreamWgsl } from "./ttAttn";

/** Shape della sonda del picco: quadrata + la shape reale del prefill del 4B. */
export const DENSE_SHAPES: Array<{ M: number; K: number; N: number; label: string }> = [
  { M: 4096, K: 4096, N: 4096, label: "quadrata 4096^3" },
  { M: 6336, K: 2560, N: 9216, label: "prefill 4B gate/up-proj, M = 6336 (~6333 token)" },
];

/** Shape reali del 4B su cui si misura il moltiplicatore multi-riga. */
export const MULTIROW_SHAPES: Array<{ K: number; N: number; label: string; Ms: number[] }> = [
  { K: 2560, N: 9216, label: "gate/up-proj", Ms: [1, 8, 16, 32] },
  { K: 9216, N: 2560, label: "down-proj", Ms: [16] },
];

export const SPLIT_K = 4;
/** Passate della sonda del picco: e' la cella con la dispersione peggiore. */
export const DENSE_PASSES = 4;
/** Copie dei pesi ruotate a ogni dispatch nella cella `coldw` (8 x 13,3 = 106 MB). */
export const WEIGHT_COPIES = 8;
/** Contesti su cui si misura l'attenzione a chunk (M = 16). */
export const ATTN_CTX = [388, 6333];
export const ATTN_M = 16;

interface Ck { sum: number; abs: number }

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
  return { compiles: true, correct: expect(v), note: `out=[${[...v].join(", ")}]`, out: [...v] };
}

export interface TtRunOpts {
  deviceLabel: string;
  hostState: string;
  onProgress: (s: string) => void;
}

/**
 * Il piano che il driver esegue per la spazzata dei limiti (sonda d / P6).
 * Porta TUTTE le forme candidate: quale vinca lo decide la misura, e la spazzata
 * deve poter rispondere sulla vincitrice vera, non su quella predetta.
 */
export interface TtSweepForm {
  variant: string;
  wgsl: string;
  workgroupStorageBytes: number;
  gx: number; gy: number;
  /** secondo dispatch (split-K): somma dei parziali */
  combineWgsl: string | null;
  combineGx: number;
  partBytes: number;
}

export interface TtSweepPlan {
  forms: TtSweepForm[];
  K: number; N: number; M: number;
  qsBytes: number; scalesBytes: number; xBytes: number; yBytes: number;
  opsPerSample: number;
  attnLegacy: { wgsl: string; workgroupStorageBytes: number; ctxMax: number };
  requestedLimits: number[];
}

export async function runTtftProbeBench(
  o: TtRunOpts,
): Promise<{ runFile: TtftRunFile; sweepPlan: TtSweepPlan }> {
  const { onProgress } = o;
  const probe = await probeWebGPU(navigator.gpu, navigator as { userAgent: string; deviceMemory?: number });
  if (!probe.webgpu) throw new Error("WebGPU non disponibile");

  const biggestDense = DENSE_SHAPES.reduce((a, b) => (b.M * b.N > a.M * a.N ? b : a));
  const { adapter, device, has } = await createEngineDevice({
    label: "tt-microbench",
    optionalFeatures: ["timestamp-query", "shader-f16", "subgroups" as GPUFeatureName,
      "chromium-experimental-subgroups" as GPUFeatureName],
    needs: {
      ctxMax: TT_ATTN_SHAPE.ctxMax,
      mlaAttention: false,
      kvBytesPerLayer: TT_ATTN_SHAPE.ctxMax * TT_ATTN_SHAPE.nKvHead * TT_ATTN_SHAPE.headDim * 4,
      extraBindings: [{
        bytes: biggestDense.M * biggestDense.N * 4,
        consumer: `tt-microbench: C della GEMM densa piu' grande (${biggestDense.label})`,
      }],
    },
  });

  const cells: TtftCell[] = [];
  const skipped: KernelDecodeSkipped[] = [];
  const useTs = has("timestamp-query");
  const timingSource: TimingSource = useTs ? "timestamp-query" : "cpu";
  const timer: Timer = { useTs, querySet: null, queryBuf: null, readBuf: null };
  if (useTs) {
    timer.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    timer.queryBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    timer.readBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  // ---------------- sonda delle feature (eredita la forma di fase 0) --------
  onProgress("sonda delle feature…");
  const wgslFeat = [...(navigator.gpu.wgslLanguageFeatures ?? [])].sort();
  const adapterFeatures = [...adapter.features].sort();
  const deviceFeatures = [...device.features].sort();
  const features: Record<string, FeatureProbeResult> = {};
  let subgroupSizeObserved: number | null = null;
  {
    const r = await probeFeature(device, SUBGROUP_PROBE_WGSL, (v) => Math.abs(v[0] - 2016) < 1e-3, 4);
    if (r.out.length > 1 && r.out[1] > 0) subgroupSizeObserved = r.out[1];
    features["subgroups"] = {
      exposed: adapterFeatures.includes("subgroups") || adapterFeatures.includes("chromium-experimental-subgroups"),
      granted: deviceFeatures.includes("subgroups") || deviceFeatures.includes("chromium-experimental-subgroups"),
      compiles: r.compiles, correct: r.correct, note: `subgroupAdd(0..63) atteso 2016 — ${r.note}`,
    };
  }
  {
    const r = await probeFeature(device, F16_PROBE_WGSL, (v) => Math.abs(v[0] - 3) < 1e-3, 1);
    features["shader-f16"] = {
      exposed: adapterFeatures.includes("shader-f16"), granted: deviceFeatures.includes("shader-f16"),
      compiles: r.compiles, correct: r.correct, note: `1.5h*2.0h atteso 3 — ${r.note}`,
    };
  }
  {
    const r = await probeFeature(device, DOT4I8_PROBE_WGSL, (v) => Math.abs(v[0] - 70) < 1e-3, 1);
    features["packed_4x8_integer_dot_product"] = {
      exposed: wgslFeat.includes("packed_4x8_integer_dot_product"), granted: null,
      compiles: r.compiles, correct: r.correct, note: `dot4I8Packed atteso 70 — ${r.note}`,
    };
  }
  const info = (adapter as unknown as { info?: Record<string, unknown> }).info;
  const kdProbe: KernelDecodeProbe = {
    adapterFeatures, deviceFeatures, wgslLanguageFeatures: wgslFeat,
    adapterInfo: info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : {},
    grantedLimits: grantedLimits(device),
    subgroupSizeObserved, features,
  };
  const adapterLimits: Record<string, number> = {};
  {
    const lim = adapter.limits as unknown as Record<string, number>;
    for (const k of [
      "maxBufferSize", "maxStorageBufferBindingSize", "maxStorageBuffersPerShaderStage",
      "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX", "maxComputeWorkgroupsPerDimension", "maxComputeWorkgroupSizeY",
      "maxComputeWorkgroupSizeZ", "minStorageBufferOffsetAlignment",
    ]) adapterLimits[k] = Number(lim[k] ?? 0);
  }

  // =========================================================================
  // (a) SONDA DEL PICCO DI CALCOLO fp32 — GEMM densa
  //
  // Le due shape si misurano INSIEME e interleavate, e su DENSE_PASSES passate:
  // un dispatch da decine di ms su un portatile power-limited oscilla col DVFS
  // (IQR 15% su 10 campioni, misurato), e con una shape alla volta la deriva
  // coinciderebbe con l'ordine delle shape. E' la cella che decide P1/P2:
  // e' l'unica che meritava piu' campioni.
  // =========================================================================
  {
    const bufs: GPUBuffer[] = [];
    const dvariants: VariantSpec[] = [];
    const dmeta = new Map<string, { shape: typeof DENSE_SHAPES[number]; ops: Dispatch[]; wgs: number; cBuf: GPUBuffer }>();
    for (const s of DENSE_SHAPES) {
      onProgress(`gemm densa ${s.label}: allocazione…`);
      const aBuf = device.createBuffer({ size: s.M * s.K * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
      fillRandomF32(new Float32Array(aBuf.getMappedRange()), 11); aBuf.unmap();
      const bBuf = device.createBuffer({ size: s.K * s.N * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
      fillRandomF32(new Float32Array(bBuf.getMappedRange()), 22); bBuf.unmap();
      const cBuf = device.createBuffer({ size: s.M * s.N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      bufs.push(aBuf, bBuf, cBuf);
      const code = gemmDenseF32Wgsl(s.M, s.K, s.N);
      const { pipeline, error } = await compile(device, code, `gemm-dense-${s.label}`);
      if (!pipeline) {
        skipped.push({ kernel: "gemm-dense-f32", variant: "tile64x64x8", shape: { M: s.M, K: s.K, N: s.N }, reason: `compilazione: ${error}` });
        continue;
      }
      const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuf } }, { binding: 1, resource: { buffer: bBuf } },
          { binding: 2, resource: { buffer: cBuf } },
        ],
      });
      const [gx, gy] = gemmDenseGrid(s.M, s.N);
      const ops: Dispatch[] = [{ pipeline, bindGroups: [bg], gx, gy }];
      const id = `M${s.M}K${s.K}N${s.N}`;
      dmeta.set(id, { shape: s, ops, wgs: workgroupStorageBytes(code), cBuf });
      dvariants.push({ id, ops, opsPerSample: 1, batched: false, rot: { i: 0 } });
    }
    if (dvariants.length > 0) {
      onProgress("gemm densa: misura interleavata…");
      const measured = await measureInterleaved(device, timer, dvariants, (m) => onProgress(`gemm densa ${m}`), DENSE_PASSES);
      for (const [id, dm] of dmeta) {
        const s = dm.shape;
        await runOnce(device, dm.ops);
        // checksum su un prefisso: C intero e' fino a 233 MB, la mappatura
        // costerebbe piu' della misura. Il prefisso copre piu' tile.
        const ckFloats = Math.min(s.M * s.N, 65536);
        const ck = await readChecksum(device, dm.cBuf, ckFloats);
        const mm = measured.get(id)!;
        const st = stats(mm.gpu.length > 0 ? mm.gpu : mm.cpu);
        const flop = 2 * s.M * s.N * s.K;
        const bytes = (s.M * s.K + s.K * s.N + s.M * s.N) * 4;
        if (!Number.isFinite(ck.sum) || ck.sum === 0) {
          skipped.push({ kernel: "gemm-dense-f32", variant: "tile64x64x8", shape: { M: s.M, K: s.K, N: s.N }, reason: `checksum sospetto (${ck.sum})` });
          continue;
        }
        cells.push({
          kernel: "gemm-dense-f32", variant: "tile64x64x8", shape: { M: s.M, K: s.K, N: s.N }, M: s.M,
          context: `${s.label} — GEMM densa fp32, tile 64x64x8, workgroup_size 256 (16x16 thread, 4x4 uscite per thread), A e B a tile in workgroup memory. E' la sonda del PICCO di calcolo (done-when a). Le due shape sono misurate interleavate fra loro.`,
          dispatchesPerOp: 1, opsPerSample: 1, warmupDiscarded: WARMUP_SAMPLES * DENSE_PASSES,
          timingSource: mm.gpu.length > 0 ? timingSource : "cpu",
          msPerOp: st, cpuMsPerOp: stats(mm.cpu),
          bytesUnique: bytes, bytesEmitted: bytes,
          effectiveGBps: bytes / 1e6 / st.p50, emittedGBps: bytes / 1e6 / st.p50,
          weightsPerSecond: null,
          weightBytesUnique: null, weightBytesEmitted: null, weightBytesPerToken: null,
          tokensPerSecond: null,
          tflops: flop / (st.p50 / 1000) / 1e12,
          workgroupStorageBytes: dm.wgs,
          checksum: ck.sum, checksumAbs: ck.abs, checksumRelDiff: null,
          hostState: o.hostState,
          notes: `un dispatch per campione, p50 su ${st.n} campioni in ${DENSE_PASSES} passate interleavate. TFLOP/s al MINIMO campione (il tetto meno sporcato dal DVFS) = ${(flop / (st.min / 1000) / 1e12).toFixed(3)}. checksum sul prefisso di ${ckFloats} uscite (C intero = ${(s.M * s.N * 4 / 2 ** 20).toFixed(0)} MiB).`,
        });
      }
    }
    for (const b of bufs) b.destroy();
  }

  // =========================================================================
  // (b) MOLTIPLICATORE MULTI-RIGA q4_0
  // =========================================================================
  let sweepPlan: TtSweepPlan | null = null;
  for (const shape of MULTIROW_SHAPES) {
    const { K, N, label } = shape;
    const Mmax = Math.max(...shape.Ms);
    onProgress(`multirow ${label} K${K}xN${N}: allocazione…`);
    const qsBytes = (N * K) / 2;
    const scBytes = (N * K / 32) * 2;
    const cold = shape.Ms.includes(16) && K === 2560; // cella coldw solo sulla shape primaria
    const copies = cold ? WEIGHT_COPIES : 1;
    const qsBuf = device.createBuffer({ size: qsBytes * copies, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    {
      const u = new Uint32Array(qsBuf.getMappedRange());
      const per = qsBytes / 4;
      // TUTTE le copie portano gli STESSI dati: cambia l'indirizzo, non il
      // contenuto — cosi' il checksum della cella coldw resta confrontabile.
      const one = new Uint32Array(per); fillRandomU32(one, 4242);
      for (let c = 0; c < copies; c++) u.set(one, c * per);
      qsBuf.unmap();
    }
    const scBuf = device.createBuffer({ size: scBytes * copies, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    {
      const u = new Uint32Array(scBuf.getMappedRange());
      const per = scBytes / 4;
      const one = new Uint32Array(per);
      const r = new Uint32Array(per); fillRandomU32(r, 99);
      for (let i = 0; i < per; i++) {
        const a = f32ToF16Bits(0.01 + ((r[i] & 0xffff) / 65536) * 0.02);
        const b = f32ToF16Bits(0.01 + ((r[i] >>> 16) / 65536) * 0.02);
        one[i] = a | (b << 16);
      }
      for (let c = 0; c < copies; c++) u.set(one, c * per);
      scBuf.unmap();
    }
    const xBuf = device.createBuffer({ size: Mmax * K * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    fillRandomF32(new Float32Array(xBuf.getMappedRange()), 1337); xBuf.unmap();
    // it.5 — ATTIVAZIONI QUANTIZZATE per la cella `splitk-idot` (docket item 11).
    // Rigenerate con lo STESSO seme di `xBuf`: la cella intera e quelle in
    // virgola mobile devono vedere gli stessi numeri, o il confronto dei
    // checksum non misura la quantizzazione ma due input diversi.
    const nBlk = K / 32;
    const xHost = new Float32Array(Mmax * K);
    fillRandomF32(xHost, 1337);
    const xqHost = new Uint32Array(Mmax * nBlk * 8);
    const xscHost = new Float32Array(Mmax * nBlk);
    for (let m = 0; m < Mmax; m++) {
      for (let b = 0; b < nBlk; b++) {
        let amax = 0;
        for (let i = 0; i < 32; i++) amax = Math.max(amax, Math.abs(xHost[m * K + b * 32 + i]));
        const sc = amax / 127;
        xscHost[m * nBlk + b] = sc;
        const inv = sc > 0 ? 1 / sc : 0;
        for (let i = 0; i < 32; i++) {
          const q = Math.max(-127, Math.min(127, Math.round(xHost[m * K + b * 32 + i] * inv)));
          xqHost[(m * nBlk + b) * 8 + (i >> 2)] |= (q & 0xff) << ((i & 3) * 8);
        }
      }
    }
    const xqBuf = device.createBuffer({ size: xqHost.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Uint32Array(xqBuf.getMappedRange()).set(xqHost); xqBuf.unmap();
    const xscBuf = device.createBuffer({ size: xscHost.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(xscBuf.getMappedRange()).set(xscHost); xscBuf.unmap();

    const yBuf = device.createBuffer({ size: Mmax * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const partBuf = device.createBuffer({ size: SPLIT_K * Mmax * N * 4, usage: GPUBufferUsage.STORAGE });

    for (const M of shape.Ms) {
      const srcs: Array<{
        id: string; code: string; grid: [number, number, number]; combine?: string;
        weightEmitFactor: number; xEmitFactor: number; ctx: string; coldw?: boolean; idot?: boolean;
        quantOnly?: boolean; quantFirst?: string; quantGx?: number;
      }> = [];
      const baseCode = gemvQuantWgsl({ kind: "q4_0", K, N, hasBias: false, batch: true });
      const [bgx, bgy] = gemvGrid(N);
      srcs.push({
        id: "base-batch-z", code: baseCode, grid: [bgx, bgy, M], weightEmitFactor: M, xEmitFactor: N,
        ctx: "FORMA ATTUALE importata da src/engine/kernels/wgsl.ts (gemvQuantWgsl con batch: true): M GEMV replicate su wid.z, un workgroup da 64 thread per (riga, m), load u32 scalari. Riuso dei pesi ZERO per costruzione.",
      });
      srcs.push({
        id: "regs", code: gemmQ4MultiRowRegsWgsl({ K, N, M }),
        grid: [...gemmQ4MultiRowRegsGrid({ K, N, M }), 1] as [number, number, number],
        weightEmitFactor: 1, xEmitFactor: Math.ceil(N / 64),
        ctx: "pesi in REGISTRI, attivazioni in workgroup memory: 64 righe di uscita per workgroup (una per thread), passo su K di 2 blocchi q4_0, tile di attivazioni M x 64 in shared. Ogni blocco di peso letto UNA volta e usato per tutte le M righe.",
      });
      if (M === 1 || M % 4 === 0) {
        srcs.push({
          id: "shared", code: gemmQ4MultiRowSharedWgsl({ K, N, M }),
          grid: [...gemmQ4MultiRowSharedGrid({ K, N, M }), 1] as [number, number, number],
          weightEmitFactor: 1, xEmitFactor: Math.ceil(N / 16),
          ctx: "pesi DEQUANTIZZATI IN WORKGROUP MEMORY: 16 righe di uscita per workgroup e quattro partizioni di M (thread t serve la riga t%16 e le m congrue a t/16 mod 4). E' la forma il cui shared scala coi PESI.",
        });
      }
      if ((K / 32) % (SPLIT_K * 2) === 0) {
        srcs.push({
          id: "splitk", code: gemmQ4MultiRowSplitKWgsl({ K, N, M, splits: SPLIT_K }),
          grid: [Math.ceil(N / 64), SPLIT_K, 1], combine: splitKCombineWgsl({ K, N, M, splits: SPLIT_K }),
          weightEmitFactor: 1, xEmitFactor: Math.ceil(N / 64),
          ctx: `come 'regs' ma con K spezzato su ${SPLIT_K} workgroup lungo wid.y + un dispatch di combinazione dei parziali. Serve a compensare l'occupancy: a N ${N} la forma 'regs' lancia solo ${Math.ceil(N / 64)} workgroup.`,
        });
      }
      // it.5 — la leva intera, autorizzata dal PI (docket item 11). Enunciati
      // P8 (tolleranza) e P9 (velocita') pre-registrati prima di scriverla.
      if ((K / 32) % (SPLIT_K * 2) === 0) {
        srcs.push({
          id: "splitk-idot", code: gemmQ4MultiRowSplitKIdotWgsl({ K, N, M, splits: SPLIT_K }),
          grid: [Math.ceil(N / 64), SPLIT_K, 1], combine: splitKCombineWgsl({ K, N, M, splits: SPLIT_K }),
          weightEmitFactor: 1, xEmitFactor: Math.ceil(N / 64), idot: true,
          ctx: "come 'splitk' ma col PRODOTTO SCALARE INTERO (dot4I8Packed): attivazioni quantizzate a i8 per blocco da 32, accumulo i32, scala applicata una volta per blocco. Via q4_0 x q8_0 di llama.cpp. NON bit-identica per costruzione: tolleranza P8 <= 2,0e-2, dal conto sull'errore di quantizzazione. ATTIVAZIONI GIA' QUANTIZZATE: questa cella misura il KERNEL, non la leva.",
        });
        // it.6 — il termine che it.5 aveva dichiarato «fuori misura»: la passata
        // che quantizza le attivazioni. `quantx-q8` da sola, e `splitk-idot-full`
        // = quantizzazione + moltiplicatore, che e' il confronto ONESTO contro
        // `splitk` perche' contiene tutto cio' che la leva costa.
        srcs.push({
          id: "quantx-q8", code: quantXQ8Wgsl({ K, N, M }), grid: [Math.ceil((M * (K / 32)) / 64), 1, 1],
          weightEmitFactor: 0, xEmitFactor: 2, quantOnly: true,
          ctx: "SOLA quantizzazione delle attivazioni a i8 per blocco da 32 (un thread per blocco). E' il costo che la via intera AGGIUNGE, misurato invece che stimato.",
        });
        srcs.push({
          id: "splitk-idot-full", code: gemmQ4MultiRowSplitKIdotWgsl({ K, N, M, splits: SPLIT_K }),
          grid: [Math.ceil(N / 64), SPLIT_K, 1], combine: splitKCombineWgsl({ K, N, M, splits: SPLIT_K }),
          weightEmitFactor: 1, xEmitFactor: Math.ceil(N / 64), idot: true,
          quantFirst: quantXQ8Wgsl({ K, N, M }), quantGx: Math.ceil((M * (K / 32)) / 64),
          ctx: "LA LEVA PER INTERO: quantizzazione delle attivazioni + moltiplicatore intero + combinazione dei parziali, nello STESSO campione. E' questo il numero da confrontare con 'splitk', non 'splitk-idot' da solo.",
        });
      }
      if (cold && M === 16) {
        srcs.push({
          id: "regs-coldw", code: gemmQ4MultiRowRegsWgsl({ K, N, M }),
          grid: [...gemmQ4MultiRowRegsGrid({ K, N, M }), 1] as [number, number, number],
          weightEmitFactor: 1, xEmitFactor: Math.ceil(N / 64), coldw: true,
          ctx: `'regs' con i pesi ruotati su ${WEIGHT_COPIES} copie a ogni dispatch (working set ${((qsBytes + scBytes) * WEIGHT_COPIES / 2 ** 20).toFixed(0)} MiB, oltre la L2): nel motore i pesi sono 2,25 GB e streammano dalla VRAM, non stanno in cache.`,
        });
        srcs.push({
          id: "base-batch-z-coldw", code: baseCode, grid: [bgx, bgy, M], weightEmitFactor: M, xEmitFactor: N, coldw: true,
          ctx: `forma attuale con i pesi ruotati su ${WEIGHT_COPIES} copie a ogni dispatch — la lettura fredda della stessa cella, pubblicata accanto a quella L2-resident.`,
        });
        // it.4: la cella fredda della VINCITRICE, che la riga 1 non ha girato.
        // Era pre-registrata come discriminante della CAUSA (traffico o
        // occupancy?) ed è stata data a `regs` e alla forma attuale, non a
        // `splitk` che ha poi vinto: senza di lei il 43,1x resta senza causa
        // accertata e la riga 2 costruirebbe su un rapporto non discriminato.
        // Enunciato P7 nel prereg (addendum it.4): atteso <= 0,0700 ms.
        if ((K / 32) % (SPLIT_K * 2) === 0) {
          srcs.push({
            id: "splitk-coldw", code: gemmQ4MultiRowSplitKWgsl({ K, N, M, splits: SPLIT_K }),
            grid: [Math.ceil(N / 64), SPLIT_K, 1], combine: splitKCombineWgsl({ K, N, M, splits: SPLIT_K }),
            weightEmitFactor: 1, xEmitFactor: Math.ceil(N / 64), coldw: true,
            ctx: `'splitk' con i pesi ruotati su ${WEIGHT_COPIES} copie a ogni dispatch (working set ${((qsBytes + scBytes) * WEIGHT_COPIES / 2 ** 20).toFixed(0)} MiB, oltre la L2). Discrimina la CAUSA del vantaggio della forma vincente: se regge a pesi freddi è occupancy, se crolla era L2-residenza del banco.`,
          });
        }
      }

      const variants: VariantSpec[] = [];
      const meta = new Map<string, { ops: Dispatch[]; ctx: string; wgs: number; emit: number; xEmit: number; disp: number }>();
      for (const s of srcs) {
        const { pipeline, error } = await compile(device, s.code, `multirow-${s.id}-M${M}`);
        if (!pipeline) {
          skipped.push({ kernel: "gemm-q4_0-multirow", variant: `${s.id}@M${M}`, shape: { K, N, M }, reason: `compilazione: ${error}` });
          continue;
        }
        const nBg = s.coldw ? copies : 1;
        const bgs = Array.from({ length: nBg }, (_, c) => device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: s.quantOnly ? [
            { binding: 0, resource: { buffer: xBuf } },
            { binding: 1, resource: { buffer: xqBuf } },
            { binding: 2, resource: { buffer: xscBuf } },
          ] : s.idot ? [
            { binding: 0, resource: { buffer: qsBuf, offset: c * qsBytes, size: qsBytes } },
            { binding: 1, resource: { buffer: scBuf, offset: c * scBytes, size: scBytes } },
            { binding: 2, resource: { buffer: xqBuf } },
            { binding: 3, resource: { buffer: s.combine ? partBuf : yBuf } },
            { binding: 4, resource: { buffer: xscBuf } },
          ] : [
            { binding: 0, resource: { buffer: qsBuf, offset: c * qsBytes, size: qsBytes } },
            { binding: 1, resource: { buffer: scBuf, offset: c * scBytes, size: scBytes } },
            { binding: 2, resource: { buffer: xBuf } },
            { binding: 3, resource: { buffer: s.combine ? partBuf : yBuf } },
          ],
        }));
        const ops: Dispatch[] = [{ pipeline, bindGroups: bgs, gx: s.grid[0], gy: s.grid[1], gz: s.grid[2] }];
        // `splitk-idot-full`: la passata di quantizzazione ENTRA nel campione,
        // davanti al moltiplicatore. E' cio' che rende il confronto con
        // `splitk` onesto invece che dedotto (it.6).
        if (s.quantFirst) {
          const q = await compile(device, s.quantFirst, `multirow-${s.id}-quant-M${M}`);
          if (!q.pipeline) {
            skipped.push({ kernel: "gemm-q4_0-multirow", variant: `${s.id}@M${M}`, shape: { K, N, M }, reason: `quant: ${q.error}` });
            continue;
          }
          const qbg = device.createBindGroup({
            layout: q.pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: xBuf } },
              { binding: 1, resource: { buffer: xqBuf } },
              { binding: 2, resource: { buffer: xscBuf } },
            ],
          });
          ops.unshift({ pipeline: q.pipeline, bindGroups: [qbg], gx: s.quantGx ?? 1, gy: 1 });
        }
        if (s.combine) {
          const c = await compile(device, s.combine, `multirow-${s.id}-combine-M${M}`);
          if (!c.pipeline) {
            skipped.push({ kernel: "gemm-q4_0-multirow", variant: `${s.id}@M${M}`, shape: { K, N, M }, reason: `combine: ${c.error}` });
            continue;
          }
          const cbg = device.createBindGroup({
            layout: c.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: partBuf } }, { binding: 1, resource: { buffer: yBuf } }],
          });
          ops.push({ pipeline: c.pipeline, bindGroups: [cbg], gx: Math.ceil((M * N) / 64), gy: 1 });
        }
        meta.set(s.id, { ops, ctx: s.ctx, wgs: workgroupStorageBytes(s.code), emit: s.weightEmitFactor, xEmit: s.xEmitFactor, disp: ops.length });
        variants.push({ id: s.id, ops, opsPerSample: GEMV_OPS_PER_SAMPLE, batched: false, rot: { i: 0 } });
        if (K === 2560 && M === 16 && (s.id === "regs" || s.id === "shared" || s.id === "splitk")) {
          const legacyCode = attnDecodeLegacyBatchWgsl(TT_ATTN_SHAPE);
          sweepPlan ??= {
            forms: [], K, N, M,
            qsBytes, scalesBytes: scBytes, xBytes: M * K * 4, yBytes: M * N * 4,
            opsPerSample: GEMV_OPS_PER_SAMPLE,
            attnLegacy: {
              wgsl: legacyCode, workgroupStorageBytes: workgroupStorageBytes(legacyCode),
              ctxMax: TT_ATTN_SHAPE.ctxMax,
            },
            requestedLimits: [16384, 24576, 32768, 49152],
          };
          sweepPlan.forms.push({
            variant: s.id, wgsl: s.code, workgroupStorageBytes: workgroupStorageBytes(s.code),
            gx: s.grid[0], gy: s.grid[1],
            combineWgsl: s.combine ?? null, combineGx: Math.ceil((M * N) / 64),
            partBytes: SPLIT_K * M * N * 4,
          });
        }
      }
      if (variants.length === 0) continue;

      onProgress(`multirow ${label} M=${M}: misura interleavata (${variants.length} varianti)…`);
      const measured = await measureInterleaved(device, timer, variants, (m) => onProgress(`multirow ${label} M=${M} ${m}`));
      const cks = new Map<string, Ck>();
      for (const [id, m] of meta) {
        await runOnce(device, m.ops);
        cks.set(id, await readChecksum(device, yBuf, M * N));
      }
      const baseCk = cks.get("base-batch-z");
      const weightUnique = qsBytes + scBytes;
      for (const [id, m] of meta) {
        const mm = measured.get(id)!;
        const ck = cks.get(id)!;
        const relDiff = baseCk && baseCk.sum !== 0 ? Math.abs(ck.sum - baseCk.sum) / Math.abs(baseCk.sum) : null;
        if (!Number.isFinite(ck.sum) || ck.sum === 0) {
          skipped.push({ kernel: "gemm-q4_0-multirow", variant: `${id}@M${M}`, shape: { K, N, M }, reason: `checksum sospetto (${ck.sum})` });
          continue;
        }
        // Tolleranza PER VARIANTE, non una sola per tutte. Le forme che cambiano
        // solo l'ordine delle somme restano a 1e-3; `splitk-idot` cambia
        // l'ARITMETICA (attivazioni quantizzate a 8 bit) e ha la sua tolleranza
        // PRE-REGISTRATA in P8, ricavata dal conto sull'errore di quantizzazione
        // e non tarata su ciò che è uscito. Resta un gate: sopra 2e-2 la cella è
        // sbagliata, non imprecisa.
        // `quantx-q8` non scrive `y`: il suo checksum e' quello lasciato dalla
        // variante precedente e NON dice niente sulla sua correttezza. Bocciarla
        // su quel numero sarebbe teatro; pubblicarla come se il gate l'avesse
        // approvata sarebbe peggio. Esente, e la ragione va nel JSON (notes).
        const noChecksum = id === "quantx-q8";
        // La via intera cambia l'ARITMETICA (attivazioni a 8 bit) e ha la sua
        // tolleranza PRE-REGISTRATA in P8, ricavata dal conto sull'errore di
        // quantizzazione: vale per il kernel E per la leva completa, che fanno
        // esattamente la stessa aritmetica.
        const tol = (id === "splitk-idot" || id === "splitk-idot-full") ? 2e-2 : 1e-3;
        if (!noChecksum && relDiff !== null && relDiff > tol) {
          skipped.push({
            kernel: "gemm-q4_0-multirow", variant: `${id}@M${M}`, shape: { K, N, M },
            reason: `checksum fuori tolleranza ${tol.toExponential(0)}: relDiff ${relDiff.toExponential(3)} (base ${baseCk!.sum}, variante ${ck.sum})`,
          });
          continue;
        }
        const st = stats(mm.gpu.length > 0 ? mm.gpu : mm.cpu);
        const wEmitted = weightUnique * m.emit;
        // traffico totale EMESSO (richieste, prima delle cache): pesi con la
        // loro ridondanza + x riletta da ogni workgroup + y. La forma attuale
        // rilegge la riga di x per OGNI riga di uscita — un termine che la
        // spiega quanto i pesi, e va contato dov'e'.
        const xEmitted = m.xEmit * M * K * 4;
        const emitted = wEmitted + xEmitted + M * N * 4;
        const unique = weightUnique + M * K * 4 + M * N * 4;
        cells.push({
          kernel: "gemm-q4_0-multirow", variant: id, shape: { K, N }, M,
          context: `${label} — ${m.ctx}`,
          dispatchesPerOp: m.disp, opsPerSample: GEMV_OPS_PER_SAMPLE, warmupDiscarded: WARMUP_SAMPLES,
          timingSource: mm.gpu.length > 0 ? timingSource : "cpu",
          msPerOp: st, cpuMsPerOp: stats(mm.cpu),
          bytesUnique: unique, bytesEmitted: emitted,
          effectiveGBps: unique / 1e6 / st.p50, emittedGBps: emitted / 1e6 / st.p50,
          weightsPerSecond: (N * K * M) / (st.p50 / 1000),
          weightBytesUnique: weightUnique, weightBytesEmitted: wEmitted, weightBytesPerToken: wEmitted / M,
          tokensPerSecond: M / (st.p50 / 1000),
          tflops: (2 * M * N * K) / (st.p50 / 1000) / 1e12,
          workgroupStorageBytes: m.wgs,
          checksum: ck.sum, checksumAbs: ck.abs, checksumRelDiff: relDiff,
          hostState: o.hostState,
          notes: `${GEMV_OPS_PER_SAMPLE} op identiche per campione (hazard WAW su y), ms per op = delta/${GEMV_OPS_PER_SAMPLE}. p50 su ${st.n} campioni interleavati. weightsPerSecond conta i pesi USATI (N*K*M), non quelli letti.`,
        });
      }
    }
    for (const b of [qsBuf, scBuf, xBuf, xqBuf, xscBuf, yBuf, partBuf]) b.destroy();
  }

  // =========================================================================
  // (c) ATTENZIONE A CHUNK DEL PREFILL
  // =========================================================================
  {
    const { nHead, nKvHead, headDim, ctxMax } = TT_ATTN_SHAPE;
    const kvDim = nKvHead * headDim;
    const M = ATTN_M;
    onProgress("attn prefill: allocazione…");
    const qBuf = device.createBuffer({ size: M * nHead * headDim * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    fillRandomF32(new Float32Array(qBuf.getMappedRange()), 7); qBuf.unmap();
    const kBuf = device.createBuffer({ size: ctxMax * kvDim * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    fillRandomF32(new Float32Array(kBuf.getMappedRange()), 1000); kBuf.unmap();
    const vBuf = device.createBuffer({ size: ctxMax * kvDim * 4, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    fillRandomF32(new Float32Array(vBuf.getMappedRange()), 90000); vBuf.unmap();
    const outBuf = device.createBuffer({ size: M * nHead * headDim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rowPastBuf = device.createBuffer({ size: M * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const legacyCode = attnDecodeLegacyBatchWgsl({ nHead, nKvHead, headDim, ctxMax });
    const attnSrcs: Array<{ id: string; code: string; grid: [number, number, number]; kvEmit: number; ctx: string }> = [
      {
        id: "legacy", code: legacyCode, grid: [nHead, M, 1], kvEmit: (nHead / nKvHead) * M,
        ctx: `BASELINE importata da src/engine/kernels/wgsl.ts (attnDecodeLegacyBatchWgsl): scores[ctxMax=${ctxMax}] in workgroup memory, letture scalari, un workgroup per (head, riga) — la riga KV si rilegge una volta per ognuna delle ${nHead / nKvHead} head del gruppo GQA e una volta per ognuna delle ${M} righe. Era la forma di PRODUZIONE del prefill a chunk fino al task T1-kernel-batch-streaming; da li' in poi e' il fallback dichiarato, e resta il termine di paragone di questa sonda con lo stesso testo byte per byte.`,
      },
      {
        id: "stream", code: attnPrefillStreamWgsl({ headsPerWg: 1, rowsPerWg: 1 }),
        grid: attnPrefillGrid({ headsPerWg: 1, rowsPerWg: 1 }, M), kvEmit: (nHead / nKvHead) * M,
        ctx: "softmax in STREAMING a tile di 64 posizioni (niente scores[ctxMax]: il workgroup storage diventa costante in ctxMax) + letture vec4. Un workgroup per (head, riga), come il legacy: isola l'effetto della sola softmax/vettorizzazione dal traffico.",
      },
      {
        id: "gqa-stream", code: attnPrefillStreamWgsl({ headsPerWg: 4, rowsPerWg: 1 }),
        grid: attnPrefillGrid({ headsPerWg: 4, rowsPerWg: 1 }, M), kvEmit: M,
        ctx: "streaming + vec4 + KV letta UNA volta per gruppo GQA (un workgroup per (gruppo, riga)): il traffico KV scende di 4x rispetto al legacy.",
      },
      {
        id: "gqa-rows2", code: attnPrefillStreamWgsl({ headsPerWg: 4, rowsPerWg: 2 }),
        grid: attnPrefillGrid({ headsPerWg: 4, rowsPerWg: 2 }, M), kvEmit: M / 2,
        ctx: "streaming + vec4 + fusione GQA + DUE righe del chunk per workgroup: la riga KV serve 4 head x 2 righe, cioe' 8 consumatori invece di 1. E' la leva specifica del prefill, che sul decode non esisteva.",
      },
    ];

    for (const n of ATTN_CTX) {
      device.queue.writeBuffer(rowPastBuf, 0, new Uint32Array(Array.from({ length: M }, (_, i) => n - M + i)));
      await device.queue.onSubmittedWorkDone();
      const variants: VariantSpec[] = [];
      const meta = new Map<string, { ops: Dispatch[]; ctx: string; wgs: number; kvEmit: number }>();
      for (const s of attnSrcs) {
        const { pipeline, error } = await compile(device, s.code, `attn-prefill-${s.id}`);
        if (!pipeline) {
          skipped.push({ kernel: "attn-prefill-chunk", variant: `${s.id}@n${n}`, shape: { nHead, nKvHead, headDim, n, ctxMax, M }, reason: `compilazione: ${error}` });
          continue;
        }
        const bg = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: qBuf } }, { binding: 1, resource: { buffer: kBuf } },
            { binding: 2, resource: { buffer: vBuf } }, { binding: 3, resource: { buffer: outBuf } },
            { binding: 4, resource: { buffer: rowPastBuf } },
          ],
        });
        const ops: Dispatch[] = [{ pipeline, bindGroups: [bg], gx: s.grid[0], gy: s.grid[1], gz: s.grid[2] }];
        meta.set(s.id, { ops, ctx: s.ctx, wgs: workgroupStorageBytes(s.code), kvEmit: s.kvEmit });
        variants.push({ id: s.id, ops, opsPerSample: 1, batched: false, rot: { i: 0 } });
      }
      onProgress(`attn prefill ctx ${n}: misura interleavata…`);
      const measured = await measureInterleaved(device, timer, variants, (m) => onProgress(`attn prefill ctx ${n} ${m}`));
      const cks = new Map<string, Ck>();
      for (const [id, m] of meta) {
        await runOnce(device, m.ops);
        cks.set(id, await readChecksum(device, outBuf, M * nHead * headDim));
      }
      const baseCk = cks.get("legacy");
      const kvUnique = n * kvDim * 2 * 4;
      for (const [id, m] of meta) {
        const mm = measured.get(id)!;
        const ck = cks.get(id)!;
        const relDiff = baseCk && baseCk.sum !== 0 ? Math.abs(ck.sum - baseCk.sum) / Math.abs(baseCk.sum) : null;
        const shape = { nHead, nKvHead, headDim, n, ctxMax };
        if (!Number.isFinite(ck.sum) || ck.sum === 0) {
          skipped.push({ kernel: "attn-prefill-chunk", variant: `${id}@n${n}`, shape: { ...shape, M }, reason: `checksum sospetto (${ck.sum})` });
          continue;
        }
        if (relDiff !== null && relDiff > 1e-3) {
          skipped.push({
            kernel: "attn-prefill-chunk", variant: `${id}@n${n}`, shape: { ...shape, M },
            reason: `checksum fuori tolleranza 1e-3: relDiff ${relDiff.toExponential(3)} (legacy ${baseCk!.sum}, variante ${ck.sum})`,
          });
          continue;
        }
        const st = stats(mm.gpu.length > 0 ? mm.gpu : mm.cpu);
        const emitted = kvUnique * m.kvEmit;
        cells.push({
          kernel: "attn-prefill-chunk", variant: id, shape, M,
          context: m.ctx,
          dispatchesPerOp: 1, opsPerSample: 1, warmupDiscarded: WARMUP_SAMPLES,
          timingSource: mm.gpu.length > 0 ? timingSource : "cpu",
          msPerOp: st, cpuMsPerOp: stats(mm.cpu),
          bytesUnique: kvUnique, bytesEmitted: emitted,
          effectiveGBps: kvUnique / 1e6 / st.p50, emittedGBps: emitted / 1e6 / st.p50,
          weightsPerSecond: null,
          weightBytesUnique: null, weightBytesEmitted: null, weightBytesPerToken: null,
          tokensPerSecond: M / (st.p50 / 1000),
          tflops: null,
          workgroupStorageBytes: m.wgs,
          checksum: ck.sum, checksumAbs: ck.abs, checksumRelDiff: relDiff,
          hostState: o.hostState,
          notes: `un chunk di ${M} righe per op; rowPast[m] = ${n - M} + m. ms per token = msPerOp/${M}. p50 su ${st.n} campioni interleavati.`,
        });
      }
    }
    for (const b of [qBuf, kBuf, vBuf, outBuf, rowPastBuf]) b.destroy();
  }

  timer.querySet?.destroy(); timer.queryBuf?.destroy(); timer.readBuf?.destroy();
  device.destroy();

  if (!sweepPlan || sweepPlan.forms.length === 0) {
    throw new Error("sweepPlan non prodotto: nessuna forma candidata compilata a K2560 M16");
  }

  return {
    runFile: {
      schemaVersion: MICROBENCH_SCHEMA_VERSION,
      kind: "microbench-ttft-riga1",
      goal: "engine-ttft riga 1 (sonde e varianti del prefill)",
      prereg: "docs/deep-dive/ttft-riga1-prereg-2026-08-13.md",
      deviceLabel: o.deviceLabel,
      hostState: { declared: o.hostState },
      ts: new Date().toISOString(),
      probe, kdProbe, adapterLimits, cells, skipped,
      limitSweep: null,
    },
    sweepPlan,
  };
}

