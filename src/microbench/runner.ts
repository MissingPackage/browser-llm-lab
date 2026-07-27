import type { MetricAggregate } from "../metrics";
import type { MicrobenchCell, MicrobenchKernelId, SkippedCell, TimingSource } from "./mbSchema";
import { aggregateSamples, bytesReadFor, effectiveGBps } from "./stats";
import { kernelSource } from "./kernels";

// Taglie decode-shape: da hidden di Qwen2.5-0.5B (896) a taglie da modello 7-8B.
// Quadrate (N=K) per semplicità di lettura della curva: la domanda del deep-dive è
// "dove si piega il roofline al crescere della taglia", non riprodurre ogni proiezione.
export const DEFAULT_SIZES: Array<{ rowsN: number; colsK: number }> = [
  { rowsN: 896, colsK: 896 },
  { rowsN: 2048, colsK: 2048 },
  { rowsN: 4096, colsK: 4096 },
  { rowsN: 8192, colsK: 8192 },
  // Working set q4 (~160 MB) sopra la L2 delle GPU desktop grandi (AD103: 64 MB):
  // senza questa taglia il q4 resta cache-resident e misura la L2, non la VRAM
  // (osservato dal vivo: f32@4096² dava ~1000 GB/s "effettivi", sopra il datasheet).
  { rowsN: 16384, colsK: 16384 },
];
export const DEFAULT_REPEATS = 10;
const WARMUP_DISPATCHES = 3;
// Chrome quantizza i timestamp GPU (~100us) per mitigazioni di timing: un singolo
// dispatch GEMV piccolo sta sotto il quanto e il delta viene 0. Si misura quindi un
// batch di dispatch per campione e si divide (osservato dal vivo: primo run 4090,
// "effectiveGBps: tempo non positivo").
export const DISPATCHES_PER_SAMPLE = 16;

// PRNG deterministico (mulberry32): dati riproducibili, niente Math.random.
export function seededData(len: number, seed: number): Float32Array {
  let a = seed >>> 0;
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    out[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  }
  return out;
}

function packQ4(weightsF: Float32Array, colsK: number): { packed: Uint32Array; scales: Float32Array } {
  const rows = weightsF.length / colsK;
  const packed = new Uint32Array((rows * colsK) / 8);
  const scales = new Float32Array((rows * colsK) / 32);
  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < colsK / 32; g++) {
      let maxAbs = 0;
      for (let i = 0; i < 32; i++) maxAbs = Math.max(maxAbs, Math.abs(weightsF[r * colsK + g * 32 + i]));
      const scale = maxAbs / 7 || 1;
      scales[r * (colsK / 32) + g] = scale;
      for (let i = 0; i < 32; i++) {
        const q = Math.max(0, Math.min(15, Math.round(weightsF[r * colsK + g * 32 + i] / scale) + 7));
        const flat = r * colsK + g * 32 + i;
        packed[flat >> 3] |= q << ((flat & 7) * 4);
      }
    }
  }
  return { packed, scales };
}

function f32ToF16Bits(v: number): number {
  // Conversione IEEE754 f32->f16 (round-to-nearest), sufficiente per dati di bench.
  const f32 = new Float32Array(1); const u32 = new Uint32Array(f32.buffer);
  f32[0] = v; const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = (x >>> 13) & 0x3ff;
  if (exp <= 0) return sign; // flush denormali/underflow a zero: ok per dati random in [-1,1]
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

export interface RunnerProgress {
  (message: string): void;
}

async function timeDispatches(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number,
  repeats: number,
  useTimestamps: boolean,
): Promise<{ gpuMs: MetricAggregate | null; cpuMs: MetricAggregate; timingSource: TimingSource }> {
  const gpuSamples: number[] = [];
  const cpuSamples: number[] = [];
  let querySet: GPUQuerySet | null = null;
  let queryBuf: GPUBuffer | null = null;
  let readBuf: GPUBuffer | null = null;
  if (useTimestamps) {
    querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    queryBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    readBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }
  for (let i = 0; i < repeats; i++) {
    const enc = device.createCommandEncoder();
    const passDesc: GPUComputePassDescriptor = {};
    if (useTimestamps && querySet) {
      passDesc.timestampWrites = { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
    }
    const pass = enc.beginComputePass(passDesc);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    // Batch di dispatch identici nello stesso pass (serializzati dal WAW hazard su y):
    // il tempo per dispatch è il delta diviso per il batch — sopra il quanto di Chrome.
    for (let d = 0; d < DISPATCHES_PER_SAMPLE; d++) pass.dispatchWorkgroups(workgroups);
    pass.end();
    if (useTimestamps && querySet && queryBuf && readBuf) {
      enc.resolveQuerySet(querySet, 0, 2, queryBuf, 0);
      enc.copyBufferToBuffer(queryBuf, 0, readBuf, 0, 16);
    }
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    cpuSamples.push((performance.now() - t0) / DISPATCHES_PER_SAMPLE);
    if (useTimestamps && readBuf) {
      await readBuf.mapAsync(GPUMapMode.READ);
      const ts = new BigUint64Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      const deltaMs = Number(ts[1] - ts[0]) / 1e6; // ns -> ms, per l'intero batch
      if (deltaMs > 0) gpuSamples.push(deltaMs / DISPATCHES_PER_SAMPLE);
      // deltaMs 0 = sotto il quanto anche col batch: campione scartato, fallback CPU.
    }
  }
  querySet?.destroy(); queryBuf?.destroy(); readBuf?.destroy();
  return {
    gpuMs: gpuSamples.length > 0 ? aggregateSamples(gpuSamples) : null,
    cpuMs: aggregateSamples(cpuSamples),
    timingSource: gpuSamples.length > 0 ? "timestamp-query" : "cpu",
  };
}

export interface MicrobenchOutcome {
  cells: MicrobenchCell[];
  skipped: SkippedCell[];
}

export async function runMicrobench(
  device: GPUDevice,
  onProgress: RunnerProgress,
  sizes = DEFAULT_SIZES,
  repeats = DEFAULT_REPEATS,
): Promise<MicrobenchOutcome> {
  const hasTs = device.features.has("timestamp-query");
  const hasF16 = device.features.has("shader-f16");
  const kernels: MicrobenchKernelId[] = hasF16
    ? ["gemv-q4f32", "gemv-f32", "gemv-f16"]
    : ["gemv-q4f32", "gemv-f32"];
  if (!hasF16) onProgress("shader-f16 non disponibile: variante f16 saltata (buco loggato)");
  if (!hasTs) onProgress("timestamp-query non disponibile: timing solo CPU-side");
  const cells: MicrobenchCell[] = [];
  const skipped: SkippedCell[] = [];

  for (const { rowsN, colsK } of sizes) {
    const wF = seededData(rowsN * colsK, 42);
    const xF = seededData(colsK, 1337);
    const { packed, scales } = packQ4(wF, colsK);
    for (const kernel of kernels) {
      onProgress(`${kernel} ${rowsN}x${colsK}…`);
      // Error scope: una cella che fallisce la validazione WebGPU (es. buffer oltre i
      // limiti del device) deve diventare uno skip esplicito, mai un numero garbage.
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      const module = device.createShaderModule({ code: kernelSource(kernel, colsK) });
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      const mk = (data: ArrayBufferView, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => {
        const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage });
        device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
        return buf;
      };
      const yBuf = device.createBuffer({
        size: rowsN * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      let entries: GPUBindGroupEntry[];
      const toFree: GPUBuffer[] = [yBuf];
      if (kernel === "gemv-q4f32") {
        const w = mk(packed); const s = mk(scales); const x = mk(xF);
        toFree.push(w, s, x);
        entries = [
          { binding: 0, resource: { buffer: w } },
          { binding: 1, resource: { buffer: s } },
          { binding: 2, resource: { buffer: x } },
          { binding: 3, resource: { buffer: yBuf } },
        ];
      } else if (kernel === "gemv-f16") {
        const wH = new Uint16Array(rowsN * colsK); const xH = new Uint16Array(colsK);
        for (let i = 0; i < wH.length; i++) wH[i] = f32ToF16Bits(wF[i]);
        for (let i = 0; i < xH.length; i++) xH[i] = f32ToF16Bits(xF[i]);
        const w = mk(wH); const x = mk(xH);
        toFree.push(w, x);
        entries = [
          { binding: 0, resource: { buffer: w } },
          { binding: 1, resource: { buffer: x } },
          { binding: 2, resource: { buffer: yBuf } },
        ];
      } else {
        const w = mk(wF); const x = mk(xF);
        toFree.push(w, x);
        entries = [
          { binding: 0, resource: { buffer: w } },
          { binding: 1, resource: { buffer: x } },
          { binding: 2, resource: { buffer: yBuf } },
        ];
      }
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });

      // warm-up fuori misura: compile/clock ramp non devono sporcare i repeats
      for (let i = 0; i < WARMUP_DISPATCHES; i++) {
        const enc = device.createCommandEncoder();
        const p = enc.beginComputePass();
        p.setPipeline(pipeline); p.setBindGroup(0, bindGroup); p.dispatchWorkgroups(rowsN); p.end();
        device.queue.submit([enc.finish()]);
      }
      await device.queue.onSubmittedWorkDone();

      const { gpuMs, cpuMs, timingSource } = await timeDispatches(
        device, pipeline, bindGroup, rowsN, repeats, hasTs,
      );

      // checksum: leggi y e somma — verifica che il kernel abbia scritto davvero
      const readY = device.createBuffer({ size: rowsN * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(yBuf, 0, readY, 0, rowsN * 4);
      device.queue.submit([enc.finish()]);
      await readY.mapAsync(GPUMapMode.READ);
      const y = new Float32Array(readY.getMappedRange().slice(0));
      readY.unmap(); readY.destroy();
      let checksum = 0;
      for (let i = 0; i < y.length; i++) checksum += y[i];

      const oomError = await device.popErrorScope();
      const valError = await device.popErrorScope();
      toFree.forEach((b) => b.destroy());
      const gpuError = valError ?? oomError;
      if (gpuError) {
        const reason = `${valError ? "validation" : "out-of-memory"}: ${gpuError.message}`;
        onProgress(`SKIP ${kernel} ${rowsN}x${colsK} — ${reason}`);
        skipped.push({ kernel, rowsN, colsK, reason });
        continue;
      }
      if (!Number.isFinite(checksum) || checksum === 0) {
        const reason = `checksum sospetto (${checksum}): il kernel potrebbe non aver scritto`;
        onProgress(`SKIP ${kernel} ${rowsN}x${colsK} — ${reason}`);
        skipped.push({ kernel, rowsN, colsK, reason });
        continue;
      }

      const bytesRead = bytesReadFor(kernel, rowsN, colsK);
      const bestMs = gpuMs ? gpuMs.mean : cpuMs.mean;
      cells.push({
        kernel, rowsN, colsK, bytesRead, repeats,
        timingSource, gpuMs, cpuMs,
        effectiveGBps: effectiveGBps(bytesRead, bestMs),
        checksum,
      });
    }
  }
  return { cells, skipped };
}
