import type { MicrobenchKernelId } from "./mbSchema";

// Kernel GEMV y[N] = W[N,K]·x[K] modellati sulla forma dei kernel TVM reali dumpati
// (.harness/goals/fase-2-deep-dive/wgsl-dump/, es. 027: un workgroup da 64 thread per
// riga di output, dequant nei registri, riduzione ad albero in workgroup). Non sono i
// kernel TVM byte-per-byte: sono la stessa *forma*, parametrica in N e K, per misurare
// il comportamento della classe di kernel, non riprodurre il compilatore.

export const WORKGROUP_SIZE = 64;

// q4 "f32 compute": 8 nibble per u32, dequant (w>>k & 15) - 7, scale f32 per gruppo di 32
// — lo schema esatto osservato in 014/027 del dump.
function gemvQ4F32(colsK: number): string {
  const wordsPerRow = colsK / 8;
  const scalesPerRow = colsK / 32;
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> weights : array<u32>;
@group(0) @binding(1) var<storage, read> scales : array<f32>;
@group(0) @binding(2) var<storage, read> x : array<f32>;
@group(0) @binding(3) var<storage, read_write> y : array<f32>;
var<workgroup> red : array<f32, ${WORKGROUP_SIZE}>;
@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x;
  let t = lid.x;
  var acc : f32 = 0.0;
  // ogni thread copre wordsPerRow/64 u32 in strided access
  for (var w : u32 = t; w < ${wordsPerRow}u; w = w + ${WORKGROUP_SIZE}u) {
    let packed = weights[row * ${wordsPerRow}u + w];
    let scale = scales[row * ${scalesPerRow}u + (w >> 2u)];
    let base = w * 8u;
    for (var i : u32 = 0u; i < 8u; i = i + 1u) {
      let q = f32((packed >> (i * 4u)) & 15u) - 7.0;
      acc = fma(x[base + i], q * scale, acc);
    }
  }
  red[t] = acc;
  workgroupBarrier();
  var stride : u32 = ${WORKGROUP_SIZE / 2}u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { y[row] = red[0]; }
}`;
}

function gemvPlain(colsK: number, ty: "f32" | "f16"): string {
  const enable = ty === "f16" ? "enable f16;\n" : "";
  return /* wgsl */ `${enable}
@group(0) @binding(0) var<storage, read> weights : array<${ty}>;
@group(0) @binding(1) var<storage, read> x : array<${ty}>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
var<workgroup> red : array<f32, ${WORKGROUP_SIZE}>;
@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x;
  let t = lid.x;
  var acc : ${ty} = ${ty}(0.0);
  for (var k : u32 = t; k < ${colsK}u; k = k + ${WORKGROUP_SIZE}u) {
    acc = acc + weights[row * ${colsK}u + k] * x[k];
  }
  red[t] = f32(acc);
  workgroupBarrier();
  var stride : u32 = ${WORKGROUP_SIZE / 2}u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { y[row] = red[0]; }
}`;
}

export function kernelSource(kernel: MicrobenchKernelId, colsK: number): string {
  if (colsK % 32 !== 0) throw new Error("kernelSource: colsK deve essere multiplo di 32");
  switch (kernel) {
    case "gemv-q4f32":
      return gemvQ4F32(colsK);
    case "gemv-f32":
      return gemvPlain(colsK, "f32");
    case "gemv-f16":
      return gemvPlain(colsK, "f16");
  }
}

export function kernelNeedsF16(kernel: MicrobenchKernelId): boolean {
  return kernel === "gemv-f16";
}
