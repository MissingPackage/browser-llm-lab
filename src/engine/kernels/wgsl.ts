// Kernel WGSL del motore — fase A, versioni "correttezza prima" (nessuna fusione:
// la fusione è la leva L3 e arriva col piano statico). Specializzazione a template:
// le shape sono COSTANTI baked nel sorgente (le conosce plan/shape, non servono
// uniform) — pattern shader-lib di LlamaWeb. L'unico stato dinamico per token è
// l'uniform TokParams {pos, nPast}.
//
// Convenzioni: f32 ovunque (niente shader-f16: le scale f16 si leggono con
// unpack2x16float, builtin sempre disponibile); un buffer qs/scales per tensore
// quantizzato (layout di repackQ4_0/repackQ8_0 in quant.ts — i kernel DEVONO
// riprodurre esattamente il dequant reference).

export const TOK_PARAMS_WGSL = `struct TokParams { pos: u32, nPast: u32 };`;

// Larghezza X della griglia 2D dei gemv (limite WebGPU: 65535 wg/dimensione).
export const GEMV_GRID_X = 32768;
export function gemvGrid(N: number): [number, number] {
  return N <= GEMV_GRID_X ? [N, 1] : [GEMV_GRID_X, Math.ceil(N / GEMV_GRID_X)];
}

// GEMV dequant-fusa: y[r] = Σ_b scale(b)·Σ_j q_j·x[...] (+ bias). Un workgroup da
// 64 thread per riga di output; riduzione in shared memory.
export function gemvQuantWgsl(opts: {
  kind: "q4_0" | "q8_0"; K: number; N: number; hasBias: boolean;
}): string {
  const { kind, K, N, hasBias } = opts;
  if (K % 32 !== 0) throw new Error("gemv: K non multiplo di 32");
  const blocksPerRow = K / 32;
  const wordsPerBlock = kind === "q4_0" ? 4 : 8;
  const biasBinding = hasBias
    ? `@group(0) @binding(4) var<storage, read> bias: array<f32>;` : "";
  const biasAdd = hasBias ? "accFinal = accFinal + bias[r];" : "";
  // corpo del dot per blocco: q4_0 = 16 byte con due nibble; q8_0 = 32 int8
  const blockDot = kind === "q4_0"
    ? `
      var dot_lo = 0.0; var dot_hi = 0.0;
      for (var w = 0u; w < 4u; w = w + 1u) {
        let word = qs[gb * 4u + w];
        for (var by = 0u; by < 4u; by = by + 1u) {
          let byte = (word >> (by * 8u)) & 0xffu;
          let j = w * 4u + by;                       // indice byte nel blocco (0..15)
          dot_lo = dot_lo + (f32(byte & 0xfu) - 8.0) * x[xBase + j];
          dot_hi = dot_hi + (f32(byte >> 4u) - 8.0) * x[xBase + 16u + j];
        }
      }
      let blockDot = dot_lo + dot_hi;`
    : `
      var bd = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let word = qs[gb * 8u + w];
        for (var by = 0u; by < 4u; by = by + 1u) {
          let v = (i32((word >> (by * 8u)) & 0xffu) << 24u) >> 24u; // int8 con segno
          bd = bd + f32(v) * x[xBase + w * 4u + by];
        }
      }
      let blockDot = bd;`;
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
${biasBinding}
const BLOCKS_PER_ROW = ${blocksPerRow}u;
const WORDS_PER_BLOCK = ${wordsPerBlock}u;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // Griglia 2D: una riga per workgroup, ma il limite WebGPU è 65535 wg per
  // dimensione (l'lm_head ha 151936 righe): r = x + y*GRID_X.
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= ${N}u) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var b = t; b < BLOCKS_PER_ROW; b = b + 64u) {
    let gb = r * BLOCKS_PER_ROW + b; // blocco globale nel tensore
    let sWord = scales[gb >> 1u];
    let sc = unpack2x16float(sWord)[gb & 1u];
    let xBase = b * 32u;
    ${blockDot}
    acc = acc + sc * blockDot;
  }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) {
    var accFinal = partial[0];
    ${biasAdd}
    y[r] = accFinal;
  }
}`;
}

// RMSNorm: out = x * (1/sqrt(mean(x^2)+eps)) * w. Un solo workgroup da 256.
export function rmsnormWgsl(D: number, eps: number): string {
  return `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
const D = ${D}u;
const EPS = ${eps};
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var ss = 0.0;
  for (var i = t; i < D; i = i + 256u) { ss = ss + x[i] * x[i]; }
  partial[t] = ss;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let rms = 1.0 / sqrt(partial[0] / f32(D) + EPS);
  for (var i = t; i < D; i = i + 256u) { out[i] = x[i] * rms * w[i]; }
}`;
}

// RoPE NEOX in-place: coppie (j, j+half) per head. pos dall'uniform.
export function ropeNeoxWgsl(nHead: number, headDim: number, freqBase: number): string {
  const half = headDim / 2;
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<uniform> P: TokParams;
const HALF = ${half}u;
const HEAD_DIM = ${headDim}u;
const N_PAIRS = ${nHead * half}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N_PAIRS) { return; }
  let h = i / HALF;
  let j = i % HALF;
  let theta = f32(P.pos) * pow(${freqBase}.0, -f32(j) / f32(HALF));
  let c = cos(theta);
  let s = sin(theta);
  let base = h * HEAD_DIM;
  let a = v[base + j];
  let b = v[base + j + HALF];
  v[base + j] = a * c - b * s;
  v[base + j + HALF] = a * s + b * c;
}`;
}

// Append di k/v correnti nella cache alla posizione pos (layout [ctx, kvDim] row-major).
export function kvAppendWgsl(kvDim: number): string {
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<uniform> P: TokParams;
const KV_DIM = ${kvDim}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= KV_DIM) { return; }
  cache[P.pos * KV_DIM + i] = src[i];
}`;
}

// Attention decode (GQA): un workgroup da 64 per head. Score seriali per thread,
// softmax con riduzioni in shared, poi accumulo di V. ctxMax baked (scores in
// workgroup storage).
export function attnDecodeWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number;
}): string {
  const { nHead, nKvHead, headDim, ctxMax } = opts;
  const groups = nHead / nKvHead;
  const kvDim = nKvHead * headDim;
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kCache: array<f32>;
@group(0) @binding(2) var<storage, read> vCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> P: TokParams;
const HEAD_DIM = ${headDim}u;
const KV_DIM = ${kvDim}u;
const GROUPS = ${groups}u;
const SCALE = ${1 / Math.sqrt(headDim)};
var<workgroup> scores: array<f32, ${ctxMax}>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let kvHead = h / GROUPS;
  let qOff = h * HEAD_DIM;
  let n = P.nPast + 1u;
  // 1) score per posizione (ogni thread un sottoinsieme di posizioni)
  for (var p = t; p < n; p = p + 64u) {
    let kOff = p * KV_DIM + kvHead * HEAD_DIM;
    var acc = 0.0;
    for (var i = 0u; i < HEAD_DIM; i = i + 1u) { acc = acc + q[qOff + i] * kCache[kOff + i]; }
    scores[p] = acc * SCALE;
  }
  workgroupBarrier();
  // 2) max (riduzione)
  var m = -3.0e38;
  for (var p = t; p < n; p = p + 64u) { m = max(m, scores[p]); }
  red[t] = m;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { red[t] = max(red[t], red[t + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let mAll = red[0];
  workgroupBarrier();
  // 3) exp + somma
  var s = 0.0;
  for (var p = t; p < n; p = p + 64u) {
    let e = exp(scores[p] - mAll);
    scores[p] = e;
    s = s + e;
  }
  red[t] = s;
  workgroupBarrier();
  stride = 32u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let sAll = red[0];
  workgroupBarrier();
  // 4) out[i] = Σ_p softmax(p)·V[p][i] — ogni thread una componente i
  for (var i = t; i < HEAD_DIM; i = i + 64u) {
    var acc = 0.0;
    for (var p = 0u; p < n; p = p + 1u) {
      acc = acc + scores[p] * vCache[p * KV_DIM + kvHead * HEAD_DIM + i];
    }
    out[qOff + i] = acc / sAll;
  }
}`;
}

// silu(gate)*up, in-place su gate.
export function siluMulWgsl(D: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
const D = ${D}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  let g = gate[i];
  gate[i] = (g / (1.0 + exp(-g))) * up[i];
}`;
}

// x += y (residual), in-place.
export function addInPlaceWgsl(D: number): string {
  return `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
const D = ${D}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  x[i] = x[i] + y[i];
}`;
}

// Argmax a due stadi su N logit. Stage 1: un wg da 256 copre 1024 elementi →
// (max, idx) parziale. Stage 2: un wg riduce i parziali. A parità di valore vince
// l'indice più basso (come il riferimento CPU).
export const ARGMAX_CHUNK = 1024;
export function argmaxStage1Wgsl(N: number): string {
  return `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> pmax: array<f32>;
@group(0) @binding(2) var<storage, read_write> pidx: array<u32>;
const N = ${N}u;
var<workgroup> vmax: array<f32, 256>;
var<workgroup> vidx: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let base = wid.x * ${ARGMAX_CHUNK}u;
  var m = -3.0e38;
  var mi = 0u;
  for (var i = base + t; i < min(base + ${ARGMAX_CHUNK}u, N); i = i + 256u) {
    if (x[i] > m) { m = x[i]; mi = i; }
  }
  vmax[t] = m; vidx[t] = mi;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (t < stride) {
      let a = vmax[t]; let b = vmax[t + stride];
      if (b > a || (b == a && vidx[t + stride] < vidx[t])) { vmax[t] = b; vidx[t] = vidx[t + stride]; }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { pmax[wid.x] = vmax[0]; pidx[wid.x] = vidx[0]; }
}`;
}

export function argmaxStage2Wgsl(nPartials: number): string {
  return `
@group(0) @binding(0) var<storage, read> pmax: array<f32>;
@group(0) @binding(1) var<storage, read> pidx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>; // [0] = argmax id
const NP = ${nPartials}u;
var<workgroup> vmax: array<f32, 256>;
var<workgroup> vidx: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var m = -3.0e38;
  var mi = 0u;
  for (var i = t; i < NP; i = i + 256u) {
    if (pmax[i] > m || (pmax[i] == m && pidx[i] < mi)) { m = pmax[i]; mi = pidx[i]; }
  }
  vmax[t] = m; vidx[t] = mi;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (t < stride) {
      let a = vmax[t]; let b = vmax[t + stride];
      if (b > a || (b == a && vidx[t + stride] < vidx[t])) { vmax[t] = b; vidx[t] = vidx[t + stride]; }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { out[0] = vidx[0]; }
}`;
}

// ============================== KERNEL FUSI (L3) ==============================
// Fusioni della spec §Piano statico. Il trucco comune: la rmsnorm si ricalcola
// per workgroup (riduzione su D elementi, ~gratis rispetto al dot) così la catena
// norm→matmul diventa un dispatch solo, senza buffer intermedi né barriere globali.

// Attention fusa: assorbe RoPE(q), RoPE(k_cur), append in cache (head "owner" per
// kv-head) e l'attention decode. Legge il buffer QKV concatenato [q|k|v].
export function attnFusedWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number; freqBase: number;
}): string {
  const { nHead, nKvHead, headDim, ctxMax, freqBase } = opts;
  const groups = nHead / nKvHead;
  const kvDim = nKvHead * headDim;
  const half = headDim / 2;
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qkv: array<f32>; // [${nHead * headDim} q | ${kvDim} k | ${kvDim} v]
@group(0) @binding(1) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(2) var<storage, read_write> vCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> P: TokParams;
const HEAD_DIM = ${headDim}u;
const HALF = ${half}u;
const KV_DIM = ${kvDim}u;
const GROUPS = ${groups}u;
const K_OFF = ${nHead * headDim}u;
const V_OFF = ${nHead * headDim + kvDim}u;
const SCALE = ${1 / Math.sqrt(headDim)};
var<workgroup> qh: array<f32, ${headDim}>;
var<workgroup> kh: array<f32, ${headDim}>;
var<workgroup> vh: array<f32, ${headDim}>;
var<workgroup> scores: array<f32, ${ctxMax}>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let kvHead = h / GROUPS;
  let pos = P.pos;
  // A) rope locale di q (slice del head) e k_cur (slice del kv-head); copia v_cur
  if (t < HALF) {
    let theta = f32(pos) * pow(${freqBase}.0, -f32(t) / f32(HALF));
    let c = cos(theta); let s = sin(theta);
    let qb = h * HEAD_DIM;
    let a = qkv[qb + t]; let b = qkv[qb + t + HALF];
    qh[t] = a * c - b * s;
    qh[t + HALF] = a * s + b * c;
    let kb = K_OFF + kvHead * HEAD_DIM;
    let ka = qkv[kb + t]; let kb2 = qkv[kb + t + HALF];
    kh[t] = ka * c - kb2 * s;
    kh[t + HALF] = ka * s + kb2 * c;
  }
  if (t < HEAD_DIM) { vh[t] = qkv[V_OFF + kvHead * HEAD_DIM + t]; }
  workgroupBarrier();
  // B) append in cache: solo il head "owner" del kv-head (nessuna dipendenza
  // cross-workgroup: la posizione corrente si legge dalle copie locali)
  if (h % GROUPS == 0u && t < HEAD_DIM) {
    kCache[pos * KV_DIM + kvHead * HEAD_DIM + t] = kh[t];
    vCache[pos * KV_DIM + kvHead * HEAD_DIM + t] = vh[t];
  }
  // C) score: passato dalla cache, corrente dalle copie locali
  let n = pos + 1u;
  for (var p = t; p < pos; p = p + 64u) {
    let kOff = p * KV_DIM + kvHead * HEAD_DIM;
    var acc = 0.0;
    for (var i = 0u; i < HEAD_DIM; i = i + 1u) { acc = acc + qh[i] * kCache[kOff + i]; }
    scores[p] = acc * SCALE;
  }
  if (t == 0u) {
    var acc = 0.0;
    for (var i = 0u; i < HEAD_DIM; i = i + 1u) { acc = acc + qh[i] * kh[i]; }
    scores[pos] = acc * SCALE;
  }
  workgroupBarrier();
  // D) softmax (max + somma, riduzioni)
  var m = -3.0e38;
  for (var p = t; p < n; p = p + 64u) { m = max(m, scores[p]); }
  red[t] = m;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { red[t] = max(red[t], red[t + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let mAll = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var p = t; p < n; p = p + 64u) {
    let e = exp(scores[p] - mAll);
    scores[p] = e;
    s = s + e;
  }
  red[t] = s;
  workgroupBarrier();
  stride = 32u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let sAll = red[0];
  workgroupBarrier();
  // E) out = Σ softmax·V (passato dalla cache, corrente da vh)
  for (var i = t; i < HEAD_DIM; i = i + 64u) {
    var acc = scores[pos] * vh[i];
    for (var p = 0u; p < pos; p = p + 1u) {
      acc = acc + scores[p] * vCache[p * KV_DIM + kvHead * HEAD_DIM + i];
    }
    out[h * HEAD_DIM + i] = acc / sAll;
  }
}`;
}

// lm_head veloce: rmsnorm fusa + GEMV Q8_0 con 4 righe per workgroup (16 lane/riga),
// x normalizzata in shared (letta una volta per 4 righe), load dei pesi in vec4<u32>.
// È il kernel dominante del decode (145 MB/token, ~42% del touch): la variante
// generica a 1 riga/wg lo serviva con metà thread inattivi e load scalari.
export function rmsGemvQ8FastWgsl(opts: { K: number; N: number; eps: number }): string {
  const { K, N, eps } = opts;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("rmsGemvQ8Fast: N non multiplo di 4");
  return `
@group(0) @binding(0) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> normW: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> red: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  // 1) rms + x normalizzata condivisa in vec4 (una volta per 4 righe)
  var ss = 0.0;
  for (var i = t; i < K; i = i + 64u) { ss = ss + x[i] * x[i]; }
  red[t] = ss;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let rms = 1.0 / sqrt(red[0] / f32(K) + ${eps});
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(x[i * 4u], x[i * 4u + 1u], x[i * 4u + 2u], x[i * 4u + 3u]) * rms *
             vec4(normW[i * 4u], normW[i * 4u + 1u], normW[i * 4u + 2u], normW[i * 4u + 3u]);
  }
  workgroupBarrier();
  // 2) 4 righe/wg, 16 lane/riga; int8 via unpack4x8snorm*127 (ESATTO: q8_0 mai -128)
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  var acc = 0.0;
  if (r < ${N}u) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];
      let x8 = b * 8u;
      var bd = 0.0;
      for (var half = 0u; half < 2u; half = half + 1u) {
        let w4 = qs[gb * 2u + half];
        bd = bd + dot(unpack4x8snorm(w4.x) * 127.0, xn4[x8 + half * 4u])
               + dot(unpack4x8snorm(w4.y) * 127.0, xn4[x8 + half * 4u + 1u])
               + dot(unpack4x8snorm(w4.z) * 127.0, xn4[x8 + half * 4u + 2u])
               + dot(unpack4x8snorm(w4.w) * 127.0, xn4[x8 + half * 4u + 3u]);
      }
      acc = acc + sc * bd;
    }
  }
  red[t] = acc;
  workgroupBarrier();
  // riduzione dentro le 16 lane di ciascuna riga
  stride = 8u;
  while (stride > 0u) {
    if (lane < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lane == 0u && r < ${N}u) { y[r] = red[sub * 16u]; }
}`;
}

// Pair gate/up veloce: rms fusa + 4 righe/wg + load vec4. Sostituisce
// rmsPairGemvSiluWgsl nel piano (stessa interfaccia di binding).
export function rmsPairGemvSiluFastWgsl(opts: { K: number; N: number; eps: number }): string {
  const { K, N, eps } = opts;
  const blocksPerRow = K / 32;
  return `
@group(0) @binding(0) var<storage, read> gQs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> gScales: array<u32>;
@group(0) @binding(2) var<storage, read> uQs4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> uScales: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read> normW: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> redG: array<f32, 64>;
var<workgroup> redU: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var ss = 0.0;
  for (var i = t; i < K; i = i + 64u) { ss = ss + x[i] * x[i]; }
  redG[t] = ss;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { redG[t] = redG[t] + redG[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let rms = 1.0 / sqrt(redG[0] / f32(K) + ${eps});
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(x[i * 4u], x[i * 4u + 1u], x[i * 4u + 2u], x[i * 4u + 3u]) * rms *
             vec4(normW[i * 4u], normW[i * 4u + 1u], normW[i * 4u + 2u], normW[i * 4u + 3u]);
  }
  workgroupBarrier();
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  var accG = 0.0;
  var accU = 0.0;
  if (r < ${N}u) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let x4 = b * 8u;
      {
        let w4 = gQs4[gb];
        var lo = 0.0; var hi = 0.0;
        for (var wi = 0u; wi < 4u; wi = wi + 1u) {
          let word = w4[wi];
        let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
        let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
          lo = lo + dot(nibLo, xn4[x4 + wi]);
          hi = hi + dot(nibHi, xn4[x4 + 4u + wi]);
        }
        accG = accG + unpack2x16float(gScales[gb >> 1u])[gb & 1u] * (lo + hi);
      }
      {
        let w4 = uQs4[gb];
        var lo = 0.0; var hi = 0.0;
        for (var wi = 0u; wi < 4u; wi = wi + 1u) {
          let word = w4[wi];
        let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
        let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
          lo = lo + dot(nibLo, xn4[x4 + wi]);
          hi = hi + dot(nibHi, xn4[x4 + 4u + wi]);
        }
        accU = accU + unpack2x16float(uScales[gb >> 1u])[gb & 1u] * (lo + hi);
      }
    }
  }
  redG[t] = accG;
  redU[t] = accU;
  workgroupBarrier();
  stride = 8u;
  while (stride > 0u) {
    if (lane < stride) { redG[t] = redG[t] + redG[t + stride]; redU[t] = redU[t] + redU[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lane == 0u && r < ${N}u) {
    let g = redG[sub * 16u];
    out[r] = (g / (1.0 + exp(-g))) * redU[sub * 16u];
  }
}`;
}

// GEMV+residual veloce: 4 righe/wg, load vec4, xin in shared.
export function gemvResidualFastWgsl(opts: { K: number; N: number }): string {
  const { K, N } = opts;
  const blocksPerRow = K / 32;
  return `
@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> xin: array<f32>;
@group(0) @binding(3) var<storage, read_write> xres: array<f32>;
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> red: array<f32, 64>;
var<workgroup> xs4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  for (var i = t; i < K / 4u; i = i + 64u) {
    xs4[i] = vec4(xin[i * 4u], xin[i * 4u + 1u], xin[i * 4u + 2u], xin[i * 4u + 3u]);
  }
  workgroupBarrier();
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  var acc = 0.0;
  if (r < ${N}u) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let x4 = b * 8u;
      let w4 = qs4[gb];
      var lo = 0.0; var hi = 0.0;
      for (var wi = 0u; wi < 4u; wi = wi + 1u) {
        let word = w4[wi];
        let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
        let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
        lo = lo + dot(nibLo, xs4[x4 + wi]);
        hi = hi + dot(nibHi, xs4[x4 + 4u + wi]);
      }
      acc = acc + unpack2x16float(scales[gb >> 1u])[gb & 1u] * (lo + hi);
    }
  }
  red[t] = acc;
  workgroupBarrier();
  var stride = 8u;
  while (stride > 0u) {
    if (lane < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lane == 0u && r < ${N}u) { xres[r] = xres[r] + red[sub * 16u]; }
}`;
}

// ======================= KERNEL PREFILL MULTI-TOKEN (B1) =======================
// Percorso chunk M≤8 (spec B1 §Forward multi-token). Stato dinamico per chunk:
// l'uniform ChunkParams {posBase, rows} — rows < M solo sull'ultimo chunk parziale,
// e OGNI kernel maschera m >= rows (le righe oltre non devono toccare cache/output).
// M (mMax) è baked come le altre shape; il valore vive in prefillplan.ts.

export const CHUNK_PARAMS_WGSL = `struct ChunkParams { posBase: u32, rows: u32 };`;

// RMSNorm per-riga su un chunk [M×D]: un workgroup per riga (grid = (mMax, 1)).
export function rmsnormChunkWgsl(D: number, eps: number, mMax: number): string {
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> C: ChunkParams;
const D = ${D}u;
const EPS = ${eps};
const M_MAX = ${mMax}u;
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wid.x;
  if (m >= C.rows) { return; } // uniforme per workgroup: barriere sotto ok
  let off = m * D;
  let t = lid.x;
  var ss = 0.0;
  for (var i = t; i < D; i = i + 256u) { ss = ss + x[off + i] * x[off + i]; }
  partial[t] = ss;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let rms = 1.0 / sqrt(partial[0] / f32(D) + EPS);
  for (var i = t; i < D; i = i + 256u) { out[off + i] = x[off + i] * rms * w[i]; }
}`;
}

// GEMM small-batch dequant-fusa: y[m][r] = Σ_b scale(b)·Σ_j q_j·x[m][...] (+bias /
// +residual). Un workgroup da 64 per RIGA DI PESO r: la riga si legge una volta e
// serve le M attivazioni (banda pesi /M — la tesi della spec). x del chunk in
// shared quando ci sta (K=896: 28.7 KB ≤ limite 32 KB richiesto al device);
// per K=4864 (down-proj) si legge da storage (broadcast fra wg ⇒ L2).
export function gemmQuantChunkWgsl(opts: {
  kind: "q4_0" | "q8_0"; K: number; N: number; mMax: number;
  hasBias: boolean; residual: boolean;
}): string {
  const { kind, K, N, mMax, hasBias, residual } = opts;
  if (K % 32 !== 0) throw new Error("gemmChunk: K non multiplo di 32");
  if (hasBias && residual) throw new Error("gemmChunk: bias+residual non previsto");
  const blocksPerRow = K / 32;
  const useShared = mMax * K * 4 <= 28672;
  const biasBinding = hasBias ? `@group(0) @binding(4) var<storage, read> bias: array<f32>;` : "";
  const uniBinding = hasBias ? 5 : 4;
  const xsDecl = useShared ? `var<workgroup> xs: array<f32, ${mMax * K}>;` : "";
  // zero-init di var<workgroup> garantita da WGSL: le righe m >= rows leggono 0
  const xsLoad = useShared
    ? `for (var i = t; i < C.rows * ${K}u; i = i + 64u) { xs[i] = x[i]; }
  workgroupBarrier();`
    : "";
  const getx = useShared
    ? `fn getx(m: u32, i: u32) -> f32 { return xs[m * ${K}u + i]; }`
    : `fn getx(m: u32, i: u32) -> f32 { return x[m * ${K}u + i]; }`;
  const blockDot = kind === "q4_0"
    ? `
      for (var w = 0u; w < 4u; w = w + 1u) {
        let word = qs[gb * 4u + w];
        for (var by = 0u; by < 4u; by = by + 1u) {
          let byte = (word >> (by * 8u)) & 0xffu;
          let j = w * 4u + by;
          let vLo = f32(byte & 0xfu) - 8.0;
          let vHi = f32(byte >> 4u) - 8.0;
          for (var m = 0u; m < M_MAX; m = m + 1u) {
            bd[m] = bd[m] + vLo * getx(m, xBase + j) + vHi * getx(m, xBase + 16u + j);
          }
        }
      }`
    : `
      for (var w = 0u; w < 8u; w = w + 1u) {
        let word = qs[gb * 8u + w];
        for (var by = 0u; by < 4u; by = by + 1u) {
          let v = f32((i32((word >> (by * 8u)) & 0xffu) << 24u) >> 24u);
          let j = w * 4u + by;
          for (var m = 0u; m < M_MAX; m = m + 1u) {
            bd[m] = bd[m] + v * getx(m, xBase + j);
          }
        }
      }`;
  const writeOut = residual
    ? `y[m * ${N}u + r] = y[m * ${N}u + r] + partial[m * 64u];`
    : hasBias
      ? `y[m * ${N}u + r] = partial[m * 64u] + bias[r];`
      : `y[m * ${N}u + r] = partial[m * 64u];`;
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
${biasBinding}
@group(0) @binding(${uniBinding}) var<uniform> C: ChunkParams;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
const M_MAX = ${mMax}u;
${xsDecl}
var<workgroup> partial: array<f32, ${64 * mMax}>;
${getx}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  let t = lid.x;
  ${xsLoad}
  var acc: array<f32, M_MAX>;
  var bd: array<f32, M_MAX>;
  if (r < ${N}u) {
    for (var b = t; b < BLOCKS_PER_ROW; b = b + 64u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let sWord = scales[gb >> 1u];
      let sc = unpack2x16float(sWord)[gb & 1u];
      let xBase = b * 32u;
      // azzeramento ESPLICITO per iterazione: il decl-in-loop contava sull'azzerato
      // implicito, che su Chrome/Tint NON avviene a ogni giro (bug trovato in fase 3:
      // con >64 blocchi/riga le iterazioni successive ereditavano bd della precedente)
      for (var m = 0u; m < M_MAX; m = m + 1u) { bd[m] = 0.0; }
      ${blockDot}
      for (var m = 0u; m < M_MAX; m = m + 1u) { acc[m] = acc[m] + sc * bd[m]; }
    }
  }
  for (var m = 0u; m < M_MAX; m = m + 1u) { partial[m * 64u + t] = acc[m]; }
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) {
      for (var m = 0u; m < M_MAX; m = m + 1u) {
        partial[m * 64u + t] = partial[m * 64u + t] + partial[m * 64u + t + stride];
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u && r < ${N}u) {
    for (var m = 0u; m < C.rows; m = m + 1u) { ${writeOut} }
  }
}`;
}

// RoPE NEOX in-place sul buffer QKV di chunk [M×qkvDim], pos per-riga = posBase+m.
// Copre q e k in un dispatch: coppie 0..QPAIRS-1 = q, il resto = k.
export function ropeChunkWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; freqBase: number; mMax: number;
}): string {
  const { nHead, nKvHead, headDim, freqBase, mMax } = opts;
  const half = headDim / 2;
  const qkvDim = (nHead + 2 * nKvHead) * headDim;
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<uniform> C: ChunkParams;
const HALF = ${half}u;
const HEAD_DIM = ${headDim}u;
const Q_PAIRS = ${nHead * half}u;
const TOTAL_PAIRS = ${(nHead + nKvHead) * half}u;
const QKV_DIM = ${qkvDim}u;
const K_OFF = ${nHead * headDim}u;
const M_MAX = ${mMax}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let m = gid.y;
  if (i >= TOTAL_PAIRS || m >= C.rows) { return; }
  var h: u32; var j: u32; var base: u32;
  if (i < Q_PAIRS) {
    h = i / HALF; j = i % HALF;
    base = m * QKV_DIM + h * HEAD_DIM;
  } else {
    let ik = i - Q_PAIRS;
    h = ik / HALF; j = ik % HALF;
    base = m * QKV_DIM + K_OFF + h * HEAD_DIM;
  }
  let theta = f32(C.posBase + m) * pow(${freqBase}.0, -f32(j) / f32(HALF));
  let c = cos(theta);
  let s = sin(theta);
  let a = qkv[base + j];
  let b = qkv[base + j + HALF];
  qkv[base + j] = a * c - b * s;
  qkv[base + j + HALF] = a * s + b * c;
}`;
}

// Append delle M righe k/v del chunk nelle cache, un dispatch per layer:
// cache[(posBase+m)*KV_DIM + i] = qkv[m][K_OFF/V_OFF + i].
export function kvAppendChunkWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; mMax: number;
}): string {
  const { nHead, nKvHead, headDim, mMax } = opts;
  const kvDim = nKvHead * headDim;
  const qkvDim = (nHead + 2 * nKvHead) * headDim;
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(2) var<storage, read_write> vCache: array<f32>;
@group(0) @binding(3) var<uniform> C: ChunkParams;
const KV_DIM = ${kvDim}u;
const QKV_DIM = ${qkvDim}u;
const K_OFF = ${nHead * headDim}u;
const V_OFF = ${nHead * headDim + kvDim}u;
const M_MAX = ${mMax}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let m = gid.y;
  if (i >= KV_DIM || m >= C.rows) { return; }
  let pos = C.posBase + m;
  kCache[pos * KV_DIM + i] = qkv[m * QKV_DIM + K_OFF + i];
  vCache[pos * KV_DIM + i] = qkv[m * QKV_DIM + V_OFF + i];
}`;
}

// Attention di chunk (GQA): un workgroup da 64 per (head, riga m); maschera causale
// intra-chunk: la riga m vede KV [0, posBase+m] (le righe del chunk sono GIÀ in
// cache via kvAppendChunk; le righe future del chunk esistono in cache ma n le
// esclude). q dal buffer QKV post-rope, out [M×nHead*headDim].
export function attnPrefillChunkWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number; mMax: number;
}): string {
  const { nHead, nKvHead, headDim, ctxMax, mMax } = opts;
  const groups = nHead / nKvHead;
  const kvDim = nKvHead * headDim;
  const qkvDim = (nHead + 2 * nKvHead) * headDim;
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> kCache: array<f32>;
@group(0) @binding(2) var<storage, read> vCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> C: ChunkParams;
const HEAD_DIM = ${headDim}u;
const KV_DIM = ${kvDim}u;
const GROUPS = ${groups}u;
const Q_DIM = ${nHead * headDim}u;
const QKV_DIM = ${qkvDim}u;
const SCALE = ${1 / Math.sqrt(headDim)};
const M_MAX = ${mMax}u;
var<workgroup> qh: array<f32, ${headDim}>;
var<workgroup> scores: array<f32, ${ctxMax}>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let m = wid.y;
  if (m >= C.rows) { return; } // uniforme per workgroup: barriere sotto ok
  let t = lid.x;
  let kvHead = h / GROUPS;
  let n = C.posBase + m + 1u; // causale: la riga m vede [0, posBase+m]
  if (t < HEAD_DIM) { qh[t] = qkv[m * QKV_DIM + h * HEAD_DIM + t]; }
  workgroupBarrier();
  for (var p = t; p < n; p = p + 64u) {
    let kOff = p * KV_DIM + kvHead * HEAD_DIM;
    var acc = 0.0;
    for (var i = 0u; i < HEAD_DIM; i = i + 1u) { acc = acc + qh[i] * kCache[kOff + i]; }
    scores[p] = acc * SCALE;
  }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var p = t; p < n; p = p + 64u) { mx = max(mx, scores[p]); }
  red[t] = mx;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { red[t] = max(red[t], red[t + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let mAll = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var p = t; p < n; p = p + 64u) {
    let e = exp(scores[p] - mAll);
    scores[p] = e;
    s = s + e;
  }
  red[t] = s;
  workgroupBarrier();
  stride = 32u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let sAll = red[0];
  workgroupBarrier();
  for (var i = t; i < HEAD_DIM; i = i + 64u) {
    var acc = 0.0;
    for (var p = 0u; p < n; p = p + 1u) {
      acc = acc + scores[p] * vCache[p * KV_DIM + kvHead * HEAD_DIM + i];
    }
    out[m * Q_DIM + h * HEAD_DIM + i] = acc / sAll;
  }
}`;
}

// silu(gate)*up per chunk [M×D], in-place su gate.
export function siluMulChunkWgsl(D: number, mMax: number): string {
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> C: ChunkParams;
const D = ${D}u;
const M_MAX = ${mMax}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let m = gid.y;
  if (i >= D || m >= C.rows) { return; }
  let g = gate[m * D + i];
  gate[m * D + i] = (g / (1.0 + exp(-g))) * up[m * D + i];
}`;
}

// QKV veloce: rms fusa + GEMV Q4_0 con bias, 4 righe/wg, load vec4 (stessa
// architettura del pair, un solo accumulatore + bias).
export function rmsGemvQ4FastBiasWgsl(opts: { K: number; N: number; eps: number }): string {
  const { K, N, eps } = opts;
  const blocksPerRow = K / 32;
  return `
@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> normW: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> red: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var ss = 0.0;
  for (var i = t; i < K; i = i + 64u) { ss = ss + x[i] * x[i]; }
  red[t] = ss;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let rms = 1.0 / sqrt(red[0] / f32(K) + ${eps});
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(x[i * 4u], x[i * 4u + 1u], x[i * 4u + 2u], x[i * 4u + 3u]) * rms *
             vec4(normW[i * 4u], normW[i * 4u + 1u], normW[i * 4u + 2u], normW[i * 4u + 3u]);
  }
  workgroupBarrier();
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  var acc = 0.0;
  if (r < ${N}u) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let x4 = b * 8u;
      let w4 = qs4[gb];
      var lo = 0.0; var hi = 0.0;
      for (var wi = 0u; wi < 4u; wi = wi + 1u) {
        let word = w4[wi];
        let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
        let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
        lo = lo + dot(nibLo, xn4[x4 + wi]);
        hi = hi + dot(nibHi, xn4[x4 + 4u + wi]);
      }
      acc = acc + unpack2x16float(scales[gb >> 1u])[gb & 1u] * (lo + hi);
    }
  }
  red[t] = acc;
  workgroupBarrier();
  stride = 8u;
  while (stride > 0u) {
    if (lane < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lane == 0u && r < ${N}u) { y[r] = red[sub * 16u] + bias[r]; }
}`;
}
