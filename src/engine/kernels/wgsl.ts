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
  kind: "q4_0" | "q4_1" | "q8_0"; K: number; N: number; hasBias: boolean;
  // MoE (C2 fase 5): y[r] += accScale[0]·dot — il down per-expert accumula il
  // contributo pesato direttamente su moe_out (pesatura DOPO il down, come
  // ffn_moe_weighted in build_moe_ffn; l'ordine delle somme sui 4 expert
  // differisce dall'oracolo solo al rounding f32).
  scaledAccum?: boolean;
}): string {
  const { kind, K, N, hasBias, scaledAccum } = opts;
  if (K % 32 !== 0) throw new Error("gemv: K non multiplo di 32");
  const blocksPerRow = K / 32;
  const wordsPerBlock = kind === "q8_0" ? 8 : 4;
  const biasBinding = hasBias
    ? `@group(0) @binding(4) var<storage, read> bias: array<f32>;` : "";
  const accBinding = scaledAccum
    ? `@group(0) @binding(${hasBias ? 5 : 4}) var<storage, read> accScale: array<f32>;` : "";
  const biasAdd = hasBias ? "accFinal = accFinal + bias[r];" : "";
  const writeY = scaledAccum ? "y[r] = y[r] + accScale[0] * accFinal;" : "y[r] = accFinal;";
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
    : kind === "q4_1"
    ? `
      // q4_1: w = q*d + m ⇒ contributo blocco = d·Σ(q·x) + m·Σx.
      var dot_q = 0.0; var sum_x = 0.0;
      for (var w = 0u; w < 4u; w = w + 1u) {
        let word = qs[gb * 4u + w];
        for (var by = 0u; by < 4u; by = by + 1u) {
          let byte = (word >> (by * 8u)) & 0xffu;
          let j = w * 4u + by;
          let xlo = x[xBase + j];
          let xhi = x[xBase + 16u + j];
          dot_q = dot_q + f32(byte & 0xfu) * xlo + f32(byte >> 4u) * xhi;
          sum_x = sum_x + xlo + xhi;
        }
      }`
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
  // q4_1: "scales" = un u32/blocco con (d, m) → accumulo dedicato
  const scaleAcc = kind === "q4_1"
    ? `
    let dm = unpack2x16float(scales[gb]);
    let xBase = b * 32u;
    ${blockDot}
    acc = acc + dm.x * dot_q + dm.y * sum_x;`
    : `
    let sWord = scales[gb >> 1u];
    let sc = unpack2x16float(sWord)[gb & 1u];
    let xBase = b * 32u;
    ${blockDot}
    acc = acc + sc * blockDot;`;
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
${biasBinding}
${accBinding}
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
    ${scaleAcc}
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
    ${writeY}
  }
}`;
}

// GEMV f32 puro: il router MoE (ffn_gate_inp) è F32 nel GGUF (spec C2 §1).
// Stessa griglia/riduzione dei gemv quant; w row-major [N righe × K].
export function gemvF32Wgsl(opts: { K: number; N: number }): string {
  const { K, N } = opts;
  return `
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= ${N}u) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var i = t; i < ${K}u; i = i + 64u) { acc = acc + w[r * ${K}u + i] * x[i]; }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { y[r] = partial[0]; }
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

// ---- GEMM small-batch FAST (architettura dei kernel fusi di fase A, per chunk) ----
// 4 righe di peso per workgroup, 16 lane/riga, load vec4<u32>; gli M accumulatori
// sono SCALARI GENERATI dal template (acc0..accM-1): niente array privati
// indicizzati, che su NVIDIA finiscono in scratch memory — la prima versione
// (1 riga/wg, array acc[m]/bd[m], load scalari) faceva ~6.6 ms/token di prefill,
// PEGGIO della baseline sequenziale; la variante vec4 1-riga/wg ~2.35 ms/token;
// questa architettura (fusioni + 4 righe/wg) serve la soglia 3× di spec.
// La riga di peso si legge una volta e serve le M attivazioni (banda pesi /M).
// x del chunk in shared vec4 dove ci sta (K=896: 28.7 KB ≤ 32 KB richiesti);
// down-proj (K=4864) legge da storage (broadcast fra wg ⇒ L2). Solo q4_0: la
// variante q8_0 M-colonne non ha consumer (lm_head solo sull'ultima posizione).

const mRange = (mMax: number) => Array.from({ length: mMax }, (_, m) => m);

// Preambolo condiviso: load di x del chunk in shared vec4 (+ rms per riga fusa,
// normalizzazione in place — le righe m ≥ rows restano a zero e sono mascherate
// in scrittura). Usa partial[0..63] come scratch di riduzione.
function chunkXsPreamble(opts: { v4PerRow: number; rms: boolean; K: number; eps?: number }): string {
  const { v4PerRow, rms, K, eps } = opts;
  const load = `for (var i = t; i < C.rows * ${v4PerRow}u; i = i + 64u) { xs4[i] = xv4[i]; }
  workgroupBarrier();`;
  if (!rms) return load;
  return `${load}
  for (var mr = 0u; mr < C.rows; mr = mr + 1u) { // bound uniforme: barriere legali
    var ss = 0.0;
    for (var i = t; i < ${v4PerRow}u; i = i + 64u) { let v = xs4[mr * ${v4PerRow}u + i]; ss = ss + dot(v, v); }
    partial[t] = ss;
    workgroupBarrier();
    var rstride = 32u;
    while (rstride > 0u) {
      if (t < rstride) { partial[t] = partial[t] + partial[t + rstride]; }
      workgroupBarrier();
      rstride = rstride >> 1u;
    }
    let rms = 1.0 / sqrt(partial[0] / ${K}.0 + ${eps});
    workgroupBarrier();
    for (var i = t; i < ${v4PerRow}u; i = i + 64u) {
      xs4[mr * ${v4PerRow}u + i] = xs4[mr * ${v4PerRow}u + i] * rms *
        vec4(normW[i * 4u], normW[i * 4u + 1u], normW[i * 4u + 2u], normW[i * 4u + 3u]);
    }
    workgroupBarrier();
  }`;
}

// Corpo del dot per blocco q4_0 (load vec4, nibble → vec4, dot con xs4/xv4).
function chunkBlockDot(ms: number[], xsExpr: (m: number, idx: string) => string, qsName: string, bd: string): string {
  return `let w4 = ${qsName}[gb];
      ${ms.map((m) => `${bd}${m} = 0.0;`).join(" ")}
      for (var wi = 0u; wi < 4u; wi = wi + 1u) {
        let word = w4[wi];
        let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
        let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
        ${ms.map((m) => `${bd}${m} = ${bd}${m} + dot(nibLo, ${xsExpr(m, "x4 + wi")}) + dot(nibHi, ${xsExpr(m, "x4 + 4u + wi")});`).join("\n        ")}
      }`;
}

// Riduzione fra le 16 lane di ciascuna (riga, m): partial[m*64 + t]. In un blocco
// a scope proprio: pairSilu la emette DUE volte nello stesso body e WGSL vieta la
// ridichiarazione di `stride` nello stesso scope (trovato in fase 6: il modulo non
// compilava e Dawn droppava in silenzio ogni submit con la pipeline invalida).
const chunkLaneReduce = (ms: number[], acc: string) => `${ms.map((m) => `partial[${m * 64}u + t] = ${acc}${m};`).join(" ")}
  workgroupBarrier();
  {
    var stride = 8u;
    while (stride > 0u) {
      if (lane < stride) {
        for (var m = 0u; m < M_MAX; m = m + 1u) {
          partial[m * 64u + t] = partial[m * 64u + t] + partial[m * 64u + t + stride];
        }
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
  }`;

// QKV di chunk: rms fusa + GEMM Q4_0 con bias.
export function rmsGemmQkvChunkFastWgsl(opts: { K: number; N: number; eps: number; mMax: number }): string {
  const { K, N, eps, mMax } = opts;
  const v4PerRow = K / 4;
  const ms = mRange(mMax);
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> xv4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> normW: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<uniform> C: ChunkParams;
const BLOCKS_PER_ROW = ${K / 32}u;
const M_MAX = ${mMax}u;
var<workgroup> xs4: array<vec4<f32>, ${mMax * v4PerRow}>;
var<workgroup> partial: array<f32, ${64 * mMax}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  ${chunkXsPreamble({ v4PerRow, rms: true, K, eps })}
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  ${ms.map((m) => `var acc${m} = 0.0; var bd${m} = 0.0;`).join(" ")}
  if (r < ${N}u) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];
      let x4 = b * 8u;
      ${chunkBlockDot(ms, (m, idx) => `xs4[${m * v4PerRow}u + ${idx}]`, "qs4", "bd")}
      ${ms.map((m) => `acc${m} = acc${m} + sc * bd${m};`).join(" ")}
    }
  }
  ${chunkLaneReduce(ms, "acc")}
  if (lane == 0u && r < ${N}u) {
    for (var m = 0u; m < C.rows; m = m + 1u) { y[m * ${N}u + r] = partial[m * 64u + sub * 16u] + bias[r]; }
  }
}`;
}

// Pair gate/up di chunk: rms fusa + due GEMM + silu, un dispatch (come pairSilu decode).
export function rmsPairGemmSiluChunkFastWgsl(opts: { K: number; N: number; eps: number; mMax: number }): string {
  const { K, N, eps, mMax } = opts;
  const v4PerRow = K / 4;
  const ms = mRange(mMax);
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> gQs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> gScales: array<u32>;
@group(0) @binding(2) var<storage, read> uQs4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> uScales: array<u32>;
@group(0) @binding(4) var<storage, read> xv4: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> normW: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> C: ChunkParams;
const BLOCKS_PER_ROW = ${K / 32}u;
const M_MAX = ${mMax}u;
var<workgroup> xs4: array<vec4<f32>, ${mMax * v4PerRow}>;
var<workgroup> partial: array<f32, ${64 * mMax}>;
var<workgroup> gRes: array<f32, ${4 * mMax}>; // g ridotto, per (m, sub)
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  ${chunkXsPreamble({ v4PerRow, rms: true, K, eps })}
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  ${ms.map((m) => `var accG${m} = 0.0; var accU${m} = 0.0; var bdG${m} = 0.0; var bdU${m} = 0.0;`).join(" ")}
  if (r < ${N}u) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let x4 = b * 8u;
      {
        ${chunkBlockDot(ms, (m, idx) => `xs4[${m * v4PerRow}u + ${idx}]`, "gQs4", "bdG")}
        let sc = unpack2x16float(gScales[gb >> 1u])[gb & 1u];
        ${ms.map((m) => `accG${m} = accG${m} + sc * bdG${m};`).join(" ")}
      }
      {
        ${chunkBlockDot(ms, (m, idx) => `xs4[${m * v4PerRow}u + ${idx}]`, "uQs4", "bdU")}
        let sc = unpack2x16float(uScales[gb >> 1u])[gb & 1u];
        ${ms.map((m) => `accU${m} = accU${m} + sc * bdU${m};`).join(" ")}
      }
    }
  }
  ${chunkLaneReduce(ms, "accG")}
  if (lane == 0u) {
    for (var m = 0u; m < M_MAX; m = m + 1u) { gRes[m * 4u + sub] = partial[m * 64u + sub * 16u]; }
  }
  workgroupBarrier();
  ${chunkLaneReduce(ms, "accU")}
  if (lane == 0u && r < ${N}u) {
    for (var m = 0u; m < C.rows; m = m + 1u) {
      let g = gRes[m * 4u + sub];
      out[m * ${N}u + r] = (g / (1.0 + exp(-g))) * partial[m * 64u + sub * 16u];
    }
  }
}`;
}

// GEMM+residual di chunk (o-proj e down-proj): niente rms; shared se K ci sta.
export function gemmResidChunkFastWgsl(opts: { K: number; N: number; mMax: number }): string {
  const { K, N, mMax } = opts;
  const v4PerRow = K / 4;
  const ms = mRange(mMax);
  const useShared = mMax * K * 4 <= 28672;
  const xsDecl = useShared ? `var<workgroup> xs4: array<vec4<f32>, ${mMax * v4PerRow}>;` : "";
  const xsLoad = useShared ? chunkXsPreamble({ v4PerRow, rms: false, K }) : "";
  const xsExpr = (m: number, idx: string) => useShared
    ? `xs4[${m * v4PerRow}u + ${idx}]`
    : `xv4[${m * v4PerRow}u + ${idx}]`;
  return `${CHUNK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> xv4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> C: ChunkParams;
const BLOCKS_PER_ROW = ${K / 32}u;
const M_MAX = ${mMax}u;
${xsDecl}
var<workgroup> partial: array<f32, ${64 * mMax}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  ${xsLoad}
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  ${ms.map((m) => `var acc${m} = 0.0; var bd${m} = 0.0;`).join(" ")}
  if (r < ${N}u) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];
      let x4 = b * 8u;
      ${chunkBlockDot(ms, xsExpr, "qs4", "bd")}
      ${ms.map((m) => `acc${m} = acc${m} + sc * bd${m};`).join(" ")}
    }
  }
  ${chunkLaneReduce(ms, "acc")}
  if (lane == 0u && r < ${N}u) {
    for (var m = 0u; m < C.rows; m = m + 1u) { y[m * ${N}u + r] = y[m * ${N}u + r] + partial[m * 64u + sub * 16u]; }
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

// Attention decode SPLIT sul contesto (fase B2, stile flash-decoding) — sostituisce
// il collo di attnFusedWgsl al crescere di kvLen: quel kernel ha nHead workgroup
// totali (GPU quasi vuota) e nella fase output ogni thread itera SEQUENZIALMENTE
// su tutto il contesto (~pos iterazioni). Qui il contesto è partizionato in blocchi
// da CHUNK posizioni: griglia FISSA (nHead, sMax) — il piano statico resta
// immutabile, le partizioni oltre n escono subito — pass 1 scrive per partizione
// {max locale m, somma locale l, out parziale non normalizzato} nel buffer
// partials [nHead × sMax × (headDim+2)], pass 2 (attnSplitReduceWgsl) combina in
// log-sum-exp esatto. La partizione che contiene pos fa anche rope di k_cur e
// append in cache (stesso owner-rule di attnFusedWgsl: nessuna dipendenza
// cross-workgroup, il corrente si legge dalle copie locali).
export function attnSplitPartWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number; freqBase: number; chunkP: number;
}): string {
  const { nHead, nKvHead, headDim, ctxMax, freqBase, chunkP } = opts;
  const groups = nHead / nKvHead;
  const kvDim = nKvHead * headDim;
  const half = headDim / 2;
  const sMax = Math.ceil(ctxMax / chunkP);
  if (chunkP !== 64) throw new Error("attnSplitPartWgsl: chunkP=64 assunto (1 posizione/thread)");
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> qkv: array<f32>; // [${nHead * headDim} q | ${kvDim} k | ${kvDim} v]
@group(0) @binding(1) var<storage, read_write> kCache: array<f32>;
@group(0) @binding(2) var<storage, read_write> vCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>; // [head][part][${headDim} out | m | l]
@group(0) @binding(4) var<uniform> P: TokParams;
const HEAD_DIM = ${headDim}u;
const HALF = ${half}u;
const KV_DIM = ${kvDim}u;
const GROUPS = ${groups}u;
const K_OFF = ${nHead * headDim}u;
const V_OFF = ${nHead * headDim + kvDim}u;
const CHUNK = ${chunkP}u;
const S_MAX = ${sMax}u;
const PART_STRIDE = ${headDim + 2}u;
const SCALE = ${1 / Math.sqrt(headDim)};
var<workgroup> qh: array<f32, ${headDim}>;
var<workgroup> kh: array<f32, ${headDim}>;
var<workgroup> vh: array<f32, ${headDim}>;
var<workgroup> ex: array<f32, ${chunkP}>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let part = wid.y;
  let t = lid.x;
  let pos = P.pos;
  let n = pos + 1u;
  let begin = part * CHUNK;
  if (begin >= n) { return; } // partizione oltre il contesto: griglia fissa, uscita immediata
  let end = min(begin + CHUNK, n);
  let kvHead = h / GROUPS;
  let isOwner = pos >= begin && pos < begin + CHUNK; // partizione del token corrente
  // A) rope locale di q (ogni wg: costo HALF, replicato per partizione); la sola
  // partizione owner fa anche rope di k_cur + copia v_cur
  if (t < HALF) {
    let theta = f32(pos) * pow(${freqBase}.0, -f32(t) / f32(HALF));
    let c = cos(theta); let s = sin(theta);
    let qb = h * HEAD_DIM;
    let a = qkv[qb + t]; let b = qkv[qb + t + HALF];
    qh[t] = a * c - b * s;
    qh[t + HALF] = a * s + b * c;
    if (isOwner) {
      let kb = K_OFF + kvHead * HEAD_DIM;
      let ka = qkv[kb + t]; let kb2 = qkv[kb + t + HALF];
      kh[t] = ka * c - kb2 * s;
      kh[t + HALF] = ka * s + kb2 * c;
    }
  }
  if (isOwner && t < HEAD_DIM) { vh[t] = qkv[V_OFF + kvHead * HEAD_DIM + t]; }
  workgroupBarrier();
  // B) append in cache: owner-rule di attnFusedWgsl (un solo wg per kv-head scrive)
  if (isOwner && h % GROUPS == 0u && t < HEAD_DIM) {
    kCache[pos * KV_DIM + kvHead * HEAD_DIM + t] = kh[t];
    vCache[pos * KV_DIM + kvHead * HEAD_DIM + t] = vh[t];
  }
  // C) score della posizione begin+t (1 posizione/thread): passato dalla cache,
  // corrente dalle copie locali (mai visibile cross-wg dentro il dispatch)
  let p = begin + t;
  var sc = -3.0e38;
  if (p < end) {
    var acc = 0.0;
    if (p == pos) {
      for (var i = 0u; i < HEAD_DIM; i = i + 1u) { acc = acc + qh[i] * kh[i]; }
    } else {
      let kOff = p * KV_DIM + kvHead * HEAD_DIM;
      for (var i = 0u; i < HEAD_DIM; i = i + 1u) { acc = acc + qh[i] * kCache[kOff + i]; }
    }
    sc = acc * SCALE;
  }
  // D) max locale (riduzione) + exp condivise
  red[t] = sc;
  workgroupBarrier();
  {
    var stride = 32u;
    while (stride > 0u) {
      if (t < stride) { red[t] = max(red[t], red[t + stride]); }
      workgroupBarrier();
      stride = stride >> 1u;
    }
  }
  let m = red[0];
  workgroupBarrier();
  var e = 0.0;
  if (p < end) { e = exp(sc - m); }
  ex[t] = e;
  red[t] = e;
  workgroupBarrier();
  {
    var stride = 32u;
    while (stride > 0u) {
      if (t < stride) { red[t] = red[t] + red[t + stride]; }
      workgroupBarrier();
      stride = stride >> 1u;
    }
  }
  let l = red[0];
  workgroupBarrier();
  // E) out parziale NON normalizzato: thread t = dimensione t, loop sulle sole
  // posizioni della partizione (≤CHUNK, vs tutto il contesto di attnFusedWgsl)
  let base = (h * S_MAX + part) * PART_STRIDE;
  if (t < HEAD_DIM) {
    var acc = 0.0;
    for (var pp = begin; pp < end; pp = pp + 1u) {
      if (pp == pos) {
        acc = acc + ex[pp - begin] * vh[t];
      } else {
        acc = acc + ex[pp - begin] * vCache[pp * KV_DIM + kvHead * HEAD_DIM + t];
      }
    }
    partials[base + t] = acc;
  }
  if (t == 0u) {
    partials[base + HEAD_DIM] = m;
    partials[base + HEAD_DIM + 1u] = l;
  }
}`;
}

// Pass 2 dello split: combina le partizioni in log-sum-exp esatto (stessa
// matematica di una softmax monolitica: M=max m_p, L=Σ l_p·exp(m_p−M),
// out=Σ o_p·exp(m_p−M)/L). Griglia (nHead,1), lavoro minuscolo.
export function attnSplitReduceWgsl(opts: {
  nHead: number; headDim: number; ctxMax: number; chunkP: number;
}): string {
  const { headDim, ctxMax, chunkP } = opts;
  const sMax = Math.ceil(ctxMax / chunkP);
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> P: TokParams;
const HEAD_DIM = ${headDim}u;
const S_MAX = ${sMax}u;
const CHUNK = ${chunkP}u;
const PART_STRIDE = ${headDim + 2}u;
var<workgroup> mAll: f32;
var<workgroup> lAll: f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let nParts = P.pos / CHUNK + 1u; // n = pos+1 ⇒ ceil(n/CHUNK)
  let base = h * S_MAX * PART_STRIDE;
  // M e L su un solo thread (nParts ≤ ${sMax}: riduzione seriale più corta del
  // costo di una riduzione parallela)
  if (t == 0u) {
    var m = -3.0e38;
    for (var p2 = 0u; p2 < nParts; p2 = p2 + 1u) { m = max(m, partials[base + p2 * PART_STRIDE + HEAD_DIM]); }
    var l = 0.0;
    for (var p2 = 0u; p2 < nParts; p2 = p2 + 1u) {
      l = l + partials[base + p2 * PART_STRIDE + HEAD_DIM + 1u] * exp(partials[base + p2 * PART_STRIDE + HEAD_DIM] - m);
    }
    mAll = m;
    lAll = l;
  }
  workgroupBarrier();
  if (t < HEAD_DIM) {
    var acc = 0.0;
    for (var p2 = 0u; p2 < nParts; p2 = p2 + 1u) {
      let b2 = base + p2 * PART_STRIDE;
      acc = acc + partials[b2 + t] * exp(partials[b2 + HEAD_DIM] - mAll);
    }
    out[h * HEAD_DIM + t] = acc / lAll;
  }
}`;
}

// Embedding gather on-GPU (fase B2 §Decode loop multi-step): dequant della riga
// `ids[0]` di token_embd (Q4_0 repackato) direttamente in x — il token id viene
// dal buffer amaxOut scritto dall'argmax dello step precedente (feedback on-GPU,
// niente readback per token). Stessa aritmetica ESATTA di dequantQ4_0Row (quant.ts):
// la dequant f32 non arrotonda oltre la scala f16 ⇒ x identica al percorso CPU.
export function embedGatherQ4Wgsl(opts: { K: number }): string {
  const { K } = opts;
  if (K % 32 !== 0) throw new Error("embedGatherQ4: K deve essere multiplo di 32");
  return `
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>; // [0] = token id (amaxOut o seed)
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
const K = ${K}u;
const BPR = ${K / 32}u; // blocchi per riga
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= K) { return; }
  let row = ids[0];
  let b = e >> 5u;         // blocco nella riga
  let j = e & 31u;         // peso nel blocco
  let gb = row * BPR + b;  // blocco globale nel tensore
  // layout repackQ4_0: peso j<16 = low nibble del byte j; j>=16 = high del byte j-16
  let lo = j < 16u;
  let jj = select(j - 16u, j, lo);
  let word = qs[gb * 4u + (jj >> 2u)];
  let byte = (word >> ((jj & 3u) * 8u)) & 0xffu;
  let nib = select(byte >> 4u, byte & 0xfu, lo);
  let scale = unpack2x16float(scales[gb >> 1u])[gb & 1u];
  x[e] = (f32(nib) - 8.0) * scale;
}`;
}

// --- GEMV K-quant (superblocco QK_K=256, goal C2 fase 4) ---
//
// Layout repack: repackKQuant (quant.ts) — superblocco GGUF grezzo in u32 LE.
// "Correttezza prima" (spec C2 §9 rischio 1): un workgroup da 64 thread per
// riga, un superblocco per thread per giro; niente vec4/tiling finché il floor
// tok/s non lo chiede. I riferimenti bit-esatti sono dequantQ5_K/dequantQ6_K.

// Q5_K: 44 word/superblocco = [d|dmin][scales 12 B][qh 32 B][qs 128 B].
// w = d·sc6bit·(nibble + bit_alto·16) − dmin·min6bit.
export function gemvQ5KWgsl(opts: { K: number; N: number }): string {
  const { K, N } = opts;
  if (K % 256 !== 0) throw new Error("gemvQ5K: K non multiplo di 256");
  const sbPerRow = K / 256;
  return `
@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
const SB_PER_ROW = ${sbPerRow}u;
var<workgroup> partial: array<f32, 64>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= ${N}u) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var sb = t; sb < SB_PER_ROW; sb = sb + 64u) {
    let wb = (r * SB_PER_ROW + sb) * 44u;   // base word del superblocco
    let dm = unpack2x16float(blocks[wb]);   // (d, dmin)
    let xBase = sb * 256u;
    var u1 = 1u; var u2 = 2u;
    for (var j = 0u; j < 4u; j = j + 1u) {  // 4 gruppi da 64 elementi
      let is = 2u * j;
      // get_scale_min_k4(is) e (is+1) — scales a byte offset 4 dentro il superblocco
      var sc1: u32; var mn1: u32; var sc2: u32; var mn2: u32;
      if (is < 4u) {
        sc1 = sbyte(wb, 4u + is) & 63u;
        mn1 = sbyte(wb, 4u + is + 4u) & 63u;
        sc2 = sbyte(wb, 4u + is + 1u) & 63u;
        mn2 = sbyte(wb, 4u + is + 5u) & 63u;
      } else {
        sc1 = (sbyte(wb, 4u + is + 4u) & 0xfu) | ((sbyte(wb, 4u + is - 4u) >> 6u) << 4u);
        mn1 = (sbyte(wb, 4u + is + 4u) >> 4u) | ((sbyte(wb, 4u + is) >> 6u) << 4u);
        sc2 = (sbyte(wb, 4u + is + 5u) & 0xfu) | ((sbyte(wb, 4u + is - 3u) >> 6u) << 4u);
        mn2 = (sbyte(wb, 4u + is + 5u) >> 4u) | ((sbyte(wb, 4u + is + 1u) >> 6u) << 4u);
      }
      let d1 = dm.x * f32(sc1); let min1 = dm.y * f32(mn1);
      let d2 = dm.x * f32(sc2); let min2 = dm.y * f32(mn2);
      var dot1 = 0.0; var sx1 = 0.0; var dot2 = 0.0; var sx2 = 0.0;
      for (var l = 0u; l < 32u; l = l + 1u) {
        let ql = sbyte(wb, 48u + j * 32u + l);   // qs a byte offset 48
        let qh = sbyte(wb, 16u + l);             // qh a byte offset 16
        let x1 = x[xBase + j * 64u + l];
        let x2 = x[xBase + j * 64u + 32u + l];
        var q1 = f32(ql & 0xfu);
        if ((qh & u1) != 0u) { q1 = q1 + 16.0; }
        var q2 = f32(ql >> 4u);
        if ((qh & u2) != 0u) { q2 = q2 + 16.0; }
        dot1 = dot1 + q1 * x1; sx1 = sx1 + x1;
        dot2 = dot2 + q2 * x2; sx2 = sx2 + x2;
      }
      acc = acc + d1 * dot1 - min1 * sx1 + d2 * dot2 - min2 * sx2;
      u1 = u1 << 2u; u2 = u2 << 2u;
    }
  }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { y[r] = partial[0]; }
}`;
}

// Q6_K: 53 word/superblocco (210 B + 2 pad) = [ql 128 B][qh 64 B][scales int8
// 16 B][d f16]. w = d·sc_int8·(q6 − 32), q6 = nibble | 2 bit alti.
export function gemvQ6KWgsl(opts: { K: number; N: number }): string {
  const { K, N } = opts;
  if (K % 256 !== 0) throw new Error("gemvQ6K: K non multiplo di 256");
  const sbPerRow = K / 256;
  return `
@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
const SB_PER_ROW = ${sbPerRow}u;
var<workgroup> partial: array<f32, 64>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
fn sint8(base: u32, i: u32) -> f32 {
  return f32((i32(sbyte(base, i)) << 24u) >> 24u);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= ${N}u) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var sb = t; sb < SB_PER_ROW; sb = sb + 64u) {
    let wb = (r * SB_PER_ROW + sb) * 53u;
    let d = unpack2x16float(blocks[wb + 52u]).x; // d f16 a byte offset 208
    let xBase = sb * 256u;
    for (var n = 0u; n < 2u; n = n + 1u) {       // 2 gruppi da 128
      let qlO = n * 64u;        // byte offset dentro ql (0 o 64)
      let qhO = 128u + n * 32u; // byte offset qh
      let scO = 192u + n * 8u;  // byte offset scales
      for (var l = 0u; l < 32u; l = l + 1u) {
        let is = l >> 4u;
        let qlA = sbyte(wb, qlO + l);
        let qlB = sbyte(wb, qlO + l + 32u);
        let qh = sbyte(wb, qhO + l);
        let q1 = f32((qlA & 0xfu) | (((qh >> 0u) & 3u) << 4u)) - 32.0;
        let q2 = f32((qlB & 0xfu) | (((qh >> 2u) & 3u) << 4u)) - 32.0;
        let q3 = f32((qlA >> 4u) | (((qh >> 4u) & 3u) << 4u)) - 32.0;
        let q4 = f32((qlB >> 4u) | (((qh >> 6u) & 3u) << 4u)) - 32.0;
        let e = xBase + n * 128u + l;
        acc = acc + d * (sint8(wb, scO + is) * q1 * x[e]
                       + sint8(wb, scO + is + 2u) * q2 * x[e + 32u]
                       + sint8(wb, scO + is + 4u) * q3 * x[e + 64u]
                       + sint8(wb, scO + is + 6u) * q4 * x[e + 96u]);
      }
    }
  }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { y[r] = partial[0]; }
}`;
}

// --- Kernel MLA absorbed (goal C2 fase 4, slice 2; semantica da deepseek2.cpp
// commit oracolo 5f55650, verificata nel sorgente: rope NORM (coppie consecutive)
// sulle sole 64 dim rope, kq_scale = 1/sqrt(n_embd_head_k_mla=256) — NON 576 —
// con mscale=1 (niente yarn nel GGUF GLM), V = c_kv normata (512). ---

// RoPE NORM in-place sul segmento rope di nVec vettori: vettore h a base
// h*stride, coppia i -> (offset+2i, offset+2i+1), theta = pos·base^(-2i/dims).
// Usi: q (nVec=20, stride=256, offset=192) e kv_cmpr_pe (nVec=1, stride=576,
// offset=512 — il rope va applicato PRIMA della kv_a_norm, che tocca solo le
// prime 512 componenti).
export function ropeMlaNormWgsl(opts: {
  nVec: number; stride: number; offset: number; ropeDims: number; freqBase: number;
}): string {
  const { nVec, stride, offset, ropeDims, freqBase } = opts;
  const half = ropeDims / 2;
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<uniform> P: TokParams;
const HALF = ${half}u;
const N_PAIRS = ${nVec * half}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N_PAIRS) { return; }
  let h = i / HALF;
  let j = i % HALF;
  let theta = f32(P.pos) * pow(${freqBase}.0, -f32(2u * j) / ${ropeDims}.0);
  let c = cos(theta);
  let s = sin(theta);
  let base = h * ${stride}u + ${offset}u + 2u * j;
  let a = v[base];
  let b = v[base + 1u];
  v[base] = a * c - b * s;
  v[base + 1u] = a * s + b * c;
}`;
}

// GEMV Q8_0 "per head": il tensore [K, rowsPerHead, nHead] (repack q8_0) moltiplica
// un x DIVERSO per head (x della head h a base h*xStride + xOffset, lungo K).
// Usi (dims GLM): wk_b assorbimento (K=192, rows=512, x=q_nope della head) e
// wv_b uscita (K=512, rows=256, x=attn·c_kv della head).
export function gemvQ8HeadsWgsl(opts: {
  K: number; rowsPerHead: number; nHead: number; xStride: number; xOffset: number;
}): string {
  const { K, rowsPerHead, nHead, xStride, xOffset } = opts;
  if (K % 32 !== 0) throw new Error("gemvQ8Heads: K non multiplo di 32");
  const blocksPerRow = K / 32;
  const N = rowsPerHead * nHead;
  return `
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= ${N}u) { return; }
  let head = r / ${rowsPerHead}u;
  let xBaseHead = head * ${xStride}u + ${xOffset}u;
  let t = lid.x;
  var acc = 0.0;
  for (var b = t; b < BLOCKS_PER_ROW; b = b + 64u) {
    let gb = r * BLOCKS_PER_ROW + b;
    let sWord = scales[gb >> 1u];
    let sc = unpack2x16float(sWord)[gb & 1u];
    let xBase = xBaseHead + b * 32u;
    var bd = 0.0;
    for (var w = 0u; w < 8u; w = w + 1u) {
      let word = qs[gb * 8u + w];
      for (var by = 0u; by < 4u; by = by + 1u) {
        let v = (i32((word >> (by * 8u)) & 0xffu) << 24u) >> 24u;
        bd = bd + f32(v) * x[xBase + w * 4u + by];
      }
    }
    acc = acc + sc * bd;
  }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { y[r] = partial[0]; }
}`;
}

// Attention decode MLA (MQA sulla cache compressa): q per head = [abs 512|pe 64]
// (576, layout Qcur di deepseek2), cache per token = [c_kv 512|k_rope 64] (576),
// score = dot completo a 576; V = prime 512 componenti della stessa riga di
// cache. out = [nHead, 512] (in spazio c_kv; wv_b si applica dopo, gemvQ8Heads).
export function mlaAttnDecodeWgsl(opts: {
  nHead: number; kvLora: number; ropeDims: number; ctxMax: number; scale: number;
}): string {
  const { kvLora, ropeDims, ctxMax, scale } = opts; // nHead = griglia del dispatch
  const width = kvLora + ropeDims;
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> P: TokParams;
const WIDTH = ${width}u;
const KV_LORA = ${kvLora}u;
const SCALE = ${scale};
var<workgroup> scores: array<f32, ${ctxMax}>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let qOff = h * WIDTH;
  let n = P.nPast + 1u;
  for (var p = t; p < n; p = p + 64u) {
    let cOff = p * WIDTH;
    var acc = 0.0;
    for (var i = 0u; i < WIDTH; i = i + 1u) { acc = acc + q[qOff + i] * cache[cOff + i]; }
    scores[p] = acc * SCALE;
  }
  workgroupBarrier();
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
  // out[h, j] = Σ_p softmax(p) · c_kv[p, j] — ogni thread un sottoinsieme di j
  for (var j = t; j < KV_LORA; j = j + 64u) {
    var acc = 0.0;
    for (var p = 0u; p < n; p = p + 1u) {
      acc = acc + scores[p] * cache[p * WIDTH + j];
    }
    out[h * KV_LORA + j] = acc / sAll;
  }
}`;
}

// Copy strided per-vettore: nVec segmenti di `len` f32 da src(srcStride, srcOff)
// a dst(dstStride, dstOff). Serve all'assemblaggio MLA absorbed (C2 fase 4):
// q576 per head = [q_ckv 512 (da gemvQ8Heads, stride 512) | q_rope 64 (da q
// 5120, stride 256 offset 192)] — copyBufferToBuffer non ha stride.
export function stridedCopyWgsl(opts: {
  nVec: number; len: number; srcStride: number; srcOffset: number;
  dstStride: number; dstOffset: number;
}): string {
  const { nVec, len, srcStride, srcOffset, dstStride, dstOffset } = opts;
  const total = nVec * len;
  return `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${total}u) { return; }
  let v = i / ${len}u;
  let j = i % ${len}u;
  dst[v * ${dstStride}u + ${dstOffset}u + j] = src[v * ${srcStride}u + ${srcOffset}u + j];
}`;
}

// ===================== FAMIGLIA FUSA PORTATA SU GLM (C3a fase 4b) ============
// Ruling PI 2026-08-02: "porta su glm tutte le ottimizzazioni che ci hanno dato
// tante soddisfazioni su qwen". La misura che lo motiva (state-2026-08-02 §2):
// a parita' di device il path GLM usa 29.3 GB/s utili contro i 155.6 del path
// Qwen — 5.1% del picco contro 27.0%.
//
// La causa NON e' la fusione in se': e' la STRUTTURA del gemv. Il generico
// `gemvQuantWgsl` che GLM usa ovunque fa 4 load scalari per blocco, rilegge x
// dalla memoria globale per ogni riga, estrae i nibble byte per byte e assegna
// UNA riga per workgroup. La famiglia fast fa una load `vec4<u32>` per blocco
// (16 B), normalizza x UNA volta in shared come `vec4<f32>`, usa `dot()` e
// tiene QUATTRO righe per workgroup. Questi kernel portano quella struttura sul
// blocco expert, che da solo vale ~902 MB dei 2220 MB letti per token.
//
// Differenza dai gemelli Qwen: niente rms fusa. Nel MoE la ffn_norm e' gia'
// stata applicata a monte del router (`preRouter`) e l'input e' condiviso dai
// 4 expert — rifarla qui la ricalcolerebbe 4 volte per layer.

// gate+up+silu del blocco expert in UN dispatch (sostituisce 3 gemvQuant+siluMul).
// x e' gia' normalizzato: si carica in shared cosi' com'e'.
export function pairGemvSiluFastWgsl(opts: { K: number; N: number }): string {
  const { K, N } = opts;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("pairGemvSiluFast: N non multiplo di 4");
  if (K % 32 !== 0) throw new Error("pairGemvSiluFast: K non multiplo di 32");
  return `
@group(0) @binding(0) var<storage, read> gQs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> gScales: array<u32>;
@group(0) @binding(2) var<storage, read> uQs4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> uScales: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> redG: array<f32, 64>;
var<workgroup> redU: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(x[i * 4u], x[i * 4u + 1u], x[i * 4u + 2u], x[i * 4u + 3u]);
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
  var stride = 8u;
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

// down del blocco expert: y[r] += accScale[0]·dot, struttura fast. Stesso ordine
// di binding del generico con scaledAccum, cosi' i bind group non cambiano forma.
export function gemvAccumFastWgsl(opts: { kind: "q4_0" | "q4_1"; K: number; N: number }): string {
  const { kind, K, N } = opts;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("gemvAccumFast: N non multiplo di 4");
  if (K % 32 !== 0) throw new Error("gemvAccumFast: K non multiplo di 32");
  // q4_0: w = (q-8)·d. q4_1: w = q·d + m ⇒ servono Σ(q·x) e Σx separati.
  const body = kind === "q4_0"
    ? `
      let w4 = qs4[gb];
      var lo = 0.0; var hi = 0.0;
      for (var wi = 0u; wi < 4u; wi = wi + 1u) {
        let word = w4[wi];
        let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
        let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
        lo = lo + dot(nibLo, xn4[x4 + wi]);
        hi = hi + dot(nibHi, xn4[x4 + 4u + wi]);
      }
      acc = acc + unpack2x16float(scales[gb >> 1u])[gb & 1u] * (lo + hi);`
    : `
      let w4 = qs4[gb];
      var dq = 0.0; var sx = 0.0;
      for (var wi = 0u; wi < 4u; wi = wi + 1u) {
        let word = w4[wi];
        let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu));
        let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu));
        let xl = xn4[x4 + wi];
        let xh = xn4[x4 + 4u + wi];
        dq = dq + dot(nibLo, xl) + dot(nibHi, xh);
        sx = sx + dot(vec4(1.0, 1.0, 1.0, 1.0), xl) + dot(vec4(1.0, 1.0, 1.0, 1.0), xh);
      }
      let dm = unpack2x16float(scales[gb]);
      acc = acc + dm.x * dq + dm.y * sx;`;
  return `
@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> accScale: array<f32>;
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> red: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(x[i * 4u], x[i * 4u + 1u], x[i * 4u + 2u], x[i * 4u + 3u]);
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
      ${body}
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
  if (lane == 0u && r < ${N}u) { y[r] = y[r] + accScale[0] * red[sub * 16u]; }
}`;
}

// Router top-k su GPU (C3a fase 4, strato 1 di spec §3.2-bis): replica esatta di
// `routerSelect` (moe.ts), che a sua volta replica build_moe_ffn. Scrive gli id
// selezionati e i pesi di mixing in DUE buffer GPU, così la selezione smette di
// essere una decisione CPU: e' il pezzo che toglie il readback dal percorso
// decisionale (il binding fisso da solo non basta — serve che gli id vivano su
// GPU, pattern ORT #27998 / ggml-webgpu `mul_mat_id_vec`).
//
// Fedelta': la CPU calcola in f64, qui si calcola in f32 — non e' una replica
// bit-identica ed e' misurato, non assunto (ktest `router-top4-*`). L'ordine
// delle operazioni e' pero' lo stesso: sigmoid, +bias per la sola SELEZIONE,
// scan lineare con `>` stretto (pareggio ⇒ indice minore, come l'argsort
// stabile dell'oracolo), somma dei probs SENZA bias nell'ordine di selezione,
// denominatore clampato, ×weightsScale.
//
// La selezione gira su un thread solo: sono nExpert×nUsed confronti (64×4 = 256)
// e la serializzazione E' la specifica — un top-k parallelo con riduzione
// cambierebbe l'ordine dei confronti e quindi il tie-break.
export function routerTopKWgsl(opts: {
  nExpert: number; nUsed: number; weightsScale: number; clampMin: number;
}): string {
  const { nExpert, nUsed, weightsScale, clampMin } = opts;
  const WG = 64;
  return `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> wts: array<f32>;
const NE = ${nExpert}u;
const NU = ${nUsed}u;
var<workgroup> probs: array<f32, ${nExpert}>;
var<workgroup> sel: array<f32, ${nExpert}>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  for (var i = t; i < NE; i = i + ${WG}u) {
    let p = 1.0 / (1.0 + exp(-logits[i]));
    probs[i] = p;
    sel[i] = p + bias[i];   // il bias entra SOLO nella selezione
  }
  workgroupBarrier();
  if (t != 0u) { return; }
  var taken: array<bool, ${nExpert}>;
  for (var i = 0u; i < NE; i = i + 1u) { taken[i] = false; }  // var in loop: azzerata a mano
  var sum = 0.0;
  for (var k = 0u; k < NU; k = k + 1u) {
    var best = NE;                                  // NE = "nessuno" (sentinella)
    for (var i = 0u; i < NE; i = i + 1u) {
      if (!taken[i] && (best == NE || sel[i] > sel[best])) { best = i; }
    }
    taken[best] = true;
    ids[k] = best;
    sum = sum + probs[best];                        // probs SENZA bias
  }
  let denom = max(sum, ${clampMin.toExponential()});
  for (var k = 0u; k < NU; k = k + 1u) {
    wts[k] = (probs[ids[k]] / denom) * ${weightsScale.toFixed(6)};
  }
}`;
}
