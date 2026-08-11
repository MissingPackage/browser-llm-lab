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
  /** batch (fase 5): wid.z = riga, x/y a offset di riga — testo non-batch invariato */
  batch?: boolean;
  // MoE (C2 fase 5): y[r] += accScale[0]·dot — il down per-expert accumula il
  // contributo pesato direttamente su moe_out (pesatura DOPO il down, come
  // ffn_moe_weighted in build_moe_ffn; l'ordine delle somme sui 4 expert
  // differisce dall'oracolo solo al rounding f32).
  scaledAccum?: boolean;
}): string {
  const { kind, K, N, hasBias, scaledAccum, batch } = opts;
  if (K % 32 !== 0) throw new Error("gemv: K non multiplo di 32");
  const blocksPerRow = K / 32;
  const wordsPerBlock = kind === "q8_0" ? 8 : 4;
  const biasBinding = hasBias
    ? `@group(0) @binding(4) var<storage, read> bias: array<f32>;` : "";
  const accBinding = scaledAccum
    ? `@group(0) @binding(${hasBias ? 5 : 4}) var<storage, read> accScale: array<f32>;` : "";
  const biasAdd = hasBias ? "accFinal = accFinal + bias[r];" : "";
  const rowPre = batch ? `\n  let xRB = wid.z * ${K}u;\n  let yRB = wid.z * ${N}u;` : "";
  const xRB = batch ? "xRB + " : "";
  const yI = batch ? "y[yRB + r]" : "y[r]";
  const writeY = scaledAccum ? `${yI} = ${yI} + accScale[0] * accFinal;` : `${yI} = accFinal;`;
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
    let xBase = ${xRB}b * 32u;
    ${blockDot}
    acc = acc + dm.x * dot_q + dm.y * sum_x;`
    : `
    let sWord = scales[gb >> 1u];
    let sc = unpack2x16float(sWord)[gb & 1u];
    let xBase = ${xRB}b * 32u;
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
  let t = lid.x;${rowPre}
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
export function gemvF32Wgsl(opts: { K: number; N: number; batch?: boolean }): string {
  // batch (fase 5): wid.z = riga — testo non-batch invariato (idioma it.27-30)
  const { K, N, batch } = opts;
  const rowPre = batch ? `\n  let xRB = wid.z * ${K}u;\n  let yRB = wid.z * ${N}u;` : "";
  const xRB = batch ? "xRB + " : "";
  const yI = batch ? "y[yRB + r]" : "y[r]";
  return `
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= ${N}u) { return; }
  let t = lid.x;${rowPre}
  var acc = 0.0;
  for (var i = t; i < ${K}u; i = i + 64u) { acc = acc + w[r * ${K}u + i] * x[${xRB}i]; }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u) { ${yI} = partial[0]; }
}`;
}

// RMSNorm: out = x * (1/sqrt(mean(x^2)+eps)) * w. Un solo workgroup da 256.
export function rmsnormWgsl(D: number, eps: number, batch?: boolean, strides?: { x: number; out: number }): string {
  // batch (fase 5): un workgroup per riga (wid.x = riga) — testo non-batch invariato.
  // strides: larghezza di RIGA dei buffer quando ≠ D (rmsKvA norma i primi 512
  // f32 di righe larghe 576) — solo batch.
  const sig = batch
    ? "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {"
    : "fn main(@builtin(local_invocation_id) lid: vec3<u32>) {";
  const sx = strides?.x ?? D, so = strides?.out ?? D;
  const rowPre = batch ? `\n  let rB = wid.x * ${sx}u;\n  let oB = wid.x * ${so}u;` : "";
  const xI = batch ? "x[rB + i]" : "x[i]";
  const outI = batch ? "out[oB + i]" : "out[i]";
  return `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
const D = ${D}u;
const EPS = ${eps};
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
${sig}
  let t = lid.x;${rowPre}
  var ss = 0.0;
  for (var i = t; i < D; i = i + 256u) { ss = ss + ${xI} * ${xI}; }
  partial[t] = ss;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let rms = 1.0 / sqrt(partial[0] / f32(D) + EPS);
  for (var i = t; i < D; i = i + 256u) { ${outI} = ${xI} * rms * w[i]; }
}`;
}

// RoPE NEOX in-place: coppie (j, j+half) per head. pos dall'uniform.
// ropeDims (q1 fase 4): rotazione PARZIALE sui primi ropeDims canali di ogni
// head (qwen35: 64 su 256, partial_rotary 0.25 — il mrope text-only collassa
// qui, spec q1 + cpuref ropeText), il resto passa invariato. Default =
// headDim: testo storico INVARIATO per i chiamanti esistenti.
export function ropeNeoxWgsl(nHead: number, headDim: number, freqBase: number, ropeDims = headDim, batch?: boolean): string {
  const half = ropeDims / 2;
  // batch (fase 4): gid.y = riga, posizione per riga da `rowPos` e vettore a
  // offset di riga — stesso idioma di `kvAppendWgsl`. Senza `batch` il testo
  // emesso e' IDENTICO a prima, byte per byte.
  const pBind = batch
    ? "@group(0) @binding(1) var<storage, read> rowPos: array<u32>;"
    : "@group(0) @binding(1) var<uniform> P: TokParams;";
  const posE = batch ? "rowPos[gid.y]" : "P.pos";
  const rowOff = batch ? "gid.y * ROW_STRIDE + " : "";
  const rowConst = batch ? `\nconst ROW_STRIDE = ${nHead * headDim}u;` : "";
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
${pBind}
const HALF = ${half}u;
const HEAD_DIM = ${headDim}u;
const N_PAIRS = ${nHead * half}u;${rowConst}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N_PAIRS) { return; }
  let h = i / HALF;
  let j = i % HALF;
  let theta = f32(${posE}) * pow(${freqBase}.0, -f32(j) / f32(HALF));
  let c = cos(theta);
  let s = sin(theta);
  let base = ${rowOff}h * HEAD_DIM;
  let a = v[base + j];
  let b = v[base + j + HALF];
  v[base + j] = a * c - b * s;
  v[base + j + HALF] = a * s + b * c;
}`;
}

// Append di k/v correnti nella cache alla posizione pos (layout [ctx, kvDim] row-major).
export function kvAppendWgsl(kvDim: number, batch?: boolean): string {
  // batch (fase 5): gid.y = riga, posizione per riga da rowPos — testo non-batch invariato
  const nBind = batch
    ? "@group(0) @binding(2) var<storage, read> rowPos: array<u32>;"
    : "@group(0) @binding(2) var<uniform> P: TokParams;";
  const posE = batch ? "rowPos[gid.y]" : "P.pos";
  const srcI = batch ? "src[gid.y * KV_DIM + i]" : "src[i]";
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
${nBind}
const KV_DIM = ${kvDim}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= KV_DIM) { return; }
  cache[${posE} * KV_DIM + i] = ${srcI};
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
export function siluMulWgsl(D: number, batch?: boolean): string {
  // batch (fase 4): gid.y = riga, indici a offset di riga. Senza, testo identico.
  const e = batch ? "gid.y * D + i" : "i";
  return `
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
const D = ${D}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  let g = gate[${e}];
  gate[${e}] = (g / (1.0 + exp(-g))) * up[${e}];
}`;
}

// x *= sigmoid(g) — output gate dell'attention qwen35 (q1 fase 4:
// attn_output_gate, il gate viaggia fuso in attn_q e NON riceve norm/rope).
export function sigmoidMulWgsl(D: number, batch?: boolean): string {
  const e = batch ? "gid.y * D + i" : "i";
  return `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
const D = ${D}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  x[${e}] = x[${e}] / (1.0 + exp(-g[${e}]));
}`;
}

// out += s·x con s scalare da buffer (q1 fase 7: combine pesato del MoE —
// il peso dell'expert arriva dal router; con sigmoidGate il gate del shared
// expert resta su GPU: s = sigmoid(sBuf[0])).
export function axpyWgsl(D: number, sigmoidGate = false): string {
  const s = sigmoidGate ? "1.0 / (1.0 + exp(-sBuf[0]))" : "sBuf[0]";
  return `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> sBuf: array<f32>;
const D = ${D}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  out[i] = out[i] + (${s}) * x[i];
}`;
}

// x += y (residual), in-place.
export function addInPlaceWgsl(D: number, batch?: boolean): string {
  const e = batch ? "gid.y * D + i" : "i";
  return `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
const D = ${D}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  x[${e}] = x[${e}] + y[${e}];
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
// Q4_K: 36 word/superblocco (144 B) = [d,dmin f16][scales 12 B 6-bit][qs 128 B].
// È gemvQ5K SENZA il piano qh (q1 fase 7: expert del 35B-A3B UD-Q4_K_S).
/**
 * SPARTIZIONE DEL LAVORO nei kernel K-quant (fase 4-bis, it.22). L'unita' non e'
 * piu' il SUPERBLOCCO ma un pezzo di gruppo: `lpu` valori dell'indice interno
 * `l`. Si sceglie `lpu` perche' le unita' totali arrivino a 64 — il numero di
 * thread del workgroup — quando la riga lo consente.
 *
 * Perche' serve: prima i thread si spartivano i superblocchi
 * (`for sb = t; sb < SB_PER_ROW`), e sul 35B SB_PER_ROW vale 8 per gate/up
 * (K=2048) e 2 per il down (K=512): **8 lane attive su 64 e 2 su 64**. Il
 * workgroup occupava uno slot per far lavorare due thread, con banda efficace
 * misurata a 17,2 GB/s su ~500 disponibili (it.21).
 *
 * `groupsPerSb` e' 4 per q4_K (i gruppi j da 64 elementi) e 2 per q6_K (i
 * gruppi n da 128). `lpu` e' una potenza di 2 <= 32, cosi' `32 % lpu == 0` e i
 * pezzi coprono il gruppo esattamente.
 */
export function kquantWorkSplit(sbPerRow: number, groupsPerSb: number): {
  lpu: number; chunks: number; unitsPerSb: number; units: number;
} {
  const want = (sbPerRow * groupsPerSb) / 2;
  const lpu = Math.max(1, Math.min(32, 2 ** Math.floor(Math.log2(Math.max(1, want)))));
  const chunks = 32 / lpu;
  const unitsPerSb = groupsPerSb * chunks;
  return { lpu, chunks, unitsPerSb, units: sbPerRow * unitsPerSb };
}

/**
 * CODA ACCUMULANTE, opzione `accum` (goal fase-D fase 4, it.21). Senza, il
 * kernel SCRIVE la riga: `y[r] = dot`. Con, ci ACCUMULA sopra il proprio
 * contributo pesato — `y[r] = y[r] + sel.w * dot` — col peso preso dalla `Sel`
 * che il preambolo d'arena ha gia' letto per sapere quale slot indirizzare.
 *
 * Serve a togliere un dispatch per expert: la catena era gate/up/silu/down/axpy
 * e l'axpy spariva solo per non essere mai esistito. Sul 35B sono 320 dispatch
 * per token (8 expert x 40 layer), su 2384 misurati in it.19 a ~21 us l'uno.
 *
 * BIT-IDENTICO per costruzione, e non "atteso identico": il dispatch che
 * sostituisce (l'axpy col peso da Sel, nato in it.17 e rimosso qui perche'
 * diventato codice morto) calcolava `out[i] + w * x[i]` in f32 con `x[i]`
 * uguale ESATTAMENTE a questo `partial[0]`, che e' il valore che il kernel
 * scriveva prima. Stessa espressione, stesse operazioni f32, stesso ordine.
 * Sul MISS non si arriva qui — il preambolo d'arena esce prima — quindi il
 * contributo e' zero, come lo era col guard esplicito di quell'axpy.
 *
 * Esige il regime d'arena: senza `Sel` non c'e' nessun peso da leggere.
 */
export function gemvQ4KWgsl(opts: { K: number; N: number; arena?: KArenaOpts; accum?: boolean }): string {
  const { K, N, arena, accum } = opts;
  if (accum === true && !arena) throw new Error("q4K: accum esige il regime d'arena (il peso viene da Sel)");
  if (K % 256 !== 0) throw new Error("gemvQ4K: K non multiplo di 256");
  const sbPerRow = K / 256;
  // OCCUPAZIONE (fase 4-bis, it.22). Prima l'unita' di lavoro era il
  // SUPERBLOCCO e i thread se li spartivano: `for sb = t; sb < SB_PER_ROW`.
  // Sul 35B SB_PER_ROW vale 8 per gate/up (K=2048) e 2 per il down (K=512),
  // cioe' 8 lane attive su 64 e 2 su 64 — il workgroup occupava uno slot per
  // far lavorare due thread, e la banda efficace misurata era 17,2 GB/s su
  // ~500 (it.21). L'unita' diventa un PEZZO del superblocco: il gruppo j
  // (64 elementi, due scale) diviso in blocchi da `lpu` valori di `l`, scelti
  // perche' le unita' totali arrivino a 64 quando la riga lo consente.
  const { lpu, chunks, unitsPerSb, units } = kquantWorkSplit(sbPerRow, 4);
  // UN SOLO corpo aritmetico per i due regimi: cambia da dove arrivano le
  // parole del blocco. Senza `arena` l'accesso resta `blocks[i]` diretto.
  const head = arena
    ? `${arenaHeadWgsl(arena, [
      "var<storage, read> x: array<f32>",
      "var<storage, read_write> y: array<f32>",
    ])}
var<private> gBase: u32;
var<private> gBi: u32;
fn blkw(i: u32) -> u32 { return ldw(gBi, gBase + i); }`
    : `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn blkw(i: u32) -> u32 { return blocks[i]; }`;
  // preambolo di main: nel regime d'arena lo slot di `Sel` diventa (binding, base)
  const pre = arena
    ? `${arenaSlotWgsl}
  gBi = bi;
  gBase = base + ${arena.tensorWords}u;
  if (!ok) { return; }`
    : "";
  return `
${head}
const SB_PER_ROW = ${sbPerRow}u;
const LPU = ${lpu}u;
const CHUNKS = ${chunks}u;
const UNITS_PER_SB = ${unitsPerSb}u;
const UNITS = ${units}u;
var<workgroup> partial: array<f32, 64>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blkw(base + (i >> 2u)) >> ((i & 3u) * 8u)) & 0xffu;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= ${N}u) { return; }${pre}
  let t = lid.x;
  var acc = 0.0;
  for (var u = t; u < UNITS; u = u + 64u) {
    let sb = u / UNITS_PER_SB;
    let rem = u % UNITS_PER_SB;
    let j = rem / CHUNKS;                   // gruppo da 64 elementi
    let lo = (rem % CHUNKS) * LPU;          // primo l dell'unita'
    let wb = (r * SB_PER_ROW + sb) * 36u;   // base word del superblocco
    let dm = unpack2x16float(blkw(wb));   // (d, dmin)
    let xBase = sb * 256u;
    let is = 2u * j;
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
    for (var l = lo; l < lo + LPU; l = l + 1u) {
      let ql = sbyte(wb, 16u + j * 32u + l);   // qs a byte offset 16 (niente qh)
      let x1 = x[xBase + j * 64u + l];
      let x2 = x[xBase + j * 64u + 32u + l];
      dot1 = dot1 + f32(ql & 0xfu) * x1; sx1 = sx1 + x1;
      dot2 = dot2 + f32(ql >> 4u) * x2; sx2 = sx2 + x2;
    }
    acc = acc + d1 * dot1 - min1 * sx1 + d2 * dot2 - min2 * sx2;
  }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  // coda accumulante (accum): vedi la nota sopra la funzione
  if (t == 0u) { ${accum === true ? "y[r] = y[r] + sel.w * partial[0];" : "y[r] = partial[0];"} }
}`;
}

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
export function gemvQ6KWgsl(opts: { K: number; N: number; arena?: KArenaOpts; accum?: boolean }): string {
  const { K, N, arena, accum } = opts;
  if (accum === true && !arena) throw new Error("q6K: accum esige il regime d'arena (il peso viene da Sel)");
  // OCCUPAZIONE (fase 4-bis, it.22): stessa correzione del q4_K. Qui il
  // superblocco si divide in 2 gruppi da 128 e ogni giro di `l` copre 4 quant;
  // l'unita' diventa un pezzo da `lpu6` valori di l dentro un gruppo, scelto
  // perche' le unita' arrivino a 64 quando la riga lo consente.
  if (K % 256 !== 0) throw new Error("gemvQ6K: K non multiplo di 256");
  const sbPerRow = K / 256;
  const { lpu: lpu6, chunks: chunks6, unitsPerSb: unitsPerSb6, units: units6 } = kquantWorkSplit(sbPerRow, 2);
  // UN SOLO corpo aritmetico per i due regimi: cambia da dove arrivano le
  // parole del blocco. Senza `arena` l'accesso resta `blocks[i]` diretto.
  const head = arena
    ? `${arenaHeadWgsl(arena, [
      "var<storage, read> x: array<f32>",
      "var<storage, read_write> y: array<f32>",
    ])}
var<private> gBase: u32;
var<private> gBi: u32;
fn blkw(i: u32) -> u32 { return ldw(gBi, gBase + i); }`
    : `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn blkw(i: u32) -> u32 { return blocks[i]; }`;
  // preambolo di main: nel regime d'arena lo slot di `Sel` diventa (binding, base)
  const pre = arena
    ? `${arenaSlotWgsl}
  gBi = bi;
  gBase = base + ${arena.tensorWords}u;
  if (!ok) { return; }`
    : "";
  return `
${head}
const SB_PER_ROW = ${sbPerRow}u;
const LPU = ${lpu6}u;
const CHUNKS = ${chunks6}u;
const UNITS_PER_SB = ${unitsPerSb6}u;
const UNITS = ${units6}u;
var<workgroup> partial: array<f32, 64>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blkw(base + (i >> 2u)) >> ((i & 3u) * 8u)) & 0xffu;
}
fn sint8(base: u32, i: u32) -> f32 {
  return f32((i32(sbyte(base, i)) << 24u) >> 24u);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= ${N}u) { return; }${pre}
  let t = lid.x;
  var acc = 0.0;
  for (var u = t; u < UNITS; u = u + 64u) {
    let sb = u / UNITS_PER_SB;
    let rem = u % UNITS_PER_SB;
    let n = rem / CHUNKS;                       // gruppo da 128
    let lo = (rem % CHUNKS) * LPU;              // primo l dell'unita'
    let wb = (r * SB_PER_ROW + sb) * 53u;
    let d = unpack2x16float(blkw(wb + 52u)).x; // d f16 a byte offset 208
    let xBase = sb * 256u;
    {
      let qlO = n * 64u;        // byte offset dentro ql (0 o 64)
      let qhO = 128u + n * 32u; // byte offset qh
      let scO = 192u + n * 8u;  // byte offset scales
      for (var l = lo; l < lo + LPU; l = l + 1u) {
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
  // coda accumulante (accum): vedi la nota sopra la funzione
  if (t == 0u) { ${accum === true ? "y[r] = y[r] + sel.w * partial[0];" : "y[r] = partial[0];"} }
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
  /** batch (fase 5): gid.y = riga, pos per riga da rowPos (storage al posto di P) */
  batch?: boolean;
}): string {
  const { nVec, stride, offset, ropeDims, freqBase, batch } = opts;
  const half = ropeDims / 2;
  const nBind = batch
    ? "@group(0) @binding(1) var<storage, read> rowPos: array<u32>;"
    : "@group(0) @binding(1) var<uniform> P: TokParams;";
  const rowPre = batch ? `\n  let vRB = gid.y * ${nVec * stride}u;` : "";
  const posE = batch ? "rowPos[gid.y]" : "P.pos";
  const vRB = batch ? "vRB + " : "";
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
${nBind}
const HALF = ${half}u;
const N_PAIRS = ${nVec * half}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N_PAIRS) { return; }${rowPre}
  let h = i / HALF;
  let j = i % HALF;
  let theta = f32(${posE}) * pow(${freqBase}.0, -f32(2u * j) / ${ropeDims}.0);
  let c = cos(theta);
  let s = sin(theta);
  let base = ${vRB}h * ${stride}u + ${offset}u + 2u * j;
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
  /** batch (fase 5): wid.z = riga; x per riga (stride nHead·xStride), y per riga */
  batch?: boolean;
}): string {
  const { K, rowsPerHead, nHead, xStride, xOffset, batch } = opts;
  const rowPre = batch ? `\n  let xRB = wid.z * ${nHead * xStride}u;\n  let yRB = wid.z * ${rowsPerHead * nHead}u;` : "";
  const xRB = batch ? "xRB + " : "";
  const yI = batch ? "y[yRB + r]" : "y[r]";
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
  let head = r / ${rowsPerHead}u;${rowPre}
  let xBaseHead = ${xRB}head * ${xStride}u + ${xOffset}u;
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
  if (t == 0u) { ${yI} = partial[0]; }
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

// Attention MLA SPLIT sul contesto (C3a fase 4c, stile flash-decoding) —
// sostituisce mlaAttnDecodeWgsl nel forward di produzione. Il monolitico gira su
// nHead=20 workgroup da 64 thread (GPU quasi vuota) e ognuno riscorre TUTTA la
// cache serialmente due volte: in MLA la cache è UNA sola (MQA sulla compressa),
// quindi quelle 20 riletture sono lo stesso traffico moltiplicato per 20 — il
// 74,5% del tempo GPU per token (51,2 ms, misura it.11).
//
// Qui il contesto è partizionato in blocchi da CHUNK posizioni: griglia FISSA
// (sMax, 1), le partizioni oltre n escono subito (condizione UNIFORME per
// workgroup, prima di ogni barrier). Ogni workgroup elabora il suo chunk per
// TUTTE le head — è il punto dell'esercizio: la cache si legge una volta per
// chunk invece di nHead volte. Pass 1 scrive per partizione {max locale m,
// somma locale l, out parziale non normalizzato}; pass 2
// (mlaAttnSplitReduceWgsl) combina in log-sum-exp esatto. Layout partials
// identico alla convenzione Qwen: [head][part][kvLora out | m | l].
export function mlaAttnSplitPartWgsl(opts: {
  nHead: number; kvLora: number; ropeDims: number; ctxMax: number; scale: number; chunk: number;
  /**
   * batch (fase 5): M righe di prefill nello stesso dispatch — wid.z = riga,
   * q da qM[riga], partials a offset di riga, e il PASSATO per riga arriva da
   * `rowPast` (storage al posto dell'uniform P): la causalita' intra-chunk e'
   * n = rowPast[m]+1 crescente per costruzione. Senza `batch` il testo emesso
   * e' IDENTICO a prima, byte per byte (idioma arena/batch di it.27-28).
   */
  batch?: boolean;
}): string {
  const { nHead, kvLora, ropeDims, ctxMax, scale, chunk, batch } = opts;
  const width = kvLora + ropeDims;
  const TW = 64;   // larghezza del tile, sia sui 576 dello score sia sui 512 dell'out
  const TWP = TW + 1; // stride PADDATO in shared: rompe i conflitti di banco
  const WG = 256;
  if (width % TW !== 0 || kvLora % TW !== 0) {
    throw new Error(`mlaAttnSplitPart: width ${width} e kvLora ${kvLora} devono essere multipli di ${TW}`);
  }
  const sMax = Math.ceil(ctxMax / chunk);
  const nAcc = Math.ceil((nHead * chunk) / WG); // accumulatori privati per thread
  const nBind = batch
    ? "@group(0) @binding(3) var<storage, read> rowPast: array<u32>;"
    : "@group(0) @binding(3) var<uniform> P: TokParams;";
  const rowPre = batch
    ? "\n  let qB = wid.z * N_HEAD * WIDTH;\n  let pB = wid.z * N_HEAD * S_MAX * PART_STRIDE;"
    : "";
  const nExpr = batch ? "rowPast[wid.z] + 1u" : "P.nPast + 1u";
  const qB = batch ? "qB + " : "";
  const pB = batch ? "pB + " : "";
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>; // [head][part][${kvLora} out | m | l]
${nBind}
const N_HEAD = ${nHead}u;
const WIDTH = ${width}u;
const KV_LORA = ${kvLora}u;
const CHUNK = ${chunk}u;
const S_MAX = ${sMax}u;
const PART_STRIDE = ${kvLora + 2}u;
const SCALE = ${scale};
const TW = ${TW}u;
const TWP = ${TWP}u;
const WG = ${WG}u;
const N_TILE_W = ${width / TW}u; // tile sulla larghezza dello score
const N_TILE_J = ${kvLora / TW}u; // tile sulle dimensioni di out
var<workgroup> cTile: array<f32, ${chunk * TWP}>;  // [CHUNK × TWP] di cache
var<workgroup> qTile: array<f32, ${nHead * TWP}>;  // [N_HEAD × TWP] di q
var<workgroup> sc: array<f32, ${nHead * chunk}>;   // score, poi exp (in place)
var<workgroup> mS: array<f32, ${nHead}>;
var<workgroup> lS: array<f32, ${nHead}>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let part = wid.x;
  let t = lid.x;${rowPre}
  let n = ${nExpr};
  let begin = part * CHUNK;
  if (begin >= n) { return; } // uniforme per workgroup: PRECEDE ogni workgroupBarrier
  let end = min(begin + CHUNK, n);
  // A) score[h][p] = SCALE · Σ_i q[h][i]·cache[begin+p][i], a tile di TW sulla
  // larghezza: il tile di cache si carica UNA volta e serve tutte le head.
  var acc: array<f32, ${nAcc}>;
  var d = 0.0; // dichiarato fuori dai loop: i var-in-loop non si ri-azzerano (landmine Tint)
  var cv = 0.0;
  var a = 0u;
  for (var i = 0u; i < ${nAcc}u; i = i + 1u) { acc[i] = 0.0; }
  for (var tile = 0u; tile < N_TILE_W; tile = tile + 1u) {
    let off = tile * TW;
    for (var i = t; i < CHUNK * TW; i = i + WG) {
      let p = i / TW; let c = i % TW;
      let pp = begin + p;
      cv = 0.0; // chunk parziale: la coda oltre end non si legge nemmeno
      if (pp < end) { cv = cache[pp * WIDTH + off + c]; }
      cTile[p * TWP + c] = cv;
    }
    for (var i = t; i < N_HEAD * TW; i = i + WG) {
      let h = i / TW; let c = i % TW;
      qTile[h * TWP + c] = q[${qB}h * WIDTH + off + c];
    }
    workgroupBarrier();
    a = 0u;
    for (var idx = t; idx < N_HEAD * CHUNK; idx = idx + WG) {
      let h = idx / CHUNK; let p = idx % CHUNK;
      d = 0.0;
      for (var c = 0u; c < TW; c = c + 1u) { d = d + qTile[h * TWP + c] * cTile[p * TWP + c]; }
      acc[a] = acc[a] + d;
      a = a + 1u;
    }
    workgroupBarrier(); // prima di riusare le shared del tile successivo
  }
  a = 0u;
  for (var idx = t; idx < N_HEAD * CHUNK; idx = idx + WG) {
    let p = idx % CHUNK;
    // posizione fuori dal contesto (chunk parziale): sentinella, come nel Qwen
    sc[idx] = select(-3.0e38, acc[a] * SCALE, begin + p < end);
    a = a + 1u;
  }
  workgroupBarrier();
  // B) softmax PARZIALE per head: riduzione seriale sulle sole CHUNK posizioni
  // del blocco, una head per thread (N_HEAD thread attivi, ~CHUNK iterazioni)
  if (t < N_HEAD) {
    let b = t * CHUNK;
    var m = -3.0e38;
    for (var p = 0u; p < CHUNK; p = p + 1u) { m = max(m, sc[b + p]); }
    var l = 0.0;
    var e = 0.0;
    for (var p = 0u; p < CHUNK; p = p + 1u) {
      e = select(0.0, exp(sc[b + p] - m), begin + p < end);
      sc[b + p] = e;
      l = l + e;
    }
    mS[t] = m;
    lS[t] = l;
  }
  workgroupBarrier();
  // C) out parziale NON normalizzato: out[h][j] = Σ_p e[h][p]·cache[begin+p][j].
  // La cache si rilegge una seconda volta dal global (come nel kernel Qwen): il
  // traffico totale resta 2× la cache invece di 2×nHead×.
  for (var tj = 0u; tj < N_TILE_J; tj = tj + 1u) {
    let off = tj * TW;
    for (var i = t; i < CHUNK * TW; i = i + WG) {
      let p = i / TW; let c = i % TW;
      let pp = begin + p;
      cv = 0.0; // chunk parziale: la coda oltre end non si legge nemmeno
      if (pp < end) { cv = cache[pp * WIDTH + off + c]; }
      cTile[p * TWP + c] = cv;
    }
    workgroupBarrier();
    for (var idx = t; idx < N_HEAD * TW; idx = idx + WG) {
      let h = idx / TW; let j = idx % TW;
      d = 0.0;
      for (var p = 0u; p < CHUNK; p = p + 1u) { d = d + sc[h * CHUNK + p] * cTile[p * TWP + j]; }
      partials[${pB}(h * S_MAX + part) * PART_STRIDE + off + j] = d;
    }
    workgroupBarrier();
  }
  if (t < N_HEAD) {
    let b2 = ${pB}(t * S_MAX + part) * PART_STRIDE;
    partials[b2 + KV_LORA] = mS[t];
    partials[b2 + KV_LORA + 1u] = lS[t];
  }
}`;
}

// Pass 2 dello split MLA: combina le partizioni in log-sum-exp esatto (M=max
// m_p, L=Σ l_p·exp(m_p−M), out=Σ o_p·exp(m_p−M)/L — stessa matematica di
// attnSplitReduceWgsl e del riferimento JS lseReduce). Griglia (nHead,1).
// NON è attnSplitReduceWgsl riusato: quello assume headDim ≤ 64 (un thread per
// dimensione), qui le dimensioni sono 512.
export function mlaAttnSplitReduceWgsl(opts: {
  nHead: number; kvLora: number; ctxMax: number; chunk: number;
  /** batch: v. mlaAttnSplitPartWgsl — wid.z = riga, nParts per riga da rowPast */
  batch?: boolean;
}): string {
  const { nHead, kvLora, ctxMax, chunk, batch } = opts; // nHead: griglia del dispatch (e stride di riga in batch)
  const sMax = Math.ceil(ctxMax / chunk);
  const WG = 256;
  const nBind = batch
    ? "@group(0) @binding(2) var<storage, read> rowPast: array<u32>;"
    : "@group(0) @binding(2) var<uniform> P: TokParams;";
  const rowPre = batch
    ? `\n  let pB = wid.z * ${nHead}u * S_MAX * PART_STRIDE;\n  let oB = wid.z * ${nHead}u * KV_LORA;`
    : "";
  const nPartsExpr = batch ? "rowPast[wid.z] / CHUNK + 1u" : "P.nPast / CHUNK + 1u";
  const pB = batch ? "pB + " : "";
  const oB = batch ? "oB + " : "";
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
${nBind}
const KV_LORA = ${kvLora}u;
const S_MAX = ${sMax}u;
const CHUNK = ${chunk}u;
const PART_STRIDE = ${kvLora + 2}u;
const WG = ${WG}u;
// shared O(1): SOLO i due scalari della riduzione. Memoizzare i pesi
// exp(m_p − M) in un array [S_MAX] costerebbe 4·S_MAX B, cioè shared che
// CRESCE col contesto — esattamente il difetto del kernel monolitico che lo
// split esiste per togliere (a ctxMax 65536 sarebbero 16 388 B, sopra il
// default di spec). Si ricalcola l'exp nel loop di output, come attnSplitReduce.
var<workgroup> mAll: f32;
var<workgroup> lAll: f32;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;${rowPre}
  let nParts = ${nPartsExpr}; // n = nPast+1 ⇒ ceil(n/CHUNK)
  let base = ${pB}h * S_MAX * PART_STRIDE;
  // M e L su un solo thread (nParts ≤ ${sMax}: riduzione seriale più corta del
  // costo di una parallela)
  if (t == 0u) {
    var m = -3.0e38;
    for (var p = 0u; p < nParts; p = p + 1u) { m = max(m, partials[base + p * PART_STRIDE + KV_LORA]); }
    var l = 0.0;
    for (var p = 0u; p < nParts; p = p + 1u) {
      let b2 = base + p * PART_STRIDE;
      l = l + partials[b2 + KV_LORA + 1u] * exp(partials[b2 + KV_LORA] - m);
    }
    mAll = m;
    lAll = l;
  }
  workgroupBarrier();
  var acc = 0.0;
  for (var j = t; j < KV_LORA; j = j + WG) {
    acc = 0.0;
    for (var p = 0u; p < nParts; p = p + 1u) {
      let b2 = base + p * PART_STRIDE;
      acc = acc + partials[b2 + j] * exp(partials[b2 + KV_LORA] - mAll);
    }
    out[${oB}h * KV_LORA + j] = acc / lAll;
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
  /** batch (fase 5): gid.y = riga, src/dst a offset di riga (stride nVec·) */
  batch?: boolean;
}): string {
  const { nVec, len, srcStride, srcOffset, dstStride, dstOffset, batch } = opts;
  const total = nVec * len;
  const rowPre = batch ? `\n  let sRB = gid.y * ${nVec * srcStride}u;\n  let dRB = gid.y * ${nVec * dstStride}u;` : "";
  const sRB = batch ? "sRB + " : "";
  const dRB = batch ? "dRB + " : "";
  return `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${total}u) { return; }${rowPre}
  let v = i / ${len}u;
  let j = i % ${len}u;
  dst[${dRB}v * ${dstStride}u + ${dstOffset}u + j] = src[${sRB}v * ${srcStride}u + ${srcOffset}u + j];
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

// ====================== ARENA EXPERT A BINDING FISSO (fase 4, strato 1) ======
// Il regime a SOTTO-RANGE (uno slot = un GPUBufferBinding {offset, size} per
// segmento) costringe a un bind group per slot: la cache ne ha migliaia, e
// soprattutto il bind group si puo' costruire solo quando la CPU SA gia' quale
// expert e' stato scelto — cioe' dopo il sync col router. E' quel vincolo, non
// la banda, a tenere in piedi i 46 sync per token.
//
// In modo arena la classe espone N buffer INTERI, bindati una volta sola in un
// bind group statico. Lo slot smette di essere un binding e diventa un
// INDIRIZZO: bufIdx = slot / SLABS_PER_BUF, base = (slot % SLABS_PER_BUF)·SLAB_W
// (costanti compile-time ⇒ mul+shift). Quale slot leggere lo dice
// `selBuf[moeIdx.selIdx]`, con selIdx da uniform a dynamic offset: e' un valore
// CPU-NOTO (la coppia (layer MoE, k)), mai l'expert. Chi riempie `Sel` e'
// intercambiabile — la CPU oggi, il router+resolve su GPU nello slice B — e i
// kernel non cambiano fra i due regimi.
//
// Gli offset dei sei segmenti arrivano in WORD (u32) da `SlabLayout` (moe.ts):
// qui non si riscrive nessuna costante di layout.
/**
 * Arena per i kernel K-quant (goal fase-D fase 3b): un solo offset, perche' un
 * tensore K-quant e' UN segmento (niente scale separate). `tensorWords` e'
 * l'offset del tensore dentro lo slab, in parole.
 */
export interface KArenaOpts {
  nBuf: number;
  slabWords: number;
  slabsPerBuf: number;
  tensorWords: number;
}

export interface ArenaOpts {
  nBuf: number;          // buffer d'arena della classe: binding 0..nBuf-1
  slabWords: number;     // SlabLayout.bytes / 4
  slabsPerBuf: number;
  qsWords: number;       // segmenti letti da gemvAccumFast (down)
  scalesWords: number;
  gateQsWords: number;   // segmenti letti da pairGemvSiluFast
  gateScWords: number;
  upQsWords: number;
  upScWords: number;
}

// Le due struct dell'indirezione expert (design §2.1 e §2.2). Stanno qui come
// costanti e non come stringhe ripetute perche' le scrivono DUE generatori —
// i kernel d'arena che leggono `Sel` e il router+resolve che la riempie: se i
// due layout divergessero, il resolve scriverebbe campi che il kernel legge
// spostati, e nessun errore lo direbbe.
const SEL_STRUCT_WGSL = "struct Sel { id: u32, slot: u32, w: f32, flags: u32 }";
const MOE_IDX_STRUCT_WGSL = "struct MoeIdx { selIdx: u32, tableBase: u32, moeLayer: u32, pad: u32 }";

/**
 * Le TAGLIE delle due struct qui sopra, esportate perche' chi riempie i buffer
 * sta fuori da questo file (glmmodel, q35gpumodel). Stanno accanto al WGSL che
 * le definisce e non nei modelli: due famiglie che scrivono la stessa `Sel` con
 * due costanti proprie sono due ABI che nessun compilatore confronta.
 */
export const SEL_BYTES = 16;
export const MOE_IDX_BYTES = 16;
/**
 * Le entry di `MoeIdx` vanno spaziate a `minUniformBufferOffsetAlignment`
 * perche' l'uniform si binda a dynamic offset. 256 e' il default di spec e il
 * massimo che un device possa chiedere (chi la usa lo ASSERTA sul device).
 */
export const MOE_IDX_STRIDE = 256;

/**
 * Testa comune dei kernel d'arena: struct, i nBuf binding d'arena, i binding
 * propri del kernel (`mid`, che partono da nBuf), `Sel` e l'uniform a dynamic
 * offset, i due accessori e le costanti di classe.
 * `ld4` fa uno switch sui binding: il ramo e' UNIFORME sull'intero dispatch (un
 * expert per dispatch) ⇒ scalare, senza divergenza fra lane.
 */
function arenaHeadWgsl(a: Pick<ArenaOpts, "nBuf" | "slabWords" | "slabsPerBuf">, mid: string[]): string {
  const lines = [SEL_STRUCT_WGSL, MOE_IDX_STRUCT_WGSL];
  for (let j = 0; j < a.nBuf; j++) {
    lines.push(`@group(0) @binding(${j}) var<storage, read> arena${j}: array<vec4<u32>>;`);
  }
  mid.forEach((d, i) => lines.push(`@group(0) @binding(${a.nBuf + i}) ${d};`));
  lines.push(`@group(0) @binding(${a.nBuf + mid.length}) var<storage, read> selBuf: array<Sel>;`);
  lines.push(`@group(0) @binding(${a.nBuf + mid.length + 1}) var<uniform> moeIdx: MoeIdx;`);
  lines.push("fn ld4(b: u32, i: u32) -> vec4<u32> {");
  lines.push("  switch b {");
  for (let j = 0; j < a.nBuf; j++) lines.push(`    case ${j}u: { return arena${j}[i]; }`);
  lines.push("    default: { return arena0[i]; }");
  lines.push("  }");
  lines.push("}");
  lines.push("fn ldw(b: u32, w: u32) -> u32 { return ld4(b, w >> 2u)[w & 3u]; }");
  lines.push(`const SLAB_W = ${a.slabWords}u;`);
  lines.push(`const SLABS_PER_BUF = ${a.slabsPerBuf}u;`);
  return lines.join("\n");
}

/**
 * Preambolo di `main`: legge la Sel e trasforma lo slot in (binding, base).
 * `slot == 0xffffffff` e' il MISS dichiarato nel layout di Sel: `ok` lo propaga
 * alle guardie invece di uscire con un `return`, che romperebbe l'analisi di
 * uniformita' dei workgroupBarrier a valle. In slice A non puo' accadere (la
 * CPU fa `ensure` prima di riempire Sel); esiste perche' il degrado sia
 * DEFINITO — niente uscita, non un indirizzo a caso.
 */
const arenaSlotWgsl = `
  let sel = selBuf[moeIdx.selIdx];
  let ok = sel.slot != 0xffffffffu;
  let slot = select(0u, sel.slot, ok);
  let bi = slot / SLABS_PER_BUF;
  let base = (slot % SLABS_PER_BUF) * SLAB_W;`;

// gate+up+silu del blocco expert in UN dispatch (sostituisce 3 gemvQuant+siluMul).
// x e' gia' normalizzato: si carica in shared cosi' com'e'.
// Senza `arena` il testo emesso e' IDENTICO a quello di prima dello strato 1,
// byte per byte: il corpo aritmetico esiste una volta sola per i due regimi.
export function pairGemvSiluFastWgsl(opts: { K: number; N: number; arena?: ArenaOpts }): string {
  const { K, N, arena } = opts;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("pairGemvSiluFast: N non multiplo di 4");
  if (K % 32 !== 0) throw new Error("pairGemvSiluFast: K non multiplo di 32");
  const head = arena
    ? arenaHeadWgsl(arena, [
        "var<storage, read> x: array<f32>",
        "var<storage, read_write> out: array<f32>",
      ])
    : `@group(0) @binding(0) var<storage, read> gQs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> gScales: array<u32>;
@group(0) @binding(2) var<storage, read> uQs4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> uScales: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;`;
  const pre = arena
    ? `${arenaSlotWgsl}
  let gQ4 = (base + ${arena.gateQsWords}u) >> 2u;
  let gSc = base + ${arena.gateScWords}u;
  let uQ4 = (base + ${arena.upQsWords}u) >> 2u;
  let uSc = base + ${arena.upScWords}u;`
    : "";
  const live = arena ? " && ok" : "";
  const gQs = arena ? "ld4(bi, gQ4 + gb)" : "gQs4[gb]";
  const gSca = arena ? "ldw(bi, gSc + (gb >> 1u))" : "gScales[gb >> 1u]";
  const uQs = arena ? "ld4(bi, uQ4 + gb)" : "uQs4[gb]";
  const uSca = arena ? "ldw(bi, uSc + (gb >> 1u))" : "uScales[gb >> 1u]";
  return `
${head}
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> redG: array<f32, 64>;
var<workgroup> redU: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;${pre}
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(x[i * 4u], x[i * 4u + 1u], x[i * 4u + 2u], x[i * 4u + 3u]);
  }
  workgroupBarrier();
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  var accG = 0.0;
  var accU = 0.0;
  if (r < ${N}u${live}) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {
      let gb = r * BLOCKS_PER_ROW + b;
      let x4 = b * 8u;
      {
        let w4 = ${gQs};
        var lo = 0.0; var hi = 0.0;
        for (var wi = 0u; wi < 4u; wi = wi + 1u) {
          let word = w4[wi];
          let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
          let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
          lo = lo + dot(nibLo, xn4[x4 + wi]);
          hi = hi + dot(nibHi, xn4[x4 + 4u + wi]);
        }
        accG = accG + unpack2x16float(${gSca})[gb & 1u] * (lo + hi);
      }
      {
        let w4 = ${uQs};
        var lo = 0.0; var hi = 0.0;
        for (var wi = 0u; wi < 4u; wi = wi + 1u) {
          let word = w4[wi];
          let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
          let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
          lo = lo + dot(nibLo, xn4[x4 + wi]);
          hi = hi + dot(nibHi, xn4[x4 + 4u + wi]);
        }
        accU = accU + unpack2x16float(${uSca})[gb & 1u] * (lo + hi);
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
  if (lane == 0u && r < ${N}u${live}) {
    let g = redG[sub * 16u];
    out[r] = (g / (1.0 + exp(-g))) * redU[sub * 16u];
  }
}`;
}

// down del blocco expert: y[r] += accScale[0]·dot, struttura fast. Stesso ordine
// di binding del generico con scaledAccum, cosi' i bind group non cambiano forma.
// In modo arena il peso di mixing NON e' piu' un binding suo: e' `sel.w`, gia'
// ×1.8, dentro la Sel — sono i quattro `wExp[k]` che collassano nella stessa
// indirezione degli slot.
export function gemvAccumFastWgsl(opts: { kind: "q4_0" | "q4_1"; K: number; N: number; arena?: ArenaOpts }): string {
  const { kind, K, N, arena } = opts;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("gemvAccumFast: N non multiplo di 4");
  if (K % 32 !== 0) throw new Error("gemvAccumFast: K non multiplo di 32");
  const qsE = arena ? "ld4(bi, qsW4 + gb)" : "qs4[gb]";
  const scE = arena
    ? (kind === "q4_0" ? "ldw(bi, scW + (gb >> 1u))" : "ldw(bi, scW + gb)")
    : (kind === "q4_0" ? "scales[gb >> 1u]" : "scales[gb]");
  const wE = arena ? "sel.w" : "accScale[0]";
  const live = arena ? " && ok" : "";
  const head = arena
    ? arenaHeadWgsl(arena, [
        "var<storage, read> x: array<f32>",
        "var<storage, read_write> y: array<f32>",
      ])
    : `@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> accScale: array<f32>;`;
  const pre = arena
    ? `${arenaSlotWgsl}
  let qsW4 = (base + ${arena.qsWords}u) >> 2u;
  let scW = base + ${arena.scalesWords}u;`
    : "";
  // q4_0: w = (q-8)·d. q4_1: w = q·d + m ⇒ servono Σ(q·x) e Σx separati.
  const body = kind === "q4_0"
    ? `
      let w4 = ${qsE};
      var lo = 0.0; var hi = 0.0;
      for (var wi = 0u; wi < 4u; wi = wi + 1u) {
        let word = w4[wi];
        let nibLo = vec4(f32(word & 0xfu), f32((word >> 8u) & 0xfu), f32((word >> 16u) & 0xfu), f32((word >> 24u) & 0xfu)) - 8.0;
        let nibHi = vec4(f32((word >> 4u) & 0xfu), f32((word >> 12u) & 0xfu), f32((word >> 20u) & 0xfu), f32((word >> 28u) & 0xfu)) - 8.0;
        lo = lo + dot(nibLo, xn4[x4 + wi]);
        hi = hi + dot(nibHi, xn4[x4 + 4u + wi]);
      }
      acc = acc + unpack2x16float(${scE})[gb & 1u] * (lo + hi);`
    : `
      let w4 = ${qsE};
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
      let dm = unpack2x16float(${scE});
      acc = acc + dm.x * dq + dm.y * sx;`;
  return `
${head}
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> red: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;${pre}
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(x[i * 4u], x[i * 4u + 1u], x[i * 4u + 2u], x[i * 4u + 3u]);
  }
  workgroupBarrier();
  let sub = t >> 4u;
  let lane = t & 15u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  var acc = 0.0;
  if (r < ${N}u${live}) {
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
  if (lane == 0u && r < ${N}u${live}) { y[r] = y[r] + ${wE} * red[sub * 16u]; }
}`;
}

// ============ PREFILL BATCHED M>1 — CATENA EXPERT SU UNIONE (fase 5, it.27) =
// Il piano (moeprefillplan) raggruppa le selezioni del chunk per expert; ogni
// dispatch processa UN expert su tutte le righe che lo selezionano. La riga
// raccolta arriva da `wid.z` via il buffer `gather` (u32: m | k<<16). Il corpo
// aritmetico e' QUELLO dei gemelli decode (pairGemvSiluFast/gemvAccumFast):
// stesso ordine di riduzione ⇒ dot bit-identici per riga. La differenza
// strutturale e' il contratto degli SLOT: il down SCRIVE y[m][k] non pesato
// (niente accumulo), e `moeCombine` somma w·y in ordine k partendo dallo
// shexp — la stessa catena di somme del decode (glmmodel: shexp scrive
// moeOut, gemvAccumFast accumula in ordine k, addMoe fa x += moeOut).
// Varianti PLAIN (bindings espliciti): il modo arena arriva col wiring,
// con la stessa testa `arenaHeadWgsl` dei gemelli.

// gate+up+silu dell'expert per le righe raccolte: h[m][k] = silu(g)·u
export function pairGemvSiluGatherWgsl(opts: { K: number; N: number }): string {
  const { K, N } = opts;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("pairGemvSiluGather: N non multiplo di 4");
  if (K % 32 !== 0) throw new Error("pairGemvSiluGather: K non multiplo di 32");
  return `
@group(0) @binding(0) var<storage, read> gQs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> gScales: array<u32>;
@group(0) @binding(2) var<storage, read> uQs4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> uScales: array<u32>;
@group(0) @binding(4) var<storage, read> xM: array<f32>;
@group(0) @binding(5) var<storage, read> gather: array<u32>;
@group(0) @binding(6) var<storage, read_write> hSlots: array<f32>;
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> redG: array<f32, 64>;
var<workgroup> redU: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let gk = gather[wid.z];
  let m = gk & 0xffffu;
  let kslot = gk >> 16u;
  let mB = m * K;
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(xM[mB + i * 4u], xM[mB + i * 4u + 1u], xM[mB + i * 4u + 2u], xM[mB + i * 4u + 3u]);
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
    hSlots[(m * 4u + kslot) * ${N}u + r] = (g / (1.0 + exp(-g))) * redU[sub * 16u];
  }
}`;
}

// down dell'expert per le righe raccolte: SCRIVE y[m][k] NON pesato — il peso
// lo applica moeCombine, in ordine k (contratto di identita' del piano).
export function gemvDownSlotsWgsl(opts: { kind: "q4_0" | "q4_1"; K: number; N: number }): string {
  const { kind, K, N } = opts;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("gemvDownSlots: N non multiplo di 4");
  if (K % 32 !== 0) throw new Error("gemvDownSlots: K non multiplo di 32");
  const scE = kind === "q4_0" ? "scales[gb >> 1u]" : "scales[gb]";
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
      acc = acc + unpack2x16float(${scE})[gb & 1u] * (lo + hi);`
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
      let dm = unpack2x16float(${scE});
      acc = acc + dm.x * dq + dm.y * sx;`;
  return `
@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> hSlots: array<f32>;
@group(0) @binding(3) var<storage, read> gather: array<u32>;
@group(0) @binding(4) var<storage, read_write> ySlots: array<f32>;
const K = ${K}u;
const BLOCKS_PER_ROW = ${blocksPerRow}u;
var<workgroup> red: array<f32, 64>;
var<workgroup> xn4: array<vec4<f32>, ${K / 4}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let gk = gather[wid.z];
  let m = gk & 0xffffu;
  let kslot = gk >> 16u;
  let hB = (m * 4u + kslot) * K;
  for (var i = t; i < K / 4u; i = i + 64u) {
    xn4[i] = vec4(hSlots[hB + i * 4u], hSlots[hB + i * 4u + 1u], hSlots[hB + i * 4u + 2u], hSlots[hB + i * 4u + 3u]);
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
  if (lane == 0u && r < ${N}u) { ySlots[(m * 4u + kslot) * ${N}u + r] = red[sub * 16u]; }
}`;
}

// combine per riga: xM[m] += shexp[m] + Σ_k w[m][k]·y[m][k], somme in ordine k
// — la stessa catena del decode (v. commento di famiglia). grid: (ceil(D/64), M).
export function moeCombineWgsl(opts: { D: number; nUsed?: number }): string {
  const { D } = opts;
  const nUsed = opts.nUsed ?? 4;
  return `
@group(0) @binding(0) var<storage, read_write> xM: array<f32>;
@group(0) @binding(1) var<storage, read> sM: array<f32>;
@group(0) @binding(2) var<storage, read> ySlots: array<f32>;
@group(0) @binding(3) var<storage, read> wBuf: array<f32>;
const D = ${D}u;
const N_USED = ${nUsed}u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wid.x * 64u + lid.x;
  if (i >= D) { return; }
  let m = wid.y;
  var t = sM[m * D + i];
  for (var k = 0u; k < N_USED; k = k + 1u) {
    t = t + wBuf[m * N_USED + k] * ySlots[(m * N_USED + k) * D + i];
  }
  xM[m * D + i] = xM[m * D + i] + t;
}`;
}

// ============ FAMIGLIA FAST PER I K-QUANT (C3a fase 4b, it.13) ==============
// Stessa terapia della famiglia Q4, applicata a Q5_K/Q6_K — che dopo l'it.12
// sono le due categorie piu' grosse rimaste (shexp 14,6 ms/token, head 9,6).
// CAVEAT sui due numeri: vengono dal bycat a 948 MHz (journal it.12) e sono
// gonfiati dal clock — la QUOTA e' quella, dominante dopo l'attention, ma il
// valore assoluto non e' iso-clock e non va confrontato con misure a clock
// pieno senza rinormalizzare.
// I tre difetti dei gemelli lenti (gemvQ5KWgsl/gemvQ6KWgsl), letti nel codice:
//   (a) UN superblocco per thread con `for sb = t; sb < SB_PER_ROW; sb += 64`:
//       a K=2048 i superblocchi per riga sono 8, quindi 56 thread su 64 non
//       fanno NIENTE. Il workgroup e' pieno all'12,5%;
//   (b) `sbyte()` fa una load u32 dal global PER OGNI BYTE: la stessa word
//       viene riletta 4 volte, e le scale Q6_K una volta per peso;
//   (c) x riletto dal global da ogni riga.
//
// La struttura nuova, comune ai due formati: un workgroup da 64 thread per
// riga, e il lavoro tagliato per SOTTOGRUPPI DI 32 PESI invece che per
// superblocco. Un superblocco K-quant sono 256 pesi = 8 sottogruppi, quindi 8
// thread per superblocco: a K=2048 sono 64 sottogruppi per riga, cioe' un
// thread ciascuno, workgroup pieno. Le word che servono al thread si caricano
// una volta in registri (8 word di quanti + le scale), i byte si estraggono di
// li'. x sta in shared, letto una volta per workgroup.
//
// PADDING DI x IN SHARED (+1 f32 ogni 32): senza, i 32 thread di un warp
// leggono indirizzi tutti congrui mod 32 — i sottogruppi distano esattamente 32
// f32 — cioe' TUTTI nello stesso banco, conflitto a 32 vie sull'accesso piu'
// caldo del kernel. Con l'indice `e + (e >> 5)` la distanza diventa 33 e i
// banchi si separano. Costa K/32 f32 di shared (64 su 2048).
//
// L'aritmetica delle SCALE e' bit-identica ai riferimenti dequantQ5_K /
// dequantQ6_K (get_scale_min_k4 e sint8, stesse espressioni dei gemelli lenti):
// e' un vincolo di conformance, non un'ottimizzazione da rivedere. Cambia solo
// il RAGGRUPPAMENTO delle somme (per sottogruppo invece che per peso), come
// gia' fatto dalla famiglia Q4 — ed e' per questo che i ktest confrontano
// contro gli stessi riferimenti CPU con le stesse tolleranze.

// Q5_K gate+up + silu·mul fusi (shexp): sostituisce gemvShexpGU ×2 + siluExp,
// tre dispatch in uno. Superblocco Q5_K = 44 word: [d|dmin][scales 12 B]
// [qh 32 B][qs 128 B]. Il sottogruppo g del superblocco e' la coppia
// (gruppo j = g>>1, meta' g&1) della struttura a 4 gruppi da 64 — e l'indice di
// scala `is` coincide con g, che e' esattamente 2j + meta'.
export function pairGemvSiluQ5KFastWgsl(opts: { K: number; N: number; batch?: boolean }): string {
  // batch: M righe (wid.z = riga, x da xM[mB], out per riga) — fase 5, shexp
  // del chunk. Senza `batch` il testo emesso e' IDENTICO a prima, byte per
  // byte: il corpo aritmetico esiste una volta sola per i due regimi (idioma
  // arena). Per riga il corpo e' lo stesso ⇒ dot bit-identici al per-riga.
  const { K, N, batch } = opts;
  const mB = batch ? "\n  let mB = wid.z * K;" : "";
  const xSrc = batch ? "x[mB + i]" : "x[i]";
  const outIdx = batch ? `out[wid.z * ${N}u + r]` : "out[r]";
  if (K % 256 !== 0) throw new Error("pairGemvSiluQ5KFast: K non multiplo di 256");
  const sbPerRow = K / 256;
  const nSub = K / 32; // sottogruppi da 32 pesi per riga
  return `
@group(0) @binding(0) var<storage, read> gBlocks: array<u32>;
@group(0) @binding(1) var<storage, read> uBlocks: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
const K = ${K}u;
const SB_PER_ROW = ${sbPerRow}u;
const N_SUB = ${nSub}u;
var<workgroup> redG: array<f32, 64>;
var<workgroup> redU: array<f32, 64>;
var<workgroup> xs: array<f32, ${K + K / 32}>; // x paddato: +1 f32 ogni 32
fn scByte(sw: vec3<u32>, i: u32) -> u32 { return (sw[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu; }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;${mB}
  for (var i = t; i < K; i = i + 64u) { xs[i + (i >> 5u)] = ${xSrc}; }
  workgroupBarrier();
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  var accG = 0.0;
  var accU = 0.0;
  var dotq = 0.0; // non "dot": e' il nome di un builtin WGSL
  var sx = 0.0;
  var q = 0.0;
  var sc = 0u;
  var mn = 0u;
  if (r < ${N}u) {
    // un sottogruppo da 32 pesi per thread (a K=2048: 64 sottogruppi, uno
    // ciascuno; a K non multiplo di 2048 il loop stride copre il resto)
    for (var su = t; su < N_SUB; su = su + 64u) {
      let sb = su >> 3u;
      let g = su & 7u;
      let j = g >> 1u;
      let half = g & 1u;
      let mask = (1u + half) << (2u * j); // qh: u1 = 1<<2j, u2 = 2<<2j
      let xb = sb * 256u + j * 64u + half * 32u;
      let sbBase = (r * SB_PER_ROW + sb) * 44u;
      for (var side = 0u; side < 2u; side = side + 1u) {
        // side 0 = gate, side 1 = up: stesse posizioni, buffer diverso
        var dm = vec2(0.0, 0.0);
        var sw = vec3(0u, 0u, 0u);
        if (side == 0u) {
          dm = unpack2x16float(gBlocks[sbBase]);
          sw = vec3(gBlocks[sbBase + 1u], gBlocks[sbBase + 2u], gBlocks[sbBase + 3u]);
        } else {
          dm = unpack2x16float(uBlocks[sbBase]);
          sw = vec3(uBlocks[sbBase + 1u], uBlocks[sbBase + 2u], uBlocks[sbBase + 3u]);
        }
        // get_scale_min_k4(is) con is = g — espressioni identiche al riferimento
        if (g < 4u) {
          sc = scByte(sw, g) & 63u;
          mn = scByte(sw, g + 4u) & 63u;
        } else {
          sc = (scByte(sw, g + 4u) & 0xfu) | ((scByte(sw, g - 4u) >> 6u) << 4u);
          mn = (scByte(sw, g + 4u) >> 4u) | ((scByte(sw, g) >> 6u) << 4u);
        }
        dotq = 0.0;
        sx = 0.0;
        for (var wI = 0u; wI < 8u; wI = wI + 1u) {
          var qsWord = 0u;
          var qhWord = 0u;
          if (side == 0u) {
            qsWord = gBlocks[sbBase + 12u + j * 8u + wI]; // qs a byte 48
            qhWord = gBlocks[sbBase + 4u + wI];           // qh a byte 16
          } else {
            qsWord = uBlocks[sbBase + 12u + j * 8u + wI];
            qhWord = uBlocks[sbBase + 4u + wI];
          }
          for (var b = 0u; b < 4u; b = b + 1u) {
            let sh = b * 8u;
            let ql = (qsWord >> sh) & 0xffu;
            let qhb = (qhWord >> sh) & 0xffu;
            q = f32(select(ql >> 4u, ql & 0xfu, half == 0u));
            if ((qhb & mask) != 0u) { q = q + 16.0; }
            let e = xb + wI * 4u + b;
            let xv = xs[e + (e >> 5u)];
            dotq = dotq + q * xv;
            sx = sx + xv;
          }
        }
        let d1 = dm.x * f32(sc);
        let min1 = dm.y * f32(mn);
        if (side == 0u) { accG = accG + d1 * dotq - min1 * sx; }
        else { accU = accU + d1 * dotq - min1 * sx; }
      }
    }
  }
  redG[t] = accG;
  redU[t] = accU;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { redG[t] = redG[t] + redG[t + stride]; redU[t] = redU[t] + redU[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (t == 0u && r < ${N}u) {
    let gv = redG[0];
    ${outIdx} = (gv / (1.0 + exp(-gv))) * redU[0];
  }
}`;
}

// Q6_K con la stessa struttura (shexp down E output head: stessa famiglia di
// shape). Firma dei binding IDENTICA a gemvQ6KWgsl, così il wiring cambia solo
// la pipeline. Superblocco = 53 word: [ql 128 B][qh 64 B][scales int8 16 B]
// [d f16 in coda]. Il sottogruppo g e' la coppia (meta' n = g>>2, quarto
// k = g&3) dei 4 termini q1..q4 del riferimento.
export function gemvQ6KFastWgsl(opts: { K: number; N: number; batch?: boolean }): string {
  // batch: v. pairGemvSiluQ5KFastWgsl — stesso regime, testo non-batch invariato
  const { K, N, batch } = opts;
  const mB = batch ? "\n  let mB = wid.z * K;" : "";
  const xSrc = batch ? "x[mB + i]" : "x[i]";
  const outIdx = batch ? `y[wid.z * ${N}u + r]` : "y[r]";
  if (K % 256 !== 0) throw new Error("gemvQ6KFast: K non multiplo di 256");
  const sbPerRow = K / 256;
  const nSub = K / 32;
  return `
@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
const K = ${K}u;
const SB_PER_ROW = ${sbPerRow}u;
const N_SUB = ${nSub}u;
var<workgroup> partial: array<f32, 64>;
var<workgroup> xs: array<f32, ${K + K / 32}>; // x paddato: +1 f32 ogni 32
// scala int8 dal byte bi (0..7) del gruppo, letta dalle due word gia' in registri
fn s8(w0: u32, w1: u32, bi: u32) -> f32 {
  let w = select(w1, w0, bi < 4u);
  return f32((i32((w >> ((bi & 3u) * 8u)) & 0xffu) << 24u) >> 24u);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;${mB}
  for (var i = t; i < K; i = i + 64u) { xs[i + (i >> 5u)] = ${xSrc}; }
  workgroupBarrier();
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  var acc = 0.0;
  var acc0 = 0.0; // pesi l<16 (scala is=0)
  var acc1 = 0.0; // pesi l>=16 (scala is=1)
  var q = 0.0;
  if (r < ${N}u) {
    for (var su = t; su < N_SUB; su = su + 64u) {
      let sb = su >> 3u;
      let g = su & 7u;
      let n = g >> 2u;
      let k = g & 3u;
      let wb = (r * SB_PER_ROW + sb) * 53u;
      let d = unpack2x16float(blocks[wb + 52u]).x; // d f16 a byte 208
      // scale del gruppo n: 8 byte a partire da byte 192+n*8 = word 48+n*2
      let scW0 = blocks[wb + 48u + n * 2u];
      let scW1 = blocks[wb + 49u + n * 2u];
      let sc0 = s8(scW0, scW1, 2u * k);      // is = 0
      let sc1 = s8(scW0, scW1, 2u * k + 1u); // is = 1
      let qlW = wb + n * 16u + (k & 1u) * 8u; // ql: byte n*64 + (k&1)*32
      let qhW = wb + 32u + n * 8u;            // qh: byte 128 + n*32
      let xb = sb * 256u + n * 128u + k * 32u;
      acc0 = 0.0;
      acc1 = 0.0;
      for (var wI = 0u; wI < 8u; wI = wI + 1u) {
        let qlWord = blocks[qlW + wI];
        let qhWord = blocks[qhW + wI];
        for (var b = 0u; b < 4u; b = b + 1u) {
          let sh = b * 8u;
          let ql = (qlWord >> sh) & 0xffu;
          let qh = (qhWord >> sh) & 0xffu;
          // nibble basso per k<2 (q1,q2), alto per k>=2 (q3,q4); 2 bit alti a k*2
          let lo = select(ql >> 4u, ql & 0xfu, k < 2u);
          q = f32(lo | (((qh >> (k * 2u)) & 3u) << 4u)) - 32.0;
          let e = xb + wI * 4u + b;
          let xv = xs[e + (e >> 5u)];
          if (wI < 4u) { acc0 = acc0 + q * xv; } else { acc1 = acc1 + q * xv; }
        }
      }
      acc = acc + d * (sc0 * acc0 + sc1 * acc1);
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
  if (t == 0u && r < ${N}u) { ${outIdx} = partial[0]; }
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
//
// RESOLVE (slice B): con `resolve` il kernel non si ferma a id+pesi — scrive
// direttamente `Sel`, cioe' la stessa struttura che la CPU riempie in slice A.
// Lo slot lo prende dalla `slotTable` (expertKey → slot, mantenuta da
// ExpertCache), quindi la traduzione expert → indirizzo smette di essere una
// decisione CPU. E' un blocco IN CODA: la selezione — e con lei il tie-break —
// resta scritta una volta sola, e senza `resolve` il testo emesso e' identico
// byte per byte a quello di it.9 (i ktest router-top4-* restano gate senza
// riscrittura).
export function routerTopKWgsl(opts: {
  nExpert: number; nUsed: number; weightsScale: number; clampMin: number;
  /**
   * Funzione di gating (goal fase-D fase 3b). GLM-4.7-Flash usa sigmoid +
   * bias di selezione; qwen35moe usa softmax puro. E' lo stesso parametro che
   * `RouterConfig` porta su CPU in `moe.ts`: qui e' la sua trascrizione in
   * WGSL, non una seconda verita'.
   *
   * Il binding `bias` resta dichiarato in ENTRAMBI i casi: chi non lo usa ci
   * lega un buffer di zeri, e `probs[i] + 0.0` e' esatto — esattamente quello
   * che `routerSelect` fa con `sel.set(probs)` quando `usesBias` e' false.
   * Cosi' il layout dei binding non dipende dalla famiglia, e con
   * `gating: "sigmoid"` il testo emesso resta BYTE-IDENTICO a prima.
   */
  gating?: "sigmoid" | "softmax";
  /**
   * Coda di resolve. I due campi ripetono nExpert/nUsed perche' qui contano
   * come PASSI di indicizzazione (stride della slotTable per layer, stride di
   * Sel per layer MoE) e non come dimensioni del router: si asserta che
   * coincidano, cosi' la ripetizione non puo' diventare una seconda verita'.
   *
   * `dirty` (C3b decode ottimistico, spec §3b): aggiunge il binding 7
   * `dirtyB` — [0] = primo layer MoE con miss (atomicMin), [1] = conteggio
   * miss (atomicAdd) — scritto SOLO quando il resolve trova almeno un MISS.
   * Opt-in: senza, il WGSL emesso e' byte-identico a prima (shadow/gpu non
   * cambiano). Il sentinel di [0] e' 0xffffffff: lo azzera la CPU per token.
   */
  resolve?: { nExpert: number; nUsed: number; dirty?: boolean };
}): string {
  const { nExpert, nUsed, weightsScale, clampMin } = opts;
  const gating = opts.gating ?? "sigmoid";
  const WG = 64;
  const res = opts.resolve;
  const dirty = res?.dirty === true;
  if (res && (res.nExpert !== nExpert || res.nUsed !== nUsed)) {
    throw new Error(
      `routerTopK resolve: (${res.nExpert}, ${res.nUsed}) != (${nExpert}, ${nUsed}) — ` +
      "la slotTable e Sel devono avere gli stessi passi del router");
  }
  // Le due dichiarazioni in piu' e la coda esistono SOLO con resolve: senza,
  // entrambe sono la stringa vuota e il template torna quello di it.9.
  const resDecl = res
    ? `
@group(0) @binding(4) var<storage, read_write> selBuf: array<Sel>;
@group(0) @binding(5) var<storage, read> slotTable: array<u32>;
@group(0) @binding(6) var<uniform> moeIdx: MoeIdx;${dirty ? `
@group(0) @binding(7) var<storage, read_write> dirtyB: array<atomic<u32>>;` : ""}`
    : "";
  // le STESSE struct dei kernel d'arena, dalla stessa costante
  const resStruct = res ? `\n${SEL_STRUCT_WGSL}\n${MOE_IDX_STRUCT_WGSL}` : "";
  // `moeIdx.selIdx` e' la prima entry di Sel del layer (k=0) e `tableBase` la
  // base del layer nella slotTable: due valori CPU-noti che arrivano
  // dall'uniform a dynamic offset, mai dall'expert scelto.
  // Il binding si chiama `selBuf` e non `sel` perche' in questo kernel `sel` e'
  // gia' l'array workgroup dei punteggi di selezione (probs+bias); `selBuf` e'
  // anche il nome che la stessa struttura ha nei kernel d'arena.
  // sigmoid: probs e score di selezione si calcolano in parallelo, il bias
  // entra SOLO nella selezione. softmax: serve il massimo su tutti gli expert
  // PRIMA di esponenziare, quindi il prefill parallelo mette via i logit
  // grezzi e la normalizzazione la fa il thread 0 (tre passate su nExpert:
  // niente rispetto alle nExpert*nUsed della selezione che segue).
  const gatingFill = gating === "sigmoid"
    ? `
    let p = 1.0 / (1.0 + exp(-logits[i]));
    probs[i] = p;
    sel[i] = p + bias[i];   // il bias entra SOLO nella selezione`
    : `
    probs[i] = logits[i];   // grezzi: la softmax ha bisogno del massimo globale`;
  const gatingNorm = gating === "sigmoid" ? "" : `
  var mx = probs[0];
  for (var i = 1u; i < NE; i = i + 1u) { if (probs[i] > mx) { mx = probs[i]; } }
  var z = 0.0;
  for (var i = 0u; i < NE; i = i + 1u) { let e2 = exp(probs[i] - mx); probs[i] = e2; z = z + e2; }
  for (var i = 0u; i < NE; i = i + 1u) { probs[i] = probs[i] / z; sel[i] = probs[i] + bias[i]; }`;
  // Con `dirty` la coda cambia in 3 punti (var, conteggio, atomiche); senza,
  // il testo emesso resta BYTE-IDENTICO a prima — shadow e gpu non cambiano.
  const resTail = res
    ? (dirty
      ? `
  var nMiss = 0u;
  for (var k = 0u; k < NU; k = k + 1u) {
    let e = ids[k];
    let slot = slotTable[moeIdx.tableBase + e];
    let o = moeIdx.selIdx + k;
    selBuf[o].id = e;
    selBuf[o].slot = slot;
    selBuf[o].w = wts[k];
    selBuf[o].flags = select(0u, 1u, slot == 0xffffffffu); // bit 0 = miss risolto
    nMiss = nMiss + select(0u, 1u, slot == 0xffffffffu);
  }
  if (nMiss > 0u) {
    atomicMin(&dirtyB[0], moeIdx.moeLayer);
    atomicAdd(&dirtyB[1], nMiss);
  }`
      : `
  for (var k = 0u; k < NU; k = k + 1u) {
    let e = ids[k];
    let slot = slotTable[moeIdx.tableBase + e];
    let o = moeIdx.selIdx + k;
    selBuf[o].id = e;
    selBuf[o].slot = slot;
    selBuf[o].w = wts[k];
    selBuf[o].flags = select(0u, 1u, slot == 0xffffffffu); // bit 0 = miss risolto
  }`)
    : "";
  return `${resStruct}
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> wts: array<f32>;${resDecl}
const NE = ${nExpert}u;
const NU = ${nUsed}u;
var<workgroup> probs: array<f32, ${nExpert}>;
var<workgroup> sel: array<f32, ${nExpert}>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  for (var i = t; i < NE; i = i + ${WG}u) {${gatingFill}
  }
  workgroupBarrier();
  if (t != 0u) { return; }${gatingNorm}
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
  }${resTail}
}`;
}
