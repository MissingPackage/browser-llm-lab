// Kernel WGSL della linear attention Qwen 3.5/3.6 (Gated DeltaNet) — fase 3
// q1, spec §4. Semantica = cpuref-f64 (src/engine/q35cpuref.ts, a sua volta
// dalla fonte llama.cpp b10333); il ktest confronta OGNI kernel col cpuref.
//
// Decomposizione (le proiezioni qkv/z/β/α e ssm_out usano i gemv esistenti,
// già ktestati):
//   1. deltaNetConvWgsl  — conv causale k=4 per canale + SiLU + SHIFT dello
//      stato conv (un thread per canale: nessuna race, colonna propria).
//   2. deltaNetGatesWgsl — β = sigmoid(βraw), g = a·softplus(αraw + dtBias)
//      (softplus con gomito a 20, come ggml).
//   3. deltaNetCoreWgsl  — un WORKGROUP per v-head (thread = colonna j dello
//      stato): L2-norm di q,k del k-head (h mod nK, floor eps), q/√hd,
//      ricorrenza decay→sk→delta→update→output IN UN SOLO passaggio dello
//      stato (il decay è fuso nelle due passate: mai una sweep separata),
//      gated norm RMS(o)·silu(z) con riduzione di workgroup.
//
// Stato: S in storage f32 (obbligo di config `mamba_ssm_dtype: float32`),
// layout [h][i·hd+j] i=key j=value — ogni thread j legge/scrive la SUA
// colonna (stride hd), zero conflitti.
//
// Regola della casa (landmine WGSL/Tint): ogni accumulatore dichiarato nei
// loop è inizializzato ESPLICITAMENTE.

export interface DeltaNetDimsWgsl {
  hd: number; // head dim (128 sul 4B; 16 nel campione)
  nK: number; // k-head
  nV: number; // v-head
  eps: number;
}

/**
 * conv causale per canale + SiLU + shift stato. Dispatch: ceil(C/64).
 *
 * `rows` (fase 4, it.30): x e outv sono matrici [M, C] e la RIGA arriva da un
 * uniform a dynamic offset — non da `gid.y` come negli altri kernel batched.
 * La ragione e' che questo kernel NON si batcha: e' la ricorrenza, e le righe
 * vanno eseguite IN ORDINE perche' la riga m legge lo stato che ha lasciato la
 * m−1. Quindi restano M dispatch, e cio' che serve e' solo dire a ciascuno su
 * quale riga lavorare. Lo stato NON e' per riga: e' la memoria che attraversa
 * il chunk. Senza `rows` il testo emesso e' IDENTICO byte per byte.
 */
export function deltaNetConvWgsl(qkvDim: number, convK: number, rows?: boolean): string {
  const K1 = convK - 1;
  const rBind = rows ? "\n@group(0) @binding(4) var<uniform> rowIdx: vec4<u32>;" : "";
  const rOff = rows ? "rowIdx.x * C + " : "";
  return `
@group(0) @binding(0) var<storage, read_write> state: array<f32>; // [${K1}*C] tempo-major
@group(0) @binding(1) var<storage, read> x: array<f32>;           // [C] qkv del token
@group(0) @binding(2) var<storage, read> w: array<f32>;           // [C*${convK}] riga per canale
@group(0) @binding(3) var<storage, read_write> outv: array<f32>;  // [C] post-SiLU${rBind}
const C = ${qkvDim}u;
const CK = ${convK}u;
const K1 = ${K1}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= C) { return; }
  var s = 0.0;
  for (var i = 0u; i < CK; i = i + 1u) {
    var v = 0.0;
    if (i < K1) { v = state[i * C + c]; } else { v = x[${rOff}c]; }
    s = s + w[c * CK + i] * v;
  }
  outv[${rOff}c] = s / (1.0 + exp(-s)); // silu
  // shift della storia del canale c (colonna propria: nessuna race)
  for (var i = 0u; i + 1u < K1; i = i + 1u) { state[i * C + c] = state[(i + 1u) * C + c]; }
  state[(K1 - 1u) * C + c] = x[${rOff}c];
}`;
}

/** β = sigmoid(βraw); g = a·softplus(αraw + dtBias). Dispatch: ceil(nV/64). */
export function deltaNetGatesWgsl(nV: number, batch?: boolean): string {
  // batch (fase 4): gid.y = riga. Le gate sono ROW-PARALLEL — dipendono solo da
  // beta/alpha della riga — a differenza di conv e core, che sono la ricorrenza
  // vera e restano per riga (misurata: 0,588 ms/token in tutto, it.23).
  // Senza `batch` il testo emesso e' IDENTICO a prima, byte per byte.
  const e = batch ? "gid.y * NV + h" : "h";
  return `
@group(0) @binding(0) var<storage, read> betaRaw: array<f32>;
@group(0) @binding(1) var<storage, read> alphaRaw: array<f32>;
@group(0) @binding(2) var<storage, read> aCoef: array<f32>;   // −exp(A_log), dal file
@group(0) @binding(3) var<storage, read> dtBias: array<f32>;
@group(0) @binding(4) var<storage, read_write> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> g: array<f32>;
const NV = ${nV}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= NV) { return; }
  beta[${e}] = 1.0 / (1.0 + exp(-betaRaw[${e}]));
  let sp = alphaRaw[${e}] + dtBias[h];
  var spv = sp;
  if (sp <= 20.0) { spv = log(1.0 + exp(sp)); } // softplus, gomito ggml
  g[${e}] = aCoef[h] * spv;
}`;
}

/**
 * Ricorrenza per v-head + gated norm. Dispatch: [nV] workgroup, wg size hd.
 * convOut layout: [q(nK·hd) | k(nK·hd) | v(nV·hd)].
 */
export function deltaNetCoreWgsl(d: DeltaNetDimsWgsl, rows?: boolean): string {
  const keyDim = d.nK * d.hd;
  // `rows` (fase 4, it.30): stessa ragione del conv — la ricorrenza non si
  // batcha, quindi la riga arriva da un uniform e i dispatch restano M, in
  // ordine. `S` non e' per riga: e' la memoria che attraversa il chunk.
  const rBind = rows ? "\n@group(0) @binding(7) var<uniform> rowIdx: vec4<u32>;" : "";
  const cvOff = rows ? `rowIdx.x * ${2 * keyDim + d.nV * d.hd}u + ` : "";
  const nvOff = rows ? `rowIdx.x * ${d.nV}u + ` : "";
  const vecOff = rows ? `rowIdx.x * ${d.nV * d.hd}u + ` : "";
  return `
@group(0) @binding(0) var<storage, read> convOut: array<f32>;   // [${2 * keyDim + d.nV * d.hd}]
@group(0) @binding(1) var<storage, read_write> S: array<f32>;   // [NV*HD*HD], [h][i*HD+j]
@group(0) @binding(2) var<storage, read> beta: array<f32>;      // [NV]
@group(0) @binding(3) var<storage, read> g: array<f32>;         // [NV]
@group(0) @binding(4) var<storage, read> z: array<f32>;         // [NV*HD] gate della norm
@group(0) @binding(5) var<storage, read> normW: array<f32>;     // [HD]
@group(0) @binding(6) var<storage, read_write> outv: array<f32>; // [NV*HD]${rBind}
const HD = ${d.hd}u;
const NK = ${d.nK}u;
const KEYDIM = ${keyDim}u;
const EPS = ${d.eps};
const QSCALE = ${1 / Math.sqrt(d.hd)};
var<workgroup> qSh: array<f32, ${d.hd}>;
var<workgroup> kSh: array<f32, ${d.hd}>;
var<workgroup> red: array<f32, ${d.hd}>;
@compute @workgroup_size(${d.hd})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;            // = colonna j dello stato
  let h = wid.x;            // v-head
  let kHead = h % NK;       // broadcast ggml_repeat: TILING, non gruppi
  let qRaw = convOut[${cvOff}kHead * HD + t];
  let kRaw = convOut[${cvOff}KEYDIM + kHead * HD + t];
  let vj = convOut[${cvOff}2u * KEYDIM + h * HD + t];

  // L2-norm di k (floor eps, semantica ggml_l2_norm)
  red[t] = kRaw * kRaw;
  workgroupBarrier();
  var stride = HD / 2u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let kScale = 1.0 / max(sqrt(red[0]), EPS);
  workgroupBarrier();
  // L2-norm di q + scala 1/√hd
  red[t] = qRaw * qRaw;
  workgroupBarrier();
  stride = HD / 2u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let qScale = (1.0 / max(sqrt(red[0]), EPS)) * QSCALE;
  workgroupBarrier();
  kSh[t] = kRaw * kScale;
  qSh[t] = qRaw * qScale;
  workgroupBarrier();

  let base = h * HD * HD;
  let decay = exp(g[${nvOff}h]);
  // passata 1: sk_j con S decaduto (decay fuso, niente sweep separata)
  var sk = 0.0;
  for (var i = 0u; i < HD; i = i + 1u) { sk = sk + S[base + i * HD + t] * decay * kSh[i]; }
  let dlt = beta[${nvOff}h] * (vj - sk);
  // passata 2: update colonna j + output o_j = Σ_i S_new[i][j]·q[i]
  var o = 0.0;
  for (var i = 0u; i < HD; i = i + 1u) {
    let sN = S[base + i * HD + t] * decay + kSh[i] * dlt;
    S[base + i * HD + t] = sN;
    o = o + sN * qSh[i];
  }

  // gated norm: RMS(o) per head · normW · silu(z)
  red[t] = o * o;
  workgroupBarrier();
  stride = HD / 2u;
  while (stride > 0u) {
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let inv = 1.0 / sqrt(red[0] / f32(HD) + EPS);
  let zj = z[${vecOff}h * HD + t];
  outv[${vecOff}h * HD + t] = o * inv * normW[t] * (zj / (1.0 + exp(-zj)));
}`;
}
