// Varianti di ATTENZIONE per la fase 0 di engine-kernel-decode.
//
// La BASELINE non vive qui: si importa `attnDecodeWgsl` da
// src/engine/kernels/wgsl.ts, cosi' la "forma attuale" e' il kernel vero e non
// una sua imitazione (il difetto che src/microbench/kernels.ts ha per
// costruzione: modella la forma TVM, non la nostra).
//
// Qui vivono SOLO le varianti candidate. Nessun file di src/engine/** e'
// toccato: le riscritture sono la fase 1, questa fase decide SE farle.

export const ATTN_SHAPE = {
  nHead: 16,
  nKvHead: 4,
  headDim: 256,
  ctxMax: 6400,
  n: 6333,
} as const;

const GROUPS = ATTN_SHAPE.nHead / ATTN_SHAPE.nKvHead; // 4 head per gruppo GQA
const HD4 = ATTN_SHAPE.headDim / 4;                    // 64 vec4 per head
const KV4 = (ATTN_SHAPE.nKvHead * ATTN_SHAPE.headDim) / 4; // 256 vec4 per posizione
const SCALE = 1 / Math.sqrt(ATTN_SHAPE.headDim);
const TOK_PARAMS = `struct TokParams { pos: u32, nPast: u32 };`;

/**
 * VARIANTE `vec4`: identica alla forma attuale (stesso `scores[ctxMax]`, stesso
 * un-workgroup-per-head, stessa mappatura thread->posizione), UNICA differenza
 * le letture `vec4<f32>` su headDim. Serve a isolare il costo delle load
 * scalari dal resto.
 */
export function attnVec4Wgsl(): string {
  return `${TOK_PARAMS}
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kCache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vCache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outv: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> P: TokParams;
const HD4 = ${HD4}u;
const KV4 = ${KV4}u;
const GROUPS = ${GROUPS}u;
const SCALE = ${SCALE};
var<workgroup> scores: array<f32, ${ATTN_SHAPE.ctxMax}>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let kvHead = h / GROUPS;
  let qOff = h * HD4;
  let kvBase = kvHead * HD4;
  let n = P.nPast + 1u;
  for (var p = t; p < n; p = p + 64u) {
    let kOff = p * KV4 + kvBase;
    var acc = 0.0;
    for (var i = 0u; i < HD4; i = i + 1u) { acc = acc + dot(q[qOff + i], kCache[kOff + i]); }
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
  // 64 thread, 64 vec4 di headDim: esattamente uno per thread, letture coalescenti
  var acc = vec4<f32>(0.0);
  for (var p = 0u; p < n; p = p + 1u) {
    acc = acc + scores[p] * vCache[p * KV4 + kvBase + t];
  }
  outv[qOff + t] = acc / sAll;
}`;
}

export interface StreamOpts {
  /** head per workgroup: 1 = una per head; 4 = un workgroup per gruppo GQA (KV letta 1 volta) */
  headsPerWg: 1 | 4;
  /** numero di chunk in cui si spezza il contesto (1 = niente split) */
  splits: number;
}

/**
 * FAMIGLIA STREAMING: softmax online a tile di 64 posizioni. Nessun
 * `scores[ctxMax]`: la workgroup memory diventa costante in ctxMax (1,5 KB con
 * headsPerWg=1, ~6 KB con 4) — che e' il requisito di PORTABILITA' del contratto,
 * indipendente dalla velocita'.
 *
 * `headsPerWg=4` fonde il gruppo GQA: la riga KV si legge UNA volta e serve le 4
 * head, invece di 4 volte. `splits>1` spezza il contesto su piu' workgroup e
 * produce parziali (acc non normalizzato, m, s) da ricombinare in log-sum-exp
 * con `attnCombineWgsl`.
 */
export function attnStreamWgsl(o: StreamOpts): string {
  const H = o.headsPerWg;
  const split = o.splits > 1;
  const chunkLen = split ? Math.ceil(Math.ceil(ATTN_SHAPE.n / o.splits) / 64) * 64 : 0;
  // indice del parziale: (head, chunk) -> riga di partOut / partMS
  const outBind = split
    ? `@group(0) @binding(3) var<storage, read_write> partOut: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> partMS: array<f32>;`
    : `@group(0) @binding(3) var<storage, read_write> outv: array<vec4<f32>>;`;
  const unroll = (body: (hh: number) => string): string =>
    Array.from({ length: H }, (_, i) => body(i)).join("\n    ");
  return `${TOK_PARAMS}
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kCache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vCache: array<vec4<f32>>;
${outBind}
@group(0) @binding(4) var<uniform> P: TokParams;
const HD4 = ${HD4}u;
const KV4 = ${KV4}u;
const GROUPS = ${GROUPS}u;
const SCALE = ${SCALE};
const NEG = -3.0e38;
var<workgroup> qsh: array<vec4<f32>, ${64 * H}>;
var<workgroup> sc: array<f32, ${64 * H}>;
var<workgroup> red: array<f32, ${64 * H}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let hBase = wid.x * ${H}u;
  let kvHead = hBase / GROUPS;
  let kvBase = kvHead * HD4;
  let nAll = P.nPast + 1u;
${split
  ? `  let chunk = wid.y;
  let pStart = chunk * ${chunkLen}u;
  let pEnd = min(nAll, pStart + ${chunkLen}u);`
  : `  let pStart = 0u;
  let pEnd = nAll;`}
  // q in memoria di gruppo una volta sola: il ciclo interno lo rilegge n volte
  ${unroll((hh) => `qsh[${hh * 64}u + t] = q[(hBase + ${hh}u) * HD4 + t];`)}
  workgroupBarrier();
  ${unroll((hh) => `var m${hh} = NEG; var s${hh} = 0.0; var acc${hh} = vec4<f32>(0.0);`)}
  var tile = pStart;
  loop {
    if (tile >= pEnd) { break; }
    let p = tile + t;
    ${unroll((hh) => `var d${hh} = NEG;`)}
    if (p < pEnd) {
      let kOff = p * KV4 + kvBase;
      ${unroll((hh) => `d${hh} = 0.0;`)}
      for (var i = 0u; i < HD4; i = i + 1u) {
        let kv = kCache[kOff + i];
        ${unroll((hh) => `d${hh} = d${hh} + dot(qsh[${hh * 64}u + i], kv);`)}
      }
      ${unroll((hh) => `d${hh} = d${hh} * SCALE;`)}
    }
    ${unroll((hh) => `sc[${hh * 64}u + t] = d${hh}; red[${hh * 64}u + t] = d${hh};`)}
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
      if (t < stride) {
        ${unroll((hh) => `red[${hh * 64}u + t] = max(red[${hh * 64}u + t], red[${hh * 64}u + t + stride]);`)}
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    ${unroll((hh) => `let tm${hh} = red[${hh * 64}u];`)}
    workgroupBarrier();
    ${unroll((hh) => `let nm${hh} = max(m${hh}, tm${hh}); let rs${hh} = exp(m${hh} - nm${hh}); m${hh} = nm${hh};`)}
    ${unroll((hh) => `let e${hh} = exp(sc[${hh * 64}u + t] - nm${hh}); sc[${hh * 64}u + t] = e${hh}; red[${hh * 64}u + t] = e${hh};`)}
    workgroupBarrier();
    stride = 32u;
    while (stride > 0u) {
      if (t < stride) {
        ${unroll((hh) => `red[${hh * 64}u + t] = red[${hh * 64}u + t] + red[${hh * 64}u + t + stride];`)}
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    ${unroll((hh) => `s${hh} = s${hh} * rs${hh} + red[${hh * 64}u];`)}
    workgroupBarrier();
    ${unroll((hh) => `acc${hh} = acc${hh} * rs${hh};`)}
    let cnt = min(64u, pEnd - tile);
    for (var j = 0u; j < cnt; j = j + 1u) {
      // la riga V si legge UNA volta e serve tutte le head del workgroup
      let vv = vCache[(tile + j) * KV4 + kvBase + t];
      ${unroll((hh) => `acc${hh} = acc${hh} + sc[${hh * 64}u + j] * vv;`)}
    }
    workgroupBarrier();
    tile = tile + 64u;
  }
${split
  ? `  ${unroll((hh) => `partOut[((hBase + ${hh}u) * ${o.splits}u + chunk) * HD4 + t] = acc${hh};`)}
  if (t == 0u) {
    ${unroll((hh) => `partMS[((hBase + ${hh}u) * ${o.splits}u + chunk) * 2u] = m${hh};
    partMS[((hBase + ${hh}u) * ${o.splits}u + chunk) * 2u + 1u] = s${hh};`)}
  }`
  : `  ${unroll((hh) => `outv[(hBase + ${hh}u) * HD4 + t] = acc${hh} / s${hh};`)}`}
}`;
}

/** Combinazione log-sum-exp dei parziali dello split: un workgroup per head. */
export function attnCombineWgsl(splits: number): string {
  return `
@group(0) @binding(0) var<storage, read> partOut: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> partMS: array<f32>;
@group(0) @binding(2) var<storage, read_write> outv: array<vec4<f32>>;
const HD4 = ${HD4}u;
const S = ${splits}u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  var gm = -3.0e38;
  for (var c = 0u; c < S; c = c + 1u) { gm = max(gm, partMS[(h * S + c) * 2u]); }
  var num = vec4<f32>(0.0);
  var den = 0.0;
  for (var c = 0u; c < S; c = c + 1u) {
    let w = exp(partMS[(h * S + c) * 2u] - gm);
    num = num + w * partOut[(h * S + c) * HD4 + t];
    den = den + w * partMS[(h * S + c) * 2u + 1u];
  }
  outv[h * HD4 + t] = num / den;
}`;
}

/** Lunghezza di chunk effettiva usata dallo split (multipli di 64, tile del kernel). */
export function splitChunkLen(splits: number): number {
  return Math.ceil(Math.ceil(ATTN_SHAPE.n / splits) / 64) * 64;
}
