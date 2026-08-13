// Varianti dell'ATTENZIONE A CHUNK DEL PREFILL per la riga 1 di engine-ttft.
//
// Pre-registrazione: docs/deep-dive/ttft-riga1-prereg-2026-08-13.md (P5).
//
// La forma attuale non vive qui: si importa `attnDecodeWgsl({ batch: true })` da
// src/engine/kernels/wgsl.ts, che instrada su `attnDecodeLegacyWgsl` (switch a
// wgsl.ts:530) — `scores: array<f32, ctxMax>` in workgroup memory, letture
// scalari, un workgroup per (head, riga) e quindi la riga KV riletta una volta
// per head del gruppo GQA. Sono gli stessi tre difetti gia' chiusi sul decode e
// mai portati al prefill.
//
// Qui vivono le candidate: softmax in STREAMING (workgroup storage costante in
// ctxMax), letture vec4, KV letta UNA volta per gruppo GQA, e la fusione su piu'
// RIGHE del chunk (che sul decode non esisteva: e' la leva specifica del
// prefill, dove M righe guardano quasi la stessa KV).

export const TT_ATTN_SHAPE = {
  nHead: 16,
  nKvHead: 4,
  headDim: 256,
  /** ctxMax del 4B: e' il numero che dimensiona `scores[ctxMax]` del legacy */
  ctxMax: 6400,
} as const;

const GROUPS = TT_ATTN_SHAPE.nHead / TT_ATTN_SHAPE.nKvHead;
const HD4 = TT_ATTN_SHAPE.headDim / 4;
const KV4 = (TT_ATTN_SHAPE.nKvHead * TT_ATTN_SHAPE.headDim) / 4;
const ROWQ4 = (TT_ATTN_SHAPE.nHead * TT_ATTN_SHAPE.headDim) / 4;
const SCALE = 1 / Math.sqrt(TT_ATTN_SHAPE.headDim);

export interface PrefillAttnOpts {
  /** head per workgroup: 1 = una per head (come il legacy); 4 = un gruppo GQA intero */
  headsPerWg: 1 | 4;
  /** righe del chunk per workgroup: 1 = una per riga (come il legacy); >1 fonde le righe */
  rowsPerWg: 1 | 2;
}

/**
 * FAMIGLIA STREAMING per il prefill a chunk. Griglia (X = head o gruppo GQA,
 * Y = 1, Z = riga del chunk / rowsPerWg), workgroup da 64 thread.
 *
 * Ogni riga m del chunk ha il SUO `nPast` (`rowPast[m]`, la causalita' arriva da
 * li' come nel legacy): con `rowsPerWg = 2` il tile scorre fino al massimo delle
 * due righe e le posizioni oltre il proprio `nPast` entrano con score -inf, che
 * la softmax online annulla senza NaN (exp(-3e38 - m) = 0 con m finito).
 *
 * Workgroup storage: qsh + sc + red, tutti proporzionali a headsPerWg·rowsPerWg
 * e COSTANTI in ctxMax — 1.536 B a (1,1), 6.144 B a (4,1), 12.288 B a (4,2).
 */
export function attnPrefillStreamWgsl(o: PrefillAttnOpts): string {
  const H = o.headsPerWg;
  const R = o.rowsPerWg;
  const P = H * R; // coppie (head, riga) servite dal workgroup
  const pairs = <T>(f: (hh: number, rr: number, i: number) => T): T[] => {
    const out: T[] = [];
    for (let rr = 0; rr < R; rr++) for (let hh = 0; hh < H; hh++) out.push(f(hh, rr, rr * H + hh));
    return out;
  };
  const un = (f: (hh: number, rr: number, i: number) => string): string => pairs(f).join("\n    ");
  return `@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kCache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vCache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> rowPast: array<u32>;
const HD4 = ${HD4}u;
const KV4 = ${KV4}u;
const ROWQ4 = ${ROWQ4}u;
const GROUPS = ${GROUPS}u;
const SCALE = ${SCALE};
const NEG = -3.0e38;
var<workgroup> qsh: array<vec4<f32>, ${64 * P}>;
var<workgroup> sc: array<f32, ${64 * P}>;
var<workgroup> red: array<f32, ${64 * P}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let hBase = wid.x * ${H}u;
  let rBase = wid.z * ${R}u;
  let kvHead = hBase / GROUPS;
  let kvBase = kvHead * HD4;
  ${Array.from({ length: R }, (_, rr) => `let n${rr} = rowPast[rBase + ${rr}u] + 1u;`).join("\n  ")}
  let nMax = ${Array.from({ length: R }, (_, rr) => `n${rr}`).reduce((a, b) => `max(${a}, ${b})`)};
  // q in workgroup memory una volta sola: il ciclo interno la rilegge nMax volte
  ${un((hh, rr, i) => `qsh[${i * 64}u + t] = q[(rBase + ${rr}u) * ROWQ4 + (hBase + ${hh}u) * HD4 + t];`)}
  workgroupBarrier();
  ${un((_h, _r, i) => `var m${i} = NEG; var s${i} = 0.0; var acc${i} = vec4<f32>(0.0);`)}
  var tile = 0u;
  loop {
    if (tile >= nMax) { break; }
    let p = tile + t;
    ${un((_h, _r, i) => `var d${i} = NEG;`)}
    if (p < nMax) {
      let kOff = p * KV4 + kvBase;
      ${un((_h, rr, i) => `if (p < n${rr}) { d${i} = 0.0; }`)}
      for (var i = 0u; i < HD4; i = i + 1u) {
        let kv = kCache[kOff + i];
        ${un((_h, _r, i) => `d${i} = d${i} + dot(qsh[${i * 64}u + i], kv);`)}
      }
      ${un((_h, rr, i) => `if (p < n${rr}) { d${i} = d${i} * SCALE; } else { d${i} = NEG; }`)}
    }
    ${un((_h, _r, i) => `sc[${i * 64}u + t] = d${i}; red[${i * 64}u + t] = d${i};`)}
    workgroupBarrier();
    var stride = 32u;
    while (stride > 0u) {
      if (t < stride) {
        ${un((_h, _r, i) => `red[${i * 64}u + t] = max(red[${i * 64}u + t], red[${i * 64}u + t + stride]);`)}
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    ${un((_h, _r, i) => `let tm${i} = red[${i * 64}u];`)}
    workgroupBarrier();
    ${un((_h, _r, i) => `let nm${i} = max(m${i}, tm${i}); let rs${i} = exp(m${i} - nm${i}); m${i} = nm${i};`)}
    ${un((_h, _r, i) => `let e${i} = exp(sc[${i * 64}u + t] - nm${i}); sc[${i * 64}u + t] = e${i}; red[${i * 64}u + t] = e${i};`)}
    workgroupBarrier();
    stride = 32u;
    while (stride > 0u) {
      if (t < stride) {
        ${un((_h, _r, i) => `red[${i * 64}u + t] = red[${i * 64}u + t] + red[${i * 64}u + t + stride];`)}
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    ${un((_h, _r, i) => `s${i} = s${i} * rs${i} + red[${i * 64}u];`)}
    workgroupBarrier();
    ${un((_h, _r, i) => `acc${i} = acc${i} * rs${i};`)}
    let cnt = min(64u, nMax - tile);
    for (var j = 0u; j < cnt; j = j + 1u) {
      // la riga V si legge UNA volta e serve tutte le coppie (head, riga) del wg
      let vv = vCache[(tile + j) * KV4 + kvBase + t];
      ${un((_h, _r, i) => `acc${i} = acc${i} + sc[${i * 64}u + j] * vv;`)}
    }
    workgroupBarrier();
    tile = tile + 64u;
  }
  ${un((hh, rr, i) => `outv[(rBase + ${rr}u) * ROWQ4 + (hBase + ${hh}u) * HD4 + t] = acc${i} / s${i};`)}
}`;
}

/** Griglia della famiglia streaming: (head o gruppi, 1, righe/rowsPerWg). */
export function attnPrefillGrid(o: PrefillAttnOpts, M: number): [number, number, number] {
  const x = o.headsPerWg === 4 ? TT_ATTN_SHAPE.nKvHead : TT_ATTN_SHAPE.nHead;
  return [x, 1, M / o.rowsPerWg];
}
