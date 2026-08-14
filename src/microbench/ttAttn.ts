// Varianti dell'ATTENZIONE A CHUNK DEL PREFILL per la riga 1 di engine-ttft.
//
// Pre-registrazione: docs/deep-dive/ttft-riga1-prereg-2026-08-13.md (P5).
//
// La BASELINE non vive qui: si importa da src/engine/kernels/wgsl.ts, cosi' il
// termine di paragone e' un kernel vero e non una sua imitazione. Fino al task
// T1-kernel-batch-streaming quel kernel era `attnDecodeWgsl({ batch: true })`,
// cioe' la forma che il prefill del 4B usava DAVVERO — `scores: array<f32,
// ctxMax>` in workgroup memory, letture scalari, un workgroup per (head, riga) e
// quindi la riga KV riletta una volta per head del gruppo GQA: gli stessi tre
// difetti gia' chiusi sul decode e mai portati al prefill.
//
// DA OGGI QUELLA FORMA E' IN PRODUZIONE, e la baseline si importa da
// `attnDecodeLegacyBatchWgsl` — lo stesso testo di prima, byte per byte, sotto
// il nome che dichiara cos'e' diventato: un fallback. Il cambio di nome NON e'
// cosmetico: chiamare ancora `attnDecodeWgsl({ batch: true })` farebbe misurare
// alla sonda streaming contro streaming, e il numero uscirebbe lo stesso — solo
// falso. Lo tiene fermo tests/ttattn-prefill-prodvslegacy.test.ts («la lista
// dice il vero su se stessa»), perche' questo file gira solo su GPU.
//
// Qui vivono le candidate: softmax in STREAMING (workgroup storage costante in
// ctxMax), letture vec4, KV letta UNA volta per gruppo GQA, e la fusione su piu'
// RIGHE del chunk (che sul decode non esisteva: e' la leva specifica del
// prefill, dove M righe guardano quasi la stessa KV). La vincitrice misurata e'
// `headsPerWg: 1, rowsPerWg: 1` — la fusione GQA PEGGIORA (occupancy).
//
// E DA OGGI VIVE QUI ANCHE LA LISTA, `prefillAttnVariants()`: il runner la
// consuma invece di tenerne una sua. Non e' un riordino estetico — e' cio' che
// rende il PRIMA/DOPO di AC2 leggibile da un test senza GPU. I due bracci che
// contano sono `legacy` (il kernel di ieri, che il prefill del 4B usava davvero)
// e `prod` (il kernel di produzione di oggi, dopo T1-kernel-batch-streaming):
// entrambi importati da src/engine/kernels/wgsl.ts, perche' un prima/dopo fatto
// con due imitazioni locali non misura il motore, misura questo file.

import { attnDecodeLegacyBatchWgsl, attnDecodeWgsl } from "../engine/kernels/wgsl";

export const TT_ATTN_SHAPE = {
  nHead: 16,
  nKvHead: 4,
  headDim: 256,
  /** ctxMax del 4B: e' il numero che dimensiona `scores[ctxMax]` del legacy */
  ctxMax: 6400,
} as const;

/**
 * Righe del chunk misurate dalla sonda. Vive QUI e non nel runner perche' le
 * griglie dei bracci la contengono ([nHead, M, 1]): il runner la ri-esporta come
 * `ATTN_M`, valore invariato, cosi' il 16 esiste in un posto solo.
 */
export const TT_ATTN_M = 16;

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

/**
 * LA LISTA CHE IL RUNNER MISURA — un solo posto, e quindi un solo prima/dopo.
 *
 * I due bracci di AC2:
 *   `legacy` — `attnDecodeLegacyBatchWgsl`, il kernel che il prefill a chunk del
 *     4B usava DAVVERO fino a T1-kernel-batch-streaming: `scores[ctxMax]` in
 *     workgroup memory (4·ctxMax + 256 B, che cresce col contesto), letture
 *     scalari, un workgroup per (head, riga). E' il PRIMA, e resta il fallback
 *     dichiarato del motore: per questo si misura ancora.
 *   `prod` — `attnDecodeWgsl({ batch: true })`, cioe' cio' che il motore compila
 *     OGGI: softmax in streaming a tile di 64, letture vec4, workgroup storage
 *     COSTANTE in ctxMax (1.536 B). E' il DOPO.
 *
 * I due si dispacciano sulla STESSA griglia [nHead, M, 1] e sugli STESSI
 * binding ([q, kCache, vCache, outv, rowPast]): cio' che cambia fra le due celle
 * dell'artefatto e' il kernel, non il banco.
 *
 * Le altre tre restano come RECORD MISURATO, non come candidate: `stream` e' la
 * gemella-microbench di `prod` (stessa forma, codegen di questo file — se le due
 * celle divergessero, la porta nel motore avrebbe perso qualcosa per strada), e
 * `gqa-stream`/`gqa-rows2` sono la fusione GQA che la misura ha BOCCIATO (meno
 * traffico KV, meno workgroup: sul device di riferimento vince il secondo
 * effetto — docket item 21, cifre in results/microbench/ttft-riga1-*.json).
 * Restano perche' un numero che ha deciso qualcosa si rimisura quando il banco
 * cambia; non diventano `prod` per nessun motivo.
 *
 * Nota sulle stringhe `ctx`: finiscono NELL'ARTEFATTO come descrizione del
 * braccio, quindi dicono cosa il kernel E' (forma, storage, traffico) e mai
 * quanti ms ha fatto — un millisecondo scritto qui resterebbe congelato alla run
 * che lo produsse. Lo tiene fermo tests/ttattn-prefill-prodvslegacy.test.ts.
 */
export function prefillAttnVariants(): Array<{
  id: string; code: string; grid: [number, number, number]; kvEmit: number; ctx: string;
}> {
  const { nHead, nKvHead, ctxMax } = TT_ATTN_SHAPE;
  const M = TT_ATTN_M;
  const groups = nHead / nKvHead;
  return [
    {
      id: "legacy", code: attnDecodeLegacyBatchWgsl(TT_ATTN_SHAPE), grid: [nHead, M, 1], kvEmit: groups * M,
      ctx: `PRIMA (baseline) importata da src/engine/kernels/wgsl.ts (attnDecodeLegacyBatchWgsl): scores[ctxMax=${ctxMax}] in workgroup memory, letture scalari, un workgroup per (head, riga) — la riga KV si rilegge una volta per ognuna delle ${groups} head del gruppo GQA e una volta per ognuna delle ${M} righe. Era la forma di PRODUZIONE del prefill a chunk fino al task T1-kernel-batch-streaming; da li' in poi e' il fallback dichiarato, e resta il termine di paragone di questa sonda con lo stesso testo byte per byte.`,
    },
    {
      id: "prod", code: attnDecodeWgsl({ ...TT_ATTN_SHAPE, batch: true }), grid: [nHead, M, 1], kvEmit: groups * M,
      ctx: `DOPO (produzione) importata da src/engine/kernels/wgsl.ts (attnDecodeWgsl con batch: true): softmax in STREAMING a tile di 64 posizioni — niente scores[ctxMax], il workgroup storage e' 1.536 B COSTANTI in ctxMax — e letture vec4. Griglia e binding identici alla baseline: un workgroup per (head, riga), quindi il traffico KV EMESSO e' lo stesso (${groups} head del gruppo GQA x ${M} righe) e cio' che cambia e' solo il kernel.`,
    },
    {
      id: "stream", code: attnPrefillStreamWgsl({ headsPerWg: 1, rowsPerWg: 1 }),
      grid: attnPrefillGrid({ headsPerWg: 1, rowsPerWg: 1 }, M), kvEmit: groups * M,
      ctx: "gemella-microbench del braccio `prod`: stessa forma (streaming + vec4, un workgroup per (head, riga)) ma generata da src/microbench/ttAttn.ts. Non e' una candidata — e' il controllo che la porta nel motore non abbia perso niente per strada: se questa cella e `prod` divergono, la differenza sta nel codegen del motore, non nella forma.",
    },
    {
      id: "gqa-stream", code: attnPrefillStreamWgsl({ headsPerWg: 4, rowsPerWg: 1 }),
      grid: attnPrefillGrid({ headsPerWg: 4, rowsPerWg: 1 }, M), kvEmit: M,
      ctx: "streaming + vec4 + KV letta UNA volta per gruppo GQA (un workgroup per (gruppo, riga)): il traffico KV scende di 4x rispetto alla baseline. MISURATA E BOCCIATA (docket item 21): meno traffico, ma un quarto dei workgroup, e sul device di riferimento vince il secondo effetto. Resta come record.",
    },
    {
      id: "gqa-rows2", code: attnPrefillStreamWgsl({ headsPerWg: 4, rowsPerWg: 2 }),
      grid: attnPrefillGrid({ headsPerWg: 4, rowsPerWg: 2 }, M), kvEmit: M / 2,
      ctx: "streaming + vec4 + fusione GQA + DUE righe del chunk per workgroup: la riga KV serve 4 head x 2 righe, cioe' 8 consumatori invece di 1. E' la leva specifica del prefill, che sul decode non esisteva — e che la misura ha bocciato con la stessa dinamica di `gqa-stream` (occupancy). Resta come record.",
    },
  ];
}
