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

import { q35AttnSplitPlan } from "../q35attnsplit";

export const TOK_PARAMS_WGSL = `struct TokParams { pos: u32, nPast: u32 };`;

// Larghezza X della griglia 2D dei gemv (limite WebGPU: 65535 wg/dimensione).
export const GEMV_GRID_X = 32768;
export function gemvGrid(N: number): [number, number] {
  return N <= GEMV_GRID_X ? [N, 1] : [GEMV_GRID_X, Math.ceil(N / GEMV_GRID_X)];
}

export interface GemvQuantOpts {
  kind: "q4_0" | "q4_1" | "q8_0"; K: number; N: number; hasBias: boolean;
  /** batch (fase 5): wid.z = riga, x/y a offset di riga — testo non-batch invariato */
  batch?: boolean;
  // MoE (C2 fase 5): y[r] += accScale[0]·dot — il down per-expert accumula il
  // contributo pesato direttamente su moe_out (pesatura DOPO il down, come
  // ffn_moe_weighted in build_moe_ffn; l'ordine delle somme sui 4 expert
  // differisce dall'oracolo solo al rounding f32).
  scaledAccum?: boolean;
  /**
   * Forma `vec4-rows2` misurata in fase 0 (src/microbench/kdGemv.ts): load
   * vettoriali + `dot()`, DUE righe per workgroup da 64 thread. Ammessa solo su
   * q4_0 nudo — vedi `gemvQuantVec4Rows2Wgsl`. Assente/false ⇒ il testo generato
   * è byte per byte quello di prima (è così che GLM non regredisce).
   */
  vec4Rows2?: boolean;
  /**
   * Riduzione di riga via `subgroupAdd` invece che ad albero. NON è una feature
   * che questo modulo decide: arriva già decisa da `gemvCapsFor` (src/engine/
   * gemvcaps.ts), che la concede solo con subgroup fisso a 32 lane. Qui si
   * genera soltanto. Richiede `vec4Rows2`: fuori da quella forma non esiste un
   * kernel dove il subgroup mappi una riga.
   */
  sg?: boolean;
}

// GEMV dequant-fusa: y[r] = Σ_b scale(b)·Σ_j q_j·x[...] (+ bias). Un workgroup da
// 64 thread per riga di output; riduzione in shared memory.
export function gemvQuantWgsl(opts: GemvQuantOpts): string {
  if (opts.vec4Rows2 === true) return gemvQuantVec4Rows2Wgsl(opts);
  if (opts.sg === true) {
    throw new Error(
      "gemvQuantWgsl: sg=true senza vec4Rows2 — la riduzione per subgroup esiste " +
      "solo nella forma a 2 righe per workgroup misurata in fase 0; il kernel a 1 " +
      "riga per workgroup non ha una mappatura riga->subgroup da usare");
  }
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

// ─────────────────────────────────────────────────────────────────────────────
// FORMA `vec4-rows2` DEL GEMV q4_0.
//
// Cosa cambia rispetto al kernel sopra, e perché. Il kernel a 1 riga per
// workgroup fa 4 load scalari di u32 per blocco e legge x un f32 alla volta: in
// fase 0 (src/microbench/kdGemv.ts, celle `vec4`/`vec4-rows2-sg`) la variante
// con UNA load `vec4<u32>` per blocco, `dot()` per quarto di blocco e DUE righe
// per workgroup da 64 thread ha misurato il salto. La forma qui sotto è quella,
// riportata identica: la misura vale per la forma misurata, non per una sua
// parente. Il ramo `sg` è BYTE PER BYTE il testo di `gemvVec4Rows2SgWgsl`
// (src/microbench/kdGemv.ts) — non "ispirato a": identico, così che un test
// possa confrontarli senza normalizzare e una divergenza si veda subito.
//
// PERCHÉ SOLO q4_0 NUDO. Fase 0 ha misurato q4_0 senza bias, senza batch e
// senza accumulo scalato. q8_0/q4_1/K-quant hanno layout di blocco diversi (8
// word, oppure d+m per blocco): il corpo andrebbe riscritto e la misura rifatta.
// Finché non esiste, l'opzione lancia invece di generare un kernel non misurato.
//
// VINCOLO DI CALL-SITE (WebGPU, non opinione). `array<vec4<u32>>` e
// `array<vec4<f32>>` hanno stride 16 B: la size della *binding* deve essere
// multipla di 16. Un tensore le cui righe di x, o il cui buffer qs, non
// arrivano a un multiplo di 16 B NON può usare questo kernel — l'esclusione si
// decide al call-site (censimento a parte), non qui: questo modulo genera testo
// e non vede i buffer.
//
// DUE RAMI DI RIDUZIONE, non uno parametrico:
//  - `sg`: una riga per subgroup da 32 lane, la riduzione è UN `subgroupAdd` e
//    nessuna barriera. Nessun early return: `subgroupAdd` vuole control flow
//    subgroup-uniform e l'analisi di Tint non prova che `r` lo sia (dipende da
//    t/sgSize). Da qui il `if (r < N_ROWS)` attorno al solo loop.
//  - fallback: 2 righe × 32 lane, riduzione ad albero su `partial[64]` con
//    `lane < stride`. NON è il codice a 64 lane per riga riusato: con 2 righe
//    per workgroup ogni riga possiede metà workgroup, e lo stride parte da 16.
// La guardia `r >= N_ROWS` c'è su ENTRAMBI i rami: con N dispari (e N dispari
// capita) l'ultimo workgroup ha una riga sola.
//
// DOVE STA LA DIFESA CONTRO subgroupSize ≠ 32 (e perché non è qui). Il ramo
// `sg` assume 32 lane per riga: su wave64 `sub = t / sgSize` vale 0 per tutti i
// thread e la riga dispari non verrebbe MAI scritta (y resta al valore vecchio,
// in silenzio). Quel controllo è già scritto, una volta sola, in
// src/engine/gemvcaps.ts: `gemvCapsFor` concede sg SOLO con
// subgroupMinSize === subgroupMaxSize === 32, e chi passa `sg: true` senza
// esserselo fatto dire da lì sta rompendo quel contratto. In-shader NON si
// aggiunge un `if (sgSize != 32u)`: sarebbe un branch su un valore che l'analisi
// di uniformità di Tint non tratta come uniforme (è la ragione per cui la forma
// misurata evita anche l'early return su `r`), e qui non c'è alcun Tint in CI
// per verificare che compili ancora. Difesa a monte, dove è verificabile.
// ─────────────────────────────────────────────────────────────────────────────

/** Righe di output per workgroup: 2 nella forma vec4-rows2, 1 in tutte le altre. */
/**
 * La combinazione e' ammissibile per il kernel a 2 righe? UNA definizione, usata
 * dal generatore E dalla griglia.
 *
 * Prima erano due: `gemvQuantRowsPerWg` diceva 2 sulla sola base di
 * `vec4Rows2`, mentre il generatore LANCIA su q8_0/q4_1/bias/batch/scaledAccum.
 * Per una combinazione rifiutata la griglia usciva dimezzata — meta' delle
 * righe non calcolate — per un kernel che non esiste. Il test lo ha preso
 * (fase 2, it.7): due posti che decidono la stessa cosa finiscono per
 * deciderla diversamente.
 */
export function gemvQuantVec4Rows2Ok(opts: GemvQuantOpts): boolean {
  return opts.vec4Rows2 === true
    && opts.kind === "q4_0"
    && opts.hasBias !== true
    && opts.batch !== true
    && opts.scaledAccum !== true;
}

export function gemvQuantRowsPerWg(opts: GemvQuantOpts): number {
  return gemvQuantVec4Rows2Ok(opts) ? 2 : 1;
}

/**
 * Griglia di dispatch del gemv quant. Sta accanto al kernel apposta: le righe
 * per workgroup sono una proprietà del TESTO generato, e un call-site che
 * calcolasse `gemvGrid(N)` a mano su un kernel a 2 righe dispatcherebbe il
 * doppio dei workgroup necessari (metà a vuoto, ma con la guardia che regge).
 */
export function gemvQuantGrid(opts: GemvQuantOpts): [number, number] {
  return gemvGrid(Math.ceil(opts.N / gemvQuantRowsPerWg(opts)));
}

function gemvQuantVec4Rows2Wgsl(opts: GemvQuantOpts): string {
  const { kind, K, N, hasBias, batch, scaledAccum, sg } = opts;
  const incompat: string[] = [];
  if (kind !== "q4_0") incompat.push(`kind "${kind}"`);
  if (hasBias) incompat.push("hasBias");
  if (batch) incompat.push("batch");
  if (scaledAccum) incompat.push("scaledAccum");
  if (incompat.length > 0) {
    throw new Error(
      `gemvQuantWgsl: vec4Rows2 non è ammesso con ${incompat.join(" + ")} — ` +
      "fase 0 ha misurato questa forma solo sul q4_0 nudo del decode, e un " +
      "kernel non misurato non si genera");
  }
  if (K % 32 !== 0) throw new Error("gemv: K non multiplo di 32");

  // Corpo per blocco: UNA load vec4<u32> (16 B), dot() per quarto di blocco.
  // Nel blocco q4_0 il byte j porta il nibble basso dell'elemento j e quello
  // alto dell'elemento j+16 — da cui le due dot() su x4 sfalsate di 4 vec4
  // (`xb + wi` e `xb + 4u + wi`, con xb = b*8: 8 vec4<f32> = 32 pesi per blocco).
  const blockBody = `
      let gb = r * BLOCKS_PER_ROW + b;
      let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];
      let w = qs4[gb];
      let xb = b * 8u;
      var bd = 0.0;
      for (var wi = 0u; wi < 4u; wi = wi + 1u) {
        let by = (vec4<u32>(w[wi]) >> vec4<u32>(0u, 8u, 16u, 24u));
        let lo = vec4<f32>(by & vec4<u32>(15u)) - vec4<f32>(8.0);
        let hi = vec4<f32>((by >> vec4<u32>(4u)) & vec4<u32>(15u)) - vec4<f32>(8.0);
        bd = bd + dot(lo, x4[xb + wi]) + dot(hi, x4[xb + 4u + wi]);
      }
      acc = acc + sc * bd;`;
  // Testa identica a `HEAD` di kdGemv.ts, newline iniziale compresa: senza
  // `enable subgroups;` la prima riga del testo resta vuota, ed è così anche là.
  const head = `${sg === true ? "enable subgroups;\n" : ""}
@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
const BLOCKS_PER_ROW = ${K / 32}u;
const N_ROWS = ${N}u;`;

  if (sg === true) {
    return `${head}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgSize: u32, @builtin(subgroup_invocation_id) sgId: u32) {
  let t = lid.x;
  let sub = t / sgSize;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 2u + sub;
  var acc = 0.0;
  // niente early return: 'subgroupAdd' vuole control flow subgroup-uniform e
  // l'analisi di Tint non prova che 'r' lo sia (dipende da t/sgSize)
  if (r < N_ROWS) {
    for (var b = sgId; b < BLOCKS_PER_ROW; b = b + sgSize) {${blockBody}
    }
  }
  let tot = subgroupAdd(acc);
  if (sgId == 0u && r < N_ROWS) { y[r] = tot; }
}`;
  }
  return `${head}
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let lane = t & 31u;
  let sub = t >> 5u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 2u + sub;
  var acc = 0.0;
  if (r < N_ROWS) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 32u) {${blockBody}
    }
  }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 16u;
  while (stride > 0u) {
    if (lane < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lane == 0u && r < N_ROWS) { y[r] = partial[sub << 5u]; }
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

// ─────────────────────────────────────────────────────────────────────────────
// ATTENTION DECODE (GQA).
//
// Il decode e' `attnDecodeWgsl` senza `batch`: split sul contesto + softmax in
// STREAMING (nessun array di workgroup dimensionato su ctxMax). Il prefill e'
// lo stesso nome con `batch: true` — e dal task T1-kernel-batch-streaming e'
// anche lui in streaming, a tile di 64 posizioni e con letture vec4.
//
// PERCHE' IL PREFILL E' STATO PORTATO ADESSO E NON PRIMA. Prima mancava la
// misura: il collo del prefill non e' ovvio come quello del decode, e una
// riscrittura senza sonda e' rischio puro. Le sonde della riga 1 di engine-ttft
// (src/microbench/ttAttn.ts, banco `attn-prefill-chunk`) l'hanno data, e
// hanno detto anche cio' che NON va fatto: fondere le teste del gruppo GQA per
// leggere la KV una volta sola PEGGIORA (2,0879 contro 1,8207 ms a ctx 6333),
// perche' taglia il traffico 4x ma porta i workgroup da 256 a 64 su 76 SM.
// Quindi: streaming e vec4 SI, un workgroup per (head, riga) resta (docket
// item 21).
//
// IL LEGACY NON SI CANCELLA. `attnDecodeLegacyBatchWgsl` resta esportata come
// fallback dichiarato e come termine di paragone: e' la BASELINE della sonda
// (src/microbench/ttRunner.ts) e cio' su cui gpulimits.test.ts misura quanto
// valeva il debito. `attnDecodeRefWgsl` e' il riferimento per-riga del ktest.
// Il testo di entrambi e' inchiodato da un golden
// (tests/fixtures/attn-decode-legacy.golden.json): se qualcuno li tocca per
// sbaglio, il test lo dice.
//
// COSA IL CAMBIO COSTA A CHI VERIFICA. Il ramo batch non e' piu' bit-identico
// al riferimento per-riga — riassocia le somme (rescale online fra tile,
// `dot()` su vec4) — quindi il banco `dense-batch-attn-chunk` del ktest e'
// passato al confronto a TOLLERANZA DICHIARATA, come previsto dalla riga 3 (d)
// di PHASES del goal engine-ttft, e ha guadagnato un caso a n > 64 senza il
// quale il rescale fra tile non verrebbe mai eseguito.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Workgroup storage di `attnDecodeWgsl` — di ENTRAMBI i rami, decode e batch:
 * `qsh[HD4]` vec4 + `sc[64]` + `red[64]` f32 = 16·ceil(headDim/4) + 512 byte.
 * Dal task T1-kernel-batch-streaming anche il ramo batch ha questa forma, e
 * questa resta LA formula: non ne esiste una seconda da nessun'altra parte.
 *
 * COSTANTE IN ctxMax — ed e' il punto dello split: prima era `4·ctxMax + 256`
 * (gli score dell'intero contesto in shared), quindi il limite del device
 * tagliava ctxMax. Il parametro `ctxMax` RESTA nella firma perche' e' cosi' che
 * lo chiama gpulimits.ts, ma non influenza piu' il risultato: e' un residuo di
 * compatibilita', non un ingresso.
 *
 * Vive QUI, accanto al kernel che la consuma, e non nel modulo dei limiti: il
 * difetto che ha reso necessaria questa funzione e' che i due posti erano
 * separati e uno dei due credeva che il path Qwen non dipendesse dal contesto
 * (goal engine-kernel-decode, docket item 2). Una formula sola, dove sta il
 * consumatore.
 */
/**
 * Workgroup storage dei kernel FUSI del path Qwen2.5-0.5B, accanto ai kernel che
 * lo consumano — stessa regola di `attnDecodeWorkgroupStorageBytes`: una formula
 * sola, dove sta il consumatore.
 *
 * SERVIVANO PERCHE' `gpulimits.ts` PORTAVA UN NUMERO SCRITTO A MANO, e quel
 * numero era giusto per COINCIDENZA (it.24). Dichiarava un solo consumatore
 * sopra i 16.384 B garantiti da WebGPU; misurandoli sono QUATTRO, e i 30.848
 * erano semplicemente il massimo dei quattro. Bastava portare `pairSilu` alla
 * forma multi-riga perche' la costante scendesse mentre `rmsQkv` continuava a
 * chiederne 30.720: il motore avrebbe SOTTO-chiesto il limite, e
 * `createComputePipeline` sarebbe fallito in validazione su OGNI device — 4090
 * compreso. Un massimo calcolato non puo' sbagliarsi cosi'.
 *
 * `x` in memoria di gruppo e' il termine che domina: 4·K·mMax.
 *
 * E' il MAGGIORANTE DELLA FAMIGLIA, non la misura esatta di ciascuno: sul
 * WGSL generato a K=896 e mMax=8 `rmsPairGemmSiluChunkFast` usa 30.848 B
 * mentre `rmsGemmQkvChunkFast` e `gemmResidChunkFast` ne usano 30.720 (128 B
 * in meno, una riduzione in meno). Un limite si chiede al massimo, quindi
 * maggiorare e' corretto e sotto-stimare no — ma chi legge deve sapere che
 * questi 128 B sono margine, non consumo.
 */
export const qwenFusedChunkWorkgroupStorageBytes = (o: { K: number; mMax: number }): number =>
  4 * o.K * o.mMax + 256 * o.mMax + 16 * o.mMax;

/** `gemvResidualFast`: x in memoria di gruppo a M=1, piu' la riduzione. */
export const qwenGemvResidualWorkgroupStorageBytes = (K: number): number => 4 * K + 256;

export const attnDecodeWorkgroupStorageBytes = (_ctxMax: number, headDim = 256): number =>
  16 * Math.ceil(headDim / 4) + 512;

/**
 * Vincolo di codegen del ramo vec4: headDim multiplo di 4 (le load sono vec4) e
 * al piu' 64 vec4 per head (un thread del workgroup da 64 possiede UNA
 * componente vec4 di headDim). Nessuna shape del motore chiede di piu': la
 * famiglia Qwen ha headDim 256 (HD4 64) e MLA/GLM ha il suo kernel.
 */
function attnHd4(headDim: number): number {
  if (headDim % 4 !== 0) {
    throw new Error(`attnDecodeWgsl: headDim ${headDim} non e' multiplo di 4 (load vec4)`);
  }
  const hd4 = headDim / 4;
  if (hd4 > 64) {
    throw new Error(
      `attnDecodeWgsl: headDim ${headDim} => ${hd4} vec4 > 64 thread per workgroup ` +
      "(nessuna shape del motore lo richiede; MLA/GLM ha il suo kernel)");
  }
  return hd4;
}

export function attnDecodeWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number;
  /**
   * batch (fase 4, it.26): `wid.y` = riga del chunk, `nPast` PER RIGA da
   * `rowPast`, q e out a offset di riga. E' l'attenzione a chunk del prefill,
   * e la CAUSALITA' viene gratis: la riga m guarda le posizioni 0..rowPast[m],
   * quindi vede se stessa e tutto cio' che la precede — comprese le righe
   * precedenti dello STESSO chunk, che `kvAppend` ha gia' scritto in cache — e
   * non vede quelle dopo. Non serve una maschera: serve che l'append preceda
   * l'attenzione, e nell'encoder e' cosi'.
   *
   * `sc` e `red` sono di workgroup, e i workgroup sono (head, riga): ogni riga
   * ha i suoi.
   *
   * FIRMA INVARIATA, forma nuova (T1-kernel-batch-streaming): questo ramo non
   * instrada piu' a `attnDecodeLegacyWgsl`. Cio' che l'assemblatore vede non
   * cambia — binding [q, kCache, vCache, out, rowPast], `@workgroup_size(64)`,
   * griglia [nHead, M, 1] con wid.x = head e wid.y = riga — quindi
   * q35gpumodel.ts non si tocca.
   */
  batch?: boolean;
}): string {
  return opts.batch === true ? attnDecodeBatchStreamWgsl(opts) : attnDecodeStreamWgsl(opts);
}

/**
 * Il ramo batch di IERI: `scores[ctxMax]` in workgroup memory, letture scalari.
 * NON e' piu' cio' che `attnDecodeWgsl({ batch: true })` emette — e' il
 * FALLBACK dichiarato e il termine di paragone (AC2 del task
 * T1-kernel-batch-streaming): se un device rifiutasse la forma nuova, o se si
 * volesse rimisurare quanto valeva il debito di workgroup storage, il testo di
 * ieri si genera da qui, byte per byte. Per questo non si cancella.
 */
export function attnDecodeLegacyBatchWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number;
}): string {
  return attnDecodeLegacyWgsl(opts, true);
}

/**
 * Il ramo NON-batch di IERI: `scores[ctxMax]` in workgroup memory, un workgroup
 * per head, letture scalari. Non lo usa piu' il motore — e' il RIFERIMENTO
 * per-riga del ktest `dense-batch-attn-chunk`, che confronta il prefill a chunk
 * con M esecuzioni del kernel per-riga. Esce dallo STESSO template di
 * `attnDecodeLegacyBatchWgsl`, quindi fra quei due l'identita' strutturale e'
 * garantita dal codice e non da una coincidenza fra due copie.
 *
 * Col ramo batch DI OGGI quella parentela non c'e' piu': il confronto del ktest
 * e' a tolleranza dichiarata, non bit a bit. Ed e' proprio per questo che il
 * riferimento resta indipendente — se seguisse la riscrittura non verificherebbe
 * piu' niente.
 */
export function attnDecodeRefWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number;
}): string {
  return attnDecodeLegacyWgsl(opts, false);
}

/**
 * Il template legacy (prefill a chunk + riferimento del ktest). `batch` sceglie
 * il binding 4 e l'offset di riga: nient'altro cambia. Il testo emesso dai due
 * rami e' congelato dal golden.
 */
function attnDecodeLegacyWgsl(
  opts: { nHead: number; nKvHead: number; headDim: number; ctxMax: number },
  batch: boolean,
): string {
  const { nHead, nKvHead, headDim, ctxMax } = opts;
  const groups = nHead / nKvHead;
  const kvDim = nKvHead * headDim;
  const pBind = batch
    ? "@group(0) @binding(4) var<storage, read> rowPast: array<u32>;"
    : "@group(0) @binding(4) var<uniform> P: TokParams;";
  const nExpr = batch ? "rowPast[wid.y] + 1u" : "P.nPast + 1u";
  const rowOff = batch ? `wid.y * ${nHead * headDim}u + ` : "";
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kCache: array<f32>;
@group(0) @binding(2) var<storage, read> vCache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
${pBind}
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
  let qOff = ${rowOff}h * HEAD_DIM;
  let n = ${nExpr};
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

/**
 * DECODE: split sul contesto + softmax in streaming (pass 1 di 2).
 *
 * Griglia (nHead, splits, 1), workgroup da 64. `wid.y` e' il chunk: il
 * workgroup (h, c) copre le posizioni [c·chunkLen, min(nPast+1, (c+1)·chunkLen))
 * e le scorre a tile di 64 (un thread una posizione), tenendo {m, s, acc} in
 * registri con l'aggiornamento online della softmax. La workgroup memory e'
 * `qsh` (la q della head, letta una volta e riletta n volte dal ciclo interno)
 * piu' due array da 64: COSTANTE in ctxMax.
 *
 * Non normalizza: scrive i PARZIALI, `partOut[(h·splits+c)·HD4 + i] = acc` e
 * `partMS[(h·splits+c)·2 + {0,1}] = {m, s}`. Li combina `attnDecodeCombineWgsl`.
 *
 * I chunk oltre `nPast+1` non sono un errore e non vanno saltati dal dispatch:
 * escono col loop vuoto e scrivono m = -3.0e38, s = 0, acc = 0, che il combine
 * annulla (exp(-3e38 − gm) = 0, nessun NaN). La griglia resta FISSA in ctxMax.
 *
 * Con HD4 < 64 (shape del ktest, headDim 32) i thread t ≥ HD4 non hanno una
 * componente di headDim: NON scrivono, ma attraversano tutti i
 * `workgroupBarrier()` — le barriere restano in control flow uniforme, si
 * guardano solo gli accessi in memoria. Restano invece pieni partecipanti alla
 * fase score, dove i 64 thread sono 64 POSIZIONI, non 64 componenti.
 */
function attnDecodeStreamWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number;
}): string {
  const { nHead, nKvHead, headDim, ctxMax } = opts;
  const hd4 = attnHd4(headDim);
  const { splits, chunkLen } = q35AttnSplitPlan(ctxMax);
  return `${TOK_PARAMS_WGSL}
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kCache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vCache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> partOut: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> partMS: array<f32>;
@group(0) @binding(5) var<uniform> P: TokParams;
// dispatch (${nHead}, ${splits}, 1): X = head, Y = chunk del contesto
const HD4 = ${hd4}u;
const KV4 = ${(nKvHead * headDim) / 4}u;
const GROUPS = ${nHead / nKvHead}u;
const SCALE = ${1 / Math.sqrt(headDim)};
const NEG = -3.0e38;
var<workgroup> qsh: array<vec4<f32>, ${hd4}>;
var<workgroup> sc: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let kvBase = (h / GROUPS) * HD4;
  let chunk = wid.y;
  let pStart = chunk * ${chunkLen}u;
  let pEnd = min(P.nPast + 1u, pStart + ${chunkLen}u);
  // q della head in memoria di gruppo: il ciclo sulle posizioni la rilegge
  if (t < HD4) { qsh[t] = q[h * HD4 + t]; }
  workgroupBarrier();
  var m = NEG;
  var s = 0.0;
  var acc = vec4<f32>(0.0);
  var tile = pStart;
  loop {
    if (tile >= pEnd) { break; }
    let p = tile + t;
    var d = NEG; // ri-azzerato a OGNI giro: una var dichiarata nel loop non lo e'
    if (p < pEnd) {
      let kOff = p * KV4 + kvBase;
      d = 0.0;
      for (var i = 0u; i < HD4; i = i + 1u) { d = d + dot(qsh[i], kCache[kOff + i]); }
      d = d * SCALE;
    }
    sc[t] = d;
    red[t] = d;
    workgroupBarrier();
    // max del tile
    var stride = 32u;
    while (stride > 0u) {
      if (t < stride) { red[t] = max(red[t], red[t + stride]); }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    let tm = red[0];
    workgroupBarrier();
    // rescale online: il massimo corrente sale, i vecchi contributi si sgonfiano
    let nm = max(m, tm);
    let rs = exp(m - nm);
    m = nm;
    let e = exp(sc[t] - nm);
    sc[t] = e;
    red[t] = e;
    workgroupBarrier();
    stride = 32u;
    while (stride > 0u) {
      if (t < stride) { red[t] = red[t] + red[t + stride]; }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    s = s * rs + red[0];
    workgroupBarrier();
    acc = acc * rs;
    let cnt = min(64u, pEnd - tile);
    // ogni thread una componente vec4 di headDim: letture di V coalescenti
    if (t < HD4) {
      for (var j = 0u; j < cnt; j = j + 1u) {
        acc = acc + sc[j] * vCache[(tile + j) * KV4 + kvBase + t];
      }
    }
    workgroupBarrier();
    tile = tile + 64u;
  }
  let row = h * ${splits}u + chunk;
  if (t < HD4) { partOut[row * HD4 + t] = acc; }
  if (t == 0u) {
    partMS[row * 2u] = m;
    partMS[row * 2u + 1u] = s;
  }
}`;
}

/**
 * PREFILL A CHUNK (ramo `batch` di `attnDecodeWgsl`): softmax in STREAMING a
 * tile di 64 posizioni, letture vec4, UN workgroup per (head, riga).
 *
 * Griglia [nHead, M, 1]: `wid.x` = head, `wid.y` = RIGA del chunk. Il modello
 * di riga 1 (src/microbench/ttAttn.ts) usa `wid.z` per la riga; qui si tiene
 * `wid.y` perche' e' cio' che l'assemblatore di q35gpumodel gia' dispatcha, e
 * questo task non tocca il cablaggio.
 *
 * CAUSALITA': dalla sola `rowPast`, come nel legacy. La riga m guarda le
 * posizioni [0, rowPast[m]] — vede se stessa e tutto cio' che la precede,
 * comprese le righe precedenti dello STESSO chunk che `kvAppend` ha gia'
 * scritto in cache. Nessuna maschera: serve solo che l'append preceda
 * l'attenzione, e nell'encoder e' cosi'.
 *
 * PERCHE' UN WORKGROUP PER (head, riga) E NON UNO PER GRUPPO GQA. Misurato
 * (docket item 21, banco `attn-prefill-chunk` di src/microbench/ttRunner.ts,
 * bracci `stream` contro `gqa-stream`): fondere le 4 teste del
 * gruppo per leggere la KV una volta sola fa 2,0879 ms contro 1,8207 a ctx
 * 6333. Taglia il traffico 4x ma porta i workgroup da 256 a 64 su 76 SM, e a
 * quel punto la GPU e' mezza vuota. E' un vincolo NEGATIVO: non si fondono le
 * teste, non si legge la KV una volta per gruppo.
 *
 * Workgroup storage: `qsh[HD4]` vec4 + `sc[64]` + `red[64]` — la stessa
 * `attnDecodeWorkgroupStorageBytes` del ramo decode, COSTANTE in ctxMax (era
 * 4·ctxMax + 256).
 *
 * Con HD4 < 64 (shape del ktest, headDim 32) valgono le stesse regole del ramo
 * decode: i thread t ≥ HD4 non possiedono una componente vec4, quindi non
 * leggono ne' scrivono `qsh`/`out`/`vCache`, ma attraversano TUTTE le barriere
 * — che restano in control flow uniforme. Nella fase score restano pieni
 * partecipanti: li' i 64 thread sono 64 POSIZIONI, non 64 componenti.
 */
function attnDecodeBatchStreamWgsl(opts: {
  nHead: number; nKvHead: number; headDim: number; ctxMax: number;
}): string {
  const { nHead, nKvHead, headDim } = opts;
  const hd4 = attnHd4(headDim);
  return `
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kCache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vCache: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> rowPast: array<u32>;
// dispatch (${nHead}, M, 1): X = head, Y = riga del chunk
const HD4 = ${hd4}u;
const KV4 = ${(nKvHead * headDim) / 4}u;
const ROWQ4 = ${(nHead * headDim) / 4}u;
const GROUPS = ${nHead / nKvHead}u;
const SCALE = ${1 / Math.sqrt(headDim)};
const NEG = -3.0e38;
var<workgroup> qsh: array<vec4<f32>, ${hd4}>;
var<workgroup> sc: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let row = wid.y;
  let kvBase = (h / GROUPS) * HD4;
  let qOff = row * ROWQ4 + h * HD4;
  let n = rowPast[row] + 1u;
  // q della (head, riga) in memoria di gruppo: il ciclo sulle posizioni la rilegge
  if (t < HD4) { qsh[t] = q[qOff + t]; }
  workgroupBarrier();
  var m = NEG;
  var s = 0.0;
  var acc = vec4<f32>(0.0);
  var tile = 0u;
  loop {
    if (tile >= n) { break; }
    let p = tile + t;
    var d = NEG; // ri-azzerato a OGNI giro: una var dichiarata nel loop non lo e'
    if (p < n) {
      let kOff = p * KV4 + kvBase;
      d = 0.0;
      for (var i = 0u; i < HD4; i = i + 1u) { d = d + dot(qsh[i], kCache[kOff + i]); }
      d = d * SCALE;
    }
    sc[t] = d;
    red[t] = d;
    workgroupBarrier();
    // max del tile
    var stride = 32u;
    while (stride > 0u) {
      if (t < stride) { red[t] = max(red[t], red[t + stride]); }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    let tm = red[0];
    workgroupBarrier();
    // rescale online: il massimo corrente sale, i vecchi contributi si sgonfiano
    let nm = max(m, tm);
    let rs = exp(m - nm);
    m = nm;
    let e = exp(sc[t] - nm);
    sc[t] = e;
    red[t] = e;
    workgroupBarrier();
    stride = 32u;
    while (stride > 0u) {
      if (t < stride) { red[t] = red[t] + red[t + stride]; }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    s = s * rs + red[0];
    workgroupBarrier();
    acc = acc * rs;
    let cnt = min(64u, n - tile);
    // ogni thread una componente vec4 di headDim: letture di V coalescenti
    if (t < HD4) {
      for (var j = 0u; j < cnt; j = j + 1u) {
        acc = acc + sc[j] * vCache[(tile + j) * KV4 + kvBase + t];
      }
    }
    workgroupBarrier();
    tile = tile + 64u;
  }
  if (t < HD4) { out[qOff + t] = acc / s; }
}`;
}

/**
 * DECODE: combinazione log-sum-exp dei parziali (pass 2 di 2).
 *
 * Griglia (nHead, 1, 1), workgroup da 64, un thread per componente vec4 di
 * headDim. Nessun uniform: quanti chunk esistono lo dice il piano baked, e i
 * chunk vuoti si annullano da soli (m = -3e38 ⇒ peso 0), quindi il kernel non
 * ha bisogno di sapere `nPast`.
 *
 * Il riferimento JS di questa aritmetica e' `q35AttnLseReduce` in
 * q35attnsplit.ts, che le unit provano equivalente alla softmax monolitica.
 */
export function attnDecodeCombineWgsl(opts: {
  nHead: number; headDim: number; ctxMax: number;
}): string {
  const hd4 = attnHd4(opts.headDim);
  const { splits } = q35AttnSplitPlan(opts.ctxMax);
  return `
@group(0) @binding(0) var<storage, read> partOut: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> partMS: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
// dispatch (${opts.nHead}, 1, 1): un workgroup per head, nessuna shared
const HD4 = ${hd4}u;
const S = ${splits}u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  if (t >= HD4) { return; }
  var gm = -3.0e38;
  for (var c = 0u; c < S; c = c + 1u) { gm = max(gm, partMS[(h * S + c) * 2u]); }
  var num = vec4<f32>(0.0);
  var den = 0.0;
  for (var c = 0u; c < S; c = c + 1u) {
    let w = exp(partMS[(h * S + c) * 2u] - gm);
    num = num + w * partOut[(h * S + c) * HD4 + t];
    den = den + w * partMS[(h * S + c) * 2u + 1u];
  }
  out[h * HD4 + t] = num / den;
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
export function axpyWgsl(D: number, sigmoidGate = false, batch = false): string {
  // batch (fase 4, it.34): gid.y = riga; `out`, `x` e lo SCALARE sono per riga
  // (lo scalare del shexp e' un GEMV per riga, quindi anche lui e' [M]).
  const e = batch ? "gid.y * D + i" : "i";
  const sIdx = batch ? "gid.y" : "0";
  const s = sigmoidGate ? `1.0 / (1.0 + exp(-sBuf[${sIdx}]))` : `sBuf[${sIdx}]`;
  return `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> sBuf: array<f32>;
const D = ${D}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= D) { return; }
  out[${e}] = out[${e}] + (${s}) * x[${e}];
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
 * Modo GATHER dei GEMV K-quant: `wid.z` indicizza `gather`, che porta
 * `m | kslot<<16`. `nUsed` e' lo stride degli slot — lo STESSO numero che
 * `moeCombineWgsl` riceve, altrimenti i due kernel indirizzano slot diversi
 * senza che niente lanci (v. `tests/engine-moegather-nused.test.ts`).
 */
export interface KGatherOpts { nUsed: number }

/**
 * KFAN (goal engine-velocita-decode, riga 2c): `wid.z` = **il k del top-K**,
 * non la riga. Un dispatch solo copre tutti gli expert selezionati dal token
 * invece di uno per expert.
 *
 * PERCHE' ESISTE, con i numeri che l'hanno chiesto. Sul token pulito del 35B
 * (`q35-optimistic-35b-cleantoken-2026-08-15.json`) il decode fa **1.320
 * dispatch** — (8 expert x 4 op) + 1 add, per 40 layer — in **40,753 ms**, cioe'
 * ~31 us l'uno e **13,9 GB/s effettivi** sui 566 MB di pesi expert per token.
 * A 576 GB/s quegli stessi byte costerebbero **0,98 ms**: il decode e' **41x
 * sopra il pavimento di banda**, e non e' ne' compute- ne' bandwidth-bound —
 * e' dispatch/occupancy-bound (32-64 workgroup su 128 SM). Col kfan i dispatch
 * scendono a **200** e l'occupazione per dispatch va x8.
 *
 * NON e' `batch`, ed e' per questo che e' un modo A SE' invece di un
 * allentamento del divieto `batch && arena`: `batch` mette le RIGHE su `wid.z`
 * (x e y diventano matrici [M,K] e [M,N]) e in regime d'arena resta vietato,
 * perche' la `Sel` e' per (riga, layer, k) e una sola non basterebbe. Il kfan
 * tiene UNA riga e mette i k su `wid.z`, che e' l'asse per cui le entry di
 * `Sel` sono gia' contigue (`selIdx = (row*nMoeLayer + m)*topK + k`,
 * `q35gpumodel.ts:1377`) — quindi `selBuf[moeIdx.selIdx + wid.z]`.
 *
 * SCRIVE NELLO SLOT, non accumula, e non e' un'alternativa: 8 dispatch che
 * facessero `y[r] += w*partial` nello stesso dispatch **corrono**. Col
 * contratto a slot ogni k scrive il suo, e `moeCombineWgsl` somma `w*y` in
 * ordine k — che a M=1 con `nUsed = topK` e' **esattamente** la catena di somme
 * del decode di oggi (`encodeExperts`, `for k2` ascendente): bit-identico per
 * costruzione, non "atteso identico".
 *
 * IL MISS. Il ramo d'arena normale esce con `if (!ok) { return; }`, e col
 * peso-in-accumulo e' giusto: un expert mancante contribuisce zero. Con gli
 * slot **no**: uscire lascerebbe lo slot con il valore del token precedente e
 * la combine lo sommerebbe. Nel kfan non si esce — si scrive **zero**
 * (`select(0.0, partial[0], ok)`), che e' lo stesso contributo nullo ma
 * DICHIARATO. E' la stessa filosofia del degrado definito di `arenaSlotWgsl`.
 */
export interface KFanOpts {
  nUsed: number;
  /**
   * L'ingresso `x` e' PER-K invece che condiviso.
   *
   * Gate e up leggono tutti lo STESSO x — l'hidden del token, una riga di K
   * valori: `xPerK` falso. Il DOWN no: il suo ingresso e' l'`h` che gate/up/silu
   * hanno prodotto **per ciascun k**, e vive nello slot `k` di un array
   * [topK, K]. Senza questo flag il down leggerebbe l'h del k = 0 per tutti e
   * otto gli expert.
   *
   * MISURATO il 2026-08-15 (it.9), perche' non e' un difetto teorico: col down
   * senza offset il kfan girava 1,295x piu' veloce e produceva argmax
   * DIVERSI — 14/39 uguali, prima divergenza al token 2. Il gate dell'A/B
   * l'ha preso; un gate a sola velocita' avrebbe promosso un motore rotto.
   */
  xPerK?: boolean;
}

/** le combinazioni di modi che non esistono, con la ragione nel messaggio */
function assertGatherCombo(
  fam: string,
  o: { arena: boolean; accum?: boolean; batch?: boolean; gather?: KGatherOpts; kfan?: KFanOpts },
): void {
  if (o.gather && o.kfan) throw new Error(`${fam}: gather e kfan non si combinano — userebbero entrambi wid.z (righe l'uno, k l'altro)`);
  if (o.kfan) {
    if (!o.arena) throw new Error(`${fam}: kfan esige il regime d'arena — l'expert di ogni k lo risolve Sel, e senza arena non c'e' Sel`);
    if (o.batch === true) throw new Error(`${fam}: kfan e batch non si combinano — userebbero entrambi wid.z`);
    if (o.accum === true) throw new Error(`${fam}: kfan e accum non si combinano — 8 k nello stesso dispatch che accumulano su y[r] CORRONO; il contratto a slot esiste per questo`);
    const n = o.kfan.nUsed;
    if (!Number.isInteger(n) || n < 1) throw new Error(`${fam}: kfan.nUsed non valido (${n})`);
  }
  if (!o.gather) return;
  if (o.arena) {
    throw new Error(`${fam}: gather e arena non si combinano — nel gather l'expert lo fissa il dispatch (bindings a sotto-range), l'arena lo risolve da Sel`);
  }
  if (o.batch === true) throw new Error(`${fam}: gather e batch non si combinano — userebbero entrambi wid.z`);
  if (o.accum === true) throw new Error(`${fam}: gather e accum non si combinano — il contratto a slot vuole la scrittura NON pesata, e' moeCombine a sommare w*y in ordine k`);
  const n = o.gather.nUsed;
  if (!Number.isInteger(n) || n < 1) throw new Error(`${fam}: gather.nUsed non valido (${n})`);
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
 *
 * MODO GATHER (goal engine-velocita-decode, riga 2). Terza variante degli
 * STESSI assi di `batch`: `wid.z` non e' piu' la riga, e' l'indice nel buffer
 * `gather` che porta `m | kslot<<16`. Il kernel legge la riga `m` di una
 * matrice [M,K] e scrive nello slot `(m*nUsed + kslot)` — il contratto a slot
 * di `moeprefillplan.ts:29-36`: il down SCRIVE non pesato, e' `moeCombine` a
 * sommare `w·y` in ordine k. Serve al path expert raggruppato per expert
 * (un dispatch per expert sulle sole righe che lo selezionano) invece che per
 * riga.
 *
 * E' PLAIN, non d'arena, ed e' una scelta e non una mancanza: nel gather
 * l'expert e' fissato DAL DISPATCH, quindi i pesi arrivano da un binding a
 * sotto-range (`slotTensorRanges`, `residency.ts:189`) e non c'e' nessuna `Sel`
 * da risolvere. E' la stessa forma dei gemelli q4_0 del GLM
 * (`pairGemvSiluGatherWgsl`), che il commento di famiglia dichiara PLAIN.
 *
 * L'ARITMETICA NON CAMBIA DI UN CARATTERE: `gather` tocca solo da dove viene
 * `x` e dove va `y`. `tests/engine-kquant-gather.test.ts` lo verifica
 * confrontando il corpo generato con quello del kernel senza gather.
 */
export function gemvQ4KWgsl(opts: {
  K: number; N: number; arena?: KArenaOpts; accum?: boolean; batch?: boolean;
  gather?: KGatherOpts; kfan?: KFanOpts;
}): string {
  const { K, N, arena, accum, batch, gather, kfan } = opts;
  if (batch === true && arena) throw new Error("q4K: batch e arena non si combinano (l'arena e' il path expert, il batch il prefill)");
  if (accum === true && !arena) throw new Error("q4K: accum esige il regime d'arena (il peso viene da Sel)");
  assertGatherCombo("q4K", { arena: arena !== undefined, accum, batch, gather, kfan });
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
    : gather
      // ORDINE DEI BINDING CONGELATO nel modo gather: [0 blocks, 1 x,
      // 2 gather, 3 y]. `y` si sposta da 2 a 3 e `gather` ne prende il posto:
      // chi costruisce il bind group segue QUESTO elenco, non quello plain.
      ? `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gather: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
fn blkw(i: u32) -> u32 { return blocks[i]; }`
      : `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn blkw(i: u32) -> u32 { return blocks[i]; }`;
  // preambolo di main: nel regime d'arena lo slot di `Sel` diventa (binding, base)
  const pre = arena
    ? `${arenaSlotWgsl(kfan ? "moeIdx.selIdx + wid.z" : undefined)}
  gBi = bi;
  gBase = base + ${arena.tensorWords}u;${kfan ? "" : `
  if (!ok) { return; }`}`
    : "";
  // batch (fase 4): wid.z = riga; x e y sono matrici [M, K] e [M, N].
  // gather (engine-velocita-decode riga 2): wid.z indicizza `gather`, che porta
  // `m | kslot<<16`; x resta [M,K], y e' l'array degli slot e la scrittura NON
  // e' pesata (contratto a slot: e' moeCombine a sommare w*y in ordine k).
  // Senza nessuno dei due il testo emesso e' identico byte per byte.
  const xPerK = kfan?.xPerK === true;
  const XR = batch || gather || xPerK ? "xR + " : "";
  const YR = gather
    ? `(mRow * ${gather.nUsed}u + kslot) * ${N}u + `
    : batch ? `wid.z * ${N}u + ` : "";
  const xRowPre = gather
    ? `\n  let gk = gather[wid.z];\n  let mRow = gk & 0xffffu;\n  let kslot = gk >> 16u;\n  let xR = mRow * ${K}u;`
    : batch || xPerK ? `\n  let xR = wid.z * ${K}u;` : "";
  const tailAcc = kfan
    // slot `wid.z` della riga 0 (il decode e' M=1): stesso layout che
    // `moeCombineWgsl` legge, `(m*nUsed + k)*N + r` con m = 0. Sul miss si
    // scrive ZERO invece di uscire: uscire lascerebbe lo slot col valore del
    // token precedente e la combine lo sommerebbe.
    ? `y[wid.z * ${N}u + r] = select(0.0, partial[0], ok);`
    : accum === true
      ? `y[${YR}r] = y[${YR}r] + sel.w * partial[0];`
      : `y[${YR}r] = partial[0];`;
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
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;${xRowPre}
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
      let x1 = x[${XR}xBase + j * 64u + l];
      let x2 = x[${XR}xBase + j * 64u + 32u + l];
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
  if (t == 0u) { ${tailAcc} }
}`;
}

export function gemvQ5KWgsl(opts: { K: number; N: number; batch?: boolean }): string {
  const { K, N, batch } = opts;
  if (K % 256 !== 0) throw new Error("gemvQ5K: K non multiplo di 256");
  const sbPerRow = K / 256;
  // batch (fase 4): wid.z = riga; x e y sono matrici [M, K] e [M, N].
  // Senza, il testo emesso e' identico byte per byte.
  const XR = batch ? "xR + " : "";
  const YR = batch ? `wid.z * ${N}u + ` : "";
  const xRowPre = batch ? `\n  let xR = wid.z * ${K}u;` : "";
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
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;${xRowPre}
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
        let x1 = x[${XR}xBase + j * 64u + l];
        let x2 = x[${XR}xBase + j * 64u + 32u + l];
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
  if (t == 0u) { y[${YR}r] = partial[0]; }
}`;
}

// Q6_K: 53 word/superblocco (210 B + 2 pad) = [ql 128 B][qh 64 B][scales int8
// 16 B][d f16]. w = d·sc_int8·(q6 − 32), q6 = nibble | 2 bit alti.
export function gemvQ6KWgsl(opts: {
  K: number; N: number; arena?: KArenaOpts; accum?: boolean; batch?: boolean;
  gather?: KGatherOpts; kfan?: KFanOpts;
}): string {
  const { K, N, arena, accum, batch, gather, kfan } = opts;
  if (batch === true && arena) throw new Error("q6K: batch e arena non si combinano (l'arena e' il path expert, il batch il prefill)");
  if (accum === true && !arena) throw new Error("q6K: accum esige il regime d'arena (il peso viene da Sel)");
  assertGatherCombo("q6K", { arena: arena !== undefined, accum, batch, gather, kfan });
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
    : gather
      // ORDINE DEI BINDING CONGELATO nel modo gather: [0 blocks, 1 x,
      // 2 gather, 3 y]. `y` si sposta da 2 a 3 e `gather` ne prende il posto:
      // chi costruisce il bind group segue QUESTO elenco, non quello plain.
      ? `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gather: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
fn blkw(i: u32) -> u32 { return blocks[i]; }`
      : `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn blkw(i: u32) -> u32 { return blocks[i]; }`;
  // preambolo di main: nel regime d'arena lo slot di `Sel` diventa (binding, base)
  const pre = arena
    ? `${arenaSlotWgsl(kfan ? "moeIdx.selIdx + wid.z" : undefined)}
  gBi = bi;
  gBase = base + ${arena.tensorWords}u;${kfan ? "" : `
  if (!ok) { return; }`}`
    : "";
  // batch (fase 4): wid.z = riga; x e y sono matrici [M, K] e [M, N].
  // gather (engine-velocita-decode riga 2): wid.z indicizza `gather`, che porta
  // `m | kslot<<16`; x resta [M,K], y e' l'array degli slot e la scrittura NON
  // e' pesata (contratto a slot: e' moeCombine a sommare w*y in ordine k).
  // Senza nessuno dei due il testo emesso e' identico byte per byte.
  const xPerK = kfan?.xPerK === true;
  const XR = batch || gather || xPerK ? "xR + " : "";
  const YR = gather
    ? `(mRow * ${gather.nUsed}u + kslot) * ${N}u + `
    : batch ? `wid.z * ${N}u + ` : "";
  const xRowPre = gather
    ? `\n  let gk = gather[wid.z];\n  let mRow = gk & 0xffffu;\n  let kslot = gk >> 16u;\n  let xR = mRow * ${K}u;`
    : batch || xPerK ? `\n  let xR = wid.z * ${K}u;` : "";
  const tailAcc = kfan
    // slot `wid.z` della riga 0 (il decode e' M=1): stesso layout che
    // `moeCombineWgsl` legge, `(m*nUsed + k)*N + r` con m = 0. Sul miss si
    // scrive ZERO invece di uscire: uscire lascerebbe lo slot col valore del
    // token precedente e la combine lo sommerebbe.
    ? `y[wid.z * ${N}u + r] = select(0.0, partial[0], ok);`
    : accum === true
      ? `y[${YR}r] = y[${YR}r] + sel.w * partial[0];`
      : `y[${YR}r] = partial[0];`;
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
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;${xRowPre}
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
        acc = acc + d * (sint8(wb, scO + is) * q1 * x[${XR}e]
                       + sint8(wb, scO + is + 2u) * q2 * x[${XR}e + 32u]
                       + sint8(wb, scO + is + 4u) * q3 * x[${XR}e + 64u]
                       + sint8(wb, scO + is + 6u) * q4 * x[${XR}e + 96u]);
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
  if (t == 0u) { ${tailAcc} }
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
const arenaSlotWgsl = (selIdx = "moeIdx.selIdx"): string => `
  let sel = selBuf[${selIdx}];
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
    ? `${arenaSlotWgsl()}
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
    ? `${arenaSlotWgsl()}
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
//
// `nUsed` = il top-K della famiglia, ed e' lo STRIDE degli slot: la riga m
// occupa `nUsed` slot consecutivi e `kslot` sceglie quale. Era scritto `4u` a
// mano (GLM e' top-4); il 35B e' top-8. I TRE kernel della famiglia devono
// essere generati con lo STESSO `nUsed` — un disaccordo non lancia, indirizza
// slot sbagliati e produce numeri plausibili. Il test
// `engine-moegather-nused.test.ts` lo verifica.
export function pairGemvSiluGatherWgsl(opts: { K: number; N: number; nUsed?: number }): string {
  const { K, N } = opts;
  const nUsed = opts.nUsed ?? 4;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("pairGemvSiluGather: N non multiplo di 4");
  if (K % 32 !== 0) throw new Error("pairGemvSiluGather: K non multiplo di 32");
  if (!Number.isInteger(nUsed) || nUsed < 1) throw new Error(`pairGemvSiluGather: nUsed non valido (${nUsed})`);
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
    hSlots[(m * ${nUsed}u + kslot) * ${N}u + r] = (g / (1.0 + exp(-g))) * redU[sub * 16u];
  }
}`;
}

// down dell'expert per le righe raccolte: SCRIVE y[m][k] NON pesato — il peso
// lo applica moeCombine, in ordine k (contratto di identita' del piano).
export function gemvDownSlotsWgsl(opts: { kind: "q4_0" | "q4_1"; K: number; N: number; nUsed?: number }): string {
  const { kind, K, N } = opts;
  const nUsed = opts.nUsed ?? 4;
  const blocksPerRow = K / 32;
  if (N % 4 !== 0) throw new Error("gemvDownSlots: N non multiplo di 4");
  if (K % 32 !== 0) throw new Error("gemvDownSlots: K non multiplo di 32");
  if (!Number.isInteger(nUsed) || nUsed < 1) throw new Error(`gemvDownSlots: nUsed non valido (${nUsed})`);
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
  let hB = (m * ${nUsed}u + kslot) * K;
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
  if (lane == 0u && r < ${N}u) { ySlots[(m * ${nUsed}u + kslot) * ${N}u + r] = red[sub * 16u]; }
}`;
}

// combine per riga: xM[m] += shexp[m] + Σ_k w[m][k]·y[m][k], somme in ordine k
// — la stessa catena del decode (v. commento di famiglia). grid: (ceil(D/64), M).
export function moeCombineWgsl(opts: { D: number; nUsed?: number; weightsFromSel?: boolean }): string {
  const { D, weightsFromSel } = opts;
  const nUsed = opts.nUsed ?? 4;
  // VARIANTE `weightsFromSel` (goal engine-velocita-decode, riga 2c): il peso
  // arriva dalla `Sel` che il motore ha GIA' scritto in VRAM per il layer,
  // invece che da un `wBuf` a parte. Serve al kfan del decode: senza, ogni
  // layer pagherebbe una `writeBuffer` in piu' nel ciclo caldo — 40 per token
  // — per ricopiare pesi che stanno gia' sul device.
  //
  // SOLO M = 1 (il decode). L'indirizzo e' `moeIdx.selIdx + k`, cioe' i topK
  // contigui di UN (riga, layer) (`selIdx = (row*nMoeLayer + m)*topK + k`,
  // `q35gpumodel.ts:1377`). Per M > 1 servirebbe anche `nMoeLayer` per saltare
  // di riga in riga, e il kernel non ce l'ha: chi dispaccia questa variante usa
  // `y = 1`. Il prefill a chunk continua a usare il `wBuf` esplicito.
  const wTerm = weightsFromSel ? "selBuf[moeIdx.selIdx + k].w" : "wBuf[m * N_USED + k]";
  const wBind = weightsFromSel
    ? `${SEL_STRUCT_WGSL}
${MOE_IDX_STRUCT_WGSL}
@group(0) @binding(3) var<storage, read> selBuf: array<Sel>;
@group(0) @binding(4) var<uniform> moeIdx: MoeIdx;`
    : "@group(0) @binding(3) var<storage, read> wBuf: array<f32>;";
  return `
@group(0) @binding(0) var<storage, read_write> xM: array<f32>;
@group(0) @binding(1) var<storage, read> sM: array<f32>;
@group(0) @binding(2) var<storage, read> ySlots: array<f32>;
${wBind}
const D = ${D}u;
const N_USED = ${nUsed}u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = wid.x * 64u + lid.x;
  if (i >= D) { return; }
  let m = wid.y;
  var t = sM[m * D + i];
  for (var k = 0u; k < N_USED; k = k + 1u) {
    t = t + ${wTerm} * ySlots[(m * N_USED + k) * D + i];
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

// ---------------------------------------------------------------------------
// PREFILL GEMM q4_0 SPLIT-K — port dal banco (src/microbench/ttGemm.ts).
//
// Provenienza: riga 1 di engine-ttft, pre-registrazione
// docs/deep-dive/ttft-riga1-prereg-2026-08-13.md. Le due forme che arrivano qui
// sono quelle MISURATE: `splitk-idot` (via intera q4_0 x q8_0 con
// `dot4I8Packed`, 1,745x su `splitk` f32 in it.6) e `splitk` f32, portata come
// FALLBACK DICHIARATO per i device che non espongono la language feature
// `packed_4x8_integer_dot_product`.
//
// REGOLA DEL PORT: il testo WGSL e' quello del banco, riga per riga. La misura
// e' una proprieta' del TESTO, non dell'intenzione: ogni riga riscritta e' un
// numero che non parla piu' del codice in produzione. Le divergenze ammesse
// stanno in `PREFILL_GEMM_PORT_DIFFS` con la loro ragione, e
// tests/engine-prefillgemm.test.ts fallisce su qualunque riga divergente non
// dichiarata (in entrambe le direzioni: niente chiavi morte).
//
// ORDINE DEI BINDING (congelato — lo consumano il piano e il wiring):
//   idot    [0 qs4, 1 scales, 2 xq(u32), 3 part(rw), 4 xsc(f32)]
//   f32     [0 qs4, 1 scales, 2 x4,      3 part(rw)]
//   quantX  [0 x4,  1 xq(rw), 2 xsc(rw)]
//   combine [0 part, 1 y(rw)]
// Layout: y[m*N + r]; part[(s*M + m)*N + r] — row-major per riga di chunk.
// ---------------------------------------------------------------------------

/**
 * I FORMATI CHE LA VIA VELOCE DI PREFILL SA LEGGERE, in una sede sola.
 *
 * Non e' un'unione scritta a mano dentro `PrefillGemmOpts`: e' un array
 * ESPORTATO, perche' il predicato di ammissibilita' (`prefillGemmCheck`) lo
 * legge da qui e il messaggio di rifiuto lo NOMINA.
 *
 * q4_0 = riga 1 di engine-ttft (1,745x della via intera sulla f32).
 * q5_K = riga 2 di engine-kquant (28,10x sulla legacy a M=16 su `ssm_out`).
 * q4_1 = riga 3 di engine-kquant (`blk.0-3.ffn_down` del 4B, 71% dei byte del
 *        segmento `gemm:ffn-down`).
 * q4_K = riga 4 di engine-kquant (35B: 117 tensori expert = 17,67 GB). IL
 *        KERNEL C'E', IL PIANO NON LO INSTRADA — v. `wired` piu' sotto.
 * q6_K = riga 4 (35B: down degli expert di 3 layer). Idem: non instradato.
 * q8_0 = riga 4 (35B: 100 tensori attn = 1,09 GB). Idem: non instradato.
 *
 * L'ORDINE E' PARTE DELL'INTERFACCIA, e i kind nuovi stanno IN CODA: chi
 * consuma questo elenco per costruire un piano o una tabella lo legge
 * posizionalmente, e un kind infilato in mezzo sposterebbe tutto in silenzio.
 *
 * STARE IN QUESTO ELENCO NON VUOL DIRE ESSERE INSTRADATI. Da riga 4 l'elenco
 * dice «di questo formato esiste un kernel multi-riga misurato», e chi decide
 * se un sito ci passa davvero e' il flag `wired` di `PREFILL_GEMM_SPEC`, letto
 * dal piano (`prefillgemmplan.ts`). Le due domande erano la stessa finche' ogni
 * kernel portato veniva anche cablato; da qui in poi non lo sono, e tenerle
 * separate e' cio' che permette di verificare una forma sul device senza
 * cambiare di una riga cio' che il 4B esegue.
 *
 * AGGIUNGERE UN KIND QUI NON BASTA, ED E' VOLUTO: ogni numero che dipende dal
 * formato vive in `PREFILL_GEMM_SPEC`, che e' un `Record<PrefillGemmKind, …>`.
 * Chi allunga questo array senza scrivere la riga corrispondente laggiu' non
 * ottiene una geometria sbagliata in silenzio: ottiene un errore di
 * compilazione. E' la lezione del cast `as "q4_0"` che stava nel piano —
 * un formato che passa per un altro non fa rumore, fa numeri.
 */
export const PREFILL_GEMM_KINDS = ["q4_0", "q5_K", "q4_1", "q4_K", "q6_K", "q8_0"] as const;
export type PrefillGemmKind = typeof PREFILL_GEMM_KINDS[number];

export interface PrefillGemmOpts {
  kind: PrefillGemmKind;
  K: number;
  N: number;
  M: number;
  splits: number;
}

/**
 * Divergenze TESTUALI fra il kernel di produzione e quello di banco: chiave =
 * la riga COME STA (rientro compreso) che diverge, valore = perche'. Vuoto
 * significa una cosa sola e forte — il port e' byte-per-byte il testo che la
 * riga 1 ha misurato — e il test lo verifica nelle due direzioni, quindi non
 * puo' restare vuoto per distrazione.
 *
 * NON e' il guardiano dell'interfaccia: una riga dichiarata qui passa il gate
 * qualunque cosa dica la ragione. Ordine dei binding e layout dei buffer sono
 * pinnati a parte, sul testo generato ([g] in engine-prefillgemm.test.ts).
 */
export const PREFILL_GEMM_PORT_DIFFS: Record<string, string> = {};

/** Il numero di fette MISURATO in riga 1. Nessun altro valore e' stato misurato. */
export const PREFILL_SPLITS_MEASURED = 4;

/**
 * Il ripiego quando le 4 fette misurate non dividono i blocchi: 1 = nessuno
 * split-K, la forma su cui lo split-K ha misurato il suo guadagno. Non e' una
 * seconda misura, e' il punto di partenza — dichiarato qui perche' un numero di
 * fette che appare senza nome e' un numero inventato.
 */
export const PREFILL_SPLITS_UNSPLIT = 1;

// Dequant di UN blocco q4_0 in registri, identico al banco (indentazione
// compresa: si porta il testo, non la sua idea).
const PREFILL_DEQ_BLOCK = `
    var lo: array<vec4<f32>, 4>;
    var hi: array<vec4<f32>, 4>;
    for (var wi = 0u; wi < 4u; wi = wi + 1u) {
      let by = (vec4<u32>(w[wi]) >> vec4<u32>(0u, 8u, 16u, 24u));
      lo[wi] = vec4<f32>(by & vec4<u32>(15u)) - vec4<f32>(8.0);
      hi[wi] = vec4<f32>((by >> vec4<u32>(4u)) & vec4<u32>(15u)) - vec4<f32>(8.0);
    }`;

/** Le due vie del moltiplicatore, come le nomina chi negozia i limiti. */
export type PrefillGemmWgVia = "idot" | "f32";

/**
 * TUTTO CIO' CHE DIPENDE DAL FORMATO, in una riga per formato.
 *
 * Non e' astrazione per gusto: e' l'unico modo perche' «un kind nuovo si
 * aggiunge in un posto» sia vero e VERIFICATO DAL COMPILATORE. Le quattro cose
 * che cambiano fra q4_0 e q5_K — l'unita' di taglio di K, quante unita' conta
 * il kernel per riga, come si dividono in fette, quanti byte di workgroup
 * storage servono per via — stavano altrimenti in quattro `if (kind ===
 * "q5_K")` sparsi, tutti con un ramo `else` che dava per scontato il q4_0. Un
 * terzo formato (il q4_1 della riga 3) sarebbe caduto in quegli `else`
 * prendendosi la geometria di un altro: fette sbagliate, storage sbagliato,
 * nessun errore. Qui invece `Record<PrefillGemmKind, …>` non compila finche' la
 * riga non c'e'.
 */
interface PrefillGemmKindSpec {
  /** Pesi per unita' INDIVISIBILE: K dev'essere multiplo di questo. */
  kUnit: number;
  /** Perche' e' quella l'unita' — finisce nel messaggio di rifiuto geometrico. */
  kUnitWhy: string;
  /** Che forma ha il formato: entra nel rifiuto dei kind fuori elenco. */
  formatWhy: string;
  /** Le unita' che il KERNEL conta per riga (e che scrive nel WGSL). */
  unitsPerRow(K: number): number;
  /** Vero se `splits` fette dividono `units` unita'. */
  slices(units: number, splits: number): boolean;
  /** Il rifiuto quando non dividono, con i nomi giusti per il formato. */
  sliceWhy(units: number, splits: number): string;
  /** La scala delle fette MISURATA per questo formato. */
  splitsFor(units: number): number;
  /** Workgroup storage per via, in byte, a M righe di chunk. */
  wgBytes: Record<PrefillGemmWgVia, (M: number) => number>;
  /**
   * IL KERNEL ESISTE ED E' MISURATO — MA SOLO I `wired` VENGONO INSTRADATI DAL
   * PIANO IN PRODUZIONE.
   *
   * `false` non significa "non funziona" e non significa "non c'e'": il
   * generatore e' portato dal banco byte per byte, il suo testo e' quello che
   * la fase 0 ha cronometrato e il ktest lo verifica contro il cpuref. Significa
   * che NESSUN SITO DI PRODUZIONE CI PASSA: `kernelVerdict` in
   * `prefillgemmplan.ts` legge questo flag e rende `legacy` con la sua ragione.
   *
   * Perche' serve un flag invece dell'assenza dal kernel: una forma verificata
   * sul device e non instradata NON e' una forma inventata — e' esattamente
   * cio' che la riga 4 del goal consegna al goal 35B. L'alternativa (tenere il
   * kernel fuori dall'elenco fino al giorno del cablaggio) vorrebbe dire
   * portarlo, misurarlo e verificarlo su un ramo che nessun test della suite
   * vede.
   *
   * Il flag si consuma in UN POSTO SOLO, il piano. Un secondo predicato di
   * ammissibilita' altrove e' proprio il difetto che i gate strutturali di
   * `tests/engine-prefillwiring-q5k.test.ts` [d] e `[6f]` sorvegliano.
   */
  wired: boolean;
  /**
   * Perche' e' cablato, o perche' non lo e'. Non e' decorazione: e' la stringa
   * che il piano riporta nella ragione della rotta, cioe' cio' che si legge in
   * telemetria quando un sito resta legacy. Stessa postura di `kUnitWhy` e
   * `sliceWhy` — accanto a ogni decisione, il suo motivo.
   */
  wiredWhy: string;
}

const PREFILL_GEMM_SPEC: Record<PrefillGemmKind, PrefillGemmKindSpec> = {
  q4_0: {
    // il loop avanza a BK = 2 blocchi da 32, quindi l'unita' di taglio e' 64
    kUnit: 64,
    kUnitWhy: "il loop avanza a BK = 2 blocchi da 32",
    formatWhy: "q4_0 nibble a 4 bit + UNA scala f16 per blocco da 32, impacchettati in quartetti i8",
    unitsPerRow: (K) => K / 32,
    slices: (units, splits) => units % (splits * 2) === 0,
    sliceWhy: (units, splits) => `${units} blocchi non divisibili in ${splits} fette da BK=2`,
    // 4 fette se i blocchi si dividono a BK=2, altrimenti nessuno split-K. Il 2
    // NON compare: sul q4_0 non e' stato misurato, e un numero di fette che
    // appare senza misura e' un numero inventato.
    splitsFor: (units) =>
      units % (PREFILL_SPLITS_MEASURED * 2) === 0 ? PREFILL_SPLITS_MEASURED : PREFILL_SPLITS_UNSPLIT,
    wgBytes: {
      // `xs` M*16 u32 + `xsc` M*2 f32 = 72·M B (1.152 a M=16)
      idot: (M) => M * 16 * 4 + M * 2 * 4,
      // `xs` M*16 vec4<f32> = 256·M B (4.096 a M=16)
      f32: (M) => M * 16 * 16,
    },
    wired: true,
    wiredWhy: "riga 1 di engine-ttft: e' il formato di quasi tutto il 4B (172 siti su 248) "
      + "e il cablaggio e' verificato dal ktest e dal checkpoint di TTFT",
  },
  q5_K: {
    // il SUPERBLOCCO da 256: le scale a 6 bit sono condivise dagli otto
    // sotto-blocchi, e una fetta che ne prendesse meta' dovrebbe rileggere
    // l'header comunque — falsificando proprio il conto dei byte del goal.
    kUnit: 256,
    kUnitWhy: "il superblocco Q5_K e' l'unita' indivisibile: le sue scale a 6 bit "
      + "sono condivise dagli otto sotto-blocchi",
    formatWhy: "q5_K superblocco da 256 con le scale a 6 bit e il piano del 5o bit",
    unitsPerRow: (K) => K / 256,
    slices: (units, splits) => units % splits === 0,
    sliceWhy: (units, splits) => `${units} superblocchi da 256 non divisibili in ${splits} fette`,
    // La scala del banco (`kquantSplitsFor` in ttKQuant.ts). Il 2 c'e' e NON e'
    // inventato: sui K-quant le unita' per riga sono otto volte meno numerose
    // (K/256 invece di K/32), quindi il ripiego secco a 1 lascerebbe senza
    // split-K shape che il banco ha misurato divise — `ssm_out` ha K=4096 = 16
    // superblocchi, ma la down degli expert del 35B ne ha DUE (K=512), e li'
    // l'alternativa e' fra 2 fette e nessuna.
    splitsFor: (units) => {
      if (units % PREFILL_SPLITS_MEASURED === 0) return PREFILL_SPLITS_MEASURED;
      if (units % 2 === 0) return 2;
      return PREFILL_SPLITS_UNSPLIT;
    },
    wgBytes: {
      // `xs` M*64 u32 + `xss` M*8 f32 + `xsum` M*8 f32 = 320·M B (5.120 a M=16)
      idot: (M) => M * 64 * 4 + M * 8 * 4 + M * 8 * 4,
      // `xs` M*32 f32 = 128·M B (2.048 a M=16): UN sotto-blocco per volta, non
      // il superblocco intero (che a M=16 sarebbe 16.384 B, il minimo di spec)
      f32: (M) => M * 32 * 4,
    },
    wired: true,
    wiredWhy: "riga 2 di engine-kquant: le 24 `ssm_out` del 4B erano il 37,9% del tempo del "
      + "prefill sul percorso legacy, e il cablaggio e' verificato dal ktest (maxRel 2,61e-7)",
  },
  q4_1: {
    // la GEOMETRIA e' quella del q4_0 — blocchi da 32, il loop avanza a BK = 2
    // — perche' il q4_1 e' un formato a blocchi, non a superblocchi. Quello che
    // cambia e' l'ARITMETICA, e sta tutta nel testo del kernel.
    kUnit: 64,
    kUnitWhy: "il loop avanza a BK = 2 blocchi da 32",
    formatWhy: "q4_1 nibble a 4 bit SENZA l'offset -8 del q4_0, piu' d e m in f16 = UNA parola di scale per blocco da 32",
    unitsPerRow: (K) => K / 32,
    slices: (units, splits) => units % (splits * 2) === 0,
    sliceWhy: (units, splits) => `${units} blocchi non divisibili in ${splits} fette da BK=2`,
    // stessa scala del q4_0: 4 fette se i blocchi si dividono a BK=2, altrimenti
    // nessuno split-K. Il 2 non compare perche' sul q4_1 non e' stato misurato,
    // e i blocchi per riga qui sono tanti quanti sul q4_0 (K/32), non otto volte
    // meno come sui K-quant — il ripiego secco non lascia scoperto niente.
    splitsFor: (units) =>
      units % (PREFILL_SPLITS_MEASURED * 2) === 0 ? PREFILL_SPLITS_MEASURED : PREFILL_SPLITS_UNSPLIT,
    wgBytes: {
      // `xs` M*16 u32 + `xss` M*2 f32 + `xsum` M*2 f32 = 80·M B (1.280 a M=16):
      // sono i 72·M del q4_0 PIU' gli 8·M di `xsum`, cioe' il termine
      // `m * Sigma(x)` che il q4_0 non ha perche' i suoi nibble sono centrati
      // su zero.
      idot: (M) => M * 16 * 4 + M * 2 * 4 + M * 2 * 4,
      // `xs` M*16 vec4<f32> = 256·M B (4.096 a M=16), identico al q4_0: la via
      // f32 legge le attivazioni dense e la somma Sigma(x) la fa in registri.
      f32: (M) => M * 16 * 16,
    },
    wired: true,
    wiredWhy: "riga 3 di engine-kquant: i 4 `blk.0-3.ffn_down` del 4B sono il 71% dei byte del "
      + "segmento `gemm:ffn-down`, e il cablaggio e' verificato dal ktest (maxRel 1,73e-5)",
  },
  // -------------------------------------------------------------------------
  // LE TRE FORME DELLA RIGA 4: kernel in produzione, NESSUN CABLAGGIO.
  //
  // Sono le famiglie del 35B (17,67 GB di q4_K, 1,09 GB di q8_0, 0,66 GB di
  // q6_K contro ZERO byte di q4_0), portate qui dal banco perche' il goal
  // successivo le trovi misurate e verificate — non perche' il 4B ci passi.
  // Ogni numero di queste tre righe viene dal FORMATO o dal testo del kernel di
  // banco (`src/microbench/ttKQuant.ts`), mai scelto.
  // -------------------------------------------------------------------------
  q4_K: {
    // il SUPERBLOCCO da 256, esattamente come il q5_K: `dequantQ4_K`
    // (engine/quant.ts) legge `QK_K = 256` pesi con 12 byte di scale/min a 6 bit
    // condivisi dagli otto sotto-blocchi. Il q4_K e' il q5_K senza `qh`, e il
    // piano del 5o bit non c'entra con l'unita' di taglio.
    kUnit: 256,
    kUnitWhy: "il superblocco Q4_K e' l'unita' indivisibile: le sue scale a 6 bit "
      + "sono condivise dagli otto sotto-blocchi",
    formatWhy: "q4_K superblocco da 256 con le scale a 6 bit, cioe' il q5_K SENZA il piano del 5o bit",
    unitsPerRow: (K) => K / 256,
    slices: (units, splits) => units % splits === 0,
    sliceWhy: (units, splits) => `${units} superblocchi da 256 non divisibili in ${splits} fette`,
    // la scala K-quant del banco (`kquantSplitsFor`), identica al q5_K: le
    // unita' per riga sono otto volte meno numerose che sui formati a blocchi,
    // e il ripiego secco a 1 lascerebbe senza split-K la down degli expert del
    // 35B (K=512 = DUE superblocchi), che il banco ha misurato a 2 fette.
    splitsFor: (units) => {
      if (units % PREFILL_SPLITS_MEASURED === 0) return PREFILL_SPLITS_MEASURED;
      if (units % 2 === 0) return 2;
      return PREFILL_SPLITS_UNSPLIT;
    },
    wgBytes: {
      // `xs` M*64 u32 + `xss` M*8 f32 + `xsum` M*8 f32 = 320·M B (5.120 a M=16):
      // gli stessi tre array del q5_K, perche' e' la stessa forma con un piano
      // di bit in meno — e il piano di bit sta nei PESI, non nelle attivazioni.
      idot: (M) => M * 64 * 4 + M * 8 * 4 + M * 8 * 4,
      // `xs` M*32 f32 = 128·M B (2.048 a M=16): UN sotto-blocco per volta, come
      // il q5_K, per stare sotto il minimo di spec anche a M grandi.
      f32: (M) => M * 32 * 4,
    },
    wired: false,
    wiredWhy: "NON CABLATO: il 4B non ha un byte di q4_K (e' la forma degli expert del 35B, "
      + "117 tensori = 17,67 GB). Cablarla e' il goal successivo, e li' il collo e' la "
      + "residency, non il kernel: serve il piano MoE, l'arena e una baseline 35B fresca, "
      + "nessuna delle quali esiste oggi. Il kernel c'e', e' portato dal banco byte per byte "
      + "ed e' verificato dal ktest: e' una forma misurata e non instradata, non una inventata",
  },
  q6_K: {
    // superblocco da 256 anche qui (`dequantQ6_K`: `QK_K` pesi, 16 scale int8 e
    // `d` in coda). I SOTTO-blocchi delle scale sono da 16, non da 32 — e' una
    // proprieta' dell'ARITMETICA del kernel (per questo `xh` e' per mezzo
    // sotto-blocco), non dell'unita' di taglio di K, che resta il superblocco.
    kUnit: 256,
    kUnitWhy: "il superblocco Q6_K e' l'unita' indivisibile: le sue 16 scale int8 e il `d` "
      + "in coda valgono per tutti i 256 pesi",
    formatWhy: "q6_K superblocco da 256 con `ql`+`qh` a 6 bit, 16 scale int8 e i pesi centrati su -32",
    unitsPerRow: (K) => K / 256,
    slices: (units, splits) => units % splits === 0,
    sliceWhy: (units, splits) => `${units} superblocchi da 256 non divisibili in ${splits} fette`,
    // stessa scala K-quant del q5_K/q4_K: la shape misurata del 35B (K=512) ha
    // DUE superblocchi per riga, cioe' proprio il caso in cui il 2 serve.
    splitsFor: (units) => {
      if (units % PREFILL_SPLITS_MEASURED === 0) return PREFILL_SPLITS_MEASURED;
      if (units % 2 === 0) return 2;
      return PREFILL_SPLITS_UNSPLIT;
    },
    wgBytes: {
      // `xs` M*64 u32 + `xss` M*8 f32 + `xh` M*16 f32 = 352·M B (5.632 a M=16).
      // I 32·M in piu' del q5_K sono `xh`: la somma delle attivazioni per MEZZO
      // sotto-blocco (16 elementi), che e' la granularita' delle scale del Q6_K.
      idot: (M) => M * 64 * 4 + M * 8 * 4 + M * 16 * 4,
      // `xs` M*32 f32 = 128·M B (2.048 a M=16): un sotto-blocco da 32 per volta.
      f32: (M) => M * 32 * 4,
    },
    wired: false,
    wiredWhy: "NON CABLATO: il 4B non ha un byte di q6_K nei GEMM di prefill (il suo unico q6_K "
      + "e' `token_embd`, che non e' una moltiplicazione di prefill). E' la down degli expert "
      + "di 3 layer del 35B, e vale la stessa cosa del q4_K: il cablaggio e' il goal successivo. "
      + "Il kernel c'e' ed e' verificato dal ktest",
  },
  q8_0: {
    // LA GEOMETRIA E' QUELLA DEL q4_0, e viene dal TESTO del kernel: il ciclo
    // avanza `b0 = b0 + 2u` su blocchi da 32 (`dequantQ8_0`: 32 pesi per
    // blocco), quindi l'unita' di taglio e' 64 pesi e le fette devono dividere i
    // blocchi A COPPIE. Il `check` del banco e' piu' largo (conta in blocchi da
    // 32 e chiede solo `upr % splits == 0`): sulla shape misurata i due danno
    // gli stessi numeri (K=2048 -> 64 blocchi, 4 fette, PER=16), ma una PER
    // dispari farebbe leggere alla fetta un blocco della fetta dopo. Qui si
    // tiene la soglia del kernel, non quella del banco.
    kUnit: 64,
    kUnitWhy: "il loop avanza a BK = 2 blocchi da 32",
    formatWhy: "q8_0 pesi GIA' in i8, 8 parole per blocco da 32 e le scale a DUE f16 per parola come il q4_0",
    unitsPerRow: (K) => K / 32,
    slices: (units, splits) => units % (splits * 2) === 0,
    sliceWhy: (units, splits) => `${units} blocchi non divisibili in ${splits} fette da BK=2`,
    // stessa scala del q4_0 e del q4_1: 4 fette se i blocchi si dividono a
    // BK=2, altrimenti nessuno split-K. Il 2 non compare perche' sul q8_0 non e'
    // stato misurato, e i blocchi per riga qui sono tanti quanti sul q4_0
    // (K/32), non otto volte meno come sui K-quant.
    splitsFor: (units) =>
      units % (PREFILL_SPLITS_MEASURED * 2) === 0 ? PREFILL_SPLITS_MEASURED : PREFILL_SPLITS_UNSPLIT,
    wgBytes: {
      // `xs` M*16 u32 + `xss` M*2 f32 = 72·M B (1.152 a M=16): gli STESSI due
      // array del q4_0, e nessun `xsum` — il q8_0 non ha termine costante,
      // perche' i suoi pesi sono gia' interi con segno.
      idot: (M) => M * 16 * 4 + M * 2 * 4,
      // `xs` M*64 f32 = 256·M B (4.096 a M=16): la via f32 del q8_0 tiene DUE
      // blocchi da 32 in virgola mobile (M*64 f32), non 16 vec4 come il q4_0.
      // Il numero coincide, la dichiarazione no — ed e' la dichiarazione che il
      // test confronta col testo.
      f32: (M) => M * 64 * 4,
    },
    wired: true,
    wiredWhy: "CABLATO il 2026-08-15 (goal engine-velocita-decode, riga 2d), e cio' che l'ha reso "
      + "sicuro NON e' una misura nuova: e' il predicato su N in `kernelVerdict`. La ragione "
      + "vecchia diceva «non cablato perche' i 48 siti `ssm_alpha`/`ssm_beta` del 4B hanno N=32, "
      + "mezzo workgroup sulla forma split-K che ne produce 64» — vera, ma risolta nel posto "
      + "sbagliato: escludeva una FAMIGLIA intera per proteggere una SHAPE. Ora l'esclusione e' "
      + "sulla shape (`PREFILL_GEMM_ROWS_PER_WG`, casi in engine-prefillgemm-nmin.test.ts): quei "
      + "48 siti restano legacy per la loro geometria, e i 100 tensori attn del 35B (1,09 GB, "
      + "N=4096) passano. Misura, banco fase 0 di engine-kquant su [2048,4096]: a M=16 "
      + "splitk-idot 0,0376 ms contro 1,3224 della legacy = 35,2x; a M=1 0,0698 contro 0,2278 = "
      + "3,26x, cioe' paga anche nel regime del decode",
  },
};

/** Il restringimento al formato, letto dall'elenco esportato e non da una copia. */
export function isPrefillGemmKind(k: string): k is PrefillGemmKind {
  return (PREFILL_GEMM_KINDS as readonly string[]).includes(k);
}

/**
 * I FORMATI CHE IL PIANO INSTRADA DAVVERO — derivati dal flag, non riscritti.
 *
 * Chi vuole sapere «quante vie veloci sono accese in produzione» chiede QUESTO,
 * non `PREFILL_GEMM_KINDS.length`: da riga 4 i due numeri sono diversi, e
 * confonderli e' il modo in cui una forma non cablata finisce contata come se
 * lo fosse.
 */
export const PREFILL_GEMM_WIRED_KINDS: readonly PrefillGemmKind[] =
  PREFILL_GEMM_KINDS.filter((k) => PREFILL_GEMM_SPEC[k].wired);

/**
 * Il flag di cablaggio DEL FORMATO, con la sua ragione, in una risposta sola.
 *
 * L'unico consumatore previsto e' `kernelVerdict` in `prefillgemmplan.ts`: e'
 * li' che si decide la via di un sito, ed e' l'unico posto dove un predicato di
 * ammissibilita' puo' stare senza diventare una seconda soglia che diverge in
 * silenzio (la lezione di it.7). Un booleano nudo non basterebbe: la ragione
 * finisce nella rotta e quindi in telemetria, dove serve a capire PERCHE' un
 * sito e' rimasto legacy senza aprire il sorgente.
 */
export function prefillGemmWiring(kind: PrefillGemmKind): { wired: boolean; why: string } {
  const spec = PREFILL_GEMM_SPEC[kind];
  return { wired: spec.wired, why: spec.wiredWhy };
}

/**
 * Contorni del kernel: si RIFIUTA quello che non e' stato misurato invece di
 * generare una forma inventata (la regola che regge gia' il GEMV veloce).
 *
 * UNA SEDE SOLA, e in QUEST'ORDINE: prima il FORMATO (il kind sta fra quelli
 * misurati?), poi la GEOMETRIA (K si divide nell'unita' del formato? le unita'
 * si dividono nelle fette?). L'ordine non e' estetico: su un kind non
 * supportato con K storto, la ragione strutturale e' il kind — e' quella su cui
 * si puo' agire, mentre il K sarebbe una risposta vera e inutile.
 *
 * `units` = le unita' che il kernel conta per riga (blocchi da 32 sul q4_0,
 * superblocchi da 256 sul q5_K); `per` = quante ne tocca a ogni fetta.
 */
/**
 * RIGHE DI USCITA PER WORKGROUP della forma split-K. Non e' una scelta: e' la
 * geometria del kernel, ed e' la stessa che `prefillGemmGrid` usa per la
 * griglia (`ceil(N / 64)`). Sta qui perche' il predicato di ammissibilita' la
 * legga dallo stesso posto della griglia invece di ricopiarla — due copie di
 * questo numero sono due forme che divergono in silenzio.
 *
 * SI CONSUMA IN `kernelVerdict` (prefillgemmplan.ts), non qui. `prefillGemmCheck`
 * e' il contorno del KERNEL — «questa forma si genera?» — e la risposta su N e'
 * si': il kernel guarda `r < N` e produce il valore giusto anche a N=32. Cio'
 * che N decide e' se la forma CONVIENE, cioe' l'AMMISSIBILITA' AL PIANO, e
 * quella ha una sede sola dichiarata. Mettere il controllo qui romperebbe anche
 * le query di dimensionamento (`prefillGemmWorkgroupStorageBytes` si interroga
 * a N=1 apposta, per provare che il fabbisogno dipende solo da M).
 */
export const PREFILL_GEMM_ROWS_PER_WG = 64;

function prefillGemmCheck(o: PrefillGemmOpts, who: string): { units: number; per: number } {
  const kind = o.kind as string;
  if (!isPrefillGemmKind(kind)) {
    throw new Error(
      `${who}: kind "${kind}" non supportato — le vie veloci di prefill sono ` +
      `${PREFILL_GEMM_KINDS.join(" e ")} e basta: sono le sole forme MISURATE (` +
      PREFILL_GEMM_KINDS.map((k) => PREFILL_GEMM_SPEC[k].formatWhy).join("; ") +
      "), e ogni altra famiglia resta sulla via legacy finche' non ha la sua misura");
  }
  const spec = PREFILL_GEMM_SPEC[kind];
  if (o.K % spec.kUnit !== 0) {
    throw new Error(`${who}: K=${o.K} non multiplo di ${spec.kUnit} (${spec.kUnitWhy})`);
  }
  const units = spec.unitsPerRow(o.K);
  if (!spec.slices(units, o.splits)) {
    throw new Error(`${who}: ${spec.sliceWhy(units, o.splits)}`);
  }
  return { units, per: units / o.splits };
}

/**
 * VIA INTERA q4_0 x q8_0 (`dot4I8Packed`) — LA FORMA VINCENTE della riga 1.
 *
 * Pesi gia' a 4 bit, attivazioni quantizzate a i8 per blocco da 32 (vedi
 * `prefillQuantXQ8Wgsl`), accumulo in i32, `sc_w * sc_x` applicata UNA VOLTA
 * PER BLOCCO. Dal ciclo interno sparisce il dequant in virgola mobile: restano
 * otto `dot4I8Packed`.
 *
 * NIENTE `enable packed_4x8_integer_dot_product;`: e' una LANGUAGE FEATURE
 * (`navigator.gpu.wgslLanguageFeatures`), non un'estensione. Scriverlo fa
 * fallire la compilazione con «expected extension» — costato una run in it.5.
 */
export function prefillGemmQ4SplitKIdotWgsl(o: PrefillGemmOpts): string {
  const { N, M } = o;
  const { units: bpr, per } = prefillGemmCheck(o, "prefillGemmQ4SplitKIdotWgsl");
  return `@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> xsc: array<f32>;
const BPR = ${bpr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
var<workgroup> xs: array<u32, ${M * 16}>;
var<workgroup> xss: array<f32, ${M * 2}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let bStart = s * PER;
  let bEnd = bStart + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var b0 = bStart;
  loop {
    if (b0 >= bEnd) { break; }
    // due blocchi per passata, come 'splitk': 16 u32 di attivazioni per riga
    for (var idx = t; idx < ${M * 16}u; idx = idx + 64u) {
      let m = idx / 16u;
      let qi = idx % 16u;
      xs[idx] = xq[(m * BPR + b0) * 8u + qi];
    }
    for (var idx = t; idx < ${M * 2}u; idx = idx + 64u) {
      let m = idx / 2u;
      xss[idx] = xsc[m * BPR + b0 + (idx % 2u)];
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      for (var bi = 0u; bi < 2u; bi = bi + 1u) {
        let gb = r * BPR + b0 + bi;
        let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];
        let w = qs4[gb];
        // i 32 pesi del blocco come QUATTRO+QUATTRO quartetti i8 impacchettati:
        // (nibble - 8) in complemento a due su 8 bit = (nibble + 248) & 255.
        var lo: array<u32, 4>;
        var hi: array<u32, 4>;
        for (var wi = 0u; wi < 4u; wi = wi + 1u) {
          let word = w[wi];
          let nl = (vec4<u32>(word) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(15u);
          let nh = (vec4<u32>(word) >> vec4<u32>(4u, 12u, 20u, 28u)) & vec4<u32>(15u);
          let sl = (nl + vec4<u32>(248u)) & vec4<u32>(255u);
          let sh = (nh + vec4<u32>(248u)) & vec4<u32>(255u);
          lo[wi] = sl.x | (sl.y << 8u) | (sl.z << 16u) | (sl.w << 24u);
          hi[wi] = sh.x | (sh.y << 8u) | (sh.z << 16u) | (sh.w << 24u);
        }
        let xb = bi * 8u;
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          let xo = m * 16u + xb;
          var idot = 0i;
          for (var wi = 0u; wi < 4u; wi = wi + 1u) {
            idot = idot + dot4I8Packed(lo[wi], xs[xo + wi]);
            idot = idot + dot4I8Packed(hi[wi], xs[xo + 4u + wi]);
          }
          // la scala si applica UNA VOLTA per blocco, non per elemento
          acc[m] = acc[m] + f32(idot) * sc * xss[m * 2u + bi];
        }
      }
    }
    workgroupBarrier();
    b0 = b0 + 2u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * VIA f32 — FALLBACK DICHIARATO. Stessa mappatura (64 righe di uscita per
 * workgroup, pesi in registri, attivazioni in workgroup memory), K spezzato in
 * S fette lungo `wid.y`, dequant in virgola mobile nel ciclo interno. E' il
 * termine di paragone su cui la via intera ha misurato 1,745x: si porta per i
 * device senza `packed_4x8_integer_dot_product`, non come alternativa
 * preferibile.
 */
export function prefillGemmQ4SplitKWgsl(o: PrefillGemmOpts): string {
  const { K, N, M } = o;
  const { units: bpr, per } = prefillGemmCheck(o, "prefillGemmQ4SplitKWgsl");
  return `@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
const BPR = ${bpr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const K4 = ${K / 4}u;
const M_ROWS = ${M}u;
var<workgroup> xs: array<vec4<f32>, ${M * 16}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let bStart = s * PER;
  let bEnd = bStart + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var b0 = bStart;
  loop {
    if (b0 >= bEnd) { break; }
    for (var idx = t; idx < ${M * 16}u; idx = idx + 64u) {
      let m = idx / 16u;
      let qi = idx % 16u;
      xs[idx] = x4[m * K4 + b0 * 8u + qi];
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      for (var bi = 0u; bi < 2u; bi = bi + 1u) {
        let gb = r * BPR + b0 + bi;
        let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];
        let w = qs4[gb];${PREFILL_DEQ_BLOCK}
        let xb = bi * 8u;
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          let xo = m * 16u + xb;
          var bd = 0.0;
          for (var wi = 0u; wi < 4u; wi = wi + 1u) {
            bd = bd + dot(lo[wi], xs[xo + wi]) + dot(hi[wi], xs[xo + 4u + wi]);
          }
          acc[m] = acc[m] + sc * bd;
        }
      }
    }
    workgroupBarrier();
    b0 = b0 + 2u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

// ---------------------------------------------------------------------------
// PREFILL GEMM q5_K SPLIT-K — port dal banco (src/microbench/ttKQuant.ts).
//
// Provenienza: riga 2 di engine-kquant, fase 0 chiusa in it.1-it.3. Le due
// forme che arrivano qui sono quelle MISURATE su `blk.*.ssm_out` del 4B
// (K=4096, N=2560, M=16): 1,2700 -> 0,0452 ms = 28,10x sulla legacy
// (results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json,
// celle `gemm-kquant-multirow` `q5_K/*`).
//
// STESSA REGOLA DEL PORT q4_0: il testo WGSL e' quello del banco, riga per
// riga, e ogni divergenza sta in `PREFILL_GEMM_PORT_DIFFS` con la sua ragione.
//
// ORDINE DEI BINDING (congelato — lo consumano il piano e il wiring):
//   idot [0 blocks, 1 xq(u32), 2 part(rw), 3 xsc(f32)]
//   f32  [0 blocks, 1 x(f32 row-major M x K), 2 part(rw)]
// UN SOLO buffer di pesi: il superblocco GGUF copiato in parole. E' la
// differenza strutturale col q4_0, dove `repackQ4_0` spezza in `qs` + `scales`.
// Il quantizzatore delle attivazioni e la combine sono gli STESSI del q4_0
// (`prefillQuantXQ8Wgsl`, `prefillSplitKCombineWgsl`): i sotto-blocchi K-quant
// sono anch'essi da 32, otto per superblocco.
// ---------------------------------------------------------------------------

/**
 * VIA INTERA q5_K x q8_0 (`dot4I8Packed`) — LA FORMA VINCENTE della riga 2.
 *
 * IL PUNTO, in una riga: il piano del 5o bit si somma IMPACCHETTATO. I 32 pesi
 * di un sotto-blocco stanno in 32 nibble = 8 parole; il loro 5o bit sta in
 * `qh`, un bit per elemento nella posizione `is` del proprio byte. Estrarlo
 * elemento per elemento costerebbe 32 test; in forma impacchettata e'
 * `((qhw >> is) & 0x01010101) << 4`, cioe' "+16 a ogni byte che ce l'ha", e i
 * quattro pesi restano in una parola pronta per `dot4I8Packed`.
 *
 * IL TERMINE CHE LA q4_0 NON HA: Q5_K e' `w = d*sc_j*q - dmin*mn_j`, quindi al
 * prodotto scalare va sottratto `dmin*mn_j` per la SOMMA delle attivazioni del
 * sotto-blocco. Quella somma non dipende dalla riga di pesi: si calcola UNA
 * volta in memoria di gruppo (`xsum`), non una per thread.
 *
 * NIENTE `enable packed_4x8_integer_dot_product;`: e' una LANGUAGE FEATURE, non
 * un'estensione. Scriverlo fa fallire la compilazione con «expected extension».
 */
export function prefillGemmQ5KSplitKIdotWgsl(o: PrefillGemmOpts): string {
  const { N, M } = o;
  const { units: upr, per } = prefillGemmCheck(o, "prefillGemmQ5KSplitKIdotWgsl");
  return `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<storage, read> xsc: array<f32>;
const SBPR = ${upr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
const WORDS = 44u;
var<workgroup> xs: array<u32, ${M * 64}>;
var<workgroup> xss: array<f32, ${M * 8}>;
var<workgroup> xsum: array<f32, ${M * 8}>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let sbEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var sb = s * PER;
  loop {
    if (sb >= sbEnd) { break; }
    // attivazioni del superblocco: M righe x 8 sotto-blocchi x 8 parole
    for (var idx = t; idx < ${M * 64}u; idx = idx + 64u) {
      let m = idx / 64u;
      let jj = (idx % 64u) / 8u;
      let ii = idx % 8u;
      xs[idx] = xq[((m * SBPR + sb) * 8u + jj) * 8u + ii];
    }
    for (var idx = t; idx < ${M * 8}u; idx = idx + 64u) {
      let m = idx / 8u;
      let jj = idx % 8u;
      xss[idx] = xsc[(m * SBPR + sb) * 8u + jj];
    }
    workgroupBarrier();
    // Sigma(x) per (riga, sotto-blocco): non dipende dalla riga di PESI, quindi
    // una volta per workgroup e non una per thread.
    for (var idx = t; idx < ${M * 8}u; idx = idx + 64u) {
      var sq = 0i;
      for (var ii = 0u; ii < 8u; ii = ii + 1u) {
        sq = sq + dot4I8Packed(0x01010101u, xs[idx * 8u + ii]);
      }
      xsum[idx] = f32(sq) * xss[idx];
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      let wb = (r * SBPR + sb) * WORDS;
      let dm = unpack2x16float(blocks[wb]);
      for (var g = 0u; g < 4u; g = g + 1u) {
        let is = 2u * g;
        // get_scale_min_k4, identico al GEMV di produzione (gemvQ5KWgsl)
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
        // gli 8+8 quartetti del gruppo, in registri: cosi' le M righe li
        // riusano invece di rileggerli
        var lo: array<u32, 8>;
        var hi: array<u32, 8>;
        for (var ii = 0u; ii < 8u; ii = ii + 1u) {
          let word = blocks[wb + 12u + g * 8u + ii];
          let qhw = blocks[wb + 4u + ii];
          lo[ii] = (word & 0x0f0f0f0fu) + (((qhw >> is) & 0x01010101u) << 4u);
          hi[ii] = ((word >> 4u) & 0x0f0f0f0fu) + (((qhw >> (is + 1u)) & 0x01010101u) << 4u);
        }
        let d1 = dm.x * f32(sc1); let min1 = dm.y * f32(mn1);
        let d2 = dm.x * f32(sc2); let min2 = dm.y * f32(mn2);
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          let bLo = m * 64u + is * 8u;
          let bHi = bLo + 8u;
          var i1 = 0i; var i2 = 0i;
          for (var ii = 0u; ii < 8u; ii = ii + 1u) {
            i1 = i1 + dot4I8Packed(lo[ii], xs[bLo + ii]);
            i2 = i2 + dot4I8Packed(hi[ii], xs[bHi + ii]);
          }
          acc[m] = acc[m]
            + d1 * f32(i1) * xss[m * 8u + is] - min1 * xsum[m * 8u + is]
            + d2 * f32(i2) * xss[m * 8u + is + 1u] - min2 * xsum[m * 8u + is + 1u];
        }
      }
    }
    workgroupBarrier();
    sb = sb + 1u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * VIA f32 — FALLBACK DICHIARATO per i device senza
 * `packed_4x8_integer_dot_product`. Stessa mappatura, attivazioni in virgola
 * mobile lette DENSE (row-major M x K, binding 1).
 *
 * Il tile e' di UN SOTTO-BLOCCO da 32 per volta (M x 32 f32 = 2.048 B a M=16),
 * non del superblocco intero: M x 256 f32 sarebbero 16.384 B a M=16, cioe'
 * l'INTERO minimo di spec WebGPU per un solo array — e questa forma deve poter
 * girare anche dove il tetto e' quello garantito.
 *
 * E' un fallback, non un'alternativa preferibile: si porta perche' senza di lui
 * i device senza la language feature resterebbero sulla legacy.
 */
export function prefillGemmQ5KSplitKWgsl(o: PrefillGemmOpts): string {
  const { N, M, K } = o;
  const { units: upr, per } = prefillGemmCheck(o, "prefillGemmQ5KSplitKWgsl");
  return `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
const SBPR = ${upr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
const K_DIM = ${K}u;
const WORDS = 44u;
var<workgroup> xs: array<f32, ${M * 32}>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let sbEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var sb = s * PER;
  loop {
    if (sb >= sbEnd) { break; }
    // UN SOTTO-BLOCCO DA 32 PER VOLTA, non un gruppo da 64: il tile scende a
    // M x 32 f32 (2.048 B a M=16) e i pesi del sotto-blocco stanno in 32
    // registri invece di 64. Il sotto-blocco is copre gli elementi
    // [is*32, is*32+32) — contigui, perche' g*64 + h*32 = (2g+h)*32.
    for (var is = 0u; is < 8u; is = is + 1u) {
      let g = is >> 1u;
      let hiHalf = (is & 1u) == 1u;
      let base = sb * 256u + is * 32u;
      for (var idx = t; idx < ${M * 32}u; idx = idx + 64u) {
        let m = idx / 32u;
        xs[idx] = x[m * K_DIM + base + (idx % 32u)];
      }
      workgroupBarrier();
      if (r < N_ROWS) {
        let wb = (r * SBPR + sb) * WORDS;
        let dm = unpack2x16float(blocks[wb]);
        var sc: u32; var mn: u32;
        if (is < 4u) {
          sc = sbyte(wb, 4u + is) & 63u;
          mn = sbyte(wb, 4u + is + 4u) & 63u;
        } else {
          sc = (sbyte(wb, 4u + is + 4u) & 0xfu) | ((sbyte(wb, 4u + is - 4u) >> 6u) << 4u);
          mn = (sbyte(wb, 4u + is + 4u) >> 4u) | ((sbyte(wb, 4u + is) >> 6u) << 4u);
        }
        let dsc = dm.x * f32(sc);
        let dmn = dm.y * f32(mn);
        var q: array<f32, 32>;
        for (var l = 0u; l < 32u; l = l + 1u) {
          let ql = sbyte(wb, 48u + g * 32u + l);
          let qh = sbyte(wb, 16u + l);
          var a: f32;
          if (hiHalf) { a = f32(ql >> 4u); } else { a = f32(ql & 0xfu); }
          if ((qh & (1u << is)) != 0u) { a = a + 16.0; }
          q[l] = a;
        }
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          var qx = 0.0; var sx = 0.0;
          for (var l = 0u; l < 32u; l = l + 1u) {
            let xv = xs[m * 32u + l];
            qx = qx + q[l] * xv; sx = sx + xv;
          }
          acc[m] = acc[m] + dsc * qx - dmn * sx;
        }
      }
      workgroupBarrier();
    }
    sb = sb + 1u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

// ---------------------------------------------------------------------------
// PREFILL GEMM q4_1 SPLIT-K — port dal banco (src/microbench/ttKQuant.ts,
// `kquantQ41MultiRowSplitKIdotWgsl` / `kquantQ41MultiRowSplitKWgsl`).
//
// Provenienza: riga 3 di engine-kquant, su `blk.0-3.ffn_down` del 4B (K=9216 =
// 288 blocchi da 32, N=2560, M=16) — il 71% dei byte del segmento
// `gemm:ffn-down` (`KQUANT_SHAPES` in ttKQuant.ts).
//
// STESSA REGOLA DEL PORT: il testo WGSL e' quello del banco, riga per riga, e
// ogni divergenza sta in `PREFILL_GEMM_PORT_DIFFS` con la sua ragione.
//
// ORDINE DEI BINDING (congelato — lo consumano il piano e il wiring): sono gli
// STESSI del q4_0, perche' `repackQ4_1` spezza anche questo formato in due
// buffer (qs + scales), al contrario del superblocco unico del Q5_K.
//   idot [0 qs4, 1 scales, 2 xq(u32), 3 part(rw), 4 xsc(f32)]
//   f32  [0 qs4, 1 scales, 2 x4,      3 part(rw)]
// Quantizzatore delle attivazioni e combine sono quelli del q4_0
// (`prefillQuantXQ8Wgsl`, `prefillSplitKCombineWgsl`): stessi blocchi da 32.
//
// COSA DISTINGUE QUESTO FORMATO DAL q4_0, e sono DUE fatti aritmetici che la
// forma non tradisce (stessi binding, stessa mappatura, stesso passo):
//   1. `w = d*q + m` con q in [0,15]: i nibble vanno in i8 SENZA la correzione
//      -8, e restano positivi;
//   2. le scale sono UNA parola per blocco (d nei 16 bit bassi, m negli alti),
//      quindi `unpack2x16float(scales[gb])` — non `scales[gb >> 1u][gb & 1u]`,
//      che e' la forma q4_0 con due f16 per parola.
// Dal punto 1 discende il terzo termine: senza offset serve `m * Sigma(x)` per
// blocco, e quella somma non dipende dalla riga di pesi — si calcola UNA volta
// in memoria di gruppo (`xsum`), come fa il Q5_K con `dmin*mn_j`.
// ---------------------------------------------------------------------------

/**
 * VIA INTERA q4_1 x q8_0 (`dot4I8Packed`) — la forma vincente della riga 3.
 *
 * Passo di due blocchi per giro, come la forma q4_0 misurata in riga 1 di
 * engine-ttft: stessa mappatura, stessa occupancy, cambia solo l'aritmetica del
 * formato. I nibble si estraggono impacchettati (`w & 0x0f0f0f0f` e
 * `(w >> 4) & 0x0f0f0f0f`) e restano quattro per parola, pronti per
 * `dot4I8Packed`; nessuno supera 15, quindi la somma per byte non trabocca.
 *
 * NIENTE `enable packed_4x8_integer_dot_product;`: e' una LANGUAGE FEATURE, non
 * un'estensione. Scriverlo fa fallire la compilazione con «expected extension».
 */
export function prefillGemmQ41SplitKIdotWgsl(o: PrefillGemmOpts): string {
  const { N, M } = o;
  const { units: bpr, per } = prefillGemmCheck(o, "prefillGemmQ41SplitKIdotWgsl");
  return `@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> xsc: array<f32>;
const BPR = ${bpr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
var<workgroup> xs: array<u32, ${M * 16}>;
var<workgroup> xss: array<f32, ${M * 2}>;
var<workgroup> xsum: array<f32, ${M * 2}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let bEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var b0 = s * PER;
  loop {
    if (b0 >= bEnd) { break; }
    for (var idx = t; idx < ${M * 16}u; idx = idx + 64u) {
      let m = idx / 16u;
      let qi = idx % 16u;
      xs[idx] = xq[(m * BPR + b0) * 8u + qi];
    }
    for (var idx = t; idx < ${M * 2}u; idx = idx + 64u) {
      let m = idx / 2u;
      xss[idx] = xsc[m * BPR + b0 + (idx % 2u)];
    }
    workgroupBarrier();
    for (var idx = t; idx < ${M * 2}u; idx = idx + 64u) {
      let m = idx / 2u;
      let bi = idx % 2u;
      var sq = 0i;
      for (var ii = 0u; ii < 8u; ii = ii + 1u) {
        sq = sq + dot4I8Packed(0x01010101u, xs[m * 16u + bi * 8u + ii]);
      }
      xsum[idx] = f32(sq) * xss[idx];
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      for (var bi = 0u; bi < 2u; bi = bi + 1u) {
        let gb = r * BPR + b0 + bi;
        let dm = unpack2x16float(scales[gb]);
        let w = qs4[gb];
        var lo: array<u32, 4>;
        var hi: array<u32, 4>;
        for (var wi = 0u; wi < 4u; wi = wi + 1u) {
          lo[wi] = w[wi] & 0x0f0f0f0fu;
          hi[wi] = (w[wi] >> 4u) & 0x0f0f0f0fu;
        }
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          let xo = m * 16u + bi * 8u;
          var idot = 0i;
          for (var wi = 0u; wi < 4u; wi = wi + 1u) {
            idot = idot + dot4I8Packed(lo[wi], xs[xo + wi]);
            idot = idot + dot4I8Packed(hi[wi], xs[xo + 4u + wi]);
          }
          acc[m] = acc[m] + dm.x * f32(idot) * xss[m * 2u + bi] + dm.y * xsum[m * 2u + bi];
        }
      }
    }
    workgroupBarrier();
    b0 = b0 + 2u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * VIA f32 — FALLBACK DICHIARATO per i device senza
 * `packed_4x8_integer_dot_product`. Stessa mappatura, attivazioni lette dense
 * (`x4`, vec4<f32>) come nella via f32 del q4_0.
 *
 * Il dequant NON sottrae 8.0: e' la stessa differenza di formato della via
 * intera, scritta in virgola mobile. E il termine costante `m * Sigma(x)` qui
 * si accumula in registri (`sx`), perche' senza `dot4I8Packed` non c'e' niente
 * da condividere fra le righe.
 */
export function prefillGemmQ41SplitKWgsl(o: PrefillGemmOpts): string {
  const { N, M, K } = o;
  const { units: bpr, per } = prefillGemmCheck(o, "prefillGemmQ41SplitKWgsl");
  return `@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
const BPR = ${bpr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
const K4 = ${K / 4}u;
var<workgroup> xs: array<vec4<f32>, ${M * 16}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let bEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var b0 = s * PER;
  loop {
    if (b0 >= bEnd) { break; }
    for (var idx = t; idx < ${M * 16}u; idx = idx + 64u) {
      let m = idx / 16u;
      let qi = idx % 16u;
      xs[idx] = x4[m * K4 + b0 * 8u + qi];
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      for (var bi = 0u; bi < 2u; bi = bi + 1u) {
        let gb = r * BPR + b0 + bi;
        let dm = unpack2x16float(scales[gb]);
        let w = qs4[gb];
        var lo: array<vec4<f32>, 4>;
        var hi: array<vec4<f32>, 4>;
        for (var wi = 0u; wi < 4u; wi = wi + 1u) {
          let by = (vec4<u32>(w[wi]) >> vec4<u32>(0u, 8u, 16u, 24u));
          lo[wi] = vec4<f32>(by & vec4<u32>(15u));
          hi[wi] = vec4<f32>((by >> vec4<u32>(4u)) & vec4<u32>(15u));
        }
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          let xo = m * 16u + bi * 8u;
          var qx = 0.0;
          var sx = 0.0;
          for (var wi = 0u; wi < 4u; wi = wi + 1u) {
            let xa = xs[xo + wi];
            let xb = xs[xo + 4u + wi];
            qx = qx + dot(lo[wi], xa) + dot(hi[wi], xb);
            sx = sx + dot(vec4<f32>(1.0), xa) + dot(vec4<f32>(1.0), xb);
          }
          acc[m] = acc[m] + dm.x * qx + dm.y * sx;
        }
      }
    }
    workgroupBarrier();
    b0 = b0 + 2u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

// ---------------------------------------------------------------------------
// PREFILL GEMM q4_K / q6_K / q8_0 SPLIT-K — port dal banco
// (src/microbench/ttKQuant.ts, `kquantQ4K*` / `kquantQ6K*` / `kquantQ80*`).
//
// PORTATE MA NON CABLATE, ed e' il punto della riga 4: `wired: false` in
// `PREFILL_GEMM_SPEC` tiene il piano fuori da questi tre formati, quindi cio'
// che il 4B esegue non cambia di un dispatch. Il goal successivo (35B) trova le
// forme gia' portate, misurate e verificate col ktest invece di doverle
// riscrivere — e le trova QUI, dentro la suite, non su un ramo.
//
// Provenienza dei numeri (fase 0, p50 a M=16, legacy/veloce nella forma
// `idot | f32`, artefatto
// results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json):
//   q4_K [2048,512]  0,070016 -> 0,016832 | 0,061504 =  4,16x | 1,14x
//   q4_K [512,2048]  0,049536 -> 0,009472 | 0,026752 =  5,23x | 1,85x
//   q6_K [512,2048]  0,152896 -> 0,024960 | 0,092160 =  6,13x | 1,66x
//   q8_0 [2048,4096] 1,322432 -> 0,037568 | 0,075008 = 35,20x | 17,63x
// (Sul q4_K la via f32 a [2048,512] non passa la regola di stop dell'1,5x: e'
// un fatto della fase 0, registrato qui perche' chi cabla lo veda.)
//
// STESSA REGOLA DEL PORT delle tre famiglie gia' in produzione: il testo WGSL e'
// quello del banco, riga per riga, e ogni divergenza sta in
// `PREFILL_GEMM_PORT_DIFFS` con la sua ragione.
//
// ORDINE DEI BINDING (congelato — lo consumeranno il piano e il wiring del 35B):
//   q4_K/q6_K idot [0 blocks, 1 xq(u32), 2 part(rw), 3 xsc(f32)]
//   q4_K/q6_K f32  [0 blocks, 1 x(f32 row-major M x K), 2 part(rw)]
//   q8_0      idot [0 qs, 1 scales, 2 xq(u32), 3 part(rw), 4 xsc(f32)]
//   q8_0      f32  [0 qs, 1 scales, 2 x(f32 row-major M x K), 3 part(rw)]
// I due K-quant hanno UN buffer di pesi (il superblocco GGUF copiato in parole),
// come il q5_K; il q8_0 ne ha DUE, perche' `repackQ8_0` (quant.ts) spezza in qs
// + scales a due f16 per parola, come il q4_0. Il quantizzatore delle
// attivazioni e la combine sono gli STESSI di tutte le altre famiglie
// (`prefillQuantXQ8Wgsl`, `prefillSplitKCombineWgsl`): blocchi da 32 ovunque.
// ---------------------------------------------------------------------------

/**
 * VIA INTERA q4_K x q8_0 (`dot4I8Packed`) — il Q5_K SENZA il piano del 5o bit.
 *
 * Superblocco da 144 B = 36 parole: `d,dmin` nella parola 0, 12 byte di scale a
 * 6 bit (byte 4..15), `qs` a byte **16** — dove il Q5_K tiene il suo `qh`.
 * Stessa `get_scale_min_k4`, stessa aritmetica `d*sc*q - dmin*mn`, un piano di
 * bit in meno. Gli offset sono quelli di `dequantQ4_K` (engine/quant.ts) e di
 * `gemvQ4KWgsl`: e' li' che un port distratto sbaglia (48 invece di 16, 44
 * parole invece di 36).
 *
 * NIENTE `enable packed_4x8_integer_dot_product;`: e' una LANGUAGE FEATURE, non
 * un'estensione. Scriverlo fa fallire la compilazione con «expected extension».
 */
export function prefillGemmQ4KSplitKIdotWgsl(o: PrefillGemmOpts): string {
  const { N, M } = o;
  const { units: upr, per } = prefillGemmCheck(o, "prefillGemmQ4KSplitKIdotWgsl");
  return `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<storage, read> xsc: array<f32>;
const SBPR = ${upr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
const WORDS = 36u;
var<workgroup> xs: array<u32, ${M * 64}>;
var<workgroup> xss: array<f32, ${M * 8}>;
var<workgroup> xsum: array<f32, ${M * 8}>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let sbEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var sb = s * PER;
  loop {
    if (sb >= sbEnd) { break; }
    for (var idx = t; idx < ${M * 64}u; idx = idx + 64u) {
      let m = idx / 64u;
      let jj = (idx % 64u) / 8u;
      let ii = idx % 8u;
      xs[idx] = xq[((m * SBPR + sb) * 8u + jj) * 8u + ii];
    }
    for (var idx = t; idx < ${M * 8}u; idx = idx + 64u) {
      let m = idx / 8u;
      let jj = idx % 8u;
      xss[idx] = xsc[(m * SBPR + sb) * 8u + jj];
    }
    workgroupBarrier();
    for (var idx = t; idx < ${M * 8}u; idx = idx + 64u) {
      var sq = 0i;
      for (var ii = 0u; ii < 8u; ii = ii + 1u) {
        sq = sq + dot4I8Packed(0x01010101u, xs[idx * 8u + ii]);
      }
      xsum[idx] = f32(sq) * xss[idx];
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      let wb = (r * SBPR + sb) * WORDS;
      let dm = unpack2x16float(blocks[wb]);
      for (var g = 0u; g < 4u; g = g + 1u) {
        let is = 2u * g;
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
        var lo: array<u32, 8>;
        var hi: array<u32, 8>;
        for (var ii = 0u; ii < 8u; ii = ii + 1u) {
          let word = blocks[wb + 4u + g * 8u + ii];
          lo[ii] = word & 0x0f0f0f0fu;
          hi[ii] = (word >> 4u) & 0x0f0f0f0fu;
        }
        let d1 = dm.x * f32(sc1); let min1 = dm.y * f32(mn1);
        let d2 = dm.x * f32(sc2); let min2 = dm.y * f32(mn2);
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          let bLo = m * 64u + is * 8u;
          let bHi = bLo + 8u;
          var i1 = 0i; var i2 = 0i;
          for (var ii = 0u; ii < 8u; ii = ii + 1u) {
            i1 = i1 + dot4I8Packed(lo[ii], xs[bLo + ii]);
            i2 = i2 + dot4I8Packed(hi[ii], xs[bHi + ii]);
          }
          acc[m] = acc[m]
            + d1 * f32(i1) * xss[m * 8u + is] - min1 * xsum[m * 8u + is]
            + d2 * f32(i2) * xss[m * 8u + is + 1u] - min2 * xsum[m * 8u + is + 1u];
        }
      }
    }
    workgroupBarrier();
    sb = sb + 1u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * VIA f32 q4_K — FALLBACK DICHIARATO. Il Q5_K f32 senza il piano del 5o bit,
 * con `qs` a byte 16 e il superblocco da 36 parole.
 *
 * Il tile e' di UN SOTTO-BLOCCO da 32 per volta (M x 32 f32 = 2.048 B a M=16),
 * non del superblocco intero: M x 256 f32 sarebbero 16.384 B a M=16, cioe'
 * l'INTERO minimo di spec WebGPU per un solo array.
 *
 * ATTENZIONE, e' un fatto della fase 0 e non un difetto del port: su
 * `[2048,512]` questa via misura 1,14x sulla legacy, cioe' SOTTO la regola di
 * stop dell'1,5x. Si porta perche' il contratto chiede che ogni via intera
 * abbia il suo fallback dichiarato, non perche' su quella shape paghi.
 */
export function prefillGemmQ4KSplitKWgsl(o: PrefillGemmOpts): string {
  const { N, M, K } = o;
  const { units: upr, per } = prefillGemmCheck(o, "prefillGemmQ4KSplitKWgsl");
  return `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
const SBPR = ${upr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
const K_DIM = ${K}u;
const WORDS = 36u;
var<workgroup> xs: array<f32, ${M * 32}>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let sbEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var sb = s * PER;
  loop {
    if (sb >= sbEnd) { break; }
    for (var is = 0u; is < 8u; is = is + 1u) {
      let g = is >> 1u;
      let hiHalf = (is & 1u) == 1u;
      let base = sb * 256u + is * 32u;
      for (var idx = t; idx < ${M * 32}u; idx = idx + 64u) {
        let m = idx / 32u;
        xs[idx] = x[m * K_DIM + base + (idx % 32u)];
      }
      workgroupBarrier();
      if (r < N_ROWS) {
        let wb = (r * SBPR + sb) * WORDS;
        let dm = unpack2x16float(blocks[wb]);
        var sc: u32; var mn: u32;
        if (is < 4u) {
          sc = sbyte(wb, 4u + is) & 63u;
          mn = sbyte(wb, 4u + is + 4u) & 63u;
        } else {
          sc = (sbyte(wb, 4u + is + 4u) & 0xfu) | ((sbyte(wb, 4u + is - 4u) >> 6u) << 4u);
          mn = (sbyte(wb, 4u + is + 4u) >> 4u) | ((sbyte(wb, 4u + is) >> 6u) << 4u);
        }
        let dsc = dm.x * f32(sc);
        let dmn = dm.y * f32(mn);
        var q: array<f32, 32>;
        for (var l = 0u; l < 32u; l = l + 1u) {
          let ql = sbyte(wb, 16u + g * 32u + l);
          if (hiHalf) { q[l] = f32(ql >> 4u); } else { q[l] = f32(ql & 0xfu); }
        }
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          var qx = 0.0; var sx = 0.0;
          for (var l = 0u; l < 32u; l = l + 1u) {
            let xv = xs[m * 32u + l];
            qx = qx + q[l] * xv; sx = sx + xv;
          }
          acc[m] = acc[m] + dsc * qx - dmn * sx;
        }
      }
      workgroupBarrier();
    }
    sb = sb + 1u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * VIA INTERA q6_K x q8_0 (`dot4I8Packed`) — la piu' diversa delle sei, per due
 * ragioni che il testo porta con se'.
 *
 * (1) I SOTTO-BLOCCHI DELLE SCALE SONO DA 16, non da 32: 16 scale int8 per 256
 *     pesi. Le attivazioni pero' sono quantizzate per 32, quindi le due meta' di
 *     un blocco condividono la scala di x e hanno scale di peso diverse —
 *     servono accumuli separati per meta', ed e' per questo che qui la somma
 *     delle attivazioni (`xh`) e' per MEZZO sotto-blocco (M x 16 f32).
 *
 * (2) I PESI SONO CENTRATI SU -32, e sottrarre 32 in forma impacchettata
 *     traboccherebbe fra i byte. Non si sottrae: si tiene `q` senza segno in
 *     [0,63] — che sta in i8 positivo — e l'offset esce dal prodotto scalare,
 *     `Sigma (q-32)x = Sigma q*x - 32*Sigma x`. E' lo stesso Sigma(x) che le
 *     altre famiglie usano per il loro termine costante: un solo meccanismo.
 *
 * Layout (`dequantQ6_K` in quant.ts, e `gemvQ6KWgsl`): 53 parole sul device —
 * 210 B nel file, **212 col pad alla parola** — `ql` a byte 0, `qh` a byte 128,
 * le 16 scale int8 a byte 192, `d` f16 nella parola 52.
 *
 * NIENTE `enable packed_4x8_integer_dot_product;`: e' una LANGUAGE FEATURE.
 */
export function prefillGemmQ6KSplitKIdotWgsl(o: PrefillGemmOpts): string {
  const { N, M } = o;
  const { units: upr, per } = prefillGemmCheck(o, "prefillGemmQ6KSplitKIdotWgsl");
  return `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<storage, read> xsc: array<f32>;
const SBPR = ${upr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
const WORDS = 53u;
var<workgroup> xs: array<u32, ${M * 64}>;
var<workgroup> xss: array<f32, ${M * 8}>;
var<workgroup> xh: array<f32, ${M * 16}>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
fn sint8(base: u32, i: u32) -> f32 {
  return f32((i32(sbyte(base, i)) << 24u) >> 24u);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let sbEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var sb = s * PER;
  loop {
    if (sb >= sbEnd) { break; }
    for (var idx = t; idx < ${M * 64}u; idx = idx + 64u) {
      let m = idx / 64u;
      let jj = (idx % 64u) / 8u;
      let ii = idx % 8u;
      xs[idx] = xq[((m * SBPR + sb) * 8u + jj) * 8u + ii];
    }
    for (var idx = t; idx < ${M * 8}u; idx = idx + 64u) {
      let m = idx / 8u;
      let jj = idx % 8u;
      xss[idx] = xsc[(m * SBPR + sb) * 8u + jj];
    }
    workgroupBarrier();
    // somma delle attivazioni per MEZZO sotto-blocco (16 elementi = 4 parole):
    // e' la granularita' delle scale del Q6_K
    for (var idx = t; idx < ${M * 16}u; idx = idx + 64u) {
      let half = idx % 2u;
      var sq = 0i;
      for (var ii = 0u; ii < 4u; ii = ii + 1u) {
        sq = sq + dot4I8Packed(0x01010101u, xs[(idx / 2u) * 8u + half * 4u + ii]);
      }
      xh[idx] = f32(sq);
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      let wb = (r * SBPR + sb) * WORDS;
      let d = unpack2x16float(blocks[wb + 52u]).x;
      for (var n = 0u; n < 2u; n = n + 1u) {
        let scO = 192u + n * 8u;
        for (var c = 0u; c < 4u; c = c + 1u) {
          let blk = n * 4u + c;                  // sotto-blocco da 32 dentro il superblocco
          let sh = 2u * c;                       // bit del piano alto per questo quarto
          var w8: array<u32, 8>;
          for (var ii = 0u; ii < 8u; ii = ii + 1u) {
            // c pari  -> ql "A" (byte l), c dispari -> ql "B" (byte l+32)
            // c < 2   -> nibble basso,    c >= 2    -> nibble alto
            let qlw = blocks[wb + n * 16u + (c & 1u) * 8u + ii];
            let qhw = blocks[wb + 32u + n * 8u + ii];
            let nib = select(qlw & 0x0f0f0f0fu, (qlw >> 4u) & 0x0f0f0f0fu, c >= 2u);
            w8[ii] = nib | (((qhw >> sh) & 0x03030303u) << 4u);
          }
          for (var m = 0u; m < M_ROWS; m = m + 1u) {
            let b0 = m * 64u + blk * 8u;
            var iA = 0i; var iB = 0i;
            for (var ii = 0u; ii < 4u; ii = ii + 1u) {
              iA = iA + dot4I8Packed(w8[ii], xs[b0 + ii]);
              iB = iB + dot4I8Packed(w8[ii + 4u], xs[b0 + 4u + ii]);
            }
            // i pesi restano senza segno: l'offset -32 esce dal prodotto
            // scalare come -32*Sigma(x), per ciascuna meta'
            let scA = sint8(wb, scO + 2u * c);
            let scB = sint8(wb, scO + 1u + 2u * c);
            let xsm = xss[m * 8u + blk];
            acc[m] = acc[m] + d * xsm * (
                scA * (f32(iA) - 32.0 * xh[(m * 8u + blk) * 2u])
              + scB * (f32(iB) - 32.0 * xh[(m * 8u + blk) * 2u + 1u]));
          }
        }
      }
    }
    workgroupBarrier();
    sb = sb + 1u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * VIA f32 q6_K — FALLBACK DICHIARATO. Qui l'offset -32 si sottrae DIRETTAMENTE
 * dal peso: il traboccamento fra byte esiste solo nella forma impacchettata, e
 * portarsi dietro il giro di `Sigma(x)` senza motivo sarebbe riscrivere.
 * Le due scale del sotto-blocco si scelgono come in produzione (`gemvQ6KWgsl`).
 */
export function prefillGemmQ6KSplitKWgsl(o: PrefillGemmOpts): string {
  const { N, M, K } = o;
  const { units: upr, per } = prefillGemmCheck(o, "prefillGemmQ6KSplitKWgsl");
  return `@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
const SBPR = ${upr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
const K_DIM = ${K}u;
const WORDS = 53u;
var<workgroup> xs: array<f32, ${M * 32}>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
fn sint8(base: u32, i: u32) -> f32 {
  return f32((i32(sbyte(base, i)) << 24u) >> 24u);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let sbEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var sb = s * PER;
  loop {
    if (sb >= sbEnd) { break; }
    for (var blk = 0u; blk < 8u; blk = blk + 1u) {
      let n = blk >> 2u;
      let c = blk & 3u;
      let base = sb * 256u + blk * 32u;
      for (var idx = t; idx < ${M * 32}u; idx = idx + 64u) {
        let m = idx / 32u;
        xs[idx] = x[m * K_DIM + base + (idx % 32u)];
      }
      workgroupBarrier();
      if (r < N_ROWS) {
        let wb = (r * SBPR + sb) * WORDS;
        let d = unpack2x16float(blocks[wb + 52u]).x;
        let scO = 192u + n * 8u;
        let qlO = n * 64u + (c & 1u) * 32u;
        let qhO = 128u + n * 32u;
        let sh = 2u * c;
        var q: array<f32, 32>;
        for (var l = 0u; l < 32u; l = l + 1u) {
          let ql = sbyte(wb, qlO + l);
          let qh = sbyte(wb, qhO + l);
          let nib = select(ql & 0xfu, ql >> 4u, c >= 2u);
          q[l] = f32(nib | (((qh >> sh) & 3u) << 4u)) - 32.0;
        }
        let scA = sint8(wb, scO + 2u * c);
        let scB = sint8(wb, scO + 1u + 2u * c);
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          var a = 0.0; var b = 0.0;
          for (var l = 0u; l < 16u; l = l + 1u) {
            a = a + q[l] * xs[m * 32u + l];
            b = b + q[l + 16u] * xs[m * 32u + l + 16u];
          }
          acc[m] = acc[m] + d * (scA * a + scB * b);
        }
      }
      workgroupBarrier();
    }
    sb = sb + 1u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * VIA INTERA q8_0 x q8_0 (`dot4I8Packed`) — la piu' semplice delle sei: i pesi
 * SONO gia' i8.
 *
 * Niente unpack e niente termine costante — il ciclo interno e' `dot4I8Packed`
 * nudo su otto parole, e la scala del peso per la scala di x si applica una
 * volta per blocco. Passo di due blocchi per giro, come il q4_0 e il q4_1.
 *
 * Layout `repackQ8_0` (quant.ts): 8 parole di i8 per blocco da 32, scale a DUE
 * f16 per parola come il q4_0 — quindi `scales[gb >> 1u][gb & 1u]`, che sul
 * q4_1 sarebbe invece l'errore di formato.
 *
 * NIENTE `enable packed_4x8_integer_dot_product;`: e' una LANGUAGE FEATURE.
 */
export function prefillGemmQ80SplitKIdotWgsl(o: PrefillGemmOpts): string {
  const { N, M } = o;
  const { units: bpr, per } = prefillGemmCheck(o, "prefillGemmQ80SplitKIdotWgsl");
  return `@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> xsc: array<f32>;
const BPR = ${bpr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
var<workgroup> xs: array<u32, ${M * 16}>;
var<workgroup> xss: array<f32, ${M * 2}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let bEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var b0 = s * PER;
  loop {
    if (b0 >= bEnd) { break; }
    for (var idx = t; idx < ${M * 16}u; idx = idx + 64u) {
      let m = idx / 16u;
      let qi = idx % 16u;
      xs[idx] = xq[(m * BPR + b0) * 8u + qi];
    }
    for (var idx = t; idx < ${M * 2}u; idx = idx + 64u) {
      let m = idx / 2u;
      xss[idx] = xsc[m * BPR + b0 + (idx % 2u)];
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      for (var bi = 0u; bi < 2u; bi = bi + 1u) {
        let gb = r * BPR + b0 + bi;
        let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];
        var w8: array<u32, 8>;
        for (var ii = 0u; ii < 8u; ii = ii + 1u) { w8[ii] = qs[gb * 8u + ii]; }
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          let xo = m * 16u + bi * 8u;
          var idot = 0i;
          for (var ii = 0u; ii < 8u; ii = ii + 1u) {
            idot = idot + dot4I8Packed(w8[ii], xs[xo + ii]);
          }
          acc[m] = acc[m] + sc * f32(idot) * xss[m * 2u + bi];
        }
      }
    }
    workgroupBarrier();
    b0 = b0 + 2u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * VIA f32 q8_0 — FALLBACK DICHIARATO: i pesi si convertono da i8, nient'altro.
 *
 * Le attivazioni si leggono DENSE row-major M x K (binding 2), non come `x4`
 * vec4 alla maniera del q4_0: il tile e' di due blocchi da 32 per giro, cioe'
 * M x 64 f32 (4.096 B a M=16).
 */
export function prefillGemmQ80SplitKWgsl(o: PrefillGemmOpts): string {
  const { N, M, K } = o;
  const { units: bpr, per } = prefillGemmCheck(o, "prefillGemmQ80SplitKWgsl");
  return `@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
const BPR = ${bpr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const M_ROWS = ${M}u;
const K_DIM = ${K}u;
var<workgroup> xs: array<f32, ${M * 64}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = wid.x * 64u + t;
  let s = wid.y;
  let bEnd = s * PER + PER;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var b0 = s * PER;
  loop {
    if (b0 >= bEnd) { break; }
    for (var idx = t; idx < ${M * 64}u; idx = idx + 64u) {
      let m = idx / 64u;
      xs[idx] = x[m * K_DIM + b0 * 32u + (idx % 64u)];
    }
    workgroupBarrier();
    if (r < N_ROWS) {
      for (var bi = 0u; bi < 2u; bi = bi + 1u) {
        let gb = r * BPR + b0 + bi;
        let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];
        // i 32 pesi del blocco in registri: le M righe li riusano
        var q: array<f32, 32>;
        for (var ii = 0u; ii < 8u; ii = ii + 1u) {
          let w = qs[gb * 8u + ii];
          for (var by = 0u; by < 4u; by = by + 1u) {
            // il byte "by" della parola, come i8 con segno
            q[ii * 4u + by] = f32((i32(w << ((3u - by) * 8u)) >> 24u));
          }
        }
        for (var m = 0u; m < M_ROWS; m = m + 1u) {
          var qx = 0.0;
          for (var l = 0u; l < 32u; l = l + 1u) {
            qx = qx + q[l] * xs[m * 64u + bi * 32u + l];
          }
          acc[m] = acc[m] + sc * qx;
        }
      }
    }
    workgroupBarrier();
    b0 = b0 + 2u;
  }
  if (r < N_ROWS) {
    for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }
  }
}`;
}

/**
 * QUANTIZZAZIONE DELLE ATTIVAZIONI a i8 per blocco da 32 — il termine che la
 * via intera AGGIUNGE, e che in it.5 era dichiarato «fuori misura» invece di
 * essere misurato.
 *
 * Un thread per blocco: amax dei 32 valori, scala `amax/127`, otto u32 di byte
 * con segno. Il layout d'uscita e' quello naturale del blocco — u32 0..3 =
 * elementi 0..15, u32 4..7 = elementi 16..31 — cioe' l'ordine in cui il
 * moltiplicatore appaia i nibble basso e alto di `qs4`: nessun rimescolamento.
 */
export function prefillQuantXQ8Wgsl(o: { K: number; M: number }): string {
  const { K, M } = o;
  if (K % 64 !== 0) {
    throw new Error(
      `prefillQuantXQ8Wgsl: K=${K} non multiplo di 64 (il moltiplicatore che consuma questi blocchi avanza a BK = 2)`);
  }
  const bpr = K / 32;
  return `@group(0) @binding(0) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
const BLOCKS = ${M * bpr}u;
const K4 = ${K / 4}u;
const BPR = ${bpr}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= BLOCKS) { return; }
  let m = b / BPR;
  let blk = b % BPR;
  let base = m * K4 + blk * 8u;          // 8 vec4<f32> = 32 valori
  var amax = 0.0;
  for (var i = 0u; i < 8u; i = i + 1u) {
    let v = abs(x4[base + i]);
    amax = max(amax, max(max(v.x, v.y), max(v.z, v.w)));
  }
  let sc = amax / 127.0;
  let inv = select(0.0, 1.0 / sc, sc > 0.0);
  xsc[b] = sc;
  for (var i = 0u; i < 8u; i = i + 1u) {
    let q = clamp(round(x4[base + i] * inv), vec4<f32>(-127.0), vec4<f32>(127.0));
    let u = vec4<u32>(u32(i32(q.x) & 255), u32(i32(q.y) & 255), u32(i32(q.z) & 255), u32(i32(q.w) & 255));
    xq[b * 8u + i] = u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);
  }
}`;
}

/** Somma le S fette dello split-K: un thread per uscita (m, r). */
export function prefillSplitKCombineWgsl(o: { N: number; M: number; splits: number }): string {
  const { N, M, splits } = o;
  return `@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
const TOTAL = ${M * N}u;
const S = ${splits}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= TOTAL) { return; }
  var v = 0.0;
  for (var s = 0u; s < S; s = s + 1u) { v = v + part[s * TOTAL + i]; }
  y[i] = v;
}`;
}

/** Griglia del moltiplicatore: 64 righe di uscita per workgroup in X, una fetta di K per Y. */
export function prefillGemmGrid(o: PrefillGemmOpts): [number, number, number] {
  prefillGemmCheck(o, "prefillGemmGrid");
  return [Math.ceil(o.N / PREFILL_GEMM_ROWS_PER_WG), o.splits, 1];
}

/** Griglia della quantizzazione: un thread per blocco da 32, workgroup da 64. */
export function prefillQuantXGrid(o: { K: number; M: number }): [number, number, number] {
  if (o.K % 64 !== 0) {
    throw new Error(`prefillQuantXGrid: K=${o.K} non multiplo di 64`);
  }
  return [Math.ceil((o.M * (o.K / 32)) / 64), 1, 1];
}

/** Griglia della combine: un thread per uscita (m, r). */
export function prefillCombineGrid(o: { N: number; M: number }): [number, number, number] {
  return [Math.ceil((o.M * o.N) / 64), 1, 1];
}

/**
 * Fette di K. 4 e' il valore MISURATO in riga 1 sulle due shape del 4B
 * (K2560xN9216 e K9216xN2560, M=16): serve a compensare l'occupancy, perche' a
 * N=9216 il moltiplicatore lancia 144 workgroup e su 128 SM e' poco piu' di uno
 * per SM.
 *
 * QUANDO LE 4 FETTE NON DIVIDONO si ripiega su 1, cioe' NESSUNO split-K — la
 * forma da cui lo split-K partiva, mai piu' veloce del misurato e sempre
 * corretta. Non 2 e non 3: quelli sarebbero numeri inventati, e qui si rifiuta
 * di inventare. Il ripiego serve a una shape gia' in albero: QWEN25_05B ha
 * dModel = 896, cioe' 28 blocchi, che in 4 fette da BK=2 non ci stanno; con il
 * rifiuto secco quel modello resterebbe senza prefill veloce per una divisione,
 * non per una misura.
 *
 * SUL q5_K LA SCALA E' UN'ALTRA, ed e' quella misurata al banco
 * (`kquantSplitsFor` in ttKQuant.ts): 4 fette se le unita' per riga si dividono
 * per 4, 2 se per 2, altrimenti 1. Il 2 non e' un'eccezione arbitraria: sui
 * K-quant le unita' per riga sono otto volte meno numerose (K/256 invece di
 * K/32), quindi il ripiego secco a 1 lascerebbe senza split-K shape che il
 * banco ha misurato divise — la down degli expert del 35B ha K=512, cioe' DUE
 * superblocchi, e li' l'alternativa e' fra 2 fette e nessuna.
 *
 * IL TERZO PARAMETRO E' IL FORMATO, ed e' OPZIONALE col default `q4_0`: i
 * chiamanti scritti prima della riga 2 non cambiano una riga e non cambiano un
 * numero. La scala vera vive in `PREFILL_GEMM_SPEC[kind].splitsFor` — qui non
 * c'e' un secondo `if` per formato.
 */
export function prefillGemmSplitsFor(K: number, N: number, kind: PrefillGemmKind = "q4_0"): number {
  const spec = PREFILL_GEMM_SPEC[kind];
  if (K % spec.kUnit !== 0) {
    throw new Error(
      `prefillGemmSplitsFor: K=${K} non multiplo di ${spec.kUnit} (${spec.kUnitWhy})`);
  }
  // N non entra nella scelta: le 4 fette sono state misurate su ENTRAMBE le
  // shape di riga 1 (N=9216 e N=2560), quindi non c'e' una soglia su N che sia
  // stata misurata. Resta nella firma perche' e' li' che andra' se un domani la
  // si misura.
  void N;
  return spec.splitsFor(spec.unitsPerRow(K));
}

/** Taglia (in f32) del buffer dei parziali: part[(s*M + m)*N + r]. */
export function prefillPartialFloats(o: PrefillGemmOpts): number {
  prefillGemmCheck(o, "prefillPartialFloats");
  return o.splits * o.M * o.N;
}

/**
 * Workgroup storage del moltiplicatore, come formula chiusa accanto al kernel
 * che la consuma (stessa scelta di `attnDecodeWorkgroupStorageBytes`).
 *
 * Le due vie hanno fabbisogni DIVERSI e quale delle due giri si decide a
 * runtime sulla language feature. Percio' la via e' un ARGOMENTO: chi ha gia'
 * deciso di accendere la pipeline intera chiede il suo numero e non l'altro.
 * Senza argomento si risponde il PEGGIORE dei due, che e' il numero giusto per
 * chi negozia i limiti prima di sapere quale via avra'.
 *
 * q4_0 — intera 72·M B (1.152 a M=16), f32 256·M B (4.096 a M=16): il peggiore
 *        e' la via f32, che sarebbe 3,55x lo storage della via intera.
 * q5_K — intera 320·M B (5.120 a M=16), f32 128·M B (2.048 a M=16): qui il
 *        peggiore e' la via INTERA, il contrario del q4_0. E' il motivo per cui
 *        il `max` va CALCOLATO e non assunto.
 *
 * Le formule stanno in `PREFILL_GEMM_SPEC[kind].wgBytes`, una riga per formato:
 * un formato nuovo senza il suo fabbisogno non compila.
 *
 * Il test confronta entrambi con la scansione del testo WGSL vero, cosi' la
 * formula non puo' derivare dal kernel.
 */
export function prefillGemmWorkgroupStorageBytes(
  o: PrefillGemmOpts,
  via?: PrefillGemmWgVia,
): number {
  prefillGemmCheck(o, "prefillGemmWorkgroupStorageBytes");
  const { wgBytes } = PREFILL_GEMM_SPEC[o.kind];
  if (via !== undefined) return wgBytes[via](o.M);
  return Math.max(wgBytes.idot(o.M), wgBytes.f32(o.M));
}
