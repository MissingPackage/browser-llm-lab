// Varianti GEMV q4_0 per la fase 0 di engine-kernel-decode.
//
// La BASELINE si importa da src/engine/kernels/wgsl.ts (`gemvQuantWgsl`, kind
// q4_0): forma attuale = kernel vero, non imitazione. Qui solo le varianti.
//
// Layout q4_0 (identico a quello che legge il kernel del motore): un blocco =
// 32 pesi in 16 byte di nibble + una scala f16. `qs` e' array<u32> (4 word per
// blocco), `scales` e' array<u32> con DUE f16 impacchettate (unpack2x16float).
// Nel blocco, il byte j porta il nibble basso dell'elemento j e il nibble alto
// dell'elemento j+16.

/** Larghezza X della griglia 2D dei gemv: identica al motore (limite 65535/dim). */
export const GEMV_GRID_X = 32768;

export function gemvGrid(rows: number): [number, number] {
  return rows <= GEMV_GRID_X ? [rows, 1] : [GEMV_GRID_X, Math.ceil(rows / GEMV_GRID_X)];
}

const HEAD = (K: number, N: number, extraEnable: string) => `${extraEnable}
@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
const BLOCKS_PER_ROW = ${K / 32}u;
const N_ROWS = ${N}u;`;

// Corpo comune: dot() su vec4 dequantizzate, UNA load vec4<u32> (16 B) per blocco.
const BLOCK_BODY = `
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

/** VARIANTE `vec4`: una riga per workgroup (come la base), load vec4 + dot(). */
export function gemvVec4Wgsl(K: number, N: number): string {
  return `${HEAD(K, N, "")}
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= N_ROWS) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var b = t; b < BLOCKS_PER_ROW; b = b + 64u) {${BLOCK_BODY}
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

/** VARIANTE `vec4-sg`: come vec4 ma la riduzione passa da `subgroupAdd`. */
export function gemvVec4SubgroupWgsl(K: number, N: number): string {
  return `${HEAD(K, N, "enable subgroups;\n")}
var<workgroup> partial: array<f32, 8>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgSize: u32, @builtin(subgroup_invocation_id) sgId: u32) {
  let r = wid.x + wid.y * ${GEMV_GRID_X}u;
  if (r >= N_ROWS) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var b = t; b < BLOCKS_PER_ROW; b = b + 64u) {${BLOCK_BODY}
  }
  let sgSum = subgroupAdd(acc);
  let nSg = 64u / sgSize;
  if (sgId == 0u) { partial[t / sgSize] = sgSum; }
  workgroupBarrier();
  if (t == 0u) {
    var tot = 0.0;
    for (var i = 0u; i < nSg; i = i + 1u) { tot = tot + partial[i]; }
    y[r] = tot;
  }
}`;
}

/** VARIANTE `vec4-rows4`: 4 righe per workgroup, 16 lane per riga. */
export function gemvVec4Rows4Wgsl(K: number, N: number): string {
  return `${HEAD(K, N, "")}
var<workgroup> partial: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let lane = t & 15u;
  let sub = t >> 4u;
  let r = (wid.x + wid.y * ${GEMV_GRID_X}u) * 4u + sub;
  var acc = 0.0;
  if (r < N_ROWS) {
    for (var b = lane; b < BLOCKS_PER_ROW; b = b + 16u) {${BLOCK_BODY}
    }
  }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 8u;
  while (stride > 0u) {
    if (lane < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lane == 0u && r < N_ROWS) { y[r] = partial[sub << 4u]; }
}`;
}

/**
 * VARIANTE `vec4-rows2-sg`: 2 righe per workgroup, UNA riga per subgroup — la
 * riduzione di riga e' un solo `subgroupAdd`, senza barriere. Valida solo se il
 * subgroup misura 32: altrimenti la cella si salta con motivo.
 */
export function gemvVec4Rows2SgWgsl(K: number, N: number): string {
  return `${HEAD(K, N, "enable subgroups;\n")}
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
    for (var b = sgId; b < BLOCKS_PER_ROW; b = b + sgSize) {${BLOCK_BODY}
    }
  }
  let tot = subgroupAdd(acc);
  if (sgId == 0u && r < N_ROWS) { y[r] = tot; }
}`;
}

// ---------------------------------------------------------------------------
// SONDE (P1): shader minimi che USANO la feature. Presente/assente si dichiara
// misurando: compila? e produce il risultato atteso?
// ---------------------------------------------------------------------------

/** somma 0+1+...+63 = 2016 via subgroupAdd, piu' il combine fra subgroup. */
export const SUBGROUP_PROBE_WGSL = `enable subgroups;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
var<workgroup> partial: array<f32, 16>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_size) sgSize: u32, @builtin(subgroup_invocation_id) sgId: u32) {
  let t = lid.x;
  let s = subgroupAdd(f32(t));
  if (sgId == 0u) { partial[t / sgSize] = s; }
  workgroupBarrier();
  if (t == 0u) {
    var tot = 0.0;
    for (var i = 0u; i < 64u / sgSize; i = i + 1u) { tot = tot + partial[i]; }
    out[0] = tot;
    out[1] = f32(sgSize);
  }
}`;

/** dot4I8Packed su valori noti: (1,2,3,4)·(5,6,7,8) = 5+12+21+32 = 70. */
export const DOT4I8_PROBE_WGSL = `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let a = 0x04030201u;
  let b = 0x08070605u;
  out[0] = f32(dot4I8Packed(a, b));
}`;

/** shader-f16 minimo: se la feature manca, la compilazione deve fallire. */
export const F16_PROBE_WGSL = `enable f16;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let a: f16 = 1.5h;
  let b: f16 = 2.0h;
  out[0] = f32(a * b);
}`;
