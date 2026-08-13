// Varianti di MOLTIPLICATORE MULTI-RIGA per la riga 1 di engine-ttft.
//
// Pre-registrazione: docs/deep-dive/ttft-riga1-prereg-2026-08-13.md
//
// La FORMA ATTUALE non vive qui: si importa `gemvQuantWgsl` (con `batch: true`)
// da src/engine/kernels/wgsl.ts, esattamente come ha fatto la fase 0 del goal
// precedente. Qui vivono solo (a) la sonda del picco di calcolo fp32 e (b) le
// tre forme candidate: pesi in registri, pesi in shared, split-K.
//
// Layout q4_0 (identico a quello che legge il motore): un blocco = 32 pesi in
// 16 byte di nibble + una scala f16. `qs` e' array<vec4<u32>> (UNA vec4 per
// blocco), `scales` e' array<u32> con DUE f16 impacchettate. Nel blocco, il byte
// j porta il nibble basso dell'elemento j e il nibble alto dell'elemento j+16 —
// da cui le due `dot()` sfalsate di 4 vec4.

/** Larghezza X della griglia 2D: identica al motore (limite 65535/dim). */
export const GEMM_GRID_X = 32768;

export function gemmGrid(wgs: number): [number, number] {
  return wgs <= GEMM_GRID_X ? [wgs, 1] : [GEMM_GRID_X, Math.ceil(wgs / GEMM_GRID_X)];
}

// ---------------------------------------------------------------------------
// SONDA (a): PICCO DI CALCOLO fp32 — GEMM densa, tiling classico shared+registri
// ---------------------------------------------------------------------------
/**
 * GEMM densa fp32 C = A·B, A M×K, B K×N, tutte row-major. Tile 64×64×8,
 * `workgroup_size` 256 (16×16 thread, 4×4 uscite per thread): la forma che la
 * pre-registrazione dichiara al punto P1. Workgroup storage: due tile da 64×8
 * f32 = 4.096 B, COSTANTE nelle shape.
 *
 * Vincoli: M e N multipli di 64, K multiplo di 8.
 */
export function gemmDenseF32Wgsl(M: number, K: number, N: number): string {
  if (M % 64 !== 0 || N % 64 !== 0 || K % 8 !== 0) {
    throw new Error(`gemmDenseF32Wgsl: shape ${M}x${K}x${N} non allineata (M,N mult. 64; K mult. 8)`);
  }
  return `@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
const K_DIM = ${K}u;
const N_DIM = ${N}u;
var<workgroup> As: array<f32, 512>;
var<workgroup> Bs: array<f32, 512>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let tx = t & 15u;
  let ty = t >> 4u;
  let row0 = wid.y * 64u;
  let col0 = wid.x * 64u;
  // WGSL: un accumulatore dichiarato dentro un loop NON viene ri-azzerato dal
  // compilatore — qui e' fuori dal loop e azzerato esplicitamente.
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }
  var k0 = 0u;
  loop {
    if (k0 >= K_DIM) { break; }
    for (var s = 0u; s < 2u; s = s + 1u) {
      let idx = t + s * 256u;
      let mm = idx >> 3u;
      let kk = idx & 7u;
      As[kk * 64u + mm] = A[(row0 + mm) * K_DIM + k0 + kk];
    }
    for (var s = 0u; s < 2u; s = s + 1u) {
      let idx = t + s * 256u;
      let kk = idx >> 6u;
      let nn = idx & 63u;
      Bs[kk * 64u + nn] = B[(k0 + kk) * N_DIM + col0 + nn];
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 8u; kk = kk + 1u) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i = i + 1u) { a[i] = As[kk * 64u + ty * 4u + i]; }
      for (var j = 0u; j < 4u; j = j + 1u) { b[j] = Bs[kk * 64u + tx * 4u + j]; }
      for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
          acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
        }
      }
    }
    workgroupBarrier();
    k0 = k0 + 8u;
  }
  for (var i = 0u; i < 4u; i = i + 1u) {
    for (var j = 0u; j < 4u; j = j + 1u) {
      C[(row0 + ty * 4u + i) * N_DIM + col0 + tx * 4u + j] = acc[i * 4u + j];
    }
  }
}`;
}

/** Griglia della GEMM densa: (N/64, M/64). */
export function gemmDenseGrid(M: number, N: number): [number, number] {
  return [N / 64, M / 64];
}

// ---------------------------------------------------------------------------
// Corpo comune delle varianti q4_0: dequantizza UN blocco in registri.
// ---------------------------------------------------------------------------
const DEQ_BLOCK = `
    var lo: array<vec4<f32>, 4>;
    var hi: array<vec4<f32>, 4>;
    for (var wi = 0u; wi < 4u; wi = wi + 1u) {
      let by = (vec4<u32>(w[wi]) >> vec4<u32>(0u, 8u, 16u, 24u));
      lo[wi] = vec4<f32>(by & vec4<u32>(15u)) - vec4<f32>(8.0);
      hi[wi] = vec4<f32>((by >> vec4<u32>(4u)) & vec4<u32>(15u)) - vec4<f32>(8.0);
    }`;

const GEMM_BINDINGS = `@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;`;

export interface MultiRowOpts {
  K: number;
  N: number;
  /** righe del chunk di prefill trattate insieme */
  M: number;
}

/**
 * VARIANTE `regs` — PESI IN REGISTRI, ATTIVAZIONI IN SHARED (la (ii) di P3).
 *
 * 64 thread per workgroup, UNA riga di output per thread (64 righe/wg). Il loop
 * su K avanza a passi di BK = 2 blocchi (64 elementi): il tile di attivazioni
 * M×64 va in workgroup memory (M×16 vec4 = M·256 B — 4.096 B a M=16), i pesi
 * NON ci vanno mai. Ogni blocco di peso si legge UNA volta e serve tutte le M
 * righe del chunk: e' il riuso che la forma attuale non ha per costruzione.
 */
export function gemmQ4MultiRowRegsWgsl(o: MultiRowOpts): string {
  const { K, N, M } = o;
  if (K % 64 !== 0) throw new Error("regs: K non multiplo di 64 (BK = 2 blocchi)");
  const bpr = K / 32;
  const k4 = K / 4;
  return `${GEMM_BINDINGS}
const BPR = ${bpr}u;
const N_ROWS = ${N}u;
const K4 = ${k4}u;
const M_ROWS = ${M}u;
var<workgroup> xs: array<vec4<f32>, ${M * 16}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let r = (wid.x + wid.y * ${GEMM_GRID_X}u) * 64u + t;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var b0 = 0u;
  loop {
    if (b0 >= BPR) { break; }
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
        let w = qs4[gb];${DEQ_BLOCK}
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
    for (var m = 0u; m < M_ROWS; m = m + 1u) { y[m * N_ROWS + r] = acc[m]; }
  }
}`;
}

export function gemmQ4MultiRowRegsGrid(o: MultiRowOpts): [number, number] {
  return gemmGrid(Math.ceil(o.N / 64));
}

/**
 * VARIANTE `shared` — PESI DEQUANTIZZATI IN WORKGROUP MEMORY (la (iii) di P3).
 *
 * 64 thread, 16 righe di output per workgroup, quattro partizioni di M: il
 * thread t serve la riga t%16 e le righe del chunk m ≡ t/16 (mod 4). I pesi del
 * passo (16 righe × 64 elementi = 4.096 B) si dequantizzano UNA volta in shared
 * e li leggono le quattro partizioni; le attivazioni M×64 stanno accanto
 * (M·256 B). E' la forma che paga shared proporzionale ai PESI, ed e' quella che
 * la riga 4 teme (docket item 1).
 */
export function gemmQ4MultiRowSharedWgsl(o: MultiRowOpts): string {
  const { K, N, M } = o;
  if (K % 64 !== 0) throw new Error("shared: K non multiplo di 64");
  if (M % 4 !== 0 && M !== 1) throw new Error("shared: M dev'essere 1 o multiplo di 4");
  const bpr = K / 32;
  const k4 = K / 4;
  const parts = M === 1 ? 1 : 4;
  return `${GEMM_BINDINGS}
const BPR = ${bpr}u;
const N_ROWS = ${N}u;
const K4 = ${k4}u;
const M_ROWS = ${M}u;
const PARTS = ${parts}u;
// 16 righe x 16 vec4 (64 elementi) di pesi dequantizzati
var<workgroup> ws: array<vec4<f32>, 256>;
var<workgroup> wsc: array<f32, 32>;
var<workgroup> xs: array<vec4<f32>, ${M * 16}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let row = t & 15u;
  let part = t >> 4u;
  let r = (wid.x + wid.y * ${GEMM_GRID_X}u) * 16u + row;
  var acc: array<f32, ${M}>;
  for (var m = 0u; m < M_ROWS; m = m + 1u) { acc[m] = 0.0; }
  var b0 = 0u;
  loop {
    if (b0 >= BPR) { break; }
    // stage pesi: 32 thread (16 righe x 2 blocchi), uno per blocco
    if (t < 32u) {
      let rr = t >> 1u;
      let bi = t & 1u;
      let gr = (wid.x + wid.y * ${GEMM_GRID_X}u) * 16u + rr;
      if (gr < N_ROWS) {
        let gb = gr * BPR + b0 + bi;
        wsc[t] = unpack2x16float(scales[gb >> 1u])[gb & 1u];
        let w = qs4[gb];${DEQ_BLOCK}
        for (var wi = 0u; wi < 4u; wi = wi + 1u) {
          ws[rr * 16u + bi * 8u + wi] = lo[wi];
          ws[rr * 16u + bi * 8u + 4u + wi] = hi[wi];
        }
      } else {
        wsc[t] = 0.0;
        for (var wi = 0u; wi < 8u; wi = wi + 1u) { ws[rr * 16u + bi * 8u + wi] = vec4<f32>(0.0); }
      }
    }
    for (var idx = t; idx < ${M * 16}u; idx = idx + 64u) {
      let m = idx / 16u;
      let qi = idx % 16u;
      xs[idx] = x4[m * K4 + b0 * 8u + qi];
    }
    workgroupBarrier();
    for (var m = part; m < M_ROWS; m = m + PARTS) {
      var bd0 = 0.0;
      var bd1 = 0.0;
      for (var wi = 0u; wi < 8u; wi = wi + 1u) {
        bd0 = bd0 + dot(ws[row * 16u + wi], xs[m * 16u + wi]);
        bd1 = bd1 + dot(ws[row * 16u + 8u + wi], xs[m * 16u + 8u + wi]);
      }
      acc[m] = acc[m] + wsc[row * 2u] * bd0 + wsc[row * 2u + 1u] * bd1;
    }
    workgroupBarrier();
    b0 = b0 + 2u;
  }
  if (r < N_ROWS) {
    for (var m = part; m < M_ROWS; m = m + PARTS) { y[m * N_ROWS + r] = acc[m]; }
  }
}`;
}

export function gemmQ4MultiRowSharedGrid(o: MultiRowOpts): [number, number] {
  return gemmGrid(Math.ceil(o.N / 16));
}

/**
 * VARIANTE `splitk` — `regs` con K spezzato su S workgroup (la (iv) di P3).
 *
 * Stessa mappatura di `regs` (64 righe per workgroup, pesi in registri,
 * attivazioni in shared), ma il contesto K e' diviso in S fette lungo `wid.y` e
 * ogni fetta scrive il proprio parziale in `part[(s·M + m)·N + r]`. Un secondo
 * dispatch (`splitKCombineWgsl`) somma le S fette. Serve a compensare
 * l'occupancy: a N = 9216 la forma `regs` lancia 144 workgroup, che su 128 SM
 * e' poco piu' di uno per SM.
 */
export function gemmQ4MultiRowSplitKWgsl(o: MultiRowOpts & { splits: number }): string {
  const { K, N, M, splits } = o;
  const bpr = K / 32;
  if (bpr % (splits * 2) !== 0) throw new Error(`splitk: ${bpr} blocchi non divisibili in ${splits} fette da BK=2`);
  const per = bpr / splits;
  const k4 = K / 4;
  return `${GEMM_BINDINGS.replace(
    "@group(0) @binding(3) var<storage, read_write> y: array<f32>;",
    "@group(0) @binding(3) var<storage, read_write> part: array<f32>;")}
const BPR = ${bpr}u;
const PER = ${per}u;
const N_ROWS = ${N}u;
const K4 = ${k4}u;
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
        let w = qs4[gb];${DEQ_BLOCK}
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

/** Somma le S fette dello split-K: un thread per uscita (m, r). */
export function splitKCombineWgsl(o: MultiRowOpts & { splits: number }): string {
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

// ---------------------------------------------------------------------------
// WORKGROUP STORAGE: LETTO dal testo WGSL generato, non dedotto a mano.
// ---------------------------------------------------------------------------
const SCALAR_BYTES: Record<string, { size: number; align: number }> = {
  f32: { size: 4, align: 4 },
  u32: { size: 4, align: 4 },
  i32: { size: 4, align: 4 },
  "vec2<f32>": { size: 8, align: 8 },
  "vec2<u32>": { size: 8, align: 8 },
  "vec3<f32>": { size: 12, align: 16 },
  "vec4<f32>": { size: 16, align: 16 },
  "vec4<u32>": { size: 16, align: 16 },
  "atomic<u32>": { size: 4, align: 4 },
};

/**
 * Somma i `var<workgroup>` dichiarati nel sorgente, con l'allineamento della
 * spec WGSL (§ address space layout). Non e' una stima: e' il testo che il
 * device ricevera'. Il numero viene poi VERIFICATO creando device con
 * `requiredLimits` espliciti (la spazzata di P6 vive nel driver).
 */
export function workgroupStorageBytes(wgsl: string): number {
  const re = /var<workgroup>\s+\w+\s*:\s*(array\s*<\s*([\w<>]+)\s*,\s*(\d+)\s*>|[\w<>]+)\s*;/g;
  let total = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(wgsl)) !== null) {
    const elem = m[2] ?? m[1];
    const count = m[3] ? Number(m[3]) : 1;
    const info = SCALAR_BYTES[elem.replace(/\s+/g, "")];
    if (!info) throw new Error(`workgroupStorageBytes: tipo non modellato "${elem}"`);
    total = Math.ceil(total / info.align) * info.align;
    total += info.size * count;
  }
  return total;
}
