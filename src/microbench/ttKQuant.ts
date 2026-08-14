// FASE 0 del goal engine-kquant: le forme MULTI-RIGA delle famiglie che oggi
// restano sul percorso vecchio (Q5_K, Q4_1 — e in it.2 Q4_K, Q6_K, Q8_0).
//
// PERCHE' UN FILE NUOVO E NON `ttGemm.ts`. Le forme di ttGemm sono q4_0-only
// per costruzione: leggono `qs4: array<vec4<u32>>` + `scales` a DUE f16 per
// parola, cioe' il layout che `repackQ4_0` produce. I K-quant hanno UN buffer
// solo (`blocks`, il superblocco GGUF copiato in parole) e il q4_1 ha una
// parola di scale PER blocco invece che ogni due. Tenere insieme geometrie
// diverse dentro lo stesso generatore e' il modo di scrivere un kernel che
// nessuna delle due famiglie legge davvero.
//
// IL BRACCIO DI PARAGONE E' IL KERNEL DI PRODUZIONE, IMPORTATO. `gemvQ5KWgsl`
// e `gemvQuantWgsl` con `batch: true` sono cio' che il motore emette OGGI su
// questi tensori (`q35gpumodel.ts` — ramo K-quant di `loadW`, e il fallback di
// `gemvB`). Riscriverli qui misurerebbe la mia copia, non il motore: e' la
// stessa regola per cui `ttAttn` importa il legacy invece di ricopiarlo.
//
// COSA MISURA QUESTA FASE (regola di stop del contratto, riga 1 di PHASES):
// se per una famiglia nessuna variante supera la legacy di >= 1,5x, quella
// famiglia si chiude col numero e NON si cabla.
import { gemvQ5KWgsl, gemvQuantWgsl, gemvGrid } from "../engine/kernels/wgsl";

/** Le famiglie che questa fase 0 misura. Q4_K/Q6_K/Q8_0 entrano in it.2. */
export type KQuantFamily = "q5_K" | "q4_1";

/**
 * Geometria di una famiglia, come la vede il KERNEL (non come sta nel file):
 * `unit` = pesi che condividono l'unita' indivisibile del formato, `words` =
 * parole a 32 bit che quell'unita' occupa nel buffer del device.
 *
 * E' la correzione C0-1 del goal, scritta come dato invece che come commento:
 * lo split-K del q4_0 conta in blocchi da 32, ma su un K-quant la fetta non
 * puo' tagliare un superblocco — le scale a 6 bit sono condivise dai suoi otto
 * sotto-blocchi, e una fetta che ne prendesse meta' dovrebbe rileggere
 * l'header comunque, falsificando proprio il conto dei byte su cui poggia il
 * goal.
 */
export const KQUANT_GEOM: Record<KQuantFamily, { unit: number; words: number; deviceBytes: number }> = {
  // Q5_K: superblocco da 256 pesi in 176 B = 44 parole.
  //   parola 0      : d, dmin (2 x f16)
  //   byte 4..15    : 12 byte di scale/min a 6 bit (get_scale_min_k4)
  //   byte 16..47   : qh, il piano del 5o bit (un bit per elemento per sotto-blocco)
  //   byte 48..175  : qs, un nibble per elemento
  q5_K: { unit: 256, words: 44, deviceBytes: 176 },
  // Q4_1: blocco da 32 pesi. `repackQ4_1` produce DUE buffer — qs (4 parole di
  // nibble) e scales (UNA parola per blocco: d nei 16 bit bassi, m negli alti).
  q4_1: { unit: 32, words: 4, deviceBytes: 20 },
};

/** Passo dello split-K, in unita' della famiglia. */
export const KQUANT_SPLIT_K = 4;

/**
 * Fette per questa (famiglia, K). Zero significa: lo split-K non si applica a
 * questa shape, e la cella NON va emessa — e' la correzione C0-4, cioe' la
 * cella degenere che altrimenti verrebbe eseguita per conformita' e committata
 * come se avesse misurato qualcosa (il down-proj degli expert del 35B ha
 * K=512 = due superblocchi per riga).
 */
export function kquantSplitsFor(family: KQuantFamily, K: number): number {
  const { unit } = KQUANT_GEOM[family];
  if (K % unit !== 0) throw new Error(`kquantSplitsFor: K=${K} non multiplo di ${unit} (${family})`);
  const upr = K / unit;                       // unita' per riga
  if (upr % KQUANT_SPLIT_K === 0) return KQUANT_SPLIT_K;
  if (upr % 2 === 0) return 2;
  return 1;
}

/** Griglia del moltiplicatore: 64 righe di uscita per workgroup, fette su y. */
export function kquantMultiRowGrid(o: { N: number; splits: number }): [number, number, number] {
  return [Math.ceil(o.N / 64), o.splits, 1];
}

export interface KQuantKernelOpts {
  family: KQuantFamily;
  K: number;
  N: number;
  M: number;
  splits: number;
}

function check(o: KQuantKernelOpts, who: string): { upr: number; per: number } {
  const { unit } = KQUANT_GEOM[o.family];
  if (o.K % unit !== 0) throw new Error(`${who}: K=${o.K} non multiplo di ${unit} (${o.family})`);
  const upr = o.K / unit;
  if (upr % o.splits !== 0) {
    throw new Error(`${who}: ${upr} unita' da ${unit} non divisibili in ${o.splits} fette`);
  }
  return { upr, per: upr / o.splits };
}

// ---------------------------------------------------------------------------
// Q5_K — VIA INTERA
// ---------------------------------------------------------------------------
/**
 * IL PUNTO DI QUESTA FORMA, in una riga: il piano del 5o bit si somma
 * IMPACCHETTATO.
 *
 * I 32 pesi di un sotto-blocco stanno in 32 nibble = 8 parole; il loro 5o bit
 * sta in `qh`, un bit per elemento nella posizione `is` del proprio byte.
 * Estrarlo elemento per elemento costerebbe 32 test; in forma impacchettata e'
 * `((qhw >> is) & 0x01010101) << 4`, cioe' "+16 a ogni byte che ce l'ha", e i
 * quattro pesi restano in una parola pronta per `dot4I8Packed`. Nessun peso
 * supera 31 (15 + 16), quindi stanno in i8 senza segno e la somma per byte non
 * trabocca.
 *
 * IL TERMINE CHE LA q4_0 NON HA (correzione C0-3 del goal): Q5_K e' `w =
 * d*sc_j*q - dmin*mn_j`, quindi al prodotto scalare va sottratto `dmin*mn_j`
 * per la SOMMA delle attivazioni del sotto-blocco. Quella somma non dipende
 * dalla riga di pesi: calcolarla nel ciclo di ogni thread la ricalcolerebbe 64
 * volte per workgroup, quindi si calcola UNA volta in memoria di gruppo
 * (`xsum`) con lo stesso `dot4I8Packed` contro `0x01010101`.
 *
 * ATTIVAZIONI: si riusano TALI E QUALI quelle che il prefill gia' produce
 * (`prefillQuantXQ8Wgsl`, blocchi da 32) — i sotto-blocchi K-quant sono
 * anch'essi da 32, otto per superblocco (correzione C0-2). Nessun secondo
 * quantizzatore, nessun dispatch in piu'.
 *
 * NIENTE `enable packed_4x8_integer_dot_product`: e' una language feature, non
 * un'estensione, e scriverla fa fallire la compilazione.
 */
export function kquantQ5KMultiRowSplitKIdotWgsl(o: KQuantKernelOpts): string {
  const { N, M } = o;
  const { upr, per } = check(o, "kquantQ5KMultiRowSplitKIdotWgsl");
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

// ---------------------------------------------------------------------------
// Q5_K — VIA f32 (fallback dichiarato)
// ---------------------------------------------------------------------------
/**
 * Stessa mappatura, attivazioni in virgola mobile. Il tile e' di UN GRUPPO da
 * 64 elementi per volta (M x 64 f32 = 4.096 B a M=16), non del superblocco
 * intero: M x 256 f32 sarebbero 16.384 B a M=16, cioe' l'INTERO minimo di spec
 * WebGPU per un solo array — e questa forma deve poter girare anche dove il
 * tetto e' quello garantito.
 *
 * Serve ai device senza `packed_4x8_integer_dot_product`. E' un fallback
 * dichiarato, non un'alternativa preferibile: quale sia il rapporto fra le due
 * su questa famiglia lo dice la misura di questa fase.
 */
export function kquantQ5KMultiRowSplitKWgsl(o: KQuantKernelOpts): string {
  const { N, M, K } = o;
  const { upr, per } = check(o, "kquantQ5KMultiRowSplitKWgsl");
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
// Q4_1 — VIA INTERA
// ---------------------------------------------------------------------------
/**
 * Il q4_1 e' `w = d*q + m` con `q` in [0,15] (niente offset -8: e' la
 * differenza col q4_0, che centra i nibble su zero e non ha termine costante).
 * Quindi:
 *   - i nibble vanno in i8 SENZA correzione, e restano positivi;
 *   - serve il termine `m * Sigma(x)` per blocco — lo stesso Sigma(x) del Q5_K,
 *     calcolato una volta per workgroup;
 *   - le scale sono UNA parola per blocco (d, m), non una ogni due blocchi come
 *     nel q4_0: `repackQ4_1` (quant.ts) mette d nei 16 bit bassi e m negli alti.
 *
 * Passo di due blocchi per giro, come la forma q4_0 misurata in riga 1 di
 * engine-ttft: e' il termine di paragone piu' onesto — stessa mappatura, stessa
 * occupancy, cambia solo l'aritmetica del formato.
 */
export function kquantQ41MultiRowSplitKIdotWgsl(o: KQuantKernelOpts): string {
  const { N, M } = o;
  const { upr, per } = check(o, "kquantQ41MultiRowSplitKIdotWgsl");
  return `@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> xsc: array<f32>;
const BPR = ${upr}u;
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

// ---------------------------------------------------------------------------
// Q4_1 — VIA f32
// ---------------------------------------------------------------------------
export function kquantQ41MultiRowSplitKWgsl(o: KQuantKernelOpts): string {
  const { N, M, K } = o;
  const { upr, per } = check(o, "kquantQ41MultiRowSplitKWgsl");
  return `@group(0) @binding(0) var<storage, read> qs4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
const BPR = ${upr}u;
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
// La lista dei bracci — leggibile SENZA GPU, che e' cio' che la rende
// verificabile in CI (stessa postura di `prefillAttnVariants` in ttAttn.ts).
// ---------------------------------------------------------------------------

export interface KQuantShape {
  family: KQuantFamily;
  K: number;
  N: number;
  /** da dove viene questa shape: e' un tensore vero, non un numero tondo */
  label: string;
  Ms: number[];
}

/**
 * Le shape REALI su cui si misura. Provenienza: inventario per-layer pinnato in
 * tests/engine-prefillgemmplan.test.ts (4B) e header dump 2026-08-10 (35B).
 */
export const KQUANT_SHAPES: readonly KQuantShape[] = [
  { family: "q5_K", K: 4096, N: 2560, label: "4B blk.*.ssm_out (24 tensori, 8,45% dei byte del prefill, 37,9% del TEMPO)", Ms: [1, 8, 16] },
  { family: "q4_1", K: 9216, N: 2560, label: "4B blk.0-3.ffn_down (4 tensori, 71% dei byte del segmento gemm:ffn-down)", Ms: [1, 8, 16] },
];

export interface KQuantVariant {
  id: string;
  code: string;
  grid: [number, number, number];
  /** true = la cella ha bisogno delle attivazioni quantizzate a i8 */
  idot: boolean;
  /** true = e' il braccio di paragone, cioe' cio' che il motore emette OGGI */
  legacy: boolean;
  splits: number;
  ctx: string;
}

/**
 * I bracci per una (famiglia, shape, M). Il primo e' SEMPRE il legacy di
 * produzione: se la lista non lo contenesse, il rapporto della regola di stop
 * non avrebbe denominatore.
 */
export function kquantVariants(o: { family: KQuantFamily; K: number; N: number; M: number }): KQuantVariant[] {
  const { family, K, N, M } = o;
  const splits = kquantSplitsFor(family, K);
  const [lgx, lgy] = gemvGrid(N);
  const out: KQuantVariant[] = [];
  if (family === "q5_K") {
    out.push({
      id: "base-batch-z", code: gemvQ5KWgsl({ K, N, batch: true }), grid: [lgx, lgy, M],
      idot: false, legacy: true, splits: 0,
      ctx: "FORMA ATTUALE, importata da src/engine/kernels/wgsl.ts (gemvQ5KWgsl con batch: true): e' cio' che il motore emette oggi su ssm_out. M GEMV replicate su wid.z, riuso dei pesi ZERO — ogni fetta z rilegge l'intera matrice.",
    });
    out.push({
      id: "splitk-idot", code: kquantQ5KMultiRowSplitKIdotWgsl({ family, K, N, M, splits }),
      grid: kquantMultiRowGrid({ N, splits }), idot: true, legacy: false, splits,
      ctx: "multi-riga split-K con prodotto scalare intero: il piano del 5o bit sommato IMPACCHETTATO, il termine del min una volta per workgroup. Ogni superblocco di peso letto UNA volta per tutte le M righe.",
    });
    out.push({
      id: "splitk-f32", code: kquantQ5KMultiRowSplitKWgsl({ family, K, N, M, splits }),
      grid: kquantMultiRowGrid({ N, splits }), idot: false, legacy: false, splits,
      ctx: "stessa mappatura, attivazioni in virgola mobile: FALLBACK DICHIARATO per i device senza packed_4x8_integer_dot_product. Tile di un gruppo da 64 per volta, cosi' il fabbisogno di memoria di gruppo resta sotto il minimo di spec WebGPU.",
    });
    return out;
  }
  out.push({
    id: "base-batch-z", code: gemvQuantWgsl({ kind: "q4_1", K, N, hasBias: false, batch: true }), grid: [lgx, lgy, M],
    idot: false, legacy: true, splits: 0,
    ctx: "FORMA ATTUALE, importata da src/engine/kernels/wgsl.ts (gemvQuantWgsl kind q4_1 con batch: true): e' cio' che il motore emette oggi su ffn_down dei layer 0-3. Riuso dei pesi ZERO.",
  });
  out.push({
    id: "splitk-idot", code: kquantQ41MultiRowSplitKIdotWgsl({ family, K, N, M, splits }),
    grid: kquantMultiRowGrid({ N, splits }), idot: true, legacy: false, splits,
    ctx: "multi-riga split-K intera: nibble senza offset (q4_1 non centra su zero) piu' il termine m*Sigma(x), calcolato una volta per workgroup.",
  });
  out.push({
    id: "splitk-f32", code: kquantQ41MultiRowSplitKWgsl({ family, K, N, M, splits }),
    grid: kquantMultiRowGrid({ N, splits }), idot: false, legacy: false, splits,
    ctx: "stessa mappatura in virgola mobile: FALLBACK DICHIARATO.",
  });
  return out;
}
