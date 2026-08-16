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
import { gemvQ5KWgsl, gemvQ4KWgsl, gemvQ6KWgsl, gemvQuantWgsl, gemvGrid } from "../engine/kernels/wgsl";

/**
 * Le cinque famiglie che il motore legge e che la via veloce q4_0 non copre.
 * DUE si cablano in questo goal (q5_K, q4_1: sono quelle che il 4B ha);
 * TRE si misurano soltanto, e sono l'eredita' del goal 35B — dove valgono
 * 17,67 GB (q4_K), 1,09 GB (q8_0) e 0,66 GB (q6_K) contro ZERO byte di q4_0.
 */
export type KQuantFamily = "q5_K" | "q4_1" | "q4_K" | "q6_K" | "q8_0";

/** Le due che questo goal porta in produzione. Il resto e' misura. */
export const KQUANT_WIRED: readonly KQuantFamily[] = ["q5_K", "q4_1"];

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
  // Q4_K: 256 pesi in 144 B = 36 parole. Come il Q5_K ma senza `qh`, e `qs`
  // sta a byte 16 (dove il Q5_K tiene il piano del 5o bit).
  q4_K: { unit: 256, words: 36, deviceBytes: 144 },
  // Q6_K: 210 B nel file, **212 sul device** — il repack allinea il superblocco
  // alla parola (53 parole, 2 byte di pad) e il kernel indicizza con QUELLO
  // stride. Contare 210 sottostimerebbe ogni riga dello 0,95%.
  q6_K: { unit: 256, words: 53, deviceBytes: 212 },
  // Q8_0: blocco da 32 pesi gia' in i8. `repackQ8_0` produce qs (8 parole) e
  // scales a DUE f16 per parola, come il q4_0.
  q8_0: { unit: 32, words: 8, deviceBytes: 34 },
};

/**
 * Le famiglie che tengono i pesi in UN buffer solo (il superblocco GGUF
 * copiato in parole) contro quelle che il repack spezza in `qs` + `scales`.
 * E' una proprieta' del FORMATO, e decide i binding: tenerla qui evita che il
 * banco e il motore la deducano ognuno per conto suo.
 */
export function kquantSingleBuffer(family: KQuantFamily): boolean {
  return family === "q5_K" || family === "q4_K" || family === "q6_K";
}

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
// Q4_K — VIA INTERA. E' il Q5_K SENZA il piano del 5o bit.
// ---------------------------------------------------------------------------
/**
 * Superblocco da 144 B = 36 parole: `d,dmin` (parola 0), 12 byte di scale a 6
 * bit (byte 4..15), `qs` a byte **16** — dove il Q5_K ha il suo `qh`. Stessa
 * `get_scale_min_k4`, stessa aritmetica `d*sc*q - dmin*mn`, un piano di bit in
 * meno. Gli offset sono verificati contro `gemvQ4KWgsl` di produzione: e' li'
 * che un port distratto sbaglia (16 invece di 48, 36 parole invece di 44).
 *
 * NON viene cablata da questo goal: e' la forma che eredita il goal 35B, dove
 * questi 117 tensori sono 17,67 GB — cioe' quasi tutto il modello.
 */
export function kquantQ4KMultiRowSplitKIdotWgsl(o: KQuantKernelOpts): string {
  const { N, M } = o;
  const { upr, per } = check(o, "kquantQ4KMultiRowSplitKIdotWgsl");
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

// ---------------------------------------------------------------------------
// Q6_K — VIA INTERA. La piu' diversa delle cinque, per due ragioni.
// ---------------------------------------------------------------------------
/**
 * (1) I SOTTO-BLOCCHI SONO DA 16, non da 32: 16 scale int8 per 256 pesi. Le
 *     attivazioni sono pero' quantizzate per 32, quindi le due meta' di un
 *     blocco condividono la scala di x e hanno scale di peso diverse — servono
 *     accumuli separati per meta', ed e' per questo che qui `xsum` e' per MEZZO
 *     sotto-blocco (M x 16 f32) invece che per sotto-blocco.
 *
 * (2) I PESI SONO CENTRATI SU -32, e sottrarre 32 in forma impacchettata
 *     traboccherebbe fra i byte. Non si sottrae: si tiene `q` senza segno in
 *     [0,63] — che sta in i8 positivo — e si porta l'offset FUORI dal prodotto
 *     scalare, `Sigma (q-32)x = Sigma q*x - 32*Sigma x`. E' lo stesso Sigma(x)
 *     che le altre famiglie usano per il loro termine costante: un solo
 *     meccanismo, due usi.
 *
 * Layout (verificato contro `gemvQ6KWgsl`): 53 parole; `ql` byte 0, `qh` byte
 * 128, scale int8 byte 192, `d` f16 nella parola 52.
 */
export function kquantQ6KMultiRowSplitKIdotWgsl(o: KQuantKernelOpts): string {
  const { N, M } = o;
  const { upr, per } = check(o, "kquantQ6KMultiRowSplitKIdotWgsl");
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

// ---------------------------------------------------------------------------
// Q8_0 — VIA INTERA. La piu' semplice: i pesi SONO gia' i8.
// ---------------------------------------------------------------------------
/**
 * Niente unpack, niente termine costante: il ciclo interno e' `dot4I8Packed`
 * nudo su otto parole, e la scala del peso per la scala di x si applica una
 * volta per blocco. Se sbaglia questa, sbaglia il banco.
 *
 * Layout `repackQ8_0`: 8 parole di i8 per blocco da 32, scale a DUE f16 per
 * parola (come il q4_0), quindi `scales[gb >> 1][gb & 1]`.
 */
export function kquantQ80MultiRowSplitKIdotWgsl(o: KQuantKernelOpts): string {
  const { N, M } = o;
  const { upr, per } = check(o, "kquantQ80MultiRowSplitKIdotWgsl");
  return `@group(0) @binding(0) var<storage, read> qs: array<u32>;
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

// ---------------------------------------------------------------------------
// I FALLBACK f32 DELLE TRE FAMIGLIE EREDITATE
//
// Perche' esistono, dopo che it.2 aveva deciso di non scriverli: il contratto
// dice «ogni via intera nuova va accompagnata dal suo fallback f32 DICHIARATO,
// come la q4_0», ed e' un CONSTRAINT del PI, non una preferenza di meccanismo.
// La decisione di saltarli era mia e fuori dalla mia autorita' — rilievo del
// verificatore in it.2, accolto: costava meno scriverli che chiedere il
// permesso di non farlo.
// ---------------------------------------------------------------------------

/** Q4_K in virgola mobile: il Q5_K f32 senza il piano del 5o bit. */
export function kquantQ4KMultiRowSplitKWgsl(o: KQuantKernelOpts): string {
  const { N, M, K } = o;
  const { upr, per } = check(o, "kquantQ4KMultiRowSplitKWgsl");
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
 * Q6_K in virgola mobile. Qui l'offset −32 si sottrae direttamente sul peso:
 * il problema del traboccamento fra byte esiste SOLO nella forma impacchettata.
 * Le due scale del sotto-blocco si scelgono con `l >> 4`, come in produzione.
 */
export function kquantQ6KMultiRowSplitKWgsl(o: KQuantKernelOpts): string {
  const { N, M, K } = o;
  const { upr, per } = check(o, "kquantQ6KMultiRowSplitKWgsl");
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

/** Q8_0 in virgola mobile: i pesi si convertono da i8, nient'altro. */
export function kquantQ80MultiRowSplitKWgsl(o: KQuantKernelOpts): string {
  const { N, M, K } = o;
  const { upr, per } = check(o, "kquantQ80MultiRowSplitKWgsl");
  return `@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
const BPR = ${upr}u;
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
  // --- le due CABLATE da questo goal: shape del 4B, tre M ciascuna ---
  { family: "q5_K", K: 4096, N: 2560, label: "4B blk.*.ssm_out (24 tensori, 8,45% dei byte del prefill, 37,9% del TEMPO)", Ms: [1, 8, 16] },
  { family: "q4_1", K: 9216, N: 2560, label: "4B blk.0-3.ffn_down (4 tensori, 71% dei byte del segmento gemm:ffn-down)", Ms: [1, 8, 16] },
  // --- le tre EREDITATE dal goal 35B. `Ms` e' [1, 8, 16] come per le altre:
  //     it.2 le aveva ristrette a M=16 e il done-when del contratto dice
  //     esplicitamente «a M = 1, 8, 16, su TUTTE le famiglie». Restringere un
  //     done-when e' del PI, non mio (rilievo del verificatore, it.2).
  { family: "q4_K", K: 2048, N: 512, label: "35B expert gate/up (117 tensori Q4_K = 17,67 GB, il modello quasi tutto)", Ms: [1, 8, 16] },
  { family: "q4_K", K: 512, N: 2048, label: "35B expert down — DUE superblocchi per riga, split-K a 2 fette (C0-4)", Ms: [1, 8, 16] },
  { family: "q6_K", K: 512, N: 2048, label: "35B expert down di 3 layer (860.160 B per expert, verificato sull'header)", Ms: [1, 8, 16] },
  { family: "q8_0", K: 2048, N: 4096, label: "35B attn q-proj (100 tensori Q8_0 = 1,09 GB) — e' anche attn_gate del DECODE, 33,16% dei pesi dei quattro (it.17)", Ms: [1, 8, 16] },
  // La shape che mancava, ed e' la PIU' GRANDE dei quattro tensori di `ssmGemv`:
  // N = (2*group_count + time_step_rank)*state_size = (2*16+32)*128 = 8192,
  // K = dModel = 2048 (q35shape.ts:86-89 + meta dell'header, it.17). Da sola il
  // 66,32% dei byte dei quattro; con attn_gate, il 99,48%.
  // `ssmGemv` e' il primo termine del decode del 35B dopo il kfan (6,99 ms/token
  // su 30 dispatch = 233 us a layer) e ce l'hanno anche 4B e 9B: e' li' che una
  // leva sarebbe globale per costruzione.
  { family: "q8_0", K: 2048, N: 8192, label: "35B attn_qkv del DECODE (66,32% dei pesi di ssmGemv, il primo termine)", Ms: [1, 8, 16] },
  // LE SHAPE PICCOLE DEL Q8_0 — `shexp` del 35B (it.38), e mancavano.
  //
  // Il predicato ammette tutto cio' che ha N >= PREFILL_GEMM_ROWS_PER_WG = 64,
  // quindi la rotta split-K del decode instrada ANCHE questi due. Ma il banco
  // aveva misurato il q8_0 solo a N=4096 e N=8192, dove rende 3,3-3,9x. Sulle
  // shape piccole analoghe dei K-quant la stessa forma fa 0,20-0,56x — cioe'
  // PERDE, e di molto. Se perde anche qui, la rotta sta rallentando un segmento
  // da 2,089 ms/token e nessuno se n'e' accorto perche' i guadagni sulle shape
  // grandi lo coprono.
  //
  // E' esattamente la cella che il banco non aveva e il motore esegue: la
  // ragione per cui questa riga esiste.
  //
  // Ms = [1, 8, 16] e non solo [1] benche' la domanda sia sul DECODE: il caso
  // [b2] pretende che ogni shape copra le tre M, ed e' un invariante giusto —
  // anche il prefill instrada questi due tensori, quindi misurarli a una M sola
  // lascerebbe scoperto proprio il regime in cui la forma multi-riga dovrebbe
  // vincere. Dare tre M costa qualche secondo di banco e non indebolisce niente.
  { family: "q8_0", K: 2048, N: 512, label: "35B ffn_gate/up_shexp — INSTRADATA dalla rotta, mai misurata qui", Ms: [1, 8, 16] },
  { family: "q8_0", K: 512, N: 2048, label: "35B ffn_down_shexp — INSTRADATA dalla rotta, mai misurata qui", Ms: [1, 8, 16] },
];

export interface KQuantVariant {
  id: string;
  code: string;
  grid: [number, number, number];
  /** true = la cella ha bisogno delle attivazioni quantizzate a i8 */
  idot: boolean;
  /**
   * true = e' un braccio di paragone, cioe' cio' che il motore emette OGGI.
   * NON e' la stessa cosa di "il denominatore": da it.21 i paragoni sono DUE,
   * uno per regime (v. `regime`). Il runner usa questo campo per scegliere il
   * buffer di uscita e il fattore di emissione, non per fare i rapporti.
   */
  legacy: boolean;
  /**
   * Il regime di cui questo braccio e' il paragone — presente SOLO sui bracci
   * `legacy`, assente sui candidati.
   *
   * Esiste perche' i due regimi emettono kernel DIVERSI, e confonderli e'
   * costato la credibilita' di un numero: il «3,26x a M=1» con cui la riga 2d
   * di `engine-velocita-decode` giustificava 2-3 iterazioni era misurato contro
   * `base-batch-z`, cioe' la forma del PREFILL (M righe su `wid.z`), mentre la
   * leva che doveva giustificare era nel DECODE, che emette lo stesso
   * generatore SENZA `batch`. Alla verifica i due coincidono entro l'1,4% e il
   * rapporto ha retto (3,30x) — ma non lo sapeva nessuno, ed e' un caso in cui
   * l'esito buono non riscatta il metodo.
   *
   * La regola: **un denominatore per regime, mai due nello stesso**.
   */
  regime?: "prefill" | "decode";
  splits: number;
  ctx: string;
}

/**
 * I bracci per una (famiglia, shape, M). Il primo e' SEMPRE il paragone del
 * PREFILL: se la lista non lo contenesse, il rapporto della regola di stop non
 * avrebbe denominatore. A M=1 la lista porta anche il paragone del DECODE.
 */
export function kquantVariants(o: { family: KQuantFamily; K: number; N: number; M: number }): KQuantVariant[] {
  const { family, K, N, M } = o;
  const splits = kquantSplitsFor(family, K);
  const [lgx, lgy] = gemvGrid(N);
  const out: KQuantVariant[] = [];
  // BRACCIO DEL DECODE (goal engine-velocita-decode, it.21), scritto UNA volta
  // per tutte e cinque le famiglie invece che ricopiato in ogni ramo.
  // Solo a M=1: a M>1 il decode non esiste, e un braccio senza `batch` con M
  // righe misurerebbe una forma che nessuno emette.
  const pushDecodeArm = (): void => {
    if (M !== 1) return;
    const code = family === "q5_K"
      ? gemvQ5KWgsl({ K, N })
      : family === "q4_K"
        ? gemvQ4KWgsl({ K, N })
        : family === "q6_K"
          ? gemvQ6KWgsl({ K, N })
          : gemvQuantWgsl({ kind: family, K, N, hasBias: false });
    out.push({
      id: "base-decode", code, grid: [lgx, lgy, 1],
      idot: false, legacy: true, regime: "decode", splits: 0,
      ctx: `IL KERNEL DEL DECODE della famiglia ${family}: lo stesso generatore del paragone di prefill ma SENZA batch, un workgroup da 64 thread per riga di uscita. E' cio' che q35gpumodel.ts:862 emette a M=1, e il solo termine di paragone legittimo per una leva di decode.`,
    });
  };
  if (family === "q5_K") {
    out.push({
      id: "base-batch-z", code: gemvQ5KWgsl({ K, N, batch: true }), grid: [lgx, lgy, M],
      idot: false, legacy: true, regime: "prefill", splits: 0,
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
    pushDecodeArm();
    return out;
  }
  if (family === "q4_1") {
    out.push({
      id: "base-batch-z", code: gemvQuantWgsl({ kind: "q4_1", K, N, hasBias: false, batch: true }), grid: [lgx, lgy, M],
      idot: false, legacy: true, regime: "prefill", splits: 0,
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
    pushDecodeArm();
    return out;
  }

  // --- le TRE non cablate: via intera E fallback f32, come le altre due.
  // it.2 aveva deciso di misurare la sola via intera («una forma senza
  // consumatore e' una forma che nessuno esegue per mesi»): decisione
  // RITIRATA in it.3, perche' il contratto dice «ogni via intera nuova va
  // accompagnata dal suo fallback f32 DICHIARATO» ed e' un vincolo del PI, non
  // una preferenza di meccanismo. Chiedere il permesso di saltarli sarebbe
  // costato piu' che scriverli.
  const legacyCode = family === "q4_K"
    ? gemvQ4KWgsl({ K, N, batch: true })
    : family === "q6_K"
      ? gemvQ6KWgsl({ K, N, batch: true })
      : gemvQuantWgsl({ kind: "q8_0", K, N, hasBias: false, batch: true });
  const fastCode = family === "q4_K"
    ? kquantQ4KMultiRowSplitKIdotWgsl({ family, K, N, M, splits })
    : family === "q6_K"
      ? kquantQ6KMultiRowSplitKIdotWgsl({ family, K, N, M, splits })
      : kquantQ80MultiRowSplitKIdotWgsl({ family, K, N, M, splits });
  out.push({
    id: "base-batch-z", code: legacyCode, grid: [lgx, lgy, M],
    idot: false, legacy: true, regime: "prefill", splits: 0,
    ctx: `FORMA ATTUALE della famiglia ${family}, importata da src/engine/kernels/wgsl.ts: e' cio' che il motore emetterebbe oggi su questi tensori. Riuso dei pesi ZERO.`,
  });
  const f32Code = family === "q4_K"
    ? kquantQ4KMultiRowSplitKWgsl({ family, K, N, M, splits })
    : family === "q6_K"
      ? kquantQ6KMultiRowSplitKWgsl({ family, K, N, M, splits })
      : kquantQ80MultiRowSplitKWgsl({ family, K, N, M, splits });
  out.push({
    id: "splitk-idot", code: fastCode,
    grid: kquantMultiRowGrid({ N, splits }), idot: true, legacy: false, splits,
    ctx: family === "q6_K"
      ? "multi-riga split-K intera: sotto-blocchi da SEDICI (non 32) e pesi centrati su -32, con l'offset portato fuori dal prodotto scalare come -32*Sigma(x) invece di sottratto in forma impacchettata, dove traboccherebbe fra i byte."
      : family === "q4_K"
        ? "multi-riga split-K intera: il Q5_K senza il piano del 5o bit, con qs a byte 16 e il superblocco da 36 parole."
        : "multi-riga split-K intera: i pesi SONO gia' i8, quindi niente unpack e niente termine costante — il ciclo interno e' dot4I8Packed nudo.",
  });
  out.push({
    id: "splitk-f32", code: f32Code,
    grid: kquantMultiRowGrid({ N, splits }), idot: false, legacy: false, splits,
    ctx: "stessa mappatura in virgola mobile: FALLBACK DICHIARATO per i device senza packed_4x8_integer_dot_product. Sul Q6_K l'offset -32 qui si sottrae direttamente dal peso — il traboccamento fra byte esiste solo nella forma impacchettata.",
  });
  pushDecodeArm();
  return out;
}
