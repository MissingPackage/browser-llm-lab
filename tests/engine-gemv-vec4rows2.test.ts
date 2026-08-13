// GEMV q4_0 "vec4 + 2 righe per workgroup" (opt-in `vec4Rows2`) — contratto di
// FORMA. Nessuna GPU qui: si asserisce sul TESTO WGSL generato e sulla griglia
// di dispatch, che sono gli unici osservabili di wgsl.ts fuori dal browser.
//
// Le quattro cose che questo file inchioda:
//  1. NON-REGRESSIONE. I call-site di glmforward.ts/glmmodel.ts non passano
//     `vec4Rows2`: il testo che ricevono deve restare IDENTICO a quello del
//     commit 5114160. Il confronto e' sul TESTO INTERO contro una copia
//     congelata del generatore di 5114160 inlinata qui sotto (cosi' un
//     fallimento mostra il diff, non solo "hash diverso"); la fedelta' della
//     copia e' a sua volta inchiodata dagli sha256 misurati su
//     `git show 5114160:src/engine/kernels/wgsl.ts`.
//  2. 2 RIGHE PER WORKGROUP NEL TESTO. Il punto del kernel non e' la griglia
//     dimezzata: e' che il corpo copra DUE righe. Se la griglia si dimezza e il
//     corpo resta a una riga, meta' di y non viene mai scritta e il kernel e'
//     silenziosamente sbagliato. Qui si legge la mappa riga->workgroup dal
//     sorgente (base scalata x2 + componente per-thread) e il rilevatore e'
//     tarato sul corpo a 1 riga di 5114160, che deve risultare NON scalato.
//  3. GRIGLIA. `vec4Rows2` calcola 2 righe per workgroup, quindi la griglia e'
//     gemvGrid(ceil(N/2)); senza il flag resta 1 riga e gemvGrid(N).
//  4. GUARDIE. Le combinazioni non misurate (q4_1/q8_0, bias, batch,
//     scaledAccum, `sg` da solo) devono THROW, non generare WGSL plausibile —
//     e chi rifiuta il testo deve rifiutare anche la griglia, altrimenti
//     l'helper di dispatch e il generatore restano in disaccordo silenzioso.
import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import {
  GEMV_GRID_X, gemvGrid, gemvQuantGrid, gemvQuantRowsPerWg, gemvQuantWgsl,
  type GemvQuantOpts,
} from "../src/engine/kernels/wgsl";

// `vec4Rows2` e' il simbolo che questo test pretende da wgsl.ts. Finche' non
// esiste nell'interfaccia, l'intersezione lo tiene tipabile senza `any`: il
// ROSSO deve arrivare dal COMPORTAMENTO mancante, non da un errore di tipi.
type Vec4Opts = GemvQuantOpts & { vec4Rows2?: boolean; sg?: boolean };

const wgsl = (o: Vec4Opts): string => gemvQuantWgsl(o);
const rowsPerWg = (o: Vec4Opts): number => gemvQuantRowsPerWg(o);
const grid = (o: Vec4Opts): [number, number] => gemvQuantGrid(o);
const sha = (s: string) => createHash("sha256").update(s).digest("hex");

// ---------------------------------------------------------------------------
// 0. Copia congelata del generatore a 5114160
// ---------------------------------------------------------------------------

// Trascrizione VERBATIM di `gemvQuantWgsl` da
// `git show 5114160:src/engine/kernels/wgsl.ts`, con TOK_PARAMS_WGSL e
// GEMV_GRID_X risolti nei loro valori di allora (`32768u`): il freeze deve
// rompersi anche se cambia una di quelle costanti, non solo il corpo.
type FrozenOpts = {
  kind: "q4_0" | "q4_1" | "q8_0"; K: number; N: number; hasBias: boolean;
  batch?: boolean; scaledAccum?: boolean;
};
function wgsl5114160(opts: FrozenOpts): string {
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
  return `struct TokParams { pos: u32, nPast: u32 };
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
  let r = wid.x + wid.y * 32768u;
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

// ---------------------------------------------------------------------------
// 1. Non-regressione dei call-site GLM-4.7-Flash
// ---------------------------------------------------------------------------

// Le opts ESATTE dei call-site, con le shape di GLM47_FLASH gia' risolte
// (dModel 2048, qLora 768, nHead 20, HL = qkNope 192 + ropeDims 64 = 256 ⇒
// nHead*HL = 5120, keyLen 576, headLenMla 256 ⇒ nHead*headLenMla = 5120,
// dFfnDense 10240, dFfnExpert 1536). I numeri sono inlinati apposta: questo
// file importa SOLO wgsl.ts, cosi' un cambio di shape non maschera un cambio
// di kernel.
const GLM_CALLSITES: Record<string, Vec4Opts> = {
  // glmforward.ts:100 / glmmodel.ts:580 — attn_q_a
  qA: { kind: "q4_0", K: 2048, N: 768, hasBias: false },
  // glmforward.ts:101 / glmmodel.ts:581 — attn_q_b
  qB: { kind: "q4_0", K: 768, N: 5120, hasBias: false },
  // glmforward.ts:102 / glmmodel.ts:582 — attn_kv_a_mqa
  kvA: { kind: "q8_0", K: 2048, N: 576, hasBias: false },
  // glmforward.ts:111 / glmmodel.ts:599 — attn_output
  o: { kind: "q4_0", K: 5120, N: 2048, hasBias: false },
  // glmforward.ts:112 / glmmodel.ts:601 — ffn gate/up denso
  gateUp: { kind: "q4_0", K: 2048, N: 10240, hasBias: false },
  // glmforward.ts:113 / glmmodel.ts:602 — ffn down denso (Q4_1)
  down: { kind: "q4_1", K: 10240, N: 2048, hasBias: false },
  // glmmodel.ts:625-644 — stessi tensori, decode batch (wid.z = riga)
  "qA.batch": { kind: "q4_0", K: 2048, N: 768, hasBias: false, batch: true },
  "qB.batch": { kind: "q4_0", K: 768, N: 5120, hasBias: false, batch: true },
  "kvA.batch": { kind: "q8_0", K: 2048, N: 576, hasBias: false, batch: true },
  "o.batch": { kind: "q4_0", K: 5120, N: 2048, hasBias: false, batch: true },
  "gateUp.batch": { kind: "q4_0", K: 2048, N: 10240, hasBias: false, batch: true },
  "down.batch": { kind: "q4_1", K: 10240, N: 2048, hasBias: false, batch: true },
  // MoE (ktest.worker.ts:1519): il down per-expert accumula pesato, Q4_0 e Q4_1
  "expDown40.scaled": { kind: "q4_0", K: 1536, N: 2048, hasBias: false, scaledAccum: true },
  "expDown41.scaled": { kind: "q4_1", K: 1536, N: 2048, hasBias: false, scaledAccum: true },
};

// sha256 del testo prodotto da gemvQuantWgsl a 5114160 per quelle opts. Sono
// misurati sul sorgente del commit, non sulla copia qui sopra: se la
// trascrizione avesse un refuso, il test "copia fedele" lo scopre.
const FROZEN_5114160: Record<string, string> = {
  qA: "9ad3ff454f36257ce90146d51357dafe0c4e37b6201a412361fecbe7373ddc89",
  qB: "dbf10689d66b255d68bbf09edcb5593851b6b89c476c24c724bff06d4a8b1133",
  kvA: "2f52bdc7097407b64b2765dfe20196b2917b85ce1ea6e1c2b6c5e9dd5a6719e5",
  o: "23f339a1f08cdde660cdc737aede1c353b1f1a21645c69c9f155d6809a5a14be",
  gateUp: "01b3802c37edec2bdd5849ab67e9a3f848422bb6a03e16fe5ad1f56437994c7e",
  down: "b370efc982086c15ed892b8b77f9cdbd3b5d03b4e8adedd03079732a44610c76",
  "qA.batch": "2632aae087e92bfb753886b7827b101fe7ac280977a8a85176d3f8681ced1b70",
  "qB.batch": "959c0a659e9592b3e25c7f2ee7203ed963975a4541ffd9c2c4ad4ed2a26da9db",
  "kvA.batch": "e33edd849182f1dbe1a59f854e0ffeaab8fed18f7df84bf7510829f8808b3951",
  "o.batch": "1f612d8be5eca6b63c6483b3c19804f45b46f5a5ba1f2563ab1d65c0cd2b512a",
  "gateUp.batch": "014372ff32ec0b2e85a7ce94a9a5d83436929ae4253d6da654450b778d42ebfa",
  "down.batch": "1b52ae2d475f243a05686ee3bddab00d96aa8754ee57081cc3b1ee7d559794ff",
  "expDown40.scaled": "3b9b8719e920fdf477e3cee3997323413bbe668729c8d98f0cc3584f2bc7eabc",
  "expDown41.scaled": "f2afd84fafcab53ef3180ff10bb7ba699f57c7189dfaa04b92d873373a258b46",
};

describe("gemvQuantWgsl — call-site GLM: testo congelato a 5114160", () => {
  it("la copia congelata e' fedele al commit (sha256 misurati su 5114160)", () => {
    for (const [name, opts] of Object.entries(GLM_CALLSITES)) {
      expect(sha(wgsl5114160(opts as FrozenOpts)), `copia infedele per ${name}`)
        .toBe(FROZEN_5114160[name]);
    }
  });

  for (const [name, opts] of Object.entries(GLM_CALLSITES)) {
    it(`${name}: WGSL identico al commit 5114160`, () => {
      // toBe su STRINGHE INTERE: se cambia, vitest stampa il diff riga per riga.
      expect(wgsl(opts)).toBe(wgsl5114160(opts as FrozenOpts));
    });
  }

  it("vec4Rows2 e' opt-in: false/undefined lasciano il testo di 5114160", () => {
    for (const name of ["qA", "o", "gateUp"]) {
      const opts = GLM_CALLSITES[name];
      expect(wgsl({ ...opts, vec4Rows2: false }), name).toBe(wgsl5114160(opts as FrozenOpts));
      expect(wgsl({ ...opts, vec4Rows2: undefined }), name).toBe(wgsl5114160(opts as FrozenOpts));
    }
  });
});

// ---------------------------------------------------------------------------
// 2. Righe per workgroup e griglia di dispatch
// ---------------------------------------------------------------------------

describe("gemvQuantRowsPerWg / gemvQuantGrid", () => {
  it("1 riga senza vec4Rows2, 2 con vec4Rows2", () => {
    expect(rowsPerWg({ kind: "q4_0", K: 2048, N: 1536, hasBias: false })).toBe(1);
    expect(rowsPerWg({ kind: "q4_0", K: 2048, N: 1536, hasBias: false, vec4Rows2: false })).toBe(1);
    expect(rowsPerWg({ kind: "q4_0", K: 2048, N: 1536, hasBias: false, vec4Rows2: true })).toBe(2);
    expect(rowsPerWg({ kind: "q4_0", K: 2048, N: 1536, hasBias: false, vec4Rows2: true, sg: true })).toBe(2);
  });

  // N piccolo (q_a), N dispari (l'ultima riga resta scoperta a meta' wg),
  // N > 2*GEMV_GRID_X = 65536: le teste di vocabolario (154880 righe di
  // GLM47_FLASH, 248320 righe della cella lm_head di fase 0), dove la griglia
  // DEVE passare per la seconda dimensione.
  const NS = [768, 2561, 154880, 248320];

  it("senza vec4Rows2 la griglia e' quella di sempre: gemvGrid(N)", () => {
    for (const N of NS) {
      const o: Vec4Opts = { kind: "q4_0", K: 2048, N, hasBias: false };
      expect(grid(o), `N=${N}`).toEqual(gemvGrid(N));
    }
  });

  it("con vec4Rows2 la griglia e' gemvGrid(ceil(N/2))", () => {
    for (const N of NS) {
      const o: Vec4Opts = { kind: "q4_0", K: 2048, N, hasBias: false, vec4Rows2: true };
      expect(grid(o), `N=${N}`).toEqual(gemvGrid(Math.ceil(N / rowsPerWg(o))));
      expect(grid(o), `N=${N}`).toEqual(gemvGrid(Math.ceil(N / 2)));
    }
  });

  it("l'lm_head da 248320 righe sfonda GEMV_GRID_X e resta coperto", () => {
    expect(248320).toBeGreaterThan(GEMV_GRID_X * 2);
    const flat: Vec4Opts = { kind: "q4_0", K: 2048, N: 248320, hasBias: false };
    expect(grid(flat)).toEqual([GEMV_GRID_X, Math.ceil(248320 / GEMV_GRID_X)]);
    const half = Math.ceil(248320 / 2);
    const g = grid({ ...flat, vec4Rows2: true });
    expect(g).toEqual([GEMV_GRID_X, Math.ceil(half / GEMV_GRID_X)]);
    // copertura: i workgroup lanciati × 2 righe coprono tutte le righe
    expect(g[0] * g[1] * 2).toBeGreaterThanOrEqual(248320);
  });

  it("N dispari: si arrotonda per eccesso, mai per difetto", () => {
    const g = grid({ kind: "q4_0", K: 2048, N: 2561, hasBias: false, vec4Rows2: true });
    expect(g).toEqual(gemvGrid(1281));
    expect(g[0] * g[1] * 2).toBeGreaterThanOrEqual(2561);
  });
});

// ---------------------------------------------------------------------------
// 3. La mappa riga -> workgroup DEV'ESSERE nel testo
// ---------------------------------------------------------------------------

// Il fallimento che questo blocco esiste per impedire: griglia dimezzata
// (gemvQuantGrid) + corpo a UNA riga (quello di 5114160). Suite verde, e le
// righe da ceil(N/2) in poi di y non vengono scritte da nessuno.
//
// `rowMapping` guarda le ESPRESSIONI DI RIGA del sorgente — l'indice con cui si
// scrive `y[...]` e le espressioni confrontate con N nella guardia — e le
// classifica propagando le assegnazioni `let/var X = ...;`:
//   - `scaled`: la riga viene da (indice di workgroup) × 2 (o × ROWS_PER_WG),
//     scritto inline o in due passi (`let wg = wid.x + ...; let r = wg * 2u;`);
//   - `covers2`: le DUE righe del workgroup sono distinte, cioe' la riga
//     dipende anche da qualcosa che varia dentro il workgroup (id di thread o
//     di subgroup, o l'indice di un ciclo sulle 2 righe) — oppure ci sono due
//     espressioni di riga scalate diverse (forma srotolata, `y[r]` e `y[r+1u]`).
// Il rilevatore e' tarato sotto: sul corpo a 1 riga di 5114160 deve dare
// scaled=false, e su una griglia dimezzata con riga NON spezzata per thread
// deve dare covers2=false. Se accettasse quei due, non discriminerebbe piu'
// niente e questo blocco sarebbe teatro.
const THREAD_VARYING = /lid\.|local_invocation|subgroup|\bsg[A-Za-z0-9_]*\b|\bsub[A-Za-z0-9_]*\b/;
const DOUBLED = /\*\s*(?:2u|ROWS_PER_WG)/;

function rowMapping(src: string, N: number): { scaled: boolean; covers2: boolean } {
  const body = src.slice(src.indexOf("fn main"));
  const idents = (e: string) => [...e.matchAll(/[A-Za-z_]\w*/g)].map((m) => m[0]);
  const has = (set: Set<string>, e: string) => idents(e).some((i) => set.has(i));
  const wgV = new Set<string>();      // dipende dall'indice di workgroup
  const scaledV = new Set<string>();  // ... moltiplicato per le righe per workgroup
  const spreadV = new Set<string>();  // varia DENTRO il workgroup (thread o ciclo sulle righe)
  // ciclo sulle 2 righe: `for (var i = 0u; i < 2u; ...)`
  for (const m of body.matchAll(/for\s*\(\s*(?:var|let)\s+(\w+)\s*=\s*0u\s*;\s*\w+\s*<\s*(?:2u|ROWS_PER_WG)/g)) {
    spreadV.add(m[1]);
  }
  for (const m of body.matchAll(/\b(?:let|var)\s+([A-Za-z_]\w*)\s*(?::[^=]+)?=\s*([^;]*);/g)) {
    const name = m[1], rhs = m[2];
    const wg = /wid\.[xy]/.test(rhs) || has(wgV, rhs);
    if (has(scaledV, rhs) || (wg && DOUBLED.test(rhs))) scaledV.add(name);
    else if (wg) wgV.add(name);
    if (THREAD_VARYING.test(rhs) || has(spreadV, rhs)) spreadV.add(name);
  }
  // espressioni di riga: l'indice con cui si scrive y, e cio' che la guardia
  // confronta con N (`if (r >= 1537u)`, `sgId == 0u && r < 1537u`, ...)
  const yExpr = [...body.matchAll(/\by\s*\[([^\]]*)\]/g)].map((m) => m[1].trim());
  const guardExpr = [...body.matchAll(new RegExp(`([\\w\\s.+*/()>-]+?)\\s*(?:<|>=)\\s*${N}u`, "g"))]
    .map((m) => m[1].replace(/^[\s(]*if\s*\(/, "").trim());
  const isScaled = (e: string) => has(scaledV, e) || (/wid\.[xy]/.test(e) && DOUBLED.test(e));
  const isSpread = (e: string) => THREAD_VARYING.test(e) || has(spreadV, e);
  const scaledRows = [...yExpr, ...guardExpr].filter(isScaled);
  // due scritture di y su righe scalate DIVERSE = forma srotolata (`y[r]`, `y[r+1u]`)
  const distinctY = new Set(yExpr.filter(isScaled).map((e) => e.replace(/\s+/g, "")));
  return { scaled: scaledRows.length > 0, covers2: scaledRows.some(isSpread) || distinctY.size >= 2 };
}

describe("rowMapping — il rilevatore discrimina davvero", () => {
  it("il corpo a 1 riga di 5114160 NON risulta scalato", () => {
    // `let r = wid.x + wid.y * 32768u;` — nessuna moltiplicazione per le righe
    // per workgroup: se questo desse scaled=true, i test sotto passerebbero
    // anche riusando il corpo vecchio sotto una griglia dimezzata.
    const oneRow = rowMapping(wgsl5114160({ kind: "q4_0", K: 2048, N: 1536, hasBias: false }), 1536);
    expect(oneRow.scaled).toBe(false);
    expect(oneRow.covers2).toBe(false);
  });

  it("la griglia dimezzata con UNA riga scritta per workgroup non passa", () => {
    // Il bug esatto: base scalata ×2, ma il workgroup scrive una riga sola.
    const halfWritten = `fn main() {
  let r = (wid.x + wid.y * 32768u) * 2u;
  if (r >= 1536u) { return; }
  let t = lid.x;
  y[r] = acc;
}`;
    expect(rowMapping(halfWritten, 1536)).toEqual({ scaled: true, covers2: false });
  });

  it("riconosce le forme plausibili della mappa a 2 righe", () => {
    const inline = `fn main() {
  let r = (wid.x + wid.y * 32768u) * 2u + lid.x / 32u;
  if (r >= 1536u) { return; }
  y[r] = acc;
}`;
    const twoSteps = `fn main() {
  let wg = wid.x + wid.y * 32768u;
  let base = wg * ROWS_PER_WG;
  let row = lid.x >> 5u;
  let r = base + row;
  if (r < 1536u) { y[r] = acc; }
}`;
    const unrolled = `fn main() {
  let r = (wid.x + wid.y * 32768u) * 2u;
  if (r < 1536u) { y[r] = a0; }
  if (r + 1u < 1536u) { y[r + 1u] = a1; }
}`;
    const loop = `fn main() {
  let base = (wid.x + wid.y * 32768u) * 2u;
  for (var i = 0u; i < 2u; i = i + 1u) {
    let r = base + i;
    if (r < 1536u) { y[r] = acc; }
  }
}`;
    for (const [n, src] of [["inline", inline], ["twoSteps", twoSteps], ["unrolled", unrolled], ["loop", loop]] as const) {
      expect(rowMapping(src, 1536), n).toEqual({ scaled: true, covers2: true });
    }
  });
});

// Generati DENTRO i test (non a modulo): se un ramo lancia, deve fallire il
// test che lo riguarda, non la raccolta dell'intero file.
const SG = (N = 1536) => wgsl({ kind: "q4_0", K: 2048, N, hasBias: false, vec4Rows2: true, sg: true });
const FB = (N = 1536) => wgsl({ kind: "q4_0", K: 2048, N, hasBias: false, vec4Rows2: true });

describe("vec4Rows2 — due righe per workgroup nel TESTO, non solo nella griglia", () => {
  for (const [name, gen] of [["sg", SG], ["fallback", FB]] as const) {
    it(`${name}: la riga e' (indice di workgroup)×2 + riga nel workgroup`, () => {
      const map = rowMapping(gen(), 1536);
      expect(map.scaled, `${name}: l'indice di workgroup non viene moltiplicato per le 2 righe`)
        .toBe(true);
      expect(map.covers2, `${name}: il workgroup scrive una riga sola — meta' di y resta vuota`)
        .toBe(true);
    });

    it(`${name}: workgroup da 64 thread (2 righe × 32 lane)`, () => {
      expect(gen()).toContain("@workgroup_size(64)");
    });

    it(`${name}: le righe per workgroup del testo sono quelle dell'helper di griglia`, () => {
      // Il testo dichiara/usa il fattore 2; l'helper deve dire lo stesso, o
      // griglia e kernel parlano di due kernel diversi.
      const o: Vec4Opts = { kind: "q4_0", K: 2048, N: 1536, hasBias: false, vec4Rows2: true, sg: name === "sg" };
      expect(rowsPerWg(o)).toBe(2);
      expect(grid(o)).toEqual(gemvGrid(768));
      expect(rowMapping(gen(), 1536).scaled).toBe(true);
    });

    it(`${name}: N dispari — la guardia sulle righe valide c'e' e copre le 2 righe`, () => {
      // Con N dispari l'ultimo workgroup ha una riga sola dentro il tensore:
      // la guardia deve stare sulla riga vera, non sull'indice di workgroup.
      const src = gen(1537);
      expect(src).toContain("1537u");
      // il confronto puo' essere col letterale o con una costante che lo lega
      // (`const N_ROWS = 1537u;` + `if (r < N_ROWS)`): si asserisce la GUARDIA,
      // non il nome scelto per la costante — il testo e' un dettaglio, il
      // confronto no.
      expect(src, `${name}: nessun confronto della riga con N`)
        .toMatch(/(?:<|>=)\s*(?:1537u|[A-Z_][A-Z0-9_]*)\b/);
      expect(rowMapping(src, 1537), `${name}: guardia non sulle 2 righe`)
        .toEqual({ scaled: true, covers2: true });
    });
  }
});

describe("vec4Rows2 — ramo subgroup", () => {
  it("abilita i subgroup e riduce la riga con subgroupAdd", () => {
    const src = SG();
    expect(src).toContain("enable subgroups;");
    expect(src).toContain("subgroupAdd(");
    // la direttiva `enable` deve precedere qualunque dichiarazione WGSL
    expect(src.indexOf("enable subgroups;")).toBe(0);
  });
});

describe("vec4Rows2 — ramo fallback (senza sg)", () => {
  it("non nomina i subgroup", () => {
    const src = FB();
    expect(src).not.toContain("enable subgroups;");
    expect(src).not.toContain("subgroupAdd(");
  });

  it("riduce ad albero in workgroup memory", () => {
    const src = FB();
    expect(src).toContain("var<workgroup> partial");
    expect(src).toContain("workgroupBarrier();");
    expect(src).toMatch(/stride\s*=\s*stride\s*>>\s*1u;/);
  });
});

describe("vec4Rows2 — invarianti comuni ai due rami", () => {
  for (const [name, gen] of [["sg", SG], ["fallback", FB]] as const) {
    it(`${name}: binding vec4 per qs e x`, () => {
      const src = gen();
      expect(src).toContain("array<vec4<u32>>");
      expect(src).toContain("array<vec4<f32>>");
      // il nome del binding e' libero (`qs4`, `x4`): quello che deve valere e'
      // che il binding 0 sia vec4<u32> e il 2 vec4<f32>, cioe' che le letture
      // siano vettoriali. Asserire l'identificatore avrebbe legato il gate a
      // una scelta di stile.
      expect(src).toMatch(/@group\(0\) @binding\(0\) var<storage, read> \w+: array<vec4<u32>>;/);
      expect(src).toMatch(/@group\(0\) @binding\(2\) var<storage, read> \w+: array<vec4<f32>>;/);
    });
  }
});

// ---------------------------------------------------------------------------
// 4. Guardie: le combinazioni non misurate devono esplodere
// ---------------------------------------------------------------------------

const base = { K: 2048, N: 1536, hasBias: false } as const;

// Le cinque combinazioni che gemvQuantWgsl deve rifiutare.
const REJECTED: Record<string, Vec4Opts> = {
  "q8_0 + vec4Rows2": { ...base, kind: "q8_0", vec4Rows2: true },
  "q4_1 + vec4Rows2": { ...base, kind: "q4_1", vec4Rows2: true },
  "hasBias + vec4Rows2": { ...base, kind: "q4_0", hasBias: true, vec4Rows2: true },
  "batch + vec4Rows2": { ...base, kind: "q4_0", batch: true, vec4Rows2: true },
  "scaledAccum + vec4Rows2": { ...base, kind: "q4_0", scaledAccum: true, vec4Rows2: true },
  "sg senza vec4Rows2": { ...base, kind: "q4_0", sg: true },
};

describe("vec4Rows2 — combinazioni rifiutate", () => {
  it("q8_0 e q4_1 non hanno il corpo vec4: throw", () => {
    expect(() => wgsl(REJECTED["q8_0 + vec4Rows2"])).toThrow();
    expect(() => wgsl(REJECTED["q4_1 + vec4Rows2"])).toThrow();
  });

  it("hasBias non e' supportato dal ramo a 2 righe: throw", () => {
    expect(() => wgsl(REJECTED["hasBias + vec4Rows2"])).toThrow();
  });

  it("batch non e' supportato dal ramo a 2 righe: throw", () => {
    expect(() => wgsl(REJECTED["batch + vec4Rows2"])).toThrow();
  });

  it("scaledAccum non e' supportato dal ramo a 2 righe: throw", () => {
    expect(() => wgsl(REJECTED["scaledAccum + vec4Rows2"])).toThrow();
  });

  it("sg senza vec4Rows2 non e' un ramo esistente: throw", () => {
    expect(() => wgsl(REJECTED["sg senza vec4Rows2"])).toThrow();
  });

  it("chi rifiuta il testo non dimezza la griglia: helper e generatore d'accordo", () => {
    // Altrimenti gemvQuantGrid({kind:'q8_0', vec4Rows2:true, N}) lancia ceil(N/2)
    // workgroup per un kernel che non esiste (o, peggio, per il kernel a 1 riga).
    for (const [name, opts] of Object.entries(REJECTED)) {
      let rows: number | "throw";
      try { rows = rowsPerWg(opts); } catch { rows = "throw"; }
      expect(rows, `${name}: gemvQuantWgsl lancia ma gemvQuantRowsPerWg dice ${String(rows)}`)
        .not.toBe(2);
      let g: [number, number] | "throw";
      try { g = grid(opts); } catch { g = "throw"; }
      if (g !== "throw") {
        expect(g, `${name}: griglia dimezzata per una combinazione rifiutata`)
          .toEqual(gemvGrid(base.N));
      }
    }
  });
});
