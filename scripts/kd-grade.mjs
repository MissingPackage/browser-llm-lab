// GRADE INDIPENDENTE della fase 0 di engine-kernel-decode.
//
// Non rinegozia niente: legge il JSON di results/microbench/, prende p50 per
// cella, calcola lo speedup contro la cella "base" della stessa forma e lo
// confronta con le soglie PRE-REGISTRATE in
// docs/deep-dive/kernel-decode-fase0-prereg-2026-08-13.md. Le soglie sono
// costanti letterali qui sotto: se qualcuno le cambia, il diff lo mostra.
//
// Uso: node scripts/kd-grade.mjs [results/microbench/kernel-decode-....json]
import { readFileSync, readdirSync } from "node:fs";

const path = process.argv[2] ?? "results/microbench/" + readdirSync("results/microbench")
  .filter((f) => f.startsWith("kernel-decode-fase0-")).sort().pop();
const j = JSON.parse(readFileSync(path, "utf8"));
console.log(`[grade] file: ${path}`);
console.log(`[grade] device: ${j.deviceLabel} · host: ${j.hostState.declared} · ts: ${j.ts}`);
console.log(`[grade] prereg: ${j.prereg}\n`);

// --- SOGLIE PRE-REGISTRATE (copiate dal testo, non derivate dai dati) -------
const TH = {
  stopRule: 1.5,                       // P0
  attnBaseMsBand: [4.9, 11.5],         // P2
  attnBaseMsHalf: 4.1,                 // P2, ramo "molto piu' veloce"
  attnBestSpeedup: 4.0,                // P3
  gemvBaseGwsBand: [133 * 0.75, 133 * 1.25], // P4: 133 +/- 25%
  gemvBestSpeedup: 1.5,                // P4
  llamaCppGws: 500,                    // P4: riferimento esterno da NON superare
  gemvCacheResidentFloorGws: 133,      // P4
};

const cells = j.cells;
const pre = (c) => !c.variant.endsWith("-coldkv"); // celle PRE-REGISTRATE
const attn = cells.filter((c) => c.kernel === "attn-decode" && pre(c));
const gemv = cells.filter((c) => c.kernel === "gemv-q4_0");
const byVar = (list, v) => list.find((c) => c.variant === v);
const shapeKey = (c) => JSON.stringify(c.shape);
const fmt = (x, d = 3) => Number(x).toFixed(d);
const results = [];
const check = (id, pass, detail) => { results.push({ id, pass, detail }); console.log(`[grade] ${pass ? "PASS" : "FAIL"}  ${id}: ${detail}`); };

// speedup di ogni variante contro la "base" della stessa forma
const speedups = new Map();
for (const c of cells) {
  const base = cells.find((b) => b.kernel === c.kernel && shapeKey(b) === shapeKey(c) && b.variant === "base");
  if (base) speedups.set(`${c.kernel}|${c.variant}|${shapeKey(c)}`, base.msPerOp.p50 / c.msPerOp.p50);
}
const su = (c) => speedups.get(`${c.kernel}|${c.variant}|${shapeKey(c)}`);

// ---------------------------------------------------------------- P1 -------
const f = j.kdProbe.features;
check("P1.subgroups", f.subgroups.exposed && f.subgroups.compiles === true && f.subgroups.correct === true,
  `esposta=${f.subgroups.exposed} compila=${f.subgroups.compiles} corretto=${f.subgroups.correct} (sgSize ${j.kdProbe.subgroupSizeObserved})`);
check("P1.shader-f16-ASSENTE", f["shader-f16"].exposed === false,
  `esposta=${f["shader-f16"].exposed} — ${f["shader-f16"].note.slice(0, 90)}`);
check("P1.dot4I8Packed-PRESENTE", f.packed_4x8_integer_dot_product.exposed === true
  && f.packed_4x8_integer_dot_product.correct === true,
  `in wgslLanguageFeatures=${f.packed_4x8_integer_dot_product.exposed}, esegue corretto=${f.packed_4x8_integer_dot_product.correct}`);
// L'enunciato P1 sulla subgroup-matrix diceva "esposta ma NON istanziabile in
// configurazione u8/f32". Sono due fatti diversi e vanno graduati separati,
// altrimenti un booleano solo nasconde quale dei due e' caduto.
const sm = f["chromium-experimental-subgroup-matrix"];
const smCfg = JSON.parse(sm.note.replace(/^subgroupMatrixConfigs: /, ""));
check("P1.subgroup-matrix-NON-istanziabile", sm.exposed === true && sm.granted === false,
  `esposta=${sm.exposed} concessa=${sm.granted} (richiesta esplicitamente fra le optionalFeatures)`);
check("P1.subgroup-matrix-nessuna-config-con-risultato-f32",
  smCfg.length > 0 && !smCfg.some((c) => c.result === "f32"),
  `config esposte: ${smCfg.map((c) => `${c.component}->${c.result} ${c.M}x${c.N}x${c.K}`).join(", ")}`);

// ---------------------------------------------------------------- P2 -------
const aBase = byVar(attn, "base");
const t = aBase.msPerOp.p50;
check("P2.baseline-in-banda", t >= TH.attnBaseMsBand[0] && t <= TH.attnBaseMsBand[1],
  `base = ${fmt(t)} ms/dispatch (banda ${TH.attnBaseMsBand[0]}-${TH.attnBaseMsBand[1]}) · ${fmt(aBase.effectiveGBps, 2)} GB/s su byte unici · IQR ${fmt(aBase.msPerOp.iqr)}`);
check("P2.NON-molto-piu-veloce", t >= TH.attnBaseMsHalf,
  `base = ${fmt(t)} ms >= ${TH.attnBaseMsHalf} ms ⇒ la pendenza del motore E' questo kernel`);

// ---------------------------------------------------------------- P3 -------
const aBest = attn.filter((c) => c.variant !== "base").sort((a, b) => a.msPerOp.p50 - b.msPerOp.p50)[0];
check("P3.attn-best>=4.0x", su(aBest) >= TH.attnBestSpeedup,
  `migliore = "${aBest.variant}" ${fmt(aBest.msPerOp.p50)} ms ⇒ ${fmt(su(aBest), 2)}x (soglia ${TH.attnBestSpeedup}x) · ${fmt(aBest.effectiveGBps, 1)} GB/s unici, ${fmt(aBest.emittedGBps, 1)} GB/s emessi`);
const vSplit = byVar(attn, "split"), vGqa = byVar(attn, "gqa-stream"), vVec4 = byVar(attn, "vec4");
check("P3.causa-split-piu-veloce-di-gqa-e-vec4",
  vSplit.msPerOp.p50 < vGqa.msPerOp.p50 && vSplit.msPerOp.p50 < vVec4.msPerOp.p50,
  `split ${fmt(vSplit.msPerOp.p50)} · gqa-stream ${fmt(vGqa.msPerOp.p50)} · vec4 ${fmt(vVec4.msPerOp.p50)} (ms)`);

// ---------------------------------------------------------------- P4 -------
const head = gemv.filter((c) => c.shape.N === 248320);
const gBase = byVar(head, "base");
const gws = (c) => c.weightsPerSecond / 1e9;
check("P4.lm_head-base-in-banda", gws(gBase) >= TH.gemvBaseGwsBand[0] && gws(gBase) <= TH.gemvBaseGwsBand[1],
  `base lm_head = ${fmt(gws(gBase), 1)} G pesi/s (banda ${fmt(TH.gemvBaseGwsBand[0], 1)}-${fmt(TH.gemvBaseGwsBand[1], 1)})`);
const gBest = head.filter((c) => c.variant !== "base").sort((a, b) => a.msPerOp.p50 - b.msPerOp.p50)[0];
check("P4.lm_head-best>=1.5x", su(gBest) >= TH.gemvBestSpeedup,
  `migliore = "${gBest.variant}" ${fmt(gws(gBest), 1)} G pesi/s ⇒ ${fmt(su(gBest), 2)}x (soglia ${TH.gemvBestSpeedup}x)`);
check("P4.best-resta-sotto-llama.cpp-500", gws(gBest) < TH.llamaCppGws,
  `${fmt(gws(gBest), 1)} < ${TH.llamaCppGws} G pesi/s`);
const cacheRes = gemv.filter((c) => c.variant === "base" && c.shape.N !== 248320);
check("P4.forme-cache-resident-sopra-133", cacheRes.every((c) => gws(c) > TH.gemvCacheResidentFloorGws),
  cacheRes.map((c) => `K${c.shape.K}xN${c.shape.N} ${fmt(gws(c), 1)}`).join(" · ") + ` (soglia > ${TH.gemvCacheResidentFloorGws})`);

// ---------------------------------------------------------------- P5 -------
check("P5.attn-guadagna-piu-del-gemv", su(aBest) > su(gBest),
  `attn ${fmt(su(aBest), 2)}x vs gemv ${fmt(su(gBest), 2)}x ⇒ ordine delle fasi 1/2 ${su(aBest) > su(gBest) ? "CONFERMATO" : "DA INVERTIRE"}`);

// ---------------------------------------------------------------- P0 -------
const best = Math.max(...[...speedups.values()]);
check("P0.regola-di-stop-NON-scatta", best >= TH.stopRule,
  `migliore speedup dell'intera fase = ${fmt(best, 2)}x (soglia ${TH.stopRule}x) ⇒ la fase ${best >= TH.stopRule ? "NON chiude" : "CHIUDE"} il goal`);

// --------------------------------------------------------- integrita' ------
const ckBad = cells.filter((c) => c.checksumRelDiff !== null && c.checksumRelDiff > 1e-3);
check("GATE.checksum-entro-1e-3", ckBad.length === 0 && cells.every((c) => Number.isFinite(c.checksum) && c.checksum !== 0),
  `${cells.length} celle, max relDiff ${Math.max(...cells.map((c) => c.checksumRelDiff ?? 0)).toExponential(2)}, ${j.skipped.length} skipped`);
check("GATE.timing-timestamp-query", cells.every((c) => c.timingSource === "timestamp-query"),
  `sorgenti: ${[...new Set(cells.map((c) => c.timingSource))].join(", ")}`);

const nFail = results.filter((r) => !r.pass).length;
console.log(`\n[grade] ${results.length - nFail}/${results.length} enunciati confermati, ${nFail} caduti`);
console.log("[grade] TABELLA SPEEDUP (p50 base / p50 variante, stessa forma, stessa sessione)");
for (const c of cells) {
  if (c.variant === "base") continue;
  console.log(`  ${c.kernel.padEnd(11)} ${c.variant.padEnd(20)} ${JSON.stringify(c.shape).padEnd(52)} ${fmt(su(c), 2)}x`);
}
process.exit(0);
