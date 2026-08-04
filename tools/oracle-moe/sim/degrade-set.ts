// Degrade set della fase 4c (design 2026-08-04-engine-fase4c-residenza-design §1, §6, §8.2, §9).
// Sceglie e PINNA l'ordine con cui gli expert vengono degradati a Q3_K/Q2_K:
// dal piu' freddo al piu' caldo, sul pool blk.5-46, con ranking calcolato
// SOLO sui 6 prompt di derivazione (p4 e p7 restano campione ratificato).
// L'artefatto e' committabile e il suo sha256Order e' il `degradeSetSha256`
// che il design §6 vuole nell'header del file slab: se il set cambia, il file
// e' un altro modello.
//
// Node, nessuna dipendenza esterna.
// Uso: npx tsx tools/oracle-moe/sim/degrade-set.ts results/engine/moe-oracle/trace-<data>.jsonl.gz [out.json]
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { usageCounts, key, type TraceRow } from "./simulate.js";

// ---- costanti del design ------------------------------------------------

/** blk.1-4 (classe q4_1) fuori dal pool degradabile (§1): denseLead=1 => moeIdx 0..3. */
const EXCLUDED_MOE_LAYERS = 4;
/** prompt su cui si CALCOLA il ranking (§8.2 anti-leakage). */
const PROMPTS_RANKING = [0, 1, 2, 3, 5, 6];
/** campione ratificato: mai usato per ordinare, serve a misurare la generalizzazione. */
const PROMPTS_HELD_OUT = [4, 7];
/** i P della ladder del pilota CPU (§8 passo 0). */
const LADDER_P = [355, 530, 627, 935, 1024];
/** risparmio per expert, Q4_0 -> Q3_K a sezioni (§2.3). */
const SAVE_Q3K_BYTES = 1_253_376;
/** risparmio per expert, Q4_0 -> Q2_K (§2.3). */
const SAVE_Q2K_BYTES = 2_211_840;
/** range dichiarato nel design §1 per il fattore di generalizzazione held-out/in-sample. */
const DESIGN_RATIO_RANGE = [1.36, 1.78] as const;

const TIE_BREAK = "conteggio full crescente, poi key = moeLayer*64+expert crescente";
const MASS_DENOMINATOR = "tutte le selezioni dei 46 layer MoE";
const SHA256_ORDER_INPUT = 'sha256(utf8) di order.map(o => o.key).join(",") — la sola lista di chiavi, in ordine di degrado';

// ---- input --------------------------------------------------------------

const tracePath = process.argv[2];
if (!tracePath) {
  console.error("uso: degrade-set.ts <trace.jsonl.gz> [out.json]");
  process.exit(1);
}
const today = new Date().toISOString().slice(0, 10);
const outPath = process.argv[3] ?? `results/engine/moe-degrade-set-${today}.json`;

const traceBytes = readFileSync(tracePath);
const traceSha256 = createHash("sha256").update(traceBytes).digest("hex");
const raw = tracePath.endsWith(".gz") ? gunzipSync(traceBytes).toString("utf8") : traceBytes.toString("utf8");
const lines = raw.split("\n").filter((l) => l.length > 0);
const header = JSON.parse(lines[0]) as {
  kind: string; nLayer: number; denseLead: number; nMoe: number; nExpert: number;
  nExpertUsed: number; ggufSha256: string; llamaCppCommit: string; corpusHash: string; arch: string;
};
const rows: TraceRow[] = lines.slice(1).map((l) => JSON.parse(l) as TraceRow);
const { nMoe, nExpert, nExpertUsed, denseLead } = header;

// ---- sanity: il degrade set non si deriva da una traccia storta ----------

if (header.kind !== "moe-route-trace") throw new Error(`header kind ${header.kind}, atteso moe-route-trace`);
if (denseLead !== 1) throw new Error(`denseLead ${denseLead}: la mappa moeIdx -> blk assunta qui (blk = moeIdx+1) non vale piu'`);
if (nExpertUsed !== 4) throw new Error(`nExpertUsed ${nExpertUsed}, atteso 4`);
for (let r = 0; r < rows.length; r++) {
  if (rows[r].e.length !== nMoe * nExpertUsed) {
    throw new Error(`riga ${r}: e.length ${rows[r].e.length}, attesi ${nMoe * nExpertUsed}`);
  }
}
const promptsSeen = [...new Set(rows.map((r) => r.p))].sort((a, b) => a - b);
const promptsExpected = [0, 1, 2, 3, 4, 5, 6, 7];
if (promptsSeen.join(",") !== promptsExpected.join(",")) {
  throw new Error(`prompt presenti ${promptsSeen.join(",")}, attesi ${promptsExpected.join(",")}`);
}

// ---- pool degradabile ---------------------------------------------------

const poolKeys: number[] = [];
for (let s = EXCLUDED_MOE_LAYERS; s < nMoe; s++) {
  for (let e = 0; e < nExpert; e++) poolKeys.push(key(s, e, nExpert));
}
const POOL_EXPECTED = (nMoe - EXCLUDED_MOE_LAYERS) * nExpert; // 42 * 64
if (poolKeys.length !== POOL_EXPECTED || POOL_EXPECTED !== 2688) {
  throw new Error(`pool ${poolKeys.length} expert (atteso ${POOL_EXPECTED}, design §1 dice 2688): indicizzazione layer sbagliata`);
}
if (LADDER_P.some((p) => p > poolKeys.length)) {
  throw new Error(`ladder chiede P > pool (${poolKeys.length})`);
}

// ---- sottoinsiemi di righe ----------------------------------------------

const inRanking = new Set(PROMPTS_RANKING);
const inHeldOut = new Set(PROMPTS_HELD_OUT);
const rowsInSample = rows.filter((r) => inRanking.has(r.p));
const rowsHeldOut = rows.filter((r) => inHeldOut.has(r.p));
if (rowsInSample.length + rowsHeldOut.length !== rows.length) {
  throw new Error("partizione in-sample/held-out non copre la traccia");
}
const decode = (rs: TraceRow[]) => rs.filter((r) => r.ph === "d");

/** Un regime di misura: conteggi per chiave + denominatore = TUTTE le selezioni dei 46 layer. */
interface Regime {
  name: string;
  rows: number;
  counts: Map<number, number>;
  total: number;
}
const regime = (name: string, rs: TraceRow[]): Regime => {
  const counts = usageCounts(rs, nExpert, nMoe);
  const total = rs.length * nMoe * nExpertUsed;
  let sum = 0;
  for (const v of counts.values()) sum += v;
  if (sum !== total) throw new Error(`regime ${name}: somma conteggi ${sum} != selezioni attese ${total}`);
  return { name, rows: rs.length, counts, total };
};

const inFull = regime("inSample/full", rowsInSample);
const inDecode = regime("inSample/decode", decode(rowsInSample));
const outFull = regime("heldOut/full", rowsHeldOut);
const outDecode = regime("heldOut/decode", decode(rowsHeldOut));
const perPrompt = PROMPTS_HELD_OUT.map((p) => {
  const rs = rows.filter((r) => r.p === p);
  return { prompt: p, full: regime(`p${p}/full`, rs), decode: regime(`p${p}/decode`, decode(rs)) };
});

// ---- ordine di degrado ---------------------------------------------------
// FREDDEZZA = conteggio crescente sul regime FULL dei 6 prompt di ranking.
// Tie-break sulla chiave crescente: a parita' di conteggio (e a conteggio 0,
// che qui non capita ma capiterebbe su tracce piu' corte) l'ordine sarebbe
// altrimenti quello di inserimento nella Map — cioe' dipendente dal percorso
// di lettura. La chiave e' l'unico criterio riproducibile da chi rifa' il
// conto senza avere il nostro processo.
const order = poolKeys
  .map((k) => ({
    key: k,
    moeLayer: Math.floor(k / nExpert),
    blk: Math.floor(k / nExpert) + denseLead,
    expert: k % nExpert,
    cntFull: inFull.counts.get(k) ?? 0,
    cntDecode: inDecode.counts.get(k) ?? 0,
  }))
  .sort((a, b) => a.cntFull - b.cntFull || a.key - b.key);

const orderKeys = order.map((o) => o.key);
const sha256Order = createHash("sha256").update(orderKeys.join(","), "utf8").digest("hex");

// ---- massa esposta -------------------------------------------------------

interface MassPoint {
  /** frazione di selezioni che cadono sugli expert degradati */
  exposed: number;
  /** selezioni sui primi P expert dell'ordine */
  numerator: number;
  /** selezioni totali del regime, su tutti i 46 layer (blk.1-4 inclusi) */
  denominator: number;
}
const mass = (r: Regime, p: number): MassPoint => {
  let num = 0;
  for (let i = 0; i < p; i++) num += r.counts.get(orderKeys[i]) ?? 0;
  return { exposed: +(num / r.total).toFixed(6), numerator: num, denominator: r.total };
};

const ladder = LADDER_P.map((P) => ({
  P,
  inSample: { full: mass(inFull, P), decode: mass(inDecode, P) },
  heldOut: { full: mass(outFull, P), decode: mass(outDecode, P) },
}));

// fattore di generalizzazione: il design §1 lo dichiara 1.36-1.78x
const ladderRatio = ladder.map((l) => ({
  P: l.P,
  full: +(l.heldOut.full.exposed / l.inSample.full.exposed).toFixed(4),
  decode: +(l.heldOut.decode.exposed / l.inSample.decode.exposed).toFixed(4),
}));
const ratiosOutOfRange = ladderRatio.filter(
  (r) => r.decode < DESIGN_RATIO_RANGE[0] || r.decode > DESIGN_RATIO_RANGE[1],
);

// p4 e' 05-math-en, l'outlier sistematico del design §1: tenuto separato da p7.
const perPromptHeldOut = perPrompt.map((pp) => ({
  prompt: pp.prompt,
  rowsFull: pp.full.rows,
  rowsDecode: pp.decode.rows,
  points: LADDER_P.map((P) => ({ P, full: mass(pp.full, P), decode: mass(pp.decode, P) })),
}));

const bytes = LADDER_P.map((P) => ({
  P,
  q3kSavedBytes: P * SAVE_Q3K_BYTES,
  q3kSavedGiB: +((P * SAVE_Q3K_BYTES) / (1 << 30)).toFixed(4),
  q2kSavedBytes: P * SAVE_Q2K_BYTES,
  q2kSavedGiB: +((P * SAVE_Q2K_BYTES) / (1 << 30)).toFixed(4),
}));

// ---- artefatto -----------------------------------------------------------

const out = {
  kind: "moe-degrade-set",
  schemaVersion: 1,
  date: today,
  generatedBy: "tools/oracle-moe/sim/degrade-set.ts",
  tracePath,
  traceSha256,
  ggufSha256: header.ggufSha256,
  llamaCppCommit: header.llamaCppCommit,
  corpusHash: header.corpusHash,
  arch: header.arch,
  shape: { nLayer: header.nLayer, denseLead, nMoe, nExpert, nExpertUsed, rows: rows.length },
  promptsRanking: PROMPTS_RANKING,
  promptsHeldOut: PROMPTS_HELD_OUT,
  excludedLayers: "blk.1-4 (moeIdx 0-3)",
  poolSize: poolKeys.length,
  tieBreak: TIE_BREAK,
  sha256Order,
  sha256OrderInput: SHA256_ORDER_INPUT,
  massDenominator: MASS_DENOMINATOR,
  regimes: {
    inSampleFullRows: inFull.rows, inSampleDecodeRows: inDecode.rows,
    heldOutFullRows: outFull.rows, heldOutDecodeRows: outDecode.rows,
    note: "regime full = prefill+decode; i conteggi cntFull/cntDecode dell'ordine sono IN-SAMPLE (6 prompt di ranking)",
  },
  ladder,
  ladderRatio,
  ladderRatioDesignRange: DESIGN_RATIO_RANGE,
  ladderRatioOutOfRange: ratiosOutOfRange.map((r) => r.P),
  perPromptHeldOut,
  bytes,
  bytesBasis: {
    q3kSectionsPerExpert: SAVE_Q3K_BYTES,
    q2kPerExpert: SAVE_Q2K_BYTES,
    note: "design §2.3: slab Q4_0 5 308 416 B; Q3_K a sezioni 4 055 040 (risparmio 1 253 376); Q2_K 3 096 576 (risparmio 2 211 840)",
  },
  order,
};

// L'array `order` e' lungo 2688: una riga per elemento invece di sei, altrimenti
// il diff dell'artefatto e' illeggibile. Il contenuto e' quello di JSON.stringify.
const json = JSON.stringify(out, null, 2).replace(
  /\{\n\s+"key": (\d+),\n\s+"moeLayer": (\d+),\n\s+"blk": (\d+),\n\s+"expert": (\d+),\n\s+"cntFull": (\d+),\n\s+"cntDecode": (\d+)\n\s+\}/g,
  '{ "key": $1, "moeLayer": $2, "blk": $3, "expert": $4, "cntFull": $5, "cntDecode": $6 }',
);
writeFileSync(outPath, json + "\n");

// ---- report a stdout -----------------------------------------------------

const pct = (x: number) => (x * 100).toFixed(2) + "%";
console.log(`[degrade] traccia ${tracePath} (sha256 ${traceSha256.slice(0, 16)}…), ${rows.length} righe`);
console.log(`[degrade] pool ${poolKeys.length} expert (blk.5-46), ranking su p${PROMPTS_RANKING.join(",p")}, held-out p${PROMPTS_HELD_OUT.join(",p")}`);
console.log(`[degrade] righe: in-sample ${inFull.rows} (decode ${inDecode.rows}) | held-out ${outFull.rows} (decode ${outDecode.rows})`);
console.log(`[degrade] sha256Order ${sha256Order}`);
console.log("");
console.log("    P | in-sample dec | held-out dec |  ratio | in-sample full | held-out full | Q3_K risparmiati");
console.log("------+---------------+--------------+--------+----------------+---------------+-----------------");
for (let i = 0; i < ladder.length; i++) {
  const l = ladder[i], r = ladderRatio[i], b = bytes[i];
  console.log(
    ` ${String(l.P).padStart(4)} | ${pct(l.inSample.decode.exposed).padStart(13)} | ${pct(l.heldOut.decode.exposed).padStart(12)}` +
    ` | ${r.decode.toFixed(2).padStart(6)} | ${pct(l.inSample.full.exposed).padStart(14)} | ${pct(l.heldOut.full.exposed).padStart(13)}` +
    ` | ${b.q3kSavedBytes.toLocaleString("en-US").padStart(13)} B (${b.q3kSavedGiB.toFixed(2)} GiB)`,
  );
}
console.log("");
for (const pp of perPromptHeldOut) {
  const line = pp.points.map((pt) => `P${pt.P} ${pct(pt.decode.exposed)}`).join("  ");
  console.log(`[degrade] held-out p${pp.prompt} decode: ${line}`);
}
if (ratiosOutOfRange.length) {
  console.log(`[degrade] ATTENZIONE: ratio decode fuori dal range ${DESIGN_RATIO_RANGE[0]}-${DESIGN_RATIO_RANGE[1]}x del design §1 a P = ${ratiosOutOfRange.map((r) => `${r.P} (${r.decode.toFixed(2)}x)`).join(", ")} — riportato, NON aggiustato`);
}
console.log(`[degrade] scritto ${outPath}`);
