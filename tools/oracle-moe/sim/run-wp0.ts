// WP-0: runner della simulazione tassa-di-replay (docket item 18b).
//
// Uso: npx tsx tools/oracle-moe/sim/run-wp0.ts \
//        results/engine/moe-oracle/trace-2026-07-31.jsonl.gz \
//        [--out results/engine/moe-oracle/wp0-replay-sim-<data>.json]
//
// Confronta, sulla traccia C1 (predizioni LOOKA reali incluse):
//   - opt / opt+lookaK : regime ottimistico a inserimento differito (WP-0)
//   - sync-lru         : ancora di continuita' col simulatore C1 (immediato)
//   - belady           : ceiling assoluto per-accesso
// e produce il COST MODEL dichiarato della tassa di replay + proiezioni tok/s.
// Le proiezioni sono PROIEZIONI: il modello e' scritto nel payload, non e'
// una misura del motore.
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { simulate, type TraceRow } from "./simulate.js";
import { simulateOptimistic, simulateBelady, crossTokenLocality, type Wp0Result } from "./wp0.js";

const tracePath = process.argv[2];
if (!tracePath) { console.error("uso: run-wp0.ts <trace.jsonl.gz> [--out FILE]"); process.exit(1); }
const outIdx = process.argv.indexOf("--out");
const outPath = outIdx >= 0 ? process.argv[outIdx + 1]
  : `results/engine/moe-oracle/wp0-replay-sim-${new Date().toISOString().slice(0, 10)}.json`;

const raw = tracePath.endsWith(".gz")
  ? gunzipSync(readFileSync(tracePath)).toString("utf8")
  : readFileSync(tracePath, "utf8");
const lines = raw.split("\n").filter((l) => l.length > 0);
const header = JSON.parse(lines[0]) as {
  nMoe: number; nExpert: number; ggufSha256: string; llamaCppCommit: string; corpusHash: string; arch: string;
};
const rows: TraceRow[] = lines.slice(1).map((l) => JSON.parse(l) as TraceRow);
const { nMoe, nExpert } = header;
const totalExperts = nMoe * nExpert;
const decodeRows = rows.filter((r) => r.ph === "d");
for (const r of decodeRows.slice(0, 1).concat(decodeRows.slice(-1)))
  if (!r.pr || r.pr.length !== nMoe * 8) throw new Error("predizioni LOOKA mancanti nella traccia");

// ---- budgets: 2596 = tetto MISURATO a sessione minima (probe it.19: 14.25
// GiB residenti − non-expert − KV@525 − riserva, in slot da 5.3255 MB);
// 2765 = proiezione no-session (non misurabile da qui); 2866 = massimo
// ARITMETICO (total − reserved), sopra il misurato: riportato come bound
// superiore, NON come punto di lavoro. Poi slab 12 GiB attuale (2419),
// punto C1 (2208) e regimi telefono 50%/25%.
const BUDGETS = [736, 1472, 2208, 2419, 2596, 2765, 2866];
const PREFETCH_KS = [0, 4, 8];
const WARMUP = 32;

// ---- COST MODEL dichiarato (tutte le fonti sono misure committate)
const COST = {
  syncLogitsMs: 7.6,            // floor readback misurato (it.1, probe indipendente)
  gpuBusyScenariosMs: [54.2, 35, 20], // oggi (it.14) / clock recuperati (proiezione) / parita' Qwen (state §6)
  fetchSerialMsPerMiss: 3.74,   // OPFS freddo p50 per expert (bound OS 2026-07-31)
  fetchCoalescedMsPerMiss: 2.66, // 5.33 MB a 2.0 GB/s dichiarati (coalescenza ottimista)
  totalModelLayers: 47,         // 1 denso + 46 MoE: replayFrac = (46 - firstMissMoeLayer) / 47
  note: "tax/token = E[nMiss]*fetchMs + P(dirty)*E[replayFrac|dirty]*gpuBusy; wallClean = gpuBusy + syncLogits (encode assunto in pipeline). PROIEZIONE, non misura.",
};

const replayFracFromHist = (hist: number[], dirty: number) => {
  if (!dirty) return 0;
  let s = 0;
  for (let l = 0; l < hist.length; l++) s += hist[l] * ((46 - l) / COST.totalModelLayers);
  return s / dirty;
};

interface Projection {
  gpuBusyMs: number;
  fetchModel: "serial" | "coalesced";
  taxMsPerToken: number;
  wallCleanMs: number;
  tokPerSec: number;
}

const project = (r: Wp0Result): Projection[] => {
  const st = r.steady;
  const eReplayFrac = replayFracFromHist(st.firstMissLayerHist, st.dirtyTokens);
  const out: Projection[] = [];
  for (const g of COST.gpuBusyScenariosMs) {
    for (const [fm, perMiss] of [["serial", COST.fetchSerialMsPerMiss], ["coalesced", COST.fetchCoalescedMsPerMiss]] as const) {
      const tax = st.meanMissPerToken * perMiss + st.pDirty * eReplayFrac * g;
      const wall = g + COST.syncLogitsMs;
      out.push({ gpuBusyMs: g, fetchModel: fm, taxMsPerToken: tax, wallCleanMs: wall, tokPerSec: 1000 / (wall + tax) });
    }
  }
  return out;
};

// ---- ancora di continuita' C1: lru immediato a 2208 deve riprodurre 0.9643
const anchor = simulate(rows, { budget: 2208, policy: "lru", nExpert, nMoe });
console.log(`[wp0] ancora C1 lru@2208: decodeHitRate ${anchor.decodeHitRate.toFixed(4)} (atteso ~0.9643)`);

// ---- matrice ottimistica
const optResults: Wp0Result[] = [];
for (const b of BUDGETS) {
  for (const k of PREFETCH_KS) {
    const r = simulateOptimistic(rows, { budget: b, nExpert, nMoe, prefetchK: k, warmupTokens: WARMUP });
    optResults.push(r);
    const s = r.steady;
    console.log(
      `[wp0] ${r.policy.padEnd(11)} b=${String(b).padStart(4)} hit=${(r.decodeHitRate * 100).toFixed(2)}% ` +
      `P(dirty)=${(s.pDirty * 100).toFixed(1)}% missE=${s.meanMissPerToken.toFixed(2)} p95=${s.p95MissPerToken} ` +
      `late=${(s.dirtyLateHalfFraction * 100).toFixed(0)}%`,
    );
  }
}

// ---- belady ceiling
const belady = BUDGETS.map((b) => {
  const r = simulateBelady(rows, b, nExpert, nMoe);
  console.log(`[wp0] belady      b=${String(b).padStart(4)} hit=${(r.decodeHitRate * 100).toFixed(2)}%`);
  return r;
});

// ---- localita' temporale cross-token
const locality = crossTokenLocality(rows, [1, 4, 16, 64], nExpert, nMoe);
for (const l of locality) console.log(`[wp0] localita' W=${String(l.window).padStart(2)}: ${(l.fraction * 100).toFixed(2)}%`);

// ---- proiezioni per il punto di lavoro di questo box (2866) e per i regimi
const PROJECT_BUDGETS = new Set([736, 1472, 2419, 2596, 2765, 2866]);
const projections = optResults
  .filter((r) => PROJECT_BUDGETS.has(r.budget))
  .map((r) => ({ policy: r.policy, budget: r.budget, projections: project(r) }));

for (const p of projections.filter((p) => p.budget === 2596 || p.budget === 2765)) {
  const best = p.projections.find((x) => x.gpuBusyMs === 54.2 && x.fetchModel === "serial")!;
  console.log(`[wp0] proiezione ${p.policy}@${p.budget} (kernel OGGI, fetch serial): tax ${best.taxMsPerToken.toFixed(1)} ms => ${best.tokPerSec.toFixed(2)} tok/s`);
}

const report = {
  kind: "wp0-optimistic-replay-sim",
  schemaVersion: 1,
  date: new Date().toISOString(),
  source: {
    trace: tracePath,
    ggufSha256: header.ggufSha256,
    llamaCppCommit: header.llamaCppCommit,
    corpusHash: header.corpusHash,
    arch: header.arch,
  },
  shape: { nMoe, nExpert, totalExperts, rows: rows.length, decodeRows: decodeRows.length, warmupTokens: WARMUP },
  semantics: "decode a inserimento differito al confine di token (1 submit/token); prefill sincrono; repair = miss inseriti al confine; prefetch LOOKA (pr della traccia, K primi) inserito al confine",
  costModel: COST,
  anchorC1: { policy: "lru", budget: 2208, decodeHitRate: anchor.decodeHitRate, expected: 0.9643 },
  optimistic: optResults,
  belady,
  crossTokenLocality: locality,
  projections,
};
writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(`[wp0] report: ${outPath}`);
