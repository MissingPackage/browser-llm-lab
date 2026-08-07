// Fase 5 C3b: confronto della tassa di replay MISURATA (bench optimistic)
// con la PROIEZIONE WP-0 allo stesso budget slot. Uso:
//   node scripts/wp0-compare.mjs [--out results/engine/wp0-vs-measured-....json]
// Unita' verificate (journal it.4/it.5): replayFrac ha la STESSA definizione
// nei due mondi — sim (46 - m)/47 con m indice MoE del primo sporco; motore
// (47 - l)/47 con l = m + 1 assoluto ⇒ identiche. La differenza DICHIARATA:
// il sim non modella la cascata dei round (selezioni fisse dalla traccia),
// il motore si' (item 7) — i termini si confrontano dove confrontabili.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const SIM = join(ROOT, "results/engine/moe-oracle/wp0-replay-sim-2026-08-06.json");
const BENCH = join(ROOT, "results/engine/bench-glm-4090-b12-optimistic-2026-08-07.json");
const OUT = arg("out", "results/engine/wp0-vs-measured-2026-08-07.json");
const TOL = 0.25; // [ASSUMED] contratto C3b: entro = modello confermato

const sim = JSON.parse(readFileSync(SIM, "utf8"));
const bench = JSON.parse(readFileSync(BENCH, "utf8"));

// ---- budget slot: la riga sim piu' vicina al misurato, delta dichiarato ----
const slotsMeas = bench.optimistic.slots.q4_0 + bench.optimistic.slots.q4_1;
const rows = sim.optimistic.filter((o) => o.policy === "opt");
const row = rows.reduce((a, b) => (Math.abs(b.budget - slotsMeas) < Math.abs(a.budget - slotsMeas) ? b : a));
const st = row.steady;

// E[replayFrac|dirty] del sim, ricalcolato con la STESSA formula di run-wp0
const eReplayFracSim = st.firstMissLayerHist.reduce((s, n, l) => s + n * ((46 - l) / 47), 0) / st.dirtyTokens;

// ---- misure dal bench ----
const meas = {
  pDirty: bench.optimistic.pDirty,
  replaysPerToken: bench.optimistic.replaysPerToken,
  replayFracPerRound: bench.optimistic.replayFracPerDirty, // media PER ROUND (nome storico del campo)
  missPerToken: bench.telemetry.decode.missesPerToken,     // ensure del repair (headline, tutti i round)
  repairMsPerToken: bench.optimistic.repairMsPerToken,
  gpuBusyMsPerToken: bench.attribution2.gpuBusyMsPerToken,
  wallMsPerTokenHeadline: 1000 / bench.decodeToksPerSec.median,
  tailWaitMsPerToken: bench.attribution2.misure.tailWaitMsPerToken,
  decodeToksPerSec: bench.decodeToksPerSec.median,
};

// ---- confronti, campo per campo ----
const cmp = (name, simV, measV, note) => {
  const ratio = simV === 0 ? null : measV / simV;
  return {
    name, sim: simV, measured: measV, ratio,
    withinTol: ratio !== null && Math.abs(ratio - 1) <= TOL,
    note,
  };
};
// termine GPU del replay: il sim ha 1 round (pDirty*frac), il motore N round
// (replays/token * frac/round) — si confronta il TERMINE, che e' cio' che
// moltiplica gpuBusy nella tassa.
const replayTermSim = st.pDirty * eReplayFracSim;
const replayTermMeas = meas.replaysPerToken * meas.replayFracPerRound;
// tassa: formula del sim coi SUOI input a gpuBusy MISURATO, contro la tassa
// misurata = wall - wallClean(gpuBusy misurato + syncLogits del cost model).
const taxSimAtMeasuredGpu = st.meanMissPerToken * sim.costModel.fetchSerialMsPerMiss
  + st.pDirty * eReplayFracSim * meas.gpuBusyMsPerToken;
const wallClean = meas.gpuBusyMsPerToken + sim.costModel.syncLogitsMs;
const taxMeasured = meas.wallMsPerTokenHeadline - wallClean;
// proiezione tok/s pura del sim (2419, gpuBusy 54.2 serial) vs misurato
const projPure = sim.projections
  .find((p) => p.policy === "opt" && p.budget === row.budget)
  .projections.find((q) => q.gpuBusyMs === 54.2 && q.fetchModel === "serial");

// seconda derivazione della tassa: wallClean col syncLogits MISURATO da
// questa run (probe 0.08 ms) invece del 7.6 dell'era C3a — la differenza
// (~7.5 ms) e' overhead CPU/event-loop che il sim assume "in pipeline".
const taxMeasuredStrict = meas.wallMsPerTokenHeadline - (meas.gpuBusyMsPerToken + bench.syncFloorProbe.mapRoundTripMs.median);

const comparisons = [
  cmp("pDirty", st.pDirty, meas.pDirty,
    "quota token con >= 1 miss; il misurato include l'effetto dell'ordine di preload (landmine it.2) e della LRU reale vs LruFast"),
  cmp("replayFracPerRound", eReplayFracSim, meas.replayFracPerRound,
    "frazione di layer rigiocati per round: unita' identiche (sim (46-m)/47, motore (47-l)/47) — il confronto standalone della 'frazione layer ripetuti' del done-when"),
  cmp("missPerToken", st.meanMissPerToken, meas.missPerToken,
    "IL SIM NON MODELLA LA CASCATA: i suoi miss sono le sole selezioni della traccia; i misurati includono i fetch dei round successivi (ri-selezioni a valle del repair, docket item 7) — fuori tolleranza ATTESO, spiegazione nel journal it.5"),
  cmp("replayGpuTerm (pDirty*frac vs replays/token*frac)", replayTermSim, replayTermMeas,
    "il moltiplicatore di gpuBusy nella tassa: sim a 1 round, motore a N round — confrontato il termine aggregato"),
  cmp("taxMsPerToken (formula sim a gpuBusy misurato vs wall-wallClean)", taxSimAtMeasuredGpu, taxMeasured,
    "wallClean = gpuBusy misurato + syncLogits 7.6 del cost model; il misurato assorbe encode/event-loop non modellati"),
  cmp("taxMsPerToken STRICT (wallClean col probe misurato 0.08 ms)", taxSimAtMeasuredGpu, taxMeasuredStrict,
    "la lettura severa: syncLogits dal probe di QUESTA run — il delta vs la riga sopra (~7.5 ms/token) e' l'overhead CPU/event-loop che il cost model assume 'in pipeline' e che il wall reale contiene; se FUORI, la spiegazione e' questa, dichiarata"),
  cmp("tokPerSec (proiezione pura sim@54.2 serial vs misurato)", projPure.tokPerSec, meas.decodeToksPerSec,
    "il misurato batte la proiezione perche' gpuBusy e' sceso 54.2 -> 39.4 (clock recovery, previsto da it.1 e non modellato dal sim)"),
];

const out = {
  kind: "wp0-vs-measured", schemaVersion: 1, date: "2026-08-07",
  inputs: {
    sim: "results/engine/moe-oracle/wp0-replay-sim-2026-08-06.json",
    bench: "results/engine/bench-glm-4090-b12-optimistic-2026-08-07.json",
  },
  budget: {
    simSlots: row.budget, measuredSlots: slotsMeas, deltaSlots: slotsMeas - row.budget,
    note: "riga sim piu' vicina; il delta (q4_1-first vs riparto proporzionale) e' dichiarato, non corretto",
  },
  toleranza: TOL,
  eReplayFracSim,
  comparisons,
  verdict: {
    within: comparisons.filter((c) => c.withinTol).map((c) => c.name),
    outside: comparisons.filter((c) => !c.withinTol).map((c) => c.name),
    note: "contratto C3b: entro ±25% il modello e' confermato; fuori, la spiegazione nel journal e' parte del done-when (it.5)",
  },
  caveats: [
    "il sim non modella la cascata dei round (selezioni fisse dalla traccia C1): missPerToken e replay term misurati la includono",
    "LRU del sim = LruFast, non byte-identica alla nostra; ordine di preload reale = (layer, expert) crescente (landmine it.2)",
    "gpuBusy misurato 39.4 vs scenario sim 54.2: clock recovery reale — le proiezioni pure del sim vanno lette a scenario, non a valore",
    "syncLogitsMs 7.6 del cost model e' dell'era C3a; il probe quiescente di questa run misura 0.08 ms round-trip — la tassa misurata e' quindi un limite SUPERIORE",
  ],
};
writeFileSync(join(ROOT, OUT), JSON.stringify(out, null, 1));
for (const c of comparisons) {
  console.log(`[wp0cmp] ${c.name}: sim ${c.sim.toFixed(3)} vs meas ${c.measured.toFixed(3)} (x${c.ratio.toFixed(3)}) ${c.withinTol ? "IN" : "FUORI"}`);
}
console.log(`[wp0cmp] verdict: ${out.verdict.within.length} entro ±25%, ${out.verdict.outside.length} fuori (${out.verdict.outside.join("; ")})`);
