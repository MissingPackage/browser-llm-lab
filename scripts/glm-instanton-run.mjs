// Instant-on (C3c fase 7, spec §6 + ruling docket item 1 opzione (a)):
// TTFT a FREDDO ≤ 1.25 × TTFT a CALDO della STESSA config (auto-ancorato).
//
// Definizione operativa (spec §6): OPFS popolata, VRAM vuota (sessione Chrome
// fresca), page cache OS FREDDA — eviction col protocollo del WP fase 1
// (fsync + posix_fadvise DONTNEED ×2 sui backing file OPFS del profilo,
// residenza fincore PRIMA/DOPO nel JSON: la freddezza si prova, non si
// asserisce). Composizione del TTFT (identica nei due rami, dichiarata):
//   ttft = buildMs (upload non-expert: router+shared+attention+head)
//        + warmup.prefill.ms (primo prefill della sessione)
//        + warmup.decode.msPerTokenP50 (primo token)
// Import/SHA-check esclusi: "OPFS popolata" è la precondizione, non il costo.
//
// Protocollo v2 (PRE-dichiarato, pattern B2): N sessioni FREDDE (eviction
// provata prima di ciascuna) e N CALDE, reps 1 l'una, config identica
// (sync + tier + prefetch, budget ctx-aware auto); mediane per ramo.
//
// Uso: node scripts/glm-instanton-run.mjs [--host-state ...] [--out ...]
import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const ROOT = new URL("..", import.meta.url).pathname;
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-glmroute-profile");
const hostStateArg = arg("host-state", "undeclared");
const OUT = arg("out", `results/engine/instanton-glm-4090-${new Date().toISOString().slice(0, 10)}.json`);

// costanti del modello di banda (src/engine/bandmodel.ts — stessa aritmetica,
// citata: coldTtftMs; qui inline perche' lo script e' .mjs)
const NON_EXPERT_BYTES = 1_354_078_720;
const PARK_BYTES = 15_678_308_352;
const BAND_COLD_SEQ_BPS = 3.51e9;
const BAND_WARM_BPS = 11.41e9;

// ---- eviction provata (protocollo WP fase 1) ----
const PY_EVICT = `
import os, sys, time
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    time.sleep(1)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
`;
const backingFiles = () =>
  execFileSync("find", [PROFILE, "-type", "f", "-size", "+512M"], { encoding: "utf8" })
    .split("\n").filter(Boolean);
const residentBytes = (files) => files.reduce((tot, f) =>
  tot + Number(execFileSync("fincore", ["--bytes", "--noheadings", "--output", "RES", f], { encoding: "utf8" }).trim()), 0);

const files = backingFiles();
if (files.length === 0) throw new Error(`nessun backing file >512M in ${PROFILE}: OPFS non popolata?`);
const evict = () => {
  const before = residentBytes(files);
  execFileSync("python3", ["-c", PY_EVICT, ...files]);
  const after = residentBytes(files);
  console.log(`[instanton] evict: ${files.length} file, resident ${(before / 1e9).toFixed(2)} GB -> ${(after / 1e6).toFixed(1)} MB`);
  if (after > 64 * 1024 * 1024) throw new Error(`eviction NON efficace (${after} B residenti)`);
  return { before, after };
};

// ---- sessioni con la stessa config di produzione ----
const bench = (label, reps, out) => {
  console.log(`[instanton] sessione ${label} (reps ${reps})…`);
  const r = spawnSync("node", [
    "scripts/glm-bench-run.mjs", "--budget-gib", "auto", "--select", "cpu",
    "--policy", "tier", "--prefetch", "inforward", "--prefill-batch", "1",
    "--reps", String(reps), "--attrib", "0", "--host-state", hostStateArg,
    "--out", out,
  ], { cwd: ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  if (r.status !== 0 && r.status !== 4) { // 4 = gate floor FAIL, atteso in sync
    console.error(r.stdout?.slice(-2000)); console.error(r.stderr?.slice(-1000));
    throw new Error(`bench ${label}: exit ${r.status}`);
  }
  return JSON.parse(readFileSync(join(ROOT, out), "utf8"));
};
const ttftOf = (j) => j.buildMs + j.warmup.prefill.ms + j.warmup.decode.msPerTokenP50;
const median = (v) => { const s = [...v].sort((a, b) => a - b); return s[s.length >> 1]; };

// Protocollo v2 (PRE-DICHIARATO prima delle run, pattern B2: la mediana di
// repliche è la difesa contro il rumore di sessione singola — il verdetto è
// quello delle mediane, qualunque sia): N sessioni fredde (eviction provata
// per ciascuna) e N calde, mediane per ramo.
const SESSIONS = Number(arg("sessions", "3"));
const evidence = [];
const colds = [], warms = [];
for (let i = 0; i < SESSIONS; i++) {
  evidence.push(evict());
  colds.push(bench(`C${i + 1} (fredda)`, 1, OUT.replace(/\.json$/, `-sessionC${i + 1}.json`)));
}
for (let i = 0; i < SESSIONS; i++) {
  warms.push(bench(`W${i + 1} (calda)`, 1, OUT.replace(/\.json$/, `-sessionW${i + 1}.json`)));
}
const C = colds[colds.map(ttftOf).indexOf(median(colds.map(ttftOf)))];
const W = warms[warms.map(ttftOf).indexOf(median(warms.map(ttftOf)))];
const ttftCold = median(colds.map(ttftOf));
const ttftWarm = median(warms.map(ttftOf));
console.log(`[instanton] fredde: ${colds.map((j) => ttftOf(j).toFixed(0)).join(" / ")} — calde: ${warms.map((j) => ttftOf(j).toFixed(0)).join(" / ")}`);
const ratio = ttftCold / ttftWarm;
const GATE = 1.25; // ruling docket item 1, opzione (a) — auto-ancorato
// predizione del modello (fase 6, coldTtftMs a overlap 0):
const uniqueBytes = Math.min(C.warmup.prefill.residency?.bytesRead ?? PARK_BYTES, PARK_BYTES);
const predColdMs = ttftWarm + (NON_EXPERT_BYTES / BAND_COLD_SEQ_BPS) * 1000
  + uniqueBytes * (1 / BAND_COLD_SEQ_BPS - 1 / BAND_WARM_BPS) * 1000;
const predErrPct = 100 * (predColdMs - ttftCold) / ttftCold;

const record = {
  schemaVersion: 1, kind: "instanton-ttft", date: new Date().toISOString(),
  ruling: "docket c3c item 1 opzione (a), 2026-08-09: ttftCold <= 1.25 x ttftWarm STESSA config (auto-ancorato); gap dai 4 s UX riportato (fase D)",
  config: {
    sameFor: "entrambe le sessioni: sync + tier + prefetch in-forward, prefill chunked, budget ctx-aware auto",
    composition: "ttft = buildMs (upload non-expert) + warmup.prefill.ms (primo prefill) + warmup.decode.msPerTokenP50 (primo token); import/SHA esclusi (OPFS popolata = precondizione)",
    profile: PROFILE,
  },
  protocolV2: `PRE-DICHIARATO: mediane su ${SESSIONS} sessioni per ramo (pattern B2), run fresche; il verdetto e' delle mediane`,
  coldReplicasTtftMs: colds.map((j) => +ttftOf(j).toFixed(0)),
  warmReplicasTtftMs: warms.map((j) => +ttftOf(j).toFixed(0)),
  coldnessProof: {
    protocol: "fsync + posix_fadvise(DONTNEED) x2 sui backing file OPFS del profilo (WP fase 1) PRIMA DI OGNI sessione fredda; VRAM vuota = sessione Chrome fresca",
    backingFiles: files.length, evictions: evidence,
  },
  cold: {
    ttftMs: +ttftCold.toFixed(0), buildMs: +C.buildMs.toFixed(0),
    warmupPrefillMs: +C.warmup.prefill.ms.toFixed(0), firstTokenMs: +C.warmup.decode.msPerTokenP50.toFixed(0),
    prefillBytesRead: C.warmup.prefill.residency?.bytesRead ?? null,
    budgetGiB: C.config.budgetGiB, reports: colds.map((_, i) => OUT.replace(/\.json$/, `-sessionC${i + 1}.json`)),
  },
  warm: {
    ttftMs: +ttftWarm.toFixed(0), buildMs: +W.buildMs.toFixed(0),
    warmupPrefillMs: +W.warmup.prefill.ms.toFixed(0), firstTokenMs: +W.warmup.decode.msPerTokenP50.toFixed(0),

    budgetGiB: W.config.budgetGiB, reports: warms.map((_, i) => OUT.replace(/\.json$/, `-sessionW${i + 1}.json`)),
  },
  gate: { threshold: GATE, ratio: +ratio.toFixed(3), pass: ratio <= GATE },
  gapUx4s: { budgetMs: 4000, coldFactor: +(ttftCold / 4000).toFixed(2), note: "obiettivo fase D, non gate" },
  bandModelPrediction: {
    source: "src/engine/bandmodel.ts coldTtftMs (fase 6), overlap 0",
    predictedColdTtftMs: +predColdMs.toFixed(0), measuredColdTtftMs: +ttftCold.toFixed(0),
    errPct: +predErrPct.toFixed(1), within15pct: Math.abs(predErrPct) <= 15,
  },
};
writeFileSync(join(ROOT, OUT), JSON.stringify(record, null, 1));
console.log(`[instanton] COLD ${ttftCold.toFixed(0)} ms vs WARM ${ttftWarm.toFixed(0)} ms — ratio ${ratio.toFixed(3)} (gate <= ${GATE}: ${ratio <= GATE ? "PASS" : "FAIL"})`);
console.log(`[instanton] modello: pred ${predColdMs.toFixed(0)} ms (err ${predErrPct.toFixed(1)}%, ±15%: ${Math.abs(predErrPct) <= 15 ? "PASS" : "FAIL"}) — gap UX 4 s: ${(ttftCold / 4000).toFixed(2)}x`);
console.log(`[instanton] scritto ${OUT}`);
process.exit(ratio <= GATE ? 0 : 4);
