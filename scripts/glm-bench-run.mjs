// Runner bench GLM (C2 fase 6 slice 2): copia i golden in public/, apre
// /glmbench.html (profilo su disco, stesso OPFS di routing/conformance) e
// scrive il report con i gate tok/s. Uso:
//   node scripts/glm-bench-run.mjs [--prompt 6] [--ngen 64] [--reps 3]
//     [--budget-gib 11|auto] [--attrib 1] [--select cpu|optimistic]
//     [--prefill-batch 0|1] [--prefetch inforward] [--policy tier]
//     [--park-frac F] [--ctx-max N] [--host-state quiescent|...]
//     [--out results/engine/...json] [--timeout-min 120]
// --attrib N: repliche DEDICATE di attribuzione (telemetria liv.1+2 accesa;
// il loro wall NON entra nei gate). 0 le disattiva.
// --prefill-batch e' un BOOLEANO (0/1), non una M: la M del prefill a chunk e'
// la costante GLM_PREFILL_M nel worker, e la pagina legge `prefillbatch==="1"`.
// Passare "16" lascia il prefill SEQUENZIALE e il report lo dice nel campo
// config.prefillPath — che e' l'unico posto dove ci si accorge dello sbaglio
// (costato due run in it.40, dove il riferimento da riprodurre e' a M=16).
// Exit: 0 gate PASS, 4 gate FAIL (report scritto), 2 errore, 3 timeout.
// ATTENZIONE (it.40): i gate del runner sono il FLOOR CPU di llama.cpp
// (13.43/56.58) e lo strutturale <=2 submit/token — NON la banda di
// non-regressione. Il riferimento b12 optimistic 2026-08-09 li fallisce
// entrambi, quindi exit 4 e' l'esito NORMALE di questa run: la
// non-regressione si legge confrontando le mediane col riferimento, non
// dall'exit code.
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { hostState } from "./lib/hoststate.mjs";
import { watchGpuErrors, invalidPath } from "./lib/gpuerrors.mjs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const prompt = arg("prompt", "6");
const nGen = arg("ngen", "64");
const reps = arg("reps", "3");
const budget = arg("budget-gib", "11");
const attrib = arg("attrib", "1");
const out = arg("out", null);
const timeoutMin = Number(arg("timeout-min", "120"));
// hostState (schema 4d): --host-state quiescent per le run di gate; il default
// "undeclared" dice che nessuno ha controllato l'host (informazione, non errore)
const host = hostState(arg("host-state", process.env.HOST_STATE ?? "undeclared"));

const ROOT = new URL("..", import.meta.url).pathname;
const GOLDEN = join(ROOT, "results/engine/golden/glm47flash/golden-glm47flash-q4_0-2026-07-31.json");
const GOLDEN_PUB = join(ROOT, "public/models/glm-conf-golden.json");
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-glmroute-profile");
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";

if (!existsSync(GOLDEN_PUB)) copyFileSync(GOLDEN, GOLDEN_PUB);
mkdirSync(PROFILE, { recursive: true });

const prefillBatch = arg("prefill-batch", "0");
// C3b fase 4: --select optimistic (decode a 1 submit + repair/replay)
const select = arg("select", "cpu");
// C3c fase 6: meccanismi del regime sync ai budget stretti
const prefetchArg = arg("prefetch", null);
const policyArg = arg("policy", null);
const parkFracArg = arg("park-frac", null);
// C3c fase 3: --budget-gib auto ⇒ tetto allocabile MISURATO ORA con nvidia-smi
// (total − used − reserved) e budget calcolato dalla formula ctx-aware nel
// worker; --ctx-max N alloca KV/partials per contesto lungo.
const ctxMaxArg = arg("ctx-max", null);
const smiMem = () => {
  const [total, used, reserved] = execFileSync("nvidia-smi",
    ["--query-gpu=memory.total,memory.used,memory.reserved", "--format=csv,noheader,nounits"],
    { encoding: "utf8" }).trim().split(",").map(Number);
  return { totalMiB: total, usedMiB: used, reservedMiB: reserved };
};
let ceilingMeasured = null;
const qs = new URLSearchParams({ prompt, ngen: nGen, reps, attrib, prefillbatch: prefillBatch, select });
if (prefetchArg) qs.set("prefetch", prefetchArg);
if (policyArg) qs.set("policy", policyArg);
if (parkFracArg) qs.set("parkfrac", parkFracArg);
if (budget !== "auto") qs.set("budget", budget);
if (ctxMaxArg) qs.set("ctxmax", ctxMaxArg);
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
// docket item 24
const gpu = watchGpuErrors(page, "glmbench");
if (budget === "auto") {
  // il ceiling si misura A CHROME LANCIATO: il footprint VRAM del browser
  // (~200-300 MiB di compositor) deve stare dentro "used", non dentro il
  // budget — stessa contabilita' del probe it.19 (Chrome incluso in used)
  await new Promise((r) => setTimeout(r, 2000));
  ceilingMeasured = smiMem();
  const ceilingBytes = (ceilingMeasured.totalMiB - ceilingMeasured.usedMiB - ceilingMeasured.reservedMiB) * 2 ** 20;
  qs.set("ceiling", String(ceilingBytes));
  console.log(`[glmbench] ceiling misurato (post-lancio Chrome): ${JSON.stringify(ceilingMeasured)} => ${(ceilingBytes / 2 ** 30).toFixed(2)} GiB allocabili`);
}
await page.goto(`${BASE_URL}/glmbench.html?${qs}`, { waitUntil: "load" });

let lastLive = "";
let vramPeakMiB = 0;
const t0 = Date.now();
for (;;) {
  if (Date.now() - t0 > timeoutMin * 60_000) {
    console.error("[glmbench] TIMEOUT");
    await browser.close();
    process.exit(3);
  }
  const status = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
  const live = await page.evaluate(() => document.querySelector("#live")?.textContent ?? "");
  if (live && live !== lastLive) {
    console.log(`[glmbench] ${live}`);
    lastLive = live;
  }
  if (status.startsWith("done") || status.startsWith("ERROR")) {
    const report = await page.evaluate(() => window.__report ?? null);
    await browser.close();
    if (!report) {
      console.error(`[glmbench] ${status} — nessun report`);
      process.exit(2);
    }
    report.hostState = host.close(); // schema 4d: dichiarato + campioni smi before/after
    // C3c fase 3: ceiling misurato al lancio (se --budget-gib auto) + picco
    // VRAM osservato durante la run (campionato ogni 5 s dal poll)
    if (ceilingMeasured) report.allocCeilingMeasured = ceilingMeasured;
    if (vramPeakMiB > 0) report.vramPeakMiB = vramPeakMiB;
    if (gpu.dirty) {
      const bad = invalidPath(join(ROOT, out ?? "results/engine/glmbench-run.json"));
      writeFileSync(bad, JSON.stringify({ ...report, gpuErrors: gpu.errors, invalid: true }, null, 1));
      console.error(`[glmbench] RUN CONTAMINATA: ${gpu.errors.length} errori GPU — report in ${bad}`);
      process.exit(5);
    }
    if (out) writeFileSync(join(ROOT, out), JSON.stringify({ ...report, gpuErrors: gpu.errors }, null, 1));
    const g = report.gates;
    console.log(
      `[glmbench] ${status} — decode ${g.decodeMedian.toFixed(2)} tok/s (gate ${g.decodeGateToksPerSec}: ${g.decodePass ? "PASS" : "FAIL"}) ` +
      `— prefill ${g.prefillMedian.toFixed(2)} tok/s (gate ${g.prefillGateToksPerSec}: ${g.prefillPass ? "PASS" : "FAIL"})`);
    const dp = g.dispatchPlan;
    if (dp) {
      console.log(
        `[glmbench] piano dispatch: planned ${dp.planned} (+2 testa = ${dp.expectedWithHead}) vs misurato ${dp.measured} — ${dp.pass ? "PASS" : "DRIFT (FAIL)"}`);
    }
    // gate STRUTTURALE C3b (solo select=optimistic)
    const sg = g.structural;
    if (sg) {
      console.log(
        `[glmbench] gate strutturale C3b: ${sg.syncsPerToken.toFixed(3)} sync / ${sg.submitsPerToken.toFixed(3)} submit per token ` +
        `(gate <= ${sg.gate}: ${sg.pass ? "PASS" : "FAIL"}) — sync path C3a: ${sg.vsC3aSyncPath.syncsPerToken}/token`);
    }
    const os = report.optimistic;
    if (os) {
      console.log(
        `[glmbench] ottimistico: slot ${os.slots.q4_0}+${os.slots.q4_1} = ${(100 * os.residencyRatio).toFixed(1)}% del parco, ` +
        `P(dirty) ${os.pDirty === null ? "n/d" : (100 * os.pDirty).toFixed(1) + "%"}, ` +
        `replayFrac|dirty ${os.replayFracPerDirty === null ? "n/d" : (100 * os.replayFracPerDirty).toFixed(1) + "%"}, ` +
        `repair ${os.repairMsPerToken === null ? "n/d" : os.repairMsPerToken.toFixed(1) + " ms/token"}`);
    }
    const d = report.telemetry.decode;
    console.log(
      `[glmbench] telemetria decode: ${report.telemetry.dispatchesPerTokenPlanned} dispatch/token (piano), ${report.telemetry.syncsPerTokenExpected} sync/token (atteso), ` +
      `hit ${(100 * d.hitRate).toFixed(2)}% (retention ${d.retention == null ? "n/d" : (100 * d.retention).toFixed(2) + "%"}), stallo ${d.stallMsPerToken.toFixed(1)} ms/token ` +
      `(read ${d.readMsPerToken.toFixed(1)} + pack ${d.packMsPerToken.toFixed(1)} + upload ${d.uploadMsPerToken.toFixed(1)}), ` +
      `residuo ${d.residuoMsPerToken.toFixed(1)} ms/token`);
    // Il regime di lettura va DETTO a chi guarda la console, non solo scritto
    // nel JSON: e' la differenza fra 15,3 e 11,3 tok/s sullo stesso codice
    // (it.20). `os-cache` = questo decode ha letto dalla RAM del sistema, e non
    // e' confrontabile con un riferimento preso a cache fredda.
    console.log(
      `[glmbench] regime di lettura: ${d.readRegime ?? "n/d"}`
      + (d.readGiBs == null ? "" : ` (${d.readGiBs.toFixed(2)} GiB/s)`)
      + (d.readRegime === "os-cache"
        ? " — ATTENZIONE: i byte NON sono arrivati dal disco. Questo decode non e' confrontabile con uno a cache fredda."
        : ""));
    // gap dalla funzione obiettivo: NON gate, ma obbligatorio nel report (C3a)
    const o = report.objective;
    if (o) {
      console.log(
        `[glmbench] obiettivo: decode ${o.gapDecode30.measuredToksPerSec.toFixed(2)} vs 30 tok/s (gap ${o.gapDecode30.factor?.toFixed(2)}x` +
        `, thinking 60: ${o.gapDecode60.factor?.toFixed(2)}x) — TTFT ${(o.gapTtft4s.measuredMs / 1000).toFixed(2)} s vs 4 s ` +
        `(gap ${o.gapTtft4s.factor.toFixed(2)}x)`);
    } else {
      console.log("[glmbench] ATTENZIONE: report senza sezione objective (checklist C3a: assenza = FAIL)");
    }
    const a = report.attribution2;
    if (a) {
      console.log(
        `[glmbench] attribuzione decode: wall ${a.wallMsPerToken.toFixed(1)} = gpuBusy ${a.gpuBusyMsPerToken?.toFixed(1) ?? "n/d"} ` +
        `+ stallo ${a.stallResidenzaMsPerToken.toFixed(1)} + sync/CPU ${a.syncCpuMsPerToken?.toFixed(1) ?? "n/d"} ms/token ` +
        `(fuori GPU ${a.quotaFuoriGpu === null ? "n/d" : (100 * a.quotaFuoriGpu).toFixed(1) + "%"}; ` +
        `MISURATI: ${a.dispatchesPerTokenMeasured.toFixed(0)} dispatch, ${a.routerSyncsPerToken.toFixed(1)} sync, ${a.submitsPerToken.toFixed(1)} submit per token)`);
      const p = report.syncFloorProbe;
      console.log(
        `[glmbench] floor sync: mapAsync ${p.mapRoundTripMs.median.toFixed(2)} ms × ${a.routerSyncsPerToken.toFixed(0)} = ` +
        `${p.floorMsPerToken.toFixed(1)} ms/token irriducibili senza cambiare meccanismo`);
      // In optimistic non ci sono sync router da batchare: projectionByK è null
      // e il crash qui mascherava l'exit code del gate (report già scritto).
      for (const pk of (a.projectionByK ?? []).filter(Boolean)) {
        console.log(`[glmbench]   proiezione sync batchato K=${pk.K}: ${pk.msPerToken.toFixed(1)} ms/token = ${pk.toksPerSec.toFixed(2)} tok/s`);
      }
    }
    process.exit(status === "done" ? 0 : status === "done-gate-fail" ? 4 : 2);
  }
  // picco VRAM osservato (spec c3c §2: il report del regime ctx-aware mostra
  // budget calcolato E VRAM di picco) — campione a ogni poll, costo ~nulla
  try { vramPeakMiB = Math.max(vramPeakMiB, smiMem().usedMiB); } catch { /* smi assente: resta 0 */ }
  await new Promise((r) => setTimeout(r, 5000));
}
