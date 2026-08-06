// Runner bench GLM (C2 fase 6 slice 2): copia i golden in public/, apre
// /glmbench.html (profilo su disco, stesso OPFS di routing/conformance) e
// scrive il report con i gate tok/s. Uso:
//   node scripts/glm-bench-run.mjs [--prompt 6] [--ngen 64] [--reps 3]
//     [--budget-gib 11] [--attrib 1] [--out results/engine/...json]
//     [--timeout-min 120]
// --attrib N: repliche DEDICATE di attribuzione (telemetria liv.1+2 accesa;
// il loro wall NON entra nei gate). 0 le disattiva.
// Exit: 0 gate PASS, 4 gate FAIL (report scritto), 2 errore, 3 timeout.
import { chromium } from "playwright";
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { hostState } from "./lib/hoststate.mjs";

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

const qs = new URLSearchParams({ prompt, ngen: nGen, reps, budget, attrib });
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[glmbench][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/glmbench.html?${qs}`, { waitUntil: "load" });

let lastLive = "";
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
    if (out) writeFileSync(join(ROOT, out), JSON.stringify(report, null, 1));
    const g = report.gates;
    console.log(
      `[glmbench] ${status} — decode ${g.decodeMedian.toFixed(2)} tok/s (gate ${g.decodeGateToksPerSec}: ${g.decodePass ? "PASS" : "FAIL"}) ` +
      `— prefill ${g.prefillMedian.toFixed(2)} tok/s (gate ${g.prefillGateToksPerSec}: ${g.prefillPass ? "PASS" : "FAIL"})`);
    const dp = g.dispatchPlan;
    if (dp) {
      console.log(
        `[glmbench] piano dispatch: planned ${dp.planned} (+2 testa = ${dp.expectedWithHead}) vs misurato ${dp.measured} — ${dp.pass ? "PASS" : "DRIFT (FAIL)"}`);
    }
    const d = report.telemetry.decode;
    console.log(
      `[glmbench] telemetria decode: ${report.telemetry.dispatchesPerTokenPlanned} dispatch/token (piano), ${report.telemetry.syncsPerTokenExpected} sync/token (atteso), ` +
      `hit ${(100 * d.hitRate).toFixed(2)}% (retention ${d.retention == null ? "n/d" : (100 * d.retention).toFixed(2) + "%"}), stallo ${d.stallMsPerToken.toFixed(1)} ms/token ` +
      `(read ${d.readMsPerToken.toFixed(1)} + pack ${d.packMsPerToken.toFixed(1)} + upload ${d.uploadMsPerToken.toFixed(1)}), ` +
      `residuo ${d.residuoMsPerToken.toFixed(1)} ms/token`);
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
      for (const pk of a.projectionByK.filter(Boolean)) {
        console.log(`[glmbench]   proiezione sync batchato K=${pk.K}: ${pk.msPerToken.toFixed(1)} ms/token = ${pk.toksPerSec.toFixed(2)} tok/s`);
      }
    }
    process.exit(status === "done" ? 0 : status === "done-gate-fail" ? 4 : 2);
  }
  await new Promise((r) => setTimeout(r, 5000));
}
