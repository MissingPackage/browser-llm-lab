// Runner conformance logits GLM (C2 fase 6): copia i golden in public/,
// apre /glmconf.html (profilo su disco, stesso OPFS del routing) e scrive il
// report. Uso:
//   node scripts/glm-conf-run.mjs [--prompts 7] [--max-gen 128]
//     [--budget-gib 12] [--out results/engine/...json] [--timeout-min 240]
import { chromium } from "playwright";
import { watchGpuErrors, invalidPath } from "./lib/gpuerrors.mjs";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const prompts = arg("prompts", null);
const maxGen = arg("max-gen", null);
const budget = arg("budget-gib", "12");
const out = arg("out", null);
const timeoutMin = Number(arg("timeout-min", "240"));

const ROOT = new URL("..", import.meta.url).pathname;
const GOLDEN = join(ROOT, "results/engine/golden/glm47flash/golden-glm47flash-q4_0-2026-07-31.json");
const GOLDEN_PUB = join(ROOT, "public/models/glm-conf-golden.json");
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-glmroute-profile");
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";

if (!existsSync(GOLDEN_PUB)) copyFileSync(GOLDEN, GOLDEN_PUB);
mkdirSync(PROFILE, { recursive: true });

// Dump argmax cpuref-f64 del campione ratificato (fase 4d: il 256/256 diventa
// campo JSON — gateCpuref — invece di un confronto assemblato a mano). Il file
// merged si RIGENERA a ogni run (derivato, costa nulla); se i dump mancano il
// worker emette gateCpuref null e lo dichiara.
const CPUREF_DUMPS = [4, 7].map((p) => join(ROOT, `results/engine/logits-cpuref-p${p}-2026-08-01.json`));
const cpurefMerged = { kind: "glm-cpuref-argmax", source: [], prompts: {} };
for (const f of CPUREF_DUMPS) {
  if (!existsSync(f)) continue;
  const d = JSON.parse(readFileSync(f, "utf8"));
  cpurefMerged.source.push(f.slice(ROOT.length));
  cpurefMerged.prompts[d.prompt] = d.positions.map((x) => x.argmax);
}
writeFileSync(join(ROOT, "public/models/glm-cpuref-argmax.json"), JSON.stringify(cpurefMerged));

const qs = new URLSearchParams();
const prefetch = arg("prefetch", null);
if (prefetch) qs.set("prefetch", prefetch);
if (prompts) qs.set("prompts", prompts);
if (maxGen) qs.set("maxgen", maxGen);
qs.set("budget", budget);

const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
// docket item 24
const gpu = watchGpuErrors(page, "glmconf");
await page.goto(`${BASE_URL}/glmconf.html?${qs}`, { waitUntil: "load" });

let lastLive = "";
const t0 = Date.now();
for (;;) {
  if (Date.now() - t0 > timeoutMin * 60_000) {
    console.error("[glmconf] TIMEOUT");
    await browser.close();
    process.exit(3);
  }
  const status = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
  const live = await page.evaluate(() => document.querySelector("#live")?.textContent ?? "");
  if (live && live !== lastLive) {
    console.log(`[glmconf] ${live}`);
    lastLive = live;
  }
  if (status.startsWith("done") || status.startsWith("ERROR")) {
    const report = await page.evaluate(() => window.__report ?? null);
    await browser.close();
    if (!report) {
      console.error(`[glmconf] ${status} — nessun report`);
      process.exit(2);
    }
    if (gpu.dirty) {
      const bad = invalidPath(join(ROOT, out ?? "results/engine/glmconf-run.json"));
      writeFileSync(bad, JSON.stringify({ ...report, gpuErrors: gpu.errors, invalid: true }, null, 1));
      console.error(`[glmconf] RUN CONTAMINATA: ${gpu.errors.length} errori GPU — report in ${bad}`);
      process.exit(5);
    }
    if (out) writeFileSync(join(ROOT, out), JSON.stringify({ ...report, gpuErrors: gpu.errors }, null, 1));
    const g = report.gateGolden ?? {};
    console.log(`[glmconf] ${status} — top1 ${g.top1Ok}/${g.top1Tot} (${g.pct?.toFixed(3)}%) klMean ${report.secondary?.klMeanTop32?.toExponential(2)} maxDl ${report.secondary?.maxAbsDeltaLogit?.toFixed(3)}`);
    const c = report.gateCpuref;
    console.log(c
      ? `[glmconf] gateCpuref: ${c.agree}/${c.total} — ${c.pass === null ? "non valutato (nessuna posizione del campione)" : c.pass ? "PASS" : "FAIL (divergenza da cpuref-f64)"}`
      : "[glmconf] gateCpuref: dump non serviti (null, dichiarato)");
    process.exit(status === "done" ? 0 : status === "done-gate-fail" ? 4 : 2);
  }
  await new Promise((r) => setTimeout(r, 5000));
}
