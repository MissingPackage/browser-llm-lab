// Runner identità prefillChunk (C3a fase 5 fetta b): copia il golden in
// public/, apre /glmprefill.html e scrive il report col gate di identità.
// Uso: node scripts/glm-prefill-run.mjs [--prompt 4] [--ngen 8] [--budget-gib 12]
//   [--out results/engine/...json] [--timeout-min 60]
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
const prompt = arg("prompt", "4");
const nGen = arg("ngen", "8");
const budget = arg("budget-gib", "12");
const out = arg("out", null);
const timeoutMin = Number(arg("timeout-min", "60"));

const ROOT = new URL("..", import.meta.url).pathname;
const GOLDEN = join(ROOT, "results/engine/golden/glm47flash/golden-glm47flash-q4_0-2026-07-31.json");
const GOLDEN_PUB = join(ROOT, "public/models/glm-conf-golden.json");
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-glmroute-profile");
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";
const host = hostState(arg("host-state", process.env.HOST_STATE ?? "undeclared"));

if (!existsSync(GOLDEN_PUB)) copyFileSync(GOLDEN, GOLDEN_PUB);
mkdirSync(PROFILE, { recursive: true });

const qs = new URLSearchParams({ prompt, ngen: nGen, budget });
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[glmprefill][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/glmprefill.html?${qs}`, { waitUntil: "load" });

let lastLive = "";
const t0 = Date.now();
for (;;) {
  if (Date.now() - t0 > timeoutMin * 60_000) {
    console.error("[glmprefill] TIMEOUT");
    await browser.close();
    process.exit(3);
  }
  const status = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
  const live = await page.evaluate(() => document.querySelector("#live")?.textContent ?? "");
  if (live && live !== lastLive) {
    console.log(`[glmprefill] ${live}`);
    lastLive = live;
  }
  if (status.startsWith("done") || status.startsWith("ERROR")) {
    const report = await page.evaluate(() => window.__report ?? null);
    await browser.close();
    if (!report) {
      console.error(`[glmprefill] ${status} — nessun report`);
      process.exit(2);
    }
    report.hostState = host.close();
    if (out) writeFileSync(join(ROOT, out), JSON.stringify(report, null, 1));
    const g = report.gate;
    console.log(
      `[glmprefill] ${status} — PASS=${g.pass}: hidden bit-eq ${g.hiddenBitEqual} (maxAbs ${g.hiddenMaxAbs}, diffs ${g.hiddenDiffs}) | ` +
      `logits bit-eq ${g.logitsBitEqual} (maxAbs ${g.logitsMaxAbs}, diffs ${g.logitsDiffs}) | argmax ${g.boundaryArgmaxEqual} | ` +
      `ids ${g.generatedIdsEqual} | routing mismatch ${g.routingMismatch}/${g.routingTot}`);
    console.log(
      `[glmprefill] prefill: seq ${report.prefillMs.sequential.toFixed(0)} ms vs chunked ${report.prefillMs.chunked.toFixed(0)} ms ` +
      `(speedup ${report.prefillMs.speedup.toFixed(2)}x su ${report.config.promptTokens} pos)`);
    process.exit(status === "done" ? 0 : status === "done-gate-fail" ? 4 : 2);
  }
  await new Promise((r) => setTimeout(r, 3000));
}
