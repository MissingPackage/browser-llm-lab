// Microbench attention split in isolamento (fase 2 B2). Uso: node scripts/attn-bench.mjs
// (dev server attivo su BASE_URL, default :5199)
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[attnbench][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/engine.html?attnbench=1`, { waitUntil: "load" });
await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 600000, polling: 1000 },
);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const report = await page.evaluate(() => window.__report ?? null);
await browser.close();
if (status !== "done" || !report) { console.error("[attnbench] FALLITO:", status); process.exit(2); }
for (const c of report.cases)
  console.log(`[attnbench] ctx=${c.ctx}: fused ${(c.fusedMsPerOp * 1000).toFixed(1)} µs vs split ${(c.splitMsPerOpPair * 1000).toFixed(1)} µs = ${c.speedup.toFixed(2)}x — maxΔ cpu: fused ${c.fusedVsCpu.toExponential(2)}, split ${c.splitVsCpu.toExponential(2)}`);
const p = report.projection;
console.log(`[attnbench] proiezione ctx576: gpuBusy ${p.attrInputs.gpuBusy} − ${p.attnSavedMsPerToken.toFixed(2)} = ${p.gpuBusyNewMsPerToken.toFixed(2)} ms/tok ⇒ K=1 ${p.toksPerSecK1.toFixed(0)} tok/s, K=8 ${p.toksPerSecK8.toFixed(0)} tok/s`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
writeFileSync(`results/engine/attn-bench-4090-${ts}.json`, JSON.stringify({ ...report, deviceLabel: "4090-linux", ts }, null, 2));
console.log("[attnbench] scritto results/engine/attn-bench-4090-" + ts + ".json");
