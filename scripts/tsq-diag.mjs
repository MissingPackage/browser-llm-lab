// Simulazione fase B1: diagnosi timestamp-query liv.2 (matrice A/B).
// Uso: node scripts/tsq-diag.mjs   (dev server già attivo su BASE_URL)
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[diag][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/engine.html?tsqdiag=1`, { waitUntil: "load" });

let last = "";
const poll = setInterval(async () => {
  try {
    const s = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    if (s !== last) { last = s; console.log("[diag]", s); }
  } catch { /* chiusa */ }
}, 3000);

await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 1200000, polling: 1000 },
);
clearInterval(poll);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const report = await page.evaluate(() => window.__report ?? null);
await browser.close();
if (status !== "done" || !report) { console.error("[diag] FALLITO:", status); process.exit(2); }
for (const v of report.variants) console.log(`[diag] ${v.label}: idsMatch=${v.idsMatch} maxDlogit=${v.maxDlogit?.toFixed(4)} gpuMs=${v.gpuMsPerToken}`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
writeFileSync(`results/engine/tsq-diag-4090-${ts}.json`, JSON.stringify({ ...report, deviceLabel: "4090-linux", ts }, null, 2));
console.log("[diag] scritto results/engine/tsq-diag-4090-" + ts + ".json");
