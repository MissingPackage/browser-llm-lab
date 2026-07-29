// Simulazione fase B1: floor de-sync-only del prefill (per la soglia di spec).
// Uso: node scripts/prefill-sim.mjs   (dev server già attivo su BASE_URL)
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[sim][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/engine.html?prefillsim=1`, { waitUntil: "load" });

let last = "";
const poll = setInterval(async () => {
  try {
    const s = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    if (s !== last) { last = s; console.log("[sim]", s); }
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
if (status !== "done" || !report) { console.error("[sim] FALLITO:", status); process.exit(2); }
for (const v of report.variants) console.log(`[sim] ${v.label}: ${v.msPerToken.toFixed(2)} ms/token (match=${v.decodeMatch})`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
writeFileSync(`results/engine/prefill-sim-4090-${ts}.json`, JSON.stringify({ ...report, deviceLabel: "4090-linux", ts }, null, 2));
console.log("[sim] scritto results/engine/prefill-sim-4090-" + ts + ".json");
