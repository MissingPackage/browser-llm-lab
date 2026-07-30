// Gate token-identity multi-step (fase 4 B2): K>1 vs per-token, greedy, ≥256
// token IDENTICI. Uso: node scripts/token-identity.mjs (dev server su BASE_URL)
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[idcheck][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/engine.html?idcheck=1`, { waitUntil: "load" });
await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 900000, polling: 1000 },
);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const report = await page.evaluate(() => window.__report ?? null);
await browser.close();
if (status !== "done" || !report) { console.error("[idcheck] FALLITO:", status); process.exit(2); }
for (const c of report.checks) console.log(`[idcheck] ${c.name}: match=${c.match}${c.divergeAt !== null ? ` divergeAt=${c.divergeAt}` : ""}`);
const t = report.msPerTokenInformal;
console.log(`[idcheck] ms/token informali: per-token ${t.perToken.toFixed(2)} · K=8 ${t.k8.toFixed(2)} · K=5 ${t.k5.toFixed(2)} · K=1-batch ${t.k1.toFixed(2)}`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
writeFileSync(`results/engine/token-identity-4090-${ts}.json`, JSON.stringify({ ...report, deviceLabel: "4090-linux", ts }, null, 2));
console.log("[idcheck] scritto results/engine/token-identity-4090-" + ts + ".json — pass:", report.pass);
process.exit(report.pass ? 0 : 1);
