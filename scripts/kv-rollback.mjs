// Prova meccanica rollback KV (fase 4 B1, done-when): "genera N, crop a P, rigenera"
// vs run fresco — sequenze token IDENTICHE, exit 0, report JSON committato.
// Uso: node scripts/kv-rollback.mjs (dev server attivo su BASE_URL)
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[rollback][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/engine.html?rollback=1`, { waitUntil: "load" });

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
if (status !== "done" || !report) { console.error("[rollback] FALLITO:", status); process.exit(2); }

for (const c of report.checks) console.log(`[rollback] ${c.name}: ${c.match ? "OK" : `DIVERGE a ${c.divergeAt}`}`);
console.log(`[rollback] kvLen dopo generazione: ${report.kvLenAfterGen} (ok=${report.kvLenOk}) — prefix ${report.prefixTokens}, gen ${report.genTokens}, cropMid ${report.cropMid}`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
const file = `results/engine/kv-rollback-4090-${ts}.json`;
writeFileSync(file, JSON.stringify({ ...report, deviceLabel: "4090-linux", ts }, null, 2));
console.log("[rollback] scritto", file);
console.log(`[rollback] ${report.pass ? "PASS" : "FAIL"}`);
process.exit(report.pass ? 0 : 1);
