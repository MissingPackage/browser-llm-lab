// Diag parità prefill chunked (fase 3 B1, TEMPORANEO come la modalità worker che pilota).
// Uso: node scripts/prefill-diag.mjs (dev server attivo su BASE_URL)
import { chromium } from "playwright";
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[pdiag][pageerror]", e.message.slice(0, 300)));
page.on("console", (m) => { if (m.text().includes("[engine]")) console.log("[pdiag]", m.text().slice(0, 300)); });
await page.goto(`${BASE_URL}/engine.html?prefilldiag=1`, { waitUntil: "load" });
await page.waitForFunction(() => {
  const s = document.querySelector("#status")?.textContent ?? "";
  return s === "done" || s.startsWith("ERROR");
}, null, { timeout: 1200000, polling: 1000 });
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const report = await page.evaluate(() => window.__report ?? null);
await browser.close();
if (status !== "done" || !report) { console.error("[pdiag] FALLITO:", status); process.exit(2); }
for (const v of report.lens) console.log(`[pdiag] L=${v.L}: argmaxMatch=${v.argmaxMatch} (ref=${v.refArgmax} got=${v.gotArgmax}) maxDlogit=${v.maxDlogit.toFixed(4)} tapDiff=${JSON.stringify(v.tapDiff)}`);
