// Runner ktest: apre /ktest.html su Chrome headed (SwiftShader landmine) e
// riporta i risultati kernel. Uso: node scripts/ktest-run.mjs (vite su :5199).
import { chromium } from "playwright";
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[ktest][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/ktest.html`, { waitUntil: "load" });
await page.waitForFunction(() => {
  const s = document.querySelector("#status")?.textContent ?? "";
  return s === "done" || s.startsWith("ERROR");
}, null, { timeout: 300000, polling: 500 });
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const report = await page.evaluate(() => window.__report ?? null);
await browser.close();
console.log("[ktest]", JSON.stringify({ status, report }, null, 1));
if (status !== "done") process.exit(2);
