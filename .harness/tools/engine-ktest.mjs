// Driver del kernel-test motore: apre ktest.html headed (GPU reale), aspetta done,
// stampa la tabella. Uso: node .harness/tools/engine-ktest.mjs
import { chromium } from "playwright";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[ktest][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/ktest.html`, { waitUntil: "load" });
await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 120000, polling: 500 },
);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const adapter = await page.evaluate(() => document.querySelector("#probe-box").textContent);
const table = await page.evaluate(() => document.querySelector("#results").innerText);
console.log("[ktest] adapter:", adapter);
console.log(table);
console.log("[ktest] STATUS:", status);
await browser.close();
process.exit(status === "done" ? 0 : 1);
