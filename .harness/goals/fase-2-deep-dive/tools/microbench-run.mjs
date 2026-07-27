// Driver del run micro-bench sulla 4090 (fase 6) — stesso pattern di wgsl-dump.mjs.
// Uso: HEADED=1 BASE_URL=http://localhost:5177 CHANNEL=chrome DEVICE_LABEL=4090-linux \
//        node .harness/goals/fase-2-deep-dive/tools/microbench-run.mjs
import { chromium } from "playwright";
import { mkdirSync, copyFileSync } from "node:fs";

const OUT_DIR = "results/microbench";
const HEADED = process.env.HEADED === "1";
const CHANNEL = process.env.CHANNEL;
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const DEVICE_LABEL = process.env.DEVICE_LABEL ?? "unknown-device";

const browser = await chromium.launchPersistentContext(PROFILE, {
  headless: !HEADED, args, ...(CHANNEL ? { channel: CHANNEL } : {}),
});
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message.slice(0, 300)}`));
page.on("console", (m) => console.log(`[console:${m.type()}] ${m.text().slice(0, 200)}`));

await page.goto(`${BASE_URL}/microbench.html`, { waitUntil: "load" });
await page.fill("#device-label", DEVICE_LABEL);
await page.click("#run");
console.log("[mb] run avviato…");
await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 300000, polling: 1000 },
);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
console.log("[mb] STATUS:", status);
if (status.startsWith("ERROR")) { await browser.close(); process.exit(3); }

// La guardia isReal come nei driver ufficiali: il file va in results/ solo da GPU vera.
const table = await page.evaluate(() => document.querySelector("#results")?.textContent ?? "");
console.log("[mb] tabella:", table.slice(0, 400));

const [download] = await Promise.all([
  page.waitForEvent("download", { timeout: 30000 }),
  page.click("#export"),
]);
mkdirSync(OUT_DIR, { recursive: true });
const out = `${OUT_DIR}/${download.suggestedFilename()}`;
copyFileSync(await download.path(), out);
console.log(`[mb] EXPORT salvato: ${out}`);
await browser.close();
