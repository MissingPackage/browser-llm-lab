// Bench decode del motore (protocollo repo: warmup + 3 repliche, prompt bench-512-v1
// chat-templated in token-id). Uso: node scripts/engine-bench.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { hostState } from "./lib/hoststate.mjs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
// hostState (schema 4d): dichiarato con HOST_STATE=quiescent (o altro), campionato via smi
const host = hostState(process.env.HOST_STATE ?? "undeclared");
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[bench][pageerror]", e.message.slice(0, 300)));
page.on("console", (m) => { if (m.text().includes("[engine]")) console.log("[bench]", m.text().slice(0, 300)); });
await page.goto(`${BASE_URL}/engine.html?bench=1`, { waitUntil: "load" });

let last = "";
const poll = setInterval(async () => {
  try {
    const s = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    if (s !== last) { last = s; console.log("[bench]", s); }
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
if (status !== "done" || !report) { console.error("[bench] FALLITO:", status); process.exit(2); }
console.log(`[bench] decode ${report.decodeToksPerSec.mean.toFixed(1)} tok/s (±${report.decodeToksPerSec.stdev.toFixed(1)}) — ${report.dispatchesPerToken} dispatch/token`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
writeFileSync(`results/engine/bench-4090-${ts}.json`, JSON.stringify({ ...report, deviceLabel: "4090-linux", ts, hostState: host.close() }, null, 2));
console.log("[bench] scritto results/engine/bench-4090-" + ts + ".json");
