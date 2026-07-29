// Conformance del motore vs golden llama.cpp (gate della spec fase A: top-1 >= 99%).
// Uso (dev server attivo, GGUF symlinkato in public/models): node scripts/conformance-engine.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const GATE_PCT = 99;
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[conf][pageerror]", e.message.slice(0, 300)));
page.on("console", (m) => {
  const t = m.text();
  if (t.includes("[engine]") || m.type() === "error" || m.type() === "warning") {
    console.log(`[conf][console:${m.type()}]`, t.slice(0, 500));
  }
});
await page.goto(`${BASE_URL}/engine.html?conformance=1`, { waitUntil: "load" });

let lastStatus = "";
const poll = setInterval(async () => {
  try {
    const s = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    if (s !== lastStatus) { lastStatus = s; console.log("[conf]", s); }
  } catch { /* chiusa */ }
}, 2000);

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

if (status !== "done" || !report) {
  console.error("[conf] FALLITO:", status);
  process.exit(2);
}
console.log(`[conf] top-1 ${report.top1Pct.toFixed(2)}% (${report.agree}/${report.total}) — maxΔlogit ${report.maxDlogitSampled.toFixed(4)} — ${report.dispatchesPerToken} dispatch/token — ${(report.wallMs / 1000).toFixed(1)}s`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
const file = `results/engine/conformance-4090-${ts}.json`;
writeFileSync(file, JSON.stringify({ ...report, deviceLabel: "4090-linux", ts }, null, 2));
console.log("[conf] scritto", file);
process.exit(report.top1Pct >= GATE_PCT ? 0 : 1);
