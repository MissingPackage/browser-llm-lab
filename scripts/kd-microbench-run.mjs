// Driver del micro-bench di fase 0 (engine-kernel-decode): apre kdbench.html su
// Chrome headed (landmine SwiftShader: launch() semplice non espone
// navigator.gpu), aspetta done, scrive il run file in results/microbench/.
// Uso: node scripts/kd-microbench-run.mjs [--label 4090-linux] [--host quiescent]
// ATTENZIONE: due runner playwright sullo stesso profilo si bloccano a vicenda —
// i bench browser vanno eseguiti UNO ALLA VOLTA.
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
};
const LABEL = arg("label", "4090-linux");
const HOST = arg("host", "quiescent");
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[kd][pageerror]", e.message.slice(0, 400)));
page.on("console", (m) => { if (m.type() === "error") console.log("[kd][console]", m.text().slice(0, 400)); });
await page.goto(`${BASE_URL}/kdbench.html?label=${encodeURIComponent(LABEL)}&host=${encodeURIComponent(HOST)}`, { waitUntil: "load" });

let last = "";
const iv = setInterval(async () => {
  try {
    const s = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    if (s && s !== last) { last = s; console.log("[kd]", s); }
  } catch { /* pagina in navigazione */ }
}, 2000);

let status = "TIMEOUT";
try {
  await page.waitForFunction(() => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  }, null, { timeout: 1_800_000, polling: 1000 });
  status = await page.evaluate(() => document.querySelector("#status").textContent);
} catch (e) {
  console.error("[kd] attesa fallita:", String(e).slice(0, 300));
}
clearInterval(iv);

const report = await page.evaluate(() => window.__report ?? null);
await browser.close();

if (status !== "done" || !report) {
  console.error("[kd] FALLITO:", status);
  process.exit(2);
}

mkdirSync("results/microbench", { recursive: true });
const ts = report.ts.replace(/[:.]/g, "-");
const path = `results/microbench/kernel-decode-fase0-${LABEL}-${ts}.json`;
writeFileSync(path, JSON.stringify(report, null, 2));
console.log("[kd] scritto", path);
for (const c of report.cells) {
  const shape = Object.entries(c.shape).map(([k, v]) => `${k}=${v}`).join(",");
  console.log(
    `[kd] ${c.kernel.padEnd(11)} ${c.variant.padEnd(14)} ${shape.padEnd(28)} ` +
    `p50 ${c.msPerOp.p50.toFixed(4)} ms  ${c.effectiveGBps.toFixed(1)} GB/s` +
    (c.weightsPerSecond ? `  ${(c.weightsPerSecond / 1e9).toFixed(1)} G pesi/s` : ""),
  );
}
for (const s of report.skipped) console.log(`[kd] SKIP ${s.kernel} ${s.variant} ${JSON.stringify(s.shape)} — ${s.reason}`);
