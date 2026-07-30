// Conformance del motore vs golden llama.cpp (gate della spec fase A: top-1 >= 99%).
// Uso (dev server attivo, GGUF symlinkato in public/models): node scripts/conformance-engine.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
// Gate doppio (ruling PI 2026-07-29, docket engine-fase-a item 3, opzione A):
// parita' vera vs cpuref-f64 >= 99% E sanity vs golden llama.cpp >= 97%.
const GATE_VS_CPUREF = 99;
const GATE_VS_GOLDEN = 97;
const CPUREF_GOLDEN = "results/engine/golden/cpuref-argmax-qwen25-05b-q4_0.json";
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
await page.goto(`${BASE_URL}/engine.html?conformance=1${process.env.TSQ === "1" ? "&tsq=1" : ""}`, { waitUntil: "load" });

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
// gate 2: engine vs cpuref-f64 (per-posizione, dagli argmax riportati dal worker)
const cpuref = JSON.parse(readFileSync(CPUREF_GOLDEN, "utf8"));
let cpuAgree = 0, cpuTotal = 0;
for (const p of report.perPrompt) {
  const ref = cpuref.prompts.find((x) => x.id === p.id);
  if (!ref || !p.got) { console.error(`[conf] manca got/cpuref per ${p.id}`); process.exit(2); }
  for (let i = 0; i < p.got.length; i++) {
    cpuTotal++;
    if (p.got[i] === ref.argmax[i]) cpuAgree++;
  }
}
const vsCpurefPct = (cpuAgree / cpuTotal) * 100;
console.log(`[conf] vs golden llama.cpp: ${report.top1Pct.toFixed(2)}% (gate ≥${GATE_VS_GOLDEN}) — vs cpuref-f64: ${vsCpurefPct.toFixed(2)}% (${cpuAgree}/${cpuTotal}, gate ≥${GATE_VS_CPUREF}) — maxΔlogit ${report.maxDlogitSampled.toFixed(4)} — ${report.dispatchesPerToken} dispatch/token`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
const file = `results/engine/conformance-4090-${ts}.json`;
writeFileSync(file, JSON.stringify({ ...report, vsCpurefPct, cpuAgree, cpuTotal, gates: { vsCpuref: GATE_VS_CPUREF, vsGolden: GATE_VS_GOLDEN }, deviceLabel: "4090-linux", ts }, null, 2));
console.log("[conf] scritto", file);
const pass = vsCpurefPct >= GATE_VS_CPUREF && report.top1Pct >= GATE_VS_GOLDEN;
console.log(`[conf] GATE DOPPIO: ${pass ? "PASS" : "FAIL"}`);
process.exit(pass ? 0 : 1);
