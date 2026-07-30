// Attribuzione del decode wall (fase 1 B2): scomposizione encode/gpuBusy/sync +
// probe sync floor + predizione K∈{2,4,8}. Uso: node scripts/decode-attrib.mjs
// (dev server attivo su BASE_URL, default :5199)
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";
// PREFIX/GEN: finestra a contesto corto (confronto col dato tsq-diag B1: il GPU
// busy scala con kvLen). Default: contesto pieno del bench.
const PREFIX = process.env.PREFIX ? Number(process.env.PREFIX) : null;
const GEN = process.env.GEN ? Number(process.env.GEN) : null;
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[attrib][pageerror]", e.message.slice(0, 300)));
page.on("console", (m) => { if (m.text().includes("[engine]")) console.log("[attrib]", m.text().slice(0, 300)); });
const q = `attrib=1${PREFIX ? `&prefixLen=${PREFIX}` : ""}${GEN ? `&gen=${GEN}` : ""}`;
await page.goto(`${BASE_URL}/engine.html?${q}`, { waitUntil: "load" });

let last = "";
const poll = setInterval(async () => {
  try {
    const s = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    if (s !== last) { last = s; console.log("[attrib]", s); }
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
if (status !== "done" || !report) { console.error("[attrib] FALLITO:", status); process.exit(2); }
const d = report.decomposition;
console.log(`[attrib] wall ${d.wallMsPerToken.toFixed(2)} ms/tok = encode ${d.encodeCpuMsPerToken.toFixed(2)} + gpuBusy ${d.gpuBusyMsPerToken.toFixed(2)} + sync ${d.syncMsPerToken.toFixed(2)} — quota fuori-GPU ${(d.quotaFuoriGpu * 100).toFixed(1)}%`);
console.log(`[attrib] probe: mapRoundTrip ${report.syncFloorProbe.mapRoundTrip.mean.toFixed(2)} ms (p50 ${report.syncFloorProbe.mapRoundTrip.p50.toFixed(2)}), workDone ${report.syncFloorProbe.workDoneRoundTrip.mean.toFixed(2)} ms`);
for (const p of report.predictionByK)
  console.log(`[attrib] K=${p.K}: batched ${p.batched.toksPerSec.toFixed(0)} tok/s (quota ${(p.batched.quotaFuoriGpu * 100).toFixed(0)}%) · pipelined ${p.pipelined.toksPerSec.toFixed(0)} tok/s (quota ${(p.pipelined.quotaFuoriGpu * 100).toFixed(0)}%)`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
const tag = PREFIX ? `-ctx${PREFIX}` : "";
writeFileSync(`results/engine/decode-attrib-4090${tag}-${ts}.json`, JSON.stringify({ ...report, deviceLabel: "4090-linux", ts }, null, 2));
console.log(`[attrib] scritto results/engine/decode-attrib-4090${tag}-${ts}.json`);
