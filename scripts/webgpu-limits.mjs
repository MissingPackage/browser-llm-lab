// Probe dei limiti WebGPU REALI dell'adapter (goal C3a, primo task della fase 4
// per spec §3.1). Nasce perché il motore negozia `min(limite, 2 GiB)` senza mai
// leggere il limite: il design della leva 2 (eliminare i 46 submit/token del
// MoE) dipende da quanti slab expert entrano in UN singolo storage binding.
// Uso:
//   node scripts/webgpu-limits.mjs [--out results/engine/webgpu-limits-<host>.json]
// Exit: 0 ok, 2 errore (niente adapter / niente WebGPU).
import { chromium } from "playwright";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const out = arg("out", null);
const ROOT = new URL("..", import.meta.url).pathname;
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-glmroute-profile");
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";

mkdirSync(PROFILE, { recursive: true });
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[limits][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/index.html`, { waitUntil: "load" });

const report = await page.evaluate(async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { error: "niente adapter WebGPU" };
  const lim = {};
  for (const k in adapter.limits) lim[k] = Number(adapter.limits[k]);
  const info = adapter.info ?? {};
  // Quanto si ottiene DAVVERO chiedendo il massimo: il limite dell'adapter è
  // una promessa, il device è ciò che si ottiene. Li riportiamo entrambi.
  // Si chiede TUTTO quello che l'adapter espone: è l'unico modo di sapere se un
  // limite è davvero CONCEDIBILE e non solo annunciato. (Prima il probe ne
  // chiedeva 3 e riportava i default sugli altri, facendo sembrare non
  // negoziabili limiti che non erano mai stati chiesti.)
  let deviceLimits = null, deviceError = null;
  try {
    const dev = await adapter.requestDevice({ requiredLimits: { ...lim } });
    deviceLimits = {};
    for (const k in dev.limits) deviceLimits[k] = Number(dev.limits[k]);
    dev.destroy();
  } catch (e) {
    deviceError = String(e);
  }
  return {
    adapterInfo: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    features: [...adapter.features].sort(),
    adapterLimits: lim,
    deviceLimits,
    deviceError,
  };
});

await browser.close();
if (report.error) {
  console.error(`[limits] ${report.error}`);
  process.exit(2);
}

// Il conto che interessa alla spec §3: quanti slab expert entrano in un binding.
const SLAB_Q4_0 = 5_308_416, SLAB_Q4_1 = 5_505_024;
const eff = report.deviceLimits ?? report.adapterLimits;
const bind = eff.maxStorageBufferBindingSize;
const buf = eff.maxBufferSize;
report.derived = {
  slabBytesQ4_0: SLAB_Q4_0,
  slabBytesQ4_1: SLAB_Q4_1,
  slabsPerBinding_q4_0: Math.floor(bind / SLAB_Q4_0),
  slabsPerBinding_q4_1: Math.floor(bind / SLAB_Q4_1),
  slabsPerBuffer_q4_0: Math.floor(buf / SLAB_Q4_0),
  bindingsToCoverPark: Math.ceil(2944 / Math.floor(bind / SLAB_Q4_0)),
  note: "parco completo = 2944 expert (2688 q4_0 + 256 q4_1); il residente a slab 12 GiB e' ~2419 slot",
};

// ADAPTER vs DEVICE: un limite dell'adapter è solo una promessa — il device
// riceve il DEFAULT della spec se non lo si chiede in `requiredLimits`. È la
// trappola in cui è caduto il motore (chiede solo 3 limiti su tanti), quindi
// il probe stampa le due colonne affiancate e segnala le differenze.
const KEYS = [
  "maxBufferSize", "maxStorageBufferBindingSize", "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup", "maxStorageBuffersPerShaderStage",
  "maxComputeWorkgroupSizeX", "maxBindGroups", "maxBindingsPerBindGroup",
];
console.log(`[limits] ${report.adapterInfo.description ?? report.adapterInfo.device ?? "adapter"}`);
console.log(`[limits] ${"limite".padEnd(38)} ${"adapter".padStart(12)} ${"device".padStart(12)}`);
const lasciati = [];
for (const k of KEYS) {
  const a = report.adapterLimits[k], d = report.deviceLimits ? report.deviceLimits[k] : null;
  const flag = d !== null && d < a ? "  <-- NON negoziato" : "";
  if (flag) lasciati.push(`${k} (${d} su ${a})`);
  console.log(`[limits] ${k.padEnd(38)} ${String(a).padStart(12)} ${String(d ?? "n/d").padStart(12)}${flag}`);
}
report.derived.limitiNonNegoziati = lasciati;
console.log(`[limits] slab expert per binding: ${report.derived.slabsPerBinding_q4_0} (q4_0)`);
console.log(`[limits] binding necessari a coprire il parco (2944): ${report.derived.bindingsToCoverPark}`);
console.log(`[limits] features: ${report.features.join(", ")}`);
if (report.deviceError) console.log(`[limits] ATTENZIONE device: ${report.deviceError}`);

if (out) {
  const p = join(ROOT, out);
  if (!existsSync(dirname(p))) mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, JSON.stringify({ kind: "webgpu-limits", schemaVersion: 1, date: new Date().toISOString(), ...report }, null, 1));
  console.log(`[limits] scritto ${out}`);
}
