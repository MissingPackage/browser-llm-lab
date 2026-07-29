// M2 di estimates §6 — attribuzione dei 7 submit/token per call-site (fase 2 del goal
// engine-fase-a). Domanda: quali dei 7 flush per token sono eliminabili con scratch
// preallocato (L2)? Metodo: patch di GPUQueue.submit nel worker che cattura
// Error().stack e bucketizza per call-site (file:riga del bundle WebLLM, normalizzato).
// Il costo della cattura stack (~7k stack/run) è irrilevante per il conteggio.
//
// Uso (dev server attivo): node .harness/tools/submit-callsites.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const OUT_DIR = "results/dispatch-profile";
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const DEVICE_LABEL = process.env.DEVICE_LABEL ?? "4090-linux";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("console", (m) => { if (m.text().startsWith("[m2]")) console.log(m.text()); });
await page.goto(BASE_URL, { waitUntil: "load" });

let worker = page.workers()[0];
for (let i = 0; !worker && i < 50; i++) {
  await new Promise((r) => setTimeout(r, 200));
  worker = page.workers()[0];
}
if (!worker) { console.error("[m2] nessun worker"); await browser.close(); process.exit(1); }

await worker.evaluate(() => {
  const buckets = new Map();
  self.__m2 = buckets;
  const orig = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function (...a) {
    // Frame 0 = questa wrapper, frame 1 = flushCommands, frame 2+ = chi l'ha forzata.
    // Si tiene la catena 1-4 normalizzata a "riga:colonna" del bundle (l'URL è unico).
    const stack = (new Error().stack ?? "").split("\n").slice(2, 6)
      .map((l) => (l.match(/:(\d+:\d+)\)?\s*$/) ?? [])[1] ?? l.trim())
      .join(" <- ");
    buckets.set(stack, (buckets.get(stack) ?? 0) + 1);
    return orig.apply(this, a);
  };
  console.log("[m2] patch submit installata");
});

await page.waitForFunction(
  () => document.querySelector("#probe-box")?.textContent?.includes("webgpu"),
  null, { timeout: 30000 },
);
const probe = JSON.parse(await page.evaluate(() => document.querySelector("#probe-box").textContent));
const vendor = (probe.adapterInfo?.vendor ?? "").toLowerCase();
if (!(probe.webgpu && vendor === "nvidia")) {
  console.error(`[m2] adapter non reale (${vendor})`); await browser.close(); process.exit(2);
}

await page.fill("#device-label", DEVICE_LABEL);
await page.click("#run");
console.log("[m2] run avviato");
await page.waitForFunction(
  () => { const s = document.querySelector("#status")?.textContent ?? ""; return s === "done" || s.startsWith("ERROR"); },
  null, { timeout: 900000, polling: 1000 },
);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const histogram = await worker.evaluate(() => Object.fromEntries(self.__m2));

const total = Object.values(histogram).reduce((a, b) => a + b, 0);
console.log(`[m2] status=${status} — ${total} submit in ${Object.keys(histogram).length} call-site:`);
for (const [site, n] of Object.entries(histogram).sort((a, b) => b[1] - a[1])) {
  console.log(`[m2]   ${n}\t(${((n / total) * 100).toFixed(1)}%)\t${site}`);
}

mkdirSync(OUT_DIR, { recursive: true });
const record = {
  schemaVersion: 1, kind: "submit-callsites", deviceLabel: DEVICE_LABEL,
  ts: new Date().toISOString(), status, totalSubmits: total, histogram, probe,
};
const file = `${OUT_DIR}/submit-callsites-${DEVICE_LABEL}-${record.ts.replace(/[:.]/g, "-")}.json`;
writeFileSync(file, JSON.stringify(record, null, 2));
console.log("[m2] scritto", file);
await browser.close();
process.exit(status === "done" ? 0 : 3);
