// Dispatch profiler — sessione di stima nuovo motore (2026-07-28).
//
// Domanda: quanti dispatch WebGPU costa REALMENTE un token di decode, e quanto
// costa il lato CPU dell'encoding (createBindGroup + writeBuffer + beginComputePass
// + dispatchWorkgroups + end) per ciascuno?
//
// Perche' serve: i doc di deep-dive usano "~34 dispatch per token", che e' il numero
// di kernel DISTINTI nel dump WGSL, non il numero di invocazioni. Con 24 layer il
// conteggio vero e' un ordine di grandezza sopra, e ogni stima di leva che tocca
// il termine N_dispatch cambia in proporzione.
//
// Metodo: patch dei prototype WebGPU dentro il worker (stesso pattern di
// wgsl-dump.mjs, nessun file della SPA toccato), campionamento dei contatori dal
// driver ogni SAMPLE_MS. Nel decode WebLLM fa un submit per token generato
// (compute-shader-dispatch.md), quindi durante la finestra di decode
//   dispatch/token = Δdispatch / Δsubmit
// senza bisogno di conoscere il numero di token.
//
// Uso (dalla root, dev server attivo):
//   HEADED=1 CHANNEL=chrome BASE_URL=http://localhost:5173 node tools/harness/dispatch-profile.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const OUT_DIR = "results/dispatch-profile";
const HEADED = process.env.HEADED === "1";
const CHANNEL = process.env.CHANNEL;
const SAMPLE_MS = Number(process.env.SAMPLE_MS ?? 250);
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const DEVICE_LABEL = process.env.DEVICE_LABEL ?? "4090-linux";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, {
  headless: !HEADED,
  args,
  ...(CHANNEL ? { channel: CHANNEL } : {}),
});
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message.slice(0, 300)}`));
page.on("console", (m) => {
  const t = m.text();
  if (!t.startsWith("[prof]")) return;
  console.log(`[worker] ${t.slice(0, 200)}`);
});

await page.goto(BASE_URL, { waitUntil: "load" });

let worker = page.workers()[0];
for (let i = 0; !worker && i < 50; i++) {
  await new Promise((r) => setTimeout(r, 200));
  worker = page.workers()[0];
}
if (!worker) {
  console.error("[prof] nessun worker trovato — struttura SPA cambiata?");
  await browser.close();
  process.exit(1);
}
console.log("[prof] worker agganciato:", worker.url());

// --- patch dentro il worker, prima che il device esista ---
await worker.evaluate(() => {
  const c = {
    dispatch: 0, bindGroup: 0, computePass: 0, writeBuffer: 0,
    submit: 0, commandEncoder: 0, passEnd: 0, mapAsync: 0, onSubmittedWorkDone: 0,
    tDispatch: 0, tBindGroup: 0, tComputePass: 0, tWriteBuffer: 0, tSubmit: 0,
    tCommandEncoder: 0, tPassEnd: 0,
  };
  self.__prof = c;

  // Risoluzione di performance.now() in questo contesto: i timer aggregati vanno
  // letti sapendo il quanto (Chrome coarsening). Misurata, non assunta.
  {
    const deltas = [];
    for (let i = 0; i < 5000 && deltas.length < 50; i++) {
      const a = performance.now(), b = performance.now();
      if (b > a) deltas.push(b - a);
    }
    self.__profClockMinDelta = deltas.length ? Math.min(...deltas) : null;
  }

  const wrap = (obj, name, counterKey, timerKey) => {
    const orig = obj[name];
    if (typeof orig !== "function") { console.log(`[prof] MANCA ${name}`); return; }
    obj[name] = function (...a) {
      const t0 = performance.now();
      const r = orig.apply(this, a);
      c[timerKey] += performance.now() - t0;
      c[counterKey]++;
      return r;
    };
  };

  wrap(GPUComputePassEncoder.prototype, "dispatchWorkgroups", "dispatch", "tDispatch");
  wrap(GPUDevice.prototype, "createBindGroup", "bindGroup", "tBindGroup");
  wrap(GPUCommandEncoder.prototype, "beginComputePass", "computePass", "tComputePass");
  wrap(GPUQueue.prototype, "writeBuffer", "writeBuffer", "tWriteBuffer");
  wrap(GPUQueue.prototype, "submit", "submit", "tSubmit");
  wrap(GPUDevice.prototype, "createCommandEncoder", "commandEncoder", "tCommandEncoder");
  wrap(GPUComputePassEncoder.prototype, "end", "passEnd", "tPassEnd");

  const origMapAsync = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function (...a) { c.mapAsync++; return origMapAsync.apply(this, a); };
  const origOSWD = GPUQueue.prototype.onSubmittedWorkDone;
  GPUQueue.prototype.onSubmittedWorkDone = function (...a) { c.onSubmittedWorkDone++; return origOSWD.apply(this, a); };

  console.log("[prof] patch installata");
});
console.log("[prof] patch installata nel worker");

await page.waitForFunction(
  () => document.querySelector("#probe-box")?.textContent?.includes("webgpu"),
  null, { timeout: 30000 },
);
const probe = JSON.parse(await page.evaluate(() => document.querySelector("#probe-box").textContent));
const vendor = (probe.adapterInfo?.vendor ?? "").toLowerCase();
const desc = `${probe.adapterInfo?.description ?? ""} ${probe.adapterInfo?.architecture ?? ""}`;
if (!(probe.webgpu && vendor === "nvidia" && !/swiftshader|llvmpipe|software/i.test(desc))) {
  console.error(`[prof] STOP: adapter non reale (vendor=${vendor}, desc=${desc.trim()}) — serve HEADED=1.`);
  await browser.close();
  process.exit(2);
}
console.log("[prof] adapter reale:", vendor, probe.adapterInfo?.architecture);

await page.fill("#device-label", DEVICE_LABEL);
await page.click("#run");
console.log("[prof] run avviato — campiono i contatori ogni", SAMPLE_MS, "ms");

const samples = [];
const t0 = Date.now();
const sampler = setInterval(async () => {
  try {
    const s = await worker.evaluate(() => ({ ...self.__prof }));
    const status = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    samples.push({ tMs: Date.now() - t0, status, ...s });
  } catch { /* pagina/worker chiusi */ }
}, SAMPLE_MS);

await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 900000, polling: 1000 },
);
clearInterval(sampler);
await new Promise((r) => setTimeout(r, 400));

const status = await page.evaluate(() => document.querySelector("#status").textContent);
const final = await worker.evaluate(() => ({ ...self.__prof }));
const clockMinDelta = await worker.evaluate(() => self.__profClockMinDelta);
const resultsJson = await page.evaluate(() => {
  const rows = document.querySelector("#results")?.textContent ?? "";
  return rows.slice(0, 4000);
});

console.log("[prof] STATUS:", status);
console.log("[prof] totali:", JSON.stringify(final));
console.log("[prof] performance.now() min delta (quanto):", clockMinDelta, "ms");

mkdirSync(OUT_DIR, { recursive: true });
const out = {
  schemaVersion: 1,
  kind: "dispatch-profile",
  deviceLabel: DEVICE_LABEL,
  ts: new Date().toISOString(),
  probe,
  status,
  sampleMs: SAMPLE_MS,
  clockMinDeltaMs: clockMinDelta,
  totals: final,
  samples,
  resultsTableText: resultsJson,
};
const file = `${OUT_DIR}/dispatch-profile-${DEVICE_LABEL}-${out.ts.replace(/[:.]/g, "-")}.json`;
writeFileSync(file, JSON.stringify(out, null, 2));
console.log("[prof] scritto", file, `(${samples.length} campioni)`);

await browser.close();
process.exit(status === "done" ? 0 : 3);
