// Dump del WGSL generato da TVM per i kernel WebLLM — fase 4 del goal fase-2-deep-dive.
// Autorità: GOAL.md, "intercept createShaderModule in a local instrumented run to dump
// generated WGSL". Non tocca nessun file della SPA: la patch è iniettata a runtime nel
// worker via Playwright worker.evaluate().
//
// Uso (dalla root del repo, dev server già attivo su :5173):
//   CHANNEL=chrome node .harness/goals/fase-2-deep-dive/tools/wgsl-dump.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const OUT_DIR = ".harness/goals/fase-2-deep-dive/wgsl-dump";
const HEADED = process.env.HEADED === "1";
const CHANNEL = process.env.CHANNEL; // "chrome" per il branded, come nei run reali
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
// Stesso profilo del driver e2e: pesi in cache → load warm, niente download.
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";

const browser = await chromium.launchPersistentContext(PROFILE, {
  headless: !HEADED,
  args,
  ...(CHANNEL ? { channel: CHANNEL } : {}),
});
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message.slice(0, 300)}`));
page.on("console", (m) => console.log(`[console:${m.type()}] ${m.text().slice(0, 200)}`));

const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
await page.goto(BASE_URL, { waitUntil: "load" });
// Logga lo stato UI ogni 15s per diagnosi (il primo run è morto in timeout senza dirci dove).
const statusLogger = setInterval(async () => {
  try {
    const s = await page.evaluate(() => ({
      status: document.querySelector("#status")?.textContent ?? "?",
      progress: document.querySelector("#progress")?.textContent?.slice(0, 120) ?? "",
    }));
    console.log(`[dump:poll] status=${s.status} progress=${s.progress}`);
  } catch { /* pagina chiusa */ }
}, 15000);

// Il bench worker parte al load della pagina; aspettiamo che Playwright lo veda.
let worker = page.workers()[0];
for (let i = 0; !worker && i < 50; i++) {
  await new Promise((r) => setTimeout(r, 200));
  worker = page.workers()[0];
}
if (!worker) {
  console.error("[dump] nessun worker trovato — la SPA ha cambiato struttura?");
  await browser.close();
  process.exit(1);
}
console.log("[dump] worker agganciato:", worker.url());

// Patch DENTRO il worker, prima che il device/gli shader esistano (il load parte col click).
await worker.evaluate(() => {
  self.__wgsl = [];
  const orig = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (desc) {
    self.__wgsl.push({ code: desc.code ?? "", ts: performance.now() });
    return orig.call(this, desc);
  };
});
console.log("[dump] patch createShaderModule installata nel worker");

// Probe pronto → label device → run webllm (stack default) → attendi done.
await page.waitForFunction(
  () => document.querySelector("#probe-box")?.textContent?.includes("webgpu"),
  null, { timeout: 30000 },
);
// Stessa guardia del driver e2e: su SwiftShader/software il warm-up non finisce mai
// (limite 256 invocations = spia del rasterizer software; il primo tentativo headless
// è morto esattamente così).
const probe = JSON.parse(await page.evaluate(() => document.querySelector("#probe-box").textContent));
const vendor = probe.adapterInfo?.vendor ?? "";
const desc = (probe.adapterInfo?.description ?? "") + " " + (probe.adapterInfo?.architecture ?? "");
if (!(probe.webgpu && vendor.toLowerCase() === "nvidia" && !/swiftshader|llvmpipe|software/i.test(desc))) {
  console.error(`[dump] STOP: adapter non reale (vendor=${vendor}, desc=${desc.trim()}) — usa HEADED=1 sulla 4090.`);
  clearInterval(statusLogger);
  await browser.close();
  process.exit(2);
}
await page.fill("#device-label", process.env.DEVICE_LABEL ?? "wgsl-dump-run");
await page.click("#run");
console.log("[dump] run avviato (serve solo il load+bench per compilare tutti gli shader)…");
await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 900000, polling: 2000 },
);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
console.log("[dump] STATUS:", status);

clearInterval(statusLogger);
const shaders = await worker.evaluate(() => self.__wgsl);
console.log(`[dump] shader catturati: ${shaders.length}`);

mkdirSync(OUT_DIR, { recursive: true });
const index = [];
shaders.forEach((s, i) => {
  // Nome dal primo entry point/fn WGSL riconoscibile, altrimenti indice.
  const m = s.code.match(/fn\s+([A-Za-z0-9_]+)\s*\(/);
  const name = `${String(i).padStart(3, "0")}-${(m?.[1] ?? "anon").slice(0, 80)}`;
  writeFileSync(`${OUT_DIR}/${name}.wgsl`, s.code);
  index.push({ file: `${name}.wgsl`, bytes: s.code.length, ts: Math.round(s.ts) });
});
writeFileSync(`${OUT_DIR}/INDEX.json`, JSON.stringify({ status, count: shaders.length, shaders: index }, null, 2));
console.log(`[dump] scritti ${shaders.length} file in ${OUT_DIR}`);
await browser.close();
process.exit(status === "done" ? 0 : 3);
