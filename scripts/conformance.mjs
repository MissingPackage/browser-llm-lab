// Driver Playwright on-demand per il conformance harness — stessa forma di
// scripts/e2e-bench.mjs (profilo persistente, log console/pageerror, GPU-reality check),
// ma esegue il contratto condiviso (src/conformance/contract.ts) contro ENTRAMBI gli
// adapter reali via conformance.html invece del bench UI.
//
// A differenza di e2e-bench.mjs (che assume il dev server Vite già attivo, cosa che ha
// causato confusione in passato), questo script CONTROLLA che il dev server risponda su
// localhost:5173 prima di procedere, e fallisce con un messaggio esplicito se non lo è —
// non lo avvia/ferma da solo, per non gestire un secondo processo lifecycle.
import { chromium } from "playwright";

const HEADED = process.env.HEADED === "1";
const CHANNEL = process.env.CHANNEL; // es. "chrome" per il branded; assente = chromium playwright
const args = process.env.CHROME_ARGS
  ? process.env.CHROME_ARGS.split(" ")
  : ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

// Stesso profilo default di scripts/e2e-bench.mjs: la Cache API dei pesi (incl. il modello
// WebLLM già scaricato nei run precedenti) sopravvive tra i due script.
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.CONFORMANCE_URL ?? "http://localhost:5173";

async function devServerReachable(url) {
  try {
    await fetch(url, { method: "GET" });
    return true;
  } catch {
    return false;
  }
}

if (!(await devServerReachable(BASE_URL))) {
  console.error(`[conformance] Vite dev server non raggiungibile su ${BASE_URL}.`);
  console.error("[conformance] Avvialo in un altro terminale con: npm run dev");
  console.error("[conformance] (questo script non lo avvia/ferma da solo — vedi commento in scripts/conformance.mjs)");
  process.exit(1);
}

const browser = await chromium.launchPersistentContext(PROFILE, {
  headless: !HEADED,
  args,
  ...(CHANNEL ? { channel: CHANNEL } : {}),
});
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("console", (m) => console.log(`[console:${m.type()}] ${m.text().slice(0, 300)}`));
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message.slice(0, 400)}`));

await page.goto(`${BASE_URL}/conformance.html`, { waitUntil: "load" });
console.log("[conformance] crossOriginIsolated =", await page.evaluate(() => crossOriginIsolated));

// GPU-reality check — stessa euristica di scripts/e2e-bench.mjs: un pass su rasterizzatore
// software (SwiftShader/llvmpipe) sarebbe fuorviante, quindi rifiutiamo di riportarlo come tale.
const gpuInfo = await page.evaluate(async () => {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter().catch(() => null);
  if (!adapter) return null;
  const a = adapter;
  const info = a.info ?? (a.requestAdapterInfo ? await a.requestAdapterInfo().catch(() => null) : null);
  return info ? { vendor: info.vendor ?? "", architecture: info.architecture ?? "", description: info.description ?? "" } : null;
});
console.log("[conformance] GPU adapter info:", JSON.stringify(gpuInfo));

const vendor = (gpuInfo?.vendor ?? "").toLowerCase();
const desc = `${gpuInfo?.description ?? ""} ${gpuInfo?.architecture ?? ""}`;
const isReal = !!gpuInfo && vendor === "nvidia" && !/swiftshader|llvmpipe|software/i.test(desc);
console.log(`[conformance] GPU reale (nvidia, non-software)? ${isReal}`);

if (!isReal) {
  if (process.env.ALLOW_UNVERIFIED === "1") {
    console.log("[conformance] WARN: adapter non verificabile via JS (es. Firefox nasconde vendor) — si procede su richiesta esplicita.");
  } else {
    console.log("[conformance] STOP: adapter non-NVIDIA o software rasterizer — un pass qui non sarebbe attendibile come conformance reale.");
    await browser.close();
    process.exit(2);
  }
}

// Attendi il completamento del run (entrambi gli adapter, eseguiti in sequenza in src/conformance/page.ts)
try {
  await page.waitForFunction(
    () => window.__conformanceResults?.status === "done" || window.__conformanceResults?.status === "error",
    null,
    { timeout: 900000, polling: 2000 },
  );
} catch (e) {
  console.error(`[conformance] timeout in attesa del completamento: ${e.message}`);
  await browser.close();
  process.exit(4);
}

const run = await page.evaluate(() => window.__conformanceResults);
await browser.close();

if (run.status === "error") {
  console.error(`[conformance] ERRORE fatale nello script di pagina: ${run.error}`);
  process.exit(3);
}

// Report leggibile per adapter, per check.
let totalChecks = 0;
let totalFail = 0;
for (const r of run.results) {
  const passCount = r.checks.filter((c) => c.pass).length;
  console.log(`\n=== ${r.adapterId} (${r.modelId}) — ${passCount}/${r.checks.length} passed ===`);
  for (const c of r.checks) {
    totalChecks++;
    if (!c.pass) totalFail++;
    console.log(`  [${c.pass ? "PASS" : "FAIL"}] ${c.name} — ${c.detail}`);
  }
}

console.log(`\n[conformance] ${totalFail === 0 ? "ALL CHECKS PASSED" : `${totalFail}/${totalChecks} check(s) FAILED`}`);
process.exit(totalFail === 0 ? 0 : 1);
