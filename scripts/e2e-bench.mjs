// E2E smoke/bench driver per Task 9 — lancia chromium con flag WebGPU,
// legge il probe, e se la GPU è reale esegue il bench ed esporta il JSON.
import { chromium, firefox } from "playwright";
import { writeFileSync } from "node:fs";

const BROWSER = process.env.BROWSER ?? "chromium"; // chromium | firefox (firefox: run effimero, solo cold)

const HEADED = process.env.HEADED === "1";
const CHANNEL = process.env.CHANNEL; // es. "chrome" per il branded; assente = chromium playwright
const args = process.env.CHROME_ARGS
  ? process.env.CHROME_ARGS.split(" ")
  : ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

// Profilo persistente (chromium): la Cache API dei pesi sopravvive tra i run (cold vs warm reali).
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
let browser, page;
if (BROWSER === "firefox") {
  browser = await firefox.launch({ headless: !HEADED, firefoxUserPrefs: { "dom.webgpu.enabled": true } });
  page = await browser.newPage();
} else {
  browser = await chromium.launchPersistentContext(PROFILE, { headless: !HEADED, args, ...(CHANNEL ? { channel: CHANNEL } : {}) });
  page = browser.pages()[0] ?? (await browser.newPage());
}
page.on("console", (m) => console.log(`[console:${m.type()}] ${m.text().slice(0, 200)}`));
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message.slice(0, 300)}`));

await page.goto("http://localhost:5173", { waitUntil: "load" });
console.log("[e2e] crossOriginIsolated =", await page.evaluate(() => crossOriginIsolated));

// Attendi il probe (BenchServer risponde al probe iniziale)
await page.waitForFunction(
  () => document.querySelector("#probe-box")?.textContent?.includes("webgpu"),
  null,
  { timeout: 30000 },
);
const probeText = await page.evaluate(() => document.querySelector("#probe-box").textContent);
const probe = JSON.parse(probeText);
console.log("[e2e] PROBE:", JSON.stringify(probe));

const vendor = probe.adapterInfo?.vendor ?? "";
const desc = (probe.adapterInfo?.description ?? "") + " " + (probe.adapterInfo?.architecture ?? "");
const isReal = probe.webgpu && vendor.toLowerCase() === "nvidia" && !/swiftshader|llvmpipe|software/i.test(desc);
console.log(`[e2e] GPU reale (nvidia, non-software)? ${isReal}`);

if (!isReal) {
  if (process.env.ALLOW_UNVERIFIED === "1") {
    console.log("[e2e] WARN: adapter non verificabile via JS (es. Firefox nasconde vendor) — si procede su richiesta esplicita; verifica hardware demandata (about:support).");
  } else {
    console.log("[e2e] STOP: adapter non-NVIDIA o software rasterizer — il run NON vale come 4090-linux.");
    await browser.close();
    process.exit(2);
  }
}

// Override modello via env (es. quant alternativa se il browser non espone shader-f16)
if (process.env.MODEL_ID) {
  await page.evaluate(
    ([id, q]) => {
      const sel = document.querySelector("#model");
      const o = document.createElement("option");
      o.value = id;
      o.dataset.quant = q;
      o.textContent = id;
      sel.appendChild(o);
      sel.value = id;
    },
    [process.env.MODEL_ID, process.env.QUANT ?? "unknown"],
  );
  console.log(`[e2e] modello override: ${process.env.MODEL_ID}`);
}

// Run bench (cold o warm a seconda della cache dell'ambiente)
await page.click("#run");
console.log("[e2e] bench avviato, attendo risultato (download modello al primo giro)…");
await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null,
  { timeout: 900000, polling: 2000 },
);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
console.log("[e2e] STATUS:", status);
if (status.startsWith("ERROR")) {
  await browser.close();
  process.exit(3);
}

// Estrai il RunFile direttamente ricostruendolo dal blob export: intercetta il download
// Più semplice: rileggi run dallo stato UI — esportiamo via click e catturiamo il download.
const [download] = await Promise.all([
  page.waitForEvent("download", { timeout: 30000 }),
  page.click("#export"),
]);
const path = await download.path();
const suggested = download.suggestedFilename();
const { copyFileSync } = await import("node:fs");
const out = `/tmp/${suggested}`;
copyFileSync(path, out);
console.log(`[e2e] EXPORT salvato: ${out}`);

await browser.close();
