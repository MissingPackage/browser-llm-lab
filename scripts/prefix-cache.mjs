// Prova end-to-end prefix-cache OPFS (fase 5 B1, done-when): save del checkpoint,
// RESTORE IN WORKER NUOVO (seconda page load = worker fresco, sessione fredda per
// il motore; l'OPFS persiste per origin), continuazione token-identica al run
// ininterrotto, e tempo restore < tempo re-prefill (entrambi nel JSON).
// Uso: node scripts/prefix-cache.mjs (dev server attivo su BASE_URL)
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[pc][pageerror]", e.message.slice(0, 300)));

const runMode = async (mode) => {
  await page.goto(`${BASE_URL}/engine.html?${mode}=1`, { waitUntil: "load" });
  await page.waitForFunction(
    () => {
      const s = document.querySelector("#status")?.textContent ?? "";
      return s === "done" || s.startsWith("ERROR");
    },
    null, { timeout: 600000, polling: 1000 },
  );
  const status = await page.evaluate(() => document.querySelector("#status").textContent);
  if (status !== "done") { console.error(`[pc] ${mode} FALLITO:`, status); process.exit(2); }
  return page.evaluate(() => window.__report);
};

const save = await runMode("pcsave");
console.log(`[pc] save: prefill ${save.prefillMs.toFixed(0)} ms · checkpoint ${(save.checkpointBytes / 1e6).toFixed(1)} MB in ${save.saveMs.toFixed(1)} ms · ${save.file}`);
const restore = await runMode("pcrestore"); // page load nuova ⇒ WORKER NUOVO
await browser.close();

const identical = restore.contTokens.length === save.contTokens.length &&
  restore.contTokens.every((t, i) => t === save.contTokens[i]);
const faster = restore.restoreMs < restore.reprefillMs;
console.log(`[pc] restore (worker nuovo): ${restore.restoreMs.toFixed(1)} ms vs re-prefill ${restore.reprefillMs.toFixed(0)} ms — continuazione identica: ${identical} · restore<reprefill: ${faster} (hit #${restore.hitCount})`);

mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
const file = `results/engine/prefix-cache-4090-${ts}.json`;
const pass = identical && faster;
writeFileSync(file, JSON.stringify({
  schemaVersion: 1, kind: "engine-prefix-cache", deviceLabel: "4090-linux", ts,
  note: "misura warm (page-cache OS calda: stesso profilo, save appena eseguito); il regime freddo è disco-bound, dichiarato in spec §Rischi",
  save, restore, checks: { continuationIdentical: identical, restoreFasterThanReprefill: faster }, pass,
}, null, 2));
console.log("[pc] scritto", file);
console.log(`[pc] ${pass ? "PASS" : "FAIL"}`);
process.exit(pass ? 0 : 1);
