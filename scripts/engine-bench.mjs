// Bench decode del motore (protocollo repo: warmup + 3 repliche, prompt bench-512-v1
// chat-templated in token-id). Uso: node scripts/engine-bench.mjs
import { chromium, firefox } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { hostState } from "./lib/hoststate.mjs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
// hostState (schema 4d): dichiarato con HOST_STATE=quiescent (o altro), campionato via smi
const host = hostState(process.env.HOST_STATE ?? "undeclared");
// BROWSER=chrome|chrome-vanilla|firefox. Il default resta `chrome` con i flag
// storici, cosi' i riferimenti gia' presi restano confrontabili.
//
// PERCHE' ESISTONO GLI ALTRI DUE, e non e' comodita':
//  - `chrome-vanilla` toglie `--enable-unsafe-webgpu`. Quel flag e' un artefatto
//    di Chrome SU LINUX (su Windows e macOS WebGPU e' di serie), ma finche' ogni
//    nostro numero viene da un browser con flag non sappiamo se il motore gira
//    per un utente vero. Un motore dietro flag non e' un prodotto web.
//  - `firefox` e' l'unico altro runtime WebGPU indipendente. ATTENZIONE al
//    reperto che ha motivato questo braccio: i «9,9 tok/s di Firefox» misurati
//    il 2026-07-25 e il 2026-08-17 sono di WEBLLM, non nostri — noi su Firefox
//    non abbiamo MAI girato. E i limiti sono diversi: Firefox 153 dichiara
//    maxBufferSize 1 GiB contro i 4 di Chrome, il che e' un vincolo di
//    STRUTTURA prima che di velocita' per l'arena degli expert.
const BROWSER = process.env.BROWSER ?? "chrome";
const args = BROWSER === "chrome-vanilla"
  ? ["--enable-features=Vulkan,WebGPUService"]
  : ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = BROWSER === "firefox"
  ? await firefox.launchPersistentContext(`${PROFILE}-ff`, { headless: false, firefoxUserPrefs: { "dom.webgpu.enabled": true, "gfx.webgpu.force-enabled": true, "dom.webgpu.workers.enabled": true } })
  : await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[bench][pageerror]", e.message.slice(0, 300)));
page.on("console", (m) => { if (m.text().includes("[engine]")) console.log("[bench]", m.text().slice(0, 300)); });
await page.goto(`${BASE_URL}/engine.html?bench=1`, { waitUntil: "load" });

let last = "";
const poll = setInterval(async () => {
  try {
    const s = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    if (s !== last) { last = s; console.log("[bench]", s); }
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
// Il BRACCIO si dichiara nell'artefatto, come `prefillPath` e `model.levers`:
// un decode di Chrome-con-flag e uno di Firefox non sono confrontabili, e senza
// questo campo sembrano una regressione. Le feature dell'adapter ci vanno
// accanto perche' sono il PERCHE' della differenza (subgroups qui, f16 li').
const adapter = await page.evaluate(async () => {
  const a = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  return a ? { arch: a.info?.architecture ?? null, features: [...a.features].sort(), maxBufferSize: a.limits.maxBufferSize } : null;
}).catch(() => null);
await browser.close();
if (status !== "done" || !report) { console.error("[bench] FALLITO:", status); process.exit(2); }
console.log(`[bench] decode ${report.decodeToksPerSec.mean.toFixed(1)} tok/s (±${report.decodeToksPerSec.stdev.toFixed(1)}) — ${report.dispatchesPerToken} dispatch/token`);
mkdirSync("results/engine", { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, "-");
writeFileSync(`results/engine/bench-4090-${BROWSER}-${ts}.json`, JSON.stringify({ ...report, deviceLabel: "4090-linux", browserArm: BROWSER, chromeArgs: BROWSER === "firefox" ? null : args, adapter, ts, hostState: host.close() }, null, 2));
console.log(`[bench] scritto results/engine/bench-4090-${BROWSER}-${ts}.json  [braccio ${BROWSER}]`);
