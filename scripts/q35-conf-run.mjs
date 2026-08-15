// Runner conformance logits Qwen3.5-4B (q1 fase 4, chiusura): copia il golden
// full-corpus in public/, apre /q35conf.html e scrive il report. Pattern
// glm-conf-run.mjs. Uso:
//   node scripts/q35-conf-run.mjs [--golden results/...json] [--prompts 0,1]
//     [--max-gen 128] [--out results/engine/...json] [--timeout-min 240]
import { chromium } from "playwright";
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const ROOT = new URL("..", import.meta.url).pathname;
const modelTag = arg("model", "4b");
const goldenKind = arg("golden-kind", "full");
const golden = arg("golden", join(ROOT, `results/engine/golden/q35/golden-q35-${modelTag}-${goldenKind}-2026-08-10.json`));
const prompts = arg("prompts", null);
const maxGen = arg("max-gen", null);
const out = arg("out", join(ROOT, `results/engine/q35-conf-${modelTag}-${new Date().toISOString().slice(0, 10)}.json`));
const timeoutMin = Number(arg("timeout-min", "240"));
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-q35conf-profile");
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";

if (!existsSync(golden)) {
  console.error(`[q35conf] golden assente: ${golden} (genera con tools/oracle-moe/run-golden-q35.sh)`);
  process.exit(2);
}
copyFileSync(golden, join(ROOT, "public/models/q35/golden-full.json"));
mkdirSync(PROFILE, { recursive: true });

const qs = new URLSearchParams();
if (modelTag !== "4b") qs.set("model", modelTag);
if (prompts) qs.set("prompts", prompts);
if (maxGen) qs.set("maxgen", maxGen);
// Budget dell'arena expert in GiB (solo MoE; la pagina lo legge da `?arena=`).
// Serve DICHIARATO nei run di correttezza: il budget decide quanti slot ci
// sono, quindi hits/misses — due bracci a budget diverso NON sono confrontabili.
const arenaGiB = arg("arena-gib", null);
if (arenaGiB) qs.set("arena", arenaGiB);
// Router GPU in OMBRA (fase-D 3b, fetta 3b): la selezione di produzione resta
// quella della CPU, il resolve GPU scrive una regione parallela di Sel e il
// report porta la fedelta' misurata sui layer VERI.
if (process.argv.includes("--shadow")) qs.set("shadow", "1");
// Profilo dei miss per token (2 passate: fredda e calda) — misura che precede
// la fetta 3c: il decode ottimistico paga solo se i token sporchi sono pochi.
if (process.argv.includes("--misstrace")) qs.set("misstrace", "1");
// GATE della fetta 3c: passata sync (fredda, scalda la cache) + passata a
// submit unico (calda) sulla STESSA cache, submit/token e readback/token
// riportati SEPARATI per passata.
if (process.argv.includes("--optimistic")) qs.set("optimistic", "1");
// Variante del gate: la passata FREDDA la fa il path ottimistico invece del
// sync — cioe' il repair+replay nel suo regime peggiore (a cache vuota ogni
// token e' sporco). Sta in un run a parte perche' la cache fredda esiste una
// volta sola per processo.
if (process.argv.includes("--opt-cold")) { qs.set("optimistic", "1"); qs.set("optcold", "1"); }
// Decomposizione del tempo GPU del token per categoria (fase 4, it.19). PERTURBA:
// spezza il pass di ogni layer in tre, quindi tre barriere invece di una.
if (process.argv.includes("--gpu-time")) qs.set("gputime", "1");
// I due path misurati entrambi a FREDDO nello stesso processo (cache svuotata
// fra i bracci): e' il confronto che la policy d'ingresso della fase 5 esige.
if (process.argv.includes("--cold-both")) qs.set("coldboth", "1");
// Policy di residenza: "tier" = LRU + AUTOPIN (fase 5). Default lru.
if (process.argv.includes("--tier")) qs.set("policy", "tier");
// KFAN (goal engine-velocita-decode, riga 2c): A/B del collasso dei k nel
// decode ottimistico. Implica --optimistic: i due bracci sono entrambi
// ottimistici e differiscono SOLO per il kfan, altrimenti il confronto
// misurerebbe due cose insieme.
if (process.argv.includes("--kfan")) { qs.set("optimistic", "1"); qs.set("kfan", "1"); }
// MISURA della fase 4-ter: quanto costa lo snapshot dello stato ricorrente
// (62,8 MiB per token). Rompe il replay: solo a cache calda.
if (process.argv.includes("--no-snapshot")) qs.set("nosnap", "1");
// GATE della fase 4: prefill a chunk di M righe contro step() sequenziale, coi
// logits confrontati BIT PER BIT.
// Tetto VRAM: il budget dell'arena expert si DERIVA da qui meno cio' che il
// modello ha davvero allocato meno la riserva (docket item 11).
const vramGiB = arg("vram-gib", null);
if (vramGiB) qs.set("vram", vramGiB);
const prefillM = arg("prefill-m", null);
if (prefillM) qs.set("prefillm", prefillM);
// --conf-prefill-m 16: la CONFORMANCE golden prefilla il prompt a chunk (it.21).
// Separato da --prefill-m, che attiva il gate dei due bracci ed esce senza mai
// arrivare al replay golden.
const confPrefillM = arg("conf-prefill-m", null);
if (confPrefillM) qs.set("confprefillm", confPrefillM);
// --gpu-time: accende la sonda per segmento del prefill (riga 5). PERTURBA:
// un pass per segmento invece di uno solo aggiunge barriere.
if (process.argv.includes("--gpu-time")) qs.set("gputime", "1");
const tap = arg("tap", null);
if (tap !== null) qs.set("tap", tap);

const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[q35conf][pageerror]", e.message.slice(0, 300)));
page.on("console", (m) => { const t = m.text(); if (/gpu-error|GPU error|WGSL|validation/i.test(t)) console.log("[q35conf][console]", t.slice(0, 400)); });
await page.goto(`${BASE_URL}/q35conf.html?${qs}`, { waitUntil: "load" });

let lastLive = "";
const t0 = Date.now();
for (;;) {
  if (Date.now() - t0 > timeoutMin * 60_000) {
    console.error("[q35conf] TIMEOUT");
    await browser.close();
    process.exit(3);
  }
  const status = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
  const live = await page.evaluate(() => document.querySelector("#live")?.textContent ?? "");
  if (live && live !== lastLive) {
    console.log(`[q35conf] ${live}`);
    lastLive = live;
  }
  if (status.startsWith("done") || status.startsWith("ERROR")) {
    const report = await page.evaluate(() => window.__report ?? null);
    await browser.close();
    if (!report) {
      console.error(`[q35conf] ${status} — nessun report`);
      process.exit(1);
    }
    writeFileSync(out, JSON.stringify(report, null, 1));
    const r = report.top1 ?? { ok: "-", positions: "-", rate: 0 };
    console.log(`[q35conf] done: top1 ${r.ok}/${r.positions} = ${(r.rate * 100).toFixed(3)}% -> ${out}`);
    process.exit(0);
  }
  await new Promise((res) => setTimeout(res, 2000));
}
