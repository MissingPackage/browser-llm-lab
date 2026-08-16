// Smoke della chat q35: carica il 4B, manda un prompt, verifica che i token
// arrivino e che l'export JSON contenga i parametri.
import { chromium } from "playwright";
import { homedir } from "node:os";
import { join } from "node:path";
import { readFileSync, mkdirSync } from "node:fs";

const BASE = process.env.BASE_URL ?? "http://localhost:5199";
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1] : d; };
const MODEL = arg("model", "4b");
const CTX = arg("ctx", "1024");
const VRAM = arg("vram", "13");
const MAXNEW = arg("maxnew", "48");
// --prompt: il testo del PRIMO turno. Serve a riprodurre un turno vero invece
// del prompt corto dello smoke — un turno da 60 token e uno da 800 pagano la
// stessa tassa di residenza fissa, quindi danno tok/s molto diversi e NON sono
// confrontabili fra loro (it.40).
// --policy tier: autopin dei top-usage (it.48). Il default resta "lru", che e'
// cio' che la chat fa oggi — cambiarlo qui e' una MISURA, non una decisione.
const POLICY = arg("policy", null);
const PROMPT = arg("prompt", "Scrivi una frase sola: perche' il cielo e' blu?");
const PROFILE = join(homedir(), ".cache/blab-glmroute-profile");
mkdirSync(PROFILE, { recursive: true });
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist", "--disable-gpu-sandbox"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args, acceptDownloads: true });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[pageerror]", e.message.slice(0, 300)));
page.on("console", (m) => { if (m.type() === "error") console.log("[console.error]", m.text().slice(0, 300)); });

let code = 0;
try {
  await page.goto(`${BASE}/chat.html`, { waitUntil: "load" });
  await page.selectOption("#model", MODEL);
  await page.fill("#ctx", CTX);
  await page.fill("#vram", VRAM);
  await page.fill("#maxnew", MAXNEW);
  await page.fill("#temp", "0");
  if (POLICY) await page.selectOption("#policy", POLICY);
  await page.click("#load");
  await page.waitForFunction(() => document.getElementById("status").textContent.startsWith("pronto"), null, { timeout: 240000 });
  console.log("[smoke] caricato:", await page.textContent("#status"));

  await page.fill("#input", PROMPT);
  await page.click("#send");
  await page.waitForSelector(".meta", { timeout: 240000 });
  const reply = await page.textContent(".msg.assistant .body");
  const meta = await page.textContent(".meta");
  console.log("[smoke] risposta:", JSON.stringify(reply.slice(0, 240)));
  console.log("[smoke] metriche:", meta);
  if (!reply || reply.trim().length === 0) throw new Error("risposta VUOTA");

  // secondo turno: verifica il riuso della KV (posStart deve ripartire da posEnd)
  await page.fill("#input", "E di notte?");
  await page.click("#send");
  await page.waitForFunction(() => document.querySelectorAll(".meta").length === 2, null, { timeout: 240000 });
  console.log("[smoke] turno 2:", (await page.textContent("#status")));

  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 30000 }), page.click("#export-json")]);
  const p = `/tmp/chat-smoke-export-${MODEL}.json`;
  await dl.saveAs(p);
  const j = JSON.parse(readFileSync(p, "utf8"));
  const t = j.turns.filter((x) => x.role === "assistant");
  console.log("[smoke] export:", {
    kind: j.kind,
    model: j.model.file,
    sha: j.model.sha256.slice(0, 8),
    load: j.params.load,
    sampling: j.params.sampling,
    hostState: j.hostState,
    turni: j.turns.length,
    tokS: t.map((x) => x.stats.decodeTokS?.toFixed(2)),
    posizioni: t.map((x) => `${x.stats.posStart}→${x.stats.posEnd}`),
    stop: t.map((x) => x.stats.stopReason),
    dispatchTotal: j.model.dispatchBreakdown.total,
    vramPlanGiB: j.final?.vramPlan ? (j.final.vramPlan.allocatedBytes / 2 ** 30).toFixed(2) : null,
    renderedPrompt0: t[0].stats.renderedPrompt,
    chatTemplatePresente: j.params.chatTemplateRaw !== null,
  });
  if (t[1].stats.posStart !== t[0].stats.posEnd) throw new Error(`KV non riusata fra i turni: ${t[0].stats.posEnd} → ${t[1].stats.posStart}`);
  if (j.params.load.vramGiB !== Number(VRAM)) throw new Error("parametri di carico non esportati");
  console.log("[smoke] OK");
} catch (e) {
  console.error("[smoke] FALLITO:", e.message);
  code = 1;
} finally {
  await browser.close();
}
process.exit(code);
