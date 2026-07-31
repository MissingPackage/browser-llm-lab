// Runner conformance routing GLM (C2 fase 5 slice 3b): prepara la traccia
// (jsonl.gz → JSON servibile), il symlink del GGUF, apre /glmroute.html su
// Chrome headed (profilo SU DISCO: l'OPFS da 17.2 GB non può stare nel tmpfs
// di /tmp) e scrive il report. Uso:
//   node scripts/glm-route-run.mjs [--cap N] [--prompts 0,1] [--budget-gib 12]
//     [--out results/engine/routing-....json] [--timeout-min 180]
// La vite su :5199 va avviata dal chiamante (come ktest-run).
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { homedir } from "node:os";
import { join } from "node:path";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const cap = arg("cap", null);
const prompts = arg("prompts", null);
const budget = arg("budget-gib", "12");
const out = arg("out", null);
const timeoutMin = Number(arg("timeout-min", "180"));

const ROOT = new URL("..", import.meta.url).pathname;
const TRACE_GZ = join(ROOT, "results/engine/moe-oracle/trace-2026-07-31.jsonl.gz");
const TRACE_JSON = join(ROOT, "public/models/glm-route-trace.json");
const GGUF = join(homedir(), ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf");
const GGUF_LINK = join(ROOT, "public/models/GLM-4.7-Flash-Q4_0.gguf");
// profilo SU DISCO (OPFS 17.2 GB): mai /tmp (tmpfs 16 GB)
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-glmroute-profile");
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";

// --- prep ---
if (!existsSync(GGUF)) throw new Error(`GGUF assente: ${GGUF}`);
if (!existsSync(GGUF_LINK)) symlinkSync(GGUF, GGUF_LINK);
if (!existsSync(TRACE_JSON)) {
  console.log("[glmroute] preparo la traccia JSON…");
  const lines = gunzipSync(readFileSync(TRACE_GZ)).toString("utf8").trimEnd().split("\n");
  const header = JSON.parse(lines[0]);
  const rows = lines.slice(1).map((l) => JSON.parse(l));
  writeFileSync(TRACE_JSON, JSON.stringify({ header, rows }));
  console.log(`[glmroute] traccia: ${rows.length} righe`);
}
mkdirSync(PROFILE, { recursive: true });

const qs = new URLSearchParams();
if (cap) qs.set("cap", cap);
if (prompts) qs.set("prompts", prompts);
qs.set("budget", budget);

const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[glmroute][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/glmroute.html?${qs}`, { waitUntil: "load" });

let lastLive = "";
const t0 = Date.now();
for (;;) {
  if (Date.now() - t0 > timeoutMin * 60_000) {
    console.error("[glmroute] TIMEOUT");
    await browser.close();
    process.exit(3);
  }
  const status = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
  const live = await page.evaluate(() => document.querySelector("#live")?.textContent ?? "");
  if (live && live !== lastLive) {
    console.log(`[glmroute] ${live}`);
    lastLive = live;
  }
  if (status.startsWith("done") || status.startsWith("ERROR")) {
    const report = await page.evaluate(() => window.__report ?? null);
    await browser.close();
    if (!report) {
      console.error(`[glmroute] ${status} — nessun report`);
      process.exit(2);
    }
    if (out) {
      writeFileSync(join(ROOT, out), JSON.stringify(report, null, 1));
      // determinismo del report: sha del file scritto nel log
      const sha = execFileSync("sha256sum", [join(ROOT, out)]).toString().split(" ")[0];
      console.log(`[glmroute] report → ${out} (sha256 ${sha})`);
    }
    const g = report.gate ?? {};
    const d = report.setMatch?.decode ?? {};
    const p = report.setMatch?.prefill ?? {};
    console.log(`[glmroute] ${status} — decode ${d.match}/${d.total} (${d.pct?.toFixed(4)}%) prefill ${p.match}/${p.total} (${p.pct?.toFixed(4)}%) gate=${g.pass}`);
    process.exit(status === "done" ? 0 : status === "done-gate-fail" ? 4 : 2);
  }
  await new Promise((r) => setTimeout(r, 5000));
}
