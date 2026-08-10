// Riferimenti full-resident 4B (q1 fase 4, it.10): apre /q35conf.html?bench=
// e scrive il JSON con hostState PRIMA/DOPO dichiarato (regola bench: mai
// numeri senza host state). Uso:
//   node scripts/q35-bench-run.mjs [--prompt-idx 4] [--n-decode 64]
//     [--declared user-session-light] [--out results/engine/...json]
import { chromium } from "playwright";
import { copyFileSync, existsSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { hostState } from "./lib/hoststate.mjs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const ROOT = new URL("..", import.meta.url).pathname;
const promptIdx = arg("prompt-idx", "4");
const nDecode = arg("n-decode", "64");
const declared = arg("declared", "undeclared");
const out = arg("out", join(ROOT, `results/engine/q35-bench-4b-fullresident-${new Date().toISOString().slice(0, 10)}.json`));
const golden = join(ROOT, "results/engine/golden/q35/golden-q35-4b-full-2026-08-10.json");
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-q35conf-profile");
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";

if (!existsSync(golden)) {
  console.error(`[q35bench] golden assente: ${golden}`);
  process.exit(2);
}
copyFileSync(golden, join(ROOT, "public/models/q35/golden-full.json"));

const hostBefore = hostState(declared);
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[q35bench][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/q35conf.html?bench=${promptIdx},${nDecode}`, { waitUntil: "load" });

const t0 = Date.now();
for (;;) {
  if (Date.now() - t0 > 30 * 60_000) {
    console.error("[q35bench] TIMEOUT");
    await browser.close();
    process.exit(3);
  }
  const status = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
  if (status.startsWith("done") || status.startsWith("ERROR")) {
    const report = await page.evaluate(() => window.__report ?? null);
    await browser.close();
    if (!report || status.startsWith("ERROR")) {
      console.error(`[q35bench] ${status} — nessun report`);
      process.exit(1);
    }
    // hostState() restituisce già {declared, before, after}: niente doppio
    // annidamento (nota verifier it.10) — before campionato all'avvio, after qui
    const full = { ...report, hostState: { declared, before: hostBefore.state?.before ?? hostBefore, after: hostState(declared).state?.before ?? null } };
    writeFileSync(out, JSON.stringify(full, null, 1));
    console.log(`[q35bench] done: decode ${report.decode.tokS.toFixed(2)} tok/s (p50 ${report.decode.msPerTokenP50.toFixed(1)} ms), prefill ${report.prefill.tokS.toFixed(1)} tok/s, TTFT ${(report.ttftMs / 1000).toFixed(1)} s -> ${out}`);
    process.exit(0);
  }
  await new Promise((res) => setTimeout(res, 1500));
}
