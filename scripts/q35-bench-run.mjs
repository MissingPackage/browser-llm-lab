// Riferimenti full-resident 4B (q1 fase 4, it.10): apre /q35conf.html?bench=
// e scrive il JSON con hostState PRIMA/DOPO dichiarato (regola bench: mai
// numeri senza host state). Uso:
//   node scripts/q35-bench-run.mjs [--prompt-idx 4] [--n-decode 64]
//     [--model 4b|9b|35b] [--vram-gib 12] [--arena-gib 11]
//     [--declared user-session-light] [--prefill-m 16] [--out results/engine/...json]
//
// --prefill-m instrada il prompt su `prefillChunk` (M righe per submit) invece
// che su `step` una posizione alla volta. Senza il flag il bench prefilla
// SEQUENZIALMENTE, che e' il path di sempre: il braccio si legge in
// `prefillPath` del JSON, mai dedotto dai numeri (goal engine-ttft riga 0).
//
// --vram-gib e --arena-gib NON sono sinonimi, e la differenza e' il senso della
// fase 5 (it.35, docket item 11): `--arena-gib` ASSERISCE il budget dell'arena
// expert, `--vram-gib` dichiara il TETTO da cui il budget si DERIVA (tetto meno
// l'allocato vero meno la riserva). I riferimenti "ai tier" del CHECKPOINT A
// vogliono il secondo: un tier e' un tetto di VRAM, non un budget che qualcuno
// ha indovinato. Aggiunto in it.43 — la pagina leggeva gia' `?vram=`, mancava
// solo il passaggio qui, e senza il done-when della riga 6 non era eseguibile.
// NOTA: sui modelli DENSI (4b/9b) il tetto non ha effetto — `vramCeilingBytes`
// vive solo nel ramo MoE (q35gpumodel: budget derivato -> ExpertCache) e sui
// densi `vramPlan()` e' null. Un 4B a tier 8 e uno a tier 16 sono lo STESSO run.
import { chromium } from "playwright";
import { copyFileSync, existsSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join } from "node:path";
import { hostState } from "./lib/hoststate.mjs";
import { watchGpuErrors, invalidPath } from "./lib/gpuerrors.mjs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const ROOT = new URL("..", import.meta.url).pathname;
const promptIdx = arg("prompt-idx", "4");
const nDecode = arg("n-decode", "64");
const declared = arg("declared", "undeclared");
const modelTag = arg("model", "4b");
const prefillM = arg("prefill-m", null);
const arenaGiB = arg("arena-gib", null);
const vramGiB = arg("vram-gib", null);
const outTag = vramGiB ? `tier${vramGiB}` : arenaGiB ? `arena${arenaGiB}` : "fullresident";
// `--out` ASSOLUTO o RELATIVO alla root, indifferentemente: i runner di questa
// famiglia scrivevano `out` grezzo (quindi un relativo finiva nella CWD) e
// quelli GLM fanno `join(ROOT, out)` (quindi un assoluto veniva mangiato, e a
// goal precedente e' costato una run GPU da 20 minuti). Una convenzione sola.
const outArg = arg("out", `results/engine/q35-bench-${modelTag}-${outTag}-${new Date().toISOString().slice(0, 10)}.json`);
const out = isAbsolute(outArg) ? outArg : join(ROOT, outArg);
const golden = arg("golden", join(ROOT, `results/engine/golden/q35/golden-q35-${modelTag}-full-2026-08-10.json`));
const PROFILE = process.env.E2E_PROFILE ?? join(homedir(), ".cache/blab-q35conf-profile");
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";

if (!existsSync(golden)) {
  console.error(`[q35bench] golden assente: ${golden}`);
  process.exit(2);
}
copyFileSync(golden, join(ROOT, "public/models/q35/golden-full.json"));

// Il CONTESTO in cui il decode gira e' l'asse che domina il ms/token su questa
// architettura (it.59: ~10,4 us per posizione, cioe' 65,8 ms su 100,5 a
// contesto 6333) — e due sensori che non lo dichiarano SEMBRANO contraddirsi.
// Il JSON lo porta esplicito accanto al numero, non solo dedotto dal prompt.
const hostBefore = hostState(declared);
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
// docket item 24: gli errori GPU non si stampano soltanto — decidono l'esito.
const gpu = watchGpuErrors(page, "q35bench");
await page.goto(`${BASE_URL}/q35conf.html?bench=${promptIdx},${nDecode}${modelTag !== "4b" ? `&model=${modelTag}` : ""}${arenaGiB ? `&arena=${arenaGiB}` : ""}${vramGiB ? `&vram=${vramGiB}` : ""}${prefillM ? `&prefillm=${prefillM}` : ""}`, { waitUntil: "load" });

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
    // `decodeContext`: il punto di lavoro del decode, ESPLICITO. Il ms/token di
    // questa architettura cresce di ~10,4 us per posizione di contesto (it.59:
    // 38,72 ms a ctx 388 contro 100,52 a ctx 6333, stesso host e stessa
    // configurazione), quindi un tok/s senza il suo contesto non e' una misura
    // confrontabile — ed e' il modo in cui il ktest e questo bench sembravano
    // contraddirsi (34,6 contro 38,6) mentre dicevano la stessa cosa.
    const ctxStart = report.prompt?.tokens ?? null;
    const decodeContext = ctxStart === null ? null : {
      startPositions: ctxStart,
      endPositions: ctxStart + (report.decode?.n ?? 0),
      note: "ms/token cresce ~10,4 us per posizione (it.59): confrontare solo a contesto dichiarato",
    };
    const full = { ...report, decodeContext, arenaGiB: arenaGiB ? Number(arenaGiB) : null, vramTierGiB: vramGiB ? Number(vramGiB) : null, gpuErrors: gpu.errors, hostState: { declared, before: hostBefore.state?.before ?? hostBefore, after: hostState(declared).state?.before ?? null } };
    // RUN CONTAMINATA (docket item 24): i numeri sono la velocita' di NON fare
    // il lavoro. Il report si scrive lo stesso — l'evidenza serve — ma MAI al
    // percorso nominale, o un glob su results/engine lo raccoglie come
    // riferimento; e l'exit e' non-zero, cosi' un batch si ferma qui.
    if (gpu.dirty) {
      writeFileSync(invalidPath(out), JSON.stringify({ ...full, invalid: true }, null, 1));
      console.error(`[q35bench] RUN CONTAMINATA: ${gpu.errors.length} errori GPU — i numeri NON sono una misura`);
      for (const e of gpu.errors.slice(0, 3)) console.error(`  [${e.src}] ${e.text.slice(0, 160)}`);
      console.error(`[q35bench] report scritto in ${invalidPath(out)} (fuori dal percorso nominale, di proposito)`);
      process.exit(5);
    }
    writeFileSync(out, JSON.stringify(full, null, 1));
    console.log(`[q35bench] done: decode ${report.decode.tokS.toFixed(2)} tok/s (p50 ${report.decode.msPerTokenP50.toFixed(1)} ms) a contesto ${decodeContext ? `${decodeContext.startPositions}-${decodeContext.endPositions}` : "?"}, prefill ${report.prefill.tokS.toFixed(1)} tok/s [${report.prefillPath ?? "?"}], TTFT ${(report.ttftMs / 1000).toFixed(1)} s -> ${out}`);
    process.exit(0);
  }
  await new Promise((res) => setTimeout(res, 1500));
}
