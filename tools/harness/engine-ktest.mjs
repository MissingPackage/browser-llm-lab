// Driver del kernel-test motore: apre ktest.html headed (GPU reale), aspetta done,
// stampa la tabella. Uso: node tools/harness/engine-ktest.mjs
//   BASE_URL   (default http://localhost:5173) — il server vite di questa sessione
//   E2E_PROFILE (default /tmp/blab-e2e-profile) — profilo Chrome, UNO ALLA VOLTA
//   KTEST_MIN_PASS (default 90) — soglia di plausibilita', v. sotto
//   KTEST_TIMEOUT_MS (default 600000) — la suite reale gira ~5 min su GPU vera
//
// PERCHE' QUESTO FILE ORA ESCE NON-ZERO ANCHE QUANDO "NON FALLISCE NIENTE"
// (goal engine-ttft, docket item 13). Prima usciva 0/1 solo sullo `status` della
// pagina, e ogni errore PRIMA di quel punto — profilo Chrome occupato da un
// altro runner, server sulla porta sbagliata, timeout — usciva come eccezione
// non gestita: nessun test eseguito, tabella vuota, e un gate scritto come
//     node engine-ktest.mjs | grep -c FAIL   ->   0
// leggeva ZERO FAIL, che e' esattamente cio' che si vede quando va tutto bene.
// E' successo davvero, in it.5: il gate e' stato dichiarato verde due volte su
// una run che non aveva eseguito un solo kernel.
//
// La regola che questo progetto si e' data altrove (sentinella sugli errori GPU
// nei runner di bench: un JSON che mente e' peggio di nessun JSON) vale anche
// qui. Tre difese, in ordine di quanto sono seccanti da aggirare:
//   1. ogni fallimento e' CATTURATO e stampato come "[ktest] FALLITO: <causa>",
//      con l'exit code a 2 — mai un'eccezione nuda;
//   2. ASSERZIONE DI PLAUSIBILITA': se i PASS contati sono meno di
//      KTEST_MIN_PASS, la run e' sospetta anche se lo status dice "done" —
//      zero kernel eseguiti NON e' "nessun fallimento";
//   3. il conteggio PASS/FAIL lo stampa il driver, cosi' chi legge non deve
//      dedurlo da un grep sulla tabella.
import { chromium } from "playwright";
import { copyFileSync, existsSync } from "node:fs";
import { join } from "node:path";

// IL FIXTURE DEL KTEST SE LO PROVVEDE IL KTEST (it.19 di engine-velocita-decode).
// `q35-mtp-draft-4b` confronta un accept-rate col riferimento CPU preso sui
// primi 64 token del prompt 0 del golden FULL del 4B. Prima leggeva
// `/models/q35/golden-full.json` — che e' lo scratch in cui q35-conf-run.mjs e
// q35-bench-run.mjs copiano il golden della LORO run: bastava una run
// `--golden-kind smoke` perche' il ktest misurasse su 37 token invece che su
// 62 e riportasse un FAIL che sembrava una regressione del modello.
// Il fixture ora ha un nome suo, e a metterlo li' e' questo runner: copiarlo
// dal repo a ogni avvio costa un `copyFileSync` e toglie la dipendenza da
// quale bench sia girato per ultimo.
const KTEST_ROOT = new URL("../..", import.meta.url).pathname;
const GOLDEN_SRC = join(KTEST_ROOT, "results/engine/golden/q35/golden-q35-4b-full-2026-08-10.json");
const GOLDEN_DST = join(KTEST_ROOT, "public/models/q35/golden-q35-4b-full.json");

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const MIN_PASS = Number(process.env.KTEST_MIN_PASS ?? 90);
// Il parco kernel gira su GPU reale e comprende i test a livello di modello:
// 120 s NON bastano e il primo disegno di questa difesa ci e' inciampato,
// facendo fallire una run BUONA. Un gate che grida al lupo a run sana viene
// disattivato da chi lo usa, ed e' il modo piu' efficace di tornare al punto
// di partenza. Misurato su 4090 Laptop: ~5 min per 100 kernel.
const TIMEOUT_MS = Number(process.env.KTEST_TIMEOUT_MS ?? 600000);
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

/** Esce dichiarando la causa. Mai un'eccezione nuda: v. l'intestazione. */
const die = (why, code = 2) => {
  console.error(`[ktest] FALLITO: ${why}`);
  console.error("[ktest] NESSUN VERDETTO — questa run non e' un gate superato.");
  process.exit(code);
};

if (existsSync(GOLDEN_SRC)) {
  copyFileSync(GOLDEN_SRC, GOLDEN_DST);
} else if (!existsSync(GOLDEN_DST)) {
  // non e' un `die`: il resto del parco kernel non dipende da questo file, e
  // il caso che lo usa fallisce da solo dicendo cosa manca
  console.error(`[ktest] ATTENZIONE: golden 4B full assente (${GOLDEN_SRC}) — q35-mtp-draft-4b fallira' dichiarandolo`);
}

let browser = null;
try {
  browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
} catch (e) {
  const msg = String(e?.message ?? e);
  // La causa piu' frequente, e quella che in it.5 ha prodotto il falso verde.
  if (/existing browser session|already in use/i.test(msg)) {
    die(`il profilo Chrome ${PROFILE} e' gia' in uso da un altro runner.\n` +
      `  I bench e i ktest sono SERIALI: uno alla volta sullo stesso profilo.\n` +
      `  Libera il profilo, oppure passa E2E_PROFILE=/tmp/blab-ktest-<qualcosa>.\n` +
      `  Dettaglio: ${msg.slice(0, 300)}`);
  }
  die(`avvio del browser: ${msg.slice(0, 400)}`);
}

try {
  const page = browser.pages()[0] ?? (await browser.newPage());
  page.on("pageerror", (e) => console.log("[ktest][pageerror]", e.message.slice(0, 300)));

  try {
    await page.goto(`${BASE_URL}/ktest.html`, { waitUntil: "load" });
  } catch (e) {
    const msg = String(e?.message ?? e);
    if (/ERR_CONNECTION_REFUSED|ECONNREFUSED/i.test(msg)) {
      die(`nessun server su ${BASE_URL}.\n` +
        `  Il default e' la porta 5173; se il tuo vite gira altrove passa\n` +
        `  BASE_URL=http://localhost:<porta> node tools/harness/engine-ktest.mjs\n` +
        `  Dettaglio: ${msg.slice(0, 200)}`);
    }
    die(`navigazione a ${BASE_URL}/ktest.html: ${msg.slice(0, 400)}`);
  }

  try {
    await page.waitForFunction(
      () => {
        const s = document.querySelector("#status")?.textContent ?? "";
        return s === "done" || s.startsWith("ERROR");
      },
      null, { timeout: TIMEOUT_MS, polling: 500 },
    );
  } catch {
    const partial = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "(nessuno status)").catch(() => "(pagina non leggibile)");
    die(`TIMEOUT dopo ${(TIMEOUT_MS / 1000).toFixed(0)} s. Ultimo status: ${partial}`, 3);
  }

  const status = await page.evaluate(() => document.querySelector("#status").textContent);
  const adapter = await page.evaluate(() => document.querySelector("#probe-box").textContent);
  const table = await page.evaluate(() => document.querySelector("#results").innerText);
  await browser.close();
  browser = null;

  console.log("[ktest] adapter:", adapter);
  console.log(table);

  // Il conteggio lo fa il driver: chi legge non deve dedurlo da un grep, e la
  // soglia di plausibilita' e' l'unica difesa contro una tabella vuota che
  // "non ha fallimenti".
  const nPass = (table.match(/\tPASS\t/g) ?? []).length;
  const nFail = (table.match(/\tFAIL\t/g) ?? []).length;
  console.log(`[ktest] STATUS: ${status} — PASS ${nPass} · FAIL ${nFail}`);

  if (status !== "done") die(`la pagina ha riportato status "${status}"`, 1);
  if (nFail > 0) die(`${nFail} kernel FALLITI su ${nPass + nFail}`, 1);
  if (nPass < MIN_PASS) {
    die(`solo ${nPass} kernel eseguiti, sotto la soglia di plausibilita' ${MIN_PASS}.\n` +
      `  ZERO FALLIMENTI SU ZERO TEST NON E' UN GATE SUPERATO. Se il parco kernel\n` +
      `  e' legittimamente calato, abbassa KTEST_MIN_PASS DICHIARANDOLO.`, 4);
  }
  console.log(`[ktest] OK — ${nPass} PASS, 0 FAIL (soglia di plausibilita' ${MIN_PASS}).`);
  process.exit(0);
} catch (e) {
  if (browser) await browser.close().catch(() => {});
  die(`errore inatteso: ${String(e?.message ?? e).slice(0, 400)}`);
}
