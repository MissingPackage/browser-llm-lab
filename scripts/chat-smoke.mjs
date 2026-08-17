// LA CHAT VERA, SU UNA CONVERSAZIONE VERA — non uno smoke da due turni.
//
// PERCHE' DIECI TURNI (ruling del PI, 2026-08-17). Con due turni si misura la
// fase in cui l'arena si RIEMPIE, non il regime in cui la chat vive: il primo
// turno di un MoE paga i miss di popolamento (7.930 sul 35B), il secondo li ha
// gia' quasi tutti residenti. Una conversazione di dieci turni con domande di
// approfondimento mostra la CURVA — dove il tok/s si stabilizza, se il contesto
// che cresce lo mangia, e se i miss tendono a zero o no.
//
// Il thread e' scelto per assomigliare all'uso vero: una domanda tecnica e nove
// follow-up che approfondiscono, in italiano, con un turno di riassunto che
// obbliga il modello a rileggere tutto il contesto. Il PRIMO turno e' quello dei
// bracci precedenti, cosi' il confronto storico regge.
//
// COSA MISURA CHE UN TURNO SOLO NON PUO':
//   - il tok/s a regime, quando l'arena e' calda;
//   - il costo del contesto che cresce (la KV si riusa fra i turni);
//   - se i miss tendono a zero (parco che ci sta) o si ripresentano (non ci sta).
//
// Uso:
//   node scripts/chat-smoke.mjs --model 35b-q2k --ctx 8192 --vram 13 --maxnew 400
//     [--turns 10] [--policy tier] [--prompt "…"] [--conversation file.json]
//     [--out PATH]
import { chromium } from "playwright";
import { homedir } from "node:os";
import { join } from "node:path";
import { readFileSync, mkdirSync, copyFileSync } from "node:fs";

const BASE = process.env.BASE_URL ?? "http://localhost:5199";
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1] : d; };
const MODEL = arg("model", "4b");
const CTX = arg("ctx", "8192");
const VRAM = arg("vram", "13");
const MAXNEW = arg("maxnew", "400");
const TURNS = Number(arg("turns", "10"));
const POLICY = arg("policy", null);
const OUT = arg("out", null);
// --thinking auto|1|0. `auto` (default) lascia decidere il template del file, che
// sul Qwen3.6 vuol dire ragionamento ACCESO da quando la polarita' si deriva
// (2026-08-17). NON e' un dettaglio di comodo: due run in modalita' diverse
// generano quantita' di token diverse e non si confrontano — il 34,97 tok/s del
// 2026-08-17 fu misurato PRIMA che il default cambiasse, quindi per confrontarsi
// con lui serve `--thinking 0` ESPLICITO. L'artefatto lo riporta da solo in
// `model.chatTemplate`.
const THINKING = arg("thinking", "auto");

/**
 * LA CONVERSAZIONE. Il turno 1 e' quello dei bracci del 2026-08-17 (confronto
 * storico); i nove seguenti sono follow-up che approfondiscono, non domande
 * indipendenti — e' la differenza fra una chat e una batteria di prompt.
 * Il turno 9 chiede un RIASSUNTO: costringe a rileggere tutto il contesto, che
 * e' il caso peggiore per la KV e per il prefill.
 */
const CONVERSAZIONE = [
  "Che relazione c'e' tra entropia dell'informazione e compressione?",
  "Puoi farmi un esempio numerico concreto con un alfabeto di quattro simboli?",
  "E se le probabilita' fossero tutte uguali, cosa cambierebbe in quell'esempio?",
  "Come si collega alla codifica di Huffman? Mostrami l'albero per l'esempio di prima.",
  "Perche' la codifica aritmetica riesce a fare meglio di Huffman?",
  "Nei modelli linguistici si parla di bit per token: e' la stessa entropia di cui parlavamo?",
  "Quindi una perplessita' di 6 quanti bit per token sono, e come la interpreto?",
  "Se quantizzo i pesi di un modello, cosa succede a quei bit per token?",
  "Riassumi in cinque punti quello che ci siamo detti finora.",
  "Dove cadrebbe questo ragionamento se i dati non fossero stazionari?",
];
const CONV_FILE = arg("conversation", null);
const conv = CONV_FILE ? JSON.parse(readFileSync(CONV_FILE, "utf8")) : CONVERSAZIONE.slice();
const P1 = arg("prompt", null);
if (P1) conv[0] = P1;
const turni = conv.slice(0, Math.max(1, TURNS));

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
  await page.selectOption("#thinking", THINKING);
  // Stessa logica dell'asserzione sul modello qui sotto: il selettore si rilegge
  // DOPO averlo impostato. Un `selectOption` su un valore che l'option non ha
  // lancia, ma un controllo che qualcuno rinomina lascerebbe la run girare nella
  // modalita' sbagliata senza dirlo — ed e' invisibile nei numeri.
  const thinkSel = await page.$eval("#thinking", (e) => e.value);
  if (thinkSel !== THINKING) throw new Error(`ragionamento "${thinkSel}" != richiesto "${THINKING}"`);
  await page.click("#load");
  await page.waitForFunction(() => document.getElementById("status").textContent.startsWith("pronto"), null, { timeout: 300000 });
  const caricato = await page.textContent("#status");
  console.log(`[chat] caricato: ${caricato}`);
  // IL MODELLO CARICATO DEV'ESSERE QUELLO CHIESTO. Il 2026-08-17 una run
  // lanciata con `--model 35b-q2k` ha caricato il 4B e ha misurato per tre
  // turni senza che niente protestasse: i numeri erano plausibili e di un altro
  // modello. Il selettore della pagina e' l'unica fonte, e va riletto DOPO il
  // caricamento, non solo impostato.
  const selezionato = await page.$eval("#model", (e) => e.value);
  if (selezionato !== MODEL) {
    throw new Error(`modello caricato "${selezionato}" != richiesto "${MODEL}" — la run avrebbe misurato un altro modello`);
  }
  console.log(`[chat] ${turni.length} turni · ctx ${CTX} · vram ${VRAM} · maxnew ${MAXNEW}${POLICY ? ` · policy ${POLICY}` : ""}\n`);

  for (const [i, q] of turni.entries()) {
    const t0 = Date.now();
    await page.fill("#input", q);
    await page.click("#send");
    // il turno e' finito quando compare la sua riga di metriche
    await page.waitForFunction((n) => document.querySelectorAll(".meta").length === n, i + 1, { timeout: 600000 });
    const meta = (await page.$$eval(".meta", (els) => els.map((e) => e.textContent)))[i];
    console.log(`[chat] turno ${String(i + 1).padStart(2)}/${turni.length} · ${((Date.now() - t0) / 1000).toFixed(0)}s · ${q.slice(0, 58)}…`);
    console.log(`         ${meta?.replace(/\s+/g, " ").trim().slice(0, 150)}`);
  }

  const [dl] = await Promise.all([page.waitForEvent("download", { timeout: 60000 }), page.click("#export-json")]);
  const p = `/tmp/chat-smoke-export-${MODEL}.json`;
  await dl.saveAs(p);
  if (OUT) { copyFileSync(p, OUT); console.log(`[chat] artefatto → ${OUT}`); }

  const j = JSON.parse(readFileSync(p, "utf8"));
  const t = j.turns.filter((x) => x.role === "assistant");
  console.log(`\n[chat] ${j.model.file} · sha ${j.model.sha256.slice(0, 8)} · leve ${JSON.stringify(j.model.levers ?? null)}`);
  console.log("[chat] turno   tok/s   TTFT ms    gen   ctx        miss   dirty  replayLayers");
  for (const [i, x] of t.entries()) {
    const s = x.stats, m = s.moe ?? {}, e = s.engine ?? {};
    // I contatori sono PER TURNO, non cumulativi di sessione: la pagina li
    // azzera a ogni invio. Verificato sull'artefatto del 2026-08-17 — la prima
    // versione di questa tabella li differenziava e stampava miss NEGATIVI, che
    // e' il modo in cui un'assunzione sbagliata si annuncia da sola.
    const miss = m.misses ?? 0, dirty = e.dirtyTokens ?? 0, rep = e.replayLayers ?? 0;
    console.log(
      `[chat]  ${String(i + 1).padStart(4)}  ${(s.decodeTokS ?? 0).toFixed(2).padStart(6)}  ${String(Math.round(s.ttftMs ?? 0)).padStart(8)}`
      + `  ${String(s.genTokens).padStart(5)}  ${String(s.posEnd).padStart(5)}  ${String(miss).padStart(9)}  ${String(dirty).padStart(6)}  ${String(rep).padStart(12)}`);
  }
  // La riga che dice se il regime esiste: la media degli ULTIMI turni, quando
  // l'arena e' calda. La media su tutti mescola il popolamento col regime.
  const coda = t.slice(Math.floor(t.length / 2));
  const medio = coda.reduce((a, x) => a + (x.stats.decodeTokS ?? 0), 0) / Math.max(1, coda.length);
  console.log(`[chat] regime (media degli ultimi ${coda.length} turni): ${medio.toFixed(2)} tok/s`);

  if (t.length >= 2 && t[1].stats.posStart !== t[0].stats.posEnd) {
    throw new Error(`KV non riusata fra i turni: ${t[0].stats.posEnd} → ${t[1].stats.posStart}`);
  }
  if (j.params.load.vramGiB !== Number(VRAM)) throw new Error("parametri di carico non esportati");
  if (t.some((x) => !x.content || x.content.trim().length === 0)) throw new Error("un turno ha risposta VUOTA");
  console.log("[chat] OK");
} catch (e) {
  console.error("[chat] FALLITO:", e.message);
  code = 1;
} finally {
  await browser.close();
}
process.exit(code);
