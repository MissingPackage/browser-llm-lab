// CHECKPOINT B (goal fase-D riga 8): l'artefatto dello speculative decoding MTP.
//
// Non e' un bench nuovo: guida il ktest, che le misure le fa gia' (fase 7,
// it.53-56), e ne estrae le tre righe MTP con lo stato dell'host DICHIARATO
// accanto. Esiste perche' un numero che vive solo in un log non e' un
// artefatto: la riga 8 chiede un JSON committato, e il ramo che si sta
// esercitando e' quello dell'ESCLUSIONE COI NUMERI.
//
// Uso: npx vite (porta 5173) in un terminale, poi
//   node scripts/q35-specdec-run.mjs [--host-state quiescent]
//
// Exit: 0 solo se le tre righe MTP sono PASS e la GPU non ha sporcato la run;
// 5 se contaminata (report scritto su .INVALID, mai al percorso nominale).
import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright";
import { hostState } from "./lib/hoststate.mjs";
import { invalidPath, watchGpuErrors } from "./lib/gpuerrors.mjs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const argv = process.argv.slice(2);
const declared = argv.includes("--host-state") ? argv[argv.indexOf("--host-state") + 1] : "undeclared";
const label = process.env.DEVICE_LABEL ?? "4090";

const KEYS = ["q35-mtp-draft-4b", "q35-mtp-specdec-invariance", "q35-mtp-costo-verifica"];

const hs = hostState(declared);
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
const gpu = watchGpuErrors(page, "specdec");
await page.goto(`${BASE_URL}/ktest.html`, { waitUntil: "load" });
await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 900000, polling: 500 },
);
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const adapter = await page.evaluate(() => document.querySelector("#probe-box").textContent);
const report = await page.evaluate(() => window.__report ?? []);
await browser.close();
hs.close();

const rows = KEYS.map((k) => report.find((r) => r.kernel === k) ?? null);
const missing = KEYS.filter((_, i) => rows[i] === null);
const allPass = rows.every((r) => r && r.pass);
const m = Object.fromEntries(KEYS.map((k, i) => [k, rows[i]]));
const met = (k, f) => m[k]?.metrics?.[f] ?? null;

const seqMs = met("q35-mtp-specdec-invariance", "msPerTokenSeq");
const specBMs = met("q35-mtp-specdec-invariance", "msPerTokenSpecBatched");
const out = {
  schemaVersion: 1,
  kind: "q35-specdec-mtp-checkpoint-b",
  decision: "excluded-by-numbers",
  // La decisione in chiaro, perche' un JSON di numeri senza la frase che li
  // lega si rilegge fra un mese come "dati" invece che come "verdetto".
  verdict:
    "Lo speculative decoding con la testa MTP e' COSTRUITO, CORRETTO E VERIFICATO " +
    "(gate secco: i token generati col draft sono identici a quelli sequenziali), ma NON PAGA " +
    "su questo motore: i kernel batch mettono la riga su wid.z e ogni riga RILEGGE i pesi, " +
    "quindi verificare due posizioni costa 1,7-1,9 volte una invece di ~1. " +
    "Il codice resta in albero, gated e testato, e torna utile se/quando esistera' un GEMM " +
    "a piu' righe con riuso vero (premio proiettato: ~23,5 ms/token = ~42 tok/s).",
  model: "Qwen3.5-4B-MTP-Q4_0.gguf",
  msPerToken: {
    sequential: seqMs,
    specPerRow: met("q35-mtp-specdec-invariance", "msPerTokenSpec"),
    specBatched: specBMs,
    ratioSpecBatchedOverSequential: seqMs && specBMs ? specBMs / seqMs : null,
  },
  acceptRate: {
    corpusWindow64Pct: met("q35-mtp-draft-4b", "acceptPct"),
    cpuRefWindow64Pct: 50.0, // riferimento f64 di it.51, stessa finestra
    freeGenerationPct: met("q35-mtp-specdec-invariance", "acceptedPct"),
    msPerDraft: met("q35-mtp-draft-4b", "msPerDraft"),
  },
  costBreakdownMs: {
    tokenBody: met("q35-mtp-costo-verifica", "bodyMs"),
    tokenTail: met("q35-mtp-costo-verifica", "tailMs"),
    verifyPass: met("q35-mtp-costo-verifica", "passMs"),
    verifyBody2Rows: met("q35-mtp-costo-verifica", "body2Ms"),
    body2OverBody: met("q35-mtp-costo-verifica", "body2Ratio"),
  },
  gates: { allPass, missing, ktestStatus: status },
  notes: rows.map((r) => (r ? { kernel: r.kernel, pass: r.pass, note: r.note } : null)),
  repro: [
    "npx vite  # porta 5173",
    "node scripts/q35-specdec-run.mjs --host-state quiescent",
    "riferimento CPU f64: Q35_MTP=1 npx vitest run tests/engine-q35-mtp-accept.test.ts",
  ],
  deviceLabel: label,
  adapter,
  ts: new Date().toISOString(),
  hostState: hs.state,
};

mkdirSync("results/engine", { recursive: true });
const nominal = `results/engine/specdec-${label}-${out.ts.replace(/[:.]/g, "-")}.json`;
const path = gpu.dirty ? invalidPath(nominal) : nominal;
writeFileSync(path, JSON.stringify(out, null, 1));
console.log(`[specdec] scritto ${path}`);
console.log(`[specdec] sequenziale ${seqMs?.toFixed(1)} ms/token · spec batch ${specBMs?.toFixed(1)} · rapporto ${out.msPerToken.ratioSpecBatchedOverSequential?.toFixed(2)}x`);
if (gpu.dirty) {
  console.log(`[specdec] RUN CONTAMINATA: ${gpu.errors.length} errori GPU, report in quarantena`);
  process.exit(5);
}
if (!allPass || status !== "done") {
  console.log(`[specdec] righe mancanti o FAIL: ${JSON.stringify(missing)} · status ${status}`);
  process.exit(2);
}
