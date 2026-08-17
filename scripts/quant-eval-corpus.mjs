// IL CORPUS SU CUI SI GRADUANO DUE QUANTIZZAZIONI DELLO STESSO MODELLO.
//
// PERCHE' NON BASTA UN CORPUS SOLO. Il danno di una quantizzazione **non e'
// uniforme**: un modello che perde poco su prosa corrente puo' perdere molto su
// codice o su aritmetica, dove la distribuzione del prossimo token e' molto piu'
// piccata e mezzo bit sposta l'argmax. Un numero unico medierebbe proprio la
// cosa che decide se il modello e' ancora usabile.
//
// PERCHE' L'ITALIANO C'E'. La funzione obiettivo di questo progetto e' un modello
// che risponde in chat, e le chat del PI sono in italiano su argomenti tecnici.
// Un corpus di sola prosa inglese misurerebbe un regime che non e' il nostro.
//
// DUE FAMIGLIE DI MISURA, e servono a domande diverse:
//   prose  -> bits/token teacher-forced: quanta sorpresa in piu' ha il modello
//             quantizzato sullo STESSO testo. Deterministico, niente
//             generazione, niente varianza di campionamento.
//   tasks  -> log-prob e RANGO della risposta giusta. Non si contano le risposte
//             corrette: e' la landmine gia' pagata da questo repo (un campione da
//             22 posizioni non distingue niente, ±1 colpo vale ±4,5 punti). La
//             log-prob del bersaglio porta la stessa informazione con varianza
//             molto piu' bassa, e a costo zero perche' e' teacher-forced.
//
// OGNI SORGENTE PORTA IL SUO SHA. Un confronto fra due modelli su corpus diversi
// non e' un confronto, e senza sha nessuno se ne accorgerebbe.
//
// Uso:
//   node scripts/quant-eval-corpus.mjs [--out PATH] [--offline] [--tokens-per-domain N]
//
//   --offline   solo le sorgenti LOCALI (repo). Il file dichiara cosa manca.
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..");
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1] : d; };
const has = (n) => process.argv.includes(`--${n}`);
const OUT = arg("out", join(REPO, "results/eval/quant-corpus.json"));
const OFFLINE = has("offline");
// ~4 caratteri per token e' la regola spannometrica del BPE Qwen su testo
// latino: qui serve solo a tagliare le sorgenti a una taglia confrontabile, e la
// taglia VERA in token la misura lo script che tokenizza.
const CHARS_PER_DOMAIN = Number(arg("chars-per-domain", "60000"));

const sha256 = (s) => createHash("sha256").update(s).digest("hex");
const clip = (s, n) => (s.length > n ? s.slice(0, n) : s);

const prose = [];
const tasks = [];
const mancanti = [];

// ---- sorgenti LOCALI: italiano tecnico e codice --------------------------
// Sono file del repo, quindi il corpus e' riproducibile senza rete e senza
// licenze da discutere. Il loro sha finisce nell'artefatto: se il file cambia,
// due misure non sono piu' confrontabili e si vede.
const locali = [
  { id: "it-tecnico", domain: "prosa-it", file: ".harness/goals/engine-velocita-decode/journal.md" },
  { id: "code-ts", domain: "codice", file: "src/engine/residency.ts" },
  // LA CONVERSAZIONE VERA, aggiunta il 2026-08-17. Le altre sezioni sono prosa e
  // codice: nessuna misura il regime in cui il modello viene davvero usato —
  // turni di chat con le marche `<|im_start|>`, il contesto che cresce, e il
  // token di fine turno. Il danno di una quantizzazione puo' benissimo essere
  // diverso li'. La sorgente e' il TRASCRITTO del braccio di riferimento
  // (llama.cpp), reso in testo dallo script `chat-transcript.mjs`.
  { id: "chat-10turni", domain: "conversazione", file: "results/eval/conversazione-10turni.txt" },
];
for (const l of locali) {
  const raw = readFileSync(join(REPO, l.file), "utf8");
  // LA FETTA SI PRENDE DAL MEZZO, e non e' un dettaglio: in questo repo ogni
  // sorgente si apre con un commento lungo in italiano che spiega il PERCHE'.
  // Tagliando dalla testa, la sezione "codice" misurava prosa italiana — e nel
  // fumo si vedeva, perche' costava piu' bit dell'italiano vero. Dal mezzo il
  // testo e' quello che l'etichetta dice.
  const start = Math.max(0, Math.floor((raw.length - CHARS_PER_DOMAIN) / 2));
  const text = raw.slice(start, start + CHARS_PER_DOMAIN);
  prose.push({
    id: l.id, domain: l.domain, source: `${l.file} [${start}..${start + text.length}]`,
    sourceSha256: sha256(raw), chars: text.length, text,
  });
}

// ---- sorgenti DI RETE ------------------------------------------------------
async function getJson(url) {
  const r = await fetch(url, { headers: { accept: "application/json" } });
  if (!r.ok) throw new Error(`${r.status} su ${url}`);
  return r.json();
}
async function getText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} su ${url}`);
  return r.text();
}

if (!OFFLINE) {
  // prosa inglese: wikitext-2 (split di test), che e' il corpus CANONICO su cui
  // la comunita' llama.cpp riporta la perplessita' dei quant. Sta qui perche' e'
  // il solo numero di questo banco che qualcuno fuori dal progetto puo'
  // confrontare con i propri.
  try {
    // l'endpoint `rows` serve al massimo 100 righe per chiamata, e 100 righe di
    // wikitext sono ~24k caratteri: troppo poche per stare alla pari con le
    // altre sezioni. Tre pagine, dichiarate nel nome della sorgente.
    const pagine = await Promise.all([0, 100, 200].map((off) => getJson(
      `https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=test&offset=${off}&length=100`)));
    const body = pagine.flatMap((d) => d.rows.map((r) => r.row.text)).join("");
    prose.push({
      id: "en-prosa", domain: "prosa-en", source: "Salesforce/wikitext wikitext-2-raw-v1 test[0:300]",
      sourceSha256: sha256(body), chars: clip(body, CHARS_PER_DOMAIN).length,
      text: clip(body, CHARS_PER_DOMAIN),
    });
  } catch (e) {
    mancanti.push({ id: "en-prosa", perche: String(e.message ?? e) });
  }

  // GSM8K: la risposta e' un NUMERO dopo «#### », quindi il bersaglio e'
  // verificabile senza giudizio umano e la sua log-prob e' ben definita
  try {
    const d = await getJson("https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main&split=test&offset=0&length=100");
    for (const [i, r] of d.rows.entries()) {
      const q = r.row.question, a = String(r.row.answer);
      const fin = a.split("####").pop().trim();
      tasks.push({
        id: `gsm8k-${i}`, domain: "matematica", source: "openai/gsm8k test",
        prompt: `${q}\n\nRispondi solo con il numero finale.`, answer: fin,
      });
    }
  } catch (e) {
    mancanti.push({ id: "gsm8k", perche: String(e.message ?? e) });
  }

  // MMLU: scelta multipla. Il bersaglio e' una LETTERA, cioe' un token solo —
  // la forma in cui log-prob e rango sono piu' puliti da leggere.
  try {
    const d = await getJson("https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu&config=all&split=test&offset=0&length=100");
    for (const [i, r] of d.rows.entries()) {
      const row = r.row;
      const ch = row.choices;
      const lettere = ["A", "B", "C", "D"];
      tasks.push({
        id: `mmlu-${i}`, domain: "conoscenza", source: "cais/mmlu test",
        subject: row.subject,
        prompt: `${row.question}\n${ch.map((c, k) => `${lettere[k]}. ${c}`).join("\n")}\n\nRispondi con una sola lettera.`,
        answer: lettere[row.answer],
        choices: lettere.slice(0, ch.length),
      });
    }
  } catch (e) {
    mancanti.push({ id: "mmlu", perche: String(e.message ?? e) });
  }
}

const corpus = {
  schemaVersion: 1,
  kind: "quant-eval-corpus",
  date: new Date().toISOString(),
  // il CONTENUTO decide l'identita': due corpus con lo stesso sha sono lo stesso
  // corpus, e il confronto fra due modelli lo pretende
  prose, tasks,
  mancanti,
  charsPerDomain: CHARS_PER_DOMAIN,
};
corpus.corpusSha256 = sha256(JSON.stringify({ prose: corpus.prose, tasks: corpus.tasks }));

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(corpus, null, 1));

console.log(`[corpus] ${prose.length} sezioni di prosa, ${tasks.length} compiti`);
for (const p of prose) console.log(`[corpus]   ${p.id.padEnd(10)} ${p.domain.padEnd(9)} ${String(p.chars).padStart(6)} char  ${p.source}`);
const perDom = {};
for (const t of tasks) perDom[t.domain] = (perDom[t.domain] ?? 0) + 1;
for (const [d, n] of Object.entries(perDom)) console.log(`[corpus]   compiti ${d.padEnd(12)} ${n}`);
if (mancanti.length) {
  console.log(`[corpus] MANCANTI (dichiarati nell'artefatto, non taciuti):`);
  for (const m of mancanti) console.log(`[corpus]   ${m.id}: ${m.perche}`);
}
console.log(`[corpus] sha ${corpus.corpusSha256.slice(0, 16)}… → ${OUT}`);
