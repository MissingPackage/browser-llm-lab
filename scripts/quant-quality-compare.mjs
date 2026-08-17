// IL CONFRONTO FRA DUE QUANTIZZAZIONI, appaiato posizione per posizione.
//
// Prende due artefatti `quant-quality` (uno per GGUF) e risponde a UNA domanda:
// **quanto costa, in bit per token e in log-prob sulla risposta giusta, passare
// dal quant A al quant B?** — con l'intervallo di confidenza accanto, perche' un
// delta senza incertezza non decide niente.
//
// LE TRE PRECONDIZIONI, verificate e non assunte. Se cadono, lo script si
// RIFIUTA invece di produrre un numero che sembra buono:
//   1. stesso corpus (sha);
//   2. stessi token nelle stesse posizioni — due tokenizer diversi renderebbero
//      l'appaiamento una finzione, e la differenza misurerebbe il tokenizer;
//   3. stesso schema di finestra (taglia, BOS).
//
// COSA STAMPA, e perche' ognuna delle quattro colonne serve:
//   bit/token A e B   il livello: dove sta ciascuno dei due
//   Δbit [lo, hi]     il costo appaiato, con bootstrap a blocchi (la media da
//                     sola non dice se e' distinguibile dal rumore)
//   vittorie          test di SEGNO: su quante posizioni B e' meglio. Immune
//                     alla coda che trascina la media
//   top-1 diverso     quanto spesso i due modelli direbbero un token diverso:
//                     e' la traduzione della perdita in «si comporta uguale?»
//
// Uso:
//   node scripts/quant-quality-compare.mjs A.json B.json [--out C.json] [--block 512]
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { blockBootstrapCI, mean, pairedNll, quantiles, toBits, winRate } from "./lib/pairedstats.mjs";

const argv = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1] : d; };
if (argv.length !== 2) {
  console.error("uso: node scripts/quant-quality-compare.mjs A.json B.json [--out C.json] [--block 512]");
  process.exit(2);
}
const A = JSON.parse(readFileSync(argv[0], "utf8"));
const B = JSON.parse(readFileSync(argv[1], "utf8"));
const BLOCK = Number(arg("block", "512"));
const OUT = arg("out", null);

const fail = (m) => { console.error(`[cmp] RIFIUTO: ${m}`); process.exit(3); };
if (A.kind !== "quant-quality" || B.kind !== "quant-quality") fail("uno dei due file non e' un artefatto quant-quality");
if (A.corpus.sha256 !== B.corpus.sha256) {
  fail(`corpus diversi (${A.corpus.sha256.slice(0, 12)} vs ${B.corpus.sha256.slice(0, 12)}) — non e' un confronto`);
}
if (A.model.sha256 && A.model.sha256 === B.model.sha256) fail("i due artefatti vengono dallo STESSO GGUF");
// stesso ORACOLO: due versioni di llama.cpp possono differire nei kernel, e la
// differenza finirebbe nel delta attribuita al quant. La 0.3.16 non carica
// nemmeno queste architetture, quindi il caso non e' teorico.
if (A.oracle.version !== B.oracle.version) {
  fail(`oracoli diversi (llama-cpp-python ${A.oracle.version} vs ${B.oracle.version}) — il delta misurerebbe anche il motore`);
}
if (A.oracle.nCtx !== B.oracle.nCtx) fail(`n_ctx diverso (${A.oracle.nCtx} vs ${B.oracle.nCtx}): le finestre non coincidono`);

console.log(`[cmp] A = ${A.model.file}  (${(A.model.bytes / 2 ** 30).toFixed(2)} GiB)`);
console.log(`[cmp] B = ${B.model.file}  (${(B.model.bytes / 2 ** 30).toFixed(2)} GiB)`);
console.log(`[cmp] corpus ${A.corpus.sha256.slice(0, 16)}… · blocco bootstrap ${BLOCK} token\n`);

// ---- prosa: bits/token appaiati -------------------------------------------
const byId = new Map(B.prose.map((s) => [s.id, s]));
const prose = [];
console.log("sezione      dominio    token    bit/tok A   bit/tok B      Δbit  [IC 95%]           B meglio   top-1 diverso");
for (const a of A.prose) {
  const b = byId.get(a.id);
  if (!b) fail(`la sezione "${a.id}" manca nell'artefatto B`);
  if (a.windowTokens !== b.windowTokens || a.addBos !== b.addBos) fail(`sezione "${a.id}": schema di finestra diverso`);
  if (a.tokens.length !== b.tokens.length) fail(`sezione "${a.id}": ${a.tokens.length} vs ${b.tokens.length} token`);
  for (let i = 0; i < a.tokens.length; i++) {
    if (a.tokens[i] !== b.tokens[i]) {
      fail(`sezione "${a.id}": i token differiscono alla posizione ${i} (${a.tokens[i]} vs ${b.tokens[i]}) — `
        + "i due modelli non hanno visto lo stesso testo, e l'appaiamento sarebbe una finzione");
    }
  }
  const r = pairedNll(a.nll, b.nll, { blockLen: BLOCK });
  let diversi = 0;
  for (let i = 0; i < a.top1.length; i++) if (a.top1[i] !== b.top1[i]) diversi++;
  const rec = {
    id: a.id, domain: a.domain, nTokens: r.n,
    bitsA: r.bitsA, bitsB: r.bitsB, deltaBits: r.deltaBits,
    win: r.win, top1Diff: diversi / a.top1.length, deltaTail: r.deltaTail,
  };
  prose.push(rec);
  const ic = `[${r.deltaBits.lo >= 0 ? "+" : ""}${r.deltaBits.lo.toFixed(4)}, ${r.deltaBits.hi >= 0 ? "+" : ""}${r.deltaBits.hi.toFixed(4)}]`;
  console.log(
    `${a.id.padEnd(12)} ${a.domain.padEnd(9)} ${String(r.n).padStart(6)}   ${r.bitsA.toFixed(4).padStart(8)}    ${r.bitsB.toFixed(4).padStart(8)}  `
    + `${(r.deltaBits.mean >= 0 ? "+" : "") + r.deltaBits.mean.toFixed(4)}  ${ic.padEnd(20)} `
    + `${(100 * r.win.fracBetter).toFixed(1).padStart(6)}%   ${(100 * rec.top1Diff).toFixed(1).padStart(6)}%`);
}

// ---- compiti: log-prob e rango del bersaglio ------------------------------
const bTask = new Map(B.tasks.map((t) => [t.id, t]));
const perDomain = new Map();
for (const a of A.tasks) {
  const b = bTask.get(a.id);
  if (!b) fail(`il compito "${a.id}" manca nell'artefatto B`);
  if (a.answerTokens.length !== b.answerTokens.length
    || a.answerTokens.some((t, i) => t !== b.answerTokens[i])) {
    fail(`compito "${a.id}": la risposta e' tokenizzata diversamente — l'appaiamento non e' definito`);
  }
  const d = perDomain.get(a.domain) ?? { dLogprob: [], rankA: [], rankB: [], peggiora: 0, n: 0, top1DiffA: 0 };
  // per-TOKEN della risposta normalizzato: risposte lunghe non devono pesare di
  // piu' solo perche' sono lunghe
  d.dLogprob.push((b.logprobSum - a.logprobSum) / a.answerTokens.length);
  d.rankA.push(a.rank1);
  d.rankB.push(b.rank1);
  if (b.rank1 > a.rank1) d.peggiora++;
  if (a.top1 !== b.top1) d.top1DiffA++;
  d.n++;
  perDomain.set(a.domain, d);
}
if (perDomain.size) {
  console.log("\ncompiti      n     Δlogprob/token  [IC 95%]            rango1 mediano A→B   1° token diverso");
  for (const [dom, d] of perDomain) {
    const ci = blockBootstrapCI(d.dLogprob, { blockLen: 1, resamples: 2000 }); // item indipendenti: blocco 1
    const mA = quantiles(d.rankA, [0.5])["0.5"], mB = quantiles(d.rankB, [0.5])["0.5"];
    const ic = `[${ci.lo >= 0 ? "+" : ""}${ci.lo.toFixed(4)}, ${ci.hi >= 0 ? "+" : ""}${ci.hi.toFixed(4)}]`;
    console.log(
      `${dom.padEnd(12)} ${String(d.n).padStart(3)}   ${(ci.mean >= 0 ? "+" : "") + ci.mean.toFixed(4)}        ${ic.padEnd(21)}`
      + `${String(mA).padStart(6)} → ${String(mB).padEnd(8)}  ${(100 * d.top1DiffA / d.n).toFixed(1).padStart(6)}%`);
  }
}

console.log("\n[cmp] LETTURA: Δbit > 0 = B piu' sorpreso = piu' danno. Δlogprob < 0 = B da' meno");
console.log("[cmp] probabilita' alla risposta giusta. Se l'IC contiene lo zero, la differenza non e'");
console.log("[cmp] distinguibile dal rumore su QUESTO corpus — non che non esista.");

if (OUT) {
  const out = {
    schemaVersion: 1, kind: "quant-quality-compare", date: new Date().toISOString(),
    a: { file: A.model.file, bytes: A.model.bytes, sha256: A.model.sha256 },
    b: { file: B.model.file, bytes: B.model.bytes, sha256: B.model.sha256 },
    corpusSha256: A.corpus.sha256, blockLen: BLOCK,
    oracle: { a: A.oracle, b: B.oracle },
    prose,
    tasks: [...perDomain.entries()].map(([dom, d]) => ({
      domain: dom, n: d.n,
      deltaLogprobPerToken: blockBootstrapCI(d.dLogprob, { blockLen: 1, resamples: 2000 }),
      rank1MedianA: quantiles(d.rankA, [0.5])["0.5"], rank1MedianB: quantiles(d.rankB, [0.5])["0.5"],
      rank1MeanA: mean(d.rankA), rank1MeanB: mean(d.rankB),
      peggiorati: d.peggiora, top1Diversi: d.top1DiffA,
    })),
  };
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, JSON.stringify(out, null, 1));
  console.log(`[cmp] scritto ${OUT}`);
}
