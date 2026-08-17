// L'ISTOGRAMMA DEI TIPI DI UN GGUF, per categoria di tensore.
//
// A COSA SERVE. Prima di valutare una quantizzazione candidata servono due
// numeri che NON stanno nella taglia del file: quanto pesa il **parco expert**
// (l'unica parte che deve stare nell'arena) e **quali formati** contiene (cioe'
// quali kernel servono). La taglia del file li mescola con embedding, attn e
// shared expert, che stanno in VRAM comunque.
//
// PERCHE' RIESISTE. `results/engine/q35-header-dump-2026-08-10.json` — che
// `q35shape.ts` e `q35prefillsites.ts` citano come la fonte dei formati pinnati
// — era stato prodotto da un comando ad hoc, sparito con la sessione che lo
// scrisse. Un artefatto che due sorgenti citano deve avere uno strumento che lo
// rigenera.
//
// Usa il parser del MOTORE (`gguf.ts`): se un giorno il parser sbagliasse,
// questo dump sbaglierebbe allo stesso modo — ed e' la proprieta' giusta,
// perche' la domanda e' «cosa ci vede il nostro motore», non «cosa c'e'».
//
// SI PUO' PUNTARE A UN URL, ed e' la ragione per cui questo script vale piu' del
// comando ad hoc che sostituisce: l'header sta nei primi MB del file, quindi
// **valutare un quant candidato costa 64 MB invece di 15 GB**. La domanda «quanto
// pesa il parco e in che formati e'» si risponde prima di scaricare.
//
// Uso:
//   node scripts/q35-header-dump.mjs FILE.gguf [https://…/altro.gguf] [--out PATH]
import { openSync, readSync, closeSync, writeFileSync, mkdirSync, statSync } from "node:fs";
import { dirname } from "node:path";

const { importaTs } = await import("./lib/tsimport.mjs");
const { parseGguf, tensorByteSize, GGML_TYPE } = await importaTs("src/engine/gguf.ts");

const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1] : d; };
const files = process.argv.slice(2).filter((a) => a.endsWith(".gguf") && !a.startsWith("--"));
const HDR = 64 * 1024 * 1024;

/** I primi HDR byte, da disco o via Range HTTP (stesso trattamento a valle). */
async function testata(f) {
  if (/^https?:\/\//.test(f)) {
    const r = await fetch(f, { headers: { Range: `bytes=0-${HDR - 1}` } });
    if (r.status !== 206) throw new Error(`${f}: Range non onorato (${r.status})`);
    const totale = Number(/\/(\d+)$/.exec(r.headers.get("Content-Range") ?? "")?.[1] ?? 0);
    const ab = await r.arrayBuffer();
    return { ab, bytes: totale };
  }
  const fd = openSync(f, "r");
  const buf = Buffer.alloc(HDR);
  readSync(fd, buf, 0, HDR, 0);
  closeSync(fd);
  return { ab: buf.buffer.slice(buf.byteOffset, buf.byteOffset + HDR), bytes: statSync(f).size };
}
if (files.length === 0) { console.error("uso: node scripts/q35-header-dump.mjs FILE.gguf [...] [--out PATH]"); process.exit(2); }
const OUT = arg("out", null);

const NOME_TIPO = Object.fromEntries(Object.entries(GGML_TYPE).map(([k, v]) => [v, k]));

// I TIPI CHE IL MOTORE NON CONOSCE, per poterli comunque PESARE.
//
// `tensorByteSize` lancia sui tipi che il motore non sa leggere — ed e' giusto
// che lanci nel motore. Ma qui la domanda e' esattamente «quanto pesa cio' che
// non sappiamo leggere», quindi serve una tabella di ripiego. Sono i type-trait
// di ggml (blocco = 256 pesi per tutte le K e le I, tranne IQ4_NL a 32).
//
// LA TABELLA SI VERIFICA DA SOLA: la somma dei tensori piu' l'header deve dare
// la taglia del file. Lo script lo controlla e lo stampa — una tabella scritta a
// memoria che non tornasse coi byte veri sarebbe peggio di nessuna tabella.
const RIPIEGO = {
  10: ["Q2_K", 256, 84], 11: ["Q3_K", 256, 110], 15: ["Q8_K", 256, 292],
  16: ["IQ2_XXS", 256, 66], 17: ["IQ2_XS", 256, 74], 18: ["IQ3_XXS", 256, 98],
  19: ["IQ1_S", 256, 50], 20: ["IQ4_NL", 32, 18], 21: ["IQ3_S", 256, 110],
  22: ["IQ2_S", 256, 82], 23: ["IQ4_XS", 256, 136], 29: ["IQ1_M", 256, 56],
  // non-quant: compaiono nei quant aggressivi, che tengono il ROUTER alto
  // (bartowski lascia `ffn_gate_inp` in BF16) — e un tensore non pesato
  // falserebbe il conto del parco
  24: ["I8", 1, 1], 25: ["I16", 1, 2], 26: ["I32", 1, 4], 27: ["I64", 1, 8],
  28: ["F64", 1, 8], 30: ["BF16", 1, 2],
};

const nomeDi = (t) => NOME_TIPO[t.type] ?? RIPIEGO[t.type]?.[0] ?? `ggml${t.type}`;

/** byte del tensore: dal motore se li sa, dalla tabella di ripiego se no. */
function bytesDi(t) {
  try {
    return { bytes: tensorByteSize(t), noto: true };
  } catch {
    const r = RIPIEGO[t.type];
    if (!r) throw new Error(`tipo ggml ${t.type} sconosciuto anche alla tabella di ripiego (${t.name})`);
    const [, wpb, bpb] = r;
    const elems = t.dims.reduce((a, b) => a * b, 1);
    return { bytes: (elems / wpb) * bpb, noto: false };
  }
}

/** La categoria di un tensore dal suo NOME: e' la partizione che conta per la VRAM. */
function categoria(name) {
  if (/^(token_embd|output)/.test(name)) return "embd/head";
  if (/_exps\./.test(name)) return "expert";            // il PARCO: l'unica parte pagata a slot
  if (/shexp/.test(name)) return "ffn/shexp";           // shared expert: sempre residente
  if (/attn/.test(name)) return "attn";
  if (/ssm|conv1d|dt_|A_log|linear/.test(name)) return "linear_attn";
  return "other";
}

const out = [];
for (const f of files) {
  const { ab, bytes: fileBytes } = await testata(f);
  const g = parseGguf(ab);

  const hist = {};
  let expertBytes = 0, sommaTensori = 0;
  const ignoti = new Set();
  for (const t of g.tensors) {
    const { bytes, noto } = bytesDi(t);
    const k = `${categoria(t.name)}:${nomeDi(t)}${noto ? "" : " *"}`;
    const e = hist[k] ?? (hist[k] = { n: 0, bytes: 0, lettoDalMotore: noto });
    e.n++;
    e.bytes += bytes;
    sommaTensori += bytes;
    if (!noto) ignoti.add(nomeDi(t));
    if (categoria(t.name) === "expert") expertBytes += bytes;
  }
  const meta = {};
  for (const [k, v] of Object.entries(g.metadata)) {
    if (Array.isArray(v)) meta[`${k}#count`] = v.length;
    else if (typeof v === "string" && v.length > 200) meta[`${k}#chars`] = v.length;
    else meta[k] = v;
  }
  const rec = {
    file: f, bytes: fileBytes, nTensors: g.tensors.length, meta,
    typeHistogram: Object.fromEntries(Object.entries(hist).sort((a, b) => b[1].bytes - a[1].bytes)),
    // IL NUMERO CHE DECIDE la residenza: il parco, non il file
    expertParkBytes: expertBytes,
    expertParkGiB: +(expertBytes / 2 ** 30).toFixed(3),
    nonExpertGiB: +((fileBytes - expertBytes) / 2 ** 30).toFixed(3),
    // i formati che il MOTORE non sa leggere: e' la lista dei kernel mancanti
    formatiNonSupportati: [...ignoti],
    // controllo della tabella di ripiego: somma dei tensori vs taglia del file
    sommaTensoriBytes: sommaTensori,
    scartoVsFileFrazione: +((fileBytes - sommaTensori) / fileBytes).toFixed(5),
  };
  out.push(rec);

  console.log(`\n${f}`);
  console.log(`  ${(rec.bytes / 2 ** 30).toFixed(2)} GiB · ${rec.nTensors} tensori · arch ${meta["general.architecture"]}`);
  for (const [k, v] of Object.entries(rec.typeHistogram)) {
    console.log(`    ${k.padEnd(22)} ${String(v.n).padStart(4)} tensori  ${(v.bytes / 2 ** 30).toFixed(3).padStart(8)} GiB`);
  }
  console.log(`  PARCO EXPERT ${rec.expertParkGiB} GiB · resto ${rec.nonExpertGiB} GiB`);
  if (rec.formatiNonSupportati.length) {
    console.log(`  ⚠ FORMATI CHE IL MOTORE NON LEGGE (marcati *): ${rec.formatiNonSupportati.join(", ")}`);
  }
  // se la tabella di ripiego fosse sbagliata, questo scarto non sarebbe ~0
  console.log(`  controllo taglie: somma tensori ${(sommaTensori / 2 ** 30).toFixed(3)} GiB su file ${(rec.bytes / 2 ** 30).toFixed(3)} — scarto ${(100 * rec.scartoVsFileFrazione).toFixed(3)}% (header e allineamenti)`);
}

if (OUT) {
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, JSON.stringify(out, null, 1));
  console.log(`\n[dump] scritto ${OUT}`);
}
