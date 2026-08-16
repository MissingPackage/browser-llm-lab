// CONVERTITORE OFFLINE GGUF → file slab, per la famiglia Qwen 3.5/3.6.
//
// PERCHE' OFFLINE E NON AL PRIMO CARICAMENTO. Il ruling del PI (docket item 10)
// dice «accetta entrambi i formati e converti al primo caricamento». La seconda
// meta' ha incontrato un muro misurato in it.43: la quota OPFS dell'origine e'
// **10,00 GiB**, `persist()` viene NEGATA, e il file slab del 35B e' **17,07
// GiB**. Convertire lato client vuole un posto dove scrivere, e in un browser
// quel posto e' OPFS. Quindi il 35B prende la prima meta' del ruling: uno slab
// **gia' convertito**, servito come il GGUF via HTTP Range. Zero OPFS, zero
// quota.
//
// COSA TOGLIE, misurato (it.40 sul turno vero del PI):
//   pack 7,11 s per sessione  ->  0
//   38.625 richieste Range    ->  12.875   (uno slab si legge in un colpo)
// `packExpertSlab` e' una funzione pura dei byte grezzi: stesso input, stesso
// output, sempre. E' cio' che rende lecito precalcolarla.
//
// SICUREZZA. Si scrive su un TEMPORANEO e si rinomina alla fine: un'interruzione
// non lascia mai un file valido a meta'. E' la stessa postura di `ensureSlabs`.
//
// Uso:
//   node scripts/q35-slab-build.mjs --model 35b [--out PATH] [--layers 0-1]
//                                   [--verify 8] [--dry-run]
//
//   --layers a-b   converte solo quei layer (VERIFICA: il file NON e' completo
//                  e il suo header lo dichiara con una geometria parziale)
//   --verify N     rilegge N slab a campione e li confronta con packExpertSlab
//   --dry-run      calcola geometria e taglia, non scrive niente
//
// ATTENZIONE ALLO SPAZIO: il file completo del 35B e' ~17 GiB. Lo script lo
// DICHIARA e rifiuta di partire se il disco non ha almeno il doppio (il
// temporaneo piu' il rinominato coesistono per un istante).
import { openSync, readSync, writeSync, closeSync, existsSync, statSync, renameSync, unlinkSync, mkdirSync } from "node:fs";
import { statfsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1] : d; };
const has = (n) => process.argv.includes(`--${n}`);

const MODELS = {
  "4b": "q35/Qwen3.5-4B-Q4_0.gguf",
  "9b": "q35/Qwen3.5-9B-Q4_0.gguf",
  "35b": "q35/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
};
const tag = arg("model", "35b");
if (!MODELS[tag]) { console.error(`[slab] modello "${tag}" sconosciuto: ${Object.keys(MODELS).join(", ")}`); process.exit(2); }
const GGUF = join(homedir(), ".cache/blab-models", MODELS[tag]);
if (!existsSync(GGUF)) { console.error(`[slab] GGUF assente: ${GGUF}`); process.exit(2); }

const { parseGguf } = await import("../src/engine/gguf.ts");
const { validateQwen35 } = await import("../src/engine/q35shape.ts");
const { q35SlabDesc, q35ExpertTensor } = await import("../src/engine/q35expertstore.ts");
const { slabGeometry, slabRangeOf } = await import("../src/engine/slabgeom.ts");
const { buildSlabHeader, slabFileBytes, SLAB_HEADER_BYTES } = await import("../src/engine/slabfile.ts");
const { packExpertSlab } = await import("../src/engine/moe.ts");
const { createHash } = await import("node:crypto");

// ---- header del GGUF e SHA ------------------------------------------------
const fd = openSync(GGUF, "r");
const hdr = Buffer.alloc(64 * 1024 * 1024);
readSync(fd, hdr, 0, hdr.length, 0);
const f = parseGguf(hdr.buffer.slice(hdr.byteOffset, hdr.byteOffset + hdr.length));
const { shape, byName } = validateQwen35(f);
const info = (n) => { const t = byName.get(n); if (!t) throw new Error(`tensore ${n} assente`); return t; };

// LO SHA DEL SORGENTE finisce nell'header dello slab e nel nome del file: e' la
// sola cosa che impedisce di leggere uno slab prodotto da un ALTRO GGUF, che
// darebbe pesi validi e sbagliati invece di un errore.
process.stdout.write("[slab] SHA-256 del GGUF… ");
const sha = await new Promise((res, rej) => {
  const h = createHash("sha256");
  const fdh = openSync(GGUF, "r");
  const buf = Buffer.alloc(64 * 1024 * 1024);
  let off = 0;
  try {
    for (;;) {
      const n = readSync(fdh, buf, 0, buf.length, off);
      if (n <= 0) break;
      h.update(buf.subarray(0, n));
      off += n;
    }
    res(h.digest("hex"));
  } catch (e) { rej(e); } finally { closeSync(fdh); }
});
console.log(sha.slice(0, 16) + "…");

const desc = q35SlabDesc(shape, info, sha);
const geo = slabGeometry(desc);
const total = slabFileBytes(geo);
const OUT = arg("out", join(homedir(), ".cache/blab-models/q35", desc.fileName));

console.log(`[slab] ${tag}: ${shape.nLayer} layer × ${shape.nExpert} expert = ${geo.nSlabs} slab`);
for (const c of geo.classes) {
  console.log(`[slab]   classe ${c.id.padEnd(6)} ${String(c.nSlabs).padStart(6)} slab × ${c.bytes} B = ${(c.nSlabs * c.bytes / 2 ** 30).toFixed(2)} GiB`);
}
console.log(`[slab] file: ${(total / 2 ** 30).toFixed(2)} GiB  →  ${OUT}`);

// ---- il conto dello spazio, PRIMA di scrivere ------------------------------
const layersArg = arg("layers", null);
const [L0, L1] = layersArg
  ? layersArg.split("-").map(Number)
  : [desc.denseLead, desc.nLayer - 1];
const parziale = layersArg !== null;
if (parziale) console.log(`[slab] PARZIALE: solo i layer ${L0}-${L1} — il file NON e' utilizzabile dal motore`);

if (has("dry-run")) { console.log("[slab] dry-run: niente scritto."); closeSync(fd); process.exit(0); }

mkdirSync(dirname(OUT), { recursive: true });
const fsst = statfsSync(dirname(OUT));
const free = fsst.bavail * fsst.bsize;
const serve = parziale ? total / 8 : total * 2; // il tmp e il rinominato coesistono
if (free < serve) {
  console.error(`[slab] spazio insufficiente: ${(free / 2 ** 30).toFixed(1)} GiB liberi, ne servono ~${(serve / 2 ** 30).toFixed(1)}`);
  console.error("[slab] (il temporaneo e il file finale coesistono per un istante)");
  process.exit(2);
}

// ---- conversione -----------------------------------------------------------
const TMP = `${OUT}.tmp`;
if (existsSync(TMP)) unlinkSync(TMP);
const out = openSync(TMP, "w+");
const t0 = Date.now();
let done = 0, bytesOut = 0;
try {
  writeSync(out, Buffer.from(buildSlabHeader(geo, sha)), 0, SLAB_HEADER_BYTES, 0);

  const nE = shape.nExpert;
  const readRaw = (name, off, len) => {
    const b = Buffer.alloc(len);
    const n = readSync(fd, b, 0, len, f.dataOffset + info(name).offset + off);
    if (n !== len) throw new Error(`lettura corta su ${name}: ${n}/${len}`);
    return new Uint8Array(b.buffer, b.byteOffset, len);
  };

  for (let l = L0; l <= L1; l++) {
    const per = {};
    for (const w of ["gate", "up", "down"]) {
      const t = info(q35ExpertTensor(l, w));
      per[w] = (t.dims[0] * t.dims[1] * bytesPerElem(t)) | 0;
    }
    const lay = desc.layoutOf(l);
    for (let e = 0; e < nE; e++) {
      const slab = packExpertSlab(
        readRaw(q35ExpertTensor(l, "gate"), e * per.gate, per.gate),
        readRaw(q35ExpertTensor(l, "up"), e * per.up, per.up),
        readRaw(q35ExpertTensor(l, "down"), e * per.down, per.down),
        lay);
      const r = slabRangeOf(geo, l, e);
      writeSync(out, Buffer.from(slab.buffer, slab.byteOffset, slab.length), 0, slab.length,
        SLAB_HEADER_BYTES + r.offset);
      bytesOut += slab.length;
      if (++done % 512 === 0) {
        const s = (Date.now() - t0) / 1000;
        process.stdout.write(`\r[slab] ${done} slab · ${(bytesOut / 1e9).toFixed(2)} GB · ${(bytesOut / 1e6 / s).toFixed(0)} MB/s   `);
      }
    }
  }
  const s = (Date.now() - t0) / 1000;
  console.log(`\r[slab] ${done} slab · ${(bytesOut / 1e9).toFixed(2)} GB in ${s.toFixed(1)} s · ${(bytesOut / 1e6 / s).toFixed(0)} MB/s        `);

  // ---- verifica a campione: i byte scritti == packExpertSlab --------------
  const nVerify = Number(arg("verify", "8"));
  if (nVerify > 0) {
    let bad = 0;
    for (let i = 0; i < nVerify; i++) {
      const l = L0 + Math.floor((i * (L1 - L0 + 1)) / nVerify);
      const e = Math.floor((i * nE) / nVerify);
      const lay = desc.layoutOf(l);
      const per = {};
      for (const w of ["gate", "up", "down"]) {
        const t = info(q35ExpertTensor(l, w));
        per[w] = (t.dims[0] * t.dims[1] * bytesPerElem(t)) | 0;
      }
      const atteso = packExpertSlab(
        readRaw(q35ExpertTensor(l, "gate"), e * per.gate, per.gate),
        readRaw(q35ExpertTensor(l, "up"), e * per.up, per.up),
        readRaw(q35ExpertTensor(l, "down"), e * per.down, per.down),
        lay);
      const r = slabRangeOf(geo, l, e);
      const got = Buffer.alloc(r.bytes);
      readSync(out, got, 0, r.bytes, SLAB_HEADER_BYTES + r.offset);
      let diff = -1;
      for (let k = 0; k < atteso.length; k++) if (got[k] !== atteso[k]) { diff = k; break; }
      if (diff >= 0) { bad++; console.error(`[slab] MISMATCH blk.${l}/${e} al byte ${diff}`); }
    }
    if (bad) { console.error(`[slab] ${bad}/${nVerify} slab NON combaciano: il file NON viene rinominato`); process.exit(4); }
    console.log(`[slab] verifica: ${nVerify}/${nVerify} slab identici a packExpertSlab`);
  }
} finally {
  closeSync(out);
  closeSync(fd);
}

if (parziale) {
  console.log(`[slab] PARZIALE: lascio il temporaneo ${TMP} (non e' un file valido)`);
} else {
  renameSync(TMP, OUT);
  console.log(`[slab] scritto ${OUT} — ${(statSync(OUT).size / 2 ** 30).toFixed(2)} GiB`);
}

function bytesPerElem(t) {
  // byte per elemento del formato, dal blocco: e' la stessa aritmetica di
  // q35TensorBytes ma qui serve la fetta di UN expert
  const G = { 2: [32, 18], 3: [32, 20], 8: [32, 34], 12: [256, 144], 13: [256, 176], 14: [256, 210] };
  const g = G[t.type];
  if (!g) throw new Error(`formato ggml ${t.type} non gestito dal convertitore`);
  return g[1] / g[0];
}
