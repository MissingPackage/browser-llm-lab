// Fixture per il ktest GPU della TESTA MTP (goal fase-D, fase 7, it.52).
// Estrae i byte RAW (quantizzati) dei 15 tensori di blk.<nLayer> del 4B-MTP
// pinnato + input sintetici seeded (T posizioni: embedding e hidden GREZZI) +
// output attesi dal cpuref-f64 (`mtpHeadRef`: la FONTE UNICA, mai una copia).
// Output: public/models/q35-mtp/{meta.json,fixture.bin} — bin NON committato
// (rigenerabile: SHA del GGUF pinnata; stesso pattern di q35-attn).
//
// L'atteso e' il vettore che entra nella lm_head CONDIVISA, cioe' il blocco
// senza la proiezione sul vocabolario: la lm_head e' gia' coperta dal ktest del
// modello pieno, e tenerla fuori fa stare il fixture in pochi MB invece che in
// 1,3 GB di token_embd.
// Uso: npx vite-node scripts/q35-mtp-fixture-gen.mjs
import { closeSync, fstatSync, mkdirSync, openSync, readSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const { Q35CpuRefModel } = await import("../src/engine/q35cpurefmodel.ts");
const { sampleLcg } = await import("../src/engine/q35sample.ts");
const { tensorByteSize } = await import("../src/engine/gguf.ts");

function readLargeFile(path) {
  const fd = openSync(path, "r");
  const size = fstatSync(fd).size;
  const ab = new ArrayBuffer(size);
  const CHUNK = 1 << 30;
  for (let off = 0; off < size; off += CHUNK) {
    const len = Math.min(CHUNK, size - off);
    readSync(fd, new Uint8Array(ab, off, len), 0, len, off);
  }
  closeSync(fd);
  return ab;
}

const MODEL = join(homedir(), ".cache/blab-models/q35/Qwen3.5-4B-MTP-Q4_0.gguf");
const buf = readLargeFile(MODEL);
const model = new Q35CpuRefModel(buf);
const S = model.shape;
if (S.mtpLayers < 1) throw new Error("il GGUF non porta la testa MTP");
const T = 3;
const d = S.dModel;

// Input sintetici seeded. Scale DIVERSE per i due ingressi e non per capriccio:
// l'embedding grezzo e l'hidden finale del modello vivono su ordini di
// grandezza diversi, e le due RMSNorm della testa esistono proprio per questo —
// dare loro la stessa scala nasconderebbe uno scambio fra enorm e hnorm.
const r = sampleLcg(20260812 ^ 0x77);
const embIn = Array.from({ length: T }, () => Float64Array.from({ length: d }, () => r() * 0.05));
const hidIn = Array.from({ length: T }, () => Float64Array.from({ length: d }, () => r() * 2.0));

const SUFFIXES = [
  "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.eh_proj.weight", "nextn.shared_head_norm.weight",
  "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
  "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight",
  "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
];

const priv = model;
const f = priv.f;
const byName = priv.byName;
const bytes = new Uint8Array(buf);

const parts = [];
let off = 0;
const meta = {
  note: "fixture testa MTP qwen35 4B (fase-D fase 7 it.52). Rigenerabile: scripts/q35-mtp-fixture-gen.mjs. Riferimento = cpuref mtpHeadRef.",
  ggufSha256: "14e6ef39302330c63c2c1a1ab548c7f6f1b7e36b3150ca8b42cab7193b0c3669",
  dims: { d, nHead: S.nHead, nKvHead: S.nKvHead, headDim: S.headDim, ropeDims: S.ropeDims, freqBase: S.ropeFreqBase, rmsEps: S.rmsEps, dFfn: S.dFfn },
  T, blk: S.nLayer, tensors: [], emb: null, hidden: null, expected: null,
};

for (const suffix of SUFFIXES) {
  const name = `blk.${S.nLayer}.${suffix}`;
  const t = byName.get(name);
  if (!t) throw new Error(`tensore assente: ${name}`);
  const nBytes = tensorByteSize(t);
  parts.push(bytes.subarray(f.dataOffset + t.offset, f.dataOffset + t.offset + nBytes));
  meta.tensors.push({ suffix, type: t.type, dims: t.dims, offset: off, bytes: nBytes });
  off += nBytes;
}

// input f32 (la GPU legge questi), atteso f32 dal cpuref f64 sugli STESSI f32
const embF32 = new Float32Array(T * d), hidF32 = new Float32Array(T * d);
embIn.forEach((x, t) => embF32.set(Float32Array.from(x), t * d));
hidIn.forEach((x, t) => hidF32.set(Float32Array.from(x), t * d));
meta.emb = { offset: off, bytes: embF32.byteLength };
parts.push(new Uint8Array(embF32.buffer));
off += embF32.byteLength;
meta.hidden = { offset: off, bytes: hidF32.byteLength };
parts.push(new Uint8Array(hidF32.buffer));
off += hidF32.byteLength;

const embF64 = Array.from({ length: T }, (_, t) => Float64Array.from(embF32.subarray(t * d, (t + 1) * d)));
const hidF64 = Array.from({ length: T }, (_, t) => Float64Array.from(hidF32.subarray(t * d, (t + 1) * d)));
const outs = model.mtpHeadRef(embF64, hidF64, true);
const outF32 = new Float32Array(T * d);
outs.forEach((o, t) => outF32.set(Float32Array.from(o), t * d));
meta.expected = { offset: off, bytes: outF32.byteLength };
parts.push(new Uint8Array(outF32.buffer));
off += outF32.byteLength;

mkdirSync("public/models/q35-mtp", { recursive: true });
const bin = new Uint8Array(off);
let w = 0;
for (const p of parts) { bin.set(p, w); w += p.length; }
writeFileSync("public/models/q35-mtp/fixture.bin", bin);
writeFileSync("public/models/q35-mtp/meta.json", JSON.stringify(meta, null, 1));
console.log(`OK: fixture ${(off / 1e6).toFixed(1)} MB, ${meta.tensors.length} tensori, T=${T}, blk.${S.nLayer}`);
