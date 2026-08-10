// Fixture per il ktest GPU dei layer attention qwen35 (q1 fase 4 slice 2).
// Estrae i byte RAW (quantizzati) dei tensori attn di blk.0 (linear/DeltaNet)
// e blk.3 (full/GQA) del 4B pinnato + input sintetici seeded (T posizioni) +
// output attesi dal cpuref-f64 (attnLayerRef: la FONTE UNICA, mai una copia).
// Output: public/models/q35-attn/{meta.json,fixture.bin} — bin NON committato
// (rigenerabile: SHA del GGUF pinnata; pattern glm-layer0).
// Uso: npx vite-node scripts/q35-attn-fixture-gen.mjs
import { closeSync, fstatSync, mkdirSync, openSync, readSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const { Q35CpuRefModel } = await import("../src/engine/q35cpurefmodel.ts");
const { sampleLcg } = await import("../src/engine/q35sample.ts");

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

const MODEL = join(homedir(), ".cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf");
const buf = readLargeFile(MODEL);
const model = new Q35CpuRefModel(buf);
const S = model.shape;
const T = 3;

// input sintetici seeded, scala realistica delle hidden post-embedding
const r = sampleLcg(20260810 ^ 0x51);
const inputs = Array.from({ length: T }, () => Float64Array.from({ length: S.dModel }, () => r() * 0.5));

const LAYERS = [
  { l: 0, kind: "linear", tensors: ["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight", "ssm_conv1d.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight", "ssm_out.weight"] },
  { l: 3, kind: "full", tensors: ["attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight"] },
];

// accesso agli interni del reader (stessa classe: f/byName privati via cast)
const priv = model;
const f = priv.f;
const byName = priv.byName;
const bytes = new Uint8Array(buf);

const parts = [];
let off = 0;
const meta = { note: "fixture attn qwen35 4B (q1 fase 4 slice 2). Rigenerabile: scripts/q35-attn-fixture-gen.mjs. Riferimento = cpuref attnLayerRef.", ggufSha256: "298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb", dims: { d: S.dModel, nHead: S.nHead, nKvHead: S.nKvHead, headDim: S.headDim, ropeDims: S.ropeDims, freqBase: S.ropeFreqBase, rmsEps: S.rmsEps, nK: S.linKHead, nV: S.linVHead, hd: S.linHeadDim, convK: S.linConvK }, T, layers: [], inputs: null, expected: [] };

for (const L of LAYERS) {
  const entry = { l: L.l, kind: L.kind, tensors: [] };
  for (const suffix of L.tensors) {
    const name = `blk.${L.l}.${suffix}`;
    const t = byName.get(name);
    if (!t) throw new Error(`tensore assente: ${name}`);
    const { tensorByteSize } = await import("../src/engine/gguf.ts");
    const nBytes = tensorByteSize(t);
    const src = bytes.subarray(f.dataOffset + t.offset, f.dataOffset + t.offset + nBytes);
    entry.tensors.push({ suffix, type: t.type, dims: t.dims, offset: off, bytes: nBytes });
    parts.push(src);
    off += nBytes;
  }
  meta.layers.push(entry);
}

// input f32 + attesi f32 (dal cpuref f64)
const inF32 = new Float32Array(T * S.dModel);
inputs.forEach((x, t) => inF32.set(Float32Array.from(x), t * S.dModel));
meta.inputs = { offset: off, bytes: inF32.byteLength };
parts.push(new Uint8Array(inF32.buffer));
off += inF32.byteLength;

for (const L of LAYERS) {
  // il cpuref consuma f64 degli STESSI f32 (parità con la GPU)
  const inF64 = Array.from({ length: T }, (_, t) => Float64Array.from(inF32.subarray(t * S.dModel, (t + 1) * S.dModel)));
  const outs = model.attnLayerRef(L.l, inF64);
  const outF32 = new Float32Array(T * S.dModel);
  outs.forEach((o, t) => outF32.set(Float32Array.from(o), t * S.dModel));
  meta.expected.push({ l: L.l, offset: off, bytes: outF32.byteLength });
  parts.push(new Uint8Array(outF32.buffer));
  off += outF32.byteLength;
}

mkdirSync("public/models/q35-attn", { recursive: true });
const bin = new Uint8Array(off);
let w = 0;
for (const p of parts) { bin.set(p, w); w += p.length; }
writeFileSync("public/models/q35-attn/fixture.bin", bin);
writeFileSync("public/models/q35-attn/meta.json", JSON.stringify(meta, null, 1));
console.log(`OK: fixture ${(off / 1e6).toFixed(1)} MB, ${meta.layers.length} layer, T=${T}`);
