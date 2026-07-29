// Shape hardcoded di Qwen2.5-0.5B-Instruct (GGUF ufficiale Qwen, q4_0) + validazione
// hard del file (postura ds4: ogni divergenza è throw, mai "best effort").
//
// Inventario VERIFICATO sul file reale (2026-07-29, parse di
// qwen2.5-0.5b-instruct-q4_0.gguf): 291 tensori. Due sorprese rispetto alla
// config HF che la spec deve recepire:
//   1. output.weight ESISTE separato (niente tied embeddings nel GGUF) ed è Q8_0 —
//      la ricetta "q4_0" di llama.cpp tiene l'output a precisione più alta;
//   2. attn q/k/v hanno BIAS F32 (architettura qwen2), [896]/[128]/[128].
import { GGML_TYPE, type GgufFile, type GgufTensorInfo } from "./gguf";

export interface ModelShape {
  arch: "qwen2";
  nLayer: number;
  dModel: number;
  nHead: number;
  nKvHead: number;
  headDim: number;
  dFfn: number;
  vocab: number;
  ctxTrain: number;
  ropeFreqBase: number;
  rmsEps: number;
}

export const QWEN25_05B: ModelShape = {
  arch: "qwen2",
  nLayer: 24,
  dModel: 896,
  nHead: 14,
  nKvHead: 2,
  headDim: 64, // 896/14
  dFfn: 4864,
  vocab: 151936,
  ctxTrain: 32768,
  ropeFreqBase: 1e6,
  rmsEps: 1e-6,
};

// Tensori attesi: nome (N = indice layer), dims in ordine GGUF (ne[0] = interna), tipo.
type Expect = { dims: number[]; type: number };
const S = QWEN25_05B;
const PER_LAYER: Record<string, Expect> = {
  "attn_norm.weight": { dims: [S.dModel], type: GGML_TYPE.F32 },
  "attn_q.weight": { dims: [S.dModel, S.dModel], type: GGML_TYPE.Q4_0 },
  "attn_q.bias": { dims: [S.dModel], type: GGML_TYPE.F32 },
  "attn_k.weight": { dims: [S.dModel, S.nKvHead * S.headDim], type: GGML_TYPE.Q4_0 },
  "attn_k.bias": { dims: [S.nKvHead * S.headDim], type: GGML_TYPE.F32 },
  "attn_v.weight": { dims: [S.dModel, S.nKvHead * S.headDim], type: GGML_TYPE.Q4_0 },
  "attn_v.bias": { dims: [S.nKvHead * S.headDim], type: GGML_TYPE.F32 },
  "attn_output.weight": { dims: [S.dModel, S.dModel], type: GGML_TYPE.Q4_0 },
  "ffn_norm.weight": { dims: [S.dModel], type: GGML_TYPE.F32 },
  "ffn_gate.weight": { dims: [S.dModel, S.dFfn], type: GGML_TYPE.Q4_0 },
  "ffn_up.weight": { dims: [S.dModel, S.dFfn], type: GGML_TYPE.Q4_0 },
  "ffn_down.weight": { dims: [S.dFfn, S.dModel], type: GGML_TYPE.Q4_0 },
};
const TOP_LEVEL: Record<string, Expect> = {
  "token_embd.weight": { dims: [S.dModel, S.vocab], type: GGML_TYPE.Q4_0 },
  "output.weight": { dims: [S.dModel, S.vocab], type: GGML_TYPE.Q8_0 },
  "output_norm.weight": { dims: [S.dModel], type: GGML_TYPE.F32 },
};

function expectTensor(byName: Map<string, GgufTensorInfo>, name: string, e: Expect): void {
  const t = byName.get(name);
  if (!t) throw new Error(`shape: tensore mancante ${name}`);
  if (t.type !== e.type) throw new Error(`shape: ${name} tipo ${t.type}, atteso ${e.type}`);
  if (t.dims.length !== e.dims.length || t.dims.some((d, i) => d !== e.dims[i])) {
    throw new Error(`shape: ${name} dims [${t.dims}], attese [${e.dims}]`);
  }
}

function expectMeta(f: GgufFile, key: string, want: unknown): void {
  const got = f.metadata[key];
  const ok = typeof want === "number" && typeof got === "number"
    ? Math.abs(got - want) <= Math.abs(want) * 1e-6
    : got === want;
  if (!ok) throw new Error(`shape: metadata ${key} = ${String(got)}, atteso ${String(want)}`);
}

// Valida il GGUF contro la shape. Ritorna la mappa nome→info per il loader.
export function validateQwen25_05B(f: GgufFile): Map<string, GgufTensorInfo> {
  expectMeta(f, "general.architecture", "qwen2");
  expectMeta(f, "qwen2.block_count", S.nLayer);
  expectMeta(f, "qwen2.embedding_length", S.dModel);
  expectMeta(f, "qwen2.feed_forward_length", S.dFfn);
  expectMeta(f, "qwen2.attention.head_count", S.nHead);
  expectMeta(f, "qwen2.attention.head_count_kv", S.nKvHead);
  expectMeta(f, "qwen2.rope.freq_base", S.ropeFreqBase);

  const byName = new Map(f.tensors.map((t) => [t.name, t]));
  for (const [name, e] of Object.entries(TOP_LEVEL)) expectTensor(byName, name, e);
  for (let l = 0; l < S.nLayer; l++) {
    for (const [suffix, e] of Object.entries(PER_LAYER)) {
      expectTensor(byName, `blk.${l}.${suffix}`, e);
    }
  }
  const expected = S.nLayer * Object.keys(PER_LAYER).length + Object.keys(TOP_LEVEL).length;
  if (f.tensors.length !== expected) {
    throw new Error(`shape: ${f.tensors.length} tensori nel file, attesi ${expected}`);
  }
  return byName;
}
