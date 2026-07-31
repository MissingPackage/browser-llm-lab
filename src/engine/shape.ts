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

// --- GLM-4.7-Flash (GGUF unsloth Q4_0, arch deepseek2) — spec C2 §2 ---
//
// Inventario VERIFICATO sul file reale (dump header 2026-07-31, 844 tensori,
// ri-verificato per-layer dopo il FAIL del verifier it.1): quant MISTA — vedi
// tabella spec §1. blk.0 è DENSO (leading_dense=1, ffn 10240); blk.1-46 sono
// MoE (64 routed top-4 + 1 shared); ffn_down_exps è Q4_1 SOLO su blk.1-4.

export interface Glm47Shape {
  arch: "deepseek2";
  nLayer: number;      // 47, primo denso
  denseLead: number;   // 1
  dModel: number;      // 2048
  nHead: number;       // 20 (MLA: 1 kv head compresso)
  qLora: number;       // 768
  kvLora: number;      // 512
  ropeDims: number;    // 64
  keyLen: number;      // 576 = kvLora + ropeDims (cache absorbed per token)
  valLen: number;      // 512
  headLenMla: number;  // 256 (k e v per head nel path absorbed)
  qkNope: number;      // 192: dim nope di q/k per head (head q = 192 nope + 64 rope)
  dFfnDense: number;   // 10240 (solo blk.0)
  nExpert: number;     // 64
  nExpertUsed: number; // 4
  nShared: number;     // 1
  dFfnExpert: number;  // 1536
  gatingFunc: number;  // 2 = sigmoid
  weightsNorm: boolean;// true: pesi mixing normalizzati a somma 1
  weightsScale: number;// 1.8
  vocab: number;       // 154880
  ctxTrain: number;    // 202752
  ropeFreqBase: number;// 1e6
  rmsEps: number;      // 1e-5
}

export const GLM47_FLASH: Glm47Shape = {
  arch: "deepseek2",
  nLayer: 47,
  denseLead: 1,
  dModel: 2048,
  nHead: 20,
  qLora: 768,
  kvLora: 512,
  ropeDims: 64,
  keyLen: 576,
  valLen: 512,
  headLenMla: 256,
  qkNope: 192,
  dFfnDense: 10240,
  nExpert: 64,
  nExpertUsed: 4,
  nShared: 1,
  dFfnExpert: 1536,
  gatingFunc: 2,
  weightsNorm: true,
  weightsScale: 1.8,
  vocab: 154880,
  ctxTrain: 202752,
  ropeFreqBase: 1e6,
  rmsEps: 1e-5,
};

// SHA-256 del GGUF canonico (pinnato dal goal C1; ogni report lo registra).
export const GLM47_FLASH_SHA256 =
  "d0bbdfcde6e323ebf90a8b9e95da57100e972be1ec6f0bfa0fad0feaa426557e";

// Ultimo layer MoE (incluso) col down_exps in Q4_1 (blk.1-4); da blk.5 è Q4_0.
export const GLM47_DOWN_EXPS_Q4_1_LAST = 4;

const G = GLM47_FLASH;
// Comuni a TUTTI i 47 layer (attention MLA + norm ffn).
const GLM_PER_LAYER: Record<string, Expect> = {
  "attn_norm.weight": { dims: [G.dModel], type: GGML_TYPE.F32 },
  "attn_q_a.weight": { dims: [G.dModel, G.qLora], type: GGML_TYPE.Q4_0 },
  "attn_q_a_norm.weight": { dims: [G.qLora], type: GGML_TYPE.F32 },
  "attn_q_b.weight": { dims: [G.qLora, G.nHead * (G.qkNope + G.ropeDims)], type: GGML_TYPE.Q4_0 },
  "attn_kv_a_mqa.weight": { dims: [G.dModel, G.keyLen], type: GGML_TYPE.Q8_0 },
  "attn_kv_a_norm.weight": { dims: [G.kvLora], type: GGML_TYPE.F32 },
  "attn_k_b.weight": { dims: [G.qkNope, G.kvLora, G.nHead], type: GGML_TYPE.Q8_0 },
  "attn_v_b.weight": { dims: [G.kvLora, G.headLenMla, G.nHead], type: GGML_TYPE.Q8_0 },
  "attn_output.weight": { dims: [G.nHead * G.headLenMla, G.dModel], type: GGML_TYPE.Q4_0 },
  "ffn_norm.weight": { dims: [G.dModel], type: GGML_TYPE.F32 },
};
// Solo blk.0 (denso).
const GLM_DENSE_LAYER: Record<string, Expect> = {
  "ffn_gate.weight": { dims: [G.dModel, G.dFfnDense], type: GGML_TYPE.Q4_0 },
  "ffn_up.weight": { dims: [G.dModel, G.dFfnDense], type: GGML_TYPE.Q4_0 },
  "ffn_down.weight": { dims: [G.dFfnDense, G.dModel], type: GGML_TYPE.Q4_1 },
};
// Solo blk.1-46 (MoE); down_exps ha tipo per-layer (vedi validate).
const GLM_MOE_LAYER: Record<string, Expect> = {
  "ffn_gate_inp.weight": { dims: [G.dModel, G.nExpert], type: GGML_TYPE.F32 },
  "exp_probs_b.bias": { dims: [G.nExpert], type: GGML_TYPE.F32 },
  "ffn_gate_exps.weight": { dims: [G.dModel, G.dFfnExpert, G.nExpert], type: GGML_TYPE.Q4_0 },
  "ffn_up_exps.weight": { dims: [G.dModel, G.dFfnExpert, G.nExpert], type: GGML_TYPE.Q4_0 },
  "ffn_gate_shexp.weight": { dims: [G.dModel, G.dFfnExpert], type: GGML_TYPE.Q5_K },
  "ffn_up_shexp.weight": { dims: [G.dModel, G.dFfnExpert], type: GGML_TYPE.Q5_K },
  "ffn_down_shexp.weight": { dims: [G.dFfnExpert, G.dModel], type: GGML_TYPE.Q6_K },
};
const GLM_TOP_LEVEL: Record<string, Expect> = {
  "token_embd.weight": { dims: [G.dModel, G.vocab], type: GGML_TYPE.Q4_0 },
  "output.weight": { dims: [G.dModel, G.vocab], type: GGML_TYPE.Q6_K },
  "output_norm.weight": { dims: [G.dModel], type: GGML_TYPE.F32 },
};

// Valida il GGUF GLM-4.7-Flash contro la shape. Ritorna la mappa nome→info.
export function validateGlm47Flash(f: GgufFile): Map<string, GgufTensorInfo> {
  expectMeta(f, "general.architecture", "deepseek2");
  expectMeta(f, "deepseek2.block_count", G.nLayer);
  expectMeta(f, "deepseek2.leading_dense_block_count", G.denseLead);
  expectMeta(f, "deepseek2.embedding_length", G.dModel);
  expectMeta(f, "deepseek2.attention.head_count", G.nHead);
  expectMeta(f, "deepseek2.attention.q_lora_rank", G.qLora);
  expectMeta(f, "deepseek2.attention.kv_lora_rank", G.kvLora);
  expectMeta(f, "deepseek2.attention.key_length", G.keyLen);
  expectMeta(f, "deepseek2.attention.value_length", G.valLen);
  expectMeta(f, "deepseek2.attention.key_length_mla", G.headLenMla);
  expectMeta(f, "deepseek2.attention.value_length_mla", G.headLenMla);
  expectMeta(f, "deepseek2.rope.dimension_count", G.ropeDims);
  expectMeta(f, "deepseek2.rope.freq_base", G.ropeFreqBase);
  expectMeta(f, "deepseek2.feed_forward_length", G.dFfnDense);
  expectMeta(f, "deepseek2.expert_count", G.nExpert);
  expectMeta(f, "deepseek2.expert_used_count", G.nExpertUsed);
  expectMeta(f, "deepseek2.expert_shared_count", G.nShared);
  expectMeta(f, "deepseek2.expert_feed_forward_length", G.dFfnExpert);
  expectMeta(f, "deepseek2.expert_gating_func", G.gatingFunc);
  expectMeta(f, "deepseek2.expert_weights_norm", G.weightsNorm);
  expectMeta(f, "deepseek2.expert_weights_scale", G.weightsScale);
  expectMeta(f, "deepseek2.vocab_size", G.vocab);
  expectMeta(f, "deepseek2.context_length", G.ctxTrain);
  expectMeta(f, "deepseek2.attention.layer_norm_rms_epsilon", G.rmsEps);

  const byName = new Map(f.tensors.map((t) => [t.name, t]));
  for (const [name, e] of Object.entries(GLM_TOP_LEVEL)) expectTensor(byName, name, e);
  for (let l = 0; l < G.nLayer; l++) {
    for (const [suffix, e] of Object.entries(GLM_PER_LAYER)) {
      expectTensor(byName, `blk.${l}.${suffix}`, e);
    }
    if (l < G.denseLead) {
      for (const [suffix, e] of Object.entries(GLM_DENSE_LAYER)) {
        expectTensor(byName, `blk.${l}.${suffix}`, e);
      }
    } else {
      for (const [suffix, e] of Object.entries(GLM_MOE_LAYER)) {
        expectTensor(byName, `blk.${l}.${suffix}`, e);
      }
      expectTensor(byName, `blk.${l}.ffn_down_exps.weight`, {
        dims: [G.dFfnExpert, G.dModel, G.nExpert],
        type: l <= GLM47_DOWN_EXPS_Q4_1_LAST ? GGML_TYPE.Q4_1 : GGML_TYPE.Q4_0,
      });
    }
  }
  const expected =
    G.nLayer * Object.keys(GLM_PER_LAYER).length +
    G.denseLead * Object.keys(GLM_DENSE_LAYER).length +
    (G.nLayer - G.denseLead) * (Object.keys(GLM_MOE_LAYER).length + 1) +
    Object.keys(GLM_TOP_LEVEL).length;
  if (f.tensors.length !== expected) {
    throw new Error(`shape: ${f.tensors.length} tensori nel file, attesi ${expected}`);
  }
  return byName;
}

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
