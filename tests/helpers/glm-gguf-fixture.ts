// Fixture GGUF sintetica GLM-4.7-Flash (inventario 844 tensori, dati vuoti) —
// estratta da engine-gguf-glm.test.ts nella fase 4d perche' la usano anche i
// test di glmsource. Stesso formato del builder di engine-gguf.test.ts.
import { GGML_TYPE, GGUF_KV_TYPE } from "../../src/engine/gguf";
import { GLM47_FLASH, GLM47_DOWN_EXPS_Q4_1_LAST } from "../../src/engine/shape";

const G = GLM47_FLASH;

export type KvVal =
  | { t: "u32"; v: number } | { t: "f32"; v: number } | { t: "str"; v: string }
  | { t: "bool"; v: boolean };
export type TensorDecl = { name: string; dims: number[]; type: number; offset: number };

export function buildGguf(kv: Record<string, KvVal>, tensors: TensorDecl[]): ArrayBuffer {
  const bytes: number[] = [];
  const enc = new TextEncoder();
  const pushU32 = (x: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, x, true); bytes.push(...b); };
  const pushU64 = (x: number) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(x), true); bytes.push(...b); };
  const pushF32 = (x: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, x, true); bytes.push(...b); };
  const pushStr = (s: string) => { const u = enc.encode(s); pushU64(u.length); bytes.push(...u); };
  pushU32(0x46554747);
  pushU32(3);
  pushU64(tensors.length);
  pushU64(Object.keys(kv).length);
  for (const [key, val] of Object.entries(kv)) {
    pushStr(key);
    if (val.t === "u32") { pushU32(GGUF_KV_TYPE.UINT32); pushU32(val.v); }
    else if (val.t === "f32") { pushU32(GGUF_KV_TYPE.FLOAT32); pushF32(val.v); }
    else if (val.t === "str") { pushU32(GGUF_KV_TYPE.STRING); pushStr(val.v); }
    else { pushU32(GGUF_KV_TYPE.BOOL); bytes.push(val.v ? 1 : 0); }
  }
  for (const t of tensors) {
    pushStr(t.name);
    pushU32(t.dims.length);
    for (const d of t.dims) pushU64(d);
    pushU32(t.type);
    pushU64(t.offset);
  }
  return new Uint8Array(bytes).buffer;
}

export function glmMeta(): Record<string, KvVal> {
  return {
    "general.architecture": { t: "str", v: "deepseek2" },
    "deepseek2.block_count": { t: "u32", v: G.nLayer },
    "deepseek2.leading_dense_block_count": { t: "u32", v: G.denseLead },
    "deepseek2.embedding_length": { t: "u32", v: G.dModel },
    "deepseek2.attention.head_count": { t: "u32", v: G.nHead },
    "deepseek2.attention.q_lora_rank": { t: "u32", v: G.qLora },
    "deepseek2.attention.kv_lora_rank": { t: "u32", v: G.kvLora },
    "deepseek2.attention.key_length": { t: "u32", v: G.keyLen },
    "deepseek2.attention.value_length": { t: "u32", v: G.valLen },
    "deepseek2.attention.key_length_mla": { t: "u32", v: G.headLenMla },
    "deepseek2.attention.value_length_mla": { t: "u32", v: G.headLenMla },
    "deepseek2.rope.dimension_count": { t: "u32", v: G.ropeDims },
    "deepseek2.rope.freq_base": { t: "f32", v: G.ropeFreqBase },
    "deepseek2.feed_forward_length": { t: "u32", v: G.dFfnDense },
    "deepseek2.expert_count": { t: "u32", v: G.nExpert },
    "deepseek2.expert_used_count": { t: "u32", v: G.nExpertUsed },
    "deepseek2.expert_shared_count": { t: "u32", v: G.nShared },
    "deepseek2.expert_feed_forward_length": { t: "u32", v: G.dFfnExpert },
    "deepseek2.expert_gating_func": { t: "u32", v: G.gatingFunc },
    "deepseek2.expert_weights_norm": { t: "bool", v: G.weightsNorm },
    "deepseek2.expert_weights_scale": { t: "f32", v: G.weightsScale },
    "deepseek2.vocab_size": { t: "u32", v: G.vocab },
    "deepseek2.context_length": { t: "u32", v: G.ctxTrain },
    "deepseek2.attention.layer_norm_rms_epsilon": { t: "f32", v: G.rmsEps },
  };
}

export function glmTensors(): TensorDecl[] {
  const ts: TensorDecl[] = [
    { name: "token_embd.weight", dims: [G.dModel, G.vocab], type: GGML_TYPE.Q4_0, offset: 0 },
    { name: "output.weight", dims: [G.dModel, G.vocab], type: GGML_TYPE.Q6_K, offset: 0 },
    { name: "output_norm.weight", dims: [G.dModel], type: GGML_TYPE.F32, offset: 0 },
  ];
  for (let l = 0; l < G.nLayer; l++) {
    ts.push(
      { name: `blk.${l}.attn_norm.weight`, dims: [G.dModel], type: GGML_TYPE.F32, offset: 0 },
      { name: `blk.${l}.attn_q_a.weight`, dims: [G.dModel, G.qLora], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.attn_q_a_norm.weight`, dims: [G.qLora], type: GGML_TYPE.F32, offset: 0 },
      { name: `blk.${l}.attn_q_b.weight`, dims: [G.qLora, G.nHead * (G.qkNope + G.ropeDims)], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.attn_kv_a_mqa.weight`, dims: [G.dModel, G.keyLen], type: GGML_TYPE.Q8_0, offset: 0 },
      { name: `blk.${l}.attn_kv_a_norm.weight`, dims: [G.kvLora], type: GGML_TYPE.F32, offset: 0 },
      { name: `blk.${l}.attn_k_b.weight`, dims: [G.qkNope, G.kvLora, G.nHead], type: GGML_TYPE.Q8_0, offset: 0 },
      { name: `blk.${l}.attn_v_b.weight`, dims: [G.kvLora, G.headLenMla, G.nHead], type: GGML_TYPE.Q8_0, offset: 0 },
      { name: `blk.${l}.attn_output.weight`, dims: [G.nHead * G.headLenMla, G.dModel], type: GGML_TYPE.Q4_0, offset: 0 },
      { name: `blk.${l}.ffn_norm.weight`, dims: [G.dModel], type: GGML_TYPE.F32, offset: 0 },
    );
    if (l < G.denseLead) {
      ts.push(
        { name: `blk.${l}.ffn_gate.weight`, dims: [G.dModel, G.dFfnDense], type: GGML_TYPE.Q4_0, offset: 0 },
        { name: `blk.${l}.ffn_up.weight`, dims: [G.dModel, G.dFfnDense], type: GGML_TYPE.Q4_0, offset: 0 },
        { name: `blk.${l}.ffn_down.weight`, dims: [G.dFfnDense, G.dModel], type: GGML_TYPE.Q4_1, offset: 0 },
      );
    } else {
      ts.push(
        { name: `blk.${l}.ffn_gate_inp.weight`, dims: [G.dModel, G.nExpert], type: GGML_TYPE.F32, offset: 0 },
        { name: `blk.${l}.exp_probs_b.bias`, dims: [G.nExpert], type: GGML_TYPE.F32, offset: 0 },
        { name: `blk.${l}.ffn_gate_exps.weight`, dims: [G.dModel, G.dFfnExpert, G.nExpert], type: GGML_TYPE.Q4_0, offset: 0 },
        { name: `blk.${l}.ffn_up_exps.weight`, dims: [G.dModel, G.dFfnExpert, G.nExpert], type: GGML_TYPE.Q4_0, offset: 0 },
        { name: `blk.${l}.ffn_down_exps.weight`, dims: [G.dFfnExpert, G.dModel, G.nExpert],
          type: l <= GLM47_DOWN_EXPS_Q4_1_LAST ? GGML_TYPE.Q4_1 : GGML_TYPE.Q4_0, offset: 0 },
        { name: `blk.${l}.ffn_gate_shexp.weight`, dims: [G.dModel, G.dFfnExpert], type: GGML_TYPE.Q5_K, offset: 0 },
        { name: `blk.${l}.ffn_up_shexp.weight`, dims: [G.dModel, G.dFfnExpert], type: GGML_TYPE.Q5_K, offset: 0 },
        { name: `blk.${l}.ffn_down_shexp.weight`, dims: [G.dFfnExpert, G.dModel], type: GGML_TYPE.Q6_K, offset: 0 },
      );
    }
  }
  return ts;
}

export function glmFixture(
  mutateTensors?: (ts: TensorDecl[]) => void,
  mutateMeta?: (kv: Record<string, KvVal>) => void,
): ArrayBuffer {
  const kv = glmMeta();
  const ts = glmTensors();
  mutateTensors?.(ts);
  mutateMeta?.(kv);
  return buildGguf(kv, ts);
}

