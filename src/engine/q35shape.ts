// Shape della famiglia Qwen 3.5/3.6 (arch GGUF `qwen35` denso / `qwen35moe`)
// + validazione hard (postura ds4: ogni divergenza è throw, mai best effort).
//
// A differenza di shape.ts (GLM/Qwen2.5: shape hardcoded del singolo file),
// qui la shape si DERIVA dai metadata e i tensori si validano contro le dims
// calcolate: è la parametrizzazione che il goal q1 costruisce (spec §3). I
// TIPI però non si assumono: ogni classe di tensore ha un allow-list chiuso
// ricavato dall'inventario reale dei 3 file pinnati (header dump 2026-08-10,
// results/engine/q35-header-dump-2026-08-10.json) e tutto il resto è throw —
// mai fallback silenzioso (lezione LlamaWeb, cliff supports_op).
//
// Struttura verificata sui file (q1 it.3):
// - layer ibridi 3:1: full attention quando l % interval === interval-1
//   (config HF layer_types: full a 3,7,11,… 0-based), linear (Gated DeltaNet)
//   altrove;
// - layer FULL: attn_q [d, 2·nHead·headDim] (gate fuso: attn_output_gate),
//   attn_k/v [d, nKvHead·headDim], QK-norm per-head [headDim], attn_output
//   [nHead·headDim, d];
// - layer LINEAR: attn_qkv fusa [d, (2·nK+nV)·hd], conv1d [convK, (2·nK+nV)·hd],
//   attn_gate [d, nV·hd], alpha/beta [d, nV], a/dt [nV], ssm_norm [hd],
//   ssm_out [nV·hd, d];
// - ffn denso su OGNI layer dei densi; MoE (routed+shared) su OGNI layer del
//   35B; embeddings tied solo sul 4B (output.weight assente).
import { GGML_TYPE, type GgufFile, type GgufTensorInfo } from "./gguf";

export interface Q35Shape {
  arch: "qwen35" | "qwen35moe";
  name: string;
  nLayer: number;
  fullInterval: number; // 4 ⇒ full quando l % 4 === 3
  nFull: number;
  dModel: number;
  nHead: number;
  nKvHead: number;
  headDim: number; // key_length = value_length = 256
  ropeDims: number; // 64 = headDim × partial_rotary 0.25
  ropeFreqBase: number;
  rmsEps: number;
  vocab: number;
  ctxTrain: number;
  // Gated DeltaNet (metadata ssm.*)
  linConvK: number; // 4
  linHeadDim: number; // state_size 128
  linKHead: number; // group_count 16
  linVHead: number; // time_step_rank 32
  linInner: number; // nV·hd 4096
  // denso
  dFfn: number | null;
  // MoE
  nExpert: number | null;
  nExpertUsed: number | null;
  dFfnExpert: number | null;
  tiedEmbeddings: boolean;
  /**
   * Teste MTP (NextN) presenti nel file: `qwen35.nextn_predict_layers`, 0 se
   * assente. I GGUF base non ce l'hanno; i repo `*-MTP-GGUF` sì, e la testa sta
   * nei blocchi `blk.<nLayer + i>` (sul 4B: blk.32, con blk.0..31 il modello).
   */
  mtpLayers: number;
}

const num = (f: GgufFile, k: string): number => {
  const v = f.metadata[k];
  if (typeof v !== "number") throw new Error(`q35shape: metadata ${k} mancante o non numerico`);
  return v;
};

/** Deriva la shape dai metadata; throw su architettura non-famiglia. */
export function q35ShapeFromGguf(f: GgufFile): Q35Shape {
  const arch = f.metadata["general.architecture"];
  if (arch !== "qwen35" && arch !== "qwen35moe") {
    throw new Error(`q35shape: architettura ${String(arch)} non è qwen35/qwen35moe`);
  }
  const p = arch;
  // `block_count` CONTA la testa MTP quando c'e' (4B-MTP: 33 = 32 layer + 1
  // testa), quindi i layer del MODELLO sono block_count - nextn_predict_layers.
  // Trovato dal test sul file vero, non dedotto: avevo assunto 32 leggendo i
  // nomi dei tensori, e il file dice 33.
  const mtpLayers = typeof f.metadata[`${p}.nextn_predict_layers`] === "number"
    ? (f.metadata[`${p}.nextn_predict_layers`] as number) : 0;
  const nLayer = num(f, `${p}.block_count`) - mtpLayers;
  const fullInterval = num(f, `${p}.full_attention_interval`);
  if (nLayer % fullInterval !== 0) throw new Error(`q35shape: layer del modello ${nLayer} (block_count ${num(f, `${p}.block_count`)} - ${mtpLayers} MTP) non multiplo di interval ${fullInterval}`);
  const headDim = num(f, `${p}.attention.key_length`);
  if (num(f, `${p}.attention.value_length`) !== headDim) throw new Error("q35shape: key_length ≠ value_length");
  const linKHead = num(f, `${p}.ssm.group_count`);
  const linVHead = num(f, `${p}.ssm.time_step_rank`);
  const linHeadDim = num(f, `${p}.ssm.state_size`);
  const linInner = num(f, `${p}.ssm.inner_size`);
  if (linInner !== linVHead * linHeadDim) throw new Error(`q35shape: inner_size ${linInner} ≠ nV·hd ${linVHead * linHeadDim}`);
  const tokens = f.metadata["tokenizer.ggml.tokens"];
  if (!Array.isArray(tokens)) throw new Error("q35shape: tokenizer.ggml.tokens mancante");
  const moe = p === "qwen35moe";
  return {
    arch: p,
    name: String(f.metadata["general.name"] ?? p),
    nLayer,
    fullInterval,
    nFull: nLayer / fullInterval,
    dModel: num(f, `${p}.embedding_length`),
    nHead: num(f, `${p}.attention.head_count`),
    nKvHead: num(f, `${p}.attention.head_count_kv`),
    headDim,
    ropeDims: num(f, `${p}.rope.dimension_count`),
    ropeFreqBase: num(f, `${p}.rope.freq_base`),
    rmsEps: num(f, `${p}.attention.layer_norm_rms_epsilon`),
    vocab: tokens.length,
    ctxTrain: num(f, `${p}.context_length`),
    linConvK: num(f, `${p}.ssm.conv_kernel`),
    linHeadDim,
    linKHead,
    linVHead,
    linInner,
    dFfn: moe ? null : num(f, `${p}.feed_forward_length`),
    nExpert: moe ? num(f, `${p}.expert_count`) : null,
    nExpertUsed: moe ? num(f, `${p}.expert_used_count`) : null,
    dFfnExpert: moe ? num(f, `${p}.expert_feed_forward_length`) : null,
    tiedEmbeddings: !f.tensors.some((t) => t.name === "output.weight"),
    mtpLayers,
  };
}

// Allow-list dei tipi per classe (inventario reale 2026-08-10; fuori lista = throw).
const T = GGML_TYPE;
const W_QUANT = [T.Q4_0, T.Q4_1, T.Q4_K, T.Q5_K, T.Q6_K, T.Q8_0]; // pesi matmul
const W_SMALL = [T.F32, T.Q8_0]; // alpha/beta (Q8_0 su 4B/9B, F32 su 35B)
const F32_ONLY = [T.F32];

type Expect = { dims: number[]; types: readonly number[] };

export function q35IsFullAttn(S: Q35Shape, l: number): boolean {
  return l % S.fullInterval === S.fullInterval - 1;
}

/** Valida tutti i tensori contro la shape derivata. Ritorna nome→info. */
export function validateQwen35(f: GgufFile): { shape: Q35Shape; byName: Map<string, GgufTensorInfo> } {
  const S = q35ShapeFromGguf(f);
  const byName = new Map(f.tensors.map((t) => [t.name, t]));
  const seen = new Set<string>();
  const expect = (name: string, e: Expect): void => {
    const t = byName.get(name);
    if (!t) throw new Error(`q35shape: tensore mancante ${name}`);
    seen.add(name);
    if (!e.types.includes(t.type)) throw new Error(`q35shape: ${name} tipo ${t.type} fuori allow-list [${e.types}]`);
    if (t.dims.length !== e.dims.length || t.dims.some((d, i) => d !== e.dims[i])) {
      throw new Error(`q35shape: ${name} dims [${t.dims}], attese [${e.dims}]`);
    }
  };

  const d = S.dModel;
  const qkvDim = (2 * S.linKHead + S.linVHead) * S.linHeadDim;
  expect("token_embd.weight", { dims: [d, S.vocab], types: W_QUANT });
  expect("output_norm.weight", { dims: [d], types: F32_ONLY });
  if (!S.tiedEmbeddings) expect("output.weight", { dims: [d, S.vocab], types: W_QUANT });

  // Un blocco = attn (full o deltanet) + ffn (denso o MoE). Estratto in una
  // funzione perche' lo usano DUE chiamanti: i layer del modello e la testa MTP,
  // che e' un blocco identico. Duplicarlo sarebbe una seconda verita'.
  const expectBlock = (b: string, full: boolean): void => {
    expect(`${b}attn_norm.weight`, { dims: [d], types: F32_ONLY });
    expect(`${b}post_attention_norm.weight`, { dims: [d], types: F32_ONLY });
    if (full) {
      expect(`${b}attn_q.weight`, { dims: [d, 2 * S.nHead * S.headDim], types: W_QUANT }); // q + output gate fusi
      expect(`${b}attn_k.weight`, { dims: [d, S.nKvHead * S.headDim], types: W_QUANT });
      expect(`${b}attn_v.weight`, { dims: [d, S.nKvHead * S.headDim], types: W_QUANT });
      expect(`${b}attn_q_norm.weight`, { dims: [S.headDim], types: F32_ONLY });
      expect(`${b}attn_k_norm.weight`, { dims: [S.headDim], types: F32_ONLY });
      expect(`${b}attn_output.weight`, { dims: [S.nHead * S.headDim, d], types: W_QUANT });
    } else {
      expect(`${b}attn_qkv.weight`, { dims: [d, qkvDim], types: W_QUANT });
      expect(`${b}attn_gate.weight`, { dims: [d, S.linInner], types: W_QUANT });
      expect(`${b}ssm_conv1d.weight`, { dims: [S.linConvK, qkvDim], types: F32_ONLY });
      expect(`${b}ssm_alpha.weight`, { dims: [d, S.linVHead], types: W_SMALL });
      expect(`${b}ssm_beta.weight`, { dims: [d, S.linVHead], types: W_SMALL });
      expect(`${b}ssm_a`, { dims: [S.linVHead], types: F32_ONLY });
      expect(`${b}ssm_dt.bias`, { dims: [S.linVHead], types: F32_ONLY });
      expect(`${b}ssm_norm.weight`, { dims: [S.linHeadDim], types: F32_ONLY });
      expect(`${b}ssm_out.weight`, { dims: [S.linInner, d], types: W_QUANT });
    }
    if (S.arch === "qwen35") {
      const dFfn = S.dFfn as number;
      expect(`${b}ffn_gate.weight`, { dims: [d, dFfn], types: W_QUANT });
      expect(`${b}ffn_up.weight`, { dims: [d, dFfn], types: W_QUANT });
      expect(`${b}ffn_down.weight`, { dims: [dFfn, d], types: W_QUANT });
    } else {
      const dE = S.dFfnExpert as number;
      const nE = S.nExpert as number;
      expect(`${b}ffn_gate_inp.weight`, { dims: [d, nE], types: F32_ONLY });
      expect(`${b}ffn_gate_exps.weight`, { dims: [d, dE, nE], types: W_QUANT });
      expect(`${b}ffn_up_exps.weight`, { dims: [d, dE, nE], types: W_QUANT });
      expect(`${b}ffn_down_exps.weight`, { dims: [dE, d, nE], types: W_QUANT });
      expect(`${b}ffn_gate_inp_shexp.weight`, { dims: [d], types: F32_ONLY });
      expect(`${b}ffn_gate_shexp.weight`, { dims: [d, dE], types: W_QUANT });
      expect(`${b}ffn_up_shexp.weight`, { dims: [d, dE], types: W_QUANT });
      expect(`${b}ffn_down_shexp.weight`, { dims: [dE, d], types: W_QUANT });
    }
  };

  for (let l = 0; l < S.nLayer; l++) expectBlock(`blk.${l}.`, q35IsFullAttn(S, l));

  // TESTA MTP (fase D fase 7). Il blocco e' identico a un layer del modello, ma
  // e' FULL-attention SEMPRE e non per la regola dell'interval: sul 4B sta a
  // blk.32 e 32 % 4 = 0, che `q35IsFullAttn` direbbe deltanet — mentre il GGUF
  // pinnato porta attn_q/k/v. Verificato sull'header di
  // unsloth/Qwen3.5-4B-MTP-GGUF (it.47), non dedotto.
  // I 4 tensori suoi: eh_proj proietta [emb(token); hidden] concatenati (2*d) su
  // d, enorm/hnorm normalizzano i due ingressi separatamente, e
  // shared_head_norm precede la lm_head CONDIVISA col modello.
  for (let i = 0; i < S.mtpLayers; i++) {
    const b = `blk.${S.nLayer + i}.`;
    expectBlock(b, true);
    expect(`${b}nextn.eh_proj.weight`, { dims: [2 * d, d], types: W_QUANT });
    expect(`${b}nextn.enorm.weight`, { dims: [d], types: F32_ONLY });
    expect(`${b}nextn.hnorm.weight`, { dims: [d], types: F32_ONLY });
    expect(`${b}nextn.shared_head_norm.weight`, { dims: [d], types: F32_ONLY });
  }

  if (seen.size !== f.tensors.length) {
    const extra = f.tensors.filter((t) => !seen.has(t.name)).map((t) => t.name);
    throw new Error(`q35shape: ${extra.length} tensori inattesi nel file: ${extra.slice(0, 5).join(", ")}…`);
  }
  return { shape: S, byName };
}

// SHA-256 dei GGUF canonici q1 (pinnati in spec §1 / scripts/q35-manifest.json).
export const Q35_SHA256 = {
  "Qwen3.5-4B": "298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb",
  "Qwen3.5-9B": "17670346b4260ddcb0173965145155885024f3c9a4a24389a3370751edbcde24",
  "Qwen3.6-35B-A3B": "a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c",
} as const;
