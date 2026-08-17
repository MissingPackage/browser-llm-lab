// BF16 (ggml 30) — il tipo che bloccava il caricamento del `bartowski Q2_K`.
//
// PERCHE' ESISTE QUESTO FILE: il GGUF bersaglio tiene i router MoE in BF16 e
// `validateQwen35` lanciava sull'allow-list. BF16 non e' un quant: e' la META'
// ALTA di un float32, quindi 2 byte per elemento SENZA blocchi e una
// conversione che e' uno SHIFT, non una ricomposizione di esponente e mantissa
// (a differenza di f16, dove l'esponente ha 5 bit e va riscalato).
//
// L'ARTEFATTO, NON L'ASSUNZIONE: i tensori BF16 del file sono DUE, non uno —
// `ffn_gate_inp` (1 048 576 B = 2048x256x2) e `ffn_gate_inp_shexp` (4 096 B =
// 2048x2), entrambi sul blocco MTP. Sta scritto nel dump committato
// (results/eval/q35-header-dump-bartowski-2026-08-17.json, typeHistogram
// `other:BF16 *` + `ffn/shexp:BF16 *`), e il primo giro di questo file ne aveva
// visto solo uno: da qui la fixture con la testa MTP e il caso esaustivo che
// prova BF16 su OGNI tensore invece che sui due che mi aspettavo.
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { GGML_TYPE, tensorByteSize, type GgufFile, type GgufTensorInfo } from "../src/engine/gguf";
import { bf16ToF32 } from "../src/engine/quant";
import { validateQwen35 } from "../src/engine/q35shape";

describe("BF16: il tipo nel sistema dei tipi", () => {
  it("GGML_TYPE.BF16 e' 30 (l'id ggml, non una scelta nostra)", () => {
    expect(GGML_TYPE.BF16).toBe(30);
  });

  it("tensorByteSize BF16 = 2 B/elemento, anche con ne[0] NON multiplo di 256", () => {
    // Dims apposta "scomode": BF16 non ha superblocchi, quindi il controllo di
    // multiplo che vale per i K-quant qui sarebbe un falso errore.
    const t: GgufTensorInfo = { name: "test.bf16", dims: [100, 7], type: GGML_TYPE.BF16, offset: 0 };
    expect(() => tensorByteSize(t)).not.toThrow();
    expect(tensorByteSize(t)).toBe(100 * 7 * 2);
  });

  it("i byte dei DUE tensori BF16 reali tornano col dump committato", () => {
    // Il dump da' n e byte per CLASSE (non i nomi): `other:BF16` e'
    // ffn_gate_inp, `ffn/shexp:BF16` e' ffn_gate_inp_shexp. Se un domani il
    // file bersaglio ne portasse un terzo, questo caso lo fa notare.
    const dump = JSON.parse(readFileSync(
      join(process.cwd(), "results/eval/q35-header-dump-bartowski-2026-08-17.json"), "utf8",
    )) as Array<{ file: string; typeHistogram: Record<string, { n: number; bytes: number }> }>;
    const q2k = dump.find((e) => e.file.endsWith("Q2_K.gguf"))!;
    const bf16 = Object.entries(q2k.typeHistogram).filter(([k]) => k.includes("BF16"));
    expect(bf16.map(([k]) => k).sort()).toEqual(["ffn/shexp:BF16 *", "other:BF16 *"]);
    expect(bf16.reduce((a, [, v]) => a + v.n, 0)).toBe(2);

    // dModel 2048, nExpert 256 (metadata dello stesso dump): le due dims della
    // shape 35B, passate per tensorByteSize, devono dare quei byte esatti.
    const bytes = (dims: number[]) => tensorByteSize({ name: "x", dims, type: GGML_TYPE.BF16, offset: 0 });
    expect(bytes([2048, 256])).toBe(q2k.typeHistogram["other:BF16 *"].bytes);
    expect(bytes([2048])).toBe(q2k.typeHistogram["ffn/shexp:BF16 *"].bytes);
  });
});

describe("bf16ToF32: i bit attesi sui casi limite", () => {
  // Ogni atteso e' il float32 i cui 16 bit ALTI sono l'input e i 16 bassi zero.
  it("+0 e -0 restano distinti (Object.is, non ==)", () => {
    expect(bf16ToF32(0x0000)).toBe(0);
    expect(Object.is(bf16ToF32(0x0000), 0)).toBe(true);
    expect(bf16ToF32(0x8000)).toBe(-0);
    expect(Object.is(bf16ToF32(0x8000), -0)).toBe(true);
  });

  it("normali con segno: 0x3F80 = +1, 0xBF80 = -1, 0xC049 = -3,140625", () => {
    expect(bf16ToF32(0x3f80)).toBe(1);
    expect(bf16ToF32(0xbf80)).toBe(-1);
    expect(bf16ToF32(0xc049)).toBe(-3.140625); // 0xC0490000 = -pi troncato a bf16
  });

  it("subnormale: 0x0001 = 2^-133 (float32 0x00010000 = 2^16 x 2^-149)", () => {
    expect(bf16ToF32(0x0001)).toBe(2 ** -133);
    expect(bf16ToF32(0x8001)).toBe(-(2 ** -133));
  });

  it("infiniti: 0x7F80 = +Inf, 0xFF80 = -Inf", () => {
    expect(bf16ToF32(0x7f80)).toBe(Infinity);
    expect(bf16ToF32(0xff80)).toBe(-Infinity);
  });

  it("NaN: 0x7FC0 e 0xFFC1 non sono numeri (Number.isNaN, perche' NaN !== NaN)", () => {
    expect(Number.isNaN(bf16ToF32(0x7fc0))).toBe(true);
    expect(Number.isNaN(bf16ToF32(0xffc1))).toBe(true);
  });
});

// --- header sintetico qwen35moe (oggetto GgufFile, non byte: validateQwen35
// --- prende il file GIA' parsato, quindi non serve un terzo builder GGUF).
// La fixture ha la TESTA MTP perche' ce l'ha il file vero, ed e' li' che stanno
// i suoi due BF16: una fixture senza MTP avrebbe testato un file che non esiste.

const D = 8;
const N_HEAD = 2, N_KV = 1, HEAD_DIM = 4;
const N_EXPERT = 4, D_FFN_EXPERT = 8;
const LIN_K = 2, LIN_V = 2, LIN_HD = 4, LIN_INNER = LIN_V * LIN_HD; // 8
const CONV_K = 4;
const QKV = (2 * LIN_K + LIN_V) * LIN_HD; // 24
const N_LAYER = 4, INTERVAL = 4, N_MTP = 1; // full solo l'ultimo (l % 4 === 3); MTP = blk.4
const VOCAB = 16;

const Q = GGML_TYPE.Q4_K; // un qualunque tipo di W_QUANT: qui non e' l'oggetto del test
const F = GGML_TYPE.F32;

const MTP = `blk.${N_LAYER}.`; // il blocco della testa MTP: blk.40 sul file vero
/** I due gate di routing: gli UNICI tensori che possono essere BF16. */
const ROUTER_NAMES = ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"];

/** Header minimo del 35B MoE; `bf16` forza a BF16 i tensori elencati per nome. */
function moeHeader(bf16: readonly string[] = []): GgufFile {
  const forced = new Set(bf16);
  const t = (name: string, dims: number[], type: number): GgufTensorInfo =>
    ({ name, dims, type: forced.has(name) ? GGML_TYPE.BF16 : type, offset: 0 });
  const tensors: GgufTensorInfo[] = [
    t("token_embd.weight", [D, VOCAB], Q),
    t("output_norm.weight", [D], F),
    t("output.weight", [D, VOCAB], Q),
  ];
  for (let l = 0; l < N_LAYER + N_MTP; l++) {
    const b = `blk.${l}.`;
    const full = l >= N_LAYER || l % INTERVAL === INTERVAL - 1; // la testa MTP e' SEMPRE full
    tensors.push(t(`${b}attn_norm.weight`, [D], F), t(`${b}post_attention_norm.weight`, [D], F));
    if (full) {
      tensors.push(
        t(`${b}attn_q.weight`, [D, 2 * N_HEAD * HEAD_DIM], Q),
        t(`${b}attn_k.weight`, [D, N_KV * HEAD_DIM], Q),
        t(`${b}attn_v.weight`, [D, N_KV * HEAD_DIM], Q),
        t(`${b}attn_q_norm.weight`, [HEAD_DIM], F),
        t(`${b}attn_k_norm.weight`, [HEAD_DIM], F),
        t(`${b}attn_output.weight`, [N_HEAD * HEAD_DIM, D], Q),
      );
    } else {
      tensors.push(
        t(`${b}attn_qkv.weight`, [D, QKV], Q),
        t(`${b}attn_gate.weight`, [D, LIN_INNER], Q),
        t(`${b}ssm_conv1d.weight`, [CONV_K, QKV], F),
        t(`${b}ssm_alpha.weight`, [D, LIN_V], F),
        t(`${b}ssm_beta.weight`, [D, LIN_V], F),
        t(`${b}ssm_a`, [LIN_V], F),
        t(`${b}ssm_dt.bias`, [LIN_V], F),
        t(`${b}ssm_norm.weight`, [LIN_HD], F),
        t(`${b}ssm_out.weight`, [LIN_INNER, D], Q),
      );
    }
    tensors.push(
      t(`${b}ffn_gate_inp.weight`, [D, N_EXPERT], F),
      t(`${b}ffn_gate_exps.weight`, [D, D_FFN_EXPERT, N_EXPERT], Q),
      t(`${b}ffn_up_exps.weight`, [D, D_FFN_EXPERT, N_EXPERT], Q),
      t(`${b}ffn_down_exps.weight`, [D_FFN_EXPERT, D, N_EXPERT], Q),
      t(`${b}ffn_gate_inp_shexp.weight`, [D], F),
      t(`${b}ffn_gate_shexp.weight`, [D, D_FFN_EXPERT], Q),
      t(`${b}ffn_up_shexp.weight`, [D, D_FFN_EXPERT], Q),
      t(`${b}ffn_down_shexp.weight`, [D_FFN_EXPERT, D], Q),
    );
    if (l >= N_LAYER) {
      tensors.push(
        t(`${b}nextn.eh_proj.weight`, [2 * D, D], Q),
        t(`${b}nextn.enorm.weight`, [D], F),
        t(`${b}nextn.hnorm.weight`, [D], F),
        t(`${b}nextn.shared_head_norm.weight`, [D], F),
      );
    }
  }
  return {
    version: 3,
    alignment: 32,
    metadata: {
      "general.architecture": "qwen35moe",
      "general.name": "bf16-router-fixture",
      "qwen35moe.block_count": N_LAYER + N_MTP,
      "qwen35moe.nextn_predict_layers": N_MTP,
      "qwen35moe.full_attention_interval": INTERVAL,
      "qwen35moe.embedding_length": D,
      "qwen35moe.attention.head_count": N_HEAD,
      "qwen35moe.attention.head_count_kv": N_KV,
      "qwen35moe.attention.key_length": HEAD_DIM,
      "qwen35moe.attention.value_length": HEAD_DIM,
      "qwen35moe.attention.layer_norm_rms_epsilon": 1e-6,
      "qwen35moe.rope.dimension_count": 2,
      "qwen35moe.rope.freq_base": 1e6,
      "qwen35moe.context_length": 64,
      "qwen35moe.ssm.conv_kernel": CONV_K,
      "qwen35moe.ssm.state_size": LIN_HD,
      "qwen35moe.ssm.group_count": LIN_K,
      "qwen35moe.ssm.time_step_rank": LIN_V,
      "qwen35moe.ssm.inner_size": LIN_INNER,
      "qwen35moe.expert_count": N_EXPERT,
      "qwen35moe.expert_used_count": 2,
      "qwen35moe.expert_feed_forward_length": D_FFN_EXPERT,
      "tokenizer.ggml.tokens": Array.from({ length: VOCAB }, (_, i) => `t${i}`),
    },
    tensors,
    dataOffset: 0,
  };
}

describe("validateQwen35: BF16 passa SOLO dai due gate di routing", () => {
  it("l'header di controllo (tutto F32/quant) e' valido — la fixture non mente", () => {
    const { shape } = validateQwen35(moeHeader());
    expect(shape.arch).toBe("qwen35moe");
    expect(shape.nLayer).toBe(N_LAYER);
    expect(shape.mtpLayers).toBe(N_MTP);
    expect(shape.nExpert).toBe(N_EXPERT);
  });

  it("ACCETTA i DUE router BF16 sul blocco MTP (la forma del file bartowski Q2_K)", () => {
    const { shape } = validateQwen35(moeHeader(ROUTER_NAMES.map((n) => MTP + n)));
    expect(shape.arch).toBe("qwen35moe");
  });

  it.each(ROUTER_NAMES)("ACCETTA %s BF16 anche da solo", (n) => {
    expect(() => validateQwen35(moeHeader([MTP + n]))).not.toThrow();
  });

  it("LANCIA se un tensore di PESI (ffn_gate_exps) e' BF16: W_QUANT non si allarga", () => {
    expect(() => validateQwen35(moeHeader([`${MTP}ffn_gate_exps.weight`])))
      .toThrow(/ffn_gate_exps\.weight tipo 30 fuori allow-list/);
  });

  // IL CASO CHE AVREBBE PRESO IL DIFETTO DEL PRIMO GIRO: invece di elencare i
  // tensori che mi aspetto tollerino BF16, li provo TUTTI. Passa solo se
  // l'insieme accettato coincide ESATTAMENTE coi due gate di routing.
  it("ogni altro tensore in BF16 lancia: l'insieme accettato e' esattamente quello", () => {
    const tutti = moeHeader().tensors.map((t) => t.name);
    expect(tutti.length).toBeGreaterThan(40); // la fixture copre davvero il modello
    const accettati: string[] = [];
    for (const name of tutti) {
      try {
        validateQwen35(moeHeader([name]));
        accettati.push(name);
      } catch { /* atteso per tutto cio' che non e' un gate */ }
    }
    const attesi = tutti.filter((n) => ROUTER_NAMES.some((r) => n.endsWith(r)));
    expect(attesi.length).toBe((N_LAYER + N_MTP) * 2);
    expect(accettati.sort()).toEqual(attesi.sort());
  });
});
