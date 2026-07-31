// MoE GLM-4.7-Flash (goal C2 fase 5): router CPU e layout slab expert per la
// residenza minima (spec §4-§5).
//
// Router — replica di build_moe_ffn (llama-graph.cpp, oracolo 5f55650,
// verifica riga-per-riga it.6):
//   probs = sigmoid(logits)                        [gating func=2, r.1861-1864]
//   selezione top-4 su probs + exp_probs_b         [bias solo selezione, r.1878-1882]
//   pesi di mixing = probs dei selezionati SENZA bias [get_rows su probs, r.1945]
//   normalizzati a somma 1 con denominatore clampato a 6.103515625e-5
//   (min f16 normale)                              [norm_w, r.1949-1964]
//   × 1.8 (expert_weights_scale)                   [r.1965-1968]
// Pareggi nella selezione: indice minore (argsort stabile — stesso tie-break
// della replica C1, tools/oracle-moe/trace.cpp). n_expert_groups=1 nel GGUF ⇒
// niente logica gruppi. La selezione gira su CPU: i top-4 decidono QUALI slab
// bindare (residenza), quindi devono essere noti prima dell'encode del blocco
// expert.
import { GLM47_FLASH as G } from "./shape";
import { repackQ4_0, repackQ4_1 } from "./quant";

// Clamp del denominatore della normalizzazione (ggml_clamp in build_moe_ffn:
// il più piccolo f16 normale, evita la divisione per zero).
export const WEIGHTS_SUM_CLAMP_MIN = 6.103515625e-5;

export interface RouterSelection {
  experts: Int32Array;   // top-4 id expert, ordine decrescente di score biased
  weights: Float64Array; // pesi di mixing allineati a experts (già ×1.8)
}

export function routerSelect(
  logits: ArrayLike<number>, bias: ArrayLike<number>, nUsed = G.nExpertUsed,
): RouterSelection {
  const n = logits.length;
  const probs = new Float64Array(n);
  const sel = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    probs[i] = 1 / (1 + Math.exp(-logits[i]));
    sel[i] = probs[i] + bias[i];
  }
  const experts = new Int32Array(nUsed);
  const taken = new Uint8Array(n);
  for (let k = 0; k < nUsed; k++) {
    let best = -1;
    for (let i = 0; i < n; i++) {
      if (!taken[i] && (best < 0 || sel[i] > sel[best])) best = i; // pareggio ⇒ resta l'indice minore
    }
    experts[k] = best;
    taken[best] = 1;
  }
  let sum = 0;
  for (let k = 0; k < nUsed; k++) sum += probs[experts[k]];
  const denom = Math.max(sum, WEIGHTS_SUM_CLAMP_MIN);
  const weights = new Float64Array(nUsed);
  for (let k = 0; k < nUsed; k++) weights[k] = (probs[experts[k]] / denom) * G.weightsScale;
  return { experts, weights };
}

// --- Slab expert (spec §5): gate+up+down di UN expert nel layout repack del
// motore ([qs | scales] per tensore). Tutti gli offset sono multipli di 256:
// i bind group per-slot bindano sotto-range del buffer di classe e WebGPU
// richiede l'allineamento minStorageBufferOffsetAlignment (≤256 garantito).
// Due size-class ESATTE (spec §1): down Q4_0 (blk.5-46, 2.688 expert) e down
// Q4_1 (blk.1-4, 256 expert). Il repack non cambia i byte totali del GGUF.

// gate [2048→1536], up [2048→1536], down [1536→2048]: stesso conteggio blocchi.
const EXPERT_TENSOR_BLOCKS = (G.dModel / 32) * G.dFfnExpert; // 98.304
const QS_BYTES = EXPERT_TENSOR_BLOCKS * 16;                  // 1.572.864
const SCALES_Q4_0_BYTES = EXPERT_TENSOR_BLOCKS * 2;          // 196.608 (1 f16/blocco)
const SCALES_Q4_1_BYTES = EXPERT_TENSOR_BLOCKS * 4;          // 393.216 (d+m f16/blocco)

export interface SlabLayout {
  downKind: "q4_0" | "q4_1";
  gateQs: number; gateScales: number;
  upQs: number; upScales: number;
  downQs: number; downScales: number;
  qsBytes: number;         // taglia di ogni segmento qs
  gateScalesBytes: number; // = upScalesBytes (gate/up sempre Q4_0)
  downScalesBytes: number;
  bytes: number;           // taglia totale slab
}

function mkLayout(downKind: "q4_0" | "q4_1"): SlabLayout {
  const downScalesBytes = downKind === "q4_0" ? SCALES_Q4_0_BYTES : SCALES_Q4_1_BYTES;
  const gateQs = 0;
  const gateScales = gateQs + QS_BYTES;
  const upQs = gateScales + SCALES_Q4_0_BYTES;
  const upScales = upQs + QS_BYTES;
  const downQs = upScales + SCALES_Q4_0_BYTES;
  const downScales = downQs + QS_BYTES;
  const bytes = downScales + downScalesBytes;
  for (const off of [gateQs, gateScales, upQs, upScales, downQs, downScales, bytes]) {
    if (off % 256 !== 0) throw new Error(`slab layout: offset ${off} non allineato a 256`);
  }
  return {
    downKind, gateQs, gateScales, upQs, upScales, downQs, downScales,
    qsBytes: QS_BYTES, gateScalesBytes: SCALES_Q4_0_BYTES, downScalesBytes, bytes,
  };
}

// 5.308.416 B e 5.505.024 B: le due size-class misurate sul file (spec §1;
// media pesata 256/2688 = l'expertBytesQ4 di residency-sim C1).
export const SLAB_DOWN_Q4_0: SlabLayout = mkLayout("q4_0");
export const SLAB_DOWN_Q4_1: SlabLayout = mkLayout("q4_1");

// Impacchetta i byte GREZZI GGUF dei tre tensori di un expert nello slab
// (repack CPU al momento dell'upload RAM→VRAM: è parte del costo di miss che
// la telemetria deve misurare). Ritorna un buffer pronto per writeBuffer.
export function packExpertSlab(
  gateRaw: Uint8Array, upRaw: Uint8Array, downRaw: Uint8Array, layout: SlabLayout,
): Uint8Array {
  if (gateRaw.length !== EXPERT_TENSOR_BLOCKS * 18) throw new Error("slab: gate bytes inattesi");
  if (upRaw.length !== EXPERT_TENSOR_BLOCKS * 18) throw new Error("slab: up bytes inattesi");
  const downBlockBytes = layout.downKind === "q4_0" ? 18 : 20;
  if (downRaw.length !== EXPERT_TENSOR_BLOCKS * downBlockBytes) throw new Error("slab: down bytes inattesi");
  const out = new Uint8Array(layout.bytes);
  const put = (data: Uint32Array, off: number) =>
    out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), off);
  const gate = repackQ4_0(gateRaw, 0, EXPERT_TENSOR_BLOCKS);
  put(gate.qs, layout.gateQs);
  put(gate.scales, layout.gateScales);
  const up = repackQ4_0(upRaw, 0, EXPERT_TENSOR_BLOCKS);
  put(up.qs, layout.upQs);
  put(up.scales, layout.upScales);
  const down = (layout.downKind === "q4_0" ? repackQ4_0 : repackQ4_1)(downRaw, 0, EXPERT_TENSOR_BLOCKS);
  put(down.qs, layout.downQs);
  put(down.scales, layout.downScales);
  return out;
}
