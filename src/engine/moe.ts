// MoE PARAMETRICO — una meccanica, una implementazione (goal fase-D fase 1,
// ruling PI 2026-08-10 / direction §7-ter: il codice si UNIFORMA, non si
// affianca un path di serie B per ogni famiglia nuova).
//
// Qui vivono TRE cose, tutte parametriche sul modello:
//   1. la geometria dei formati quant (una tabella, non un if per famiglia);
//   2. il router (gating + selezione + pesi di mixing);
//   3. il layout dello slab expert e il suo packer.
// GLM-4.7-Flash e Qwen 3.5/3.6 sono CONFIGURAZIONI di questa meccanica.
//
// --- Router, le due semantiche verificate sulla fonte ---
// GLM (build_moe_ffn, oracolo 5f55650, verifica riga-per-riga C2 it.6):
//   probs = sigmoid(logits)                        [gating func=2, r.1864]
//   selezione top-4 su probs + exp_probs_b         [bias solo selezione, r.1883]
//   pesi = probs dei selezionati SENZA bias        [get_rows su probs, r.1940]
//   normalizzati a somma 1, denominatore clampato  [norm_w, r.1958-1960]
//   × 1.8 (expert_weights_scale)                   [r.1967]
// Qwen35moe (qwen35moe.cpp + build_moe_ffn @b10333, verifica q1 it.14):
//   probs = softmax(logits) su TUTTI gli expert; NIENTE bias (exp_probs_b
//   nullptr); selezione top-8 su probs; pesi normalizzati con lo STESSO
//   clamp; NIENTE scale (expert_weights_scale non letto dall'arch ⇒ 0 ⇒
//   il ramo di scale è saltato).
// Pareggi: indice minore in entrambe (argsort stabile — tie-break della
// replica C1, tools/oracle-moe/trace.cpp, e di ggml_argsort_top_k).
import { GLM47_FLASH as G } from "./shape";
import { repackKQuantInto, repackQ4_0, repackQ4_1, repackQ8_0 } from "./quant";

// Clamp del denominatore della normalizzazione (ggml_clamp in build_moe_ffn:
// il più piccolo f16 normale, evita la divisione per zero). Identico nelle
// due famiglie: è del framework, non del modello.
export const WEIGHTS_SUM_CLAMP_MIN = 6.103515625e-5;

// ---------------------------------------------------------------------------
// 1. Geometria dei formati quant
// ---------------------------------------------------------------------------

export type QuantKind = "q4_0" | "q4_1" | "q8_0" | "q2_K" | "q3_K" | "q4_K" | "q5_K" | "q6_K";

interface QuantGeom {
  /** pesi per blocco (32 legacy, 256 K-quant) */
  weightsPerBlock: number;
  /** byte per blocco NEL GGUF */
  srcBytesPerBlock: number;
  /**
   * forma del repack: i formati legacy si spezzano in due binding (qs +
   * scales, come li vuole `gemvQuantWgsl`); i K-quant restano un binding
   * unico di superblocchi paddati a word (come li vuole `gemvQ{4,5,6}KWgsl`).
   */
  split: { qsBytesPerBlock: number; scalesBytesPerBlock: number } | null;
}

const QUANT: Record<QuantKind, QuantGeom> = {
  q4_0: { weightsPerBlock: 32, srcBytesPerBlock: 18, split: { qsBytesPerBlock: 16, scalesBytesPerBlock: 2 } },
  q4_1: { weightsPerBlock: 32, srcBytesPerBlock: 20, split: { qsBytesPerBlock: 16, scalesBytesPerBlock: 4 } },
  q8_0: { weightsPerBlock: 32, srcBytesPerBlock: 34, split: { qsBytesPerBlock: 32, scalesBytesPerBlock: 2 } },
  // Q2_K e Q3_K entrano il 2026-08-17: sono i formati degli expert del quant
  // che fa stare il parco nell'arena. Taglie da `quant.ts`, che li dequantizza
  // gia' (fase 4c) con byte-identita' verificata contro llama-quantize.
  q2_K: { weightsPerBlock: 256, srcBytesPerBlock: 84, split: null },
  q3_K: { weightsPerBlock: 256, srcBytesPerBlock: 110, split: null },
  q4_K: { weightsPerBlock: 256, srcBytesPerBlock: 144, split: null },
  q5_K: { weightsPerBlock: 256, srcBytesPerBlock: 176, split: null },
  q6_K: { weightsPerBlock: 256, srcBytesPerBlock: 210, split: null },
};

/** byte del segmento monolitico dopo `repackKQuant` (superblocchi paddati a word). */
const kquantPackedBytes = (kind: QuantKind, nBlocks: number): number =>
  nBlocks * Math.ceil(QUANT[kind].srcBytesPerBlock / 4) * 4;

// ---------------------------------------------------------------------------
// 2. Router parametrico
// ---------------------------------------------------------------------------

export interface RouterSelection {
  experts: Int32Array;   // top-K id expert, ordine decrescente di score di selezione
  weights: Float64Array; // pesi di mixing allineati a experts (già scalati)
}

export interface RouterConfig {
  gating: "sigmoid" | "softmax";
  nUsed: number;
  /** expert_weights_scale (GLM 1.8; qwen35moe non lo legge ⇒ 1 = nessuno scale) */
  weightsScale: number;
  /** il bias entra SOLO nella selezione, mai nei pesi (DeepSeek-V3 / GLM) */
  usesBias: boolean;
}

export const ROUTER_GLM47: RouterConfig = {
  gating: "sigmoid", nUsed: G.nExpertUsed, weightsScale: G.weightsScale, usesBias: true,
};
export const ROUTER_QWEN35MOE: RouterConfig = {
  gating: "softmax", nUsed: 8, weightsScale: 1, usesBias: false,
};

/**
 * Selezione + pesi di mixing. `bias` è ignorato (e può essere null) quando
 * `cfg.usesBias` è false: la firma resta una sola per entrambe le famiglie.
 */
export function routerSelect(
  logits: ArrayLike<number>,
  bias: ArrayLike<number> | null,
  cfg: RouterConfig,
): RouterSelection {
  const n = logits.length;
  const nUsed = cfg.nUsed;
  const probs = new Float64Array(n);
  if (cfg.gating === "sigmoid") {
    for (let i = 0; i < n; i++) probs[i] = 1 / (1 + Math.exp(-logits[i]));
  } else {
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
    let sum = 0;
    for (let i = 0; i < n; i++) { probs[i] = Math.exp(logits[i] - mx); sum += probs[i]; }
    for (let i = 0; i < n; i++) probs[i] /= sum;
  }
  // score di SELEZIONE: probs (+ bias dove previsto). I pesi restano sui probs.
  const sel = new Float64Array(n);
  if (cfg.usesBias) {
    if (!bias) throw new Error("routerSelect: la config richiede il bias (exp_probs_b)");
    for (let i = 0; i < n; i++) sel[i] = probs[i] + bias[i];
  } else {
    sel.set(probs);
  }
  const experts = new Int32Array(nUsed);
  const taken = new Uint8Array(n);
  for (let k = 0; k < nUsed; k++) {
    let best = -1;
    for (let i = 0; i < n; i++) {
      if (!taken[i] && (best < 0 || sel[i] > sel[best])) best = i; // pareggio ⇒ indice minore
    }
    experts[k] = best;
    taken[best] = 1;
  }
  let sum = 0;
  for (let k = 0; k < nUsed; k++) sum += probs[experts[k]];
  const denom = Math.max(sum, WEIGHTS_SUM_CLAMP_MIN);
  const weights = new Float64Array(nUsed);
  for (let k = 0; k < nUsed; k++) weights[k] = (probs[experts[k]] / denom) * cfg.weightsScale;
  return { experts, weights };
}

// ---------------------------------------------------------------------------
// 3. Slab expert parametrico
// ---------------------------------------------------------------------------
//
// Slab = gate+up+down di UN expert nel layout repack del motore. Tutti gli
// offset sono multipli di 256: i bind group per-slot bindano sotto-range del
// buffer di classe e WebGPU esige `minStorageBufferOffsetAlignment` (≤256
// garantito). Una CLASSE per combinazione di formati: GLM ne ha due (down
// Q4_0 su blk.5-46, down Q4_1 su blk.1-4), qwen35moe ne ha due (down Q4_K,
// down Q6_K). Il repack non cambia i byte totali del GGUF.

/** Un tensore dentro lo slab: 1 segmento (K-quant) o 2 (legacy qs+scales). */
export interface SlabTensorLayout {
  kind: QuantKind;
  nBlocks: number;
  /** offset del segmento principale: qs (legacy) o superblocchi (K-quant) */
  data: number;
  dataBytes: number;
  /** offset del segmento scale — null sui K-quant (nessun segmento separato) */
  scales: number | null;
  scalesBytes: number;
}

export interface SlabLayout {
  /** identificatore della classe, per i log e le mappe di arena */
  id: string;
  gate: SlabTensorLayout;
  up: SlabTensorLayout;
  down: SlabTensorLayout;
  bytes: number;

  // --- COMPAT (goal fase-D fase 1): i call site GLM storici leggono questi
  // campi piatti. Sono DERIVATI dai tre SlabTensorLayout sopra — nessuna
  // seconda fonte di verità. Spariscono quando i call site migrano alla
  // vista generica (docket fase-D). ---
  downKind: QuantKind;
  gateQs: number; gateScales: number;
  upQs: number; upScales: number;
  downQs: number; downScales: number;
  qsBytes: number; gateScalesBytes: number; downScalesBytes: number;
}

export interface ExpertTensorSpec { kind: QuantKind; elems: number }

/** Costruisce una classe di slab da (gate, up, down). UNICO builder. */
export function mkSlabLayout(
  id: string, gate: ExpertTensorSpec, up: ExpertTensorSpec, down: ExpertTensorSpec,
): SlabLayout {
  let off = 0;
  const place = (spec: ExpertTensorSpec): SlabTensorLayout => {
    const geom = QUANT[spec.kind];
    if (spec.elems % geom.weightsPerBlock !== 0) {
      throw new Error(`slab: ${spec.kind} con ${spec.elems} elementi non multiplo di ${geom.weightsPerBlock}`);
    }
    const nBlocks = spec.elems / geom.weightsPerBlock;
    const data = off;
    const dataBytes = geom.split ? nBlocks * geom.split.qsBytesPerBlock : kquantPackedBytes(spec.kind, nBlocks);
    off += dataBytes;
    let scales: number | null = null;
    let scalesBytes = 0;
    if (geom.split) {
      scales = off;
      scalesBytes = nBlocks * geom.split.scalesBytesPerBlock;
      off += scalesBytes;
    }
    return { kind: spec.kind, nBlocks, data, dataBytes, scales, scalesBytes };
  };
  const g = place(gate), u = place(up), d = place(down);
  const bytes = off;
  for (const o of [g.data, g.scales, u.data, u.scales, d.data, d.scales, bytes]) {
    if (o !== null && o % 256 !== 0) throw new Error(`slab layout ${id}: offset ${o} non allineato a 256`);
  }
  const out: SlabLayout = {
    id, gate: g, up: u, down: d, bytes,
    downKind: d.kind,
    gateQs: g.data, upQs: u.data, downQs: d.data,
    qsBytes: g.dataBytes, gateScalesBytes: g.scalesBytes, downScalesBytes: d.scalesBytes,
  } as SlabLayout;
  // I campi `*Scales` esistono SOLO per i formati legacy (2 segmenti). Sui
  // K-quant non c'è un segmento scale: leggerli darebbe un offset finto e un
  // binding di taglia 0 SENZA errore (trappola trovata dal verifier it.2).
  // Qui diventano getter che FALLISCONO, invece di mentire.
  const scaleField = (name: string, t: SlabTensorLayout): void => {
    Object.defineProperty(out, name, {
      // NON enumerabile (verifier it.3): altrimenti {...layout} e JSON.stringify
      // su un layout K-quant lancerebbero con un messaggio fuorviante.
      enumerable: false,
      get(): number {
        if (t.scales === null) {
          throw new Error(
            `slab ${id}: ${name} non esiste sul formato ${t.kind} (K-quant: un solo segmento). ` +
            "Usa la vista generica (layout.gate/up/down) invece dei campi compat legacy.");
        }
        return t.scales;
      },
    });
  };
  scaleField("gateScales", g);
  scaleField("upScales", u);
  scaleField("downScales", d);
  return out;
}

// --- Classi GLM-4.7-Flash (spec C2 §1): 5.308.416 B e 5.505.024 B, le due
// size-class misurate sul file. Ora DERIVATE dal builder generico. ---
const GLM_EXPERT_ELEMS = G.dModel * G.dFfnExpert; // 3.145.728
export const SLAB_DOWN_Q4_0: SlabLayout = mkSlabLayout(
  "glm-down-q4_0",
  { kind: "q4_0", elems: GLM_EXPERT_ELEMS },
  { kind: "q4_0", elems: GLM_EXPERT_ELEMS },
  { kind: "q4_0", elems: GLM_EXPERT_ELEMS },
);
export const SLAB_DOWN_Q4_1: SlabLayout = mkSlabLayout(
  "glm-down-q4_1",
  { kind: "q4_0", elems: GLM_EXPERT_ELEMS },
  { kind: "q4_0", elems: GLM_EXPERT_ELEMS },
  { kind: "q4_1", elems: GLM_EXPERT_ELEMS },
);

/**
 * Impacchetta i byte GREZZI GGUF dei tre tensori di un expert nello slab.
 * PARAMETRICO sui formati: legacy → repack a 2 segmenti; K-quant → segmento
 * unico. (Il repack al momento dell'upload è il costo di miss che la fase 2
 * del goal sposta all'import.)
 */
export function packExpertSlab(
  gateRaw: Uint8Array, upRaw: Uint8Array, downRaw: Uint8Array, layout: SlabLayout,
): Uint8Array {
  const out = new Uint8Array(layout.bytes);
  const putWords = (data: Uint32Array, off: number): void =>
    out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), off);
  const one = (raw: Uint8Array, t: SlabTensorLayout, name: string): void => {
    const geom = QUANT[t.kind];
    const expect = t.nBlocks * geom.srcBytesPerBlock;
    if (raw.length !== expect) throw new Error(`slab: ${name} ${raw.length} byte, attesi ${expect} (${t.kind})`);
    if (!geom.split) {
      // DIRETTAMENTE nello slab: nessun array temporaneo, nessun zero-fill in
      // piu' (fase D fase 2 — erano 3 passate sugli stessi byte).
      repackKQuantInto(raw, 0, t.nBlocks, geom.srcBytesPerBlock, out, t.data);
      return;
    }
    const r = t.kind === "q4_1" ? repackQ4_1(raw, 0, t.nBlocks)
      : t.kind === "q8_0" ? repackQ8_0(raw, 0, t.nBlocks)
      : repackQ4_0(raw, 0, t.nBlocks);
    putWords(r.qs, t.data);
    putWords(r.scales, t.scales as number);
  };
  one(gateRaw, layout.gate, "gate");
  one(upRaw, layout.up, "up");
  one(downRaw, layout.down, "down");
  return out;
}
