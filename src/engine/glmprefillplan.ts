// Piano MoE del prefill batched M>1 (C3a fase 5, spec §5) — puro CPU-side,
// testabile senza GPU (convenzione fase A, come prefillplan del path Qwen).
//
// DECISIONE DI DESIGN (it.26, col criterio dei numeri di fase 4 — journal):
// fra "Sel per token" (ripetere la catena expert del decode per ogni riga del
// chunk: 4M dispatch/layer, pesi riletti una volta PER TOKEN selezionante) e
// "batching per expert con gather" (unione degli expert selezionati nel chunk,
// un GEMM per expert sulle righe che lo selezionano), vince la seconda:
//   - dispatch/layer: |unione| ≤ min(4M, 64) contro 4M — a M=16 l'attesa è
//     E[|unione|] = 64·(1−(1−1/64)^64) ≈ 40 contro 64; il costo per dispatch
//     misurato in fase 1 è ~43 µs, ed è la voce che la 4b ha dovuto abbattere;
//   - traffico pesi expert: ogni expert dell'unione si legge UNA volta per
//     chunk invece che una per token selezionante ⇒ ÷ molteplicità media
//     (≈1.6× a M=16, ≈4× a M=64) — è il regime memory-bound che fase 1 ha
//     misurato 20× sopra il floor (2.22 GB/token su 576 GB/s).
// La spec §5 prescrive già "per unione ... con maschera per token": qui la
// maschera diventa gather esplicito (rows) + combine per token.
//
// IDENTITÀ BIT-A-BIT COL DECODE (la ragione della struttura a slot): il decode
// accumula i 4 expert in ordine k (gemvAccumFast k=0..3 su moeOut, poi addMoe).
// Se il path batched accumulasse per-expert nell'ordine dell'unione, le somme
// f32 riordinate potrebbero flippare gli argmax near-tie (classe misurata in
// slice B: rate 9e-6). Quindi: i kernel batched SCRIVONO y[m][k] in slot
// separati (niente accumulo), e una combine per token somma w[m][k]·y[m][k]
// in ordine k crescente — lo STESSO ordine di somma del decode. La classe di
// rischio è eliminata per costruzione, non tollerata.
import type { RouterSelection } from "./moe";
import { GLM47_FLASH as G } from "./shape";

/** M iniziale del prefill GLM (spec §5, `[ASSUMED]`; si tara sul TTFT). */
export const GLM_PREFILL_M = 16;

export interface MoeExpertBatch {
  /** id expert (0..63) */
  expert: number;
  /** righe del chunk che lo selezionano, ordine crescente (gather) */
  rows: Int32Array;
  /** per ogni riga: lo slot k (0..3) della selezione di quel token — dice
   * alla scrittura del kernel DOVE mettere y[m][k] per la combine */
  slots: Int32Array;
}

export interface MoeChunkPlan {
  m: number;
  /** unione degli expert del chunk, id crescente (ordine deterministico) */
  experts: MoeExpertBatch[];
  /**
   * Pesi di mixing per la combine, layout [m*4+k] in f32 — la stessa
   * precisione che la Sel di produzione porta oggi (selF32, glmmodel).
   */
  weights: Float32Array;
}

/**
 * Raggruppa le selezioni top-4 di un chunk di M token per expert.
 * Invariante (testato): ogni coppia (m, k) compare ESATTAMENTE una volta
 * nell'unione — il piano è una biiezione delle selezioni, non un campione.
 */
export function planMoeChunk(selections: RouterSelection[], mMax = GLM_PREFILL_M): MoeChunkPlan {
  const m = selections.length;
  if (m < 1 || m > mMax) throw new Error(`planMoeChunk: chunk di ${m} righe (ammesso 1..${mMax})`);
  const byExpert = new Map<number, Array<{ row: number; slot: number }>>();
  const weights = new Float32Array(m * G.nExpertUsed);
  for (let row = 0; row < m; row++) {
    const sel = selections[row];
    if (sel.experts.length !== G.nExpertUsed) {
      throw new Error(`planMoeChunk: riga ${row} con ${sel.experts.length} expert (attesi ${G.nExpertUsed})`);
    }
    for (let k = 0; k < G.nExpertUsed; k++) {
      const e = sel.experts[k];
      if (e < 0 || e >= G.nExpert) throw new Error(`planMoeChunk: expert ${e} fuori range (riga ${row}, k ${k})`);
      let a = byExpert.get(e);
      if (!a) { a = []; byExpert.set(e, a); }
      a.push({ row, slot: k });
      weights[row * G.nExpertUsed + k] = sel.weights[k]; // f64→f32, come selF32
    }
  }
  const experts: MoeExpertBatch[] = [...byExpert.keys()].sort((a, b) => a - b).map((e) => {
    const entries = byExpert.get(e)!; // già in ordine di riga (loop crescente)
    return {
      expert: e,
      rows: Int32Array.from(entries.map((x) => x.row)),
      slots: Int32Array.from(entries.map((x) => x.slot)),
    };
  });
  return { m, experts, weights };
}

/**
 * Combine per token, in ordine k CRESCENTE — l'ordine di somma del decode.
 * `base[m]` è x della riga (il residuo su cui il decode fa addMoe), `y[m][k]`
 * gli output per-slot scritti dai kernel batched. Referenza CPU della combine
 * GPU: il test di identità la confronta col percorso per-token.
 * Tutto in f32 (Math.fround su ogni op, come farà il kernel).
 */
export function combineMoeRow(
  base: Float32Array, y: ReadonlyArray<Float32Array>, weights: Float32Array, row: number,
): Float32Array {
  const out = Float32Array.from(base);
  for (let k = 0; k < G.nExpertUsed; k++) {
    const w = weights[row * G.nExpertUsed + k];
    const yk = y[k];
    for (let i = 0; i < out.length; i++) {
      out[i] = Math.fround(out[i] + Math.fround(w * yk[i]));
    }
  }
  return out;
}
