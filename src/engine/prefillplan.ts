// Piano di chunking del prefill multi-token (spec B1 §Forward multi-token) — puro
// CPU-side, testabile in CI senza GPU (convenzione fase A). M è un parametro del
// piano, mai hardcodato nei kernel-call site: lo stesso percorso servirà alla
// verifica spec-dec in fase D.

// Righe per chunk (ultimo chunk parziale ammesso). CONVENZIONE del piano, non
// legge fisica: M=8 e' degenere per l'obiettivo (troppo poco riuso della riga di
// peso per token) — alzata a 16 su ratifica del PI, PHASES C7-2.
export const PREFILL_M = 16;

// ECCEZIONE PINNATA — path DENSO Qwen2.5-0.5B (gpuforward.ts + i report di
// engine.worker.ts che quel motore produce). Non segue la convenzione per una
// ragione NUMERICA, non storica: il consumatore massimo di workgroup storage del
// path fuso e' rmsPairGemmSiluChunkFast con K = dModel 896, e il suo fabbisogno
// e' LINEARE in mMax —
//     4·K·m + 256·m + 16·m  byte
//   m = 8  →  28 672 + 2 048 + 128 =  30 848 B  ≤ cap del device
//   m = 16 →  57 344 + 4 096 + 256 =  61 696 B  >  cap  ⇒ pipeline invalida
// (lo stesso conto e' in gpulimits.ts, QWEN_WORKGROUP_STORAGE_BYTES = 30 848, che
// e' anche il cap effettivamente negoziato col device).
// Portare il denso a 16 richiede prima di ri-tilare quel kernel, non di cambiare
// questa costante. Il conto vive come aritmetica ESEGUIBILE in
// tests/engine-chunking.test.ts: se cambia, il test lo dice.
export const PREFILL_M_DENSE05B = 8;

// Knee misurato dalla sim (prefill-sim-4090-*.json): submit ogni ~64 token; il
// submit unico peggiora (4.22 vs 3.78 ms/token).
// GIUSTIFICAZIONE PINNATA (fase 4d, censimento orfani): i due
// results/engine/prefill-sim-4090-2026-07-29T*.json sono l'UNICA evidenza di
// questo knee — il produttore (sim prefillBatched, f18ff73) e' stato rimosso
// deliberatamente alla chiusura della fase 3 B1 (e9dcb4a). I file si tengono;
// il 64 NON si cambia senza ri-misurare (servirebbe riscrivere la sim).
export const PREFILL_SUBMIT_TOKENS = 64;

export interface PrefillChunk {
  start: number;      // indice nel vettore token del prefill (relativo, 0-based)
  rows: number;       // token nel chunk (≤ mMax; parziale solo l'ultimo)
  posBase: number;    // posizione assoluta della prima riga (= posStart + start)
  submitAfter: boolean; // submit del command buffer dopo questo chunk
}

// Lunghezza causale della riga `row` di un chunk: la riga i vede KV [0, posBase+i],
// cioè posBase+i+1 posizioni (maschera causale intra-chunk dei kernel).
export function causalLen(posBase: number, row: number): number {
  if (posBase < 0 || row < 0) throw new Error("causalLen: argomenti negativi");
  return posBase + row + 1;
}

export function planPrefill(
  nTokens: number,
  posStart: number,
  mMax: number = PREFILL_M,
  submitEvery: number = PREFILL_SUBMIT_TOKENS,
): PrefillChunk[] {
  if (!Number.isInteger(nTokens) || nTokens < 1) throw new Error(`planPrefill: nTokens non valido (${nTokens})`);
  if (!Number.isInteger(posStart) || posStart < 0) throw new Error(`planPrefill: posStart non valido (${posStart})`);
  if (!Number.isInteger(mMax) || mMax < 1) throw new Error(`planPrefill: mMax non valido (${mMax})`);
  if (!Number.isInteger(submitEvery) || submitEvery < 1) throw new Error(`planPrefill: submitEvery non valido (${submitEvery})`);
  const chunks: PrefillChunk[] = [];
  for (let start = 0; start < nTokens; start += mMax) {
    const rows = Math.min(mMax, nTokens - start);
    const done = start + rows;
    chunks.push({
      start,
      rows,
      posBase: posStart + start,
      // il conteggio per la granularità di submit è in TOKEN processati, non in
      // chunk: con l'ultimo chunk parziale il boundary resta sui multipli di 64
      submitAfter: done % submitEvery === 0 || done === nTokens,
    });
  }
  return chunks;
}
