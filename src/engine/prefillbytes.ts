// Contatore del TRAFFICO PESI del prefill (ondata TTFT, t7) — puro CPU-side,
// nessuna GPU e nessuna allocazione di memoria device, testabile in CI su
// qualunque runner (convenzione di `prefillplan.ts` e `moeprefillplan.ts`).
// Il nome dell'atto di allocazione non compare NEPPURE nei commenti: il gate
// strutturale (`tests/engine-one-mechanism.test.ts`) scansiona il sorgente
// grezzo, e questo modulo deve restare fuori dalla sua allowlist.
//
// PERCHE' ESISTE. Il prefill a M righe e' memory-bound sui pesi: cio' che
// decide il TTFT non e' il numero di FLOP ma quante volte i byte dei pesi
// attraversano la memoria. Le due forme del GEMM che il motore emette hanno
// riusi DIVERSI PER COSTRUZIONE, non per taratura:
//   - `legacy`: M GEMV replicate su `wid.z`. Ogni fetta z rilegge l'intera
//     matrice: riuso dei pesi ZERO ⇒ M · N · bytesPerRow. E' la forma che il
//     motore emette OGGI per il piano gemello (`pushB` in `q35gpumodel.ts`).
//   - `multirow`: una passata sui pesi per chunk, le M righe di attivazione
//     stanno nei registri/workgroup ⇒ N · bytesPerRow, e M NON entra.
// Il rapporto fra le due e' quindi ESATTAMENTE M, ed e' verificabile prima di
// qualunque run (v. `tests/engine-prefillbytes.test.ts`).
//
// PERCHE' UN METER E NON UNA FORMULA. Il done-when del goal chiede "RIUSO
// misurato non dedotto": una formula scritta a parte misura la formula, non il
// motore. Il consumatore (`q35gpumodel`) alimenta il meter AL MOMENTO DEL PUSH
// di ogni dispatch del piano — cosi' il numero e' contato sul piano che il
// motore emette davvero, e un dispatch dimenticato o rimasto legacy si vede nel
// totale invece di sparire nella derivazione.

/**
 * Formati dei pesi che i GEMM del prefill leggono.
 *
 * L'elenco copre TUTTI i rami di `loadW` del consumatore (`q35gpumodel.ts`):
 * F32, Q4_0/Q4_1, Q8_0 e i tre K-quant Q4_K/Q5_K/Q6_K — il 4B ha `ssm_out` in
 * Q5_K e la down MoE del 35B e' q4_K oppure q6_K, e per entrambi il motore
 * spinge un dispatch del piano gemello. Un kind mancante qui non e' un buco
 * innocuo: il consumatore non avrebbe nulla da passare ad `add()` e quei byte
 * sparirebbero dal totale, cioe' proprio il modo di fallire che il meter
 * esiste per rendere visibile.
 * NOTA PER IL PI: `q4_K`/`q5_K` ESTENDONO l'unione fissata nell'interfaceFreeze
 * (che elencava q4_0|q4_1|q8_0|q6_K|f32). E' un superset — nessun chiamante
 * scritto sull'unione precedente si rompe — ma resta un emendamento, non una
 * scelta di implementazione: e' segnalato, non consegnato in silenzio.
 */
export type PrefillQuantKind = "q4_0" | "q4_1" | "q8_0" | "q4_K" | "q5_K" | "q6_K" | "f32";

/** Le due forme del GEMM del prefill: v. il commento in testa. */
export type PrefillGemmForm = "legacy" | "multirow";

/**
 * Geometria del blocco quantizzato. I byte/riga si contano SUL BLOCCO, non su
 * una densita' media: e' la differenza fra il conteggio dei byte davvero letti
 * dal kernel e una stima.
 *
 * `gguf` = byte del blocco nel file; `device` = byte che il blocco occupa (e
 * che il kernel attraversa) nel layout in memoria video. I due numeri NON
 * coincidono sempre, e qui conta il secondo:
 *   q4_0: 32 pesi · gguf 18 = device 18 (repack in due buffer: 16 B di nibble
 *         in `qs` + 2 B di scala f16 in `scales`, `quant.ts` §Repack)
 *   q4_1: 32 pesi · gguf 20 = device 20 (16 B di nibble + d,m f16)
 *   q8_0: 32 pesi · gguf 34 = device 34 (32 B int8 + scala f16)
 *   q4_K: 256 pesi · gguf 144 = device 144 (gia' multiplo di 4)
 *   q5_K: 256 pesi · gguf 176 = device 176 (gia' multiplo di 4)
 *   q6_K: 256 pesi · gguf 210 → device 212 — il repack K-quant allinea il
 *         superblocco alla parola (`quant.ts:301`, `stride = ceil(bb/4)*4`, 53
 *         word con 2 B di pad) e il kernel indicizza con QUELLO stride. Contare
 *         210 sottostimerebbe di ~0,95% ogni riga q6_K.
 *   f32:  1 peso · 4 B (blocco unitario)
 * `tests/engine-prefillbytes.test.ts` verifica questi valori CONTRO le costanti
 * di `quant.ts`, non contro se stessi. Qui `quant.ts` non si importa di
 * proposito: questo file deve restare senza dipendenze per non trascinare il
 * repack dentro un contatore.
 */
const BLOCK: Record<PrefillQuantKind, { weights: number; device: number }> = {
  q4_0: { weights: 32, device: 18 },
  q4_1: { weights: 32, device: 20 },
  q8_0: { weights: 32, device: 34 },
  q4_K: { weights: 256, device: 144 },
  q5_K: { weights: 256, device: 176 },
  q6_K: { weights: 256, device: 212 },
  f32: { weights: 1, device: 4 },
};

/**
 * Byte di peso letti per UNA riga di lunghezza K nel formato `kind`.
 * K non multiplo del blocco e' un errore: un arrotondamento silenzioso qui
 * falsifica il rapporto di riuso che tutto il resto misura.
 */
export function weightBytesPerRow(kind: PrefillQuantKind, K: number): number {
  const b = BLOCK[kind];
  if (b === undefined) throw new Error(`weightBytesPerRow: kind sconosciuto (${String(kind)})`);
  if (!Number.isInteger(K) || K < 1) throw new Error(`weightBytesPerRow: K non valido (${K})`);
  if (K % b.weights !== 0) {
    throw new Error(`weightBytesPerRow: K=${K} non e' multiplo del blocco ${kind} (${b.weights} pesi)`);
  }
  return (K / b.weights) * b.device;
}

/** Un dispatch di GEMM del prefill: forma, formato dei pesi e geometria. */
export interface PrefillDispatch {
  form: PrefillGemmForm;
  kind: PrefillQuantKind;
  /** lunghezza della riga di pesi (dimensione contratta) */
  K: number;
  /** righe di pesi (uscite del GEMM) */
  N: number;
  /**
   * ESTENSIONE z DEL DISPATCH EMESSO — non le righe utili del chunk.
   *
   * Nel motore la z del piano gemello e' la costante `M_MAX`
   * (`q35gpumodel.ts:444,469,720` passano tutti `[gx, gy, M_MAX]`), e il kernel
   * batch non ha early-out per riga non valida (`kernels/wgsl.ts`: `let xRB =
   * wid.z * K; let yRB = wid.z * N;`, nessuna guardia su un conteggio di righe
   * valide). Su un chunk parziale — 20 token con M_MAX=8 danno chunk 8,8,4
   * (`prefillplan.ts`) — le 8 fette rileggono comunque l'intera matrice: il
   * traffico reale dell'ultimo chunk e' 8·N·bytesPerRow, non 4·N·bytesPerRow.
   * Passare `chunk.rows` invece di `M_MAX` sottostima l'ultimo chunk di quasi
   * ogni prompt, e la sottostima colpisce solo il ramo `legacy` (l'unico dove M
   * entra): falserebbe il rapporto legacy/multirow verso il basso.
   * Regola: M = il terzo argomento del dispatch, letto dal sito di push.
   */
  M: number;
}

/**
 * Byte di peso che UN dispatch fa attraversare la memoria.
 *   legacy   = M · N · bytesPerRow  (M fette z: riuso ZERO)
 *   multirow = N · bytesPerRow      (una passata sui pesi per chunk)
 * Il traffico di ATTIVAZIONI non entra: e' O(M·K) contro O(N·K) byte di pesi e
 * nel regime del prefill non e' la voce che decide (contarlo qui mescolerebbe
 * due grandezze con riusi diversi).
 */
export function dispatchWeightBytes(o: PrefillDispatch): number {
  if (o.form !== "legacy" && o.form !== "multirow") {
    throw new Error(`dispatchWeightBytes: form sconosciuta (${String(o.form)})`);
  }
  if (!Number.isInteger(o.N) || o.N < 1) throw new Error(`dispatchWeightBytes: N non valido (${o.N})`);
  if (!Number.isInteger(o.M) || o.M < 1) throw new Error(`dispatchWeightBytes: M non valido (${o.M})`);
  const perRow = weightBytesPerRow(o.kind, o.K);
  return o.form === "legacy" ? o.M * o.N * perRow : o.N * perRow;
}

/**
 * Accumulatore dei byte di peso su una SEQUENZA di dispatch. Si alimenta al
 * push di ogni dispatch del piano: il totale e' quello del piano emesso — di
 * norma TUTTI i chunk del prefill, non uno solo (`planPrefill` produce
 * ⌈nTokens/M_MAX⌉ chunk e il motore li spinge nello stesso comando).
 */
export interface PrefillWeightMeter {
  add(o: PrefillDispatch): void;
  totalBytes(): number;
  /**
   * Byte di peso per TOKEN DI PREFILL: totalBytes / M, dove M e' il numero
   * TOTALE di token accumulati nel meter — non l'M del singolo dispatch.
   *
   * Il meter accumula l'intero piano: su 512 token con M_MAX=8 ci sono 64
   * chunk dentro lo stesso totale, e `perToken(8)` pubblicherebbe un valore 64
   * volte piu' grande dei byte per token reali. Chi misura un chunk solo passa
   * l'M di quel chunk perche' li' i due numeri coincidono; chi misura il
   * prefill passa i token del prefill. Il meter non puo' dedurlo da solo (non
   * sa quali dispatch appartengano allo stesso chunk), quindi il numero e'
   * responsabilita' del chiamante — ed e' per questo che sta nella firma.
   */
  perToken(M: number): number;
  /** ripartizione per forma — serve alla COPERTURA: i pesi non-q4_0 restano legacy */
  byForm(): Record<PrefillGemmForm, number>;
  reset(): void;
}

export function createPrefillWeightMeter(): PrefillWeightMeter {
  // stato per ISTANZA (nessun globale): due meter possono contare due piani
  // alternativi nella stessa run senza contaminarsi
  let legacy = 0;
  let multirow = 0;
  return {
    add(o: PrefillDispatch): void {
      const b = dispatchWeightBytes(o);
      if (o.form === "legacy") legacy += b;
      else multirow += b;
    },
    totalBytes(): number {
      return legacy + multirow;
    },
    perToken(M: number): number {
      if (!Number.isInteger(M) || M < 1) throw new Error(`perToken: M non valido (${M})`);
      return (legacy + multirow) / M;
    },
    byForm(): Record<PrefillGemmForm, number> {
      return { legacy, multirow };   // COPIA: mutarla non falsifica il meter
    },
    reset(): void {
      legacy = 0;
      multirow = 0;
    },
  };
}
