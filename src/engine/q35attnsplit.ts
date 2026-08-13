// Piano dello split streaming dell'attention Qwen (goal engine-kernel-decode) —
// puro CPU-side, testabile in CI senza GPU. Stessa convenzione di attnsplit.ts
// e mlasplit.ts: qui vive la verita' testabile del sizing e della geometria,
// l'aritmetica gemella sta nel kernel WGSL (`attnDecodeWgsl` non-batch +
// `attnDecodeCombineWgsl`), che IMPORTA queste funzioni invece di ricopiarle.
//
// PERCHE' UN TERZO FILE E NON attnsplit.ts. attnsplit.ts descrive lo split del
// path gpuforward (CHUNK=64, una posizione per thread, partials [out|m|l] a
// stride headDim+2). Questo descrive una geometria diversa e incompatibile: il
// contesto e' spezzato in POCHI chunk lunghi (>=512 posizioni), ciascuno con
// softmax in streaming a tile di 64, e i parziali stanno in DUE buffer separati
// (partOut vec4, partMS f32) perche' il kernel scrive gli out con store vec4
// coalescenti e gli {m,s} con un solo thread. Riusare ATTN_CHUNK_P vorrebbe
// dire una costante che significa due cose diverse.

/**
 * Posizioni per chunk, minimo. Multiplo di 64 (il tile del kernel: 64 thread,
 * una posizione per thread per iterazione). E' una scelta di OCCUPANCY contro
 * OVERHEAD: sotto le ~512 posizioni per chunk il combine e i parziali costano
 * piu' di quanto rendano i workgroup in piu'.
 */
export const Q35_ATTN_MIN_CHUNK = 512;

/**
 * Tetto ai chunk, cioe' alla dimensione Y del dispatch e all'altezza dei buffer
 * parziali. Oltre questo il chunk cresce e il numero di workgroup no: i buffer
 * `partOut`/`partMS` restano COSTANTI in ctxMax, che e' il punto dello split.
 */
export const Q35_ATTN_MAX_SPLITS = 64;

/**
 * Piano dello split per un ctxMax dato. Invarianti (unit [f]):
 *   chunkLen % 64 === 0        — il tile del kernel divide il chunk
 *   splits = ceil(ctxMax/chunkLen) <= Q35_ATTN_MAX_SPLITS
 *   splits * chunkLen >= ctxMax — nessuna posizione scoperta
 *
 * I chunk che cadono oltre `nPast+1` a runtime NON sono un errore: il kernel li
 * esegue col loop vuoto e scrive m = -3.0e38, s = 0, acc = 0, che il combine
 * annulla (exp(-3e38 - gm) = 0). La griglia e' fissa in ctxMax, il lavoro no.
 */
export function q35AttnSplitPlan(ctxMax: number): { splits: number; chunkLen: number } {
  if (!Number.isInteger(ctxMax) || ctxMax < 1) {
    throw new Error(`q35AttnSplitPlan: ctxMax ${ctxMax} non e' un intero positivo`);
  }
  // chunk minimo che tiene splits sotto il tetto, arrotondato al tile
  const perSplit = Math.ceil(Math.ceil(ctxMax / Q35_ATTN_MAX_SPLITS) / 64) * 64;
  const chunkLen = Math.max(Q35_ATTN_MIN_CHUNK, perSplit);
  return { splits: Math.ceil(ctxMax / chunkLen), chunkLen };
}

/**
 * f32 dei due buffer parziali del pass 1:
 *   partOut = [nHead x splits x headDim]  (l'accumulo NON normalizzato, letto vec4)
 *   partMS  = [nHead x splits x 2]        ({m, s} per chunk)
 * Sono separati perche' il kernel li scrive con pattern diversi: partOut da
 * tutti i thread con store vec4 coalescenti, partMS dal solo thread 0.
 */
export function q35AttnPartialsFloats(
  o: { nHead: number; headDim: number; ctxMax: number },
): { out: number; ms: number } {
  const { splits } = q35AttnSplitPlan(o.ctxMax);
  return { out: o.nHead * splits * o.headDim, ms: o.nHead * splits * 2 };
}

/**
 * Riferimento JS della combinazione log-sum-exp (stessa forma di `lseReduce` in
 * attnsplit.ts, con `s` al posto di `l` perche' e' il nome che il kernel usa in
 * partMS). Proprieta' testata [g]: equivale alla softmax MONOLITICA sugli score
 * concatenati, entro 1e-12 in f64.
 *
 * `out` di ciascuna parte e' Sum_p exp(score_p - m)*v_p, `s` e' Sum_p exp(score_p - m):
 * esattamente cio' che il pass 1 lascia in partOut/partMS.
 */
export function q35AttnLseReduce(parts: { m: number; s: number; out: Float64Array }[]): Float64Array {
  if (parts.length < 1) throw new Error("q35AttnLseReduce: nessun chunk");
  const dim = parts[0].out.length;
  let gm = -Infinity;
  for (const p of parts) gm = Math.max(gm, p.m);
  let den = 0;
  const num = new Float64Array(dim);
  for (const p of parts) {
    const w = Math.exp(p.m - gm);
    den += p.s * w;
    for (let i = 0; i < dim; i++) num[i] += p.out[i] * w;
  }
  for (let i = 0; i < dim; i++) num[i] /= den;
  return num;
}
