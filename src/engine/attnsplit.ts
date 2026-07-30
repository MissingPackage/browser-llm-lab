// Piano dell'attention split sul contesto (spec B2 §Attention split) — puro
// CPU-side, testabile in CI senza GPU (convenzione fase A). La stessa aritmetica
// vive nei kernel WGSL (attnSplitPartWgsl/attnSplitReduceWgsl): qui è la verità
// testabile per sizing dei buffer e per le unit sul piano.

export const ATTN_CHUNK_P = 64; // posizioni per partizione (1 posizione/thread)

// partizioni della griglia fissa per un ctxMax dato (S_MAX del kernel)
export const attnSMax = (ctxMax: number): number => Math.ceil(ctxMax / ATTN_CHUNK_P);

// f32 per il buffer partials: [nHead × sMax × (headDim + 2)] (out | m | l)
export const attnPartialsLen = (nHead: number, headDim: number, ctxMax: number): number =>
  nHead * attnSMax(ctxMax) * (headDim + 2);

// partizioni ATTIVE per una posizione corrente (n = pos+1 ⇒ floor(pos/CHUNK)+1);
// il kernel reduce usa la stessa formula (nParts = P.pos / CHUNK + 1)
export const attnActiveParts = (pos: number): number => Math.floor(pos / ATTN_CHUNK_P) + 1;

// partizione owner del token corrente (fa rope k_cur + append in cache)
export const attnOwnerPart = (pos: number): number => Math.floor(pos / ATTN_CHUNK_P);

// range [begin, end) delle posizioni coperte dalla partizione part a contesto n
export const attnPartRange = (part: number, n: number): { begin: number; end: number } => {
  const begin = part * ATTN_CHUNK_P;
  return { begin, end: Math.min(begin + ATTN_CHUNK_P, n) };
};

// Riduzione log-sum-exp del pass 2 (riferimento JS per le unit, spec §Unit):
// combina partial {m, l, out[]} in una softmax esatta — proprietà testata:
// equivale alla softmax monolitica sugli score concatenati.
export const lseReduce = (
  parts: { m: number; l: number; out: Float64Array }[],
): Float64Array => {
  if (parts.length < 1) throw new Error("lseReduce: nessuna partizione");
  const dim = parts[0].out.length;
  let M = -Infinity;
  for (const p of parts) M = Math.max(M, p.m);
  let L = 0;
  for (const p of parts) L += p.l * Math.exp(p.m - M);
  const out = new Float64Array(dim);
  for (const p of parts) {
    const w = Math.exp(p.m - M);
    for (let i = 0; i < dim; i++) out[i] += p.out[i] * w;
  }
  for (let i = 0; i < dim; i++) out[i] /= L;
  return out;
};
