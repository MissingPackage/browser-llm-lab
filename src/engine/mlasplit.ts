// Piano dell'attention MLA split sul contesto (C3a fase 4c) — puro CPU-side,
// testabile in CI senza GPU (stessa convenzione di attnsplit.ts, che fa lo
// stesso lavoro per il path Qwen). Qui vive la verità testabile per il sizing
// dei buffer; l'aritmetica gemella sta nei kernel WGSL
// (mlaAttnSplitPartWgsl/mlaAttnSplitReduceWgsl).
//
// PERCHÉ UN FILE SEPARATO DA attnsplit.ts. Le due geometrie non coincidono e
// non devono: lo split Qwen è "1 posizione per thread" (CHUNK=64 su workgroup
// da 64, una head per workgroup), quello MLA è "un chunk di cache per TUTTE le
// head" (la cache MLA è condivisa fra le 20 head: rileggerla 20 volte era il
// 74,5% del tempo GPU per token, misura it.11). Fondere i due moduli
// significherebbe una costante CHUNK che vuol dire due cose diverse.

// Posizioni di cache per partizione. La scelta è di OCCUPANCY, non di banda: al
// bench (pos ~525) servono abbastanza workgroup per riempire la GPU — con 16
// sono 33 partizioni attive, con 64 sarebbero 9 (meno dei 20 workgroup del
// kernel monolitico che stiamo sostituendo, cioè lo stesso difetto).
export const MLA_CHUNK_P = 16;

// Larghezza del tile (in f32) con cui il pass 1 scorre sia i 576 della fase
// score sia i 512 della fase output. Il tile in shared è memorizzato a stride
// PADDATO (+1 f32): senza il padding le letture per posizione cadono tutte
// nello stesso banco (stride 64 f32 = multiplo di 32 banchi).
export const MLA_TILE_W = 64;
export const MLA_TILE_STRIDE = MLA_TILE_W + 1;

// partizioni della griglia fissa per un ctxMax dato (S_MAX del kernel)
export const mlaSMax = (ctxMax: number): number => Math.ceil(ctxMax / MLA_CHUNK_P);

// f32 per il buffer partials: [nHead × sMax × (kvLora + 2)] (out | m | l) —
// stessa convenzione di layout dello split Qwen, così il pass 2 ha la stessa
// aritmetica log-sum-exp.
export const mlaPartialsLen = (nHead: number, kvLora: number, ctxMax: number): number =>
  nHead * mlaSMax(ctxMax) * (kvLora + 2);

// partizioni ATTIVE per una posizione corrente (n = pos+1 ⇒ floor(pos/CHUNK)+1);
// il kernel reduce usa la stessa formula (nParts = P.nPast / CHUNK + 1)
export const mlaActiveParts = (pos: number): number => Math.floor(pos / MLA_CHUNK_P) + 1;

// range [begin, end) delle posizioni coperte dalla partizione part a contesto n
export const mlaPartRange = (part: number, n: number): { begin: number; end: number } => {
  const begin = part * MLA_CHUNK_P;
  return { begin, end: Math.min(begin + MLA_CHUNK_P, n) };
};

/**
 * Workgroup storage del PASS 1, in byte. È il fabbisogno del `var<workgroup>`
 * dichiarato da `mlaAttnSplitPartWgsl`:
 *   tile di cache [CHUNK × TILE_STRIDE] + tile di q [nHead × TILE_STRIDE]
 *   + score/exp [nHead × CHUNK] + m[nHead] + l[nHead]
 * A nHead 20 fa 10 800 B, sotto il default di spec (16 KiB).
 */
export const mlaSplitPartWorkgroupStorageBytes = (nHead: number): number =>
  4 * (MLA_CHUNK_P * MLA_TILE_STRIDE + nHead * MLA_TILE_STRIDE + nHead * MLA_CHUNK_P + 2 * nHead);

/**
 * Workgroup storage del PASS 2: i due scalari della riduzione (M e L), niente
 * altro. Il pass 2 NON memoizza i pesi exp(m_p − M) in un array [sMax] apposta:
 * quell'array sarebbe shared che cresce col contesto (a ctxMax 65536 farebbe
 * 16 388 B, sopra il default di spec), cioè il difetto che lo split esiste per
 * togliere. Si paga un exp in più per (j, partizione) e si compra la
 * costanza in ctxMax.
 */
export const MLA_SPLIT_REDUCE_WORKGROUP_STORAGE_BYTES = 8;

/**
 * Il fabbisogno di workgroup storage dello split, cioè il MASSIMO fra i due
 * kernel che lo compongono (domina il pass 1). È questo che il consumatore deve
 * confrontare col limite del device: i due pass girano sullo stesso device e
 * basta che uno sfondi.
 *
 * DIFFERENZA SOSTANZIALE dal monolitico (`mlaWorkgroupStorageBytes` in
 * gpulimits.ts, 4·ctxMax+256): lì gli score dell'INTERO contesto stavano in
 * shared, quindi il limite del device tagliava ctxMax (a 8128 sul default). Qui
 * il contesto è partizionato ed ENTRAMBI i pass hanno fabbisogno costante:
 * nessun vincolo su ctxMax. `tests/engine-mlasplit.test.ts` ri-deriva i numeri
 * scansionando il WGSL generato di tutti e due, così le formule non possono
 * scollarsi dai kernel — e la costanza si verifica su entrambi, non su metà.
 */
export const mlaSplitWorkgroupStorageBytes = (nHead: number): number =>
  Math.max(mlaSplitPartWorkgroupStorageBytes(nHead), MLA_SPLIT_REDUCE_WORKGROUP_STORAGE_BYTES);
