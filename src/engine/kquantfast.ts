// Sizing e tolleranze della famiglia FAST per i K-quant (C3a fase 4b it.13) —
// puro CPU-side, sul modello di mlasplit.ts. Qui vivono i numeri che devono
// stare in UN posto solo: il fabbisogno di workgroup storage dei due kernel
// (che glmmodel confronta col limite del device) e le tolleranze con cui il
// ktest li giudica (che il floor test ri-deriva da un'emulazione f32).
//
// I kernel gemelli sono `pairGemvSiluQ5KFastWgsl` e `gemvQ6KFastWgsl` in
// kernels/wgsl.ts; `tests/engine-kquant-f32floor.test.ts` scansiona il WGSL
// generato e verifica che le formule qui sotto non si siano scollate.

/** Thread per workgroup: un workgroup per riga di output. */
export const KQUANT_FAST_WG = 64;

/** Pesi per sottogruppo, cioè per thread e per giro (8 sottogruppi/superblocco). */
export const KQUANT_FAST_SUB = 32;

/**
 * f32 occupati da x in shared. Il `+K/32` è il PADDING: i sottogruppi distano
 * esattamente 32 f32, quindi senza di esso i 32 thread di un warp leggono
 * indirizzi tutti congrui mod 32 — stesso banco, conflitto a 32 vie
 * sull'accesso più caldo del kernel. Con l'indice `e + (e >> 5)` la distanza
 * diventa 33 e i banchi si separano.
 */
export const kquantFastXsLen = (K: number): number => K + K / KQUANT_FAST_SUB;

/**
 * Workgroup storage dei due kernel, in byte: x paddato + le riduzioni (due
 * array da 64 per il pair gate/up, uno solo per il gemv semplice).
 *
 * A DIFFERENZA dello split MLA questo fabbisogno CRESCE con K (8448 B di solo
 * x a K=2048): è per questo che entra nella guardia di glmmodel invece di
 * essere dato per scontato.
 */
export const pairGemvSiluQ5KFastWorkgroupStorageBytes = (K: number): number =>
  4 * (kquantFastXsLen(K) + 2 * KQUANT_FAST_WG);

export const gemvQ6KFastWorkgroupStorageBytes = (K: number): number =>
  4 * (kquantFastXsLen(K) + KQUANT_FAST_WG);

/**
 * Tolleranza relativa del ktest per il pair Q5_K + silu·mul.
 * DERIVAZIONE (tests/engine-kquant-f32floor.test.ts, che la importa da qui):
 * il pavimento aritmetico f32 di questo kernel è 3,432e-4 senza contrazione
 * FMA e 5,869e-4 con — e quest'ultimo è ESATTAMENTE il valore che il device
 * misura, a quattro cifre. La tolleranza sta 1,70× sopra il caso peggiore, e
 * ordini di grandezza sotto ciò che produrrebbe un bug strutturale (≥1,6e-2).
 */
export const KQUANT_FAST_Q5K_PAIR_REL_TOL = 1e-3;

/**
 * Tolleranza relativa del ktest per il Q6_K fast (down shexp e testa).
 * DERIVAZIONE: qui il caso peggiore è l'ASSENZA di contrazione — 2,413e-4
 * senza FMA contro 4,711e-5 con (il device di sviluppo fonde, ed è per questo
 * che a 2e-4 il ktest passava). La spec WGSL permette la contrazione ma non la
 * impone: a 2e-4 un driver conforme che non fonde darebbe FAIL su un kernel
 * corretto. 5e-4 sta 2,07× sopra il pavimento no-FMA, cioè sopra il caso
 * peggiore LECITO e non sopra quello osservato su una macchina sola.
 */
export const KQUANT_FAST_Q6K_REL_TOL = 5e-4;
