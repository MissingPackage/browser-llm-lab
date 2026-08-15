// Casi e tolleranze del ktest per il PREFILL GEMM K-quant/q4_1 — riga 2
// (q5_K) e riga 3 (q4_1) di engine-kquant, puro CPU-side, sul modello di
// kquantfast.ts.
//
// Qui vivono i numeri che devono stare in UN posto solo: la SHAPE di ogni caso
// di conformita' (che altrimenti il ktest e il floor test scriverebbero due
// volte, descrivendo due esperimenti diversi sotto lo stesso nome) e le quattro
// tolleranze per formato con cui il ktest giudica le due vie del
// moltiplicatore.
//
// I kernel gemelli sono `prefillGemmQ5KSplitKIdotWgsl` /
// `prefillGemmQ41SplitKIdotWgsl` (via intera `dot4I8Packed`) e
// `prefillGemmQ5KSplitKWgsl` / `prefillGemmQ41SplitKWgsl` (fallback f32) in
// kernels/wgsl.ts; `tests/engine-prefillkquant-f32floor.test.ts` genera quei
// kernel VERI, ne asserisce le costanti e RI-DERIVA da un'emulazione f32 i
// pavimenti che le tolleranze qui sotto citano — e fallisce se una tolleranza
// scende sotto il proprio pavimento o sale sopra 20 volte il pavimento.
//
// I DUE BLOCCHI NON CONDIVIDONO UN NUMERO, e non e' una convenzione estetica:
// il pavimento RELATIVO del q4_1 sta ~39x SOPRA quello del q5_K, e quello
// ASSOLUTO ~20x SOTTO. Due formati, due geometrie (blocchi da 32 contro
// superblocchi da 256) e due ordini di grandezza d'uscita (|y| medio 1,93e2
// contro 5,37e3): non esiste un fattore di conversione fra le due famiglie, e
// ogni tolleranza si legge dal proprio pavimento. Il floor test asserisce che i
// pavimenti dei due formati sono DIVERSI confrontando le variabili misurate,
// non i letterali scritti qui.

/**
 * SHAPE E SEED DEL CASO KTEST, in una sede sola.
 *
 * K=4096 e' la K VERA di `blk.*.ssm_out` del 4B — la shape su cui la riga 2 ha
 * misurato il suo 28,10x — quindi il caso di CONFORMITA' esercita la stessa
 * geometria del caso di VELOCITA': 16 superblocchi per riga, che
 * `prefillGemmSplitsFor` divide in 4 fette da 4 (`const PER = 4u`).
 *
 * N=200 e' ridotta rispetto ai 2560 veri, come negli altri casi ktest: il costo
 * del riferimento f64 cresce con N e la conformita' non e' una misura di
 * velocita'. Ma N%64 != 0 e' VOLUTO e non e' un ripiego: il moltiplicatore
 * lancia workgroup da 64 righe di uscita, quindi con N=200 l'ultimo workgroup ne
 * ha 56 oltre il bordo — e' l'unico modo perche' la guardia `r < N_ROWS` sia
 * ESERCITATA invece che creduta.
 *
 * M=16 e' la riga di chunk del prefill (`M_MAX` del wiring).
 *
 * I due seed sono separati perche' separati sono i due generatori: i byte dei
 * superblocchi (`randBytes(seedBlocks)`, scale f16 sanate a offset [1,3] con
 * `0x2c | (b & 0x03)`, la stessa preparazione di testGemvC2/fixScalesAt) e le
 * attivazioni (`randF32(seedX)` su M x K row-major).
 */
export const PREFILL_Q5K_KTEST_CASE = {
  K: 4096, N: 200, M: 16, splits: 4, seedBlocks: 5501, seedX: 5502,
} as const;

/**
 * Tolleranza RELATIVA del ktest per la via intera q5_K (`dot4I8Packed`).
 *
 * DERIVAZIONE (tests/engine-prefillkquant-f32floor.test.ts, che la importa da
 * qui e ri-misura il pavimento a ogni run): l'emulazione f32 del kernel su
 * `PREFILL_Q5K_KTEST_CASE` da' 4,382e-7 SENZA contrazione FMA e 4,382e-7 CON. Il
 * pavimento e' l'INVILUPPO dei due, 4,382e-7, e non il valore di un modello
 * solo: la spec WGSL PERMETTE di fondere `a*b + c` ma non lo impone, quindi due
 * device conformi danno due risultati diversi ed entrambi vanno coperti. E' la
 * lezione gia' pagata sul Q6_K fast, dove la vecchia tolleranza 2e-4 reggeva
 * SOLO perche' il device di sviluppo fondeva. 5e-6 sta 11,4x sopra.
 *
 * PERCHE' COSI' BASSO rispetto ai GEMV fast (1e-3): qui il prodotto scalare del
 * ciclo interno e' INTERO ed ESATTO (pesi a 5 bit x attivazioni i8, accumulo
 * i32). L'unico errore f32 sta nell'accumulo dei 128 termini per riga e nella
 * somma delle 4 fette — manca la cancellazione elemento per elemento che domina
 * il pavimento della famiglia fast.
 */
export const PREFILL_GEMM_Q5K_IDOT_REL_TOL = 5e-6;

/**
 * Tolleranza ASSOLUTA del ktest per la via intera q5_K. Serve perche' la
 * `compare` del ktest passa su `maxAbs <= absTol OPPURE maxRel <= relTol`: senza
 * il ramo assoluto un'uscita quasi nulla (denominatore `max(|ref|, 1e-6)`)
 * farebbe fallire un kernel corretto.
 *
 * DERIVAZIONE: pavimento 1,491e-3 SENZA contrazione FMA e 1,914e-3 CON —
 * inviluppo 1,914e-3, su uscite di modulo ~3e3. Qui la contrazione PEGGIORA il
 * caso peggiore, il contrario di quel che si assume: sull'accumulo per fetta
 * l'FMA sposta il rounding, non lo riduce. 2e-2 sta 10,4x sopra.
 */
export const PREFILL_GEMM_Q5K_IDOT_ABS_TOL = 2e-2;

/**
 * Tolleranza RELATIVA del ktest per il fallback f32 q5_K.
 *
 * DERIVAZIONE: pavimento 6,446e-7 SENZA contrazione FMA e 3,087e-7 CON —
 * inviluppo 6,446e-7, cioe' il modello SENZA contrazione. Su questa via i due
 * modelli si scambiano il posto fra le due metriche (sul REL vince il senza-FMA,
 * sull'ABS il con-FMA): e' la dimostrazione concreta che «il device di sviluppo
 * fonde» non basta a tarare una tolleranza. 8e-6 sta 12,4x sopra.
 *
 * E' PIU' ALTO della via intera, e la ragione e' strutturale: qui il prodotto
 * `q[l] * x[l]` avviene in virgola mobile per ogni elemento, mentre la via
 * intera lo fa in i32 esatto. Le due vie calcolano la stessa cosa ma NON con lo
 * stesso errore, e una tolleranza sola per entrambe sarebbe o troppo larga per
 * una o troppo stretta per l'altra.
 */
export const PREFILL_GEMM_Q5K_F32_REL_TOL = 8e-6;

/**
 * Tolleranza ASSOLUTA del ktest per il fallback f32 q5_K.
 *
 * DERIVAZIONE: pavimento 1,871e-3 SENZA contrazione FMA e 2,279e-3 CON —
 * inviluppo 2,279e-3. 3e-2 sta 13,2x sopra.
 */
export const PREFILL_GEMM_Q5K_F32_ABS_TOL = 3e-2;

// ---------------------------------------------------------------------------
// q4_1 — riga 3 di engine-kquant. Stessa struttura del blocco q5_K qui sopra e
// NESSUN numero in comune con esso: i kernel gemelli sono
// `prefillGemmQ41SplitKIdotWgsl` / `prefillGemmQ41SplitKWgsl`, e lo stesso
// floor test che deriva le quattro tolleranze q5_K deriva anche queste, sul
// proprio caso e con i propri seed.
// ---------------------------------------------------------------------------

/**
 * SHAPE E SEED DEL CASO KTEST q4_1, in una sede sola.
 *
 * K=9216 e' la K VERA di `blk.0-3.ffn_down` del 4B — la shape su cui la riga 3
 * ha misurato il suo guadagno, e il 71% dei byte del segmento `gemm:ffn-down`.
 * Il q4_1 e' un formato a BLOCCHI da 32, non a superblocchi: 9216/32 = 288
 * blocchi per riga, che `prefillGemmSplitsFor(K, N, "q4_1")` divide in 4 fette
 * da PER=72 (`const BPR = 288u;` / `const PER = 72u;` nei due kernel). E' una
 * geometria diversa da quella del caso q5_K — 16 superblocchi in 4 fette da 4 —
 * e la differenza si vede nel pavimento: la catena di somme f32 dentro una
 * fetta e' 18 volte piu' lunga in unita'.
 *
 * N=200 e' ridotta rispetto ai 2560 veri, come negli altri casi ktest: il costo
 * del riferimento f64 cresce con N e la conformita' non e' una misura di
 * velocita'. Ma N%64 != 0 e' VOLUTO e non e' un ripiego: il moltiplicatore
 * lancia workgroup da 64 righe di uscita, quindi con N=200 l'ultimo workgroup ne
 * ha 56 oltre il bordo — e' l'unico modo perche' la guardia `r < N_ROWS` sia
 * ESERCITATA invece che creduta.
 *
 * M=16 e' la riga di chunk del prefill (`M_MAX` del wiring).
 *
 * I due seed sono separati perche' separati sono i due generatori: i byte dei
 * blocchi (`randBytes(nBlocks * Q4_1_BLOCK_BYTES, seedBlocks)`, con le DUE scale
 * f16 `d` e `m` sanate a offset [1,3] da `fixScalesAt`) e le attivazioni
 * (`randF32(seedX)` su M x K row-major). E sono DIVERSI dai due del caso q5_K:
 * quattro seed distinti in tutto, cosi' i due esperimenti non condividono un
 * flusso e i due pavimenti non possono coincidere per costruzione.
 */
export const PREFILL_Q41_KTEST_CASE = {
  K: 9216, N: 200, M: 16, splits: 4, seedBlocks: 5511, seedX: 5512,
} as const;

/**
 * Tolleranza RELATIVA del ktest per la via intera q4_1 (`dot4I8Packed`).
 *
 * DERIVAZIONE (tests/engine-prefillkquant-f32floor.test.ts, che la importa da
 * qui e ri-misura il pavimento a ogni run): l'emulazione f32 del kernel su
 * `PREFILL_Q41_KTEST_CASE` da' 1,462e-5 SENZA contrazione FMA e 1,693e-5 CON.
 * Il pavimento e' l'INVILUPPO dei due, 1,693e-5, e non il valore di un modello
 * solo: la spec WGSL PERMETTE di fondere `a*b + c` ma non lo impone, quindi due
 * device conformi danno due risultati diversi ed entrambi vanno coperti. 2e-4
 * sta 11,8x sopra.
 *
 * PERCHE' NON E' IL NUMERO DEL q5_K: il pavimento relativo omologo li' e'
 * 4,382e-7, 39 volte piu' basso, e non e' lo stesso fenomeno visto peggio —
 * sono due misure su due geometrie (72 blocchi da 32 per fetta contro 4
 * superblocchi da 256) e su uscite di modulo diverso (|y| medio 1,93e2 qui,
 * 5,37e3 li'). Ricopiare il 5e-6 del q5_K avrebbe dichiarato una banda 3,4
 * volte SOTTO il pavimento misurato, e il ktest avrebbe bocciato un kernel
 * corretto. Il floor test lo impedisce confrontando i pavimenti fra loro, non i
 * letterali.
 */
export const PREFILL_GEMM_Q41_IDOT_REL_TOL = 2e-4;

/**
 * Tolleranza ASSOLUTA del ktest per la via intera q4_1. Serve perche' la
 * `compare` del ktest passa su `maxAbs <= absTol OPPURE maxRel <= relTol`: senza
 * il ramo assoluto un'uscita quasi nulla (denominatore `max(|ref|, 1e-6)`)
 * farebbe fallire un kernel corretto.
 *
 * DERIVAZIONE: pavimento 9,708e-5 SENZA contrazione FMA e 8,273e-5 CON —
 * inviluppo 9,708e-5, su uscite di modulo 1,93e2 in media (3,09e2 al massimo) —
 * due ordini di grandezza sotto le uscite del caso q5_K. Sulla metrica ASSOLUTA il
 * modello senza contrazione sta SOPRA, sulla relativa sotto: i due modelli si
 * scambiano il posto fra le due metriche, ed e' per questo che il pavimento va
 * preso su ognuna separatamente. 1e-3 sta 10,3x sopra.
 */
export const PREFILL_GEMM_Q41_IDOT_ABS_TOL = 1e-3;

/**
 * Tolleranza RELATIVA del ktest per il fallback f32 q4_1.
 *
 * DERIVAZIONE: pavimento 1,489e-5 SENZA contrazione FMA e 1,715e-5 CON —
 * inviluppo 1,715e-5. 2e-4 sta 11,7x sopra.
 *
 * E' LO STESSO NUMERO DICHIARATO PER LA VIA INTERA, e non perche' sia stato
 * ricopiato: i due pavimenti MISURATI distano l'1,3% (1,693e-5 sulla via intera
 * contro 1,715e-5 qui), quindi lo stesso valore sta nella banda [pavimento,
 * 20x] di entrambe. Sul q5_K le due vie divergevano di 1,47x e infatti li' le
 * due tolleranze sono diverse; qui la misura dice che cio' che distingue le due
 * vie — prodotto per elemento intero ESATTO contro prodotto in virgola mobile —
 * non emerge sopra l'errore che le due CONDIVIDONO, cioe' l'accumulo f32 lungo
 * i 72 blocchi della fetta piu' la somma delle 4 fette. Se un domani i due
 * pavimenti divergessero, il floor test lo direbbe con la coppia di asserzioni
 * a 20x prima che le due tolleranze restino uguali per inerzia.
 */
export const PREFILL_GEMM_Q41_F32_REL_TOL = 2e-4;

/**
 * Tolleranza ASSOLUTA del ktest per il fallback f32 q4_1.
 *
 * DERIVAZIONE: pavimento 9,120e-5 SENZA contrazione FMA e 8,666e-5 CON —
 * inviluppo 9,120e-5. 1e-3 sta 11,0x sopra. Stesso valore dichiarato della via
 * intera per la stessa ragione MISURATA della tolleranza relativa qui sopra: i
 * due pavimenti assoluti distano il 6,4% (9,708e-5 contro 9,120e-5) e la banda
 * a 20x li copre entrambi.
 */
export const PREFILL_GEMM_Q41_F32_ABS_TOL = 1e-3;
