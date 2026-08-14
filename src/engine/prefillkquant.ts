// Caso e tolleranze del ktest per il PREFILL GEMM q5_K (riga 2 di
// engine-kquant) — puro CPU-side, sul modello di kquantfast.ts.
//
// Qui vivono i numeri che devono stare in UN posto solo: la SHAPE del caso di
// conformita' (che altrimenti il ktest e il floor test scriverebbero due volte,
// descrivendo due esperimenti diversi sotto lo stesso nome) e le quattro
// tolleranze con cui il ktest giudica le due vie del moltiplicatore.
//
// I kernel gemelli sono `prefillGemmQ5KSplitKIdotWgsl` (via intera
// `dot4I8Packed`) e `prefillGemmQ5KSplitKWgsl` (fallback f32) in
// kernels/wgsl.ts; `tests/engine-prefillkquant-f32floor.test.ts` genera quei
// kernel VERI, ne asserisce le costanti e RI-DERIVA da un'emulazione f32 i
// pavimenti che le tolleranze qui sotto citano — e fallisce se una tolleranza
// scende sotto il proprio pavimento o sale sopra 20 volte il pavimento.

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
