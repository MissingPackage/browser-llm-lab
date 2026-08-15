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

// ===========================================================================
// LE TRE FORME DELLA RIGA 4 — q4_K, q6_K, q8_0.
//
// Sono le famiglie del 35B, portate dal banco e NON CABLATE (`wired: false` in
// `PREFILL_GEMM_SPEC`): il piano non ci instrada un dispatch, e il 4B esegue
// esattamente quello di ieri. Quello che i tre blocchi qui sotto consegnano non
// e' un cablaggio ma la fetta verticale che nessun test di aritmetica puo'
// dedurre — «quel kernel gira su una GPU vera e da' i numeri giusti» — nella
// stessa forma con cui le righe 2 e 3 l'hanno consegnata per q5_K e q4_1.
//
// La regola che ha retto le due righe precedenti vale identica qui: LE
// TOLLERANZE SI DERIVANO DAL PAVIMENTO, e il pavimento lo misura
// `tests/engine-prefillkquant-f32floor.test.ts` sul CASO dichiarato in questo
// file, con le DUE emulazioni di contrazione FMA. Nessuno dei dodici numeri
// qui sotto e' stato scelto: ognuno cita il pavimento da cui esce e quante
// volte gli sta sopra, e il floor test fallisce se scende sotto il pavimento o
// sale oltre 20 volte.
//
// SEI SEED NUOVI, tutti distinti dai quattro delle righe 2 e 3: dieci in tutto.
// Non e' pignoleria — due casi che condividono un flusso di byte hanno
// pavimenti che possono coincidere per costruzione, e l'anti-ricopiatura del
// floor test confronta VARIABILI misurate.
// ===========================================================================

/**
 * SHAPE E SEED DEL CASO KTEST q4_K.
 *
 * K=2048 e' la K VERA di `ffn_gate/up_exps` del 35B — la prima delle due shape
 * q4_K misurate in fase 0 (`[2048,512]`, 4,16x sulla legacy via intera) — e non
 * la seconda (`[512,2048]`): a K=2048 il superblocco da 256 sta 8 volte per
 * riga e `prefillGemmSplitsFor` le divide in 4 fette da PER=2, cioe' le QUATTRO
 * fette misurate. La `[512,2048]` avrebbe dato 2 superblocchi per riga e
 * splits=2, che e' il ramo di ripiego dei K-quant: quel ramo lo esercita il
 * caso q6_K qui sotto, che ha K=512 per davvero, e averli entrambi in due casi
 * diversi vale piu' che averli due volte nello stesso.
 *
 * N=200 e' ridotta rispetto ai 512 veri, come in tutti i casi ktest: il costo
 * del riferimento f64 cresce con N e la conformita' non e' una misura di
 * velocita'. N%64 != 0 e' VOLUTO: il moltiplicatore lancia workgroup da 64
 * righe di uscita, quindi con N=200 l'ultimo ne ha 56 oltre il bordo — e' l'unico
 * modo perche' la guardia `r < N_ROWS` sia ESERCITATA invece che creduta.
 *
 * M=16 e' la riga di chunk del prefill (`M_MAX` del wiring).
 *
 * I due seed sono separati perche' separati sono i due generatori: i byte dei
 * superblocchi da 144 B (`randBytes(seedBlocks)`, con `d` e `dmin` f16 sanate a
 * offset [1,3] da `fixScalesAt`, la stessa preparazione del q5_K perche' e' lo
 * stesso header) e le attivazioni (`randF32(seedX)` su M x K row-major).
 */
export const PREFILL_Q4K_KTEST_CASE = {
  K: 2048, N: 200, M: 16, splits: 4, seedBlocks: 5521, seedX: 5522,
} as const;

/**
 * Tolleranza RELATIVA del ktest per la via intera q4_K (`dot4I8Packed`).
 *
 * DERIVAZIONE (tests/engine-prefillkquant-f32floor.test.ts, che la importa da
 * qui e ri-misura il pavimento a ogni run): l'emulazione f32 del kernel su
 * `PREFILL_Q4K_KTEST_CASE` da' 7,841e-5 SENZA contrazione FMA e 3,851e-5 CON.
 * Il pavimento e' l'INVILUPPO dei due, 7,841e-5, e non il valore di un modello
 * solo: la spec WGSL PERMETTE di fondere `a*b + c` ma non lo impone, quindi due
 * device conformi danno due risultati diversi ed entrambi vanno coperti. 1e-3
 * sta 12,8x sopra.
 *
 * QUI I DUE MODELLI DISTANO 2,0x, il divario piu' largo delle sei vie di questo
 * file: sul q4_K la contrazione MIGLIORA il caso peggiore relativo di un fattore
 * due. Tarare sul device di sviluppo — quello che fonde — avrebbe dichiarato una
 * banda che un driver conforme che NON fonde sfonda subito.
 */
export const PREFILL_GEMM_Q4K_IDOT_REL_TOL = 1e-3;

/**
 * Tolleranza ASSOLUTA del ktest per la via intera q4_K. Serve perche' la
 * `compare` del ktest passa su `maxAbs <= absTol OPPURE maxRel <= relTol`: senza
 * il ramo assoluto un'uscita quasi nulla (denominatore `max(|ref|, 1e-6)`)
 * farebbe fallire un kernel corretto.
 *
 * DERIVAZIONE: pavimento 4,191e-4 SENZA contrazione FMA e 6,216e-4 CON —
 * inviluppo 6,216e-4, su uscite di modulo 1,049e3 in media. Sulla metrica
 * ASSOLUTA i due modelli si scambiano il posto rispetto alla relativa (li' vince
 * il senza-FMA, qui il con-FMA): e' per questo che il pavimento va preso su ogni
 * metrica separatamente. 8e-3 sta 12,9x sopra.
 */
export const PREFILL_GEMM_Q4K_IDOT_ABS_TOL = 8e-3;

/**
 * Tolleranza RELATIVA del ktest per il fallback f32 q4_K.
 *
 * DERIVAZIONE: pavimento 4,984e-5 SENZA contrazione FMA e 6,266e-5 CON —
 * inviluppo 6,266e-5. 8e-4 sta 12,8x sopra.
 *
 * E' PIU' BASSA della via intera (6,266e-5 contro 7,841e-5), ed e' l'ORDINE
 * OPPOSTO a quello del q5_K, dove il fallback f32 sta sopra la via intera e il
 * floor test lo asserisce. Non e' una misura sospetta: i due bracci del q4_K
 * producono uscite dello stesso modulo (|y| medio 1,049e3 e 1,050e3, misurati
 * accanto ai pavimenti), quindi il salto non viene dalla scala del risultato ma
 * dall'aritmetica — con nibble a 4 bit i due addendi `d*sc*Sigma(q x)` e
 * `dmin*mn*Sigma(x)` sono piu' vicini in modulo che a 5 bit, e la cancellazione
 * fra loro pesa piu' del prodotto intero esatto che la via idot guadagna. Il
 * PERCHE' esatto non e' stato isolato con un esperimento; il FATTO e' misurato,
 * e basta a dire che ogni tolleranza si legge dal proprio pavimento e non
 * dall'ordine che aveva su un altro formato.
 */
export const PREFILL_GEMM_Q4K_F32_REL_TOL = 8e-4;

/**
 * Tolleranza ASSOLUTA del ktest per il fallback f32 q4_K.
 *
 * DERIVAZIONE: pavimento 5,991e-4 SENZA contrazione FMA e 5,991e-4 CON —
 * inviluppo 5,991e-4. E' l'unica delle dodici vie in cui i due modelli danno lo
 * STESSO caso peggiore assoluto: qui il massimo cade su un'uscita in cui la
 * contrazione non cambia nessuno degli arrotondamenti decisivi. Resta un
 * inviluppo di due misure, non una misura sola. 7e-3 sta 11,7x sopra.
 */
export const PREFILL_GEMM_Q4K_F32_ABS_TOL = 7e-3;

/**
 * SHAPE E SEED DEL CASO KTEST q6_K.
 *
 * K=512 e' la K VERA — e l'unica misurata in fase 0 per questo formato:
 * `[512,2048]`, la down degli expert di 3 layer del 35B, 6,13x sulla legacy via
 * intera. Non e' ridotta e non poteva esserlo: 512 pesi sono DUE superblocchi da
 * 256 per riga, e `prefillGemmSplitsFor` li divide in 2 fette da PER=1 — cioe'
 * il RAMO A DUE FETTE della scala K-quant, che nessun altro caso di questo file
 * esercita (q5_K e q4_K stanno entrambi sul ramo a 4). Accorciare K non era
 * possibile e allungarla avrebbe descritto una shape che il 35B non ha.
 *
 * PER=1 non e' un caso degenere: e' la fetta piu' corta che il piano possa
 * produrre, quindi qui la catena di somme f32 dentro una fetta e' la piu' breve
 * delle sei vie e il peso relativo della `prefillSplitKCombineWgsl` — la somma
 * delle DUE fette — e' il piu' alto. E' un pavimento diverso per costruzione, e
 * i numeri lo dicono.
 *
 * N=200 e' ridotta rispetto ai 2048 veri, come in tutti i casi ktest: il costo
 * del riferimento f64 cresce con N. N%64 != 0 e' VOLUTO: e' l'unico modo perche'
 * la guardia `r < N_ROWS` sia ESERCITATA invece che creduta.
 *
 * M=16 e' la riga di chunk del prefill (`M_MAX` del wiring).
 *
 * I due seed sono separati perche' separati sono i due generatori. I byte dei
 * superblocchi da 210 B si sanano a offset [209] e NON a [1,3] come i due
 * K-quant precedenti: il Q6_K tiene la sua unica scala f16 IN CODA, e le 16
 * scale per sotto-blocco sono int8 — byte casuali vanno benissimo per quelle,
 * mentre un esponente f16 casuale su `d` darebbe NaN/Inf.
 */
export const PREFILL_Q6K_KTEST_CASE = {
  K: 512, N: 200, M: 16, splits: 2, seedBlocks: 5531, seedX: 5532,
} as const;

/**
 * Tolleranza RELATIVA del ktest per la via intera q6_K (`dot4I8Packed`).
 *
 * DERIVAZIONE: pavimento 8,810e-5 SENZA contrazione FMA e 8,810e-5 CON —
 * inviluppo 8,810e-5. 1e-3 sta 11,3x sopra.
 *
 * E' LO STESSO VALORE DICHIARATO PER LA VIA INTERA DEL q4_K, e non perche' sia
 * stato ricopiato: i due pavimenti MISURATI distano il 12% (8,810e-5 qui contro
 * 7,841e-5 li'), quindi lo stesso numero sta nella banda [pavimento, 20x] di
 * entrambi. Sono comunque due misure su due formati che non condividono niente
 * dell'aritmetica — scale int8 per sotto-blocco da 16 e offset -32 qui, scale a
 * 6 bit per sotto-blocco da 32 e minimo f16 li' — e se un domani divergessero,
 * la coppia di asserzioni a 20x lo direbbe prima che restino uguali per inerzia.
 */
export const PREFILL_GEMM_Q6K_IDOT_REL_TOL = 1e-3;

/**
 * Tolleranza ASSOLUTA del ktest per la via intera q6_K. Serve perche' la
 * `compare` del ktest passa su `maxAbs <= absTol OPPURE maxRel <= relTol`.
 *
 * DERIVAZIONE: pavimento 8,006e-4 SENZA contrazione FMA e 5,929e-4 CON —
 * inviluppo 8,006e-4, su uscite di modulo 1,340e3 in media. Qui la contrazione
 * MIGLIORA il caso peggiore assoluto di 1,35x mentre lascia identico quello
 * relativo: due metriche, due comportamenti, e il pavimento va preso su ognuna
 * separatamente. 1e-2 sta 12,5x sopra.
 */
export const PREFILL_GEMM_Q6K_IDOT_ABS_TOL = 1e-2;

/**
 * Tolleranza RELATIVA del ktest per il fallback f32 q6_K.
 *
 * DERIVAZIONE: pavimento 1,118e-2 SENZA contrazione FMA e 5,626e-3 CON —
 * inviluppo 1,118e-2. 1,5e-1 sta 13,4x sopra.
 *
 * E' DUE ORDINI SOPRA OGNI ALTRA TOLLERANZA RELATIVA DI QUESTO FILE, e la
 * ragione e' misurata, non congetturata: il caso peggiore relativo di questo
 * braccio cade sull'uscita di modulo piu' PICCOLO dell'intera griglia —
 * 4,395e-2 contro una media di 1,340e3, cinque ordini sotto — dove l'errore
 * ASSOLUTO vale 4,914e-4, cioe' dentro il pavimento assoluto di questa stessa
 * via. Non e' il kernel a essere impreciso: e' un denominatore quasi nullo. Il
 * braccio intero non ci cade perche' il suo riferimento e' un altro (le
 * attivazioni dopo il giro a i8) e la sua uscita piu' piccola vale 7,469e-1.
 *
 * E' PROPRIO IL CASO PER CUI LA `compare` USA L'OR: su questo formato e' il ramo
 * ASSOLUTO a portare il giudizio, e la banda relativa larga non e' compiacenza —
 * un kernel che sbagliasse il formato sfonda comunque entrambe (la mutazione
 * dell'offset -32 da' 7,838e4 sul relativo e 1,147e4 sull'assoluto, cinque e sei
 * ordini oltre le rispettive tolleranze). Il floor test asserisce la forma del
 * fenomeno — REL divergente di piu' di un ordine fra le due vie, ABS coincidente
 * entro il 20% — cosi' che se un domani a divergere fosse anche l'assoluto,
 * quello NON passerebbe per «e' la solita cancellazione».
 */
export const PREFILL_GEMM_Q6K_F32_REL_TOL = 1.5e-1;

/**
 * Tolleranza ASSOLUTA del ktest per il fallback f32 q6_K — la metrica che su
 * questo formato porta davvero il giudizio (v. la tolleranza relativa qui sopra).
 *
 * DERIVAZIONE: pavimento 8,753e-4 SENZA contrazione FMA e 8,799e-4 CON —
 * inviluppo 8,799e-4. 1e-2 sta 11,4x sopra, e sta il 10% sopra il pavimento
 * assoluto della via intera (8,006e-4): sull'ASSOLUTO le due vie del q6_K
 * misurano la stessa cosa, ed e' il relativo a separarle.
 */
export const PREFILL_GEMM_Q6K_F32_ABS_TOL = 1e-2;

/**
 * SHAPE E SEED DEL CASO KTEST q8_0.
 *
 * K=2048 e' la K VERA di `attn_q/k/v/output` del 35B — la sola shape q8_0
 * misurata in fase 0 (`[2048,4096]`, 35,20x sulla legacy via intera, il
 * guadagno piu' grande delle quattro shape della riga 4). Il q8_0 e' un formato
 * a BLOCCHI da 32: 2048/32 = 64 blocchi per riga, che `prefillGemmSplitsFor`
 * divide in 4 fette da PER=16.
 *
 * IL VINCOLO DURO NON E' SU K MA SU PER: il ciclo del kernel avanza `b0 = b0 +
 * 2u`, quindi le fette devono dividere i blocchi A COPPIE — PER=16 e' pari e la
 * fetta finisce dove comincia la successiva. Una PER dispari farebbe leggere
 * alla fetta un blocco della fetta dopo, e nessun controllo di forma se ne
 * accorgerebbe: e' `slices` in `PREFILL_GEMM_SPEC` a tenerlo, e la struttura del
 * floor test lo verifica sul testo generato.
 *
 * N=200 e' ridotta rispetto ai 4096 veri, come in tutti i casi ktest. N%64 != 0
 * e' VOLUTO: e' l'unico modo perche' la guardia `r < N_ROWS` sia ESERCITATA.
 *
 * M=16 e' la riga di chunk del prefill (`M_MAX` del wiring).
 *
 * I byte dei blocchi da 34 B si sanano a offset [1] — UNA scala f16 per blocco,
 * come il q4_0 — e non a [1,3] come i formati a due scale ne' a [209] come il
 * q6_K.
 */
export const PREFILL_Q80_KTEST_CASE = {
  K: 2048, N: 200, M: 16, splits: 4, seedBlocks: 5541, seedX: 5542,
} as const;

/**
 * Tolleranza RELATIVA del ktest per la via intera q8_0 (`dot4I8Packed`).
 *
 * DERIVAZIONE: pavimento 7,611e-4 SENZA contrazione FMA e 1,756e-4 CON —
 * inviluppo 7,611e-4. 1e-2 sta 13,1x sopra.
 *
 * QUI I DUE MODELLI DISTANO 4,3x, il divario piu' largo delle dodici vie di
 * questo file. E' la dimostrazione piu' netta che il pavimento non puo' essere
 * la misura di un device solo: tarare su una macchina che fonde avrebbe
 * dichiarato ~2e-3, e un driver conforme che NON fonde avrebbe dato FAIL su un
 * kernel corretto — la lezione gia' pagata sul Q6_K fast, qui con quattro volte
 * il margine di errore.
 */
export const PREFILL_GEMM_Q80_IDOT_REL_TOL = 1e-2;

/**
 * Tolleranza ASSOLUTA del ktest per la via intera q8_0. Serve perche' la
 * `compare` del ktest passa su `maxAbs <= absTol OPPURE maxRel <= relTol`.
 *
 * DERIVAZIONE: pavimento 1,089e-4 SENZA contrazione FMA e 6,908e-5 CON —
 * inviluppo 1,089e-4, su uscite di modulo 1,493e2 in media. Sono le uscite piu'
 * piccole dei cinque casi (contro 1,049e3 del q4_K e 1,340e3 del q6_K), e infatti
 * questo e' il pavimento assoluto piu' basso della riga 4: e' la ragione per cui
 * un pavimento assoluto non si trasporta MAI da un formato all'altro. 1e-3 sta
 * 9,2x sopra.
 */
export const PREFILL_GEMM_Q80_IDOT_ABS_TOL = 1e-3;

/**
 * Tolleranza RELATIVA del ktest per il fallback f32 q8_0.
 *
 * DERIVAZIONE: pavimento 1,859e-4 SENZA contrazione FMA e 1,678e-4 CON —
 * inviluppo 1,859e-4. 2e-3 sta 10,8x sopra.
 *
 * E' QUATTRO VOLTE PIU' BASSA della via intera (1,859e-4 contro 7,611e-4), che
 * e' il contrario di quel che ci si aspetta — sulla via intera il prodotto
 * elemento per elemento e' esatto. Non lo e' pero' cio' che le due vie
 * CONFRONTANO: i due bracci hanno due riferimenti diversi (attivazioni dopo il
 * giro a i8 contro attivazioni grezze), quindi i loro casi peggiori cadono su
 * uscite diverse, e su questo formato quella del braccio intero e' piu' vicina
 * allo zero. Il fatto e' misurato; l'ordine delle due vie NON e' una proprieta'
 * che si eredita, ed e' per questo che ogni tolleranza si legge dal proprio
 * pavimento.
 */
export const PREFILL_GEMM_Q80_F32_REL_TOL = 2e-3;

/**
 * Tolleranza ASSOLUTA del ktest per il fallback f32 q8_0.
 *
 * DERIVAZIONE: pavimento 1,447e-4 SENZA contrazione FMA e 1,086e-4 CON —
 * inviluppo 1,447e-4. 2e-3 sta 13,8x sopra.
 */
export const PREFILL_GEMM_Q80_F32_ABS_TOL = 2e-3;
