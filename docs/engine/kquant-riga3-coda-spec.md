# SPEC — coda della riga 3: la conformita' del Q4_1

Il cablaggio della riga 3 e' **fatto e in albero** (commit `54f909c`): i kernel
`prefillGemmQ41SplitKIdotWgsl` / `prefillGemmQ41SplitKWgsl` esistono, `gemvB` ha
il suo sito q4_1 esplicito, il piano instrada, la copertura e' salita a
**15,5247x** e la suite e' verde (791 passed).

**Manca solo la CONFORMITA' del kernel nuovo**, che e' esattamente la parte che
nessun test di aritmetica puo' dedurre: il kernel gira davvero e da' i numeri
giusti su una GPU vera?

## Cosa fare, e su quale modello ricalcarlo

**Tutto esiste gia' per il q5_K**: questa e' una gemellazione, non un progetto.
Chi lo fa deve LEGGERE i tre file del q5_K e produrne il gemello q4_1, non
inventare una forma nuova.

### (1) `src/engine/prefillkquant.ts` — il caso e le tolleranze

Aggiungere `PREFILL_Q41_KTEST_CASE` e le quattro tolleranze
(`PREFILL_GEMM_Q41_{IDOT,F32}_{REL,ABS}_TOL`), sul modello letterale di quelle
q5_K che stanno nello stesso file, **con la loro derivazione scritta**.

Geometria del caso, e ognuna ha la sua ragione:
- `K: 9216` — la K VERA di `blk.0-3.ffn_down` del 4B, la shape su cui la fase 0
  ha misurato 22,57x. 288 blocchi da 32 per riga, che
  `prefillGemmSplitsFor` divide in **4 fette** (288 % 8 == 0);
- `N: 200` — ridotta rispetto ai 2560 veri (il riferimento f64 costa), e
  **NON multiplo di 64 apposta**: il moltiplicatore lancia workgroup da 64
  righe di uscita, quindi l'ultimo ne ha 56 oltre il bordo ed e' l'unico modo
  perche' la guardia `r < N_ROWS` sia ESERCITATA invece che creduta;
- `M: 16` — la riga di chunk del prefill;
- due seed separati, come nel caso q5_K: uno per i byte dei blocchi, uno per le
  attivazioni.

**Preparazione dei pesi**: `randBytes(nBlocks * Q4_1_BLOCK_BYTES, seed)` +
`fixScalesAt(src, Q4_1_BLOCK_BYTES, [1, 3])` — sono `d` e `m`, entrambi f16, e
sono gli stessi offset che `testGemvC2` usa gia' per il q4_1 in
`ktest.worker.ts`. Riferimento in f64 via `dequantQ4_1`.

### (2) `src/engine/ktest/ktest.worker.ts` — i due casi

Gemello di `testPrefillGemmQ5KMultiRow` (riga ~3741). Le differenze, tutte
meccaniche:
- il q4_1 ha **DUE buffer** (`repackQ4_1` → `qs` + `scales`) dove il q5_K ne ha
  uno (`blocks`): i binding della via intera sono
  `[qs, scales, xq, part, xsc]` e quelli della f32 `[qs, scales, x, part]`;
- `kind: "q4_1"` negli `PrefillGemmOpts`;
- nomi dei casi: `prefill-gemm-q41-multirow-idot` e `-f32`.
Registrarli in `main()` accanto a `testPrefillGemmQ5KMultiRow`.

Il braccio intero si confronta con un riferimento calcolato sulle attivazioni
**ri-espanse dopo la quantizzazione a i8** (amax/127 per blocco da 32), come fa
il caso q5_K: cosi' il caso misura il KERNEL e non la quantizzazione.

### (3) `tests/engine-prefillkquant-f32floor.test.ts` — il pavimento

Estendere con la sezione q4_1: **genera i kernel VERI**, ne asserisce le
costanti, e RI-DERIVA da un'emulazione f32 il pavimento aritmetico con e senza
contrazione FMA. Le tolleranze di (1) devono stare **sopra il pavimento** e
**sotto 20 volte il pavimento** — sono le due barre che il file q5_K usa gia'.

L'aritmetica da emulare per il q4_1: `w = d*q + m` con `q` in [0,15] senza
offset, prodotto scalare intero ESATTO (nibble x i8, accumulo i32), piu' il
termine `m * Sigma(x)` per blocco; l'errore f32 sta nell'accumulo dei 288
termini per riga e nella somma delle 4 fette. **Attenzione**: i termini per riga
sono 2,25 volte quelli del caso q5_K (288 contro 128), quindi un pavimento
IDENTICO a quello del q5_K sarebbe sospetto, non rassicurante.

### (4) Il `pbCat` dei quattro siti, verificato senza GPU

`q35gpumodel.ts` timbra ora `gemm:ffn-down-q41` sui soli siti Q4_1 (layer 0-3) e
`gemm:ffn-down` sugli altri 28. Serve un test che lo verifichi **sul sorgente o
sull'inventario del piano**, non a occhio: senza, la riga 5 misurerebbe un
segmento la cui composizione nessuno ha inchiodato. Il file
`tests/engine-prefillwiring-q5k.test.ts` ha gia' la macchina per scansionare il
sorgente biancato: riusarla.

## DONE WHEN

1. `npx tsc --noEmit` pulito · `npx vitest run` verde (oggi **791 passed | 10
   skipped**: il numero sale, i falliti restano zero).
2. Il floor test passa e **stampa** i pavimenti misurati e il rapporto
   tolleranza/pavimento per entrambe le vie, come fa gia' per il q5_K.
3. Il test del `pbCat` c'e' ed e' verde.
4. I due casi ktest sono registrati in `main()`.

**Il ktest su GPU lo esegue il chiamante**, non chi implementa: qui non c'e' ne'
browser ne' server. Un task che dichiarasse verde un ktest non eseguito e' il
difetto piu' grave che questo repo abbia registrato.

## VINCOLI

- Non toccare i kernel gia' in produzione (q4_0, q5_K, q4_1): questa coda
  aggiunge conformita', non cambia comportamento.
- Nessun `enable packed_4x8_integer_dot_product`.
- Le tolleranze si DERIVANO dal pavimento, non si tarano su cio' che esce.
- Nessuna attribuzione AI nei commit.
