# SPEC — riga 2 di `engine-kquant`: il Q5_K in produzione

Goal: `.harness/goals/engine-kquant/GOAL.md`. Riga 2 di `PHASES.md`.
Fase 0 chiusa (it.1-it.3): la forma e' MISURATA, questa riga la cabla.

## 1. Il fatto di partenza

`gemm:deltanet-out` e' il **37,9% del tempo del prefill** (12.169 ms su 32.101)
con soli 24 dispatch, perche' i 24 tensori `blk.*.ssm_out` sono **Q5_K** e
cadono sul fallback legacy di `gemvB` (`src/engine/q35gpumodel.ts:778`): M GEMV
replicate su `wid.z`, **riuso dei pesi ZERO**, 16 riletture per chunk.

La forma multi-riga per Q5_K esiste gia', misurata al banco:
**1,2700 → 0,0452 ms a M=16 = 28,10x**
(`results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json`,
celle `gemm-kquant-multirow` `q5_K/*`). Proiezione sul segmento: 24 × 395 ×
0,0452 = **429 ms**, cioe' **−11,7 s** sul tempo al primo token.

## 2. Cosa si porta, e da dove

Sorgente: `src/microbench/ttKQuant.ts`
- `kquantQ5KMultiRowSplitKIdotWgsl` — via intera (la vincente)
- `kquantQ5KMultiRowSplitKWgsl` — via f32, **fallback dichiarato** per i device
  senza `packed_4x8_integer_dot_product`

Destinazione: `src/engine/kernels/wgsl.ts`, accanto a
`prefillGemmQ4SplitKIdotWgsl` / `prefillGemmQ4SplitKWgsl`.

**REGOLA DEL PORT, gia' vigente in questo file**: il testo WGSL e' quello
misurato, riga per riga. Ogni divergenza va dichiarata in
`PREFILL_GEMM_PORT_DIFFS` con la sua ragione; `tests/engine-prefillgemm.test.ts`
fallisce su qualunque riga divergente non dichiarata, **in entrambe le
direzioni** (niente chiavi morte). La misura e' una proprieta' del TESTO.

## 3. I quattro pezzi, e l'ordine fra loro

**(a) `prefillGemmSplitsFor` va reso PARAMETRICO SULLA FAMIGLIA.** Oggi
(`wgsl.ts:4150`) conta in blocchi da 32 (`bpr = K/32`) e rifiuta K non multiplo
di 64. Sui K-quant l'unita' indivisibile e' il **superblocco da 256**: le scale
a 6 bit sono condivise dagli otto sotto-blocchi, e una fetta che ne tagliasse
meta' dovrebbe rileggere l'header comunque — falsificando il conto dei byte su
cui poggia tutto il goal. La forma gia' scritta e' `kquantSplitsFor` in
`ttKQuant.ts`: 4 fette se le unita' per riga sono divisibili per 4, 2 se per 2,
altrimenti 1. `ssm_out` ha K=4096 = **16 superblocchi ⇒ 4 fette**.

**(b) Il PREDICATO DI AMMISSIBILITA' resta in UNA sede sola.**
`prefillgemmplan.ts` chiede al kernel (`prefillGemmCheck`) quali kind accetta;
`q35gpumodel.ts` **non deve** ri-derivare la condizione. Un secondo posto che
decide la stessa cosa e' il bug di it.7 del goal precedente, e il gate
strutturale che lo rileva esiste gia'. Il predicato passa da «q4_0-only» a
«q4_0 oppure q5_K», con il messaggio d'errore che continua a nominare i kind
NON supportati invece di dire genericamente "non supportato".

**(c) IL CABLAGGIO. Attenzione: `ssm_out` NON passa da `gemvB`.** Oggi il ramo
K-quant di `loadW` (`q35gpumodel.ts:487-505`) costruisce da se' il suo `pushB`
con `gemvQ5KWgsl({batch: true})`, e non tocca mai `gemvB`. Il cablaggio va
fatto **li'**, non nel fallback di `gemvB`: cioe' il ramo K-quant di `loadW`
deve chiedere la rotta al piano esattamente come fa `gemvB`, e prendere la via
veloce quando il piano la concede. Chi cabla `gemvB` e basta non cambia una
riga del comportamento reale, e i test di copertura passerebbero comunque
perche' contano i SITI, non i dispatch: e' il modo silenzioso di sbagliare
questa riga.

**(d) IL BUFFER DEI PARZIALI.** La via veloce scrive in `prefillPart` e poi
combina. La taglia oggi e' dimensionata sul massimo N che `gemvB` puo'
ricevere (`q35gpumodel.ts:629`): `ssm_out` ha N=2560, dentro il massimo
esistente, **ma la formula va ri-verificata** perche' adesso il consumatore non
e' piu' solo `gemvB`. Se il massimo cambia, cambia dichiarandolo.

## 4. DONE WHEN (mechanical — sono le clausole della riga 2 di PHASES.md)

1. `npx vitest run tests/engine-prefillgemmplan.test.ts` → il test `[6c]`
   riporta un rapporto **≥ 10,9x** sull'inventario per-layer INTERO del 4B a
   M=16 (oggi **5,8593x**), con le 24 `ssm_out` fra i siti `multirow` e non piu'
   fra le `exceptions`.
2. Il test `[6d]` — che dice di se' «se un giorno la copertura sale, questo test
   fallisce e va cancellato con la sua ragione» — e' **cancellato con quella
   ragione scritta nel commit**, non aggirato.
3. **Caso ktest nuovo** per il GEMM multi-riga Q5_K contro il riferimento CPU:
   `node tools/harness/engine-ktest.mjs` con tutti PASS (oggi 101 PASS / 0
   FAIL — il numero sale, gli zero FAIL restano).
4. **Floor test**: un test che ri-deriva la tolleranza del caso ktest dal
   PAVIMENTO ARITMETICO f32 del testo WGSL generato, **con e senza contrazione
   FMA**, sul modello di `src/engine/kquantfast.ts` +
   `tests/engine-kquant-f32floor.test.ts`. Una tolleranza tarata sul device di
   sviluppo e' un gate che passa per fortuna.
5. `npx vitest run tests/gpulimits.test.ts` verde, con il fabbisogno di
   workgroup storage del kernel nuovo espresso come **formula esportata accanto
   al kernel** ed entrante nel `Math.max` CALCOLATO di
   `QWEN_WORKGROUP_STORAGE_BYTES`. Misurato al banco: **5.120 B a M=16** — sotto
   i 16.384 garantiti da WebGPU, quindi non alza il tetto; ma va DICHIARATO, o
   il prossimo kernel che lo alza non trovera' il posto dove dirlo.
6. `npx tsc --noEmit` pulito · `npx vitest run` verde.
7. Il piano **non instrada M=1** sulla via multi-riga: a M=1 il banco la misura
   PIU' LENTA della legacy (0,56x sulla via f32). Oggi e' vero per costruzione
   (la via veloce vive solo nel piano gemello del prefill); serve un test che lo
   inchiodi, non un commento.

**CHI ESEGUE QUALE GATE.** I gate SENZA GPU (`npx tsc --noEmit`,
`npx vitest run`) li esegue chi implementa, a ogni task. I gate CON GPU — il
ktest (che vuole un server vite e un Chrome vero) e qualunque bench — li esegue
il chiamante dopo l'integrazione: un task che dichiara verde un ktest che non
ha girato e' il difetto piu' grave che questo repo abbia registrato
(`engine-ktest.mjs` usciva 0 senza aver eseguito nulla).

## 5. VINCOLI (dal contratto — non negoziabili in questa riga)

- **Nessun `enable packed_4x8_integer_dot_product`**: e' una language feature,
  non un'estensione. Scriverla fa fallire la compilazione con «expected
  extension» (costo' una run intera).
- **Ogni via intera nuova va accompagnata dal suo fallback f32 DICHIARATO.**
  Entrambe si portano: sono entrambe misurate.
- **I gate di correttezza non si allargano**: ktest tutti PASS, top-1 contro
  l'oracolo llama.cpp ≥ 1012/1024 = 98,828% su ENTRAMBI i bracci, sequenze
  generate identiche 8/8.
- **Non-regressione**: decode 4B ≥ 45,5 tok/s a ctx 6333 (banda ±5%).
- Nessuna attribuzione AI in commit o PR.
- Non toccare: `PREFILL_M`, il path 0.5B, il cablaggio di 35B/GLM, i goal in
  standby.

## 6. FUORI DA QUESTA RIGA

- Il **Q4_1** e' la riga 3, con la stessa forma. Non anticiparlo qui: la riga 2
  deve poter essere misurata da sola, o non si sapra' quale delle due leve ha
  prodotto quale pezzo del guadagno.
- Le tre famiglie del 35B (Q4_K, Q6_K, Q8_0) sono misurate e **non si cablano**
  (riga 4: ktest + scheda di consegna).
- La misura end-to-end del TTFT e' la riga 5, su codice finale.
