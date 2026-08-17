# SPEC — riga 3 di `engine-kquant`: il Q4_1 in produzione

Goal: `.harness/goals/engine-kquant/GOAL.md`. Riga 3 di `PHASES.md`.
La riga 2 (Q5_K) e' **chiusa e in produzione**: questa riga ne riusa la
struttura per l'ultima famiglia che il 4B ha sul percorso vecchio.

## 1. Il fatto di partenza

I 4 tensori `blk.0-3.ffn_down` sono **Q4_1** (K=9216, N=2560) e sono
**il 71% dei byte** del segmento `gemm:ffn-down` (4.971 ms, 15,5% del prefill):
le altre 28 `ffn_down` sono gia' q4_0 e gia' multi-riga. I quattro Q4_1 cadono
sul fallback legacy — M gemv replicate su `wid.z`, riuso pesi ZERO.

Forma misurata al banco (fase 0, it.1-it.3):
**2,3474 → 0,1040 ms a M=16 = 22,57x** (via intera), **16,63x** (via f32).
Artefatto: `results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json`,
celle `q4_1/*`.

## 2. Cosa si porta

Sorgente: `src/microbench/ttKQuant.ts`
- `kquantQ41MultiRowSplitKIdotWgsl` — via intera
- `kquantQ41MultiRowSplitKWgsl` — via f32, fallback dichiarato

Destinazione: `src/engine/kernels/wgsl.ts`, accanto ai gemelli q4_0 e q5_K.
Vale la **REGOLA DEL PORT** gia' vigente: testo riga per riga, ogni divergenza
in `PREFILL_GEMM_PORT_DIFFS` con la sua ragione, e
`tests/engine-prefillgemm.test.ts` fallisce sulle righe divergenti non
dichiarate in ENTRAMBE le direzioni.

## 3. I pezzi

**(a) `PREFILL_GEMM_KINDS` diventa `["q4_0", "q5_K", "q4_1"]`.** Il
`Record<PrefillGemmKind, PrefillGemmKindSpec>` in `PREFILL_GEMM_SPEC`
(`wgsl.ts:3935`) **non compila** finche' il kind nuovo non porta la sua
specifica: e' la struttura che la riga 2 ha lasciato apposta. Per il q4_1:
- `kUnit: 64` e `unitsPerRow: (K) => K / 32` — identici al q4_0: il blocco e'
  da 32 pesi e il loop avanza a BK = 2 blocchi;
- `splitsFor` identica al q4_0 (4 fette se i blocchi si dividono a BK=2);
- `formatWhy`: nibble a 4 bit **senza offset** (il q4_0 centra su zero, il q4_1
  no) piu' `d` e `m` f16 per blocco, cioe' UNA parola di scale per blocco
  invece di una ogni due;
- `wgBytes.idot`: `xs` M*16 u32 + `xss` M*2 f32 + `xsum` M*2 f32 = **80·M B**
  (1.280 a M=16, misurato al banco);
- `wgBytes.f32`: `xs` M*16 vec4<f32> = **256·M B** (4.096 a M=16).

**(b) L'ARITMETICA CHE IL q4_0 NON HA.** `w = d*q + m` con `q` in [0,15]:
- i nibble vanno in i8 **senza** la correzione `-8` del q4_0;
- serve il termine `m * Σ(x)` per blocco, che nella via intera si ottiene con
  `dot4I8Packed(xq, 0x01010101)` calcolato **una volta per workgroup** in
  memoria di gruppo (`xsum`), non una volta per thread;
- le scale si leggono `unpack2x16float(scales[gb])` — **una parola per blocco**
  — e non `scales[gb >> 1][gb & 1]` come nel q4_0. Sbagliare questo da' un
  binding valido e un kernel che legge le scale della riga sbagliata: nessun
  errore, logit storti.

**(c) IL CABLAGGIO E' IN `gemvB`, non in `loadW`.** Al contrario del q5_K, le
`ffn_down` q4_1 passano da `q40()` e poi da `gemvB` (`q35gpumodel.ts:778`), che
la rotta al piano **la chiede gia'**: `planPrefillGemm({kind: kk, ...})` con
`kk` che vale gia' `"q4_1"` per quei tensori. Oggi cade sul fallback solo
perche' il piano rifiuta il kind. Il lavoro qui e' emettere i kernel giusti nel
ramo veloce quando `route.via !== "legacy"` e `kk === "q4_1"` — con la stessa
doppia condizione difensiva che la riga 2 ha messo nel ramo K-quant, e per la
stessa ragione (il kind del kernel dev'essere quello con cui si e' chiesta la
rotta, per costruzione e non per convenzione).

**(d) UNA CATEGORIA DI MISURA PROPRIA PER I QUATTRO SITI — obbligatoria.**
Oggi `gemm:ffn-down` mescola 28 tensori q4_0 (gia' veloci) e 4 Q4_1 (lenti), e
la quota dei secondi e' **dedotta dal banco, non misurata**: il verificatore
l'ha marcata come l'anello debole della proiezione della riga 3. Serve un
`pbCat` distinto — es. `gemm:ffn-down-q41` — assegnato ai soli siti Q4_1 in
`q35gpumodel.ts`, cosi' il checkpoint della riga 5 **attribuisce** quel tempo
invece di dedurlo. Va fatto **PRIMA** che la riga 5 misuri, o il prima/dopo non
sara' confrontabile.

## 4. DONE WHEN (mechanical)

1. `npx vitest run tests/engine-prefillgemmplan.test.ts` → il test `[6c]`
   riporta **≥ 15,5x** sull'inventario per-layer INTERO a M=16 (oggi
   **10,9376x**), con le 4 `ffn_down` Q4_1 fra i siti `multirow`. La copertura
   sale da 196/248 a **200/248 siti** e da 96,914% a **99,796% dei byte**.
2. **Caso ktest nuovo** per il GEMM multi-riga Q4_1 contro il riferimento CPU,
   su ENTRAMBE le vie (intera e f32): `node tools/harness/engine-ktest.mjs`
   tutti PASS (oggi **103 PASS / 0 FAIL** — il numero sale, gli zero FAIL
   restano).
3. **Floor test**: le tolleranze del caso ktest ri-derivate dal pavimento
   aritmetico f32 del testo WGSL generato, **con e senza contrazione FMA**,
   sul modello di `src/engine/prefillkquant.ts` +
   `tests/engine-prefillkquant-f32floor.test.ts` (la riga 2 ha gia' la forma:
   riusarla, non re-inventarla).
4. `npx vitest run tests/gpulimits.test.ts` verde, col fabbisogno del kernel
   nuovo come formula accanto al kernel dentro il `Math.max` calcolato.
5. Il `pbCat` dei quattro siti Q4_1 e' distinto, e un test senza GPU lo verifica
   sull'inventario del piano (`prefillPlanInventory` o l'equivalente puro).
6. `npx tsc --noEmit` pulito · `npx vitest run` verde.
7. **M=1 resta legacy su ogni kind**: e' gia' clausola del piano dalla riga 2 e
   non va indebolita per far entrare il q4_1.

**CHI ESEGUE QUALE GATE**: i gate senza GPU li esegue chi implementa, a ogni
task. Il ktest e i bench li esegue il chiamante dopo l'integrazione — un task
che dichiara verde un ktest che non ha girato e' il difetto piu' grave che
questo repo abbia registrato.

## 5. VINCOLI

- Nessun `enable packed_4x8_integer_dot_product` (language feature, non
  estensione: scriverla fa fallire la compilazione).
- Entrambe le vie si portano: sono entrambe misurate.
- Gate di correttezza non si allargano; non-regressione decode ≥ 45,5 tok/s.
- Nessuna attribuzione AI nei commit.
- Non toccare `PREFILL_M`, il path 0.5B, il cablaggio 35B/GLM, i goal in
  standby, ne' il kernel q5_K appena messo in produzione.

## 6. FUORI DA QUESTA RIGA

- Le tre famiglie del 35B (Q4_K, Q6_K, Q8_0): misurate, **non cablate** — sono
  la riga 4 (ktest + scheda di consegna).
- La misura end-to-end del TTFT e dei segmenti: riga 5, su codice finale.
