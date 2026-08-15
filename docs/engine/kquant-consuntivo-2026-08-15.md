# Consuntivo del goal `engine-kquant` — 2026-08-15

**Tempo al primo token a modello caldo, prompt-idx 0 (6333 token), Qwen3.5-4B:
32.127 → 17.153 ms. Discesa 1,873×.**

Le due barre del contratto (ruling PI 2026-08-14) sono **entrambe passate**:
la barra del goal **< 22.500 ms** e il nice-to-have **< 18.000 ms**
(`results/engine/q35-ttft-kernel-checkpoint-4b-2026-08-15.json`,
`metrica.barraPassata: true`, `metrica.niceToHavePassato: true`,
`metrica.quotaDellaBarra: 0.762`).

È il primo goal di questa serie che chiude la sua barra. Questo documento dice
voce per voce cosa è stato ottenuto e **con quale artefatto**, dove il tempo è
finito prima e dopo, cosa la fase 0 aveva previsto bene e cosa no, e quali
trappole il goal successivo eredita.

La scheda di consegna al goal 35B è un documento a parte e **non è duplicata
qui**: `docs/engine/kquant-consegna-35b-2026-08-15.md` (423 righe).

---

## 1. Il DONE WHEN del contratto, clausola per clausola

Sorgente della checklist: `.harness/goals/engine-kquant/GOAL.md`, sezione
`DONE WHEN (all measurable)`, quindici clausole. Una clausola senza artefatto
accanto è una clausola non chiusa, e qui è dichiarata tale.

| # | clausola | esito | artefatto |
|---|---|---|---|
| 1 | **FASE 0 prima di qualunque cablaggio**: micro-bench isolato, JSON in `results/microbench/`, G pesi/s e GB/s per variante, M ∈ {1,8,16}, tutte le famiglie, shape reali; regola di stop ≥ 1,5× per famiglia | **SÌ** | `results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json` — `kind: microbench-kquant-fase0`, 54 celle `gemm-kquant-multirow`, `skipped: []`, `hostState.declared: quiescent`, adapter `nvidia lovelace`. Tabella in §4.1. Zero file sotto `src/engine/` toccati dalla riga (journal it.3) |
| 2 | **COPERTURA DEL 4B in CI senza GPU**: `[6c]` da **5,8593×** a **≥ 15,5×**, con le 24 `ssm_out` Q5_K e le 4 `ffn_down` Q4_1 fra i `multirow`; `[6d]` cancellato con la sua ragione scritta | **SÌ** | `tests/engine-prefillgemmplan.test.ts:442` `[6c]`. Eseguito oggi: **15,5247×**, legacy 32,751 GB → piano 2,110 GB, copertura **200/248 siti = 99,796% dei byte**, 48 eccezioni = 0,204%. `[6d]` non c'è più, e la ragione è scritta al suo posto (`:533-542`) |
| 3 | **I 48 SITI Q8_0 `ssm_alpha`/`ssm_beta` esclusi COI NUMERI** (0,204% dei byte, N=32 contro le 64 righe per workgroup della forma split-K) | **SÌ** | `[6c]` pinna `exceptions` a un solo kind (`["q8_0"]`), 48 siti, `onePass = 4.177.920 B`; e `tests/engine-prefillgemmplan-notwired.test.ts` `[w3]` verifica il **caso concreto** (q8_0 K=2560 N=32: il kernel accetta la shape, il piano la lascia comunque legacy, 48 dispatch legacy su 48 siti) |
| 4 | **SEGMENTI** sullo stesso JSON, con byte e GB/s accanto: `gemm:deltanet-out` ≤ 2.000 ms, `gemm:ffn-down` ≤ 2.000 ms | **SÌ** | checkpoint 2026-08-15: `gemm:deltanet-out` **572,8 ms** (68,3 GB, 119,3 GB/s); `gemm:ffn-down` **1.187,6 + 225,8 = 1.413,4 ms** (146,8 + 23,3 GB, 123,6 e 103,2 GB/s). Il `+225,8` è la categoria `gemm:ffn-down-q41` scorporata in it.5: il confronto onesto col 4.971 di prima è la SOMMA |
| 5 | **TTFT a caldo (a) < 22.500 ms** — la barra del goal | **SÌ** | 17.153 ms = `prefill.ms 17.126 + decode.firstMs 27`, checkpoint 2026-08-15 |
| 5b | **TTFT a caldo (b) < 18.000 ms** — nice-to-have | **SÌ** | stesso numero, `niceToHavePassato: true`. Il margine sulla (b) è 847 ms, cioè il 4,9%: è un bersaglio di ottimo raggiunto per poco, non con comodità |
| 6 | **`prefill.tokS > 282`** | **SÌ** | **369,72** tok/s (era 197,25 — checkpoint 2026-08-14) |
| 7 | **`prefill.tokS > decode.tokS`** | **SÌ** | 369,72 > 47,06 |
| 8 | **CORRETTEZZA con tolleranza DERIVATA**: un caso ktest per OGNI forma prodotta (Q5_K, Q4_1, Q4_K, Q6_K, Q8_0, anche le non cablate) + floor test che ri-deriva la tolleranza dal pavimento f32 del testo WGSL, con e senza contrazione FMA | **SÌ** | **dieci bracci** (due per formato) su GPU vera: ktest **111 PASS / 0 FAIL** (era 101 a inizio goal). Numeri in §5. Floor test: `tests/engine-prefillkquant-f32floor.test.ts` (1.928 righe, tutti e cinque i formati, `floor = max(conFMA, senzaFMA)`, tolleranza vincolata fra 1× e 20× il pavimento, più una mutazione discriminante per formato) |
| 8b | **una forma verificata ma non cablata non è una forma inventata**: il piano instrada SOLO i kind misurati **e** cablati | **SÌ** | flag `wired` + `wiredWhy` in `PREFILL_GEMM_SPEC` (`src/engine/kernels/wgsl.ts:4003, 4036, 4066, 4109, 4143, 4181`), consumato in una sede sola dentro `kernelVerdict`; `PREFILL_GEMM_WIRED_KINDS` (`:4203`) è un sottoinsieme proprio di `PREFILL_GEMM_KINDS`. Gate: `tests/engine-prefillgemmplan-notwired.test.ts`, 17 test, provato mutando `wired: true` su q8_0 (8 test rossi in 3 file, journal it.8) |
| 9 | **GATE SECCHI A OGNI MERGE**: ktest tutti PASS; top-1 vs oracolo llama.cpp ≥ 1012/1024 su **entrambi** i bracci; sequenze generate identiche 8/8; `vitest` verde; `tsc --noEmit` pulito | **SÌ** | `results/engine/ratchet-correttezza-2026-08-15.json`: ktest **111/0**; top-1 **1012/1024 = 98,828%** sequenziale (`q35-conf-4b-riga6-seq-2026-08-15.json`) e **1012/1024** a chunk (`q35-conf-4b-riga6-chunk-2026-08-15.json`), identici anche nella ripartizione per prompt `[125,125,125,127,128,127,127,128]`; sequenze **8/8 identiche, zero token di differenza su 1024 posizioni**; vitest **1017 passed \| 10 skipped**; tsc exit 0. Tutti misurati sull'albero finale, nessuno ricopiato |
| 10 | **NON-REGRESSIONE, decode 4B ≥ 45,5 tok/s a ctx 6333** | **SÌ** | **47,06** tok/s, stesso checkpoint (`metrica.decodeTokS`) |
| 10b | **NON-REGRESSIONE, GLM b12 optimistic entro ±5% di 13.172 / 31,26 / 14,74** | **INTERPRETATA** — v. §6 | `results/engine/bench-glm-4090-b12-riga6-2026-08-15.json` contro `bench-glm-4090-b12-optimistic-nonreg-2026-08-09.json`: 15,330 / 37,542 / 12.279, cioè **fuori banda da tutte e tre le parti, in meglio**. Letta alla lettera, questa clausola FALLISCE |
| 11 | **PORTABILITÀ**: il fabbisogno di workgroup storage di OGNI kernel nuovo è una formula esportata accanto al kernel **ed entra nel `Math.max` calcolato**; `tests/gpulimits.test.ts` verde | **SÌ** (chiusa in it.11 — v. §7.2) | La formula esportata c'è per tutti e sei i formati (`PREFILL_GEMM_SPEC[kind].wgBytes`, verificata contro il TESTO del WGSL generato in `tests/engine-prefillgemm.test.ts:518-548`). Il `Math.max` di `engineNeeds` prende ora **un termine per ogni kind di `PREFILL_GEMM_KINDS`**, derivato dall'elenco via `PREFILL_GEMM_SHAPES` (`Record<PrefillGemmKind, …>`: un kind nuovo senza la sua shape non compila). `tests/gpulimits.test.ts` **38 test verdi**, incluso il gate che asserisce che ogni kind abbia il suo termine |
| 12 | **UNA SOLA SEDE PER LA ROTTA**: l'ammissibilità resta in `prefillgemmplan.ts`, che la chiede al kernel; nessun secondo predicato in `q35gpumodel.ts` | **SÌ** | gate strutturale `[w4] prefillGemmWiring si legge solo nel piano` (`tests/engine-prefillgemmplan-notwired.test.ts:190-241`), con il guard contro la scansione vuota. Il predicato di ammissibilità non vive nel piano: è sondato dal kernel (`prefillgemmplan.ts:174-193`) |
| 13 | **DISCIPLINA DEL PORT**: il WGSL di produzione è quello misurato al banco, riga per riga; ogni divergenza in `PREFILL_GEMM_PORT_DIFFS` con la ragione | **SÌ** | `PREFILL_GEMM_PORT_DIFFS` è **`{}`** (`src/engine/kernels/wgsl.ts:3893`) **con sei kernel dentro**, confrontato su 14 coppie banco/produzione e nelle due direzioni (`tests/engine-prefillgemm.test.ts:680-740`: niente righe divergenti non dichiarate, niente chiavi morte). Un port riscritto invece che portato non avrebbe potuto lasciare quel record vuoto |
| 14 | **I BYTE DEL SEGMENTO SI CHIEDONO AL METER**, non si ricopiano (`build-ttft-checkpoint.mjs:108` ricopiava `173_015_040`) | **SÌ, e il difetto sotto era più grande** | l'inventario dei siti è uscito da un file di TEST ed è ora `src/engine/q35prefillsites.ts`, modulo puro importato da script e test; i byte li conta `dispatchWeightBytes` (`src/engine/prefillbytes.ts`), la forma la decide `planPrefillGemm`. Gate: `tests/engine-ttft-checkpoint-banda.test.ts`. Cosa il debito nascondeva: §7.1 |
| 15 | **CONSUNTIVO** voce per voce + **scheda di consegna al goal 35B** | **SÌ** | questo file; `docs/engine/kquant-consegna-35b-2026-08-15.md` |

**Bilancio: tredici clausole soddisfatte con l'artefatto accanto, una
interpretata e dichiarata (10b), una soddisfatta solo in parte (11).** Nessuna
clausola è stata ammorbidita, e nessuna è chiusa senza artefatto.

---

## 2. La nuova ripartizione del tempo per segmento — prima e dopo

Il senso del goal è lo **spostamento**, quindi le due colonne stanno accanto.
Prima: `results/engine/q35-ttft-kernel-checkpoint-4b-2026-08-14.json`
(prefill 32.101 ms). Dopo:
`results/engine/q35-ttft-kernel-checkpoint-4b-2026-08-15.json`
(prefill 17.126 ms). Stesso prompt, stesso `chunkM: 16`, stessi `chunks: 395`,
stesso `hostState.declared: "quiescent"`, stesso modello
(`modelSha256: 298fcb5f…`).

| segmento | ms PRIMA | % | ms DOPO | % | Δ |
|---|---|---|---|---|---|
| **`gemm:deltanet-out`** (24 `ssm_out` Q5_K) | **12.169,1** | **37,91** | **572,8** | 3,34 | **−95,3%** |
| **`gemm:ffn-down`** (28 `ffn_down` Q4_0) | **4.971,1** | **15,49** | **1.187,6** | 6,93 | v. nota |
| **`gemm:ffn-down-q41`** (4 `ffn_down` Q4_1) | *dentro il precedente* | — | **225,8** | 1,32 | — |
| *somma dei due `ffn-down`* | *4.971,1* | *15,49* | *1.413,4* | *8,25* | **−71,6%** |
| `deltanet:recurrence` | 1.601,8 | 4,99 | **1.732,1** | **10,11** | +8,1% |
| `gemm:ffn` | 1.233,8 | 3,84 | 1.303,0 | 7,61 | +5,6% |
| `deltanet:gemm` | 972,6 | 3,03 | 1.030,8 | 6,02 | +6,0% |
| `attn:core` | 804,6 | 2,51 | 848,3 | 4,95 | +5,4% |
| `gemm:qkv` | 315,6 | 0,98 | 332,6 | 1,94 | +5,4% |
| `gemm:attn-out` | 144,8 | 0,45 | 155,7 | 0,91 | +7,5% |
| `norm:attn` | 132,9 | 0,41 | 140,9 | 0,82 | +6,0% |
| `norm:ffn` | 132,6 | 0,41 | 139,9 | 0,82 | +5,5% |
| `ffn:act` | 83,5 | 0,26 | 87,1 | 0,51 | +4,3% |
| `resid` | 71,6 | 0,22 | 74,7 | 0,44 | +4,3% |
| `deltanet:gates` | 51,9 | 0,16 | 54,2 | 0,32 | +4,4% |
| `attn:prep` | 38,4 | 0,12 | 40,5 | 0,24 | +5,5% |
| `attn:rope+kv` | 30,5 | 0,10 | 32,4 | 0,19 | +6,2% |
| **dentro i pass GPU** | **22.754,8** | **70,9** | **7.958,4** | **46,5** | **−65,0%** |
| **fuori dai pass GPU** | **9.346** | **29,1** | **9.168** | **53,5** | −1,9% |

**Tre cose che questa tabella dice e che vanno lette insieme:**

1. **Tutto il guadagno è dentro i pass.** −14.796 ms dentro, −178 ms fuori
   (che è dentro il rumore). I 9,35 s fuori dai pass che il contratto dichiarava
   invarianti per questa leva **lo sono stati**.
2. **I segmenti che la leva non tocca sono cresciuti del 4-8%, sistematicamente,
   tutti.** Non lo attribuisco: le due misure vengono da run diverse, la sonda
   per segmento perturba (spezza il pass di ogni layer) e i totali sono
   per-chunk misurati × 395. Il verso è coerente su quindici segmenti su
   quindici, quindi è deriva sistematica e non rumore per segmento — ma **le
   differenze piccole di questa tabella non sono interpretabili**, e le due che
   contano (−95,3% e −71,6%) stanno un ordine di grandezza fuori da quella
   deriva.
3. **Le percentuali hanno denominatori diversi** (32.101 e 17.126 ms): un
   segmento può salire in quota restando fermo in millisecondi. È esattamente
   ciò che è successo a tutto ciò che non è `gemm:deltanet-out`.

### 2.1 La banda, prima e dopo

| segmento | GB PRIMA | GB/s PRIMA | GB DOPO | GB/s DOPO |
|---|---|---|---|---|
| `gemm:deltanet-out` | **1.093,5** | **89,9** | **68,3** | **119,3** |
| `gemm:ffn-down` (28 q4_0) | 147,2 | — | 146,8 | 123,6 |
| `gemm:ffn-down-q41` (4 q4_1) | 373,7 | — | 23,3 | 103,2 |
| *somma `ffn-down`* | *520,9* | *104,8* | *170,1* | *120,3* |

- I byte "prima" di `gemm:deltanet-out` e i suoi 89,9 GB/s stanno
  nell'artefatto del 14 agosto (`banda[0]`, `forma: "legacy"`). Quelli "prima"
  dei due `ffn-down` **non ci sono in quell'artefatto**: lo schema 1 pubblicava
  due sole categorie. Sono l'aritmetica del contratto
  (`GOAL.md:45-47`), ricalcolabile dall'inventario pinnato — e il contratto la
  calcolava su **396** chunk invece di 395, da cui i 373,7 GB contro i 372,8
  che escono a 395. Lo scrivo perché la colonna "prima" di quelle due righe è
  **derivata, non misurata**.
- **1.093,5 → 68,3 GB è esattamente 16×**, cioè `PREFILL_M`. È la definizione
  della leva vista dal lato dei byte: la forma legacy rileggeva la matrice una
  volta per riga di chunk.
- **La banda per byte migliora poco** (89,9 → 119,3 GB/s). Il guadagno non
  viene dall'andare più veloce sui byte: viene dal **non leggerli**. Ed è il
  motivo per cui la proiezione «alla stessa banda misurata oggi» del contratto
  era conservativa e ha comunque tenuto.
- I dispatch di `gemm:deltanet-out` passano da **24 a 72** (la forma veloce ne
  emette tre dove la legacy ne emetteva uno) e quelli dei due `ffn-down` da
  **120 a 128**: il costo di dispatch che il verificatore di it.1 aveva messo
  nel conto è reale e vale decine di ms, non secondi.
- `calcolo.tflopsSostenuti` **1,578 → 2,958** su un picco fp32 misurato di 9,26
  (17% → 31,9%). Non è un'efficienza — il prefill gira su pesi quantizzati e
  quel tetto non lo tocca per costruzione. Dice la sola cosa che serve: **il
  collo non è l'ALU, né prima né dopo.**

---

## 3. Il termine che diventa PRIMO, e il fatto più grande accanto

### 3.1 `deltanet:recurrence` — 1.732,1 ms = 10,11% del prefill

Misurato, non ipotizzato:
`results/engine/q35-ttft-kernel-checkpoint-4b-2026-08-15.json`,
`segmenti[0]` — `msPerChunk: 4.3852`, `msTotale: 1732.1`,
`quotaPrefillPct: 10.11`, **768 dispatch**, 80 workgroup per dispatch
(`wgMin: 32`, `wgMax: 128`).

È il primo termine del prefill dopo questa leva, e nel checkpoint precedente
era il terzo (1.601,8 ms = 4,99%). **Non è cresciuto: è rimasto fermo mentre
tutto il resto scendeva** — i +8,1% sono la deriva sistematica di §2, non un
effetto del goal.

Due avvertenze per chi volesse attaccarlo:

- **768 dispatch su 32 layer sono il conteggio più alto del prefill**, e il
  goal `engine-ttft` ha già pagato una volta il prezzo di scambiare i dispatch
  per il tempo (it.23 di quel goal concluse «il quarto collo è la ricorrenza,
  47% del piano» contando i dispatch; col cronometro era il 5,0% —
  `docs/engine/ttft-consuntivo-2026-08-14.md` §4). Oggi il numero è il **10,11%
  del tempo**, misurato col cronometro, ed è quello il termine di paragone.
- `deltanet:recurrence` è in `bandaNonAttribuita` del checkpoint: non muove
  pesi di GEMM, quindi non ha una banda per segmento e **questa leva non lo
  tocca né lo toccherebbe**. Chi lo attacca lavora su un'altra famiglia di
  problema (occupancy e forma della ricorrenza), non sui byte dei pesi.

### 3.2 Il fatto più grande: **GPU nei pass 46,5%, fuori 53,5%**

`doveFinisceIlTempo` del checkpoint 2026-08-15: `dentroIPassGpuMs: 7958`
(46,5%), `fuoriDaiPassMs: 9168` (**53,5%**). Nel checkpoint precedente erano
70,9% e 29,1%.

**Più della metà del prefill non è più calcolo.** Il contratto lo aveva
registrato come fuori scope e aveva predetto che «DOPO questa leva diventa oltre
il 50% del TTFT» (`GOAL.md:260-263`): la previsione è stata verificata, ed è il
53,5%.

Cosa sono quei 9.168 ms, alla lettera dell'artefatto: «encode CPU (i bind group
sono già costruiti al load, restano `setPipeline`/`setBindGroup`/
`dispatchWorkgroups`), submit, e i buchi fra un submit e il successivo». E cosa
NON sono: **attribuiti più finemente**. Servirebbe una sonda CPU per dispatch
che oggi non esiste.

**Conseguenza operativa, ed è la consegna principale di questo consuntivo:
chi volesse ancora TTFT dopo questo goal non lo trova nei kernel.** Anche
azzerando tutti e sedici i segmenti GPU si arriverebbe a 9.168 ms di prefill;
sommato ai 27 ms di `decode.firstMs` resta un pavimento di ~9,2 s, cioè il
53,5% di quello che si paga oggi. Il termine di kernel più grande rimasto vale
1,7 s. **Il rapporto fra le due leve è 5,3 a 1, e non è a favore dei kernel.**

---

## 4. Cosa la fase 0 aveva previsto — inclusa la parte che ha sbagliato

### 4.1 La previsione che ha tenuto all'1,5%

Il micro-banco di fase 0 (it.2) proiettava un TTFT di **~16,9 s**; il cronometro
di riga 5 ha misurato **17,153 s**. **Errore 1,5%.**

E ha tenuto anche la **decomposizione** predetta nel contratto prima di
iniziare — «9.350 ms fuori dai pass GPU (invarianti per questa leva) + 8,05 s
di pass residui = 17,4 s» (`GOAL.md:49-51`) — contro i **9.168 fuori + 7.958
dentro = 17.126** misurati.

È il fatto metodologico più forte del goal: una previsione **fatta prima,
scritta in un artefatto pre-registrato**
(`docs/deep-dive/kquant-fase0-prereg-2026-08-14.md`, referenziato dal campo
`prereg` del JSON di fase 0) **e verificata dopo su un cronometro diverso**.
La proiezione era anche conservativa per costruzione: assumeva la **stessa
banda per byte** misurata sul percorso vecchio, e la banda è invece migliorata
del 33% (§2.1).

Il controllo che l'aveva resa credibile prima ancora della verifica, e che vale
più dei rapporti: il braccio legacy del banco riproduce il segmento di
produzione **in millisecondi**, non solo in banda —
`24 × 395 × 1,2700 = 12.039 ms` contro i **12.169 ms** misurati su
`gemm:deltanet-out`, **1,1% di scarto** (correzione del verificatore
indipendente, it.1: avevo scelto l'evidenza più debole, il confronto fra bande
91 e 89,9 GB/s).

### 4.2 Le tre previsioni su cinque che sono cadute

Pre-registrate in `docs/deep-dive/kquant-fase0-prereg-2026-08-14.md`, giudicate
in it.2 sull'artefatto:

| previsione | esito |
|---|---|
| **P1** — ≥ 10× su tutte e tre le famiglie ereditate | **CADUTA**: Q4_K 4,16-5,23×, Q6_K 6,13× |
| **P2** — Q8_0 la meno migliorata | **CADUTA, e rovesciata**: è la più migliorata (35,20×) |
| **P3** — K=512 rende meno di K=2048, stessa famiglia | **CADUTA**: Q4_K a K=512 fa 5,23× contro 4,16×, ed è anche più veloce in assoluto |
| **P4** — nessuna cella scartata | confermata: `skipped: []` su 54 celle |
| **P5** — memoria di gruppo sotto i 16.384 B garantiti | confermata: massimo **5.632 B** (Q6_K idot a M=16) |

**Le tre cadute dicono tutte la stessa cosa, ed è la lezione del goal: avevo
attribuito il guadagno al FORMATO, e il guadagno è una proprietà della SHAPE.**
La leva non è l'unpack: è quante volte la forma legacy rilegge la matrice e
quanto costa rileggerla. I tensori grandi (7-15 MB) rendono 22-35×; quelli degli
expert del 35B sono da 0,59-0,87 MB, stanno in cache, e le riletture se le serve
la L2 quasi gratis.

**E su P2 la correzione è più severa del verdetto**: non era falsificabile come
l'avevo scritta — confronta «chi migliora di più» fra formati misurati su shape
che differiscono di 10× in byte. Il rovesciamento è del disegno
dell'esperimento, non del mondo, e in it.2 l'avevo graduata come scoperta.

Nella stessa iterazione ho pubblicato **tre affermazioni sbagliate**, corrette
in it.2/it.3: che il `relDiff` del Q6_K fosse il più alto di tutti (il più alto
è il q8_0, 1,376e-2 contro 1,123e-2 — il numero era nell'artefatto e non
l'avevo guardato); P2 come sopra; e P5 dichiarata confermata mentre era stata
misurata al solo M=16 sulle tre famiglie nuove.

### 4.3 L'affermazione RITRATTATA

In it.2 avevo scritto, e messo nel digest: **«il 4-5× del 35B è un LIMITE
INFERIORE, non una stima»**. È stata **ritrattata in it.3**, per tre ragioni
indipendenti tutte del verificatore:

1. il crollo da 28× a 5× ha **due** cause di peso simile, non una: il legacy è
   ~2,2× più veloce per peso sulle shape piccole (cache) **e** la forma veloce è
   ~2,3× più lenta (32-64 workgroup su 128 SM: fame di parallelismo). 2,2 × 2,3
   ≈ 5. Solo la prima migliora in produzione;
2. il legacy a `[512,2048]` legge 9,44 MB in 49,6 µs = **190 GB/s** su una
   scheda che ne fa ~1.000: quella cella non è limitata dalla banda, quindi «in
   produzione la rilettura costa il prezzo pieno» presuppone un regime che la
   misura non mostra;
3. **la più seria**: per Q4_K e Q6_K il braccio legacy misurato **non è il
   percorso di produzione**. Il motore instrada gli expert in regime d'ARENA con
   `accum`, e `wgsl.ts:2175`/`:2359` vietano per costruzione la combinazione
   `batch && arena`. In più, nel prefill MoE cade la premessa stessa della leva:
   token diversi selezionano expert diversi, quindi la "stessa matrice riletta M
   volte" non c'è.

**Il 4-6× non è né un tetto né un pavimento: vale per una shape su un braccio
ipotetico.** È scritto così nella scheda di consegna, ed è la cosa che il goal
35B deve leggere prima del numero nudo.

### 4.4 Il reperto che i fallback f32 hanno comprato

Le tre famiglie del 35B erano state misurate in it.2 **senza** il fallback f32 e
al solo M=16 — una restrizione del done-when decisa da me e non registrata, che
il verificatore ha bocciato (FAIL di it.2) perché una delle due violava un
CONSTRAINT esplicito del contratto («ogni via intera nuova va accompagnata dal
suo fallback f32 DICHIARATO»). Correzione presa eseguendo, non negoziando: it.3
ha scritto i tre kernel mancanti.

**Cosa si è scoperto scrivendoli: sul Q4_K la via f32 NON supera la regola di
stop.** 1,14× sulla shape gate/up `[2048,512]` e 1,85× sulla down `[512,2048]`,
contro la barra di 1,5×. Su un device senza
`packed_4x8_integer_dot_product` la forma multi-riga, su quelle shape, non vale
la pena. Non l'avremmo saputo saltando quei kernel.

**E a M=1 la forma multi-riga PERDE** (0,91× sul Q4_K gate/up, 0,56× sul Q5_K
f32): il combine dei parziali non si ammortizza su una riga sola. È il numero
che vieta di offrire questa forma al decode, e serve anche a leggere il §6.

---

## 5. Le tolleranze derivate hanno predetto il silicio tre volte di fila

Dieci bracci ktest, cinque formati, tutti su GPU vera. La tolleranza non è mai
stata scelta: è **derivata** dal pavimento aritmetico f32 del testo WGSL
generato, calcolato con e senza contrazione FMA e preso come inviluppo
(`floor = max(conFMA, senzaFMA)`).

| forma | pavimento derivato | misurato su GPU | tolleranza | iterazione |
|---|---|---|---|---|
| `prefill-gemm-q5k-multirow-idot` | — | **2,61e-7** | — | it.4 |
| `prefill-gemm-q5k-multirow-f32` | — | **4,28e-7** | — | it.4 |
| `prefill-gemm-q41-multirow-idot` | 1,693e-5 | **1,73e-5** | 2e-4 (11,8×) | it.7 |
| `prefill-gemm-q41-multirow-f32` | 1,715e-5 | **1,51e-5** | 2e-4 (11,7×) | it.7 |
| `prefill-gemm-q4k-multirow-idot` | 7,841e-5 | **7,39e-5** | 1e-3 | it.8 |
| `prefill-gemm-q4k-multirow-f32` | 6,266e-5 | **3,86e-5** | 8e-4 | it.8 |
| `prefill-gemm-q6k-multirow-idot` | 8,810e-5 | **8,08e-5** | 1e-3 | it.8 |
| `prefill-gemm-q6k-multirow-f32` | 1,118e-2 | **5,63e-3** | 1,5e-1 | it.8 |
| `prefill-gemm-q80-multirow-idot` | 7,611e-4 | **5,96e-4** | 1e-2 | it.8 |
| `prefill-gemm-q80-multirow-f32` | 1,859e-4 | **2,04e-4** | 2e-3 | it.8 |

**Gli errori misurati stanno sul pavimento, non a metà strada verso la
tolleranza.** Vuol dire due cose insieme: il margine (11,8× sul q4_1) è reale ma
è **tutto pavimento**, non slack — il caso è dominato dall'aritmetica del
formato, non da un difetto del kernel; e derivare la tolleranza invece di
sceglierla è **ciò che permette di dirlo**. Un numero scelto a occhio l'avrebbe
messa troppo stretta (falso rosso permanente) o troppo larga, e in nessuno dei
due casi avremmo saputo quale dei due.

**Il discriminante regge**, ed è quello che rende il test un gate e non un
timbro: togliere `m·Σ(x)` dal kernel q4_1 porta l'errore a **5,233** (idot) e
**4,192** (f32), quattro-cinque ordini sopra la tolleranza. Per i tre formati del
35B ogni mutazione porta l'errore a oltre 10× la tolleranza
(`tests/engine-prefillkquant-f32floor.test.ts:1236` e gemelli).

**Il caso che meritava di essere sospettato, e la sua assoluzione**: il q6_K via
f32 ha pavimento relativo 1,118e-2 e tolleranza **1,5e-1** — una tolleranza
relativa del 15% ha l'aria della compiacenza. Non lo è: il caso peggiore cade
sull'uscita di modulo **più piccolo dell'intera griglia** (|ref| = 4,395e-2
contro una media di 1,340e3), dove l'errore assoluto vale 4,914e-4, dentro il
pavimento assoluto della stessa via. È una cancellazione, non un kernel
impreciso; la tolleranza è forzata dal pavimento (13,4× sopra, sotto il tetto di
20× che il test impone) e su GPU vera il misurato è la metà dell'inviluppo.

---

## 6. La clausola INTERPRETATA — GLM b12 fuori banda in meglio

Il contratto chiede il GLM b12 optimistic «**entro ±5%** di 13,172 / 31,26 /
14,74». Misurato su questo albero
(`results/engine/bench-glm-4090-b12-riga6-2026-08-15.json`, mediane su 3
ripetizioni):

    decode    13,172 -> 15,330 tok/s   +16,4%
    prefill   31,265 -> 37,542 tok/s   +20,1%
    TTFT      14.745 -> 12.279 ms      -16,7%

**Letta alla lettera, quella clausola fallisce da tutte e tre le parti.** La
leggo come **banda di RUMORE per la non-regressione** — che è il ruling
permanente di questo progetto («le metriche misurate non peggiorano mai; banda
rumore ±5%») — e non come un requisito a due code che vieta di migliorare.
Decisione mia, registrata, non un ruling richiesto: se non arrivasse nessuna
risposta farei esattamente questo.

**E non è una vittoria di questo goal.** Il confronto è **confuso dall'host
state** e lo dichiaro: stessa config sui due lati (chunked M=16, budget 12 GiB,
ctxMax 525, policy `lru`) ma **host diverso** — il riferimento
(`bench-glm-4090-b12-optimistic-nonreg-2026-08-09.json`) è
`user-session-light`, questa run è `quiescent`. Non ho modo di togliere il
confondimento senza rimisurare il riferimento su un albero che non esiste più.
Di più, spaccando il numero in due:

- il **prefill a +20,1% è plausibilmente vero**: il GLM ha 256 slot q4_1, che la
  riga 3 ha portato alla forma multi-riga, e il suo prefill passa da
  `prefillChunk`;
- il **decode a +16,4% NON è spiegabile da questo goal**: la forma multi-riga a
  M=1 perde (§4.4) ed è esclusa dal decode per costruzione. Quella parte è host,
  non codice.

Scritto così anche nel ratchet (`glmB12NonRegressione`), dove lo troverà chi
rifarà il confronto.

---

## 7. Cosa il goal consegna al successivo — le trappole, non i trofei

### 7.1 Il flag `wired` è per FORMATO, non per shape

`q4_K`, `q6_K` e `q8_0` sono **portati, verificati su GPU vera e NON cablati**:
il kernel esiste in produzione, il piano non lo instrada, e la ragione sta in
`wiredWhy` per ciascuno.

**La trappola**: il giorno in cui il goal 35B accenderà `q8_0` per i suoi tensori
attn (N=4096), **i 48 siti `ssm_alpha`/`ssm_beta` del 4B con N=32 entreranno
nello stesso istante**, perché `prefillGemmCheck` controlla kind, K e fette ma
**non guarda N**. Sono mezzo workgroup per dispatch, e sono esclusi coi numeri
dal contratto di questo goal. Serve, in quel momento, un predicato sulla shape.

Sta scritta in tre posti (`wiredWhy` del q8_0, la scheda di consegna, e qui) —
**ma una stringa non è un gate**, e chi cabla deve saperlo prima di girare il
flag. È l'unico difetto noto consegnato aperto.

**Il debito del checkpoint era peggio di come il contratto lo dichiarava.** Il
contratto chiamava `build-ttft-checkpoint.mjs:108` «due posti che decidono lo
stesso numero»; sotto c'era che **l'inventario dei siti viveva dentro un file di
TEST**, e uno script non può importare un file di test — per questo i byte erano
stati ricopiati. E il checkpoint del 2026-08-14 pubblicava **due numeri falsi**:
`gemm:deltanet-out` dichiarato `forma: "legacy"` col suo `× M` (vero quando fu
scritto, falso da quando la riga 2 ha portato le `ssm_out` a multi-riga: 16
volte i byte veri) e `gemm:qkv` contato sull'intera famiglia attn Q4_0 (80
tensori) mentre quel segmento ne cronometra 24 — **i 738 GB/s pubblicati erano
5×**. Nello stesso blocco `metrica` c'erano ancora `baselineWarmMs: 87618` e
`barraContrattoMs: 21905`, cioè i numeri del goal **engine-ttft**: il primo
checkpoint di `engine-kquant` avrebbe pubblicato una discesa di **5,108×**
invece di 1,873×, mescolando due contratti in un campo che si legge come un
risultato. Ora baseline e barra arrivano obbligatoriamente dal file di ratchet e
un test impedisce che tornino nel sorgente.

**Un residuo di quella stessa malattia è stato trovato scrivendo questo
consuntivo, e chiuso in it.11**: il checkpoint portava `"goal": "engine-ttft"` e
`"phase": "riga 5 — la discesa massima, contabilizzata"` al livello ALTO
(costanti incise nel builder) mentre `metrica.goal` diceva correttamente
`engine-kquant` — **lo stesso file si auto-attribuiva a due goal diversi**, e chi
avesse letto il primo campo avrebbe accreditato questo risultato al goal
sbagliato. Ora anche quei due campi arrivano dal ratchet (`contratto.goal` e
`contratto.fase`), come già facevano baseline e barra nel blocco accanto: il
builder non incide più nessun nome di goal. Checkpoint ricostruito e verificato
(`goal: engine-kquant`, `phase: riga 5 — la misura di chiusura, contabilizzata`,
`metrica.goal: engine-kquant`).

### 7.2 La portabilità: il difetto trovato scrivendo questo consuntivo, e chiuso

Il fabbisogno di workgroup storage è una formula esportata accanto a ogni
kernel (`PREFILL_GEMM_SPEC[kind].wgBytes`, una riga per formato e per via) e ogni
formula è **confrontata col testo WGSL generato**, non asserita a mano
(`tests/engine-prefillgemm.test.ts:518-548`). A M=16:

| formato | idot | f32 | cablato | termine proprio in `engineNeeds` |
|---|---|---|---|---|
| q4_0 | 1.152 | 4.096 | sì | sì |
| q5_K | **5.120** | 2.048 | sì | sì |
| q4_1 | 1.280 | 4.096 | **sì** | **sì, da it.11** |
| q4_K | 5.120 | 2.048 | no | **sì, da it.11** |
| q6_K | **5.632** | 2.048 | no | **sì, da it.11** |
| q8_0 | 1.152 | 4.096 | no | **sì, da it.11** |

**La prima stesura di questo consuntivo ha trovato la clausola 11 soddisfatta
solo in parte**, ed è il motivo per cui un consuntivo si scrive leggendo il
codice e non il journal: nel `Math.max` di `engineNeeds` entravano **due termini
scritti a mano** — q4_0 e q5_K — mentre i kind del prefill erano diventati sei.
Il **q4_1, che è CABLATO e in produzione dalla riga 3**, non aveva un termine
proprio; non l'avevano i tre formati del 35B.

Non rompeva niente, e il numero che lo dice è che il massimo dichiarato resta
dominato da `QWEN_WORKGROUP_STORAGE_BYTES = 30.848` (il path fuso 0.5B): 5.632 <
30.848, quindi nessun kernel chiedeva più del dichiarato e
`createComputePipeline` non falliva da nessuna parte. Ma la differenza si vedeva
sulla **soglia**: il termine q5_K era pinnato nei test perché sfonda il tetto a
**M=97** (320·97 = 31.040 > 30.848), mentre il q6_K (352·M) lo sfonda a **M=88**
— *prima* — e **nessun test se ne sarebbe accorto**, perché quel termine nel
`Math.max` non entrava.

**Chiuso in it.11, e non con un termine in più scritto a mano**: i termini ora
vengono da `PREFILL_GEMM_KINDS`, con le shape in un
`Record<PrefillGemmKind, (M) => PrefillGemmOpts>` — la stessa garanzia di
`PREFILL_GEMM_SPEC`, cioè chi allunga l'elenco senza scrivere la shape ottiene
un errore di compilazione e non un tetto sbagliato in silenzio. I test che
pinnavano la soglia sono stati riscritti sulla verità nuova: la soglia del q5_K
resta come proprietà della *sua* formula, e il valore negoziato lo alza il
termine più ripido, che è il q6_K a M=88. Più un gate nuovo che asserisce che
**ogni** kind dell'elenco abbia il suo termine e che il valore sia il loro
massimo — così il difetto non può ripetersi con un settimo formato.

### 7.3 Le altre cose già misurate che il successivo eredita

- **Il refuso corretto nel contratto** (it.8): il prefill del 35B **non** gira su
  `moeprefillplan.ts` — `planMoeChunk` ha un solo consumatore di produzione,
  `glmmodel.ts:1368`, che è il GLM. Il 35B ripete per riga la catena del DECODE
  (`q35gpumodel.ts:2743-2778`): readback CPU dei router logits per riga,
  `pinUnion` che calcola già l'unione ma solo per pinnare gli slot, poi
  `prepLayer` + `encodeExperts` per riga — 40 round-trip per chunk, 512 dispatch
  per layer. **Il lavoro è il ramo `moe` di `q35gpumodel.ts`**, non il piano
  CPU-side, che è già parametrico su `{nExpert, nExpertUsed}`.
- **Il Qwen3.5-9B ha la stessa identica struttura del 4B** (`linear_attn:Q5_K`
  276,8 MB su 24 tensori, `ffn/shexp:Q4_1` 125,8 MB su 4 — header dump, it.0):
  questa leva vale lì **senza una riga di codice in più**. Nessuna baseline 9B
  esiste, quindi è un reperto, non una promessa.
- **La guardia doppia del cablaggio non si tocca.** La condizione
  `route.via !== "legacy" && kk === "<formato>"` **ha intercettato un caso vero
  il giorno dopo essere stata scritta**: quando il piano ha accettato il q4_1
  prima che `gemvB` emettesse i suoi kernel, quei tensori sono ricaduti sulla
  legacy invece di essere letti col kernel del q4_0 — nibble senza offset e
  scale col passo sbagliato, cioè **logit storti senza nessun errore WebGPU e
  nessuna eccezione** (it.5).
- **La trappola della riga 2, evitata perché era nella spec**: `ssm_out` **non
  passa da `gemvB`**. Il ramo K-quant di `loadW` (`q35gpumodel.ts:487-505`) si
  costruisce da sé il proprio `pushB`. Cablare `gemvB` — la cosa che sembrava
  ovvia, ed è quella che il contratto stesso nominava — non avrebbe cambiato
  **una riga** di comportamento, e il test di copertura sarebbe passato lo
  stesso perché conta i SITI, non i dispatch.

---

## 8. Cosa è costato, e la regola che ne esce

### 8.1 it.6 — cinque run di GPU su un gate che era l'ambiente

Dopo l'ultima run valida (ktest 103 PASS / 0 FAIL alle 00:52 del 15 agosto),
**cinque fallimenti consecutivi con tre sintomi diversi**: `Failed to fetch`,
`Target crashed`, `A valid external Instance reference no longer exists` (device
WebGPU perso), due timeout a 600 s. Punto di rottura stabile, sempre subito dopo
`q35-mtp-head-real-blk32`, il primo caso che carica pesi reali del 4B.

L'attribuzione per esclusione è in `docket.md` item 2 (non il banco nuovo: la
run fallisce lo stesso col banco disattivato; non il lock del profilo; non il
server, `curl` risponde 200 prima e dopo; non il Chrome zombie del Playwright
MCP, ucciso e riprovato). **Il PI ha eseguito il discriminante — riavvio della
macchina — e il ktest è tornato verde al primo tentativo utile: 105 PASS / 0
FAIL, senza che fra l'ultimo fallimento e quella run fosse cambiata una riga di
codice** (stesso commit `aff171b`). Item chiuso dall'evidenza: stato accumulato
dell'host in una sessione lunga. Il banco è scagionato e non diventa lavoro suo.

**Il costo vero sono cinque run da ~10 minuti l'una, e ne bastavano tre.** La
terza — quella che disattiva il codice nuovo e vede fallire lo stesso — aveva
già chiuso l'attribuzione. Da lì la mossa corretta era una frase al PI
(«riavvia»), non altri due tentativi.

> **Regola, e vale oltre questo goal**: quando la disattivazione del codice nuovo
> non cambia l'esito, l'ipotesi ambiente è già provata. Il ritentativo non è una
> misura.

### 8.2 Il 15 agosto, tre difetti della stessa malattia

1. **`BASE_URL` mancante** nel comando di ripresa lasciato in `HANDOFF.md`: vite
   sulla 5199, il runner del ktest parla di default alla 5173. Prima cosa fatta
   alla ripresa, primo comando, fallito. Costo reale zero — il runner rifiuta
   subito e dichiara la porta, senza spendere GPU — e proprio per questo
   istruttivo: **il runner era scritto bene, ero io ad aver scritto male la
   consegna** (it.7).
2. **`--prefill-m 16` mancante nella riga EVIDENCE del contratto**: il suo
   default è `null`, il prompt è andato su `step` per posizione e ha dato
   **91.230 ms** di prefill invece di 17.126. **Una run di GPU buttata** (it.9).
   L'ho intercettata solo perché il runner **dichiara** il path
   (`prefillPath: "chunked M=16 (395 chunk + coda 12 via step)"`) invece di
   lasciarlo dedurre dai numeri — una decisione presa nella riga 0 del goal
   `engine-ttft`, che quel giorno ha pagato per la prima volta.
3. **Un `--help` dato a un runner che non lo conosce**, che ha eseguito il bench
   coi default **e ucciso una run di conformità in corso** (HANDOFF.md §1).

> **La regola che ne esce, e sta in `HANDOFF.md` perché è lì che serve: un
> comando lasciato per la ripresa si esegue come sta scritto PRIMA di lasciarlo,
> e i flag di un runner si leggono dal SORGENTE — mai eseguendolo.**

I flag non opzionali oggi noti: `BASE_URL` sul ktest; `--prefill-m 16` sul bench;
`--conf-prefill-m` sul conf, che **non** è `--prefill-m` (quest'ultimo esce al
gate dei due bracci senza arrivare al replay golden); e `--prefill-batch` del GLM
che è un **booleano 0/1**, non una M.

### 8.3 Altre note di processo, registrate

- **it.1: ho committato prima che il verificatore rientrasse**, invertendo i
  passi 4 e 6 del protocollo. Il contenuto verificato era byte-identico, ma
  l'ordine era sbagliato.
- **it.2: ho chiuso una riga su un done-when che avevo ristretto io** (§4.4), e
  una delle due restrizioni violava un CONSTRAINT esplicito del contratto. Il
  test «se non arrivasse mai un ruling farei così» **non si applica a ciò che il
  contratto copre in senso opposto**.
- **it.5: il workflow ha consegnato tre task su quattro** lasciando in albero un
  test in fase rossa (22 falliti) — la forma TDD portata a metà. Il quarto l'ho
  scritto io.
- **it.10: leggere l'artefatto prima di progettare la misura.** Il piano diceva
  di confrontare `perPrompt[i].engineArgmax` fra i due artefatti di conformità;
  **quel campo non esiste** — il worker lo sposta al livello alto in
  `engineArgmaxByPrompt` (`q35conf.worker.ts:722-723`). Scoperto guardando la
  struttura PRIMA di lanciare il secondo braccio: dopo, avrei avuto due run buone
  e nessun modo di confrontarle senza rifarne una.

---

## 9. Il conto finale

| | prima (2026-08-14) | dopo (2026-08-15) |
|---|---|---|
| TTFT a caldo | 32.127 ms | **17.153 ms** (1,873×) |
| prefill | 32.101 ms · 197,25 tok/s | **17.126 ms · 369,72 tok/s** |
| `decode.firstMs` | 26 ms | 27 ms |
| primo segmento | `gemm:deltanet-out` 37,91% | **`deltanet:recurrence` 10,11%** |
| dentro / fuori i pass GPU | 70,9% / 29,1% | **46,5% / 53,5%** |
| copertura del piano, M=16 | 5,8593× · 172/248 siti | **15,5247× · 200/248 siti · 99,796% dei byte** |
| formati sulla via veloce | 1 (q4_0) | **3 cablati** (q4_0, q5_K, q4_1) **+ 3 portati e non cablati** |
| ktest | 101 PASS / 0 FAIL | **111 PASS / 0 FAIL** |
| vitest | 680 passed | **1017 passed \| 10 skipped** |

**La leva è esaurita, e la misura lo dice invece di lasciarlo dedurre.** Restano
legacy 48 siti su 248, lo **0,204% dei byte**, esclusi coi numeri: il tetto
teorico che resta sul traffico dei pesi è 16× contro i 15,5247× già presi. Il
prossimo millisecondo di TTFT non sta lì.

**Sta nel 53,5% che non è più calcolo** (§3.2), e in seconda battuta nella
ricorrenza DeltaNet a 1,7 s (§3.1). Il cablaggio del 35B — già deciso dal PI come
priorità successiva — è un altro problema ancora: lì il collo è la **residency**
di 17,67 GB di expert su 16 GiB di scheda, non il kernel, e le forme di kernel
arrivano da qui già misurate e verificate
(`docs/engine/kquant-consegna-35b-2026-08-15.md`).
