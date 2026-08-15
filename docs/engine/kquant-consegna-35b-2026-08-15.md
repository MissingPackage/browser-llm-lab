# Scheda di consegna — da `engine-kquant` al goal 35B — 2026-08-15

**Il collo del 35B non è il kernel. È la residency.** Gli expert pesano
17.666.408.448 B = **17,67 GB = 16,45 GiB**
(`results/engine/q35-header-dump-2026-08-10.json`, entry `[2]`,
`typeHistogram["expert:Q4_K"].bytes`); il modello intero ne fa 20.882.024.960 =
**20,88 GB = 19,45 GiB** (somma dell'istogramma). La scheda di sviluppo è una
RTX 4090 mobile con **~16 GB di VRAM**
(`docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md:22`; il
problema è già nominato così in
`docs/superpowers/specs/2026-07-31-engine-fase-c2-design.md:99` — «il MINIMO che
fa girare 17 GB in 16 GiB»). Chi apre questo goal partendo dai kernel lavora sul
termine sbagliato.

Questo documento consegna tre cose per ciascuna delle tre forme misurate
(Q4_K, Q6_K, Q8_0): **i numeri della fase 0**, **le shape vere e la loro
provenienza**, **cosa manca per cablarle**. Chiude la riga 4 di
`.harness/goals/engine-kquant/PHASES.md`.

> **Nota sui riferimenti di riga.** I numeri di riga di
> `src/engine/kernels/wgsl.ts` e `src/engine/prefillgemmplan.ts` sono verificati
> il 2026-08-15 su un albero in cui la riga 4 del goal stava **atterrando** su
> quei due file. I nomi dei simboli citati sono stabili; le righe vanno
> ri-verificate.
>
> **Cosa la riga 4 ha già consegnato** (albero al 2026-08-15): le tre forme
> multi-riga sono **portate dal banco dentro `wgsl.ts`** (sezione «PREFILL GEMM
> q4_K / q6_K / q8_0 SPLIT-K — port dal banco», `wgsl.ts:4842`), col testo WGSL
> riga per riga del banco e l'ordine dei binding congelato e dichiarato lì. Sono
> **misurate e non instradate**: `PREFILL_GEMM_SPEC` porta per ciascuna
> `wired: false` con la sua ragione (`wgsl.ts:4108`, `:4142`, `:4180`), e
> `planPrefillGemm` rifiuta con `from: "wiring"` prima ancora di guardare la
> geometria (`prefillgemmplan.ts:216-226`). L'elenco dei formati **instradati**
> è `PREFILL_GEMM_WIRED_KINDS` (`wgsl.ts:4203`), che da qui in poi è diverso da
> `PREFILL_GEMM_KINDS` — non vanno confusi.

---

## 1. I numeri della fase 0

Artefatto: `results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json`
— `kind: "microbench-kquant-fase0"`, `deviceLabel: "4090-linux"`,
`hostState.declared: "quiescent"`, `prereg:
"docs/deep-dive/kquant-fase0-prereg-2026-08-14.md"`. Celle della famiglia
`gemm-kquant-multirow`: **54**, `skipped: []`. Tempi in `msPerOp.p50`,
`timingSource: "timestamp-query"`.

Braccio di paragone («legacy», id `base-batch-z`): il kernel di **produzione**
importato, non una sua imitazione — `gemvQ4KWgsl`/`gemvQ6KWgsl` con
`batch: true` e `gemvQuantWgsl({kind:"q8_0", batch:true})`
(`src/microbench/ttKQuant.ts:1137-1141`). È la forma che dispaccia M GEMV su
`wid.z`: ogni riga rilegge l'intera matrice, riuso dei pesi zero.

### 1.1 Rapporto legacy / veloce a M = 16 — la tabella della regola di stop

| famiglia, shape `[K,N]` | tensore del 35B | legacy (ms) | `idot` (ms) | **rapporto** | `f32` (ms) | **rapporto** |
|---|---|---|---|---|---|---|
| Q4_K `[2048, 512]` | expert gate/up | 0,0700 | 0,0168 | **4,16×** | 0,0615 | **1,14×** ❌ |
| Q4_K `[512, 2048]` | expert down | 0,0495 | 0,0095 | **5,23×** | 0,0268 | **1,85×** |
| Q6_K `[512, 2048]` | expert down di 3 layer | 0,1529 | 0,0250 | **6,13×** | 0,0922 | **1,66×** |
| Q8_0 `[2048, 4096]` | attn q-proj | 1,3224 | 0,0376 | **35,20×** | 0,0750 | **17,63×** |

**Regola di stop applicata (barra 1,5× sulla legacy, done-when della riga 1 del
contratto): sette esiti su otto la passano; ne cade uno.**

- **Passano** tutte e quattro le vie intere (`idot`), da 4,16× a 35,20×.
- **Passano** le vie `f32` di Q4_K `[512,2048]` (1,85×), Q6_K (1,66×) e Q8_0
  (17,63×).
- **NON passa** la via `f32` del **Q4_K `[2048,512]`: 1,14× contro 1,5×.** Su un
  device senza `packed_4x8_integer_dot_product` la forma multi-riga, su
  quella shape, **non paga**. Le due shape del Q4_K vanno quindi trattate
  separatamente nel piano: gate/up ha una sola via ammissibile (`idot`), down ne
  ha due. È il reperto che i fallback f32 hanno comprato (journal it.3,
  `.harness/goals/engine-kquant/journal.md:282-288`); nel disegno originale
  quei tre kernel non sarebbero stati scritti.

Nessuna famiglia è chiusa dalla regola di stop: **nessuna delle tre è
"esclusa coi numeri"**. Il cablaggio resta aperto per tutte e tre, con la
riserva del §3.

### 1.2 M = 1 e M = 8 — e il divieto che ne esce

Rapporto `idot | f32`, stesso artefatto:

| famiglia, shape | M=1 | M=8 | M=16 |
|---|---|---|---|
| Q4_K `[2048,512]` | **0,91** \| 0,20 | 3,20 \| 0,88 | 4,16 \| 1,14 |
| Q4_K `[512,2048]` | 1,38 \| 0,57 | 3,95 \| 1,53 | 5,23 \| 1,85 |
| Q6_K `[512,2048]` | 1,29 \| 0,41 | 4,38 \| 1,36 | 6,13 \| 1,66 |
| Q8_0 `[2048,4096]` | 3,26 \| 3,21 | 22,68 \| 13,66 | 35,20 \| 17,63 |

**A M=1 la forma multi-riga PERDE sul Q4_K gate/up (0,91×)** e perde su tutte e
tre le vie f32: il combine dei parziali non si ammortizza su una riga sola.
Vale come divieto per il cablaggio — **il piano non deve offrire queste forme al
decode**, che sul 35B è il regime dominante (40 layer × 8 expert = 320
selezioni per token, `results/engine/q35-vramplan-35b-it35.json`,
`selectionsPerToken`).

### 1.3 Cose dell'artefatto che servono a chi cabla

- **Fabbisogno di workgroup storage** (`workgroupStorageBytes`), a M=16:
  Q4_K 5.120 B · Q6_K **5.632 B** (il massimo delle sei famiglie: ha un array in
  più per le somme a mezzo sotto-blocco) · Q8_0 1.152 B. Tutti sotto i 16.384 B
  garantiti da WebGPU, a ogni M ≤ 16. La formula va comunque nel `Math.max`
  calcolato di `QWEN_WORKGROUP_STORAGE_BYTES` (vincolo di portabilità del
  contratto).
- **Split-K**: Q4_K `[2048,512]` e Q8_0 girano a **4 fette**; le due shape con
  K=512 a **2 fette** — due superblocchi per riga, a 4 fette il kernel
  rifiuterebbe (correzione C0-4, `PHASES.md`). Il campo `notes` di ogni cella
  lo dichiara.
- **Scarto aritmetico** (`checksumRelDiff`, contro il GEMV di produzione della
  famiglia): il più alto è il **Q8_0 idot, 1,38e-2** — il kernel senza unpack,
  senza offset e senza termine costante. Il residuo scala col numero di termini
  accumulati (K·N = 8,4 M prodotti), non con la complessità del formato.
  Q6_K 1,12e-2, Q4_K 1,44e-4 … 2,08e-3. Le vie f32 stanno a 1e-8 … 1e-7.
- **Il Q8_0 legacy è la cella più rumorosa dell'artefatto**: `p50` 1,3224 ms
  contro `min` 1,0323 (e a M=8, `p50` 1,2830 contro `min` 0,6682). Sul `min` il
  rapporto scende da 35,20× a 34,4× — la conclusione non cambia, ma chi
  ricalcola non deve sorprendersi.

---

## 2. Le shape vere del 35B, e da dove vengono

Fonte: `results/engine/q35-header-dump-2026-08-10.json`, entry `[2]`, file
`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`, 733 tensori. Dal blocco `meta`:

| chiave GGUF | valore | uso |
|---|---|---|
| `qwen35moe.embedding_length` | **2048** | dModel — la K di gate/up e la N del down |
| `qwen35moe.expert_feed_forward_length` | **512** | dFfnExpert — la N di gate/up e la K del down |
| `qwen35moe.expert_count` | **256** | expert per layer |
| `qwen35moe.expert_used_count` | **8** | top-K |
| `qwen35moe.block_count` | **40** | layer, tutti MoE (`denseLead: 0`, `src/engine/q35expertstore.ts:74`) |

Dall'istogramma `typeHistogram` della stessa entry (byte esatti, non
arrotondati):

| voce | n tensori | byte | GB |
|---|---|---|---|
| `expert:Q4_K` | 117 | 17.666.408.448 | **17,67** |
| `attn:Q8_0` | 100 | 1.091.829.760 | **1,09** |
| `expert:Q6_K` | 3 | 660.602.880 | **0,66** |
| `embd/head:Q8_0` | 1 | 540.344.320 | **0,54** |
| `embd/head:Q6_K` | 1 | 417.177.600 | 0,42 |
| `linear_attn:Q8_0` | 30 | 267.386.880 | **0,27** |
| `ffn/shexp:Q8_0` | 120 | 133.693.440 | 0,13 |
| (resto F32) | 361 | 104.581.632 | 0,10 |

**Tutti i numeri della sezione «IL 35B» di `GOAL.md:55-57` sono verificati
contro l'header dump: 17,67 · 0,66 · 1,09 · 0,27 · 0,54 GB, dModel 2048,
dFfnExpert 512, nExpert 256.** Il 35B **non ha un byte di q4_0**: i formati che
il piano instrada davvero sono `q4_0`, `q5_K` e `q4_1`
(`PREFILL_GEMM_WIRED_KINDS`, `src/engine/kernels/wgsl.ts:4203`), e ne coprono lo
**0%**.

Le shape misurate corrispondono ai siti veri, e i siti sono in codice:

- **Q4_K `[2048,512]`** — `pGate`/`pUp` della catena expert:
  `gemvQ4KWgsl({ K: d, N: dE, arena })`, `src/engine/q35gpumodel.ts:1418-1419`.
- **Q4_K / Q6_K `[512,2048]`** — `pDown`, con `accum: true`:
  `src/engine/q35gpumodel.ts:1423-1425`. La classe di slab si sceglie sul
  formato del down (`src/engine/q35expertstore.ts:55-70`): **37 layer in classe
  q4_K, 3 in classe q6_K**.
- **Q8_0 `[2048,4096]`** — pesi statici, `loadW` ramo Q8_0
  (`src/engine/q35gpumodel.ts:487-489`) → `gemvB(..., "q8_0")`, che oggi cade
  sul fallback legacy `gemvQuantWgsl({kind:"q8_0", batch:true})`
  (`src/engine/q35gpumodel.ts:941-943`).

Byte per tensore sul device, dall'artefatto (`weightBytesUnique`): Q4_K
**589.824 B** (144 B per 256 pesi) · Q6_K **868.352 B** (212 B sul device, non i
210 del file: il repack allinea il superblocco alla parola) · Q8_0
**8.912.896 B**.

---

## 3. La cosa che chi legge deve sapere PRIMA di usare i numeri del §1

**Il 4-6× di Q4_K e Q6_K non è un limite inferiore. È un'affermazione già
ritrattata dentro questo goal**, in it.3
(`.harness/goals/engine-kquant/journal.md:305-324`), dopo che il digest di it.2
l'aveva scritta al contrario. Le ragioni della ritrattazione, che restano
valide:

1. **Il braccio legacy misurato non è il percorso di produzione degli expert.**
   Il motore instrada gli expert in **regime d'ARENA** con `accum`
   (`q35gpumodel.ts:1418-1425`), e la combinazione `batch && arena` è **vietata
   per costruzione**:
   - `src/engine/kernels/wgsl.ts:2175` —
     `if (batch === true && arena) throw new Error("q4K: batch e arena non si combinano (l'arena e' il path expert, il batch il prefill)");`
   - `src/engine/kernels/wgsl.ts:2359` — la stessa riga per il q6K.

   La GEMV `batch: true` su binding diretto, su quei 117 tensori, il motore
   **non la emette e non può emetterla**. Il commento del banco lo dice già al
   condizionale: «è ciò che il motore *emetterebbe* oggi su questi tensori»
   (`src/microbench/ttKQuant.ts:1150`).
2. **Nel prefill MoE la premessa stessa della leva cade**: token diversi
   selezionano expert diversi, quindi «la stessa matrice riletta M volte» non
   c'è. Quantificato: con 256 expert, top-8 e routing uniforme, l'unione degli
   expert di un chunk da M=16 righe è
   `256·(1−(1−1/256)^128) ≈ 101`, cioè **molteplicità media 1,27 righe per
   expert** (128/101). *Calcolo mio dagli ingressi qui sopra, non una misura —
   e il routing vero non è uniforme: l'istogramma `moe.routing` di
   `results/engine/q35-vramplan-35b-it35.json` è marcatamente sbilanciato.*
   Il fattore di riuso a M=16 sul path expert vale **~1,27×**, non 16×.
   Per confronto: M=8 → 1,13× · M=32 → 1,58× · M=64 → 2,31×.
3. Il crollo da 28× a 5× ha **due** cause di peso simile: sulle shape piccole
   (0,59-0,87 MB, che stanno in cache) il legacy è ~2,2× più veloce per peso, e
   la forma veloce è ~2,3× più lenta (32-64 workgroup su 128 SM: fame di
   parallelismo). Solo la prima migliora in produzione.

**Conclusione operativa: il 4-6× vale per una shape su un braccio ipotetico. Non
è né un tetto né un pavimento del guadagno atteso sul 35B.** L'unico dei tre
numeri che si trasferisce così com'è è il **Q8_0**: i tensori di attn sono
statici, densi, letti da tutte le M righe, e il braccio legacy misurato è
*letteralmente* quello che `gemvB` emette oggi (`q35gpumodel.ts:941-943`).

---

## 4. Cosa manca per cablare, forma per forma

### 4.1 Q8_0 attn `[2048,4096]` — la sola che si trasferisce 1:1

Il kernel **esiste già in produzione** (`wgsl.ts:4842` e seguenti, port dal
banco) e non è instradato. **Prima di fidarsene, verifica il gate**: al momento
in cui scrivo `src/engine/ktest/**` non era ancora toccato dalla riga 4, quindi
il caso ktest su GPU vera per le tre forme — clausola della riga 4 — non
risultava eseguito. Fatto quello, per il Q8_0 manca:

- **girare `wired` a `true`** per `q8_0` in `PREFILL_GEMM_SPEC`
  (`wgsl.ts:4180`), sostituendo la `wiredWhy` con la ragione nuova. Attenzione:
  la ragione attuale **non è "non serve"**, è numerica e riguarda il 4B — i 48
  siti `ssm_alpha`/`ssm_beta` hanno **N=32** contro le 64 righe per workgroup
  della forma split-K, sono lo 0,204% dei byte del prefill del 4B, e
  `prefillGemmCheck` **non guarda N** (controlla kind, K e fette). Girare il
  flag senza aggiungere una condizione sulla shape **cambia ciò che il 4B
  esegue oggi**, e lo cambia in peggio. Sul 35B N vale 4096 e l'esclusione non
  si applica: la distinzione va fatta sulla shape, non sulla famiglia;
- **un terzo ramo nel sito emittente**: `gemvB` (`q35gpumodel.ts:852`) ha già la
  forma a due guardie usata per q4_0 e q4_1 — rotta chiesta al piano **più**
  verifica del formato nel punto in cui si emette il kernel. Serve un ramo
  `kk === "q8_0"` esplicito, non un ternario dentro quelli esistenti: il gate
  strutturale legge il sorgente e verifica che il kernel sia emesso con gli
  stessi `opts` con cui si è chiesta la rotta.

Questa è anche la sola delle tre che tocca il **prefill denso**, cioè il piano
che il 4B usa già: nessuna interazione con l'arena.

### 4.2 e 4.3 Q4_K `[2048,512]` / `[512,2048]` e Q6_K `[512,2048]` — expert

**La forma portata dalla riga 4 non è la forma che serve al path expert.** Le
due forme `q4_K`/`q6_K` split-K stanno in `wgsl.ts:4842` e seguenti, con
l'ordine dei binding congelato (`[0 blocks, 1 xq, 2 part, 3 xsc]` per la via
intera, `[0 blocks, 1 x, 2 part]` per la f32) — ma misurano un GEMM multi-riga
su M righe *contigue* contro *la stessa* matrice. Il path expert ha bisogno di
una **forma a gather**: un dispatch per expert, sulle sole righe del chunk che
lo selezionano. Il precedente esiste già, funziona, ed è in produzione — **su
GLM, non sul 35B**:

- `pairGemvSiluGatherWgsl` (`wgsl.ts:3221`) — gate+up+silu sulle righe raccolte,
  riga presa da `wid.z` via il buffer `gather` (`u32: m | k<<16`);
- `gemvDownSlotsWgsl` (`wgsl.ts:3303`) — il down **scrive** `y[m][k]` non
  pesato, in slot separati;
- `moeCombineWgsl` (`wgsl.ts:3381`) — `xM += sM + Σ_k w·y` in ordine k, la
  stessa catena di somme del decode (è il contratto di identità bit-a-bit
  descritto in `src/engine/moeprefillplan.ts:29-36`);
- cablaggio: `src/engine/glmmodel.ts:650-653` (pipeline) e `:1368-1412` (il
  loop per expert dell'unione, con bind group a sotto-range dello slot).

Cosa manca, in concreto, perché quella forma valga sul 35B:

1. **Le tre versioni K-quant dei due kernel a gather non esistono.**
   `gemvDownSlotsWgsl` accetta `kind: "q4_0" | "q4_1"` per firma
   (`wgsl.ts:3303`); `pairGemvSiluGatherWgsl` ha il layout q4_0 **cablato nel
   corpo** (`gQs4: array<vec4<u32>>` + `gScales`, nibble − 8). I K-quant hanno
   un buffer solo (`blocks`) e un superblocco da 256. È qui che l'aritmetica
   verificata dalla riga 4 di questo goal si riusa — unpack, offset, termine
   `Σx` — **non la geometria multi-riga**.
2. **I due kernel a gather sono cablati a top-4.** `m * 4u` è scritto a mano
   negli indici di `hSlots` e `ySlots` (`wgsl.ts:3296`, `:3351`, `:3375`),
   mentre `moeCombineWgsl` è già parametrico (`nUsed`, default 4). Il 35B ha
   **top-8**: vanno parametrizzati i primi due.
3. **Servono bindings a sotto-range su slab K-quant** — e ci sono già:
   `slotTensorRanges` (`src/engine/residency.ts:189`) esiste esattamente per
   questo, mentre la vista legacy a sei range `slotBindRanges` **lancia** sui
   K-quant. È la funzione che i call site nuovi devono usare.
4. **In alternativa, la forma a gather in regime d'arena**: il commento di
   famiglia di `wgsl.ts:3216-3218` dice «Varianti PLAIN (bindings espliciti): il
   modo arena arriva col wiring». Se si sceglie l'arena invece dei sotto-range,
   il divieto `batch && arena` di `wgsl.ts:2175`/`:2359` va **rimosso con una
   ragione scritta**, non aggirato: oggi è corretto perché protegge da una
   combinazione che nessuno ha misurato. E il conto dei binding va rifatto —
   `expertArenaBindings(geo.nBuf)` è già confrontato con
   `maxStorageBuffersPerShaderStage` e lancia se sfora
   (`q35gpumodel.ts:1371-1377`): un kernel a gather aggiunge almeno il buffer
   `gather` e lo slot di uscita.

### 4.4 Il punto d'aggancio, e perché `moeprefillplan.ts` da solo non basta

**`src/engine/moeprefillplan.ts` non è cablato sul 35B.** L'unico consumatore di
`planMoeChunk` in produzione è `src/engine/glmmodel.ts:1368`; da `q35gpumodel.ts`
non lo chiama nessuno (`grep planMoeChunk src/` → glmmodel, ktest.worker,
i test). Il file è già **parametrico sulla famiglia** (`MoePlanShape =
{nExpert, nExpertUsed}`, `moeprefillplan.ts:45-48`) e la config del 35B lo
soddisfa per struttura: `planMoeChunk(sels, 16, {nExpert: 256, nExpertUsed: 8})`
funziona così com'è. **Il piano CPU-side non è il lavoro. Il lavoro è il
consumatore GPU-side.**

Il consumatore oggi è `q35gpumodel.ts:2743-2778`, dentro `prefillChunk`. Per
ogni layer MoE:

1. `runSegB(...)` chiude il segmento statico e copia `routerLogits` nello
   staging;
2. `readStaging(cr.routerStagingM, M_MAX * nE * 4)` — **un readback CPU per
   layer**: 16 × 256 × 4 = 16.384 B, **40 round-trip GPU→CPU per chunk**
   (`:2757-2758`);
3. `moe.pinUnion(li, lgAll, M_MAX, nE)` — l'unione degli expert del chunk viene
   già calcolata, ma **solo per pinnare gli slot**, non per raggrupparvi i
   dispatch (`:2761`, corpo a `:2024-2032`);
4. `for m2 in 0..M_MAX`: `await moe.prepLayer(li, …, m2, pinAll)` — selezione,
   `ensure`, `writeBuffer` della `Sel` **per riga**, con un `await` dentro il
   loop; poi due `copyBufferToBuffer` che portano la riga nei buffer per-riga, e
   `moe.encodeExperts(pz, li, cr.bgAddRow[m2], m2)` (`:2763-2774`).

`encodeExperts` (`:2086-2105`) è **la catena del decode ripetuta M volte**:
`topK` × (gate, up, silu, down) + un add. Sul 35B, per layer e per chunk da
M=16: **16 × 8 × 4 = 512 dispatch + 16 add**, cioè **20.480 + 640 per chunk** su
40 layer. Con la forma a gather diventerebbero `2 × |unione| + 1 ≈ **203** per
layer (~8.120 per chunk): **circa 2,6× in meno di dispatch**, più il ~1,27× di
traffico pesi del §3. *Conti miei dalle strutture citate; nessuno dei due è
misurato sul 35B.*

**Dove va agganciata la forma multi-riga, in concreto**: al posto del loop
`for m2` di `:2763-2774`, con `planMoeChunk` che produce `experts[]` (rows +
slots) e `weights`, un `writeBuffer` del `gather` per expert e un dispatch per
expert — esattamente la struttura di `glmmodel.ts:1388-1411`. **Cosa lo blocca
oggi**: (a) i kernel a gather K-quant non esistono (§4.2); (b) sono cablati a
top-4 (§4.2.2); (c) il down del 35B usa `accum: true` con il peso letto dalla
`Sel`, mentre il contratto a slot pretende che il down **scriva** non pesato e
che sia la combine a sommare in ordine k — sono due contratti diversi sullo
stesso tensore, e il gate del 35B è la **bit-identità** col path sequenziale
(`results/engine/q35-prefillmoe-35b-it34.json`, `gate.bitIdentical: true` su
993.280 valori); (d) il readback per layer resta comunque, perché la selezione
la fa la CPU.

### 4.5 Residency — il termine che decide il goal

Ingressi verificati: expert 17,67 GB su una scheda da ~16 GB (§ intestazione);
budget d'arena di default **12 GiB** (`q35gpumodel.ts:396`,
`arenaBudgetBytes = 12 * (1 << 30)`), che è anche quello dell'ultima misura sul
35B (`results/engine/q35-vramplan-35b-it35.json`, `arenaGiB: 12`); il budget
vero è **calcolato** dal tetto meno KV, work e riserva
(`src/engine/residency.ts:403`), quindi a ctx lungo scende ancora.

Aritmetica derivata da quegli ingressi (**calcolo mio**, con la riserva che
`layout(c).bytes` può includere allineamenti che non ho letto):

- slab per expert: classe q4_K = 3 × 589.824 = **1.769.472 B**; classe q6_K =
  2 × 589.824 + 868.352 = **2.048.000 B**;
- parco: 37 × 256 = 9.472 expert in classe q4_K, 3 × 256 = 768 in classe q6_K,
  **10.240 in tutto**;
- residency piena, in byte di **device**: 18.333.302.784 B = **17,07 GiB**
  (6.291.456 B in più dei 18.327.011.328 di file: è il pad del Q6_K, 212 B per
  superblocco invece di 210);
- a 12 GiB, col riparto pro-quota di `expertSlots` (`residency.ts:294-314`):
  **6.735 slot su 9.472 (71,1%)** in classe q4_K e **471 su 768 (61,3%)** in
  classe q6_K.

E il dato che rende la cosa un problema di prefill e non di decode: **un chunk
da 16 token tocca ~101 expert distinti per layer**, cioè ~178 MB per layer e
**~7,14 GB di pesi distinti per chunk** su 40 layer — più della metà dell'arena
ricambiata a ogni chunk. Il costo dei miss è misurato:
`q35-vramplan-35b-it35.json` a freddo fa **3.341 miss su 39 token**,
**5.965.004.800 B caricati**, `packMs` **1.857** e `uploadMs` **442**.

---

## 5. Cosa NON c'è, e va prodotto per primo

- **Non esiste una baseline del 35B dopo i kernel nuovi.** Gli artefatti del 35B
  sono tutti del **2026-08-11**: `q35-prefillmoe-35b-it34.json` (prefill a
  chunk, `chunkM: 8`, 32 token — chunked 34,95 ms/token contro 131,08
  sequenziale, **3,75×**), `q35-vramplan-35b-it35.json` (decode: sync-warm
  134,10 ms/token, optimistic-warm **44,32** ms/token), `q35-sensors-35b-it38`,
  `q35-onepass-35b-it29`. Il Q5_K e il Q4_1 sono entrati in produzione il
  **2026-08-15** (journal it.4 e it.7). **Nessuno di quei numeri descrive il
  motore di oggi**, e nessuno è sul prompt da 6333 token né a M=16. La prima
  cosa che il goal 35B deve produrre è una misura fresca con host dichiarato e
  i tre termini scomposti — non per completezza: perché senza non si sa quale
  dei termini del §4 sia il primo.
- **Non è misurato** quanto pesi il readback per layer (§4.4 punto 2) dentro il
  prefill del 35B. Sul 4B la sonda per segmento esiste già
  (`prefillGpuTime`, `q35gpumodel.ts:2141-2149`); sul MoE le categorie ci sono
  ma non sono mai state lette su una run lunga.
- **Non è verificato** che le forme multi-riga della fase 0 compilino in regime
  d'arena: nessuno le ha mai generate con `arenaHeadWgsl`, e il divieto di
  `wgsl.ts:2175`/`:2359` ha impedito perfino di provarci.
- **Non risultava eseguito**, al momento di scrivere, il caso **ktest su GPU**
  delle tre forme portate: `src/engine/ktest/**` non era fra i file toccati
  dalla riga 4. È la prima cosa da controllare in `git log` prima di costruirci
  sopra.
- **Non è verificato** che l'unione degli expert di un chunk stia nell'arena a
  ctx lungo: i 7,14 GB per chunk del §4.5 sono un conto, non una misura, e il
  budget calcolato a ctx 6333 non è mai stato ispezionato sul 35B.

---

## 6. Ordine di lavoro che questi numeri suggeriscono

Non è un ruling: è la lettura di chi ha misurato.

1. **Baseline fresca del 35B** (prefill a chunk + decode, host dichiarato,
   termini scomposti, budget d'arena dichiarato). Senza, ogni ottimizzazione è
   una scommessa.
2. **Residency**: il conto del §4.5 dice che il prefill a chunk ricambia più di
   metà dell'arena per chunk. Se il numero regge alla misura, è lì che sta il
   tempo — e nessun kernel lo tocca.
3. **Q8_0 attn** (§4.1): il solo cablaggio in cui i numeri della fase 0 si
   trasferiscono, 1,09 GB di pesi, lavoro contenuto e già nella forma dei due
   rami esistenti.
4. **Forma a gather K-quant per gli expert** (§4.2-4.4): è il pezzo grosso, e
   il suo guadagno atteso è ~2,6× di dispatch e ~1,27× di traffico pesi — non i
   4-6× della tabella del §1.
