# MEMO fase 0 — `engine-kernel-decode`: le leve esistono, e la più grande non è quella che il piano indicava

Verdetto in una riga: **la regola di stop NON scatta**. Il migliore speedup della
fase è **33,8×** (attenzione), contro la soglia di 1,5× che avrebbe chiuso il
goal. 16 enunciati pre-registrati su 17 confermati; l'unico caduto è caduto
_a favore_ del progetto.

- pre-registrazione: `docs/deep-dive/kernel-decode-fase0-prereg-2026-08-13.md`
  (commit `1d08061`, **prima** di qualunque esecuzione)
- dati: `results/microbench/kernel-decode-fase0-4090-linux-2026-08-12T23-49-24-010Z.json`
  (32 celle, 0 skipped, timing `timestamp-query` su tutte)
- grade meccanico: `node scripts/kd-grade.mjs`
- harness: `src/microbench/{kdAttn,kdGemv,kdRunner,kdPage}.ts`, `kdbench.html`,
  `scripts/kd-microbench-run.mjs`. **Nessun file di `src/engine/**` toccato**: la
  forma attuale dei due kernel è *importata* da `src/engine/kernels/wgsl.ts`, non
  ricopiata.
- host: 4090-linux, `hostState.declared = quiescent`, Chrome/Linux, NVIDIA
  Lovelace, subgroup size 32.

## 1. Le due varianti vincenti

| kernel | forma attuale | variante vincente | fattore |
|---|---|---|---|
| `attnDecodeWgsl` n=6333 | **9,980 ms**/dispatch · 5,20 GB/s su byte unici | **`split`** — softmax in streaming + vec4 + contesto spezzato su 16 workgroup per head + combinazione log-sum-exp | **33,7×** → 0,296 ms · 175,3 GB/s unici, 701,2 GB/s emessi |
| `gemvQuantWgsl` q4_0 lm_head K2560×N248320 | **108,6 G pesi/s** | **`vec4-rows2-sg`** — una load `vec4<u32>` per blocco + `dot()`, 2 righe per workgroup, una riga per subgroup, riduzione con un solo `subgroupAdd` | **4,00×** → 434,2 G pesi/s |

Il GEMV vincente vince su **tutte e quattro** le forme (4,07× · 4,27× · 3,96× ·
4,00×) — non è un artefatto di una taglia.

## 2. Il grade, enunciato per enunciato

| # | enunciato | esito | misura |
|---|---|---|---|
| P0 | la regola di stop non scatta (≥ 1,5×) | **CONFERMATO** | migliore 33,84× |
| P1a | `subgroups` presente, `subgroupAdd` compila e dà il risultato giusto | **CONFERMATO** | esposta+concessa, `subgroupAdd(0..63)=2016`, sgSize 32 |
| P1b | `shader-f16` ASSENTE | **CONFERMATO** | `extension 'f16' is not allowed in the current environment` |
| P1c | `packed_4x8_integer_dot_product` presente [confidenza bassa] | **CONFERMATO** | in `wgslLanguageFeatures`, `dot4I8Packed` esegue e dà 70 |
| P1d | subgroup-matrix esposta ma **non istanziabile** | **CADUTO** | esposta **e concessa**; config `u8→u32` e `i8→i32` 16×16×32 e 16×8×32 |
| P1d' | ...nessuna config con risultato f32 | confermato (lettura letterale di "u8/f32") | nessun `result: f32` fra le quattro |
| P2 | baseline 8,23 ms ± 40% (banda 4,9–11,5) | **CONFERMATO** | 9,980 ms, IQR 0,219 |
| P2' | NON molto più veloce (< 4,1 ms) ⇒ il bersaglio è giusto | **CONFERMATO** | 9,980 ≫ 4,1 |
| P3a | migliore variante di attenzione ≥ 4,0× | **CONFERMATO** (stima puntuale 6–8× **sottostimata di 4×**) | 33,72× |
| P3b | lo split è più veloce della fusione GQA e delle vec4 prese da sole | **CONFERMATO** | split 0,296 · gqa-stream 4,128 · vec4 2,639 ms |
| P4a | lm_head base 133 G pesi/s ± 25% | **CONFERMATO** | 108,6 |
| P4b | migliore variante GEMV ≥ 1,5× | **CONFERMATO** (stima puntuale 2–2,5× sottostimata) | 4,00× |
| P4c | resta sotto i 500 di llama.cpp Vulkan | **CONFERMATO** | 434,2 < 500 (87% del riferimento esterno) |
| P4d | le tre forme cache-resident stanno sopra 133 G pesi/s | **CONFERMATO** | 133,9 · 143,7 · 157,1 |
| P5 | l'attenzione guadagna più del GEMV ⇒ ordine delle fasi 1/2 | **CONFERMATO** | 33,72× vs 4,00× |
| gate | checksum entro 1e-3 relativo alla base | **PASSA** | max relDiff **2,99e-6** su 32 celle |

**P2 è l'enunciato che vale di più**: l'ADDENDUM it.59 attribuiva i 10,4
µs/posizione ad `attnDecodeWgsl` lasciando l'attribuzione `[DA VERIFICARE]`. La
forma attuale, isolata e senza modello attorno, misura **5,20 GB/s** sulla
scansione KV contro i **6,3 GB/s** dedotti dal motore. L'attribuzione è
**confermata**: la fase 1 punta il bersaglio giusto.

## 3. La causa, separata dalla velocità

La domanda del contratto era: *quanto della ridondanza GQA (4 workgroup che
leggono le stesse righe KV) se la mangia già la L2?* La risposta è **quasi
tutta**, e va letta insieme al parallelismo:

| variante | ms | fattore | traffico emesso | note |
|---|---|---|---|---|
| `base` | 9,980 | 1,00× | 4× (207,5 MB) | 16 workgroup, `scores[6400]` in memoria di gruppo |
| `vec4` | 2,639 | 3,78× | 4× | **solo** le letture vec4, tutto il resto identico |
| `stream` | 3,478 | 2,87× | 4× | softmax online, niente `scores[ctxMax]` |
| `gqa-stream` | 4,128 | 2,42× | **1× (51,9 MB)** | KV letta una volta per gruppo, ma solo **4 workgroup** |
| `split` | **0,296** | **33,72×** | 4× | 256 workgroup |
| `split-gqa` | 0,325 | 30,74× | **1×** | 64 workgroup |

Due letture, entrambe controintuitive rispetto al piano:

1. **Togliere il 75% del traffico non fa guadagnare niente.** `gqa-stream` legge
   un quarto dei byte di `stream` ed è **più lenta** (4,13 contro 3,48 ms). La
   dedup GQA da sola è una perdita, perché paga il traffico risparmiato con la
   riduzione del numero di workgroup da 16 a 4. Stesso ordine fra `split` (4×
   traffico, 256 wg) e `split-gqa` (1× traffico, 64 wg): vince quella che legge
   **quattro volte tanto**. Il kernel non è limitato dalla banda: è limitato dal
   **parallelismo**. Con 16 (o 4) workgroup su 128 SM la GPU è ferma al 12% (3%).
2. **Le letture scalari costano da sole 3,8×.** È la variante più economica da
   scrivere del lotto e vale il 78% del guadagno di un rewrite completo — ma non
   toglie il tetto di contesto (v. §5).

Nota di onestà sul confronto `split` vs `split-gqa`: i due non differiscono solo
per la fusione GQA, differiscono anche per il numero di workgroup (256 vs 64).
**Il confronto è confuso**, e la fase 1 deve rifarlo a parità di occupancy
(`split-gqa` con 64 chunk invece di 16). Registrato come item di docket.

## 4. La landmine trovata strada facendo: la L2 da 72 MB si mangia la cella

La KV di **un** layer a n=6333 pesa **51,9 MB**; la L2 della Lovelace ne tiene
**72**. Un micro-bench che alloca una sola KV e la martella misura la cache, non
la VRAM. La pre-registrazione aveva dichiarato questo rischio per il GEMV (ed è
per questo che la lm_head è l'unica cella confrontabile col motore) e **non**
per l'attenzione: era un buco.

Il run definitivo lo chiude allocando **8 layer di KV (415 MB)** e ruotandoli a
ogni dispatch, come fa il motore vero (ogni layer legge la SUA KV una volta per
token). Le celle `-coldkv` sono un'**aggiunta post-hoc dichiarata**, non
pre-registrata. La prova del fatto sta dentro lo stesso file, nel confronto fra
la misura a dispatch singolo e quella in batch da 16:

| | batch16 sullo stesso layer (L2) | batch16 a layer ruotanti (VRAM) | fattore L2 |
|---|---|---|---|
| `base` | 4,772 ms | 9,750 ms | **2,04×** |
| `split` | 0,132 ms | 0,290 ms | **2,21×** |

Un primo run (scartato: aveva anche un difetto di riempimento della KV) misurava
`base` a 4,97 ms e `split` a 0,134 ms — cioè esattamente i numeri "caldi" — e
attribuiva alla variante 386 GB/s su byte unici, **1547 GB/s emessi**, sopra il
picco di targa della scheda. È il modo in cui un micro-bench mente: nessun
checksum lo prende, solo l'aritmetica sulla banda di picco.

**Conseguenza per chi legge i numeri di questa fase**: le celle riportate sono
quelle a working set 415 MB. Sono la stima *conservativa*.

## 5. Portabilità: il tetto di contesto sparisce, e sparisce guadagnando

`attnDecodeWgsl` chiede `4·ctxMax + 256` B di memoria di gruppo — 25 856 B a
ctxMax 6400, contro i **16 384 B garantiti da WebGPU** (docket item 2: il
pareggio col limite negoziato è a ctxMax 7648, sopra il quale la pipeline non
parte). Tutta la famiglia streaming chiede una quantità **costante in ctxMax**:

- `stream` / `split`: 1 536 B (`qsh` 1 KB + `sc` 256 B + `red` 256 B)
- `gqa-stream` / `split-gqa`: 6 144 B

Il requisito di portabilità del contratto (`< 16384 B a QUALUNQUE ctxMax`) non
costa velocità: la variante che lo soddisfa è anche quella che va 33,7× più
forte.

## 6. Cosa consegna la fase 1 e cosa la fase 2

Proiezione sulla scomposizione it.59 a ctx 6333 (100,5 ms/token = 65,8 KV + 27,1
corpo + 7,6 coda). **Proiezione, non promessa**: il micro-bench misura un
dispatch isolato, il motore paga anche overhead di dispatch e sincronizzazioni.

| scenario | KV | corpo | coda | token | tok/s |
|---|---|---|---|---|---|
| oggi | 65,8 | 27,1 | 7,6 | 100,5 | 9,95 |
| conservativo (KV 10×, corpo 3×) | 6,6 | 9,0 | 7,6 | 23,2 | **43** |
| a fattore misurato (33,7× e 4,0×) | 1,95 | 6,8 | 7,6 | 16,4 | **61** |

Nello scenario conservativo la **coda** (7,6 ms: lm_head + argmax + readback)
diventa il 33% del token, e nello scenario a fattore misurato il 46%. Non è nel
piano di questo goal: **va registrata come la prossima pendenza**, prima che la
fase 3 ci arrivi di sorpresa.

Ordine delle fasi: **P5 confermato**, la fase 1 (attenzione) resta prima della
fase 2 (GEMV).

## 7. Ricette per la fase 1 e la fase 2 (dal micro-bench, non dalla teoria)

**Attenzione** (`src/microbench/kdAttn.ts::attnStreamWgsl({headsPerWg:1, splits:16})`
+ `attnCombineWgsl`):
- softmax **online a tile di 64 posizioni**: max di tile → rescale
  dell'accumulatore → somma di tile; nessun `scores[ctxMax]`;
- fase punteggi: un thread per posizione, `dot()` su `vec4<f32>`, `q` messa in
  memoria di gruppo una volta sola;
- fase output: il thread `t` possiede il `vec4` numero `t` di headDim (64 thread
  × 4 componenti = 256): la riga V si legge in modo **coalescente**;
- split del contesto in 16 chunk (448 posizioni), parziali `(acc non
  normalizzato, m, s)` e combinazione **log-sum-exp** in un secondo dispatch.
- La fase 1 **non è bit-identica per costruzione** (cambia l'ordine delle
  somme): qui lo scarto misurato sul checksum è **3,0e-6 relativo**, un dato
  utile per fissare la tolleranza del ktest prima di scriverlo.

**GEMV q4_0** (`src/microbench/kdGemv.ts::gemvVec4Rows2SgWgsl`):
- una load `vec4<u32>` (16 B = un blocco q4_0 intero) al posto di 4 load `u32`;
- estrazione dei nibble **vettoriale**: `(vec4<u32>(word) >> vec4(0,8,16,24)) & 15`
  per i bassi, `>> 4 & 15` per gli alti, poi due `dot()` con le `vec4<f32>` di x;
- 2 righe per workgroup, **una riga per subgroup**: la riduzione di riga è un
  solo `subgroupAdd`, senza barriere e senza memoria di gruppo;
- ATTENZIONE `subgroupAdd` vuole control flow **subgroup-uniform**: l'early
  `return` sulla riga fuori range fa fallire la compilazione con
  `'subgroupAdd' must only be called from subgroup uniform control flow`. Si
  guarda l'accumulo con un `if`, si chiama `subgroupAdd` fuori, si guarda la
  scrittura. (Osservato dal vivo, primo run.)
- Se si vuole una variante **senza dipendenza da `subgroups`**: `vec4-rows4`
  (16 lane per riga, riduzione ad albero) vale 3,77× sulla lm_head, e `vec4`
  liscia 3,89×. Il subgroup vale l'ultimo 3%, non il grosso.

## 8. Cosa NON è stato misurato (dichiarato, non taciuto)

- **`dot4I8Packed` come variante di velocità**: usarlo per q4_0 richiede
  attivazioni quantizzate a int8, cioè un algoritmo diverso, che violerebbe la
  tolleranza 1e-3 sul checksum. Pre-registrato come sola sonda; la sonda dice che
  l'istruzione **c'è e funziona**. È materiale per un goal di prefill/GEMM.
- **subgroup-matrix**: esposta **e concessa**, config `u8→u32` / `i8→i32`
  16×16×32 e 16×8×32. Nessuna con risultato f32. È l'unica riga di P1 caduta, ed
  è caduta nella direzione favorevole: sul tavolo c'è un tensor-core path intero
  per il prefill, che questo goal non tocca.
- **l'attenzione a chunk del prefill** (`attnDecodeWgsl` con `batch`): stesso
  kernel, goal diverso (TTFT). Le stesse tre leve si applicano.
- **il numero di layer di attenzione piena del 4B**: 65,8 ms di scansione KV
  misurati dal motore contro 9,98 ms per dispatch qui fanno ~6,6 dispatch per
  token. Il conto non è stato chiuso; se il modello non ha ~6-7 layer di
  attenzione piena a questa forma, uno dei due numeri va rivisto.
