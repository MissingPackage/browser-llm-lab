# Consuntivo del goal `engine-ttft` — 2026-08-14

**Tempo al primo token a modello caldo, prompt da 6333 token, Qwen3.5-4B:
87.618 → 32.127 ms. Discesa 2,727×.**

La barra meccanica del contratto era **< 21.905 ms** e **non è stata raggiunta**:
manca **1,467×**. Questo documento dice voce per voce cosa è stato ottenuto,
cosa no, e — la parte che vale di più — **perché**, con la misura che lo
dimostra.

---

## 1. Il DONE WHEN del contratto, voce per voce

| # | clausola | esito | evidenza |
|---|---|---|---|
| 1 | Baseline rimisurata prima di ogni riscrittura, prompt-idx 0, tre termini distinti, host dichiarato | ✅ | `results/engine/ttft-baseline-4b-prompt0-2026-08-13.json` — 87.618 ms = load 10.892 + prefill 87.582 + firstMs 36, `quiescent` |
| 2 | **TTFT a caldo < 21.905 ms** | ❌ **NON RAGGIUNTA** | 32.127 ms (`q35-ttft-kernel-checkpoint-4b-2026-08-14.json`). Manca 1,467× |
| 3 | Esaurimento delle leve: ognuna in produzione e misurata prima/dopo, **oppure esclusa coi numeri** | ✅ | v. §2 |
| 4 | Contabilità del tetto residuo: JSON `q35-ttft-kernel-checkpoint` con banda per segmento, workgroup in volo per dispatch, e il conto di dove finisce il tempo | ✅ | `q35-ttft-kernel-checkpoint-4b-2026-08-14.json`, v. §3 |
| 5 | `prefill.tokS > 289` | ❌ | 197,25 tok/s. È la clausola 2 vista dal lato del rate: cade con lei |
| 6 | Il prefill resta più veloce del decode | ✅ | 197,25 > 48,15 |
| 7 | Fase 0 superata prima di ogni riscrittura, con regola di stop | ✅ | `results/microbench/ttft-riga1-4090-linux-2026-08-13T13-06-23-120Z.json`, 34 celle. Regola di stop **non scattata** |
| 8 | Leva 1 (GEMM multi-riga con riuso vero) misurata prima/dopo | ✅ | riuso **5,8593×** sull'inventario per-layer intero (barra del ruling: ≥ 5,5×), test `[6c]` |
| 9 | Leva 2 (attenzione a chunk del prefill) | ✅ | ramo `batch` non instrada più al legacy; ktest `dense-batch-attn-chunk` + `-multitile` |
| 10 | Portabilità (e2) | ⚠️ **parziale** | (e2b) chiusa: il ramo batch è il quinto consumatore dichiarato, costante in ctxMax. (e2a) **aperta al PI** — v. §5 |
| 11 | Correttezza, gate secchi a ogni merge | ✅ | ktest **101 PASS / 0 FAIL**; top-1 contro l'oracolo llama.cpp **1012/1024 = 98,828%** su entrambi i bracci; sequenze generate **identiche 8/8** |
| 12 | Non-regressione, banda ±5% | ✅ | decode **48,15** tok/s a ctx 6333 (soglia 45,5; baseline 47,79) |

**Due clausole su dodici non sono soddisfatte, e sono la stessa: la barra.**

---

## 2. Le leve — cosa è entrato e cosa è stato escluso coi numeri

**In produzione, misurate prima/dopo:**

| leva | effetto misurato | dove |
|---|---|---|
| Moltiplicatore multi-riga `splitk` | prefill 34,36 → 111,16 tok/s sul braccio a chunk | it.14 |
| Via intera `packed_4x8_integer_dot_product` | 111,16 → 123,26 tok/s | it.15 |
| Attenzione del prefill in streaming | 123,26 → 196,41 tok/s | it.17 |

**Escluse coi numeri — e questa lista è l'eredità più utile:**

- **Fusione delle teste GQA sul prefill: PIÙ LENTA.** 2,0879 contro 1,8207 ms a
  contesto 6333. Taglia il traffico KV di 4× e perde comunque, perché scende da
  256 a 64 workgroup su 76 processori. *Sul decode aveva funzionato*: chi
  riprova deve sapere che il verso è opposto.
- **Il tetto di memoria di gruppo negoziabile non è una leva**: spread 0,1-2,3%
  spazzando {16.384 … 49.152}. La forma vincente ne chiede 4.096, sotto il
  pavimento di spec.
- **La ricorrenza DeltaNet NON è il collo**: 5,0% del tempo, contro il 47,3% dei
  dispatch. V. §4, è una lezione di metodo prima che un risultato.

---

## 3. Dove finisce il tempo

Prefill **32.101 ms**, scomposto con `timestamp-query` un pass per segmento:

| segmento | ms | % prefill | dispatch | wg/dispatch |
|---|---|---|---|---|
| **`gemm:deltanet-out`** | **12.169** | **37,9** | 24 | 40.960 |
| `gemm:ffn-down` | 4.971 | 15,5 | 120 | 1.739 |
| `deltanet:recurrence` | 1.602 | 5,0 | 768 | 80 |
| `gemm:ffn` | 1.234 | 3,8 | 192 | 967 |
| resto | ~2.775 | 8,7 | 520 | — |
| **fuori dai pass GPU** | **9.350** | **29,1** | — | — |

- **70,9% dentro i pass, 29,1% fuori** — encode CPU, submit, e i buchi fra un
  submit e il successivo. Non attribuito più finemente, e lo dichiaro:
  servirebbe una sonda CPU per dispatch che oggi non esiste.
- **1,578 TFLOP/s sostenuti contro il picco fp32 misurato di 9,26 = 17%.** Non è
  un'efficienza: il prefill gira su pesi quantizzati e quel tetto non lo tocca
  per costruzione. Dice una cosa sola, ed è quella che serve: **il collo non è
  l'ALU.**
- **Banda per segmento**: `gemm:deltanet-out` muove 1.093 GB a **89,9 GB/s**, su
  un motore che ne ha dimostrati ~300 sulla scansione KV. `gemm:qkv` sta a 738
  GB/s — sopra la DRAM, cioè servito dalla cache fra un chunk e il successivo:
  il numero dice che il riuso funziona, non che la memoria vada così.

---

## 4. Perché la barra non è stata raggiunta — la scoperta che conta

**Il primo termine del prefill è un moltiplicatore che questo goal non poteva
toccare.** `gemm:deltanet-out` è il 37,9% del tempo con soli 24 dispatch: uno
per layer DeltaNet, mentre la via veloce ne emette tre. Cade sul **fallback
legacy**, perché `ssm_out` è **Q5_K** e la via veloce è **q4_0-only per
costruzione**. Riuso dei pesi zero: li rilegge 16 volte per chunk.

**E questo ri-inquadra un ruling già dato.** Il ruling sul riuso (docket item 19)
aveva abbassato la barra a 5,5× perché «l'11,54% dei byte del 4B resta sul
percorso vecchio», e aveva assegnato quel residuo al goal K-quant **come una
coda**. La misura dice che quell'11,54% dei byte è il **37,9% del tempo**: la
quota di byte sottostimava massicciamente il peso, proprio perché la forma
legacy rilegge i pesi M volte — la stessa asimmetria che rende il riuso una
leva. **Non è una coda: è la leva più grande rimasta sul tempo al primo token**,
e chi pianifica il prossimo goal deve saperlo.

**La proiezione di riga 1 (pavimento ~8.665 ms) non era ottimistica: era cieca**
su un termine che dichiarava di non contare. Sommava le moltiplicazioni q4_0 e
l'attenzione. Il `ssm_out` Q5_K non è nessuna delle due.

**LEZIONE DI METODO, e mi è costata due iterazioni.** In it.23 avevo concluso
«il quarto collo è la ricorrenza DeltaNet, 47% del piano» contando i **dispatch**.
Col cronometro la ricorrenza è il **5,0%**. Contare i dispatch non è misurare il
tempo, e presentarlo come se lo fosse è un errore che si propaga: era già finito
in un HANDOFF e in un digest prima che il cronometro lo smentisse.

---

## 5. Cosa eredita il goal successivo

**Aperto al PI, non deciso qui:**

1. **Il porting del path 0.5B** (docket item 26). Non è un kernel: sono quattro,
   e uno — il down-proj del decode — **non è raggiungibile dalla forma
   multi-riga**, perché gira a M=1 e non è un GEMM di prefill. Finché c'è lui,
   nessun porting del prefill porta quel path sotto i 16.384 B garantiti da
   WebGPU. Proposto come goal suo.
2. **Cosa promette il DONE WHEN sulla portabilità** (docket item 25), ora
   informato dal punto sopra.
3. **La priorità del goal K-quant**, alla luce del 37,9%.

**Fatti utili, già misurati:**

- La via veloce accetta **tutte** le shape del 0.5B (kind q4_0, K%64 == 0
  verificato): il blocco non è il dominio, è il numero di kernel.
- `QWEN_WORKGROUP_STORAGE_BYTES` era giusta **per coincidenza** ed è ora un
  massimo calcolato: portare un solo kernel alla forma nuova avrebbe fatto
  **sotto-chiedere** il limite, rompendo la creazione delle pipeline su ogni
  device. Chi attacca quel lavoro parta da qui.
- `gemmResidChunkFast` sta **esattamente** sul confine della sua soglia di
  memoria condivisa (28.672 = 28.672): un `mMax` diverso cambierebbe ramo in
  silenzio.

**Strumenti nuovi, riusabili:** `model.prefillPlanInventory()` (dispatch e
workgroup in volo per segmento, senza spendere un submit),
`model.prefillGpuTime()` (tempo GPU per segmento), e
`scripts/build-ttft-checkpoint.mjs`, che unisce i due artefatti con la
sentinella sul `kind` invece che sul nome del file.
