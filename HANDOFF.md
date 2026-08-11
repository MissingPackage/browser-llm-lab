# HANDOFF — browser-llm-lab   (updated 2026-08-11, sessione 28 — goal engine-fase-d in corso: core unificato + gate a invarianti, GLM bit-identico; fasi 1-2-3 chiuse, FASE 3b CHIUSA (it.11-18: 81->1 submit/token, 41->1 readback, argmax 39/39 identico anche col replay a freddo, -68,60 ms/token a parita'); fase 4 in corso: misurato che il token e' DISPATCH-BOUND (~20 us/dispatch), proiezione 2,02x a M=16, headCut fatto, piano di prefill parametrizzato it.20)

## 1. Next decidable

**AL LAVORO: goal `engine-fase-d`** (chartered 2026-08-10, tag
`goal-engine-fase-d-start`, PHASES 9 fasi). Nasce dal ruling PI
2026-08-10 (direction §7-ter) e dalla riapertura di q1 (docket q1 item
14): una famiglia nuova non è importata finché non ha le stesse
ottimizzazioni delle esistenti; il codice si UNIFORMA.

**Fatto (it.1-3, ogni iterazione verificata)**: `moe.ts` e `residency.ts`
sono una meccanica sola — tabella di geometria dei quant, UN router con
due configurazioni (GLM sigmoid+bias+1.8 / Qwen softmax), UN builder di
slab, e il MOTORE della cache (stati di classe, ripartizione del budget,
arena, repin, stats, path caldo `ensure`) guidato da `MoeModelConfig`.
GLM è una configurazione e resta **BIT-IDENTICO** (ktest 84/84:
2layer L2rel 2.07e-7, arena-vs-slotrange BIT-A-BIT, layer0 2.35e-7).
Suite 387. Nato il GATE STRUTTURALE (`tests/engine-one-mechanism.test.ts`)
che vieta un secondo router/layout/arena.

**Gate anti-duplicazione (it.4-6)**: il verifier ha bocciato TRE versioni
di un gate a scansione del sorgente, l'ultima con 5 evasioni eseguite —
di cui una (un router Qwen legittimo: softmax puro, niente clamp, niente
nomi di tensori, niente VRAM) NON catturabile da nessuna impronta
testuale, perché la differenza fra duplicazione e seconda famiglia è
SEMANTICA. Esito: **l'invariante vive nel SISTEMA DI TIPI** — marchio di
conio (`unique symbol`) su `SlotRef`, che solo `residency.ts` può CONIARE
(11 sonde ostili del verifier rifiutate da `tsc`). Ferma la CONTRAFFAZIONE,
non l'indifferenza: un'arena che non usa `SlotRef` non viene sfiorata. È
diventato PORTANTE con it.7, da quando anche q35 binda gli expert passando
da `SlotRef`/`slotTensorRanges`. E
`tests/types/slotref-brand.ts` (`@ts-expect-error`) va rosso se il
marchio sparisce. `tests/engine-one-mechanism.test.ts` resta come
**RATCHET** su impronte note (scansione di tutto `src/`, ancorata,
estensioni incluse) con la pretesa ridimensionata per iscritto: non è una
prova, ed è sbagliato usarlo come tale. Docket item 4 CHIUSO.

**FASE 1 CHIUSA (it.7)**: `q35gpumodel` non possiede più niente della
residenza expert — arena a chunk, LRU, `ensure`, repack e router propri
sono spariti, sostituiti dalla `ExpertCache` di GLM guidata da una
`MoeModelConfig` dedotta dal GGUF e da `routerSelect`. Nato
`src/engine/q35expertstore.ts`, gemello di `expertstore.ts`: le tre voci
DEBITO NOTO sono sparite DAVVERO (il gate asserisce `debiti == []`).
Prove: parità slab CPU-only 6/6 (stessi byte agli stessi offset), ktest
**84/84** con GLM bit-identico (arena-vs-slotrange maxRel 0), conformance
smoke 35B sul path migrato **top1 5/5** con hits/misses/uploadedBytes
IDENTICI al pre-migrazione (8846 / 3314 / 5.916.950.528).

**FASE 2 (it.8-9)**: misurato prima di scrivere. Il repack K-quant costava
3,50 ms/miss = **il 22% del prompt** sullo smoke 35B, per due ragioni
entrambe di scrittura: `repackKQuant` ricostruiva le parole con un `|=` per
BYTE (ma su little-endian è una COPIA travestita da aritmetica) e
`packExpertSlab` passava da un array temporaneo (ogni byte toccato 3 volte).
Ora memcpy + `repackKQuantInto` diretto nello slab: **pack 11.585 → 1.856 ms
(6,24x)**, 3,50 → 0,56 ms/miss, prompt 53,0 → 42,5 s, con top1 5/5 e
hits/miss IDENTICI. ktest 84/84 con valori uguali cifra per cifra: è
bit-a-bit lo stesso repack. Micro-bench committato e riproducibile
(`PACK_BENCH=1`, JSON con prima+dopo). Eliminata la precondizione "dst
azzerato" di `repackKQuantInto` (rilievo del verifier: un generatore di slab
che riusi il buffer avrebbe prodotto pesi Q6_K sporchi IN SILENZIO).

**PI-GATED, docket item 5**: il done-when della fase 2 dice "il repack esce
dal path on-miss (import-time)". Non l'ho fatto: l'ho reso quasi gratis
dov'era. Residuo ~1,7 s su 45 (~4%); spostarlo all'import costa **~18 GB di
slab su disco**. Parere: non conviene, e la riga andrebbe riscritta come "il
costo di pack per miss scende di >=4x, misurato". **Riscrivere un done-when
è un cambio di contratto, non una decisione di meccanismo: decide il PI.**
Se dice "fallo all'import", la fase 2 si riapre e il lavoro fatto resta buono.

**FASE 3 (it.10-12), densi — E UNA CORREZIONE MIA**. `decodeBatch` c'e' e
funziona: K token teacher-forced in un submit, argmax su GPU, un readback di
K·4 byte, con **argmax IDENTICO** al path a readback su tutti i 39 token
(dentro il `pass` del ktest). **Ma il "-15,0 ms/token" che avevo scritto era
un ARTEFATTO DI MISURA** — il braccio lento era la prima passata dopo il
load (a freddo), quello veloce la terza (a caldo): ~8 ms/token di warm-up
tutti da una parte. Trovato dal verifier. Micro-bench riscritto (warm-up
scartato, bracci interleavati, 3 ripetizioni, mediana + dispersione):

| 4B, a caldo | ms/token | [min-max] |
|---|---|---|
| `step` con sync | 40,02 | 40,0-40,2 |
| `decodeBatch` | 36,69 | 36,7-36,8 |
| pavimento senza sync | 35,44 | 35,4-35,5 |
| **delta** | **-3,33 (-8,3%)** | spread 0,19/0,10 |

**RULING PI 2026-08-11 (docket item 9): opzione (a) — CHIUSO.** La riga 3 non
chiede piu' `>= -5,1 ms`: chiede il delta MISURATO a caldo con dispersione,
qualunque sia. Il -5,1 era una stima ex-ante di q1, non un requisito del
prodotto. **Fase 3 chiusa sui densi anche per contratto**, col numero vero
(-3,33). Il guadagno residuo sui densi sta nei kernel (pavimento 35,44
ms/token con 562 dispatch), non nel batching del decode.

**AL LAVORO: fase 3b** (resolve MoE su GPU, ruling PI 2026-08-10). Fetta 1
FATTA (it.11): `gemvQ4K`/`gemvQ6K` accettano l'indirizzamento d'ARENA (slot
da `Sel`, buffer bindato intero, testa di GLM riusata) e il gate nuovo sul
35B reale dice **BIT-A-BIT identico** contro il binding a sotto-range, su
entrambe le classi. ktest 86/86. Fetta 2 FATTA (it.13): `routerTopKWgsl` parametrico sul gating
(sigmoid GLM / softmax Qwen), col binding `bias` tenuto in entrambi i casi e
zeri per chi non lo usa — per GLM il testo emesso resta byte-identico. Gate
su 64 estrazioni 256x8: **0 flip d'insieme, 0 d'ordine, 0 resolve errati**,
errore sui pesi 2,32e-7 (soglia 1e-5); il resolve si prova con una slotTable
che ha UN MISS di proposito. ktest 87/87.

**FETTA 3a FATTA (it.14)**: `q35gpumodel` monta la cache in regime
`arena + slotTable`. I buffer di classe si bindano INTERI e l'indirizzo dello
slab lo ricava il KERNEL dallo slot che legge in `Sel` (riempita ancora dalla
CPU); bind group layout ESPLICITO perche' `hasDynamicOffset` non esiste con
`layout: "auto"` ⇒ **3 bind group per classe** invece di 3 per slot. Le
costanti dell'ABI (`SEL_BYTES`, `MOE_IDX_BYTES`, `MOE_IDX_STRIDE`) si spostano
accanto alle struct WGSL: una sola verita' per le due famiglie.
**GATE, due bracci nella stessa sessione** (worktree su HEAD vs modificato,
stesso golden smoke 35B, budget 10 GiB DICHIARATO): **istogramma di routing
IDENTICO chiave per chiave** (3314 chiavi, 12.160 selezioni), argmax generati
identici, top1 5/5, hits/misses 8846/3314, uploadedBytes identici, 782
dispatch/token. Il rischio previsto in it.13 era reale: la classe q4k si
spezza in 5-6 buffer ⇒ 8-9 storage binding contro i 7 del path Qwen, e
`q35conf.worker` ora negozia `arenaNeeds`. NON coperto, detto: nessuna
eviction in questo run (3314 miss = 3314 expert distinti su 6006 slot).

**FETTA 3b FATTA (it.15)**: router+resolve su GPU in OMBRA sui layer VERI —
`Sel` raddoppia (produzione CPU + ombra GPU nello stesso buffer), il dispatch
gira nello stesso submit del segmento statico e si confrontano. Opt-in
(`routerShadow` / `--shadow`), spento di default. **Smoke 35B, 1520 confronti
(38 token x 40 layer), 12.160 selezioni: 0 flip d'insieme, 0 flip d'ordine,
errore max sui pesi 3,80e-7 (soglia 1e-5), slotMismatch 0, miss GPU/CPU
3314/3314 con 0 disaccordi**, e produzione IDENTICA a it.14. Il numero che
qualifica il gate NON sono gli zeri ma il **margine minimo: 1,0e-5** fra
l'ultimo expert preso e il primo scartato (sui logit) — una decade sopra la
risoluzione dell'f32 a quella scala. Sotto ~1e-6 sarebbe testa-o-croce e qui
non e' capitato: da rifare sul corpus pieno alla fase 6.

**MISURA PRIMA DELLA 3c (it.16)**: il path ottimistico di GLM ha una
precondizione (`optimisticMinResidency` 0.8) e il 35B a 10 GiB sta al 58,7% del
parco. Misurato con `--misstrace` (due passate sullo stesso prompt; `resetState`
azzera lo stato ricorrente, NON la cache expert): pass FREDDO **39/39 token
sporchi**, mediana 68 miss su 320 selezioni, hit 73,2%; pass CALDO **0/39
sporchi, 0 miss, hit 100%**. Due conclusioni opposte, da tenere insieme: a
residenza raggiunta l'ottimistico da' **1 submit/token** (ed e' misurabile su
questo stesso smoke, seconda passata); a cache fredda il repair+replay e' il
REGIME e il path ottimistico NUDO sarebbe una regressione sul prefill.

**FETTA 3c FATTA (it.17)**: il token intero in **UN submit**. La `Sel` di
produzione la scrive il ROUTER SU GPU (resolve expert->slot dalla `slotTable`
nello stesso dispatch, `dirtyB` con atomicMin sul primo layer MoE sporco e
atomicAdd sul conteggio); 40 segmenti statici + 40 router + 320 dispatch expert
in un encoder solo; il `routing` si ricostruisce alla fine dalla `Sel` copiata
in coda, perche' mentre il token gira nessuno sulla CPU vede la selezione. **I
kernel expert non cambiano di una riga** — cambia CHI riempie `Sel`, ed e' il
pagamento dell'indirezione della fetta 3a. Pezzi nuovi: `clearBuffer(moeAcc)`
DENTRO l'encoder (la `writeBuffer` e' ordinata prima dell'INTERO submit: i 40
layer vedrebbero un solo azzeramento), `axpySelWgsl` (il peso di mixing nasce su
GPU; sul MISS il contributo e' ZERO, degrado definito) e
`ExpertCache.noteResidentHit` (senza, `ensure` non viene mai chiamata e la LRU
degraderebbe a FIFO: il path nuovo misurerebbe miss diversi per un motivo che
non c'entra col meccanismo).

**GATE 3c (smoke 35B, 39 token, 320 sel/token, 10 GiB)** — tre regimi separati:
submit/token **81 -> 1**, readback/token **41 -> 1**, argmax **39/39 IDENTICO**,
routing **identico chiave per chiave** (3341 chiavi, 0 diff), miss 0 vs 0,
hit+miss = 12.480 = 39x320 in tutte le passate. ms/token A PARITA' di residenza
(bracci interleavati, 4 rip., prima coppia scartata — docket 10): **143,49
[142,51-145,94] -> 71,50 [71,05-71,57] = -71,99 ms/token (-50,2%)**. Il primo
giro del gate confrontava freddo (1192,9) contro caldo: e' servita una TERZA
passata perche' il numero fosse onesto. ktest 87/87, suite 410|9, tsc pulito.
JSON: `results/engine/q35-optimistic-35b-it17.json`.

**REPAIR+REPLAY FATTO — FASE 3b CHIUSA (it.18)**. La precondizione che GLM non
aveva: 30 layer su 40 sono deltanet e aggiornano `convSt`/`stateS` IN PLACE, quindi
rigiocarli applicherebbe l'aggiornamento DUE VOLTE (senza errore, con numeri
plausibili). Snapshot per token — costa poco perche' ogni layer tocca il proprio
stato una volta sola, quindi stato-all-ingresso-del-layer == stato-all-ingresso-
del-token — **62,8 MiB**, restore dei soli layer >= `firstDirty`.

BUG PRESO DAL GATE, non da una rilettura: avevo usato `startLayer === 0` come
sinonimo di "primo giro". A cache vuota il primo layer sporco E' lo zero, quindi
il replay finiva nel ramo dello snapshot e `x` restava l'uscita del giro prima
invece dell'embedding — il router di layer 0 sceglieva altri 8 expert, non
residenti, e il round dopo ritrovava il layer 0 sporco ("progresso violato").
Ipotesi formulata dal codice PRIMA di toccarlo, corretta separando i due
concetti, ri-verificata contro lo stesso repro.

**GATE it.18 (smoke 35B, 39 token, 10 GiB)**, regime freddo in un run a parte
(`--opt-cold`: la cache fredda esiste una volta sola per processo) —
ottimistico FREDDO 3,79 submit/token · 39/39 token sporchi · 109 replay
(2,79/token, **32,3 layer su 40 = 80,7% del token rigiocato**) · 3742 miss
contro i 3341 del sync (**+12,0%**, 910 fetch mai usati: un expert riparato puo'
non finire nella Sel definitiva) · repair 484,3 ms/token = **65,6% del token** ·
738,6 ms/token. CALDO **81 -> 1 submit/token, 41 -> 1 readback**, 141,18
[140,61-144,64] -> 72,58 [72,45-72,73] = **-68,60 ms/token (-48,6%)**.
**argmax IDENTICO 39/39 anche fra freddo-ottimistico e sync-caldo** — 109
restore dello stato ricorrente senza deriva, che su un modello ricorrente
sarebbe cumulativa. ktest 87/87, suite 410|9, tsc pulito. JSON:
`q35-optimistic-35b-it17.json` (freddo sync) e
`q35-optimistic-cold-35b-it18.json` (freddo ottimistico).

**Done-when 3b voce per voce**: (a) 81->1 submit e 41->1 readback misurati nello
stesso JSON, residenza piena a 1 submit/token ✓ · (b) argmax 39/39 ✓ · (c) ktest
87/87, GLM bit-identico ✓ · (d) miss e routing identici A RESIDENZA PIENA, che e'
il campione nominato da (a); **a freddo NO e non puo' esserlo** — il +12,0% di
fetch e' inerente al meccanismo e si pubblica come fatto misurato.

**UNA MIA PREVISIONE SMENTITA** (docket item 13): in it.16 avevo scritto che a
cache fredda il path ottimistico nudo sarebbe stato una REGRESSIONE. Misurato
738,6 contro 1192,9 ms/token del sync freddo. Non lo prendo per buono — un
campione per braccio, due run diversi, passata dominata dall'I/O: per il docket
item 10 non e' una misura. Ma la soglia della fase 5 va tarata su un bench fatto
apposta, non su quell'intuizione e nemmeno su questi due numeri.

**FASE 4 — MISURA PRIMA DELLA FETTA (it.19)**. Previsione registrata e
committata PRIMA di misurare ("expert >= 60% del tempo GPU del token") e
**SBAGLIATA**: e' 58,0% del tempo GPU e 45,3% di quello di parete. Sonda nuova
(`--gpu-time`, opt-in, perturbazione misurata +1,7%): expert 33,44 ms (58,0%,
1600 dispatch) · statico 14,53 (25,2%, 742) · coda 6,86 (11,9%) · router 2,83
(4,9%, 40 dispatch); totale GPU 57,67 su 73,85 ms di token, quindi **16,18 ms
(21,9%) stanno fuori dai pass** (encode CPU, submit, readback, argmax, embed).

**IL NUMERO CHE REGGE: ~20 us per dispatch, uguale fra statico (19,6) ed expert
(20,9)**, contro 1-2 us di lavoro stimato per GEMV. Due famiglie di kernel con
taglie e traffico diversi che costano lo stesso per dispatch ⇒ il costo e' il
per-dispatch, non il lavoro: il token e' DISPATCH-BOUND, e la fase 4 (che toglie
dispatch) e' la leva giusta. Proiezione rifatta sui numeri veri a M=16: 57,67 ->
**28,55 ms GPU (2,02x)**, col path expert al 92% del residuo perche' l'unione si
comprime solo 1,27x (256 expert e top-8 contro i 64/top-4 di GLM).

**I DUE VINCOLI STRUTTURALI DELLA FASE 4**, derivati prima della misura: (1) 30
layer su 40 sono deltanet, RICORRENTI sul tempo — le righe di un chunk sono
token consecutivi, quindi i 2 dispatch ricorrenti per layer restano M mentre
tutto il resto diventa 1 (la forma chunkwise del delta rule esiste ma e' un
kernel di ricerca, non un port); (2) l'unione degli expert comprime 1,27x a
M=16 contro l'1,6x di GLM.

**GIA' FATTO in it.19**: `headCut` — la coda (norma finale + head) vale 6,86
ms/token (9,3%) e nel prefill veniva BUTTATA (`read=false` non legge i logits).
Tolta su MoE e densi; la norma finale scrive `xn`, uno scratch, e il residuo `x`
non viene sfiorato. NON misurata end-to-end (il gate gira `read=true`): la
misura e' il micro-bench della fase 4. Provata la non-regressione: argmax 39/39,
routing identico, miss 0.

**PASSO 1 DELLA FETTA FATTO (it.20)**: il piano del prefill non e' piu' di GLM.
`glmprefillplan.ts` era cablato su `GLM47_FLASH` (64 expert, top-4) in SETTE
punti; ora e' `moeprefillplan.ts` ed e' parametrico su `MoePlanShape
{nExpert, nExpertUsed}` — config STRUTTURALE e non un import di
`MoeModelConfig`, perche' `residency.ts` importa `GLM_PREFILL_M` da qui e
dipendere di la' a runtime chiuderebbe un CICLO (`MoeModelConfig` e' assegnabile
per struttura: l'unificazione la fa il sistema di tipi, senza dipendenza).
I test girano su DUE famiglie (`describe.each`: GLM 64/top-4 e q35 256/top-8) su
tutte le proprieta', identita' bit-a-bit con la catena del decode inclusa, piu'
la controprova che l'ordine dell'unione DEVE divergere in f32 e un test che il
default senza cfg e' ancora esattamente GLM. Suite **417|9**, ktest **87/87**
(`prefill-moe-batched-vs-decode-chain` BIT-IDENTICO): il prefill di GLM non si e'
mosso di un bit.

**PROSSIMO: i kernel a M righe di q35** — statico/router/expert, gather per
expert dell'unione, combine in ordine k (e' cosi' che GLM tiene l'identita'
bit-a-bit col decode), e la scelta di M coi numeri di it.19 (proiezione 2,02x a
M=16), non per analogia con GLM. Vincoli gia' noti e da rispettare: i 2 dispatch
ricorrenti per layer deltanet restano M (30 layer su 40), e l'unione comprime
solo 1,27x a M=16.

**Regola dell'harness (docket 10)**: il primo passaggio dopo il load non si
misura mai — si scarta una passata, si interleavano i bracci, si riporta
mediana e dispersione.

**Landmine nuova (it.14, docket 11)**: il budget dell'arena del 35B NON e'
ctx-aware come quello di GLM — e' un parametro fisso, default 12 GiB, e su
questo host (14,4 GiB liberi) sfonda con `VK_ERROR_OUT_OF_DEVICE_MEMORY`. I
buffer diventano invalidi e il modello CONTINUA a girare producendo numeri
plausibili (top1 1/5): senza il listener `uncapturederror` sarebbe muto. I run
di correttezza sul 35B dichiarano il budget (`q35-conf-run.mjs --arena-gib`,
nuovo in it.14).

## 2. State delta (sessione 27, 2026-08-10 — goal q1 intero, it.0-21)

- 21 iterazioni in ~14 h, verifier gate su ognuna (2 FAIL sanati: journal
  it.15, run-morta it.17). Storia completa nel journal del goal.
- Moduli nuovi in `src/engine/`: q35shape (shape dai metadata),
  q35tokenizer (PRIMO tokenizer in-engine, BPE byte-level), q35cpuref +
  q35cpurefmodel (cpuref-f64 famiglia, reader lazy), q35sample,
  kernels/deltanet (conv+gates+core WGSL), q35gpumodel (orchestratore
  denso+MoE con arena a chunk e LRU), q35conf/ (conformance+bench+tap).
  gguf/quant/wgsl estesi SOLO additivi (Q4_K, axpy, sigmoidMul, ropeDims).
  GLM (glmmodel/residency/moe/bandmodel) INTATTO.
- Strumenti: run-golden-q35.sh (provenance, CORPUS_DIR, arch fix),
  q35-conf-run/q35-bench-run (--model/--arena-gib), q35-looka-run,
  q35-bandmodel-fit, q35-tier-mobile-gen, webgpu-subgroup-matrix-probe,
  q35-verify-sha, fixture generators.
- Soglie ratchet nel docket q1 (item 8/9/12); leve e convergenza int8
  (item 10/11); non-reg host-gated (item 13).
- Docs: direction §7-bis (generalizzazione coi numeri), ledger +4 righe,
  spec q1 con §7 aggiornata dal probe, gap-decomposition §5.

## 3. Open threads (fuori goal, non bloccanti)

- **Rerun non-reg a boot pulito** (docket q1 item 13): glm-bench
  optimistic autobudget (>=13.43, banda vs 15.641) + glm-conf full +
  Qwen2.5. È il gate d'apertura della prossima sessione.
- Golden q35 4B/9B: campo arch "deepseek2" stale nei metadata (fix del
  tool fatto; rigenerare i CAMPI prima del paper — docket q1 item 7).
- BASE_URL :5173 in engine-bench.mjs e conformance (thread c3a).
- Ratifiche c3a pendenti (14b, 2, 19-21); campi power hostState scambiati
  (c3c item 7); profilo bench ~6.4 GiB in ~/.cache cancellabile.
- Goal fase-1b/fase-2 = STANDBY deliberato benchmark (non stale).
- Disco: +50 GB questa sessione (29 GGUF q35 + profili/golden).

## 4. Landmines

- **Non-reg NUMERICA solo a host comparabile DICHIARATO** (lezione it.21):
  page cache OPFS e VRAM baseline muovono stallResidenza 4.3→29 ms/token
  e P(dirty) 0.81→0.98; mai confrontare bande fra host diversi (c3c).
- MoE q35: prefill bench con `await` (mai fire-and-forget: mapAsync del
  router); arena SOLO a chunk ≤2 GiB (adapter cap 4 GB, il monolitico
  fallisce SILENZIOSO); slot in byte REPACKED (Q6_K 210→212).
- fetch/ArrayBuffer >2 GiB: sempre Range/pread (3 pareti trovate).
- recall@4 lookahead ha denominatore 8 (tetto 50%): mai confrontarlo
  con @8 senza dichiararlo.
- Storiche: run GPU ad albero congelato + 60 s; niente pipe sui runner;
  near-tie mai gateati; f32-first; var WGSL azzerate; full-corpus solo
  per riferimenti; KV GLM 108 288 B/token; q35: 40 960 (35B) / 65 536
  (densi).

## 5. Docket (user decisions pending)

- **Prossimo passo post-q1**: release (split+paper+blog, sequenza c3c
  item 10) vs fase D (moltiplicatori) — la decisione di §1.
- Rerun non-reg a boot pulito (q1 item 13) — operativo, primo atto.
- Hero-demo M4 (c3c item 8) — PI-gated per hardware.
- Timing del blog (prima del paper o insieme) — aperto, non bloccante.
