# Spec — engine-fase-c3b: decode ottimistico + repair esatto

Data: 2026-08-07 · Goal: `.harness/goals/engine-fase-c3b/GOAL.md` (contratto v2)
Input: WP-0 (journal c3a it.20, `results/engine/moe-oracle/wp0-replay-sim-2026-08-06.json`),
select:"gpu" (journal c3a it.17), ruling item 18a (proposta accettata nel tradeoff).

## 0. Obiettivo e principio

Eliminare i 46 sync router/token del decode a residenza PARZIALE, nel regime
near-total (>= ~88%). Il token si sottomette OTTIMISTICAMENTE come se tutti gli
expert selezionati fossero residenti (1 submit, 0 sync intermedi — la struttura
di `select:"gpu"` it.17); i miss vengono MARCATI dalla GPU e letti nel readback
di coda che esiste gia'; se il token e' sporco, la CPU ripara (fetch dei
mancanti) e RIGIOCA dal primo layer sporco. **La qualita' non si tocca mai: i
logits di un token sporco non vengono MAI campionati — si campiona solo l'esito
del replay. Il costo e' latenza rara, mai correttezza** (item 18a).

Perche' funziona: localita' cross-token misurata W=1 32%, W=16 79%, W=64 95.4%
(WP-0) — il riuso tra token consecutivi e' reale, quindi P(dirty) e' basso nel
regime near-total e il replay e' l'eccezione, non la regola.

## 1. Invarianti (non negoziabili, da assert)

- **I1 — slotTable congelata in volo.** Dopo `queue.submit` di un token,
  slotTable (shadow CPU e buffer GPU) e' INTOCCABILE fino al readback di coda
  di quel token. Ogni insert/evict avviene SOLO al confine di token (semantica
  WP-0: "inserimento differito al confine"). Assert: flag `inFlight` in
  ExpertCache; `ensure`/`flushSlotTable` con `inFlight=true` ⇒ throw.
- **I2 — un token sporco non emette.** Il campionamento (argmax/sampling) legge
  esclusivamente logits di un forward PULITO (0 miss dichiarati dal readback).
- **I3 — il replay e' pulito per costruzione.** Dopo il repair, il replay dello
  stesso token NON puo' produrre nuovi miss (v. §4, pin-for-replay). Un replay
  sporco e' un bug strutturale ⇒ throw con diagnosi, mai un secondo replay.
- **I4 — determinismo del replay.** Il router di layer L nel replay riceve un
  hidden BIT-IDENTICO a quello del giro ottimistico (checkpoint, §3) e pesi
  identici ⇒ stessa selezione. E' cio' che rende valido il pin di §4.
- **I5 — prefill invariato.** Il path prefill (chunked M=16, sincrono, ensure
  per layer) non cambia di una riga. L'ottimistico e' SOLO decode.

## 2. Modo nuovo: `select: "optimistic"`

`select:"gpu"` resta com'e' (residenza totale, throw — it.17 non si tocca: e'
il riferimento e il caso M4). Modo nuovo con la stessa struttura a 1 submit:

- **Precondizione (build-time)**: `slots/park >= OPTIMISTIC_MIN_RESIDENCY`
  per ciascuna classe. Default **0.88** [ASSUMED: la soglia del regime WP-0;
  override esplicito `optimisticMinResidency` per i test, mai abbassabile in
  produzione senza docket]. Sotto soglia ⇒ throw con messaggio esplicito
  (specchio del messaggio it.17: qui il rimedio indicato e' "usare select
  'cpu' (sync) o aspettare C3c — il regime di scarsita' e' l'altro segmento
  della Pareto").
- **Rifiuto runtime**: nessuno switch automatico. La telemetria riporta
  P(dirty) osservato; se la finestra mobile [ASSUMED 64 token] supera
  `2*(1-minResidency)` si LOGGA un warning strutturato (una volta), non si
  cambia strada — il fallback dinamico e' materia C3c.

## 3. Il giro ottimistico (GPU)

Identico a `select:"gpu"` (un encoder, un submit, router+resolve per layer,
routing[] dal readback Sel di coda) con tre aggiunte:

- **3a — Guardia di miss nei kernel expert.** Il resolve puo' ora scrivere
  `Sel.slot = SLOT_TABLE_MISS` (0xffffffff). I kernel della catena expert
  (`pairGemvSiluFast`, `gemvAccumFast` in variante arena) con slot == MISS
  **contribuiscono zero e ritornano** (guard a inizio kernel, prima di ogni
  load): niente letture clamped di slab altrui, il valore del token sporco e'
  comunque spazzatura CONTROLLATA che il replay butta (I2). Costo nel path
  pulito: un confronto u32 per workgroup.
- **3b — Flag di miss piggyback.** Buffer `dirtyB` (2 × u32: `firstDirtyLayer`
  via atomicMin, `missCount` via atomicAdd), azzerato via `clearBuffer`
  nell'encoder del token (non writeBuffer: I1 non c'entra ma l'ordine si').
  Lo scrive il **resolve** (che gia' vede slot e layer): slot == MISS ⇒
  atomicMin(firstDirtyLayer, layerIdx), atomicAdd(missCount, 1). Copiato nello
  staging di coda INSIEME a Sel/hidden/logits: **stessa mapAsync, zero sync in
  piu'**. La lista degli expert mancanti non serve al flag: la CPU la ricava
  dal readback Sel (id con slot == MISS), che gia' legge.
- **3c — Checkpoint hidden.** Buffer `hiddenCkpt` da `nLayer × dModel × f32`
  (47 × 2048 × 4 = 385 KB ≈ il 376 KB/token del ruling 18a). Prima di ogni
  layer, `copyBufferToBuffer(x → hiddenCkpt[L])` nell'encoder (copy, non
  dispatch: il conteggio dispatch/token NON cambia). E' l'input che rende
  possibile il replay dal primo layer sporco e l'invariante I4.

## 4. Il confine di token (CPU): repair + replay

Al readback di coda (mapAsync unica su logits+hidden+Sel+dirty):

1. **Token pulito** (`missCount == 0`): si campiona, si aggiorna la LRU dalla
   Sel letta (touch degli expert usati — al confine, I1 rispettata), si passa
   al token successivo. Path identico a oggi + il touch.
2. **Token sporco**: (a) lista miss = { (layer, expertId) con slot == MISS }
   dal readback Sel; (b) **fetch** dei mancanti da expertstore (read OPFS +
   writeBuffer negli slot liberi/evinti — il costo modellato: 3.74 ms/miss
   serial, 2.66 coalesced, WP-0); (c) **pin-for-replay**: l'eviction di
   questo repair NON puo' toccare gli slot referenziati dalla Sel dei layer
   `>= firstDirtyLayer` (al piu' 4 × 46 layer MoE = 184 slot, sempre
   disponibili a budget >= 2419) — con I4 garantisce che il replay ri-selezioni gli stessi
   expert e li trovi TUTTI residenti ⇒ I3; (d) `flushSlotTable()` UNA volta,
   DOPO le writeBuffer degli slab (ordine R5: il dato prima della tabella);
   (e) **replay**: encoder nuovo da `hiddenCkpt[firstDirtyLayer]` (copy in
   `x`), layer firstDirty..47 + head, submit, readback di coda. Assert I3:
   `missCount == 0` nel replay, altrimenti throw. (f) campiona dal replay.
3. **LRU al confine**: touch nell'ordine osservato; insert dei fetch di
   repair. NIENTE prefetch predittivo al confine (esclusione §7).

Contabilita' attesa: path pulito = 1 submit / 1 sync; token sporco = 2 / 2.
Media steady-state = `1 + P(dirty)` ⇒ **<= 2 sync/token e <= 2 submits/token
gia' a P(dirty) <= 100%** — il gate strutturale del contratto e' soddisfatto
per costruzione se il meccanismo funziona; cio' che il bench misura davvero
e' che i contatori lo CONFERMINO in produzione (e la tassa, fase 5).

## 5. Telemetria (schema unico, it.22)

Contatori cumulativi nuovi in `CoreCounters` (null contagioso invariato):
`dirtyTokens`, `replaySubmits`, `misses`, `replayLayers` (somma dei layer
rigiocati), `repairFetchMs`, `repairUploadMs`. Il report glmbench deriva:
P(dirty) = dirtyTokens/forwards, miss/token, E[replayFrac|dirty] =
replayLayers/(dirtyTokens·47), tassa ms/token = repair + replay attribuiti.
Confronto WP-0 (fase 5): stesse definizioni del cost model del JSON
(`tax = E[nMiss]·fetchMs + P(dirty)·E[replayFrac|dirty]·gpuBusy`).

## 6. Test (fasi 2-3 di PHASES)

- **ktest flag+guard** (fase 2): expert forzati non-residenti (slotTable
  manipolata ad arte) ⇒ dirtyB riporta ESATTAMENTE quei miss (firstDirtyLayer
  e missCount attesi; 0 falsi positivi su run pulita); guard 3a: output della
  catena con slot MISS == output con contributo zero di riferimento.
- **ktest checkpoint** (fase 2): `hiddenCkpt[L]` bit-identico all'hidden di
  input del layer L del forward sincrono, per ogni L.
- **Identita' ottimistico-vs-sincrono** (fase 3, il test del contratto):
  stessa sequenza [ASSUMED >= 64 token, ctx ~500, GPU reale], argmax identico
  su TUTTE le posizioni; almeno 1 caso con miss forzato che attraversa il
  replay (e assert I3 esercitato) e 1 caso di rifiuto precondizione (throw
  con messaggio, pattern del test it.17).
- **Assert I1**: unit che verifica il throw di `ensure` a token in volo.

## 7. Cosa NON si costruisce (esclusioni del contratto)

- **Predictor GPU al confine di token** (LOOKA-al-confine): falsificato 3
  volte (WASTE, K3, WP-0: b=736 hit 63.2% → 59.2%). Il prefetch IN-FORWARD e'
  materia C3c e non entra qui.
- **Repair batched** (accumulare token sporchi): WP-0 dice che non serve nel
  regime near-total.
- **Variante a 2 segmenti** (1 sync mid-token per ridurre late-half 34-49%):
  resta nel cassetto; si riapre via docket SOLO se la tassa misurata in fase
  5 sfora la proiezione.
- Nessuna modifica a kernel non-expert, al path Qwen, al prefill.

## 8. Numeri attesi (proiezioni WP-0 — riferimento fase 5, non gate)

Al tetto misurato 2596 slot: P(dirty) 65.4%, 1.53 miss/token, 11.33 tok/s a
gpuBusy 54.2 ms. A slab 12 GiB (2419): 85.2%, 3.25, 9.55. Il confronto di
fase 5 avviene ALLO STESSO budget slot della run, su TUTTE le quantita' del
contratto (P(dirty), miss/token, frazione layer ripetuti, ms/token di tassa),
tolleranza [ASSUMED ±25%]; nota interpretativa: la LRU del sim e' LruFast,
non identica byte-a-byte alla nostra — uno scostamento su P(dirty)/miss va
prima attribuito li' e poi, se resta, dichiarato come da contratto.

## 9. Landmine note (ereditate, si applicano qui)

mapAsync SEMPRE dopo il submit del buffer (tsq-diag); mai mapAsync su staging
di un submit non emesso — nel replay lo staging di coda va unmappato PRIMA di
ri-encodare; error scope su ogni submit; var WGSL nei loop da azzerare (vale
per la guard 3a); run GPU lunghe ad albero congelato e ~60 s tra run
consecutive; bench a macchina scarica con hostState nel JSON.
