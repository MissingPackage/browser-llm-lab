# Journal — engine-fase-c3b (decode ottimistico + repair)

## it.0 — 2026-08-07 — setup (goal-setup)

Contratto v2 scritto e committato (dfa8cec) su delega PI ("Scegli tu su questi
punti e poi procedi con il loop"); decisioni registrate in docket item 2 (split
C3b/C3c, gate strutturale, policy>LRU in C3c) e item 3 (banda di rumore
provvisoria). Tag goal-engine-fase-c3b-start su dfa8cec. PHASES.md: 7 fasi
sequenziali, nessuna docket-born (authority delta = none ovunque). Plan-check:
pre-autorizzato dalla stessa delega (docket item 4, pattern c3a item 3) — il
revert e' un edit di PHASES prima della fase 2.

## it.1 — 2026-08-07 — fase 1: la spec del decode ottimistico

### Cosa

`docs/superpowers/specs/2026-08-07-engine-fase-c3b-decode-ottimistico.md`,
scritta dal codice (glmmodel select:"gpu"/resolve/tail-readback, residency
slotTable/flush, telemetry schema unico) e dai numeri WP-0. Fissa i 7 punti
del contratto: flag miss piggyback (dirtyB 2xu32 atomici scritti dal resolve,
letto nella STESSA mapAsync di coda), checkpoint hidden 47x2048xf32 = 376 KiB
via copyBufferToBuffer (dispatch/token invariati), replay dal primo layer
sporco, slotTable congelata in volo (assert inFlight, insert/evict SOLO al
confine), prefill invariato, rifiuto build-time a slots/park < 0.88 [ASSUMED]
senza fallback dinamico, esclusioni (no predictor di confine, no repair
batched, variante 2-segmenti nel cassetto). Scelte nuove registrate a docket
item 5: modo `select:"optimistic"` (it.17 intatto), pin-for-replay (eviction
del repair mai sugli slot Sel dei layer >= firstDirtyLayer, <= 184 slot) =>
replay pulito per costruzione, max 1 replay/token, replay sporco = throw.
5 invarianti I1-I5 con assert; contabilita' attesa 1+P(dirty) submit e sync
per token => gate strutturale <= 2 per costruzione, il bench lo conferma.

### Verifica

loop-verifier (a32b0b61): griglia 7/7 punti del contratto FISSATI, riferimenti
al codice accurati riga per riga (glmmodel 339-361, residency 234/245/336-339,
telemetry 19/40-49), numeri coerenti col JSON WP-0 (steady 2596/2419, cost
model, localita'). Primo verdetto FAIL di FORMA (journal/digest/PHASES/HANDOFF
non ancora aggiornati — questa entry e il commit li saldano) + 2 note recepite:
188 -> 184 slot (46 layer MoE, non 47), confronto fase 5 su TUTTE le quantita'
del contratto (non solo tok/s), con attribuzione LruFast come nota
interpretativa. Nessuna violazione di constraint, nessun drift, gate/soglie
del contratto intoccati.

### Prossimo pezzo

Fase 2 (blocked-by-1 sciolto): guard MISS nei kernel arena + dirtyB +
checkpoint hidden, misurati in ktest su GPU reale.

## it.2 — 2026-08-07 — fase 2: flag di miss + checkpoint hidden, misurati in ktest

### Cosa

La scoperta che restringe la fase: la guardia MISS (spec 3a) ESISTEVA GIA'
(`arenaSlotWgsl`: `ok` propagato alle guardie, degrado definito, it.15) e il
resolve scriveva gia' `Sel.flags` bit 0. Costruito il resto:
- `routerTopKWgsl` opzione `resolve.dirty` -> binding 7 `dirtyB` (atomicMin
  del primo layer MoE sporco + atomicAdd dei miss); senza l'opzione il WGSL
  emesso resta BYTE-IDENTICO (shadow/gpu invariati, verificato dal verifier
  con un diff HEAD-vs-tree sul testo emesso).
- glmmodel `select:"optimistic"`: struttura a 1 submit di it.17 con
  precondizione near-total (slots/park >= 0.88 default, override solo test),
  preload a capacita' SENZA evict, reset dirtyB per token via writeBuffer
  (sentinel 0xffffffff — clearBuffer renderebbe ambiguo il layer 0; spec 3b
  allineata), staging dirty nella STESSA Promise.all di coda, `dirty` nel
  ritorno del forward con CROSS-CHECK kernel-vs-Sel (divergenza = throw).
- checkpoint hidden (`opts.checkpointHidden`): copyBufferToBuffer x ->
  hiddenCkpt[l] per layer nell'encoder (zero dispatch in piu'),
  `readHiddenCkpt()` per l'harness. `debugMarkMiss` (glmmodel+residency,
  SOLO harness) per miss deterministici.
- ktest: kit `mkMiniModelKit` estratto da testGlmModel2Layer (pura
  estrazione, +`zeroExperts` nel ref MoE = il degrado definito) + 3 casi.

### Verifica (GPU reale, /tmp/ktest-c3b-it2.log)

**ktest 68/68 PASS** (65 esistenti invariati +3):
- `glm-optimistic-forced-miss`: run pulita L2rel 5.98e-8 vs ref pieno,
  **1 submit/0 sync**, dispatch = piano+testa, **0 falsi positivi**; forzati
  {3,2} + controllo 4 marcato-ma-non-selezionato: missCount=2, first=1, miss
  esatti sui k giusti, controllo ASSENTE, flags per-k nella Sel di produzione;
  degrado == ref f64 a expert azzerati (L2rel 5.15e-8, argmax e logit in banda).
  Selezione attesa dal riferimento f64 INDIPENDENTE (no tautologia).
- `glm-hidden-checkpoint`: gpu==optimistic **4096/4096 bit**, riga0==xIn
  **2048/2048 bit**, riga1 vs input f64 del layer 1 L2rel 3.17e-7,
  dispatch/token invariati (le copy non sono dispatch).
- `glm-optimistic-precondizione`: throw con frammenti asseriti (7/64 slot).
`npm test` **337+7** invariata; `tsc --noEmit` pulito. Verifier (a2991b813):
PASS su tutti i 7 punti, incluse byte-identita' WGSL e assenza di regressioni
cpu/shadow/gpu (metriche identiche a c3a).

### Note di merito (dal verifier, recepite)

- Done-when (b) "bit-identico al forward sincrono": l'hidden intermedio NON e'
  bit-confrontabile fra path diversi (gia' in c3a gpu-vs-cpu 8309/12288). Il
  test usa gli ancoraggi piu' forti disponibili — gpu==optimistic bit-exact
  (il checkpoint del giro sincrono E' quello), riga0==xIn esatta, riga1 vs f64
  — che sono quelli che servono a I4 (replay = stesso input bit-identico).
  DICHIARATO QUI perche' il verifier di fase 7 non lo riapra.
- Landmine per la fase 4: il preload optimistic riempie in ordine (layer,
  expert) crescente — sul modello vero a ~88% lascia MISS sistematici gli
  expert alti degli ultimi layer. Se P(dirty) esce dalla proiezione WP-0,
  guardare PRIMA l'ordine di preload, poi la LRU.

### Prossimo pezzo

Fase 3: repair al confine (fetch dei mancanti, pin-for-replay, flush unico,
replay da hiddenCkpt[firstDirty], assert I1/I3) + test di identita'
ottimistico-vs-sincrono con miss forzato e caso di rifiuto.
