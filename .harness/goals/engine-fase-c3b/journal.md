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
