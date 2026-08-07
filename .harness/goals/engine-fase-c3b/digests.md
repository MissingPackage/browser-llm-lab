# Digests — engine-fase-c3b

## it.0 (2026-08-07) — setup: contratto v2 + PHASES, goal APERTO

- **Contratto v2** (sostituisce il chartered 2026-08-01): decode ottimistico +
  repair, gate STRUTTURALE (sync/token ≤ 2 in produzione + tassa entro WP-0
  ±25%), niente gate tok/s; paging → goal nuovo engine-fase-c3c (chartered).
- **7 fasi**: 1 spec → 2 flag miss+checkpoint (ktest) → 3 repair+identita' →
  4 glmbench+gate strutturale → 5 tassa vs WP-0 → 6 non-regressione piena →
  7 chiusura. Nessuna fase docket-born.
- **Primo target**: fase 1, la spec — punti fissati dal contratto, WP-0 e it.17
  come input.
- Decisioni prese su delega PI e registrate (docket item 2-3); plan-check
  pre-autorizzato (item 4).

## it.1 (2026-08-07) — fase 1 DONE: spec depositata e verificata

- **Spec scritta dal codice e da WP-0**: 1 submit ottimistico, miss marcati
  dal resolve (dirtyB piggyback sulla mapAsync di coda: zero sync in piu'),
  repair al confine con pin-for-replay => replay pulito per costruzione
  (max 1/token, sporco = throw), checkpoint hidden 376 KiB, prefill intatto.
- **Qualita' mai toccata per costruzione** (I2: un token sporco non emette);
  contabilita' 1+P(dirty) => il gate strutturale <= 2 e' della struttura,
  il bench di fase 4 deve solo confermarlo.
- **Verifier**: 7/7 punti contratto, riferimenti codice accurati, numeri
  coerenti col JSON WP-0; 2 note recepite (184 slot; confronto fase 5 su
  tutte le quantita'). Nessun gate/soglia toccato => nessun ruling richiesto.
- **Prossimo**: fase 2 — guard MISS + dirtyB + checkpoint in ktest, GPU reale.

## it.2 (2026-08-07) — fase 2 DONE: il meccanismo del miss e' misurato

- **ktest 68/68 su GPU reale** (+3): miss forzati marcati ESATTI (0 falsi
  positivi, controllo non-selezionato assente), degrado == riferimento f64 a
  expert azzerati (5.15e-8), checkpoint bit-identico gpu<->optimistic
  4096/4096 e riga0==xIn bit-exact; precondizione 0.88 con throw esplicito.
- La guardia MISS nei kernel ESISTEVA gia' (it.15, degrado definito): la fase
  si e' ridotta a dirtyB (binding 7 opt-in, WGSL esistenti byte-invariati),
  modo `select:"optimistic"`, checkpoint, hook harness.
- **1 submit/0 sync anche in optimistic** sul token pulito; dispatch/token
  invariati. Suite 337+7, tsc pulito, verifier PASS su 7/7 punti.
- **Prossimo**: fase 3 — repair + replay + identita' ottimistico-vs-sincrono.

## it.3 (2026-08-07) — fase 3 DONE: repair+replay, identita' BIT-esatta

- **Il claim del GOAL e' misurato**: 64 posizioni (3 con replay dal primo
  layer sporco), hidden e logits **bit-identici 131072/131072** vs decode
  sincrono, argmax 64/64, submits 67=64+3, 0 sync router. ktest 69/69.
- Replay = secondo giro da hiddenCkpt[firstDirty]; pin-for-replay ⇒ replay
  pulito per costruzione (replay sporco = throw I3, mai un secondo); I1
  (slotTable congelata in volo) con guard nel path + unit node.
- Deviazione dichiarata: identita' a ctx 64 (limite ktest), non ~500 — la
  scala vera e' fase 4/6. Suite 338+7, tsc pulito, verifier PASS 8/8.
- **Prossimo**: fase 4 — glmbench ottimistico sul modello vero, gate
  strutturale in produzione.

## it.4 (2026-08-07) — fase 4: l'ottimistico in produzione — 11.60 tok/s (+123%)

- **Prima misura assoluta**: decode 5.211 → **11.60 tok/s**, TTFT 17.88 →
  16.79 s, prefill 25.78 → 27.45, gpuBusy 54.2 → 39.4 (clock recovery
  reale). **Token generati 64/64 identici al sync** a ctx 525.
- **Due finding dal modello vero**, risolti e docketati: preload R4 (device
  perso ⇒ chunked async) e la CASCATA del replay (I3 falsificata ⇒ repair
  iterativo per prefisso, spec emendata).
- **Gate strutturale 2.188 > 2: FAIL dichiarato, ruling al PI (item 8)** —
  P(dirty) 93.8% × round 1.27; al tetto 2596 l'aritmetica e' 1.82 PASS.
  Opzione raccomandata: una run a sessione minima (azione host tua).
- **Prossimo**: fase 5 — tassa vs WP-0 (analisi, indipendente dal ruling).
