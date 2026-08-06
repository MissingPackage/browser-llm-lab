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
