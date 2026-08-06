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
