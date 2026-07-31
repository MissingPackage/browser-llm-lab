# PHASES — engine-fase-c1 (v1, contratto approvato dal PI 2026-07-30)

Decomposizione del contratto GOAL.md. Sequenziale (nessun parallel-group: le fasi
condividono tools/oracle-moe/ e results/engine/moe-oracle/). Cambiabile solo via
docket dopo l'iterazione 0. Gate d'ingresso: plan-check (docket item 1) — STOP
prima dell'iterazione 1.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | **Risorse oracolo + recon** — download GGUF GLM-4.7-Flash q4 (autorizzato dal PI), checkout+build llama.cpp dedicato, smoke run greedy, recon dei punti di tap nel sorgente upstream (routing MoE e hidden post-attention) | GGUF in `~/.cache/blab-models/` con SHA-256 registrato; build llama.cpp con commit registrato; smoke run greedy exit 0 con log/JSON in `results/engine/moe-oracle/` (incl. tok/s dell'oracolo, per dimensionare il corpus); recon nel journal con file:riga upstream dei punti di tap | none | `tools/oracle-moe/`, `results/engine/moe-oracle/`, `~/.cache/blab-models/` (fuori repo), checkout llama.cpp (fuori repo) | **done** (2026-07-30 it.1: GGUF unsloth Q4_0 sha d0bbdf…, oracolo 5f55650a CPU-only, smoke exit 0, pp512 56.6/tg64 13.4 t/s, recon zero-patch via cb_eval — arch GGUF=deepseek2, 46 layer MoE confermati, tap `ffn_norm`; verifier PASS) |
| 2 | **Spec C1** — metodo di strumentazione (patch counters-only, tap hidden), corpus definitivo, definizioni metriche, policy da simulare, sanity-gate, timebox fase 4 con clausola fallback | `docs/superpowers/specs/*engine-fase-c1-design.md` esiste con sezioni grep-abili ("Strumentazione", "Corpus", "Metriche", "Policy simulate", "Sanity-gate", "Timebox e fallback", "Rischi"); entry richiesta ruling appesa a `docket.md` | none | `docs/superpowers/specs/`, `.harness/goals/engine-fase-c1/` | **done** (2026-07-30 it.2: spec scritta, 7 sezioni, ruling richiesto — docket 3; verifier PASS. Fasi 3-6 gated dal ruling) |
| 3 | **ROUTE_TRACE** — patch counters-only che logga il routing vero (posizione, layer, top-4 id) sul corpus fisso | patch-file committato in `tools/oracle-moe/`; run documentata exit 0; traccia in `results/engine/moe-oracle/` con SHA-256 GGUF, commit llama.cpp, n_threads, greedy; sanity: righe = posizioni × 46 layer routed esatte | none | `tools/oracle-moe/`, `results/engine/moe-oracle/`, `scripts/` | **done** (2026-07-31 it.3: trace.cpp zero-patch via cb_eval, 31.274 posizioni [26.154p+5.120d] gate 2×, sanity PASS anche indipendenti; finding inp_out_ids→logits ovunque; verifier PASS) |
| 4 | **LOOKA recall** — tap hidden post-attention di L, router di L+1 applicato, recall@K vs routing vero; baseline "stessi expert del token precedente" sulla stessa traccia | report JSON con recall aggregato e per-layer per K∈{4,6,8} E baseline token-precedente; sanity-gate di spec verde (la_tot coerente) | none | `tools/oracle-moe/`, `results/engine/moe-oracle/` | **done** (2026-07-31 it.4, 1 it. su 3 di timebox — fallback NON esercitato: autotest 0.999997, decode R@4 0.775 / R@6 0.877 / R@8 0.920 vs baseline 0.323, per-layer 45+46; verifier PASS con baseline riprodotta indipendentemente) |
| 5 | **Residenza + simulatore** — statistiche offline dalla traccia (usage per (layer,expert), cumulativa top-N, working-set a finestra, co-attivazione se gratis) + simulatore trace-driven delle policy | JSON statistiche committato; simulatore committato con test verdi (`npm test` o `uv run pytest`, scelta in spec); JSON curve hit-rate vs budget per {LRU, LFRU tier.h, LFRU+pin, LFRU+pin+prefetch@recall}; punto di lavoro 4090 evidenziato; verdetto su "modello ~2× la memoria" nel report | none | `tools/oracle-moe/`, `tests/`, `results/engine/moe-oracle/` | pending |
| 6 | **Sintesi + chiusura** — la decisione PREPARATA, non presa | docket con proposta go/no-go PILOT per C2 (numero misurato + costo-miss da results/opfs-bench); ledger §A/§I e direction §7 aggiornati dove stale; checklist DONE WHEN 6/6 nel journal; HANDOFF aggiornato; merge+push a goal chiuso DOPO verifier gate PASS (ruling permanente) | none | `docs/engine/`, `HANDOFF.md`, docket, journal | pending |

Note di spine:
- Fase 1 PRIMA della spec by design (pattern B2 fase 1): la spec ha bisogno di due
  fatti misurati — il throughput dell'oracolo (dimensiona il corpus: ≥16k posizioni
  a N tok/s = ore di run?) e i punti di tap reali nel sorgente upstream (il rischio
  tecnico del goal sta lì). Fase 1 non scrive strumentazione, solo recon.
- Il tap hidden (fase 4) è separato dal ROUTE_TRACE (fase 3) perché il fallback
  pre-negoziato del contratto scarta SOLO il recall: skew, working-set e
  simulazione (senza prefetch) sopravvivono a un fallimento della fase 4.
- La baseline token-precedente sta in fase 4 col recall (stesso report, stessa
  traccia — contratto), ma si calcola offline dalla traccia di fase 3: se la
  fase 4 va in fallback, la baseline migra al report di fase 5 via docket.
- src/engine/** MAI toccato in questo goal (must-docket): l'unica scrittura nel
  repo principale è tools/, tests/, scripts/, results/, docs/.
- Benchmark pubblico e hero-demo M4: fuori dal goal by contract.
