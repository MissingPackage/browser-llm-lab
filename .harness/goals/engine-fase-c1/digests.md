# Digests — engine-fase-c1

2026-07-30 — it.0 (setup). Goal aperto: misure sull'oracolo desktop che
dimensionano il paging esperti su GLM-4.7-Flash (metodo LOOKA di colibri,
contatori-only) — recall lookahead, skew/heat/working-set, curve hit-rate vs
budget da simulazione trace-driven. Nessun meccanismo browser: src/engine/**
è must-docket; C2 (MoE+MLA nel motore) e C3 (paging+instant-on) sono goal
futuri, hero-demo M4 PI-gated. 6 fasi sequenziali: risorse+recon → spec →
ROUTE_TRACE → LOOKA recall (timebox, fallback pre-negoziato) → residenza+
simulatore → sintesi+chiusura. Primo target: fase 1 (GGUF ~17 GB autorizzato,
build llama.cpp, smoke, recon punti di tap). Docket-born: plan-check (item 1)
— STOP prima dell'iterazione 1.

2026-07-30 — it.1 (fase 1 DONE, verifier PASS). Risorse oracolo pronte: GGUF
unsloth GLM-4.7-Flash-Q4_0 (17.2 GB, sha registrato — Q4_0 = layout del motore,
ruling PI), llama.cpp 5f55650a build CPU-only dedicata fuori repo, smoke greedy
exit 0. Throughput oracolo: pp512 56.6 / tg64 13.4 t/s (16 thread) ⇒ corpus 16k
posizioni ≈ 9 min/run. FINDING 1: zero patch necessarie — cb_eval + tensori già
nominati (ffn_moe_topk, ffn_moe_logits, ffn_norm) ⇒ rischio fase 4 ridotto.
FINDING 2 (docket item 2, PI-gated): il GGUF è arch deepseek2 e SENZA testa
NextN/MTP — implicazioni per fase D (spec-dec) e per il reader GGUF di C2.
46 layer MoE confermati dai metadati (sanity-gate fase 3 ok). Landmine nuova:
llama-cli UI chat ignora -no-cnv e loop-a su stdin chiuso — sempre -st
--simple-io nelle run scriptate. Next: fase 2 (spec C1) → ruling PI.

2026-07-30 — it.2 (fase 2 DONE, verifier PASS, STOP BY DESIGN). Spec C1 scritta
e ancorata ai numeri di it.1: tool C++ standalone via cb_eval (zero patch),
LOOKA calcolato online (pesi router ~24 MB estratti alla load, replica esatta
sigmoid→+bias→top-k — verificata dal verifier nel codice E nei tensori del
GGUF), corpus 8 prompt ≈16k posizioni (~9-12 min/run), recall@{4,6,8}
decode-only + baseline, simulatore TS+vitest con 4 policy × 6 budget, autotest
predittore ≥0.999 come gate hard dello strumento, NESSUN gate numerico sul
recall (misura, non target). RULING PI RICHIESTO: docket item 3, decisioni
(a)-(f). Fasi 3-6 gated — il goal riprende al ruling.
