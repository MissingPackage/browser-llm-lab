# Journal — engine-fase-c2

## it.1 — fasi 1+2 in parallelo (2026-07-31)

**Fase 2 (spec) DONE-pending-ruling**: dump header GGUF (metadata deepseek2
completi + inventario 844 tensori) ⇒ TRE fatti che la spec recepisce:
1. **quant MISTA nel file "Q4_0"** (ffn_down_exps Q4_1 su tutti i 2944
   expert; shexp Q5_K/Q6_K; output Q6_K; proiettori MLA Q8_0) ⇒ servono
   kernel dequant nuovi, repack escluso (romperebbe la conformance);
2. metadati MLA confermano la formulazione absorbed come design point
   (cache 576/token/layer = 54 KB/token f16, direction §3);
3. gating: sigmoid (func=2) + exp_probs_b per la selezione,
   weights_norm=true, weights_scale=1.8 per il mixing.
Spec: `docs/superpowers/specs/2026-07-31-engine-fase-c2-design.md` (10
sezioni); RULING RICHIESTO a docket item 2 con 6 decisioni (a)-(f), incluse
2 deviazioni dichiarate dal contratto: MLA absorbed (supera l'ASSUMED
naive-first) e OPFS come sorgente dei byte expert (il contratto diceva
staging RAM: 15.6 GB di ArrayBuffer in un tab su host 31 GB non reggono).
Fasi 3-6 restano gated dal ruling.

**Fase 1 (floor+golden)**: floor registrato nel docket allo scaffold
(decode 13.43 / prefill 56.58 tok/s, llama-bench C1). Tool
`tools/oracle-moe/golden.cpp` (nuovi file, zero modifiche ai file C1;
owns esteso rispetto alla riga PHASES — deviazione minore registrata qui):
prefill chunked 512 + decode greedy 128/prompt, argmax + top-32 logit f32
per posizione, envelope sha/commit/threads/corpusHash. Run canonica
lanciata sul corpus C1 (8 prompt); esito in coda a questa entry.

**Fase 1 esito (run 2026-07-31, exit 0, ~10 min)**: golden scritto —
`results/engine/golden/glm47flash/golden-glm47flash-q4_0-2026-07-31.json`
(647 KB): 26.154 prefill (IDENTICO alla traccia C1: stesso corpus/template,
sanity incrociata), 1.024 posizioni golden (8×128, zero EOS anticipati),
argmax+top-32 f32 ordinati, envelope sha/commit/corpusHash verificati.
Cross-check indipendente: i 128 token greedy per prompt coincidono 8/8 coi
primi 128 decode della traccia C1 (stesso oracolo, tool diverso ⇒ il golden
e la traccia sono mutuamente coerenti). Checkout oracolo pulito, commit
5f55650a. Done-when fase 1: run exit 0 ✓, JSON con envelope ✓, floor nel
docket ✓ (registrato allo scaffold). FASE 1 DONE (pending verifier).
