# Journal — engine-fase-c1

(append-only; una entry per iterazione, con evidenza e verdetto verifier)

## it.1 — fase 1: risorse oracolo + recon (2026-07-30, IN CORSO — gated dal download)

Plan-check RISOLTO (docket 1): ruling PI in chat, GGUF = layout del motore ⇒ Q4_0.

**Scelta GGUF**: `unsloth/GLM-4.7-Flash-GGUF` → `GLM-4.7-Flash-Q4_0.gguf` (17.22 GB).
Motivazione: Q4_0 puro = il layout quant che i kernel dequant-fusi del motore leggono
(criterio del ruling; stesso formato di qwen2.5-0.5b-instruct-q4_0.gguf dell'oracolo
fase A); unsloth già citato in direction §3; bartowski ha lo stesso Q4_0 (17.39 GB)
come alternativa. Download in corso in `~/.cache/blab-models/` (~2 MB/s ⇒ ~2h);
SHA-256 da registrare a download finito.

**Build oracolo**: checkout dedicato `~/Projects/llama.cpp-oracle` (FUORI repo),
`ggml-org/llama.cpp` commit `5f55650a78f92aff4d48d671423e888fac0469ff` (2026-07-30,
shallow). Build CPU-only Release (`-DGGML_CUDA=OFF -DLLAMA_CURL=OFF`, nvcc assente),
32 core: `llama-cli` e `llama-quantize` compilati, exit 0. CPU-only è coerente col
metodo colibri (thread pinnati, determinismo; niente varianza numerica GPU).

**Recon punti di tap (file:riga al commit sopra) — FINDING: zero patch necessarie.**
llama.cpp espone un eval-callback di scheduling (`cb_eval`, `include/llama.h:375`,
`common/common.h:488`) con pattern ask/observe già dimostrato in
`common/debug.cpp:143-185` (filtro regex sul nome tensore + `ggml_backend_tensor_get`)
e `examples/eval-callback/`. I tensori che servono sono GIÀ nominati per layer via
`cb(...)`:
- `ffn_moe_logits` — logits del router (`src/llama-graph.cpp:1846`);
- `ffn_moe_probs` / `ffn_moe_probs_biased` — probs sigmoid + bias di selezione
  (gating GLM = SIGMOID di default, `src/models/glm4-moe.cpp:17-20`; exp_probs_b
  stile DeepSeek-V3, `llama-graph.cpp:1883-1887`);
- `ffn_moe_topk` — expert selezionati `[n_expert_used, n_tokens]`
  (`src/llama-graph.cpp:1929`, via `ggml_argsort_top_k`);
- `post_attn_norm` — hidden post-attention normato, INPUT del router
  (`src/models/glm4-moe.cpp:219`) = lo stato su cui LOOKA applica il router di L+1;
- `l_out` — output di layer (`glm4-moe.cpp:262`), alternativa per la definizione
  esatta del predittore (decisione di spec, fase 2).
⇒ Il tool di traccia = eseguibile C++ NOSTRO in `tools/oracle-moe/` che linka la
build (pattern eval-callback), contatori/dump nel tool. Il constraint "llama.cpp
SOLO oracolo, mai vendored" resta intatto: nessuna modifica upstream. Il rischio
tecnico della fase 4 (tap hidden) si è ridotto: il tap è un filtro sul nome.

**Conteggio layer routed (per sanity-gate fase 3)**: da confermare allo smoke coi
metadati GGUF — `n_layer_dense_lead` e `n_layer_nextn` sono hparams
(`glm4-moe.cpp:12,23`); il builder SALTA il layer NextN nel forward
(`glm4-moe.cpp:52-53`). Il "46 MoE attesi" del GOAL va verificato (47 layer di
config upstream − dense lead − NextN: la spec fissa il numero esatto dai metadati).

**Completamento (stessa iterazione, dopo il download):**

- **GGUF verificato**: 17.22 GB in `~/.cache/blab-models/GLM-4.7-Flash-Q4_0.gguf`,
  SHA-256 `d0bbdfcde6e323ebf90a8b9e95da57100e972be1ec6f0bfa0fad0feaa426557e`.
- **CORREZIONE recon**: il GGUF unsloth è arch **`deepseek2`**, NON `glm4moe`
  (gguf-dump). Il builder effettivo è `src/models/deepseek2.cpp`; `build_moe_ffn`
  è condiviso ⇒ `ffn_moe_logits`/`ffn_moe_probs`/`ffn_moe_topk` restano validi
  (llama-graph.cpp:1847/1874/1929); il tap hidden post-attention in deepseek2 si
  chiama **`ffn_norm`** (`deepseek2.cpp:376`; alternative: `attn_out`:231,
  `l_out`:419). MLA confermata dai metadati (q_lora 768, kv_lora 512, rope 64 —
  direction §3 ok). **Niente NextN nel GGUF** (testa MTP droppata dalla
  conversione) → FINDING fuori-scope a docket item 2 (implicazioni fase D e C2).
- **Conteggio per sanity-gate fase 3 CONFERMATO**: block_count 47, dense lead 1,
  niente layer NextN nel forward ⇒ **46 layer MoE routed** (64 expert top-4 +
  1 shared, gating sigmoid, exp_probs_b stile DeepSeek-V3).
- **Smoke greedy exit 0**: `llama-cli -st --simple-io --temp 0 -t 16 -n 64` —
  output coerente (Rayleigh), log committato. NOTA STRUMENTO: la nuova UI chat
  di llama-cli IGNORA `-no-cnv` e loop-a all'infinito su stdin chiuso (676 MB di
  "> " nel primo tentativo, processo killato) ⇒ per run scriptate SEMPRE
  `-st --simple-io`, o llama-bench. Landmine da riportare in HANDOFF.
- **Throughput oracolo (llama-bench, 16 thread, r=2, cache calda)**:
  **pp512 56.58 ± 3.74 t/s; tg64 13.43 ± 0.10 t/s** (stddev decode minima ⇒
  niente thrashing nonostante host RAM 31 GB — il working set MoE per token è
  piccolo, effetto-colibri osservato di passaggio). Fattibilità corpus: 16k
  posizioni ≈ 9 min/run ⇒ la spec può permettersi corpus 2-3× senza problemi.
- Artefatti: `results/engine/moe-oracle/{oracle-smoke-2026-07-30.json,
  smoke-glm47flash-q4_0-2026-07-30.log, llama-bench-glm47flash-q4_0-2026-07-30.json}`.

**Done-when fase 1**: GGUF+SHA ✓, build+commit registrati ✓, smoke exit 0 con
log/JSON e tok/s ✓, recon file:riga nel journal ✓ (corretto deepseek2).
Verifier: **PASS** (agent loop-verifier, 2026-07-30 — tutti i punti con evidenza;
drift none; violazioni none; note minori applicate: sizeBytes esatto 17216676192,
ffn_moe_logits riga 1846). FASE 1 DONE.

## it.2 — fase 2: spec C1 (2026-07-30)

Spec scritta: `docs/superpowers/specs/2026-07-30-engine-fase-c1-design.md`,
ancorata ai fatti misurati di it.1 (arch deepseek2, 46 MoE, gating
sigmoid+exp_probs_b, pp/tg misurati). Sezioni: Strumentazione (tool C++
standalone via cb_eval, zero patch; LOOKA online con estrazione pesi router e
replica esatta dell'ordine di selezione di build_moe_ffn), Corpus (8 prompt,
~16k posizioni, prefill/decode separati), Metriche (recall@K decode-only,
baseline_prev, skew, working-set), Policy simulate (TS+vitest; LRU, LFRU
tier.h, +pin split-temporale, +prefetch con replay e guard), Sanity-gate
(la_tot 45/46 da metadati asseriti a runtime; autotest predittore ≥0.999),
Timebox e fallback (3 it.), Rischi (5, incl. fedeltà predittore e osservatore).
Richiesta ruling appesa: docket item 3, decisioni (a)-(f). Done-when fase 2:
spec esiste con sezioni grep-abili ✓, entry docket ✓.
STOP BY DESIGN dopo verifier: fasi 3-6 gated dal ruling PI.
Verifier: **PASS** (loop-verifier, 2026-07-30 — 7 sezioni presenti; coerenza
spec↔journal e spec↔GOAL verificate; claim tecnico sigmoid→+bias→top-k
verificato NEL CODICE a commit 5f55650a e nei metadati GGUF, incl. esistenza di
blk.N.ffn_gate_inp.weight/exp_probs_b.bias e ASSENZA di ffn_gate_inp.bias;
src/engine intatto; drift none). FASE 2 DONE, ruling pendente (docket 3).
