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
- `ffn_moe_logits` — logits del router (`src/llama-graph.cpp:1847`);
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

**Resta per chiudere la fase 1**: SHA-256 del GGUF, smoke run greedy exit 0 con
log in results/engine/moe-oracle/ + tok/s dell'oracolo (dimensiona il corpus in
spec). Poi verifier e chiusura iterazione.
