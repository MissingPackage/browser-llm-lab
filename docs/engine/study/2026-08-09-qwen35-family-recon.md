# Recon — famiglia Qwen 3.5/3.6 come target della generalizzazione

Data: 2026-08-09 · Input della direzione post-fase-C (docket c3c item 9).
Fonti: config.json ufficiali HF (Qwen3.6-35B-A3B, Qwen3.5-9B), ricerca hub.
Complemento: baseline nativa llama.cpp misurata lo stesso giorno
(`results/engine/native-baseline-llamacpp-vulkan-2026-08-09.json`).

## 1. La mappa (tutto Apache 2.0, tutto multimodale nativo)

| Serie | Densi | MoE | Note |
|---|---|---|---|
| **Qwen3.5** (feb 2026) | 0.8B / 2B / 4B / 9B / 27B | 35B-A3B / 122B-A10B / 397B-A17B | la scala completa |
| **Qwen3.6** (apr 2026) | 27B | 35B-A3B | refresh dei tier consumer top |
| Qwen3.8 | — | — | **solo `qwen3.8-max-preview` API: niente pesi open** (le "3.8" su HF sono finetune comunitarie di Qwen3-4B) |

Visione: tower separata nel config (`text_config` autosufficiente) ⇒ il
path testo si fa senza toccare la visione. GGUF: ecosistema maturo
(unsloth ~1M download per taglia, perfino GGUF con MTP) ⇒ **llama.cpp
supporta `qwen3_5`/`qwen3_5_moe` ⇒ l'oracolo di conformance del nostro
metodo si trasferisce intatto**.

## 2. L'architettura (IL fatto della recon): TUTTA la famiglia è ibrida

Densi E MoE condividono lo stesso scheletro (verificato su 9B e 35B-A3B):

- **`layer_types` 3:1** — 3 layer di **linear attention** (famiglia Gated
  DeltaNet: conv kernel 4, 16 K-head / 32 V-head @128, stato ricorrente,
  `mamba_ssm_dtype: float32` — combacia con la nostra regola f32-first)
  ogni 1 di **full attention** (GQA: head_dim 256, 2-4 KV head, partial
  rotary 0.25, mrope interleaved). Ctx 262 144.
- **KV solo sui layer full**: 10/40 (35B-A3B) o 8/32 (9B) ⇒ la KV crolla
  di ~4× a parità di contesto e lo stato lineare è O(1) —
  **il contesto lungo nel browser diventa quasi gratis** (il nostro slab
  ctx-aware ne beneficia direttamente).
- **MTP head nativa** (1 layer) su TUTTE le taglie ⇒ la fase D
  (spec-dec) ha il draft in casa, uniforme sulla famiglia.

## 3. Il MoE 35B-A3B (il modello-tesi successore di GLM-4.7-Flash)

- 40 layer (tutti MoE), **256 expert top-8** + 1 shared; hidden 2048,
  `moe_intermediate` 512 ⇒ **expert ≈ 3×2048×512 param ≈ 1.7 MB Q4**
  (⅓ del nostro 5.3 MB) — parco ≈ 10 240 slot piccoli, ~17-19 GB Q4.
- Conseguenze per la nostra cache: granularità 3× più fine ⇒ miss più
  economici (~1 ms freddo vs 2.9) e residenza più adattiva; router
  256-wide top-8 (vs 64 top-4) ⇒ recall del prefetch da rimisurare;
  slotTable e classi expert da parametrizzare (oggi GLM-shaped).

## 4. La lista dei prezzi (delta di engineering, in ordine di costo)

1. **Kernel linear attention WGSL** (DeltaNet gated: conv + delta rule +
   stato) — il pezzo nuovo grosso, scala ~fase-4b; non esiste nulla di
   simile nel motore. Prior art: ggml (implementazione CPU/CUDA di
   qwen3_5), kernel Triton ufficiali.
2. **Reader GGUF `qwen3_5(_moe)`** + tokenizer famiglia (vocab 248 320 ⇒
   head Q6_K ~2× la nostra: ~400-500 MB).
3. **Path full-attention GQA** (variante del path Qwen2.5 esistente:
   head_dim 256, partial rope 0.25, mrope interleaved, attn output gate).
4. **Parametrizzazione MoE** (nExpert/topK/shapes/classi slab oggi
   costanti GLM in residency/moe/glmmodel) — la meccanica C3c
   (ctx-aware, prefetch, tier, bandmodel) si trasferisce concettualmente
   intatta, i numeri vanno rimisurati.
5. Conformance harness per modello (il metodo c'è: golden+cpuref+firma).

## 5. Mappa taglie → baseline hardware consumer ("buone baseline")

| Tier | Modello | Q4 | Regime |
|---|---|---|---|
| 8 GB GPU | 9B denso (~5.5 GB) full-resident; 35B-A3B pagato (~45% residente) | il regime C3c di scarsità VERA | prefill-bound: il collo noto |
| 12 GB | 35B-A3B pagato (~65% residente) | C3c regime medio |
| 16 GB (dev-box) | 35B-A3B (~85-90% residente) | near-total: decode ottimistico |
| Apple M (unificata) | 35B-A3B full-resident | hero-demo (docket c3c item 8) |

## 6. Baseline nativa (il numero accanto al quale scriveremo il nostro)

llama.cpp b10333 Vulkan, stessa GPU/driver, stesso GGUF GLM, p512/n64:
**decode 66.6 tok/s / prefill 1230** (best, -ncmoe 8); perfino a expert
TUTTI su CPU fa 26.7/338. Noi: 15.6/35.7. Gap 4.3×/34×, ed è un limite
INFERIORE (Vulkan, non CUDA). La differenza architetturale chiave:
llama.cpp gli expert spillover li **computa dove stanno** (CPU,
compute-at-data), noi li muoviamo. Direzione futura legittima per il
browser: path expert WASM-SIMD (compute-at-data nel tab) — da valutare
nel goal di generalizzazione o dopo, NON prometterlo nel writeup.

## 7. Rischi dichiarati della scelta

1. Il kernel DeltaNet è il rischio tecnico dominante (nuovo, numerica
   ricorrente, f32 obbligato dal config — almeno coerente con noi).
2. mrope/multimodale: il path testo deve provare che il tokenizer/rope
   testo-solo è bit-fedele all'oracolo (mrope_section anche in text-only).
3. 256-expert top-8: il recall LOOKA 92% di GLM NON si assume — si
   rimisura (il metodo c'è).
4. La "3.8" che il PI cita come eccellente è API-only: la famiglia open
   di riferimento è 3.5/3.6 — se escono pesi 3.8 open, stessa architettura
   attesa, il lavoro si trasferisce.
