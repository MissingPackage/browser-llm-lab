# Ideas ledger — nuovo motore di inferenza browser

Registro append-only delle idee emerse nelle sessioni di stima/brainstorm (2026-07-28/29),
perché non si perdano durante lo sviluppo. NON è un piano: è l'inventario da cui i piani
pescheranno. Ogni idea ha: asse (su quale budget agisce), prior art, stato.

**Cornice** (brain dump PI 2026-07-28): obiettivo = massima intelligenza sotto vincolo di
rate "sufficiente" (soglia morbida ~30 tok/s, più alta per reasoning; l'oggetto vero è la
curva di Pareto qualità-vs-rate per device) e memoria ≤ host. Due budget:
`touch/token = BW_eff × (1/rate − T_fisso)` e `residenza ≤ M_host`. Terzo asse:
intelligenza per byte (specializzazione). Contesto: tecnologia+benchmark+ricerca, non
prodotto — moonshot legittimi. Deliverable: motore, benchmark pubblico, ricerca/evals.

Stati: `stimata` (numeri in estimates.md) · `da-verificare` (prior art/fattibilità da
controllare) · `da-misurare` (esperimento definibile subito) · `moonshot` (frontiera).

> **Sweep di verifica eseguito 2026-07-29** (`verification-sweep-2026-07-29.md`):
> statuses aggiornati sotto. Fatti nuovi principali: llama.cpp ha un backend WebGPU
> upstream (LlamaWeb, con le nostre L1/L2 già dentro — valida la diagnosi, apre la
> decisione fork-vs-scratch); un MoE full-residency gira già in browser (LFM2);
> QuantSpec (Apple) è il prior art della cascata di precisione; SharedWorker+WebGPU
> non esiste ancora (fallback: leader-tab via Web Locks).

## A. Budget residenza — modelli più grandi a parità di spazio

| Idea | Prior art | Stato |
|---|---|---|
| Quantizzazione 4-bit fusa nel matmul (baseline) | MLC/GGML | stimata |
| Double-quant delle scale (~−8-10% memoria, gratis) | QLoRA double quantization | da-misurare |
| Sub-4-bit + residuo low-rank (`W ≈ Q + LR`, tutto quantizzato, <2.5 bit/param; fattori LoRA-finetunabili → compone con la primitiva adapter) | CALDERA (NeurIPS'24) ✅ | verificata (sweep 29/07) |
| KV quantizzata int4/int8 (a 8B/32k: KV f16 4.3 GB → ~1.1) | llama.cpp cache-type, KIVI | stimata (leva primaria) |
| KV architetturale: MLA (latente ~10-15× più piccolo); **TransMLA converte modelli GQA esistenti in MLA post-training** | DeepSeek/Kimi K2/LongCat; TransMLA ✅ | verificata; stato llama.cpp da leggere nel codice |
| KV cross-layer: CLA / YOCO (2-3× / un layer solo) | Character.AI, MSR YOCO | da-verificare |
| Ibridi SSM/linear-attention (stato O(1), KV quasi nulla) | Jamba, Granite 4.0, Qwen3-Next | moonshot (scan Mamba2 in WGSL) |
| KV eviction: H2O / SnapKV / sinks+sliding window | Xiao et al., H2O NeurIPS'23 | da-verificare + eval |
| MoE: residenza ≫ touch (cuneo 5× su M4 48 GB); per-expert buffers aggirano il binding cap | Qwen3-30B-A3B, DeepSeek-MoE | **candidato hero-demo** |
| Paging esperti GPU↔RAM↔OPFS con prefetch predittivo dal router (**routing predicibile 71.6% un layer avanti** — colibri/PILOT; learned pinning su routing heat persistito; spec-dec per nascondere la latenza di offload) | ds4, colibri, MoE-Infinity, DAOP, arXiv 2508.21706 ✅ | verificata (sweep 29/07); target realistico: modello ~2× la memoria, non 30× |
| Esperti lazy al load ("instant-on MoE": router+attention+esperti caldi subito, resto on-demand; usage skew + shared experts) | ds4 (non-routed residenti + cache on-miss, budget 80%) ✅ | verificata |
| ~~Motore in SharedWorker~~ → **leader-tab**: un tab detiene il motore (Web Locks), gli altri proxyano via BroadcastChannel | SharedWorker+WebGPU MAI shippato (intent fermo dal 2024) | corretta (sweep 29/07) |
| Store OPFS content-addressed, delta update tra fine-tune | — | da-verificare |
| requiredLimits negoziati dall'adapter (M4: binding 2³²−4) | probe di progetto | stimata (regola v0) |
| Split tensori oversize + indirezione | llama.cpp split-mode row | stimata (solo se target >8B dense) |
| **Quantizzazione asimmetrica per classe di tensore**: routed expert aggressivi (q2-q3), attention/shared conservativi; compone col residuo low-rank | antirez/ds4 (validata in produzione) | da-verificare (recon 2026-07-29) |
| Picco di load = stato finale (upload a chunk con backpressure; OPFS SyncAccessHandle come SSD) | — | **misurata (2026-07-29, 4090/NVMe)**: write 2.2 GB/s; read via API 7.5-11.7 GB/s a page-cache calda ⇒ l'overhead API non è il bottleneck, il freddo è disco-bound (`results/opfs-bench/`) |

## B. Budget touch — velocità a parità di modello e spazio

| Idea | Prior art | Stato |
|---|---|---|
| Piano statico + bind group precomputati (L1: −15% su 4090) | — (misura nostra) | stimata |
| Un submit/token, scratch preallocato (L2) | — | stimata (M2 per quantificarla) |
| Fusione grafo 270→~100 dispatch + sampling in 1 kernel (L3) | megakernel prior art | stimata — **esistenziale su mobile** (T_fisso 40 ms > budget 33) |
| Multi-step decode N=4 (L4: −15% 4090, −3/4% M4/S22) | vLLM num-scheduler-steps | stimata |
| Kernel dequant migliore (L5: −6% 4090, ~0 altrove — q4 già a 69-76% banda su M4/S22) | Marlin, mul_mat_q | stimata (ultima leva, non prima) |
| Percorso compute f16 dove c'è (+66% misurato S22) | run PI 2026-07-27 | misurata ✅ |
| Self-speculative: testine draft sullo stesso tronco, ~zero memoria extra, lossless con verifica corretta | EAGLE, Medusa, LayerSkip | da-verificare (leva regina desktop) |
| **Riuso draft model DeepSpec già trainati** (Eagle3/DFlash/DSpark × Qwen3-4/8/14B + Gemma-4-12B, checkpoint su HF, MIT) | deepseek-ai/DeepSpec ✅ | verificata; architettura draft da leggere nel codice |
| **Verifica schedulata su confidenza** (verificare solo quando il batch K ripaga T_fisso; soglia decisa dal roofline autotarato) | DSpark ✅ | verificata |
| Cascata di precisione: draft = stesso modello a INT4 (pesi+KV gerarchica INT4/INT8), accettazione >90%, ~2.5× | **QuantSpec (Apple, ICML'25)** ✅ | prior art trovato — da tradurre in browser |
| **Lezione eval dal campo**: il draft è quant-sensibile (MTP int4 → accettazione 0-4%, int8 ok — colibri) | colibri ✅ | acquisita → sez. D |
| Sparsità contestuale FFN (predictor → leggi solo righe attive) | PowerInfer, TEAL, Deja Vu | moonshot + eval (capitolo paper) |
| Pesi come texture su mobile ("image-backed weight banks" per GEMV caldi; attacca il 43% banda S22) | llama.cpp OpenCL Adreno, ufficiale Qualcomm + IWOCL'26 ✅ | verificata → **da-misurare** (una cella nel micro-bench) |
| Quest-style sparse attention: riassunti pagine su GPU, top-k pagine lette | Quest (Tang'24) | da-verificare |
| KV tiering GPU→RAM→OPFS guidato dalla selezione sparsa | CacheGen (SIGCOMM'24), LMCache | moonshot |
| Timestamp-query nativa + roofline su banda misurata (autotuning) | infra micro-bench nostra | stimata (regola v0) |

## C. Terzo asse — intelligenza per byte

| Idea | Prior art | Stato |
|---|---|---|
| **LoRA primitiva di prima classe**: `y = Q(W)x + B(Ax)` fusa nel kernel, adapter 10-40 MB hot-swap su base cachata | ardesia-gguf (lineage diretto); **gap browser confermato**: WebLLM no, wllama TODO, MediaPipe solo Gemma | verificata come gap (sweep 29/07) |
| Multi-adapter nello stesso batch | S-LoRA | da-verificare |
| Composizione adapter a runtime (somma pesata) | MoLE, model soups | moonshot + eval |
| **Training LoRA in-browser** su base quantizzata congelata (dati mai fuori; ardesia completo su WebGPU) | ardesia 35B/13GB; sweep 29/07: nessun prior art WebGPU-browser trovato (adiacenti: QVAC su Vulkan; MambaKit "WSLA" da ispezionare) | moonshot (claim sopravvissuto) |
| Decoding vincolato da grammatica/JSON-schema (CPU-side, rende affidabili modelli piccoli) | llama.cpp grammars, outlines | da-verificare |
| Prefix/KV cache persistente in OPFS (ri-entrata contesto in ~0.5 s vs re-prefill) | vLLM prefix caching, CacheGen, **ds4 "live graph KV checkpoint" su disco (semantica conversazionale completa)** | da-verificare |
| KV precomputata shippata via CDN (prefill zero per contesti noti; portabilità numerica cross-device = research question) | CacheGen | moonshot |
| Load progressivo pipelined (prefill layer 0 durante il download dei layer alti; TTFT ≈ max, non somma) | WebAssembly compileStreaming (principio) | da-verificare |

## D. Evals — misurare la perdita di intelligenza (richiesta PI 2026-07-29)

| Idea | Nota | Stato |
|---|---|---|
| KL / top-k agreement sui logit vs riferimento f16, corpus fisso | richiede logits: WebLLM non li espone, il motore nostro sì **by design** | da-progettare (WP) |
| Perplexity harness in-browser | il motore come loop di eval, non solo chat | da-progettare |
| Micro-suite per task (GSM8K subset, micro-NER anonimizzazione) | validità di faccia | da-progettare |
| Telemetria acceptance-rate speculative come proxy live | DeepSpec (eval draft su 9 benchmark: GSM8K, MATH500, HumanEval, LiveCodeBench…) | da-progettare |
| Cosa è lossless by construction: spec-dec con rejection sampling corretto | Leviathan/Chen 2023 | fatto (chiarito) |
| Cosa richiede eval: quant pesi/KV, eviction, sparse attention, sparsità FFN, composizione LoRA, ibridi | — | inventariato |
| Aggancio: docket #10 fase-1b (qualityScore mai collegato) | era in anticipo, non orfano | da-collegare |

## E. Benchmark pubblico (indipendente dal motore)

- Multi-metrica, utile agli utenti a prescindere dal motore. Headline **non decisa** —
  la curva di frontiera (max qualità sostenibile sopra soglia, per device) è UNA candidata.
  Opzioni aperte per ruling PI futuro.

## G. Motori di riferimento da studiare in profondità (recon 2026-07-29, indicati dal PI)

| Motore | Cosa insegna a noi | Fonte |
|---|---|---|
| **antirez/ds4** — DeepSeek 4 Flash/PRO su Metal/CUDA/ROCm, self-contained, narrow by design | Filosofia single-model; KV su disco con checkpoint conversazionale; quant asimmetrica routed-expert; contesti lunghi via KV compressa+SSD. La versione nativa del nostro paging OPFS | github.com/antirez/ds4 |
| **JustVugg/colibri** — GLM-5.2 744B MoE su 25 GB RAM, C puro, esperti streammati da disco, CPU-only, 1-2 tok/s | L'estremo dell'asse residenza; valida la formula dei due budget (touch bounded da banda SSD → regime 1-2 tok/s); da studiare la gestione località/cache esperti | github.com/JustVugg/colibri |
| **DSpark** (DeepSeek+PKU, giu 2026) — spec-dec semi-autoregressivo, +60-85% su V4 vs MTP-1 | Scheduling della verifica basato su confidenza; checkpoint V4 DSpark-flavored su HF | venturebeat + repo |
| **DeepSpec** (MIT) — train+eval di draft model (DSpark, DFlash, Eagle3) | Draft già trainati per Qwen3-4/8/14B e Gemma-4-12B = le nostre taglie desktop; harness eval spec-dec per la sez. D | github.com/deepseek-ai/DeepSpec |

Sintesi del recon: il mondo native converge sui nostri assi (MoE narrow + streaming esperti
+ KV su disco). **Gap statement raffinato dallo sweep 29/07**: nel browser esistono già
l'esecuzione MoE full-residency (LFM2 via Transformers.js), GGUF+WebGPU e il load OPFS
streaming (LlamaWeb/wllama) — ciò che NON esiste è il **sistema di memoria** (esperti
paginati con prefetch predittivo, KV tiering/checkpoint, adapter hot-swap, spec-dec) e la
**telemetria/eval integrata**. Riferimenti aggiunti dallo sweep: LlamaWeb (arXiv
2605.20706, backend WebGPU upstream di llama.cpp — apre la decisione fork-vs-scratch) e
LFM2-MoE (Liquid AI). Lo studio in profondità dei codebase resta un WP a sé.

## H. Modello target — ruling PI 2026-07-29 (docket #13)

**Modello-tesi: GLM-4.7-Flash** (30B-A3B, MLA, MIT, GGUF unsloth disponibili) · dev
rungs: Qwen3.5-0.8B/2B (Apache) · first-light: Qwen2.5-0.5B (grafo già noto al
dispatch-level) · v2 dichiarato: ibridi (Qwen3.5-35B-A3B / Nemotron-3-Nano).

Panorama architetturale che ha deciso (recon 2026-07-29, fonte Kaitchup + verifiche):

| Candidato | Attention | KV B/token | Licenza | Verdetto |
|---|---|---|---|---|
| Qwen3-30B-A3B (2025) | GQA classica | 98 304 | Apache | fallback: ultimo della specie, già battuto in intell./byte |
| **GLM-4.7-Flash** | **MLA** | 54 144 | **MIT** | **tesi**: KV compressa (allineata alla sintesi WP), ds4+colibri = reference GLM in casa, MTP nativa di famiglia [VERIFY su 4.7], kernel scope senza SSM |
| Qwen3.5-35B-A3B (2026) | ibrida (30 DeltaNet + 10 attn) | 20 480 | Apache | v2: mette i kernel linear-attention nel critical path di v0 |
| Nemotron-3-Nano | ibrida Mamba-2 | 6 144 | NVIDIA | v2 |
| LFM2.5-8B-A1B | ibrida conv | piccola | LFM custom | classe 4-6B dense: eventuale gradino mobile, non tesi |

Nota (domanda PI sui piccoli MoE): il rischio "stupidità" è di taglia, non di quant —
la quant estrema (Q2/Q3) non serve all'hero su M4 (q4 ~17 GB in 48 GB); serve solo per
la 4090/16 GB, dove le risposte sono paging o ricetta asimmetrica ds4 + evals.

## F. Decisioni di cornice già prese (brain dump 2026-07-28/29)

- Obiettivo = intelligenza sotto vincolo di rate sufficiente, non max tok/s.
- Desktop first (4090 = dev-loop scriptabile, M4 = cuneo MoE/hero-demo); mobile = sogno
  dichiarato, da affrontare con la guerra al floor (fusione dispatch) quando sarà il momento.
- È ricerca/tecnologia/benchmark, non prodotto: Chrome built-in non è un concorrente,
  la tassa-browser sulla residenza è un parametro, i moonshot sono capitoli legittimi.
- Le evals di perdita d'intelligenza sono parte del progetto, non un afterthought.

## I. Rimandi espliciti di fase (registro; aperto in B1, ruling PI 2026-07-30)

Ogni riga è una generalità deliberatamente NON costruita, col momento in cui torna
utile e il trigger che la riattiva. Chi apre una fase nuova legge questa tabella.

| Rimando | Deciso in | Riprende in | Trigger di riattivazione | Riferimento |
|---|---|---|---|---|
| Lookup longest-prefix nella prefix-cache (ds4 `find_text_prefix`: vince il prefisso salvato più lungo) | B1 (v1 = match esatto) | fase C | il paging rende normali i prompt che crescono tra sessioni; >~50 entry in cache | spec B1 §Formato prefix-cache; study/ds4.md §2 |
| Chiave della cache su testo renderizzato (anti-ritokenizzazione BPE) + migrazione della chiave token-id | B1 (chiave = hash token-id) | quando arriva il tokenizer (D/evals) | primo caso d'uso con testo libero in ingresso | spec B1 §Formato prefix-cache (layoutVersion nell'envelope = punto di migrazione) |
| Logits dell'ultima posizione nel checkpoint KV (ds4 li salva per riprendere il sampling senza forward extra) | B1 (si ricalcola 1 forward, ~8 ms) | D (spec-dec) o instant-on fase C | il costo del forward di ripresa smette di essere trascurabile (batch di sessioni, modelli grandi) | spec B1 §Formato prefix-cache |
| Eviction con scoring a densità (ds4: half-life hit 6h, fattore 2× per anchor) | B1 (LRU semplice) | fase C | la cache supera ~100 entry o compaiono pattern di riuso non-LRU | study/ds4.md §2; spec B1 §Quota/eviction |
| Floor dispatch ≤100/token | split B1/B2 (PI 2026-07-29) | B2 | goal-brief B2 — DA RI-INQUADRARE: col liv.2 funzionante, GPU busy ≈2.2 ms/token vs 8.1 wall ⇒ ~73% del decode è sync/encode, non dispatch | docket goal B1 item 2; tsq-diag-2026-07-29.md §Conseguenze |
| Fusioni cross-layer/megakernel viste durante B1 ma non implementate | B1 (must-docket da contratto) | B2 | si annotano nel docket del goal B1 man mano | GOAL.md B1 §must-docket |
| Parametro M del piano prefill riusato come batch di verifica spec-dec | A (spec §Piano statico) | D | costruzione del verificatore speculativo | spec A §Piano statico; spec B1 §Forward multi-token |
| Telemetria livello 3 (un pass per step, timestamp per-op) | A (mai default) | diagnostica ad-hoc | solo per indagini; mai nei numeri pubblicati | spec A §Telemetria |
