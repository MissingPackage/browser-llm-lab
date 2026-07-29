# Verification sweep — 2026-07-29

Sweep search-level delle righe `da-verificare` del ledger (`ideas-ledger.md`), approvato
dal PI. Metodo: WebSearch + lettura diretta di repo/paper (WebFetch). Non sostituisce lo
studio in profondità dei codebase (WP separato): verifica esistenza, forma e numeri dei
prior art, e la tenuta dei gap statement.

## 1. I quattro fatti che cambiano il quadro strategico

### 1a. llama.cpp ha un backend WebGPU upstream ("LlamaWeb", mag 2026) — il panorama è cambiato

Paper UCSC + MSR (Reese Levine et al., [arXiv 2605.20706](https://arxiv.org/html/2605.20706v1)):
backend WebGPU per llama.cpp, **upstreamato con 100+ PR**, 23 formati quant con dequant
templata nei kernel. Tecniche: **allocazione statica dei buffer + arena rotante di slot
preallocati** (= le nostre L1/L2, indipendentemente diagnosticate) e **load via OPFS con
streaming asincrono** (= la nostra idea di load). Risultati: decode **45-69% più veloce**
di WebLLM/Transformers.js, memoria −29-33%. E wllama v3.1 lo shippa già
([github](https://github.com/ngxson/wllama), WebGPU auto-abilitato).

Tre implicazioni:
- **Valida la nostra diagnosi**: la loro accelerata da "static allocation + arena" è
  nello stesso ordine del nostro 1.44× stimato per L1-L3. Due strade indipendenti,
  stessa conclusione.
- **Chiude due nostre righe come novità**: "GGUF+WebGPU nel browser" e "load OPFS
  streaming" esistono già.
- **Fianco scoperto**: il loro prefill è 21-51% PIÙ LENTO dei competitor browser, e
  fino a 10× sotto CUDA/Metal su alcuni device. Il prefill WebGPU resta un problema
  aperto per tutti.
- Dichiarati assenti: **MoE/expert streaming, LoRA, speculative decoding** (KV quant
  parziale). Il nostro spazio di ricerca vive esattamente lì.

**Nuova decisione PI (docket)**: motore from-scratch o costruito sopra
llama.cpp-WebGPU? Da scratch: controllo totale del piano statico/fusione/telemetria
(la nostra tesi), nessun vincolo ggml. Sopra llama.cpp: 23 quant gratis, zoo GGUF,
MLA/MoE del core ggml, e si eredita il lavoro di Levine — ma il grafo è ggml, non
nostro, e le leve L3 (fusione aggressiva) e telemetria nativa vanno negoziate con
l'architettura loro. Input per la decisione: lo studio in profondità (WP).

### 1b. Un MoE gira già nel browser — ma full-residency

Liquid AI **LFM2-MoE 8.3B-A1.5B** interamente client-side via Transformers.js/WebGPU
([webgpu.com showcase](https://www.webgpu.com/showcase/lfm2-moe-8b-language-model-browser-webgpu/)).
Nessun paging/streaming: il modello sta tutto in memoria. **Il gap statement si
raffina e sopravvive**: ciò che non esiste nel browser è il *sistema di memoria*
(esperti paginati da OPFS, prefetch predittivo, KV tiering, adapter, spec-dec) — non
l'esecuzione MoE in sé.

### 1c. "Cascata di precisione" ha già un nome: QuantSpec (Apple, ICML 2025)

Self-speculative col **draft = stesso modello a INT4 (pesi e KV gerarchica INT4/INT8)**:
accettazione >90%, speedup ~2.5×, kernel custom
([arXiv 2502.10424](https://arxiv.org/abs/2502.10424)). La nostra riga moonshot
diventa "prior art da tradurre in browser". Nota: la gerarchia INT4/INT8 della KV
senza ri-quantizzazione on-the-fly è un'idea riusabile anche fuori dallo speculative.

### 1d. SharedWorker + WebGPU: non esiste ancora in nessun browser

Approvato dal Community Group, Intent-to-Prototype Chrome fermo dal 2024, **mai
shippato** ([chromestatus](https://chromestatus.com/feature/4875951026733056),
[mozilla/standards-positions#971](https://github.com/mozilla/standards-positions/issues/971)).
La riga "una copia del modello per N tab" cambia forma: **fallback leader-tab** — un
tab detiene il motore (elezione via Web Locks), gli altri proxyano via
BroadcastChannel/postMessage. Funziona oggi, va solo progettato.

## 2. Verdetti sui motori di riferimento (letture dirette)

### ds4 ("DwarfStar", antirez) — [github](https://github.com/antirez/ds4)
- **Quant asimmetrica, ricetta precisa**: SOLO i routed expert quantizzati (up/gate
  IQ2_XXS, down Q2_K); shared expert, proiezioni, routing a precisione piena. ✅
- **KV checkpoint su disco**: snapshot SHA1-keyed con testo renderizzato + token id +
  righe compresse per layer; salvataggi incrementali a boundary configurabili
  (default allineamento 2048, minimo 512). Design direttamente traducibile in OPFS. ✅
- **Streaming**: non-routed residenti; routed in cache RAM con load-on-miss dal GGUF;
  budget cache = 80% del working set; prefill streaming sovrapposto con riserva di 2
  layer routed. ✅
- Perf: M5 Max 128 GB, V4 Flash Q2: **~26 tok/s** gen; PRO Q2 streaming ~9.6.
  Speculative opzionale = **DSpark come modello ausiliario**. ✅

### colibri (JustVugg) — [github](https://github.com/JustVugg/colibri)
- GLM-5.2: 744B totali, **~40B attivi/token**, 19 456 routed expert; densi ~17B
  residenti int4 (~9.9 GB), esperti streammati da ~370 GB su disco.
- **Il numero che ci serviva**: routing **predicibile al 71.6% un layer avanti**
  (thread di lookahead "PILOT") — il prefetch predittivo del nostro ledger ha una
  costante misurata. ✅
- **Learned pinning su "routing heat"** persistito su disco (`.coli_usage`): il motore
  "diventa più veloce con l'uso". Versione browser: profilo di routing in OPFS. ✅
- **Lezione per le evals**: il draft head MTP a int4 collassa l'accettazione a 0-4%
  (int8 ok) — **il draft è quant-sensibile**, va eval-ato a parte. ✅
- Perf reali: 0.05-0.1 tok/s a 25 GB baseline; ~1.8 a 128 GB CPU; 5.8-6.8 con
  residenza piena su 6× RTX 5090. La formula dei due budget predice il regime. ✅

### DeepSpec — [github](https://github.com/deepseek-ai/DeepSpec)
MIT ✅; tre algoritmi (DSpark, DFlash, Eagle3) × quattro target (**Qwen3-4B/8B/14B,
Gemma-4-12B-it**) con checkpoint su HF ✅; harness eval su 9 benchmark ✅.
Portabilità dell'architettura draft: non documentata nel README → **codice da leggere
nel WP di studio**.

## 3. Verdetti sulle altre righe del ledger

| Riga | Verdetto | Fonte / nota |
|---|---|---|
| Sub-4bit + residuo low-rank | ✅ CONFERMATA e migliore del previsto: CALDERA (NeurIPS'24) fa W≈Q+LR con Q,L,R tutti quantizzati, regime <2.5 bit/param, e **i fattori low-rank sono LoRA-finetunabili** → compone con la primitiva adapter | [arXiv 2405.18886](https://arxiv.org/abs/2405.18886), [pilancilab/caldera](https://github.com/pilancilab/caldera) |
| MLA | ✅ adozione ampia (DeepSeek, Kimi K2, LongCat); **TransMLA converte modelli GQA esistenti in MLA post-training** (leva nuova); stato llama.cpp non confermato dallo sweep → codice da leggere | [TransMLA](https://arxiv.org/pdf/2502.07864) |
| Paging esperti + prefetch | ✅ letteratura ricca oltre a ds4/colibri: MoE-Infinity, DAOP, survey caching/prefetching; e **spec-dec usato per NASCONDERE la latenza di offload esperti** (sinergia nuova tra due nostre righe) | [arXiv 2508.21706](https://arxiv.org/pdf/2508.21706), [arXiv 2511.05814](https://arxiv.org/pdf/2511.05814) |
| Pesi come texture su mobile | ✅ CONFERMATA, ufficiale: backend OpenCL llama.cpp per Adreno con "image-backed weight banks" per GEMV caldi (Q4_0/Q4_K_M), half4/half8, subgroup ops; paper IWOCL 2026 | [Qualcomm blog](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu), [IWOCL'26](https://www.iwocl.org/wp-content/uploads/IWOCL-2026-Wang-Llamacpp.pdf) |
| LoRA nel browser | ✅ GAP CONFERMATO: WebLLM nessun supporto trovato; wllama lo lista esplicitamente nel TODO; unica eccezione MediaPipe (solo Gemma, adapter statici) | [wllama](https://github.com/ngxson/wllama), [MediaPipe/Gemma](https://medium.com/google-developer-experts/fine-tuning-gemma-with-lora-for-on-device-inference-android-ios-web-with-separate-lora-weights-f05d1db30d86) |
| Training LoRA in-browser | ✅ MOONSHOT SOPRAVVIVE: nessun prior art WebGPU-in-browser trovato; adiacenti: QVAC (Vulkan edge, non browser), MambaKit menziona "WSLA fine-tuning" browser-side → da ispezionare prima di scrivere il claim | [QVAC](https://huggingface.co/blog/qvac/fabric-llm-finetune) |
| SSM/ibridi in browser | parzialmente esistente: MambaKit (SDK browser Mamba-1/2/3), runtime RWKV pure-WebGPU → prior art c'è, non è terra vergine | ricerca blocco 4 |
| OPFS banda | datapoint singolo: ~1.1 GB/s in scrittura (100 MB/90 ms); letture "native-speed" senza numeri pubblici → **resta da-misurare in casa** (20 righe) | [renderlog](https://renderlog.in/blog/origin-private-file-system-opfs/) |
| Quest / CacheGen / H2O / SnapKV / YOCO / CLA / S-LoRA / grammars | confermate da conoscenza consolidata pre-cutoff (riferimenti citati corretti); non ricontrollate una a una in questo sweep | — |
| Sparsità contestuale (TEAL/PowerInfer) | NON verificata in questo sweep (fuori budget) → resta `da-verificare` | — |

## 4. Aritmetica nuova resa possibile dallo sweep

Paging esperti in browser, fattibilità: per un 30B-A3B (~1.7 GB attivi/token q4), a
~1 GB/s di OPFS un miss completo costerebbe ~1.7 s/token → **il paging vive o muore
con l'hit-rate della cache**, e colibri dice che l'hit-rate c'è (71.6% di
predicibilità + pinning per heat). Con hit-rate 95% e miss ammortizzati dal prefetch,
il costo scende a ~85 ms/token di tassa I/O — ancora tanto: serve la combinazione
pinning (esperti caldi SEMPRE residenti) + prefetch + eventualmente spec-dec per
nascondere la latenza (arXiv 2508.21706). Su M4 48 GB il problema non si pone
(residenza piena); si pone su 16 GB e sotto. La conclusione di progetto non cambia:
**il sistema di memoria è il cuore del motore**, ma il primo target realistico del
paging è "modello 2× la memoria", non "744B su 8 GB".

## 5. Cosa resta aperto dopo lo sweep

1. Stato MLA in llama.cpp/ggml (codice da leggere — rientra nel WP studio motori).
2. Architettura dei draft head DeepSpec e portabilità WebGPU (codice da leggere).
3. Banda OPFS in lettura reale (da-misurare, tool da 20 righe nel nostro harness).
4. MambaKit "WSLA fine-tuning": cosa fa davvero (tocca il claim training in-browser).
5. Portabilità numerica cross-device di KV/golden logits (esperimento nostro).
6. Sparsità contestuale TEAL/PowerInfer (search non fatta, resta in coda).
