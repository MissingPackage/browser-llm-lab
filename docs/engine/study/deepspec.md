# Studio deepseek-ai/DeepSpec — riuso dei draft model su WebGPU

Fonti: clone `/tmp/study-deepspec` (commit HEAD shallow, 2026-07-29), config.json + dimensioni reali dei checkpoint su HF.
Paper: DSpark arXiv:2607.05147; DFlash arXiv:2602.06036; Eagle3 arXiv:2503.01840.

## Sintesi

Riuso fattibile, e più semplice del previsto. I tre "algoritmi" condividono un solo scheletro: DFlash è
letteralmente DSpark con markov head e confidence head spenti (stessa classe `Qwen3DSparkModel`, stesso trainer).
L'inference dei draft usa SOLO ops standard (matmul, RMSNorm, RoPE, attention SDPA, SiLU-MLP, softmax/argmax):
**niente tree attention, niente maschere dinamiche a inference** — la verifica è una catena lineare con rejection
sampling esatto (distribution-preserving). Il prezzo vero è l'accoppiamento: tutti e tre i draft consumano gli
hidden states di 5 layer intermedi del target, quindi il nostro motore deve esporre 5 "tap" interni sia al prefill
sia a ogni forward di verifica. Embeddings e lm_head del draft sono copie congelate di quelli del target: nel
browser si condividono i buffer e il costo incrementale reale del draft Qwen3-8B scende a ~630 MB (DSpark q4)
o ~170 MB (Eagle3 q4). Verdetto: portabile; DSpark è il candidato migliore (1 forward per blocco da 7 vs 7
forward sequenziali di Eagle3).

## Le tre architetture a confronto (draft per Qwen3-8B)

| | DSpark | DFlash | Eagle3 |
|---|---|---|---|
| Classe | `Qwen3DSparkModel` | `Qwen3DSparkModel` (markov/conf off) | `Qwen3Eagle3Model` |
| Layer draft | 5 | 5 | 1 (asserted ==1 in eval) |
| Parametri totali | 2 371 M | 2 293 M | 1 547 M |
| — di cui embed+lm_head (copie del target, frozen) | 1 245 M | 1 245 M | 1 245 M |
| — parametri "propri" | 1 126 M | 1 049 M | 302 M |
| Checkpoint HF (bf16, un solo safetensors) | 4 742 MB | 4 587 MB | 3 093 MB |
| Peso proprio f16 (embed condivisi col target) | ~2 253 MB | ~2 097 MB | ~604 MB |
| Peso proprio q4 (~4,5 bit/w, stima) | **~634 MB** | ~590 MB | **~170 MB** |
| Proposta | blocco di 7 token in 1 forward (mask tokens, semi-AR) | idem | 7 forward AR da 1 token |
| Input dal target | hidden dei layer {1,9,17,25,33} concatenati → `fc` 5h→h | idem | idem + embedding del token corrente (concat 2h in q/k/v) |
| Extra head | Markov head rank-256 + confidence head (linear→sigmoid) | nessuna | nessuna |
| Attention a inference | non-causale, K/V = [ctx proiettato ‖ blocco] con KV-cache propria | idem | causale standard con KV-cache propria |

Tutti: `block_size`/`ttt_length` = 7, `target_layer_ids` 5 tap (Qwen3-4B [1,9,17,25,33], 14B [1,10,19,28,37],
Gemma4-12B [5,17,29,41,46], `mask_token_id` 151669 per Qwen / 4 per Gemma). Vietato il tap sull'ultimo layer
(`assert_no_final_target_layer`, `deepspec/eval/base_evaluator.py:100`).

## 1. Architettura dei draft e inventario ops

**DSpark/DFlash** (`deepspec/modeling/dspark/qwen3/modeling.py`, classe `Qwen3DSparkModel`):
`embed_tokens` (151936×4096, copia frozen del target — `initialize_embeddings_and_head(freeze=True)`,
`trainer/base_trainer.py:276`), `fc` Linear(5·h→h, no bias) + `hidden_norm` sul concat dei 5 tap del target,
5× `Qwen3DSparkDecoderLayer` (attn con q/k/v/o + q_norm/k_norm RMS + Qwen3MLP SiLU + 2 RMSNorm), `norm` finale,
`lm_head` (copia frozen del target). L'attention (`Qwen3DSparkAttention.forward`) è ibrida: le query vengono dai
7 mask-token del blocco; K/V = concat(`k_proj(target_hidden_proiettati)`, `k_proj(hidden_blocco)`) — a inference
`attention_mask=None, is_causal=False` (`eval/dspark/draft_ops.py:35-43`), cioè full attention bidirezionale;
il K/V del contesto si accumula in una `DynamicCache` propria, con `crop(start)` che scarta i K/V dei mask-token
dopo ogni proposta. Un solo forward per blocco da 7.
- **Markov head** (solo DSpark, `modeling/dspark/markov_head.py`, `VanillaMarkov`): `w1` Embedding(vocab,256) +
  `w2` Linear(256→vocab); a ogni step logits_corretti = logits_base + w2(w1[token_precedente]); campionamento
  sequenziale sui 7 step (`sample_block_tokens`) — dipendenza sequenziale ma costo ~39 MFLOP/step.
- **Confidence head** (solo DSpark): `AcceptRatePredictor` = 1 Linear((h+256)→1) su hidden+embedding markov;
  sigmoid → tasso d'accettazione previsto per posizione (allenata a predire 1 − ½·TV(draft,target),
  `modeling/dspark/loss.py:_compute_accept_rate_3d`).

**Eagle3** (`deepspec/modeling/eagle3/qwen3/modeling.py`, `Qwen3Eagle3Model`, adattato da SpecForge): 1 decoder
layer i cui q/k/v prendono in input concat(embed(token), hidden_prec) di dim 2h; `fc` 5h→h applicato una volta
per iniettare i tap del target; poi loop autoregressivo: 7 forward da 1 token (`eval/eagle3/evaluator.py:_propose`),
KV-cache draft che a ogni ciclo viene `crop` e ri-estesa con i token committati usando i NUOVI hidden del target
(`_update` → `extend_draft_cache`).

**Inventario ops a inference (tutti):** embedding gather, matmul (GEMM a 7–8 righe), RMSNorm, RoPE, SDPA
(causale per Eagle3, senza maschera per DSpark), SiLU-MLP, softmax/argmax su vocab, multinomial (temp>0),
gather di probabilità, cumprod su ≤7 elementi, residual sampling. Nient'altro.

## 2. Algoritmo di verifica

`generate_decoding_sample` + `verify_draft_tokens` (`eval/base_evaluator.py:186-441`): ciclo **lineare a catena**
(bsz=1 hard-asserted), NESSUN albero. Il target fa un forward su [token_corrente ‖ K draft] (K≤7) con la sua
KV-cache + `crop(start)` per il rollback. Accettazione = **rejection sampling esatto** alla Leviathan:
`accept_prob = min(1, p_target/q_draft)` per token, prefisso via `cumprod`; al primo rifiuto si campiona dal
**residuo** normalizzato `clamp(p−q,0)` (`utils/sampling.py:sample_residual`); se tutto accettato, bonus token
da p_target. La distribuzione del target è preservata esattamente (a temp→0 diventa match greedy, sempre esatto).
Non è configurabile in modalità approssimata. Il **confidence-based scheduling** di DSpark sta in
`eval/dspark/draft_ops.py:_confident_prefix_length`: con `--confidence-threshold t>0` la proposta viene troncata
alla prima posizione con sigmoid(conf) < t (default 0.0 = proponi sempre 7). FlexAttention/BlockMask
(`create_dspark_attention_mask`, `create_eagle3_attention_mask`) servono SOLO al training.

## 3. Accoppiamento col target

Forte, stile EAGLE, per tutti e tre: servono gli **hidden states interni** di 5 layer del target
(`extract_context_feature`: `hidden_states[layer_id+1]`, cioè l'output del decoder layer `layer_id`), non solo
i logits. Il motore deve: (a) al prefill, emettere i 5 tap per TUTTE le posizioni del prompt (transiente:
per 4k ctx a h=4096 sono ~170 MB f16, consumati subito dalla KV draft e liberabili); (b) a ogni verifica,
emettere i 5 tap per le ≤8 posizioni verificate; (c) KV-cache del target con rollback (`crop`) — banale con
buffer preallocato e length pointer; (d) forward multi-token in decode (8 posizioni = mini-prefill). In più
il draft riusa embed_tokens e lm_head del target: se nel nostro motore li teniamo condivisi ma quantizzati,
i logits draft divergono leggermente da come è stato allenato (rischio basso: incide solo sull'accept rate,
mai sulla correttezza, grazie alla verifica esatta).

## 4. Portabilità WebGPU — verdetto e sforzo

**Verdetto: portabile, sforzo moderato.** Facile: tutte le ops draft coincidono con quelle che un motore
target-side deve già avere (matmul/RMSNorm/RoPE/SDPA/MLP); l'attention DSpark è più semplice di quella causale
(nessuna maschera, K/V concatenati); la confidence head è un dot product. Media difficoltà: (1) kernel di
verifica (softmax fp32 su vocab 152k ×8 posizioni, gather, confronto con rand, residual sampling — un paio di
kernel o CPU-side sul solo residuo); (2) markov head: 7 step sequenziali dove il token campionato alimenta lo
step successivo → tenere l'argmax su GPU e passare l'indice via buffer per evitare 7 readback; (3) plumbing dei
5 tap di hidden states nel motore (va progettato da subito, non retrofittato). Nessun componente "difficile"
tipo tree attention con maschere dinamiche: non esiste in questo repo.
Costo per proposta (Qwen3-8B): DSpark ≈ 1 forward da 7 token su 1,05 GFLOP-weights propri (~2·1,05G·7 ≈ 15 GFLOP,
trascurabile vs target) + logits 7×152k; Eagle3 ≈ 7 forward latency-bound da 1 token (più kernel launch, meno
FLOP). Su WebGPU, dove il decode è bandwidth-bound, verificare 8 token costa ~quanto generarne 1: lo speedup
atteso ≈ acceptance length. **Stima sforzo** sopra un motore target funzionante: DSpark 1–2 settimane
(attention ibrida + markov chain + verify kernel), Eagle3 ~1 settimana; il grosso è esporre i tap e il
rollback KV nel design del motore. Preferenza: **DSpark** (o DFlash come primo step senza markov/conf head).

## 5. Eval harness — cosa copiare

`eval.py` + `BaseEvaluator`: 9 task jsonl in `eval_datasets/` (gsm8k 500, math500 500, aime25 30, humaneval 164,
mbpp 256, livecodebench 500, mt-bench 80, alpaca 500, arena-hard-v2 500), campo `turns`, primo turno solo,
seed per-sample (`seed + idx`), chat template con `enable_thinking=False`. Metriche (`build_metrics_row`):
`acceptance_length` (media token committati per verifica, incluso bonus), `draft_tokens_per_proposal`,
`verify_rate` = accettati/(proposti+verifiche), `accept_rate@pos` per posizione 0..6. **Non misura wall-clock**,
solo accettazione. In più per DSpark: calibrazione della confidence head (`eval/dspark/confidence_head.py`) con
ECE, Brier, AUROC per posizione + reliability bins (20 coarse/1000 fine) → tensorboard e artifact JSON.
Per la nostra sezione evals: riusabili subito i 9 jsonl + lo schema di metriche per-posizione; l'idea giusta da
copiare per misurare la "perdita d'intelligenza" di quant/eviction è usare **l'accept rate del target pieno come
giudice a costo zero** (ogni calo di accept/verify_rate del motore quantizzato vs riferimento bf16 è un segnale
di divergenza dalla distribuzione), più il pattern seed-per-sample per la riproducibilità.

## 6. Licenze

Codice: **MIT** (LICENSE, "Copyright (c) 2026 The DeepSpec Authors"). NOTICE: parti Eagle3 adattate da SpecForge
(Apache-2.0), design DFlash da z-lab/dflash (MIT). Checkpoint HF: **nessuna licenza dichiarata** — i 12 repo
hanno solo config.json + model.safetensors, senza README/model card né tag di licenza (verificato su
dspark/dflash/eagle3_qwen3_8b: tag = safetensors, qwen3, region:us). Non trovato = non c'è. Attenzione inoltre:
ogni draft CONTIENE una copia degli embeddings/lm_head del target, quindi i draft Gemma incorporano pesi Gemma
(Gemma Terms of Use si applicano a prescindere); i draft Qwen incorporano pesi Apache-2.0 (ok).

## Dubbi aperti

1. **Numeri di accettazione**: il repo non contiene i risultati di Table 1 (solo nel paper 2607.05147, non letto
   qui). Da estrarre prima di stimare lo speedup atteso per device.
2. **Threshold di confidenza ottimale**: default 0.0 (mai schedulato); il valore usato nel paper per lo
   scheduling non è nel repo.
3. **Sensibilità alla quantizzazione**: i draft sono allenati in bf16 contro target bf16; quanto cala l'accept
   rate con target q4 + draft q4 (e tap di hidden states quantizzati?) è da misurare, non documentato.
4. **Gemma4-12B-it**: variante `Gemma4DSparkModel` con vocab 262k (embed/lm_head più pesanti) e attention
   local/global; non ho verificato le dimensioni dei checkpoint Gemma su HF.
5. I checkpoint sono allenati in **non-thinking mode** (README): con target in thinking mode il README stesso
   raccomanda un re-finetune del draft.
