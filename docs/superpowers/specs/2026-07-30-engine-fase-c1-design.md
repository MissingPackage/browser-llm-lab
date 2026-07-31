# Spec — engine-fase-c1: misure sull'oracolo per il paging esperti (2026-07-30)

Goal contract: `.harness/goals/engine-fase-c1/GOAL.md`. Ancorata ai fatti di fase 1
(journal it.1): oracolo llama.cpp `5f55650a` CPU-only, GGUF unsloth Q4_0 arch
**deepseek2** (46 layer MoE routed: 47 blocchi, dense lead 1, niente NextN),
64 expert routed top-4 + 1 shared, gating **sigmoid** + `exp_probs_b`
(bias di selezione stile DeepSeek-V3), `expert_weights_norm=true`, scale 1.8;
throughput pp512 56.6 / tg64 13.4 t/s a 16 thread. Stato: **APPROVATA**
(ruling PI 2026-07-31, "ok (a)-(f)", docket item 3).

## Strumentazione

Tool C++ autonomo `tools/oracle-moe/` (CMake standalone che linka
`~/Projects/llama.cpp-oracle/build` — **zero modifiche upstream**, constraint
"solo oracolo" intatto), pattern `examples/eval-callback`:

- `cb_eval` (ask/observe, `include/llama.h:375`): in ask ritorna true SOLO per
  `ffn_moe_topk` e `ffn_norm` (filtro sul nome, `common/debug.cpp:143-185`);
  in observe legge i dati (backend CPU ⇒ `is_host`, accesso diretto).
- **Routing vero**: `ffn_moe_topk` = `[4, n_tokens]` int32 per layer
  (`src/llama-graph.cpp:1929`) → append alla traccia.
- **LOOKA online** (niente dump da 6 GB): alla load il tool estrae i pesi router
  `blk.{L}.ffn_gate_inp.weight` (64×2048 f32, ~24 MB totali) e `exp_probs_b` per
  layer via lookup nel modello caricato. In observe di `ffn_norm` (layer L,
  `src/models/deepseek2.cpp:376`) calcola la predizione per L+1:
  `top-K( sigmoid(W_router[L+1] · h_L) + exp_probs_b[L+1] )` — replica ESATTA
  dell'ordine di selezione di `build_moe_ffn` (sigmoid → +bias → argsort_top_k;
  `n_expert_groups=1` ⇒ niente logica gruppi). Confronto con `ffn_moe_topk` di
  L+1 nello stesso passo ⇒ contatori `la_hit[K]/la_tot` per layer.
- **Definizione del predittore** (decisione (a) del ruling): primario = router
  di L+1 su `ffn_norm(L)` (l'analogo del "post-attention di L" di colibri
  LOOKA=1, kind 1). Variante two-step (kind 2, +2.3% in colibri) NON in v1:
  rimando nel report se il recall primario delude.
- Generazione: loop `llama_decode` greedy nel tool (temp 0), chat template
  applicato via `llama_chat_apply_template` del modello; token-id del corpus
  registrati nella traccia (riproducibilità). `n_threads=16` pinnato, seed
  fisso, 1 sequenza per volta, `n_ubatch` default (512) — la mappa
  posizione←colonna del tensore è l'offset dell'ubatch corrente (il tool
  traccia `n_past` per ubatch).
- Output in `results/engine/moe-oracle/`:
  `trace-<data>.jsonl.gz` (per posizione: token id, fase prefill/decode, 46×top-4)
  + `recall-<data>.json` (contatori aggregati e per layer) — envelope con
  sha256 GGUF, commit llama.cpp, n_threads, corpus hash.

## Corpus

8 prompt chat-templated eterogenei (file `tools/oracle-moe/corpus.json`,
committato; hash nell'envelope): 2 codice (en), 2 prosa (en/it), 2 matematica/
reasoning (en), 1 multilingua (it→en), 1 estrazione/JSON. Prefill target
~1.5k token/prompt (testi lunghi inclusi nel file, non generati), decode 512
token greedy ciascuno ⇒ **~12k prefill + 4k decode ≈ 16k posizioni routed**
(gate contrattuale ≥16k) ≈ 9-12 min/run alla velocità misurata. Prefill e
decode SEPARATI in tutte le metriche (la località del routing differisce tra
batch e autoregressivo; il paging browser vive sul decode — il prefill entra
nelle statistiche di residenza, non nel recall headline).

## Metriche

- `recall@K`, K∈{4,6,8}: `|pred_topK(L+1) ∩ true_top4(L+1)| / 4`, media su
  posizioni di decode, aggregata e per layer (46 valori). Headline: recall@4
  e recall@8 decode-only (confronto diretto con colibri 71.6% @K hint-8).
- `baseline_prev`: `|true_top4(L, t−1) ∩ true_top4(L, t)| / 4` decode-only
  (colibri: 41.3%) — dalla sola traccia.
- Skew: per layer, quota cumulativa di selezioni coperta dai top-N expert
  (N∈{4,8,16,32,64}); Gini opzionale nel report.
- Working-set: expert unici (layer,expert) in finestre scorrevoli di
  W∈{32,128,512} posizioni di decode.
- Co-attivazione COUPLE-style: SOLO se gratis dalla traccia (post-processing);
  non gate.

## Policy simulate (fase 5)

Simulatore **TypeScript + vitest** (`tools/oracle-moe/sim/`, unit in `tests/` —
convenzione repo, `npm test`), input = trace.jsonl.gz, cache model a slot di
expert interi (unità = 1 expert ≈ 5.3 MB q4; slab per-expert come da study
colibri §"cosa rubiamo"):
1. **LRU** puro;
2. **LFRU tier.h**: `heat<<8|recency`, decay right-shift periodico, isteresi
   25%+4 (copiato da study/colibri.md §2);
3. **LFRU + pin**: pin top-N per layer da usage della PRIMA metà della traccia,
   valutato sulla seconda (split temporale anti-leakage), budget pin = 50%
   degli slot;
4. **LFRU + pin + prefetch**: al passo t (layer L) prefetch dei predetti per
   L+1 col recall MISURATO (replay delle predizioni vere dalla traccia, non
   un recall sintetico), guard anti-eviction (una speculazione non evicta un
   residente con ≥2 accessi più caldo — colibri PILOT_EVICT_GUARD).
Griglia budget: slot ∈ {184, 368, 736, 1472, 2208, 2944} (= 1/16, 1/8, 1/4,
1/2, 3/4, tutto il parco routed; ~1-15.6 GB). Output: hit-rate per (policy,
budget) + curva; punto di lavoro del device dev annotato nel report (VRAM
utile da misurare col probe adapter esistente, meno residente non-expert —
numero da estimates, non gate). Verdetto esplicito nel report su "modello ~2×
la memoria" (ledger §A).

## Sanity-gate

- `la_tot == (n_moe−1) × posizioni_totali × 4` (lookahead definito su ogni layer
  MoE che ha un predecessore MoE — prefill incluso, contato in slot; la
  componente decode vale `45 × posizioni_decode × 4`) e conteggi traccia
  `== posizioni_totali × 46` per il routing vero. I numeri esatti (46/45) sono
  ricavati dai metadati a runtime e asseriti, non hardcoded.
- Autotest del predittore: applicato a `ffn_norm(L)` per predire lo STESSO
  layer L (`W_router[L]`) deve dare recall ≈ 1.0 (≥0.999) — valida
  l'estrazione pesi e la replica dell'ordine di selezione. Gate hard del tool.
- top-4 senza duplicati; expert id ∈ [0,63]; exit ≠ 0 su qualsiasi violazione.
- Simulatore: hit-rate(budget=2944) == 1.0 per ogni policy; LRU ≤ LFRU+pin a
  parità di budget atteso ma NON gate (è un risultato, non un invariante).

## Timebox e fallback

Fase 4 (LOOKA online): **timebox 3 iterazioni**. Allo sforo: clausola
pre-negoziata del contratto — proposta di fallback ROUTE_TRACE-only via
docket (recall→via alternativa documentata), decide il PI. La `baseline_prev`
in quel caso migra al report di fase 5 (nota in PHASES).

## Rischi

1. **Fedeltà del predittore replicato**: l'ordine di selezione del tool deve
   restare identico a `build_moe_ffn` (sigmoid+bias+top-k). Mitigazione:
   autotest recall-su-se-stesso ≥0.999 (sanity-gate), pinning del commit.
2. **Mappa posizioni↔colonne ubatch**: errori di offset darebbero recall
   sistematicamente sbagliato. Mitigazione: autotest + confronto del numero
   di colonne osservate con n_tokens dell'ubatch.
3. **RAM host 31 GB** (modello 17 GB mmap): il regime resta senza thrashing
   (tg64 stddev 0.10 misurata), ma run lunghe con browser aperti possono
   degradare — le run di traccia si fanno senza carichi concorrenti; i tempi
   NON sono metriche (contatori-only), solo il completamento conta.
4. **Osservatore**: `cb_eval` forza la materializzazione dei tensori osservati;
   con backend CPU è una lettura diretta (~zero costo). I tok/s della run
   strumentata NON si pubblicano né si confrontano (direction §8.6).
5. **Recall = proprietà del modello**: se molto sotto il 71.6% di GLM-5.2,
   NON è un fallimento della fase — è il numero che decide il go/no-go PILOT
   (fase 6, PI). Nessun gate numerico sul recall by design.

## Soglie (riepilogo dei gate meccanici)

Run tool exit 0 con envelope completo; ≥16k posizioni (≥4k decode); sanity
tutti verdi (autotest ≥0.999 incluso); report recall con aggregato+46 layer+
baseline; simulatore `npm test` verde + curve per 4 policy × 6 budget;
nessun file sotto `src/engine/**` toccato.
