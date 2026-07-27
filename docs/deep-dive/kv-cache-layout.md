# KV cache — layout, paging e il costo del contesto

Quarto doc di deep-dive sul percorso WebGPU di WebLLM/MLC (spec:
`docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md`). Fonti: bundle
non-minificato `node_modules/@mlc-ai/web-llm/lib/index.js` di **@mlc-ai/web-llm 0.2.84**
per l'allocazione, e i **kernel WGSL reali** del dump
(`.harness/goals/fase-2-deep-dive/wgsl-dump/`, run live 4090,
Qwen2.5-0.5B-Instruct-q4f32_1) per il layout — che qui si legge direttamente
dall'aritmetica di indicizzazione, non da documentazione.

## Cosa fa

### L'allocazione: paged in teoria, dimensionata sul massimo in pratica

Al load, `LLMChatPipeline` crea la cache con `create_tir_paged_kv_cache` (bundle, righe
~9979-9995) passando:

- `max_num_sequence` = **1, hardcoded** (`defaultMaxNumSequence`) — una sola conversazione;
- `max_total_sequence_length` = `slidingWindowSize` se configurata, altrimenti **l'intera
  `contextWindowSize`** del modello;
- `prefill_chunk_size` = dai metadata del modello compilato (riga ~9930);
- `page_size` = **16, hardcoded** (il commento nel bundle dice "hard coded for now");
- un flag per la sliding window.

La cache è quindi *paged* nel layout (sotto), ma il budget è deciso al load sul
**contesto massimo configurato**, non sull'uso reale. Quanto eager sia l'allocazione
della slab TVM sotto quella chiamata non è decidibile dal solo bundle JS:
questione aperta: la firma passa il budget massimo, ma se il commit fisico della slab sia eager o lazy si decide nei sorgenti TVM/relax upstream — instradata a `engine-design-notes.md`.

### Il layout delle pagine, letto dal kernel

`009-tir_kv_cache_transpose_append_kernel.wgsl` scrive K e V nelle pagine a ogni token.
L'offset di scrittura è (verbatim dal kernel, riordinato):

```
pages[ pagina×4096 + head×1024 + (pos & 15)×64 + dim ]          // K
pages[ pagina×4096 + head×1024 + (pos & 15)×64 + dim + 2048 ]   // V
```

dove `pagina = pos >> 4` (page_size 16) e ogni thread scrive un elemento f32. Da qui il
layout di una pagina, senza bisogno di alcun documento: **`[pagina][K|V][kv-head][pos][dim]`**
— 2 (K/V) × 2 kv-head × 16 posizioni × 64 dim = 4096 f32 = 16 KiB per pagina, coerente
con Qwen2.5-0.5B (2 KV-head, head_dim 64, GQA 14:2). La `position_map` traduce posizione
logica → slot fisico: è il mattoncino del paging (le pagine di una sequenza possono non
essere contigue), ereditato dal design PagedAttention.

### L'attention che la legge

- **Prefill**: `008-batch_prefill_ragged_kv_kernel.wgsl` — attention "ragged" sul batch
  di token del prompt (chunked da `prefillChunkSize`), 250 righe.
- **Decode**: `028-batch_decode_paged_kv_kernel.wgsl` — attention paged in stile
  FlashAttention/Flash-Decoding: tile di K e V in shared memory (`K_smem`/`V_smem`,
  `vec4<f32>`), softmax online con riduzione di max e denominatore (`md_allreduce`),
  workgroup `(16, 7, 2)` — 16 lane sul head_dim (64/4 in vec4), 7×2 = i **14 Q-head**
  del GQA serviti insieme contro i 2 KV-head condivisi.

A ogni token di decode, questo kernel rilegge la KV di **tutte** le posizioni accumulate:
il costo dell'attention cresce linearmente col contesto, mentre il costo dei matmul dei
pesi resta fisso (`dequant-kernels.md`).

## Perché i numeri sono quelli

- **La KV è f32 pieno, non quantizzata**: nessun percorso KV-quant nell'ABI del bundle.
  Parametri verificati da fonte primaria (HF `Qwen/Qwen2.5-0.5B-Instruct/config.json`:
  `num_hidden_layers: 24`, `num_key_value_heads: 2`, `head_dim: 64`;
  `mlc-ai/Qwen2.5-0.5B-Instruct-q4f32_1-MLC/mlc-chat-config.json`:
  `context_window_size: 32768`, `sliding_window_size: -1`, `prefill_chunk_size: 2048`):
  2 kv-head × 64 dim × 2 (K+V) × 4 byte = 1 KiB per token per layer, × 24 layer =
  **24 KiB/token**. Sul contesto massimo configurato (32768): **768 MiB** di sola KV —
  ~2.9× i pesi q4 (~0.28 GB, `dequant-kernels.md`). Coi 469 token del prompt di bench:
  ~11 MB, irrilevante. È il motivo per cui nei run 1b la KV non si vede: i numeri
  committati esercitano contesti corti — e con `prefill_chunk_size` 2048 il prompt da
  469 token non attiva nemmeno il chunking: il prefill è davvero un singolo mega-dispatch.
- **TTFT 290–344 ms (4090) vs 8.2 s con varianza 104% (S22)**
  (`results/4090-linux-2026-07-26T19-54-55-278Z.json`,
  `results/s22-ultra-2026-07-27T00-34-09-931Z.json`): il prefill è il singolo blocco di
  lavoro GPU più grande del run (512 token in un chunk), ed è proprio dove la varianza
  mobile esplode (5.3→10.9 s a decode stabilissimo). L'ipotesi DVFS/thermal del SoC —
  il governor non ha ancora alzato i clock quando il mega-dispatch parte — resta la più
  plausibile e **non è misurabile dal runtime** (niente timestamp-query richiesta; ma il
  probe S22 espone la feature, quindi il micro-bench di fase 6 può misurarla davvero su
  mobile). L'attribuzione DVFS non è ancora misurata: è l'oggetto del candidato
  esperimento docket #6. Si lega al docket #12
  ereditato: `high-variance` oggi guarda solo il decode, cioè il posto sbagliato per il
  mobile.
- **Su memoria unificata la KV compete coi pesi nel pool fisico** (S22: ~2 GiB di
  `maxBufferSize` dentro 8 GB condivisi con OS e compositor — `buffer-limit-2gb.md`):
  allungare il contesto riservato = meno margine per i pesi, un trade-off che sulla
  4090 con VRAM dedicata non si vede. Per l'M4 Pro (48 GB unificati) vale il discorso
  opposto: la capienza c'è, il collo è altrove (binding cap — stesso doc).
  probe M4 in attesa dello sweep manuale: buco dichiarato

## Bottleneck & vie d'uscita

### 1. Il TTFT ha una varianza che nessuna telemetria del runtime segnala — 104% min→max sull'S22, invisibile accanto a un decode stabilissimo

Sul run S22 Ultra (`Qwen2.5-0.5B-Instruct-q4f32_1-MLC`, 469 prompt token, `results/s22-ultra-2026-07-27T00-34-09-931Z.json`) il TTFT ha tre repliche — 5354.9 / 10936.9 / 8292.8 ms, mean 8194.9, stdev 2792.3 — con un salto min→max del **104,2%** e CV **34,1%**. Nello stesso run il decode è quasi immobile: 6.91-7.09 tok/s, CV **1,3%**. Il gap è già stato registrato come `docket #12` ereditato (`.harness/goals/fase-1b-matrice/docket.md`): "high-variance guarda solo il decode… la varianza del TTFT non è segnalata da nulla" — nessuna metrica del laboratorio la cattura oggi.

La ragione strutturale sta nel dispatch: verificato che `prefill_chunk_size` per questo modello è **2048** (`mlc-chat-config.json` del repo `mlc-ai/Qwen2.5-0.5B-Instruct-q4f32_1-MLC`), letto a runtime da `metadata.prefill_chunk_size` (bundle riga 9930). Il prompt del benchmark è 469 token — sotto soglia, quindi il prefill gira **come un unico mega-dispatch non chunkato** seguito da una sola sync (`compute-shader-dispatch.md`, TTFT 290-344 ms sulla 4090 vs 8.2 s sull'S22). Il decode è invece tanti dispatch piccoli e ripetuti: la stessa volatilità per-step, mediata su decine di token, si spegne per legge dei grandi numeri; sul prefill non c'è nulla su cui mediare. Ipotesi più probabile per la causa di fondo — DVFS/thermal ramp-up del governor GPU Android sul primo colpo di carico pesante — resta un'ipotesi non misurata (candidato esperimento docket #6): nessuna riga di `web-llm` gestisce clock/DVFS (fuori dal controllo WebGPU), e senza `timestamp-query` (mai richiesta dal runtime, `compute-shader-dispatch.md` bottleneck #2) non si può separare ramp-clock, contesa con compositor/OS Android e throttling termico residuo — tre cause plausibili oggi indistinguibili.

### 2. La KV cache dimensionata sul contesto massimo pesa quasi 3× i pesi del modello, su una memoria unificata già stretta

Verificato dal bundle (righe 9979-9990): `create_tir_paged_kv_cache` riceve `page_size=16` (hardcoded, commento "hard coded for now"), `max_num_sequence=1`, `max_total_sequence_length = slidingWindowSize != -1 ? slidingWindowSize : contextWindowSize` (righe 9981-9983, `slidingWindowSize`/`contextWindowSize`/`attentionSinkSize` letti da `config.*` alle righe 9936-9938). Per questo modello, dal `mlc-chat-config.json`: `context_window_size: 32768`, `sliding_window_size: -1` → `maxTotalSeqLen = 32768`. Col layout f32 decodificato sopra: **768 MiB** di KV al tetto di contesto contro **~0.28 GB** di pesi+scale — la KV al massimo configurato pesa **~2.9×** i pesi: per un modello così piccolo non è la taglia del modello a dominare il budget memoria, è la capacità di contesto configurata. Il working set realmente scritto dal run osservato (469 token — e nessun `results/*.json` committato supera 469 prompt token) è **≈11 MB**, due ordini di grandezza sotto il tetto. Se l'allocazione TVM/relax sia eager sull'intero `max_total_sequence_length` o cresca lazy per pagina non è decidibile dal solo bundle JS: questione aperta nei sorgenti TVM/relax upstream, instradata a `engine-design-notes.md`.

Il cap per-buffer WebGPU (~2 GiB) **non è il vincolo che morde qui**: 768 MiB ci sta comodamente sotto su entrambi i device. Il vincolo reale è il **pool fisico condiviso**: 8 GB unificati sull'S22 contro la VRAM dedicata della 4090, dove 768 MiB + 0.28 GB è rumore. Verificato anche lato adapter di progetto: `src/adapters/webllm.ts` oggi non tocca in nessun punto `context_window_size`/`sliding_window_size`/`attention_sink_size` — il default arriva intatto dal `mlc-chat-config.json` del repo modello, quindi qualunque scelta device-aware richiede aggiungere quella logica nell'adapter (rework, non solo config).

| Idea | Prior art | Fattibilità / costo | Rischio | Instradamento |
|---|---|---|---|---|
| Warm-up dedicato pre-ramp clock, separato dal warmup shader/cache: burst di compute scartato prima di avviare il timer TTFT | Prassi comune di benchmarking mobile (dummy pass prima della misura) attribuzione specifica a singoli motori non ri-verificata: prassi generica di benchmarking mobile | Basso: harness/driver-level, zero modifiche a `webllm.ts` | Basso; non risolve jitter da contesa OS/compositor né da throttling termico residuo, solo il ramp DVFS | **esperimento** |
| Prefill chunking device-aware: tunare `prefillChunkSize` sotto il default 2048 per tagliare la coda worst-case da preemption dello scheduler Android | llama.cpp: batch processing configurabile del prompt | Medio: parametro già letto da metadata (riga 9930), nessun nuovo codice, solo sweep di misure — **ma nessun run committato ha mai un prompt >469 token**, quindi l'effetto chunking non è mai stato esercitato nel corpus attuale: servono prompt nuovi prima di poterlo misurare | Medio: trade-off throughput/tail-latency esplicito, e dipendenza aggiuntiva (nuovo corpus prompt) assente nell'idea #1 | esperimento |
| Contesto/sliding-window device-aware: `context_window_size` più basso o `sliding_window_size`+`attention_sink_size` (ABI già presente, righe 9936-9938) quando il probe rileva memoria unificata piccola | vLLM PagedAttention: alloca pagine on-demand, non il budget massimo a priori | Medio: l'ABI espone già i campi; la selezione del default va aggiunta in `src/adapters/webllm.ts` (verificato: oggi non la tocca) | Basso tecnico; alto di prodotto — contesto più corto su mobile è una regressione funzionale visibile, va comunicata non nascosta | engine-notes |
| KV cache quantizzata (int8/int4 per K/V, non solo pesi) | llama.cpp `--cache-type-k/-v q8_0`/`q4_0` | Alto: non esposto nell'ABI TVM/relax attuale di web-llm esistenza di un branch upstream non verificata — check instradato a engine-notes; richiede kernel dequant aggiuntivi nel path attention paged, oltre il vincolo "niente rework del motore" della fase | Alto/fuori scope — leva col miglior guadagno teorico ma non aggirabile senza toccare il runtime MLC | engine-notes |
| Allocazione pagine KV lazy invece che upfront su `max_total_sequence_length` | vLLM PagedAttention (paging dinamico reale) | Alto: tocca l'allocatore TVM/relax, non il layer JS | Alto/fuori scope — stesso discorso della quantizzazione | engine-notes |
| Pattern ONNX Runtime Web / transformers.js per `past_key_values` (tensore GPU persistente via IOBinding, riusato fra chiamate) | onnxruntime-web + transformers.js, KV cache come tensore WebGPU persistente comportamento esatto IOBinding-web non ri-verificato — irrilevante per la decisione: idea scartata | N/A per il progetto (motore diverso) | Nessun vantaggio dimostrato: stesso problema di allocazione statica a un `max_length` fisso di web-llm, solo in un runtime diverso — conferma che la tensione è di categoria "browser inference", non specifica di MLC | **scartata** — stessa famiglia di problema osservata altrove, non una soluzione |
| Swap OPFS delle pagine KV "fredde" fuori dalla finestra attiva (stile disk-swap `antirez/ds4`) | `antirez/ds4`: swap su disco delle strutture fredde, hot set in RAM — già valutato per i pesi in `buffer-limit-2gb.md` (scartata lì per latenza IndexedDB/OPFS), qui riadattata alla KV | Alto: serializzazione pagine + invalidazione su cambio quant/modello | Incoerente come proposta a sé: con `sliding_window_size=-1` (attenzione piena, verificato) ogni step di decode attende su **tutte** le posizioni passate — non si può evincere fisicamente una pagina dal calcolo senza cambiare la semantica di attenzione, il che è di fatto reinventare l'idea "sliding window" con un giro I/O più lento sopra. Una variante coerente (OPFS come persistenza di prefix-cache *abbinata* a una sliding window già attiva, per continuità conversazionale multi-sessione) è concettualmente diversa e andrebbe proposta a parte | **scartata** (per come formulata qui; la variante compatibile richiede prima l'idea sliding-window) |
| Offload KV "freddo" a RAM host stile DeepSpeed-Inference/FlexGen (gerarchia GPU-VRAM/CPU-RAM/disco) | DeepSpeed-Inference ZeRO-Inference, FlexGen (offload multi-tier per generazione a batch grande) | Alto: staging buffer WebGPU↔CPU per pagina, orchestrazione dedicata | Su S22 (memoria unificata, bottleneck 2) GPU-RAM e CPU-RAM **sono lo stesso pool fisico**: l'offload non libera nulla, sposta solo byte nello stesso serbatoio. Ha senso solo su device con VRAM dedicata separata — dove però 768 MiB di KV non è mai vicino a saturare nulla: nessun run l'ha reso necessario | **scartata** — zero beneficio proprio sul device (S22) dove la memoria è più stretta; nessuna evidenza che lo giustifichi sulla 4090 |

**Raccomandazione**: propongo come esperimento il **warm-up dedicato pre-ramp clock** (prima riga). Rispetto al chunking device-aware (unico altro candidato "esperimento"), ha due vantaggi verificati: è indipendente dal corpus prompt attuale — il chunking tuning richiederebbe prima prompt >2048 token, che oggi non esistono in nessun run committato, quindi non è misurabile senza lavoro propedeutico — e attacca direttamente il bug già docket-ato (#12, "la varianza del TTFT non è segnalata da nulla") con il costo più basso della tabella: nessuna modifica a `webllm.ts`, solo un burst scartato a livello harness/driver prima di avviare il timer. Non è detto che risolva l'intera varianza (il 104% ha probabilmente più di una causa concorrente — DVFS, contesa OS/compositor, throttling residuo, tutte oggi indistinguibili senza `timestamp-query`), ma è il modo più economico per falsificare o confermare l'ipotesi DVFS con un solo cambiamento alla volta, e va deciso in docket, non eseguito da questa sezione.
