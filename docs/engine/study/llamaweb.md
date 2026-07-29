# Studio: backend WebGPU di llama.cpp ("LlamaWeb", arXiv 2605.20706)

Fonte codice: `ggml/src/ggml-webgpu/` @ llama.cpp `e9fa0781` (2026-07-28, clone shallow in `/tmp/study-llamacpp`).
Fonte paper: https://arxiv.org/html/2605.20706v1 (Levine et al., UCSC+MSR).

## Sintesi

Il backend è un traduttore op-per-op del grafo ggml verso WebGPU: **un dispatch per nodo**, bind group e
parametri uniform creati **ad ogni dispatch di ogni token**, grafo ri-codificato da zero a ogni forward.
L'unica fusione esistente è RMS_NORM+MUL (peephole scritta a mano). I punti forti sono i kernel: dequant
fusa nel matmul per ~23 formati, FlashAttention/FlashDecoding, MMVQ int8, subgroup-matrix (solo nativo).
La memoria è delegata all'allocatore ggml, con due trucchi eleganti (arena uniform a slot, scratch
"oltre-dst" via `get_alloc_size`). Telemetria: timestamp-query per-kernel ma solo compile-time e solo
utile in nativo; profilare cambia la struttura di esecuzione. Prefill lento per assenza di
subgroup-matrix nei browser + niente fusione + costo CPU per-dispatch. MoE supportato nel codice
(oltre il paper); MLA non citata da nessuna parte. La nostra tesi (fusione aggressiva, piano di
esecuzione cachato, telemetria nativa) risulta **non espressa e strutturalmente ostica** nel loro design.

## 1. Modello di esecuzione (la domanda)

Percorso: `ggml_backend_webgpu_graph_compute` (ggml-webgpu.cpp:3419) → `ggml_webgpu_encode` (:3245,
switch sugli op) → `ggml_backend_webgpu_build_multi` (:576) → `ggml_backend_webgpu_submit_commands` (:548).

- **Un dispatch per op.** `ggml_webgpu_encode` mappa 1 nodo ggml → 1 kernel (eccezioni: MMVQ = 2 dispatch
  perché quantizza prima le attivazioni in q8; FlashDecoding = split+reduce; MoE prefill = gather+matmul).
  View/permute/reshape/transpose sono no-op (:3266-3271).
- **Fusione: una sola.** `ggml_webgpu_can_fuse_rms_norm_mul` (:3171) usa `ggml_can_fuse` su una finestra
  di 2 nodi e produce `rms_norm_mul.wgsl`. Nient'altro: niente dequant+bias+act, niente rope+attn,
  niente catene. Il paper stesso elenca "dynamic kernel fusion" come future work.
- **Bind group: creati per-dispatch, mai cachati.** In `build_multi` (:583-599) ogni dispatch fa
  `device.CreateBindGroup(...)` fresco, a ogni token. Nessuna cache keyed sul nodo.
- **Parametri: un `queue.WriteBuffer` per dispatch** (:602-605) verso uno slot dell'arena uniform.
  Quindi per ogni op di ogni token: 1 CreateBindGroup + 1 WriteBuffer + SetPipeline/SetBindGroup/Dispatch.
  Nel browser ognuna di queste è una chiamata attraverso emdawnwebgpu (wire JS↔wasm).
- **Batching:** un `CommandEncoder` con **un unico ComputePass** per batch (`batch_compute_passes=true`,
  :3441); submit ogni 64 kernel (`WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE 64`, :89). Submit per token ≈
  ⌈n_dispatch/64⌉ (ordine ~5-15 per un grafo decode tipico), più il sync finale sul readback dei logits
  (`buffer_get_tensor` → staging + MapAsync con mutex globale, :3695). `max_inflight_batches` è
  UINT32_MAX di default (:503) — non aspettano mai a metà grafo.
- **Nessun riuso tra token:** il grafo viene ri-attraversato e ri-codificato integralmente a ogni
  forward. Nessun piano statico, nessun command buffer persistente, nessun indirect dispatch.
- **Pipeline: cachate** (unica cosa cachata) in `unordered_map` per variante di specializzazione
  (shader-lib.hpp:1176-1226), compilate lazy al primo uso via pre-processore.

**Verdetto per la nostra tesi:** confermata dal codice. A taglie piccole (decode) il costo per-op è
CPU-bound e ricorrente: creazione bind group, WriteBuffer, ri-encoding — tutto ripetuto per centinaia di
op a token, per costruzione. L'interfaccia ggml (il backend vede i nodi uno alla volta dentro
`graph_compute`) rende ogni fusione un caso speciale scritto a mano e rende impossibile un piano di
esecuzione compilato una volta e riusato.

## 2. Arena parametri / allocazione

- `webgpu_param_arena` (:117-156): **un solo buffer uniform** di `slot_count = 64+10` slot da 128 byte
  (`WEBGPU_PARAMS_BUF_SIZE_BYTES`), stride allineato a `minUniformBufferOffsetAlignment` (init a :4139).
  Totale ~19 KB. Risolve l'assenza di push constants senza allocare buffer piccoli a runtime.
- Politica: cursore lineare (`alloc_slot`), **reset subito dopo il submit del batch** (:3471). La
  sicurezza non viene da fence ma dall'ordinamento di coda: i `WriteBuffer` del batch successivo sono
  ordinati dopo il submit del precedente. Zero sincronizzazione esplicita. Esaurimento = abort (:140).
- Scratch intermedi: trucco **`get_alloc_size`** (:3813) — il tensore dst viene sovra-allocato
  dall'allocatore ggml e lo scratch vive "oltre la fine" di dst, con offset allineati: q8 delle
  attivazioni per MMVQ (MUL_MAT), liste gather per MoE (MUL_MAT_ID), parziali FlashDecoding + blocchi
  maschera (FLASH_ATTN_EXT), doppio buffer ARGSORT/TOP_K. Niente allocatore scratch separato.
- Tensori: un `wgpu::Buffer` per ggml_buffer; il disallineamento rispetto ai 256B di
  `minStorageBufferOffsetAlignment` è gestito bindando all'offset allineato e passando il resto come
  offset in elementi nei params (`ggml_webgpu_tensor_misalignment`).

## 3. Kernel (43 file .wgsl + 6 .tmpl, ~11.500 righe)

- **Preprocessore proprio** (`pre_wgsl.hpp`, header-only): #define/#ifdef/#include per WGSL, che non ne
  ha. I kernel sono template specializzati a runtime con defines (es. mul_mat_reg_tile.wgsl:
  `TILE_M/TILE_N/WORKGROUP_SIZE_M/...`).
- **Dequant fusa nel matmul**: `mul_mat_decls.tmpl` + `quant_inner_loops.tmpl` generano loader per
  ~23 formati (legacy, K-quants, I-quants, q1_0, MXFP4, NVFP4 — supports_op :4316-4356). Mai un pass
  di dequant separato.
- **Tre percorsi matmul**: `mul_mat_reg_tile` (shmem f16 + tile registri, acc f32; tile fissi 4x4 per
  thread, wg 8x8 — shader-lib.hpp:33-38), `mul_mat_subgroup_matrix` (tensor core, **solo build nativa**:
  la detection è sotto `#ifndef __EMSCRIPTEN__`, :4028), `mul_mat_vec` per N≤4 (:1608) con riduzione
  subgroup o shmem. **MMVQ**: attivazioni quantizzate on-the-fly in q8 + `dot4I8Packed`, abilitato per
  vendor amd/intel/nvidia (shader-lib.hpp:1149).
- **FlashAttention**: tile path per prefill (subgroups richiesti), subgroup-matrix path, FlashDecoding
  vettoriale (split+reduce, parziali in scratch) con skip dei blocchi interamente mascherati
  (`flash_attn_vec_blk`). KV-cache quantizzata q4_0/q8_0 dequantizzata al load (supports_op :4399-4457).
- f16 ovunque (shmem, accumuli FA), **ShaderF16 richiesto hard** (:4054). Qualità: alta, leggibile,
  ottima letteratura per i nostri kernel — soprattutto mul_mat_decls e flash_attn_vec_*.
- Paper: i parametri "performance-portable" vengono da uno sweep su 4 GPU (+41% medio vs valori
  hand-picked) — ma nel codice sono **costanti compile-time**, non tuning per-device a runtime.

## 4. Perché il prefill è lento (−21/−51% vs WebLLM/Transformers.js)

Spiegazione loro (paper §eval): (1) WebLLM ha la fusione kernel di TVM, loro no; (2) Transformers.js ha
matmul subgroup-ottimizzato; (3) il loro percorso subgroup-matrix **non esiste nei browser** (in nativo
battono tutti: +88% geo-mean su WebLLM). In più i safety check WebGPU costano 14-23% medio (fino a 42%
su RTX 5080 in prefill); i toggle Dawn `disable_robustness`, `disable_workgroup_init` (:4096) sono
solo nativi.
Diagnosi mia dal codice, in aggiunta: nel browser resta solo `mul_mat_reg_tile` con tile fissi 32x32
per workgroup da 64 thread (piccolo per GPU discrete); zero fusione ⇒ ogni epilogo (bias, act, norm)
ricarica i tensori dalla VRAM; e il costo CPU per-dispatch della sezione 1 pesa anche in prefill su
grafi lunghi. La lentezza non è "strutturale del web": è il costo di non possedere il grafo.

## 5. Telemetria

- **Sì, timestamp-query per-kernel** — ma solo con build `GGML_WEBGPU_GPU_PROFILE`: QuerySet da 4096
  timestamp (:83), 2 per dispatch, resolve+copy+map in `collect_profile_results` (:3364), tempi aggregati
  per nome pipeline. Richiede feature TimestampQuery (:4068) e disabilita `timestamp_quantization`
  (toggle Dawn, nativo).
- **Effetto osservatore**: con profiling attivo `batch_compute_passes=false` (:4151) — ogni dispatch va
  in un ComputePass separato per avere i timestamp. Ciò che misuri non è ciò che esegui in produzione.
- Profiling CPU separato a macro chrono (`GGML_WEBGPU_CPU_PROFILE`, :58-80), con TODO aperto di rework
  (:208, ref PR 22050). Nulla è esposto a runtime/API: telemetria da sviluppatore, non da prodotto.

## 6. Limiti e feature negoziate

`create_webgpu_device` (:4004): `adapter.GetLimits()` e poi **`requiredLimits` = limiti interi
dell'adapter** (:4072) — greedy, chiedono tutto il massimo. Features: `ShaderF16` sempre richiesta
(niente fallback: device senza f16 = fail), `Subgroups` se presente, `ChromiumExperimentalSubgroupMatrix`
solo nativo (config accettata solo f16×f16, :4036), `ImplicitDeviceSynchronization` (Dawn, nativo),
`TimestampQuery` solo build profile. `Packed4x8IntegerDotProduct` rilevata come WGSL language feature
(:4024). Toggle adapter nativi: `vulkan_enable_f16_on_nvidia`, `use_vulkan_memory_model`.

## 7. Load OPFS / streaming

**Non è nel backend: non trovato.** Zero occorrenze di "opfs" nell'intero repo llama.cpp. Il backend
espone solo `buffer_set_tensor` = `queue.WriteBuffer` sincrono (:3663) — compatibile con streaming a
chunk ma senza logica di streaming. Le tre strategie del paper (cache OPFS, streaming con quattro buffer
da 1 MB via l'interfaccia di loading asincrono di llama.cpp, mai materializzare nel heap WASM) vivono
nel wrapper **wllama** (repo ngxson/wllama), fuori da questo albero. Motivazione: heap WASM grow-only e
limiti Safari. `docs/build.md:730-745` documenta la build Emscripten via emdawnwebgpu.

## 8. MoE e MLA (stato reale nel codice)

- **MoE: supportato, oltre il paper** (che lo dava come future work). `GGML_OP_MUL_MAT_ID` completo:
  decode fast-path `mul_mat_id_vec` (:1718, se `dst->ne[2]==1`), prefill a due dispatch
  `mul_mat_id_gather` + `mul_mat_id` (:1776, liste per-expert in scratch oltre-dst); `GGML_OP_ADD_ID`
  presente; stessa copertura quant del matmul denso incl. MXFP4/NVFP4 (:4358-4397). gpt-oss-20b era già
  nel paper via emulazione mxfp4.
- **MLA: non gestita esplicitamente.** Zero occorrenze di "mla"/"deepseek" nel backend. FLASH_ATTN_EXT
  gestisce dimensionalmente head-dim K≠V (usa `src0->ne[0]` e `src2->ne[0]` separati in
  `max_kv_tile`, :4454), ma con head 576/512 il gate su `maxComputeWorkgroupStorageSize` molto
  probabilmente dà `max_kv_tile==0` ⇒ supports_op false ⇒ llama.cpp ripiega sull'attention non-FA
  (mul_mat+soft_max, comunque su GPU). Non verificato a runtime: inferenza dal codice.

## 9. Stato del progetto

Molto attivo: commit continui fino a luglio 2026 (HEAD 2026-07-28). Contributor principali: yomaytk,
reeselevine (autore paper), nikhilJain17, Constannnnnt. Temi recenti: refactor/tuning FlashAttention,
matmul quantizzati, prefill via percorsi mat-vec per batch piccoli, NVFP4/i-quants, subgroup matrix,
fix di aliasing dei binding. TODO nel codice (9): rework profiling CPU (:208), tuning per-piattaforma di
batch size/inflight (:502), `cpy_tensor` non implementato (:3765), memoria riportata fake
(maxBufferSize come free e total, :3927), richiesta di una "fast mode" WebGPU senza check (:4094).

## Cosa imparare / cosa evitare / cosa possiamo fare noi

**Imparare (da copiare quasi alla lettera):**
- Arena uniform a slot con riuso garantito dall'ordine di coda (niente fence) — :117-156 + :3471.
- Scratch "oltre-dst" via `get_alloc_size` — elimina un allocatore scratch (:3813).
- Trucco misalignment: bind a offset 256-allineato + offset residuo in elementi nei params.
- `pre_wgsl` + shader-lib che restituisce pipeline **e metadata delle decisioni** (workgroup, tile)
  usati per calcolare la dispatch shape — buon pattern di specializzazione.
- Dequant-in-matmul templata (`mul_mat_decls.tmpl`), MMVQ con `dot4I8Packed`, FlashDecoding
  split/reduce con skip dei blocchi mascherati.
- I numeri sul costo dei safety check (14-42% prefill) e la lista toggle Dawn: argomento quantitativo
  per progettare noi il bounds-checking (layout che li rende eliminabili dal compilatore).

**Evitare:**
- CreateBindGroup + WriteBuffer per-op per-token senza cache; ri-encoding integrale del grafo a ogni
  forward; nessun command buffer persistente.
- Fusione peephole caso-per-caso: una fusione = una funzione C++ + uno shader dedicato.
- Telemetria compile-time che cambia la struttura di esecuzione quando la accendi.
- `requiredLimits` = limiti interi dell'adapter (fragile nel browser) e ShaderF16 hard senza fallback.
- Costanti di tiling compile-time uguali per tutti i device (il loro stesso paper quantifica +41% dal
  tuning per-device).
- Cliff di `supports_op`: tensore > maxStorageBufferBindingSize ⇒ fallback CPU (:4261) — nel browser
  significa WASM, catastrofico; noi dobbiamo splittare i pesi, non ripiegare.

**Cosa la nostra tesi può fare che loro strutturalmente non possono (evidenza dal codice):**
1. **Piano di esecuzione compilato una volta**: loro sono vincolati dall'interfaccia ggml — il backend
   riceve `cgraph` e lo attraversa nodo-per-nodo dentro `graph_compute` (:3445), con `ggml_can_fuse` a
   finestra minima; non possono cachare encoding/bind group tra token né fondere catene senza riscrivere
   il contratto ggml. Un motore from-scratch possiede l'IR: fusione di catene
   (dequant+matmul+bias+act+norm), bind group precostruiti (indirizzi tensori statici), pochi submit per
   token, uniform ring con dynamic offset al posto di N WriteBuffer.
2. **Telemetria nativa always-on**: la loro è un ifdef che riplasma i pass; noi possiamo progettare ring
   di query set per batch, resolve asincrona, budget per-op senza alterare il batching.
3. **Sistema di memoria**: loro delegano a ggml (un buffer per ggml_buffer, cap per-binding, memoria
   riportata fake); noi possiamo pianificare placement, splitting dei pesi oltre il cap di binding,
   streaming/evizione integrati con OPFS (che da loro vive fuori, in wllama).

## Dubbi aperti

- Numero reale di submit/token e costo del wire emdawnwebgpu in browser: da misurare, non l'ho eseguito.
- Quale sia esattamente "l'interfaccia di loading asincrono di llama.cpp" usata da wllama (upstream o
  patch loro): serve lettura di wllama per chiudere il punto 7.
- MLA: la mia conclusione (FA rifiutata a head 576 ⇒ percorso non-FA) è inferita, non testata.
- Il paper è v1 e precede parte del codice attuale (MoE, NVFP4): i numeri di prefill/decode potrebbero
  essere già cambiati.
- Disponibilità browser di subgroup-matrix (oggi dietro flag Chromium): è la variabile che più cambia il
  loro gap di prefill — e il vantaggio residuo dei nostri kernel fusi.
