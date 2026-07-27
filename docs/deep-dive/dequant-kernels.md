# Kernel dequant — come i pesi 4-bit diventano matmul

Terzo doc di deep-dive sul percorso WebGPU di WebLLM/MLC (spec:
`docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md`). Fonte primaria: i **34
kernel WGSL reali** catturati da un run live sulla 4090 (Qwen2.5-0.5B-Instruct-q4f32_1,
Chrome branded, `@mlc-ai/web-llm 0.2.84`) con intercettazione di `createShaderModule` —
dump committato in `.harness/goals/fase-2-deep-dive/wgsl-dump/`, tool in
`.harness/goals/fase-2-deep-dive/tools/wgsl-dump.mjs`. Non ricostruzioni: sorgenti
generati dal codegen WebGPU di TVM, come arrivano al driver.

## Cosa fa

### Lo schema q4: 8 pesi per u32, offset −7, una scale ogni 32

Dal kernel `014-fused_dequantize_NT_matmul14_kernel.wgsl` (proiezione finale sul
vocabolario, decode):

- i pesi vivono in un `array<u32>` storage: **8 pesi da 4 bit impacchettati per u32**,
  estratti con shift a passi di 4 (`(w >> 0u) & 15u`, `(w >> 4u) & 15u`, … `>> 28u`);
- la dequant è `f32(nibble) − 7.0` moltiplicato per una **scale f32 per gruppo di 32
  pesi** (l'indice scale è `threadIdx.x >> 2`: 4 u32 consecutivi = 32 pesi → stessa
  scale). Range effettivo del peso: −7…+8 per scale;
- ogni estrazione alimenta direttamente una `fma(attivazione, peso_dequant, acc)`: la
  dequant è **fusa nel matmul**, il peso dequantizzato vive solo in un registro — nessun
  buffer f16/f32 dei pesi viene mai scritto in VRAM.

La variante `q4f16_1` differisce solo nel dtype di calcolo (f16 invece di f32, stesso
packing 4-bit e group size 32 — vedi `baseline/run-C-green-dequant.md` e il catalogo
quantizzazione upstream mlc-llm); nessun run committato la usa (docket #2).

### Due forme di kernel, generate per due regimi

Il codegen TVM emette per ogni proiezione due kernel distinti, visibili nel dump:

| | Prefill (suffisso `_kernel_2`) | Decode (suffisso `_kernel`) |
|---|---|---|
| Esempio | `006-fused_dequantize1_fused_NT_matmul5_add2_kernel_2.wgsl` | `027-fused_dequantize1_fused_NT_matmul10_add4_kernel.wgsl` |
| Workgroup | `8×8` | `64×1` |
| Forma | GEMM tiled: tile di attivazioni e di pesi dequantizzati messi in **shared memory** (`dequantize_reindex_shared`, 256 f32) e riusati dal tile | GEMV: ogni thread legge i suoi u32 packed, dequantizza **nei registri** (catena di 8 fma per u32), riduzione finale in workgroup (`red_buf0`, 64 f32) |
| Perché | batch di token: ogni peso dequantizzato serve a più righe → conviene pagare la shared memory | un token solo: ogni peso serve una volta → nessun riuso, si minimizza il traffico |

È la stessa dicotomia GEMM/GEMV dei motori nativi (llama.cpp, cuBLAS-vs-gemv), qui
generata automaticamente dal compilatore a partire dallo stesso IR.

### La catena completa nel decode

Per ogni token: rms_norm → per ogni layer, i `fused_dequantize*_NT_matmul*` delle
proiezioni (qkv, out, gate/up, down — kernel 027, 029, 030, 032) intervallati da rope
(007), attention paged (028 `batch_decode_paged_kv`), silu (031); in coda la proiezione
sul vocabolario (014) e i kernel di sampling (argsort/cumsum/top-p, 017–025). Tutti
accodati nello stesso command encoder e flushati con un submit unico + una sync per token
(`compute-shader-dispatch.md`).

## Perché i numeri sono quelli

Il decode legge l'intera matrice dei pesi una volta per token (nessun riuso a batch=1 —
sopra). Qwen2.5-0.5B q4f32_1 pesa **~0.28 GB** di pesi+scale (4 bit/peso + una scale f32
ogni 32; coerente col fetch osservato nel run di dump: ~110 MB al 41%). Da lì il conto
roofline, coi numeri misurati:

| Device | Banda dichiarata | Tetto teorico (banda/0.28 GB) | Misurato | % del tetto |
|---|---|---|---|---|
| 4090 laptop | 576 GB/s di targa (256-bit GDDR6 18 Gbps — [VideoCardz](https://videocardz.net/nvidia-geforce-rtx-4090-laptop-gpu), [TechSpot](https://www.techspot.com/review/2624-nvidia-geforce-rtx-4090-laptop-gpu/)); ~435 GB/s misurati dal micro-bench (`micro-bench-matmul.md`) | ~2000 tok/s | 101–116 tok/s (`results/4090-linux-2026-07-26T19-54-55-278Z.json`) | **~5%** |
| S22 Ultra | 51.2 GB/s di targa (Exynos 2200, LPDDR5 4×16 — [nanoreview](https://nanoreview.net/en/soc/samsung-exynos-2200), [Notebookcheck](https://www.notebookcheck.net/Samsung-Exynos-2200-Processor-Benchmarks-and-Specs.610323.0.html)) | ~180 tok/s | 6.99 tok/s (`results/s22-ultra-2026-07-27T00-34-09-931Z.json`) | **~4%** |

Due letture, entrambe importanti per la pagina pubblica:

1. **Il rapporto 4090/S22 (~14-16×) è vicino al rapporto di banda (~11×)**: la scala
   *relativa* tra device è spiegata bene dalla banda memoria, coerente con la natura
   memory-bound (bassa intensità aritmetica) del decode q4.
2. **Ma in assoluto entrambi i device stanno a ~5% del tetto di banda pesi.** Il "decode
   è memory-bound" da manuale va quindi sfumato: per un modello così piccolo il cap
   reale è altrove — candidati: overhead di dispatch/orchestrazione per token (una
   manciata di ms fissi pesa moltissimo quando il lavoro utile è ~0.14 ms/kernel),
   occupancy bassa su matrici piccole (hidden 896), attivazioni f32, latenza non
   nascosta nei GEMV. Risposta di prima battuta dal micro-bench di fase 6: ~75-85%
   orchestrazione (`micro-bench-matmul.md`); il breakdown per-kernel nel runtime è
   instradato a engine-notes.

Corollario: i tok/s dei run 1b **non misurano la banda dei device** su questo modello —
misurano l'overhead della pipeline a taglia piccola. Un modello più grande (3B/8B)
dovrebbe avvicinarsi molto di più al roofline; è un'ipotesi verificabile nello sweep
e nel micro-bench. Verificato: tutti i run in `results/` sono su Qwen2.5-0.5B —
l'ipotesi resta aperta per lo sweep multi-device.

## Bottleneck & vie d'uscita

### 1. Il gap col roofline di banda-pesi (bottleneck primario): entrambi i device restano al 4-6% del tetto teorico

Numeri già inchiodati sopra (tabella "Perché i numeri sono quelli"): 4090 laptop 576 GB/s di targa (fonti nella tabella sopra; ~435 GB/s misurati) → tetto teorico ~2000 tok/s per ~0.28 GB di pesi+scale, misurato **101-116 tok/s** su 4 configurazioni (`results/4090-linux-2026-07-26T19-54-55-278Z.json`, cella `webllm`, mean 101.32/105.31/107.81/115.64) = **~5-6% del tetto**. S22 Ultra 51.2 GB/s LPDDR5 di targa (fonti nella tabella sopra) → tetto ~180 tok/s, misurato **6.99 tok/s** (stdev 0.09, `results/s22-ultra-2026-07-27T00-34-09-931Z.json`) = **~4% del tetto**.

Il rapporto 4090/S22 misurato (~14-16×) è vicino al rapporto di banda dichiarata (~11×) — la scala *relativa* tra device è spiegata bene dalla banda memoria, coerente con la bassa intensità aritmetica del decode q4 (roofline testbook). Ma **in assoluto** nessuno dei due device si avvicina al proprio tetto: se il costo fosse davvero dominato dal solo trasferimento pesi, ci aspetteremmo cifre a ridosso del 100%, non al 4-6%. Il "decode è memory-bound" da manuale va quindi sfumato — memory-bound nel senso dell'intensità aritmetica, ma il cap misurato non è la banda: è qualcos'altro, e questo qualcos'altro è il buco che il micro-bench di fase 6 deve chiudere (vedi anche `compute-shader-dispatch.md`, bottleneck #1-#2, sulla sync-per-token e sull'assenza di `timestamp-query`).

### 2. Il kernel GEMV di decode ha grana di parallelismo minima: un workgroup da 64 thread per riga d'output, zero riuso del peso dequantizzato

Verificato in `027-fused_dequantize1_fused_NT_matmul10_add4_kernel.wgsl`: `@compute @workgroup_size(64, 1, 1)`, un solo elemento di output (`v__1`) calcolato per workgroup, riduzione ad albero in `red_buf0 : array<f32, 64>` (halving 32→16→8→4→2→1, 6 `workgroupBarrier`). La riga di pesi letta è lunga 112 `u32` = 896 nibble, che combacia con hidden 896 di Qwen2.5-0.5B (`docs/deep-dive/buffer-limit-2gb.md`) — un peso dequantizzato vive per la durata di una sola `fma` e non viene mai riletto da un altro thread o da un altro token.

Contrasto diretto col prefill (`006-fused_dequantize1_fused_NT_matmul5_add2_kernel_2.wgsl`, verificato): `@compute @workgroup_size(8, 8, 1)`, due buffer `array<f32, 256>` in shared memory (`rms_norm98_reindex_pad_shared`, `dequantize_reindex_shared`), un tile di pesi dequantizzati scritto una volta e riusato da 16 accumulatori × 8 fma per iterazione del loop su `ax3_0` (112 iterazioni) = 128 fma/iterazione — lì il peso serve più righe d'output nello stesso workgroup, quindi vale la pena pagare la shared memory. Nel decode (batch=1) quel riuso è strutturalmente assente: non è un difetto di scheduling di TVM, è la natura del GEMV — ma la contropartita è che la griglia di dispatch resta piccola (un workgroup per riga d'output, ordine hidden/intermediate-size) e nessun passo aggiuntivo prova a spezzare ulteriormente il lavoro per aumentare l'occupancy su una GPU con molti più SM del numero di workgroup lanciati. Non misurabile direttamente oggi: nessun `timestamp-query`/`GPUQuerySet` richiesto dal runtime (`compute-shader-dispatch.md`, bottleneck #2) — stima da lettura del kernel, nessuna istrumentazione occupancy diretta; misura instradata a engine-notes.

| Idea | Prior art | Fattibilità / costo | Rischio | Instradamento |
|---|---|---|---|---|
| Micro-bench isolato del kernel dequant+GEMV a taglie di output crescenti (hidden/intermediate sintetici), per separare quanto del gap ~95% dal tetto sia occupancy-bound vs overhead di dispatch | Metodologia roofline (Williams et al. 2009) applicata a granularità di singolo kernel invece che a livello applicazione | Media: riusa la route SPA micro-bench matmul già pianificata per fase 6; non richiede toccare `src/adapters/webllm.ts` | Basso — solo misura, nessuna modifica al motore | **esperimento** |
| Passare a `q4f16_1` sui device con `shader-f16` confermato (S22 lo espone, mai usato — `results/s22-ultra-2026-07-27T00-34-09-931Z.json`, `probe.features`) | Catalogo quantizzazione MLC-LLM, stesso group size 32 (docket #2, `baseline/run-C-green-dequant.md`) | Bassissima — solo `modelId` in `scripts/seq-bench.mjs:20`/`src/conformance/page.ts:26` | Basso, ma **ridimensionato dai fatti nuovi**: il gap misurato è ~20× sotto il tetto di banda, un delta f32→f16 su registri/ALU intermedi spiega al più un fattore 2× — necessario da verificare, non sufficiente da solo a spiegare il bottleneck #1 | esperimento (secondario) |
| Split-K / multi-workgroup per riga d'output: spezzare la riduzione su K tra più workgroup invece di uno solo da 64 thread, per aumentare il numero di blocchi lanciati sulle GEMV di decode | Split-K GEMM, CUTLASS: "se M o N sono piccoli... pochi threadblock lanciati, non si sfruttano tutti gli SM" — esattamente il caso hidden=896/batch=1 ([NVIDIA CUTLASS, Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)); stesso principio dietro Flash-Decoding per l'attention in decode ([PyTorch blog](https://pytorch.org/blog/flash-decoding/)) | Alta — richiede rework dello schedule TVM/MLC (compile-time), non solo config JS | Medio-alto: un secondo passo di riduzione aggiunge overhead fisso (altro dispatch + atomics/barrier) che su un modello così piccolo potrebbe azzerare il guadagno | engine-notes |
| Kernel dequant+GEMM fuso con overlap latenza/calcolo via async copy e layout dedicato (stile Marlin) | Marlin, GPTQModel/vLLM ([arXiv:2408.11743](https://arxiv.org/pdf/2408.11743)) | Alta — WGSL custom, TVM/MLC oggi non genera questo pattern per WebGPU | Alto — subgroup ops WebGPU ancora parziali su GPU mobile | engine-notes |
| Block layout coalesced-access stile `mul_mat_q` di llama.cpp | llama.cpp, kernel quantizzati con layout a blocchi ottimizzato per warp/coalescing sorgente llama.cpp non ri-verificato in questa fase — check instradato a engine-notes | Media-alta — tocca il layout pesi generato da TVM | Medio | engine-notes |
| Persistent kernel / megakernel: un solo dispatch per l'intero forward pass invece di ~34 kernel separati per token, con workgroup che restano vivi e si passano il lavoro invece di rilanciare la griglia ogni volta | Pattern persistent-kernel/megakernel da ray-tracing e serving LLM nativo: kernel-launch overhead misurato **~14.6% del tempo end-to-end su Qwen2.5-1.5B sotto TensorRT-LLM** (stessa famiglia di modello del nostro benchmark) ([TaxBreak, arXiv:2603.12465](https://arxiv.org/pdf/2603.12465)); MegaKernel/MPK eliminano il round-trip in HBM tra operatori fondendo tutto in un kernel persistente ([Compiling LLMs into a MegaKernel, Zhihao Jia](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)) — nessun motore lo fa oggi su WebGPU: è il transfer fuori dagli schemi di questo sweep | Molto bassa nel breve periodo — WebGPU non espone primitive di sincronizzazione persistenti/grid-wide tra dispatch in modo portabile; richiederebbe riscrivere il runtime di dispatch di MLC, non solo lo schedule di un kernel | Alto: pattern immaturo su WebGPU, nessun prior art diretto nel dominio browser | engine-notes (idea fuori dagli schemi, valutata e non scartata: il precedente TensorRT-LLM è solido, ma il gap WebGPU→primitive-necessarie è troppo ampio per un esperimento a basso costo) |
| Delegare il matmul quantizzato a WebNN (driver nativo, operatori QDQ dedicati) | WebNN espone `quantizeLinear`/`dequantizeLinear` ([webnn#623](https://github.com/webmachinelearning/webnn/issues/623)) | Molto alta — adapter WebNN completo, nuovo | Alto — supporto mobile/Android incompleto | **scartata** (fuori scope: WebNN è nei "Deferred" dello spec madre) |
| Dequantizzare una volta e cachare i pesi f16/f32 in un buffer persistente tra token | Pattern da sistemi di offload (FlexGen) per batching, non per batch=1 nessun prior art trovato per il caso singolo token nello sweep fatto | Bassa da implementare | Alto — a batch=1 quadruplica i byte da leggere/scrivere in VRAM, aggravando esattamente il bottleneck #1 | **scartata** (aggrava, non risolve) |

**Raccomandazione**: l'esperimento da proporre come prima voce docket è il **micro-bench isolato a taglie crescenti** (prima riga). Il run-C originale puntava a "cambiare quant" come unica leva; i fatti nuovi la ridimensionano: il gap osservato (~95% sotto il tetto di banda, su entrambi i device) è troppo grande per essere spiegato da un delta f32→f16 sulle attivazioni, e resta ignoto se il vero freno sia l'occupancy minima del GEMV di decode (bottleneck #2, un workgroup da 64 thread per riga d'output, nessun riuso) o l'overhead di dispatch per i ~34 kernel/token già discusso in `compute-shader-dispatch.md`. Il micro-bench è l'unico esperimento che produce il dato mancante per discriminare tra le due ipotesi — con costo marginale quasi nullo (riusa la route SPA già pianificata per fase 6) e senza toccare il motore di produzione. Lo swap `q4f16_1` resta un esperimento valido (docket #2) ma va trattato come secondario: verifica una leva concreta, non la causa primaria del gap.
