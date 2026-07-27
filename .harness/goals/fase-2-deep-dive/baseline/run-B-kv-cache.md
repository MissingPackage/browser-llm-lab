# Baseline run B — bottleneck: varianza TTFT S22 + KV cache vs pesi (per fase 5)

<!-- Output verbatim del subagent, 2026-07-27, senza skill. Da ri-verificare prima dell'uso. -->

## Bottleneck & vie d'uscita

### Bottleneck 1 — TTFT del prefill volatile, decode no

Sul Samsung S22 Ultra (Xclipse 920, RDNA2 mobile, 8 GB unificati) il run `webllm` su
`Qwen2.5-0.5B-Instruct-q4f32_1-MLC` (469 prompt token, `results/s22-ultra-2026-07-27T00-34-09-931Z.json`)
mostra tre replicate di TTFT — 5354.9 ms, 10936.9 ms, 8292.8 ms — con un salto min→max del
**104,2%** e una CV del 34,1% (stdev/mean). Nello stesso run il decode è quasi immobile:
6.91–7.09 tok/s, CV **1,3%**. La stessa metrica di dispersione, applicata alla 4090 (es.
`results/4090-linux-2026-07-26T19-54-55-278Z.json`), mostra CV per-cella anche più alte in
relativo (48–87%) ma su una base di ~300–400 ms: irrilevante in assoluto. Sull'S22 la stessa
volatilità si traduce in **secondi**, cioè nella parte di latenza che l'utente percepisce di persona
prima del primo token.

Il prefill è l'unica fase esposta a questa variabilità perché è un singolo dispatch grande
(tutto il prompt in un colpo, salvo chunking — vedi `prefillChunkSize`,
`node_modules/@mlc-ai/web-llm/lib/index.js:9930`) mentre il decode è tanti dispatch piccoli e
ripetuti, quindi mediamente stabile anche se il singolo passo ha jitter. Ipotesi più probabile
(non verificata a fondo, marcata `[VERIFY]`): DVFS/thermal ramp-up della GPU mobile — un SoC
Android non tiene le frequenze GPU fisse come una scheda desktop, e il primo colpo di carico
pesante (il prefill) è anche quello più esposto al tempo di ramp del governor, alla
contesa con compositor/altri processi Android, e a eventuale throttling termico residuo da
run precedenti. Nessuna riga di codice in `web-llm` gestisce DVFS lato browser: è fuori dal
controllo di WebGPU, e il warm-up applicato (`protocol.warmupPolicy: "always"`) scalda la
shader pipeline ma non necessariamente i clock della GPU.

### Bottleneck 2 — KV cache e pesi in competizione per una memoria GPU piccola e unificata

`LLMChatPipeline` alloca la paged KV cache con `create_tir_paged_kv_cache`
(`node_modules/@mlc-ai/web-llm/lib/index.js:9985-9990`) passando
`max_total_sequence_length` (= `context_window_size` o `sliding_window_size`,
`:9981-9983`), `page_size` fisso a 16 (`:9979`, commentato "hard coded for now") e
`max_num_sequence` fisso a 1. I parametri passati mostrano che la cache viene dimensionata
sul **budget massimo di contesto configurato**, non sull'uso reale corrente — quanto
effettivamente eager sia l'allocazione della slab TVM sotto quel binding non è verificabile
dal solo bundle JS: `[VERIFY]` da sorgenti TVM/relax upstream.

Quello che è certo dal probe (`results/s22-ultra-...json`): `maxBufferSize` e
`maxStorageBufferBindingSize` sono ~2 GiB, stesso ordine di grandezza già visto sulla 4090 —
ma lì i 2 GiB sono un limite per-buffer dentro 24 GB di VRAM dedicata, mentre sull'S22 sono
~2 GiB dentro **8 GB di memoria unificata** condivisa con OS, compositor Android, e il resto
del browser. Pesi e KV cache non competono per lo stesso vincolo (il cap 2 GiB), competono
per lo stesso **pool fisico** sottostante, molto più piccolo. Su mobile, allungare il
contesto (più KV cache riservata) lascia meno margine ai pesi e viceversa — un compromesso
che sulla 4090 semplicemente non si vede.

### Vie d'uscita — idee valutate, con prior art cross-engine

| Idea | Prior art | Fattibilità / costo | Rischio |
|---|---|---|---|
| **Warm-up dedicato pre-ramp clock**, separato dal warm-up di shader/cache: un burst di compute scarti prima di far partire il timer TTFT, per forzare il governor GPU a salire di frequenza prima della misura reale | Prassi comune nelle app mobile di inferenza (MLC-LLM Android/iOS, ExecuTorch) che fanno un "dummy pass" prima del primo prompt utente | Basso costo, nessuna modifica al runtime WebLLM: uno script/harness-level nel driver di bench o nella futura app pubblica | Rischio basso; non risolve jitter da contesa OS/compositor, solo quello da DVFS ramp |
| **Prefill chunking più aggressivo** (il parametro `prefillChunkSize` esiste già, `:9930`): spezzare il prefill in dispatch più piccoli per ridurre la coda worst-case da preemption dello scheduler Android | llama.cpp: batch processing configurabile del prompt; riduce i picchi di latenza a scapito di un piccolo overhead di throughput | Medio: parametro già esposto, andrebbe solo *tunato* per device-class mobile (non richiede nuovo codice, richiede sweep di misure) | Rischio medio: trade-off esplicito throughput/tail-latency, va misurato non assunto |
| **KV cache dimensionata sul device, non sul default**: usare `sliding_window_size`/`attention_sink_size` (già nell'ABI, `:9942-9944`, `:12149-12150`) o un `context_window_size` più basso quando il probe rileva `deviceMemoryGB` piccolo o `maxBufferSize` ~2 GiB su memoria unificata | vLLM PagedAttention: alloca pagine on-demand invece di riservare il budget massimo a priori | Medio: la sliding window è già nell'ABI del bundle, serve solo scegliere il default in base al probe già raccolto in fase 1b | Rischio basso tecnico; rischio di prodotto: contesto più corto su mobile è una regressione funzionale visibile, va comunicata non nascosta |
| **KV cache quantizzata** (int8/int4 per K/V, non solo per i pesi) | llama.cpp `--cache-type-k/-v q8_0`/`q4_0`: dimezza o quarta la dimensione della cache a parità di contesto | Alto costo: non esposto nell'ABI TVM/relax attuale di web-llm (`[VERIFY]` se esiste un branch upstream); richiederebbe kernel dequant aggiuntivi nel path attention, fuori dallo scope "niente rework del motore reale" di questa fase | Rischio alto/fuori scope: è la leva con il rapporto guadagno/costo più alto sulla carta, ma non aggirabile senza toccare `src/adapters/webllm.ts` o il runtime MLC — buon candidato per `engine-design-notes.md`, non per un esperimento di fattibilità qui |
| **Allocazione pagine KV lazy invece che upfront su `max_total_sequence_length`** | vLLM PagedAttention (paging dinamico) | Alto costo: tocca l'allocatore TVM/relax, non il layer JS — stesso discorso della quantizzazione, va oltre "nessun rework del motore" | Rischio alto/fuori scope fase corrente; da portare in `engine-design-notes.md` come requisito per un motore custom |

Le prime due idee (warm-up dedicato, chunking tunato) sono le uniche a basso costo/rischio
compatibili con il vincolo "niente rework di `webllm.ts`" di questa fase, e sono i candidati
più sensati per l'unico/i due esperimenti di fattibilità previsti nell'intero goal. Le altre
tre (sliding window come default device-aware, KV quantizzata, paging lazy) richiedono
modifiche più profonde all'ABI o al runtime e sono più adatte a informare
`engine-design-notes.md` — il terreno per un futuro motore custom — che a un esperimento
isolato qui.
