# Compute shader dispatch — come i kernel TVM arrivano a girare via WebGPU

Primo dei quattro doc di deep-dive sul percorso WebGPU di WebLLM/MLC (spec:
`docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md`). Tutte le citazioni al
codice puntano al bundle non-minificato `node_modules/@mlc-ai/web-llm/lib/index.js` di
**@mlc-ai/web-llm 0.2.84** — i numeri di riga valgono per quella versione. I numeri di
performance vengono dai run committati in `results/`.

## Cosa fa

### I kernel non sono nel JavaScript: vivono nel wasm compilato da TVM

WebLLM non contiene sorgenti WGSL dei kernel come stringhe nel bundle (l'unico WGSL
letterale è `canvasRenderWGSL`, riga 4165, usato solo per il render di anteprima su canvas
— nulla a che vedere con l'inferenza). I kernel del modello sono generati dal codegen
WebGPU di TVM **a tempo di compilazione del modello** e incorporati nel binario wasm che
accompagna i pesi. A runtime escono così (`asyncLoadWebGPUPipelines`, righe 7434-7456):

1. `webgpu.get_fmap` (funzione esportata dal wasm) restituisce una mappa JSON dei kernel:
   per ognuno nome, `launch_param_tags` e tipi degli argomenti (riga 7440-7441);
2. `webgpu.get_shader(key)` restituisce la stringa WGSL di quel kernel (riga 7452);
3. per ogni kernel il runtime chiama `createShaderAsync` e, a pipeline pronta,
   `webgpu.update_prebuild` reinstalla la closure di dispatch **dentro la function table
   del wasm** (righe 7454-7456) — da lì in poi è il codice compilato TVM a invocare i
   kernel, non il layer JS di chat.

Esiste anche il percorso sincrono lazy (`wasm.WebGPUCreateShader`, righe 7505-7508): il
codice wasm può chiedere la compilazione di uno shader on-demand. Il grosso passa però
dal preload asincrono al load del modello (righe 12310-12312, 12564).

### Dalla stringa WGSL alla pipeline

`WebGPUContext` (classe da riga 4359) fa il lavoro WebGPU vero:

- **bind group layout**: un binding storage per ogni argomento-buffer del kernel, più un
  binding uniform in coda per gli argomenti scalari/POD (righe 4599-4635);
- **shader module**: `createShaderModule` con `compilationHints` sul layout (righe
  4738-4746) — l'hint permette al driver di compilare per il layout giusto al primo colpo;
- **pipeline**: `createComputePipelineAsync` nel percorso di preload (righe 4747-4757),
  variante sincrona come fallback (righe 4758-4767).

Il bind group invece **non** è precotto: viene creato a ogni dispatch
(`createBindGroup` dentro il closure di submit, righe 4720-4723), perché i buffer
argomento cambiano tra una chiamata e l'altra.

### Il dispatch a runtime: un encoder condiviso, un submit per molti kernel

Il punto architetturale più importante del sotto-sistema (e il meno raccontato nei
writeup in giro, che spesso descrivono ancora "un submit per kernel"):

- ogni invocazione kernel appende un compute pass a un **command encoder condiviso**
  (`pendingEncoder`, righe 4644-4649): `beginComputePass` → `setPipeline` →
  `setBindGroup` → `dispatchWorkgroups` → `end` (righe 4649, 4720-4725);
- le dimensioni dei workgroup vengono dai `launch_param_tags` del fmap
  (`blockIdx.*`/`threadIdx.*`, righe 4580-4598), con un workaround per il limite WebGPU
  di 65535 sul singolo asse: `blockIdx.x` oltre soglia viene ripacchettato su
  `blockIdx.z`, riservato a uso interno (righe 4657-4679);
- gli argomenti POD finiscono in un uniform buffer preso da un pool (righe 4551-4568)
  scritto via `queue.writeBuffer` (riga 4712);
- il `queue.submit` vero avviene solo in `flushCommands()` (righe 4411-4418), forzato da:
  una `sync()`, una copia GPU↔CPU, o una free di buffer (righe 4490, 4832, 4839, 4887,
  4905 — il commento a riga 4829-4831 spiega il perché: liberare un buffer ancora dentro
  un submit pendente è un errore WebGPU).

### La sincronizzazione col JS: una sola volta per token

Nel decode loop, i kernel dell'intero forward pass (attention, matmul, norm, softmax,
sampling top-p) si accumulano nell'encoder senza alcuna sync intermedia. La sync arriva
**una volta per token generato**, quando il layer JS deve leggere l'id del token
campionato: `copyFrom(sampledTokensDevice)` + `yield this.device.sync()` (righe
11126-11135). `sync()` (righe 4445-4458) flusha l'encoder e attende o la promise della
copia GPU→CPU pendente (fast path) o `queue.onSubmittedWorkDone()`. Il readback usa
staging buffer pooled con `mapAsync` (righe 4855-4901).

Non esiste un tetto sul numero di dispatch in-flight nell'encoder: si accumula finché
qualcosa forza il flush. Gli unici contatori (`shaderSubmitCounter`,
`debugShaderSubmitLimit`, righe 4392-4394) sono diagnostici; `debugLogFinish` (riga 4396,
usato a 4727-4732) forza flush+wait per-kernel a scopo di debug — di fatto disattiva il
batching.

### Cosa il runtime NON fa: misurarsi

Il runtime non richiede mai la feature `timestamp-query` e non crea alcun `GPUQuerySet`:
le `requiredFeatures` chieste al device sono solo `shader-f16` e `subgroups`, se
disponibili (righe 4105-4112, dentro `detectGPUDevice`). Tutto il "profiling" interno è
JS-side: `performance.now()` nel chat pipeline (`enable_latency_breakdown`, es. righe
10998-11001) e il contatore di submit. Il tempo GPU per-kernel è **invisibile** — un
fatto che pesa sul micro-bench di fase 6 e sulla sezione bottleneck qui sotto.

## Perché i numeri sono quelli

I run di riferimento: `results/4090-linux-2026-07-26T19-54-55-278Z.json` (Chrome 150,
adapter nvidia/lovelace) e `results/s22-ultra-2026-07-27T00-34-09-931Z.json` (S22 Ultra,
Xclipse 920), stesso modello `Qwen2.5-0.5B-Instruct-q4f32_1-MLC`.

- **Decode 101–116 tok/s sulla 4090 = 8.6–9.9 ms per token.** Ogni token è: N dispatch
  accumulati in un encoder (l'intero forward pass), un submit, una sync col JS per il
  readback dell'id campionato. Il costo per token include quindi un round-trip CPU→GPU→CPU
  fisso per il sampling, non solo il calcolo. Quanto pesi quel round-trip dentro i ~9 ms
  non è misurabile oggi dal runtime stesso (nessun timestamp-query — sopra): è
  esattamente il buco che il micro-bench matmul di fase 6 può colmare. `[VERIFY: quota
  sync/dispatch-overhead per token da misurare col micro-bench]`
- **Decode 6.99 tok/s su S22 = ~143 ms per token.** A parità di grafo di dispatch, il
  rapporto ~13-17× rispetto alla 4090 non viene dall'orchestrazione (identica: stesso
  bundle, stesso wasm) ma dal costo dei kernel stessi su una GPU con una frazione della
  banda — l'analisi del *perché* i kernel siano memory-bound sta nel doc dequant
  (`dequant-kernels.md`, fase 4). Qui conta il negativo: su mobile l'overhead di
  orchestrazione per token è proporzionalmente trascurabile (~9 ms sarebbero il 6% di
  143 ms anche nel caso limite in cui TUTTO il budget 4090 fosse orchestrazione).
- **TTFT warm 290–344 ms (4090) vs 8.2 s (S22).** Il prefill è un singolo mega-batch di
  dispatch (chunked da `prefillChunkSize`) + una sync finale: la varianza mobile di questo
  numero è materia del doc KV/prefill (fase 5).
- **Load warm 1.5–1.9 s (4090) vs 6.1 s (S22), a pesi già in cache.** Il load warm deve
  comunque rifare, per ogni kernel del fmap: fetch della stringa WGSL dal wasm,
  `createShaderModule` (compilazione WGSL→IR driver) e `createComputePipelineAsync`
  (specializzazione per il layout). È il candidato naturale a dominare il load warm dopo
  i pesi; i browser hanno cache interne delle pipeline compilate (Dawn/Chrome) che
  possono assorbire parte del costo tra run consecutivi. `[VERIFY: quota shader-compile
  del load warm non misurata — separabile strumentando asyncLoadWebGPUPipelines; cache
  pipeline del browser da verificare su Chrome 150]`
- **Cold load ~56–62 s (4090, Chrome)**: dominato dal download dei pesi, non dal
  sotto-sistema di questo doc (`results/4090-linux-2026-07-25T19-24-07-948Z.json`,
  `results/4090-linux-2026-07-26T18-34-31-633Z.json`).

## Bottleneck & vie d'uscita

### 1. Una sync per token: round-trip CPU→GPU→CPU nel decode loop

Nel decode loop tutti i kernel del forward pass (attention, matmul, norm, softmax,
sampling top-p) si accumulano in un unico `pendingEncoder` condiviso (righe 4644-4649) e
partono con un solo `queue.submit` dentro `flushCommands()` (righe 4411-4418). Ma quel
flush viene forzato **una volta per token generato**, perché il layer JS deve leggere
l'id campionato: `copyFrom(sampledTokensDevice)` seguito da `yield this.device.sync()`
(righe 11126-11135). `sync()` (righe 4445-4458) flusha l'encoder e attende la promise di
readback pendente o `queue.onSubmittedWorkDone()`.

Sulla 4090 il decode misurato è 101.3-115.6 tok/s
(`results/4090-linux-2026-07-26T19-54-55-278Z.json`, 4 repliche, modello
`Qwen2.5-0.5B-Instruct-q4f32_1-MLC`) = 8.65-9.87 ms/token. Quella cifra include per
costruzione un round-trip CPU→GPU→CPU fisso per il sampling ad ogni singolo token, non
solo calcolo — ma quanto pesi il round-trip dentro i ~9 ms non è misurabile oggi (nessuna
telemetria GPU-side, vedi bottleneck #2).

### 2. Zero visibilità GPU-side

Il device richiesto da WebLLM elenca solo due `requiredFeatures`: `shader-f16` e
`subgroups`, se disponibili (righe 4105-4112, dentro `detectGPUDevice`) —
`timestamp-query` non viene mai richiesta e nessun `GPUQuerySet` viene mai creato. Tutto
il "profiling" interno resta JS-side (`performance.now()`, `enable_latency_breakdown`).
Il tempo per-kernel sulla GPU è invisibile al runtime, mentre l'hardware lo esporrebbe:
il probe 4090 elenca `timestamp-query` tra le `features` dell'adapter
(`results/4090-linux-2026-07-26T19-54-55-278Z.json`, `probe.features`, verificato via
script). Risultato pratico: non possiamo separare, dentro il budget di 9 ms/token del
bottleneck #1, quanto è calcolo e quanto è overhead di sincronizzazione — il claim resta
`[VERIFY]` finché non si strumenta.

### 3. Compile pipeline al load

`asyncLoadWebGPUPipelines` (righe 7434-7456) itera **ogni kernel del fmap** e per
ciascuno fa `createShaderModule` + `createComputePipelineAsync` (righe 4738-4757),
prima che il modello sia dichiarato pronto. Verificato nel loader (righe 12549-12564):
`fetchTensorCache` (download/lettura pesi da cache) viene `yield`-ato per intero, e **solo
dopo** viene `yield`-ato `newPipeline.asyncLoadWebGPUPipelines()` — le due fasi sono
sequenziali, non sovrapposte, per struttura del codice (due `await` in fila), non per una
dipendenza dati necessaria (il modulo wasm con `webgpu.get_fmap`/`get_shader` è già
disponibile prima che i pesi finiscano di arrivare).

Numeri, a pesi già in cache (`cacheState: "warm"`): load 1.46-1.9 s sulla 4090
(`results/4090-linux-2026-07-26T19-54-55-278Z.json`) contro 6.11 s sulla S22 Ultra
(`results/s22-ultra-2026-07-27T00-34-09-931Z.json`), stesso modello. Con i pesi già
disponibili, quel tempo è quasi interamente compile pipeline — candidato naturale a
dominare il load warm.

### 4. Workaround blockIdx.x > 65535 (dormiente sulla scala attuale)

Il dispatch ripacchetta `blockIdx.x` oltre soglia su `blockIdx.z`, riservato a uso interno
(righe 4657-4679): se `workDim[0] >= 65536` lo spezzetta iterativamente (dimezzando,
arrotondando per eccesso se dispari) finché rientra nel limite. Per Qwen2.5-0.5B
(vocab 151936, hidden 896) le dimensioni di dispatch tipiche (es. proiezione finale sul
vocabolario con workgroup a 256 thread → ~594 blocchi) restano ordini di grandezza sotto
65535: il ramo di ripacchettamento con ogni probabilità non si attiva mai nei run
committati. `[VERIFY: nessuna istrumentazione diretta del branch — stima per ordine di
grandezza, non un conteggio a runtime]`. Non lo tratto come bottleneck di performance
oggi; resta un rischio latente per modelli/context futuri con dispatch grid molto più
grandi (vocab estesi, batch>1, context lunghissimi).

| Idea | Prior art | Fattibilità / costo | Rischio | Instradamento |
|---|---|---|---|---|
| Multi-step decode: accumulare N forward pass (N piccolo, 4-8) prima di leggere il token campionato, aggiornando il sampler lato JS solo a fine batch | vLLM `--num-scheduler-steps`: scheduling+input prep una volta ogni N step, CPU overhead spalmato su N step, +28% throughput Llama-70B su 4×H100 ([vLLM v0.6.0 perf update](https://blog.vllm.ai/2024/09/05/perf-update.html)) | Medio: l'encoder condiviso (`pendingEncoder`) già accumula un intero forward pass per turno, va esteso a N turni; serve gestire EOS/stop-token mid-batch (over-generazione da scartare) | Rompe lo streaming token-per-token percepito dall'utente; su prompt corti/EOS frequenti il guadagno si riduce | esperimento |
| Speculative decoding: draft model piccolo + verifica in batch dal modello target | llama.cpp (`docs/speculative.md`), 25-45% riduzione latenza tipica; draft e target competono per la stessa GPU e serializzano se condividono device ([issue #23126](https://github.com/ggml-org/llama.cpp/issues/23126)) | Alto: serve un secondo modello caricato in VRAM (raddoppia il budget memoria, già stretto su mobile) e una pipeline di verifica browser-side non banale | Su S22 il round-trip di sync pesa già ~6% nel caso limite (vedi doc sopra): il guadagno si concentra solo su desktop, mentre il costo di memoria pesa ovunque | scartata — costo-beneficio sbilanciato per un laboratorio single-model, ROI concentrato solo su un device |
| Sync coalescing adattivo: variare N (del multi-step sopra) a runtime in base alla latenza di round-trip misurata, invece di un N fisso | Interrupt coalescing / NAPI a polling adattivo (Linux, dominio non-LLM): sotto carico si passa da interrupt-per-pacchetto a polling batched, con soglie adattive throughput/latenza | Medio-alto: prerequisito sia il multi-step (riga sopra) sia la telemetria timestamp-query (bottleneck #2) per misurare la latenza da cui adattare N | Complessità di tuning (soglie, isteresi) per un guadagno probabilmente marginale se il carico è stabile; utile solo con carico variabile (multi-tab, thermal throttling) | engine-notes — combina due rework (scheduling + telemetria), non isolabile in un esperimento singolo |
| Richiedere `timestamp-query` come `requiredFeature` + creare `GPUQuerySet`, scrivere `timestampWrites` sul compute pass del round decode, risolvere via `resolveQuerySet` | Feature standard WebGPU ([MDN `GPUCommandEncoder.writeTimestamp`](https://developer.mozilla.org/docs/Web/API/GPUCommandEncoder/writeTimestamp), [MDN `resolveQuerySet`](https://developer.mozilla.org/en-US/docs/Web/API/GPUCommandEncoder/resolveQuerySet)); pattern di profiling già usato da altri stack (es. onnxruntime-web performance-diagnosis) | Alto: feature nativa già esposta dall'adapter 4090 nel probe, API stabile, nessun workaround esotico; costo per compute pass è noto essere piccolo ma non zero | Basso su desktop; su GPU TBDR (mobile tile-based) l'implementazione di timestamp-query dentro compute pass è storicamente incompleta/instabile ([gpuweb issue #2046](https://github.com/gpuweb/gpuweb/issues/2046)) — supporto S22 da verificare, non presente nel probe controllato qui | esperimento — è l'infrastruttura che la route micro-bench di fase 6 può costruire; sblocca la misura per gli altri due bottleneck |
| Fork/patch locale del bundle vendored per aggiungere `timestamp-query` e instrumentare `WebGPUContext` in modo permanente nel runtime chat (non solo nel micro-bench) | Pattern comune di vendoring/patching quando il profiling manca upstream (nessuna singola famiglia esterna, via diretta) | Medio: tocca codice vendored (`node_modules/@mlc-ai/web-llm`), va risincronizzato a ogni bump di `@mlc-ai/web-llm` | Drift dal bundle upstream, costo di manutenzione ricorrente | engine-notes — rework del motore, destinazione `engine-design-notes.md` |
| Tracciare via Chrome DevTools / `chrome://tracing` (categoria `disabled-by-default-gpu.dawn`) invece di timestamp-query in-app | Dawn (backend WebGPU di Chrome) espone tracing interno usato dal team Dawn per il proprio profiling | Basso per un laboratorio riproducibile: richiede flag Chrome, non gira headless/cross-browser | Non portabile a Firefox/Safari/S22, output non JSON-abile nei `results/` committati | scartata — non integrabile nell'infrastruttura di risultati esistente |
| Sovrapporre `fetchTensorCache` (pesi) e `asyncLoadWebGPUPipelines` (oggi sequenziali per struttura del codice, righe 12549-12564), invece di attendere i pesi prima di iniziare la compilazione | `WebAssembly.compileStreaming`: i motori JS compilano il modulo wasm mentre i byte sono ancora in arrivo dalla rete, invece di aspettare il download completo — stesso principio di overlap I/O↔compute | Medio: il modulo wasm con `webgpu.get_fmap`/`get_shader` è verificabilmente disponibile prima della fine del fetch pesi; i due path non condividono buffer | Basso: sono percorsi indipendenti (compile shader non tocca i buffer pesi); rischio principale è contesa CPU/rete difficile da isolare senza instrumentazione | esperimento — cronometrabile subito invertendo/parallelizzando le due `yield`, alto rapporto guadagno/costo se confermato |
| Demand-paging dei kernel: compilare solo i kernel toccati dal primo forward pass (prefill) e lasciare il resto al path lazy già esistente (`wasm.WebGPUCreateShader`, righe 7505-7508) invece di forzare il preload completo del fmap | Paginazione a richiesta (OS demand paging / lazy page-in) | Medio-alto: il primitivo lazy esiste già nel runtime, non serve nuova API | Il primo forward pass probabilmente tocca già la maggioranza dei kernel distinti (stessa architettura per ogni layer): guadagno reale incerto `[VERIFY: quota di kernel del fmap esclusivi di configurazioni rare vs. usati dal primo forward pass — non misurata]`; introduce anche un possibile stallo sincrono imprevedibile a metà generazione se un kernel raro compare tardi | engine-notes — cambia il contratto "modello pronto quando il fmap è compilato", va progettato prima di isolarlo come esperimento |
| Migrare l'esecuzione a WebNN (grafo compilato una volta su backend nativo — DirectML/CoreML/NNAPI/XNNPACK — niente WGSL) | W3C WebNN spec, `MLGraph` immutabile compilato una volta, nessuna compilazione shader lato pagina ([webnn.io architecture](https://webnn.io/en/faq/architecture)) | Basso nel breve periodo: WebNN ha un op-set fisso, non supporta i kernel custom di dequantizzazione INT4 group-wise generati dal codegen TVM | Alto: mismatch di feature-set (quantizzazione custom, kernel generati), supporto browser ancora parziale `[VERIFY: browser-compat WebNN su Chrome150/S22]` | scartata — incompatibile con l'architettura a kernel-custom-TVM del progetto |
| Instrumentare `submitShader` per contare quante volte il branch `workDim[0] >= 65536` si attiva realmente nei run del laboratorio | Nessuna famiglia esterna necessaria — stessa logica "misura prima di ottimizzare" del bottleneck #2 | Molto bassa: poche righe, patch locale al bundle vendored, solo dev-build | Nessuno in produzione | engine-notes — patch locale, va in `engine-design-notes.md` insieme a un eventuale fix se il branch risultasse attivo |

**Raccomandazione**: l'esperimento da proporre per primo è **timestamp-query nel micro-bench di fase 6** (riga 2 della tabella). Non è il guadagno di throughput più vistoso sulla carta — quello sarebbe il multi-step decode, con precedente esterno più solido (+28% misurato da vLLM) — ma è l'unico che sblocca la misura degli altri due: senza numeri GPU-side non possiamo dire se il multi-step decode stia davvero comprando calcolo o solo nascondendo un round-trip che pesava meno del previsto, né isolare quanto della finestra 1.46-1.9 s di load warm sia compile pipeline puro contro overhead di orchestrazione JS. È inoltre a basso rischio e basso costo di implementazione (feature standard, già esposta dall'adapter 4090 nel probe), e usa un'infrastruttura (route SPA micro-bench matmul) già pianificata per fase 6 — non richiede di toccare il bundle vendored di produzione.
