# Il muro dei "2 GB" — buffer WebGPU e spezzettamento dei pesi

Secondo doc di deep-dive sul percorso WebGPU di WebLLM/MLC (spec:
`docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md`). Citazioni al bundle
non-minificato `node_modules/@mlc-ai/web-llm/lib/index.js` di **@mlc-ai/web-llm 0.2.84**;
numeri dai run committati in `results/`.

## Cosa fa

### Il muro è a tre piani, non uno

Il numero che si vede nel probe (`maxStorageBufferBindingSize: 2147483644`, cioè 2³¹-4
byte) non è il soffitto operativo reale. È il *cap dell'adapter*: identico sulla RTX 4090
(`results/4090-linux-2026-07-26T19-54-55-278Z.json`, vendor nvidia/lovelace) e
sull'Xclipse 920 dell'S22 Ultra (`results/s22-ultra-2026-07-27T00-34-09-931Z.json`) —
architetture del tutto diverse che convergono sullo stesso valore: un vincolo dell'API
WebGPU, non del driver. Il valore identico cross-vendor (2³¹−4) punta a un vincolo di rappresentazione a 32 bit nell'API o nelle implementazioni; qui resta un'osservazione, non una spiegazione tracciata a spec.

WebLLM però non chiede mai quel cap al device. In `detectGPUDevice` **entrambi** i limiti
sono richiesti hardcoded a 1 GiB (`1 << 30`):

- `maxBufferSize`: richiesto 1 GiB, fallback 256 MiB (`1 << 28`) se il device non regge
  (righe 4050-4066);
- `maxStorageBufferBindingSize`: richiesto 1 GiB, fallback 128 MiB (`1 << 27`)
  (righe 4067-4082).

Il fallback non è teorico: il run Firefox su LLVMPIPE (rasterizer software,
`results/4090-linux-2026-07-25T20-08-26-375Z-firefox152-LLVMPIPE-CPU.json`) mostra
`maxStorageBufferBindingSize: 134217728` — esattamente `1 << 27`.

Quindi i tre piani del muro, dal più largo al più stretto:

| Piano | Valore | Dove vive |
|---|---|---|
| Limite API/adapter | ~2 GiB per binding (2³¹-4); `maxBufferSize` 4 GiB su 4090, 2 GiB su S22 | probe, `results/*.json` |
| **Richiesta WebLLM** | **1 GiB per entrambi, hardcoded** | bundle, righe 4050-4082 |
| Fallback WebLLM | 256 MiB (`maxBufferSize`) / 128 MiB (binding) | stesso blocco; osservato su LLVMPIPE |

Il device su cui girano i kernel vive sul piano di mezzo: metà dello spazio che
l'hardware offrirebbe resta sul tavolo, in cambio di portabilità verso device deboli.
Nota di misura: il probe di questo progetto legge i limiti dell'*adapter*, non del
*device* creato da WebLLM — i `results/*.json` raccontano il piano 1, il runtime vive
nel piano 2.

### Lo "spezzettamento" dei pesi: non un accorgimento, il default

Il caricamento non alloca mai un buffer monolitico per il modello: ogni record di
parametro (un singolo tensore) diventa un NDArray/buffer GPU a sé. Per ogni tensore il
loader fa: decode nello staging CPU → alloca l'array GPU (`this.empty(rec.shape,
rec.dtype, device)`) → `copyFrom` → **`yield device.sync()`** → registra nel tensor
cache (righe 7086-7106). Due conseguenze:

1. il cap per-binding non limita la **taglia totale** del modello ma la taglia del
   **singolo tensore più grande**;
2. l'upload dei pesi è **serializzato con una sync GPU completa per tensore** — centinaia
   di round-trip al load, un costo strutturale del percorso di caricamento che rientra
   nella finestra di load warm (1.5–1.9 s su 4090, 6.1 s su S22 — vedi
   `compute-shader-dispatch.md` per l'altra metà di quella finestra, la compile delle
   pipeline).

Più tensori vengono poi bindati insieme nello stesso dispatch: la libreria chiede
headroom di storage buffer per shader stage oltre il default WebGPU di 8, e i device
osservati ne offrono ampiamente di più (`maxStorageBuffersPerShaderStage`: 16 sui probe
4090 e S22, 48 su LLVMPIPE).

## Perché i numeri sono quelli

- **Perché nessun run committato ha mai sbattuto contro il muro**: con lo split
  per-tensore, per transformer densi in fascia browser (≤ ~8B, q4) nessuna matrice si
  avvicina a 1 GiB. Il tensore a rischio è l'embedding/lm_head non quantizzato: con
  vocabolario ~128k e hidden 4096 in fp16 fa `128256 × 4096 × 2 ≈ 0.978 GiB` — sotto la
  soglia WebLLM di 1 GiB con un margine del ~2% (conto a mano su un layout ipotetico:
  nessun modello del set di progetto lo ha — Qwen2.5-0.5B ha vocab 151936 × hidden 896 ≈
  0.25 GiB fp16). Vocabolari più grandi, hidden maggiori o embedding non quantizzati
  bastano a superarla: è lì che il muro diventa reale.
- **Perché il muro pesa diversamente per device**: sulla 4090 il binding cap (1-2 GiB) è
  un limite *per-tensore* dentro 16 GB di VRAM; sull'S22 `maxBufferSize` (2 GiB esatti) e
  binding cap coincidono dentro 8 GB di memoria unificata condivisa con OS e compositor —
  i muri per-allocazione e per-binding sono a ridosso, e il vincolo vero è il pool
  fisico, non il cap. Per l'M4 Pro (48 GB unificati, sweep manuale in corso), il
  binding cap da ~2 GiB è il candidato numero uno a cappare il vantaggio-capienza:
  trattato dallo spec madre come finding da misurare, non solo rischio. Probe M4 non
  ancora disponibile: buco dichiarato, in attesa dello sweep manuale.
- **Perché il fallback LLVMPIPE è il vero pavimento**: il gradino a 128 MiB non è mai
  scattato su hardware reale nei run committati — solo sul rasterizer software. Il
  vincolo stringente per la classe di device debole non è il layout dei buffer ma il
  rifiuto dell'adapter software a monte (landmine "Firefox silent-fallback" in
  `HANDOFF.md`).

## Bottleneck & vie d'uscita

### Bottleneck 1 — cap per-binding richiesto hardcoded a 1 GiB, non negoziato al cap reale del device

Come mostrato sopra, `detectGPUDevice()` chiede sempre `1 << 30` (1 GiB) sia per `maxBufferSize` che per `maxStorageBufferBindingSize`, con retrocessione a `1 << 28` (256 MiB) e `1 << 27` (128 MiB) solo se il device non regge nemmeno quello (`node_modules/@mlc-ai/web-llm/lib/index.js`, righe ~4050-4082, verificate a mano su `@mlc-ai/web-llm 0.2.84`). Il cap dell'*adapter* letto dai probe è più largo — 2147483644 (~2 GiB-4, binding) sia su 4090 che su S22 Ultra, e `maxBufferSize` 4294967292 (~4 GiB) sulla 4090 contro 2147483648 (2 GiB esatti) sull'S22 (`results/4090-linux-2026-07-26T19-54-55-278Z.json`, `results/s22-ultra-2026-07-27T00-34-09-931Z.json`) — ma il *device* che WebLLM crea e su cui girano i kernel vive comunque al piano di mezzo, 1 GiB, qualunque cosa l'adapter offra. Metà dello spazio per-binding resta sul tavolo su desktop; sull'S22, dove `maxBufferSize` è già a 2 GiB, il margine perso è minore in proporzione. Il bottleneck non è il numero di targa (~2 GiB) ma la richiesta hardcoded a monte.

### Bottleneck 2 — upload dei pesi serializzato, una sync GPU completa per tensore

Distinto dal cap: nel loop di caricamento, per ogni record/tensore il runtime fa decode CPU → alloca `gpu_arr` → `copyFrom` → **`yield device.sync()`** prima di passare al tensore successivo (stesso bundle, righe ~7086-7106). Non è un limite di taglia ma di *pipeline*: centinaia di round-trip sync-per-tensore, un costo strutturale che ricade dentro la finestra di load warm osservata (1.5-1.9 s sulla 4090, 6.1 s sull'S22 Ultra — l'altra componente di quella finestra è la compile delle pipeline, coperta in `compute-shader-dispatch.md`). Non risulta da nessun run che questo sync-per-tensore sia mai stato una scelta deliberata di throttling piuttosto che semplicità di implementazione: è un candidato a basso rischio per una misura diretta.

### Idee valutate

| Idea | Prior art | Fattibilità / costo | Rischio | Instradamento |
|---|---|---|---|---|
| Alzare la richiesta al cap reale del device (leggere `adapter.limits` invece di `1<<30` hardcoded) | Il pattern "richiedi X, negozia al ribasso se eccede" è già nello stesso blocco di codice (righe 4050-4082) | Basso concettualmente, ma tocca `detectGPUDevice()` nel bundle vendorizzato — fuori da `src/adapters/webllm.ts`, che per contratto di fase non si modifica (`docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md`, riga 104) | Basso di per sé (pattern collaudato), ma introduce drift da un pacchetto upstream terzo | engine-notes |
| Split esplicito dei tensori oversize (embedding/lm_head non quantizzato, esperti MoE concatenati) su più buffer + kernel di indirezione per riga | llama.cpp `--split-mode row` quando un device non regge un tensore intero; ONNX Runtime Web (EP WebGPU) applica split analogo ai grandi initializer di rete comportamento dell'EP WebGPU sugli initializer oversize non ri-verificato sul repo ORT — check instradato a engine-notes | Moderato: tocca i kernel TVM-generati, non solo l'init JS | Basso: il bind multiplo per-tensore è già il pattern esistente, qui solo esteso | engine-notes — nessun modello del set attuale ci si avvicina (conto a mano: embedding fp16 vocab~128k/hidden 4096 ≈ 0.978 GiB, sotto 1 GiB con ~2% di margine; nessun modello di progetto ha questo layout, es. Qwen2.5-0.5B ≈ 0.25 GiB) |
| Diradare/eliminare il `device.sync()` per-tensore nel loop di upload pesi (es. un sync ogni N tensori, o accodare tutte le `copyFrom` e sincronizzare una sola volta a fine shard) | Pattern "non sincronizzare per singola operazione, batch sulla coda di comandi" comune in Vulkan/D3D12 e nel modello a coda di comandi della stessa WebGPU; overlap copy/compute stile CUDA streams in FlexGen/DeepSpeed-Inference | Basso-medio: modifica localizzata al loop di caricamento (righe ~7086-7106), testabile con un build locale patchato del bundle vendorizzato senza toccare `src/adapters/webllm.ts` | Medio: non è noto se il sync-per-tensore serva a limitare il picco di staging CPU concorrente — ipotesi non testata, da verificare nell'eventuale esperimento (docket #4) | **esperimento** |
| Weight streaming/paging per-layer stile `antirez/ds4` (SSD streaming per sopperire alla VRAM) + FlexGen/DeepSpeed-Inference/llama.cpp `--n-gpu-layers` parziale, prefetch del layer N+1 mentre si calcola N | ds4 (storage esterno), FlexGen/DeepSpeed-Inference (offload CPU/disco), llama.cpp (caricamento parziale su GPU) | Alto: nel browser il "disco" diventa IndexedDB/OPFS, va gestito ordine di prefetch e invalidazione cache | Alto: IndexedDB/OPFS molto più lenti di un vero NVMe, margine di prefetch più stretto che nel caso ds4 originale | scartata — risolve un problema più ampio (VRAM/memoria totale insufficiente) di cui il cap per-binding è solo un caso particolare; nessun run del progetto ha mai sbattuto contro il cap, quindi il costo non è giustificato da questo bottleneck specifico |
| Backend alternativo WebNN (API a grafo, gestione buffer delegata al driver/browser) | Motori browser: WebNN come alternativa a WebGPU compute diretto | Non testato di persona se l'astrazione a grafo esponga o meno lo stesso muro per-binding — fronte deferred dallo spec madre | Alto: richiederebbe riscrivere l'intero percorso di inferenza, non solo il loader pesi | scartata — WebNN è esplicitamente deferred dallo spec madre di progetto |
| Staging-ring / double buffering per l'upload pesi (pool di 2-3 buffer di staging riciclati, upload asincroni sovrapposti, sync solo quando il pool si esaurisce) — trasferimento da grafica realtime/OS, non da un motore LLM | Multiple-buffering nelle pipeline grafiche in tempo reale (swap chain a N buffer); code di completamento asincrone stile I/O batched (`io_uring`) invece di un thread/sync per richiesta | Medio: stessa direzione dell'idea "diradare la sync" ma più invasiva — richiede gestione esplicita del ciclo di vita di un pool di buffer, non solo spostare dove cade la `yield` | Medio: complessità di gestione del pool ricreata da zero, stesso rischio di ipotesi non verificata sul perché la sync attuale sia per-tensore | engine-notes — versione più ambiziosa dell'idea "diradare la sync"; quella resta la forma minima testabile subito |

### Raccomandazione

Come esperimento di fattibilità proporrei **diradare/eliminare il `device.sync()` per-tensore nel loop di upload pesi**. È l'unica idea della tabella che: (a) attacca un bottleneck osservato oggi, non ipotetico — la finestra di load warm misurata (1.5-1.9 s / 6.1 s) — mentre il cap a 1 GiB non è mai stato toccato da nessun run committato; (b) è localizzata a ~20 righe del loop di caricamento, testabile con una build locale patchata del bundle vendorizzato senza toccare `src/adapters/webllm.ts` né i kernel TVM-generati; (c) ha un criterio di successo binario e misurabile (loadMs prima/dopo, stesso harness di `results/`). Le idee sul cap-per-binding (alzare la richiesta, split dei tensori) restano `engine-notes`: nessun run le rende urgenti oggi, ma vanno annotate per il motore custom futuro — in particolare in vista dell'M4 con 48 GB, dove il cap ~2 GiB è trattato dallo spec madre come finding di capienza, non solo rischio. Dato il tetto di 1-2 esperimenti per l'intera fase 2, questa candidatura compete con quelle degli altri tre doc di sotto-sistema: va decisa in docket, non eseguita da questa sezione.
