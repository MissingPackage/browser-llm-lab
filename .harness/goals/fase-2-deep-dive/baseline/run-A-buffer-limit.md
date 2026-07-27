# Baseline run A — bottleneck: maxStorageBufferBindingSize ~2 GiB (per fase 3)

<!-- Output verbatim del subagent, 2026-07-27, senza skill. Da ri-verificare prima dell'uso. -->

## Bottleneck & vie d'uscita

### Il muro è a tre piani, non uno

Il numero che si vede nel probe (`maxStorageBufferBindingSize: 2147483644`, cioè 2³¹-4 byte)
non è il soffitto operativo reale. È il *cap del device*: identico sulla RTX 4090
(`results/4090-linux-2026-07-25T19-24-07-948Z.json`, vendor `nvidia`/`lovelace`) e sulla GPU
Xclipse 920 dello Snapdragon dell'S22 Ultra (`results/s22-ultra-2026-07-27T00-34-09-931Z.json`,
`vendor: samsung`/`rdna-2`) — due architetture del tutto diverse che convergono sullo stesso
valore, prova che è un vincolo dell'API WebGPU (probabile ragione: dimensioni/offset dei
binding devono restare rappresentabili in 32 bit con segno), non un capriccio del driver.

Ma WebLLM non chiede mai quel cap al device. In `detectGPUDevice()` la richiesta è
hardcoded a 1 GiB (`1 << 30`), con un fallback a 128 MiB (`1 << 27`) se anche 1 GiB eccede
il limite del device (`node_modules/@mlc-ai/web-llm/lib/index.js:4067-4082`). Il fallback non
è teorico: il run Firefox 152 su LLVMPIPE (software rasterizer,
`results/4090-linux-2026-07-25T20-08-26-375Z-firefox152-LLVMPIPE-CPU.json`) mostra
`maxStorageBufferBindingSize: 134217728` — esattamente `1 << 27` — mentre `maxBufferSize`
sulla stessa riga è rimasto a `1073741824` (1 GiB pieno): i due limiti degradano in modo
indipendente, e qui solo il binding storage è caduto al gradino più basso.

Quindi sull'hardware di questo progetto il soffitto *per singolo binding* è 1 GiB, non i
~2 GiB che l'adattatore dichiara: metà dello spazio disponibile viene lasciato sul tavolo
per garantire portabilità verso device più deboli — scelta ragionevole per una libreria
general-purpose, ma è il vero numero da cui partire, non quello del probe.

C'è anche un secondo muro, distinto: `maxBufferSize` (dimensione massima di una singola
allocazione, non del binding usato in uno shader). Sulla 4090 è `4294967292` (~4 GiB-4),
sull'S22 Ultra è `2147483648` (2 GiB esatti) — quindi su mobile i due muri (allocazione
massima e binding massimo) sono praticamente a ridosso l'uno dell'altro, mentre su desktop
c'è un margine 2×. Una strategia che impacchettasse più tensori in un unico `GPUBuffer`
grande (per ridurre l'overhead di allocazione) sbatterebbe contro il muro più stretto su
mobile molto prima che su desktop.

### Perché in pratica non si sbatte quasi mai

Il caricamento dei pesi non alloca un buffer monolitico per il modello: ogni record di
parametro (`rec`, un singolo tensore) diventa un NDArray/buffer GPU a sé
(`tvm.empty(rec.shape, rec.dtype, device)`, `node_modules/@mlc-ai/web-llm/lib/index.js:7086-7106`).
Lo "spezzettamento su più buffer" descritto nel bottleneck non è quindi un accorgimento
aggiunto per aggirare il cap: è semplicemente come sono organizzati i parametri, un buffer
per tensore. Più tensori vengono poi bindati insieme nello stesso dispatch — la libreria
richiede uno headroom di 10 storage buffer per shader stage contro il default WebGPU di 8
(stesso blocco di codice, riga ~4079), e i device osservati ne offrono 16
(`maxStorageBuffersPerShaderStage` in tutti i run).

Conseguenza pratica: il cap (1 GiB, o 2 GiB, o 128 MiB a seconda del device) non limita la
taglia totale del modello, ma la taglia del *singolo tensore più grande*. Per architetture
transformer dense in fascia servibile da browser (≤ ~8B, quantizzate q4) nessuna matrice di
peso si avvicina alla soglia.

Dove invece si avvicina — stima a mano, **[VERIFY]**, non presa da un run: una tabella
embedding/lm_head in fp16 (non quantizzata) con vocabolario ~128k e hidden 4096 pesa
`128256 × 4096 × 2 byte ≈ 1.050.673.152 byte ≈ 0.978 GiB` — sotto la soglia WebLLM di 1 GiB,
ma con un margine di appena il ~2%. Bastano vocabolario più grande, hidden size maggiore, o
niente quantizzazione sull'embedding (pratica comune anche nei modelli quantizzati, che
spesso lasciano fp16 proprio embedding e lm_head) per superarla. Stesso discorso per stack
di esperti MoE se il layout li vuole concatenati in un'unica allocazione, o per kernel
fusi ad hoc che si aspettano un blocco di pesi contiguo invece che binding per-tensore.

### Vie d'uscita valutate

1. **Alzare il limite richiesto fino al cap reale del device.** Costo basso, guadagno
   asimmetrico: raddoppia lo spazio utile su desktop (1 GiB → ~2 GiB) ma non aggiunge nulla
   sull'S22 Ultra, dove `maxBufferSize` è già al pari del binding cap. Il codice esistente
   già negozia in sicurezza (richiede X, verifica contro `adapter.limits`, retrocede se
   eccede) quindi il pattern non è rischioso di per sé — ma richiederebbe una patch/fork di
   `detectGPUDevice()` in `@mlc-ai/web-llm`, fuori scope per questa fase
   (`src/adapters/webllm.ts` non si tocca, per contratto dello spec madre). Idea per
   `engine-design-notes.md`, non per ora.

2. **Split esplicito dei tensori oversize su più buffer + bind group aggiuntivi**, stesso
   principio del binding per-tensore già in uso ma esteso ai casi limite (embedding/lm_head
   giganti, esperti MoE): divisione per righe/colonne sotto soglia più un kernel di
   indirezione che sceglie il buffer corretto dall'indice di riga. Prior art diretto:
   llama.cpp fa lo split analogo quando un singolo backend (es. Vulkan) o un singolo device
   non regge l'intero tensore (`--split-mode row`); ONNX Runtime Web con l'EP WebGPU applica
   lo stesso schema ai grandi initializer di rete. Costo moderato (tocca i kernel
   TVM-generati, non solo l'inizializzazione JS); rischio basso perché il pattern di bind
   multiplo è già collaudato nel runtime.

3. **Weight streaming/paging per-layer**, ispirato a `antirez/ds4` (SSD streaming per
   sopperire alla VRAM) e a FlexGen/DeepSpeed-Inference/`llama.cpp --n-gpu-layers` parziale:
   il decode consuma i pesi layer per layer in ordine deterministico, quindi si presta al
   prefetch del layer N+1 mentre si calcola il layer N, tenendo residente solo un working
   set sotto soglia. Nel browser il "disco" diventerebbe IndexedDB/OPFS — molto più lento di
   un vero NVMe, quindi il margine di prefetch è più stretto che nel caso ds4 originale.
   Risolve un problema più ampio (VRAM totale insufficiente) di cui l'edge-case del singolo
   binding è solo un caso particolare; costo alto, non giustificato dal solo bottleneck del
   buffer da 2 GB.

4. **Backend alternativo, WebNN.** L'API a grafo astrae la gestione dei buffer al
   browser/driver e potrebbe non esporre lo stesso muro per-binding all'applicazione —
   **[VERIFY]**, non testato di persona: WebNN è esplicitamente deferred dallo spec madre di
   progetto. Nota per il motore custom futuro, non un'azione di questa fase.

### Cosa non è un bottleneck reale, qui

Dato lo split-per-tensore già in essere, il vincolo più stringente osservato sui device del
progetto non è il binding da 1-2 GiB ma il fallback a 128 MiB su GPU software/deboli
(Firefox su LLVMPIPE). È un problema di classe di device, non di layout dei buffer, e si
affronta a monte — rifiutando l'adattatore software, come già fa il progetto (vedi
landmine "Firefox silent-fallback a CPU" in `HANDOFF.md`) — non con un packing dei pesi più
furbo.
