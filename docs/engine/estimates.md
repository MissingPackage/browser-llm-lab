# Stima quantitativa delle leve — input di progetto per il nuovo motore

Prodotto dalla **sessione di stima** aperta dal ruling PI 2026-07-28 (docket
fase-2-deep-dive #9): quantificare l'incremento atteso da *tutti* i suggerimenti del
deep-dive, con trade-off, alternative e motivazioni, in funzione dell'obiettivo
dichiarato — costruire un motore di inferenza browser nuovo.

Continua `docs/deep-dive/engine-design-notes.md` (il backlog qualitativo). La differenza:
lì le idee erano **instradate**, qui sono **stimate** contro un modello di budget
misurato.

## 0. Perché il ranking del deep-dive va rifatto da zero

Le colonne "Instradamento" dei quattro doc di sotto-sistema sono state compilate sotto un
vincolo di fase esplicito — *niente rework del motore, niente tocchi a
`src/adapters/webllm.ts` né ai kernel generati da TVM*. Quel vincolo decideva da solo gran
parte delle righe: tutto ciò che chiedeva di riscrivere il runtime finiva in `engine-notes`
o in `scartata` **per costo di integrazione, non per valore atteso**.

Nel contesto "nuovo motore" quel vincolo non esiste più. Le idee vanno quindi ri-valutate
sul solo asse valore/rischio, e alcune si spostano molto (§5, §6).

## 1. La misura nuova di questa sessione

Il modello di budget del deep-dive poggiava su un parametro mai misurato: **quanti
dispatch WebGPU costa un token**. I doc usano "~34 dispatch per token", che è il numero di
kernel **distinti** nel dump WGSL, non il numero di invocazioni.

Tool: `.harness/tools/dispatch-profile.mjs` (patch dei prototype WebGPU dentro il worker,
stesso pattern di `wgsl-dump.mjs`; contatori esatti + timer aggregati campionati dal
driver). Run: `results/dispatch-profile/dispatch-profile-4090-linux-2026-07-28T18-14-25-294Z.json`
— 4090 laptop, Chrome branded, `Qwen2.5-0.5B-Instruct-q4f32_1-MLC`, warm-up + 3 repliche
= 1020 token di decode, 275 456 dispatch totali.

| Grandezza | Misurato | Il deep-dive diceva |
|---|---|---|
| dispatch / token | **270.1** | ~34 |
| submit (`queue.submit`) / token | **7.02** | 1 |
| readback `mapAsync` / token | **1.00** | 1 ✅ |
| `createBindGroup` / dispatch | **1.00** | 1 ✅ (righe 4720-4723) |
| dispatch : submit | **38.45** (269/7, stabilissimo su 63 campioni) | — |

Il rapporto dispatch/submit è costante al terzo decimale per tutta la finestra di decode:
non è rumore, è la struttura del grafo.

**Costo CPU lato renderer, per chiamata** (aggregato su 275 456 campioni; quanto di
`performance.now()` misurato = 5 µs, non assunto — l'errore di quantizzazione è non-distorto
sulla media a questa numerosità):

| Chiamata | µs/chiamata | ms/token |
|---|---|---|
| `createBindGroup` | **5.22** | 1.41 |
| `beginComputePass` | 1.13 | 0.31 |
| `queue.writeBuffer` (arg POD) | 1.13 | 0.31 |
| `dispatchWorkgroups` | 0.37 | 0.10 |
| `pass.end` | 0.37 | 0.10 |
| `submit` + `createCommandEncoder` | 10.08 | 0.07 |
| **totale encode CPU** | **8.30 / dispatch** | **2.24** |

Due fatti strutturali che questa tabella inchioda: **il 64% del costo di encoding è una
sola chiamata** (`createBindGroup`, rifatta a ogni dispatch perché l'ABI a function-table
di TVM non sa che i buffer sono gli stessi di ieri), e **l'encoder viene flushato 7 volte
per token**, non una. I trigger di flush sono documentati nel bundle (righe 4832 free
buffer, 4839 write CPU→GPU, 4887 readback, 4905 copy GPU→GPU, 4448 sync): la catena di
sampling alloca/libera scratch e copia, quindi rompe il batching 6 volte in più del
necessario. *[VERIFY: l'attribuzione per singolo call-site non è strumentata — vedi §7 M2.]*

## 2. Il modello di budget

```
T_token  =  T_gpu_work  +  N_disp × (t_floor_gpu + t_encode_cpu)  +  n_sync × t_roundtrip  +  R
```

`t_roundtrip` è misurato dai JSON del micro-bench e non era mai stato estratto: in ogni
cella, `cpuMs − gpuMs` moltiplicato per i 16 dispatch del campione dà il costo di un
round-trip `submit → onSubmittedWorkDone`, **indipendente dal lavoro nel batch**:

| Device | round-trip misurato (media su 10-15 celle) |
|---|---|
| 4090 laptop | **1.8 ms** (range 1.1-2.5) |
| M4 Pro | **0.45 ms** (range 0.34-0.89) |
| S22 Ultra | **5.0 ms** (range 4.2-6.7) |

### 4090 laptop — decomposizione (baseline 9.3 ms/token, media delle 4 configurazioni 101-116 tok/s)

| Termine | ms | % | provenienza |
|---|---|---|---|
| encode CPU (270 × 8.30 µs) | 2.24 | 24% | **misurato** (questa sessione) |
| submit CPU (7 × 10.1 µs) | 0.07 | 1% | **misurato** |
| round-trip di sync (1/token) | 1.80 | 19% | **misurato** (micro-bench) |
| lavoro GPU (270 × 5 µs floor + ~0.8 lm_head) | 2.15 | 23% | stimato dal micro-bench |
| **residuo non attribuito** | **3.04** | **33%** | — |

Il residuo è il limite di onestà di questa stima: contiene il lavoro del processo GPU
(validazione e traduzione dei command buffer, che il wrapper lato renderer non vede) e la
pipeline JS di chat. **Non lo assegno a nessuna leva.** La tesi headline del deep-dive
("75-85% orchestrazione") sopravvive — il lavoro GPU è il 23% — ma l'attribuzione interna
di quel 77%, che è ciò che decide quale leva tirare, cambia completamente.

### M4 Pro e S22 — stessi termini, regimi opposti

`N_disp = 270` si assume identico (stesso modello, stessa quant, stesso runtime → stesso
grafo). L'encode CPU non è misurato su quei device (§7 M1).

| | 4090 (9.3 ms) | M4 Pro (10.2 ms) | S22 f16 (86.2 ms) |
|---|---|---|---|
| floor GPU × 270 | 1.35 ms (14%) | **7.6 ms (75%)** ⚠︎ | **35 ms (41%)** |
| round-trip sync | 1.80 ms (19%) | 0.45 ms (4%) | 5.0 ms (6%) |
| encode CPU | 2.24 ms (24%) | non misurato | non misurato |
| regime | **encode + sync** | **floor per-dispatch** | **floor + kernel** |

⚠︎ Il floor M4 (28 µs) viene dalla cella q4 896², che contiene lavoro reale: è un limite
*superiore* del floor vero. Se il floor nudo fosse ~15 µs, la quota GPU dell'M4 scende al
40% e il resto va in encode. È la maggiore incertezza del quadro cross-device, e si chiude
con un run del tool su M4 (§7 M1).

## 3. Le leve, stimate

Applicate **in sequenza** su 4090, così non si sommano guadagni sovrapposti (ogni riga
parte dal residuo della precedente). Le prime tre sono pura architettura del motore: non
costano niente all'utente e non richiedono kernel engineering.

| # | Leva | Meccanismo | Δ ms | T dopo | cumulato |
|---|---|---|---|---|---|
| — | baseline WebLLM | | | 9.30 | 107 tok/s |
| L1 | **Bind group precomputati** (grafo statico: i bind group nascono al load, non a ogni dispatch) | −5.22 µs × 270 | −1.41 | 7.89 | **1.18×** |
| L2 | **Un submit per token** (scratch preallocato → zero free, zero copie intermedie) | −0.06 lato CPU (+ quota ignota del residuo) | −0.06 | 7.83 | 1.19× |
| L3 | **Fusione: 270 → ~100 dispatch** (norm+matmul, silu+mul nel matmul, catena sampling 27→1) | −170 × (3.08 encode residuo + 5 floor) µs | −1.37 | 6.46 | **1.44×** |
| L4 | **Multi-step decode N=4** (sync ogni 4 token) | −1.80 × ¾ | −1.35 | 5.11 | 1.82× |
| L5 | **Kernel dequant migliore** (vec4, scale in shared, layout Marlin) | solo la parte sopra-floor: lm_head ~0.8 → ~0.2 | −0.60 | 4.51 | **2.06×** |

**9.3 → 4.5 ms/token = 2.06× (107 → 222 tok/s)**, di cui **1.44× dalle sole L1-L3**, che
non hanno costo di prodotto (nessuna regressione di streaming, nessun WGSL nuovo).

Le stesse leve, sugli altri due regimi:

| Leva | 4090 | M4 Pro | S22 f16 | perché diverge |
|---|---|---|---|---|
| L1 bind group | **−15%** | −14% ᵉ | −5% ᵉ | costo CPU, scala col device |
| L3 fusione 270→100 | **−15%** | **−47%** ᵉ | **−26%** | domina dove il floor per-dispatch è alto |
| L4 multi-step N=4 | **−15%** | −3% | −4% | il round-trip pesa solo dove il token è breve |
| L5 kernel migliore | −6% | ~0% | −4% ᵐ | dove il kernel è già vicino alla banda non c'è margine |
| f16 compute path | 0 (feature assente su Chrome/Linux) | ~2× sulle GEMV | **+66% misurato** | — |

ᵉ = estrapolato: l'encode CPU non è misurato su quel device — per M4 assumo lo stesso
costo per dispatch della 4090, per S22 un fattore 3× (CPU mobile). **Assunzioni superate
da M1, eseguita: misure in §8** (M4 ~2.5 µs/dispatch, S22 ~67: la colonna S22 sottostima
L1/L3, la colonna M4 sovrastima L1) · ᵐ = limitato dalla banda: il q4
S22 è già a 16.4 di 21.7 GB/s misurati (**76% della banda**), l'M4 a 171 di 248 (69%). Il
"kernel dequant lontano dal suo tetto" del deep-dive **è un fatto solo della 4090** (86 di
436 GB/s, 20%) — e sulla 4090 i kernel sono il 23% del budget. Il margine reale di kernel
engineering è quindi ovunque piccolo: è la conclusione più controintuitiva di questa stima.

### Leve che non toccano i tok/s (e vanno valutate su un altro asse)

| Leva | Effetto stimato | Su cosa |
|---|---|---|
| Diradare la sync per-tensore nell'upload pesi (docket #4) | 272 sync × 1.8 ms ≈ **−0.49 s** su un load warm di 1.5-2.4 s → **−21…−33%** | load warm. Il contatore `onSubmittedWorkDone`=272 di questo run è esattamente la fase di load: la stima ora è misurata, non ipotizzata |
| Overlap fetch pesi ↔ compile pipeline (docket #3b) | fetch ≈ 1.3 s di 2.35 s nel run profilato (writeBuffer sale mentre dispatch=0) → **fino a −40%** del load warm | load warm |
| KV quantizzata int8/int4 | **0% sui tok/s** a contesto 469; 768 MiB → 192-384 MiB al tetto 32k | capienza contesto su memoria unificata |
| `requiredLimits` dai limiti dell'adapter | 0% | capienza (sblocca 2³²−4 di binding su Metal) |
| Contesto/sliding-window device-aware | 0% | capienza, con regressione funzionale visibile |
| Split tensori oversize + indirezione | 0% | abilita vocab/embedding grandi — nessun modello del set attuale ci arriva |
| Allocazione pagine KV lazy | 0% | picco di memoria al load |

Sono tutte "abilitanti", non acceleranti: vanno decise contro il target di modello del
motore (0.5B non le richiede; 3-8B sì), non contro i tok/s.

## 4. Il motore v0 — cosa entra e perché

**Entra (architettura, non ottimizzazione).** Sono decisioni che costano zero se prese al
giorno zero e costano un rewrite se prese dopo:

1. **Piano di esecuzione statico con bind group e scratch preallocati.** L1+L2. È la
   singola differenza strutturale con WebLLM: il suo costo per-dispatch non è un bug, è il
   prezzo dell'ABI a function-table di TVM, che non può sapere che i buffer sono invarianti.
2. **Grafo fuso ad ampiezza aggressiva**, e un kernel di sampling unico. L3. La catena di
   top-p a ~27 dispatch è il bersaglio più facile del progetto: a `temperature=0` (il caso
   del nostro bench) è un argmax, cioè 1 dispatch.
3. **Telemetria nativa** (`timestamp-query` richiesta al giorno zero, con batching dei
   dispatch per il quanto di ~100 µs di Chrome e fallback CPU). Costo ~zero, e senza di
   essa il residuo del 33% resta non attribuibile per sempre.
4. **`requiredLimits` negoziati dall'adapter + error scope + checksum** su ogni percorso di
   allocazione. Costo ~zero, e il progetto ha già sbagliato questa cosa due volte (WebLLM a
   1 GiB hardcoded; il nostro micro-bench a 128 MiB con celle garbage silenziose).
5. **Percorso di calcolo f16 dove `shader-f16` c'è.** L'unico incremento *misurato* del
   progetto (+66% decode, −68% TTFT, varianza TTFT da ±34% a ±0.4% sull'S22).
6. **Decode con dimensione di batch ≥ 1 come cittadino di prima classe.** Non serve a v0,
   ma è il prerequisito di multi-step e speculative (§6): metterlo dopo costa il rewrite
   dello scheduler.

**Entra con riserva.** *Multi-step decode* (L4): −15% sulla 4090 ma −3/−4% su M4 e S22, e
rompe lo streaming percepito. Va costruito come *politica* sopra il punto 6, non come
struttura, e acceso solo dove il round-trip pesa. La sua forma adattiva (docket
"sync coalescing") richiede prima la telemetria del punto 3: non è una leva a sé.

**Non entra in v0.** *Kernel engineering* (L5, split-K, layout Marlin): −6% sulla 4090,
~0% altrove, e costo alto. È l'ultima cosa da fare, non la prima — l'esatto opposto
dell'intuizione "un motore nuovo si giustifica coi kernel".

## 5. Alternative valutate e trade-off

**Perché non "WebLLM ma con le patch L1-L3".** Sarebbe la via a costo minore per il 1.44×.
Costo reale: le tre leve toccano l'ABI a function-table del wasm TVM, cioè il punto in cui
WebLLM *non è patchabile* senza forkare anche il codegen. Il fork dovrebbe essere
risincronizzato a ogni bump upstream. È il motivo per cui la stessa idea, nel deep-dive,
era instradata `engine-notes` e non `esperimento`.

**Perché non un backend WebNN.** Confermata la valutazione del deep-dive, ma per un motivo
diverso e più forte alla luce di questi numeri: WebNN sposterebbe l'intero grafo dentro il
driver, il che *azzererebbe* per costruzione i termini encode e floor (che qui sono il
38-75% del budget). È la sola alternativa che attacca il collo giusto. Resta fuori perché
l'op-set non copre la dequant INT4 group-wise e il supporto è parziale — ma va rivalutata,
non archiviata: se l'op-set QDQ maturasse, renderebbe questo motore obsoleto prima di
nascere. È il rischio strategico numero uno del progetto. *[VERIFY: compatibilità browser
WebNN mai verificata di persona — eredità del deep-dive.]*

**Perché il megakernel resta fuori.** WebGPU non espone sincronizzazione grid-wide né
kernel persistenti in modo portabile. Il valore ci sarebbe (annullerebbe N_disp), il
meccanismo no. Da riesaminare se e quando le primitive arrivano.

**Speculative decoding va promosso, non tenuto scartato.** Il deep-dive lo scartò con
l'argomento "il round-trip pesa ~6% su S22, il guadagno è concentrato su desktop". Quei
numeri ora dicono altro: il costo *fisso per token* (encode + floor + sync, tutto ciò che
non scala col lavoro utile) è il **77% sulla 4090** e il **~50% sull'S22**, e la verifica
speculativa produce K token pagando *una volta* quel costo fisso — perché il verify è un
GEMM a K righe su matrici che restano floor-bound. Con 2.5 token accettati per passo il
tetto teorico è −40…−50% di latenza, ben oltre qualunque leva di §3. Costo: un secondo
modello in memoria (proibitivo a 0.5B, ragionevole a 3B target + 0.5B draft) e una
pipeline di verifica non banale. **Verdetto: fuori da v0, ma il punto 6 di §4 va progettato
perché lo renda possibile senza rewrite.**

**Cosa resta scartato e perché.** Dequantizzare una volta e cachare i pesi f16 (a batch=1
quadruplica i byte letti: aggrava il termine che si vuole ridurre). Offload KV su RAM host
(su memoria unificata è lo stesso serbatoio fisico). Swap OPFS delle pagine KV (con
attenzione piena ogni step legge tutte le posizioni: è reinventare la sliding window con
un giro di I/O sopra). Tracing Dawn via `chrome://tracing` (non portabile, non
integrabile nei `results/`). Nessuno di questi cambia col venir meno del vincolo di fase.

## 6. Le misure che cambierebbero questo ranking

In ordine di rapporto informazione/costo. M1 e M2 sono le uniche che considero
propedeutiche a scrivere codice del motore.

- **M1 — `dispatch-profile` su M4 Pro e S22.** ~~Il tool è scritto e girato; servono le
  mani del PI~~ **ESEGUITA (2026-07-29)** via `prof.html` (pagina manuale, stessa
  procedura fase-1b) — esiti in §8.
- **M2 — contatore per call-site su `flushCommands`** (~10 righe sul bundle vendored).
  Dice quali dei 7 submit/token sono eliminabili con scratch preallocato, cioè quanto vale
  davvero L2 — oggi contabilizzata a −0.06 ms, quasi certamente una sottostima.
- **M3 — `timestamp-query` dentro il runtime** (fork strumentato del bundle, già in
  backlog engine-notes). È l'unica cosa che scioglie il residuo del 33%. Se il residuo
  fosse proporzionale a `N_disp`, L3 varrebbe il doppio di quanto stimato qui.
- **M4 — un run con `max_tokens` diverso** (es. 64 e 512). Verifica per differenza che i
  costi che qui tratto come fissi-per-token lo siano davvero. Costo: due run del bench.

## 7. Correzioni dovute ai doc già pubblicati

Il numero "~34 dispatch per token" è sbagliato di **8×** e compare in tre dei sei doc del
deep-dive, già mergiati su main e destinati alla pagina pubblica:

- `docs/deep-dive/compute-shader-dispatch.md` — §"Perché i numeri sono quelli" e
  bottleneck #1 ("~34 dispatch + 1 sync per token"; il submit per token è 7, non 1);
- `docs/deep-dive/dequant-kernels.md` — §"Perché i numeri sono quelli" (~0.14 ms/kernel);
- `docs/deep-dive/micro-bench-matmul.md` — §"Cosa dice sul gap del 4-6%" ("un token di
  decode ne lancia ~34").

La *conclusione* di quei passaggi regge (lavoro GPU utile ~2 ms su ~9 misurati, quindi
75-85% orchestrazione): con 270 dispatch a 5 µs di floor il conto torna a 2.15 ms invece
di 1-2 ms. È il numero intermedio a essere sbagliato, non la tesi. ~~Non ho modificato i
doc~~ **Corretti il 2026-07-29** (ruling PI: doc stale si corregge appena notato) — in
quattro doc, non tre: anche `engine-design-notes.md` citava "~34 + 1 sync".

## 8. M1 eseguita — misure cross-device (2026-07-29)

Run manuali via `prof.html` (S22: q4f16_1; M4: q4f32_1 e q4f16_1) + run 4090 di
cross-validazione della pagina contro il tool. File in `results/dispatch-profile/`.

| | 4090 (f32) | M4 Pro (f32 / f16) | S22 (f16) |
|---|---|---|---|
| dispatch totali | 275 456 | 275 456 / 275 456 | 275 456 |
| submit totali | 7 164 | 7 164 | 7 164 |
| dispatch/submit (finestra decode) | 38.4 | 38.4 | 38.4 |
| encode CPU µs/dispatch | 3.9 (tool 28/7: 8.6) | 2.7 / 2.5 | **67.4** |
| tok/s sotto patch | 112.8 | 92.5 / 98.1 | 6.9 |

Quattro fatti:

1. **`N_disp = 269/token` e 7 submit/token sono invarianti di device E di quant** —
   totali identici al byte in tutti e cinque i run. Il grafo di esecuzione è
   deterministico; la premessa strutturale di §2-3 è confermata.
2. **L'encode CPU per dispatch era la stima più sbagliata, in direzioni opposte.**
   M4 misurato ~2.5 µs (assunto = 4090: in realtà l'M4 encoda *più veloce*); S22
   misurato ~67 µs (assunto 3× la 4090 ≈ 25 µs: è ~10-25×). Solo encode CPU su S22:
   270 × 67 µs ≈ **18 ms/token** — è la voce dominante del suo budget, e le leve
   L1/L3 su S22 valgono *più* di quanto stimato in §3. **Le colonne ᵉ di §3 vanno
   ricalcolate**; i numeri grezzi bastano a `direction.md`, il ricalcolo fine può
   attendere il primo timestamp-query nel runtime (M3).
3. **Observer effect dichiarato**: i tok/s sotto patch non sostituiscono i bench.
   Su 4090/M4 il patch è neutro (112.8 vs ~107; 98.1 vs 98.3 baseline); su S22
   6.9 vs 11.7 baseline — sul device CPU-encode-bound le ~2200 chiamate
   `performance.now()`/token del patch pesano, e il run è partito a telefono già
   caldo (il 67 µs/dispatch può includere throttling: leggerlo come limite superiore).
4. **q4f32_1 su S22 non ha completato il run prof** ("empty timeline: no chunks
   received" — stream chiuso senza contenuto, nessuna eccezione engine). Non
   investigato oltre una volta riusciti su f16; coerente col pivot a f16 già fatto in
   fase-1b. Nota aperta, non bloccante: la baseline S22 del progetto è f16.
