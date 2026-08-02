# Spec — engine-fase-c3a (struttura: dal 4.68 al floor CPU)

Stato: **DRAFT, in attesa di ruling PI** (docket C3a item 6).
Contratto: `.harness/goals/engine-fase-c3a/GOAL.md` (v1 + emendamento 1).
Input di misura: `results/engine/bench-glm-4090-b12-attrib-2026-08-01.json`,
`results/engine/host-gpu-clocks-c3a-it1-2026-08-01.csv`, journal it.1.

---

## 1. Da dove si parte (misurato, non assunto)

Decode GLM-4.7-Flash, slab 12 GiB, p6 (461 token), nGen 64, macchina quiescente:
**4.678 tok/s = 215.0 ms/token**, scomposti così:

| voce | ms/token | come è stata ottenuta |
|---|---|---|
| `gpuBusy` | **78.2** | somma delle durate dei 94 compute pass (timestamp-query) |
| stallo residenza | **53.8** | contatori ExpertCache: pack 41.4 + read 4.8 + upload 7.8 |
| sync/CPU | **83.0** | residuo `wall − gpuBusy − stallo` |
| *(di cui readback vero)* | *7.6* | probe indipendente: mapAsync a GPU scarica 0.16 ms × 46 |
| encode CPU | 3.5 | segmento disgiunto misurato (identità chiusa allo 0.3%) |

Contesto host durante la run: clock SM **1746 MHz medi su 3105 di cap**,
utilizzo GPU **34.6%**, 61-66 °C.

**Il budget del gate.** 13.43 tok/s = **74.46 ms/token**. Con le leve 1 e 2 al
100% di efficacia si arriva a 98.2 ms/token = 10.18 tok/s: `gpuBusy` da solo
(78.2) eccede il budget. Da qui l'emendamento 1 e la quarta leva.

**Il budget per leva** che questa spec si dà (somma = 74.4 ms/token, gate preso
con margine zero — il margine deve venire dai clock):

| leva | da | a | risparmio |
|---|---|---|---|
| 1. repack all'import | pack 41.4 | ~0 | −41.4 |
| 2. sync router | 83.0 | 7.6 (floor) | −75.4 |
| 4. granularità dispatch | `gpuBusy` 78.2 | **≤ 54.5** | −23.7 |
| residuo residenza (read+upload) | 12.4 | 12.4 | — |

**Nota di onestà**: parte della riduzione di `gpuBusy` arriverà "gratis" dai
clock che salgono quando spariscono le bolle (34.6% di utilizzo è la firma).
La spec impone di **attribuire separatamente** le due cause nel report della
fase 4b — altrimenti si attribuisce al lavoro sui kernel un guadagno che è del
governor.

---

## 2. Leva 1 — repack all'import (fase 3)

**Problema.** `ExpertCache.ensure` chiama `packExpertSlab` a ogni miss: legge i
byte GGUF grezzi dei tre tensori e li ri-impacchetta nel layout `[qs | scales]`
del motore. Costa **9.5 ms per miss × 4.47 miss/token = 41.4 ms/token**, sul
path caldo, su CPU.

**Design.** Il repack si fa **una volta sola**, all'import, e produce un secondo
file OPFS in layout slab. `ensure` diventa `read → writeBuffer`, senza CPU.

- **File**: `GLM-4.7-Flash-Q4_0.slabs.bin` accanto al GGUF in OPFS.
- **Contenuto**: i 2.944 slab expert consecutivi, ognuno nel layout di
  `SLAB_DOWN_Q4_0` / `SLAB_DOWN_Q4_1` (invariati: il repack non cambia i byte,
  cambia solo QUANDO avviene). Indicizzati per `(layer, expert)` con l'ordine
  già usato da `expertKey`.
- **Taglia**: 2.688 × 5.308.416 + 256 × 5.505.024 = **15.68 GB**. Con il GGUF
  restano 32.89 GB in OPFS; verificato 1.2 TB liberi sul volume del profilo.
  **Il GGUF resta**: i tensori non-expert e gli harness di conformance/routing
  lo leggono ancora.
- **Header** (primo blocco da 4 KiB, allineato): magic `BLABSLAB`, versione del
  layout (u32), SHA-256 del GGUF sorgente, conteggi e taglie delle due
  size-class, offset della prima slab. **Invalidazione**: se magic, versione di
  layout o SHA non combaciano, il file si rigenera — mai si legge un file di
  cui non si è certi della provenienza.
- **Costo one-shot**: ~28 s di CPU per il repack (2.944 × 9.5 ms) più la
  lettura del GGUF e la scrittura di 15.68 GB. Accettabile: è un costo di
  import, non di inferenza. Va **misurato e riportato** (entra nel TTFT a
  freddo, che è metrica di C3b).
- **Fallimento parziale**: la generazione scrive su un file temporaneo e
  rinomina alla fine, così un'interruzione non lascia un file valido a metà.

**Done-when della fase 3** (da PHASES): `pack CPU < 1.0 ms/token` nel path
caldo, test verdi su produzione dell'artefatto e su invalidazione, correttezza
invariata (argmax ≡ cpuref-f64 256/256).

### 2.1 ESITO MISURATO (it.7) — il budget della leva va corretto

Implementata e misurata a macchina quiescente, due run
(`bench-glm-4090-b12-repack-2026-08-02.json` e `…-warm-…`):

| ms/token | pre-repack | post-repack |
|---|---|---|
| pack CPU | 42.5 | **0.0** ✅ |
| read OPFS | 5.8 | **18.4** |
| upload | 8.1 | 16.2 |
| **stallo residenza** | **56.1** | **34.5** |
| decode tok/s | 4.640 | **4.912** (+5.9%) |

**Il budget di §1 era sbagliato: la leva 1 non vale −41.4 ms/token ma −21.6.**
Il pack sparisce davvero, ma read+upload crescono di 20.7 e si mangiano metà
del beneficio.

**Causa, misurata e non congetturata** (tre ipotesi, due smentite):
- cache fredda al primo giro: SMENTITA (2ª run 18.4 vs 19.2);
- file frammentato: SMENTITA e nel verso opposto — `filefrag` dà **2 extent**
  per lo slab contro **263** per il GGUF;
- throughput del file: SMENTITA — misurato fuori dal motore, entrambi i file
  danno ~1.9-2.7 GB/s a freddo e ~10 GB/s a caldo.

Il throughput EFFETTIVO per miss dice il resto: **4.08 GB/s prima** (fra il
freddo e il caldo del GGUF ⇒ letture prevalentemente in page cache), **1.29
GB/s dopo** (sotto persino il freddo ⇒ quasi tutte fuori cache). Prima gli
expert si leggevano dallo stesso file che il motore tocca comunque a ogni build
(i tensori non-expert), quindi il GGUF restava caldo; ora ci sono **due** file —
33 GB contro 31 GB di RAM e 14 di page cache — e nessuno resta caldo.

**Il repack non elimina lavoro: scambia 41.4 ms di CPU con 12.6 ms di I/O.**

**Conseguenza sul piano delle leve**: i 18.4 ms/token di lettura sono tempo in
cui la GPU è ferma, quindi sono materia della **leva 2** (sovrapposizione via
prefetch, ammessa in perimetro dall'emendamento 2a). Repack da solo: −21.6.
Repack + sovrapposizione: fino ai −34.5 dello stallo intero. Il budget
complessivo di §1 regge solo se la leva 2 assorbe anche l'I/O migrato.

**Correttezza**: la conformance full-corpus costa ~5 h (run C2: 4.9 h) e NON è
stata eseguita. La catena di verifica usata al suo posto: (a) test unitario che
carica gli stessi expert per entrambi i percorsi e confronta **byte per byte**
ciò che finisce in VRAM; (b) test che confronta gli slab **sul disco** con
`packExpertSlab` sui byte grezzi del GGUF (7 campioni, entrambe le size-class,
estremi e centro) — byte-identici. Se i byte in VRAM sono identici, il calcolo
lo è. La conformance resta da eseguire prima della chiusura del goal (fase 6),
dove serve comunque per il gate.

**Decisione chiesta al ruling**: nessuna sul design — si segnalano il consumo
disco (+15.68 GB) e la correzione del budget.

---

## 3. Leva 2 — i 46 sync del router (fase 4)

**Problema.** La selezione top-4 vive su CPU perché decide **quali slab
bindare**, e in WebGPU i bind group sono oggetti CPU. Quindi ogni layer MoE
spezza il token in un submit a sé: 46 readback + 47 submit per token.

### 3.0 Il meccanismo vero (corretto dopo la ricerca di it.3)

La prima stesura di questa spec attribuiva gli ~75 ms alla "latenza dei
submit". **È sbagliato**: il costo API di una submit è ~13 µs (Dawn/Vulkan,
misura pubblicata), quindi 47 submit valgono **~0.6 ms/token**, non 75.

Il meccanismo dominante è che **`mapAsync` è una barriera**: da spec non si
completa finché *tutto* il lavoro GPU accodato prima non è finito. Ogni layer è
quindi un **drain completo della coda**, e la GPU non ha mai lavoro accodato
mentre la CPU decide. La nostra stessa misura lo conferma per via aritmetica:
`gpuBusy`/wall = 78.2/215 = **36.4%**, contro un'utilizzazione GPU campionata
del **34.6%** — coincidono. Non c'è overlap da recuperare: c'è serializzazione
da eliminare. Il probe a 0.16 ms misura il round-trip a coda **vuota**; sotto
carico ogni round-trip include il drain (83/46 = 1.8 ms, ~11× il probe).

**Conseguenza di design**: il bersaglio non è né "leggere più in fretta" né
"fare meno submit" — è **smettere di drenare la coda 46 volte per token**.

### 3.0-bis Il pattern che risolve, già provato da tre implementazioni

Nessuno, in nessun motore, cambia bind group in funzione dell'expert
selezionato. Il pattern condiviso è: **pesi di TUTTI gli expert in un tensore
packed** (asse 0 = expert id), **binding FISSO**, indici top-k in un **buffer
GPU**, e l'expert diventa un **offset aritmetico dentro lo shader**.

- **ONNX Runtime WebGPU**, PR #27998 (mergiata 2026-04-10), decode MoE a 1 token
  da 17 a 5 dispatch: `let actual_weight_idx = weight_index_indirect[a_global];`
  poi `let b_base_offset = actual_weight_idx * uniforms.K_of_b * uniforms.N;`
  Il gate produce gli indici come tensore GPU e il codice **prosegue senza
  readback**. Guadagni riportati: +21% su Meteor Lake, +14% su RTX 5060 Ti.
- **llama.cpp `ggml-webgpu`**: `mul_mat_id_vec.wgsl` fa
  `let expert = ids[...]` → `src0_batch_offset = ... + own_expert * stride_02`.
  Submit batchate a 64. `dispatchWorkgroupsIndirect` non compare mai.
- **MLC/TVM**: `moe_matmul.gemv` con `indptr` come tensore GPU (ma WebLLM non
  spedisce modelli MoE: nessuna prova d'esercizio su WebGPU).

Nota importante: **nessuno usa `dispatchWorkgroupsIndirect`** per questo. Con k
costante il caso peggiore è il caso esatto, quindi si dispatcha e basta.

**API che NON arriveranno in soccorso** (verificate sulla spec e su Dawn):
`binding_array` per i buffer è dichiarato non implementato in Dawn
(`BindGroupLayoutInternal.cpp`: *"bindingArraySize > 1 for a buffer binding is
not implemented yet"*); il bindless sperimentale
(`chromium_experimental_resource_table`) copre **solo texture e sampler**; i
dynamic offset sono valori CPU passati a `setBindGroup` (spec §14.1), quindi
richiedono comunque di conoscere l'indice su CPU.

### 3.0-ter L'ostacolo che nessun pattern di binding risolve: la residenza

Le tre implementazioni di riferimento assumono **tutti gli expert residenti**.
Noi no: a slab 12 GiB sono residenti **2419 slot su 2944 = 82.2%**, con 4.47
miss/token. E **solo la CPU può leggere da OPFS**: finché esiste un miss, il
readback del router è necessario per sapere cosa caricare.

**Il conto della residenza totale su questo device** (calcolato, non stimato):

| voce | GiB |
|---|---|
| parco expert COMPLETO (2688 q4_0 + 256 q4_1) | 14.60 |
| pesi non-expert (attention 47 layer, shexp 46, denso, head Q6_K) | 1.26 |
| KV a ctx 525 | 0.05 |
| **totale richiesto** | **15.91** |
| VRAM usabile (16376 MiB − 763 di desktop) | 15.25 |
| **deficit** | **0.67** |

**Mancano 0.67 GiB, cioè 135 expert su 2944 — il 4.6% del parco.** A ctx 4096
il KV sale a 0.41 GiB e il deficit diventa 1.03 GiB.

Questo trasforma la leva 2 in una **decisione sulla funzione obiettivo**, non in
un problema di implementazione: il progetto ottimizza "massima intelligenza sotto
vincolo di rate", con due assi dichiarati (capienza a parità di spazio, velocità
a parità di modello). Qui i due assi si toccano: **spendere ~0.67 GiB di qualità
compra l'eliminazione del drain**. Vie possibili, da valutare con una eval di
perdita:
- quant più aggressiva sul 5-10% di expert più freddi (la matrice usage di C1 li
  identifica), lasciando Q4_0 sul resto — quant asimmetrica, lineage ds4;
- KV quantizzato, che libera il deficit a ctx lunghi;
- head Q6_K → Q5_K (−40 MB, non basta da solo).

**Se non si paga quel prezzo**, il drain resta e la leva 2 si riduce a
**sovrapporre il lavoro CPU a quello GPU** tramite prefetch (§3.2 bis).

**RULING RICHIESTO** su questo punto: docket item 8.

### 3.1 Limiti WebGPU — MISURATI (probe eseguito, [VERIFY] sciolto)

`scripts/webgpu-limits.mjs`, artefatto
`results/engine/webgpu-limits-4090laptop-2026-08-02.json`:

| limite | adapter | device che creiamo | nota |
|---|---|---|---|
| `maxStorageBufferBindingSize` | 2 147 483 644 | 2 147 483 644 | **tetto duro NVIDIA**: Dawn lo clampa a 2 GiB−4 per un bug driver su `OpArrayLength` |
| `maxBufferSize` | 4 294 967 292 | 4 294 967 292 | il codice lo clampa a 2 GiB **senza motivo** |
| `maxStorageBuffersPerShaderStage` | **16** | **8** | mai richiesto ⇒ default di spec |
| `maxComputeInvocationsPerWorkgroup` | **1024** | **256** | mai richiesto ⇒ default di spec |
| `maxComputeWorkgroupStorageSize` | 49 152 | 49 152 | il codice ne chiede 32 768 |

**Tre capacità lasciate sul tavolo** perché non richieste in `requiredLimits`:
un limite dell'adapter è una promessa, il device riceve il default se non lo si
chiede. Ruling PI 2026-08-02: si negoziano **subito**, in fase 3.

Conseguenze dirette:
- un binding storage copre **404 slab** q4_0 ⇒ servono **8 binding** per il parco
  completo, **6** per il residente attuale (2419 slot);
- con `maxStorageBuffersPerShaderStage` a 16 (negoziato) i binding ci sono;
- con `maxBufferSize` a 4 GiB i buffer di classe passano da 7 a **4**;
- con 1024 invocazioni/workgroup il GEMV può cambiare forma (oggi
  `workgroup_size(64)` con riduzione in shared memory) — **è materia della
  leva 4**, ed è la scoperta più promettente del probe;
- **`chromium-experimental-subgroup-matrix` è ESPOSTA** su questo adapter, con
  `subgroups` e `subgroup-size-control`: direction §8 rischio 1 («subgroup-matrix
  assente nei browser») è **stale**. Attenzione al perimetro: la vediamo perché
  lanciamo Chrome con `--enable-unsafe-webgpu`, quindi vale per esplorare il
  ceiling del motore, **non** per un confronto pubblico su browser stock.
- **`chromium-experimental-timestamp-query-inside-passes`** è esposta: darebbe
  il timing **per dispatch** invece che per pass, che è la granularità che serve
  alla leva 4.

### 3.2 Vie, con il criterio di scelta esplicitato

- **(A) Selezione su GPU con parco bindato per size-class.** Il router scrive i
  4 id in un buffer storage; le catene expert leggono i pesi da un binding
  grande indicizzato per offset calcolato **sulla GPU**. Elimina readback *e*
  submit split. Praticabile solo se il parco residente sta in **poche**
  bindings; con slot da 5.3 MB, un binding da 2 GiB ne copre ~394, quindi
  servono ~7 binding per 2.419 slot. Poiché WGSL non indicizza tra binding, la
  variante realizzabile è **dispatchare la catena expert una volta per
  binding** (7×4 = 28 dispatch, di cui 24 escono subito con un test sull'id):
  legale, ma va misurata — 28 dispatch quasi-vuoti costano.
- **(B) Bind del parco per-layer.** I 64 expert di UN layer sono 340 MB: stanno
  in un binding solo. Ma richiede residenza per-layer contigua, cioè tutti e 64
  gli expert di ogni layer residenti — 15.68 GB, oltre la VRAM. Praticabile solo
  a layer parziale, quindi non elimina il caso "expert non residente".
- **(C) Pipelining a profondità >1.** Nel decode **non è applicabile fra token**
  (il token t+1 dipende da t) né fra layer (l+1 dipende da l). Resta applicabile
  al **prefill** (leva 3) e in regime speculativo (fase D). **Va scartata come
  soluzione al sync del decode**, e questa spec lo dichiara per chiudere
  l'opzione: era elencata fra le vie note nel contratto, ma la dipendenza
  sequenziale la esclude.

### 3.2-bis La via scelta: packed arena + offset in shader + prefetch overlap

Alla luce di §3.0 / §3.0-bis / §3.0-ter, e del ruling PI 2026-08-02 che ammette
il **prefetch LOOKA nel perimetro C3a**, il design è in due strati.

**Strato 1 — togliere il binding dal percorso decisionale** (pattern ORT/ggml).
I 6-8 buffer di classe restano, ma diventano **binding fissi** del bind group
delle catene expert; lo slot dell'expert diventa un **offset aritmetico**
calcolato nello shader a partire da un id letto da un buffer GPU scritto dal
router. Serve: negoziare `maxStorageBuffersPerShaderStage` a 16, sostituire
`layout: "auto"` con un bind group layout esplicito, aggiungere ai kernel GEMV
un base-offset (oggi `qs`/`scales` assumono offset 0 — `wgsl.ts:93-94, 111`), e
collassare i 4 buffer `wExp[k]` da 4 byte in uno solo indicizzato.

**Strato 2 — togliere la CPU dal percorso critico** (prefetch). Il drain resta
finché la residenza non è totale (§3.0-ter), ma oggi CPU e GPU **non si
sovrappongono mai**: per layer, 1.7 ms di GPU con CPU ferma, 1.5 ms di drain,
1.18 ms di `ensure` con GPU ferma. Predire gli expert del layer l+1 dall'hidden
del layer l (LOOKA, **recall 92% @K=8** misurato in C1) permette di far girare
le letture OPFS e gli upload **durante** il lavoro GPU del layer l: sono
~54 ms/token di lavoro CPU che scompaiono dietro la GPU.

**Questo ribalta la valutazione del docket C2 item 8**, che classificava il
prefetch come leva 4 di valore residuo piccolo (≤56 ms/token). Quella stima
considerava solo lo *stallo* come costo di un miss; non vedeva che il prefetch
abilita la **sovrapposizione**, che attacca il termine sync — molto più grosso.

**Criterio di scelta** (dichiarato come chiede il DONE WHEN §1): si sceglie la
via che **minimizza il numero di drain della coda** a parità di correttezza del
routing; a parità di drain, quella che massimizza la sovrapposizione CPU/GPU.
Il conteggio dei submit è un proxy, non l'obiettivo.

**Nota di rischio permanente** (osservazione della ricerca): finché esiste un
readback per layer, l'intera famiglia di ottimizzazioni **record/replay** resta
preclusa per costruzione — il graph capture di ORT richiede esplicitamente
nessun kernel su CPU. È un costo composto del drain, non solo di latenza.

**Done-when della fase 4**: ≤ 2 sync/token (o la soglia ammortizzata dichiarata
col suo parametro), routing conformance non peggiore del riferimento C2, **più
la ri-misura obbligatoria di `gpuBusy` e dei clock** che dimensiona la 4b.

---

## 4. Leva 4 — granularità dei dispatch (fase 4b, emendamento 1)

**Il margine.** Per token si leggono **2.22 GB** di pesi (47 layer di attention
+ 46 × (4 expert + shared) + head Q6_K); la banda del device è **576 GB/s**
(9001 MHz × 2 × 256 bit, letta dal device) ⇒ floor memory-bound **3.85
ms/token**. `gpuBusy` misurato è **20× sopra**: 1.816 dispatch a 43 µs medi.
Il tempo GPU non è un muro fisico.

**Prima si misura DOVE, poi si fonde.** La telemetria di fase 1 somma le durate
dei 94 pass in un unico numero. Il primo task della 4b è **spezzare quella somma
per categoria** (attention, router, shexp, catene expert, head) usando i
timestamp già disponibili: cambia l'aggregazione, non la strumentazione. Fondere
kernel senza sapere quale categoria pesa è la ricetta per ottimizzare il 5%.

**Ipotesi da confermare o smentire con quel dato** (in ordine di sospetto):
1. **Dispatch minuscoli serializzati dalle dipendenze RW**: `rmsnorm` gira su
   **1 workgroup**, `silu`/`add` su 24-32; dentro un pass ogni dispatch legge
   ciò che il precedente scrive, quindi il driver li serializza. Sono decine per
   layer con la GPU quasi ferma.
2. **Le 4 catene expert sono 16 dispatch/layer** che fanno la stessa forma su
   pesi diversi: candidate naturali a un dispatch unico su 4 expert (una
   dimensione di griglia in più), che è anche il presupposto di (A) in §3.
3. **Fusione delle code**: `gate/up → silu → down` è una catena a tre stadi con
   attivazioni intermedie in VRAM; fonderla riduce dispatch e traffico.

**Done-when della fase 4b**: `gpuBusy ≤ 54.5 ms/token` (soglia derivata:
74.46 − 12.4 di stallo residuo − 7.6 di floor sync), con dispatch/token e clock
SM medio riportati **e attribuiti separatamente**, correttezza invariata.

---

## 5. Leva 3 — prefill batched M>1 (fase 5)

**Problema.** Il motore non ha un percorso M>1: il prefill esegue 461 forward
sequenziali ⇒ 5.235 tok/s e **TTFT 88.06 s** contro un budget UX di 4 s. Non è
una prestazione 12× peggiore del floor, è una **capacità mancante** (precisato
in docket C2 item 6 e ratificato dal ruling che ha fissato il TTFT).

**Design.** Forward su M token insieme: le GEMV diventano GEMM su M righe;
l'attention usa il percorso chunked già esistente (pattern `prefillplan` del
path Qwen); il MoE è il punto nuovo — **token diversi selezionano expert
diversi**, quindi la catena expert va eseguita per unione degli expert
selezionati nel chunk, con maschera per token. La residenza deve reggere
l'unione (più miss per chunk, ma ammortizzati su M token).

**Condizione di identità** (il gate di correttezza della fase): i logits di un
prompt processato con M>1 devono coincidere con quelli a M=1 entro la tolleranza
fissata qui: **argmax identico su tutte le posizioni** e differenza massima sui
logit entro il rumore f32 già accettato in C2 (`maxAbsDeltaLogit` resta metrica
di scala, mai gate — landmine C2). Test in `npm test`.

**M iniziale**: 16 `[ASSUMED]`, poi taratura sul TTFT. Con M=16 il TTFT
scenderebbe, a parità di tutto il resto, verso ~5-6 s — ancora sopra i 4 s, il
che dice che il prefill ha bisogno anche delle leve 1/2/4, non solo del
batching.

---

## 6. Protocollo di bench, TTFT, telemetria

**Invariato rispetto a C2, per parità**: prompt p6 (461 token), nGen 64, 3
repliche + warmup scartato, mediana come headline, slab 12 GiB, macchina
quiescente con stato host dichiarato nel report. Nessun confronto fuori parità
(lezione B2).

**Repliche di attribuzione separate**: la telemetria liv.1+2 accende
`performance.now()` e i `timestampWrites`; il loro wall **non entra nei gate**.
Headline sempre da repliche a telemetria spenta.

**TTFT — definizione operativa** (come chiede il DONE WHEN §1): tempo dal
**primo forward di prefill** al momento in cui il **primo token generato è
disponibile** (argmax dei logits dell'ultima posizione di prompt), **modello già
residente**. Il TTFT a freddo — che include load, import e popolamento della
slab — è metrica di **C3b** (instant-on) e non si confonde con questo.

**Obbligo di riporto** (contratto, ruling PI 2026-08-01): ogni report contiene
i gap da 30 tok/s, da 60 (thinking) e dal budget TTFT di 4 s. **L'assenza è un
FAIL di checklist**, non un dettaglio di forma.

---

## 7. Rischi dichiarati

1. **Il gate resta irraggiungibile anche con quattro leve.** La forbice misurata
   è 10.2-15.6 tok/s e il gate 13.43 ci sta dentro: l'esito dipende da quanto i
   clock salgono. La clausola di fallback è stata **rimandata dal PI a fine fase
   4** (docket item 2), quando la ri-misura l'avrà sciolta.
2. **§3.1 può invalidare la via (A).** Se i limiti reali di binding non
   consentono di coprire il parco, la leva 2 va ripensata e il ruling torna al
   PI. È il motivo per cui il probe viene prima del codice.
3. **Il repack aggiunge 15.68 GB in OPFS.** Verificato spazio, ma il costo di
   import cresce e si somma al TTFT a freddo di C3b.
4. **La fase 4b è kernel engineering**, che direction §2 mette per ultima nel
   ranking delle leve. Entra qui solo perché la misura dimostra che senza non si
   prende il gate — non perché sia diventata prioritaria in generale.
5. **Correttezza sotto rifattorizzazione pesante**: fondere kernel e cambiare il
   layout dei pesi tocca i path che C2 ha validato bit per bit. Il gate doppio
   (argmax ≡ cpuref-f64 sul campione ratificato + top-1 vs golden ≥ 98.83%) va
   rieseguito a **ogni** fase, non solo alla 6.

---

## 8. Decisioni chieste al ruling

1. **§2** — repack come secondo file OPFS (+15.68 GB), GGUF mantenuto: ok?
2. **§3** — criterio di scelta "minimizzare i submit, non i readback" e
   raccomandazione (A) condizionata al probe: ok? Si ratifica anche lo
   **scarto esplicito del pipelining** (via C) come soluzione al sync del
   decode?
3. **§4** — "prima misurare dove va `gpuBusy`, poi fondere", con l'obbligo di
   attribuire separatamente kernel e clock: ok?
4. **§5** — M=16 iniziale `[ASSUMED]` e condizione di identità "argmax identico
   su tutte le posizioni": ok?
5. Si conferma che il **gate resta 13.43 / 56.58** e che il TTFT ≤4 s resta
   riportato-non-gateato anche in questa fase?
