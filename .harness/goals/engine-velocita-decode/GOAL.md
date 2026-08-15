GOAL: engine-velocita-decode — **il decode del motore diventa più veloce su
tutte le famiglie**, con tre leve GLOBALI misurate su almeno due famiglie
ciascuna: la forma a gather universale per gli expert MoE, il raggruppamento
delle richieste di I/O, e l'unità di riparazione del decode ottimistico. Il caso
più duro è il 35B e porta la barra: **≥ 30 tok/s a caldo**. Nessun modello
regredisce.

<!-- RI-SCOPATO il 2026-08-15 su ruling del PI, dopo che la misura ha demolito
     il contratto precedente. Il goal si chiamava `engine-35b-residency` e la
     sua tesi era che il collo del 35B fosse la residency. Ruling:

     «Tieni 30, allarga il goal alla forma gather (ti avevo detto che avremmo
     dovuto farlo anche nel goal dei k quant, ma non mi hai ascoltato mettendolo
     fuori scope). Aggiungi anche il raggruppamento delle richieste http se può
     dare un boost globale al motore. Non so quante volte ti ho ripetuto di non
     overingegnerizzare e applicare il principio di Pareto. Leve globali,
     massimo risultato, non piccolezze specifiche per il modello A o il quant B
     e robe simili. FACCIAMO IL MOTORE PIÙ VELOCE POSSIBILE. PUNTO»

     L'ERRORE CHE HA CORRETTO, per intero e senza attenuanti. Nel goal
     `engine-kquant` la forma a gather K-quant era stata messa fuori scope con
     la motivazione «varrebbe il 16% del token», e il PI aveva già detto allora
     che andava fatta. Il 16% era calcolato su una stima del token pulito
     (21,1 ms) ottenuta per sottrazione e mai misurata. **Misurato: il token
     pulito costa 43,585 ms e il termine che quella leva aggredisce vale il
     93,5%.** Ho escluso il termine dominante con un numero derivato — la
     stessa forma d'errore di `readMs`, che sembrava dire «I/O gratuito» perché
     misurava una finestra dove l'I/O non passa. Due istanze nello stesso goal.

     LA REGOLA CHE NE ESCE, ed è meccanica perché sulle intenzioni ho già
     fallito: **una leva vale solo se è misurata su ≥ 2 famiglie di modelli.**
     Sta nei done-when, non nelle premesse. Una leva che serve un modello solo o
     un formato solo non è candidata a riga di questo goal.
-->

<!-- CONTRATTO (chartered 2026-08-15 come `engine-35b-residency`, ri-scopato lo
     stesso giorno). La tesi v0 diceva «il lavoro NON è nei kernel, è nella
     fetch degli expert»: la prima metà regge ed è una riga di questo goal, la
     seconda è falsa — sul token pulito il 93,5% è il pass. Tutto ciò che segue
     è misurato e il ri-scopo non lo tocca: cambia l'ordine delle leve, non i
     fatti.

<!-- CONTRATTO v1 (chartered 2026-08-15, PI in chat: «Con il 4 e il 9B direi che
     siamo apposto. Con il 35 diamo ancora lenti, ma a questo punto credo sia un
     limite fisico?» → dopo la lettura dei contatori: «mi torna. andiamo».)

     PROVENIENZA. Il goal non nasce da un'intuizione: nasce dai contatori di un
     turno di chat vero, `results/chat/chat-35b-2026-08-15T16-25-12.json`
     (Qwen3.6-35B-A3B-UD-Q4_K_S, select optimistic, 1058 token generati,
     8,34 tok/s, 26,0 s di TTFT su 34 token di prompt). Il file è dichiarato
     NON-riferimento (nessun warm-up scartato, nessuna replica): i suoi tok/s
     non valgono come misura, i suoi CONTATORI sì, perché sono somme di eventi
     e non medie di tempi rumorosi.

     LA SCOMPOSIZIONE, dal JSON (`final.perf`, `final.moe`):
       tokenMs        144.744 ms   totale di parete, 1092 forward
       tailCpuMs      121.691 ms   = 84,07% del token
         repairMs      76.160 ms
           packMs        7.590 ms   packExpertSlab            (residency.ts:839)
           uploadMs      6.272 ms   queue.writeBuffer         (residency.ts:840)
           readMs            3,8 ms  ← NON misura l'I/O: vedi sotto
           NON NOMINATO 62.294 ms   = 43,0% dell'intero turno
         replay+conta  45.531 ms   per differenza, mai scomposto
       tokenMs − tailCpuMs = 23.053 ms / 1092 forward = **21,1 ms/token**

     **[RITRATTATO il 2026-08-15, poche ore dopo averlo scritto — v. C0-4 in
     `PHASES.md`.]** Il contratto diceva: «IL NUMERO CHE CHARTER-A IL GOAL:
     21,1 ms, cioè 47,4 tok/s, sopra la soglia». **Quella sottrazione non regge
     due controlli**, e la ritratto prima che diventi un ingresso di qualcun
     altro:
       (a) `readbackMs`/pass vale 37,4 ms (89.924 / 2.407 submit) — più
           dell'INTERA porzione non-tail del token (21,8 ms/decode step). Un
           pass non può contenere un'attesa più lunga di sé;
       (b) `results/engine/q35-vramplan-35b-it35.json`, pass `optimistic-warm`
           con **0 miss, 0 dirty, 0 replay** — cioè un token pulito MISURATO,
           non dedotto — costa **43,74 ms**, di cui `readbackWait` **40,98** e
           `tailCpuMs` **0,194**. Il 2026-08-11, su codice più vecchio.
     Il token pulito va MISURATO (riga 1), non ottenuto per differenza da
     aggregati che mescolano pass iniziale e replay.

     **MISURATO in it.2, e la risposta è quella sfavorevole.**
     `results/engine/q35-optimistic-35b-cleantoken-2026-08-15.json`, pass
     `optimistic-warm`, **0 miss · 0 dirty · 0 replay**:

       tokenMs         43,585 ms/token   = 22,9 tok/s   (it.35: 43,736)
       readbackWaitMs  40,753  = 93,5% del token pulito
       encodeMs         1,188 · argmaxMs 0,399 · tailCpuMs 0,200
       submitsPerToken 1 · readbacksPerToken 1

     Coincide col 2026-08-11 entro lo 0,7% — e conferma di passaggio che i
     kernel di `engine-kquant` non hanno mosso il 35B di un ms, come previsto
     (0% di copertura, consegna §2).

     **QUINDI: LE RIGHE 2 E 3 NON POSSONO CHIUDERE QUESTO GOAL.** Togliere il
     100% della tassa di residency lascia il 35B a **22,9 tok/s, sotto la barra
     dei 30**. Non è una stima: è il token pulito misurato due volte a quattro
     giorni di distanza, su due alberi diversi.

     **IL TERMINE CHE DECIDE IL GOAL È CAMBIATO, ed è quello che il contratto ha
     dichiarato fuori scope.** Sul token pulito `readbackWait` è il 93,5%, con
     UN submit e UN readback: non è overhead, è la GPU che lavora 40,75 ms sui
     320 GEMV expert del token. La leva su quel termine è la **forma a gather
     K-quant** della consegna §4.2-4.4 (~2,6× di dispatch), che questo contratto
     aveva messo fuori scope perché «varrebbe il 16% del token». Quel giudizio
     era basato sulla stima ritrattata. **Serve un ruling del PI sulla barra o
     sullo scope** — è funzione obiettivo, non meccanismo: docket item 3.

     RESTA VERO, e non dipende dalla sottrazione ritrattata: il pavimento di
     banda su questa scheda è ~2,9 ms/token (~1,66 GB di pesi attivi, 4090
     mobile ~576 GB/s — *conto mio dagli header, non una misura*), contro i ~120
     ms misurati. La lentezza del 35B **non è un limite fisico** — questo
     reperto è intatto. Ciò che è caduto è la stima di QUANTO margine ci sia
     sotto la tassa di residency, non l'esistenza della tassa.

     PERCHÉ `readMs` È ~0 E NON SIGNIFICA "I/O GRATUITO". È scritto nel sorgente
     dal 2026-08-?? e nessuno l'aveva ancora incrociato con un turno lungo:
     `residency.ts:250-257` avverte che `readMs` misura `readRaw` DENTRO
     `ensure`, che è sincrona, mentre il 35B fa l'I/O PRIMA di `ensure` — in
     `await Promise.all(misses.map(readExpert))`, `q35gpumodel.ts:1994`. Quella
     await è fuori da ogni finestra misurata. È il 43% del turno.

     COSA C'È DENTRO QUELLA AWAIT, letto per intero:
       q35gpumodel.ts:1994  await Promise.all(misses.map(readExpert))
       q35expertstore.ts:103-108  readExpert = 3 readRange (gate, up, down)
       chat.worker.ts:62-68  readRange = fetch(URL_GGUF, {Range}) + arrayBuffer()
     Cioè: **~31.000 richieste HTTP Range al dev server in un solo turno**
     (10.417 miss × 3 tensori), ciascuna awaited in serie col decode, con la GPU
     ferma. 62.294 ms / 10.417 miss = **5,98 ms per miss** per 1,77 MB di slab =
     ~300 MB/s, su una macchina che tiene 20 GB di GGUF nella page cache
     (`free -g`: 20 in buff/cache). Non è il disco. È il path.

     PERCHÉ IL REPLAY NON SI SALVA MIGLIORANDO LA CACHE. Il cache hit rate è già
     ottimo: 340.456 hit su 350.873 richieste = **miss 2,97%**, con 11,17 GB di
     budget d'arena (`vramPlan.expertBudgetBytes`) contro 18,33 GB di slab di
     device = 61% residente. Ma un token fa **320 selezioni** (40 layer × top-8)
     e ne basta UNA a sporcarlo:
       dirtyTokens   980 / 1092 = **89,7%** dei token sporchi
       replays       1.315       = 1,34 round di repair per token sporco
       replayLayers  37.977      = +34,8 layer rigiocati per token su 40
                                 → lavoro GPU **1,87×** (81.657 layer su 43.680)
     Per portare i token sporchi sotto il 20% servirebbe un miss rate dello
     **0,07%**: 43× meglio di oggi (1 − 0,8^(1/320) = 6,97e-4). *Conto mio dai
     contatori.* Nessuna messa a punto della LRU ci arriva — e va detto anche il
     rovescio, che è la buona notizia: il modello a miss INDIPENDENTI predirebbe
     il 99,99% di token sporchi (0,9703^320 = 6,5e-5 di probabilità di token
     pulito) mentre il misurato è 89,7%. Una correlazione da sfruttare esiste
     (skew del routing + LRU), ma non vale 43×. **Va cambiata l'unità di
     riparazione, non il tasso di hit.**

     CIÒ CHE QUESTO GOAL EREDITA E NON RIFÀ. La scheda
     `docs/engine/kquant-consegna-35b-2026-08-15.md` (423 righe) ha già fatto il
     lavoro di lettura: §4.5 nomina la residency come il termine che decide il
     goal, §5 dichiara che manca la baseline fresca, §6 propone l'ordine
     1-residency / 2-Q8_0 / 3-gather. Questo contratto **conferma quell'ordine e
     lo specializza con la misura che allora non c'era**: dentro "residency" il
     primo termine non è la capienza dell'arena, è la FETCH. La consegna resta
     l'ancora per tutto ciò che riguarda i kernel (§4.1-4.4): le forme Q4_K,
     Q6_K e Q8_0 sono già portate, misurate e ktest-ate, e NON instradate.

     PERCHÉ IL THINKING DEL 35B STA IN QUESTO GOAL E NON IN UNO SUO. I tre JSON
     del 2026-08-15 mostrano che il 35B emette un blocco `<think>` VUOTO. Non è
     un difetto del 35B — anche il 9B lo fa, in entrambi i run. È che il banco
     **non esegue il template Jinja** (dichiarato in `params.chatTemplate`) e la
     polarità del default è INVERTITA fra le due generazioni, verificato sui
     `chatTemplateRaw` dei tre file:
       Qwen3.5 (4B, 9B): enable_thinking is TRUE → '<think>\n'
                         altrimenti → '<think>\n\n</think>\n\n'   default OFF
       Qwen3.6 (35B):    enable_thinking is FALSE → '<think>\n\n</think>\n\n'
                         altrimenti → '<think>\n'                 default ON
     Quindi l'anomalia è il 4B (pensa contro il suo default, 4.781 caratteri, e
     in inglese mentre risponde in italiano), e il 35B è **l'unico che il
     rendering sta privando della sua modalità di ragionamento di default**.
     Per la funzione obiettivo del progetto — massima intelligenza sopra la
     soglia — è il modello più capace messo in modalità meno capace per un
     accidente di prompt. Ma col thinking acceso il 35B genera molti più token,
     e a 8,3 tok/s è insostenibile: **i due problemi sono lo stesso problema**, e
     la clausola del thinking è per costruzione l'ULTIMA riga prima del gate.
-->

DONE WHEN (all measurable):

- **LA REGOLA CHE VALE SU OGNI RIGA, e viene prima delle righe: OGNI LEVA È
  MISURATA SU ≥ 2 FAMIGLIE DI MODELLI.** Una riga che migliora un modello solo o
  un formato solo non chiude, anche se il suo numero è bello. È la forma
  meccanica del ruling del PI, messa qui perché sulle intenzioni ho già fallito
  una volta in questo stesso goal.

- **STRUMENTAZIONE PRIMA DI QUALUNQUE OTTIMIZZAZIONE**: i termini del token
  scomposti e nominati, `nominati / tailCpuMs ≥ 0,95` **nel regime sporco**.
  ✅ **FATTO in it.1-it.2**: `namedFrac` **0,9995**.

- **IL DECODE DEL 35B A CALDO ≥ 30 tok/s**, a contesto dichiarato, select
  `optimistic`, host quiescente, su artefatto di riferimento e non su una chat.
  Punto di partenza misurato: **22,9 tok/s** è il TETTO della sola residency
  (token pulito 43,585 ms), quindi la barra richiede per costruzione la leva sul
  pass. *Nice-to-have **≥ 45 tok/s**: è il token pulito diviso ~2, cioè ciò che
  la forma a gather dovrebbe rendere se il suo ~2,6× di dispatch si traduce
  anche solo per metà.*

- **LA FORMA A GATHER DIVENTA UNIVERSALE** — la leva sul 93,5%. Non «gather per
  il 35B»: **un solo path a gather che regge ogni famiglia MoE e ogni formato**.
  Oggi `pairGemvSiluGatherWgsl`/`gemvDownSlotsWgsl` esistono, funzionano, e sono
  cablati a **q4_0 e top-4** — cioè al solo GLM (consegna §4.2). Done-when:
  i due kernel parametrici su `nUsed` (il 35B è top-8) e sui formati K-quant;
  **misurati su GLM E sul 35B** (le due famiglie MoE: è qui che la regola delle
  ≥ 2 famiglie si verifica da sé); bit-identità col path sequenziale rimisurata.

- **IL RAGGRUPPAMENTO DELLE RICHIESTE DI I/O** — leva globale, il PI l'ha
  chiesta esplicitamente. **Misurato oggi: 2,1×** (6,90 ms/fetch con 24
  richieste concorrenti in `prepLayer`, 3,27 ms con qualche centinaio nel
  repair; stessi byte, stesso server). Done-when: il raggruppamento è nel path
  di I/O **condiviso** (`readRange`), non nei due call site del 35B; il tempo
  per fetch scende sotto **1,5 ms**; e l'effetto è misurato **anche sul LOAD**
  di 4B/9B/GLM, che passa dallo stesso `range()` — se lì non paga, si dichiara
  col numero.

- **L'UNITÀ DI RIPARAZIONE DEL DECODE OTTIMISTICO**:
  `replayLayers / (tokens × nLayer)` da **0,87** a **≤ 0,20**, oppure il replay a
  prefisso è sostituito e i suoi contatori dichiarati obsoleti con la ragione
  scritta. Vale su ogni famiglia MoE con arena che non contiene il working set:
  **misurata su GLM e 35B**.

- **IL 35B RICEVE LA SUA MODALITÀ DI RAGIONAMENTO PER DEFAULT**: il prompt reso
  in-page rispetta la polarità del template della famiglia, la scelta è
  DICHIARATA nel JSON, e un test senza GPU verifica le due polarità sui
  `chatTemplateRaw`. Gated sulla barra dei 30 tok/s. **Unica clausola
  ammessa a valere su un modello solo**, e per una ragione che non è di
  velocità: è un difetto di correttezza del prompt, non un'ottimizzazione.

- **GATE DI MERGE** (riga di sola verifica): `node
  .harness/tools/engine-ktest.mjs` tutti PASS; top-1 contro l'oracolo
  llama.cpp ≥ 1012/1024; sequenze generate identiche 8/8; **decode 4B ≥ 45,5
  tok/s a ctx 6333** e **`prefill.ms + decode.firstMs` < 22.500 ms** (le due
  barre che `engine-kquant` ha portato: non si regredisce); GLM b12 optimistic
  entro ±5%; `npx vitest run` verde; `npx tsc --noEmit` pulito.

- **CONSUNTIVO E CONSEGNA**: `docs/engine/velocita-decode-consuntivo-<data>.md`
  clausola per clausola con l'artefatto accanto, **la tabella dei tok/s prima e
  dopo su tutte e quattro le famiglie** (4B, 9B, 35B, GLM — è la prova che le
  leve erano globali), e il termine che diventa primo dopo, con la sua misura
  fresca; `HANDOFF.md` §1 aggiornato; `GLOSSARY.md` coi termini coniati.

NON-REGRESSIONE (ruling permanente 2026-07-31, banda ±5% su tok/s e TTFT, gate
di correttezza secchi):
- Le barre di `engine-kquant` sono ora pavimento: TTFT 4B < 22.500 ms, decode 4B
  ≥ 45,5 tok/s a ctx 6333, `gemm:deltanet-out` ≤ 2.000 ms, `gemm:ffn-down`
  ≤ 2.000 ms.
- Il 9B non regredisce sotto i 30 tok/s **e va misurato a contesto vero**: i
  30,17 tok/s del 2026-08-15 sono a ctx 34, e a ctx lungo scenderanno. Chi lo
  dichiara "a posto" senza una misura a ctx 6333 sta leggendo il numero
  sbagliato.
- La bit-identità del prefill MoE col path sequenziale
  (`q35-prefillmoe-35b-it34.json`, `gate.bitIdentical: true` su 993.280 valori)
  è un gate secco: qualunque cosa tocchi l'ordine delle somme del combine la
  rompe, e va rimisurata prima del merge.
- I bench costosi si eseguono su codice finale, non su stati intermedi.
- Ogni misura dichiara host e contesto; nessun tok/s senza `decodeContext`.
- Il server di sviluppo va avviato staccato e verificato con `curl`, mai con
  `pgrep`. Mai due runner GPU insieme. I flag dei runner si leggono dal
  sorgente: `--help` non esiste e fa partire il bench.

FUORI SCOPE (registrati, non aperti):
- **Il PREFILL a chunk del 35B** (§4.4 della consegna: readback CPU per layer,
  20.480 dispatch per chunk). Termine grosso e reale, ma questo goal ha una
  barra sul DECODE e mescolarci il prefill renderebbe illeggibile quale leva ha
  pagato. Le misure lo riportano; nessuna riga lo tocca. *Nota: la forma a
  gather che la riga sul 93,5% costruisce è la stessa che servirebbe lì — il
  prefill la eredita gratis, ed è un'altra ragione per cui è una leva globale.*
- **Il cablaggio Q8_0 attn** (§4.1 della consegna): tocca il prefill denso.
  Stesso motivo.

> **RIENTRATA IN SCOPO il 2026-08-15 su ruling del PI: la forma a gather
> K-quant.** Era qui, esclusa con «varrebbe il 16% del token». Il 16% veniva
> da una stima non misurata; sul numero vero vale il **93,5%**, ed è la leva
> che porta la barra. Il PI aveva già chiesto di farla nel goal precedente.
- **Il LOAD del modello** (7,85 s sul 35B): soglia sua, famiglia residency/IO,
  ma è una leva sul time-to-first-use e non sul tok/s.
- **Il 29,1% fuori dai pass GPU del 4B**: eredità di `engine-kquant`, resta il
  candidato dopo questo goal.

WORKING PROTOCOL: skills loop-iteration + done; verifier gate per ciclo; digest
ogni ciclo; stop-by-design quando il resto è docket-gated.
VEICOLI: `sdd-conductor` per le ondate a spec, `second-opinion` sui rami che
meritano sfiducia verso un revisore solo, `research-campaign` per le righe con
una tesi da pre-registrare.
VINCOLO: niente `/loop` con un workflow in volo finché il watchdog dell'harness
non è corretto (misurato tre volte il 2026-08-14). Lavoro lungo con umano
presente, oppure `ScheduleWakeup{stop:true}` prima di lanciare il workflow.

CONTEXT ANCHORS:
- `docs/engine/kquant-consegna-35b-2026-08-15.md` — la scheda che apre questo
  goal: §4.5 residency, §5 cosa NON c'è, §6 l'ordine di lavoro
- `results/chat/chat-35b-2026-08-15T16-25-12.json` — i contatori da cui viene
  tutta l'aritmetica del contratto (NON-riferimento: i tok/s non valgono)
- `src/engine/q35gpumodel.ts:1958-2008` — il tail ottimistico: `tTail`, il
  `while (cur.missCount > 0)`, il repair e il replay a prefisso
- `src/engine/q35gpumodel.ts:1994` e `:2050` — le DUE await della fetch
- `src/engine/q35gpumodel.ts:199-219` — il commento di `perf()` (**stale**:
  descrive `tailCpuMs` come sola contabilità, ma `tTail` sta prima del loop di
  repair e ne include replay e await)
- `src/engine/q35expertstore.ts:96-111` — `q35ExpertReader`: 3 `readRange` per
  expert
- `src/engine/chat/chat.worker.ts:62-68` — `range()`: la fetch HTTP Range
- `src/engine/residency.ts:250-258` — l'avvertenza su `readMs`, e `:820-847`
  dove pack e upload SONO misurati
- `src/engine/residency.ts:294-314` — `expertSlots`, il riparto pro-quota
- `results/engine/q35-vramplan-35b-it35.json` — l'ultima misura di decode del
  35B (2026-08-11, PRIMA dei kernel nuovi): optimistic-warm 44,32 ms/token
- `results/engine/q35-header-dump-2026-08-10.json` — l'istogramma dei tipi
