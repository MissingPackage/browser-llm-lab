GOAL: engine-35b-residency — il 35B smette di pagare la residency a ogni token:
il decode a caldo passa da 8,34 tok/s a **oltre 30 tok/s**, che è la soglia di
usabilità della funzione obiettivo del progetto. Il lavoro NON è nei kernel: è
nella fetch degli expert mancanti, oggi seriale dentro il loop del token, e
nell'unità di riparazione del decode ottimistico.

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

     **IL NUMERO CHE CHARTER-A IL GOAL: 21,1 ms.** È il token del 35B senza la
     tassa di residency, cioè **47,4 tok/s** — sopra la soglia. La lentezza del
     35B non è fisica e non è un limite di banda: il pavimento di banda su
     questa scheda è ~2,9 ms/token (~1,66 GB di pesi attivi, 4090 mobile
     ~576 GB/s — *conto mio dagli header, non una misura*). La fisica sta a 2,9,
     il codice pulito a 21,1, il misurato a 120. Due ordini di separazione.

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

- **BASELINE E STRUMENTAZIONE PRIMA DI QUALUNQUE OTTIMIZZAZIONE.** Un artefatto
  di riferimento del 35B (host dichiarato, warm-up scartato, repliche) con i
  termini del token scomposti, e i contatori nuovi che nominano ciò che oggi è
  residuo: la somma dei termini nominati copre **≥ 95% di `tailCpuMs`**. Senza
  questo, ogni riga successiva è una scommessa: oggi il 43% del turno non ha
  nome.

- **IL DECODE DEL 35B A CALDO ≥ 30 tok/s**, a contesto dichiarato, select
  `optimistic`, su host dichiarato quiescente, misurato su artefatto di
  riferimento e non su una chat. *Nice-to-have, e non è un secondo obiettivo ma
  la lettura del pavimento misurato: **≥ 45 tok/s**, cioè il token pulito di
  21,1 ms senza aver aggiunto nulla al resto.*

- **I MISS SMETTONO DI ESSERE SERIALI COL TOKEN**: il tempo per miss scende da
  5,98 ms sotto **1,5 ms** (cioè sotto il costo pack+upload già misurato, 1,33
  ms/miss), oppure la fetch esce del tutto dal path critico e il contatore che
  la misura lo dimostra. Il meccanismo lo scelgo io sui numeri della riga 1 —
  non è deciso qui.

- **IL LAVORO GPU RIPETUTO SCENDE**: `replayLayers / (tokens × nLayer)` da
  **0,87** (oggi: 37.977 su 43.680) a **≤ 0,20**, oppure il replay a prefisso è
  sostituito e il contatore che lo misura è dichiarato obsoleto con la sua
  ragione scritta.

- **IL 35B RICEVE LA SUA MODALITÀ DI RAGIONAMENTO PER DEFAULT**: il prompt reso
  in-page rispetta la polarità del template della famiglia, la scelta è
  DICHIARATA nel JSON (non implicita), e un test senza GPU verifica le due
  polarità sui `chatTemplateRaw` dei tre modelli. Clausola gated sulla barra dei
  30 tok/s: si esegue solo dopo, per la ragione scritta nel contratto.

- **GATE DI MERGE** (riga di sola verifica): `node
  .harness/tools/engine-ktest.mjs` tutti PASS; top-1 contro l'oracolo
  llama.cpp ≥ 1012/1024; sequenze generate identiche 8/8; **decode 4B ≥ 45,5
  tok/s a ctx 6333** e **`prefill.ms + decode.firstMs` < 22.500 ms** (le due
  barre che `engine-kquant` ha portato: non si regredisce); GLM b12 optimistic
  entro ±5%; `npx vitest run` verde; `npx tsc --noEmit` pulito.

- **CONSUNTIVO E CONSEGNA**: `docs/engine/35b-residency-consuntivo-<data>.md`
  clausola per clausola con l'artefatto accanto, la nuova ripartizione del
  token, e **il termine che diventa primo dopo questa leva, nominato con la sua
  misura fresca**; `HANDOFF.md` §1 aggiornato; `GLOSSARY.md` coi termini coniati.

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
- **Il PREFILL del 35B a chunk** (§4.4 della consegna: readback CPU per layer,
  `for m2` con `prepLayer`+`encodeExperts` per riga, 20.480 dispatch per chunk).
  È un termine grosso e reale, ma il TTFT del 35B non è la barra di questo goal
  e mescolarci il decode renderebbe illeggibile quale leva ha pagato. La riga 1
  lo MISURA (serve alla baseline); nessuna riga lo tocca.
- **I kernel a gather K-quant** (§4.2-4.4 della consegna, ~2,6× di dispatch).
  Oggi varrebbero il 16% del token: sono un'ottimizzazione del termine che
  diventa primo DOPO questo goal, non di quello che è primo adesso. Se la
  riga 1 li promuove a primo termine, il contratto si riapre — non si allarga in
  silenzio.
- **Il cablaggio Q8_0 attn** (§4.1): tocca il prefill denso, cioè il piano che
  il 4B usa già. Stesso motivo.
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
