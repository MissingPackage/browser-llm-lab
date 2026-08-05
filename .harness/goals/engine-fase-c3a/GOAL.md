GOAL: engine-fase-c3a — Il motore esegue GLM-4.7-Flash sopra il floor dell'oracolo
CPU di C1 (decode >=13.43 tok/s) e possiede un percorso di prefill batched M>1,
eliminando i tre costi strutturali attribuiti in C2 (pack CPU nel path caldo,
46 sync router/token, forward sequenziale in prefill) a correttezza invariata.

<!-- EMENDAMENTO 6 (ruling PI 2026-08-05, in chat + docket item 16): FASE 4d —
     RISANAMENTO DELLA BASE. "Risistemiamo la base di orchestrazione,
     telemetria, error handling, ecc. non ha senso produrre questo casino."
     Motivo strutturale (state-2026-08-02 §5): i path Qwen e GLM non
     condividono NESSUN modulo di orchestrazione, telemetria, error handling
     o negoziazione limiti, e le divergenze si sono gia' pagate (dispatch
     count sbagliato di 2 per tutto C2, `uncapturederror` assente sui worker
     GLM = la classe di errore silenzioso che il progetto teme di piu').
     Perche' serve un emendamento: il lavoro e' trasversale agli owns di fasi
     chiuse (gpuforward = fase A, glmbench = fase 1) — senza una fase propria
     sarebbe lavoro senza owner, come per l'emendamento 4.
     Perimetro della 4d (riga in PHASES): helper unico di creazione device
     (limiti negoziati + uncapturederror ovunque — chiude il residuo del
     docket item 10 con ri-baseline dichiarata); schema di telemetria unico
     (contatori cumulativi diffabili, default off, ttftMs e hostState in OGNI
     report, anche Qwen); dispatch Planned+Measured su entrambi i path;
     glmsource.ts sotto test; il gate 256/256 emesso come campo JSON; path
     non-fuso Qwen morto e artefatti orfani risolti. NON incluso: ktest dei
     kernel fusi solo-Qwen (debito registrato, si riduce in fase 5).
     I gate di chiusura del goal restano INVARIATI. -->

<!-- EMENDAMENTO 5 (ruling PI 2026-08-05, docket item 15: "andiamo su opzione
     c: trovare il GiB altrove"): la 4c CAMBIA MEZZO, non fine. La ladder di
     it.18 ha dimostrato che la degradazione non passa il gate a nessun P
     (P(pass)≈2e-9, danno unidirezionale): la qualita' del modello NON si
     spende. La residenza totale resta l'obiettivo; il deficit (0.67 GiB
     @ctx525) si colma dal bilancio VRAM dell'host (processi desktop ~347-763
     MiB su dGPU senza iGPU: si spegne la sessione per le run di gate) e la
     prima azione e' misurare il tetto allocabile VERO da Chrome/Dawn (il
     15.247 GiB del design era aritmetico, mai verificato contro i 429 MiB di
     memory.reserved del driver).
     Conseguenze sul contratto:
     (1) l'authority dell'emendamento 4 (quant asimmetrica + layout slab v2
         con regione degradata) DECADE inutilizzata: nessun kernel Q3_K,
         nessuna terza size-class, nessun file v2. Quantizzatore e ladder
         restano nel repo come strumenti di eval;
     (2) la 4c si ri-slicea: A′ probe del tetto VRAM (3 regimi host), B′
         preload asincrono + slotsExact + precondizione, C′ residenza totale
         Q4_0 puro e chiusura congiunta fase 4 + 4c;
     (3) regola pre-dichiarata: gap residuo dopo il recupero host ⇒ docket
         col numero, MAI degradazione;
     (4) l'eval di perdita della 4c decade (pesi immutati); i gate restano
         invariati e il 98.83% torna a essere non-regressione pura;
     (5) il protocollo di bench guadagna il REGIME HOST come parametro
         dichiarato nel report (sessione piena/minima/headless): i gate di
         chiusura si misurano nel regime dichiarato, a parita' di protocollo
         per ogni confronto. -->

<!-- EMENDAMENTO 4 (ruling PI 2026-08-03, docket item 8 + "fai l'emendamento"):
     la RESIDENZA TOTALE entra nel perimetro C3a come fase 4c.
     Motivo: il PI ha deciso di pagare ~0.67 GiB di qualita' per eliminare il
     drain (docket item 8). Senza miss il readback del router non si riduce,
     sparisce — ed e' la condizione che rende applicabile il pattern binding
     fisso + offset aritmetico di spec §3.2-bis, provato da ORT/ggml/MLC.
     Perche' serve un emendamento e non bastava il ruling: il lavoro tocca
     quant.ts / slabfile.ts / il percorso di import, che sono gli `owns` della
     fase 3, CHIUSA. Senza una fase propria sarebbe lavoro senza owner.
     Conseguenze sul contratto:
     (1) PHASES guadagna la fase 4c, fra la 4 e la 6;
     (2) "quant asimmetrica sugli expert piu' freddi + nuova versione di layout
         dello slab" esce dal must-docket ed entra nell'authority, MA la scelta
         di QUALI expert degradare resta vincolata a una eval di perdita: il
         ruling autorizza la spesa, non esonera dal misurarla;
     (3) i gate di correttezza restano invariati e valgono come tetto alla
         perdita ammissibile (top-1 vs golden >= 98.83% full-corpus,
         argmax = cpuref-f64 sul campione ratificato);
     (4) il deficit da colmare NON e' un numero fisso: 0.67 GiB a ctx 525,
         1.03 GiB a ctx 4096. Il dimensionamento va fatto sul contesto che C3b
         deve servire, e la scelta va scritta nel report.
     Il gate 13.43 resta INVARIATO. -->

<!-- EMENDAMENTO 2 (ruling PI 2026-08-02, docket item 7): entrano nel perimetro
     C3a due cose prima vietate o non previste.
     (a) PREFETCH LOOKA (era C3b, must-docket in C3a). Motivo: la misura di it.1
         piu' l'analisi di it.3 mostrano che CPU e GPU non si sovrappongono MAI
         (per layer: 1.7 ms di GPU con CPU ferma, 1.5 di drain, 1.18 di ensure
         con GPU ferma). Predire gli expert del layer l+1 dall'hidden del layer
         l (recall 92% @K=8 misurato in C1) sposta ~54 ms/token di lavoro CPU
         dietro il lavoro GPU. Questo RIBALTA la valutazione del docket C2
         item 8, che dava al prefetch valore residuo piccolo guardando solo lo
         stallo e non la sovrapposizione.
         Restano a C3b: slab ctx-aware, tier.h, AUTOPIN, pin learned, modello di
         banda, instant-on, WP banda fredda browser.
     (b) NEGOZIAZIONE DEI LIMITI WebGPU. Il probe di it.3 mostra che il device
         prende i default di spec su tre limiti che l'adapter offre piu' alti:
         maxStorageBuffersPerShaderStage 8 su 16, maxComputeInvocationsPerWorkgroup
         256 su 1024, maxBufferSize clampato a 2 GiB su 4 disponibili.
         Si negoziano in fase 3. -->

<!-- EMENDAMENTO 1 (ruling PI 2026-08-01, docket item 4 opzione b): la
     GRANULARITÀ/FUSIONE DEI DISPATCH entra nel perimetro C3a come QUARTA LEVA.
     Motivo: la misura di fase 1 dimostra che le tre leve originali non possono
     raggiungere il gate — con repack e sync al 100% di efficacia si arriva a
     10.18 tok/s, e `gpuBusy` da solo (78.2 ms/token) eccede il budget del gate
     (74.46). Il margine c'è: 2.22 GB di pesi/token su 576 GB/s danno un floor
     memory-bound di 3.85 ms/token, quindi `gpuBusy` è 20× sopra (1816 dispatch
     a 43 µs l'uno). Il gate 13.43 resta INVARIATO.
     Conseguenze sul contratto: (1) nuovo DONE WHEN sulla quarta leva con
     soglia DERIVATA `gpuBusy ≤ 54.5 ms/token`; (2) "fusioni cross-layer /
     megakernel / granularità dei dispatch" esce dal must-docket ed entra
     nell'authority; (3) PHASES guadagna la fase 4b e la fase 4 chiude con una
     ri-misura di clock e `gpuBusy` che dimensiona la 4b.
     Docket item 2 (clausola di fallback) RIMANDATO dal PI a fine fase 4. -->

<!-- CONTRATTO v1 — approvato dal PI 2026-08-01 in chat, in due passaggi:
     (1) "ok lo split in 2 goal" — la fase C3 di direction §7 e' splittata in
         C3a (struttura: il floor tok/s) e C3b (paging: slab ctx-aware, tier,
         AUTOPIN, PILOT-real, modello di banda, instant-on, WP banda fredda
         browser). Motivo: l'attribuzione C2 (docket item 8) separa i 158.9
         ms/token di struttura dai 56.1 di residenza — due assi con due
         condizioni di chiusura diverse; un goal unico perdeva la completion
         condition. direction §7 emendata di conseguenza.
     (2) RULING sul gate prefill (domanda lasciata aperta da docket C2 item 6):
         NON e' un requisito di sola capacita'. "alla fine abbiamo deciso di
         puntare alla massima intelligenza possibile con una velocita' minima
         accettabile. 30+ tk/s come soglia normale e di piu' se thinking. per
         garantire l'esperienza utente ottimale". Conseguenza sul contratto:
         il floor C1 resta il gate di CHIUSURA (e' misurabile oggi), ma ogni
         report deve stampare la distanza dalla soglia UX — che e' l'obiettivo
         vero. TTFT budget fissato dal PI a <=4 s ("a 2 rischiamo di essere
         troppo stringenti e non chiudere mai. alla fine 4 secondi e'
         accettabile sul mio hardware che non e' top").
     Soglie [ASSUMED] del draft confermate dal PI senza correzioni: sync/token
     <=2, pack <1.0 ms/token, 60 tok/s in regime thinking, errore del modello
     di banda +/-15% (C3b), budget slab di prova 50%/25% del parco (C3b).
     Nomi c3a/c3b confermati.
     Goal start tag: goal-engine-fase-c3a-start. -->

DONE WHEN (all measurable):
- Spec C3a scritta (docs/superpowers/specs/<data>-engine-fase-c3a-design.md) e
  ruling PI di approvazione registrato nel docket. La spec fissa: formato dei pesi
  repacked all'import (layout, versioning dell'artefatto OPFS, invalidazione su
  SHA/versione mismatch); meccanismo scelto per i sync router, con il criterio di
  scelta esplicitato tra le opzioni note (top-4 su GPU con indirect dispatch, bind
  completo del parco residente, pipelining a profondita >1); percorso prefill M>1
  per MoE (insiemi di expert diversi per token) con la sua condizione di identita;
  protocollo di bench IDENTICO a C2 (prompt p6 461 token, nGen 64, 3 repliche +
  warmup, mediana, slab 12 GiB, macchina quiescente) — nessun confronto fuori
  parita; definizione operativa del TTFT misurato (da quale evento a quale).
- Repack all'import: l'import produce l'artefatto repacked (test verdi su
  produzione dell'artefatto + invalidazione) e il report di bench riporta pack CPU
  < 1.0 ms/token nel path caldo, contro i 42.5 ms/token della baseline C2.
- Sync router: il report di bench riporta <= 2 sync CPU<->GPU per token di decode
  (baseline C2: 47 = 46 readback router + 1 logits). Se il meccanismo scelto in
  spec e' il pipelining, la spec puo' riformulare la soglia come sync/token
  AMMORTIZZATI, dichiarando la profondita: la riformulazione va in spec col
  ruling, non presa qui.
- Granularità dei dispatch (QUARTA LEVA, emendamento 1): il bench riporta
  **`gpuBusy` <= 54.5 ms/token** (baseline it.1: 78.2). La soglia e' DERIVATA
  dall'aritmetica del gate, non scelta: budget 74.46 − stallo residenza
  post-repack 12.4 − floor di sync misurato 7.6 = 54.5. Va riportato anche il
  conteggio dispatch/token (baseline 1816) e il clock SM medio campionato
  durante la run, perche' una parte della riduzione puo' arrivare dai clock che
  salgono quando spariscono le bolle, non dal lavoro sui kernel: le due cause
  vanno distinte nel report, non confuse in un unico numero.
- GATE decode (chiusura): bench JSON quiescente in results/engine/ con
  decode >= 13.43 tok/s a parita' di protocollo C2 (floor oracolo CPU C1,
  results/engine/moe-oracle/llama-bench-glm47flash-q4_0-2026-07-30.json).
- Prefill: (i) esiste il percorso batched M>1, provato da un test di identita'
  logits M=1 vs M>1 sullo stesso prompt entro la tolleranza fissata in spec; e
  (ii) GATE prefill >= 56.58 tok/s sullo stesso prompt p6.
- Distanza dalla funzione obiettivo, RIPORTATA e non gateata (ruling PI
  2026-08-01): ogni report di bench committato include, calcolati e stampati,
  (a) decode tok/s vs soglia UX 30 (e vs 60 in regime thinking) col fattore di
  gap residuo; (b) TTFT misurato vs budget <= 4 s a ctx ~500 (su p6, 461 token,
  4 s equivalgono a ~115 tok/s di prefill, ~2x il floor CPU) col fattore di gap.
  Il goal chiude sui gate del floor; questi numeri NON sono gate, ma la loro
  ASSENZA dal report e' un FAIL della checklist di chiusura.
- Correttezza invariata (nessun gate nuovo, solo non-regressione dei valori C2):
  argmax == cpuref-f64 100% sul campione ratificato 2/8 (256/256 posizioni);
  top-1 vs golden >= 98.83% full-corpus; routing conformance rigenerato e non
  peggiore di results/engine/routing-conformance-glm47flash-2026-07-31.json.
- Non-regressione del pregresso: conformance fase A verde; bench Qwen2.5-0.5B a
  macchina quiescente >= baseline permanente 2026-08-01
  (results/engine/bench-4090-2026-08-01T16-34-04-484Z.json: decode K=8 321.88,
  K=1 243.97, prefill chunked <= 600.2 ms); suite npm test verde (>= 220 passed)
  e npx tsc --noEmit pulito.
- Chiusura: docket aggiornato con gli input per C3b (costo residuo della residenza
  dopo il repack, hit-rate osservato, nuovo profilo ms/token scomposto),
  direction §7 e docs/engine/ideas-ledger.md §A aggiornati dove resi stale,
  HANDOFF refresh.

EVIDENCE OF DONE: npm test verde + npx tsc --noEmit pulito; run glmbench con
exit 0 e JSON committato in results/engine/ (decode, prefill, TTFT, sync/token,
pack ms/token, hit-rate, dispatch/token, gap vs 30 tok/s e vs 4 s); run glmconf
con exit 0 e JSON di conformance logits + routing; bench Qwen JSON di
non-regressione; test di identita' M=1/M>1 nella suite; diff di spec + docket +
direction + ledger + HANDOFF.

AUTHORITY GRANTED:
- may do autonomously: MODIFICARE src/engine/** (e' l'oggetto del goal: import e
  repack, kernel MoE/MLA, scheduling dei dispatch, percorso prefill batched,
  telemetria; **e dall'emendamento 1: fusione dei kernel, riduzione del numero
  di dispatch, megakernel parziali, batching delle catene expert** — la quarta
  leva e' ora dentro il perimetro), tests/**, tools/**, scripts/**;
  commit/push su main a fine
  iterazione VERIFICATA (ruling PI 2026-07-31, ratifica del main-diretto);
  merge su main + push a goal CHIUSO e verificato (ruling permanente 2026-07-29);
  rigenerare/invalidare gli artefatti in ~/.cache/blab-models e nel profilo
  Chrome ~/.cache/blab-glmroute-profile (inclusa la re-import da 17.2 GB);
  riuso del checkout llama.cpp di C1 per rigenerare golden (nessun download
  nuovo, nessuna spesa); run locali su 4090; aggiornare docs/engine/* quando
  stale (ruling 2026-07-29); docket/HANDOFF refresh.
- must docket (never do): tier.h, AUTOPIN, pin learned, instant-on, budget slab
  ctx-aware, WP banda fredda browser (= C3b; il PREFETCH/PILOT-real e' USCITO
  da questa lista con l'emendamento 2 ed e' ora dentro il perimetro C3a); cambio modello-tesi, quant diversa da Q4_0,
  direction §3; testa MTP/spec-dec (fase D); benchmark pubblico (ruling
  2026-07-30); run M4/S22 e hero-demo (PI-gated per hardware); ogni spesa;
  delete di codice committato >30gg senza check git log --follow; merge con
  QUALSIASI metrica misurata in regressione — la deroga e' decisione PI via
  docket; riformulazione o abbassamento dei gate 13.43/56.58 e del budget TTFT
  (decisione PI, come lo e' stata la deroga C2).

CONSTRAINTS: spec-first prima del codice; NON-REGRESSIONE PERMANENTE (ruling PI
2026-07-31): nessuna metrica gia' committata in results/ peggiora, salvo deroga
PI a docket; bench SOLO a macchina quiescente (norma PI 2026-08-01) con
dichiarazione dello stato host nel report; confronti solo a parita' di
corpus/contesto/protocollo (lezione B2); llama.cpp SOLO oracolo, mai linkato al
motore; f32-first sul dev-loop Chrome/Linux, percorso f16 dietro feature-detect;
tap hidden-states preservati nel forward (direction §4.4 — retrofit vietato per
design, e sono l'input di PILOT-real in C3b); telemetria zero-overhead se spenta;
determinismo nei report (SHA-256 del GGUF, greedy, commit oracolo); mai gate
diretti engine-vs-oracolo su selezioni near-tie (l'oracolo CPU quantizza le
attivazioni q8 — il confronto giusto e' vs cpuref-f64); maxAbsDeltaLogit e'
metrica di scala, mai gate; zero attribution AI nei commit.
FUNZIONE OBIETTIVO (direction §1, ribadita dal ruling PI 2026-08-01): "massima
intelligenza sotto vincolo di rate sufficiente" — la soglia UX e' ~30 tok/s di
decode (piu' alta, ~60, in regime thinking: i token di ragionamento sono latenza
pura prima della risposta visibile) e un TTFT <= 4 s. Il floor CPU 13.43/56.58 e'
un gate d'ingresso INTERMEDIO: mai da presentare, riportare o trattare come
obiettivo raggiunto.

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md §1/§5; .harness/goals/engine-fase-c2/docket.md item 8
(INPUT C3: baseline 215.5 ms/token scomposti, leve dimensionate) e item 6
(attribuzione + precisazione sul gate prefill); .harness/goals/engine-fase-c2/
GOAL.md (emendamento 3) e journal.md it.7 (misura del pack) / it.11 (harness
glmbench); docs/engine/direction.md §1 (funzione obiettivo), §3 (config GLM
verificata) e §7 (fase C splittata);
docs/superpowers/specs/2026-07-30-engine-fase-c1-design.md (replica esatta del
router); src/engine/{gguf,gpuforward,shape,quant}.ts;
results/engine/moe-oracle/llama-bench-glm47flash-q4_0-2026-07-30.json (il floor);
results/engine/bench-glm-4090-b12-quiesced-2026-08-01.json (la baseline);
.harness/goals/engine-fase-c3b/GOAL.md (il perimetro che questo goal NON tocca).
