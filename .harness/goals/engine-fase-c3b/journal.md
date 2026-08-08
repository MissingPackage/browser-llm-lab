# Journal — engine-fase-c3b (decode ottimistico + repair)

## it.0 — 2026-08-07 — setup (goal-setup)

Contratto v2 scritto e committato (dfa8cec) su delega PI ("Scegli tu su questi
punti e poi procedi con il loop"); decisioni registrate in docket item 2 (split
C3b/C3c, gate strutturale, policy>LRU in C3c) e item 3 (banda di rumore
provvisoria). Tag goal-engine-fase-c3b-start su dfa8cec. PHASES.md: 7 fasi
sequenziali, nessuna docket-born (authority delta = none ovunque). Plan-check:
pre-autorizzato dalla stessa delega (docket item 4, pattern c3a item 3) — il
revert e' un edit di PHASES prima della fase 2.

## it.1 — 2026-08-07 — fase 1: la spec del decode ottimistico

### Cosa

`docs/superpowers/specs/2026-08-07-engine-fase-c3b-decode-ottimistico.md`,
scritta dal codice (glmmodel select:"gpu"/resolve/tail-readback, residency
slotTable/flush, telemetry schema unico) e dai numeri WP-0. Fissa i 7 punti
del contratto: flag miss piggyback (dirtyB 2xu32 atomici scritti dal resolve,
letto nella STESSA mapAsync di coda), checkpoint hidden 47x2048xf32 = 376 KiB
via copyBufferToBuffer (dispatch/token invariati), replay dal primo layer
sporco, slotTable congelata in volo (assert inFlight, insert/evict SOLO al
confine), prefill invariato, rifiuto build-time a slots/park < 0.88 [ASSUMED]
senza fallback dinamico, esclusioni (no predictor di confine, no repair
batched, variante 2-segmenti nel cassetto). Scelte nuove registrate a docket
item 5: modo `select:"optimistic"` (it.17 intatto), pin-for-replay (eviction
del repair mai sugli slot Sel dei layer >= firstDirtyLayer, <= 184 slot) =>
replay pulito per costruzione, max 1 replay/token, replay sporco = throw.
5 invarianti I1-I5 con assert; contabilita' attesa 1+P(dirty) submit e sync
per token => gate strutturale <= 2 per costruzione, il bench lo conferma.

### Verifica

loop-verifier (a32b0b61): griglia 7/7 punti del contratto FISSATI, riferimenti
al codice accurati riga per riga (glmmodel 339-361, residency 234/245/336-339,
telemetry 19/40-49), numeri coerenti col JSON WP-0 (steady 2596/2419, cost
model, localita'). Primo verdetto FAIL di FORMA (journal/digest/PHASES/HANDOFF
non ancora aggiornati — questa entry e il commit li saldano) + 2 note recepite:
188 -> 184 slot (46 layer MoE, non 47), confronto fase 5 su TUTTE le quantita'
del contratto (non solo tok/s), con attribuzione LruFast come nota
interpretativa. Nessuna violazione di constraint, nessun drift, gate/soglie
del contratto intoccati.

### Prossimo pezzo

Fase 2 (blocked-by-1 sciolto): guard MISS nei kernel arena + dirtyB +
checkpoint hidden, misurati in ktest su GPU reale.

## it.2 — 2026-08-07 — fase 2: flag di miss + checkpoint hidden, misurati in ktest

### Cosa

La scoperta che restringe la fase: la guardia MISS (spec 3a) ESISTEVA GIA'
(`arenaSlotWgsl`: `ok` propagato alle guardie, degrado definito, it.15) e il
resolve scriveva gia' `Sel.flags` bit 0. Costruito il resto:
- `routerTopKWgsl` opzione `resolve.dirty` -> binding 7 `dirtyB` (atomicMin
  del primo layer MoE sporco + atomicAdd dei miss); senza l'opzione il WGSL
  emesso resta BYTE-IDENTICO (shadow/gpu invariati, verificato dal verifier
  con un diff HEAD-vs-tree sul testo emesso).
- glmmodel `select:"optimistic"`: struttura a 1 submit di it.17 con
  precondizione near-total (slots/park >= 0.88 default, override solo test),
  preload a capacita' SENZA evict, reset dirtyB per token via writeBuffer
  (sentinel 0xffffffff — clearBuffer renderebbe ambiguo il layer 0; spec 3b
  allineata), staging dirty nella STESSA Promise.all di coda, `dirty` nel
  ritorno del forward con CROSS-CHECK kernel-vs-Sel (divergenza = throw).
- checkpoint hidden (`opts.checkpointHidden`): copyBufferToBuffer x ->
  hiddenCkpt[l] per layer nell'encoder (zero dispatch in piu'),
  `readHiddenCkpt()` per l'harness. `debugMarkMiss` (glmmodel+residency,
  SOLO harness) per miss deterministici.
- ktest: kit `mkMiniModelKit` estratto da testGlmModel2Layer (pura
  estrazione, +`zeroExperts` nel ref MoE = il degrado definito) + 3 casi.

### Verifica (GPU reale, /tmp/ktest-c3b-it2.log)

**ktest 68/68 PASS** (65 esistenti invariati +3):
- `glm-optimistic-forced-miss`: run pulita L2rel 5.98e-8 vs ref pieno,
  **1 submit/0 sync**, dispatch = piano+testa, **0 falsi positivi**; forzati
  {3,2} + controllo 4 marcato-ma-non-selezionato: missCount=2, first=1, miss
  esatti sui k giusti, controllo ASSENTE, flags per-k nella Sel di produzione;
  degrado == ref f64 a expert azzerati (L2rel 5.15e-8, argmax e logit in banda).
  Selezione attesa dal riferimento f64 INDIPENDENTE (no tautologia).
- `glm-hidden-checkpoint`: gpu==optimistic **4096/4096 bit**, riga0==xIn
  **2048/2048 bit**, riga1 vs input f64 del layer 1 L2rel 3.17e-7,
  dispatch/token invariati (le copy non sono dispatch).
- `glm-optimistic-precondizione`: throw con frammenti asseriti (7/64 slot).
`npm test` **337+7** invariata; `tsc --noEmit` pulito. Verifier (a2991b813):
PASS su tutti i 7 punti, incluse byte-identita' WGSL e assenza di regressioni
cpu/shadow/gpu (metriche identiche a c3a).

### Note di merito (dal verifier, recepite)

- Done-when (b) "bit-identico al forward sincrono": l'hidden intermedio NON e'
  bit-confrontabile fra path diversi (gia' in c3a gpu-vs-cpu 8309/12288). Il
  test usa gli ancoraggi piu' forti disponibili — gpu==optimistic bit-exact
  (il checkpoint del giro sincrono E' quello), riga0==xIn esatta, riga1 vs f64
  — che sono quelli che servono a I4 (replay = stesso input bit-identico).
  DICHIARATO QUI perche' il verifier di fase 7 non lo riapra.
- Landmine per la fase 4: il preload optimistic riempie in ordine (layer,
  expert) crescente — sul modello vero a ~88% lascia MISS sistematici gli
  expert alti degli ultimi layer. Se P(dirty) esce dalla proiezione WP-0,
  guardare PRIMA l'ordine di preload, poi la LRU.

### Prossimo pezzo

Fase 3: repair al confine (fetch dei mancanti, pin-for-replay, flush unico,
replay da hiddenCkpt[firstDirty], assert I1/I3) + test di identita'
ottimistico-vs-sincrono con miss forzato e caso di rifiuto.

## it.3 — 2026-08-07 — fase 3: repair + replay, e l'identita' e' BIT-esatta

### Cosa

- `forward` ristrutturato in `runPass(startLayer)` (wrap meccanico a
  indentazione preservata, ordine delle operazioni INVARIATO — verificato dal
  verifier col diff): il replay e' un secondo giro dello stesso token con
  rientro `hiddenCkpt[startLayer] -> x` e `l < startLayer => continue`.
- Driver del confine (spec §4): pin-for-replay (slot Sel dei layer >=
  firstDirty, miss inclusi), `ensure` dei mancanti, UN `flushSlotTable` dopo
  le writeBuffer (R5), replay, **throw I3 se il replay e' sporco — mai un
  secondo replay**, `dirty.repaired`. `optimisticRepair:false` SOLO harness.
  hiddenCkpt ora FORZATO in optimistic (e' l'input del replay).
- I1 nel path: `ExpertCache.setInFlight` + guard su ensure/flush/debugMarkMiss
  (armato solo in optimistic; nel sync l'ensure fra submit e mapAsync E' il
  design). Unit node dedicata. `debugMarkMiss` ora EVICTION vera (lru+free+
  MISS+flush): il repair fa fetch reali.
- Telemetria: `dirtyTokens`/`replays`/`repairMs` (P(dirty) = dirtyTokens/
  forwards; la tassa GPU sta gia' in submits/dispatches/gpuBusy).

### Verifica (GPU reale, /tmp/ktest-c3b-it3.log; verifier a550b0ff PASS su sostanza 8/8)

**ktest 69/69**; il nuovo `glm-optimistic-identita` misura il claim del GOAL
("qualita' bit-invariata"), piu' forte del done-when (argmax): **64 posizioni,
3 con replay forzato (startLayer=1: rientro dal checkpoint esercitato),
hidden E logits bit-identici 131072/131072 vs sincrono**, argmax 64/64,
routing per-k identico, submits 67 = 64+3, routerSyncs 0, contatori 3/3.
La bit-identita' dell'INTERA sequenza prova anche la KV: le posizioni dopo i
replay dipendono dalle righe KV riscritte dal replay (kvAppend idempotente
per pos, verificato dal verifier sul WGSL). 68 kernel preesistenti con
metriche IDENTICHE a it.2 (solo timing OPFS variato). `npm test` **338+7**
(+1: unit I1); `tsc` pulito. Fase-2 forced-miss ora a `optimisticRepair:false`
(senza, il repair ripara prima del ritorno e il degrado non si osserva).

### Deviazione dichiarata (il verifier di fase 7 non la riapra)

Il `[ASSUMED ctx ~500]` del done-when non e' esercitabile in ktest (device
ktest ctxMax 64): identita' misurata a **ctx 64 / 64 token**. Giudizio del
verifier: accettabile — parametro [ASSUMED] non fissato dal PI, proprieta'
strutturalmente ctx-indipendente, dipendenza cross-posizione via KV inclusa
nella prova; la scala vera e' coperta dai done-when di fase 4 (glmbench sul
modello vero) e fase 6 (conformance).

### Note non bloccanti (recepite)

- Throw fra submit e readback ⇒ inFlight resta true: DELIBERATO (throw
  fatale), ora commentato nel codice.
- debugMarkMiss libera slot che la Sel in VRAM puo' ancora referenziare:
  SOLO harness, mai in produzione (documentato).
- PHASES riga 3: owns integrato con src/engine/ktest/ (correzione da
  verifier, non un re-scope: il done-when esigeva il test in ktest).

### Prossimo pezzo

Fase 4: glmbench sul ramo ottimistico (modello vero, ctx ~500) + gate
strutturale sync/token <= 2 e submits/token <= 2 in produzione, report con
P(dirty)/miss/tassa/decode/prefill/TTFT + gap UX + hostState.

## it.4 — 2026-08-07 — fase 4: il decode ottimistico in produzione — 11.60 tok/s (+123%)

### Cosa

Wiring bench: `--select optimistic` (runner/page/worker), allocazione slot
q4_1-first (`optimisticSlots`: q4_1 intera + resto a q4_0 — docket item 6b),
report esteso (gates.structural, sezione optimistic, TelemDelta con
dirty/replay/repair), dispatchPlan neutralizzato in optimistic (il replay
aggiunge dispatch legittimi). Precondizione EMENDATA a TOTALE >= 0.80
(docket item 6a: per-classe 0.88 = 13.5 GiB, oltre il tetto fisico a
qualunque host; collasso WP-0 a <= 50%).

### I due finding del modello vero (il ktest non poteva vederli)

1. **Preload R4** (run 1): 2 417 ensure sincroni = ~12 GiB di writeBuffer
   senza drenaggio ⇒ device perso ("A valid external Instance reference no
   longer exists"). La nota SCALA del codice lo prediceva. Fix: preload
   asincrono chunked (onSubmittedWorkDone ogni 64 expert), forward e
   prefillChunk attendono la promise al primo uso.
2. **La cascata del replay** (run 2): replay SPORCO (3 miss al layer 11)
   dopo il repair — **I3 "mai un secondo replay" FALSIFICATA**. I4 vale solo
   per il PRIMO layer sporco (input = checkpoint bit-identico); a valle
   l'hidden riparato cambia le selezioni ⇒ nuovi miss possibili. Il
   mini-modello (1 layer MoE) non ha valle; WP-0 ha selezioni fisse dalla
   traccia. Fix: **repair ITERATIVO per prefisso** — round finche' pulito,
   progresso stretto di firstDirty asserito (violazione = throw), cap nMoe,
   `repairRounds` nel dirty. Spec I3/I4 emendate, docket item 7.

### La misura (bench-glm-4090-b12-optimistic-2026-08-07.json, host quiescent)

- **decode 11.60 tok/s** (5.211 ⇒ +123%; sopra la proiezione WP-0 9.55),
  repliche 11.75/11.53/11.60; **prefill 27.45** (25.78); **TTFT 16.79 s**
  (17.88). Floor C1 ancora FAIL (13.43/56.58), gap UX 2.59x / TTFT 4.20x.
- **gpuBusy decode 39.4 ms/token (da 54.2)**: il clock recovery di it.1 e'
  reale — meno bolle, boost su. Wall 86.5 = 39.4 gpu + 15.4 stallo + 31.6
  sync/CPU (repair 15.5 dentro).
- **Identita' in produzione: token generati 64/64 IDENTICI al bench sync**
  (rebaseline4d, stesso p6, greedy, ctx 525) e 3 repliche identiche fra
  loro — chiude anche la deviazione ctx-64 dichiarata in it.3.
- **Gate strutturale: 2.188 sync(=submit)/token > 2 — FAIL DICHIARATO**
  (docket item 8, ruling PI): 1 + P(dirty) 93.8% × E[round|dirty] 1.267.
  Al tetto 2596 (sessione minima) l'aritmetica da' 1.82 PASS; su M4 ~1.0.
  Il termine eliminato: 47 → 2.19 sync/token (−95%).
- P(dirty) 93.8% vs 85.2% del sim a pari slot: gap da spiegare in fase 5
  (LruFast, ordine di preload — landmine it.2 — e cascata non modellata).

### Verifica

Verifier (ab93a715) PASS su 8/8 punti: report completo (nessun campo del
done-when assente), identita' ricalcolata indipendentemente, aritmetica slot
e item 8 ricontrollate, sync path invariato, suite 338+7 e tsc rilanciati.
Note recepite: hitRate 0% in optimistic = cambio di semantica (annotato nel
worker per le run future), projectionByK soppressa in optimistic, e —
LOAD-BEARING per la fase 5 — **replayLayers conta TUTTI i layer dal primo
sporco (densi inclusi) su denominatore nLayer: verificare l'unita' del sim
prima del confronto ±25%**.

### Prossimo pezzo

Fase 5: confronto della tassa con wp0-replay-sim allo stesso budget slot
(2417 vs 2419: delta dichiarato), unita' di replayFrac verificata, gap
P(dirty) attribuito. Non dipende dal ruling item 8.

## it.5 — 2026-08-07 — fase 5: la tassa misurata conferma il modello WP-0 (dove il modello arriva)

### Cosa

`scripts/wp0-compare.mjs` (nuovo) → `results/engine/wp0-vs-measured-2026-08-07.json`:
confronto campo per campo bench@2417 vs sim@2419 (riga piu' vicina, delta 2
slot dichiarato — q4_1-first vs riparto proporzionale). UNITA' di replayFrac
VERIFICATE identiche prima del confronto (nota it.4): sim (46-m)/47, motore
(47-l)/47 con l=m+1 ⇒ stessa definizione.

### Esito (tolleranza ±25% del contratto)

- **ENTRO (3)**: pDirty 0.852→0.938 (×1.10); **taxMsPerToken 34.5→39.2
  (×1.14)** — la formula del sim coi suoi input a gpuBusy MISURATO contro
  wall−wallClean: la bottom line del modello REGGE; tok/s 9.55→11.60 (×1.21,
  e il misurato BATTE la proiezione perche' gpuBusy e' sceso 54.2→39.4 —
  clock recovery non modellato, previsto da it.1).
- **FUORI (2), spiegazione quantitativa** (parte del done-when): missPerToken
  ×1.47 e replayGpuTerm ×1.35 — sono esattamente i termini che il sim NON
  puo' modellare (selezioni fisse dalla traccia ⇒ niente cascata, item 7).
  La spiegazione e' coerente nei numeri: moltiplicatore dei round misurato
  = replaysPerToken/pDirty_sim = 1.1875/0.852 = **1.39×**, che riportato sui
  miss da' primo-round ≈ 4.781/1.39 = 3.44 vs 3.25 del sim (+6%, dentro il
  rumore LRU/preload). L'eccesso e' TUTTO cascata, non un buco del motore
  ne' del sim: due meccanismi dichiaratamente diversi.

### Verifica

Script deterministico sui due JSON committati; numeri incrociati a mano
(replayTerm 0.852·0.665=0.567 vs 1.1875·0.646=0.767; tax misurata 86.23 −
(39.4+7.6) = 39.2). Caveat nel JSON: syncLogits 7.6 e' dell'era C3a (probe
quiescente di questa run: 0.08 ms) ⇒ la tassa misurata e' un limite
SUPERIORE; LruFast != LRU nostra; gpuBusy a scenario.

### Prossimo pezzo

Fase 6 (non-regressione a perimetro pieno, albero congelato, run lunghe) —
NOTA: le run di conformance/routing/Qwen girano sul path SYNC (non toccato),
piu' bench sync di non-regressione. Poi fase 7 (chiusura). Ruling item 8
ancora pendente (non blocca la 6).

### Addendum it.5 (nota verifier recepita, load-bearing)

Aggiunte al JSON due righe: (1) `replayFracPerRound` standalone — sim 0.665
vs misurato 0.646, ×0.97, ENTRO (la "frazione layer ripetuti" del done-when
ora e' autoportante); (2) `taxMsPerToken STRICT` con wallClean al syncLogits
MISURATO da questa run (probe 0.08 ms, non il 7.6 dell'era C3a): ×1.36,
FUORI — il delta (~7.5 ms/token) e' l'overhead CPU/event-loop che il cost
model assume "in pipeline" e che il wall reale contiene (attribution2:
sync/CPU 31.6 ms/token ne e' il contenitore). Le due letture stanno nel JSON
una accanto all'altra: la conclusione "modello confermato" vale per la
formula del sim COI SUOI TERMINI; il residuo di wall non modellato e'
dichiarato, ed e' materia di C3c (overlap/encode), non un buco del modello
di replay. Rimisura formale di syncLogits annotata per la fase 7 / C3c.

## it.6 — 2026-08-08 — fase 6 DONE: non-regressione a perimetro pieno, tutte le righe PASS (banda ±5%, ruling item 9)

### Contesto

Ruling PI in chat (2026-08-08, docket item 9): banda di non-regressione
fissata a ±5% sulle metriche di prestazione ("troppa variabilità nella
macchina di sviluppo"); ratifica c3a item 14 con larghezza emendata 2%→5%
(14b resta pendente); gate di correttezza secchi. Le 4 run notturne del
07-08 (lanciate a fine sessione precedente, mai committate) sono state
adjudicate sotto la banda nuova; le righe rimanenti (routing full-corpus,
ktest, suite, tsc) eseguite oggi a host APPENA RIAVVIATO (GPU a inizio
run: 210 MHz, 0%, 492 MiB — la quiescenza migliore mai avuta).

### Esito riga per riga (blocco non-regressione del contratto)

- decode sync **5.299 ≥ 5.211** PASS (`bench-glm-4090-b12-sync-nonreg-2026-08-07.json`, reps 5.188/5.299/5.325);
- prefill **25.01 vs 25.78 = −3.0%** PASS in banda ±5% (stessa run);
- TTFT **18.43 s vs 17.88 = +3.1%** PASS in banda (stessa run);
- Qwen K=8 **325.3 vs 326.2 = −0.3%** PASS (`bench-4090-2026-08-07T00-52-35-088Z.json`);
- golden full-corpus **98.828%** (1012/1024, b11) PASS secco, cpuref GLM
  **256/256** PASS (`conformance-glm47flash-full-nonreg-2026-08-07.json`);
- fase A: cpuref-f64 **512/512 (100%)** + golden 98.05% = firma storica
  esatta (`conformance-4090-2026-08-07T00-53-39-887Z.json`, campi
  vsCpurefPct/cpuAgree — il 502/512 è il vs-golden, identico a it.8 c3a);
- routing full-corpus **= firma 14b AL CONTEGGIO ESATTO**: prefill
  1 047 485/1 203 084 (87.0667%), decode 208 441/235 520 (88.5025%),
  router GPU set-match 1 438 591/1 438 604, pesi fuori tolleranza 0, Sel
  di produzione 0/5 754 416 difformi
  (`routing-conformance-glm47flash-full-nonreg-2026-08-08.json`, b11 come
  il riferimento 08-06; exit 4 del runner = gate legacy engine-vs-oracolo
  soglia 99, FALSE anche nel riferimento — landmine near-tie, non è il
  criterio di fase);
- ktest **69/69** PASS su GPU reale; suite **338 passed + 7 skipped**
  (≥ 337+7); `npx tsc --noEmit` pulito.

### Verifica

Albero congelato per tutta la finestra (nessun edit src; vite spento a
fine run). Nessuna riga fuori banda: la fase 6 chiude senza deroghe.
Verifier indipendente: PASS su tutti i punti di sostanza (suite e tsc
rieseguiti dal verifier stesso); artefatti committati nel commit di
questa iterazione.

### Prossimo pezzo

Fase 7 (chiusura): checklist DONE WHEN punto per punto, direction §7,
input C3c nel docket, HANDOFF, digest, merge+push. NOTA BLOCCANTE
PARZIALE: la riga "gate strutturale ≤ 2" del DONE WHEN dipende dal
ruling item 8 (2.188 a b12; opzione raccomandata = run a sessione
minima al tetto 2596, azione host del PI). La fase 7 può preparare
tutto, ma la chiusura del goal è user-gated su item 8.

## it.7 — 2026-08-08 — ruling item 8 opzione (a) ESEGUITO: gate strutturale PASS al tetto (1.891 ≤ 2), fase 4 CHIUSA

### Cosa

Ruling PI in chat ("vai con la a"): run di gate a sessione minima al tetto.
Host post-riavvio (VRAM desktop 529-545 MiB = regime "sessione minima" del
probe it.19). Budget 12.88 GiB ⇒ slot 2339+256 = **2595** (tetto nominale
2596: delta 1 slot, dichiarato — pattern it.5). Allocazione riuscita a
15 708 MiB, ~240 MiB sotto il tetto fisico 15 947, zero OOM.

### Esito (report `bench-glm-4090-btetto-optimistic-v2-2026-08-08.json`)

- **Gate strutturale: sync/token = submits/token = 1.890625 ≤ 2 PASS**
  (router syncs 0). P(dirty) 0.8125, replays/token 0.891, repair 3.70
  ms/token, miss/token 2.234. L'aritmetica della raccomandazione (1.82)
  era corretta a meno del 4%.
- Decode **16.64 tok/s** (mediana; reps 16.31/16.64/16.74, stdev 0.22) —
  sopra il floor 13.43 trasferito a C3c (qui è informazione, non gate);
  prefill 36.58, **TTFT 12.60 s** (gap UX 3.15×, da 4.47× di c3a).
- vs b12 (it.4): P(dirty) 0.938→0.813, miss/token 4.78→2.23, tassa di
  repair 15.5→3.7 ms/token — la capienza extra lavora come previsto da
  WP-0 in direzione, un po' meno in grandezza (sim: 0.65 a 2596).
- Per il ruling (a): **la fase 4 chiude con questo report; il b12
  (11.60 tok/s, 2.188) resta il punto di misura della tassa** (fase 5).

### Deviazione dichiarata (v1 non canonica, artefatto conservato)

La PRIMA run al tetto (`bench-glm-4090-btetto-optimistic-2026-08-08.json`)
era partita SENZA `--prefill-batch 1` (mio errore di invocazione): prefill
"decode-only" non canonico ⇒ decode con cache churnata dal prefill per
posizione ⇒ P(dirty) 0.922 e gate 2.203 FAIL. Protocollo corretto
identificato e dichiarato PRIMA di lanciare la v2 (trascrizione sessione);
il riferimento del gate è la v2 canonica (stesso protocollo di it.4 b12 e
del bench sync di non-regressione). FINDING registrato: la P(dirty) del
decode ottimistico è sensibile allo stato di cache lasciato dal prefill —
il prefill chunked scalda la LRU con gli expert del contesto reale;
rilevante per C3c (policy/AUTOPIN).

### Igiene

`scripts/glm-bench-run.mjs:124` crashava su `projectionByK null` (in
optimistic non ci sono sync router da proiettare) DOPO la scrittura del
report, mascherando l'exit code del gate (1 invece di 0/4): guard aggiunta
(`?? []`), fix a run concluse, albero scongelato. hostState.before della
v1 riporta powerW 588.21 a 210 MHz/0%: lettura spuria del sensore,
dichiarata (il resto della telemetria host è coerente).

### Verifica

Report v2 con hostState quiescent dichiarato e stabile (537→545 MiB);
gate PASS nel JSON (`gates.structural.pass = true`); repliche a stdev
0.22 tok/s. Verifier indipendente a valle (esito nel commit).

### Prossimo pezzo

Fase 7 — chiusura: checklist DONE WHEN punto per punto, direction §7,
input C3c nel docket, HANDOFF, digest, push da goal verificato. Con item
8 risolto non restano gate pendenti sul contratto C3b.

## it.8 — 2026-08-08 — fase 7: CHIUSURA — checklist DONE WHEN 7/7 PASS

### Checklist DONE WHEN del contratto v2, punto per punto

1. **Spec scritta e registrata** — PASS.
   `docs/superpowers/specs/2026-08-07-engine-fase-c3b-decode-ottimistico.md`
   (it.1, docket item 5, verifier 7/7 punti); emendamenti registrati: item 6
   (precondizione a TOTALE 0.80, q4_1-first), item 7 (repair iterativo per
   prefisso, I3/I4). Nessun gate/soglia toccato senza ruling.
2. **Gate strutturale sync/token ≤ 2 e submits/token ≤ 2 in produzione** —
   PASS (ruling item 8, opzione a): al tetto 2595 slot **1.890625 ≤ 2**
   (`bench-glm-4090-btetto-optimistic-v2-2026-08-08.json`, hostState
   quiescent dichiarato, confronto coi 47 sync C3a nel JSON); al punto
   della tassa b12: 2.188 (dichiarato, resta il riferimento della tassa).
   Il termine eliminato: 47 → 2.19/1.89 sync/token (−95/−96%).
3. **Identità del meccanismo** — PASS. ktest `glm-optimistic-identita`
   (it.3): 64 posizioni, 3 replay forzati con rientro dal checkpoint,
   hidden+logits BIT-identici 131072/131072 vs sincrono, caso di rifiuto
   precondizione; in produzione 64/64 token identici al sync (it.4).
4. **Tassa di replay vs proiezione WP-0** — PASS (it.5,
   `wp0-vs-measured-2026-08-07.json`): tassa ×1.14, pDirty ×1.10,
   replayFrac ×0.97, tok/s ×1.21 ENTRO ±25%; miss ×1.47 e replayTerm
   ×1.35 FUORI con spiegazione quantitativa nel journal (cascata, round
   1.39× — parte del done-when: modello confermato dove modella).
5. **Niente gate tok/s; doppio livello sui numeri in ogni report** — PASS.
   Tutti i bench riportano decode/prefill/TTFT, floor C1 e gap UX (campo
   `objective`); al tetto decode 16.64 (floor C3c 13.43 gia' battuto li'),
   gap UX 30 tok/s = 1.80×, TTFT 12.60 vs 4 s = 3.15×.
6. **Non-regressione permanente** — PASS (it.6, tutte le righe, banda ±5%
   ratificata docket item 9 che sostituisce la provvisoria item 3):
   decode sync 5.299 / prefill 25.01 (−3.0% in banda) / TTFT 18.43 (+3.1%
   in banda) / Qwen 325.3 / golden 98.828% / cpuref 256/256 + 512/512 /
   routing = firma 14b ai conteggi esatti / ktest 69/69 / suite 338+7 /
   tsc pulito.
7. **Chiusura** — PASS (questa iterazione): docket c3b aggiornato (item 8
   risolto con esito; nessun item aperto che blocchi); direction §7 riga
   C3b CHIUSA coi numeri; ideas-ledger NON toccato dal goal (nessun
   obbligo); HANDOFF refresh; input C3c fissati in
   `.harness/goals/engine-fase-c3c/docket.md` item 2 (profilo con tassa a
   b12 e tetto, hit-rate LRU b11/b12, landmine i-v, banda ±5%).

### Stato finale

Il goal consegna: `select:"optimistic"` in produzione (spec-first, 5
invarianti con assert), repair iterativo per prefisso con identita'
bit-esatta, telemetria della tassa, e i numeri: **decode 5.211 → 11.60
(b12) / 16.64 (tetto), TTFT 17.88 → 12.60 s, 47 → 1.89 sync/token**.
Restano fuori-goal (gia' docketate altrove): ratifiche c3a 14b/2/19/20/21;
igiene test:conformance e BASE_URL; goal stale fase-1b/fase-2.

### Prossimo pezzo

C3c (paging in scarsita'): contratto chartered, input fissati, parte con
il WP banda fredda. Apertura su /goal del PI (pre-avvio item 2 ESEGUITO;
tag goal-engine-fase-c3c-start alla rilettura del contratto).
