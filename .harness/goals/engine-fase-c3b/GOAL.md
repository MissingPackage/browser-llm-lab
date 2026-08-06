GOAL: engine-fase-c3b — Il decode ottimistico con repair esatto elimina i 46 sync
router per token nel regime near-total (residenza >= ~88%): 1 submit/token nel path
pulito, replay esatto dal primo layer sporco sui token dirty, qualita' bit-invariata,
e la tassa di replay misurata sul device sta dentro la proiezione WP-0.

<!-- CONTRATTO v2 — 2026-08-07. Sostituisce il contratto chartered 2026-08-01, che
     copriva il paging in scarsita': quel perimetro e' ora engine-fase-c3c
     (.harness/goals/engine-fase-c3c/GOAL.md). Split e decisioni di gate presi su
     delega esplicita del PI ("scegli tu su questi punti e poi procedi con il
     loop", 2026-08-07) e registrati nel docket di questo goal, item 2.
     Numeri di non-regressione fissati da .harness/goals/engine-fase-c3a/docket.md
     item 21. Dimensionamento del meccanismo: WP-0 (journal c3a it.20 +
     results/engine/moe-oracle/wp0-replay-sim-2026-08-06.json).
     Tag di avvio: goal-engine-fase-c3b-start. -->

DONE WHEN (all measurable):
- Spec scritta (docs/superpowers/specs/2026-08-07-engine-fase-c3b-decode-ottimistico.md)
  e REGISTRATA a docket (pattern plan-check pre-autorizzato, c3a item 3, + ruling
  decide-dont-escalate 2026-08-03: il meccanismo e' del loop, al PI funzione
  obiettivo e gate — ruling PI bloccante SOLO se la spec tocca gate o soglie di
  questo contratto). La spec fissa: semantica del flag di miss (piggyback sul
  readback logits, scritto dal kernel); checkpoint hidden per token (376 KB);
  replay dal primo layer sporco; inserimento in slotTable DIFFERITO al confine di
  token (dopo il submit la slotTable e' intoccabile — semantica WP-0); prefill
  sincrono invariato; criterio di rifiuto esplicito quando la residenza scende
  sotto la precondizione del regime ottimistico. NON include: predictor GPU al
  confine di token (falsificato 3 volte: WASTE, K3, WP-0) ne' repair batched.
- Gate STRUTTURALE (il gate formale della fase 4 di C3a, finalmente misurabile in
  produzione): nel bench glmbench quiescente i contatori dello schema unico
  riportano sync/token <= 2 e submits/token <= 2 in media steady-state sul ramo
  decode (path pulito: 1 submit / 0 sync; token dirty: +1 replay), con hostState
  dichiarato nel JSON. Il confronto dichiarato nel report e' coi 46 sync/token
  di chiusura C3a.
- Identita' del meccanismo: test verde che confronta decode ottimistico e decode
  sincrono sulla STESSA sequenza con argmax identico su tutte le posizioni
  (pattern del test M=1 vs M>1 della fase 5 C3a), incluso almeno un caso con miss
  FORZATO che attiva il replay e un caso di rifiuto della precondizione.
  [ASSUMED: >= 64 token decodificati, ctx ~500, su GPU reale]
- Tassa di replay misurata contro la proiezione: JSON in results/engine/ con
  P(dirty) osservato, miss/token, frazione media di layer ripetuti e ms/token di
  tassa, confrontati con wp0-replay-sim-2026-08-06.json allo stesso budget slot.
  Entro [ASSUMED +/-25%] il modello e' confermato; oltre, la spiegazione dello
  scostamento nel journal e' parte del done-when (non un FAIL automatico: un
  modello smentito con spiegazione e' un esito, un numero non confrontato no).
- NIENTE gate tok/s in questo goal (decisione docket item 2: WP-0 proietta 11.3
  tok/s al tetto misurato 2596 coi kernel di oggi — un gate 13.43 qui sarebbe di
  nuovo hardware, non struttura; il floor 13.43 passa a C3c con clausola
  pre-negoziata). OGNI bench riporta comunque decode/prefill/TTFT, il confronto
  col floor C1 (13.43/56.58) e il gap dalla soglia UX (30 tok/s decode, ~60 in
  thinking, TTFT <= 4 s) — l'assenza di quei numeri dal report e' un FAIL di
  checklist (ruling c3a item 1, doppio livello sui numeri).
- Non-regressione permanente (numeri di chiusura C3a, docket c3a item 21), con la
  banda di rumore proposta in c3a item 14 applicata IN VIA PROVVISORIA (docket
  item 3 di questo goal — pendente ratifica): regressione = mediana nuova sotto
  il riferimento oltre il 2% OPPURE sotto in modo statisticamente significativo
  a livello di repliche. Riferimenti tok/s: decode 5.211, prefill 25.78, TTFT
  17.88 s a ctx ~500, Qwen 326.2. ESATTI (nessuna banda): argmax == cpuref-f64
  256/256 (gateCpuref JSON) e 512/512 fase A, top-1 vs golden >= 98.828%
  full-corpus, routing full-corpus = firma docket c3a item 14b, ktest tutti PASS
  (>= 65), npm test senza regressioni (>= 337 passed + 7 skipped),
  npx tsc --noEmit pulito.
- Chiusura: docket aggiornato, direction §7 riga C3b coi numeri, ideas-ledger se
  toccato, HANDOFF refresh, e input di non-regressione per C3c fissati (pattern
  c3a item 21: profilo ms/token con la tassa dentro, residenza, landmine).

EVIDENCE OF DONE: file di spec + entry di registrazione nel docket; run glmbench
quiescente con JSON committato in results/engine/ che contenga sync/token,
submits/token, P(dirty), miss/token, tassa ms/token, decode/prefill/TTFT, gap UX e
hostState; JSON di confronto con wp0-replay-sim-2026-08-06.json; test di identita'
ottimistico-vs-sincrono verde in npm test (nome del file nel journal); run
scripts/conformance-engine.mjs (HEADED, BASE_URL del motore) con gateCpuref
256/256; conformance full-corpus con golden >= 98.828% (protocollo b11); routing
full-corpus con firma item 14b; scripts/ktest-run.mjs tutti PASS; bench Qwen di
non-regressione; npm test + npx tsc --noEmit; diff di docket + direction +
HANDOFF (+ ledger se toccato).

AUTHORITY GRANTED:
- may do autonomously: src/engine/**, tests/**, tools/**, scripts/**,
  docs/engine/** quando stale; commit e push su main a iterazione VERIFICATA;
  merge su main + push a goal CHIUSO e verificato; run locali sulla 4090;
  re-import e invalidazione degli artefatti in ~/.cache; PIU' l'implementazione
  del decode ottimistico, del repair, del checkpoint hidden e della precondizione
  di residenza — qui sono l'oggetto del goal.
- must docket (never do): abbassare o riformulare QUALSIASI soglia di questo
  contratto; merge con qualsiasi metrica misurata in regressione (oltre la banda
  provvisoria di item 3); costruire un predictor GPU al confine di token o il
  repair batched senza ruling che riapra WP-0; cambio modello-tesi, quant diversa
  da Q4_0, direction §3; testa MTP e spec-dec (fase D); benchmark pubblico
  (ruling 2026-07-30); run M4/S22 e hero-demo (PI-gated per hardware); ogni
  spesa; delete di codice committato >30gg senza check git log --follow.

CONSTRAINTS: spec-first; NON-REGRESSIONE PERMANENTE (ruling PI 2026-07-31); bench
SOLO a macchina quiescente (norma PI 2026-08-01) con hostState dichiarato nel
report; ~60 s fra run GPU consecutive sullo stesso profilo (OPFS handle +
rilascio VRAM, di piu' dopo un errore — landmine c3a item 21); albero CONGELATO
durante le run lunghe (lezione it.16: un edit di src/engine/** con vite HMR
attivo uccide il full-corpus); parita' di corpus/contesto/protocollo, conf full
= b11; llama.cpp solo oracolo; f32-first sul dev-loop; mai gate diretti
engine-vs-oracolo su selezioni near-tie; telemetria zero-overhead se spenta;
determinismo nei report; runner in background senza pipe su tail/grep; zero
attribution AI nei commit.

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md §1 e §5; .harness/goals/engine-fase-c3a/docket.md
item 21 (input C3b) + item 17a (clausola di chiusura C3a) + item 14/14b (banda di
rumore e firma routing); .harness/goals/engine-fase-c3a/journal.md it.20 (WP-0, i
numeri di questo contratto) e it.17 (select:"gpu", 1 submit/0 sync in ktest);
results/engine/moe-oracle/wp0-replay-sim-2026-08-06.json;
src/engine/{residency,page,expertstore,decodebatch,telemetry,glmmodel}.ts;
docs/engine/direction.md §1, §2, §2-bis, §7;
docs/engine/study/2026-08-06-waste-kimik3-streaming.md.
