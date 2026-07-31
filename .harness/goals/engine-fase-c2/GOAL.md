GOAL: engine-fase-c2 — Il motore esegue GLM-4.7-Flash (MoE+MLA, GGUF deepseek2
Q4_0) end-to-end su 4090 con conformance verificata contro l'oracolo llama.cpp
(logits e routing), con una residenza esperti MINIMA (senza prefetch/tier/pin —
quelli sono C3) e telemetria nativa che riporta i numeri che C3 userà.

<!-- EMENDAMENTO 2 (ruling PI 2026-07-31 in chat, docket item 4 opzione a):
     il punto "Conformance routing" del DONE WHEN passa da gate a soglia a
     MISURA INFORMATIVA (report in results/engine/); il gate di correttezza
     MoE è il doppio gate logits full-model (fase 6). La "soglia di spec"
     cui il punto rimanda è emendata in spec §7 con analisi e contingenza
     (oracolo f32 rigenerabile on-demand). Evidenza: it.9 + discriminatore
     cpuref-f64 (router scagionato). -->

<!-- CONTRATTO v1 — approvato dal PI 2026-07-31 in chat ("Tutto approvato, ma
     per i tok/s dobbiamo assicurarci che rimangano almeno pari a quelli
     registrati in C1. Andando avanti con le implementazioni dobbiamo sempre
     assicurarci che le metriche che abbiamo non peggiorino").
     Emendamenti recepiti rispetto al draft:
     (1) il punto "Decode E2E" passa da misura-senza-gate a GATE di
         non-regressione vs oracolo C1 (decode >=13.4 tok/s, prefill >=56.6);
     (2) principio permanente di non-regressione delle metriche misurate,
         in CONSTRAINTS (vale anche oltre questo goal — salvato in memoria).
     Gli altri [ASSUMED] del draft sono stati accettati senza correzioni.
     Goal start tag: goal-engine-fase-c2-start. -->

DONE WHEN (all measurable):
- Spec C2 scritta (docs/superpowers/specs/<data>-engine-fase-c2-design.md) e
  ruling PI di approvazione registrato nel docket. La spec fissa: subset GGUF
  deepseek2 validato (chiavi/tensori richiesti, exit su mismatch); formulazione
  MLA scelta (si parte dalla naive/decompressa; la variante absorbed va in spec
  come decisione esplicita — lineage KV ds4/colibri); design kernel MoE (router
  sigmoid+bias+top-4+norm replicato da C1, GEMV dequant-fusa per-expert, shared
  expert nel path denso); meccanismo di residenza minima (staging dell'intero
  parco routed in RAM + cache VRAM LRU on-demand a slot da 5.33 MB, ~2.573
  slot; niente prefetch, niente pin, niente OPFS — il path di load è però
  progettato per accettare il prefetch di C3, ruling docket C1 item 4);
  soglie di conformance; corpus e sanity-gate.
- Load GGUF: il reader del motore carica il GLM-4.7-Flash Q4_0 di C1 (stesso
  SHA-256 d0bbdfc…) con validazione hard ed exit 0; test verdi sul parsing
  (npm test).
- Conformance logits vs oracolo: golden logits llama.cpp (pattern
  gen-golden.py, stesso commit 5f55650 di C1) su corpus fissato in spec;
  report JSON in results/engine/ con KL/top-k entro le soglie di spec
  (stesse metriche e harness di conformance della fase A).
- Conformance routing: sul corpus della traccia C1, i top-4 expert id per
  (posizione, layer) del motore coincidono con la traccia oracolo
  (results/engine/moe-oracle/trace-2026-07-31.jsonl.gz) sopra la soglia di
  spec (target >=99% in greedy/f32; taratura fine in spec, non qui — il
  routing è il segnale da cui vive il prefetch di C3).
- Decode E2E su 4090 — GATE di non-regressione (emendamento PI 2026-07-31):
  report bench JSON con tok/s, dispatch/token, hit-rate della cache VRAM
  minima e occupazione; **decode >= 13.4 tok/s e prefill >= 56.6 tok/s**
  (= oracolo llama.cpp CPU-only di C1,
  results/engine/moe-oracle/llama-bench-glm47flash-q4_0-2026-07-30.json,
  a parità di condizioni: greedy, contesto della spec). Il confronto coi
  30 tok/s della funzione obiettivo resta informativo per C3 e va a docket.
- Non-regressione del pregresso: a fine goal la conformance fase A e il bench
  first-light Qwen2.5-0.5B restano verdi e >= agli ultimi valori committati in
  results/engine/ (bench e conformance 2026-07-30); una regressione non
  assorbibile va a docket, non si merge.
- Chiusura: docket aggiornato con gli input preparati per C3 (numeri di
  residenza minima osservati vs simulatore C1, costo reale dell'upload
  RAM→VRAM per expert), ledger/direction aggiornati dove resi stale,
  HANDOFF refresh.

EVIDENCE OF DONE: npm test verde + tsc --noEmit pulito; run conformance con
exit 0 e JSON in results/engine/ (logits e routing, comandi esatti in spec);
bench JSON committato con i due gate tok/s soddisfatti + bench Qwen di
non-regressione; diff di spec + docket + ledger + direction + HANDOFF.

AUTHORITY GRANTED:
- may do autonomously: commit/push su main a fine iterazione VERIFICATA
  (ruling PI 2026-07-31, docket item 3 — ratifica del main-diretto; supera
  la riga originale "branch engine/fase-c2");
  MODIFICARE src/engine/** (è l'oggetto del goal: reader deepseek2, kernel
  MLA/MoE, residenza minima, telemetria); tests/**, tools/**; riuso del GGUF
  già in ~/.cache/blab-models e del checkout llama.cpp di C1 per i golden
  (nessun download nuovo, nessuna spesa); run locali su 4090; aggiornare
  docs/engine/* quando stale (ruling 2026-07-29); docket/HANDOFF refresh;
  merge su main + push a goal CHIUSO e verificato (ruling permanente).
- must docket (never do): meccanismo di paging oltre la residenza minima
  (prefetch, tier.h, AUTOPIN, pin, OPFS-backed experts, instant-on = C3);
  WP banda fredda browser (ruling item 4: sta in C3); testa MTP/spec-dec
  (assente nel GGUF deepseek2 — docket C1 item 2, fase D); cambio
  modello-tesi, quant diversa da Q4_0, o direction §3; benchmark pubblico
  (ruling 2026-07-30); run M4/S22; ogni spesa; delete di codice committato
  >30gg; merge con QUALSIASI metrica misurata in regressione (vedi
  CONSTRAINTS) — la deroga è decisione PI via docket.

CONSTRAINTS: spec-first prima del codice; llama.cpp SOLO oracolo (mai linkato
al motore); f32-first sul dev-loop (Chrome/Linux senza shader-f16), percorso
f16 dietro feature-detect; tap hidden-states preservati nel forward (direction
§4.4 — retrofit vietato per design); telemetria zero-overhead se spenta;
determinismo nei report (SHA-256, greedy, commit oracolo); confronti a parità
di corpus/contesto (lezione B2); zero attribution AI nei commit;
**NON-REGRESSIONE PERMANENTE (ruling PI 2026-07-31)**: ogni implementazione
ricontrolla le metriche già misurate prima del merge — tok/s, conformance,
hit-rate: nessuna peggiora rispetto all'ultimo valore committato in results/,
salvo deroga PI a docket; il target ≤100 dispatch/token della fase B non si
eredita come gate su GLM (47 layer ⇒ floor architetturale diverso; si misura
e si riporta).

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md; docs/engine/direction.md §3 (config GLM
verificata: 47 layer/primo denso, 64 routed top-4 + 1 shared, MLA kv_lora
512 + rope 64, 1.152 B f16/layer/token, expert 5.33 MB q4) e §7 fase C;
.harness/goals/engine-fase-c1/docket.md (item 2: arch deepseek2 senza NextN;
item 4 RISOLTO: GO prefetch; item 5: matrice usage ricalcolabile);
docs/superpowers/specs/2026-07-30-engine-fase-c1-design.md (replica esatta
del router); src/engine/{gguf,gpuforward,shape,quant}.ts (stato attuale del
motore); scripts/gen-golden.py; results/engine/moe-oracle/ (traccia + sim +
llama-bench = floor tok/s); docs/engine/study/{colibri,ds4}.md (lineage KV
MLA).
