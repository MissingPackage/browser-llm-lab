GOAL: engine-fase-c1 — L'oracolo desktop (llama.cpp strumentato, contatori-only)
produce le misure che dimensionano il paging degli esperti su GLM-4.7-Flash:
recall del router lookahead, skew/heat/working-set del routing, e curve hit-rate
vs budget da simulazione trace-driven delle policy candidate — i numeri entrano
nel docket come input della spec C2, nessun meccanismo browser viene costruito.

<!-- CONTRATTO v1 — approvato dal PI 2026-07-30 in chat ("Per me tutto ok.
     Scarica pure GLM 4.7 quando ti serve. Andiamo avanti"). Split della fase C
     di direction §7 approvato col contratto: C1 = oracolo (questo goal),
     C2 = esecuzione GLM (MoE+MLA) nel motore, C3 = paging+instant-on;
     hero-demo M4 resta PI-gated per hardware. Le assunzioni [ASSUMED] del
     draft sono state accettate senza correzioni.
     Goal start tag: goal-engine-fase-c1-start. -->

DONE WHEN (all measurable):
- Spec C1 scritta (docs/superpowers/specs/<data>-engine-fase-c1-design.md) e
  ruling PI di approvazione registrato nel docket. La spec fissa: metodo di
  strumentazione (patch counters-only su checkout llama.cpp dedicato — il tap
  hidden post-attention di L per il router di L+1 è il punto tecnico da
  progettare; via alternativa solo se motivata), corpus (numero prompt, domini,
  lunghezze [provvisorio: ≥8 prompt chat-templated eterogenei — codice, prosa,
  matematica, multilingua — ≥16k posizioni routed totali tra prefill e decode]),
  definizioni esatte delle metriche (recall@K per K∈{4,6,8}, baseline "stessi
  expert del token precedente", skew cumulativo, working-set a finestra),
  policy da simulare e sanity-gate numerici.
- Traccia di routing riproducibile: patch/tool committati e run con exit 0 che
  scrive in results/engine/moe-oracle/ la traccia (posizione, layer, top-4 id)
  sul corpus fisso; il report registra SHA-256 del GGUF, n_threads pinnato,
  greedy, versione/commit di llama.cpp (pattern determinismo di gen-golden.py
  esteso).
- Recall lookahead misurato: report JSON con recall PILOT-style aggregato e
  per-layer, PIÙ la baseline token-precedente sulla STESSA traccia; sanity-gate
  della spec verde (conteggi la_tot coerenti con posizioni × layer routed —
  46 layer MoE attesi, direction §3).
- Statistiche di residenza dalla stessa traccia: JSON con distribuzione usage
  per (layer, expert), curva cumulativa top-N, working-set per finestra
  (co-attivazione stile COUPLE inclusa solo se gratis dal tracciato).
- Simulatore trace-driven committato con test verdi: curva hit-rate vs budget
  expert-cache per almeno {LRU, LFRU tier.h, LFRU+pin-da-usage,
  LFRU+pin+prefetch al recall misurato}, output JSON; evidenziato il punto
  di lavoro 4090 (budget = VRAM − residente non-expert, dalla spec) e
  la lettura "modello ~2× la memoria" confermata o corretta (ledger §A).
- Decisione PREPARATA, non presa: docket aggiornato con proposta go/no-go PILOT
  per C2 (col numero misurato e il costo-miss da results/opfs-bench),
  ledger §A/§I e direction §7 aggiornati dove resi stale dalle misure.

EVIDENCE OF DONE: run documentata del tool oracolo con exit 0 (comando esatto in
spec); JSON in results/engine/moe-oracle/ (traccia o suo digest, recall,
residenza, simulazione); test del simulatore verdi (npm test, o uv run pytest se
la spec sceglie Python); diff di spec + docket + ledger + direction.

AUTHORITY GRANTED:
- may do autonomously: branch engine/fase-c1 e commit/push sul branch; creare
  tools/oracle-moe/** (o scripts/), tests/**; checkout + patch + build LOCALE di
  llama.cpp per sola misura (fuori dal repo o vendored come patch-file, mai
  linkato al motore); download GGUF GLM-4.7-Flash q4 (~17 GB disco, nessuna
  spesa — autorizzato esplicitamente dal PI 2026-07-30) in ~/.cache/blab-models;
  run locali su 4090; aggiornare docs/engine/* quando stale (ruling 2026-07-29);
  docket/HANDOFF refresh; merge su main + push a goal CHIUSO e verificato
  (ruling permanente 2026-07-29).
- must docket (never do): QUALSIASI modifica a src/engine/** (il meccanismo —
  kernel MoE/MLA, slab, tier, AUTOPIN, PILOT-real, instant-on — è C2/C3);
  cambio modello-tesi o direction §3; rimandi §I della prefix-cache
  (longest-prefix, eviction a densità, logits-nel-checkpoint) — restano
  docketed per il goal browser; benchmark pubblico (contributo separato,
  ruling 2026-07-30); run che richiedono M4/S22; ogni spesa; delete di codice
  committato >30gg; ESERCIZIO della clausola di fallback (sotto): proposta via
  docket, decide il PI.

CONSTRAINTS: llama.cpp SOLO oracolo — la patch di misura non entra mai nel
runtime del motore; contatori zero-overhead se spenti e provabilmente isolati
(lezione colibri, direction §8.6: mai numeri da run strumentate nei confronti
pubblici); determinismo e identità del modello nel report (SHA-256, greedy,
thread pinnati); zero attribution AI nei commit; spec-first prima del codice;
confronti solo a parità di corpus/contesto (lezione B2). Clausola di fallback
pre-negoziata: se il tap hidden-state in llama.cpp sfora il timebox fissato in
spec, C1 chiude con ROUTE_TRACE-only (baseline, skew, working-set, simulazione
senza prefetch) e il recall lookahead va a docket con la via alternativa —
decisione PI, non unilaterale.

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md; docs/engine/direction.md §3 (config GLM verificata:
47 layer/primo denso, 64 routed top-4 + 1 shared, expert ≈5.3 MB q4, ~2944
expert ≈15-16 GB) e §7 fase C; docs/engine/study/colibri.md §1 (LOOKA/PILOT:
71.6% vs baseline 41.3% su GLM-5.2 — proprietà del modello, da rimisurare) e §2
(eusage/eheat/AUTOPIN, tier.h); docs/engine/study/ds4.md (instant-on, quant
asimmetrica); docs/engine/ideas-ledger.md §A (riga paging) + §I;
scripts/gen-golden.py (pattern oracolo riproducibile); results/opfs-bench/
(costo-miss: warm ~gratis, cold disco-bound).
