# Docket — engine-fase-c2 (decisioni PI pendenti)

1. ~~**PLAN-CHECK**~~ RISOLTO (2026-07-31, ruling PI in chat: "Approvo,
   parti"). Decomposizione approvata senza modifiche; iterazione 1 = fasi
   1+2 in parallelo. Testo originale:

   **PLAN-CHECK** (iteration 0, 2026-07-31): approvazione di PHASES.md prima
   dell'iterazione 1. Il contratto GOAL.md è già approvato in chat (con
   emendamento non-regressione); qui si approva la decomposizione: 7 fasi,
   gruppo parallelo A (floor+golden ‖ spec), fasi 3-6 gated dal ruling di
   spec, gate tok/s hard in fase 6 (decode ≥13.4 / prefill ≥56.6 = oracolo
   CPU C1) + non-regressione Qwen.

2. ~~**RULING RICHIESTO — spec C2**~~ RISOLTO (2026-07-31, ruling PI in chat:
   "Approvo (a)-(f)", dopo walkthrough delle decisioni rispetto all'obiettivo
   finale — intelligenza max sopra soglia, Pareto per device). Spec approvata
   senza modifiche; fasi 3-6 SBLOCCATE (3 ready, 4-6 a cascata). Testo
   originale:

   **RULING RICHIESTO — spec C2** (2026-07-31, fase 2): approvazione di
   `docs/superpowers/specs/2026-07-31-engine-fase-c2-design.md`. Decisioni
   proposte da ratificare (dettaglio in spec §10):
   (a) il GGUF "Q4_0" è a quant MISTA (FINDING dal dump, CORRETTO dopo il
       verifier it.1: ffn_down_exps Q4_1 SOLO su blk.1-4 = 256 expert, il
       resto Q4_0; shexp Q5_K/Q6_K, output Q6_K, proiettori MLA Q8_0,
       ffn_down denso Q4_1) ⇒ si implementano i kernel dequant
       Q4_1/Q5_K/Q6_K; NIENTE repack (la requant romperebbe la conformance
       con l'oracolo sullo stesso file); slab expert a DUE size-class
       esatte (5.308.416 / 5.505.024 B — media pesata = il 5.325.512 di
       residency-sim C1);
   (b) MLA in formulazione ABSORBED nel motore (cache 576/token/layer =
       54 KB/token f16, quella di direction §3); la naive solo in cpuref
       f64 — supera l'[ASSUMED naive-first] del contratto (la naive ha una
       cache 18× e non sta nel budget);
   (c) residenza minima: il GGUF vive in OPFS (copiato al primo load) e i
       miss expert leggono da lì — DEVIAZIONE dichiarata dalla lettera del
       contratto ("staging RAM", "OPFS-backed experts" a must-docket):
       15.6 GB di ArrayBuffer in un tab su host 31 GB non sono credibili;
       niente policy/tier/prefetch su OPFS (quello resta C3);
   (d) policy cache VRAM = LRU pura (a budget 87% il simulatore C1 dà
       96.4% decode hit; tuned/prefetch sono C3);
   (e) soglie: conformance argmax ≥99% vs cpuref-f64 E ≥97% vs golden
       (gate doppio, landmine oracolo-q8); routing set-match ≥99% vs
       traccia C1; bench = protocollo B2;
   (f) timebox: 4 iterazioni fase 4 (MLA) + 4 fase 5 (MoE+residenza);
       sforamento ⇒ docket con analisi.

Costanti di gate (fase 1 le formalizza nel report):
- Floor tok/s da `results/engine/moe-oracle/llama-bench-glm47flash-q4_0-2026-07-30.json`:
  decode 13.43 ±0.10 tok/s (n_gen 64), prefill 56.58 ±3.74 tok/s (n_prompt 512),
  llama.cpp 5f55650 CPU-only, i9-14900HX 16 thread.

3. ~~**RULING RICHIESTO — branch policy del goal**~~ RISOLTO (2026-07-31,
   ruling PI in chat: "Ratifica main-diretto" = opzione (a)). C2 continua su
   main con push a fine iterazione verificata; riga di authority del GOAL
   aggiornata. Testo originale:

   **RULING RICHIESTO — branch policy del goal** (2026-07-31, post-verifier
   it.2): GOAL.md §AUTHORITY dice "branch engine/fase-c2 e commit/push sul
   branch; merge su main a goal CHIUSO" (pattern C1), ma il goal ha lavorato
   finora su main direttamente (da it.0: scaffold, spec, fase 1, fase 3 —
   tutto già pushato, ruling permanenti push/merge rispettati nello spirito).
   Opzioni: (a) ratificare main-diretto per C2 (fix della riga di authority);
   (b) da fase 4 in poi si lavora su branch engine/fase-c2, merge a goal
   chiuso (lettera del contratto). Il verifier segnala; decide il PI.

4. ~~**RULING RICHIESTO — taratura del gate di conformance routing**~~
   RISOLTO (2026-07-31, ruling PI in chat: "ok andiamo con A. Se in fase 6
   dovessimo avere dei dubbi e per qualche motivo ci servisse di nuovo un
   oracolo, lo faremmo rigirare"). OPZIONE (a) ADOTTATA: routing = misura
   informativa; gate di correttezza = doppio gate logits full-model di
   fase 6. CONTINGENZA registrata: oracolo f32 + traccia nuova rigenerabili
   on-demand se fase 6 solleva dubbi (ex opzione b). Spec §7 emendata;
   fase 5 CHIUSA (timebox sanato dallo stesso ruling); fase 6 SBLOCCATA.
   Testo originale:

   **RULING RICHIESTO — taratura del gate di conformance routing** (2026-07-31,
   it.9, fase 5 slice 3b; timebox fase 5 esaurito a 4 it. ⇒ questo item vale
   anche come docket-con-analisi da ruling (f) della spec).

   **Fatto misurato**: il replay teacher-forced della traccia C1 sul motore
   (47 layer, pesi reali da OPFS, harness `glmroute` nuovo) dà set-match
   top-4 SOTTO la soglia ≥99% di spec §7: prompt 4 completo (1.004 pos) —
   prefill 90.80%, **decode 85.85%** (29.440 coppie posizione×layer);
   report `results/engine/routing-smoke-p4-2026-07-31.json`.

   **Debug eseguito (spec §7: "sotto soglia ci si ferma e si debugga il
   router")** — il router è SCAGIONATO con l'evidenza più forte disponibile:
   sul medesimo subset (prompt 4, prime 16 pos, 736 coppie) il full-model
   **cpuref f64 ESATTO** (naive, zero GPU, zero codice condiviso col motore)
   concorda con la traccia al **96.20% — IDENTICO al motore GPU, e i 28
   mismatch sono le stesse identiche coppie (pos, layer) 28/28**
   (`results/engine/routing-cpuref-analysis-2026-07-31.json`, harness
   `tests/analysis-route-cpuref.test.ts`). Firma coerente: mismatch a UN
   solo elemento (swap 4°/5°), ~2% già al layer 1, degrado liscio con
   profondità e lunghezza contesto. ⇒ Il disaccordo è tra ARITMETICA ESATTA
   e la numerica dell'oracolo CPU (attivazioni quantizzate q8 — la landmine
   già nota da C1) sui near-tie del top-4; la soglia 99% era tarata
   sull'autotest C1 (recall 0.999 usando l'hidden DELL'ORACOLO), che non
   copre il replay engine-vs-oracolo.

   **Opzioni per il PI**:
   (a) [RACCOMANDATA] La conformance routing diventa MISURA informativa
       (report + input C3), il gate di correttezza resta il doppio gate
       logits full-model di fase 6 (argmax ≥99% vs cpuref-f64 — che il
       discriminatore mostra essere il confronto giusto — e top-1 ≥97% vs
       golden, che già assorbe il drift oracolo per costruzione). Per il
       prefetch C3 conta comunque il routing REALE del motore, non quello
       dell'oracolo. Costo: emendamento spec §7.
   (b) Ritarare il gate routing con un oracolo a numerica confrontabile
       (build llama.cpp con attivazioni f32 pure) e rigenerare la traccia.
       Costo: build+trace nuovi (~1 run oracolo), rischio che la fuzziness
       ai near-tie resti (f32 GPU vs f32 CPU ordering).
   (c) Mantenere il gate com'è ⇒ FAIL formale della fase 5 e stop del goal.
   In (a) e (b): la fase 5 resta "done salvo gate" — kernel, residenza,
   forward e harness sono verificati (it.6-9, verifier PASS ciascuna).

   Numeri di contesto per la decisione: hit-rate residenza nel replay p4 =
   95.9% a 12 GiB di slab (2.216+203 slot, 82% del parco) — coerente col
   simulatore C1; throughput replay 3,7 pos/s (46 sync/token + miss, da
   confrontare col path bench fase 6 che non fa readback per-layer... il
   readback del ROUTING sì, è strutturale).

   Addendum post-verifier (stessa data): corroborazione su prompt 7 completo
   (891 pos): decode 94.11%, prefill 93.05%
   (`results/engine/routing-smoke-p7-2026-07-31.json`) — la varianza tra
   prompt (decode 85.8%↔94.1%) è coerente col meccanismo near-tie (la
   densità dei pareggi dipende dal contenuto) e INCOERENTE con un bug
   sistematico. Nota di dicitura recepita dal verifier it.9: il
   discriminatore ha "zero codice DI CALCOLO condiviso" col motore (parsing
   GGUF/indice/dequant CPU sono condivisi ma scagionati dalla firma — il
   motore usa dequant WGSL indipendenti, concordi a L2rel 2.35e-7 da it.5).

5. **NOTA PER LA CHIUSURA GOAL — copertura campionaria del gate (i)**
   (2026-08-01, it.10, segnalazione verifier): il gate (i) argmax vs
   cpuref-f64 è valutato su 2/8 prompt (p7 + p4 = 256 posizioni golden,
   risultato 256/256 = 100%) e non sull'intero corpus, per costo CPU del
   riferimento f64 (54 min p7, ~2.7h p4; gli 8 prompt ≈ 1-2 giorni CPU).
   Difendibilità del campione: include p4 (il prompt PEGGIORE al routing),
   su entrambi i prompt la divergenza dal golden è lo STESSO TOKEN per
   motore e cpuref (firma q8), e il gate poggia su layer-level a pesi reali
   (it.5) + discriminatore esaustivo it.9. Il PI ratifica o chiede
   estensione alla chiusura del goal (la contingenza oracolo-f32 resta
   disponibile). Correlata: p3 è a 96.87% per-prompt sul gate (ii) — il
   gate di spec è AGGREGATO (98.83% PASS); se si volesse un gate
   per-prompt, p3 sarebbe l'unico sotto.
