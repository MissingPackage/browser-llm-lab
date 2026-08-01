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

6. **RULING RICHIESTO — gate tok/s GLM falliti (fase 6, spec §8)**
   (2026-08-01, it.11). Il bench di produzione misura, sul prompt p6 (461
   token) con nGen 64, protocollo B2 (3 repliche + warmup, mediana):
   **decode 3.30 tok/s vs gate 13.43** (4.1× sotto) e **prefill 4.41 tok/s
   vs gate 56.58** (12.8× sotto), config migliore = slab 12 GiB
   (`results/engine/bench-glm-4090-b12-2026-08-01.json`; a 11 GiB: 2.96 /
   3.87).

   **Attribuzione misurata** (non congetturale): decode 302.7 ms/token =
   stallo residenza 112.4 (pack 71.6 + read 20.1 + upload 20.8, 4.47
   miss/token a hit 97.57%) + residuo 190.3 (1.816 dispatch + 47 sync
   CPU↔GPU/token). I token a ZERO miss costano 136.3 ms (mediana; n_eff = 3
   posizioni distinte per run, corroborate da 135.8 ms nel run a 11 GiB)
   ⇒ **PROIEZIONE: con residenza perfetta il decode sarebbe 5.3-7.3 tok/s,
   comunque sotto il floor.** Il gate decode non è raggiungibile con
   l'architettura per-token attuale.

   **Precisazione sul gate prefill (importante per lo scope di C3)**: il
   4.41 vs 56.58 NON è un divario omogeneo della stessa grandezza. Il floor
   è il `pp512` di llama-bench = prefill BATCHED (n_ubatch 512); il motore
   esegue 461 forward SEQUENZIALI perché non ha un percorso M>1. Non è una
   prestazione 12.8× peggiore: è una **capacità mancante** (MoE batched, con
   insiemi di expert diversi per token). Il PI decida se il gate prefill
   vada riformulato in C3 come requisito di capacità e non di velocità.

   **Il contratto vieta ottimizzazioni fuori scope C2**, quindi il PI decide:
   (a) [RACCOMANDATA] **deroga con misura onesta**: C2 chiude come "correttezza
       + residenza dimostrate, prestazione sotto il floor CPU con attribuzione
       quantitativa"; il floor 13.43/56.58 diventa l'obiettivo di C3, che ha
       già le leve identificate e dimensionate (repack all'import: −71.6
       ms/token; eliminazione/batching dei 47 sync: il termine dominante da
       190 ms; prefill batched: il gate prefill per costruzione);
   (b) **estendere C2** con le ottimizzazioni (viola lo scope dichiarato e il
       timebox; le leve sono materia C3 per decisione di spec);
   (c) **FAIL formale del goal C2** e ri-scope.
   In (a): serve anche il pronunciamento sulla riformulazione del gate per C3
   (il floor CPU resta il target del programma, non del goal C2).

7. **RULING RICHIESTO — non-regressione Qwen: K=8 e prefill sotto soglia,
   attribuiti all'host** (2026-08-01, it.11). Misura same-day, 2 run
   indipendenti: decode K=8 **263.5** vs soglia 282.9 (287.46−2σ) e prefill
   chunked **747.9 ms** vs soglia 726.3 (697.8+2σ) ⇒ FAIL della lettera del
   gate; K=1 **241.0/246.3** vs 226.5 PASS; conformance fase A **PASS
   identica** (98.05 golden / 100.00 cpuref, 147 dispatch/token).

   **Evidenza dell'attribuzione** (v. journal it.11): il codice del percorso
   Qwen è byte-identico al commit del baseline (engine.worker, gpuforward,
   decodebatch, prefillplan, attnsplit) e il WGSL generato per tutte e 6 le
   config Qwen di gpuforward.ts:271-276 è testualmente identico modulo righe
   vuote (check rifatto in modo indipendente dal verifier; gli altri file
   condivisi toccati — quant.ts e shape.ts in aggiunta pura, gguf.ts +18/−3
   e cpuref.ts +320/−1 con edit in-place le cui rimozioni sono un commento,
   la riga dell'enum GGML_TYPE riscritta con gli stessi valori Qwen e un
   messaggio d'errore, più una riga di import); falliscono solo i due path a
   carico sostenuto (K=8, prefill chunked) mentre i path latency-bound (K=1,
   seq prefill, encode CPU) sono uguali o migliori; l'host ha un processo
   utente al 86% di CPU da 6 giorni, package 99 °C, GPU a 76-83 °C e
   1425-1620 MHz sotto carico (max SM 3105).

   **Opzioni**: (a) [RACCOMANDATA] accettare l'attribuzione a stato-host e
   ri-misurare a macchina quiescita PRIMA della chiusura del goal — richiede
   un'azione umana (chiudere il browser `zen`), quindi la ri-misura va
   schedulata dal PI; (b) accettare l'attribuzione e NON ri-misurare
   (la conformance identica + i path latency-bound migliori bastano come
   prova di non-regressione del codice); (c) trattarlo come regressione vera
   e bloccare la chiusura finché non torna sopra soglia.
   Nota metodologica per il PI: le soglie 2σ sono tarate su repliche
   intra-run (σ 2.3 su K=8); non catturano la varianza INTER-giorno dello
   stato termico dell'host. Se (a) o (b), vale la pena registrare in spec
   che il bench di non-regressione dichiari clock/temperatura GPU.
