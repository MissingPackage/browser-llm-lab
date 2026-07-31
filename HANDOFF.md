# HANDOFF — browser-llm-lab   (updated 2026-07-31, session 14 — goal engine-fase-c1 CHIUSO)

## 1. Next decidable

**GOAL `engine-fase-c1` CHIUSO** (2026-07-31, checklist DONE WHEN 6/6 nel
journal, verifier gate a ogni fase + merge su main da ruling permanente).
Fase C splittata C1/C2/C3: C1 = misure sull'oracolo, chiusa. Risultati che
cambiano il progetto:
- **recall lookahead 92.0% @K=8** (77.5% @4) su GLM-4.7-Flash vs baseline
  temporale 32.3% — meglio del 71.6%/41.3% di colibri su GLM-5.2;
- **la residenza NON è skewed** (top-4 expert/layer = 21.8% delle selezioni,
  working-set 1.663 unici in 32 token, copertura 2944/2944) ⇒ il learned
  pinning vale poco (≤12.5% degli slot), il valore sta nel prefetch;
- **"modello ~2× la memoria" REGGE**: 90.8% hit-rate decode a cache = 50% del
  parco (config tarata: pin 12.5%, K=4) vs 84.7% LRU;
- **il box dev è una 4090 LAPTOP da 16 GiB**: 2.573 slot su 2.944 = fattore
  1.14× ⇒ il regime di paging vero è mobile/M4-condiviso, non il dev-loop.

**PI-gated ora**: (a) **docket item 4 del goal C1** — proposta go/no-go PILOT
(GO prefetch, NO-GO pinning come leva primaria) con l'aritmetica del costo-miss;
la domanda aperta è se serve prima un WP sulla banda OPFS **a freddo** (~mezza
giornata) per rendere non-condizionale il calcolo; (b) prossimo goal engine:
C2 (esecuzione GLM MoE+MLA nel motore) o C3 (paging) o un rimando §I;
(c) headline benchmark pubblico (docket 16, contributo separato); (d) igiene
goal stale (fase-1b-matrice, fase-2-deep-dive) da /weekly-maintenance.
Riancorarsi da: `.harness/goals/engine-fase-c1/{GOAL,PHASES,docket,journal,digests}.md`,
`results/engine/moe-oracle/`, `docs/engine/direction.md` §7 (aggiornata),
`docs/engine/ideas-ledger.md` §A/§I (aggiornati).

## 2. Stato del goal C1 (sessione 14, iterazioni 0-6)

- **Strumento**: `tools/oracle-moe/` — tool C++ che linka la build dell'oracolo
  (`~/Projects/llama.cpp-oracle`, commit `5f55650a`, CPU-only) e osserva
  `ffn_moe_topk`/`ffn_norm` via `cb_eval`; pesi router letti dal GGUF con
  `gguf.h`. **ZERO patch a llama.cpp** (checkout verificato pulito a ogni
  iterazione). Corpus di 8 prompt congelati + `run-trace.sh` + simulatore TS
  (`sim/`, 15 unit) + `gguf-residency.py`.
- **Artefatti**: `results/engine/moe-oracle/{trace-2026-07-31.jsonl.gz,
  -summary.json, -recall.json, residency-sim-2026-07-31.json, oracle-smoke,
  llama-bench, smoke log}`.
- **Autotest dello strumento** (gate hard): il predittore replicato riproduce
  la selezione vera dello STESSO layer a 0.999997 su 5.75M slot.
- Verifier indipendente a ogni fase: PASS ×5, con un FAIL intermedio in fase 5
  (done-when "punto di lavoro" spuntato senza artefatto) corretto e ri-verificato.

## 3. Stato precedente (sessione 13, goal B2 — decode 122.4 → 287.5 tok/s)

- **Fase 1 (attribuzione)**: modalità `attrib` (wall per-token + delta liv.1/liv.2
  + probe sync-floor + predizione K). FINDING: il "73% fuori GPU" del docket B1
  era un artefatto (gpuBusy@ctx64 vs wall@ctx570); quota reale 20%, gpuBusy scala
  con kvLen (attention 14 wg). → contratto rifatto (v2, ruling opzione (a)).
- **Fase 2 (spec+microbench)**: kernel attention SPLIT sul contesto (2 pass,
  CHUNK=64, griglia fissa (14,16), log-sum-exp esatto) misurato in isolamento:
  ~32 µs/layer PIATTO da ctx 64 a 1024 vs 29→219 del fuso; parità identica al
  fuso vs CPU f64. Spec approvata: soglia 230, token_embd su GPU, dispatch ≤100
  ARCHIVIATO-desktop (trigger mobile, ledger §I).
- **Fase 3**: split nel piano fuso (attnPart+attnReduce, `attnsplit.ts` puro) —
  conformance IDENTICA a B1, **decode 248.3 tok/s già a K=1**.
- **Fase 4**: decodeBatch K≤8 — feedback token on-GPU (`embedGatherQ4`,
  token_embd ~68 MB su GPU, dequant esatta), UNA mapAsync/batch, EOS via crop
  (`decodebatch.ts` + trimAtEos); token-identity K=8/5/1 IDENTICI su 256 token.
- **Fase 5**: liv.2 sul loop nuovo (timestampWrites per step, gpuMs reale,
  identità invariata con tsq ON); engine-prof v2 K-aware (0 bindGroup, submit
  1/8 esatto, 148 dispatch/forward); bench su decodeBatch (k=8 default).
- **Fase 6**: bench canonico — **headline K=8 287.5 ±2.3 tok/s** (gate 230),
  baseline K=1 same-day 238.3, gpuBusy 3.06 ms/token da repliche liv.2 DEDICATE
  (quota fuori-GPU 13.2%), overhead telemetria −0.002%, prefill 697.8 ms ≤810
  (gate ASSOLUTO di spec: il gate relativo alla seq è fuorviante ora che la seq
  usa lo split); rollback + prefix-cache PASS (restore worker nuovo 173 ms).
- Unit 156→166 (attnsplit 5, decodebatch 5); attnFusedWgsl resta nel sorgente
  per microbench/debug, fuori dal piano di produzione.

## 4. Open threads

- **Merge su main**: fatto a goal chiuso dopo verifier PASS (ruling permanente) —
  se questa riga è presente e il merge non risulta in `git log main`, il merge è
  stato interrotto: rifarlo (merge di engine/fase-c1).
- **Banda OPFS a FREDDO mai caratterizzata**: tutta l'aritmetica del costo-miss
  del paging (docket C1 item 4) è condizionale su questo numero. WP ~mezza
  giornata, PI-gated.
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket item),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Sweep fase 1b (wllama/transformersjs su S22) fuori goal; GLM-5 uscito → ledger §H a v2.
- Rimandi di fase in ideas-ledger §I (longest-prefix, chiave testuale, logits nel
  checkpoint, scoring eviction, parametro M, telemetria liv.3, dispatch≤100→mobile).
- Nota per il futuro bench: il rate K=1 usa la finestra fase-A (post-primo-token),
  il K>1 post-primo-batch — differenza non materiale (verifier it.6: <1σ), ma da
  uniformare se il confronto diventasse un gate.

## 5. Landmines

- **MAI confrontare gpuBusy e wall misurati a contesti/finestre diversi**: il GPU
  busy scala con kvLen (attention) — il "73% fuori GPU" di B1 era questo errore
  (tsq-diag §Conseguenze, corretta). Constraint di contratto da B2 in poi.
- **Coi tap ATTIVI il tsq copre solo il primo segmento del pass** (la copy del tap
  spezza il pass, i segmenti riaperti non hanno timestampWrites): le misure
  gpuBusy si fanno SENZA taps (il bench lo fa; engine con tap ⇒ gpuMs parziale).
- **Pipeline WebGPU invalida = submit droppati IN SILENZIO con readback STALE
  plausibili**: errore di compilazione WGSL (es. ridichiarazione di variabile
  nello stesso scope) ⇒ Dawn droppa ogni submit; le mapAsync risolvono coi dati
  del run precedente ⇒ "parità perfetta" fasulla. Difese: error scope
  validation/oom come CONTRATTO di prefillChunked E decodeBatch; diag a
  COLD-START; gli uncapturederror arrivano come pageerror ASINCRONI — mai
  troncarli via `tail`/filtri.
- **WGSL/Tint: mai fidarsi dell'azzeramento implicito di una `var` array dichiarata
  nel body di un loop** — azzerare esplicitamente (guardia: kernel-diag).
- **mai mapAsync su un buffer referenziato da un submit non ancora emesso**: Dawn
  droppa l'INTERO command buffer in silenzio (tsq-diag-2026-07-29.md).
- Chrome headless Linux/NVIDIA → SwiftShader: driver Playwright con HEADED=1; Chrome
  da shell sandboxata → SwiftShader anche headed (bench col sandbox disabilitato).
- Vite: server di sessioni vecchie su :5173+ servono CODICE STALE — porta dedicata
  (`npx vite --port 5199 --strictPort`), kill a fine sessione (`pkill -f "[v]ite --port"`).
- Device senza `requiredLimits` espliciti nasce a 128 MiB binding → garbage silenzioso;
  grid > 65535/dim ⇒ submit no-op muto.
- Chrome branded Linux/NVIDIA NON espone shader-f16 → f32-first; timestamp GPU
  quantizzati ~100 µs.
- llama.cpp SOLO oracolo (mai vendored); oracolo CPU quantizza le attivazioni (q8):
  noise floor 98.05% ⇒ gate doppio (cpuref-f64 ≥99% E golden ≥97%).
- **llama-cli (build ≥2026-07, UI chat) IGNORA `-no-cnv` e loop-a all'infinito su
  stdin chiuso** (676 MB di "> " in un log prima del kill): run scriptate SEMPRE
  con `-st --simple-io` (o llama-bench); mai fidarsi che esca da solo.
- `erasableSyntaxOnly`; contratto `pos === kvLen` hard su
  forwardToken/prefillChunked/decodeBatch (pos libere ⇒ throw; usare crop/reset).

## 6. Docket (decisioni PI pendenti)

4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10-fase-1b qualityScore
     (goal evals futuro); #8 sorveglianza wllama (v3.1 ha WebGPU — rilevante per il
     benchmark pubblico).
20. **go/no-go PILOT** (docket C1 item 4): GO prefetch / NO-GO pinning come leva
    primaria, con l'aritmetica del costo-miss; domanda aperta = WP banda fredda
    prima o dopo.
16. Headline del benchmark pubblico (ledger §E): serve prima del goal benchmark.
    Il benchmark è un CONTRIBUTO SEPARATO dall'engine (ruling PI 2026-07-30:
    repo/paper/sito propri, confronto imparziale che include il nostro engine;
    split repo rimandato alla pubblicazione) — ci si torna su iniziativa PI.
19. ~~**Prossimo goal engine da scegliere**~~ RISOLTO (2026-07-30, ruling PI in
    chat: contratto engine-fase-c1 approvato, "Per me tutto ok. Scarica pure
    GLM 4.7 quando ti serve"). Fase C splittata C1/C2/C3; goal C1 aperto →
    `.harness/goals/engine-fase-c1/`. Pendente lì: plan-check (item 1).
