# HANDOFF — browser-llm-lab   (updated 2026-07-30, session 14 — goal engine-fase-c1 aperto, it.0)

## 1. Next decidable

**GOAL `engine-fase-c1` APERTO** (2026-07-30, contratto approvato dal PI in chat,
setup it.0 fatto): misure sull'oracolo desktop (llama.cpp strumentato,
contatori-only) che dimensionano il paging esperti su GLM-4.7-Flash — recall
lookahead LOOKA, skew/heat/working-set, curve hit-rate vs budget da simulazione
trace-driven. Split della fase C approvato: C1 = oracolo (questo goal), C2 =
MoE+MLA nel motore, C3 = paging+instant-on; hero-demo M4 PI-gated.
**Fase 1 DONE (it.1, verifier PASS)**: GGUF unsloth Q4_0 17.2 GB (sha in
journal/JSON), oracolo llama.cpp 5f55650a CPU-only fuori repo
(~/Projects/llama.cpp-oracle), smoke exit 0, pp512 56.6 / tg64 13.4 t/s ⇒
corpus 16k pos ≈ 9 min/run. Finding: zero patch (cb_eval + ffn_moe_topk/
ffn_moe_logits/ffn_norm — arch GGUF = deepseek2, 46 layer MoE); testa NextN/MTP
ASSENTE nel GGUF → docket goal item 2 (PI-gated, tocca fase D e C2).
**Fase 2 DONE (it.2, verifier PASS)**: spec C1 scritta
(`docs/superpowers/specs/2026-07-30-engine-fase-c1-design.md`, 7 sezioni,
LOOKA online con replica esatta della selezione — verificata nel codice
oracolo e nei tensori GGUF). **Next: RULING PI su docket item 3 del goal
(decisioni (a)-(f))** — fasi 3-6 (ROUTE_TRACE → LOOKA → simulatore →
sintesi) gated fino al ruling. STOP BY DESIGN. Riancorarsi da:
`.harness/goals/engine-fase-c1/{GOAL,PHASES,docket,journal,digests}.md`,
`docs/engine/direction.md` §3+§7, `docs/engine/study/colibri.md` §1-2.
Restano PI-gated fuori goal: headline benchmark pubblico (docket 16) e igiene
goal stale (fase-1b-matrice, fase-2-deep-dive) da /weekly-maintenance.
Chiusura B2 (decode 122.4 → 287.5 tok/s, 2.35×): vedi §2 e
`.harness/goals/engine-fase-b2/`.

## 2. State delta (sessione 13, 2026-07-30 — goal B2 completo, iterazioni 0-6)

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

## 3. Open threads

- **Merge su main**: fatto a goal chiuso dopo verifier PASS (ruling permanente) —
  se questa riga è presente e il merge non risulta in `git log main`, il merge è
  stato interrotto: rifarlo (merge di engine/fase-b2).
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket item),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Sweep fase 1b (wllama/transformersjs su S22) fuori goal; GLM-5 uscito → ledger §H a v2.
- Rimandi di fase in ideas-ledger §I (longest-prefix, chiave testuale, logits nel
  checkpoint, scoring eviction, parametro M, telemetria liv.3, dispatch≤100→mobile).
- Nota per il futuro bench: il rate K=1 usa la finestra fase-A (post-primo-token),
  il K>1 post-primo-batch — differenza non materiale (verifier it.6: <1σ), ma da
  uniformare se il confronto diventasse un gate.

## 4. Landmines

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

## 5. Docket (decisioni PI pendenti)

4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10-fase-1b qualityScore
     (goal evals futuro); #8 sorveglianza wllama (v3.1 ha WebGPU — rilevante per il
     benchmark pubblico).
16. Headline del benchmark pubblico (ledger §E): serve prima del goal benchmark.
    Il benchmark è un CONTRIBUTO SEPARATO dall'engine (ruling PI 2026-07-30:
    repo/paper/sito propri, confronto imparziale che include il nostro engine;
    split repo rimandato alla pubblicazione) — ci si torna su iniziativa PI.
19. ~~**Prossimo goal engine da scegliere**~~ RISOLTO (2026-07-30, ruling PI in
    chat: contratto engine-fase-c1 approvato, "Per me tutto ok. Scarica pure
    GLM 4.7 quando ti serve"). Fase C splittata C1/C2/C3; goal C1 aperto →
    `.harness/goals/engine-fase-c1/`. Pendente lì: plan-check (item 1).
