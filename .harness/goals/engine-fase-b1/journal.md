# Journal — engine-fase-b1

2026-07-29 — Iterazione 2 (cont.), fase 1 DONE: spec B1 scritta
(specs/2026-07-29-engine-fase-b1-design.md, 5 sezioni contratto verificate via grep).
Decisioni proposte al ruling (docket 3): chiave su token-id, lookup v1 esatto, niente
logits nel checkpoint (1 forward al restore), LRU semplice, contratto hard pos===kvLen,
budget 512 MB. Soglia prefill 3× già PI-ruled (Pareto, it.1); resto soglie =
non-regressione fase A. Con fase 2 già chiusa, TUTTO il residuo (fasi 3-6) è
PI-gated dal ruling spec ⇒ stop-by-design da working protocol.

2026-07-29 — Iterazione 2, fase 2 DONE (diagnosi telemetria liv.2, dentro il timebox).
Matrice A/B (5 varianti, detector = 64 token greedy + logits finali): H1 timestampWrites
refutata, H3 ring refutata, H2 resolve-in-encoder confermata ma incompleta; il buffer
etichettato ha dato il colpevole: "tsq-staging used in submit while mapped". ROOT CAUSE
= bug NOSTRO fase A (mapAsync chiamata prima del submit che riempie lo staging ⇒ Dawn
droppa l'intero command buffer: token saltato = corruzione, copy mai eseguita = zeri;
"own" curava solo la matematica; il microbench attendeva il submit ⇒ funzionava).
FIX: flushTsq encoda soltanto, armTsq mappa dopo il submit. VERIFICA: matrice post-fix
tutta pulita (idsMatch=✔, maxΔ=0, gpuMs 2.17-2.26 ms/token REALI), conformance completa
con telemetryGpu:true GATE DOPPIO PASS (98.05% golden / 100.00% cpuref, = fase A),
npm test 122/122. Landmine HANDOFF aggiornata (decade "timestamp azzerati", resta la
lezione mapAsync-post-submit). Nota: docs/engine/tsq-diag-2026-07-29.md. Osservazione
fuori scope → docket 2: GPU busy 2.2 ms/token vs 8.1 wall ⇒ ~73% del decode è sync/
encode, dato che ri-inquadra il goal-brief di B2.

2026-07-29 — Iterazione 1 (parziale, pre-spec): plan-check APPROVATO dal PI ("sul
resto tutto ok"). Soglia prefill decisa per simulazione col criterio di Pareto
(mandato PI in chat). Sim su 4090 (prefill-sim-4090-*.json, 2 run concordanti):
seq-await 5.16 ms/token; batched senza readback 3.80 (submit/64); + skip lm_head
per posizioni non finali 3.38 ⇒ il floor "trucchi senza kernel nuovi" = 1.53×.
Residuo overhead-bound: ~28 µs/dispatch × 120 dispatch ⇒ analitico M=8 ≈ 5-10×.
SOGLIA FISSATA: 3× (≤1/3 della baseline seq same-day) — 2× sopra il floor dei
trucchi, metà della predizione conservativa. GOAL.md aggiornato (ruling registrato).
Metodo: prefillBatched() sim nel motore (embedding+pos pre-caricati, copy in-encoder,
submit a granularità variabile, decodeMatch=true su tutte le varianti vs run seq).
NOTA per fase 3: il knee submit-granularità è a 64 (all-in-one peggiora: 4.22).

2026-07-29 — Goal aperto. Contratto approvato dal PI in chat (goal-brief, "vai con 2
goal": approvazione in blocco, [ASSUMED] compresi). SPLIT della fase B di direction:
B1 = memoria/latenza di entrata (questo goal), B2 = floor dispatch ≤100 (goal futuro,
annotato in HANDOFF). Soglie numeriche marcate "da confermare in spec" vanno fissate
nella spec B1 prima del codice.

2026-07-29 — Verifier gate iterazione 2: PASS (8/8 punti, zero violazioni). Nota a
costo zero dal verifier per la fase 3+: aggiungere il campo `telemetryGpu` allo schema
del report di conformance, così il gate "liv.2 non perturba" diventa auto-evidente
dal JSON.

2026-07-30 — Ruling PI: spec B1 APPROVATA (docket 3, decisioni a-g in blocco).
Richiesta PI aggiuntiva recepita: i rimandi deliberati vanno documentati dove le fasi
successive li ritrovino → creato registro ideas-ledger.md §I ("Rimandi espliciti di
fase": 8 righe con fase di ripresa + trigger di riattivazione — longest-prefix,
chiave testuale/tokenizer, logits nel checkpoint, scoring eviction, re-inquadramento
B2, fusioni annotate, parametro M per spec-dec, telemetria liv.3). Fase 3 ready.

2026-07-30 — Iterazione 3, FASE 3 DONE: forward multi-token M≤8 implementato e
CONFORME. Piano prefill compilato a parte (11 dispatch/layer/chunk): prefillplan.ts
(chunking puro CPU-side, 13 unit CI), kernel nuovi in wgsl.ts (gemmQuantChunk q4/q8
a M colonne — riga di peso riusata per M attivazioni, shared-x se ≤28KB; rmsnormChunk;
ropeChunk pos per-riga; kvAppendChunk M righe/dispatch; attnPrefillChunk causale
intra-chunk; siluMulChunk), prefillChunked() in gpuforward (zero readback, embedding
CPU-side in buffer unico, submit/64 token, lm_head+argmax solo ultima posizione, tap
= ultima riga per chunk). Conformance ora gira col percorso M>1 attivo.
ROOT-CAUSE HUNT (prima conformance: FAIL 82.4%): bisezione a sensori — (1) diag
parità per lunghezza L: già L=1 divergeva ⇒ non era la maschera causale; (2) tap
per stadio 0/11/23: layer 0 già rotto; (3) stage-diag a 1 layer: qkv/attn/gate ok
a 2e-6, hidden Δ2.19 ⇒ down-proj; ipotesi strutturali (termine doppio/mancante,
range di blocchi) tutte refutate su CPU; (4) kernel in ISOLAMENTO vs dequant CPU:
K=896 perfetto, K=4864 rotto ⇒ BUG TROVATO: su Chrome/Tint una `var` array
dichiarata NEL BODY di un loop WGSL non viene ri-azzerata a ogni iterazione — con
>64 blocchi/riga (down: 152) le iterazioni ereditavano l'accumulatore bd della
precedente. FIX: azzeramento esplicito per iterazione (wgsl.ts, commento landmine).
VERIFICA: kernel-diag err ~1e-7 su 8 righe entrambi i K; parità seq-vs-chunked
L∈{1..128} argmax identici, Δlogit ≤1e-4; conformance GATE DOPPIO PASS 98.05%
golden / 100.00% cpuref (512/512) — identica a fase A — sia liscia sia con
telemetryGpu attivo; npm test 135/135 (122+13); tsc pulito. Report conformance ora
include telemetryGpu e prefill{path,mMax,submitTokens} (nota verifier fase 2).
Rimossi (spec §Struttura): sim prefillBatched + modalità prefillsim + script
prefill-sim.mjs. Restano come scaffolding di fase (rimozione a fase 6 coi knob
tsqDiag): prefill-diag.mjs (parità per lunghezza), kernel-diag.mjs (kernel vs CPU,
guardia sul bug Tint). Fase 3 chiusa in 1 iterazione (cap: 4). Next: fase 4 (crop).

2026-07-30 — Iterazione 4, FASE 4 DONE: rollback KV (crop/length-pointer) in 1
iterazione. Nuovo src/engine/kvlen.ts: pointer PURO (CI-testabile) con contratto
hard spec §Crop — assertNext(pos,count) pre-encode (pos===kvLen o throw, capacità),
advance post-forward, crop(toLen≤kvLen) zero-GPU, reset()≡crop(0). Integrato in
gpuforward: forwardToken valida pos===kvLen e avanza a forward riuscito;
prefillChunked valida posStart===kvLen e avanza di n; handle espone crop() e kvLen
(getter). Call-site aggiornati al riavvio esplicito (reset() in conformance
per-prompt, bench runOnce, prefill-diag). Prova meccanica: modalità worker rollback
+ scripts/kv-rollback.mjs — prefill 47 tok, genera 64 greedy, poi (1) crop al
prefisso e rigenera, (2) crop a metà generazione (P=62) e rigenera la coda,
(3) reset+re-prefill fresco: sequenze IDENTICHE su tutti e tre i check, kvLen
contabile esatto (111=47+64), exit 0, JSON committato
(kv-rollback-4090-2026-07-30T07-20-23-512Z.json). VERIFICA: npm test 143/143
(135+8 unit crop), tsc pulito, conformance invariante GATE DOPPIO PASS 98.05%
golden / 100.00% cpuref (spina: parità a ogni passo). Next: fase 5 (prefix-cache
OPFS — kvstore.ts codec puro + I/O SyncAccessHandle, restore in worker nuovo).

2026-07-30 — Iterazione 5, FASE 5 DONE: prefix-cache OPFS end-to-end in 1 iterazione.
Nuovo src/engine/kvstore.ts in due metà nette: (1) PURE, CI-testabili — codec
envelope BKV1 (header a offset fissi con lastUsedMs f64/hitCount per l'update in
place, meta JSON padded a 4B, payload per-layer K|V; validazione hard: magic,
version, meta vs atteso, dimensione totale ⇒ throw), chiave SHA-256 di
layoutVersion‖sha256(GGUF)‖tokenIds u32 LE (ruling a), pickEvictions LRU pura
(ruling d, budget 512 MB ruling f); (2) I/O OPFS KvStoreOpfs via SyncAccessHandle
(save con eviction proattiva a budget + retry singolo su QuotaExceededError da spec
§Quota; load con touch LRU in place). gpuforward: readKv (copy GPU→staging unica
→CPU nel layout esatto dell'envelope) e writeKv (writeBuffer per layer + pointer
reset+advance — niente logits, ruling c: si ricalcola il forward). Worker: modalità
pcsave/pcrestore; scripts/prefix-cache.mjs orchestra DUE page load ⇒ il restore
avviene in un WORKER NUOVO (sessione fredda). ESITO (JSON committato
prefix-cache-4090-2026-07-30T07-35-18-558Z.json): checkpoint 11.5 MB, save 4.4 ms,
restore 104.2 ms vs re-prefill 2977 ms (restore<reprefill ✓, margine ~29×),
continuazione 64 token IDENTICA al run ininterrotto ✓ (misura warm, dichiarato nel
JSON). Unit: 156/156 (143+13 kvstore: roundtrip, offset fissi, mismatch⇒throw,
chiave, LRU). tsc pulito.
OSSERVAZIONE LOAD-BEARING per fase 6: prefill chunked misurato ~3.0-3.2 s su 468
token (~6.6 ms/tok) nei run pcsave/pcrestore — SOPRA la baseline seq fase A
(~2.44 s): il GEMM chunk correctness-first (64 thread/riga, load scalari) non regge
la soglia 3× (~810 ms). La fase 6 deve applicare il trattamento fast-kernel di fase
A al GEMM chunk (4 righe/wg, vec4, shared già presente) e/o il fallback M della
spec §Rischi prima di rinegoziare soglie via docket. Next: fase 6 (bench +
non-regressione + chiusura goal).


2026-07-30 — Iterazione 6, FASE 6 DONE e GOAL CHIUSO. Percorso: (1) il GEMM chunk
"correctness-first" era 6.6 ms/token (peggio del seq) -> riscritto vec4 1-riga/wg
(2.35 ms/tok, reale) -> architettura FUSA di fase A per chunk (rmsGemmQkvChunkFast,
rmsPairGemmSiluChunkFast con silu, gemmResidChunkFast; 4 righe/wg, 16 lane, M
srotolato in SCALARI generati dal template — niente array privati -> scratch memory,
prima causa di lentezza) => 7 dispatch/layer (era 11). (2) ROOT-CAUSE conformance
53%: `var stride` ridichiarata nello stesso scope WGSL (chunkLaneReduce emessa 2x
in pairSilu) => modulo non compilato => pipeline invalida => Dawn DROPPAVA OGNI
SUBMIT del prefill in silenzio e i readback restituivano DATI STALE del run seq
precedente — per questo il diag "bitwise-perfetto" mentiva (misurava il seq!).
Diagnosi: cold-start + doppio giro chunk + dump KV riga-per-riga (readKv) + error
scope. FIX: scope block nella riduzione + error scope validation/oom PROMOSSO A
CONTRATTO di prefillChunked (throw sincrono e attribuibile). Lezione landmine: una
pipeline invalida non urla al submit — urla SOLO su uncapturederror (che finiva
nei pageerror troncati) e i risultati stale sembrano plausibili. (3) runBench
portato su prefillChunked + baseline seq same-day nello stesso report
(schemaVersion 3, gate auto-evidente); profiler engine-prof con FINESTRA DECODE
PURA (192 forward fra i progress i=32->224 dell'ultima replica OFF — la finestra
fase-A mescolava il prefill chunked). (4) Cleanup: knob tsqDiag+runTsqDiag+
tsq-diag.mjs RIMOSSI (diagnosi stabile, da spec); prefill-diag e kernel-diag
TENUTI e promossi a harness di parità permanente del percorso chunk (deviazione
motivata dalla nota "rimozione a fase 6": due root-cause trovate oggi grazie a
loro; il prefill-diag ora include cold-start golden + confronto KV riga-per-riga).

CHECKLIST DONE WHEN DEL GOAL (8/8):
1. [x] Spec B1 approvata (ruling PI 2026-07-30, docket 3).
2. [x] npm test 156/156 con unit nuovi: chunking M<=8 (13), crop (8), kvstore (13).
3. [x] Parità INVARIATA col prefill M>1: conformance GATE DOPPIO PASS 98.05% golden
   / 100.00% cpuref (512/512), anche con telemetryGpu attivo (run post-cleanup).
4. [x] Prefill >=3x: 700.5 ms vs 2410.9 ms seq same-day = 3.44x (soglia 804 ms) —
   bench-4090-2026-07-30T08-09-39-624Z.json (warmup+3 repliche HEADED).
5. [x] Rollback KV meccanico: 3 check sequenze IDENTICHE, exit 0
   (kv-rollback-4090-2026-07-30T08-10-42-251Z.json, kernel fusi).
6. [x] Prefix-cache OPFS e2e: restore in worker NUOVO 106.5 ms < re-prefill 699 ms,
   continuazione token-identica (prefix-cache-4090-2026-07-30T08-10-58-866Z.json).
7. [x] Non-regressione decode: 122.4 +-1.1 tok/s >=120 (fase A: 122.0); profiler
   finestra decode: createBindGroup=0, submit/token=1.005, dispatch/token 123.6
   <=130 (engine-prof-4090 JSON); overhead telemetria -0.85% (rumore).
8. [x] Diagnosi telemetria liv.2: fase 2 (root-cause mapAsync, fixata, nota
   tsq-diag-2026-07-29.md); knob diagnostici rimossi a diagnosi stabile.
