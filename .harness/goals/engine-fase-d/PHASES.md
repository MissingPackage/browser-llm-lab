# PHASES — engine-fase-d (decomposizione it.0, 2026-08-10)

Sequenziale (GPU una, albero congelato nelle run). Ordine dei BLOCCO A dal
ROI misurato in q1 (gap-decomposition §4), NON dall'ordine storico di GLM.

**Regola di misura del goal** (direction §7-ter): durante le fasi 1-5 e 7 si
usano SOLO micro-bench e ktest. I bench pieni esistono in DUE punti: fase 6
(checkpoint A) e fase 8 (checkpoint B). Una fase che vuole un bench lungo
fuori da lì è un errore di piano, non una necessità.

Gate PERMANENTI a ogni fase (impliciti in ogni done-when): ktest tutti PASS,
`npm test` verde, `npx tsc --noEmit` pulito, ratchet di correttezza intatti.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | Parametrizzazione del core (**DONE** it.1-7) | `residency.ts`/`moe.ts` parametrici su (nExpert, topK, classi slab + byte, shapes): GLM e q35 configurazioni della STESSA struttura; l'arena+LRU+slotTable q35-only di q1 it.16 RIMOSSA da `q35gpumodel.ts`; test di NON-DUPLICAZIONE in `npm test` (pattern gpudevice.test) che fallisce se ricompare una seconda implementazione; ktest tutti PASS (GLM bit-exact) | none | src/engine/**, tests/** | **done** |
| 2 | Costo di pack per miss (**done** it.8-9) | ~~il repack K-quant esce dal path on-miss (import-time)~~ → **riscritto dal ruling PI 2026-08-10 (docket item 5)**: il costo di pack per miss scende di >=4x, micro-bench PRIMA e DOPO nello stesso JSON, delta riportato; ktest MoE q35 PASS invariati | none | src/engine/**, tests/**, scripts/**, results/engine/** | **done** |
| 3 | Decode multi-step senza readback (**densi DONE it.10-12: -3,33 ms/token (-8,3%); MoE → fase 3b**) | q35 densi usa il pattern `decodeBatch` (argmax on-GPU + embed gather, K step/submit): micro-bench ms/token prima/dopo, ~~atteso ≥ −5.1 ms dal q1~~ → **riscritto dal ruling PI 2026-08-11 (docket item 9)**: delta MISURATO a caldo con dispersione dichiarata, qualunque sia (warm-up scartato, bracci interleavati — docket item 10); argmax IDENTICO al path a readback su un campione (gate di correttezza secco) | none | src/engine/**, tests/**, results/engine/** | **done (densi); MoE → 3b** |
| 3b | **Resolve MoE su GPU** (ruling PI 2026-08-10, docket item 8) | il decode del 35B non fa piu' un readback per layer: router + selezione + risoluzione expert→slot su GPU via slotTable, kernel d'arena al posto dei sotto-range calcolati da CPU, e miss rilevato su GPU con repair+replay dalla CPU. DONE WHEN: (a) submit/token e readback/token MISURATI prima e dopo nello stesso JSON, con il caso a residenza piena a **1 submit/token**; (b) argmax IDENTICO al path attuale su un campione del golden 35B (gate secco); (c) ktest MoE q35 PASS invariati e GLM bit-identico; (d) il conteggio dei miss e il routing esatto restano quelli del path CPU sullo stesso campione | none | src/engine/**, tests/**, results/engine/** | **done (it.11-18)**: fette 1 (kernel d'arena), 2 (router+resolve WGSL), 3a (cablaggio arena), 3b (router in ombra sui layer veri), 3c (submit unico, it.17), repair+replay con checkpoint dello stato ricorrente (it.18). Done-when: (a) 81→1 submit/token e 41→1 readback misurati nello stesso JSON · (b) argmax 39/39 identico · (c) ktest 87/87 · (d) miss e routing identici A RESIDENZA PIENA (il campione che (a) nomina); a freddo il fetch e' +12,0%, inerente al meccanismo e pubblicato come fatto. −68,60 ms/token (−48,6%) a parita' |
| 4 | Prefill chunked/batched q35 | prefill non più sequenziale: pattern `planMoeChunk` + gemv batch portato a q35; micro-bench tok/s di prefill prima/dopo; logits del prefill batched == logits sequenziali sul campione (gate secco) | none | src/engine/**, tests/** | ready |
| 5 | Decode ottimistico + policy sul MoE q35 | decode ottimistico (1 submit/token, repair+replay) attivo dove la residenza lo consente; prefetch in-forward + tier/AUTOPIN + budget ctx-aware parametrici e attivi su q35 — CIASCUNO con delta misurato su micro-bench O esclusione motivata coi numeri (il recall q35 è 82.67%@8: il prefetch si giustifica o si esclude); GLM invariato | none | src/engine/**, tests/** | ready |
| 6 | **CHECKPOINT A** — misura e merge gate | (host DICHIARATO, GPU scarica, albero congelato) GLM non-reg PIENA: b12 optimistic in banda ±5% vs 13.172/31.26/14.74 + golden AL PIN + cpuref + firma; RIFERIMENTI q35 NUOVI (4B, 9B, 35B ai tier 8/12/16) con hostState; gap nativo RI-MISURATO a parità; ratchet golden q35 riverificati; direction §7-bis RISCRITTO coi numeri veri (via la marcatura stale) | none | results/engine/**, docs/engine/**, scripts/** | ready |
| 7 | Spec-dec MTP (blocco B) | reader della testa MTP (`*-MTP-GGUF` pinnati) + draft/verify: accept-rate misurato per modello su micro-campione; token accettati == token del greedy (bit-invarianza, gate secco) | none | src/engine/**, tests/**, scripts/** | ready |
| 8 | **CHECKPOINT B** — speedup o esclusione | speedup end-to-end del decode misurato con hostState O esclusione motivata coi numeri se sotto soglia utile; ratchet golden invariati; JSON committato | none | results/engine/**, docs/engine/** | ready |
| 9 | Chiusura | checklist DONE WHEN del contratto voce per voce con evidenza; non-reg GLM piena fresca; direction/ledger/HANDOFF; docket triage; q1 docket item 14 CHIUSO (la parità c'è) | none | docs/**, HANDOFF.md, .harness/goals/** | ready |

Taglie stimate (1-4 it.): 1→2-4 · 2→1-2 · 3→2-3 · 4→2-3 · 5→2-4 · 6→1 ·
3b→3-5 · 7→2-4 · 8→1 · 9→1. La fase 1 è la più rischiosa (tocca il core indurito di
GLM): il paracadute è il ktest bit-exact, che gira in minuti.
