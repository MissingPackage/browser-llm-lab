GOAL: engine-fase-d — Il motore ha UN SOLO path ottimizzato, parametrico sul
modello: GLM-4.7-Flash e la famiglia Qwen 3.5/3.6 girano sulla STESSA
meccanica (paging, prefill, decode, telemetria) senza duplicazioni, con GLM
invariato nelle metriche e q35 che eredita ogni ottimizzazione esistente;
sopra la parità, i moltiplicatori: spec-dec con MTP nativa.

<!-- CONTRATTO v1 (chartered 2026-08-10, PI in chat: "vai"). Nasce dal ruling
     PI 2026-08-10 (direction §7-ter): una famiglia nuova non è importata
     finché non ha le stesse ottimizzazioni di quelle esistenti; il codice si
     UNIFORMA, non si affianca un path di serie B. Il goal q1 aveva chiuso
     senza parità (docket q1 item 14: goal RIAPERTO, checklist soddisfatta ma
     contratto sbagliato — scritto da me).
     BLOCCO A (parità) = GATE DI MERGE AUTONOMO: se il blocco B slitta, A è
     già atterrato e utilizzabile.
     Goal start tag: goal-engine-fase-d-start. -->

DONE WHEN (all measurable):

--- BLOCCO A: unificazione + parità (GATE DI MERGE AUTONOMO) ---

- UN SOLO meccanismo, non due: `residency.ts` / `moe.ts` /
  `glmprefillplan.ts` / il piano di decode diventano PARAMETRICI sul modello
  (nExpert, topK, classi slab e loro byte, shapes, gating). Verifica
  MECCANICA: test in `npm test` che FALLISCE se esistono due implementazioni
  della stessa meccanica (pattern `tests/gpudevice.test.ts`, che già vieta
  per costruzione `requestDevice` fuori dal punto unico); l'arena+LRU+slot
  table nata q35-only in q1 it.16 SPARISCE, assorbita dalla
  parametrizzazione.
- GLM NON REGREDISCE (gate secco, host comparabile DICHIARATO): b12
  optimistic in banda ±5% vs 13.172 / 31.26 / 14.74; golden AL PIN 98.828%;
  cpuref 256+512; firma routing esatta; ktest tutti PASS; suite; tsc.
- q35 EREDITA, ciascuna leva con delta misurato su MICRO-bench (mai bench
  lunghi durante lo sviluppo — direction §7-ter punto 2):
  (a) slab PRE-IMPACCHETTATI all'import (via il repack JS on-miss: oggi
      17.64 ms/miss misurati, di cui il repack è la quota dominante);
  (b) prefill CHUNKED/batched (pattern `planMoeChunk` + gemv batch);
  (c) decode MULTI-STEP senza readback per token (pattern `decodeBatch`:
      argmax on-GPU + embed gather; −5.1 ms/token misurati in q1);
  (d) decode OTTIMISTICO sul MoE (1 submit/token, repair+replay) dove la
      residenza lo consente;
  (e) prefetch in-forward, policy tier+AUTOPIN, budget slab ctx-aware;
  (f) fusioni esistenti (pairGemvSilu, gemvAccumFast, kquantfast).
  Per (e): il recall lookahead q35 è 82.67%@8 vs 91.92% GLM — se il
  beneficio misurato non c'è, si ESCLUDE coi numeri, non si assume.
- RATCHET DI CORRETTEZZA PRESERVATI, bit-fedeltà invariata: golden top-1
  >= 98.828% (4B) / 97.656% (9B) / 98.926% (35B). Le ottimizzazioni non
  spostano un token: sono i gate di q1 che sopravvivono e rendono sicuro
  l'intero blocco.
- RIFERIMENTI q35 NUOVI che SOSTITUISCONO la baseline storica stale:
  decode / prefill / TTFT per 4B, 9B e 35B (tier 8/12/16 GB) con hostState
  dichiarato — misurati UNA VOLTA, a blocco A completo.
- Gap nativo RI-MISURATO a parità (llama.cpp Vulkan, stesso GGUF): il numero
  pubblicabile del writeup nasce qui, non prima.

--- BLOCCO B: moltiplicatori (fase D) ---

- Spec-dec con MTP NATIVA (la famiglia Qwen ha la testa MTP in tutti i GGUF
  `*-MTP-GGUF`; GLM da valutare): draft+verify implementato, con accept-rate
  MISURATO per modello e speedup end-to-end sul decode.
- Qualità BIT-INVARIATA rispetto al greedy: il verificatore accetta solo
  token che il modello target avrebbe prodotto (ratchet golden invariati).
- Se lo speedup misurato non supera una soglia utile, si CHIUDE con
  l'esclusione motivata coi numeri (ramo previsto, non fallimento).

EVIDENCE OF DONE: test di non-duplicazione verde in `npm test`; ktest
completo; JSON GLM non-reg (bench + conformance) a host dichiarato;
micro-bench per ciascuna leva con delta; JSON riferimenti q35 nuovi; JSON gap
nativo a parità; JSON accept-rate/speedup spec-dec; diff di direction (§7-bis
riscritto coi numeri veri) + ledger + HANDOFF.

AUTHORITY GRANTED:
- may do autonomously: `src/engine/**` INCLUSI i file GLM (residency, moe,
  glmmodel, glmprefillplan, bandmodel) — qui la parametrizzazione è l'oggetto
  del goal, non un rischio da evitare; tests/**, scripts/**, tools/**,
  docs/engine/**; commit/push su main a iterazione VERIFICATA; merge a goal
  chiuso e verificato; run locali su 4090; micro-bench liberi.
- must docket (never do): abbassare un ratchet di correttezza o una metrica
  GLM; eseguire bench LUNGHI fuori dai due punti di misura previsti (fine
  blocco A, fine blocco B) — direction §7-ter; pubblicare numeri q35 prima
  della parità; quant nuove; hardware nuovo (M4/S22); spese; azioni di
  publishing; delete di codice committato <30 giorni.

CONSTRAINTS: le standing (spec-first, non-regressione permanente, bench a
host DICHIARATO, albero congelato + 60 s nelle run GPU, llama.cpp solo
oracolo, f32-first, near-tie mai gateati, no pipe sui runner, full-corpus
solo per riferimenti, zero attribution AI) PIÙ: **il codice si uniforma —
una meccanica, una implementazione**; ogni leva portata si misura con
MICRO-bench durante lo sviluppo, i bench pieni SOLO ai due checkpoint.

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per
cycle; digest every cycle; stop-by-design when the remaining work is
docket-gated. Blocco A merge-abile da solo.

CONTEXT ANCHORS: `.harness/goals/engine-fase-q1/docket.md` item 14 (perché
siamo qui) + item 10-13; direction §7-bis (baseline STALE) e §7-ter (le due
regole permanenti); spec e journal C3a/C3b/C3c (le ottimizzazioni da
parametrizzare); `docs/engine/study/2026-08-10-q35-gap-decomposition.md` §4
(ordine ROI misurato); recon Qwen §2 (MTP nativa ovunque);
`tests/gpudevice.test.ts` (il pattern del test di non-duplicazione).

<!-- STATUS: CHIUSO dal PI il 2026-08-12 ("penso che questo goal non abbia altro
     da dire"). Blocco A (unificazione + parita') RAGGIUNTO e verificato; blocco
     B (spec-dec MTP) COSTRUITO, verificato col gate secco e ESCLUSO COI NUMERI
     al checkpoint B (PHASES riga 8, artefatto
     results/engine/specdec-4090-2026-08-12T21-49-18-513Z.json). L'obiettivo dei
     30 tok/s NON e' raggiunto e la strada che ci arriva e' nominata: kernel a
     piu' righe con riuso vero dei pesi (~42 tok/s proiettati), che il PI ha
     approvato come GOAL SUCCESSIVO, previa revisione architetturale
     indipendente. La riga 9 (checklist formale + non-reg GLM fresca) resta
     l'unico adempimento non eseguito: dichiarata non necessaria dal PI in
     questa sessione, si esegue al tag di release se e quando serve. -->
