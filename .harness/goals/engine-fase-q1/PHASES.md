# PHASES — engine-fase-q1 (decomposizione it.0, 2026-08-10)

Base della decomposizione: GOAL.md (contratto v1, aperto 2026-08-10) +
recon §4 "lista dei prezzi". Sequenziale, NESSUN parallel-group: la GPU è
una e le run lunghe esigono albero congelato (landmine C3c) — il
parallelismo fra fasi qui è un rischio, non un'ottimizzazione.

Non-regressione GLM = gate PERMANENTE a ogni merge (contratto), non una
fase: ogni done-when la include implicitamente via `npm test` + ktest.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | Spec + ratifica contratto | `docs/superpowers/specs/2026-08-10-engine-fase-q1-design.md` esiste con: SHA GGUF pinnate dei 3 modelli (4B/9B/35B-A3B); piano numerico DeltaNet (strategia verifica ricorrenza, f32); protocollo conformance per-modello; definizione proxy tier mobile (VRAM/banda dichiarate); stato subgroup-matrix browser con fonte+data (il [VERIFY] llamaweb chiuso); ratifica formale del contratto (pattern c3a it.3/c3b it.4); entry di registrazione in docket.md | none | docs/superpowers/specs/**, .harness/goals/engine-fase-q1/** | done (it.1, verifier PASS) |
| 2 | Reader GGUF + tokenizer famiglia | GGUF dei 3 modelli scaricati, SHA verificate (script exit 0); reader `qwen3_5`/`qwen3_5_moe` carica i 3 modelli (test load PASS); token id IDENTICI all'oracolo llama.cpp sul corpus di conformance committato, per i 3 modelli (test dedicato PASS in `npm test`); `npx tsc --noEmit` pulito | none | src/engine/** (nuovi file famiglia), tests/**, scripts/**, tools/** | done (it.3) |
| 3 | Kernel DeltaNet WGSL (kernel-level) | ktest DeltaNet vs cpuref-f64 PASS sul campione fissato in spec (conv k4 + delta rule + stato ricorrente, f32); ktest preesistenti tutti PASS (69/69 = GLM path intatto) | none | src/engine/** (kernel nuovi), tests/** | done (it.4-5: 75/75) |
| 4 | Path 4B end-to-end (ibrido 3:1 + GQA variante) | forward 4B completo: argmax == cpuref-f64 sul campione ratificato; golden 4B full-corpus MISURATO e soglia FISSATA (ratchet, mai import del PIN GLM); riferimenti full-resident decode/prefill/TTFT in JSON con hostState | none | src/engine/**, results/engine/** | done (it.6-10) |
| 5 | 9B + WP decomposizione gap | 9B: argmax == cpuref + golden a soglia fissata; JSON confronto full-residency noi-vs-llama.cpp Vulkan (stessa GGUF/driver, p512/n64) in results/engine/; doc di studio con scomposizione esplicita del gap (kernel / dispatch / safety-check / paging) e leve ordinate per ROI misurato | none | src/engine/**, results/engine/**, docs/engine/study/** | done (it.11-12) |
| 6 | Leve kernel bounded (esito condizionato dal ROI di fase 5) | per dot4I8Packed e tuning tile per-device: O implementata DIETRO FLAG con delta misurato in JSON e default invariato (non-reg verde), O esclusione motivata coi numeri del WP nel doc di studio; subgroup-matrix: JSON di misura dietro flag Chromium O impraticabilita' documentata con fonte — MAI path di default (promozione = docket) | none | src/engine/**, results/engine/**, docs/engine/study/** | done (it.13: 2 esclusioni motivate + probe) |
| 7 | MoE 35B-A3B parametrizzato | slotTable/classi/nExpert/topK/shapes parametrici con GLM INVARIATO (non-reg verde); 35B forward: argmax == cpuref sul campione, golden a soglia fissata, firma routing registrata; run regime C3c al budget 16 GB senza OOM con JSON | none | src/engine/**, results/engine/** | done (it.14-18: ratchet 98.926%, paging con eviction reali) |
| 8 | Tier + recall + bandmodel | JSON recall prefetch sul router 256-wide (vs 91.92% GLM: scostamento SPIEGATO nel doc, non gateato); bandmodel rifittato coi punti nuovi, test in `npm test` verde; bench JSON per tier mobile(4B) + 8/12/16 GB(35B) emulati con hostState, prefill/TTFT e gap UX riportati; collasso in scarsita' rimisurato con attribuzione | none | src/engine/**, results/engine/**, docs/engine/study/** | done (it.19-20) |
| 9 | Chiusura — RIAPERTA (docket item 14: manca la parità di ottimizzazioni) | checklist DONE WHEN del contratto ripassata voce per voce con evidenza puntuale; non-reg GLM PIENA fresca ad albero congelato (golden AL PIN, cpuref 256+512, firma esatta, b12 in banda ±5%, Qwen2.5, ktest, suite, tsc); direction (sezione generalizzazione coi numeri) + ledger + HANDOFF refresh; docket q1 triage | none | docs/**, HANDOFF.md, .harness/goals/engine-fase-q1/** | RIAPERTA (it.21 chiusa sul contratto sbagliato) |

Taglie stimate (1-4 it. ciascuna): 1→1-2 · 2→2-3 · 3→2-4 · 4→2-3 · 5→1-2 ·
6→1-3 · 7→3-4 · 8→2-3 · 9→1. Il rischio dominante (DeltaNet) è isolato in
fase 3 kernel-level PRIMA di ogni end-to-end: se la numerica ricorrente non
regge il cpuref, si scopre lì, non a metà del 35B.
