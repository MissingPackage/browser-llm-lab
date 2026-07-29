# HANDOFF — browser-llm-lab   (updated 2026-07-29, session 10)

## 1. Next decidable

**Iterazione 0 del goal `engine-fase-a`** (contratto approvato dal PI 2026-07-29,
`.harness/goals/engine-fase-a/GOAL.md`): eseguire goal-setup (costruire PHASES.md),
poi prima azione da contratto = **spec di fase A** (spec-first: IR/piano statico,
formato interno pesi da GGUF, contratto dei tap, telemetria, soglie di conformance).
La direzione completa è in `docs/engine/direction.md` (ancorarsi a §7 ordine di
costruzione e §9 prime azioni). Branch di lavoro: `engine/fase-a` (merge su main =
ruling PI).

## 2. State delta (sessione 10)

- **`prof.html` + `src/prof/`**: dispatch profiler in-SPA (patch prototype in un worker
  che delega allo stesso BenchServer; export JSON = forma del tool + BenchCell). Rende
  i run cross-device manuali (~3 min/device). Test unit in `tests/prof.test.ts`.
- **M1 chiusa** (`estimates.md` §8): run M4 (f32+f16) e S22 (f16) + cross-validazione
  4090. Fatti: 269 dispatch/token e 7 submit/token **invarianti di device e quant**
  (totali identici al byte); encode CPU M4 ~2.5 µs/dispatch, S22 **~67 µs**
  (~18 ms/token solo encode); tok/s sotto patch non sostituiscono i bench (S22 −40%).
- **Docket #10 e #11 chiusi**; 4 doc deep-dive corretti (270/7, non ~34/1).
- **[VERIFY] ruling #13 chiusi**: GLM-4.7-Flash ha MTP nativa (`num_nextn_predict_layers:
  1` in config.json) e benchmark indipendenti (AA Intelligence Index 23, #11/130 classe
  4B-40B). Config verificata: MLA 576 float/layer/token, expert ~5.3 MB q4.
- **`docs/engine/direction.md`** scritto: tesi, narrow per iscritto, scala modelli,
  ordine fasi A-D, rischi, prime azioni.
- **Goal `engine-fase-a` aperto** (GOAL/docket/journal/digests seedati).
- Due ruling di processo (salvati in memoria harness): doc stale si corregge subito
  senza chiedere; **push su origin/main autorizzato a fine task**. Primo push fatto
  (ff5b3d7..ca65ae7 + questa sessione).

## 3. Open threads

- Fasi B-D del motore (KV/OPFS → MoE/paging → spec-dec/LoRA/evals): sequenziate in
  direction §7, ciascuna avrà spec e goal propri dopo il gate first-light.
- S22 q4f32_1 non completa i run ("empty timeline", EOS immediato) — baseline S22 = f16;
  nota aperta non bloccante (estimates §8.4).
- Sweep fase 1b (wllama/transformersjs su S22) ancora fuori goal.
- GLM-5 uscito (API): rivisitare il panorama §H del ledger quando si deciderà v2.

## 4. Landmines

- Chrome headless Linux/NVIDIA → SwiftShader: driver Playwright con HEADED=1. Chrome
  lanciato da shell sandboxata → SwiftShader anche headed (visto in sessione 10).
- Chrome quantizza i timestamp GPU (~100 µs); `performance.now()` worker quanto 5 µs.
- Device senza `requiredLimits` nasce a 128 MiB binding → garbage silenzioso.
- Chrome branded Linux/NVIDIA NON espone shader-f16 (M4/S22 sì) → f32-first sul
  dev-loop; equità da dichiarare nei confronti pubblici.
- Il patch del profiler perturba i device CPU-encode-bound (S22 −40% tok/s): mai usare
  i tok/s dei file prof come bench; telemetria del motore = zero-overhead da spenta.
- Residuo non attribuito del budget 4090 = 33%: non assegnarlo a leve senza
  timestamp-query nel runtime (si scioglie in fase A).
- `erasableSyntaxOnly` in tsconfig; righe doc valide per `@mlc-ai/web-llm 0.2.84`.
- Dev server vite di sessioni vecchie vivi su :5173-:5177 (ascoltano solo su [::1]).
- llama.cpp = SOLO oracolo (ruling #14): mai vendored/linkato nel motore.

## 5. Docket (decisioni PI pendenti)

4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10-fase-1b qualityScore
     (si aggancia alla sez. D del ledger, goal evals futuro); #8 sorveglianza wllama
     (v3.1 ha WebGPU via LlamaWeb — rilevante per i confronti del benchmark pubblico).
16. Headline del benchmark pubblico (ledger §E): curva di frontiera vs alternative —
    ruling non urgente, serve prima del goal benchmark.
