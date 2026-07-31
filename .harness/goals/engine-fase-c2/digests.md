# Digests — engine-fase-c2

## it.0 (2026-07-31, setup)
Contratto approvato in chat con emendamento non-regressione (gate tok/s =
floor oracolo C1: decode >=13.4 / prefill >=56.6 CPU-only; pregresso Qwen resta
verde). PHASES.md: 7 fasi — (1) floor+golden GLM ‖ (2) spec C2 [gruppo A],
poi 3 reader deepseek2 → 4 MLA → 5 MoE+residenza minima → 6 E2E+gate → 7
chiusura; 3-6 gated dal ruling di spec. Primo target: fasi 1+2. Docket-born:
solo il plan-check (item 1) — nessuna fase ha chiesto authority extra.

## it.1 (2026-07-31, fasi 1+2)
Fase 1 DONE: golden logits GLM dall'oracolo C1 (exit 0, 26.154 prefill,
8×128 posizioni, argmax+top-32, envelope completo; token greedy identici
8/8 ai decode della traccia C1). Fase 2 DONE-pending-ruling: spec C2 con 6
decisioni al PI (kernel Q4_1/Q5K/Q6K vs repack, MLA absorbed, OPFS sorgente
miss, LRU, soglie 99/97/99, timebox 4+4) — docket item 2. Verifier: FAIL su
un claim della spec (Q4_1 era su 4/46 layer, non su tutti) → corretto e
quadrato con residency-sim (media 5.325.512 B esatta); resto tutto PASS.
STOP BY DESIGN: fasi 3-6 gated dal ruling di spec.

## it.2 (2026-07-31, fase 3)
Reader GGUF deepseek2 DONE e verificato: shape GLM (844 tensori, quant mista
per-layer con down_exps Q4_1 solo blk.1-4), dequant CPU Q4_1/Q5_K/Q6_K
portate dall'oracolo e validate su byte REALI del GGUF contro gguf-py
(oracolo indipendente, match f32 esatto), load headless del file da 17.2 GB
(header parse + validazione + bounds). Suite 21 file / 199 test verde, tsc
pulito, zero regressioni. Verifier PASS (porting confrontato riga-per-riga
col C; fixture riverificata con dequant ex novo). A docket: branch policy
(item 3, main-diretto vs lettera del contratto). Next: fase 4 (kernel MLA).

## it.3 (2026-07-31, fase 4 slice 1)
GEMV dequant-fuse dei formati GLM: repack Q4_1/K-quant + 3 kernel WGSL nuovi
(q4_1 in gemvQuant; gemvQ5K con scaleMinK4 6-bit; gemvQ6K con d in coda) +
ktest esteso (5 casi a taglie reali) + runner scripts/ktest-run.mjs. Run su
4090: 16/16 PASS (nuovi: maxRel 5e-7…2e-4), zero regressioni sugli 11 kernel
preesistenti; npm test 199/199; tsc pulito. Verifier PASS con ri-run
indipendente. Watch: q5_K passa via absTol, margine rel sottile. Next (it.4):
cpuref MLA naive f64 + kernel absorbed (proiettori, rope, attention 576).

## it.4 (2026-07-31, fase 4 slice 2)
Kernel MLA absorbed: ropeMlaNorm (RoPE tipo NORM — catch: deepseek2 NON usa
NEOX), gemvQ8Heads (wk_b/wv_b per-head), mlaAttnDecode (MQA su cache 576,
kq_scale 1/sqrt(256), V=c_kv normata). Semantica verificata NEL sorgente
dell'oracolo prima di scrivere WGSL; ktest 21/21 su 4090 (mla-attn-decode
maxAbs 3.7e-8), 199 test node verdi, tsc pulito. Verifier PASS su codice E
semantica. Restano per fase 4 (timebox 2 it.): assemblaggio layer GLM in
gpuforward + conformance layer-level con pesi reali (gate doppio).

## it.5 (2026-07-31, fase 4 slice 3 — CHIUDE LA FASE 4)
Layer GLM assemblato su GPU (glmforward.ts: MLA absorbed, 22 dispatch/token)
+ conformance layer-level con PESI REALI blk.0 vs cpuref f64 naive: L2rel
2.35e-7 su 16 pos decode (4090) — accordo al rounding f32. Identità algebrica
naive↔absorbed in suite (maxAbs<1e-8). ktest 23/23, npm test 200/200, tsc
pulito. Verifier PASS (dualità verificata algebricamente, 22 dispatch contati
a mano). Watch fase 6: stringere il gate layer-level (1e-3 vs 2.35e-7
misurato) a canary di non-regressione. Fase 4 chiusa in 3 it. (timebox 4).

## it.6 (2026-07-31, fase 5 slice 1)
Router MoE: semantica verificata riga-per-riga in build_moe_ffn (sigmoid;
bias exp_probs_b SOLO selezione; pesi = sigmoid senza bias, norm con clamp
6.1035e-5, ×1.8; pesatura dopo il down; shexp sommato a moe_out; tie-break
indice minore). moe.ts nuovo (routerSelect CPU + slab a due size-class
5.308.416/5.505.024 B, offset 256-aligned, packExpertSlab), gemvF32 router,
scaledAccum sul down per-expert, ref f64 indipendente in cpuref. ktest 28/28
su 4090 (moe-ffn-block completo con bind group a offset per-slot), suite
208/208, tsc pulito. Verifier PASS. Next (it.7): slice 2 — residenza minima
(OPFS → staging → cache VRAM LRU per classe + telemetria).

## it.7 (2026-07-31, fase 5 slice 2)
Residenza minima: expertstore.ts (SHA-256 incrementale FIPS 180-4 — subtle
non streama su 17 GB; ExpertOpfsStore con import streaming hash-verificato e
read range SyncAccessHandle; GgufExpertIndex, expert = fetta contigua) +
residency.ts (ExpertCache: slot (classe,buffer,offset), riparto budget
∝ parco 256/2688, LRU pura per classe, pinned intra-token, telemetria gated).
ktest 29/29 su 4090 (roundtrip OPFS→pack→upload→readback byte-esatto,
contatori esatti), suite 219/219, tsc pulito. Verifier PASS (SHA su vettori
NIST indipendenti). WATCH fase 6: ~10 ms/miss con pack CPU dominante ⇒
proiezione ~66 ms/token di stallo a hit 96.4% su budget 74.5 — margine <10
ms sul gate decode; leva candidata: repack pagato all'import. Next (it.8):
slice 3 — forward multi-layer + conformance routing vs traccia C1.

## it.8 (2026-07-31, fase 5 slice 3a)
Forward GLM multi-layer sul path di produzione: glmmodel.ts (createGlmModel —
attention MLA fase 4 + router it.6 + residenza it.7; per layer MoE: submit →
mapAsync 64 logit → routerSelect CPU → ensure×4 pinned → encode shexp+expert
da slot; pipeline condivise, bind group per-slot cached, sopravvivono
all'eviction). cpuref: attention estratta (GlmMlaAttnRefF64) + nuovo
GlmMoeLayerRefF64. ktest 30/30 su 4090: glm-model-2layer (denso+MoE Q4_1,
cache stretta con 5 eviction) L2rel 2.36e-7, routing 6/6, pesi 1.8e-7, 61
dispatch esatti; suite 219/219; tsc pulito. Verifier PASS. Proiezione
full-model: 1.816 dispatch/token (misura, non gate). Next (it.9, ultima del
timebox fase 5): loader OPFS reale + import 17.2 GB + replay traccia C1 +
gate routing set-match ≥99%.

## it.9 (2026-07-31, fase 5 slice 3b — GATE ROUTING → RULING)
Primo run end-to-end sui pesi reali: glmsource.ts (OPFS 17.2 GB, SHA
streaming verificato) + harness glmroute (replay teacher-forced traccia C1).
GATE NON PASSATO: decode 85.85% (p4) / 94.11% (p7) vs ≥99%. Debug come da
spec: discriminatore cpuref-f64 esatto ⇒ 96.20% sul subset, IDENTICO al
motore, STESSI 28/28 mismatch (tutti swap singoli 4°/5°, ~2% già a layer 1)
⇒ router SCAGIONATO, il disaccordo è la numerica q8 dell'oracolo sui
near-tie; soglia 99% mal tarata (autotest C1 usava l'hidden dell'oracolo).
RULING RICHIESTO docket item 4 (racc.: routing informativo, gate su logits
fase 6); timebox fase 5 esaurito (ruling f). Hit residenza 95.9% @12 GiB —
coerente col sim C1. Verifier PASS (ipotesi alternative valutate ed
escluse). Loop in STOP BY DESIGN: fase 6 gated dal ruling.
