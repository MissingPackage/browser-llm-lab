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
