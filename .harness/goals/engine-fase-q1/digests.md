# Digests — engine-fase-q1

## it.0 (2026-08-10)

Goal APERTO al tag `goal-engine-fase-q1-start`. Spine: 9 fasi sequenziali
da recon §4 — il rischio dominante (kernel DeltaNet) isolato kernel-level in
fase 3, prima di ogni end-to-end; leve kernel condizionate al ROI del WP
gap (fase 5→6); MoE 35B in coda perché eredita reader+kernel+parametrizz.
Primo target: fase 1 = spec con SHA GGUF pinnate, piano numerico DeltaNet,
proxy tier mobile, chiusura [VERIFY] subgroup-matrix. Docket-born: nessuna;
plan-check pre-autorizzato (revisione naturale: prima della fase 3).

## it.1 (2026-08-10)

Fase 1 DONE, verifier PASS. Spec depositata con ratifica del contratto;
SHA dei 3 GGUF pinnate e cross-verificate. Finding: Q4_0 del 35B non
esiste da fonte pulita → MoE in UD-Q4_K_S (deviazione registrata, docket
item 3; prezzo = dequant Q4_K expert, K-quant già in casa per Q6_K).
[VERIFY] subgroup-matrix chiuso: niente origin trial (ago 2026), solo
Dawn sperimentale → fase 6 resta spike dietro flag. Next: fase 2 (reader
+ tokenizer, download ~29 GB).

## it.2-3 (2026-08-10)

Fase 2 DONE in 2 iterazioni, verifier PASS su entrambe. Pesi in casa (SHA
3/3), reader parametrico (shape dai metadata, inventario completo
426/427/733), PRIMO tokenizer in-engine (BPE byte-level, regex qwen35
verbatim — riscontrata nel binario oracolo). Il gate secco ha preso 2 bug
di fedeltà al protocollo (escape processing, semantica USER_DEFINED) prima
dei kernel. Suite 364 (+5), tsc pulito, GLM intatto. Next: fase 3 —
kernel DeltaNet (il rischio dominante), cpuref-f64 prima del WGSL.
