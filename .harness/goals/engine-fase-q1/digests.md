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

## it.4 (2026-08-10)

Fase 3 prima metà, verifier PASS. cpuref-f64 DeltaNet con semantica dalla
fonte llama.cpp PINNATA a b10333 (la build dell'oracolo): ordine decay →
lettura → delta → update → output, broadcast k-head h mod nK, l2norm con
eps a floor, softplus gomito 20. 5 identità algebriche in aritmetica
esatta + campione T=12 pinnato (generatore esplicito). Il verifier ha
trovato un buco di discriminanza nel test dell'ordine (variante
decay-dopo-update mascherata da cancellazione esatta) → chiuso subito con
il caso v₂≠0 che separa TUTTI e tre gli ordinamenti. Suite 371 (+7).
Next: it.5 = kernel WGSL DeltaNet + ktest vs cpuref sul campione.

## it.5 (2026-08-10)

FASE 3 DONE, verifier PASS (ktest rieseguiti indipendentemente). 3 kernel
WGSL nuovi (conv+shift, gates, core un-workgroup-per-v-head con decay fuso
— equivalenza al cpuref verificata per linearità); 6 ktest nuovi TUTTI
PASS al primo run reale, inclusa la catena T=12 con stato persistente su
GPU (maxAbs 5.4e-7) e il core a dims REALI hd128. Totale 75/75, GLM
intatto. Il rischio dominante del goal è dimezzato: la numerica ricorrente
regge kernel-level. Next: fase 4 — 4B end-to-end (GQA+mrope+ibrido+ffn,
argmax==cpuref, golden a ratchet).

## it.6 (2026-08-10)

Fase 4 slice 1, verifier PASS con controlli indipendenti forti (golden
rigenerato = identico al committato; mrope e GQA verificati dalla fonte).
**Il cpuref 4B end-to-end CONCORDA con l'oracolo al primo run numerico**:
argmax == llama.cpp su tutte le 6 posizioni generate, teacher-forced su 39
posizioni × 32 layer con pesi reali (130 s, gated Q35_E2E=1). Chiavi:
mrope text-only collassa a NEOX-64 (provato dalla fonte); q+gate fusi;
GQA grouping vs DeltaNet tiling (due mapping diversi, entrambi fedeli).
La comprensione del modello è PROVATA — da qui ogni divergenza GPU è bug
di kernel. Note per fase 4: run-golden-q35.sh con provenance per i golden
full-corpus; binario golden buildato da 5f55650 (≡ b10333 per qwen35).
Next: it.7 = forward GPU 4B (orchestratore sui kernel esistenti+deltanet).

## it.7 (2026-08-10)

Fase 4 slice 2, verifier PASS (ktest rieseguiti, fixture bit-deterministica
riprovata, rope retrocompatibile sui chiamanti storici). **Assembly GPU di
ENTRAMBI i tipi di layer con pesi reali del 4B == cpuref a L2rel ~1e-7**
(linear blk0: catena DeltaNet completa con Q5_K out; full blk3:
deinterleave + QK-norm + rope64 + attnDecode 16/4/256 + gate). Kernel
nuovi: ropeDims opzionale su ropeNeoxWgsl, sigmoidMul. ktest 79/79.
Ogni pezzo del forward è provato: it.8 = pura orchestrazione (32 layer +
embed + head) + gate argmax GPU == oracolo sul golden smoke.

## it.8 (2026-08-10)

Fase 4 slice 3, verifier PASS. **IL 4B GIRA SU GPU: argmax == oracolo
6/6** sul golden smoke (orchestratore q35gpumodel.ts, 562 dispatch/token,
39 posizioni in 1.9 s, full residency). Due pareti Chromium risolte
(fetch/ArrayBuffer 2.6 GB → Range per-tensore; head 527 MB → limiti
device). ktest 80/80. Il motore ESEGUE la famiglia Qwen 3.5 — prima
inferenza browser end-to-end del goal. Restano per fase 4: golden
full-corpus a soglia RATCHET + riferimenti full-resident (it.9).

## it.9 (2026-08-10)

Fase 4 quasi chiusa, verifier PASS (top-1 ricalcolato indipendentemente).
Golden full-corpus 4B committato (8 prompt, 1024 posizioni, provenance
piena) e **SOGLIA RATCHET FISSATA: top-1 = 1012/1024 = 98.828125%**
(docket item 8; run GPU 29 min ad albero congelato; miss sparsi, max 3 per
prompt). Curiosità verificata: stessa cifra del PIN GLM — stesso corpus
testuale, modelli e miss diversi; coincidenza onesta, non copia. Resta per
fase 4: it.10 = riferimenti decode/prefill/TTFT full-resident con
hostState (dichiarati come frame correttezza-prima).

## it.10 (2026-08-10)

**FASE 4 COMPLETA** (fasi 1-4 su 9 in 10 iterazioni), verifier PASS con
ricalcoli esatti. Riferimenti full-resident 4B committati con hostState:
**decode 22.93 tok/s · prefill seq 26.0 · TTFT 25.8 s** — frame
correttezza-prima DICHIARATO (562 dispatch/token, zero fusioni, readback
per token). Lettura: il denso full-resident fa già 23 tok/s senza alcuna
ottimizzazione (GLM paginato: 15.6 dopo tre fasi); la soglia UX 30 è a
1.3× con tutte le leve in canna. Next: fase 5 — 9B + WP decomposizione
gap vs llama.cpp Vulkan (il cuore del confronto onesto per il writeup).
