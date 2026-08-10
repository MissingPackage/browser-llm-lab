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

## it.11 (2026-08-10)

Fase 5 prima metà, verifier PASS (top-1 e near-tie ricalcolati da zero).
**9B CONFORME: soglia ratchet 1000/1024 = 97.656%** con analisi near-tie
allegata al pin (24 miss TUTTI near-tie: 23/24 = top-2 oracolo, margine
mediano 0.066 logit, zero >1 — firma benigna f32). Sorpresa dal file:
il 9B ha embd/head INVERTITI rispetto al 4B (Q4_0/Q6_K) → orchestratore
ora parametrico su entrambi gli orientamenti. Riferimenti 9B: decode
14.55 tok/s · prefill 15.4 · TTFT 40.3 s (banda-pesi-bound, come atteso).
Due modelli della famiglia CONFORMI in produzione GPU. Next: it.12 = WP
decomposizione gap (llama.cpp Vulkan stesso GGUF 9B, p512/n64).

## it.12 (2026-08-10)

**FASE 5 COMPLETA** (5/9), verifier PASS con bench nativo rieseguito.
Il WP gap del contratto è FATTO: **decode 5.18× (4B) / 4.62× (9B)** vs
llama.cpp Vulkan stesso GGUF, scomposto con MISURA (readback 5.1/3.6
ms/token = decode−prefillSeq; compute residuo 4.6×/4.4×). Prefill
183×/171× = assenza di batching, non gap. Leve per ROI (docket item 10,
solo registrazione): 1. prefill batched (pattern GLM), 2. decode
multi-step no-readback (−5.1 ms misurati, pattern GLM B2), 3. fusione
dispatch, 4. tuning+dot4I8Packed (fase 6), 5. spike subgroup-matrix.
Per il writeup: pubblicare 4.6-5.2× a parità di regime; prefill solo
post-batching. Next: it.13 = fase 6 (leve bounded da contratto).

## it.13 (2026-08-10)

**FASE 6 COMPLETA** (6/9), verifier PASS (probe rieseguito = identico;
giudizio esplicito: done-when soddisfatto in lettera e spirito).
dot4I8Packed e tuning tile: escluse per ora COI NUMERI del WP (il collo è
readback+dispatch+fusione, non l'ALU; il +41% è sul batched che non
esiste). **Spike subgroup-matrix: il probe RIBALTA la fonte** — la
feature è esposta e concessa sul nostro harness (Chrome stable + flag
runner), ma INT8-ONLY (16×16×32, {u8,i8}→{u32,i32}); la misura di
throughput richiede l'infra int8-packed = converge con dot4I8Packed,
rivalutazione dopo il prefill batched. Spec §7 corretta subito. Next:
it.14 = fase 7 — il MoE 35B-A3B parametrizzato (il pezzo grosso rimasto).

## it.14 (2026-08-10)

Fase 7 slice 1, verifier PASS (semantica confrontata riga-riga con la
fonte, oracolo gguf-py rigenerato = identico). **Q4_K expert in casa**:
dequant ancorato a gguf-py su byte REALI del 35B (<1e-7), gemv ktestato a
dims expert reali (82/82). **cpuref MoE dalla fonte** (softmax 256 senza
bias → top-8 → norm con clamp esatto → NESSUNO scale → shared con gate
sigmoid SCALARE — tutto diverso da GLM, letto non assunto), 3 property
test. Suite 375. Next: it.15 = reader lazy (20.9 GB) + cpuref e2e 35B
vs golden smoke, poi forward GPU con paging C3c.

## it.15 (2026-08-10)

Fase 7 slice 2. **cpuref 35B-A3B == ORACOLO al primo run (130 s)**: il
terzo modello della famiglia — e il primo MoE — ha la comprensione
provata (router softmax-top8-clamp, shared gate scalare, DeltaNet,
Q4_K/Q6_K, reader lazy da fd per il 20.9 GB). Verifier: FAIL
amministrativo (entry journal mancante al commit — sanata; sostanza
tutta PASS con e2e rieseguito e oracolo rigenerato identico). Elevata a
docket item 7: i golden q35 committati portano "arch":"deepseek2" falsa
in metadata (tool da sanare prima del paper). Next: it.16 = forward GPU
35B con paging C3c parametrizzato — l'ultimo grosso pezzo di fase 7.

## it.16 (2026-08-10)

Fase 7 slice 3a, verifier PASS (ktest rieseguiti, blk34 riscontrato
Q6_K-down dal file). **DECISIONE dichiarata**: residency.ts GLM resta
INTATTO (core indurito C3c); il paging q35 nasce come strato NUOVO
parametrico per costruzione — rifitting C3c pieno in fase 8. **Blocco MoE
35B su GPU con pesi reali == cpuref a L2rel 2-3e-7** (entrambe le classi
down, arena offset-binding = seme del paging; bug repack Q6_K 210→212
preso dal run). Kernel axpy nuovo. ktest 84/84, suite verde, GLM intatto.
Next: it.17 = forward GPU 35B completo (arena+LRU on-miss) + argmax gate.

## it.17 (2026-08-10)

**IL 35B-A3B GIRA SU GPU CON PAGING: 5/5 vs oracolo** (residenza 70.3% al
budget 12 GiB, firma routing esatta, zero OOM). Root-cause hunt da
manuale sul 0/5 iniziale: tap debug → selezione ok → x post-attn 9.8e-8 →
pass dinamici NO-OP silenziosi → console worker catturata → **arena
monolitica 11.8 GB oltre maxBufferSize E cap adapter 4 GB** → arena a
CHUNK ≤2 GiB + needs slabClassBytes. Esperimento discriminante: ktest MoE
in single-pass = e-7 (ordering sano). Verifier: sostanza tutta PASS con
ricalcoli al byte; FAIL puntuale (la run full era abortita: mancava il
golden full 35B) SANATO — fix arch golden.cpp (item 7), golden full in
generazione, conf GPU full (~12 h) a seguire per la soglia ratchet.
TUTTI E TRE i modelli della famiglia girano su GPU nel motore.

## it.18 (2026-08-10)

**FASE 7 COMPLETA** (7/9), verifier PASS (top-1 e paging ricalcolati al
numero). **Soglia ratchet 35B: 1013/1024 = 98.926%** — la migliore dei
tre; 11 miss tutti near-tie (mediana 0.204 logit). E il paging ha
lavorato SUL SERIO: 9.06M selezioni, hit 98.55%, **121 421 eviction LRU**,
234.7 GB on-miss, zero OOM al budget — il regime C3c del done-when con
evidenza piena. Run 122 min (stima 12h pessimista 6×). Golden full con
arch reale (fix golden.cpp). Next: it.19 = fase 8 (tier mobile/8/12/16 +
recall 256-wide + bandmodel rifittato), poi fase 9 chiusura.

## it.19 (2026-08-10)

Fase 8 slice 1, verifier PASS (numeri ricalcolati, onestà giudicata).
**Bench per tier 35B con hostState**: 8/12/16 GB → decode 0.79/2.00/3.40
tok/s, residenza 23/47/64%, hit 77/91/94% — il COLLASSO IN SCARSITÀ
misurato (4.3×) con attribuzione al hit-rate (~73 miss/token al tier 8).
Frame correttezza-prima dichiarato ovunque. Tier mobile 4B: cap 3 GiB
rispettato (2.48), proiezione parametrica → conferma prefill-bound.
Fix await-prefill (mapAsync concorrenti sul MoE); device-lost arena 12
(baseline VRAM +240 MB vs fase 7) risolto con arena 11 + margine
dichiarato. Next: it.20 = recall oracolo 256-wide + bandmodel rifit.

## it.20 (2026-08-10)

**FASE 8 COMPLETA** (8/9), verifier PASS (LSQ ricalcolata a mano, JSON
deterministici). **Recall lookahead 256-wide: 82.67% @8** (GLM 91.92%)
con scostamento SPIEGATO: softmax-256 senza bias vs sigmoid+bias-64,
bersaglio 3.1% vs 6.25%, profilo per-layer (L0 44%, max 90%) — prefetch
utilizzabile ma beneficio minore, registrato non promesso. **Bandmodel
q35: 17.64 ms/miss** (residui ≤6.3%, limiti dichiarati: 3 punti, base
negativa, regime correttezza-prima col repack JS). Note per fase 9: la
spiegazione va in direction (lettera del done-when); recall@4 ha
denominatore 8 (tetto 50%) — esplicitare nella migrazione ai doc.
Next: it.21 = FASE 9 — LA CHIUSURA (checklist contratto voce-voce,
non-reg GLM piena, direction/ledger/HANDOFF, docket triage).
