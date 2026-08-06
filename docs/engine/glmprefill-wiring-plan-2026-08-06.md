# Piano di wiring `prefillChunk` (fase 5 slice 4) — it.31, ricognizione su glmmodel@5d9fd32

Scritto per essere eseguito da un contesto fresco: ogni decisione qui è già
presa, con il razionale; l'implementazione non deve ri-derivare nulla.

## 0. Contratti già validati (non ridiscutere)

- Kernel batch TUTTI bit-identici sul device (it.27-30, ktest 64/64):
  catena expert gather+slot+combine, shexp batch, MLA split batch (rowPast),
  dense sweep 8/8 (gemvQuant×3, gemvQ8Heads, rmsnorm, ropeMlaNorm, kvAppend,
  stridedCopy — i position-dependent leggono rowPos/rowPast storage).
- Piano MoE: `planMoeChunk`/`combineMoeRow` (glmprefillplan.ts, it.26-27) —
  combine k-order = catena esatta del decode (moeOut parte dallo SHEXP).
- `GLM_PREFILL_M = 16` (spec §5 [ASSUMED]).

## 1. API (additiva — `forward()` INTOCCATO)

`GlmModel.prefillChunk(xRows: Float32Array /*M·dModel, M≤16*/, posStart: number,
readLogitsLast?: boolean): Promise<{ routing: GlmRouting[][]; logits?: Float32Array }>`
— il chiamante embedda per riga CPU-side (dequantQ4_0Row) come oggi; l'ultimo
chunk può essere parziale (M<16). `pos === kvLen` contratto ereditato dal
chiamante (gli harness passano posizioni sequenziali).

## 2. Router: CPU per chunk (decisione presa)

`select:"cpu"` con UN readback di M×64 logit per LAYER per chunk ⇒ 46
sync/chunk = **2.9 sync/token a M=16** (oggi 46/token). Il resolve GPU
batched è un'ottimizzazione SUCCESSIVA (negoziabile in C3b), non serve al
done-when. `routerSelect` per riga resta CPU f64 ⇒ selezione BIT-IDENTICA
al per-token (stessi logit per costruzione).

## 3. Pesi expert: bind group a SOTTO-RANGE per unione (decisione presa)

I kernel gather di it.27 sono in variante PLAIN (binding diretti). In
produzione i pesi vivono negli slot dell'arena, ma per il prefill si
costruiscono bind group a sotto-range {offset,size} per ogni expert
dell'unione, DOPO gli ensure (la CPU conosce lo slot; il sync col router
c'è comunque). Costo: |unione|≈40 bind group/layer/chunk, ammortizzati su
M token. NIENTE porting dell'arena head sui gather (ottimizzazione futura,
annotarla nel journal alla chiusura).
ATTENZIONE: il sotto-range punta dentro i buffer di classe della cache
(`slotRef` → buffer+offset, v. `ExpertCache.slotRef`); il layout slab dà
gli offset di gate/up/down (SLAB_DOWN_*.gateQs ecc., come nel ktest it.27).

## 4. Buffer nuovi (tutti M×, allocati una volta a ctxMax noto)

xM(M·2048) fnBM qaBM(M·qLora) qanBM qBM(M·nHead·HL) kvBM(M·576)
row576M(M·576) qCkvM(M·nHead·512) q576M(M·nHead·576)
attnPartialsM(M·partialsLen) attnCkvM(M·nHead·512) attnMlaM(M·nHead·128)
tmpM(M·2048) gateDM/upDM(M·dFfnDense, solo blk.0) hSlots(M·4·1536)
ySlots(M·4·2048) sM(M·2048) wBufM(M·4) logitsBM(M·64)+staging
rowPosB/rowPastB(u32·M, riscritti per chunk) gatherB (u32·4M, riscritto
per (layer,expert) — o un buffer per layer con offset).
≈ M·(576·20·2 + …)·4 ≈ 30 MB a M=16: trascurabile sul budget.

## 5. Sequenza per chunk (speculare a forward(), righe 877-999)

Per layer: attention batch [rmsD_b, gemvQA_b, rmsQA_b, gemvQB_b,
ropeQ_b(rowPos), gemvKvA_b, ropeKPe_b(rowPos), rmsKvA_b] → kpeCopy via
stridedCopy BATCH (non M copy nell'encoder) → kvApp_b(rowPos, stessa cache
del decode) → absorbKb_b, copyCkv_b, copyQRope_b → attnPart_b(rowPast, grid
[sMax,1,M]) → attnReduce_b → voutVb_b → gemvO_b → add su griglia M·D (stesso
kernel, size M·dModel: elementwise, nessuna variante — vale anche per
siluDense).
Dense (blk.0): rms_b + gemvDenseGU_b ×2 + siluDense(M·dFfnDense) +
gemvDenseDown_b + add.
MoE: preRouter batch (rms_b + router gemv f32 batch — VERIFICARE il
generatore del router GEMM: se f32 dedicato, serve la variante batch o M
dispatch nello stesso pass, 64 logit l'uno = costo nullo) → endPass, copy
logitsBM→staging, submit, mapAsync (il sync del chunk) → routerSelect ×M →
`planMoeChunk` → ensure UNIONE (pinned = chiavi unione) + flushSlotTable →
nuovo encoder → per expert dell'unione: bind group sotto-range + pairGather
+ downSlots (grid z = |rows|) → shexp batch (Q5K pair_b + Q6K down_b → sM)
→ moeCombine (xM += sM + Σ w·y, k-order).
Coda: se readLogitsLast, copy riga M−1 di xM → x e riusare `headSteps`
ESISTENTI (bit-identici); hidden staging dalla riga M−1.

## 6. Identità (perché l'atteso è ESATTO, non tollerato)

Ogni kernel per riga ≡ per-token (ktest); routerSelect CPU identico;
l'ORDINE degli ensure (unione asc vs per-token) cambia solo l'assegnazione
degli slot, mai i valori; combine = catena decode. ⇒ argmax identico su
TUTTE le posizioni, e i logit dell'ultima posizione bit-uguali al forward
sequenziale. Il test E2E può pretendere l'uguaglianza esatta.

## 7. Fette di verifica (ordine)

(a) `prefillChunk` compilato + suite verde (il path è additivo, nessun test
    GPU ancora) — commit;
(b) identità E2E CORTA: glmconf.html con param `prefill=M` — replay di
    poche posizioni di p4 confrontando argmax/logit del prefillChunk contro
    forward sequenziale (stesso worker, stessa sessione) — commit;
(c) identità p6 intera (461 posizioni) + gateCpuref/gateGolden verdi in
    modalità prefill — commit;
(d) bench: glmbench con prefill batched → prefill tok/s e TTFT nel report
    (attesi ~5-6 s da spec §5 a parità di kernel) — chiusura fase con
    done-when.

## 8. Landmine specifiche

- HMR: mai run GPU con edit pendenti; ogni fetta committata prima della run.
- Il monolitico mlaAttnDecode NON si tocca (consumatori glmforward/ktest).
- Contatori: submits/dispatches sempre-on (T.dispatches += … come nel
  forward); telemetria per-chunk nel campo esistente (forwards += M).
- Sel di produzione NON è usata dai gather plain: niente scritture Sel nel
  path prefill (il selMiss resta una proprietà del decode).
