# Engine fase B2 — floor di decode (attention split + multi-step): Design

**Data**: 2026-07-30 · **Stato**: **approvato** (ruling PI 2026-07-30, docket goal
item 3 — decisioni a-g con le raccomandate: soglia 230, token_embd su GPU,
dispatch ≤100 archiviato)
**Direction**: `docs/engine/direction.md` §7 (fase B) · **Contratto**: `.harness/goals/engine-fase-b2/GOAL.md` (v2, re-scope opzione (a))
**Riferimenti**: attribuzione `results/engine/decode-attrib-4090-*2026-07-30*.json`
(ctx pieno + ctx32), microbench `results/engine/attn-bench-4090-2026-07-30T14-59-*.json`,
`docs/engine/tsq-diag-2026-07-29.md` §Conseguenze (corretta), journal goal B2 it.1.

## Obiettivo

Portare il wall di decode al contesto di bench (~570) verso il floor GPU: riscrivere
l'attention decode per usare la GPU intera (split sul contesto) e amortizzare la sync
per-token col loop multi-step. Dati di partenza (attribuzione it.1): wall 8.09 ms/tok
= gpuBusy 6.46 (attention ~4.0) + sync 1.59 + encode 0.05. Tutto ciò che non serve a
questo è fuori fase.

## Attention split

- **Kernel a due pass, stile flash-decoding** (implementati e misurati in isolamento,
  fase 2): `attnSplitPartWgsl` partiziona il contesto in blocchi da **CHUNK=64**
  posizioni (1 posizione/thread, wg da 64); ogni workgroup (head h, partizione p)
  calcola scores del suo blocco, max locale m, somma locale l e out parziale NON
  normalizzato, scritti in `partials [nHead × S_MAX × (headDim+2)]` (14×16×66 f32 ≈
  58 KB). `attnSplitReduceWgsl` (nHead wg) combina in log-sum-exp esatto:
  M=max mₚ, L=Σ lₚ·exp(mₚ−M), out=Σ oₚ·exp(mₚ−M)/L.
- **Griglia FISSA (nHead, S_MAX=16)**: il piano statico resta immutabile (bind group
  precostruiti, zero dispatch dinamici); le partizioni oltre n escono su
  `begin >= n`. Niente dispatchWorkgroupsIndirect.
- **Rope/append invariati nel contratto**: rope di q per-wg (replicata per
  partizione, costo HALF trascurabile); la partizione owner di pos fa rope k_cur +
  append con lo stesso owner-rule del kernel fuso (h%GROUPS==0) — nessuna dipendenza
  cross-workgroup, il corrente si legge dalle copie locali.
- **Numeri misurati (microbench, pesi/KV sintetici deterministici, 480 op/submit)**:
  split ~28-38 µs/layer PIATTO da ctx 64 a 1024; fuso 29→219 µs (lineare). A ctx 576:
  144.3 vs 31.8 µs = 4.53×; a ctx 1024: 7.18×. Parità: maxΔ vs riferimento CPU f64
  IDENTICO al kernel fuso a ogni contesto (5.7e-8 @ ctx64, ~3e-3 @ ctx lunghi =
  accumulo f32, stesso valore per entrambi) — lo split è numericamente equivalente.
- **Costo accettato**: a ctx≤64 lo split costa +9 µs/layer (+0.2 ms/token). NIENTE
  switch runtime fuso/split (due piani = complessità; i contesti corti sono già
  veloci). Il kernel fuso `attnFusedWgsl` resta nel sorgente per il piano non-fuso
  di debug, non nel piano di produzione.
- **Dispatch**: attention passa da 1 a 2 dispatch/layer ⇒ ~123.6+24 ≈ 148
  dispatch/token. Vedi §Soglie per l'archiviazione del target ≤100.

## Decode loop multi-step

- **K forward per submit** (default K=8, parametro 1..8 del piano; K=1 ≡
  comportamento B1, tenuto come oracolo per token-identity). Per-step: uniform P da
  slot precompilati (stride 256) copiati nell'encoder — pattern pcSlots del prefill
  B1, già validato.
- **Token feedback on-GPU**: l'argmax del passo i (amaxOut) è l'input dell'embedding
  del passo i+1. Serve un kernel `embedGatherQ4` che dequantizza la riga di
  token_embd (Q4_0) direttamente su GPU leggendo il token id da amaxOut ⇒
  **token_embd sale su GPU (~68 MB q4)** — deviazione dichiarata dalla scelta
  fase A "embedding CPU-side" (era legata al readback per-token, che sparisce).
- **Readback 1/K**: dopo ogni step, copy di amaxOut in staging a offset i×4; una
  sola mapAsync a fine batch restituisce i K id. kvLen advance(K) a batch riuscito.
- **EOS/stop mid-batch**: la CPU vede i K id a fine batch; token oltre EOS si
  eliminano con `crop()` (il rollback B1 rende l'operazione esatta by design).
  Stop richiesto dall'app: tra batch.
- **Fallback pre-negoziato** (rischio embedding gather): pipelining del readback con
  doppio staging (encode step i+1 prima della mapAsync di i). Nota dal verifier
  it.1: su 4090 vale poco (encode 0.05), è il fallback di ripiego, non il piano.
- **Conformance**: teacher-forcing richiede feed del golden ⇒ gira a K=1 (esercita
  kernel split + piano nuovo); il percorso K>1 è coperto dal gate token-identity
  greedy (≥256 token identici al per-token).

## Streaming

- Contratto app: i token arrivano **a raffiche di K** (callback/postMessage con
  array di id). Latenza percepita del primo token del batch ≈ K×wall_token; per uso
  interattivo K è configurabile (K=1 ripristina il flusso B1). Nessuna API nuova
  oltre al parametro K e alla consegna a raffiche.

## Tap e telemetria

- **Tap**: semantica invariata (buffer = hidden post-layer dell'ULTIMO forward);
  con K>1 la copy avviene per step, `readTap` legge l'ultimo. Zero step emessi se
  taps=[].
- **Telemetria liv.1**: encode misurato per batch, riportato /K.
- **Telemetria liv.2**: un pass per step ⇒ timestampWrites per step, ring 64
  invariato, tsqIdx++ per step; flush/armTsq invariati (mapAsync SOLO post-submit,
  landmine tsq-diag). Zero-overhead da spenta by design.

## Metodologia gpuBusy (bench)

- Headline = repliche telemetria OFF (come B1). gpuBusy/quota fuori-GPU dal blocco
  di repliche liv.2 ON adiacenti, stesso prompt/finestra, dichiarati nel JSON con
  la finestra di misura. MAI confrontare gpuBusy e wall di finestre/contesti
  diversi (lezione it.1, ora constraint di contratto). Prerequisito feature
  timestamp-query dichiarato nel report (device futuri senza tsq: quota n/d).

## Soglie (criterio di Pareto su attribuzione + microbench)

Predizione additiva a ctx 576: gpuBusy 6.46 − 24×(0.1443−0.0318) = **3.76 ms/tok**
⇒ K=1: 185 tok/s · K=4: 238 · K=8: **249**. Discriminanti: sync-only (senza kernel)
satura a ~149; kernel-only (senza multi-step) a ~185.

- **decode ≥ 230 tok/s** (mean, telemetria OFF, baseline same-day nello stesso
  report): sopra ENTRAMBI i plateau a leva singola ⇒ prova che kernel E multi-step
  funzionano insieme; margine ~8% sotto la predizione 249. [FISSATA dal ruling PI
  2026-07-30: 230, non 240 — il gate è un minimo, non un tetto.]
- **prefill ≤ 810 ms** (soglia 3× B1 invariata — il percorso prefill non si tocca).
- **token-identity K>1 vs K=1**: ≥256 token IDENTICI, exit 0.
- **conformance**: gate doppio invariato (≥99% cpuref-f64 E ≥97% golden, ≥512 tok).
- **profiler**: createBindGroup=0; submit/token = 1/K (a K=8: ~0.13); dispatch/token
  ≤ **160** (nuovo bound sanity: 148 attesi).
- **dispatch ≤100/token: ARCHIVIATO** (proposta, da ratificare nel ruling): l'
  attribuzione mostra che su 4090 il floor GEMV (2.4 ms a ctx corto) è lavoro reale
  di banda/occupancy, non overhead di dispatch; lo split ne AGGIUNGE 24 e il wall
  scende comunque. Il target torna vivo per mobile (T_fisso/dispatch domina su S22)
  — resta rimando ledger §I con questo trigger.
- **telemetria**: overhead da spenta ≤2% nel bench; liv.2 con gpuMs reale non-null.

## Quota fuori-GPU attesa (diagnostica, non gate)

wall K=8 ≈ 4.01 = gpuBusy 3.76 + encode 0.05 + sync/8 0.20 ⇒ quota ~6%. Riportata
nel JSON accanto a gpuBusy (guard v2 = parità + non-regressione + headline, il
guard-rail v1 sul gpuBusy è rimosso: ridurlo È l'obiettivo).

## Unit CPU-side (convenzione CI-senza-GPU)

- piano split: partizioni/owner/griglia per (pos, CHUNK) — pure function testata.
- piano multi-step: slot P per batch, boundary, advance/crop su EOS mid-batch.
- reduce log-sum-exp: riferimento JS vs formula (proprietà: equivale alla softmax
  monolitica su input concatenati).

## Rischi

1. **Proiezione additiva**: l'interazione fra kernel nel pass (occupancy, hazard
   partials part→reduce) è già nel numero misurato (480 op serializzate), ma il
   piano integrato può differire — la soglia 230 tiene margine; se il bench
   integrato manca il gate con kernel conformi, il delta va attribuito PRIMA di
   rinegoziare (attrib re-run, stesso metodo it.1).
2. **embedGatherQ4**: +68 MB residenza GPU e kernel nuovo sul percorso critico di
   parità — mitigazione: unit dedicata (riga dequant vs dequantQ4_0Row CPU) +
   conformance K=1; fallback = pipelining readback (sopra).
3. **Contesti corti**: +0.2 ms/token sotto ctx~128 — accettato e dichiarato.
4. **Error-scope**: ogni nuovo percorso di encode (batch multi-step) nasce col
   contratto error-scope di B1 (throw sincrono attribuibile; diag a cold-start).
5. **Timebox fase 3**: 4 iterazioni, clausola di split B2/B3 pre-negoziata via
   docket (decide il PI).
