# Engine fase B1 — memoria I (KV e persistenza): Design

**Data**: 2026-07-29 · **Stato**: bozza — ruling PI richiesto (docket goal item 3)
**Direction**: `docs/engine/direction.md` §7/§8.1/§8.3 · **Contratto**: `.harness/goals/engine-fase-b1/GOAL.md`
**Riferimenti**: `docs/engine/study/ds4.md` §2 (KV checkpoint), `docs/deep-dive/kv-cache-layout.md`,
sim `results/engine/prefill-sim-4090-*.json`, diag `docs/engine/tsq-diag-2026-07-29.md`

## Obiettivo

Il motore prefilla a chunk multi-token (M≤8), fa rollback della KV con un length
pointer e ri-entra in un contesto salvato su OPFS più in fretta che ri-prefillando —
senza perdere né la parità numerica né i numeri di decode della fase A. Tutto ciò che
non serve a questo è fuori fase.

## Forward multi-token

- **Chunk M=8** (ultimo chunk parziale ammesso; M è un parametro del piano, mai
  hardcodato — lo stesso percorso servirà alla verifica spec-dec in fase D, vincolo
  già posto dalla spec A §Piano statico).
- **Piano prefill compilato a parte** (la spec A lo prevedeva; la fase A ha shippato
  solo il piano decode con prefill sequenziale): stessi step del piano fuso decode ma
  con kernel GEMM small-batch — x diventa [M×896], ogni GEMV fuso ha la variante a M
  colonne di output per riga di peso (stesso loader di blocco Q4_0/Q8_0, dequant fusa
  invariata: si riusa la riga di peso per M attivazioni → banda pesi /M).
- **Attention di chunk**: kernel dedicato con M righe di query, maschera causale
  intra-chunk (riga i vede KV [0, posBase+i]); KV append di M righe per layer in un
  dispatch (pos base + offset di riga). RoPE per-riga con pos per-riga.
- **lm_head + argmax SOLO sull'ultima posizione dell'ultimo chunk** (misurato in sim:
  il lm_head per-posizione costa ~0.4 ms/token dei 5.2 della baseline).
- **Zero readback durante il prefill** (l'input è il prompt, noto): embedding di tutto
  il prompt dequantizzate CPU-side in un buffer unico al via, copy per-chunk
  nell'encoder (pattern validato dalla sim, `decodeMatch=true`); una sola sync a fine
  prefill. **Submit ogni ~64 token** (8 chunk) — knee misurato dalla sim (il submit
  unico peggiora: 4.22 vs 3.78 ms/token).
- **Tap invariati** (contratto DeepSpec): semantica identica alla fase A — il buffer
  tap contiene l'hidden post-layer dell'**ultimo token processato** (896 f32); nel
  prefill la copy è una per chunk (ultima riga). Nessuna nuova API.
- **Parità**: la conformance (gate doppio fase A) gira col percorso prefill M>1
  attivo — le posizioni confrontate sono le stesse, cambia solo come si riempie la KV.

## Crop

- Il motore acquisisce un **length pointer interno `kvLen`** (fase A: pos era solo
  esterno). Contratto (postura ds4, validazione hard):
  - `crop(toLen)`: richiede `toLen ≤ kvLen`, imposta `kvLen = toLen`. Zero lavoro
    GPU: le righe KV si sovrascrivono per posizione e l'attention legge solo
    [0, pos] — le righe oltre `kvLen` sono garbage mai letto.
  - `forwardToken(token, pos)`: **`pos === kvLen` o throw** (niente più posizioni
    libere: il pointer è l'unica verità); `kvLen++` a forward riuscito. Il piano
    prefill avanza `kvLen` di M per chunk.
  - `reset()` = `crop(0)` (sostituisce il commento "basta ripartire da pos 0").
- Prova meccanica (fase 4): "genera N, crop a P, rigenera" vs run fresco dallo stesso
  prefisso → sequenze token IDENTICHE (exit 0, JSON in `results/engine/`).

## Formato prefix-cache

Traduzione browser del KV checkpoint ds4 (§2 dello studio), **ristretta al caso senza
tokenizer** (fase B1 lavora a token-id; la chiave testuale di ds4 esiste per la
ritokenizzazione BPE, problema che non abbiamo ancora):

- **Chiave**: SHA-256 (`crypto.subtle`) di `layoutVersion ‖ sha256(GGUF) ‖ tokenIds
  (u32 LE)`. File: `/kvcache/<primi-32-hex>.kvc` in OPFS.
- **Punto di salvataggio**: dopo il prefill di `prompt[0..len-2]` (kvLen = len−1) —
  il flusso di decode riparte forwardando `prompt[len-1]`, quindi i logits si
  ricalcolano naturalmente: **niente logits nel payload** (diversamente da ds4, che
  li salva per riprendere il sampling; noi rifacciamo 1 forward, ~8 ms).
- **Envelope binario little-endian** (validazione hard, mismatch ⇒ throw):
  - header fisso: magic `"BKV1"` (u32) · version (u32) · lastUsedMs (f64, offset
    fisso per l'update in place) · hitCount (u32) · metaBytes (u32);
  - meta JSON: `{ modelSha256, layoutVersion, nLayer, kvDim, ctxMax, tokenCount,
    createdAt }`;
  - payload raw: `tokens u32[tokenCount]` · per layer: K f32[tokenCount×kvDim] poi
    V f32[tokenCount×kvDim].
  - Dimensione: 469 token ⇒ ~11.5 MB (24 layer × 2 × 469 × 128 × 4 B).
- **Lookup v1: match esatto full-prefix** (hash di `prompt[0..len-2]`). La ricerca
  longest-prefix stile ds4 è v2 (fase C, quando il paging la giustifica) — narrow
  dichiarato.
- **I/O**: `FileSystemSyncAccessHandle` nel worker (banda misurata: write 2.2 GB/s,
  read warm 7.5-11.7 GB/s ⇒ ~10 ms per 11.5 MB; il regime freddo è disco-bound e va
  dichiarato nel report). Save: copy `kCache/vCache[0, kvLen)` → staging → OPFS.
  Restore: OPFS → `writeBuffer` → `kvLen` impostato.
- **Modulo**: `src/engine/kvstore.ts` — codec dell'envelope **puro** (testabile in CI
  senza OPFS/GPU) + I/O OPFS dietro interfaccia minima.
- Restore in **worker nuovo** (sessione fredda) con continuazione token-identica al
  run ininterrotto = prova end-to-end (fase 5).

## Quota/eviction

- Budget disco: **512 MB** default (configurabile alla createEngine/store init).
- Eviction **LRU su `lastUsedMs`** (update in place a ogni hit; hitCount contatore).
  Niente scoring a densità ds4 (half-life, anchor): v1 non ha abbastanza entry da
  giustificarlo — dichiarato.
- `QuotaExceededError` in save: evict LRU e retry una volta, poi throw.
- Entry corrotta/stale al restore (magic/version/modelSha/layout mismatch): **throw**
  (contratto); il chiamante può cancellare l'entry e fare fallback a prefill pieno.
  La cache non deve mai né brickare il motore né degradare in silenzio.

## Soglie

Vincolanti per i verifier da qui in poi:

1. **Prefill ≥3×**: prefillMs mean ≤ **1/3** della baseline seq same-day (fase A:
   ~2.44 s / 469 tok ⇒ soglia ~810 ms). Fissata col criterio di Pareto (ruling PI
   2026-07-29): floor dei trucchi senza kernel nuovi = 1.53× (misurato), analitico
   M=8 = 5-10×.
2. **Non-regressione decode**: decodeToksPerSec.mean **≥ 120** (fase A: 122.0,
   tolleranza ~2%); profiler in finestra decode: `createBindGroup=0`,
   `submit/token=1`, `dispatch/token ≤ 130`.
3. **Parità invariata**: gate doppio fase A (top-1 vs cpuref-f64 ≥99% E vs golden
   ≥97%, ≥512 token) col prefill M>1 attivo.
4. **Restore < re-prefill** sullo stesso prefisso, entrambi i numeri nello stesso
   JSON (warm; il cold dichiarato se misurabile).
5. **Telemetria**: liv.1 invariato (overhead da spenta <1%); liv.2 resta opt-in
   (`telemetryGpu`) — dal fix B1 è funzionante e non perturba (gate: conformance
   PASS con liv.2 attivo, già misurato in fase 2).

## Struttura

```
src/engine/
  gpuforward.ts    + kvLen/crop, piano prefill M≤8, saveKv/loadKv (copy GPU↔CPU)
  kernels/wgsl.ts  + gemmQuant M-colonne (q4_0/q8_0), attnPrefillChunk, ropeChunk,
                     kvAppendChunk (stesso pattern di specializzazione della fase A)
  kvstore.ts       envelope codec puro + I/O OPFS (SyncAccessHandle, solo worker)
  engine.worker.ts + modalità rollback-check e prefix-cache-check
tests/engine-kvstore.test.ts     codec envelope, chiave, LRU (CI, zero GPU)
tests/engine-chunking.test.ts    piano di chunking M≤8, semantica crop (CI, zero GPU)
scripts/kv-rollback.mjs          report rollback (exit 0/1)
scripts/prefix-cache.mjs         report restore-vs-reprefill (exit 0/1)
```

La sim `prefillBatched` (temporanea, fase 1) viene RIMOSSA quando il piano prefill
vero la sostituisce — stesso destino dei knob `tsqDiag` a diagnosi confermata stabile.

## Non-goals di fase B1

Tokenizer · lookup longest-prefix (v1 = match esatto) · compressione/quantizzazione
KV · sliding window · floor dispatch <130 (B2 — le occasioni di fusione viste si
annotano nel docket) · spec-dec oltre il parametro M (fase D) · salvataggio logits
nel checkpoint (ds4 lo fa, noi ricalcoliamo 1 forward) · multi-sessione concorrente
sulla stessa cache.

## Ordine di implementazione (= PHASES 3→4→5→6)

1. Kernel GEMM small-batch + piano prefill + unit chunking → conformance exit 0 con
   M>1 (fase 3).
2. `kvLen`/crop + unit semantica + report rollback (fase 4).
3. `kvstore.ts` codec + unit CI, poi I/O OPFS + report restore in worker nuovo
   (fase 5).
4. Bench prefill vs baseline same-day + non-regressione decode + profiler + chiusura
   (fase 6).

## Rischi specifici della fase

- **GEMM small-batch che non scala** (kernel M-colonne peggiore di M GEMV su GPU
  grandi — già visto con l'esperimento GQA 2-wg della fase A): mitigazione = la
  soglia 3× ha margine 2-3× rispetto all'analitico; fallback dichiarato = chunk M
  più piccolo (M=4/M=2) prima di rinegoziare la soglia via docket.
- **Restore "warm" illusorio**: la banda OPFS misurata è a page-cache calda; il
  restore in sessione davvero fredda può essere disco-bound. Il report dichiara la
  condizione di misura; il gate resta restore < re-prefill (margine ~80×, regge
  anche a freddo).
- **`pos === kvLen` hard può rompere usi interni della fase A** (conformance riparte
  da pos 0 per prompt): coperto da `reset()`/`crop(0)` espliciti nei runner — da
  verificare in fase 4 che tutti i call-site usino il pointer.
