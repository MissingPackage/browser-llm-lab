# Spec — engine-fase-c3c: paging in scarsità (slab ctx-aware, prefetch in-forward, tier+AUTOPIN, instant-on)

Data: 2026-08-08 · Goal: `.harness/goals/engine-fase-c3c/GOAL.md` (contratto v1)
Input: WP banda fredda browser (fase 1, `results/opfs-bench/opfs-cold-4090-linux-2026-08-08*.json`),
C1 (recall LOOKA, matrice usage, `results/engine/moe-oracle/`), WP-0 (Belady,
collasso in scarsità), C3b chiuso (decode ottimistico, finding P(dirty)↔prefill),
probe vram-ceiling it.19 (`results/engine/vram-ceiling-minimal-2026-08-06.json`).

## 0. Obiettivo e regime

C3b ha chiuso il regime near-total (≥ ~80% residenza): decode ottimistico,
16.64 tok/s al tetto. La Pareto di WP-0 ha DUE segmenti: sotto ~50% di residenza
il replay collassa (100% token sporchi) e il segmento giusto è **sync + overlap**.
C3c costruisce quel segmento: budget slab che si adatta al contesto invece di
andare OOM, prefetch in-forward che nasconde la latenza di miss dietro il compute,
policy che batte la LRU pura a budget stretto, avvio instant-on. La leva dichiarata
su P(dirty) per l'ottimistico (policy > LRU, Belady ~dimezza — WP-0) vive qui.

## 1. Numeri di ingresso (misurati, non estrapolati)

| Grandezza | Valore | Fonte |
|---|---|---|
| Banda fredda browser, random expert-size | **1.79-1.94 GB/s, p50 2.9-3.1 ms/expert, p95 3.7-4.5** | WP fase 1 |
| Banda fredda browser, streaming seq expert-size | 3.51-3.66 GB/s, p50 1.3-1.4 ms/expert | WP fase 1 |
| Banda calda browser (page cache) | 11.41-11.88 GB/s, p50 0.4-0.5 ms/expert | WP fase 1 (= fase A) |
| Parco expert | 2944 slot (2688 q4_0 @5 308 416 B + 256 q4_1 @5 505 024 B) = 15 678 308 352 B | `residency.ts`, probe |
| Non-expert residente | 1 354 078 720 B (1291 MiB) | probe `required.nonExpertBytes` |
| KV | **108 288 B/token** = nLayer 47 × keyLen 576 × 4 B (f32) | `glmmodel.ts:725`, probe kvBytes/525. NOTA: il "54 KB/token" di HANDOFF §5 era stale (conto f16) — corretto |
| Tetto allocazioni device | 15 947 MiB a host perfetto; sessione minima misurata 14 720 MiB allocati senza OOM | probe it.19 |
| Config di riferimento C3b | tetto 2595 slot @12.88 GiB (88.15%), decode 16.64, TTFT 12.60; b12 2417 slot (82.1%), 11.60, TTFT 16.79 | docket c3c item 2 |
| Hit-rate LRU pura | b11 0.9575, b12 0.9756 (full-corpus, path sync) | routing-conformance 2026-08-03 |
| Recall lookahead oracolo (hidden L → router L+1) | **92.0% @K=8**, 77.5% @K=4 | C1 |
| Ceiling Belady in scarsità | +9-19pp vs LRU a 50%/25%; +0.6pp a 2596 | WP-0 |
| Budget di prova fasi 5-6 | 12 GiB (≈2417), **~50% = 1472 slot**, **~25% = 736 slot** | contratto (mai rinunciarvi) |

## 2. Budget slab ctx-aware (fase 3)

Oggi `budgetGiB` è una costante di config: a ctx 6k il KV (665 MB) + partials
sfonda e il device va OOM. La formula sostituisce la costante:

```
slabBudgetBytes(ctx) = allocCeilingBytes          // tetto misurato del device (probe, sessione dichiarata nel report)
                     − nonExpertBytes             // misurato all'import (oggi 1 354 078 720)
                     − kvBytes(ctx)               // nLayer·keyLen·4·ctx = 108 288·ctx
                     − workBytes(ctx)             // buffer ctx-dipendenti dai consumatori veri: partials MLA ×(1+GLM_PREFILL_M) — il ×16 del prefill chunked, scoperto dall'OOM della prima run (fase 3) — + hiddenCkpt
                     − reserveBytes               // riserva driver/frammentazione: TARATA 512 MiB in fase 3 (era [ASSUMED 256]; due punti OOM: slack sessione viva >247 MiB = staging ring Dawn nel preload + compositor)
slots(ctx) = expertSlots(slabBudgetBytes(ctx))    // riparto esistente (q4_1-first nel modo optimistic, item c3b 6b)
```

- `allocCeilingBytes` è un input di config MISURATO (probe vram-ceiling), non
  una costante inventata: il report di ogni bench dichiara sessione host e
  ceiling usato (pattern hostState).
- Vincolo di sanità: `slots(ctx) ≥ MIN_SLOTS` (= 4·nMoe = 184, il pin-for-replay
  di c3b I3); sotto ⇒ throw esplicito con il numero (niente degradazione
  silenziosa — emendamento 5 di c3a resta).
- **Non-regressione (done-when fase 3)**: a ctx ~525 la formula riproduce il
  regime attuale (decode/TTFT in banda ±5% sul punto b12); a ctx 6144 glmbench
  NON va OOM e il report mostra budget calcolato + VRAM di picco osservata.
- La formula è pura e testata in npm test (unit: monotonia in ctx, clamp al
  parco, MIN_SLOTS, riproduzione dei punti b12/tetto dai loro input).

## 3. Prefetch in-forward (fase 4) — prefetch, NON predictor

**Distinzione di contratto.** Il predictor al CONFINE di token (LOOKA sul hidden
di fine token → selezioni del token dopo) è FALSIFICATO (WASTE, K3, WP-0 b=736:
63.2%→59.2%) e NON si costruisce. Il prefetch IN-FORWARD è un'altra cosa: DENTRO
il token, il tap dell'hidden all'uscita del layer L alimenta il router del layer
L+1 (pesi router: gemv 2048×64, ~130 KB f32 — sempre residenti) e il fetch dei
predetti parte MENTRE il layer L+1 non è ancora arrivato a consumarli. Recall
oracolo: 92.0% @K=8. Il guadagno è nascondere p50 2.9-3.1 ms/expert di miss
freddo dietro il compute dei layer.

- **Path**: decode sync e prefill (ensure per layer già sincrono — il prefetch
  anticipa di un layer la ensure). L'ottimistico di C3b NON si tocca (I1-I5
  intatte: nel path optimistic il beneficio arriva in fase 5 via policy, non
  da fetch mid-token).
- **Flag**: `prefetch: "inforward" | "off"` (default off finché fase 5 non
  misura il delta; il flag vive nel config del forward, telemetria
  zero-overhead quando off).
- **Misura del recall IN-ENGINE (done-when)**: sul corpus C1 (harness routing),
  per ogni layer L: predetti top-K(hidden L, router L+1) vs selezione vera del
  layer L+1 → recall aggregato confrontato con 92.0% @K=8 dell'oracolo.
  Scostamento SPIEGATO nel journal (f32 vs f64, hidden post-shared vs tap
  dell'oracolo), NON gateato.
- **Identità (done-when)**: il prefetch non cambia il forward — ktest esistenti
  PASS, argmax = sync sul campione. Il tap legge, non scrive.
- K di prefetch: parte a K=4 [ASSUMED, dal finding C1 "K ottimo cresce col
  budget, K=8 a budget stretto fa thrashing"]; K è un parametro della policy
  (fase 5 lo misura a 1472/736).

## 4. Policy tier.h + AUTOPIN (fase 5) — colibri §2 tradotto

Base: LRU esistente (clock per slot). Si aggiungono, dietro `policy: "tier" |
"lru"`:

- **eusage** (persistente): count per (layer, expert), u32[2944]; salvato in
  OPFS accanto allo slab file (write atomica tmp+rename, pattern colibri),
  caricato ADDITIVAMENTE all'init. È l'input dell'AUTOPIN.
- **eheat** (sessione): u16[2944], incrementato a ogni selezione, decade con
  right-shift a ogni passata di repin (tier_decay).
- **AUTOPIN**: con storia ≥ [ASSUMED 5000] selezioni, pinna i top-N di eusage
  con budget = min(**12.5% degli slot** (HARD, ruling C1 — assert nel path),
  quota proporzionale alla fiducia hist/[ASSUMED 200k] cap 1). Slot pinnati
  mai evinti (né da LRU né da repair c3b — il pin-for-replay resta separato).
- **REPIN live**: tra i token, scambio del pin più freddo col non-pinnato più
  caldo via score LFRU `heat<<8|recency`; isteresi 25%+4 anti ping-pong, max
  4 swap/passata.
- **Prefetch** (fase 4) inserisce con priorità: gli slot prefetched-non-ancora-
  usati non sono vittime preferenziali entro il token corrente.
- **Misura (done-when fase 5)**: harness routing (path sync), STESSI budget
  1472 e 736 slot, hit-rate tier+AUTOPIN+prefetch vs LRU pura, JSON per
  entrambe le policy a entrambi i budget; delta confrontato col ceiling Belady
  WP-0 (+9-19pp). La policy deve battere LRU, NON raggiungere Belady.

## 5. Modello di banda (fase 6)

Formula chiusa, committata CON TEST:

```
msPerToken(h, ctx, B) = gpuBusy(ctx) + struttura + (1−h)·nExpertUsed·nMoe_eff·missMs(banda)
tokS = 1000 / msPerToken;  missMs = expertBytes/bandaFredda (+ upload)  [componenti dal WP fase 1]
TTFT_freddo(ctx, B) = TTFT_caldo(ctx, B) + nonExpertBytes/bandaSeq + bytesFreddi(ctx, B)·(1/bandaFredda − 1/bandaCalda) − overlapMs
```

con h = hit-rate misurato (fase 5), banda fredda = numeri browser fase 1 (mai
i numeri OS). Done-when: predice i punti MISURATI entro ±15% su 3 budget (12
GiB, 1472, 736 — bench committati) e il TTFT a freddo entro ±15% (verificato in
fase 7). I coefficienti si fissano dai bench di fase 3/5, non si fitta a
posteriori sul punto che si predice.

## 6. Instant-on (fase 7) — definizione operativa

**Stato di partenza (definito, riproducibile)**: OPFS popolata (import fatto),
VRAM vuota (sessione browser fresca), **page cache OS FREDDA** — eviction col
protocollo del WP fase 1 (fadvise sui backing file OPFS, residenza fincore nel
JSON). Senza il terzo requisito "freddo" non è dimostrabile (lezione fase 1:
freddezza provata, non asserita).

**Meccanismo**: router + shared + non-expert (1.354 GB) caricati SUBITO in
streaming sequenziale (banda misurata 3.5 GB/s ⇒ ~0.4 s); esperti on-demand dal
prefill (path ensure esistente + prefetch fase 4); nessun preload del parco.

**Strumento di misura (done-when)**: glmbench con modalità cold-start (flag) —
stesso prompt/protocollo del bench canonico (ctx ~525), JSON con TTFT a freddo,
eviction dichiarata (evictionEvidence), TTFT a caldo della STESSA config nello
stesso report, gap dal budget UX 4 s, predizione del modello di banda a fianco.

**Aritmetica di proiezione (per il ruling item 1, non gate)**: al tetto il
prefill a caldo legge ~18.4 GB da OPFS (bench v2, warmup.prefill.bytesRead) a
banda calda ~11 GB/s (~1.7 s dentro i 12.60 s di TTFT). A freddo la prima
lettura di ogni expert è fredda: ~15.7 GB unici a 1.8-3.5 GB/s ⇒ +3.1-7.3 s di
I/O extra + 0.4 s non-expert ⇒ **TTFT freddo proiettato 16.1-20.3 s = 1.28-1.61×
il caldo 12.60**, SENZA overlap. Con prefetch/overlap del chunk successivo
dietro il compute del chunk corrente (pattern WILLNEED colibri, il prefill è
compute-bound: 12.6 s GPU vs 3-7 s I/O) la parte nascondibile è grande: 1.25×
è raggiungibile SE l'overlap funziona, non per aritmetica secca.

## 7. PROPOSTA per docket item 1 (budget instant-on) — ruling PI richiesto

Il contratto ha [ASSUMED 1.25× il TTFT a caldo]. Proposta, coi numeri sopra:

- **(a) RACCOMANDATA — relativo 1.25×, auto-ancorato**: TTFT_freddo ≤ 1.25 ×
  TTFT_caldo misurato NELLA STESSA config e sessione del report instant-on
  (niente scelta arbitraria fra 16.79 b12 / 12.60 tetto: l'ancora è la config
  della run, entrambe riportate). È ambizioso ma fondato: richiede che
  l'overlap I/O-dietro-compute funzioni (≥ ~60% dell'I/O extra nascosto al
  tetto). Il gap dai 4 s UX resta riportato in ogni report (fase D).
- **(b) alternativa conservativa**: relativo 1.4× (passa con overlap parziale
  ~30%; meno mordente sul meccanismo).
- **(c) status quo del chartered**: 4 s assoluto — aritmeticamente fuori
  (12.6 s di solo prefill a caldo), diventerebbe obiettivo di fase D, non gate
  C3c.

Il floor 13.43, la banda ±5% e i gate di correttezza NON sono toccati da questa
spec (nessuno STOP di soglia; l'unico ruling aperto è item 1, che blocca SOLO
la fase 7 — le fasi 3-6 procedono).

## 8. Cosa NON si costruisce (esclusioni)

- Predictor al confine di token (falsificato 3×; riaprirlo = riaprire WP-0).
- Preload dell'intero parco all'avvio (nega l'instant-on).
- Fallback dinamico optimistic→sync a runtime (resta materia futura; qui solo
  la precondizione statica c3b §2).
- Modifiche a kernel, al path Qwen, agli invarianti I1-I5 di c3b.
- Belady come gate (è ceiling di confronto).

## 9. Telemetria, test, misure collaterali

- Contatori nuovi (schema unico, null contagioso): `prefetchIssued`,
  `prefetchHits`, `prefetchWasted`, `pinSlots`, `repinSwaps`,
  `coldReadMs/coldBytes` (instant-on). Report: hit-rate per policy, recall
  in-engine, attribuzione I/O freddo.
- Unit: formula budget (monotonia, clamp, riproduzione punti noti), eusage
  persist/load additivo, AUTOPIN cap 12.5% (assert), repin isteresi,
  modello di banda (fase 6, predizioni vs punti committati).
- ktest: invarianza del forward con prefetch on (tap read-only).
- **Rimisura formale syncLogits** (landmine iv, docket item 2): nel primo
  report di fase 3, col metodo probe di c3b it.5 (atteso ~0.08 ms, il 7.6
  era dell'era C3a) — chiude la nota di apertura.
- Landmine ereditate: P(dirty) sensibile allo stato cache post-prefill (i);
  ordine preload (layer,expert) arbitrario (ii); 60 s fra run GPU, albero
  congelato (v); bench quiescenti con hostState; niente pipe sui runner.

## 10. Ordine e dipendenze (= PHASES 3→7)

Slab ctx-aware (3) → prefetch in-forward + recall (4) → policy vs LRU ai budget
stretti (5) → modello di banda sui punti misurati (6) → instant-on (7, dietro
ruling item 1) → gate + non-regressione (8) → chiusura (9). Nessuna fase tocca
soglie: l'unico PI-gate è item 1.
