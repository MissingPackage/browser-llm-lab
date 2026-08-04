# FASE 4c — Residenza totale: design

Goal engine-fase-c3a, emendamento 4 (ruling PI 2026-08-03, docket item 8).
Prodotto in it.18 da agente Plan su file e dati veri; rivisto e approvato
dall'orchestratore. La matrice usage è stata DERIVATA (non stimata) dalla
traccia C1; tutti i byte sono ricalcolati dall'inventario di shape.ts.

## 0. Il numero che ribalta l'ipotesi di partenza

Spec §3.0-ter e docket item 8 parlavano di «quant più aggressiva sul 5-10%
di expert più freddi». L'ARITMETICA DICE CHE NON BASTA: degradare Q4_0
(4.50 bpw) → Q3_K (3.4375 bpw a sezioni) libera 1 253 376 B = il 23.61%
dello slab, non il 100%. Per coprire un deficit da 135 slab interi servono
~5.4× più expert:

| necessario | expert | % parco | massa selezioni held-out (decode) |
|---|---|---|---|
| @ctx525 | **627** | 21.3% | **12.5%** |
| @ctx4096 | **935** | 31.8% | **21.0%** |

La 4c non è una limatura sulla coda: è una decisione di qualità su un quinto
del parco. Il design è costruito attorno a questo: misurare prima, degradare
dopo, scelta reversibile a costo di disco.

## 1. La matrice usage, derivata (traccia C1 trace-2026-07-31.jsonl.gz)

- 5 754 416 selezioni full / 942 080 decode su 2 944 chiavi (layer, expert).
- Distribuzione QUASI UNIFORME con coda calda: mediana 0.0257% vs media
  uniforme 0.0340%; p99/p50 = 7.8× full, 6.0× decode. Niente Zipf, niente
  ginocchio. 2944/2944 toccati in decode (caveat C1 confermato).
- Massa dei coldest-N (ranking full, misurata su tutta la traccia):
  N=135 → 0.64% decode; 300 → 2.23%; 500 → 5.24%; 900 → 13.06%.
  Il rendimento marginale peggiora con N (ratio vs uniforme: 0.074 → 0.287).
- **Generalizzazione (held-out, ranking su p0-3 misurato su p4-7, decode)**:
  la massa esposta fuori campione è **1.36-1.78× quella in-sample**
  (N=135: 0.64→1.14%; N=935: 13.9→18.9%). Il prompt 05-math-en è l'outlier
  sistematico (2× la mediana). ⇒ ogni eval fatta sul corpus di derivazione
  è ottimista ~1.5×: protocollo anti-leakage obbligatorio (§8.2).
- Il set freddo è SPARSO (a N=500: 43/46 layer, max 23/64 in un layer).
- **blk.1-4 (classe q4_1) FUORI dal pool degradabile**: compaiono solo a N
  alto, la ricetta upstream gli ha dato più bit sul down (sensibilità
  dichiarata), e le classi restano tre. Pool = blk.5-46, 2 688 expert q4_0.
- Ranking derivato ESCLUDENDO p4+p7 (campione ratificato): coincide al
  96.5-98% col full — gratis, e rende il gate del campione held-out.

## 2. Aritmetica esatta

### 2.1 Bilancio VRAM (ricalcolato, riproduce il docket)

Parco expert 15 678 308 352 B (14.6016 GiB); non-expert 1 354 078 720 B
(1.2611 GiB: attention 47×15 143 936, denso 36.7 MB, shexp+router 342.9 MB,
head Q6_K 262 676 480, output_norm 8 KB; token_embd NON in VRAM —
verificato); VRAM usabile 16 371 417 088 B (15.2471 GiB).
KV = ctx × 108 288 B.

| ctx | deficit | = slab q4_0 |
|---|---|---|
| 0 | 660 969 984 B | 124.5 — **anche a ctx 0 il parco non ci sta** |
| 525 | 717 821 184 B (0.6685 GiB) | 135.2 ✔ docket |
| 4096 | 1 104 517 632 B (1.0287 GiB) | 208.1 ✔ docket |

**VRAM_RESERVE = 64 MiB dichiarata** (allocazioni non-slab note ≈12.7 MiB
@4096 — attnPartials 10.04 MiB, staging, Sel/slotTable — + margine Dawn).

### 2.2 Formati: cosa c'è e cosa manca

quant.ts ha SOLO dequant (Q4_0/Q4_1/Q8_0/Q5_K/Q6_K) e repack. MANCANO:
`dequantQ3_K`/`dequantQ2_K` (senza, il gate argmax≡cpuref non è formulabile
sugli expert degradati); **UN QUANTIZZATORE, in qualsiasi formato** (il repo
sa solo leggere; Q3_K richiede make_qx_quants con ricerca iterativa per
sotto-blocco — la voce di costo più nascosta della fase); kernel GEMV Q3_K
(pair gate/up + accum down, varianti arena).
**Caveat R8**: non abbiamo i pesi originali — si ri-quantizza Q4_0→f32→Q3_K:
errore COMPOSTO, peggiore delle tabelle pubbliche, che NON vanno citate come
proxy. Si misura. (GGUF Q3_K esterno = download = must-docket.)

### 2.3 Byte per formato (tensore expert = 3 145 728 pesi)

| formato | B/tensore layout motore | bpw | slab (3 tensori) | risparmio |
|---|---|---|---|---|
| Q4_0 | 1 769 472 | 4.500 | 5 308 416 | — |
| **Q3_K a sezioni** (hmask 393 216 \| qs 786 432 \| scales 147 456 \| d 24 576) | **1 351 680** | **3.4375** | **4 055 040** | **1 253 376 (23.61%)** |
| Q3_K blob paddato 110→112 | 1 376 256 | 3.500 | 4 128 768 | 1 179 648 |
| Q2_K | 1 032 192 | 2.625 | 3 096 576 | 2 211 840 (41.67%) |
| Q4_K | 1 769 472 | 4.500 | — | ZERO vs Q4_0 |

Q3_K a sezioni: il pad sparisce, ogni sezione multipla di 256 e di 16
(requisito ld4). +73 728 B/expert vs blob. SI ADOTTA.

### 2.4 Opzioni quotate

(a) N expert per chiudere deficit+riserva: Q3_K 627 @525 / 935 @4096;
Q2_K 355 / 530; solo-gate+up 998 / 1490 (scartata: più massa esposta a
parità di risparmio); solo-down 1879 / 2805 (scartata).
(b) KV quantizzata: misto (kv_lora Q8_0 + rope f16) = 672 B/token/layer:
38.4 MiB @525 (5.3% del deficit — irrilevante al gate), 299.6 MiB @4096
(porta N da 935 a 696). Blast radius: kernel MLA it.12 validati + scrittura
cache + cpuref + pin gpulimits. SCARTATA per questa fase, NOMINATA E QUOTATA
come leva del ramo ctx4096 e per C3b.
(c) Head Q6_K→Q5_K: −42.54 MiB = 36 expert. Quantizzatore nuovo sul tensore
che amplifica ~10³. SCARTATA (rischio/beneficio pessimo).
(d) Q2_K puro: fallback dichiarato (355 expert, 6.3% massa held-out ma danno
per selezione 3-4×) — la ladder di slice A lo misura comunque.

## 3. Contesto di dimensionamento — RACCOMANDAZIONE

**Artefatto dimensionato su ctx 4096 (P_max=1024 nel file, copre ctx≤5128);
produzione e gate dichiarati a ctx 525 (P=627); perdita misurata a ENTRAMBI
i punti.** Il meccanismo (§6) rende P un parametro di LOAD, non di import:
cambiare contesto non richiede re-import (risposta concreta al ruling
«dimensionare sul contesto che C3b vuole servire» — C3b eredita senza
rifare 17.2 GB).
La conformance NON forza ctx≥6144: la eval di perdita gira con select:"cpu"
e cache LRU parziale (misura i PESI, non la residenza).

**Regola di decisione PRE-DICHIARATA** (per non decidere sotto pressione):
- eval passa a P=935 ⇒ produzione a ctx 4096;
- passa a P=627 ma non a 935 ⇒ 4c chiude a ctx 525, artefatto resta capace
  di 4096, gap a docket con KV misto già quotata (696 fra i due punti);
- non passa nemmeno a P=627 ⇒ si docketa (done-when: «sotto la soglia si
  docketa, non si assorbe»); fallback identificato: Q2_K N=355, già misurato
  dalla ladder.
Nel report: ctx, deficit, VRAM_RESERVE, P, sha del degrade set, massa
esposta in-sample E held-out.

## 4. Opzione raccomandata

**Q3_K a sezioni sui P expert più freddi di blk.5-46** (ranking senza p4+p7).
P=627 @525: occupazione 16 371 414 784 / 16 371 417 088 B — il margine È la
riserva, dichiarato. Perché Q3_K e non Q2_K: a parità di deficit, Q3_K
espone 12.5% held-out con salto 1.06 bpw; Q2_K 6.3% con salto 1.88 bpw e
rischio di coda (output fuori scala su expert routed, senza CPU di guardia
in select:"gpu"). Varianza minore vince.

## 5. Convivenza con le arene (il vincolo strutturale)

Slab a taglia variabile DISTRUGGEREBBERO l'offset aritmetico (base da
tabella = indirezione per GEMV): scartati senza appello. **TERZA
SIZE-CLASS q3k** (SLAB_W=1 013 760 word, SLABS_PER_BUF=529): costanti
compile-time come le altre.

**Il problema vero: la classe non è più funzione del layer** (un layer
mischia expert q4_0 e q3k, e in select:"gpu" la CPU non conosce la
selezione all'encode). VIA RACCOMANDATA — pipeline unificata:
- un dispatch = un expert ⇒ la classe è UNIFORME sul dispatch come già bi;
- classe nei bit alti della entry slotTable (cls [31:29], slot [28:0];
  0xffffffff resta MISS perché cls=7 mai prodotta) — zero byte in più;
- il resolve scrive cls in Sel.flags; il kernel expert lo legge;
- UNA famiglia di pipeline che binda i buffer di TUTTE le classi (ld4 già
  switcha su bi: la lista si allunga) + switch cls scalare in testa con
  duplicazione testuale del corpo (pattern R1 di it.15). Corpi:
  pairGemvSiluFast ×2 (Q4_0, Q3_K), gemvAccumFast ×3 (Q4_0, Q4_1, Q3_K).
- Dispatch/token: INVARIATI (1405).
FALLBACK (se R2 morde): due famiglie mascherate per layer (+368
dispatch/token quasi vuoti). SCARTATA: indirect dispatch a conteggio zero
(triplica l'encode CPU).

**Binding ri-quotati** (finestra 2 147 483 644 B): P=627 ⇒ q4_0 6 + q4_1 1 +
q3k 2 = 9 buffer, nBuf+3=12 ≤ 16 (margine 4); P=935 ⇒ 8 buffer, 11.
La terza classe RIDUCE il max per classe (7→6→5). Negoziare
maxStorageBuffersPerShaderStage a 12. ARENA_BUFFERS_MAX ri-derivata come
TOTALE = 13 (=16−3), visibile al test che scansiona il WGSL.

**expertSlots → slotsExact** a residenza totale (gli slot SONO il parco),
con precondizione Σ park[c]·slab[c] + nonExpert + KV(ctx) + RESERVE ≤ budget
PRIMA di costruire la cache; il messaggio dice di quanti expert si è corti.

## 6. Slab v2, invalidazione, import

File: header 4 KiB | indice 2944×u32 (fmt [31:28], idx [27:0] — stessa
forma dell'incapsulamento slot/cls) | REGIONE BASE byte-identica a v1
(2944 slab, stessi offset) | REGIONE DEGRADATA (P_max=1024 slab Q3_K,
ordinati per freddezza). Totale ≈19.83 GB (OPFS con GGUF: 37.05 su 1.2 TB).
- **Promozione v1→v2 senza rilettere il GGUF**: header+indice, quantizza
  1024 expert LEGGENDOLI dalla regione base (5.4 GB letti, 4.15 scritti).
- **Coesistenza delle due rappresentazioni** ⇒ P è un parametro di load.
- Header: nSlabsQ3K, slabBytesQ3K, **degradeSetSha256** (se il set cambia,
  il file è un altro modello: il legame artefatto↔eval), degradePolicyId
  (versione dell'algoritmo di quantizzazione), regioni esplicite.
  SLAB_LAYOUT_VERSION 1→2; slabFileReason esteso, ogni motivo dichiarato.
- **Degrade set = artefatto pinnato**: tool che legge la traccia, esclude
  blk.1-4 e p4+p7, emette results/engine/moe-degrade-set-<data>.json
  (lista ordinata, corpusHash, ggufSha256, statistiche in-sample/held-out)
  + modulo TS generato con lista e SHA, importato da slabfile. L'import non
  dipende da results/.
- Import (glmsource.ensureSlabs, seconda fase): leggi slab base → dequant
  Q4_0→f32 → quantize Q3_K → scrivi a sezioni; tmp+move atomico.
  **Costo da MISURARE in slice A** (37.7M superblocchi, proiezione 75-190 s;
  mitigazione: pool di worker per-superblocco, non cambia i byte).

## 7. Preload asincrono (caveat §3-ter.1 di it.17)

Ring di staging mappato ESPLICITO: 2 buffer MAP_WRITE|COPY_SRC da
C·slabBytes (C=32 ⇒ 340 MB, distrutti a fine preload); per chunk di C slab
CONSECUTIVI (senza attraversare confini di buffer d'arena): mapAsync →
read OPFS nella mappata (UNA copia) → unmap → copyBufferToBuffer →
submit → ping-pong. Lettura sequenziale: ~15 GB in 5-8 s a 2-3 GB/s (target
misurabile). flushSlotTable UNA volta alla fine. Asserzioni it.17
rafforzate: evictions===0, misses==parco, occupancy==parco per TRE classi.
**createGlmModel resta sincrona**: con select:"gpu" ritorna
residencyReady=false; `await model.preloadResidency(onProgress)`; il
forward rifiuta esplicitamente se !residencyReady (postura selMiss/R6).

## 8. Eval di perdita (il ruling autorizza la spesa, non esonera dal misurarla)

**Distinzione obbligatoria**: argmax≡cpuref-f64 = correttezza dei KERNEL
(cpuref dequantizza gli STESSI Q3_K: passerebbe con pesi rovinati);
top-1 vs golden (oracolo Q4_0 NON degradato) = l'UNICO gate di qualità.
**Sequenza**:
- Passo 0 — PILOTA CPU (slice A): cpuref.ts ha già il forward f64 completo.
  Quantizza i P expert in memoria, esegui con/senza degradazione sul
  campione, ladder su P ∈ {355,530,627,935,1024} × {Q3_K,Q2_K}. DECIDE P e
  formato PRIMA di scrivere kernel/import. Ore di CPU non presidiate.
- Passo 1 — gate campione (~15 min): argmax≡cpuref 256/256 + top-1 sul
  campione (held-out per costruzione, §8.2).
- Passo 2 — full-corpus glmconf (~4.9 h, UNA volta, albero congelato + GPU
  esclusiva — lezione it.16): top-1 ≥ 98.83%.
**Anti-leakage**: ranking sui 6 prompt p0-3,p5,p6 (esclusi p4+p7). Il gate
full-corpus resta contrattuale ma il report DICHIARA che è parzialmente
in-sample e affianca il numero held-out del campione.
Routing conformance: NON si rigenera (gate_inp/exp_probs_b intatti).

## 9. Slicing

**4c-A — quantizzatore e riferimento (zero GPU, zero import)**:
quantizeQ3_K + dequantQ3_K (+ Q2_K per la ladder); tool degrade-set +
artefatto committato; pilota CPU (ladder completa); misura del costo di
quantizzazione su un tensore reale. GATE: round-trip vs llama-quantize
--allow-requantize del checkout C1 su ≥1 tensore vero (byte-identico o RMS
dichiarata, R1); dequantQ3_K vs dequantize_row_q3_K; artefatto set con
statistiche; ladder committata; npm test + tsc; tempo/tensore proiettato.
**È la slice che decide P e formato, e non tocca il path caldo.**

**4c-B — terza classe (senza residenza totale, senza file v2)**: SLAB_Q3K
in moe.ts; spazio slot unificato; pipeline unificata + corpi Q3_K;
ExpertCache a 3 classi + slotsExact; arenaNeeds; ARENA_BUFFERS_MAX totale;
negoziazione a 12. GATE: ktest q3k-vs-ref (tolleranza derivata, pattern R2
it.15); ktest classi miste con expert NON degradati BIT-IDENTICI a prima;
round-trip node su ogni slot delle 3 classi; bench cpu-mode: contatori
IDENTICI (46/47/1405) e experts bycat entro 5% iso-clock.

**4c-C — slab v2 e preload asincrono**: versione 2, indice, regione
degradata, promozione v1→v2, ring mappato, residencyReady. GATE: test di
invalidazione per OGNI campo header (incluso degradeSetSha256); offset
indice vs ricalcolo indipendente 2944/2944; campione di slab byte-a-byte;
preload con wall time, staging di picco dichiarata, evictions 0, ring
distrutto.

**4c-D — residenza totale e i gate (= chiusura congiunta fase 4 + 4c)**:
P da ctxMax+RESERVE; select:"gpu" sul modello vero; bench quiescente con
hit 100%, 0 miss/token, routerSyncs 0, submits 1, selMiss 0; ctx nel report
con deficit/P/sha/masse; argmax≡cpuref 256/256; top-1 ≥ 98.83% full-corpus;
gpuBusy+clock ri-misurati e attribuiti; non-regressione (item 14) + Qwen ≥
baseline; suite+tsc.

## 10. Rischi (probabilità, discriminatore, mitigazione)

- **R1 quantizzatore ≠ byte ggml** (media): confronto con llama-quantize
  del checkout C1; RMS ≤1e-6 rel = esiti equivalenti, si pinna il nostro;
  degradePolicyId invalida il file a ogni cambio di algoritmo.
- **R2 switch di classe costa** (media): ktest stesso dato mono-classe vs
  unificata — la differenza È lo switch; fallback = famiglie mascherate.
- **R3 top-1 < 98.83% al P scelto** (MEDIA-ALTA): la ladder lo proietta
  prima delle 4.9 h; regola di decisione pre-dichiarata (§3). Non è deroga:
  è il done-when.
- **R4 il freddo non generalizza** (ALTA — già misurata: 1.4-1.8×):
  anti-leakage §8.2; il report dichiara l'in-sample.
- **R5 import esplode** (media): misurato in slice A su un tensore; pool di
  worker.
- **R6 ring stalla/OOM** (bassa-media): wall vs throughput OPFS; C
  parametrico.
- **R7 riserva sotto-stimata** (bassa, conseguenza alta): precondizione
  prima dell'allocazione; report stampa occupazione; MAI retry silenzioso.
- **R8 doppia quantizzazione** (CERTA, grandezza ignota): la ladder la
  misura; tabelle pubbliche MAI come proxy; dichiarata nel report.
- **R9 tre classi rompono il riparto** (bassa): slotsExact + asserzione in
  precondizione + evictions===0.

## 11. Cosa NON cambia

Gate (13.43/56.58/TTFT/54.5/98.83%/256-256); protocollo bench; router e
routing conformance; kernel MLA it.12 e KV f32; head Q6_K; classe q4_1 e
blk.1-4; i 2061+ expert non degradati (byte identici, stesso kernel);
contatori fase 4 (1405, submits 1/routerSyncs 0 in gpu); regione base del
file slab e slabRange; packExpertSlab e path raw; GGUF su disco; path Qwen;
select cpu/shadow (funzionanti a residenza parziale: ci gira la eval).

## 12. Riepilogo decisionale per il PI (docket item 15)

1. Contesto: artefatto a 4096 (P_max 1024), produzione/gate a 525 (P=627),
   perdita misurata a entrambi. Contesto = parametro di load.
2. Opzione: Q3_K a sezioni su blk.5-46; KV e head scartate con l'aritmetica
   (KV nominata per il ramo 4096/C3b).
3. IL FATTO: non è il 5-10% del parco — è il 21-32%, con massa held-out
   12.5-21%. La eval non è una formalità.
4. Rischi principali: R3 (gate 98.83%, regola pre-dichiarata) e R8 (doppia
   quantizzazione, tabelle pubbliche inutilizzabili).
