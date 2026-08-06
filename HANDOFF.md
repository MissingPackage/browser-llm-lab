# HANDOFF — browser-llm-lab   (updated 2026-08-06, sessione 22 — it.21-25: FASE 4d CHIUSA; prossimo: fase 5 prefill M>1)

## 1. Next decidable

**FATTO (it.21-25): FASE 4d CHIUSA — la base è risanata.** It.21 `gpudevice.ts`
punto unico device (8 siti, test strutturale, ktest 53/53). It.22
`telemetry.ts` core unico + retention/hit-split + hostState nei runner. It.23
Planned+Measured con gate sui due path, verificati su device (148 ≡ 148;
1405 ≡ planned+2). It.24 debiti §7: glmsource sotto test, gateCpuref JSON
(8/8 su device), path non-fuso RIMOSSO, censimento orfani (evidence-bearing
pinnati, niente da rimuovere). It.25 **ri-baseline dichiarata** (docket item
19): Qwen **322.2 ± 0.7** ≥ 321.9; GLM **5.013/5.661** — delta vs 08-03 =
RUMORE (Welch t≈1.7 p≈0.16, 3v3), deviceLimits IDENTICI byte-a-byte (rischio
item 10a escluso). Suite **331+7**, tsc pulito.

**PROSSIMO PEZZO DECIDIBILE: FASE 5 — prefill batched M>1 per MoE**
(sbloccata: ordine em.6 era 4c-A′ → 4d → 5). Spec §5 del design 08-01:
M=16 iniziale, condizione di identità = argmax identico su tutte le
posizioni M=1 vs M>1; owns: prefill path, prefillplan, moe batched, tests/.
Done-when: test di identità verde in `npm test` + bench JSON con prefill
tok/s e TTFT su p6 (oggi 5.66 tok/s, TTFT 81.4 s vs UX 4 s — è la leva del
TTFT). Nota di design: nel prefill gli insiemi di expert DIFFERISCONO per
token — il meccanismo arena+Sel della fase 4 (un dispatch = un expert) va
esteso a M>1 (Sel per token o batching per expert con gather). Il ktest dei
kernel fusi solo-Qwen (debito §7.1) si riduce qui coprendo ogni kernel che
il lavoro prefill tocca.

**FATTO (it.20): WP-0 — la tassa di replay è simulata sulla traccia vera**
(`wp0-replay-sim-2026-08-06.json`, semantica differita al confine di token,
LOOKA = predizioni reali, Belady ceiling, 7 unit test; suite 311+7). Esito
in 3 righe: (1) il decode ottimistico è il meccanismo del regime
**near-total (≥~88% residenza)** — al tetto misurato 2596 slot: P(dirty)
65%, tax 26 ms ⇒ 11.3 tok/s ai kernel di oggi; il gate 13.43 arriva con
1-2 tra {clock recovery (16.2 @35ms), kernel margin (24.4 @20ms), policy
> LRU (Belady: P(dirty) ~dimezzabile), GiB no-session (13.7 @2765)};
(2) in scarsità (50/25%) il replay collassa (100% token sporchi) ⇒ la
Pareto ha DUE segmenti: ottimistico per near-total, sync+overlap C3b per
il telefono; (3) **design semplificato**: repair semplice, NIENTE predictor
GPU (LOOKA al confine è neutro/dannoso — replica del finding WASTE),
niente repair batched. Località cross-token misurata: W=16 79%, W=64 95%.
Nota aperta: ancora C1 lru@2208 0.9633 vs 0.9643 committato (0.1pp non
spiegato, codice+traccia invariati in git).

**PROSSIMO PEZZO DECIDIBILE: FASE 4d a perimetro pieno** (em.6, item 18c)
— helper device unico (limiti+uncapturederror), telemetria a schema unico
(retention! — lezione kimi-k3-in-c), Planned+Measured ovunque, glmsource
test, 256/256 come campo JSON, path non-fuso morto, artefatti orfani,
ri-baseline dichiarata. Poi fase 5 (prefill M>1). Il meccanismo ottimistico
si spec-a DOPO la 4d come apertura C3b, sulla base risanata.

**FATTO (it.19): 4c slice A′ — IL GIB NON ESISTE SU QUESTO DEVICE.** Probe
`scripts/vram-ceiling.mjs` (3 run committate): fabbisogno residenza totale
@ctx525 = 16 362 MiB vs tetto fisico allocazioni = **15 947 MiB** (total
16 376 − memory.reserved 429, OOM Vulkan verificato) ⇒ **gap minimo 415
MiB a host perfetto** — l'opzione c non chiude a nessun regime host.
Misurato: 14.00/14.25 GiB residenti (sessione piena/minima);
oversubscription driver 128-224 MiB, post-OOM 0, instabile (LRU del
driver, non pinnabile); Chrome headless senza adapter WebGPU. **Docket
item 17 (RULING RICHIESTO)**: raccomandata la clausola item 2(a) —
fase 4+4c chiudono con meccanismo costruito e misura, gate 13.43
dichiarato irraggiungibile per hardware (−415 MiB), non per struttura;
su M4 48 GB il meccanismo è pronto e la residenza è gratis.

**FATTO (2026-08-06, chat): item 17 = clausola (a) RATIFICATA + razionale
di sviluppo in direction §2-bis** (scarsità VRAM = condizione di progetto,
residenza parziale veloce = tesi). Proposta "decode ottimistico +
riparazione esatta" accettata nel tradeoff (docket item 18a): 1 submit/
token, miss marcato dal kernel, flag piggyback sul readback logits,
replay dal primo layer sporco (checkpoint hidden 376 KB/token) — la
qualità non si tocca mai, il costo è latenza rara. Studio WASTE +
kimi-k3-in-c consegnato (`docs/engine/study/2026-08-06-waste-kimik3-
streaming.md`): LOOKA validato per la terza volta (81.4% top-6 su K3),
ladder Q3_K confermata dal loro esperimento gemello, "volatile memory is
memory you have given away" = il nostro probe it.19 su un altro stack.

**PROSSIMO PEZZO DECIDIBILE (ordine deciso, item 18b): WP-0 — simulazione
trace-driven della tassa di replay** (traccia C1, LRU+LOOKA vs Belady
ceiling, budget 97%/50%/25%, ramo decode steady-state) → poi FASE 4d a
perimetro pieno (item 18c) → fase 5. Il verdetto WP-0 dimensiona il
design del decode ottimistico prima di scrivere kernel.

**RULING (2026-08-05, sessione 21): lo stallo 4c è SCIOLTO — opzione (c),
"trovare il GiB altrove".** La degradazione è FUORI a qualunque P/formato;
la residenza totale resta l'obiettivo e il deficit (685 MiB @ctx525) si
colma dal bilancio VRAM dell'host: nessuna iGPU (display su eDP della
4090), processi desktop ≈347-763 MiB ⇒ per le run di gate la sessione si
spegne (sessione minima o headless da TTY). **Caveat misurato**: il driver
riserva 429 MiB e il "usabile 15.247 GiB" del design era aritmetico, mai
verificato ⇒ **primo task = 4c-A′, probe del tetto allocabile vero da
Chrome/Dawn nei 3 regimi host**. Regola pre-dichiarata: copre ⇒ residenza
totale Q4_0 puro, eval di perdita decaduta (pesi immutati); gap ⇒ docket
col numero, mai degradazione. Emendamento 5 in GOAL, riga 4c ri-sliceata
(A′ probe / B′ preload+slotsExact+precondizione / C′ chiusura congiunta
4+4c). Slab v2, kernel Q3_K e terza size-class NON si costruiscono;
quantizzatore e ladder restano come strumenti di eval.

**RULING (2026-08-05, sessione 21): FASE 4d — risanamento della base**
(emendamento 6, docket item 16): unificare device/limiti/uncapturederror
(chiude il residuo item 10a con ri-baseline dichiarata), telemetria a
schema unico (ttftMs+hostState in ogni report, anche Qwen), dispatch
Planned+Measured ovunque, glmsource sotto test, 256/256 come campo JSON,
path non-fuso morto e artefatti orfani (i 2 prefill-sim-* load-bearing).
NON incluso: ktest dei kernel fusi solo-Qwen (si riduce in fase 5).
**Ordine corrente: 4c-A′ (probe, breve) → 4d → 5 (prefill M>1)**; la
fase 6 ora è blocked-by-3,4,4b,4c,4d,5.

**FATTO (it.13): famiglia fast sui K-quant.** shexp 14.6→5.5, head 9.6→3.8
ms/token (bycat); decode **5.054** (nuovo massimo, medie 6-rep 4.972→5.024);
dispatch/token 1405; **`gpuBusy` 54.2 ≤ 54.5 — soglia del gate 4b raggiunta**
coi clock contro (863 MHz vs 1746 di it.1: la riduzione è tutta kernel).
Prefill 5.69 statisticamente fermo (Welch t=0.82 su 6v6 repliche; ratifica
della banda di rumore a docket item 14). Doppia review avversaria (Opus con
emulazione f64 rel≤3e-13 + Codex fault-injection 0/64 escape): kernel corretti;
la tolleranza ktest allargata è stata rifatta su derivazione onesta (il modello
FMA+associatività reale coincide col device a 4 cifre: 5.869e-4).

**FATTO (it.14): FASE 4b CHIUSA.** glmconf sul campione ratificato: **argmax ≡
cpuref-f64 256/256**, top-1 golden 99.22% (le 2 divergenze sono le stesse di
C2, stesso token di cpuref). Full-corpus resta il gate di fase 6 (item 11).

**FATTO (it.15-16): FASE 4 SLICE A+B.** Slice A: arena a binding fisso in
produzione (identità bit-a-bit, decode 5.081). Slice B: **il router GPU vive
in ombra e la sua fedeltà è misurata sul corpus vero** — set-match
**99.9991% su 1.44M confronti** (13 flip near-tie sul 4° expert, rate 9e-6 =
il numero R8 che lo slice C deve citare), pesi maxRel 4.43e-7, e la Sel di
produzione riletta dalla VRAM è **0/5.75M difforme** dalla decisione CPU (R5
chiuso con lettura diretta). Decode **5.163** (quinto massimo consecutivo),
contatori invariati (46/47/1405). Artefatti:
`routing-conformance-glm47flash-shadow-2026-08-03.json`,
`bench-glm-4090-b12-shadow-2026-08-03.json`. Il setMatch oracolo NON è
identico al 07-31 (+151 prefill/−19 decode su 1.4M: near-tie da somme
riordinate di it.12/13, non slice B — invarianza cpu-vs-shadow bit-identica)
→ docket item 14b.

**LEZIONE OPERATIVA (pagata 2 volte in it.16)**: run GPU lunghe SOLO ad
albero congelato e GPU esclusiva — una review che esegue ktest o un edit di
src/engine/** con vite HMR attivo uccidono il full-corpus (VRAM/full-reload).

**FATTO (it.17): SLICE C — il meccanismo della fase 4 è COMPLETO.**
`select:"gpu"` misurato nel ktest a residenza totale per costruzione:
**1 submit / 0 sync per token** (observer indipendente su queue.submit),
selMiss 0, preload 64/64 asserito, id expert identici per-k, precondizione
con rifiuto esplicito. Decode **5.166** / prefill **5.747** (entrambi
record). Il gate formale della fase 4 (≤2 sync/token nel bench di
produzione) è misurabile solo a residenza totale ⇒ **fase 4 e 4c chiudono
insieme**. **Item 2 RIPRESENTATO al PI nel docket** coi numeri per decidere
(batching = max 12.47, sotto gate; eliminazione via 4c = sopra per
aritmetica, incognite: clock e costo qualità).

**FATTO (it.18): FASE 4c SLICE A — il pilota dice di NON spendere.**
Quantizzatore Q3_K/Q2_K byte-identico a llama-quantize (primo del repo) +
degrade set pinnato + LADDER DI PERDITA su 10 config. Verdetto (nel dato,
`q3k-loss-ladder-2026-08-04.json`, validato da review con ricomputazione):
**nessun P passa il gate** — danno unidirezionale (5-0 appaiato al meglio),
IC95 escludono lo zero per tutte le config, P(pass)≈2e-9; e il 98.83% è un
PIN di non-regressione (1012/1024 misurato in C2), non una soglia.
~~4c in stallo su decisione PI~~ **RISOLTO 2026-08-05 (v. testa del §1):
opzione (c), niente degradazione, GiB dal bilancio host.**

**PROSSIMO PEZZO DECIDIBILE: 4c-A′ (probe tetto VRAM), poi 4d (risanamento
base), poi FASE 5 — prefill batched M>1** (sbloccata dal ruling item 6; il
gate prefill 56.58 e il TTFT 81 s vs 4 servono comunque). Spec §5: M=16
iniziale, condizione di identità = argmax identico su tutte le posizioni
M=1 vs M>1; owns: prefill path, prefillplan, moe batched, tests/. Nota di
design: nel prefill gli insiemi di expert differiscono per token — il
meccanismo arena+Sel della fase 4 (un dispatch = un expert) va esteso a
M>1 (Sel per token o batching per expert con gather). Il done-when: test di
identità M=1 vs M>1 verde + bench con prefill tok/s e TTFT su p6.

**FATTO (it.12): flash-decoding sulla MLA.** `mlaAttnSplitPartWgsl` (chunk da
16 posizioni, 256 thread/wg, cache in shared riusata da tutte le 20 head) +
`mlaAttnSplitReduceWgsl` (LSE esatto, shared O(1)) al posto del monolitico in
glmmodel. **attn 51.21 → 27.49 ms/token** (replica bycat), decode **4.982**
(≥ 4.967: non-regressione PASS), prefill 5.74, TTFT 88 → 80.4 s, `gpuBusy`
69.1 → **64.4** (gate 4b ≤ 54.5 ancora FAIL). ktest 41/41, doppia review
avversaria (Opus + Codex) convergente: zero difetti numerici. Report:
`bench-glm-4090-b12-mlasplit{,2}-2026-08-03.json`, journal it.12.

**IL DATO NUOVO: i clock.** SM medio **948 MHz** durante la run headline
(campionati, 195 campioni attivi) contro 1746 di it.1, cap 3105. Meno lavoro
GPU ⇒ più bolle ⇒ boost più basso: nella replica bycat tutte le categorie
non-attn sembrano ~raddoppiate (shexp 14.6, head 9.6, experts 8.1 ms/token) —
è inflazione da clock, non regressione. Il −46% dell'attn è un lower bound del
guadagno a iso-clock.

**PROSSIMO PEZZO, nessuna decisione pendente: chiudere il margine kernel della
4b portando shexp (Q5_K/Q6_K) e head (Q6_K) alla struttura fast** — sono le due
categorie più grosse rimaste dopo l'attention e usano `gemvQ5K`/`gemvQ6K` con
`sbyte()`: una load u32 intera PER OGNI BYTE, contro la struttura vec4 +
shared-x + 4 righe/wg della famiglia fast (it.10). Stesso profilo di rischio
basso: si porta una struttura validata, non si inventa. Poi la **fase 4** (il
drain), che è dove vive il gate.

**Il vincolo confermato per la terza volta (it.10, it.12)**: con 46 drain per
token i guadagni GPU non si convertono in tok/s — attn dimezzata, decode fermo
(+0.3%), sync/CPU salito a 121-178 ms/token e clock a 948 MHz. La fase 4 resta
obbligatoria; la 4b da sola non porta al gate 13.43 (proiezione it.10: 9.77
anche batchando tutti i sync).

**Ruling recepiti (2026-08-02/03)**: **item 11** = sostituzione della
conformance accettata, full-corpus al gate di fase 6; **item 8** = si paga la
residenza totale, la leva 2 si progetta senza miss; **item 12** = la 4b parte
subito (portare la famiglia fusa di Qwen). **Emendamento 4 scritto** (2026-08-03):
`GOAL.md` + riga **4c** in PHASES (residenza totale, `blocked-by-4`, fase 6 ora
`blocked-by-3,4,4b,4c,5`) — è l'unica fase con authority delta.

**Nessuna decisione PI pendente.** Aperto solo **item 2** (clausola di
fallback), che per ruling si ripresenta a fine fase 4.

**Contesto di fase 3 (chiusa, it.4-8).** Limiti WebGPU derivati dai consumatori
(`src/engine/gpulimits.ts`: `min(adapter, requisito)`, mai costanti né massimo
dell'adapter) + repack all'import (`src/engine/slabfile.ts`, file OPFS da
15,68 GB con header e invalidazione).
- **pack CPU 42,5 → 0,0 ms/token**, decode **4,640 → 4,912** (+5,9%), stallo
  residenza 56,1 → 34,5.
- **Il budget di spec era sbagliato**: la leva vale −21,6 ms/token, non −41,4.
  Le letture expert passano da "prevalentemente in page cache" (4,08 GB/s) a
  "quasi tutte fuori" (1,29 GB/s) perché ora ci sono **due** file da 33 GB
  contro 14 GB di page cache. Il repack **scambia CPU con I/O**; i 18,4
  ms/token di read residui sono materia della leva 2 (li nasconde il prefetch).
- Conformance full-corpus NON eseguita (~5 h): sostituita da verifica
  byte-identica in VRAM e su disco. Docket item 11.

**Contesto — le due assunzioni del contratto smentite dalla misura.**
1. Le tre leve originali **non raggiungono il gate 13.43** (it.1) ⇒ emendamento 1,
   quarta leva (granularità dispatch) nel perimetro.
2. Il costo dei 46 sync **non è latenza dei submit** (che vale 0.6 ms/token) ma
   il fatto che `mapAsync` è una **barriera**: ogni layer drena la coda (it.3).
   Confermato aritmeticamente: `gpuBusy`/wall 36.4% ≈ utilizzo GPU 34.6%, e
   `gpu_idle` attivo in 34/40 campioni mentre power-cap e thermal stanno a 1/40.
   ⇒ emendamento 2, prefetch LOOKA nel perimetro (sovrappone ~54 ms/token di
   lavoro CPU al lavoro GPU).
Il pattern che risolve è provato da ORT/ggml-webgpu/MLC: binding **fisso**,
expert come **offset aritmetico** nello shader. Ma vale solo a residenza
totale ⇒ item 8.

Dettaglio della fase 1: le tre leve originali **non raggiungono il gate 13.43**.
- Wall decode 215.0 ms/token = `gpuBusy` **78.2** + stallo residenza **53.8** +
  sync/CPU **83.0** (63.6% fuori GPU). Ma i 46 readback veri costano solo
  **7.6 ms/token** (probe indipendente): il resto è latenza di submit e bolle.
- Leve 1+2 al 100% ⇒ **10.18 tok/s**; il gate vuole ≤74.46 ms/token e `gpuBusy`
  da solo ne occupa 78.2.
- **Però** la GPU è sotto-clockata dalle bolle che la leva 2 rimuove (1746 MHz
  medi su 3105, utilizzo 34.6%) ⇒ limite ottimistico **15.63 tok/s, PASS**.
  **Forbice 10.2-15.6 col gate dentro**: la leva 2 ha un payoff di secondo
  ordine mai dimensionato.
- Cross-check: 2.22 GB pesi/token su 576 GB/s ⇒ floor memory-bound 3.85
  ms/token, `gpuBusy` è **20× sopra** (1816 dispatch a 43 µs l'uno). Quarta leva
  disponibile (granularità dispatch), fuori contratto e ultima per direction §2.
- Ruling PI: quarta leva **dentro** C3a (GOAL emendamento 1, fase 4b), gate
  invariato. La fase 4 chiude con una ri-misura obbligatoria di clock e
  `gpuBusy` che dimensiona la 4b.
- **Perimetro C3a (struttura)**: repack all'import, eliminazione/batching dei
  46 sync router, prefill batched M>1. **Gate di chiusura**: decode ≥13.43 /
  prefill ≥56.58 tok/s (floor oracolo CPU C1, ereditato dalla deroga C2).
- **Doppio livello sui numeri** (ruling PI 2026-08-01, docket C3a item 1): il
  floor è gate d'ingresso, NON obiettivo. Ogni bench riporta anche il gap
  dalla soglia UX — **decode 30 tok/s (60 in thinking) e TTFT ≤4 s**. Oggi:
  4.64 tok/s (gap 6.5×) e 88 s di TTFT. L'assenza di quei numeri dal report
  è un FAIL di checklist.
- **Baseline misurata**: 215.5 ms/token = 56.1 stallo residenza + 158.9
  struttura (docket C2 item 8). Soglie confermate dal PI: sync/token ≤2,
  pack <1.0 ms/token nel path caldo.
- **C3b (paging) è CHARTERED e non avviato**: slab ctx-aware, tier.h,
  AUTOPIN, PILOT-real, modello di banda, WP banda fredda browser, instant-on
  come TTFT a freddo. Parte a C3a chiusa, dopo aver fissato i numeri di
  non-regressione che C3a lascia (docket C3b item 1).
Riancorarsi da: `.harness/goals/engine-fase-c3a/{GOAL,docket}.md`, docket C2
item 8 (input C3 completo), direction.md §2 (funzione obiettivo a due
termini) e §7 (fase C splittata).

## 2. State delta (sessione 20 — it.12-13)

- **it.13 (fase 4b)**: `pairGemvSiluQ5KFastWgsl` + `gemvQ6KFastWgsl` (8
  thread/superblocco, word in registri, x in shared paddato); shexp 4→2 step,
  head su fast, `upE` rimosso; modulo `kquantfast.ts` (tolleranze derivate +
  sizing shared); floor test `engine-kquant-f32floor.test.ts` (metà
  strutturale sul WGSL vero + metà numerica con associatività reale); guardia
  shared glmmodel a 4 consumatori. Bench ×2 committati, docket item 14.

- **it.12 (fase 4b)**: flash-decoding MLA — kernel `mlaAttnSplitPart/Reduce`
  nuovi in `kernels/wgsl.ts`, wiring in `glmmodel.ts` (17 step attn/layer,
  buffer `attnPartials`), sizing in `mlasplit.ts` (nuovo), `attnPartials` nei
  candidates di `engineNeeds`. Monolitico conservato (glmforward/ktest).
- **Processo nuovo**: implementazione delegata a subagent Opus, doppia review
  avversaria indipendente (adversarial-reviewer Opus con simulatore
  thread-level + Codex), fix round sui 3 finding strutturali (reduce O(ctxMax)
  in shared, fail-fast tautologico, candidate mancante). Zero difetti numerici.
- Bench: 2 run quiescenti committate (`mlasplit` con bycat, `mlasplit2`
  headline), clock SM campionati (948 MHz medi). Docket item 13 (consumatore
  fantasma gpulimits, finding non-decisione), PHASES riga 4b aggiornata.
- Verifica: ktest 41/41 (6 casi nuovi), npm test 261+2, tsc pulito.

## 2-bis. State delta (sessione 17 — chiusura C2)

- **it.11**: harness `glmbench` nuovo (B2 sul forward di produzione, decode
  greedy reale, telemetria per-token); gate GLM falliti con attribuzione;
  debug non-regressione Qwen (codice scagionato byte-identico, firma
  throttling host). Verifier PASS alla terza passata.
- **Post-it.11**: PI quiesce l'host (kill zen: 5g10h CPU accumulata,
  rendering loop WebRender; brief indagine CPU in
  `~/cpu-investigation/brief.md`, fuori repo). Ri-misure: **Qwen PASS tutti
  i gate (K=8 321.9 — sopra il baseline 287.5, anch'esso contaminato;
  prefill 600.2; K=1 244.0)**; GLM quiescente 4.64/5.22 — gate ancora FAIL,
  attribuzione ripulita (pack 42.5, struttura 158.9).
- **it.12 (fase 7)**: ruling PI — item 6 = DEROGA (floor → C3), item 5 =
  campione 2/8 ratificato, coda item 7 = baseline Qwen quiescente adottato
  senza clausola termica formale ("nessuno fa i benchmark con altri processi
  aperti"). Chiusura: docket item 8 (INPUT C3), GOAL emendamento 3,
  direction §7 + ledger riga paging aggiornati, checklist DONE WHEN 7/7
  (punto 5 in deroga), digest.
- Stato test: suite 220 passed + 2 skipped, `tsc --noEmit` pulito, ktest
  30/30 (da it.10). Tutto su main, push a chiusura verificata.

## 3. Cosa c'è nel motore adesso (eredità C2)

- GLM-4.7-Flash end-to-end: reader deepseek2 (quant mista validata hard),
  MLA absorbed (cache 576/token/layer), MoE router bit-fedele a
  build_moe_ffn, residenza minima OPFS→VRAM (ExpertCache LRU due
  size-class, import SHA-streaming 17.2 GB), output head Q6_K, harness
  glmroute/glmconf/glmbench + runner.
- Correttezza: argmax ≡ cpuref-f64 100% (256/256, campione ratificato);
  top-1 vs golden 98.83% full-corpus; identità naive↔absorbed in suite.
- Qwen intatto: conformance bit-identica, bench sopra il baseline nuovo.

## 4. Open threads

- **Goal C3a**: fasi 1-2-3 DONE (it.1-8); fase 4 strato 1 avviato (it.9, router
  su GPU); fase 4b in corso (it.10 catena expert, it.11 attribuzione). Fasi
  4c (residenza totale, emendamento 4) e 5/6/7 a valle.
- **Stato it.9-11 (sessione 19)**: `routerTopKWgsl` (fedeltà f32 misurata: regge
  a 1e-6, primo flip a 1e-7); famiglia fusa sulla catena expert
  (`pairGemvSiluFastWgsl`, `gemvAccumFastWgsl`) ⇒ dispatch/token 1818 → **1450**,
  `gpuBusy` 75.9 → **69.1**, decode 4.912 → **4.967**; tagging dei pass per
  categoria (`setTelemetry(on, gpu, byCat)`). ktest **35/35**, `npm test` 252+2,
  `tsc` pulito, tutto su origin/main.
- **Quello che NON è stato portato** (e ora si sa quanto vale): shexp Q5_K/Q6_K
  10.4%, head 6.7%. `gemvQ5K`/`gemvQ6K` usano `sbyte()`, che fa una load u32
  intera **per ogni byte** — margine grosso, ma dopo l'attention.
- **Analisi complessiva del motore**: `docs/engine/state-2026-08-02.md` (it.8).
  Il numero che riorganizza la fase 4b: il path GLM usa **5.3× meno banda
  utile** del path Qwen sullo stesso device (5.1% vs 27.0% del picco), perché
  le 18 fusioni WGSL esistenti sono tutte solo-Qwen. Portarle, non inventarle.
- **[VERIFY] sciolto** (it.3): `scripts/webgpu-limits.mjs` +
  `results/engine/webgpu-limits-4090laptop-2026-08-02.json`. Il device prendeva
  i default di spec su 3 limiti — ora negoziati (`src/engine/gpulimits.ts`).
  `maxStorageBufferBindingSize` 2 GiB−4 è un tetto duro NVIDIA (workaround Dawn
  per un bug su `OpArrayLength`): lì il cap del codice era corretto.
- **`subgroup-matrix` è disponibile** su questo adapter ⇒ direction §8 rischio 1
  corretto; nuove righe in ideas-ledger §B. Vale per il ceiling, NON per i
  confronti pubblici (la vediamo solo con `--enable-unsafe-webgpu`).
- **Record/replay del grafo comandi precluso** finché esiste un readback per
  layer (graph capture richiede nessun kernel su CPU) — alza la posta di item 8.
- **Goal C3b**: chartered, parte a C3a chiusa.
- **TTFT misurato per la prima volta**: 88.06 s a ctx 461 — 22× il budget UX di
  4 s. Il prefill sequenziale (nessun percorso M>1) è la causa diretta.
- **Verifier di protocollo non lanciato** in it.1: la policy di sessione vieta
  subagent non richiesti dal PI. Se vuoi il gate indipendente a ogni ciclo,
  serve un'autorizzazione esplicita.
- **Indagine CPU host**: sessione separata col brief in
  `~/cpu-investigation/`; esito rilevante per l'igiene dei bench futuri
  (la norma operativa: bench a macchina scarica).
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Duplicato results/opfs-bench-*.json in due posizioni (segnalato, non toccato).
- Nota minore: `scripts/engine-bench.mjs` e `conformance-engine.mjs` hanno
  BASE_URL default :5173, convenzione repo è :5199 (si sovrascrive da env).

## 5. Landmines

- **Bench su questa macchina: farli a macchina SCARICA** (norma PI
  2026-08-01). Con carico utente pesante la GPU resta thermally-capped
  (osservato: 1425-1620 MHz su 3105, −8% sui path a carico sostenuto,
  path latency-bound intatti — quella firma è il discriminatore).
  Diagnostica rapida: `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,
  power.draw,utilization.gpu`.
- Oracolo CPU quantizza le attivazioni q8: MAI gate diretti engine-vs-oracolo
  su selezioni near-tie; il confronto giusto è vs cpuref-f64; golden ≥97%
  assorbe il drift. maxAbsDeltaLogit è metrica di SCALA: mai gate.
- VRAM 16.4 GB: slab 12 GiB + head OK a ctx corto (KV 54 KB/token); ctx 6k
  ⇒ KV 361 MB ⇒ OOM. Budget slab ctx-aware in C3.
- `/tmp` è tmpfs 16 GB: profilo Chrome con OPFS 17 GB in
  `~/.cache/blab-glmroute-profile` (E2E_PROFILE); import skip su size-match.
- mlaAttnDecode MONOLITICO (glmforward/glmroute, ktest): ctx>4k richiede
  maxComputeWorkgroupStorageSize negoziato — il termine 4·ctxMax+256 resta in
  gpulimits per quei consumatori (docket C3a item 13). glmmodel (split, it.12)
  non ha più il vincolo: il suo tetto di contesto è la VRAM della KV.
- Runner in background: NIENTE pipe su tail/grep (maschera exit code);
  output diretto su file.
- Mini-modello sintetico ktest: NON estendere in profondità (overflow f32).
- Storiche: error scope su ogni submit; vite :5199; f32-first Chrome/Linux;
  `pos === kvLen` hard; var WGSL nei loop da azzerare; mai mapAsync su
  buffer di submit non emesso.

## 6. Docket (user decisions pending)

- **Nessun item C2 aperto** (1-8 tutti risolti o informativi; item 8 =
  input C3, non richiede decisione).
- **C3a item 1 RISOLTO** (2026-08-01): split C3→C3a/C3b + formulazione del
  gate prefill (velocità misurata sull'UX, TTFT 4 s) + soglie confermate.
- **C3a RISOLTI 2026-08-02**: item 4 e 5 (ruling 08-01), item 6 (spec),
  item 7 (prefetch+limiti), item 10 (superato dai limiti derivati), **item 11**
  (sostituzione della conformance) e **item 8** (residenza totale: si paga).
- **C3a APERTI**: **emendamento PHASES per la fase 4c** (requant, v. §1) — il
  solo che serva prima della fase 6; **item 2** (clausola di fallback), che il
  PI ha rimandato a fine fase 4 e che la fase 4 ha in done-when l'obbligo di
  ripresentare.
- C3b: nessuna decisione pendente finché C3a non chiude.
- fase-1b-matrice (11 item) e fase-2-deep-dive (5 item): stale, da triage
  weekly-maintenance.
