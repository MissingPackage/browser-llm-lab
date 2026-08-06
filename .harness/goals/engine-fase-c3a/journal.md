# Journal — engine-fase-c3a

## it.1 (2026-08-01) — fase 1: strumentazione TTFT + attribuzione sync vs dispatch

**Passo scelto**: fase 1 di PHASES (l'unica `ready`; le 3-6 sono gated dal ruling
di spec). Obiettivo dichiarato: misurare lo split dei 158.9 ms/token di
"struttura" — l'unico numero che manca per SCEGLIERE in spec il meccanismo dei
sync, invece di sceglierlo alla cieca.

### Cosa è stato costruito

- **`glmmodel.ts` — telemetria di attribuzione, opt-in a due livelli.**
  Liv.1 (CPU): `encodeCpuMs`, `ensureMs`, `routerWaitMs`, `tailWaitMs`, più
  **sync e submit contati davvero** (fino a qui `syncsPerToken: 46+1` era una
  costante dichiarata nel report, non una misura — ora è confermata dalla
  misura: 46.0 sync e 47.0 submit per token).
  Liv.2 (GPU): `timestampWrites` su ogni compute pass, un `resolveQuerySet` per
  token dentro il submit finale, `mapAsync` **dopo** il submit — è il
  root-cause del known-issue di fase A (tsq-diag-2026-07-29), rispettato.
  Zero-overhead da spenta: senza `telemetry` non si chiama `performance.now()`.
- **Deroga consapevole al pattern Qwen**: là le repliche liv.2 girano su un
  *secondo engine dedicato*; qui un secondo modello GLM raddoppierebbe la VRAM
  e non ci sta a slab 12 GiB. Quindi la CAPACITÀ è allocata al build e la
  REGISTRAZIONE si accende a runtime (`setTelemetry`): le repliche headline
  restano a overhead nullo, quelle di attribuzione sono separate e dichiarate,
  il loro wall non entra nei gate.
- **`glmbench.worker.ts`**: TTFT misurato, sezione `objective` coi tre gap
  (30 / 60 thinking / 4 s), attribuzione ortogonale
  `wall = gpuBusy + stallo + sync/CPU`, proiezione per K, **probe del floor di
  sync** (mapAsync round-trip a GPU scarica) come cross-check indipendente, e
  un **check di identità** dei segmenti.
- **`page.ts` / `glm-bench-run.mjs`**: `?attrib=N` / `--attrib N`; il runner
  stampa gap e attribuzione e urla se `objective` manca (FAIL di checklist).

### Difetto trovato e corretto DENTRO l'iterazione

Il check di identità ha fatto il suo lavoro alla prima run: `encodeCpuMs`
contava due volte `ensureMs` (il segmento di encode non veniva avanzato oltre
il blocco `ensure`) — somma dei segmenti 260.9 contro wall 209.4. Corretto
(`tSeg += durata_ensure`) e **bench ri-eseguito**, così l'artefatto committato
corrisponde a HEAD. Dopo il fix: somma 214.34 vs wall 215.05, **non attribuito
0.70 ms = 0.3%**. `encodeCpu` vero è **3.5 ms/token**, non 55.6.

### Misure (run quiescente, slab 12 GiB, p6 461 token, nGen 64, 3 repliche + 1 di attribuzione)

Artefatto: `results/engine/bench-glm-4090-b12-attrib-2026-08-01.json`.
Stato host campionato durante la run (`/tmp/glmbench-c3a-it1b-gpu.csv`, 65
campioni a 5 s): clock SM medio **1746 MHz**, max 2250, cap 3105; utilizzo GPU
medio **34.6%**; 61-66 °C.

Headline (telemetria SPENTA), invariato rispetto al baseline C2:
- decode **4.678 tok/s** (C2: 4.640) — nessuna regressione dalla telemetria
- prefill **5.235 tok/s** (C2: 5.221)
- **TTFT 88.06 s** vs budget UX 4 s ⇒ **gap 22.0×**
- gap decode: **6.41× dai 30 tok/s**, 12.82× dai 60 (thinking)

Attribuzione del decode (finestra strumentata, wall 215.0 ms/token):

| voce | ms/token | quota |
|---|---|---|
| `gpuBusy` (somma durate dei 94 pass) | **78.2** | 36.4% |
| stallo residenza (pack 41.4 + read 4.8 + upload 7.8) | **53.8** | 25.0% |
| sync/CPU (residuo) | **83.0** | 38.6% |
| **fuori GPU** | 136.9 | **63.6%** |

Segmenti disgiunti (identità chiusa): encode CPU 3.5 · ensure 54.1 ·
**routerWait 148.8** · tailWait 8.0. Overflow del ring timestamp: **0**.

**Probe del floor di sync**: mapAsync round-trip a GPU scarica **0.16 ms** (p50)
⇒ 46 × 0.16 = **7.6 ms/token** irriducibili. Cioè: degli 83 ms di sync/CPU,
**meno del 10% è il readback vero**; il resto è latenza di submit e bolle.

### Il risultato che cambia la fase 2

Proiezione con **le leve 1 e 2 del contratto al 100% di efficacia** (repack che
azzera il pack CPU, sync ridotti al loro floor misurato):

- **98.2 ms/token = 10.18 tok/s** — il gate 13.43 vuole ≤ **74.46 ms/token**.
- `gpuBusy` **da solo (78.2) eccede il budget del gate**.

⇒ **Le tre leve del contratto non bastano per costruzione.** Non è una stima
congetturale: è il wall misurato meno i due termini che le leve rimuovono.

**Ma l'osservazione discriminante ribalta a metà la conclusione.** Ho campionato
i clock proprio per questo: la GPU gira a **1746 MHz medi su 3105 di cap**, con
utilizzo 34.6% — è sotto-clockata *dalle bolle stesse* che la leva 2 va a
togliere. Se `gpuBusy` scalasse col clock SM (limite ottimistico: i GEMV sono
in buona parte memory-bound e il clock memoria è già al massimo), diventerebbe
44.0 ms ⇒ **64.0 ms/token = 15.63 tok/s, gate PASS**.

Il vero valore sta nella forbice: **10.2 – 15.6 tok/s**, col gate 13.43 dentro.
La leva 2 ha un **payoff di secondo ordine** (far salire i clock) potenzialmente
grande quanto quello di primo ordine, e nessuno dei due era dimensionato.

**Terzo cross-check, indipendente**: i pesi letti per token sono **2.22 GB**
(calcolo dai tensori: 47 layer di attention + 46 × (4 expert + shared) + head
Q6_K); la banda reale del device è 576 GB/s (9001 MHz × 2 × 256 bit, letta dal
device) ⇒ floor memory-bound **3.85 ms/token**. `gpuBusy` misurato è **20×
sopra**. Quindi il tempo GPU non è un muro fisico: a 1816 dispatch/token siamo a
43 µs per dispatch su GEMV che dovrebbero costarne ~2. C'è una **quarta leva**
(granularità/fusione dei dispatch) con margine enorme — ma è kernel
engineering, che direction §2 mette deliberatamente per ultima.

### Done-when della fase 1: soddisfatta, con due deviazioni di forma

1. ✅ TTFT e i tre gap nel report — `objective.gapDecode30/gapDecode60/gapTtft4s`,
   `ttftMs`.
2. ✅ Attribuzione sync vs dispatch col metodo dichiarato nel payload
   (`attribution2` + `caveats` + `identity`), **ma dentro lo stesso JSON** e non
   in un secondo file come diceva la riga PHASES. Sostanza consegnata, forma no.
3. ✅ `npm test` 220 passed + 2 skipped (= baseline), `npx tsc --noEmit` pulito.
4. ⚠️ La riga PHASES chiedeva "run glmbench **exit 0**": impossibile in fase 1,
   perché exit 0 significa gate PASS e i gate passano solo in fase 6. La run è
   uscita **4** = report scritto + gate FAIL, che è l'esito corretto qui.
   Correzione della riga a docket item 4 (non presa nel loop).

**Il controfattuale a selezione registrata NON è stato costruito**: la
strumentazione con timestamp-query ha dato lo split cercato in modo diretto e
meno invasivo (nessuna modifica alla semantica del forward), quindi
l'esperimento più pesante non serviva. Scelta di metodo, dichiarata.

### Verifica

Fatta in proprio e meccanica: `npx tsc --noEmit` pulito; `npm test` 220 passed
+ 2 skipped (identico al baseline C2); identità della strumentazione chiusa allo
0.3%; headline non regredito (4.678 ≥ 4.640 di C2); overflow del ring 0;
`timestampQuery.used = true`. **Verifier indipendente NON lanciato**: la policy
di sessione vieta di spawnare subagent senza richiesta esplicita del PI — il
gate del protocollo resta formalmente scoperto per questa iterazione ed è
segnalato qui, non aggirato in silenzio.

### Prossimo passo

Fase 2 (spec) è **bloccata da un ruling**: la scelta del meccanismo dei sync ora
dipende da quale ramo della forbice si assume, e la sufficienza delle leve del
contratto è smentita. Docket item 4.

---

## it.2 (2026-08-01) — fase 2: spec C3a

**Ruling PI recepiti prima di partire** (docket item 4b): quarta leva
(granularità dei dispatch) **dentro** il perimetro C3a, gate 13.43 invariato;
clausola di fallback rimandata a fine fase 4; riga PHASES della fase 1 corretta.
GOAL emendamento 1 scritto, PHASES guadagna la fase 4b e la fase 4 chiude con
una ri-misura obbligatoria di clock e `gpuBusy`.

Nota sul recepimento della (b): ho conservato dentro l'opzione scelta il pezzo
utile della (a) che il PI non ha preso — la ri-misura subito dopo la fase 4.
Serve a dimensionare la 4b con un fatto invece che con l'assunzione ottimistica
sui clock, e non costa un ciclo di implementazione.

**Deliverable**: `docs/superpowers/specs/2026-08-01-engine-fase-c3a-design.md`,
8 sezioni, copre i 6 punti del DONE WHEN §1. Ruling richiesto a docket item 6.

### Cosa la spec decide, e sui quali numeri

- **Budget per leva** (somma 74.4 ms/token, cioè il gate con margine zero — il
  margine deve venire dai clock): repack −41.4, sync −75.4, dispatch −23.7,
  residenza residua 12.4.
- **Leva 1 (repack)**: secondo file OPFS `*.slabs.bin` da **15.68 GB** (calcolo
  esatto dalle due size-class), GGUF mantenuto perché conformance e routing lo
  leggono ⇒ 32.89 GB in OPFS. **Spazio verificato prima di speccarlo**: 1.2 TB
  liberi. Header con magic + versione di layout + SHA del sorgente,
  rigenerazione su mismatch, temporaneo + rename.
- **Leva 2 (sync)**: il criterio di scelta discende dalla misura, non dal
  contratto. Il contratto diceva "eliminare i 46 readback"; la misura dice che i
  readback valgono **7.6 degli 83 ms** — il resto è **frammentazione dei
  submit**. Quindi il criterio è **minimizzare i submit, non i readback**: una
  soluzione che toglie i readback ma lascia 47 submit non incassa quasi nulla.
- **Il pipelining è SCARTATO esplicitamente**: era fra le tre vie note del
  contratto, ma nel decode il token t+1 dipende da t e il layer l+1 da l.
  Resta valido per il prefill (leva 3) e per il regime speculativo (fase D).
  Chiuderlo in spec evita che riappaia come opzione fantasma.
- **Vincolo WebGPU marcato [VERIFY]**: `maxStorageBufferBindingSize` e
  `maxBufferSize` reali dell'adapter **non sono mai stati letti** (il codice
  negozia `min(limite, 2 GiB)` senza guardare il limite). WGSL non indicizza fra
  binding diversi e i bind group non si costruiscono da GPU: quindi la via
  raccomandata (A) è **condizionata a un probe da ~20 righe**, primo task della
  fase 4. Sceglierla senza sarebbe tirare a indovinare.
- **Leva 4 (dispatch)**: prima **spezzare `gpuBusy` per categoria** riusando i
  timestamp già raccolti (cambia l'aggregazione, non la strumentazione), poi
  fondere. Con tre ipotesi ordinate per sospetto: dispatch minuscoli
  serializzati dalle dipendenze RW (`rmsnorm` gira su **1 workgroup**), le 16
  catene expert per layer che fanno la stessa forma su pesi diversi, la fusione
  gate/up→silu→down.
- **Leva 3 (prefill M>1)**: M=16 iniziale `[ASSUMED]`, condizione di identità =
  argmax identico su tutte le posizioni M=1 vs M>1. **Conto che vale la pena
  aver fatto**: con M=16 il TTFT scenderebbe verso ~5-6 s, **ancora sopra i
  4 s** — il prefill ha bisogno anche delle leve 1/2/4, non solo del batching.

### Verifica

La fase 2 produce un documento, non codice: `npm test` e `tsc` non sono
pertinenti e non sono stati rieseguiti (ultimo stato verde a it.1, commit
74cb368). Verificato invece che la spec copra i 6 punti del DONE WHEN §1 e che
ogni numero citato abbia una fonte (report di it.1, CSV dei clock, calcolo delle
size-class, `df`). Verifier indipendente non lanciato: stessa policy di sessione
di it.1.

### Prossimo passo

STOP by design: le fasi 3-6 sono gated dal ruling di spec (docket item 6), come
da convenzione C1/C2.

---

## it.3 (2026-08-02) — assunti-non-misurati sistemati + indagine sul readback

PI autorizza l'uso dei subagent. Tre agenti in parallelo (sweep degli assunti,
ricerca sul readback, ricognizione della residenza) più il probe dei limiti
WebGPU eseguito da me.

### A. Valori assunti presentati come misure — CORRETTI

Lo sweep ha trovato il pattern in tre punti, il più grave dei quali era un
**gate tautologico**: `ktest` verificava `model.dispatchesPerToken === 61`, ma
quel valore È la formula `16·nLayer + 6·nDense + 23·nMoe` — il test confrontava
la formula con se stessa e sarebbe passato anche togliendo un pass WGSL.

Correzioni:
- `glmmodel.ts`: contatore **reale** dei dispatch (`T.dispatches`, incrementato
  in `runSteps` e nelle catene expert), esposto in `telemetry()`. Il valore
  derivato resta ma si chiama ora `dispatchesPerTokenPlanned`.
- **La formula era anche sbagliata**: non conta la testa (rms + lm_head), quindi
  ha riportato 2 dispatch in meno per tutto C2. Reale con `readLogits`: 1818,
  non 1816.
- `ktest`: il gate diventa due asserzioni distinte — il piano vale 61 **e** il
  runtime emette 63 (= piano + testa). È (b) a renderlo un test.
- Report di bench/conf/route: nomi che dichiarano la provenienza
  (`*Planned` / `*Expected` dal piano, `*Measured` dai contatori).

**Il gate nuovo ha trovato un bug alla PRIMA esecuzione** — nella strumentazione
che avevo appena scritto: `T.dispatches` contava sempre, `T.forwards` solo a
telemetria accesa, quindi il rapporto era 378 invece di 63. I due contatori ora
vivono insieme, entrambi incondizionati. ktest **30/30 PASS**, exit 0.

### B. Probe dei limiti WebGPU — [VERIFY] sciolto, tre capacità sul tavolo

`scripts/webgpu-limits.mjs` (nuovo), artefatto
`results/engine/webgpu-limits-4090laptop-2026-08-02.json`. Un limite
dell'adapter è una promessa: il device riceve il **default di spec** se non lo
si chiede in `requiredLimits`. Il motore chiede 3 limiti su tanti, quindi:
- `maxStorageBuffersPerShaderStage`: **8 su 16** disponibili
- `maxComputeInvocationsPerWorkgroup`: **256 su 1024** disponibili
- `maxBufferSize`: clampato a 2 GiB su **4** disponibili
- `maxStorageBufferBindingSize` 2 GiB−4: **tetto duro NVIDIA**, Dawn lo clampa
  per un bug driver su `OpArrayLength` — qui il cap del codice era giusto.

Inoltre l'adapter espone **`chromium-experimental-subgroup-matrix`**,
`subgroups`, `subgroup-size-control` e
`chromium-experimental-timestamp-query-inside-passes`. Direction §8 rischio 1
(«subgroup-matrix assente nei browser») è **stale** su questa macchina — con la
precisazione di perimetro: la vediamo per `--enable-unsafe-webgpu`, quindi vale
per il ceiling del motore, non per un confronto pubblico.

### C. Il meccanismo del readback: la mia attribuzione era SBAGLIATA

Avevo scritto in spec v1 che gli ~75 ms erano "latenza dei submit". Il costo API
di una submit è **~13 µs** (Dawn/Vulkan, misura pubblicata): 47 submit valgono
**0.6 ms/token**, non 75.

Il meccanismo vero: **`mapAsync` è una barriera** — da spec non si completa
finché tutto il lavoro accodato prima non è finito. Ogni layer è un **drain
completo della coda**. La nostra stessa misura lo conferma per via aritmetica:
`gpuBusy`/wall = 36.4% contro utilizzo campionato 34.6%. La GPU è idle
esattamente quando non è dentro un pass. Il probe a 0.16 ms misurava il
round-trip a coda **vuota**; sotto carico ogni round-trip include il drain
(83/46 = 1.8 ms, ~11× il probe).

**Il pattern che risolve esiste ed è provato da tre implementazioni
indipendenti** (ONNX Runtime PR #27998 mergiata 2026-04-10, llama.cpp
`ggml-webgpu`, MLC): pesi di tutti gli expert in un tensore packed, **binding
FISSO**, indici top-k in un buffer GPU, expert = **offset aritmetico** nello
shader. Nessuno usa `dispatchWorkgroupsIndirect`. Le API che sembravano
promettenti non arrivano in soccorso: `binding_array` per i buffer è dichiarato
non implementato in Dawn, il bindless sperimentale copre solo texture/sampler,
i dynamic offset sono valori CPU.

### D. L'ostacolo che nessun pattern di binding risolve

Le tre implementazioni di riferimento assumono **tutti gli expert residenti**.
Noi ne abbiamo 2419 su 2944 (82.2%), e solo la CPU legge da OPFS ⇒ finché c'è
un miss, il readback serve. Conto della residenza totale: servono **15.91 GiB**
(14.60 di parco + 1.26 di non-expert + 0.05 di KV) contro **15.25** usabili
⇒ **deficit 0.67 GiB = 135 expert = 4.6% del parco**. A ctx 4096: 1.03 GiB.

È una decisione sulla funzione obiettivo (capienza vs velocità), non di
implementazione ⇒ docket item 8.

### E. DVFS: ipotesi ora DIMOSTRATA nel meccanismo

L'agente di ricerca ha giustamente marcato come non dimostrata la mia ipotesi
che le bolle abbassassero i clock. L'ho misurata: `nvidia-smi` con le
`clocks_event_reasons` durante un bench (49 campioni,
`results/engine/host-gpu-throttle-c3a-it3-2026-08-02.csv`):

- **`gpu_idle` ATTIVO in 34/40 campioni**
- `sw_power_cap`, `hw_slowdown`, `sw_thermal_slowdown`: 1/40 ciascuno

I clock bassi (1168-1746 MHz su 3105) sono causati da **inattività**, non da
limiti termici o di potenza. Il meccanismo è quindi stabilito: il drain causa
l'idle, l'idle abbassa i clock. **Resta non misurato di quanto `gpuBusy`
scenderebbe a clock pieni** (i GEMV sono in parte memory-bound e il clock
memoria è già al massimo) — la forbice 10.2-15.6 tok/s resta, ma la sua premessa
non è più un'assunzione.

### Ruling PI recepiti in giornata

- **Prefetch LOOKA ammesso nel perimetro C3a** (GOAL emendamento 2a). Ribalta
  la valutazione del docket C2 item 8, che dava al prefetch valore residuo
  piccolo guardando solo lo stallo e non la sovrapposizione CPU/GPU.
- **I tre limiti si negoziano subito**, in fase 3 (emendamento 2b).

Spec §3 riscritta di conseguenza (§3.0 meccanismo, §3.0-bis pattern provato,
§3.0-ter residenza, §3.1 limiti misurati, §3.2-bis via scelta).

### Verifica

`npx tsc --noEmit` pulito; `npm test` 220 passed + 2 skipped; **ktest 30/30
PASS exit 0** col gate dispatch ora significativo. Probe eseguito con artefatto
committato. Verifier indipendente: ora autorizzato dal PI, da usare dalla
prossima iterazione.

---

## it.4 (2026-08-02) — fase 3 slice 1: negoziazione dei limiti WebGPU

Spec approvata (docket item 6, PI "ok andiamo avanti" dopo esposizione delle 5
decisioni con raccomandazione per ciascuna; interpretazione registrata nel
docket per essere corretta se sbagliata). Fasi 3-6 sbloccate.

**Findings di it.3 annotati dove servono** (ruling PI "annota questi 2
findings"): direction §8 rischio 1 corretto (`subgroup-matrix` non è più assente
dai browser, col perimetro `--enable-unsafe-webgpu` dichiarato);
ideas-ledger §B guadagna tre righe — GEMV con subgroup ops, binding fisso +
offset aritmetico (col prior art ORT/ggml/MLC), record/replay precluso dal
readback. Docket item 9 li registra come finding, non decisioni.

**Fatto**: `src/engine/gpulimits.ts` — `negotiateLimits(adapter, caps?)`,
`grantedLimits(device)`, `slabBufferCap(device)`. Applicato ai **4 worker GLM**
(glmbench, glmroute, glmconf, ktest). Prima ogni worker ripeteva
`Math.min(lim.X, costante)` per 3 limiti e prendeva i default di spec su tutti
gli altri; ora si chiede il massimo che l'adapter concede, e i limiti
**concessi** finiscono nel report (una prestazione misurata su limiti diversi
non è confrontabile — lezione B2).

Il cap di ktest a 1 GiB è VOLUTO (mini-modello sintetico) ed è preservato via
`caps`, non perso: senza il parametro sarebbe stata una regressione silenziosa.

**Path Qwen NON toccato di proposito**: `engine.worker.ts` ha il gate di
non-regressione hard contro il baseline quiescente 2026-08-01, e cambiare i
limiti del suo device richiederebbe un ri-bench per restare confrontabili.
Va fatto, ma come slice dichiarata, non di straforo.

**Rischio di regressione principale, escluso aritmeticamente**: alzare
`maxBufferSize` da 2 a 4 GiB poteva cambiare il numero di buffer allocati dalla
residenza (`residency.ts:88` usa `min(maxBindingBytes, maxBufferBytes)`). Non
cambia: il binding size NVIDIA (2 GiB−4) resta il vincolo minore in entrambi i
casi, quindi cap identico e stessa allocazione.
`maxComputeWorkgroupStorageSize` passa da 32768 a 49152 richiesti: è una
rilassazione del fail-fast di `glmmodel.ts` (ctx più lunghi ammessi), non un
vincolo nuovo.

**Verifica**: `npx tsc --noEmit` pulito; `npm test` 220 passed + 2 skipped;
**ktest 30/30 PASS exit 0** — quest'ultimo è il test che esercita davvero il
percorso device con i limiti nuovi. Review avversaria del diff lanciata in
parallelo (subagent, ora autorizzati dal PI).

**Prossimo**: fase 3 slice 2 — il repack all'import vero e proprio.

---

## it.5 (2026-08-02) — review avversaria della slice 1: il fix non faceva nulla

Subagent `adversarial-reviewer` sul diff di it.4. Verdetto duro e in gran parte
corretto; 3 finding sostanziali recepiti, 1 rimandato al PI.

### Critical accolto: la negoziazione non produceva NULLA

`residency.ts:88` dimensionava i buffer slab con
`min(maxBindingBytes, maxBufferBytes)`. Su NVIDIA il binding è clampato da Dawn
a 2 GiB−4 (bug driver su `OpArrayLength`), quindi il `min` **buttava via** i
4 GiB di `maxBufferSize` appena sbloccati: 7 buffer prima, 7 dopo. Il payoff
scritto in spec §3.1 (7→4) non si materializzava.

Ed era anche **semanticamente sbagliato**: `maxStorageBufferBindingSize` limita
la taglia di un `GPUBufferBinding` (il SOTTO-RANGE bindato — qui `qsBytes`,
1.5 MB), `maxBufferSize` limita la taglia del buffer. Il commento del file lo
diceva pure ("difensivo: i bind sono comunque sotto-range da ~1.5-2 MB") senza
trarne la conseguenza.

Corretto: si cappa con `maxBufferBytes` e si **asserta** che il sotto-range più
grande stia nel limite di binding (throw al costruttore, non scoperta al primo
`setBindGroup`). Effetto reale: **809 slab per buffer invece di 404**, 3 buffer
q4_0 invece di 6 a budget 12 GiB.

**La mia verifica di it.4 era giusta ma per il motivo sbagliato**: avevo
"escluso aritmeticamente" il cambio di comportamento della residenza. Era vero —
ed era esattamente il sintomo che il fix non funzionava.

### Important accolti

- **Copertura ZERO** su `gpulimits.ts`: nessun test sarebbe fallito se
  `negotiateLimits` avesse restituito `{}`. Aggiunto `tests/gpulimits.test.ts`
  (8 casi che discriminano implementazioni sbagliate reali: `caps` come minimo e
  non sostituzione, limite assente saltato invece che chiesto a 0, nessuna
  chiave sopra il massimo dell'adapter, coppie vincolate riallineate) +
  1 caso nuovo in `engine-residency.test.ts` sull'invariante del sotto-range.
- **Coppie di limiti desincronizzabili**: cappare `maxComputeInvocationsPerWorkgroup`
  senza cappare `maxComputeWorkgroupSizeX` dà un device che passa
  `requestDevice` e poi fallisce alla creazione della pipeline. Aggiunta la
  tabella `COUPLED` e il riallineamento.
- **La tesi centrale era NON VERIFICATA**: il probe chiedeva 3 limiti e
  riportava i default sugli altri, quindi "chiedendoli si ottengono 16 e 1024"
  era una congettura. Probe corretto (chiede tutto ciò che l'adapter espone) e
  **ri-eseguito: 16 e 1024 si ottengono davvero**, artefatto aggiornato.
- `maxBindingsPerBindGroup` tolto da `WANTED` (nessun consumatore, nessuna
  motivazione — la review ha ragione: il file si dà come regola di dichiarare
  il perché di ogni limite); `deviceLimits` aggiunto anche ai report di
  glmroute/glmconf; `deviceFeatures` accanto ad `adapterFeatures` (le feature
  dell'adapter sono annunciate, non abilitate); `schemaVersion` 2→3.

### Un test esistente è caduto, e andava fatto cadere

`engine-residency.test.ts` forzava la partizione su più buffer via
`maxBindingBytes`. Con la semantica corretta quella non è più la leva: cambiata
in `maxBufferBytes`, **intento del test invariato** (verifica ancora la
partizione e l'indirizzamento degli slot su più buffer). Non è "aggiustare il
test per far passare il codice": la leva vecchia testava un vincolo che la spec
WebGPU non impone.

### Rimandato al PI: docket item 10

La review ha trovato due cose che non posso chiudere da solo:
(a) i limiti sono negoziati sui 4 worker GLM ma non su `gpuforward.ts:99` (il
motore Qwen) né su 4 siti minori ⇒ **ktest diventa più permissivo del motore
che deve validare**; (b) il baseline del gate GLM è stato prodotto con limiti
diversi e non contiene `deviceLimits`, quindi il prossimo confronto non è
falsificabile. Entrambe costano un ri-bench e toccano il gate di
non-regressione hard: decisione PI.

### Verifica

`npx tsc --noEmit` pulito; `npm test` **229 passed** + 2 skipped (da 220: +8
gpulimits, +1 invariante residenza); **ktest 30/30 PASS exit 0**; probe
ri-eseguito con artefatto committato.

---

## it.6 (2026-08-02) — limiti DERIVATI dai consumatori (ruling PI)

**Ruling PI**: "non vorrei che facessimo lo stesso errore della prima volta dove
abbiamo messo dei limiti che poi abbiamo dovuto rivedere. Forse meglio capire
caso per caso dove ci possiamo spingere?"

**Root cause dei due errori** (stesso difetto, direzioni opposte):
1. i cap originali (`min(lim.X, 2 GiB)`, `32768`) erano costanti difensive
   scelte **senza consumatore dichiarato** ⇒ nessuno poteva più distinguere un
   requisito da una supposizione, e sono stati rivisti a mano;
2. la mia prima versione li ha sostituiti col **massimo dell'adapter** ⇒
   identico difetto: si chiedevano 1024 invocazioni per workgroup mentre il
   kernel più largo del repo ne usa 256. La spec WebGPU §3.6.2 avverte
   esplicitamente che chiedere più del necessario *può* costare prestazioni.

**Regola adottata**: `min(adapter, requisito_derivato)`. L'adapter è il TETTO,
non il target. Ogni requisito porta il suo consumatore, e un test ri-deriva le
costanti **scansionando il WGSL vero**.

### Inventario (subagent, numeri esatti col consumatore)

| limite | default spec | requisito REALE | consumatore |
|---|---|---|---|
| `maxComputeInvocationsPerWorkgroup` | 256 | **256** (margine 0) | rmsnorm, argmax1/2 |
| `maxComputeWorkgroupSizeX` | 256 | **256** | idem |
| `maxStorageBuffersPerShaderStage` | 8 | **7** | rmsPairGemvSiluFast (Qwen) |
| `maxStorageBufferBindingSize` | 128 MiB | **262 676 480** (250,5 MiB) | `output.weight` Q6_K GLM |
| `maxComputeWorkgroupStorageSize` | 16 KiB | **max(30 848, 4·ctxMax+256)** | Qwen pairSilu chunked / mlaAttnDecode |
| `maxBufferSize` | 256 MiB | 262 676 480 hard + packing soft | testa GLM / ExpertCache |

### Due cose che nessuno aveva scritto

1. **Il cap `32768` limitava il contesto a 8128 token.** `mlaAttnDecode` usa
   `4·ctxMax + 256` B di workgroup storage: a ctx 8129 sfora. La landmine in
   HANDOFF diceva «ctx>4k richiede 32 KB negoziati» senza il pezzo che conta,
   cioè fin dove si arriva. Ora è una formula, quindi si autodocumenta.
2. **Il consumatore massimo di workgroup storage del path Qwen non è quello che
   dice il commento.** `gpuforward.ts` cita 19,7 KB (il down-proj), ma il vero
   massimo è `rmsPairGemmSiluChunkFast` a **30 848 B** — i 32 768 richiesti a
   mano lasciavano 1 920 B di margine senza che nessuno lo sapesse.

Inoltre: la testa Q6_K (262 676 480 B) sta **appena sotto** il default di spec
su `maxBufferSize` (268 435 456), margine 2,1%. Un vocabolario più grande
avrebbe rotto il motore su qualunque device ai default.

### Cosa è stato costruito

`gpulimits.ts` riscritto: `engineNeeds(opts)` produce una lista di `LimitNeed`
(limite, valore, **consumatore**, hard/soft); `limitsFor` chiede
`min(adapter, requisito)` e lancia `UnmetLimitError` **col consumatore nel
messaggio** se un requisito hard non è servibile. I requisiti soft (packing
della ExpertCache) si troncano al disponibile invece di far fallire.

I 4 worker GLM creano ora il device **dopo** aver saputo `ctxMax` e budget, che
è la precondizione per derivare invece di indovinare. ktest dichiara il proprio
consumatore extra (binda i pesi densi VERI di blk.0, 10 485 760 B) invece del
cap inventato a 1 GiB.

### Il test che tiene insieme limite e consumatore

`tests/gpulimits.test.ts` (14 casi) **scansiona `src/engine/kernels/wgsl.ts`**
ed estrae i `@workgroup_size` e i binding `var<storage>` per kernel: se qualcuno
aggiunge un kernel `workgroup_size(512)` senza aggiornare la derivazione, il
test cade. È l'unica cosa che lega i due — finora vivevano in file diversi
senza niente in mezzo, ed è esattamente il motivo per cui l'errore è successo
due volte.

### Verifica

`npx tsc --noEmit` pulito; `npm test` **235 passed** + 2 skipped (da 229);
**ktest 30/30 PASS exit 0** col device ai limiti derivati.

**Nota su docket item 10**: l'opzione (a) che avevo raccomandato (negoziare
ovunque al massimo + ri-baseline) è **superata da questo ruling**. Con i limiti
derivati, il path Qwen (`gpuforward.ts:101-107`) chiede già oggi 256 MiB e
32 KiB, cioè valori compatibili con i requisiti calcolati: l'inversione di
permissività ktest-vs-motore si chiude allineando `gpuforward` alla stessa
derivazione, senza alzare nulla. Item 10 da riscrivere di conseguenza.

---

## it.7 (2026-08-02) — fase 3 slice 2: repack all'import

**Costruito**: `slabfile.ts` (formato: header 4 KiB con magic + versione di
layout + SHA del GGUF sorgente, poi 2.944 slab contigui; le due size-class in
ordine — 256 q4_1 poi 2.688 q4_0 — così l'offset è aritmetica chiusa senza
tabella); generazione in `GlmOpfsSource.open` su file temporaneo + rename
(un'interruzione non lascia mai un file valido a metà); percorso rapido in
`ExpertCache.ensure` che salta `packExpertSlab` quando la sorgente offre lo
slab già impacchettato, con fallback sul percorso raw per i mock dei test.

Il test che conta non è quello sugli offset: carica gli stessi expert per
**entrambi** i percorsi e confronta i byte caricati in VRAM uno per uno. Un
offset o un ordine di segmenti sbagliato non passerebbe.

### Risultato: done-when soddisfatto, guadagno metà del previsto

| ms/token | pre-repack | post-repack |
|---|---|---|
| pack CPU | 42,5 | **0,0** |
| read OPFS | 5,8 | **18,4** |
| upload | 8,1 | 16,2 |
| **stallo** | **56,1** | **34,5** |
| decode tok/s | 4,640 | **4,912** (+5,9%) |

Done-when della fase 3 (`pack CPU < 1.0 ms/token`): **soddisfatto**, 0,0.
Ma il guadagno netto è 21,6 ms/token, non i 41,4 proiettati in spec §2: read e
upload sono cresciuti di 20,7 e si sono mangiati metà del beneficio.

### Tre ipotesi, due smentite, causa trovata

Il salto di `read` da 5,8 a 18,4 ms/token andava spiegato, non accettato.

1. **Page cache fredda al primo giro** — SMENTITA: seconda run 18,4 contro
   19,2. Non è un transitorio.
2. **File slab frammentato** (l'ho scritto senza preallocare) — SMENTITA e nel
   verso opposto: `filefrag` dà **2 extent** per lo slab contro **263** per il
   GGUF. Il file meglio disposto è quello lento.
3. **Throughput del file** — SMENTITA: misurato fuori dal motore con letture
   random da 5,3 MB, entrambi i file danno ~1,9-2,7 GB/s a freddo e ~10 GB/s a
   caldo. Il file slab non ha nulla che non va.

**Causa reale**, dal throughput EFFETTIVO per miss:
- pre-repack **4,08 GB/s** (fra il freddo 2,72 e il caldo 10,6 del GGUF)
  ⇒ le letture expert erano prevalentemente **in page cache**;
- post-repack **1,29 GB/s**, sotto persino il freddo misurato ⇒ **quasi tutte
  fuori cache**.

Prima gli expert si leggevano dallo stesso file che il motore tocca comunque a
ogni build (i tensori non-expert), quindi il GGUF restava caldo. Ora ci sono
**due** file — 33 GB contro **31 GB di RAM totali e 14 di page cache** — e
nessuno dei due resta caldo.

**Il repack non ha eliminato lavoro: ha scambiato 41,4 ms di pack CPU con
12,6 ms di I/O.** Resta un guadagno netto, ma metà di quello scritto in spec.

### Conseguenza per la fase 4 (che rafforza il ruling sul prefetch)

I 18,4 ms/token di lettura sono ora il secondo termine della residenza e sono
**tempo in cui la GPU è ferma**. È esattamente ciò che il prefetch — ammesso in
perimetro dal PI con l'emendamento 2a — va a nascondere dietro il lavoro GPU.
Il repack da solo vale 21,6 ms/token; repack + sovrapposizione valgono i 34,5
dello stallo intero. La spec §2 va aggiornata: il budget della leva 1 non è
−41,4 ma −21,6, e i 12,6 di I/O migrato si recuperano solo con la leva 2.

### Verifica

`npx tsc --noEmit` pulito; `npm test` **251 passed** + 2 skipped (da 235:
+13 formato slab, +3 percorso rapido). Due bench a macchina quiescente. Il
contatore dei dispatch conferma sul modello di produzione i **1818** reali
contro i 1816 del piano (la testa che la formula non contava, trovata in it.3).
Conformance in corso: è l'ultimo pezzo del done-when.

---

## it.8 (2026-08-02) — allineamento Qwen + analisi complessiva del motore

**Allineamento `gpuforward.ts`** (docket item 10, "cosa da fare non da
decidere"): sostituita `const need = 256 * 1024 * 1024` — l'ultima costante
difensiva senza consumatore rimasta nel repo — con la derivazione. I consumatori
veri del path Qwen: `wOut.qs` (lm_head Q8_0 ripacchettata, **136 134 656 B**,
calcolata e coincidente col numero dell'inventario) e `rmsPairGemmSiluChunkFast`
(30 848 B di workgroup storage). `engineNeeds` guadagna `mlaAttention` e
`kvBytesPerLayer` per non attribuire a Qwen consumatori che non ha. La formula
del workgroup storage MLA, che era scritta due volte in notazioni diverse
(`glmmodel.ts` e `gpulimits.ts`), ora vive in un posto solo.

**Verifica del gate di non-regressione Qwen** (il device è cambiato, quindi era
obbligatoria): conformance **identica** — 98.05% vs golden, 100.00% vs
cpuref-f64 (512/512), 147 dispatch/token, gate doppio PASS. Bench **322.8 ±
5.5 tok/s** contro il baseline permanente 321.88 ± 2.60 ⇒ **PASS**, leggermente
sopra. `npm test` 252 passed, ktest 30/30.

**Tre agenti in parallelo** (storia delle metriche, architettura + copertura,
debito aperto) per l'analisi complessiva richiesta dal PI. Prodotto:
`docs/engine/state-2026-08-02.md`.

### Il risultato che riorganizza il resto

Incrociando l'inventario dei kernel con i profili di attribuzione dei due path,
a parità di device:

| | byte/token | `gpuBusy` | banda utile | % del picco | µs/dispatch |
|---|---|---|---|---|---|
| Qwen | 346 MB | 2.22 ms | 155.6 GB/s | **27.0%** | 15.1 |
| GLM | 2 220 MB | 75.86 ms | 29.3 GB/s | **5.1%** | 41.7 |

**Il path GLM usa 5.3× meno banda utile del path Qwen sullo stesso hardware.**
Non è taglia del modello: è qualità dei kernel. Dei 30 generatori WGSL, 18 sono
solo-Qwen e sono la famiglia FUSA; i 7 solo-GLM sono tutti generici.

**Il lavoro di fusione è già fatto, validato e misurato — su un path solo, e non
è mai stato portato sull'altro.** Riformula la quarta leva di C3a: non inventare
ottimizzazioni, portare quelle che esistono. Da 5.1% a 27% del picco sarebbe
`gpuBusy` da 75.9 a ~14 ms/token.

### Altre cose emerse dagli agenti, registrate

- **Nessun worker GLM installa `uncapturederror`**, che sul path Qwen esiste per
  una ragione documentata (griglia >65535 ⇒ "top-1 0.2% muto"). È la classe di
  errore silenzioso che la disciplina del progetto teme di più, non intercettata
  sul path nuovo.
- **Gli 11 kernel fusi di produzione Qwen non hanno test kernel-level**: ktest
  importa 16 generatori su 30, i 14 mancanti sono tutti Qwen e tutti fusi.
  Coperti solo end-to-end dalla conformance, che è statistica — mentre i tre
  bug storici (telemetria liv.2, `var stride`, varianti "2800 tok/s") sono stati
  presi proprio da confronti puntuali.
- **Il 256/256 del gate argmax≡cpuref-f64 GLM non esiste in nessun JSON**: è
  asserito nei digest e ricostruibile solo indirettamente.
- **13 artefatti orfani** in `results/engine/`, di cui `prefill-sim-*` è
  load-bearing (giustifica "submit ogni ~64 token" citato in `prefillplan.ts:7`,
  produttore cancellato).
- **Tre rimandi orfani** registrati in ideas-ledger §I (soglia TTFT
  high-variance, `quality.ts` mai collegato, poll wllama mai meccanizzato):
  erano in docket di goal chiusi, fuori dal registro ufficiale dei rimandi.

### Verifica

`npx tsc --noEmit` pulito; `npm test` **252 passed** + 2 skipped; ktest 30/30
exit 0; conformance Qwen gate doppio PASS; bench Qwen sopra il baseline.

---

## it.9 (2026-08-02) — fase 4 slice 1: il router top-4 vive su GPU

**Ruling recepiti** (PI: "ok item 11 e 8. ok sulla sostituzione"):
- **item 11** — opzione (a): la verifica byte-identica sostituisce la conformance
  full-corpus per la fase 3; la conformance resta gate di chiusura della fase 6.
  La (c) (campione ratificato eseguibile in minuti) NON è stata scelta: resta
  debito registrato, insieme al 256/256 che non esiste come campo in nessun JSON.
- **item 8** — si paga: la leva 2 si progetta **a residenza totale**. Conseguenza
  registrata nel docket, non decisa qui: il requant degli expert freddi tocca
  `quant.ts`/`slabfile.ts`/import, che sono gli `owns` della fase 3 (chiusa) ⇒
  serve un **emendamento a PHASES** (proposta: fase 4c fra la 4 e la 6, con la
  eval di perdita nel done-when). Finché non è confermato, la fase 4 procede su
  ciò che sta nei suoi owns.

**Costruito**: `routerTopKWgsl` (`kernels/wgsl.ts`) — sigmoid, bias nella sola
selezione, scan lineare con `>` stretto su un thread solo (il tie-break *è* la
serializzazione: un top-k parallelo con riduzione cambierebbe l'ordine dei
confronti), somma dei probs senza bias, denominatore clampato, ×1.8. Scrive id e
pesi in due buffer GPU.

**Perché è il primo pezzo e non un pezzo qualsiasi.** Il binding fisso da solo
non toglie il readback: serve che gli id vivano su GPU. È la precondizione sia
dello strato 1 (offset aritmetico) sia della residenza totale appena approvata —
senza miss il readback non "si riduce", sparisce, ma solo se nessuno su CPU deve
più sapere chi sono i top-4.

### La fedeltà è stata misurata, non dichiarata

La CPU calcola in f64, il kernel in f32: **non è una replica bit-identica**, e
questo è il numero che dice quanto conta.

| caso ktest | esito |
|---|---|
| `router-top4-gpu-vs-cpu` (64 estrazioni casuali) | insiemi identici 64/64, ordine identico 64/64, pesi maxRel **1.64e-7** |
| `router-top4-near-tie` (separazione imposta, sweep 1e-3→1e-8) | tiene fino a **eps 1e-6**, primo flip a **1e-7** |

Il primo caso da solo sarebbe stato un gate finto: con 64 expert e punteggi
sparsi la separazione fra 4o e 5o non scende mai sotto **3.43e-5** da sola, cioè
il gate dichiarato a 1e-5 non veniva mai esercitato. Il secondo caso costruisce
il pareggio (sposta il bias del primo escluso finché `sel[5o] = sel[4o] − eps`)
e trova il bordo vero: **10× di margine** fra il gate e il punto in cui f32 cede.

L'ordine dentro i 4 non è gate — id e peso viaggiano appaiati (`slots[k]` con
`wExp[k]`), quindi una permutazione non cambia la matematica del layer — ma è
riportato lo stesso, perché cambierebbe il confronto posizionale con la routing
conformance.

### Verifica

`npx tsc --noEmit` pulito; `npm test` **252 passed** + 2 skipped (invariato: i
casi nuovi sono kernel-level, vivono in ktest); **ktest 32/32 PASS exit 0**
(da 30/30 di it.6, +2 casi router).

### Non fatto, e perché

Il kernel **non è ancora nel path caldo**: cablarlo senza consumare gli id su
GPU non toglierebbe nessun sync, perché il readback oggi serve anche a sapere
cosa *caricare*. Il pezzo successivo è lo strato 1 completo — bind group layout
esplicito al posto di `layout: "auto"`, base-offset nei GEMV expert
(`wgsl.ts` gate/up/down assumono offset 0), collasso dei 4 buffer `wExp[k]` in
uno indicizzato, e tabella slot→(buffer, offset) su GPU mantenuta dalla cache.

---

## it.10 (2026-08-02) — fase 4b: famiglia fusa portata sulla catena expert

**Ruling PI**: "porta su glm tutte le ottimizzazioni che ci hanno dato tante
soddisfazioni su qwen" ⇒ la 4b esce da `blocked-by-4` (docket item 12).

### La causa del 5.3×: non è la fusione, è la struttura del gemv

Letti fianco a fianco `gemvQuantWgsl` (che GLM usa ovunque) e la famiglia fast
di Qwen, la differenza non è "un dispatch invece di tre":

| | generico (GLM) | fast (Qwen) |
|---|---|---|
| load pesi | `qs[gb*4+w]` scalare, 4 load/blocco | `array<vec4<u32>>`, 1 load da 16 B |
| input `x` | da storage globale, ri-letto per riga | shared `vec4<f32>`, una volta per wg |
| MAC | estrazione byte a byte | `dot(vec4, vec4)` |
| righe/wg | 1 | 4 |

**Costruito**: `pairGemvSiluFastWgsl` (gate+up+silu in un dispatch, **senza rms**
— nel MoE la ffn_norm è già applicata a monte del router e rifarla la
ricalcolerebbe 4 volte per layer) e `gemvAccumFastWgsl` (down q4_0/q4_1, stessa
struttura, stesso ordine di binding del generico con `scaledAccum`). Cablati
nella catena expert: **4 dispatch per expert → 2**.

### Correttezza

ktest **35/35 exit 0** (da 32/32): tre casi nuovi contro riferimento f64, più i
due gate end-to-end — `glm-model-2layer` L2rel **2.36e-7**, argmax 6/6,
logitMaxRel 3.83e-4; `glm-layer0-conformance` sui pesi **reali** di blk.0
L2rel 2.35e-7. `tsc` pulito, `npm test` 252 + 2 skipped.

**Il gate sui dispatch è caduto, ed è stato utile**: l'atteso era ancora 61
(catena a 4). Riscritto come somma dei termini invece che come numero, così la
prossima volta si legge QUALE termine è cambiato.

`gpuforward.ts` non è stato toccato ⇒ path Qwen intatto, nessun ri-bench dovuto.

### Il risultato: corretto, ma il modello che lo prevedeva è smentito

`results/engine/bench-glm-4090-b12-fused-2026-08-02.json`, macchina quiescente:

| ms/token | pre-fusione | post-fusione | Δ |
|---|---|---|---|
| dispatch/token | 1818 | **1450** | −368 (= 46×8, esatto) |
| `gpuBusy` | 75.86 | **69.11** | **−6.75 (−8.9%)** |
| stallo residenza | 34.14 | 31.09 | −3.05 |
| sync/CPU | 94.18 | **99.71** | **+5.54** |
| wall | 204.18 | 199.91 | −4.27 |
| decode tok/s | 4.912 | **4.967** | **+1.1%** |

Il modello di it.8 (`state-2026-08-02` §2) diceva: da 5.1% a 27% del picco di
banda ⇒ `gpuBusy` da 75.9 a ~14. La catena expert vale ~902 MB dei 2220 MB
letti per token, cioè **41% dei byte**, e portarla ha reso **8.9%** del tempo
GPU. Le due letture possibili — la catena expert pesa su `gpuBusy` molto meno
di quanto pesi sui byte, oppure i kernel nuovi non sono 5× più veloci — **non
sono distinguibili con i dati che ho**.

**Errore di metodo da registrare**: spec §4 mette come PRIMO task della 4b
«spezzare `gpuBusy` per categoria (attention, router, shexp, catene expert,
head), poi fondere». L'ho considerato assolto dall'analisi di it.8, che però
misura la banda a livello di **path**, non di categoria. Non è la stessa cosa,
e questo bench è il prezzo di averle confuse.

### Il risultato più importante non è il mio, è strutturale

**−6.75 ms/token di GPU sono diventati −4.27 di wall e +1.1% di decode.** Con
46 drain per token la GPU che finisce prima aspetta solo di più: sync/CPU è
SALITO a 99.7 ms/token, il 50% del wall. Le proiezioni nel report lo dicono
senza ambiguità — **anche batchando tutti e 46 i sync (K=46) si arriva a 9.77
tok/s**, sotto il gate 13.43.

⇒ **Finché il drain c'è, i guadagni della 4b non si convertono in tok/s.** È un
fatto misurato che riguarda l'ordine delle fasi appena cambiato dal ruling: la
4b conviene *dopo* la 4, o almeno insieme. Da riportare al PI, non da decidere
qui.

---

## it.11 (2026-08-02) — attribuzione per categoria: era l'attention

**Ruling PI**: "andiamo prima con le attribuzioni per categoria". Costruito il
tagging dei compute pass per categoria (`setTelemetry(on, gpu, byCat)`): in
quella modalità i pass si spezzano ai confini di categoria e `gpuBusyMs` si
scompone. Fuori da lì il path è identico a prima — il bench headline non cambia
forma. Replica dedicata, separata da `attribution2`.

### Il risultato

`results/engine/bench-glm-4090-b12-bycat-2026-08-02.json`:

| categoria | ms/token | quota |
|---|---|---|
| **attn (MLA)** | **51.21** | **74.5%** |
| shexp | 7.16 | 10.4% |
| head | 4.64 | 6.7% |
| **experts** | **3.95** | **5.8%** |
| router | 0.98 | 1.4% |
| dense | 0.50 | 0.7% |
| addMoe | 0.25 | 0.4% |

Costo dei confini di pass: **−2.54 ms** (il totale con lo split è più BASSO di
quello senza, 68.7 vs 71.24: differenza fra repliche, non un artefatto
sistematico — i confini non stanno gonfiando la misura).

### Cosa dice, e cosa smonta

**La catena expert che ho portato in it.10 vale il 5.8% del tempo GPU.** Anche
azzerandola si guadagnano 3.95 ms su 69. Il +1.1% di decode misurato in it.10
non era una delusione: era esattamente quello che questa tabella prevede.

**Il modello "byte letti ⇒ tempo" è sbagliato di suo.** La catena expert è il
41% dei byte e il 5.8% del tempo; l'attention è ~2.6% dei byte di pesi e il
74.5% del tempo. Ogni pianificazione fatta sulla banda in questo goal — inclusa
la mia di it.8 — era su una proxy che non regge.

### Perché l'attention costa così

Letto `mlaAttnDecodeWgsl`: la griglia è **nHead = 20 workgroup da 64 thread**,
cioè **1280 thread** su una GPU che ne vuole ordini di grandezza di più. Ogni
workgroup riscorre da solo **tutta** la cache KV (due volte: scores e output),
con un loop seriale di 576 iterazioni per posizione. Non è banda e non è
FLOP — è **parallelismo assente**: 20 workgroup non riempiono nemmeno gli SM.

E la cache c_kv in MLA è **condivisa fra le head** (è il senso di MLA), quindi
le 20 riletture sono anche ridondanti per costruzione.

Il pattern che risolve esiste già nel repo, sul path Qwen: `attnSplitPart` +
`attnSplitReduce` (flash-decoding: si spezza il range KV, ogni pezzo ha il suo
workgroup, poi si riduce con max/somma correnti). È il prossimo porting, ed è
quello che vale.

### Verifica

`tsc` pulito, `npm test` 252 + 2 skipped. **Nota onesta sul report committato**:
la colonna `passesPerToken` vale 1.0 per ogni categoria — contavo un pass per
batch invece che per pass. Il bug è nel contatore, non nei ms (che sono sommati
per pass e sono corretti); corretto in questo commit, il valore giusto comparirà
col prossimo bench.

## it.12 (2026-08-03) — fase 4b: flash-decoding sulla MLA

Portato lo split sul contesto (pattern Qwen B2) sull'attention MLA, il 74.5%
del tempo GPU secondo it.11. Due kernel nuovi in `kernels/wgsl.ts`:
`mlaAttnSplitPartWgsl` — griglia (sMax,1), CHUNK=16 posizioni per workgroup da
256 thread, un chunk di cache caricato in shared a tile (stride paddato 65
anti bank-conflict) e riusato da TUTTE le 20 head: via il fattore 20 di
riletture ridondanti del monolitico — e `mlaAttnSplitReduceWgsl` (log-sum-exp
esatto, shared O(1): 8 byte, costanti in ctxMax). Wiring in `glmmodel.ts` (2
step al posto di 1, buffer `attnPartials` condiviso fra i layer), sizing
CPU-side in `mlasplit.ts` nuovo. Il monolitico resta: glmforward/glmroute e
ktest lo usano ancora, ed è il riferimento del test di identità.

### Processo (nuovo per questo goal)

Implementazione delegata a subagent Opus su brief di design chiuso; doppia
review avversaria indipendente (adversarial-reviewer Opus + Codex). Esito
convergente: ZERO difetti numerici/di indicizzazione (Opus ha verificato con un
simulatore thread-level su 11 valori di nPast: ogni cella scritta esattamente
una volta, maxAbs ≤ 1.7e-16 vs riferimento f64); tre finding strutturali reali,
tutti fixati nel secondo round: (1) il reduce memoizzava i pesi in shared
O(ctxMax) — l'esatto difetto che lo split esiste per togliere, contraddiceva
commenti e test (che scansionava solo il part: test vacuo); (2) fail-fast
tautologico in glmmodel; (3) `attnPartials` assente dai candidates di
`engineNeeds`. Dopo il fix l'output dello split è bit-identico al pre-fix.

### Verifica

- ktest 41/41 (6 casi nuovi: split vs cpuref-f64 su nPast 7/15/16/40/200,
  identità split-vs-mono maxAbs 1.08e-7); `npm test` 261+2; `tsc` pulito.
- Bench quiescente (due run, protocollo C2 invariato, host: GPU 210 MHz 0%
  a riposo prima del lancio):
  - `bench-glm-4090-b12-mlasplit-2026-08-03.json` (con attribuzione+bycat):
    decode 4.946, **attn 51.21 → 27.49 ms/token (−46%)** nella replica bycat.
  - `bench-glm-4090-b12-mlasplit2-2026-08-03.json` (headline): **decode 4.982
    ≥ 4.967 (fused-2026-08-02): non-regressione PASS**; prefill 5.74 ≥ 5.50;
    TTFT 80.4 s (da 88); dispatch/token 1497 misurati (1450+47: +1
    reduce/layer); gpuBusy 64.4 (da 69.1), gate 4b ≤54.5 ancora FAIL.

### L'attribuzione kernel-vs-clock (obbligo del done-when 4b)

Clock SM campionati durante la run headline (nvidia-smi ogni 2 s; aggregatore:
media dei campioni con utilization>10%, n=195; CSV persistito in
`results/engine/clocks-sm-it12-mlasplit2-2026-08-03.csv`): **media 948 MHz**
(min 255, max 1455) contro i 1746 di it.1 —
il cap è 3105. Il paradosso apparente della replica bycat (tutte le categorie
non-attn ~raddoppiate: shexp 7.16→14.63, head 4.64→9.61, experts 3.95→8.06)
è questo: meno lavoro GPU ⇒ più bolle relative ⇒ boost più basso ⇒ ms gonfiati
UNIFORMEMENTE. Il −46% dell'attn è quindi un LOWER BOUND del guadagno kernel a
iso-clock; il decode headline resta ~fermo perché il collo è fuori GPU
(sync/CPU 121-178 ms/token con 46 drain). Conferma sperimentale del vincolo di
it.10: la fase 4 non è opzionale.

### Residui noti (dalle review, non bloccanti, per il ledger)

- CHUNK=16 è tarato sull'occupancy a pos~525; a ctx lungo (6688+) CHUNK=64
  dimezzerebbe il traffico partials (oggi ~59% del traffico cache risparmiato).
- Il reduce serializza la scansione delle nParts su thread 0 (836 load
  dipendenti a ctx 6688) e la fase A ha un imbalance 320/256 (~37% del tempo
  score): margini piccoli, misurare prima di toccare.
- gpulimits tiene il termine workgroup-storage del monolitico (consumatore
  reale: glmforward/glmroute, ktest). Il tetto di contesto di GLMMODEL non è
  più la shared: è la VRAM della KV. Decisione di rimozione/condizionamento →
  docket item 13.

## it.13 (2026-08-03) — fase 4b: famiglia fast sui K-quant (shexp + head)

Le due categorie più grosse dopo l'attention usavano `gemvQ5K`/`gemvQ6K` con
due difetti strutturali letti nel kernel: 1 riga per workgroup da 64 thread con
soli 8 superblocchi per riga (K=2048) ⇒ 56 thread su 64 fermi, e `sbyte()` che
fa una load u32 dal global PER OGNI BYTE. Due kernel nuovi (sole aggiunte):
`pairGemvSiluQ5KFastWgsl` (gate+up Q5_K + silu fusi — lo shexp passa da 4
dispatch a 2) e `gemvQ6KFastWgsl` (down shexp e head): 8 thread per
superblocco (utilizzo pieno a K=2048), word in registri, x in shared paddato
(e+(e>>5), anti bank-conflict), riduzione ad albero. Aritmetica bit-fedele ai
gemelli lenti, che restano nel repo. `upE` eliminato (nessun silu separato
sopravvive nel path MoE). Modulo nuovo `kquantfast.ts` (tolleranze e sizing).

### Review (stesso protocollo di it.12) e il caso della tolleranza

Il ktest dello shexp fuso sfondava la tolleranza storica 2e-4 (maxRel
5.87e-4). L'implementatore l'ha giustificato con un'emulazione; la review
avversaria Opus ha PROVATO la correttezza (emulazione f64 del WGSL: rel ≤3e-13
su tutte le righe; copertura esattamente-una-volta per K 256→8192; 6 mutazioni
strutturali + fault-injection Codex su tutti i 64 sottogruppi: 0 escape, il
più favorevole dei fault dà maxRel 32 contro gate 1e-3) ma ha DEMOLITO la
giustificazione: la causa non era "l'exp del device" (amplificazione ≤1,
quindi ≤3.6e-7) ma la contrazione FMA; il cond era 1.4e5 non 4e3; e "sui pesi
veri non succede" era smentito sperimentalmente. Punto decisivo: anche la
pipeline VECCHIA (2 gemv lenti + silu separato) sfonda i 2e-4 sugli stessi
dati (3.09e-4) — il pavimento era sempre stato lì, il caso vecchio non lo
vedeva perché non fondeva il silu nel confronto. Fix round: floor test
riscritto sui kernel VERI (metà strutturale che genera il WGSL + metà
numerica con l'associatività reale — il modello corretto coincide col device
a 4 cifre: 5.869e-4), tolleranze derivate dal caso peggiore LECITO
(Q5_K pair 1e-3, margine 1.70× sul device; Q6_K 5e-4, margine 2.07× sul
no-FMA — un driver spec-compliant senza contrazione non deve dare falso
FAIL), costanti condivise in kquantfast.ts, guardia shared di glmmodel estesa
a 4 consumatori, `dot`→`dotq` (shadowing del builtin).

### Verifica

ktest 44/44; `npm test` 272+2; `tsc` pulito. Bench quiescente ×2 (clock
campionati: 863 MHz medi run 1):
(Clock: aggregatore = media dei campioni nvidia-smi ogni 2 s con
utilization>10%, n=289, min 210 max 1335; CSV persistito in
`results/engine/clocks-sm-it13-kquantfast-2026-08-03.csv` — copre la run 1.)
- `bench-glm-4090-b12-kquantfast-2026-08-03.json` (attrib+bycat):
  **shexp 14.63→5.54, head 9.61→3.81** (bycat); attn 27.5→32.0 ed experts
  8.1→13.1 su clock più bassi (863 vs 948) e replica singola — rumore+clock,
  non regressione kernel (nessun kernel attn/experts toccato).
- `bench-glm-4090-b12-kquantfast2-2026-08-03.json` (headline): **decode 5.054
  ≥ 4.982: PASS, nuovo massimo** (medie 6-replicate: 4.972→5.024);
  dispatch/token 1405 (da 1497); **gpuBusy 54.2 ≤ 54.5: prima volta sotto la
  soglia derivata del gate 4b**, coi clock CONTRO (863 MHz vs 948 di it.12 vs
  1746 di it.1): la riduzione è tutta kernel, zero dai clock.

### Prefill: analisi di non-regressione (onesta, non assorbita)

Mediane headline 5.674/5.697 contro il 5.736 committato (it.12 run 2). A
livello di repliche (6 vs 6): it.12 media 5.706 sd 0.039, it.13 media 5.691
sd 0.022, Welch t=0.82 — statisticamente indistinguibili; il 5.736 è la
mediana della tripla più fortunata, cioè un massimo storico. It.13 toglie
lavoro dal prefill (stesso forward), e il decode delle stesse run migliora.
Conclusione: nessuna regressione misurabile; un effetto reale ≤0.5% non è
escludibile dai dati. Interpretazione a banda di rumore del gate → docket
item 14 per ratifica PI (rollback a un commit se non concorda).

### Fase 4b: cosa manca per chiuderla

Il done-when chiede anche correttezza invariata (argmax ≡ cpuref-f64 sul
campione ratificato, top-1 vs golden ≥ 98.83%): non verificata in it.13 (ktest
sintetico argmax 6/6 non basta). Prossima iterazione: conformance reale
(glmconf) + routing, poi la 4b chiude e si ripresenta l'item 2 come da
done-when della fase 4.

## it.14 (2026-08-03) — conformance reale: la fase 4b CHIUDE

Misura pura, zero codice. glmconf sul campione ratificato 2/8 (p4+p7, 256
posizioni generate, budget 11 GiB come da precedente C2 anti-OOM, macchina
quiescente): `logits-conformance-glm47flash-sample47-2026-08-03.json`.

- **Gate (i): argmax motore ≡ cpuref-f64 256/256** (confronto rifatto dai file
  grezzi contro `logits-cpuref-p{4,7}-2026-08-01.json`, posizione per
  posizione). Le somme f32 riordinate da it.12 (attention split) e it.13
  (K-quant fast) non spostano NESSUN argmax del campione.
- **Top-1 vs golden: 254/256 = 99.22% ≥ 98.83%.** Le 2 divergenze sono le
  STESSE di C2 (p4 k=34 → 51, p7 k=75 → 1182) e in entrambe il motore sceglie
  lo stesso token di cpuref-f64: la firma di divergenza oracolo-vs-f64 è
  intatta. klMean 2.6e-3, maxDl 29.9 (metrica di scala, non gate).
- Full-corpus: resta il gate di fase 6 (ruling item 11); qui il campione è
  stato ESEGUITO davvero (non sostituito byte-identico come in fase 3 — con la
  matematica cambiata la sostituzione non era disponibile).

**FASE 4b DONE (it.10-14).** Checklist del done-when: gpuBusy 54.2 ≤ 54.5
(kquantfast, attribuzione: tutto kernel, clock 863 MHz CONTRO — zero guadagno
dai clock, che scendono con le bolle); dispatch/token 1405 e clock riportati;
correttezza sopra; `npm test` 272+2. Nota onesta: la soglia è presa con 0.3 di
margine a clock depressi — se la fase 4 alza i clock, gpuBusy scende ancora;
il rischio inverso (clock ancora più giù) non esiste perché la fase 4 rimuove
le bolle che li deprimono.

Prossimo: **fase 4** (eliminazione dei 46 drain) — il rischio dichiarato del
goal. Proiezione it.13 aggiornata: batchare tutti i sync dà 12.47 tok/s,
ancora sotto il gate 13.43 ⇒ serve l'eliminazione (binding fisso + offset
aritmetico, spec §3.2-bis), che diventa totale solo con la 4c (residenza).
A fine fase 4: ripresentare l'item 2 (clausola di fallback) come da done-when.

### Post-verifier (chiusura ratificata)

Verifier indipendente sulla chiusura: sostanza tutta confermata (256/256
ricalcolato dai grezzi, gpuBusy, dispatch, suite), FAIL su due punti di
bookkeeping, entrambi sistemati prima del commit: digest it.11-14 scritti
(erano fermi a it.10) e CSV dei clock persistiti in results/engine/ con
aggregatore dichiarato (i numeri del journal sono riproducibili esattamente:
util>10% ⇒ 948/195 e 863/289). Due osservazioni del verifier per la fase 6,
registrate qui: (a) i report citati nelle chiusure di fase hanno
`gates.decodePass=false` per costruzione (il gate vive a fase 6) — un campo
`phaseContext` li renderebbe auto-descrittivi; (b) la coppia
headline/attribuzione è distinguibile solo dal journal — un campo `run:
"headline"|"attribution"` nel JSON sarebbe assicurazione a buon mercato per la
catena di non-regressione di fase 6.

## it.15 (2026-08-03) — fase 4 strato 1, slice A: arena a binding fisso

Design prima del codice: proposta prodotta da agente Plan sui file veri,
rivista e approvata, persistita in
`docs/superpowers/specs/2026-08-03-engine-fase4-strato1-arena.md` (addendum a
spec §3.2-bis). Due fatti scoperti nel design che cambiavano le assunzioni:
il binding cap negoziato era 250 MiB (la testa Q6_K), non 2 GiB — i buffer di
classe attuali non erano bindabili interi; e la finestra al tetto NVIDIA dà
404/390 slab per binding ⇒ 6+1 buffer a budget 12 GiB, 7+1 a parco completo.

**Slice A implementato** (Sel {id,slot,w,flags} + uniform MoeIdx a dynamic
offset, kernel expert con opzione `arena` — corpo aritmetico unico nei due
regimi —, ExpertCache in modo arena, BGL espliciti, 4 bind group statici al
posto di ~2419×5 cached, wExp e slotBgCache rimossi). La CPU comanda ancora:
lo slice A non riduce i sync per contratto — costruisce il meccanismo per cui
alla residenza totale (4c) il readback sparisce.

### Review e fix (protocollo consueto)

Opus + Codex convergenti: ZERO difetti di correttezza (byte-identità dei
kernel non-arena ri-derivata da git archive; geometria/offset/MISS/barrier/
dynamic-offset riprodotti numericamente da entrambi, inclusi i bordi slot
403/404 e 389/390). Finding veri, tutti da mutation testing sui TEST:
il round-trip node confrontava TS con TS (mutazione dell'indirizzamento WGSL
⇒ 39/39 verdi) e la regex di ld4 non legava case↔arena (tutti su arena0 ⇒
18/18 verdi). Fix round: estrazione delle costanti e delle espressioni dal
WGSL GENERATO, backreference case↔binding — entrambe le mutazioni ora
FALLISCONO (provato applicandole e rimuovendole); più assert %16, guardie
arenaNeeds sui degeneri, caso ktest MISS (slot 0xffffffff, w≠0: uscite
bit-per-bit intatte), tipo `select` allargato all'interfaccia del design.

### Verifica

- ktest **48/48** (identità arena-vs-slotrange BIT-A-BIT su entrambe le
  classi con slab nel buffer 1 — l'arco dello switch è esercitato; MISS
  bit-per-bit; glm-model-2layer a nBuf=3 coi numeri storici); `npm test`
  283+2; `tsc` pulito; kernel non-arena byte-identici a HEAD.
- **Bench arena in produzione** (`bench-glm-4090-b12-arena-2026-08-03.json`,
  quiescente, clock util>10%: 892 MHz medi n=282, CSV committato):
  **decode 5.081 ≥ 5.054 PASS** (reps 5.147/5.064/5.081); contatori
  **identici**: 46.0 sync, 47.0 submit, 1405 dispatch per token (asserzione
  del design §4 gate 4, verificata dal report); prefill mediana 5.673 (reps
  6.306/5.672/5.673 — stessa banda statistica dell'item 14, con la prima
  replica sopra il record).
- **Rischi di produzione sciolti**: R4 NO (6+1 buffer da ~2.14 GB creati
  senza OOM); R3 NO (attn bycat 31.97→31.57: i limiti alzati non degradano);
  R1 sotto il rumore (experts bycat 13.08→12.38, in discesa).
- **Discordanza documentata**: gpuBusy della replica attrib 54.2→56.6 mentre
  la somma bycat scende 58.35→56.14 — due strumenti, direzioni opposte, a
  clock diversi fra repliche: rumore, non effetto. Nessuno dei due è gate di
  questa iterazione; il gate 4b (54.5) è stato preso alla chiusura di fase
  con la sua misura, e la ri-misura di fine fase 4 è già nel done-when.

Prossimo: **slice B** — routerTopK+resolve, slotTable, modo shadow: la
fedeltà del router GPU misurata sul corpus vero di glmroute (31 274
posizioni) invece delle 64 estrazioni sintetiche di it.9.

## it.16 (2026-08-03) — fase 4 slice B: la selezione vive su GPU, in ombra

`routerTopKWgsl` + blocco `resolve` (senza resolve: testo byte-identico a
it.9, 1318/1318 B — verificato da entrambe le review), slotTable mantenuta da
ExpertCache (shadow + flush intervallo sporco per layer, ordine slab→tabella),
modo `select:"shadow"` (il router GPU scrive Sel in regione ombra via entry
MoeIdx dedicate — deroga ratificata: 2944 B non è multiplo di 256, il raddoppio
a offset di binding era illegale — mentre la CPU comanda identica), harness
glmroute con `gpuRouterAgreement`. Bonus del fix round: `GlmRouting.vram` —
la Sel di PRODUZIONE riletta dalla VRAM e confrontata con la decisione CPU.
Trovato e corretto un bug latente dello slice A: `tableBase` indicizzato sul
layer MoE invece che assoluto (inerte allora: nessun lettore).

### Review e fix

Opus + Codex convergenti: zero difetti critici; formula del resolve identica a
routerSelect al bit (emulazione f32: wMaxRel 1.65e-7, zero flip); 5/5 mutazioni
uccise sulle unit slotTable; scenario cross-layer (vittima layer 40 + ensure
layer 5 ⇒ un flush [321,2560]) riprodotto. Fix round 7/7: contatori
submits/routerSyncs ora ASSERITI nel ktest shadow (prima telemOn=false li
lasciava a 0: l'invariante §6 non era testata), Promise.all sui tre mapAsync
(il commento "viaggia nella mapAsync esistente" era falso), struct WGSL
condivise, runner con la riga del gate GPU, design doc §3-bis con le deroghe.

### Lezione operativa (pagata due volte)

Run GPU lunghe e altro lavoro NON convivono: (1) la review Opus ha eseguito
ktest durante il full-corpus ⇒ VRAM esaurita (14.5 GiB del run + Chrome ktest)
⇒ device perso a 2250/31274; (2) il fix round ha editato wgsl.ts con vite HMR
attivo ⇒ full-reload della pagina harness a metà run 2. Norma da qui in poi:
il full-corpus si lancia ad ALBERO CONGELATO e GPU esclusiva; le review che
vogliono eseguire ktest lo dichiarano e si serializzano.

### Verifica (catena finale: ktest → full-corpus → bench, albero congelato)

- ktest: done, tutti PASS (incl. shadow-invariance con Object.is su l2/argmax/
  contatori: submits e routerSyncs invariati fra cpu e shadow, Δdispatch
  esattamente +nMoeLayer).
- **Full-corpus glmroute (31 274 posizioni, 700.9 GiB letti):
  `routing-conformance-glm47flash-shadow-2026-08-03.json`**:
  - **gate router GPU PASS: set-match 1 438 591/1 438 604 = 99.9991%**
    (soglia 99.99); ordine 99.9978%; pesi maxRel 4.43e-7, fuori tolleranza 0.
  - **Sel di produzione: 0/5 754 416 difformi** — R5 chiuso con lettura
    diretta di ciò che i kernel expert hanno consumato, non per inferenza.
  - I 13 flip: tutti sul 4° expert (rate 9.0e-6), firma R8 attesa (near-tie
    f32; it.9: tiene a 1e-6). Numero che lo slice C deve citare.
  - setMatch verso l'oracolo: prefill 87.0667% (+151 match su 1.2M vs 07-31),
    decode 88.5025% (−19 su 235k). NON identico all'artefatto 07-31 e NON è
    un bug dello slice B: l'invarianza cpu-vs-shadow è bit-identica (ktest),
    l'artefatto 07-31 è pre-it.12/13 — il riordino legittimo delle somme f32
    (validato da argmax≡cpuref 256/256 in it.14) flippa i near-tie in
    entrambe le direzioni. Netto: +132. Il "non peggiore" del done-when di
    fase 4 sul componente decode (−0.008 pp) va letto con la banda di rumore
    → aggiunto come secondo caso concreto al docket item 14.
- **Bench** (`bench-glm-4090-b12-shadow-2026-08-03.json`, quiescente):
  **decode 5.163 ≥ 5.081 PASS (quinto massimo consecutivo)**; gpuBusy 53.8;
  contatori 46 sync / 47 submit / 1405 dispatch INVARIATI (il bench gira in
  modo cpu: lo shadow è opt-in dell'harness routing). Prefill mediana 5.667
  (banda invariata, item 14).
- `npm test` 288+2, `tsc` pulito (stato finale).

Prossimo: **slice C** (interruttore `select:"gpu"`) — verificabile subito nel
ktest a residenza totale per costruzione; in produzione richiede la 4c.
Poi ri-misura gpuBusy/clock di fine fase 4 + ripresentazione item 2.

## it.17 (2026-08-04) — fase 4 slice C: l'interruttore, e il meccanismo è completo

`select:"gpu"`: un solo command encoder e UN SOLO submit per token (tail
incluso), zero readback del router, routing[] ricostruito dal readback di Sel
al tail, precondizione di residenza totale PRIMA di costruire la cache (stessa
`expertSlots`/`downIsQ4_1` della cache: divergenza impossibile per
costruzione), preload al load con zero evict garantito dalla geometria,
contatore `selMiss` con throw. Misurato nel ktest a residenza totale per
costruzione: **1 submit / 0 sync per token** (osservatore INDIPENDENTE: wrap
di queue.submit — 6 osservati = 6 dichiarati), selMiss 0, preload 64/64,
id expert identici per-k al run cpu, hidden maxRel 2.50e-7 (gate 1e-6),
pesi 1.27e-7 (il narrowing f32 strutturale: la CPU non calcola più nulla),
logit 2.56e-4 in banda indipendente 1e-3. NOTA (dal reviewer): L2rel del
mini-modello è una statistica quasi-degenere (range dinamico ~1e14) —
l'evidenza dello slice è il confronto valore-per-valore, non gli aggregati.

### Review e fix (protocollo consueto)

Opus + Codex: zero difetti critici. Codex: ricerca esaustiva su 31.9M
combinazioni layer/slot — zero casi precondizione-pass-ma-evict. Il finding
più prezioso (Opus): due dei tre pilastri del claim d'identità erano
TAUTOLOGICI (implicati per disuguaglianza triangolare da gate già in vigore)
— claim riscritto sui portatori veri e soglia logit resa indipendente.
Codex: completezza del preload non asserita (ora misses==64 e occupancy) e
submit-observer non indipendente (ora monkey-patch). Tolleranze derivate
dalla misura (margini 4×/7.9×/3.9×). +3 unit node (modelExpertPark ai
confini q4_1/q4_0).

### Verifica

ktest 53/53; `npm test` 291+2; `tsc` pulito. Bench (2 run, clock CSV
committato): la prima contaminata da jitter host (probe mapAsync 2.75 ms vs
0.110 sano, stdev 0.94 — dichiarata e scartata come da norma macchina
quiescente), la seconda pulita: **decode 5.166 ≥ 5.163 PASS** (reps
5.225/5.056/5.166) e **prefill 5.747 — sopra il vecchio record 5.736**.
Contatori di produzione invariati (il default resta cpu).

### Stato della fase 4 e ripresentazione item 2

Il MECCANISMO della fase 4 è completo e verificato (slice A: arena bit-a-bit;
B: router GPU 99.9991% sul corpus vero; C: 1 submit/0 sync a residenza
totale). Il GATE formale (≤2 sync/token nel bench di produzione) è
misurabile solo a residenza totale ⇒ la fase 4 chiude INSIEME alla 4c, che
il suo blocked-by intendeva sbloccare proprio a meccanismo pronto. Il
preload sincrono attuale è a scala ktest: la 4c deve portare preload
chunked/asincrono dal file slab (caveat in §3-ter.1). Item 2 ripresentato
al PI nel docket, con la forbice misurata.

Prossimo: **fase 4c** (residenza totale, emendamento 4) — quant asimmetrica
sugli expert freddi (matrice usage C1) + nuova versione di layout slab +
eval di perdita OBBLIGATORIA (il ruling autorizza la spesa, non esonera dal
misurarla; gate: top-1 ≥ 98.83% full-corpus, argmax ≡ cpuref sul campione)
+ preload asincrono + dimensionamento sul contesto scelto (0.67 GiB @525 /
1.03 @4096, scelta da scrivere nel report).

## it.18 (2026-08-04) — fase 4c slice A: il pilota dice di non spendere

Design 4c (doc `2026-08-04-engine-fase4c-residenza-design.md`): la matrice
usage DERIVATA dalla traccia C1 smentisce l'ipotesi "5-10% di expert freddi"
— Q3_K libera il 23.6% di uno slab, servono 627 expert @ctx525 / 935 @ctx4096
(21-32% del parco), il ranking generalizza male (2.4× ai P bassi, misurato e
corretto in §1 dopo review). Ratifica a docket item 15.

**Slice 4c-A eseguita (zero GPU, zero path caldo)**: quantizeQ3_K/Q2_K +
dequant (PRIMO quantizzatore del repo, **byte-identico a llama-quantize
--allow-requantize su tensori veri** — 12 288 + 72 superblocchi, 0 diff;
f32ToF16 fuzzato su 1.09M valori, 0 mismatch), degrade set pinnato
(anti-leakage: ranking senza p4+p7, sha 93ea3d3c...), e la LADDER DI PERDITA
su 10 configurazioni (159 min, 11 worker, base che riproduce it.14 a 256/256
con gli stessi identici fallimenti).

### Il verdetto (nel dato committato: pairedAnalysis/gate/fullCorpusFeasibility)

- **Nessuna configurazione passa il gate**: il meglio (q3k a 355-627) perde
  5 posizioni su 256 appaiato, 0 guadagnate — danno strettamente
  unidirezionale, IC95 appaiato esclude lo zero per TUTTE e 10 le config.
  Proiezione: λ≈20 perdite nette su 1024 ⇒ P(passare) ≈ 2·10⁻⁹.
- **Scoperta che riorganizza la lettura del gate**: 98.83% = 1012/1024
  MISURATO dal motore in C2 — è un pin di non-regressione, non una soglia:
  qualunque perdita netta lo rompe per costruzione.
- Q3_K domina Q2_K a ogni P; "355→627 gratis" era compensazione nella banda
  (il danno cresce monotono: KL +60%, test dei segni p=0.031) — corretto
  dopo review, l'artefatto riporta l'analisi appaiata giusta.
- R8 misurato: doppia quantizzazione Q4_0→f32→Q3_K = 16.2% RMS rel.
- Full-corpus CPU NON lanciato: cammino critico ≥31 h (p5: 6 175 posizioni
  di replay, attention O(L²) memory-bound) e non necessario — l'appaiato
  chiude la domanda. In GPU costerebbe 4.9 h, ma solo DOPO aver costruito
  kernel+import: esattamente la spesa che il pilota dice di non fare.

### Review e fix (protocollo consueto)

Opus: ladder "base decisionale affidabile" dopo ricomputazione integrale
(secondo tensore, degrade set da zero, testa riprodotta, conteggi
posizione-per-posizione). Quattro condizioni, tutte chiuse: provenienza
(resume no-op bit-identico + producedBy con gli sha), fixture estesa a 2
tensori (la mutazione is<=nstep ora FALLISCE — prima era cieca), design §1
corretto (2.42×→1.60×, apples-to-apples), minori (dominio nearestInt,
policy-id nel path dei blob, unione shard, costo dichiarato 105.9 ms/tensore,
13 GB di scratch liberati).

### Conseguenza sul goal

**La 4c per degradazione dei pesi è morta a gate invariato** — la decisione
è PI (item 15 aggiornato col verdetto; interagisce con item 2). Il loop NON
si ferma: la **fase 5 (prefill batched M>1)** è sbloccata dal ruling item 6,
indipendente dalla 4c, e il gate prefill 56.58 (+ TTFT 81 s vs 4) serve
comunque, qualunque sia l'esito della 4c.

## it.19 — 2026-08-06 — fase 4c slice A′: il probe dice che il GiB NON esiste

### Cosa

`scripts/vram-ceiling.mjs` (nuovo) + `public/vramprobe.html`: probe del
tetto VRAM allocabile REALE da Chrome/Dawn — alloca storage buffer a chunk
(256 MiB coarse, 32 fine) fino all'OOM di Dawn, banda di copia per buffer
per discriminare residente (~88-116 GB/s a clock idle) da host-backed
(8-13 GB/s ≈ PCIe), curva nvidia-smi affiancata, guardia anti-SwiftShader,
verdetto meccanico contro il fabbisogno della residenza totale. Tre run
committate: `vram-ceiling-{full,minimal,minimal-overflow}-2026-08-06.json`.

### I numeri (regola pre-dichiarata em.5: gap ⇒ docket, mai degradazione)

- Fabbisogno @ctx525 con riserva 64 MiB: **16 362 MiB** (design §2.1).
- Tetto FISICO delle allocazioni: **15 947 MiB** = total 16 376 −
  `memory.reserved` 429 (confermato indisponibile: OOM
  `VK_ERROR_OUT_OF_DEVICE_MEMORY` con smi a 15 948).
- **Gap aritmetico minimo a host PERFETTO (desktop=0, Chrome=0): 415 MiB.**
  L'opzione c non può chiudere per costruzione su questo device.
- Misurato: residenti 14.00 GiB @sessione-piena, 14.25 @minimal
  (plasmashell/krunner/greeter/bridge fermati: −177 MiB desktop).
- Oversubscription driver: pool host-backed GPU-indirizzabile **128-224
  MiB**, post-OOM **0** su 24 retry, e **INSTABILE**: dopo 5 s il driver
  aveva retrocesso i PRIMI 5 buffer allocati (1.25 GiB a 8 GB/s) — LRU
  suo, non pinnabile. Il tier freddo host-backed "gratis" non è affidabile
  né capiente (serve ≥415, meglio ~700 MiB).
- Chrome headless: niente adapter WebGPU (2 config, incl.
  `--disable-vulkan-surface`) — regime no-session non misurabile da qui e
  comunque insufficiente per aritmetica.

### Errori del probe pagati (3, tutti a costo zero sul motore)

vite dep-optimization che rinavigava la pagina (fix: pagina in `public/`,
servita senza vite client); `networkidle` mai raggiunto col websocket HMR;
headless su SwiftShader che "allocava" 14 GiB di RAM host (fix: guardia
sull'adapter info — senza, il numero sarebbe stato uno degno del docket).
E una violazione della landmine "niente pipe sull'output dei runner"
(exit code mascherato da `| head` su una run headless).

### Verifica

`npm test` 304 passed + 7 skipped; `npx tsc --noEmit` pulito; desktop
ripristinato (plasmashell active, used 1258 MiB). Nessun file di
src/engine/** toccato.

### Conseguenza sul goal

**Opzione c FALSIFICATA dalla misura: la residenza totale non sta in
questo device a nessun regime host** (−415 MiB fisici). Docket item 15
aggiornato con le opzioni rimaste (si lega a item 2, la cui clausola (a)
resta la raccomandata: chiusura ordinata sotto gate con attribuzione
"hardware, non struttura"). Il loop NON si ferma: prossimo pezzo
decidibile = **fase 4d (risanamento base)**, poi fase 5 — entrambe
indipendenti dall'esito 4c.

## it.20 — 2026-08-06 — WP-0: la tassa di replay, simulata sulla traccia vera

### Cosa

`tools/oracle-moe/sim/wp0.ts` + `run-wp0.ts` (nuovi, logica pura + runner) +
`tests/oracle-moe-wp0.test.ts` (7 unit sulla semantica). Semantica del decode
ottimistico modellata onestamente: **inserimento differito al confine di
token** (dopo il submit la slotTable è intoccabile ⇒ repair dei miss e
prefetch di token t residenti solo a t+1); prefill sincrono; LOOKA = pr
reali della traccia C1 (predictor vero, non sintetico). Belady per-accesso
come ceiling assoluto; località cross-token misurata. Artefatto:
`results/engine/moe-oracle/wp0-replay-sim-2026-08-06.json` (cost model
dichiarato nel payload: tax = E[miss]·fetch + P(dirty)·E[replayFrac]·gpuBusy,
wallClean = gpuBusy + 7.6 sync; PROIEZIONI, non misure).

### I numeri (steady-state, warmup 32; fetch serial 3.74 ms/expert)

| budget slot | regime | P(dirty) | miss/tok | tok/s @54.2 | @35 | @20 |
|---|---|---|---|---|---|---|
| 2866 | max ARITMETICO (falsificato da it.19) | 13.1% | 0.16 | 15.20 | 22.0 | 34.0 |
| 2765 | proiezione no-session | 34.0% | 0.52 | **13.66** | 19.3 | 29.8 |
| 2596 | **tetto MISURATO** (sessione minima) | 65.4% | 1.53 | **11.33** | 16.2 | 24.4 |
| 2419 | slab 12 GiB attuale | 85.2% | 3.25 | 9.55 | 13.4 | 19.6 |
| 1472 / 736 | telefono 50%/25% | 100% | 27-68 | collasso | | |

- **LOOKA come prefetch di confine: neutro al meglio, dannoso a budget
  bassi** (b=736: hit 63.2%→59.2% con K=8 — il finding WASTE dei 287 slot
  replicato). Il predictor GPU NON si costruisce per il regime ottimistico.
- **Località cross-token misurata**: W=1 32%, W=16 79%, W=64 95.4% — è il
  motivo per cui la LRU regge; il riuso è reale (decode incrementale).
- **Belady vs LRU**: gap 0.6pp a 2596 (≈ P(dirty) dimezzabile con policy
  migliore: LFRU/pin di C1 valgono anche qui), 9-19pp in scarsità.
- Ancora C1: lru@2208 = 0.9633 vs 0.9643 committato (0.1pp, codice e
  traccia invariati in git — discrepanza NON spiegata nel timebox, annotata;
  non tocca WP-0 che usa LruFast indipendente).

### Il verdetto (item 18b: "la simulazione può cambiare cosa implementiamo" — l'ha fatto)

1. **Il decode ottimistico è il meccanismo del regime near-total (≥~88%
   residenza), non della scarsità**: a 50/25% ogni token è sporco e il
   replay collassa — lì serve la macchina sync+overlap di C3b. La Pareto
   ha DUE segmenti, e vanno costruiti entrambi.
2. Al tetto misurato di OGGI (2596): 11.3 tok/s ai kernel attuali — sotto
   il gate 13.43. Il gate torna raggiungibile con una qualsiasi tra: clock
   recovery (16.2 @35ms — attesa rimuovendo le bolle, non misurata),
   margine kernel residuo (24.4 @20ms), policy migliore di LRU (Belady
   dice che P(dirty) può ~dimezzare), o il GiB no-session (13.7 @2765).
   Nessuna singola assunzione porta il claim: ne bastano 1-2 su 4.
3. **Design semplificato**: repair semplice (flag + replay dal primo layer
   sporco), niente predictor GPU, niente repair batched. late-half 34-49%
   ⇒ la variante a 2 segmenti (1 sync mid-token, entro il gate ≤2) resta
   nel cassetto se la tassa andrà limata.

### Verifica

vitest wp0 7/7; suite piena **311 passed + 7 skipped**; `tsc --noEmit`
pulito (2 errori di sintassi erasable corretti); artefatto rigenerato col
codice committato; ancora C1 entro 0.1pp (annotata).

### Conseguenza sul goal

WP-0 CHIUSO: il meccanismo va costruito, il claim di gate è condizionato e
quantificato, il design è più piccolo di quello proposto. Prossimo pezzo
(ordine item 18b): **fase 4d a perimetro pieno**. Il meccanismo ottimistico
si spec-a dopo la 4d, sulla base risanata, come apertura C3b.

## it.21 — 2026-08-06 — fase 4d pezzo 1: il punto unico di creazione device

### Cosa

`src/engine/gpudevice.ts` nuovo: `createEngineDevice({label, needs,
optionalFeatures})` = adapter + `negotiateLimits` (gpulimits, invariato) +
listener `uncapturederror` che urla con la label dell'harness. Migrati TUTTI
gli 8 siti: gpuforward (Qwen), engine.worker ×2 (diag GEMV + attn bench),
glmroute, glmconf, glmbench, ktest, microbench. `needs` accetta una funzione
dell'adapter (serve ad `arenaNeeds`, che legge maxBufferSize/binding).

### La deriva che chiude (residuo item 10a)

Tre famiglie prima della 4d: negoziato+listener (solo gpuforward);
negoziato SENZA listener (glmroute/glmconf/glmbench/ktest — 4 harness dove
un errore di validazione era un no-op silenzioso, la landmine del bring-up);
ad-hoc (engine.worker `min(adapter, 32768)`, microbench al massimo
dell'adapter — il difetto che gpulimits documenta "nell'altra direzione").
I due diag ora dichiarano i binding veri come consumatori; microbench deriva
il requisito dalla cella massima di DEFAULT_SIZES (16384² f32 = 1 GiB), non
dal tetto adapter. Comportamento invariato per ogni consumatore esistente.

### Il test che lo tiene chiuso

`tests/gpudevice.test.ts` (8 test): scansione di TUTTO src/ — `requestAdapter`
/`requestDevice` fuori dal helper = test rosso (allowlist con razionale:
probe.ts adapter-only, verificato; scripts/ probe del device grezzo fuori
scansione, di proposito). Più unit del helper su fake GPU: negoziazione
passata davvero (spot-check 30 848), needs-funzione riceve l'adapter,
optionalFeatures filtrate, listener agganciato e urlante con label.

### Verifica

`npx tsc --noEmit` pulito; `npm test` **319 passed + 7 skipped** (+8);
**ktest 53/53 PASS sul device vero** attraverso il helper (arena needs
incluse, /tmp/ktest-it21.log). NON verificati su GPU in questa iterazione:
glmroute/glmconf/glmbench/microbench/diag (stessa struttura del path ktest;
li esercita la ri-baseline dichiarata a fine 4d, a macchina quiescente).

### Prossimo pezzo (4d)

Telemetria a schema unico (contatori cumulativi diffabili, retention,
ttftMs+hostState anche Qwen) + dispatch Planned+Measured; poi debiti §7
(glmsource test, 256/256 JSON, path non-fuso, artefatti orfani); ri-baseline
per ultima (albero congelato).

## it.22 — 2026-08-06 — fase 4d pezzo 2: telemetria a schema unico

### Cosa

`src/engine/telemetry.ts` nuovo: `CoreCounters` (forwards, encodeCpuMs,
submits, dispatches, gpuBusyMs|null, gpuPasses — CUMULATIVI) + `diffCounters`
con null contagioso (null = non misurato, mai 0; una finestra con un estremo
non misurato non e' una misura). `GlmTelemetry` ed `EngineTelemetry` ora
ESTENDONO il core — l'aderenza e' un fatto del build (test che le assegna a
CoreCounters), non una convenzione.

### I tre disallineamenti chiusi

1. **Qwen era default-ON** (`?? true`, unico del repo) → default OFF come GLM;
   runBench gestiva gia' setTelemetry esplicito, e le medie ON non sono piu'
   diluite dal warmup (miglioramento dichiarato, non regressione).
2. **Qwen non aveva cumulativi**: aggiunti nSubmits/nDispatches (contati
   SEMPRE, 5+7 siti, razionale it.9) e gpuBusy/gpuPasses accumulati al
   drenaggio; `gpuMsPerToken` conserva la semantica a drenaggio (load-bearing
   in runSyncProbe, dichiarata nell'interfaccia).
3. **Report Qwen senza ttftMs/deviceLimits** → engine-bench **v4**: ttftMs
   (media+repliche OFF, definizione C3a: primo token DISPONIBILE — con k>1
   atterra a fine primo batch, dichiarato) + grantedLimits.

### Retention (lezione kimi-k3-in-c, item 18)

ExpertCacheStats: `hitsResident`/`hitsPrefetch` (oggi 0 strutturale, lo
schema non cambia quando arriva il prefetch C3b), `requests` esplicito,
`retention = 1 − evictions/requests` (null a cache vergine, mai NaN).
glmbench: retention nel PhaseResidency per finestra e nelle mediane decode
del report, SEMPRE accanto all'hitRate; runner la stampa.

### hostState (schema 4d)

`scripts/lib/hoststate.mjs`: dichiarato (--host-state / HOST_STATE, default
"undeclared" = nessuno ha controllato, informazione non errore) + campioni
nvidia-smi before/after (temp, clock SM, power, mem, util, throttle bitmask).
Wired in glm-bench-run.mjs e engine-bench.mjs — e' la firma che discrimina
regressione da host contaminato (norma 2026-08-01, lezione it.11).

### Verifica

`npx tsc --noEmit` pulito; suite **324 passed + 7 skipped** (+5: telemetry 4,
retention 1). Un test scritto male corretto in corsa (pretendeva zeroCounters
neutro su gpuBusyMs: il contagio del null E' il contratto). NON verificato su
GPU: i campi nuovi nei report bench (ttftMs Qwen, hostState, retention) si
esercitano alla ri-baseline dichiarata di fine 4d.

### Prossimo pezzo (4d)

Dispatch Planned+Measured con gate su entrambi i path (Qwen ora HA il
misurato: manca nel report e nel confronto); poi debiti §7 (glmsource test,
256/256 JSON, path non-fuso, artefatti orfani); ri-baseline per ultima.

## it.23 — 2026-08-06 — fase 4d pezzo 3: Planned+Measured con gate, sui due path

### Cosa

Il piano statico e il path eseguito non possono piu' divergere in silenzio:
- **Qwen** (engine-bench v4, `gates.dispatchPlan`): finestra DEDICATA non
  temporizzata sul path della run (k=8), contatori sempre-attivi diffati con
  `diffCounters` (il core it.22 usato davvero). Planned DERIVATO: per-token =
  step compute + 1 embedGather (solo multi-step: feedback on-GPU); submit =
  1/batch. Gate a UGUAGLIANZA ESATTA (interi su piano statico: ogni scarto e'
  drift, non rumore).
- **GLM** (glmbench `gates.dispatchPlan`, cablato nell'exit code in page.ts):
  atteso = planned + 2 (testa rms+lm_head a ogni token con readLogits, nota
  it.8), confronto con `dispatchesPerTokenMeasured` dell'attribuzione; null ad
  attribuzione spenta = non valutato, neutro.

### Verifica — ENTRAMBI i gate esercitati sul device vero

- GLM (run corta --ngen 8 --reps 1 --attrib 1): **planned 1403 + 2 = 1405 ≡
  misurato 1405 — PASS esatto**. Retention nel log (93.34%). Exit 4 dai gate
  tok/s, atteso by design (2.98 tok/s non confrontabile: nGen=8, cache
  fredda — run di aritmetica, non di performance).
- Qwen (bench completo): **planned 148 ≡ misurato 148, submit 0.125 ≡ 0.125 —
  PASS**; decode 329.6 ± 6.5 ≥ baseline 321.9 (non-regressione informale, la
  formale e' la ri-baseline); ttftMs 624 ms nel report, deviceLimits 7 chiavi,
  hostState quiescent con rampa clock 210→2250 MHz e throttle bitmask —
  TUTTI i campi it.22 vivi nel JSON
  (`bench-4090-2026-08-06T15-04-48-235Z.json`, committato).
- Suite **324+7**, tsc pulito. hoststate.mjs smoke-testato standalone.

### Ricognizione §7 fatta (per it.24, nessun edit)

- `fused: false`: ZERO consumatori nel repo (solo default `?? true` e il
  ramo) — candidato alla rimozione col git-log check;
- il 256/256 cpuref: glmconf oggi emette solo gateGolden; i dump
  `logits-cpuref-p{4,7}-2026-08-01.json` esistono in results/engine/ — il
  campo JSON si costruisce servendoli come il golden e confrontando per
  posizione (gateCpuref {agree,total,pass});
- artefatti orfani: censimento da fare su results/engine/.

### Prossimo pezzo (4d)

Debiti §7: glmsource sotto test, gateCpuref come campo JSON, decisione path
non-fuso nel journal, censimento orfani. Poi ri-baseline (ultima).

## it.24 — 2026-08-06 — fase 4d pezzo 4: i quattro debiti §7, chiusi

### 1. glmsource SOTTO TEST (em.6)

Seam iniettabile `GlmSourceDeps` (openStore/moveFile) + `SlabStoreLike`
(ExpertOpfsStore la soddisfa strutturalmente): `open()` delega a `openWith()`,
comportamento identico. `tests/glmsource.test.ts` (7 unit, store in memoria,
fixture GGUF sintetica 844-tensori ESTRATTA in tests/helpers/glm-gguf-fixture
e riusata da engine-gguf-glm): size-match ⇒ import saltato, SHA canonico
all'import, byte-check hard, range nonExpert/expert per size-class, slabRange
esatto, taglia slab monca ⇒ arriva alla decisione di rigenerare. FUORI,
dichiarato: il loop di rigenerazione (~15 GB I/O; slabFileReason gia' coperto).

### 2. Il 256/256 come campo JSON (gateCpuref)

glm-conf-run.mjs rigenera a ogni run public/models/glm-cpuref-argmax.json dai
dump `logits-cpuref-p{4,7}-2026-08-01.json`; glmconf confronta per posizione
(indice ORIGINALE del golden, non post-filtro) ed emette
`gateCpuref {agree,total,pass,source}` — pass null = non valutato (dump non
serviti o nessuna posizione del campione), neutro per l'exit; false = FAIL.
**Verificato su device**: run corta prompt 4 max-gen 8 ⇒ gateCpuref 8/8 PASS,
top1 8/8, exit 0.

### 3. Path non-fuso Qwen: RIMOSSO (decisione, con git-log check)

`fused: false` aveva ZERO consumatori (grep su src+tests); il ramo era lo
scaffolding del bring-up (nato a90e8a9 first-light) e la correttezza dei
kernel e' coperta da ktest + conformance cpuref. Autorita': done-when em.6
("rimosso o reso raggiungibile") + ruling 18c (path morto = vincolo hard).
Via: opzione `fused`, ramo else del loop (16 push), tail non-fusa, 13
pipeline compilate a ogni load (il debito §7.8: compilate e irraggiungibili),
6 buffer, 7 import kernel. Header del modulo aggiornato.
**Verificato su device**: engine-bench post-rimozione decode **324.5 ± 1.5**
(≥ baseline 321.9), dispatchPlan PASS con planned 148 invariato.

### 4. Censimento artefatti orfani (risultato: NIENTE da rimuovere)

Censimento per campo `kind` (118 JSON in results/engine) contro i produttori
in repo: orfani veri = **prefill-sim ×2** (produttore sim rimosso
deliberatamente a e9dcb4a, chiusura fase 3 B1) e **tsq-diag ×3** (citati da
docs/engine/tsq-diag-2026-07-29.md come evidenza del known-issue fase A).
Entrambe le famiglie sono evidence-bearing ⇒ SI TENGONO: giustificazione
prefill-sim PINNATA in prefillplan.ts (il knee 64 non si cambia senza
ri-misurare), tsq-diag ha gia' il doc che li cita. Il "13 orfani" dello state
doc 08-02 era impreciso: engine-prof ×5 ha il produttore vivo in
`.harness/tools/engine-prof.mjs`, prefix-cache ×5 in scripts/prefix-cache.mjs.
L'insieme "da rimuovere con nota" e' VUOTO.

### Verifica

Suite **331 passed + 7 skipped** (+7); tsc pulito; 2 run GPU committabili nei
log (/tmp/qwenbench-it24.log, /tmp/glmconf-it24.log), report Qwen nuovo in
results/engine/ (bench-4090-2026-08-06T15-23-43-832Z.json, hostState
quiescent).

### Prossimo pezzo (4d, ULTIMO): ri-baseline dichiarata

2 run quiescenti nuove Qwen+GLM con deviceLimits nel payload, sostituzione
registrata a docket (item 10 opzione a / em.6). Albero congelato + GPU
esclusiva (lezione it.16). Poi la 4d CHIUDE e parte la fase 5.

## it.25 — 2026-08-06 — fase 4d pezzo 5/ULTIMO: ri-baseline dichiarata — LA 4d CHIUDE

### Le due run (albero congelato a 1a639df, host quiescente verificato: 0%,
### 210 MHz, load 0.32, solo compositor)

- **Qwen** `bench-4090-2026-08-06T15-42-19-755Z.json`: decode **322.2 ± 0.7**
  ≥ baseline 321.9 (non-regressione PASS), ttftMs 624, dispatchPlan 148 ≡ 148,
  hostState quiescent con campioni smi, deviceLimits nel payload. Exit 0.
- **GLM** `bench-glm-4090-b12-rebaseline4d-2026-08-06.json` (protocollo
  identico al riferimento: p6, nGen 64, reps 3, b12, attrib 1): decode
  mediana **5.013** [4.907/5.013/5.061], prefill **5.661**, TTFT 81.4 s,
  dispatchPlan 1405 ≡ planned+2 PASS, retention 97.57%, stallo 24.0 ms/token,
  floor sync 6.0. Exit 4 (gate tok/s FAIL, atteso sotto clausola item 17a).

### Il delta GLM: rumore, non regressione

−0.15 tok/s vs il riferimento 08-03 (5.163, run SHADOW): Welch t≈1.7,
p≈0.16 su 3v3 — non significativo; la peggior replica del riferimento
(5.005) ≈ la mediana nuova; **deviceLimits identici byte-a-byte** ⇒ la
negoziazione unificata di it.21 non ha cambiato i limiti concessi (che era
il rischio da escludere: item 10a). Nessuna deroga: sostituzione registrata
a **docket item 19** (pre-autorizzata em.6/item 16), coi numeri per
un'eventuale contestazione PI.

### FASE 4d: DONE-WHEN COMPLETO (it.21-25)

test strutturale device ✓ (it.21) · schema report ttftMs+hostState anche
Qwen ✓ (it.22-23) · Planned+Measured con gate sui due path ✓ (it.23,
verificati su device) · glmsource sotto test ✓ (it.24) · 256/256 campo JSON
✓ (it.24, gateCpuref 8/8 su device) · path non-fuso rimosso con decisione
nel journal ✓ (it.24) · artefatti orfani censiti ✓ (it.24) · ri-baseline
dichiarata ✓ (it.25, item 19) · suite verde (331+7) · tsc pulito · ktest
invariato (53/53 a it.21).

### Prossimo: FASE 5 — prefill batched M>1

Sbloccata (ordine em.6: 4c-A′ → 4d → 5). Spec §5 del design 08-01: M=16
iniziale, identità = argmax identico su tutte le posizioni M=1 vs M>1,
owns prefill path/prefillplan/moe batched/tests. Il gate: prefill tok/s e
TTFT su p6 (oggi 5.66 tok/s e 81 s vs UX 4 s). Nota design: gli insiemi di
expert differiscono per token ⇒ arena+Sel va esteso (Sel per token o
batching per expert con gather).

## it.26 — 2026-08-06 — FASE 5 aperta: design MoE batched + piano sotto test

### La decisione di design (delegata dall'ordine 18b, criterio = numeri fase 4)

Fra "Sel per token" (catena decode ripetuta per riga: 4M dispatch/layer) e
"batching per expert con gather" (unione del chunk, un GEMM per expert),
vince la seconda, quantificata:
- **dispatch/layer**: |unione| ≈ 40 attesi a M=16 contro 64 (costo per
  dispatch misurato in fase 1: ~43 µs — la voce che la 4b ha dovuto
  abbattere);
- **traffico pesi**: ogni expert dell'unione letto UNA volta per chunk ⇒
  ÷ molteplicità media (≈1.6× a M=16, ≈4× a M=64) — il regime memory-bound
  che fase 1 misurò 20× sopra il floor. Coerente con la prescrizione di
  spec §5 ("per unione ... con maschera per token").

### La struttura che compra l'identità BIT-A-BIT

Il decode accumula i 4 expert in ordine k. Il path batched esegue gli expert
in ordine di UNIONE ⇒ se accumulasse direttamente, somme f32 riordinate =
argmax near-tie flippabili (classe slice B, rate 9e-6). Design: i kernel
scrivono y[m][k] in SLOT separati, una combine per token somma in ordine k
crescente — lo stesso ordine del decode. La classe di rischio e' eliminata
per costruzione. **Controprova nel test**: l'accumulo in ordine-unione
DIVERGE davvero (il test fallirebbe se la struttura a slot fosse inutile).

### Consegnato (slice 1)

`src/engine/glmprefillplan.ts`: `GLM_PREFILL_M = 16` (spec §5 [ASSUMED]),
`planMoeChunk` (biiezione (riga,k) → unione, ordine deterministico, pesi
selF32), `combineMoeRow` (referenza CPU f32 della combine, fround per op).
`tests/engine-glmprefillplan.test.ts` (6): biiezione, ordine, raggruppamento,
validazione hard, **identita' bit-identica** M>1 vs per-token su expert finto
non banale, controprova di divergenza.

### Verifica

Suite **337 passed + 7 skipped** (+6), tsc pulito. Niente GPU in questo
slice (piano puro CPU, convenzione fase A).

### Prossimo (fase 5, slice 2)

I kernel: GEMM per-expert con gather (rows) che scrive negli slot y[m][k],
combine WGSL in ordine k, e il percorso denso del chunk (MLA prefill batched
— pattern chunked del path Qwen adattato all'attention MLA). Poi wiring in
glmmodel + test identita' su modello sintetico (ktest) + p6 reale.

## it.27 — 2026-08-06 — fase 5 slice 2: i kernel della catena batched, BIT-IDENTICI sul device

### La correzione che ha aperto l'iterazione

Scrivendo il contratto WGSL, la catena vera del decode (glmmodel) e' emersa
diversa dalla referenza it.26: `moeOut` PARTE DALLO SHEXP (gemvShexpDown
scrive), i 4 expert accumulano in ordine k, poi `addMoe: x += moeOut` —
quindi out = x + ((((s + w0y0) + w1y1) + w2y2) + w3y3), NON
(x + w0y0) + ... `combineMoeRow` riscritta sulla catena vera (accumulatore
t da s, poi x + t) e test it.26 allineati (6/6 ancora verdi). E' il motivo
per cui la referenza CPU esiste: l'errore e' morto nel piano, non nel kernel.

### I tre kernel (wgsl.ts, varianti plain; arena col wiring)

- `pairGemvSiluGatherWgsl`: corpo aritmetico IDENTICO a pairGemvSiluFast
  (stesso ordine di riduzione ⇒ dot bit-identici), riga raccolta da wid.z
  via `gather` (u32: m | k<<16), x da xM[riga], scrive h[m][k];
- `gemvDownSlotsWgsl`: corpo di gemvAccumFast ma SCRIVE y[m][k] non pesato
  (niente accumulo — il peso e' della combine);
- `moeCombineWgsl`: xM[m] += s[m] + Σ_k w·y in ordine k — la catena esatta.
`Gpu.run` di ktest esteso a griglie [x,y,z] (retrocompatibile).

### Verifica — il gate della fase, ridotto ai kernel, sul device

ktest nuovo `prefill-moe-batched-vs-decode-chain`: catena batched vs catena
DECODE vera (pairGemvSiluFast + gemvAccumFast k-order + addInPlace), M=4 su
pool di 6 expert (molteplicita' 2.67), geometria reale (K=2048, N=1536),
pesi sintetici formato slab. Esito: **BIT-IDENTICO, maxAbs 0 maxRel 0**
(fallback R2 dichiarato ma NON servito — la contrazione FMA cade uguale).
ktest **54/54**; suite **337+7**; tsc pulito; scan WGSL di gpulimits verde
(i kernel nuovi rientrano nei limiti derivati).

### Prossimo (fase 5, slice 3)

Il percorso denso del chunk: MLA prefill batched (pattern chunked Qwen
adattato alla MLA absorbed) + shexp GEMM su M righe; poi wiring in glmmodel
(prefill path con planMoeChunk + Sel/ensure per unione) e test identita' p6.

## it.28 — 2026-08-06 — fase 5 slice 3a: shexp batch su M righe, bit-identico

### Cosa

Opzione `batch?: boolean` sui due generatori K-quant fast (`pairGemvSiluQ5K`,
`gemvQ6K`): wid.z = riga, x da xM[riga], out per riga — corpo aritmetico
INVARIATO (idioma arena: il testo non-batch e' byte-identico a HEAD,
VERIFICATO con diff programmatico dei generatori). E' lo shexp del chunk:
la base `s` della combine (it.27) calcolata su M righe in 2 dispatch invece
di 2M.

### Verifica

ktest nuovo `shexp-batch-vs-per-row`: geometria reale shexp (2048→1536→2048),
M=3, blocchi K-quant sintetici — **BIT-IDENTICO (maxAbs 0)** contro i kernel
di produzione per-riga. ktest **55/55**; suite **337+7**; tsc pulito.

### Prossimo (fase 5, slice 3b — il pezzo grosso rimasto)

L'attention MLA batched del chunk (rope+kvAppend M righe, attention causale
intra-chunk sulla rappresentazione absorbed — il pattern chunked Qwen non si
trasla 1:1, la MLA ha la cache 576/token e le teste assorbite), il dense
blk.0 chunk, poi il wiring glmmodel (router per riga + planMoeChunk + ensure
unione + Sel/gather) e l'identita' argmax M=1 vs M>1 su p6.
