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

Clock SM campionati durante la run headline (nvidia-smi, 2 s, 195 campioni
GPU-attivi): **media 948 MHz** (min 255, max 1455) contro i 1746 di it.1 —
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
