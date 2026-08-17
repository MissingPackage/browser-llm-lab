# GLOSSARY — browser-llm-lab

Mezza riga per termine. Serve agli agenti e alla sessione successiva, non a
Cristiano: chat, digest e HANDOFF restano in linguaggio piano a prescindere.
Un termine si aggiunge nell'iterazione che lo conia, non quando qualcuno ci si
confonde. ⚠ = collisione dichiarata: due cose che si chiamano uguale.

## Protocollo di lavoro

**PI** — chi decide funzione obiettivo e gate (Cristiano). Tutto il resto — ordine, meccanismo, strumenti — è dell'agente.
**ruling** — decisione del PI, registrata; una volta data non si rinegozia senza nuove prove.
**docket** — le decisioni *registrate e non prese*, in attesa di ruling. `.harness/goals/<goal>/docket.md`.
**docket-born** — una fase che nasce `blocked` perché servirebbe più autorità di quella concessa dal contratto.
**riga** — una fase di `PHASES.md`. "La riga 2" = la fase 2, mai una riga di codice.
**gate secco** — condizione binaria che blocca il merge, senza banda di tolleranza (es. argmax identico al pin).
**fase 0** — la fase di sole *misure* che precede ogni riscrittura e ha il potere di chiudere il goal. ⚠ non è "la prima fase": è un tipo di fase, e in `engine-ttft` è la riga 1.
**regola di stop** — la soglia sotto cui una fase 0 chiude il goal invece di procedere (finora: nessuna variante ≥ 1,5× sulla forma attuale).
**plan-check** — il gate che ferma l'iterazione 1 finché il PI non approva `PHASES.md`.
**stop-by-design** — il loop si ferma perché il resto è docket-gated, non perché è fallito.
**verificatore** — subagente in sola lettura che grada la pretesa di un'iterazione con PASS/FAIL; non ripara niente.

## Misura

**ktest** — la suite dei kernel eseguita nel browser vero contro un riferimento CPU. `tools/harness/engine-ktest.mjs` + `src/.../ktest.worker.ts` (era `.harness/tools/`: spostato il 2026-08-17, quei cinque strumenti sono del motore, non processo).
**cpuref** — l'implementazione di riferimento in f64 su CPU contro cui il kernel GPU si confronta.
**golden / pin** — l'uscita congelata del modello (argmax per posizione) su un corpus fisso; "identico al pin" è un gate secco. `results/engine/golden/`.
**freeze** — sha256 del *testo WGSL generato*: impedisce che un kernel cambi per sbaglio. Copre solo i call-site che qualcuno ha inchiodato.
**pavimento / floor test** — l'errore che un kernel *corretto* produce comunque, derivato dal testo WGSL generato con e senza contrazione FMA (`floor = max(senzaFMA, conFMA)`). La tolleranza del ktest si **deriva** da lì e non si sceglie: troppo stretta dà falsi rossi su driver che fondono diversamente, troppo larga non discrimina. Verificato tre volte di fila che predice il silicio — gli errori misurati stanno *sul* pavimento, non a metà strada verso la tolleranza.
**mutazione (del floor test)** — il controllo che dimostra che la tolleranza discrimina davvero: si rompe di proposito il termine caratteristico del formato (`-dmin·Σx` sul q4_K, l'offset −32 sul q6_K, i pesi letti come u8 invece che i8 sul q8_0) e si verifica che l'errore esploda di ordini di grandezza. Senza, una tolleranza larga passa per rigore.
**non-reg** — misura che dimostra che un numero già raggiunto non è peggiorato. Banda ±5% su tok/s e TTFT.
**hostState** — lo stato dell'host accanto al numero: `declared` è la parola dell'operatore, `before`/`after` sono campioni di `nvidia-smi`. ⚠ nessun runner confronta i due — un `"quiescent"` mentito passa (docket `engine-ttft` item 4).
**decodeContext** — quante posizioni di contesto c'erano *mentre* si misurava il decode. Un tok/s senza questo non è confrontabile: il ms/token cresce col contesto.
**full-corpus** — tutti gli 8 prompt. Obbligatorio per firma/non-reg/riferimenti nuovi; l'esplorazione va su un sottoinsieme di prompt interi, mai su `--cap`.
**instanton** — la misura che parte da page cache *fredda*: evita i file di backing OPFS con `fincore` prima di eseguire. `scripts/glm-instanton-run.mjs`.
**b12 / b1472** — il budget di slab in GiB passato al bench GLM (`--budget-gib 12`). Non è un identificatore di build.
**attribution / attrib** — quanto di un tempo sta fuori dalla GPU (lettura, upload, API) invece che nel calcolo.

## Motore — forme e residenza

**prefill** — la fase che consuma il prompt, tutti i token noti in anticipo. ⚠ sul 4B il bench oggi lo esegue *una posizione alla volta* (`q35conf.worker.ts:207`), che è decode travestito.
**decode** — la generazione un token alla volta, ognuno dipendente dal precedente. M = 1 per necessità.
**M / mMax / PREFILL_M** — ⚠ tre nomi per la stessa cosa: quante righe di token attraversano i pesi in una passata. `M` nei discorsi, `mMax` nei kernel, `PREFILL_M` nella costante (`prefillplan.ts`).
**batch** (opzione dei kernel) — ⚠ **non** è il batch d'inferenza: è "M righe dispatchate sull'asse z", cioè M copie indipendenti dello stesso kernel. Riuso dei pesi zero per costruzione.
**workgroup storage** — lo scratchpad veloce condiviso dai thread di un workgroup. WebGPU ne garantisce 16.384 B; questa scheda ne concede 49.152.
**"il limite dei 16 KB"** — ⚠ due referenti diversi, ed è già costato una diagnosi sbagliata: (a) `maxComputeWorkgroupStorageSize`, il default di spec WebGPU sopra; (b) uno `slice(0, 16000)` nel workflow `sdd-conductor`, che troncava le patch. Dire sempre quale.
**GQA** — più teste di query condividono una testa KV (qui 4:1). Se ogni testa legge la KV per conto suo, la stessa memoria si legge 4 volte.
**K-quant** — le quantizzazioni a blocchi con super-scale (Q4_K, Q6_K, Q5_K), diverse dalla q4_0 "nuda". Il 35B è tutto K-quant e Q8_0; dal goal `engine-kquant` i loro kernel esistono ma non sono cablati (v. **wired**).
**forma multi-riga (multirow)** — il GEMM di prefill che fa attraversare i pesi da M righe di token in UNA passata. La forma vecchia (`legacy`) è M gemv replicate: riuso dei pesi ZERO, i pesi riletti M volte. ⚠ a M=1 la multi-riga *perde* (0,91× sul Q4_K): il piano non deve mai offrirla al decode.
**split-K** — la riga di uscita è divisa in `splits` fette calcolate in parallelo e ricombinate da un dispatch a parte. Il numero di fette non si sceglie: viene da `PREFILL_GEMM_SPEC[kind].splitsFor`, ed è quello *misurato* per quel formato.
**copertura del piano** — il rapporto `M·(F+L)/(F+M·L)` sull'inventario per-layer intero: quanto del prefill è passato alla multi-riga. Pinnato dal test `[6c]`. 5,8593× → 15,5247× in questo goal, cioè 200/248 siti = 99,796% dei byte.
**wired / wiredWhy** — la separazione fra *misurato* e *cablato* (`PREFILL_GEMM_SPEC`, goal `engine-kquant` riga 4): un kind può avere il kernel in produzione, portato e verificato col ktest, e non essere instradato da nessun sito. `wiredWhy` è la ragione, che finisce in telemetria — un booleano nudo non è diagnosticabile. ⚠ **è un flag per FORMATO, non per shape**, e `prefillGemmCheck` non guarda N: accendere `wired` su q8_0 per il 35B instraderebbe *nello stesso istante* i 48 siti del 4B a N=32 che il contratto esclude coi numeri.
**repack** — riordinare i pesi del GGUF nel layout che il kernel vuole, una volta al load.
**slab** — un blocco unico che impacchetta gate/up/down di una classe di expert, per bindarli insieme. `src/engine/moe.ts`.
**arena** — il regime in cui gli expert vivono in pochi buffer grandi invece che uno per tensore.
**residency-bound** — limitato dal far entrare i pesi in VRAM, non dal calcolo. GLM lo è: nessuna leva sui kernel lo sposta.
**tier** — politica di residenza LRU + autopin, contrapposta a `lru` semplice. ⚠ nei nomi dei file di bench, `tier8`/`tier12` è invece il *tetto di VRAM in GiB* passato con `--vram-gib`.
**decode ottimistico** — si sottomette il token successivo senza aspettare la conferma del precedente; insert/evict solo al confine di token.
**spec-dec MTP / predizione doppia** — il modello propone più token e se ne verifica l'accettazione. In albero e disattivato; oggi senza una funzione obiettivo che lo reclami (docket `engine-ttft` item 2).

## Metriche di prodotto

**TTFT** — ⚠ nel codice `ttftMs = loadMs + prefillMs + firstMs` (`q35conf.worker.ts:246`), quindi **include il caricamento del modello**. L'obiettivo dei 4 s vale invece a modello *caldo*, `prefill.ms + decode.firstMs` (ruling A, 2026-08-13). Citare sempre i tre termini scomposti.
**a caldo / a freddo** — a caldo = pesi già in page cache e turno non-primo; a freddo = primo turno dopo il load. Il 35B rende 22,6 contro 9,58 tok/s fra i due: il divario è paginazione, non kernel.
**tok/s** — sempre col suo `decodeContext` accanto, o non è una misura.
**µs/posizione** — la pendenza con cui il ms/token cresce col contesto. È il numero che dice se un motore regge il contesto lungo: 10,4 prima del goal sui kernel, 0,15 dopo.
**regime di lettura (`readRegime`)** — da dove sono arrivati i byte di una finestra di I/O: `disk` (banda sotto `OPFS_DEVICE_CEILING_GIBS`, 4 GiB/s misurati su questo host) oppure `os-cache` (sopra: li ha serviti la page cache del sistema operativo, non il dispositivo). ⚠ **due run in regimi diversi non sono confrontabili**: il GLM b12 ha dato 15,3 e 11,3 tok/s con lo stesso codice e gli stessi `bytesRead`, e la sola differenza era questa (it.20). Un riferimento che non lo dichiara non è un riferimento.
**kfan** — il collasso dei topK expert in un giro solo di dispatch (`wid.z = k`): 1.320 → 200 dispatch/token sul 35B.
**parallelismo effettivo** — somma delle durate delle singole letture diviso la durata di parete della finestra. ~1 = letture di fatto seriali, ~N = davvero parallele. ⚠ **non confonderlo col massimo in volo**: it.34 leggeva «picco 24, parallelismo 2,95» come «ne emette 24 e ne serve 3», mentre erano un massimo e una media dello stesso intervallo — un path che chiede per lo più 3 range alla volta dà quei numeri senza che nessuno serializzi niente.
