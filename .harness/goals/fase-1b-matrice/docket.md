# Docket — fase-1b-matrice

1. ~~plan-check: PI deve approvare `PHASES.md` prima dell'iterazione 1~~ **APPROVATO 2026-07-26** (Cristiano: "approvato"). Via libera per iterazione 1 (Fase 1 — adapter Transformers.js).

2. **Conformance test: done-when non soddisfatto** (aperto 2026-07-26, dal final review whole-branch).
   `PHASES.md` riga 1 richiede "`npm test` verde (incl. nuovo **conformance test** transformersjs)", e
   `GOAL.md` lo definisce come "load a tiny model, generate deterministic output, `capabilities()`
   matches real behavior". `tests/transformersjs-adapter.test.ts` non fa nessuna delle tre cose:
   tutti i test iniettano un engine fake, nessun modello viene caricato, nessuna assertion su
   `capabilities()`. È un buon unit test della forma DI, ma non un conformance test.
   **Nota**: lo stesso gap esiste per WebLLM — non ha mai avuto un conformance test, quindi renderlo
   bloccante per la sola Fase 1 sarebbe asimmetrico. La spec di design lo chiede per tutti gli adapter
   ("Conformance test per adapter: stesso contratto per tutti").
   **Opzioni**: (a) scrivere un vero conformance test per entrambi gli adapter (un modello ONNX tiny
   gira su CPU in Node, niente GPU — costo: runtime della suite + dipendenza di rete); (b) emendare
   GOAL/PHASES per riformulare il done-when su ciò che è stato costruito; (c) rimandare a una fase
   dedicata che copra tutti gli stack insieme.
   **Decisione del PI** — non presa autonomamente perché cambia il contratto del goal.
   **DECISO 2026-07-26** (Cristiano): scriviamo i conformance test, **opzione (c) — tutti gli stack
   nel browser**, eseguiti on-demand da uno script Playwright; `npm test` resta la suite unit
   veloce e offline. Un solo meccanismo, uniforme, già pronto per wllama in Fase 2.
   **Conseguenza sul contratto**: il done-when "verifiable via `npm test`" di GOAL.md e PHASES.md
   riga 1 è stato emendato di conseguenza — vedi nota in calce a PHASES.md.

3. ~~**Merge di `feat/fase-1b-matrice` in `main`**~~ **DECISO 2026-07-26** (Cristiano): "Va bene
   mergiare alla fine di ogni fase del goal. È una repo dove iteriamo velocemente."
   → Fase 1 mergiata in `main` con merge commit (convenzione del progetto: non fast-forward, come
   per `feat/fase-1a` e `fix/fase-1b-fixin1b`). La stessa regola vale per le Fasi 2-4: merge a fine
   fase, senza ri-chiedere.
   **`push to origin`** era rimasto fuori dal ruling originale (Cristiano aveva detto "mergiare",
   non "pushare", e `GOAL.md` li elenca come due voci distinte sotto *must docket*).
   **DECISO 2026-07-26** (Cristiano, docket): "Anche pushare è approvato su questo progetto."
   → Il push verso `origin` non richiede più di chiedere, per il resto del goal. Chiuso.

4. **`InferenceAdapter.id` allargato da `"webllm"` a `StackId`** (registrato 2026-07-26 per tracciabilità,
   non richiede azione). `GOAL.md` elenca "change the public `InferenceAdapter` contract" sotto
   *must docket*. Il cambiamento era esplicito nel piano approvato al gate plan-check, quindi è
   coperto dall'approvazione — ma nulla lo registrava, e a un audit successivo la riga di authority
   sarebbe sembrata violata. Registrato qui per chiudere il buco.
   OK

5b. **EFFETTO WARM-UP TRA CELLE — mina la comparabilità** (aperto 2026-07-26, dai dati reali di
   Cristiano in `results/4090-linux-2026-07-26T19-54-55-278Z.json`, 8 celle, stesso probe).
   Ordine di esecuzione e decode tok/s: transformersjs 31.7 → 31.0 → (webllm 101.3 → 105.3) →
   transformersjs **49.3 → 46.6** → webllm 107.8 → **115.6**. Transformers.js **+55%** e WebLLM
   **+14%** nella stessa sessione, senza alcun cambiamento nel probe.
   **Le repliche multiple non lo catturano**: le 3 repliche sono consecutive *dentro* una cella, la
   varianza intra-cella resta ±0.5–5 e il flag `high-variance` (soglia 0.15) non scatta mai. La
   soglia guarda la finestra sbagliata.
   **Conseguenza**: ogni confronto cross-stack dipende dall'ordine dei run. Il "2.04×" registrato
   in journal/HANDOFF e il "3.3×" derivato dallo screenshot sono entrambi inaffidabili per questo
   motivo; a regime il rapporto sembra ~2.3× ma senza protocollo non è dichiarabile.
   **Ipotesi non verificata**: power/clock state della GPU laptop (bassa dopo il download, sale coi
   run) + eventuale cache di compilazione shader di ONNX Runtime Web. Osservazione che la
   discriminerebbe: log dei clock via `nvidia-smi` durante una sequenza di celle.
   **Opzioni**: (a) cella di riscaldamento eseguita e scartata prima di ogni misura; (b) ordine
   randomizzato/interlacciato con più passate per stack; (c) registrare l'indice di esecuzione e
   trattare la posizione come covariata; (d) rilevare la deriva confrontando la prima e l'ultima
   cella dello stesso (stack, modello) e flaggarla come anomalia.
   **Sotto-problema**: `BenchCell` non ha timestamp — l'ordine è ricostruibile solo dalla posizione
   nell'array, e si perde se i file vengono uniti o riordinati.
   **Decisione del PI**: Cella di riscaldamento se il modello è cold (non presente in cache) e poi media su 3, 4 o 5 passate per stack

   **IMPLEMENTATO 2026-07-26** — `needsWarmup()` in `src/benchServer.ts`: se `cacheState !== "warm"`
   si esegue una generazione a carico identico e la si scarta. `"unknown"` trattato come freddo.
   Le repliche misurate restano 3 (dentro il range 3–5 del ruling).

   **VERIFICATO SU 4090 REALE — il ruling risolve il problema che mirava a risolvere, ma ne scopre
   un secondo.** 4 run, `scripts/seq-bench.mjs` (nuovo driver multi-cella), dati e log in
   `results/methodology/`:

   | run | sequenza | decode tok/s per cella | deriva |
   |-----|----------|------------------------|--------|
   | alternato | tjs, webllm, tjs, webllm | tjs 56.5 → 48.6 / webllm 113.4 → 110.5 | tjs **−13.9%**, webllm −2.6% |
   | mono ×3 | tjs ×3 | 49.9 → 45.6 → 45.1 | **−9.5%** |
   | mono ×4 | tjs ×4 | 49.2 → 45.0 → 45.4 → 46.0 | **−6.6%** |
   | warm-up **sempre** | tjs ×4 | 48.0 → 48.2 → 47.2 → 46.6 | **−2.9%** (spread 3.6%) |

   - **Il +55% è sparito.** La deriva positiva cold→warm che apriva questa voce non è riproducibile
     con il warm-up attivo.
   - **Resta una deriva negativa** di −7…−14%: la prima cella di ogni sessione browser è più veloce
     (49–50) delle successive (45–46). Riproducibile su 3 sessioni indipendenti.
   - **Non è hardware.** Strumentando `nvidia-smi` a ogni confine di cella: temperatura 48→56 °C
     (throttling di una 4090 laptop è ~87 °C), memoria GPU piatta (2177→2197 MiB), clock stabile a
     2265 MHz, `clocks_throttle_reasons` = 0x0 durante le celle (0x1 = GpuIdle solo tra una e
     l'altra). L'ipotesi power/clock-state di questa voce **è confermata solo per il primo run
     assoluto** (1455 MHz a freddo → 2265 dopo la prima cella); il residuo è stato del processo,
     non della GPU — presumibilmente compilazione shader / cache di kernel di ONNX Runtime Web.
   - **Causa del residuo**: il warm-up si applica *solo alle celle cold*, quindi ogni cella warm
     successiva paga di nuovo il primo-run lento. La cella cold del run alternato, che il warm-up
     l'aveva avuto, è la più veloce di tutte (56.5).

   **RICHIESTA DI RULING — estendere il warm-up a tutte le celle, non solo alle cold.**
   Misurato: porta la deriva da −6.6% a **−2.9%** e stabilizza il TTFT (275/270/279/290 contro
   347/403/459/429). È una riga (`needsWarmup` → `return true`), già isolata apposta. Costa una
   generazione scartata per cella (~8 s su tjs, ~2.5 s su webllm).
   Non l'ho applicato da solo perché il ruling dice esplicitamente *"se il modello è cold"*: è la
   lettera della decisione di metodologia, e cambiarla è tua.
   **Residuo dichiarato**: anche con warm-up sempre attivo resta ~3% di dipendenza dall'ordine.
   Per numeri pubblicabili al di sotto di quella soglia servirebbe l'opzione (b) del ruling
   originale — ordine randomizzato su più passate — oppure una sessione browser fresca per cella.

   **Conseguenza sui numeri già registrati**: il "2.04×" in journal/HANDOFF e il "3.3×" da
   screenshot restano non dichiarabili. Le misure pulite di oggi danno **webllm/transformersjs
   ≈ 2.0–2.3×** sulla stessa 4090 (2.01 in posizione 1, 2.27 in posizione 2) — coerenti col ~2.3×
   stimato, ma ancora ordine-dipendenti finché il ruling qui sopra non è preso.

   OK va bene l'estensione del warm-up a tutte le celle. Teniamo presente però che potrebbe non essere una situazione che non sempre si verificherebbe nella realtà. Quindi direi che conviene lasciarlo come opzione quando costruiremo la pagina pubblica (runnare con warm up o no). Il warm up per ogni cella è utile per i benchmark, mentre senza warm up è forse più utile alle persone per capire in applicazioni reali che comportamento potrebbero aspettarsi.

   **CHIUSO — implementato 2026-07-26.** `WarmupPolicy` in `src/benchServer.ts`:
   `"always"` (default, benchmark/steady-state) · `"never"` (prima esperienza reale, uso
   divulgativo) · `"cold-only"` (prima formulazione del ruling, conservata perché è ciò che ha
   prodotto i dati in `results/methodology/`). Selezionabile per singolo run via
   `MainToWorker.bench.warmup` (validato in `isMainToWorker`), con fallback sulla politica del
   server. **Nessuna UI**: la pagina pubblica dovrà solo esporre il controllo, il motore è pronto.
   Il commento sopra `WarmupPolicy` registra il punto del PI: le due modalità **misurano cose
   diverse e non vanno confrontate fra loro** — è una distinzione che va difesa anche nella UI,
   non solo nel codice, o la pagina inviterà a confronti scorretti.
   Gate: `tsc --noEmit` pulito, `npm test` 53/53, `npm run build` ok.

5. **`tsconfig.json` non ha `strict`** (pre-esistente, non introdotto da questo branch; segnalato dal
   final review). Ogni `| null` in `schema.ts`/`metrics.ts` e il `!` nell'helper `$()` di `main.ts` non
   sono controllati. Per un codebase la cui correttezza poggia sulla distinzione null-vs-zero
   (`decodeToksPerSec: null` quando non misurabile) è un buco che vale una decisione esplicita.
   **DECISO 2026-07-26** (Cristiano): "Va bene lo strict se ci risolve il problema."
   → `"strict": true` aggiunto a `tsconfig.json`. **Il codebase era già conforme**: zero errori
   introdotti, `npx tsc --noEmit` pulito al primo colpo, 50/50 test verdi, build ok. Da ora ogni
   `| null` e ogni `!` sono controllati dal compilatore, wllama incluso. Chiuso.

7. **Il protocollo di misura non ha un posto nello schema** (aperto 2026-07-26, conseguenza
   dell'implementazione del ruling #5b — richiede decisione PI).
   Il warm-up applicato viene registrato in `BenchCell.anomalies` come stringa
   `"protocol: warm-up run discarded (cacheState=cold)"`. Funziona (il campo esiste, i run restano
   auditabili) ma è un abuso semantico: `anomalies` significa "cosa è andato storto", mentre un
   warm-up applicato è il protocollo che funziona come previsto. Un consumatore che filtra le celle
   con `anomalies.length > 0` scarterebbe proprio le celle misurate correttamente.
   **Perché non l'ho deciso da solo**: `CONSTRAINTS` in `GOAL.md` dice "schema v3 implementato
   esattamente come speccato (**no ad hoc field additions beyond `qualityScore`**)". Un campo
   `BenchCell.protocol { warmupRuns, replicateCount }` sarebbe esattamente un'aggiunta ad hoc.
   **Opzioni**: (a) lasciare com'è, `anomalies` fa da log testuale (costo zero, semantica sporca);
   (b) emendare il constraint e aggiungere `protocol` in Fase 3, dove lo schema si tocca comunque
   per il bump a v3 (costo: una riga di schema + il ruling); (c) prefisso convenzionale su
   `anomalies` documentato nel README (`protocol:` = informativo, non anomalia).
   **Raccomandazione**: (b) — Fase 3 tocca già lo schema, e un campo esplicito è ciò che rende il
   protocollo verificabile a posteriori da chi legge i JSON senza conoscere la convenzione.

   **AGGIORNAMENTO 2026-07-26 — il ruling su #5b alza la posta: ora è correttezza dei dati, non
   estetica.** Con `WarmupPolicy` esistono due modalità che misurano cose diverse, e la nota in
   `anomalies` viene emessa **solo quando il warm-up avviene**. Conseguenza: un file senza note è
   ambiguo fra `policy="never"` (misura deliberata della prima esperienza) e `policy="cold-only"`
   su cache calda. Due run con numeri legittimamente diversi diventano indistinguibili nel JSON.
   Fintanto che la pagina pubblica non espone la scelta il rischio resta teorico; **nel momento in
   cui la espone, diventa reale** — chiunque scarichi due export non saprà quale confronto è
   lecito. Questo sposta la raccomandazione da "opportuna" a "da fare prima della pagina pubblica".

   OK sono d'accordo con la raccomandazione. Facciamolo in fase 3.

   **CHIUSO — implementato 2026-07-27 (Fase 3).** `BenchCell.protocol: { warmupPolicy,
   warmupApplied, replicateCount }` in `src/schema.ts`; `benchServer.ts` lo popola per ogni cella
   al posto della nota testuale in `anomalies`. Constraint di `GOAL.md` emendato di conseguenza
   (vedi riga CONSTRAINTS). Gate: `npm test` 87/87, `tsc --noEmit` pulito, `npm run build` ok.

8. **wllama: il chunk di chiusura sfora nella generate successiva — 1 check di conformance
   FALLISCE** (aperto 2026-07-27, Fase 2. Richiede una decisione: non è un bug del nostro codice).
   **Stato**: `npm run test:conformance` dà **transformersjs 8/8, webllm 8/8, wllama 7/8**, exit 1.
   Il check che fallisce è *determinism (token count) across two identical generate() calls*:
   run1=16 token, run2=15.
   **Causa radice, misurata (non ipotizzata)** — strumentando i chunk dello stream:
   - run 1: `[content=null role=assistant]` + 16 chunk con contenuto = 17 chunk
   - run 2: **`[content=undefined fin=length]`** + `[content=null role=assistant]` + 15 con
     contenuto = 17 chunk
   Il primo chunk del run 2 è il **chunk di chiusura del run 1**: wllama lo emette dopo aver già
   segnalato `has_more=false`, quindi resta in coda e viene consegnato alla chiamata successiva.
   **Il modello è deterministico**: la sequenza generata è la stessa nei due run.

   **CORREZIONE alla caratterizzazione qui sopra** (2026-07-27, dopo aver letto l'issue upstream —
   vedi sotto). Avevo scritto che "il testo generato è identico" e che l'effetto è solo sul
   conteggio: **è impreciso, e in modo che sottostima il problema**. Rileggendo i dati, il run 1
   finisce con `"8" "\n"` e il run 2 con `"8"`: il run 2 ha **perso il suo ultimo token**, che
   verrà consegnato al giro successivo. Non è solo contabilità — è la **coda della risposta che
   viene persa** e riemessa nella chiamata dopo. Per il bench significa che l'ultimo timestamp di
   ogni `generate()` manca: il decode rate è calcolato su n−1 token. Su 256 token l'effetto è
   ~0.4%, sistematico e sempre nella stessa direzione, quindi non si media via fra repliche.

   **GIÀ SEGNALATO UPSTREAM — non da noi** (verificato 2026-07-27):
   - Issue **ngxson/wllama#263** (aperta 2026-07-19, **OPEN**): "Streaming completions lose their
     final tokens; the lost tail is emitted at the start of the next completion". Descrive lo
     stesso difetto con la **stessa causa radice** che avevamo trovato per conto nostro: in
     `Wllama.getResponse()` il `break` su `has_more:false` scatta anche quando quella stessa
     risposta conteneva dati, e lato wasm `has_more` è calcolato sulla coda dei *task*, non su
     quella dei *risultati*.
   - PR **ngxson/wllama#264** (`fix/getresponse-stranded-results`, **OPEN, non merged**): rimuove
     quel `break`, continuando a pollare finché un `get_result` vuoto conferma la coda drenata.
     Costo: un round-trip vuoto per completion.
   - Ultima release: **3.5.1** (2026-06-15), precedente alla PR → **il fix non è ancora
     distribuito**, restiamo su 3.5.1 col difetto.

   **DECISO 2026-07-27** (Cristiano): opzione (a) — documentare e accettare 7/8 per ora. Niente
   workaround nel nostro codice, niente materiale da preparare per l'upstream (la segnalazione
   esiste già ed è migliore di quella che avremmo scritto). Controllo automatico ogni 3 giorni se
   il fix è arrivato; quando lo sarà, aggiornare la dipendenza e rimuovere questa deroga.
   **Sorveglianza attiva**: routine cloud `trig_018i6ZnQpZHF1tg6egjTsyST`
   ("browser-llm-lab — watch wllama streaming fix"), cron `23 7 */3 * *` UTC, ambiente
   `fedora:browser-llm-lab`. Controlla PR #264, issue #263 e la versione npm; **non fa nulla** se
   niente è cambiato, apre una PR quando il fix è rilasciato. Non può verificare il conformance
   (serve GPU reale, l'ambiente cloud non ce l'ha) e il suo prompt lo dice esplicitamente.

   **CONFERMATO ANCHE NEL BENCH REALE** (2026-07-26, `results/4090-linux-2026-07-26T22-51-39-379Z.json`):
   la cella wllama registra `completionTokens: 255` a fronte di `maxTokens: 256`. È lo stesso
   difetto visto dal contratto, ora misurato sul percorso di produzione: manca esattamente un
   token per `generate()`. Il decode risultante (25.97 tok/s, stdev 0.05) è quindi calcolato su
   254 intervalli invece di 255 — sottostima di ~0.4%, sistematica.
   **Ipotesi falsificate lungo la strada** (registrate per non rifarle):
   (a) non-determinismo numerico multi-thread — falsificata: con `n_threads: 1` il fallimento è
   identico; (b) penalità di ripetizione con storia condivisa — falsificata: con
   `penalty_last_n: 0`, `penalty_repeat: 1.0`, `seed: 42` il fallimento è identico;
   (c) prompt cache — falsificata come causa del determinismo, ma `cache_prompt: false` **è stato
   tenuto** per una ragione indipendente e più importante (vedi sotto).
   **Mitigazione già in essere**: `chunkIsToken()` scarta i chunk senza contenuto, quindi il chunk
   sforato **non contamina i timestamp né il TTFT** — l'effetto residuo è solo sul conteggio.
   **Opzioni**: (a) accettare 7/8 per wllama documentandolo, e ricordare che l'ultimo token di
   ogni generate è contabilizzato al giro dopo; (b) drenare lo stream con una lettura extra al
   termine di ogni `generate()` (costo: una chiamata in più per replica, dentro o fuori dalla
   finestra cronometrata — da decidere, perché se finisce dentro falsa la misura); (c) usare
   `usage.completion_tokens` come fonte del conteggio invece della lunghezza dei timestamp —
   ma arriva proprio sul chunk che sfora, quindi nella stessa chiamata non è disponibile;
   (d) segnalare upstream a ngxson/wllama e nel frattempo tenere (a).
   **Raccomandazione**: (a) + (d). Il contratto sta riportando un fatto vero sullo stack, e
   silenziarlo con un workaround nel nostro codice nasconderebbe un comportamento che chiunque
   usi wllama per misurare incontrerà.
   **Nota di scope**: `PHASES.md` riga 2 chiede "conformance test wllama" nel done-when. Con 7/8
   la fase **non è dichiarabile completa** finché non decidi fra le opzioni sopra.

   OK. Un subagent dedicato dovrà documentare tutto e preparare un esempio replicabile, così abbiamo il materiale per segnalarlo a wllama upstream

   **Nota successiva del PI (chat, 2026-07-27), che supera quella qui sopra**: "ti avevo scritto di
   preparare il materiale per segnalarlo upstream ma ho visto poi che lo hanno già fatto […] Lato
   nostro per il momento documentiamolo come hai suggerito." → **nessun subagent, nessun materiale
   upstream da preparare**: issue #263 e PR #264 coprono già il caso meglio di quanto avremmo
   fatto. Il done-when di Fase 2 è emendato di conseguenza — vedi nota in calce a `PHASES.md`.

9. **`@wllama/wllama` è il nome reale del pacchetto, non `wllama`** (registrato 2026-07-27,
   non richiede azione). `GOAL.md` autorizza a installare "`wllama`" e il design doc usa lo stesso
   nome, ma su npm `wllama` **non esiste** (404); il pacchetto è `@wllama/wllama` (3.5.1).
   Installato quello, che è chiaramente ciò che era inteso. Registrato perché a un audit la riga
   di authority sembrerebbe non corrispondere.
   **Difetto di packaging da conoscere**: la 3.5.1 dichiara `main: "index.js"` ma non pubblica
   nessun `index.js` alla root (solo `index.ts`) e non ha campo `exports`. Col bare specifier
   TypeScript ripiega sul sorgente `.ts` e `erasableSyntaxOnly` fallisce. L'adapter importa quindi
   da `@wllama/wllama/esm/index.js` — stessa forma che il README upstream usa per altri moduli.

6. **Minor accettati come sono** (dal final review, registrati per non perderli):
   `STACK_IDS` castato a `string[]` per `.includes` (cosmetico); `stackSel.value as StackId` non
   validato (safe: `isMainToWorker` ri-valida lato worker — il fix vero è renderizzare `#stack` da
   `STACK_IDS`, naturale quando arriva il terzo stack); `sharp`/`onnxruntime-node` transitivi da
   `@huggingface/transformers` (solo install-size/supply-chain, il build browser emette solo
   `ort-wasm-*`); `defaultEngineFactory` senza copertura unit (stesso precedente di
   `WebLLMAdapter.defaultFactory`; la verifica è il run reale).
   Altri minor da valutare in Fase 2: `TextStreamer` ri-decodifica il `token_cache` a ogni token dentro
   la finestra cronometrata (<1% stimato, stessa classe di contaminazione già corretta in `216c31c`);
   `promptTokens: null` per le celle transformersjs (WebLLM registra 469) rende non auditabile la
   comparabilità del TTFT; `completionTokens` ha provenienza diversa per stack (usage autorevole vs
   conteggio dei chunk) senza che lo schema lo dichiari; `bench.worker.ts` importa staticamente
   entrambi gli adapter (bundle worker 6.55 MB, fuori da ogni finestra cronometrata).

10. **`quality.ts` costruito e testato ma non collegato a `benchServer.ts`** (registrato
    2026-07-27, Fase 3 — ambiguità risolta con un default sicuro, non richiede ruling per
    procedere, ma la scelta va conosciuta prima di dichiarare la Fase 4).
    `PHASES.md` riga 3 chiede solo "unit test perplexity + fallback 12-prompt, `tsc` pulito,
    `qualityScore` tipato" — non chiede che una cella reale porti un punteggio. Ho quindi reso
    `BenchCell.qualityScore` **opzionale** e non ho toccato la pipeline di generazione in
    `benchServer.ts` per calcolarlo.
    **Perché non l'ho deciso da solo come "wiring incluso"**: collegare `quality.ts` a una cella
    reale significa eseguire 12 `generate()` extra (fallback exact-match) o un passaggio a
    logprobs per ogni bench — un aumento non banale del costo/tempo di ogni run reale sulle GPU
    fisiche, che nessun docket ha ancora approvato esplicitamente.
    **Conseguenza**: i run reali (Fase 4, sweep sui 3 device) **non includeranno
    `qualityScore`** finché questo collegamento non viene fatto. Se l'intento è che il README di
    Fase 4 riporti anche un confronto di qualità, questo va deciso e implementato prima o durante
    Fase 4 — altrimenti Fase 4 procede riportando solo le metriche di velocità, come oggi.
    **Opzioni per quando serve**: (a) invocare `evaluateExactMatch`/`computePerplexity` dentro
    `benchServer.ts` dopo le repliche cronometrate (costo: tempo extra per cella, ma fuori dalla
    finestra cronometrata delle metriche di velocità); (b) farlo come passo manuale separato,
    fuori dal driver di bench, se il costo per cella è ritenuto eccessivo per lo sweep sui 3
    device.

11. **Goal `fase-1b-matrice` chiuso** (2026-07-27, ruling PI: "possiamo chiudere questo goal").
    Tutte e 4 le fasi complete, mergiate e pushate su `main`. Cristiano testa lui stesso M4 Pro e
    laptop domani; per l'S22 Ultra resta da capire l'approccio (vedi HANDOFF.md — risposta su come
    raggiungere il dev server da telefono). Docket #10 e #8 restano aperti/attivi ma non bloccano
    la chiusura — non erano condizioni di DONE WHEN di questo goal.
