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
   **`push to origin` NON è coperto da questo ruling** — Cristiano ha detto "mergiare", non
   "pushare", e `GOAL.md` li elenca come due voci distinte sotto *must docket*. Il push resta da
   chiedere (è azione verso l'esterno: la repo è pubblica su GitHub).

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
   **Decisione del PI**: è una scelta di metodologia di misura, non un fix bounded.

5. **`tsconfig.json` non ha `strict`** (pre-esistente, non introdotto da questo branch; segnalato dal
   final review). Ogni `| null` in `schema.ts`/`metrics.ts` e il `!` nell'helper `$()` di `main.ts` non
   sono controllati. Per un codebase la cui correttezza poggia sulla distinzione null-vs-zero
   (`decodeToksPerSec: null` quando non misurabile) è un buco che vale una decisione esplicita.

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
