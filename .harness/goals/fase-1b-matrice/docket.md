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

3. **Merge di `feat/fase-1b-matrice` in `main`** (aperto 2026-07-26). `GOAL.md` elenca "merge to `main`"
   e "push to `origin`" sotto *must docket*. Fase 1 completa e verde (47/47, tsc pulito, build ok,
   due datapoint reali committati), ma le Fasi 2-4 del goal sono ancora `ready`. Da decidere se
   mergiare ora la sola Fase 1 o tenere il branch fino a fine goal.

4. **`InferenceAdapter.id` allargato da `"webllm"` a `StackId`** (registrato 2026-07-26 per tracciabilità,
   non richiede azione). `GOAL.md` elenca "change the public `InferenceAdapter` contract" sotto
   *must docket*. Il cambiamento era esplicito nel piano approvato al gate plan-check, quindi è
   coperto dall'approvazione — ma nulla lo registrava, e a un audit successivo la riga di authority
   sarebbe sembrata violata. Registrato qui per chiudere il buco.

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
