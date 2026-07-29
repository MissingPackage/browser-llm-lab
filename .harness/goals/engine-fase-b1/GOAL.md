GOAL: Il motore (fase B1, memoria I) prefilla multi-token, fa rollback della KV e
ri-entra in un contesto salvato su OPFS più in fretta che ri-prefillando, senza
perdere la parità numerica né il decode della fase A.

<!-- Contract approved by PI 2026-07-29 (chat, goal-brief; "vai con 2 goal" —
     approvazione in blocco, tutti gli [ASSUMED] approvati così com'erano, come per
     la fase A: soglia prefill 2× da confermare in spec; diagnosi telemetria liv.2
     timeboxed dentro B1; tokenizer fuori scope; merge a goal chiuso nei may-do).
     Direction: docs/engine/direction.md §7 (fase B) e §8.1/§8.3.
     SPLIT approvato dal PI 2026-07-29 (chat): la fase B di direction si divide in
     B1 (questo goal: memoria/latenza di entrata) e B2 (floor dispatch ≤100/token,
     goal separato futuro). Il target ≤100 NON è in questo contratto: qui vale solo
     la NON-regressione dei numeri di fase A.
     Goal start tag: goal-engine-fase-b1-start. -->

DONE WHEN (all measurable):
- Spec di fase B1 scritta (docs/superpowers/specs/<data>-engine-fase-b1-design.md) e
  ruling PI di approvazione registrato nel docket di HANDOFF.md. La spec fissa: layout
  del forward multi-token (M≤8), contratto di crop/length-pointer, formato on-disk
  della prefix-cache OPFS (chiave = hash della sequenza token-id + sha256 GGUF +
  versione layout; postura ds4: mismatch ⇒ throw), gestione quota/eviction.
- `npm test` verde con i test nuovi inclusi (unit CPU-side su: indice prefix-cache,
  piano di chunking M≤8, semantica crop — convenzione CI-senza-GPU della fase A).
- Parità INVARIATA col prefill multi-token attivo: `node scripts/conformance-engine.mjs`
  esce 0 col gate doppio della fase A (top-1 vs cpuref-f64 ≥99% E vs golden llama.cpp
  ≥97%, ≥512 token, stesso corpus committato) usando il percorso prefill M>1.
- Prefill più veloce: bench sulla 4090 (protocollo repo: warmup + 3 repliche, HEADED)
  con prefillMs mean ≤ 1/2 della baseline fase A committata (~2.44 s su 469 token)
  [ASSUMED approvato: soglia 2×; da confermare in spec B1 prima del codice] — JSON
  committato in results/engine/.
- Rollback KV provato meccanicamente: script/pagina con run "genera N, crop a pos P,
  rigenera" vs run fresco dallo stesso prefisso — sequenze token IDENTICHE (exit 0,
  report JSON committato).
- Prefix-cache OPFS provata end-to-end: salvataggio stato KV, restore in un worker
  NUOVO (sessione fredda), continuazione decode token-identica al run ininterrotto;
  nello stesso report: tempo di restore < tempo di re-prefill sullo stesso prefisso
  (entrambi i numeri nel JSON committato).
- NON-regressione decode: nel bench di cui sopra decodeToksPerSec.mean ≥ 120
  [ASSUMED approvato: fase A = 122.0, tolleranza ~2%; da confermare in spec], e run
  profiler con finestra decode invariata: createBindGroup = 0, submit/token = 1,
  dispatch/token ≤ 130.
- Diagnosi known-issue telemetria livello 2 (timestamp azzerati + corruzione compute
  su Chrome/Linux/NVIDIA), TIMEBOXED a 1 iterazione: esito accettabile anche
  "riprodotto e documentato come bug browser" con nota in docs/engine/ [ASSUMED
  approvato: era "diagnosi fase B" nel journal della fase A; B2 è solo dispatch].

EVIDENCE OF DONE: npm test; node scripts/conformance-engine.mjs (exit 0, prefill M>1);
JSON in results/engine/ (bench con prefillMs vs baseline fase A, report rollback,
report prefix-cache con restore-vs-reprefill); JSON profiler; diff spec + docket.

AUTHORITY GRANTED:
- may do autonomously: branch engine/fase-b1 e commit/push sul branch; creare/modificare
  src/engine/**, tests/**, pagine bench/prof dedicate, scripts/; run locali sulla 4090;
  aggiornare docs/engine/* quando stale (ruling 2026-07-29); docket/HANDOFF refresh;
  merge su main + push a goal CHIUSO e verificato (ruling permanente 2026-07-29).
- must docket (never do): modifiche a direction.md §3 (rulings) o alla scala modelli;
  lavoro sul floor dispatch ≤100 (è B2 — se un'occasione di fusione emerge, si annota
  nel docket, non si implementa); modifiche a src/adapters/** o allo schema results v3
  della SPA; run che richiedono M4/S22; ogni spesa; delete di codice committato >30gg.

CONSTRAINTS: zero attribution AI nei commit; llama.cpp SOLO oracolo (mai vendored);
erasableSyntaxOnly; landmine HANDOFF §4 (HEADED=1, requiredLimits espliciti, f32-first,
timestamp quantizzati); spec-first prima del codice; telemetria zero-overhead da
spenta; tap hidden-states preservati nel forward multi-token (vincolo DeepSpec);
tokenizer resta FUORI scope (la prefix-cache si chiave su token-id, non su testo);
piano fuso resta il default.

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md; docs/engine/direction.md §7 + §8.1 + §8.3;
.harness/goals/engine-fase-a/journal.md (known-issue e lezioni bring-up);
docs/superpowers/specs/2026-07-29-engine-fase-a-design.md; docs/engine/study/ds4.md
(prefix-cache/checkpoint KV); docs/deep-dive/kv-cache-layout.md (variante OPFS
"proposta a parte" = questo goal); results/opfs-bench/ (banda misurata);
src/engine/gpuforward.ts (stato attuale: KV per-layer, CTX_MAX, piano fuso).
