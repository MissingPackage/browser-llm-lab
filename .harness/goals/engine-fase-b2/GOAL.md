GOAL: Il motore (fase B2, floor di decode) porta il wall-time di decode al contesto
di bench verso il floor della GPU — riscrivendo l'attention decode per usare la GPU
intera (split sul contesto) e amortizzando la sync per-token col loop multi-step —
senza perdere parità numerica, prefill 3.44x, né persistenza e telemetria della B1.

<!-- CONTRATTO v2 — re-scope opzione (a), ruling PI 2026-07-30 in chat ("Opzione A,
     assolutamente. Non abbiamo fretta. L'obiettivo è scrivere il miglior motore in
     assoluto, o almeno uno dei migliori"). Il contratto v1 (gate sulla quota
     fuori-GPU, premessa "73% del decode è sync/encode") è stato refutato dai dati
     di fase 1 (journal it.1, decode-attrib-4090-*2026-07-30*.json): quota reale al
     ctx bench 20%; wall 8.09 ms/tok = gpuBusy 6.46 (di cui attention ~4.0: kernel a
     14 workgroup con loop ∝ kvLen) + sync 1.59 + encode 0.05. Leve ordinate:
     (1) attention decode split sul contesto (stile flash-decoding), (2) multi-step
     K forward/submit con readback 1/K (vale ~1.4-1.6 ms), (3) fusioni/floor GEMV
     (secondarie, solo se la spec le arruola). Il guard-rail anti-gaming sul gpuBusy
     del v1 è rimosso BY DESIGN: ridurre il gpuBusy ora È l'obiettivo; i guard sono
     parità + non-regressione + headline con baseline same-day. Fase 1 (attribuzione)
     resta DONE dal v1. Clausola di split pre-negoziata: se la fase kernel sfora il
     suo timebox, il goal si spacca via docket (B2 chiude con multi-step, kernel →
     B3) — decisione PI, non unilaterale.
     Goal start tag: goal-engine-fase-b2-start. -->

DONE WHEN (all measurable):
- Spec di fase B2 v2 scritta (docs/superpowers/specs/<data>-engine-fase-b2-design.md)
  e ruling PI di approvazione registrato nel docket. La spec fissa: design del kernel
  attention decode split-context (partizionamento del kvLen, softmax parziale con
  riduzione, scelta del fattore di split, layout scratch; parità come vincolo hard),
  architettura del decode loop multi-step (K forward per submit, token feedback
  on-GPU, cadenza readback, contratto streaming a raffiche di K), interazione con
  tap hidden-states e telemetria liv.2, metodologia di misura del gpuBusy nel bench,
  e le SOGLIE finali (decode tok/s e target dispatch/token: rientro del ≤100 o
  archiviazione motivata nel ledger §I) fissate col criterio di Pareto sui dati di
  attribuzione fase 1 + microbench del kernel split in isolamento.
- Attribuzione misurata del wall di decode [DONE, it.1 — v1]: 2 JSON in
  results/engine/ (ctx pieno + ctx32) con scomposizione encode/gpuBusy/sync,
  probe sync-floor e predictionByK.
- Decode più veloce: bench 4090 (warmup + 3 repliche, HEADED, baseline same-day
  nello stesso report, schemaVersion 3) con decodeToksPerSec.mean ≥ soglia fissata
  in spec [provvisoria: ≥240 ≈ 2x la B1 (122.4); plausibilità dai dati: wall
  target ~4.2 = attn split ~1.0 + GEMV 2.4 + sync/K ~0.2 — da confermare col
  microbench] e headline tok/s + ms/token; quota fuori-GPU e gpuBusy riportati
  nel JSON (diagnostica, non gate).
- Decode multi-step token-identico: run "loop K>1" vs run per-token dallo stesso
  prefisso, greedy — sequenze IDENTICHE ≥256 token, exit 0, report JSON committato.
- Parità INVARIATA col nuovo kernel attention E il nuovo loop: node
  scripts/conformance-engine.mjs esce 0 col gate doppio (top-1 vs cpuref-f64 ≥99%
  E vs golden llama.cpp ≥97%, ≥512 token, stesso corpus).
- NON-regressione prefill e persistenza: nello stesso bench prefillMs mean ≤ 810 ms
  (soglia 3x di B1 invariata); kv-rollback e prefix-cache re-run verdi (exit 0,
  JSON committati) con kernel e loop nuovi.
- Telemetria viva nel nuovo loop: run liv.2 con gpuMs reale (non null) e overhead
  da spenta ≤2% nel bench; profiler JSON committato con submit/token e
  dispatch/token, e il target dispatch fissato in spec rispettato (o archiviazione
  motivata nel ledger §I).

EVIDENCE OF DONE: npm test (unit nuovi CPU-side su piano loop multi-step e piano
split attention); node scripts/conformance-engine.mjs exit 0; JSON in
results/engine/ (attribuzione [c'è], microbench kernel split, bench decode con
baseline same-day, token-identity K>1, rollback+prefix-cache re-run, profiler,
run tsq liv.2); diff spec + docket + ledger §I aggiornato.

AUTHORITY GRANTED:
- may do autonomously: branch engine/fase-b2 e commit/push sul branch; modificare
  src/engine/**, tests/**, pagine bench/prof, scripts/; run locali sulla 4090;
  aggiornare docs/engine/* quando stale (ruling 2026-07-29); docket/HANDOFF
  refresh; merge su main + push a goal CHIUSO e verificato (ruling permanente
  2026-07-29).
- must docket (never do): modifiche a direction.md §3 o alla scala modelli;
  lavoro su paging/esperti (fase C) o spec-dec (fase D — se il loop multi-step
  suggerisce il riuso come batch di verifica, si annota nel ledger §I, non si
  implementa); QUALSIASI lavoro sul benchmark pubblico (contributo separato:
  repo/paper/sito propri, ruling PI 2026-07-30 — vive fuori dai goal engine);
  modifiche a src/adapters/** o allo schema results v3 della SPA; cambi al formato
  on-disk BKV1 della prefix-cache; run che richiedono M4/S22; ogni spesa; delete
  di codice committato >30gg; ESERCIZIO della clausola di split (proposta via
  docket, decide il PI).

CONSTRAINTS: zero attribution AI nei commit; llama.cpp SOLO oracolo; spec-first
prima del codice; erasableSyntaxOnly; landmine HANDOFF §4 (HEADED=1, requiredLimits
espliciti, f32-first, timestamp quantizzati) + landmine B1: pipeline invalida non
urla al submit — ogni nuovo percorso di encode nasce col contratto error-scope
(throw sincrono attribuibile) e i readback si validano contro dati stale (diag a
cold-start); prefill-diag/kernel-diag restano harness permanenti e si ESTENDONO al
kernel split; telemetria zero-overhead da spenta; tap hidden-states preservati
(vincolo DeepSpec); tokenizer FUORI scope; accumulatori WGSL dichiarati nei loop
azzerati esplicitamente (memoria Tint); MAI confrontare gpuBusy e wall misurati a
contesti diversi (lezione it.1).

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md; .harness/goals/engine-fase-b2/{PHASES,docket,journal}.md
(it.1 = attribuzione e diagnosi); results/engine/decode-attrib-4090-*2026-07-30*.json;
docs/engine/tsq-diag-2026-07-29.md §Conseguenze (corretta);
src/engine/kernels/wgsl.ts (attnFusedWgsl = il kernel da spaccare, 14 wg);
src/engine/gpuforward.ts (decode loop, catena sampling fusa);
.harness/goals/engine-fase-b1/GOAL.md (pattern contratto);
docs/engine/ideas-ledger.md §B (L4 multi-step) + §I (floor dispatch, fusioni).
