GOAL: Il motore (fase B2, floor di decode) avvicina il wall-time di decode al tempo
GPU busy — eliminando la sincronizzazione per-token (readback argmax) e amortizzando
l'encode — senza perdere parità numerica, prefill 3.44x, né la telemetria della B1.

<!-- Contract approved by PI 2026-07-30 (chat, goal-brief B2: "diciamo di sì …
     se siamo apposto su tutto prosegui pure" + "vai con il goal").
     Ri-inquadramento del contratto di split (PI 2026-07-29: "B2 = floor dispatch
     ≤100/token"), previsto dal docket goal B1 item 2 e dal rimando ideas-ledger §I:
     col liv.2 funzionante GPU busy ≈ 2.2 ms/token vs 8.1 ms/token wall ⇒ ~73% del
     decode è fuori GPU (sync readback argmax + encode CPU), non dispatch count
     (tsq-diag-2026-07-29.md §Conseguenze). La leva primaria è il decode loop
     multi-step con feedback del token on-GPU (ledger §B, riga L4, prior art vLLM
     num-scheduler-steps); il target ≤100 dispatch diventa soglia da fissare in
     spec sui dati di attribuzione. Gate sulla QUOTA fuori-GPU (device-neutrale,
     vale anche su UMA: su M4/S22 cambia la composizione del fuori-GPU, non la
     scomposizione) con tok/s come headline (metrica standard: llama-bench tg,
     WebLLM decode tok/s); guard-rail anti-gaming sul gpuBusy richiesto dal PI.
     Goal start tag: goal-engine-fase-b2-start. -->

DONE WHEN (all measurable):
- Spec di fase B2 scritta (docs/superpowers/specs/<data>-engine-fase-b2-design.md) e
  ruling PI di approvazione registrato nel docket. La spec fissa: architettura del
  decode loop multi-step (K forward per submit, token feedback on-GPU, cadenza di
  readback; fallback a pipelining/doppio buffering del readback se l'embedding
  gather non può leggere il token id da buffer GPU), contratto di
  interruzione/streaming dei token verso l'app (arrivano a raffiche di K),
  interazione con tap hidden-states e telemetria liv.2, metodologia di misura del
  gpuBusy nel bench (liv.2 nelle repliche di misura O repliche dedicate — scelta
  dichiarata; prerequisito feature timestamp-query, da dichiarare per i device
  futuri), e — dai dati di attribuzione della fase 1 — la soglia finale della
  quota fuori-GPU E il target dispatch/token (se l'encode CPU residuo risulta
  dispatch-bound il ≤100 rientra; altrimenti archiviazione motivata nel ledger §I).
- Attribuzione misurata del wall di decode: JSON committato in results/engine/ che
  scompone gli 8.1 ms/token della B1 in GPU busy / sync readback / encode CPU /
  altro (protocollo repo, 4090 HEADED), con la predizione analitica del guadagno
  multi-step per K ∈ {2,4,8} — l'analogo della simulazione Pareto che fissò la
  soglia 3x di B1 (stessa procedura "simulazioni + Pareto", ruling PI 2026-07-29;
  la soglia sotto è provvisoria finché questo JSON non la conferma).
- Decode più veloce: bench 4090 (warmup + 3 repliche, HEADED, baseline same-day
  nello stesso report, schemaVersion 3) in cui la quota fuori-GPU del wall di
  decode (wall − gpuBusy)/wall scende da ~73% (B1) a ≤ soglia fissata in spec col
  criterio di Pareto [provvisoria: ≤50%, che sulla 4090 equivale a ~2x ≈ 240
  tok/s]; nello stesso JSON decodeToksPerSec.mean e ms/token come headline.
  GUARD-RAIL anti-gaming: gpuBusy ms/token entro il +5% della B1 (~2.2) — la quota
  si vince riducendo il fuori-GPU, non gonfiando il busy.
- Decode multi-step token-identico: run "loop K>1" vs run per-token della B1 dallo
  stesso prefisso, greedy — sequenze IDENTICHE ≥256 token, exit 0, report JSON
  committato (stesso pattern meccanico del check rollback di B1).
- Parità INVARIATA: node scripts/conformance-engine.mjs esce 0 col gate doppio
  (top-1 vs cpuref-f64 ≥99% E vs golden llama.cpp ≥97%, ≥512 token, stesso corpus)
  attraverso il NUOVO decode loop.
- NON-regressione prefill e persistenza: nello stesso bench prefillMs mean ≤ 810 ms
  (soglia 3x di B1 invariata); kv-rollback e prefix-cache re-run verdi (exit 0,
  JSON committati) col nuovo loop.
- Telemetria viva nel nuovo loop: run liv.2 con gpuMs reale (non null) e overhead
  da spenta ≤2% nel bench; profiler JSON committato con submit/token e
  dispatch/token del nuovo loop, e il target dispatch fissato in spec rispettato
  (o l'archiviazione motivata registrata nel ledger §I).

EVIDENCE OF DONE: npm test (unit nuovi CPU-side sul piano del loop multi-step);
node scripts/conformance-engine.mjs exit 0; JSON in results/engine/ (attribuzione,
bench decode con quota fuori-GPU + baseline same-day, token-identity K>1,
rollback+prefix-cache re-run, profiler, run tsq liv.2); diff spec + docket +
ledger §I aggiornato.

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
  di codice committato >30gg.

CONSTRAINTS: zero attribution AI nei commit; llama.cpp SOLO oracolo; spec-first
prima del codice; erasableSyntaxOnly; landmine HANDOFF §4 (HEADED=1, requiredLimits
espliciti, f32-first, timestamp quantizzati) + landmine B1: pipeline invalida non
urla al submit — ogni nuovo percorso di encode nasce col contratto error-scope
(throw sincrono attribuibile) e i readback si validano contro dati stale;
prefill-diag/kernel-diag restano harness permanenti; telemetria zero-overhead da
spenta; tap hidden-states preservati nel loop multi-step (vincolo DeepSpec);
tokenizer resta FUORI scope; accumulatori WGSL dichiarati nei loop azzerati
esplicitamente (memoria Tint).

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md; .harness/goals/engine-fase-b1/{GOAL,docket,journal}.md
(chiusura, item 2, lezioni fase 6); docs/engine/tsq-diag-2026-07-29.md §Conseguenze
(il dato 2.2/8.1); docs/engine/direction.md §7 (fase B) + §8 rischi 5-6;
docs/engine/ideas-ledger.md §B (riga L4 multi-step) + §I (rimandi "floor dispatch"
e "fusioni cross-layer"; riga "parametro M → spec-dec" per il confine con D);
results/engine/ (bench/prof di chiusura B1 = baseline); src/engine/gpuforward.ts
(decode loop attuale, catena sampling fusa).
