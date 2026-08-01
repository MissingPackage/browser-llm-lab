GOAL: engine-fase-c3b — Il paging degli esperti regge il regime di scarsita' vero
(budget slab ctx-aware, policy tier+AUTOPIN+PILOT-real sul router reale) e l'avvio
e' instant-on, con un modello di banda predittivo validato sulla banda fredda
misurata DENTRO il browser.

<!-- CONTRATTO v1 — CHARTERED, NON AVVIATO. Approvato dal PI 2026-08-01 in chat
     insieme allo split di direction §7 fase C ("ok lo split in 2 goal").
     Questo goal parte DOPO la chiusura di engine-fase-c3a: le sue metriche di
     non-regressione (decode, TTFT) sono i valori che C3a chiude, quindi il
     contratto va riletto e i numeri fissati a quel momento, prima di /goal.
     Soglie [ASSUMED] gia' confermate dal PI: errore del modello di banda
     +/-15%, budget slab di prova 50%/25% del parco, TTFT <= 4 s come budget UX
     (qui applicato al caso freddo, v. punto instant-on).
     Goal start tag da creare all'avvio: goal-engine-fase-c3b-start. -->

DONE WHEN (all measurable):
- Spec C3b scritta (docs/superpowers/specs/<data>-engine-fase-c3b-design.md) e
  ruling PI di approvazione registrato nel docket. La spec fissa: politica di
  budget slab in funzione di (VRAM, residente non-expert, KV(ctx)); struttura
  tier.h + AUTOPIN col tetto <= 12.5% degli slot per il pinning (ruling C1: la
  residenza NON e' skewed, top-4/layer = 21.8% delle selezioni ⇒ il valore sta
  nel prefetch, non nel pin); integrazione PILOT-real (tap hidden di L → router
  di L+1) nel forward; definizione operativa di instant-on e del suo strumento
  di misura; budget TTFT a freddo.
- WP banda fredda browser (ruling docket C1 item 4): tool committato che misura in
  Chrome la lettura OPFS a page-cache FREDDA su blocchi expert-size (5.33 MB);
  JSON in results/opfs-bench/; confronto esplicito coi numeri OS di C1 (random
  expert-size 1.63 GB/s / 3.74 ms p50; seq 3.22 GB/s); docs/engine/ideas-ledger.md
  §A e direction §8.3 aggiornati col numero browser.
- Slab ctx-aware: run a ctx 6k che NON va OOM (oggi: KV 54 KB/token ⇒ 361 MB a
  6k ⇒ OOM, HANDOFF §5), con report che mostra il budget slab calcolato e la
  VRAM di picco osservata.
- Modello di banda: formula tok/s = f(hit-rate, banda fredda, ctx, budget slab)
  committata con test, che predice i punti misurati entro +/-15% su >= 3
  configurazioni di budget slab distinte (12 GiB, ~50% del parco, ~25% del parco).
  La stessa formula predice anche il TTFT a freddo entro la stessa tolleranza.
- tier + AUTOPIN + PILOT-real: a parita' di budget stretto (50% e 25% del parco),
  hit-rate della policy > hit-rate della LRU pura misurata sugli stessi budget,
  con delta riportato; recall lookahead misurato IN-ENGINE confrontato col 92.0%
  @K=8 dell'oracolo C1 sullo stesso corpus (scostamento spiegato, non gateato).
- Instant-on: TTFT a freddo (OPFS popolata, VRAM vuota, router + shared expert +
  esperti caldi residenti subito e il resto on-demand) <= budget fissato in spec,
  coerente col budget UX <= 4 s del ruling PI 2026-08-01; report JSON committato.
- Non-regressione: decode >= il valore chiuso da C3a (>= 13.43 tok/s) e TTFT non
  peggiore di quello chiuso da C3a a ctx ~500; correttezza C2 invariata (argmax
  == cpuref-f64 sul campione ratificato, top-1 vs golden >= 98.83%); Qwen >=
  baseline quiescente 2026-08-01; npm test verde, npx tsc --noEmit pulito.
- Chiusura: docket, direction §7 (fase C chiusa), ledger §A, HANDOFF refresh;
  input preparati per la hero-demo M4, che resta PI-gated per hardware.

EVIDENCE OF DONE: run del tool banda fredda con exit 0 + JSON in
results/opfs-bench/; run glmbench a ctx 6k e ai budget slab della spec con JSON
committati in results/engine/; JSON hit-rate per policy (LRU vs tier+AUTOPIN+
PILOT); JSON instant-on; test del modello di banda verdi in npm test; bench Qwen
di non-regressione; diff di spec + docket + direction + ledger + HANDOFF.

AUTHORITY GRANTED:
- may do autonomously: come C3a (src/engine/**, tests/**, tools/**, scripts/**;
  commit/push su main a iterazione VERIFICATA; merge su main + push a goal CHIUSO
  e verificato; run locali su 4090; re-import e invalidazione degli artefatti in
  ~/.cache; aggiornare docs/engine/* quando stale), PIU' l'implementazione di
  tier.h, AUTOPIN, pin learned, PILOT-real/prefetch, instant-on e budget slab
  ctx-aware — qui sono l'oggetto del goal; creare tools/opfs-cold/**.
- must docket (never do): cambio modello-tesi, quant diversa da Q4_0, direction
  §3; testa MTP e spec-dec (fase D); benchmark pubblico (ruling 2026-07-30);
  run M4/S22 e hero-demo (PI-gated per hardware); ogni spesa; delete di codice
  committato >30gg senza check git log --follow; merge con QUALSIASI metrica
  misurata in regressione; abbassamento delle soglie di questo contratto.

CONSTRAINTS: identiche a C3a — spec-first; NON-REGRESSIONE PERMANENTE (ruling PI
2026-07-31); bench SOLO a macchina quiescente (norma PI 2026-08-01) con
dichiarazione dello stato host; parita' di corpus/contesto/protocollo; llama.cpp
solo oracolo; f32-first sul dev-loop; tap hidden-states preservati (qui sono
l'input di PILOT-real); telemetria zero-overhead se spenta; determinismo nei
report; mai gate diretti engine-vs-oracolo su near-tie; zero attribution AI nei
commit; funzione obiettivo direction §1 (30 tok/s decode, ~60 in thinking, TTFT
<= 4 s) riportata a ogni bench col gap residuo.
IN PIU': i numeri di banda fredda del browser NON si estrapolano dai numeri OS —
si misurano (lezione colibri, direction §8.6: mai numeri da run strumentate o
derivate nei confronti).

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md; .harness/goals/engine-fase-c3a/{GOAL,docket}.md
(il contratto che precede e i suoi numeri di chiusura);
.harness/goals/engine-fase-c2/docket.md item 8;
.harness/goals/engine-fase-c1/{GOAL,docket}.md (item 4: GO prefetch; item 5:
matrice usage ricalcolabile) e i JSON in results/engine/moe-oracle/ (recall,
residenza, simulazione trace-driven); docs/engine/study/colibri.md §2
(eusage/eheat/AUTOPIN, tier.h) e §1 (LOOKA/PILOT); docs/engine/study/ds4.md
(instant-on, quant asimmetrica); docs/engine/ideas-ledger.md §A;
docs/engine/direction.md §1, §7, §8.3; results/opfs-bench/.
