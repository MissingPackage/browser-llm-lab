GOAL: engine-fase-c3c — Il paging degli esperti regge il regime di scarsita' vero
(budget slab ctx-aware, policy tier+AUTOPIN+prefetch migliore della LRU pura sul
router reale), l'avvio e' instant-on, e un modello di banda predittivo e' validato
sulla banda fredda misurata DENTRO il browser.

<!-- CONTRATTO v1 (perimetro ex-C3b chartered 2026-08-01, riscritto 2026-08-07
     dopo lo split registrato in .harness/goals/engine-fase-c3b/docket.md item 2).
     APERTO 2026-08-08 (PI in chat: "partiamo con c3c") a C3b CHIUSO (83477a9).
     Numeri di non-regressione fissati alla chiusura C3b (docket c3c item 2,
     ESEGUITO): riferimenti sync + optimistic b12/tetto, banda ±5% (ruling c3b
     item 9). Rilettura all'apertura: journal it.0 (nota: "TTFT a caldo ~18 s"
     nel commento sotto è superato — riferimenti C3b: 16.79 b12 / 12.60 tetto).
     Modifiche sostanziali rispetto al chartered 2026-08-01, da rileggere:
     (1) il decode ottimistico e' USCITO (e' C3b); (2) il floor 13.43 e' ENTRATO,
     con clausola pre-negoziata; (3) PILOT-real e' riformulato dopo WP-0
     (predictor al confine di token falsificato: qui si costruisce il prefetch
     in-forward, non quel predictor); (4) l'instant-on <= 4 s assoluto e'
     sostituito da un target relativo [ASSUMED], perche' aritmeticamente fuori
     finche' il TTFT a caldo e' ~18 s — la ri-negoziazione del budget assoluto
     e' un docket item all'avvio.
     Goal start tag da creare all'avvio: goal-engine-fase-c3c-start. -->

DONE WHEN (all measurable):
- WP banda fredda browser (ruling docket C1 item 4) — PRIMO PEZZO, blocca la
  spec: tool committato in tools/opfs-cold/** che misura in Chrome la lettura
  OPFS a page cache FREDDA su blocchi expert-size (5.33 MB); JSON in
  results/opfs-bench/; confronto esplicito coi numeri OS di C1 (random
  expert-size 1.63 GB/s / 3.74 ms p50; seq 3.22 GB/s); ideas-ledger §A e
  direction §8.3 aggiornati col numero browser. I numeri OS NON si estrapolano
  (lezione colibri, direction §8.6).
- Spec C3c scritta (docs/superpowers/specs/<data>-engine-fase-c3c-design.md) e
  registrata a docket DOPO il WP banda fredda e coi suoi numeri dentro (stesso
  regime di approvazione di C3b: ruling PI bloccante solo se tocca gate o
  soglie). La spec fissa: politica di budget slab in funzione di (VRAM,
  residente non-expert, KV(ctx)); struttura tier.h + AUTOPIN col tetto <= 12.5%
  degli slot per il pinning (ruling C1: la residenza NON e' skewed, top-4/layer
  = 21.8% delle selezioni ⇒ il valore sta nel prefetch, non nel pin);
  definizione operativa di instant-on e del suo strumento di misura; budget
  TTFT a freddo.
- Prefetch, non predictor: la spec distingue il prefetch IN-FORWARD (tap hidden
  di L → router di L+1, dentro il token) dal predictor al CONFINE di token, che
  WP-0 ha falsificato (b=736: hit 63.2% → 59.2% con K=8 — terza replica del
  finding WASTE) e che NON si costruisce. Il recall lookahead si misura
  IN-ENGINE e si confronta col 92.0% @K=8 dell'oracolo C1 sullo stesso corpus;
  scostamento spiegato, non gateato.
- Slab ctx-aware: run glmbench a ctx 6k che NON va OOM (oggi KV 54 KB/token ⇒
  361 MB a 6k ⇒ OOM, HANDOFF §5), con report che mostra il budget slab
  calcolato e la VRAM di picco osservata.
- Modello di banda: formula tok/s = f(hit-rate, banda fredda, ctx, budget slab)
  committata con test, che predice i punti misurati entro +/-15% su >= 3 budget
  slab distinti (12 GiB, ~50% del parco, ~25% del parco). La stessa formula
  predice anche il TTFT a freddo entro la stessa tolleranza.
- Policy > LRU: a parita' di budget stretto (50% e 25% del parco), hit-rate
  della policy tier+AUTOPIN+prefetch > hit-rate della LRU pura misurata sugli
  STESSI budget, con delta riportato e confrontato col ceiling Belady di WP-0
  (9-19pp in scarsita'; 0.6pp a 2596 nel regime near-total).
- Floor C1 ereditato: decode >= 13.43 tok/s alla config di budget migliore
  disponibile sul device, quiescente, CON CLAUSOLA PRE-NEGOZIATA (lezione c3a
  item 2/17a — la clausola si scrive all'avvio, non a fine goal): se a
  meccanismi completi (policy, prefetch, slab ctx-aware, decode ottimistico di
  C3b attivo) e tassa entro modello il decode resta sotto 13.43 per un limite
  che il journal attribuisce a hardware/clock con la misura (pattern probe
  it.19), il goal chiude con misura + attribuzione, la leva residua va a docket
  e la decisione al PI. Il floor NON si abbassa: o si passa, o si dichiara con
  questa clausola. Ogni bench riporta comunque il gap dalla soglia UX
  (30 tok/s, ~60 thinking, TTFT <= 4 s).
- Instant-on: TTFT a freddo (OPFS popolata, VRAM vuota, router + shared expert
  + esperti caldi residenti subito e il resto on-demand) <= [ASSUMED 1.25x] il
  TTFT a caldo chiuso da C3b, con report JSON committato e il gap dal budget UX
  4 s riportato esplicitamente. NOTA: il <= 4 s ASSOLUTO del contratto
  chartered 2026-08-01 e' aritmeticamente fuori finche' il TTFT a caldo e'
  ~18 s — la ri-negoziazione del budget assoluto e' docket item 1, da chiudere
  in fase di spec.
- Non-regressione permanente (numeri da fissare all'avvio, pattern c3b item 1):
  decode, prefill e TTFT >= i valori chiusi da C3b (banda di rumore come da
  esito ratifica c3a item 14); correttezza C2 invariata (argmax == cpuref-f64
  sul campione ratificato, top-1 vs golden >= 98.828% full-corpus); Qwen >=
  baseline corrente; ktest tutti PASS; npm test verde; npx tsc --noEmit pulito.
- Chiusura: docket, direction §7 (fase C CHIUSA), ledger §A, HANDOFF refresh;
  input preparati per la hero-demo M4, che resta PI-gated per hardware.

EVIDENCE OF DONE: run del tool banda fredda con exit 0 + JSON in
results/opfs-bench/; file di spec + entry nel docket; run glmbench a ctx 6k e ai
budget slab della spec con JSON committati in results/engine/; JSON hit-rate per
policy (LRU vs tier+AUTOPIN+prefetch) agli stessi budget; JSON instant-on; test
del modello di banda verdi in npm test; bench Qwen e conformance di
non-regressione; diff di spec + docket + direction + ledger + HANDOFF.

AUTHORITY GRANTED:
- may do autonomously: come C3b (src/engine/**, tests/**, tools/**, scripts/**,
  docs/engine/** quando stale; commit/push su main a iterazione VERIFICATA;
  merge su main + push a goal CHIUSO e verificato; run locali su 4090;
  re-import e invalidazione artefatti in ~/.cache), PIU' l'implementazione di
  tier.h, AUTOPIN, pin learned, prefetch in-forward, instant-on e budget slab
  ctx-aware — qui sono l'oggetto del goal; creare tools/opfs-cold/**.
- must docket (never do): identiche a C3b (soglie del contratto, regressioni,
  modello-tesi, quant, fase D, benchmark pubblico, M4/S22, spese, delete >30gg),
  PIU': ri-negoziare il budget instant-on assoluto (docket item 1, PI);
  costruire il predictor al confine di token (riaprirebbe WP-0); rinunciare a
  uno dei tre budget slab di prova.

CONSTRAINTS: identiche a C3b (spec-first, non-regressione permanente, bench
quiescenti con hostState, albero congelato nelle run lunghe, parita' di
protocollo, llama.cpp solo oracolo, f32-first, near-tie mai gateati, telemetria
zero-overhead, determinismo, no pipe sui runner, zero attribution AI), PIU': i
numeri di banda fredda del browser NON si estrapolano dai numeri OS — si
misurano; tap hidden-states preservati (qui sono l'input del prefetch
in-forward).

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle;
digest every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md; .harness/goals/engine-fase-c3b/{GOAL,docket}.md (il
contratto che precede e i suoi numeri di chiusura);
.harness/goals/engine-fase-c3a/docket.md item 21 e item 17a;
.harness/goals/engine-fase-c2/docket.md item 8;
.harness/goals/engine-fase-c1/{GOAL,docket}.md (item 4: GO prefetch; item 5:
matrice usage ricalcolabile) e i JSON in results/engine/moe-oracle/ (recall,
residenza, wp0-replay-sim); docs/engine/study/colibri.md §2 (eusage/eheat/
AUTOPIN, tier.h) e §1 (LOOKA/PILOT); docs/engine/study/ds4.md (instant-on,
quant asimmetrica); docs/engine/ideas-ledger.md §A; docs/engine/direction.md
§1, §7, §8.3; results/opfs-bench/.
