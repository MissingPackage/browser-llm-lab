# Digests — engine-fase-c1

2026-07-30 — it.0 (setup). Goal aperto: misure sull'oracolo desktop che
dimensionano il paging esperti su GLM-4.7-Flash (metodo LOOKA di colibri,
contatori-only) — recall lookahead, skew/heat/working-set, curve hit-rate vs
budget da simulazione trace-driven. Nessun meccanismo browser: src/engine/**
è must-docket; C2 (MoE+MLA nel motore) e C3 (paging+instant-on) sono goal
futuri, hero-demo M4 PI-gated. 6 fasi sequenziali: risorse+recon → spec →
ROUTE_TRACE → LOOKA recall (timebox, fallback pre-negoziato) → residenza+
simulatore → sintesi+chiusura. Primo target: fase 1 (GGUF ~17 GB autorizzato,
build llama.cpp, smoke, recon punti di tap). Docket-born: plan-check (item 1)
— STOP prima dell'iterazione 1.

2026-07-30 — it.1 (fase 1 DONE, verifier PASS). Risorse oracolo pronte: GGUF
unsloth GLM-4.7-Flash-Q4_0 (17.2 GB, sha registrato — Q4_0 = layout del motore,
ruling PI), llama.cpp 5f55650a build CPU-only dedicata fuori repo, smoke greedy
exit 0. Throughput oracolo: pp512 56.6 / tg64 13.4 t/s (16 thread) ⇒ corpus 16k
posizioni ≈ 9 min/run. FINDING 1: zero patch necessarie — cb_eval + tensori già
nominati (ffn_moe_topk, ffn_moe_logits, ffn_norm) ⇒ rischio fase 4 ridotto.
FINDING 2 (docket item 2, PI-gated): il GGUF è arch deepseek2 e SENZA testa
NextN/MTP — implicazioni per fase D (spec-dec) e per il reader GGUF di C2.
46 layer MoE confermati dai metadati (sanity-gate fase 3 ok). Landmine nuova:
llama-cli UI chat ignora -no-cnv e loop-a su stdin chiuso — sempre -st
--simple-io nelle run scriptate. Next: fase 2 (spec C1) → ruling PI.

2026-07-30 — it.2 (fase 2 DONE, verifier PASS, STOP BY DESIGN). Spec C1 scritta
e ancorata ai numeri di it.1: tool C++ standalone via cb_eval (zero patch),
LOOKA calcolato online (pesi router ~24 MB estratti alla load, replica esatta
sigmoid→+bias→top-k — verificata dal verifier nel codice E nei tensori del
GGUF), corpus 8 prompt ≈16k posizioni (~9-12 min/run), recall@{4,6,8}
decode-only + baseline, simulatore TS+vitest con 4 policy × 6 budget, autotest
predittore ≥0.999 come gate hard dello strumento, NESSUN gate numerico sul
recall (misura, non target). RULING PI RICHIESTO: docket item 3, decisioni
(a)-(f). Fasi 3-6 gated — il goal riprende al ruling.

2026-07-31 — it.3 (fase 3 DONE, verifier PASS). ROUTE_TRACE operativo:
trace.cpp (C++ su llama.h, zero patch — checkout oracolo verificato pulito),
corpus 8 prompt congelati (hash in envelope), run canonica exit 0 in ~21 min:
31.274 posizioni routed (26.154 prefill + 5.120 decode, zero EOS anticipati),
gate ≥16k/≥4k passato con margine 2×. Finding metodologico: llama.cpp pota
l'ultimo layer alle righe di output (inp_out_ids) — il primo run è FALLITO
correttamente sul sanity ("colonne 1 != 512 su il=46"), fix = logits=true su
ogni posizione. Traccia 5.6 MB gz = input della fase 5. Next: fase 4 (LOOKA
online, timebox 3 it.).

2026-07-31 — it.4 (fase 4 DONE in 1 it. su 3, verifier PASS). LOOKA online:
pesi router letti dal GGUF (gguf.h, 46×[2048×64] f32 + bias), predizione
sigmoid(W·h)+b top-K al tap ffn_norm. Autotest (gate hard dello strumento):
self-recall 0.999997 su 5.754.416 slot attesi esatti. RISULTATO decode
(921.600 slot): recall@4 0.7746, @6 0.8769, **@8 0.9197**, baseline
"stessi expert del token precedente" 0.3228; prefill quasi identico (@8 0.9189)
⇒ non è un artefatto del regime autoregressivo. Confronto colibri/GLM-5.2:
R@8 92.0% vs 71.6% e baseline 32.3% vs 41.3% — su GLM-4.7-Flash il router
batte la persistenza temporale di 2.8× (non 1.7×): il prefetch predittivo ha
molto più margine di quanto assunto in direction §4.3. Coda bassa localizzata
nei layer 4-9 (R@4 <0.70), max il=41 0.846. Il verifier ha ricalcolato la
baseline dalla traccia in modo indipendente (identica a <1e-5). Next: fase 5
(residenza + simulatore trace-driven).

2026-07-31 — it.5 (fase 5 DONE; verifier FAIL al primo giro, corretto). Traccia
rigenerata col dump delle predizioni (routing bit-identico, verificato riga per
riga) e simulatore TS + 15 unit verdi (suite 181/181). DUE FINDING che toccano
le assunzioni di progetto: (1) la residenza NON è skewed come in ds4/colibri —
top-4 expert per layer coprono il 21.8% delle selezioni, working-set 1.663
expert unici in 32 token, copertura 2944/2944 ⇒ il learned pinning ha poco su
cui aggrapparsi, e con i parametri di spec (pin 50%) LRU batte le policy
sofisticate; (2) le sensibilità ribaltano: con pin 12.5% e K adattivo
(2/4/8 al crescere del budget) LFRU+pin+prefetch batte LRU ovunque
(736: 67.5 vs 61.4; 1472: 90.8 vs 84.7; 2208: 98.9 vs 96.4). Verdetto "modello
2× la memoria": 90.8% di hit-rate decode a cache = 50% del parco. Punto di
lavoro: il box dev è una 4090 LAPTOP da 16 GiB ⇒ 2.573 slot = 87% del parco,
modello/cache 1.14× — sul device di sviluppo il paging è quasi inoperante, il
regime vero è mobile/M4-condiviso. Caveat dichiarato: l'hit-rate non misura il
guadagno di LATENZA del prefetch (serve il modello di banda, C3). Next: fase 6
(sintesi, proposta go/no-go PILOT, chiusura).

2026-07-31 — it.6 (fase 6 DONE, GOAL CHIUSO). Checklist DONE WHEN 6/6.
Sintesi consegnata: docket item 4 = proposta go/no-go PILOT istruita ma NON
decisa (GO prefetch — recall 92% e delta ~5 ms/token sul budget da 33 ms a
30 tok/s; NO-GO learned pinning come leva primaria — skew troppo debole;
domanda aperta al PI: WP sulla banda OPFS a FREDDO prima o dopo, perché tutta
l'aritmetica del costo-miss è condizionale su quel numero). Doc allineati ai
numeri misurati: ledger §A (2 righe riscritte: paging e instant-on), §I
(2 rimandi nuovi: K adattivo per budget/layer, traccia sostituita),
direction §3 (5.33 MB/expert e 1.53 GB non-expert misurati), §5 (4090 Laptop),
§7 (fase C splittata, C1 chiusa con i risultati). HANDOFF riscritto.
Next PI-gated: docket 4, poi C2 o C3.
