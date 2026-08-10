# Docket — engine-fase-q1

10. **WP GAP — ESITO CHE CONDIZIONA LA FASE 6 (2026-08-10, it.12, fase 5;
   registrazione, la decisione di merito è già nel contratto)** — gap
   decode full-resident vs llama.cpp Vulkan STESSO GGUF: 4B 5.18×, 9B
   4.62×; scomposto: readback+sync 5.1/3.6 ms/token MISURATI (12%,
   eliminabile col pattern GLM decodeBatch), compute 4.6×/4.4× (dentro:
   dispatch 562/token stimato 15-29%, kernel non tuned, check WebGPU).
   Prefill 183×/171× = ASSENZA di batching, non gap: leva n.1. ORDINE ROI
   per la fase 6: (1) prefill batched pattern GLM, (2) decode multi-step
   no-readback, (3) fusione/riduzione dispatch, (4) tuning+dot4I8Packed
   dietro flag (da contratto), (5) spike subgroup-matrix. Le leve 1-3 sono
   PATTERN GIÀ IN CASA (GLM): la fase 6 del contratto (dot4I8Packed/
   tuning/spike) resta com'è, ma il doc registra che il ROI più alto è nel
   PORTARE i pattern GLM, materia delle fasi 7-8/D — nessun cambio di
   contratto, solo priorità informata per il PI a valle. Doc:
   docs/engine/study/2026-08-10-q35-gap-decomposition.md. Per il WRITEUP:
   pubblicare il gap a parità di regime (4.6-5.2× decode); il prefill solo
   dopo il batching (oggi misura un'assenza).

9. **SOGLIA GOLDEN 9B FISSATA — RATCHET (2026-08-10, it.11, fase 5; stesso
   regime pre-autorizzato di item 8)** — golden top-1 full-corpus del 9B su
   GPU: **1000/1024 = 97.65625%** (q35-conf-9b-2026-08-10.json vs
   golden-q35-9b-full; per-prompt 120-128/128, load 15.6 s, 42 min).
   ANALISI NEAR-TIE allegata al pin: TUTTI i 24 miss sono near-tie
   dell'oracolo — 23/24 il motore sceglie il top-2, margine mediano 0.066
   logit, max 0.557, ZERO miss con margine >1 logit (firma benigna del
   rounding f32; regola C2 near-tie rispettata). Gate 9B da qui in avanti:
   top-1 ≥ 1000/1024 AL PIN. Riferimenti full-resident 9B (stesso JSON di
   bench, host dichiarato): decode 14.55 tok/s (p50 68.7 ms), prefill seq
   15.4 tok/s, TTFT 40.3 s — frame correttezza-prima.

8. **SOGLIA GOLDEN 4B FISSATA — RATCHET (2026-08-10, it.9, fase 4; spec §5
   punto 3, pre-autorizzata dal contratto: "soglia FISSATA alla prima run
   verificata e mai più abbassata")** — golden top-1 full-corpus del 4B su
   GPU: **1012/1024 = 98.828125%** (run q35-conf-4b-2026-08-10.json vs
   golden-q35-4b-full-2026-08-10.json, provenance piena; per-prompt:
   125-128/128, load 11.3 s, 29 min totali coi prefill sequenziali).
   Da qui in avanti il gate 4B è top-1 ≥ 1012/1024 AL PIN (near-tie mai
   gateati singolarmente: la cifra è il pin, come per GLM). Nota di colore
   VERIFICABILE: la cifra coincide col PIN GLM (1012/1024) — stesso
   corpus, 1024 posizioni, aritmetica dei near-tie q4 comparabile;
   coincidenza, non copia (i 12 miss cadono su prompt diversi).

7. **IGIENE (2026-08-10, it.6 — fuori goal, non bloccante)** — golden.cpp
   (tools/oracle-moe) scrive `"arch":"deepseek2"` HARDCODED nel JSON di
   output anche per modelli qwen35: i token e i logit sono corretti (l'eval
   usa l'arch vera), è solo il campo metadata. Da parametrizzare a igiene,
   fuori dalle run del goal.

1. **PLAN-CHECK — PRE-AUTORIZZATO (2026-08-10, PI in chat: "partiamo con la
   spec e poi direttamente il loop"; pattern c3c item 3 / c3b item 4 / c3a
   item 3).** PHASES.md è su disco al tag di apertura: 9 fasi sequenziali,
   authority delta = none ovunque, nessuna docket-born, nessun
   parallel-group (GPU singola + albero congelato). Se vuoi rivederla, il
   punto giusto è PRIMA della fase 3 (le fasi 1-2 producono spec, reader e
   test di conformance — reversibili e senza kernel nuovi). La ratifica
   formale del contratto avviene DENTRO la spec di fase 1.

2. **REGISTRAZIONE (2026-08-10, it.1, fase 1 — non richiede decisione)** —
   spec depositata: `docs/superpowers/specs/2026-08-10-engine-fase-q1-design.md`.
   Ratifica formale del contratto dentro (§0). Non tocca gate né soglie ⇒
   nessuno STOP di ruling (regime C3b/C3c). SHA GGUF pinnate (spec §1),
   [VERIFY] subgroup-matrix CHIUSO (spec §7: nessun origin trial Chrome
   146-148 ago 2026, solo feature sperimentale Dawn).

3. **DEVIAZIONE QUANT MoE — REGISTRAZIONE (2026-08-10, it.1, spec §2).**
   Il Q4_0 del 35B-A3B NON esiste da fonte pulita (verificati unsloth,
   Qwen ufficiale, hub — 2026-08-10). Deciso (meccanismo): densi Q4_0,
   MoE = unsloth UD-Q4_K_S pinnato via SHA; prezzo = dequant Q4_K expert
   (macchineria K-quant già in casa per la head Q6_K); i byte restano
   identici fra motore e oracolo. Alternativa scartata: auto-quant Q4_0
   (>100 GB disco, must-docket "quant nuove", perde il GGUF maturo). Se
   preferisci l'auto-quant, è un RULING che riapre authority disco.

4. **RISOLTO (2026-08-10, it.2)** — oracolo llama.cpp b10333 (build
   08659901c) supporta la famiglia SENZA upgrade: llama-bench sul 4B
   `qwen35` carica e genera (pp16 36.1 / tg8 17.2 t/s CPU, exit 0);
   llama-tokenize funziona (protocollo fissato: `--ids --no-bos
   --no-parse-special`). Arch GGUF reali: `qwen35`/`qwen35moe`.

6. **REGISTRAZIONE (2026-08-10, it.3 — non richiede decisione)** — due
   correzioni di protocollo tokenizer scoperte dal gate secco, entrambe
   verso la fedeltà al testo RAW: (i) protocollo oracolo v2 = `--ids
   --no-bos --no-parse-special --no-escape` — senza `--no-escape`
   llama-tokenize processa gli escape (\\→\) e il riferimento non è il
   testo che il motore tokenizza (scoperto su corpus 11); fixture
   rigenerato, cross-check 4B==35B PASS, stesso totale 27 714; (ii)
   semantica special ESATTA di llama.cpp replicata (llama-vocab.cpp:3171):
   parse_special=false salta SOLO CONTROL/UNKNOWN, i 6 USER_DEFINED della
   famiglia (<tool_call>/<tool_response>/<think> + chiusure) si matchano
   SEMPRE nel partizionamento pre-BPE (scoperto su corpus 12: <think> =
   token singolo 248068 anche con --no-parse-special). Il commento nel
   corpus 12 ("trattarli da testo") è impreciso per i USER_DEFINED ma il
   file NON si tocca: è input del fixture, i byte sono congelati.

5. **REGISTRAZIONE (2026-08-10, it.2 — non richiede decisione)** — header
   dump dei 3 GGUF (`results/engine/q35-header-dump-2026-08-10.json`):
   (i) [VERIFY] spec §3 tutti chiusi, tabella aggiornata coi valori veri
   (9B non-tied, 35B head_count_kv 2 ⇒ KV 40 960 B/token, 2.6× meno di
   GLM); (ii) mix UD del 35B enumerato: expert Q4_K 117 + Q6_K 3 — il
   dequant nuovo resta SOLO Q4_K (+ classe slot Q6_K, macchineria
   esistente); (iii) densi interamente nel set type già supportato
   (Q5_K sulle proiezioni linear-attn: kquantfast si riusa); (iv) vocab
   IDENTICO su tutta la famiglia (cross-check 4B==35B nel generatore del
   fixture, 12 file / 27 714 token).
