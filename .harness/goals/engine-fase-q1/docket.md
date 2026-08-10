# Docket — engine-fase-q1

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
