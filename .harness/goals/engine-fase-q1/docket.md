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

4. **REGISTRAZIONE CONDIZIONALE (2026-08-10, it.1)** — oracolo llama.cpp:
   il supporto `qwen3_5(_moe)` del binario locale b10333 si verifica al
   primo run di fase 2; se servisse un upgrade, build nuova PINNATA e
   dichiarata, stesso protocollo (registrazione, non ruling).
