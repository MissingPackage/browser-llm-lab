# Journal — engine-fase-q1

## it.0 (2026-08-10) — goal-setup

- Contratto v1 chartered e APERTO (PI: "partiamo con la spec e poi
  direttamente il loop"); tag `goal-engine-fase-q1-start` su 8bbd56e.
- Emendamenti PI assorbiti nel contratto: terza taglia densa = 4B
  mobile-target (2B = eventuale futuro, fuori goal).
- PHASES.md: 9 fasi sequenziali (spec → reader/tokenizer → kernel DeltaNet
  kernel-level → 4B e2e → 9B+WP gap → leve bounded → 35B parametrizzato →
  tier/recall/bandmodel → chiusura), authority delta none ovunque,
  nessun parallel-group (GPU singola, albero congelato nelle run).
- Docket item 1: plan-check pre-autorizzato; punto di revisione naturale =
  prima della fase 3.
- Next: it.1 = fase 1 (spec + ratifica).

## it.1 (2026-08-10) — fase 1 DONE (verifier PASS)

- Spec depositata: `docs/superpowers/specs/2026-08-10-engine-fase-q1-design.md`
  (9 sezioni); ratifica formale del contratto in §0 (emendamento 4B incluso).
- SHA GGUF pinnate e cross-verificate sui pointer LFS (verifier): 4B Q4_0
  298fcb5f… (2.58 GB), 9B Q4_0 17670346… (5.38 GB), 35B UD-Q4_K_S
  a8138f18… (20.9 GB). Disco ~29 GB ≤ 60 authority.
- FINDING carico-portante: il Q4_0 del 35B-A3B non esiste da fonte pulita
  (unsloth/Qwen/hub verificati) → deviazione quant DICHIARATA (spec §2,
  docket item 3): MoE = UD-Q4_K_S, prezzo = dequant Q4_K expert; byte
  identici motore/oracolo preservati. Auto-quant scartata (>100 GB, must-
  docket).
- Config 4B verificata da HF (32 layer, 8 full, hidden 2560, 16/4 head,
  tie embeddings TRUE); [VERIFY] residui (9B/35B dettagli) si chiudono in
  fase 2 dal header GGUF.
- [VERIFY] llamaweb CHIUSO (spec §7): subgroups shipped Chrome 134;
  subgroup-matrix in standardizzazione, NESSUN origin trial Chrome 146-148
  (ago 2026), solo Dawn sperimentale → fase 6 = spike empirico sul flag.
- Verifier: PASS su tutti i punti del done-when, SHA cross-check incluse;
  drift none, violazioni none.
- Next: it.2 = fase 2 (reader GGUF + tokenizer; download pinnati, enum
  type del file UD, verifica supporto oracolo b10333).

## it.2 (2026-08-10) — fase 2 IN CORSO, checkpoint strumentazione (c59c76c)

- Download dei 3 GGUF pinnati in background (~29 GB, hf CLI, sequenziale
  4B→9B→35B in ~/.cache/blab-models/q35/).
- Strumentazione committata E validata: q35-manifest.json (SHA spec §1);
  q35-verify-sha.mjs (gate secco byte+hash, FAIL pulito su assenti
  VERIFICATO); q35-header-dump.py (VALIDATO su GLM: 844 tensori,
  deepseek2, 47 block — chiuderà i [VERIFY] spec §3 + inventario UD);
  corpus-tok/09-12 (emoji/CJK/ZWJ, whitespace, numeri/code, special
  adversariali; 01-08 GLM riusati per il regime).
- Al risveglio (notifica download): verify-sha (exit 0), header-dump sui
  3 file, verifica supporto qwen3_5 dell'oracolo b10333 (llama-tokenize),
  poi reader+tokenizer (grosso di fase 2, it.3).
- Oracolo localizzato: ~/.cache/llamacpp-vulkan/llama-b10333/ (include
  llama-tokenize, llama-quantize, llama-perplexity).
