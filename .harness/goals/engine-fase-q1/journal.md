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

## it.2 seguito (2026-08-10) — download DONE, misure a valle DONE

- SHA: `node scripts/q35-verify-sha.mjs` → 3× PASS, exit 0 (done-when
  punto 1 di fase 2 CHIUSO).
- Header dump committato (`results/engine/q35-header-dump-2026-08-10.json`):
  [VERIFY] spec §3 CHIUSI (tabella aggiornata), mix UD enumerato (expert
  Q4_K 117 + Q6_K 3), densi nel set type già supportato, KV 35B
  40 960 B/token. Docket item 5.
- Oracolo b10333: supporto famiglia PROVATO (llama-bench 4B pp16 36.1 /
  tg8 17.2 CPU; llama-tokenize ok). Docket item 4 RISOLTO, no upgrade.
  Nota: llama-cli -no-cnv si impianta (probabile stdin) — llama-bench e
  llama-tokenize sono gli strumenti del protocollo, nessun impatto.
- Fixture tokenizer: `tests/fixtures/q35-tok-oracle.json` via
  `scripts/q35-tok-oracle-gen.mjs` — 12 file, 27 714 token, protocollo
  `--ids --no-bos --no-parse-special`, cross-check 4B==35B PASS in
  generazione (vocab di famiglia identico).
- RESTANO per chiudere fase 2 (it.3): reader TS `qwen35`/`qwen35moe`
  (load test sui 3), tokenizer TS con id==fixture in npm test, tsc.

## it.3 (2026-08-10) — fase 2 DONE (verifier PASS)

- Inventario nomi-tensori completo dei 3 file (q35-header-dump esteso ai
  nomi): qkv FUSA sui layer DeltaNet [d,(2·nK+nV)·hd], attn_q dei full =
  q+gate fusi [d,2·nHead·256] (attn_output_gate), QK-norm per-head, pattern
  UD sui down_exps (Q4_1 su 4 layer del 4B, Q6_K su 3 del 35B).
- `src/engine/q35shape.ts`: shape DERIVATA dai metadata + validazione hard
  dell'inventario completo (dims calcolate, allow-list tipi chiusa, throw su
  tensori extra) — la parametrizzazione di spec §3, primo pezzo. gguf.ts:
  aggiunta SOLO additiva Q4_K (superblocco 144 B).
- `src/engine/q35tokenizer.ts`: PRIMO tokenizer in-engine — BPE byte-level
  GPT-2, regex pre `qwen35` verbatim da llama.cpp (verifier: riscontrata
  IDENTICA nel binario b10333 via strings), partizionamento special fedele
  (USER_DEFINED sempre, CONTROL solo con parseSpecial — llama-vocab.cpp:3171).
- DUE correzioni di protocollo scoperte dal gate secco (docket item 6):
  `--no-escape` obbligatorio (corpus 11: \\ processato) e semantica
  USER_DEFINED (corpus 12: <think> = token singolo). Fixture v2 rigenerato,
  cross-check 4B==35B PASS. Il gate secco ha pagato: 2 bug di fedeltà presi
  PRIMA di toccare i kernel.
- Gate: SHA 3/3 exit 0; shape test 4/4 (426/427/733 tensori); tokenizer
  12 file id-identici + roundtrip decode; suite 364 PASS (+5) | 7 skip;
  tsc pulito. Verifier PASS con spot-check oracolo indipendente.
- Next: it.4 = fase 3 (kernel DeltaNet WGSL kernel-level vs cpuref-f64;
  prima il cpuref TS della catena linear-attn, spec §4).

## it.4 (2026-08-10) — fase 3 prima metà: cpuref-f64 DeltaNet (verifier PASS)

- Semantica presa dalla FONTE pinnata alla build dell'oracolo (llama.cpp
  b10333): qwen35.cpp (build_layer_attn_linear), delta-net-base.cpp (path
  AUTOREGRESSIVO = la ricorrenza; il chunked è l'ottimizzazione della stessa
  matematica), ops.cpp (ssm_conv = dot k4 su concat(stato,x); l2_norm con
  eps a FLOOR: 1/max(√Σx²,eps)), ggml-impl.h (softplus x>20?x:log1p(eˣ)).
- Ordine ESATTO della ricorrenza fissato nel cpuref: decay → lettura sk →
  delta β(v−sk) → update S → output S·q. Broadcast k-head→v-head = h mod nK
  (ggml_repeat TILA, non raggruppa); q/√hd; gated norm = RMS(o)·silu(z).
- `src/engine/q35cpuref.ts`: catena f64 completa con tap intermedi esposti
  per il ktest (qkvPreConv, convOut, q/kNorm, oCore, gated); core esportato
  puro (deltaNetStepCore).
- Test (7 PASS): 5 identità algebriche in aritmetica esatta (fit β=1 ⇒
  o(q=k)=v/√hd; ortogonalità non disturba; β=0 stato invariato; g→−∞ wipe;
  discriminante dell'ORDINE decay-prima-della-lettura) + softplus al gomito
  + campione sintetico T=12 pinnato su fixture (helper condiviso, generatore
  esplicito scripts/q35-deltanet-fixture-gen.mjs — niente auto-pin).
- Suite 371 PASS (+7) / 7 skip, tsc pulito, GLM intatto.
- RESTA per chiudere fase 3 (it.5): kernel WGSL DeltaNet + ktest vs cpuref
  sul campione (+ pesi reali 4B layer-scale), ktest 69/69 invariati.
