# Manifest dello split — cosa finisce in quale repo

Derivato dall'albero REALE del 2026-08-18, non dal piano del 2026-08-09: quel
piano lasciava tre buchi (vedi §4) e li chiude questo file. È l'input di
`git filter-repo`, e siccome definisce **cosa diventa pubblico** va letto prima
di eseguire, non dopo.

Nomi decisi (docket item 20): il motore è **`webgguf`**, sotto `MissingPackage/`.

---

## 1. `webgguf` — il motore

### Sorgente e strumenti
    src/engine/                 60 file  — il motore
    tools/                      34 file  — oracle-moe, generatori, diagnostica

### Documentazione  ⚠️ DA TRADURRE IN INGLESE (docket item 16)
    docs/engine/                21 file  — consuntivi, spec, ledger delle idee
    docs/architettura/           6 file  — MECCANISMI, QUANTIZZAZIONE, VALUTAZIONE
                                           + i due file del diagramma
    GLOSSARY.md

### Prove
    results/engine/            323 file  — i banchi del motore
    results/opfs-bench/          4 file  — il cold-read è materia motore (piano §3)
    results/chat/               15 file  — le run di chat: è QUI che vive il 34,97 tok/s
    results/eval/               13 file  — corpus e qualità dei quant (+0,13 bit/token)

### Test — 66 `tests/engine-*` più questi 19, verificati per import
    tests/fixtures/  tests/helpers/  tests/types/
    tests/analysis-conf-cpuref     tests/analysis-q3k-ladder
    tests/analysis-q3k-roundtrip   tests/analysis-route-cpuref
    tests/glmsource                tests/gpudevice          tests/gpulimits
    tests/telemetry                tests/slabfile           tests/quant-repack-fast
    tests/q35-pack-microbench      tests/q35-slab-parity
    tests/ttattn-prefill-prodvslegacy   tests/ttkquant-fase0-varianti
    tests/ttprobe
    tests/oracle-moe-sim   tests/oracle-moe-wp0   ← importano da `tools/`, non da
                                                    `src/engine`: il classificatore
                                                    per import li aveva SBAGLIATI
    tests/quant-paired-stats   ← importa `scripts/lib/pairedstats.mjs`
    tests/chat-conversazione   ← sorveglia `chat-smoke.mjs` vs `chat-llamacpp.py`

### Script — tutti tranne i quattro del bench (§2)
    q35-*  glm-*  kd-*  tt-*  ktest-run  engine-bench  conformance-engine
    attn-bench  chat-smoke  chat-llamacpp  chat-transcript  decode-attrib
    kernel-diag  prefill-diag  quant-*  seq-bench  token-identity  vram-ceiling
    webgpu-limits  webgpu-subgroup-matrix-probe  kv-rollback  prefix-cache
    gen-*  build-ttft-checkpoint  transcript-audit  q35-manifest.json  lib/*

### Pagine dell'harness motore
    engine.html  ktest.html  chat.html  q35conf.html  kdbench.html  ttbench.html
    glmbench.html  glmconf.html  glmprefill.html  glmroute.html

### Nuovi, da scrivere allo split
    LICENSE      Apache-2.0 integrale (docket item 14)
    NOTICE       non ridistribuiamo pesi; i GGUF sono di terzi (Qwen Apache-2.0,
                 quant bartowski/unsloth ripacchettati da loro)
    README.md    ⚠️ apre sul **35B MoE a 34,97 tok/s in una scheda**, NON su
                 «run GGUF in your browser» — quello è lo scaffale dei 18 webui
                 (docket item 20, sezione sul vicinato)
    package.json riscritto: name `webgguf`, license, exports pubblici

---

## 2. `webgguf-bench` — l'app di benchmark (resta in STANDBY, non si pubblica)

    src/ (radice)      main, render, schema, metrics, quality, protocol,
                       promptset, qualityPrompts, stacks, probe, benchServer,
                       bench.worker
    src/adapters/  src/conformance/  src/microbench/  src/prof/
    docs/deep-dive/                    13 file
    results/{4090-linux-*, m4-pro*, s22-ultra*, microbench, dispatch-profile,
             methodology}
    tests/  benchServer, metrics, microbench, probe, prof, protocol, quality,
            render, schema, smoke, telemetry-adapters, transformersjs-adapter,
            webllm-adapter, wllama-adapter
    scripts/  e2e-bench.mjs  bench-chrome.sh  conformance.mjs  wp0-compare.mjs
    index.html  microbench.html  prof.html  conformance.html

---

## 3. `webgguf-paper` — **NON SPLITTABILE OGGI**

Il piano del 2026-08-09 diceva: «`paper/**` nasce ORA nel lab proprio perché
filter-repo lo estragga poi con la sua storia». **Non è mai nato**: `git ls-files
paper` ritorna 0 file. Esiste solo `docs/publishing/paper-contract-draft.md`.

Quindi: o il paper si scrive prima nel lab sotto `paper/` (e lo si estrae dopo,
come previsto), o il terzo repo nasce vuoto e senza storia — che è esattamente
ciò che il piano voleva evitare. **Non lo estraggo.**

---

## 4. FUORI DA TUTTI E TRE — il processo resta privato

    .harness/          126 file  — journal, docket, GOAL/PHASES
    .claude/                     — worktree (rimossi il 2026-08-18), config
    docs/superpowers/   18 file  — spec del workflow SDD
    docs/publishing/             — questo file e il piano: sono interni
    HANDOFF.md

## 5. I tre buchi del piano del 2026-08-09, e come sono chiusi

1. **`docs/architettura/` non era assegnato a nessun repo.** Sono MECCANISMI,
   QUANTIZZAZIONE e VALUTAZIONE — cioè la documentazione che l'HANDOFF ordina di
   leggere prima di progettare e prima di credere a un numero. → **motore**.
2. **`results/chat/` e `results/eval/` non erano assegnati** (28 file). Sono la
   prova del 34,97 tok/s e del costo di +0,13 bit/token: senza, il README del
   motore afferma numeri che il repo non contiene. → **motore**.
3. **`paper/` non esiste.** → §3.

## 6. La regola che nessuna lista può catturare

Il classificatore meccanico «chi importa `src/engine` è del motore» ha
**sbagliato due file su trentadue** (`oracle-moe-sim`, `oracle-moe-wp0`:
importano da `tools/`, che è del motore, non da `src/engine`). Il manifest sopra
è il risultato del passaggio a mano, non del proxy. Se si aggiungono test prima
di eseguire lo split, vanno classificati a mano allo stesso modo.
