# Manifest dello split — cosa finisce in quale repo

Derivato dall'albero REALE del 2026-08-17, non dal piano del 2026-08-09: quel
piano lasciava tre buchi (vedi §4) e li chiude questo file. È l'input di
`git filter-repo`, e siccome definisce **cosa diventa pubblico** va letto prima
di eseguire, non dopo.

Nomi decisi (docket item 20): il motore è **`webgguf`**, sotto `MissingPackage/`.

---

## 1. `webgguf` — il motore

### Sorgente e strumenti
    src/engine/                 60 file  — il motore
    src/microbench/                      — i microbanchi tt*/kd* sono del motore
                                           (§7.2: la chiusura degli import li pretende)
    src/{metrics,probe,quality,qualityPrompts,schema}.ts
                                         — ZONA CONDIVISA col bench (§7.2)
    tools/                      39 file  — oracle-moe, generatori, diagnostica, piu'
                                           tools/harness/ (5): engine-ktest,
                                           engine-prof, opfs-bench, dispatch-profile,
                                           submit-callsites — vedi §7.3

### Documentazione  ⚠️ DA TRADURRE IN INGLESE (docket item 16)
    docs/engine/                21 file  — consuntivi, spec, ledger delle idee
    docs/deep-dive/             17 file  — CORRETTO il 2026-08-18: il manifest li
                                           dava al BENCH. Sono tutti ricerca del
                                           MOTORE (ogni file rimanda a un goal
                                           engine-*, o parla di kernel/dequant/
                                           KV/limiti). Nessuno riguarda l'app.
    docs/architettura/           6 file  — MECCANISMI, QUANTIZZAZIONE, VALUTAZIONE
                                           + i due file del diagramma
    GLOSSARY.md

### Prove
    results/engine/            323 file  — i banchi del motore
    results/opfs-bench/          4 file  — il cold-read è materia motore (piano §3)
    results/chat/               15 file  — le run di chat: è QUI che vive il 34,97 tok/s
    results/eval/               13 file  — corpus e qualità dei quant (+0,13 bit/token)
    results/microbench/                  — CORRETTO il 2026-08-18: idem. Contiene
                                           costm-decode-*, kernel-decode-fase0-*,
                                           kquant-fase0-*: artefatti di goal engine

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
    results/{4090-linux-*, m4-pro*, s22-ultra*, dispatch-profile, methodology}
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

    .harness/          121 file  — journal, docket, GOAL/PHASES. Erano 126: i 5
                                   di `.harness/tools/` sono usciti, perche' non
                                   erano processo (§7.3)
    .claude/                     — worktree (rimossi il 2026-08-17), config
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

---

## 7. COSA HA TROVATO L'ESECUZIONE, il 2026-08-17

Il manifest sopra è stato scritto leggendo l'albero. Eseguirlo ha trovato **tre
difetti che leggerlo non poteva trovare** — ognuno avrebbe prodotto un repo che
sembrava completo e non lo era.

### 7.1 Il typecheck dipendeva da un pacchetto del concorrente

Tolte le tre dipendenze runtime (`@mlc-ai/web-llm`, `@huggingface/transformers`,
`@wllama/wllama` — tutte e tre degli adapter del bench), `tsc` è passato da 0 a
**295 errori**, tutti su `node:fs`, `process`, `Buffer`.

Causa: `@wllama/wllama/esm/wllama.d.ts` contiene `/// <reference types="node" />`.
Il typecheck del MOTORE stava ereditando i tipi globali di node attraverso il
binding WASM di un concorrente, presente solo perché serviva all'app di benchmark.
`tsconfig.json` non li dichiarava.

Chiuso dichiarandoli: `"types": [..., "node"]` più `@types/node` esplicito fra le
devDependencies, con il perché scritto nel tsconfig. **Il motore resta a ZERO
dipendenze runtime** — verificato: non importa un solo pacchetto.

### 7.2 Il manifest amputava `src/microbench/` e cinque moduli della radice

Quattro test del motore importano `src/microbench/tt*` e `kd*` — i microbanchi dei
goal TTFT e kernel-decode, che importano da `../engine/`. Sono lavoro del motore,
e il manifest li aveva lasciati al bench.

Non basta aggiungerli: la **chiusura transitiva** degli import (calcolata
meccanicamente, non a occhio) pretende anche `src/{metrics,probe,quality,
qualityPrompts,schema}.ts`. Sono ZONA CONDIVISA fra i due repo, come già previsto
dal piano §3 per `scripts/lib/hoststate.mjs`.

**Regola che ne discende**: prima di uno split si calcola la chiusura degli
import, non si classifica per cartella. Due giri di `tsc` me l'hanno insegnato.

### 7.3 Cinque strumenti del MOTORE vivevano in `.harness/`

Il difetto peggiore, perché silenzioso: la regola «`.harness/` è processo, resta
privato» avrebbe buttato via `engine-ktest.mjs` (il driver della conformance dei
kernel), `engine-prof.mjs` (il profiler esterno del first light), `opfs-bench.mjs`
(il cui `results/opfs-bench/` il manifest assegnava GIÀ al motore),
`dispatch-profile.mjs` e `submit-callsites.mjs`.

Se ne sono accorti quattro test, nel repo estratto, con `ENOENT`. Corretto ALLA
RADICE nel lab (`git mv .harness/tools tools/harness`, 18 file aggiornati),
non aggirato nello split.

### 7.4 Lo stato verificato del repo estratto

    commit   359          (dai 693 del lab)
    file     701
    .git     20 MB
    tsc      exit 0
    vitest   1237 passed | 11 skipped | 0 FAILED

Assenti e verificati tali: `.harness/` (journal, docket, GOAL), `.claude/`,
`docs/superpowers/`, `docs/publishing/`, `docs/deep-dive/`, `src/adapters/`,
`src/main.ts`, `index.html`, `HANDOFF.md`, i quattro script del bench.

Nuovi: `LICENSE` (Apache-2.0), `NOTICE` (nessun peso ridistribuito),
`README.md`, `docs/RESUMING.md` (ripresa su un'altra macchina, in inglese).

### 7.5 Cosa resta

- **Il push è bloccato da un guasto GitHub** («Partial System Outage», HTTP 503
  sull'API). Il repo locale è completo in `~/Projects/webgguf`; `gh repo create
  webgguf --private --source=.` e `git push -u origin main` bastano al rientro.
- **La traduzione non è fatta**: commenti del sorgente e `docs/` sono in italiano.
  README, NOTICE e RESUMING sono in inglese. Il README lo dichiara.
- `webgguf-bench` e `webgguf-paper` NON sono stati estratti (§2, §3).

---

## 8. QUARTO DIFETTO, trovato il 2026-08-18 sincronizzando

Il manifest assegnava **`docs/deep-dive/` e `results/microbench/` al repo BENCH**.
Sono entrambi del **MOTORE**, e la verifica è banale una volta fatta: dei 17
documenti in `deep-dive/`, ognuno rimanda a un goal `engine-*` o parla di
kernel/dequant/KV/limiti di buffer — **nessuno riguarda l'app di benchmark**. E
`results/microbench/` contiene `costm-decode-*`, `kernel-decode-fase0-*`,
`kquant-fase0-*`, cioè artefatti di goal engine.

Se lo split fosse andato in produzione così, il repo motore sarebbe nato **senza
il proprio registro di ricerca** — inclusi il roofline (`headroom`), i due
prereg/memo della ricerca `cost(M)` e il micro-bench che misura la banda della
scheda. Amputazione silenziosa, come quella dei cinque strumenti in `.harness/`
(§7.3): stessa classe di errore, scoperta dallo stesso meccanismo — provare a
usare davvero il repo estratto.
