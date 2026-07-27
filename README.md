# browser-llm-lab

Fino a dove ci si può spingere per servire un LLM interamente nel browser?
Harness di benchmark (backend × modello × quant × device) + deep-dive + hero-demo.
Design completo: [`docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md`](docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md).

## Quick start

```bash
npm install
npm run dev        # http://localhost:5173 — serve COOP/COEP (crossOriginIsolated)
npm test           # unit suite (vitest), nessuna GPU richiesta
```

Nella pagina: il **probe box** mostra l'adapter WebGPU reale visto dal worker
(vendor, `maxStorageBufferBindingSize`, …) — è la riga-dati #0. **Run bench**
scarica il modello (prima volta), esegue prefill 512-tok + 256 tok greedy e
mostra load/TTFT/tok-s. **Export JSON** scarica il run file (schema v3) da
salvare in `results/`.

**Device label**: campo di testo in cima, obbligatorio di fatto — è l'unica cosa che lega un run
all'hardware su cui è girato (il probe dice cosa ha visto il browser, non su che macchina sei).
Viene ricordata per-origine (`localStorage`), quindi la scrivi una volta per device e sopravvive
alle ricariche. Se la lasci vuota il file esce come `unknown-device`: **meglio un'etichetta
palesemente assente che una sbagliata** — un `unknown-device` si nota, un run del telefono che si
dichiara `4090-linux` inquina i confronti cross-device in silenzio (successo davvero, 2026-07-27).

## Run automatizzato (Playwright)

```bash
node scripts/e2e-bench.mjs                        # headless (finisce su SwiftShader: solo smoke)
HEADED=1 DEVICE_LABEL=4090-linux node scripts/e2e-bench.mjs               # GPU reale (serve un display)
HEADED=1 DEVICE_LABEL=4090-linux MODEL_ID=Qwen2.5-0.5B-Instruct-q4f32_1-MLC QUANT=q4f32_1 node scripts/e2e-bench.mjs
```

- Il driver **si rifiuta di produrre un risultato** se l'adapter è un software
  rasterizer (SwiftShader): un run CPU etichettato come GPU sarebbe un dato falso.
- Profilo browser persistente in `/tmp/blab-e2e-profile` (override: `E2E_PROFILE`):
  è ciò che rende reale la distinzione **cold vs warm** tra run consecutivi.
- Flag per la GPU su Linux/NVIDIA: `--ignore-gpu-blocklist` (la blocklist è l'unico ostacolo
  lato Chrome) **più `--disable-gpu-sandbox` nei lanci nativi**: su Fedora/NVIDIA il sandbox
  del processo GPU nega gli ICD Vulkan a Dawn ("Found no drivers") — riprodotto col toggle
  del sandbox. I run playwright non lo soffrono solo perché playwright è no-sandbox di default.
  Driver parametrizzabile: `CHANNEL=chrome CHROME_ARGS="--ignore-gpu-blocklist"`.
- Altri env del driver: `BROWSER=firefox` (run Firefox, effimero/solo-cold),
  `ALLOW_UNVERIFIED=1` (procede quando il vendor è nascosto, es. Firefox — la verifica
  hardware va fatta fuori banda: nvidia-smi o about:support), `E2E_PROFILE=<dir>`,
  `DEVICE_LABEL=<label>` — **da passare sempre**, default `unknown-device`. Non ha un default che
  nomini una macchina precisa di proposito: sarebbe lo stesso difetto della costante cablata, solo
  spostato nel driver (con `ALLOW_UNVERIFIED=1` su hardware non-NVIDIA produrrebbe un file che si
  dichiara 4090). Sulla 4090: `DEVICE_LABEL=4090-linux node scripts/e2e-bench.mjs`.
- **Sequenze multi-cella**: `scripts/seq-bench.mjs` esegue più celle nella stessa sessione
  browser e logga i contatori `nvidia-smi` a ogni confine di cella — serve a misurare la
  dipendenza dall'ordine dei run, che una cella sola non può mostrare. Stessi env di
  `e2e-bench.mjs`, più `SEQ` (default `transformersjs,webllm,transformersjs,webllm`):
  ```bash
  SEQ=transformersjs,transformersjs,transformersjs \
    HEADED=1 CHANNEL=chrome CHROME_ARGS="--ignore-gpu-blocklist --disable-gpu-sandbox" \
    node scripts/seq-bench.mjs
  ```
  Run archiviati in `results/methodology/` (evidenza del protocollo di misura, non benchmark).
- **Protocollo di misura** (`WarmupPolicy` in `src/benchServer.ts`): prima delle repliche
  misurate si esegue una generazione di riscaldamento che viene **scartata**. Tre modalità, e
  **le prime due misurano cose diverse: non vanno confrontate fra loro**.
  - `always` (default) — riscalda sempre. Misura lo **steady state**: è ciò che serve per un
    benchmark comparabile. Senza, la prima cella di ogni sessione browser è ~7–14% più veloce
    delle successive e il confronto cross-stack dipende dall'ordine dei run. Residuo dichiarato:
    ~3% di dipendenza dall'ordine anche con il riscaldamento attivo.
  - `never` — nessun riscaldamento. È ciò che un utente vero sperimenta **al primo colpo**, che
    è l'informazione utile per capire come si comporterebbe un'applicazione reale.
  - `cold-only` — riscalda solo a cache fredda.

  Selezionabile per singolo run (`MainToWorker.bench.warmup`). Ogni cella registra la politica
  applicata in `BenchCell.protocol` (`{ warmupPolicy, warmupApplied, replicateCount }`, schema
  v3) — non più in `anomalies`, perché un warm-up applicato è il protocollo che funziona come
  previsto, non un'anomalia.
- Bench manuali sul Chrome branded: `scripts/bench-chrome.sh` (profilo dedicato `blab-bench`).
  **Mai** impostare i flag Vulkan in `chrome://flags` del profilo quotidiano: `enable-vulkan`
  corrompe il compositing su NVIDIA/Wayland, `force-enable-webgpu-interop` crasha all'avvio.
- Chrome 150 branded (launcher `bench-chrome.sh`): cold 61 s / warm 1.8 s, fino a **116.9 tok/s**
  (run manuale utente) — in linea con chromium-playwright; la varianza run-to-run (~10-25%)
  conferma la necessità di repliche multiple (introdotte in Fase 1b — fondamenta, vedi sotto).

## Run da telefono / altro device sulla LAN

Il dev server gira **sulla macchina con Node**, il telefono fa solo da client Chrome — non si
lancia `npm run dev` sul telefono.

```bash
npm run dev -- --host      # bind su tutte le interfacce; vite stampa l'URL "Network:"
```

Poi da Chrome sul device: `http://<IP-LAN-della-macchina>:5173`, stessa rete Wi-Fi.

**Il nodo è il secure context.** `http://<IP>:5173` non è HTTPS, quindi niente
`crossOriginIsolated` → niente WebGPU e niente `SharedArrayBuffer`, malgrado i header COOP/COEP
siano serviti correttamente. Per i test: su Chrome del device apri
`chrome://flags/#unsafely-treat-insecure-origin-as-secure`, aggiungi **esattamente** l'origine
(`http://192.168.x.y:5173`) e riavvia il browser. L'IP cambia se il DHCP lo riassegna: allora la
flag va aggiornata, altrimenti WebGPU sparisce senza dire perché.

Prima di fidarti di un numero, guarda il probe box: `webgpu: true` **con un vendor reale**. Il
progetto ha già visto un silent-fallback a software rasterizer che dichiarava `webgpu: true`
(Firefox/llvmpipe, vedi sotto) — su mobile vale la stessa cautela.

## Cosa abbiamo verificato dal vivo (4090 mobile, Fedora, chromium Playwright)

- **WebGPU su Linux/NVIDIA funziona** nel worker con i flag sopra (vendor
  `nvidia`, arch `lovelace`). Headless shell → SwiftShader, inutilizzabile per dati.
- **`maxStorageBufferBindingSize` ≈ 2 GiB** anche su una GPU da 16 GB: il muro
  per-buffer previsto dalla spec, confermato. `maxBufferSize` ≈ 4 GiB.
- **Feature e limiti WebGPU sono per-browser (e per-versione), non per-GPU** — la
  stessa 4090, tre stack:
  | | chromium (playwright) | Firefox 152 release | Firefox (playwright, nightly-based) |
  |---|---|---|---|
  | adapter reale | NVIDIA (lovelace) | **llvmpipe = CPU!** (verificato via about:support: NVIDIA inattiva, "acceleration blocked by platform") | NVIDIA (GPU@100% verif. nvidia-smi) |
  | `shader-f16` | **assente** → `q4f16_1` crasha (`ShaderModule` invalid) — confermato anche su **Chrome branded 150** coi flag Vulkan e persino con `--enable-webgpu-developer-features`: è Dawn/driver, non la build | presente | presente |
  | `maxStorageBufferBindingSize` | 2 GiB | 128 MiB (limite llvmpipe) | 1 GiB |
  | decode 0.5B | 106–118 tok/s (`q4f32_1`) | 1.8 tok/s (`q4f16_1`, **CPU**) | **9.9 tok/s** (`q4f16_1`) |
  **Finding chiave**: Firefox può fare **silent fallback a software rasterizer**
  riportando `webgpu: true` con vendor vuoto — l'utente non ha modo di accorgersene
  dalla pagina. Il probe ora lo rileva (fingerprint: cap 128 MiB + vendor vuoto,
  vedi Fase 1b — fondamenta sotto). Il gap GPU-vero: chromium ~110 vs Firefox ~10 tok/s (≈11×, plausibile
  ruolo di `subgroups`, assente su Firefox).
- **COEP `require-corp` convive col CDN HF**: shard scaricati senza bisogno del
  fallback `credentialless` (vale anche su Firefox).
- Primi numeri (Qwen2.5-0.5B `q4f32_1`, schema v1, in `results/`):
  load cold ~56 s → **warm ~1.6 s** (Cache API); TTFT 0.5–0.9 s (warm–cold);
  decode **~106–118 tok/s** (varianza run-to-run ~10%, repliche multiple ora in Fase 1b — fondamenta).

## Fase 1b — fondamenta (schema v2)

- **Schema v2** (`SCHEMA_VERSION = 2`, non retro-compatibile con i file v1 in `results/`,
  che restano storici): `DeviceProbe` guadagna `browser` (nome/versione parsati dalla UA),
  `features` (elenco `adapter.features`, es. `shader-f16`), `anomalies` (flag rilevati dal probe).
  `BenchCell` guadagna `replicates` (le repliche grezze) e `anomalies` (flag per-cella);
  `gen` non è più una singola misura ma un aggregato `{ mean, stdev, samples }` per metrica.
- **Repliche multiple**: ogni cella esegue 3 `generate()` sullo stesso modello già caricato
  (nessun ricaricamento tra repliche) e aggrega tok/s, TTFT, tempo totale con media e
  deviazione standard. Una cella con `stdev/mean > 0.15` sul tok/s riceve l'anomalia
  `high-variance` — risponde al finding "varianza run-to-run ~10-25%" osservato in 1a.
- **Rilevazione software-adapter**: il probe marca `anomalies: ["software-adapter: ..."]`
  quando l'adapter dichiara `vendor` vuoto **e** `maxBufferSize <= 128 MiB` — la firma
  osservata su Firefox 152 in silent-fallback a llvmpipe (vedi sopra). Un run con questa
  anomalia è un datapoint CPU, non GPU: va escluso da confronti tok/s cross-device.
- **Fuori scope qui** (piano successivo "1b — matrice"): adapter Transformers.js/wllama,
  sweep sui 3 device, modulo qualità-leggera.

## Fase 1b — matrice: stack supportati

Tre stack dietro la stessa interfaccia `InferenceAdapter`, selezionabili da UI:

| stack | runtime | modello di riferimento (Tiny) | quant |
|---|---|---|---|
| `webllm` | MLC / WebGPU | `Qwen2.5-0.5B-Instruct-q4f32_1-MLC` | q4f32_1 |
| `transformersjs` | ONNX Runtime Web / WebGPU | `onnx-community/Qwen2.5-0.5B-Instruct` | q4 |
| `wllama` | llama.cpp → WASM (**CPU**) | `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf` | Q4_K_M |

Per wllama il `modelId` è `<owner>/<repo>/<file.gguf>`, con il file GGUF **nominato per esteso**:
`loadModelFromHF({quant})` farebbe fallback silenzioso (Q4_K_M → Q8_0 → non quantizzato), e una
cella etichettata `Q4_K_M` potrebbe contenere una misura Q8_0 senza che nulla lo dica.

Prima misura sulla 4090 (Qwen2.5-0.5B, stesso prompt, 3 repliche, warm-up scartato): webllm ~110
tok/s, transformersjs ~46–48, **wllama ~26** — wllama gira su CPU via WASM, non su GPU, quindi il
confronto dice quanto costa non avere accelerazione, non che la libreria sia lenta. Il suo TTFT
(~8.4 s) è dominato dal prefill dei 512 token del prompt su CPU.

**wllama dentro un Web Worker richiede uno shim**: `absoluteUrl()` di wllama risolve i path con
`document.baseURI`, che nel worker non esiste (`ReferenceError: document is not defined`, e la
cella di bench fallisce interamente). `ensureWorkerDocumentShim()` in `src/adapters/wllama.ts`
definisce `document.baseURI = self.location.href` prima di istanziare `Wllama`. È sicuro perché
`document.baseURI` è l'unico uso di `document` nel sorgente di wllama 3.5.1. Il conformance
harness non lo esercita — gira nel main thread, dove `document` c'è.

### Difetto noto di wllama 3.5.1 — l'ultimo token di ogni risposta arriva in ritardo

`npm run test:conformance` dà **wllama 7/8** (gli altri due 8/8): il check di determinismo del
conteggio token fallisce, 16 token al primo `generate()` e 15 al secondo.

Non è l'adapter: wllama interrompe il polling dei risultati quando `has_more` è `false` anche se
quella risposta conteneva ancora dati, così la **coda di ogni risposta resta in coda e viene
consegnata all'inizio della chiamata successiva**. Conseguenza sulle misure: l'ultimo timestamp di
ogni `generate()` manca, quindi il decode rate è calcolato su n−1 token (~0.4% su 256, sistematico
e sempre nello stesso verso — non si media via fra repliche).

Si vede anche nei dati: la cella wllama in
`results/4090-linux-2026-07-26T22-51-39-379Z.json` riporta `completionTokens: 255` a fronte di
`maxTokens: 256`.

Già segnalato upstream da terzi: [issue #263](https://github.com/ngxson/wllama/issues/263) e
[PR #264](https://github.com/ngxson/wllama/pull/264), entrambe aperte al 2026-07-27; la 3.5.1 è
precedente alla PR. **Nessun workaround nel nostro codice**: silenziarlo nasconderebbe un
comportamento che chiunque usi wllama per misurare incontrerà. Quando il fix viene rilasciato,
aggiornare la dipendenza e rimuovere questa deroga.

### Fasce modello — il gap strutturale della fascia Large

| fascia | modello | WebLLM | Transformers.js | wllama |
|---|---|---|---|---|
| Tiny | Qwen2.5-0.5B-Instruct | ✓ | ✓ | ✓ (Q4_K_M ≈0.37GB) |
| Small | Llama-3.2-1B / Qwen2.5-1.5B-Instruct | ✓ | ✓ | ✓ (≈0.75–0.92GB) |
| Mid | Llama-3.2-3B / Phi-3.5-mini-instruct | ✓ | ✓ (Phi-3.5-mini: solo quant `q4f16`) | ✓ (≈1.9–2.2GB) |
| **Large** | Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct | ✓ | **✗** | **✗** |

**Fascia Large: solo WebLLM può servirla.** Non è un problema di tuning — è un gap strutturale,
verificato su Hugging Face Hub (2026-07-26):
- **Transformers.js**: nessun repo ONNX web-runnable per questi due modelli — esistono solo
  varianti vendor-locked (AMD/NVIDIA/DirectML), non eseguibili nel browser via ONNX Runtime Web.
- **wllama**: il Q4_K_M di entrambi supera il tetto WASM di 4 GB sui soli pesi (Qwen2.5-7B
  ≈4.36GB, Llama-3.1-8B ≈4.58GB, quest'ultimo con margine peggiore).

Non forziamo quant più aggressivi per farceli entrare: degraderebbe la qualità sotto la soglia
utile, il che scambierebbe un gap onesto per un numero fasullo. Il gap resta documentato qui,
mai "risolto" in silenzio.

## Fase 3 — modulo qualità + schema v3

- **Schema v3** (`SCHEMA_VERSION = 3`, non retro-compatibile con i file v2 in `results/`, che
  restano storici): `BenchCell` guadagna `protocol` (v. sopra, docket #7) e `qualityScore`
  (opzionale — vedi sotto).
- **`src/quality.ts` + `src/qualityPrompts.ts`**: modulo puro, nessuna dipendenza da DOM/worker,
  copertura completa da unit test. Due percorsi, scelti da `capabilities().logprobs`:
  - **perplexity** (`computePerplexity`) quando l'adapter espone i logprobs — `exp` della
    log-probabilità media negata sul passaggio di testo generato.
  - **fallback exact-match** (`evaluateExactMatch`) altrimenti: 12 prompt deterministici greedy
    in 4 categorie (aritmetica, factual breve, format-following, task JSON), valutati via
    regex/parsing — nessuna soglia di pass/fail, il punteggio grezzo (n/12) è quello riportato.
- **`BenchCell.qualityScore` è opzionale e oggi non è popolato da nessun run reale**: il modulo
  è pronto e testato ma non ancora collegato a `benchServer.ts`. Collegarlo alla pipeline di
  bench vera significherebbe eseguire fino a 12 `generate()` extra per cella (percorso
  exact-match) o un passaggio dedicato ai logprobs — un costo/tempo aggiuntivo per ogni run
  fisico che nessuna decisione ha ancora approvato esplicitamente. I run in `results/` restano
  quindi solo metriche di velocità, come prima di questa fase.

## Note

- I risultati in `results/` sono i dati del progetto: committati, schema
  versionato (`schemaVersion`), niente fingerprinting (label device inserita a mano, vedi sopra —
  fino al 2026-07-27 era invece una costante cablata, e i run di ogni device si dichiaravano
  `4090-linux`).
- Fase 1b — matrice piena: adapter Transformers.js (Fase 1), wllama (Fase 2) e modulo qualità +
  schema v3 (Fase 3) fatti. Resta lo sweep manuale sui 3 device (M4 Pro, Samsung S22 Ultra) — fuori
  da queste fasi per costruzione, passo separato quando i device sono fisicamente disponibili.
  Fase 2 della Fasatura originale (deep-dive kernel MLC): vedi spec, sezione Fasatura.
