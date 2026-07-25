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
mostra load/TTFT/tok-s. **Export JSON** scarica il run file (schema v1) da
salvare in `results/`.

## Run automatizzato (Playwright)

```bash
node scripts/e2e-bench.mjs                        # headless (finisce su SwiftShader: solo smoke)
HEADED=1 node scripts/e2e-bench.mjs               # GPU reale (serve un display)
HEADED=1 MODEL_ID=Qwen2.5-0.5B-Instruct-q4f32_1-MLC QUANT=q4f32_1 node scripts/e2e-bench.mjs
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
  hardware va fatta fuori banda: nvidia-smi o about:support), `E2E_PROFILE=<dir>`.
- Bench manuali sul Chrome branded: `scripts/bench-chrome.sh` (profilo dedicato `blab-bench`).
  **Mai** impostare i flag Vulkan in `chrome://flags` del profilo quotidiano: `enable-vulkan`
  corrompe il compositing su NVIDIA/Wayland, `force-enable-webgpu-interop` crasha all'avvio.
- Chrome 150 branded (launcher `bench-chrome.sh`): cold 61 s / warm 1.8 s, fino a **116.9 tok/s**
  (run manuale utente) — in linea con chromium-playwright; la varianza run-to-run (~10-25%)
  conferma la necessità di repliche multiple in 1b.

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
  dalla pagina. Il probe di 1b deve rilevarlo (fingerprint: cap 128 MiB + vendor
  vuoto). Il gap GPU-vero: chromium ~110 vs Firefox ~10 tok/s (≈11×, plausibile
  ruolo di `subgroups`, assente su Firefox).
- **COEP `require-corp` convive col CDN HF**: shard scaricati senza bisogno del
  fallback `credentialless` (vale anche su Firefox).
- Primi numeri (Qwen2.5-0.5B `q4f32_1`, schema v1, in `results/`):
  load cold ~56 s → **warm ~1.6 s** (Cache API); TTFT 0.5–0.9 s (warm–cold);
  decode **~106–118 tok/s** (varianza run-to-run ~10%, repliche multiple in 1b).

## Note

- I risultati in `results/` sono i dati del progetto: committati, schema
  versionato (`schemaVersion`), niente fingerprinting (label device manuale).
- Fase 1b (matrice piena: Transformers.js, wllama, più modelli/quant/device) e
  fase 2 (deep-dive kernel MLC): vedi spec, sezione Fasatura.
