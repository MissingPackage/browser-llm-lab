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
- Flag usati: `--enable-unsafe-webgpu --enable-features=Vulkan,WebGPUService --ignore-gpu-blocklist`.

## Cosa abbiamo verificato dal vivo (4090 mobile, Fedora, chromium Playwright)

- **WebGPU su Linux/NVIDIA funziona** nel worker con i flag sopra (vendor
  `nvidia`, arch `lovelace`). Headless shell → SwiftShader, inutilizzabile per dati.
- **`maxStorageBufferBindingSize` ≈ 2 GiB** anche su una GPU da 16 GB: il muro
  per-buffer previsto dalla spec, confermato. `maxBufferSize` ≈ 4 GiB.
- **`shader-f16` NON esposto** da questo chromium su NVIDIA/Vulkan → i modelli
  MLC `q4f16_1` falliscono la validazione shader (`index_kernel`). Fallback:
  varianti `q4f32_1`. Finding chiave: *la quant servibile dipende dalle feature
  del browser, non solo dalla GPU*. (Da ritestare su Chrome branded / flag Dawn.)
- **COEP `require-corp` convive col CDN HF**: shard scaricati senza bisogno del
  fallback `credentialless`.
- Primi numeri (Qwen2.5-0.5B `q4f32_1`, schema v1, in `results/`):
  load cold ~56 s → **warm ~1.6 s** (Cache API); TTFT 0.5–0.9 s (warm–cold);
  decode **~106–118 tok/s** (varianza run-to-run ~10%, repliche multiple in 1b).

## Note

- I risultati in `results/` sono i dati del progetto: committati, schema
  versionato (`schemaVersion`), niente fingerprinting (label device manuale).
- Fase 1b (matrice piena: Transformers.js, wllama, più modelli/quant/device) e
  fase 2 (deep-dive kernel MLC): vedi spec, sezione Fasatura.
