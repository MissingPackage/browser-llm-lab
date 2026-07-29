# HANDOFF — browser-llm-lab   (updated 2026-07-29, session 9)

## 1. Next decidable

**Piano di lavoro del motore, in una SESSIONE NUOVA** (deciso col PI, 2026-07-29):
riancorarsi da HANDOFF + `docs/engine/{estimates,ideas-ledger}.md` + i 4 study report
(`docs/engine/study/` — leggere prima `study/README.md`, la sintesi incrociata), poi
scrivere `docs/engine/direction.md` e aprire il goal. Ruling già presi: **scratch**
narrow GGUF-compatibile (#14), first-light = battere WebLLM (107 tok/s, Qwen2.5-0.5B
q4, 4090) con le sole L1-L3. Decisione ancora aperta che il piano deve forzare: #13
(modello target). Roadmap invariata: consolidamento → ceiling+hero-demo+benchmark → motore.

## 2. State delta (sessioni 8-9)

- Sessione 8 (stima): tool `.harness/tools/dispatch-profile.mjs` + run 4090 →
  **270 dispatch/token (non ~34), 7 submit/token, 8.3 µs encode CPU/dispatch**;
  round-trip per device estratto dai micro-bench (1.8/0.45/5.0 ms). Doc
  `docs/engine/estimates.md`: modello di budget, leve stimate (1.44× architettura,
  2.06× totale), kernel engineering ultima leva ovunque.
- Sessione 9 (brainstorm + verifica): nuova funzione obiettivo dal PI — **max
  intelligenza sotto rate "sufficiente" (~30 tok/s morbidi) e memoria host**; due
  budget (touch = BW×(1/rate−T_fisso), residenza). Frontiere: 4090 ~20B, M4 cuneo
  MoE 5×, S22 oggi irraggiungibile (T_fisso 40 ms > budget).
- `docs/engine/ideas-ledger.md`: ~50 idee su 3 assi + evals, con statuses.
- **Sweep di verifica eseguito** → `docs/engine/verification-sweep-2026-07-29.md`.
  Fatti chiave: llama.cpp ha backend WebGPU upstream (LlamaWeb, arXiv 2605.20706) con
  le nostre L1/L2 già dentro (valida la diagnosi; prefill loro debole −21-51%); MoE
  full-residency già in browser (LFM2); QuantSpec = prior art cascata di precisione;
  SharedWorker+WebGPU mai shippato (fallback leader-tab); colibri: routing predicibile
  71.6%, draft quant-sensibile (int4 → accettazione 0-4%); ds4: ricetta quant
  asimmetrica + KV checkpoint SHA1 su disco.
- Memoria persistente: `engine-objective-intelligence-at-threshold.md` (funzione
  obiettivo + precedente ardesia-gguf).

## 3. Open threads

- Gap statement raffinato: il vuoto browser è il **sistema di memoria** (paging
  esperti+prefetch, KV tiering/checkpoint, adapter hot-swap, spec-dec) + telemetria/
  eval integrata — non l'esecuzione MoE né GGUF+WebGPU (già esistenti).
- Idee promosse dal PI: LoRA hot-swap ("geniale"), oracolo desktop golden-logits,
  narrow-focus stile ds4 (dense = MoE degenere, un modello di riferimento).
- Aperti dallo sweep: stato MLA in llama.cpp, architettura draft DeepSpec, banda OPFS
  in lettura (tool 20 righe), MambaKit "WSLA", TEAL/PowerInfer (search non fatta).
- Sweep fase 1b (wllama/transformersjs su S22) ancora fuori goal.

## 4. Landmines

- Residuo non attribuito del budget 4090 = 33%: non assegnarlo a nessuna leva senza
  timestamp-query nel runtime. Floor M4 (28 µs) è un limite superiore.
- Paging esperti: a ~1 GB/s OPFS un miss pieno su 30B-A3B costa ~1.7 s → vive solo con
  pinning+prefetch; target realistico "modello ~2× la memoria".
- Chrome headless Linux/NVIDIA → SwiftShader: driver Playwright con HEADED=1.
- Chrome quantizza i timestamp GPU (~100 µs); `performance.now()` worker quanto 5 µs.
- Device senza `requiredLimits` nasce a 128 MiB binding → garbage silenzioso.
- Chrome branded Linux/NVIDIA NON espone shader-f16 (M4/S22 sì) — per il benchmark
  pubblico è un problema di equità da dichiarare.
- `erasableSyntaxOnly` in tsconfig; righe doc valide per `@mlc-ai/web-llm 0.2.84`.
- Dev server vite di sessioni vecchie vivi su :5173-:5177.

## 5. Docket (decisioni PI pendenti)

4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10 qualityScore (ora si
     aggancia alla sez. D del ledger), #8 sorveglianza wllama (ora rilevante: v3.1 ha
     WebGPU via LlamaWeb).
10. ~~Correzione doc pubblicati~~ FATTO (2026-07-29, ruling "doc stale si corregge
    subito"): quattro doc corretti (anche engine-design-notes), vedi estimates §7.
11. ~~Run `dispatch-profile` su M4 e S22~~ FATTO (2026-07-29) via `prof.html` (pagina
    manuale nuova). Esiti in estimates §8: N_disp/submit invarianti di device e quant,
    encode CPU S22 ~67 µs/dispatch (~18 ms/token), M4 ~2.5 µs. S22 q4f32_1 non completa
    (empty timeline) — baseline S22 resta f16.
12. ~~Commit sessioni 8-9~~ DECISO e FATTO (2026-07-29): 4 commit diretti su `main`
    (convenzione docs/data delle sessioni 7-8). Push NON fatto (non richiesto).
13. ~~Modello target~~ **DECISO (2026-07-29): GLM-4.7-Flash (30B-A3B, MLA, MIT) come
    modello-tesi**; dev rungs Qwen3.5-0.8B/2B (Apache); first-light invariato
    Qwen2.5-0.5B; v2 target dichiarato: architetture ibride (Qwen3.5-35B-A3B /
    Nemotron). Razionale e panorama architetturale in `ideas-ledger.md` §H.
    Due [VERIFY] da chiudere in sessione di piano: (a) presenza testa MTP su
    4.7-Flash (per spec-dec senza DeepSpec), (b) benchmark indipendenti non-vendor.
14. ~~From-scratch vs fork~~ **DECISO (2026-07-29): SCRATCH** — narrow,
    GGUF-compatibile, llama.cpp come oracolo non come substrato; tokenizer e draft
    DeepSpec in prestito. First-light dichiarato: Qwen2.5-0.5B q4 su 4090, target
    battere i 107 tok/s di WebLLM con le sole L1-L3.
15. ~~WP "studio in profondità"~~ ESEGUITO (2026-07-29, sessione 9): 4 studi paralleli
    → `docs/engine/study/` + sintesi incrociata in `study/README.md`. Esiti chiave:
    tesi scratch confermata dal codice ggml-webgpu; sistema di memoria già progettato
    2× in nativo (PILOT 71.6% recall, tier.h ~60 righe); KV persistente risolta 2×;
    spec-dec portabile ma ri-prioritizzato (tap hidden-states da progettare in v0,
    build dopo; draft mai sotto int8; checkpoint DeepSpec senza licenza dichiarata).
    Caveat: nessun codebase eseguito, solo lettura.
