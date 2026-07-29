# HANDOFF — browser-llm-lab   (updated 2026-07-29, session 9)

## 1. Next decidable

**Ruling PI su #14 (from-scratch vs fork)**: raccomandazione consegnata in chat
(2026-07-29) — scratch narrow GGUF-compatibile, llama.cpp come oracolo non come
substrato. Al ruling parte subito il **WP di studio #15 (già approvato)**, coi brief
tarati sull'esito. Poi `docs/engine/direction.md` e il goal.
Roadmap approvata invariata: consolidamento → ceiling + hero-demo + benchmark → motore.

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
10. **Correzione doc pubblicati** ("~34 dispatch/token" → 270; "1 submit" → 7) in tre
    doc deep-dive. Correggere ora o nel consolidamento?
11. **Run `dispatch-profile` su M4 e S22** (mani PI, ~5 min/device) — chiude le
    estrapolazioni cross-device di estimates.md.
12. ~~Commit sessioni 8-9~~ DECISO e FATTO (2026-07-29): 4 commit diretti su `main`
    (convenzione docs/data delle sessioni 7-8). Push NON fatto (non richiesto).
13. **Modello target del motore** — ora informato da: cuneo M4, DeepSpec (draft per
    Qwen3-4/8/14B), ecosistema DeepSeek V4 Flash (ds4+DSpark+MLA), narrow-focus.
14. **Motore from-scratch vs sopra llama.cpp-WebGPU** — raccomandazione consegnata
    (scratch narrow: tesi inesprimibile in ggml, economia dell'attenzione, deep-dive
    = spec del v0; paletti: GGUF come formato, llama.cpp come oracolo, tokenizer/draft
    presi in prestito). ATTESA RULING PI.
15. ~~WP "studio in profondità"~~ APPROVATO (2026-07-29): ds4, colibri, DeepSpec
    (architettura draft), LlamaWeb/ggml-webgpu (incl. stato MLA). Parte al ruling #14
    (i brief cambiano con l'esito).
