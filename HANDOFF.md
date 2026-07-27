# HANDOFF — browser-llm-lab   (updated 2026-07-28, session 7)

## 1. Next decidable

**Sessione di stima per il nuovo motore** (ruling PI 2026-07-28): quantificare
l'incremento atteso da TUTTI i suggerimenti del deep-dive (backlog engine-notes +
candidati esperimento), valutare trade-off e alternative, motivare le scelte — in
funzione dell'obiettivo dichiarato: **costruire un nuovo motore di inferenza browser**.
Input: `docs/deep-dive/engine-design-notes.md` (backlog + questioni aperte) e i 6 run
cross-device. Roadmap approvata (ruling stesso giorno): consolidamento (incl. soglia
TTFT del #12) → ceiling + hero-demo + benchmark pubblico (eventuale paper) → nuovo
motore. Goal fase-2-deep-dive: MERGIATO su main (77ed165) e pushato.

## 2. State delta (session 7)

- PI ha eseguito: bench M4 Pro, bench S22 con q4f16_1 (esperimento docket #2, slot 1),
  micro-bench su M4 e S22 → committati su `main` (340ab91, fcb54b0).
- Esiti chiave: S22 f16 = decode +66% (11.6 tok/s) e varianza TTFT collassata (±34%→±0.4%);
  M4 = 98.3 tok/s ≈ 4090 con metà banda (conferma tesi orchestrazione); micro-bench M4 =
  248 GB/s misurati (91% di targa), q4 GEMV 274 Gw/s (2× la 4090); micro-bench S22 =
  tetto effettivo ~22 GB/s (43% di targa), floor per-dispatch ~0.13 ms (24× la 4090).
- Probe M4 ha FALSIFICATO la tesi "binding cap 2³¹−4 = costante API" (su Metal è 2³²−4):
  `buffer-limit-2gb.md` corretto sul branch (c85777d).
- Docket #7-#8 registrati (esiti esperimento + findings M4); #6 (warm-up) declassato
  dai dati: la varianza TTFT era legata al percorso f32, non al DVFS puro.

## 3. Open threads

- Sweep manuale fase 1b: wllama/transformersjs su S22 ancora mancanti (fuori goal).
- Goal di consolidamento da aprire (conterrà: soglia TTFT #12, esiti della sessione di
  stima, eventuale resto sweep).

## 4. Landmines

- Chrome headless su Linux/NVIDIA cade su **SwiftShader** (spia:
  maxComputeInvocationsPerWorkgroup=256): i driver Playwright del deep-dive richiedono
  HEADED=1 (`tools/wgsl-dump.mjs`, `tools/microbench-run.mjs`).
- Chrome **quantizza i timestamp GPU** (~100 µs): il micro-bench misura batch da 16
  dispatch per campione — non togliere il batching.
- Un device WebGPU senza `requiredLimits` nasce a 128 MiB di binding → celle garbage
  silenziose. Error scope + checksum già nel runner: mantenerli.
- `erasableSyntaxOnly` in tsconfig: niente parameter properties nelle classi.
- Chrome branded su Linux/NVIDIA NON espone `shader-f16` (M4 e S22 sì).
- Dev server: il :5173 di una vecchia sessione potrebbe essere ancora vivo (pid 487440,
  non nostro); i tool del goal usano BASE_URL (:5177 nell'ultima sessione).
- I numeri di riga citati nei doc valgono per `@mlc-ai/web-llm 0.2.84`.

## 5. Docket (decisioni PI pendenti)

1. ~~Merge+push~~ DECISO e FATTO (2026-07-28): merge 77ed165, push origin/main.
2. ~~Slot esperimento #2~~ DECISO: non assegnato — sostituito dalla sessione di stima
   (vedi §1); i candidati confluiscono lì come opzioni da quantificare.
3. ~~Docket #12~~ DECISO: soglia TTFT separata nel goal di consolidamento.
4. **Promozione skill** `bottleneck-brainstorm` a `~/.claude/skills/`: non decisa,
   resta project-level.
5. Ereditati non bloccanti: #10 (qualityScore non collegato), #8 (sorveglianza wllama).
