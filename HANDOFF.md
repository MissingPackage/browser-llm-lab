# HANDOFF — browser-llm-lab   (updated 2026-07-28, session 7)

## 1. Next decidable

**Tutto è PI-gated**: il goal `fase-2-deep-dive` è COMPLETO (7/7 fasi, verifier PASS sul
DONE WHEN riga per riga) e i dati cross-device sono arrivati (bench + micro-bench su
4090, M4 Pro, S22). Le decisioni aperte sono nel docket (§5). Alla prima decisione presa,
il lavoro riparte da lì; il candidato più probabile è: aggiornare gli slot M4/S22 in
`docs/deep-dive/micro-bench-matmul.md` coi dati nuovi → merge del branch → nuovo goal
(fase 3 dello spec madre: ceiling + hero-demo).

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

- **Branch `feat/fase-2-deep-dive`** (20 commit, MAI pushato): 6 doc deep-dive, skill
  `bottleneck-brainstorm`, micro-bench (`src/microbench/` + `microbench.html`, 102/102
  test), 34 WGSL dumpati, 2 tool. In attesa di merge (docket).
- `docs/deep-dive/micro-bench-matmul.md` ha gli slot M4/S22 "pending" da riempire coi
  dati appena arrivati (lavoro pronto da fare sul branch, pre-merge).
- `main` locale avanti di 5 commit su `origin/main` (3 setup + 2 data) — push mai fatto.
- Sweep manuale fase 1b: wllama/transformersjs su S22 ancora mancanti (fuori goal).

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

1. **Merge** `feat/fase-2-deep-dive` → main (previo aggiornamento slot M4/S22 nel doc
   micro-bench). Push di main+branch a origin: mai fatto in tutto il goal.
2. **Slot esperimento #2 di 2** (lo swap q4f16 ha consumato il #1, esito ottimo):
   candidati vivi → multi-step decode (#3a) · overlap fetch/compile al load (#3b) ·
   sync-diradata upload (#4, ridimensionato: su M4 il load warm è già 0.6 s) ·
   warm-up pre-ramp (#6, declassato dai dati f16). Oppure: nessuno ora, si rimanda
   al prossimo goal.
3. **Docket #12 ereditato** (high-variance non guarda il TTFT): coi dati f16 la varianza
   S22 collassa, ma lo sweep mobile continuerà — soglia TTFT separata da implementare
   (piccola, candidabile a prima fase del prossimo goal) o chiudere come "risolto dal
   passaggio a f16"?
4. **Promozione skill** `bottleneck-brainstorm` a `~/.claude/skills/` o resta project.
5. **Prossimo orizzonte**: fase 3 dello spec madre (ceiling run + hero-demo) come nuovo
   goal (brainstorming → goal-brief), o altro.
6. Ereditati non bloccanti: #10 (qualityScore non collegato), #8 (sorveglianza wllama).
