# HANDOFF — browser-llm-lab   (updated 2026-08-09, sessione 25 — GOAL C3C CHIUSO, fase C COMPLETA; next = decisione PI post-fase-C)

## 1. Next decidable

**DECISIONE PI: direzione post-fase-C.** Il goal engine-fase-c3c è CHIUSO
E VERIFICATO (it.10, checklist 10/10, tag `goal-engine-fase-c3c-done`,
commit 58f5cf4): **floor C1 13.43 battuto SECCO in produzione — decode
15.641 a budget CALCOLATO (ctx-aware 12.737 GiB), sessione utente viva,
clausola non servita**. FASE C COMPLETA (C3a+C3b+C3c): **decode 4.64 →
15.64 tok/s (3.4×), TTFT 88 → 12.9 s (6.8×), 47 → 2 sync/token, qualità
bit-invariata** (golden 98.828% AL PIN, cpuref 256/256+512/512, firma
routing esatta). Candidati sul tavolo (nessun goal aperto):
(a) **hero-demo M4** — input pronti (docket c3c item 8), PI-gated per
hardware; (b) **fase D — moltiplicatori** (spec-dec MTP, direction §7;
gap UX residuo: 1.92× decode, 3.22× TTFT); (c) **igiene fuori-goal**
(v. §3). Riancorarsi da: `.harness/goals/engine-fase-c3c/{GOAL,journal,
docket}.md`, direction §7 (fase C chiusa coi numeri), ledger §A.

## 2. State delta (sessione 25, 2026-08-08/09 — goal C3c intero)

- Fasi 1-9 in 10 iterazioni, ogni iterazione verificata da loop-verifier
  e committata su main (10 commit, 5c2de37..58f5cf4).
- Meccanismi nuovi in `src/engine/`: `slabBudgetCtxAware` (residency.ts,
  riserva tarata 512 MiB), prefetch in-forward decode+prefill chunked
  (glmmodel.ts, tap router L+1, recall 91.92% @K=8), policy tier+AUTOPIN
  (residency.ts, pin ≤12.5% HARD), `bandmodel.ts` (±1% sui 3 punti, test
  permanente). Tutto dietro flag: default = comportamento storico.
- Tool nuovi: `tools/opfs-cold/` (banda fredda browser, freddezza provata),
  `scripts/glm-instanton-run.mjs` (instant-on, protocollo mediane 3+3).
- Riferimenti nuovi (docket c3c item 2/5-bis): b12 optimistic
  13.172/31.26/14.74; hit-rate policy 1472/736 (routing-policy-*.json);
  instant-on 1.247 ≤ 1.25× (ruling item 1a).
- Docs: direction §7 fase C CHIUSA, §8.3 banda browser; ledger §A riga
  paging CHIUSA; spec `2026-08-08-engine-fase-c3c-design.md`.
- Storia completa: `.harness/goals/engine-fase-c3c/journal.md` (it.0-10);
  lo storico delle sessioni 17-24 vive nei journal dei goal c3a/c3b.

## 3. Open threads

- **Fuori-goal, non bloccanti**: ratifiche c3a item 14b / item 2 formale /
  19-21 prese d'atto; `npm run test:conformance` punta all'harness
  benchmark; BASE_URL :5173 in 2 script; goal stale fase-1b/fase-2
  (weekly-maintenance); campi power hostState scambiati (c3c item 7);
  selezioni prefill non in eusage (c3c item 6); profilo bench ~6.4 GiB in
  `~/.cache/blab-opfs-cold-profile` (cancellabile).
- C3b/C3a/C2: chiusi, nessun item aperto che richieda azione.

## 4. Landmines

- Bench su questa macchina: gate a macchina scarica o host state DICHIARATO
  nel JSON; costo per-fetch varia ~4× fra sessioni host (upload 0.65-3.7
  ms) — mai confrontare bande ±5% fra host state diversi.
- Margine instant-on 0.3%: qualunque ritocco al prefill invalida il
  riferimento — rimisurare col protocollo v2 (mediane 3+3), non a sessione
  singola.
- Prefill in scarsità collassa (30→4 tok/s a 1472/736 slot): su device
  piccoli il TTFT è prefill-bound, primo numero da guardare in fase D.
- Run GPU lunghe SOLO ad albero congelato (HMR vite uccide la run) e 60 s
  fra run; niente pipe sull'output dei runner; `/tmp` è tmpfs (profili
  Chrome in ~/.cache); KV = 108 288 B/token (f32, NON il vecchio 54 KB).
- Full-corpus SOLO per firma/nonreg/riferimenti nuovi; sim su traccia o
  subset di PROMPT INTERI per esplorare (mai --cap: perde il decode).
- Oracolo CPU quantizza q8: mai gate su near-tie; golden ≥98.828% è un PIN.
- Storiche: error scope su ogni submit; vite :5199; f32-first; `pos ===
  kvLen` hard; var WGSL nei loop da azzerare; mai mapAsync su submit non
  emesso; lock OPFS transitorio possibile al primo accesso (retry).

## 5. Docket (user decisions pending)

- **Direzione post-fase-C** (M4 / fase D / igiene) — la decisione di §1.
- c3c item 8: input hero-demo M4 (PI-gated per hardware).
- Ratifiche c3a pendenti (14b, 2 formale, 19-21) — prese d'atto, non urgenti.
