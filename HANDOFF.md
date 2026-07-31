# HANDOFF — browser-llm-lab   (updated 2026-07-31, session 15 — goal engine-fase-c2 APERTO)

## 1. Next decidable

**Fase 5 del goal `engine-fase-c2` (MoE + residenza minima, timebox 4 it.),
slice 1**: router (sigmoid+bias+top-4+norm ×1.8, replica C1 — verifica
riga-per-riga in build_moe_ffn, llama-graph.cpp 5f55650) + GEMV per-expert
sugli slab; poi residenza minima (GGUF in OPFS, cache VRAM LRU a due
size-class 5.308.416/5.505.024 B, ~2.5k slot) + conformance routing ≥99%
vs traccia C1 (spec §4-§5, §7). **FASE 4 CHIUSA a it.5** (verifier PASS):
layer 0 GLM assemblato su GPU (`glmforward.ts`, MLA absorbed, 22
dispatch/token) + conformance layer-level con PESI REALI vs cpuref-f64
naive: L2rel 2.35e-7 su 16 pos decode (4090); identità algebrica
naive↔absorbed in suite; ktest 23/23, suite 200/200, tsc pulito. Il gate
"golden" matura al full-model (fase 6, lettura registrata a journal it.5).
Watch item fase 6: stringere il gate layer-level (ora 1e-3 vs 2.35e-7
misurato) + q5_K absTol (it.3) + assunzione mscale=1 da golden.
Riancorarsi da: `.harness/goals/engine-fase-c2/{GOAL,PHASES,docket,journal}.md`,
spec §4-§5, `src/engine/glmforward.ts` (pattern da estendere), traccia
`results/engine/moe-oracle/trace-2026-07-31.jsonl.gz`, sim residenza C1.

## 2. State delta (questa sessione, 15)

- **Docket C1 item 4 RISOLTO** (ruling PI: "Adotta, ma WP comunque"): GO
  prefetch / NO-GO pinning adottato come input C2/C3; il WP banda fredda
  BROWSER si fa comunque ma in C3, non blocca C2.
- **Banda NVMe a freddo bracketata lato OS** senza il WP browser
  (`tools/cold-read-bench.py`, fadvise DONTNEED, 990 PRO): random
  expert-size **1.63 GB/s (3.74 ms/expert p50, 8× il warm 0.46)**, seq 3.22;
  warm re-read 10-11 GB/s = coincide col bench browser (metodo validato).
  JSON: `results/opfs-bench/cold-read-os-4090-linux-2026-07-31.json`.
- **Headline "modello ~2× la memoria" CONDIZIONATA ai tier** (propagata a
  ledger §A e direction §8.3): regge come "2× la VRAM con spillover
  RAM-backed" (24% del budget); in regime disk-bound il tetto è ~18 tok/s o
  serve hit ≥94.5% con overlap perfetto ⇒ prefetch unica leva.
- **Goal engine-fase-c2 aperto**: contratto approvato con emendamento,
  PHASES.md scritto (7 fasi), plan-check a docket.
- Ruling permanente nuovo in memoria: **non-regressione delle metriche a ogni
  merge** (metrics-non-regression.md).
- Commit: `b6a3ec5` (istruttoria item 4), `cbf2523` (ruling + propagazione),
  più lo scaffold C2 (questo commit).

## 3. Open threads

- Goal C2: iterazione 1 pronta dopo il plan-check (fasi 1+2).
- **WP banda fredda browser** → precondizione di C3 (ruling item 4), non di C2.
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket), `fase-2-deep-dive`
  (5) — igiene da /weekly-maintenance.
- Duplicato da igiene: `results/opfs-bench-*.json` esiste sia in `results/`
  root sia in `results/opfs-bench/` (segnalato, non toccato).
- Rimandi §I ledger invariati; GLM-5 uscito → ledger §H a v2 (invariato).

## 4. Landmines

(invariate dalla sessione 14 — valgono tutte per C2, in particolare:)
- Pipeline WebGPU invalida = submit droppati in silenzio con readback stale
  plausibili ⇒ error scope come contratto; diag a cold-start.
- Oracolo CPU quantizza le attivazioni (q8): noise floor 98.05% ⇒ gate doppio
  (cpuref-f64 ≥99% E golden ≥97%).
- Coi tap attivi il tsq copre solo il primo segmento del pass ⇒ gpuBusy senza tap.
- MAI confrontare gpuBusy/wall a contesti diversi; mai mapAsync su buffer di
  submit non emesso; var WGSL nei loop da azzerare esplicitamente.
- llama-cli scriptato SEMPRE `-st --simple-io`; Vite porta dedicata 5199 + kill.
- Device senza requiredLimits ⇒ 128 MiB binding garbage; f32-first su Chrome/Linux.
- `erasableSyntaxOnly`; contratto `pos === kvLen` hard.
- NUOVA (sessione 15): il bench freddo lato OS usa dati urandom perché /home è
  btrfs `compress=zstd:1` — mai bench di banda con dati comprimibili lì.

## 5. Docket (decisioni PI pendenti)

1. **PLAN-CHECK C2** (docket del goal, item 1) — vedi §1.
4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10 qualityScore
     (goal evals futuro); #8 sorveglianza wllama (v3.1 WebGPU, rilevante per
     il benchmark pubblico).
16. Headline del benchmark pubblico (ledger §E) — contributo separato
    (ruling 2026-07-30), su iniziativa PI.
20. ~~go/no-go PILOT~~ RISOLTO 2026-07-31 ("Adotta, ma WP comunque") — vedi §2.
