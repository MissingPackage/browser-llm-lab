# HANDOFF — browser-llm-lab   (updated 2026-07-30, session 12)

## 1. Next decidable

**Goal `engine-fase-b1`, FASE 5: prefix-cache OPFS** — su branch `engine/fase-b1`.
Riancorarsi da: `.harness/goals/engine-fase-b1/{GOAL,PHASES,docket,journal}.md` +
spec `docs/superpowers/specs/2026-07-29-engine-fase-b1-design.md` §Formato
prefix-cache + §Quota/eviction (envelope BKV1, chiave SHA-256 di
layoutVersion‖sha256(GGUF)‖tokenIds, niente logits, LRU 512 MB, mismatch ⇒ throw).
Struttura: `src/engine/kvstore.ts` = codec envelope PURO (CI senza OPFS/GPU) + I/O
`FileSystemSyncAccessHandle` dietro interfaccia; `saveKv/loadKv` in gpuforward
(copy GPU↔CPU); restore in **worker nuovo** (sessione fredda) con continuazione
token-identica. Done-when: unit indice/codec verdi; report JSON committato:
continuazione identica al run ininterrotto + tempo restore < tempo re-prefill
(entrambi nel JSON). Poi fase 6 (bench+chiusura, merge a goal chiuso).

## 2. State delta (sessione 12, 2026-07-30)

- **FASE 3 DONE (it.3)**: forward multi-token M≤8 conforme — `prefillplan.ts`
  (chunking puro), 6 kernel chunk in `kernels/wgsl.ts`, `prefillChunked()` (zero
  readback, submit/64, lm_head solo ultima posizione). Conformance col percorso
  M>1: GATE DOPPIO PASS 98.05% golden / 100.00% cpuref, anche con `telemetryGpu`.
- **Bug WGSL/Tint trovato e fixato** (prima conformance 82%): `var` array nel body
  di un loop NON ri-azzerata per iterazione su Chrome/Tint (vedi Landmine).
- **FASE 4 DONE (it.4)**: length pointer `kvLen` — `src/engine/kvlen.ts` puro
  (contratto hard `pos===kvLen`, `crop(toLen≤kvLen)` zero-GPU, `reset()≡crop(0)`),
  integrato in forwardToken/prefillChunked, handle espone `crop()`/`kvLen`;
  call-site a riavvio esplicito (`reset()`). Prova meccanica PASS (3 check:
  crop-prefisso, crop-metà-generazione, run fresco — sequenze IDENTICHE), JSON
  `kv-rollback-4090-*.json`; conformance invariante PASS; 143/143 unit.
- **Sim rimossa** (spec): `prefillBatched`+`prefillsim`+script. Scaffolding di fase
  (rimozione a fase 6): `prefill-diag.mjs`, `kernel-diag.mjs`, modalità `rollback`
  (questa resta: è il gate di fase 4).
- Report conformance esteso: `telemetryGpu` e `prefill{path,mMax,submitTokens}`.

## 3. Open threads

- **Branch `engine/fase-b1` NON merged** (merge a goal chiuso, ruling permanente).
- Goal B1: fasi 5-6 a cascata, nessun ruling pendente.
- **B2 da ri-inquadrare al goal-brief** (docket goal B1, item 2): GPU busy ≈2.2 su
  8.1 ms/token ⇒ ~73% del decode è sync/encode, non dispatch.
- Scaffolding temporaneo da rimuovere a fase 6: knob `tsqDiag`, modalità
  `prefilldiag`/`kerneldiag` + script relativi.
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket item),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Sweep fase 1b (wllama/transformersjs su S22) fuori goal; GLM-5 uscito → ledger §H a v2.

## 4. Landmines

- **WGSL/Tint: mai fidarsi dell'azzeramento implicito di una `var` array dichiarata
  nel body di un loop** — su Chrome/Tint NON viene ri-azzerata a ogni iterazione
  (spec WGSL dice il contrario): azzerare esplicitamente. Colpiva solo kernel con
  >64 blocchi/riga (unico loop multi-iterazione); trovato in fase 3, guardia:
  `scripts/kernel-diag.mjs`.
- **mai mapAsync su un buffer referenziato da un submit non ancora emesso**: Dawn
  droppa l'INTERO command buffer in silenzio (tsq-diag-2026-07-29.md).
- Chrome headless Linux/NVIDIA → SwiftShader: driver Playwright con HEADED=1; Chrome
  da shell sandboxata → SwiftShader anche headed (bench col sandbox disabilitato).
- Vite: server di sessioni vecchie su :5173+ servono CODICE STALE — porta dedicata
  (`npx vite --port 5199 --strictPort`), kill a fine sessione (`pkill -f "[v]ite --port"`).
- Device senza `requiredLimits` espliciti nasce a 128 MiB binding → garbage silenzioso;
  grid > 65535/dim ⇒ submit no-op muto (uncapturederror fatale già nel motore).
- Chrome branded Linux/NVIDIA NON espone shader-f16 → f32-first; timestamp GPU
  quantizzati ~100 µs.
- I tok/s sotto patch del profiler non sostituiscono i bench (S22 −40%); telemetria
  liv.2 opt-in `telemetryGpu`, zero-overhead da spenta.
- llama.cpp SOLO oracolo (mai vendored); oracolo CPU quantizza le attivazioni (q8):
  noise floor 98.05% ⇒ gate doppio (cpuref-f64 ≥99% E golden ≥97%).
- `erasableSyntaxOnly` in tsconfig; contratto B1 ATTIVO da it.4: `pos === kvLen`
  hard su forwardToken/prefillChunked — chi usa pos libere si rompe by design
  (`crop`/`reset` espliciti; i runner del repo sono già stati adeguati).

## 5. Docket (decisioni PI pendenti)

4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10-fase-1b qualityScore
     (goal evals futuro); #8 sorveglianza wllama (v3.1 ha WebGPU — rilevante per il
     benchmark pubblico).
16. Headline del benchmark pubblico (ledger §E): serve prima del goal benchmark.
17. RISOLTO (2026-07-30): ruling spec B1 (a-g in blocco) — dettaglio nel docket del
    goal, item 3; rimandi in ideas-ledger §I.
