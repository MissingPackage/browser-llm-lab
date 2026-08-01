# PHASES — engine-fase-c3a

Decomposizione iteration-0 (2026-08-01). Sequenziale: gli `owns` delle fasi 3-5
si sovrappongono sul path caldo (il repack cambia il layout che i kernel MoE
leggono, e sync-router e prefill batched toccano entrambi il forward), quindi
nessun `parallel-group`. Le fasi 3-6 sono gated dal ruling di spec (fase 2),
come da convenzione C1/C2. Nessuna fase richiede authority oltre il grant di
GOAL.md.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | Strumentazione & attribuzione fine: glmbench misura il TTFT e stampa i tre gap dalla funzione obiettivo (30 / 60 thinking / 4 s); run controfattuale a **selezione expert registrata** (nessun readback del router) che separa i 158.9 ms/token di struttura in sync vs dispatch | run glmbench exit 0 + JSON in `results/engine/` con campi `ttftMs`, `gapDecode30`, `gapDecode60`, `gapTtft4s`; secondo JSON di attribuzione con ms/token imputati a sync e a dispatch, metodo dichiarato nel payload; `npm test` verde, `npx tsc --noEmit` pulito | none | scripts/, glmbench.html, telemetria in src/engine/ (solo lettura/contatori, nessuna modifica al path caldo) | ready |
| 2 | Spec C3a (spec-first): formato repacked all'import (layout, versioning, invalidazione su SHA/versione), meccanismo dei sync router **scelto col criterio esplicitato** sui numeri di fase 1 (top-4 su GPU / bind del parco residente / pipelining a profondità >1), percorso prefill M>1 per MoE + sua condizione di identità, protocollo bench identico a C2, definizione operativa del TTFT | `docs/superpowers/specs/2026-08-XX-engine-fase-c3a-design.md` esiste e copre i 5 punti del DONE WHEN §1; entry `RULING RICHIESTO` nel docket | none | docs/superpowers/specs/ | blocked-by-1 |
| 3 | Repack all'import: l'import produce l'artefatto repacked (layout dai kernel, non dal GGUF), con versioning e invalidazione; i kernel MoE leggono il nuovo layout | `npm test` verde su produzione dell'artefatto + invalidazione su SHA/versione mismatch; `npx tsc --noEmit` pulito; bench JSON con **pack CPU < 1.0 ms/token** nel path caldo (baseline C2: 42.5); correttezza invariata (argmax ≡ cpuref-f64 256/256 sul campione ratificato) | none | src/engine/{gguf,quant,shape}.ts + import/OPFS, kernel MoE lato lettura pesi, tests/ | blocked-by-2 |
| 4 | Sync router: eliminazione o batching dei 46 readback per token col meccanismo di spec | bench JSON con **≤ 2 sync CPU↔GPU per token di decode** (baseline 47), o la soglia ammortizzata fissata in spec col suo parametro di profondità dichiarato nel payload; routing conformance rigenerato **non peggiore** di `routing-conformance-glm47flash-2026-07-31.json`; `npm test` verde | none | src/engine/ (router/moe, gpuforward, scheduling dispatch), tests/ | blocked-by-2 |
| 5 | Prefill batched M>1 per MoE (insiemi di expert diversi per token) | test di identità logits M=1 vs M>1 sullo stesso prompt entro la tolleranza di spec, verde in `npm test`; bench JSON con prefill tok/s e TTFT misurati sullo stesso p6 | none | src/engine/ (prefill path, prefillplan, moe batched), tests/ | blocked-by-2 |
| 6 | E2E & GATE: bench quiescente coi due gate di chiusura + i numeri di gap + correttezza invariata + non-regressione del pregresso | bench JSON quiescente con **decode ≥ 13.43** e **prefill ≥ 56.58 tok/s**, e coi campi di gap di fase 1 presenti (assenza = FAIL); conformance logits (argmax ≡ cpuref-f64 256/256; top-1 vs golden ≥ 98.83% full-corpus) e routing ≥ valori C2; conformance fase A verde; bench Qwen ≥ baseline 2026-08-01 (K=8 321.88, K=1 243.97, prefill chunked ≤ 600.2 ms); `npm test` ≥ 220 passed, `tsc --noEmit` pulito | none | src/engine/, results/engine/ | blocked-by-3,4,5 |
| 7 | Chiusura: docket con gli input per C3b (residenza residua dopo il repack, hit-rate osservato, nuovo profilo ms/token scomposto), direction §7 + ideas-ledger §A ove stale, HANDOFF refresh, merge+push | checklist DONE WHEN completa nel journal; verifier gate finale PASS; merge+push eseguiti (ruling permanente 2026-07-29) | none | docs/, .harness/, HANDOFF.md | blocked-by-6 |

Note:
- **Fase 1 è loop-runnabile subito** e non tocca il path caldo: gira mentre il
  ruling di spec non esiste ancora. Non è solo strumentazione — è la misura che
  la fase 2 usa per SCEGLIERE il meccanismo dei sync, oggi l'unico numero
  mancante (i 158.9 ms/token di struttura sono sync *più* 1.816 dispatch, senza
  split noto). Scriverla dopo la spec sarebbe scegliere alla cieca.
- **La fase 4 è il rischio del goal.** Il tetto senza di essa è ~7 tok/s
  (docket C2 item 8) ⇒ il gate 13.43 vive o muore qui. Può sforare le 4
  iterazioni: in quel caso lo split della fase va a docket, non preso nel loop
  (PHASES si modifica solo via ruling dopo iteration 0).
- **Nessuna clausola di fallback pre-negoziata** (a differenza di C1): se dopo
  le fasi 3-5 il decode resta sotto 13.43, il contratto non autorizza né
  scorciatoie né auto-deroga — si docketa. Proposta di clausola aperta a
  docket item 2.
- Gate non-regressione: HARD (contratto v1 C2, principio permanente ruling PI
  2026-07-31). Regressione non assorbibile ⇒ docket, non merge.
- Ogni bench va eseguito a **macchina quiescente** con lo stato host dichiarato
  nel report (norma PI 2026-08-01).
