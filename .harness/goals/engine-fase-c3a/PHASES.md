# PHASES — engine-fase-c3a

Decomposizione iteration-0 (2026-08-01). Sequenziale: gli `owns` delle fasi 3-5
si sovrappongono sul path caldo (il repack cambia il layout che i kernel MoE
leggono, e sync-router e prefill batched toccano entrambi il forward), quindi
nessun `parallel-group`. Le fasi 3-6 sono gated dal ruling di spec (fase 2),
come da convenzione C1/C2. Le fasi 1-4b/5-7 non richiedono authority oltre il
grant di GOAL.md; **la 4c sì**, ed è concessa dall'emendamento 4 (quant
asimmetrica + nuova versione di layout dello slab, prima must-docket).

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | Strumentazione & attribuzione fine: glmbench misura il TTFT e stampa i tre gap dalla funzione obiettivo (30 / 60 thinking / 4 s); esperimento che separa i 158.9 ms/token di struttura in sync vs dispatch | run glmbench che **scrive il report** (exit 0 o 4 — exit 0 significa gate PASS, impossibile in fase 1; corretto da docket item 5) con campi `ttftMs`, `gapDecode30`, `gapDecode60`, `gapTtft4s`; sezione di attribuzione **nel report di bench** con ms/token imputati a sync e a dispatch e metodo dichiarato nel payload; `npm test` verde, `npx tsc --noEmit` pulito | none | scripts/, glmbench.html, telemetria in src/engine/ (solo lettura/contatori, nessuna modifica al path caldo) | **done (it.1)** — attribuzione dentro il report di bench invece che in un secondo JSON; exit 4 (report scritto + gate FAIL), non 0: correzione della riga a docket item 5 |
| 2 | Spec C3a (spec-first): formato repacked all'import (layout, versioning, invalidazione su SHA/versione), meccanismo dei sync router **scelto col criterio esplicitato** sui numeri di fase 1 (top-4 su GPU / bind del parco residente / pipelining a profondità >1), percorso prefill M>1 per MoE + sua condizione di identità, **piano della quarta leva (granularità/fusione dei dispatch, emendamento 1) col target derivato `gpuBusy` ≤ 54.5 ms/token**, protocollo bench identico a C2, definizione operativa del TTFT | `docs/superpowers/specs/2026-08-XX-engine-fase-c3a-design.md` esiste e copre i 6 punti del DONE WHEN §1; entry `RULING RICHIESTO` nel docket | none | docs/superpowers/specs/ | **done (it.2)** — `2026-08-01-engine-fase-c3a-design.md`, ruling pendente a docket item 6 |
| 3 | Repack all'import: l'import produce l'artefatto repacked (layout dai kernel, non dal GGUF), con versioning e invalidazione; i kernel MoE leggono il nuovo layout | `npm test` verde su produzione dell'artefatto + invalidazione su SHA/versione mismatch; `npx tsc --noEmit` pulito; bench JSON con **pack CPU < 1.0 ms/token** nel path caldo (baseline C2: 42.5); correttezza invariata (argmax ≡ cpuref-f64 256/256 sul campione ratificato) | none | src/engine/{gguf,quant,shape}.ts + import/OPFS, kernel MoE lato lettura pesi, tests/ | **done (it.4-7)** — limiti derivati (it.6) + repack (it.7): pack 42.5→**0.0** ms/token, decode 4.640→**4.912**. Guadagno netto 21.6 non 41.4: read+upload crescono di 20.7 per cache miss (causa misurata, spec §2.1). Conformance full-corpus NON eseguita (~5 h): sostituita da verifica byte-identica VRAM + file, resta per la fase 6 |
| 4 | Sync router: eliminazione o batching dei 46 readback per token col meccanismo di spec | bench JSON con **≤ 2 sync CPU↔GPU per token di decode** (baseline 47), o la soglia ammortizzata fissata in spec col suo parametro di profondità dichiarato nel payload; routing conformance rigenerato **non peggiore** di `routing-conformance-glm47flash-2026-07-31.json`; `npm test` verde; **+ ri-misura obbligatoria di `gpuBusy` e del clock SM medio subito dopo** (è il pezzo utile dell'opzione (a) conservato dentro la (b): dice quanto della forbice 10.2-15.6 si è chiusa da sola coi clock, e dimensiona la fase 4b con un fatto invece che con l'assunzione ottimistica). A fine fase: **ripresentare al PI il docket item 2** (clausola di fallback, rimandata qui dal ruling) | none | src/engine/ (router/moe, gpuforward, scheduling dispatch), tests/ | **in corso (it.9, it.15)** — it.9: router top-4 su GPU (`routerTopKWgsl`, fedeltà misurata). it.15: **slice A del design `2026-08-03-engine-fase4-strato1-arena.md`** — arena a binding fisso, Sel+dynamic offset, identità bit-a-bit, contatori invariati (46 sync/47 submit/1405 dispatch), decode 5.081. It.16: **slice B** — router GPU in ombra, gate **99.9991% su 1.44M confronti full-corpus** (13 flip near-tie sul 4° expert, rate 9e-6 = R8), Sel di produzione 0/5.75M difformi (R5 chiuso), decode 5.163. It.17: **slice C** — 1 submit / 0 sync per token a residenza totale (ktest, observer indipendente), selMiss 0, precondizione con rifiuto esplicito, decode 5.166 / prefill 5.747 (entrambi record). **MECCANISMO COMPLETO; il gate ≤2 sync/token è misurabile solo a residenza totale ⇒ la fase chiude INSIEME alla 4c** (che il blocked-by sblocca ora: il suo valore era nel meccanismo, che esiste). Item 2 ripresentato al PI (docket). Ri-misure gpuBusy/clock fatte a ogni iterazione (53.8-56.6 a 863-892 MHz, CSV committati) |
| 4b | **Guerra ai dispatch (quarta leva, emendamento 1)**: ridurre il tempo GPU per token col meccanismo di spec — batching delle 4 catene expert in un dispatch, fusione dei GEMV di attention, meno pass per layer. Dimensionata dalla ri-misura che chiude la fase 4 | bench JSON con **`gpuBusy` ≤ 54.5 ms/token** (baseline it.1 78.2; soglia derivata dall'aritmetica del gate) + dispatch/token e clock SM medio riportati e **attribuiti separatamente** (quanto viene dai kernel, quanto dai clock che salgono); correttezza invariata (argmax ≡ cpuref-f64 256/256, top-1 vs golden ≥ 98.83%); `npm test` verde | none | src/engine/kernels/, src/engine/ (moe, gpuforward, glmmodel), tests/ | **done (it.10-14)** — it.10 famiglia fusa expert (dispatch 1818→1450); it.12 flash-decoding MLA (attn 51.2→27.5); it.13 famiglia fast K-quant (shexp 14.6→5.5, head 9.6→3.8, dispatch 1405, decode 5.054); it.14 conformance reale: **gpuBusy 54.2 ≤ 54.5** (attribuzione: tutto kernel, clock 863 MHz contro), **argmax ≡ cpuref-f64 256/256** sul campione ratificato, top-1 golden 99.22% (full-corpus a fase 6, item 11) |
| 4c | **Residenza totale (emendamento 4)**: azzerare il deficit di VRAM che tiene 135 expert su 2944 fuori dallo slab, con quant asimmetrica sugli expert più freddi (matrice usage C1) e nuova versione di layout dello slab. È la precondizione perché il binding fisso di spec §3.2-bis elimini il readback invece di ridurlo | bench JSON con **hit-rate expert = 100%** e **0 miss/token** a decode nel contesto dichiarato, e il contesto scelto (525 o 4096) **scritto nel report** col deficit corrispondente (0.67 / 1.03 GiB); **eval di perdita** eseguita e riportata: top-1 vs golden **≥ 98.83%** full-corpus e argmax ≡ cpuref-f64 sul campione ratificato — sotto la soglia si docketa, non si assorbe; invalidazione della versione di layout coperta da test come in fase 3; `npm test` verde, `tsc --noEmit` pulito | authority: quant asimmetrica + nuova versione di layout slab (era must-docket) | src/engine/{quant,slabfile,glmsource}.ts + import/OPFS, src/engine/residency.ts, tests/ | blocked-by-4 |
| 5 | Prefill batched M>1 per MoE (insiemi di expert diversi per token) | test di identità logits M=1 vs M>1 sullo stesso prompt entro la tolleranza di spec, verde in `npm test`; bench JSON con prefill tok/s e TTFT misurati sullo stesso p6 | none | src/engine/ (prefill path, prefillplan, moe batched), tests/ | blocked-by-docket-6 |
| 6 | E2E & GATE: bench quiescente coi due gate di chiusura + i numeri di gap + correttezza invariata + non-regressione del pregresso | bench JSON quiescente con **decode ≥ 13.43** e **prefill ≥ 56.58 tok/s**, e coi campi di gap di fase 1 presenti (assenza = FAIL); conformance logits (argmax ≡ cpuref-f64 256/256; top-1 vs golden ≥ 98.83% full-corpus) e routing ≥ valori C2; conformance fase A verde; bench Qwen ≥ baseline 2026-08-01 (K=8 321.88, K=1 243.97, prefill chunked ≤ 600.2 ms); `npm test` ≥ 220 passed, `tsc --noEmit` pulito | none | src/engine/, results/engine/ | blocked-by-3,4,4b,4c,5 |
| 7 | Chiusura: docket con gli input per C3b (residenza residua dopo il repack, hit-rate osservato, nuovo profilo ms/token scomposto), direction §7 + ideas-ledger §A ove stale, HANDOFF refresh, merge+push | checklist DONE WHEN completa nel journal; verifier gate finale PASS; merge+push eseguiti (ruling permanente 2026-07-29) | none | docs/, .harness/, HANDOFF.md | blocked-by-6 |

Note:
- **La fase 4c nasce dall'emendamento 4** (ruling PI 2026-08-03, docket item 8):
  il PI paga ~0.67 GiB di qualità per la residenza totale. Sta DOPO la 4 perché
  il suo valore è nel meccanismo che la 4 costruisce (binding fisso + offset
  aritmetico): senza quello, la residenza totale è capienza sprecata. Sta PRIMA
  della 6 perché il gate finale gira sul motore che si vuole spedire.
  Il done-when chiede il contesto dichiarato nel report perché il deficit non è
  una costante: 0.67 GiB a ctx 525, 1.03 a ctx 4096.
- **Fase 1 è loop-runnabile subito** e non tocca il path caldo: gira mentre il
  ruling di spec non esiste ancora. Non è solo strumentazione — è la misura che
  la fase 2 usa per SCEGLIERE il meccanismo dei sync, oggi l'unico numero
  mancante (i 158.9 ms/token di struttura sono sync *più* 1.816 dispatch, senza
  split noto). Scriverla dopo la spec sarebbe scegliere alla cieca.
- **La fase 4 è il rischio del goal.** Il tetto senza di essa è ~7 tok/s
  (docket C2 item 8) ⇒ il gate 13.43 vive o muore qui. Può sforare le 4
  iterazioni: in quel caso lo split della fase va a docket, non preso nel loop
  (PHASES si modifica solo via ruling dopo iteration 0).
  **Aggiornamento it.1**: la misura dice che la fase 4 da sola NON basta —
  leve 1+2 al 100% danno 10.18 tok/s, e `gpuBusy` (78.2 ms/token) eccede da
  solo il budget del gate (74.46). La forbice reale è 10.2-15.6 tok/s e
  dipende da quanto i clock GPU salgono quando spariscono le bolle (misurati
  1746 MHz medi su 3105 di cap, utilizzo 34.6%). Docket item 4.
  **Ruling PI 2026-08-01 (item 4b)**: la quarta leva entra nel perimetro ⇒
  fase 4b. Il gate 13.43 resta invariato.
- **Nessuna clausola di fallback ANCORA**: il PI l'ha rimandata a fine fase 4
  (docket item 2), quando la ri-misura dei clock avrà sciolto la forbice.
  La fase 4 ha in done-when l'obbligo di ripresentarla.
- Se il decode resta sotto 13.43 a fine goal, il contratto non autorizza né
  scorciatoie né auto-deroga — si docketa (la clausola è rimandata, v. sopra).
- Gate non-regressione: HARD (contratto v1 C2, principio permanente ruling PI
  2026-07-31). Regressione non assorbibile ⇒ docket, non merge.
- Ogni bench va eseguito a **macchina quiescente** con lo stato host dichiarato
  nel report (norma PI 2026-08-01).
