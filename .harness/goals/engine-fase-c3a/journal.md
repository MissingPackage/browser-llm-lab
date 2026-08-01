# Journal — engine-fase-c3a

## it.1 (2026-08-01) — fase 1: strumentazione TTFT + attribuzione sync vs dispatch

**Passo scelto**: fase 1 di PHASES (l'unica `ready`; le 3-6 sono gated dal ruling
di spec). Obiettivo dichiarato: misurare lo split dei 158.9 ms/token di
"struttura" — l'unico numero che manca per SCEGLIERE in spec il meccanismo dei
sync, invece di sceglierlo alla cieca.

### Cosa è stato costruito

- **`glmmodel.ts` — telemetria di attribuzione, opt-in a due livelli.**
  Liv.1 (CPU): `encodeCpuMs`, `ensureMs`, `routerWaitMs`, `tailWaitMs`, più
  **sync e submit contati davvero** (fino a qui `syncsPerToken: 46+1` era una
  costante dichiarata nel report, non una misura — ora è confermata dalla
  misura: 46.0 sync e 47.0 submit per token).
  Liv.2 (GPU): `timestampWrites` su ogni compute pass, un `resolveQuerySet` per
  token dentro il submit finale, `mapAsync` **dopo** il submit — è il
  root-cause del known-issue di fase A (tsq-diag-2026-07-29), rispettato.
  Zero-overhead da spenta: senza `telemetry` non si chiama `performance.now()`.
- **Deroga consapevole al pattern Qwen**: là le repliche liv.2 girano su un
  *secondo engine dedicato*; qui un secondo modello GLM raddoppierebbe la VRAM
  e non ci sta a slab 12 GiB. Quindi la CAPACITÀ è allocata al build e la
  REGISTRAZIONE si accende a runtime (`setTelemetry`): le repliche headline
  restano a overhead nullo, quelle di attribuzione sono separate e dichiarate,
  il loro wall non entra nei gate.
- **`glmbench.worker.ts`**: TTFT misurato, sezione `objective` coi tre gap
  (30 / 60 thinking / 4 s), attribuzione ortogonale
  `wall = gpuBusy + stallo + sync/CPU`, proiezione per K, **probe del floor di
  sync** (mapAsync round-trip a GPU scarica) come cross-check indipendente, e
  un **check di identità** dei segmenti.
- **`page.ts` / `glm-bench-run.mjs`**: `?attrib=N` / `--attrib N`; il runner
  stampa gap e attribuzione e urla se `objective` manca (FAIL di checklist).

### Difetto trovato e corretto DENTRO l'iterazione

Il check di identità ha fatto il suo lavoro alla prima run: `encodeCpuMs`
contava due volte `ensureMs` (il segmento di encode non veniva avanzato oltre
il blocco `ensure`) — somma dei segmenti 260.9 contro wall 209.4. Corretto
(`tSeg += durata_ensure`) e **bench ri-eseguito**, così l'artefatto committato
corrisponde a HEAD. Dopo il fix: somma 214.34 vs wall 215.05, **non attribuito
0.70 ms = 0.3%**. `encodeCpu` vero è **3.5 ms/token**, non 55.6.

### Misure (run quiescente, slab 12 GiB, p6 461 token, nGen 64, 3 repliche + 1 di attribuzione)

Artefatto: `results/engine/bench-glm-4090-b12-attrib-2026-08-01.json`.
Stato host campionato durante la run (`/tmp/glmbench-c3a-it1b-gpu.csv`, 65
campioni a 5 s): clock SM medio **1746 MHz**, max 2250, cap 3105; utilizzo GPU
medio **34.6%**; 61-66 °C.

Headline (telemetria SPENTA), invariato rispetto al baseline C2:
- decode **4.678 tok/s** (C2: 4.640) — nessuna regressione dalla telemetria
- prefill **5.235 tok/s** (C2: 5.221)
- **TTFT 88.06 s** vs budget UX 4 s ⇒ **gap 22.0×**
- gap decode: **6.41× dai 30 tok/s**, 12.82× dai 60 (thinking)

Attribuzione del decode (finestra strumentata, wall 215.0 ms/token):

| voce | ms/token | quota |
|---|---|---|
| `gpuBusy` (somma durate dei 94 pass) | **78.2** | 36.4% |
| stallo residenza (pack 41.4 + read 4.8 + upload 7.8) | **53.8** | 25.0% |
| sync/CPU (residuo) | **83.0** | 38.6% |
| **fuori GPU** | 136.9 | **63.6%** |

Segmenti disgiunti (identità chiusa): encode CPU 3.5 · ensure 54.1 ·
**routerWait 148.8** · tailWait 8.0. Overflow del ring timestamp: **0**.

**Probe del floor di sync**: mapAsync round-trip a GPU scarica **0.16 ms** (p50)
⇒ 46 × 0.16 = **7.6 ms/token** irriducibili. Cioè: degli 83 ms di sync/CPU,
**meno del 10% è il readback vero**; il resto è latenza di submit e bolle.

### Il risultato che cambia la fase 2

Proiezione con **le leve 1 e 2 del contratto al 100% di efficacia** (repack che
azzera il pack CPU, sync ridotti al loro floor misurato):

- **98.2 ms/token = 10.18 tok/s** — il gate 13.43 vuole ≤ **74.46 ms/token**.
- `gpuBusy` **da solo (78.2) eccede il budget del gate**.

⇒ **Le tre leve del contratto non bastano per costruzione.** Non è una stima
congetturale: è il wall misurato meno i due termini che le leve rimuovono.

**Ma l'osservazione discriminante ribalta a metà la conclusione.** Ho campionato
i clock proprio per questo: la GPU gira a **1746 MHz medi su 3105 di cap**, con
utilizzo 34.6% — è sotto-clockata *dalle bolle stesse* che la leva 2 va a
togliere. Se `gpuBusy` scalasse col clock SM (limite ottimistico: i GEMV sono
in buona parte memory-bound e il clock memoria è già al massimo), diventerebbe
44.0 ms ⇒ **64.0 ms/token = 15.63 tok/s, gate PASS**.

Il vero valore sta nella forbice: **10.2 – 15.6 tok/s**, col gate 13.43 dentro.
La leva 2 ha un **payoff di secondo ordine** (far salire i clock) potenzialmente
grande quanto quello di primo ordine, e nessuno dei due era dimensionato.

**Terzo cross-check, indipendente**: i pesi letti per token sono **2.22 GB**
(calcolo dai tensori: 47 layer di attention + 46 × (4 expert + shared) + head
Q6_K); la banda reale del device è 576 GB/s (9001 MHz × 2 × 256 bit, letta dal
device) ⇒ floor memory-bound **3.85 ms/token**. `gpuBusy` misurato è **20×
sopra**. Quindi il tempo GPU non è un muro fisico: a 1816 dispatch/token siamo a
43 µs per dispatch su GEMV che dovrebbero costarne ~2. C'è una **quarta leva**
(granularità/fusione dei dispatch) con margine enorme — ma è kernel
engineering, che direction §2 mette deliberatamente per ultima.

### Done-when della fase 1: soddisfatta, con due deviazioni di forma

1. ✅ TTFT e i tre gap nel report — `objective.gapDecode30/gapDecode60/gapTtft4s`,
   `ttftMs`.
2. ✅ Attribuzione sync vs dispatch col metodo dichiarato nel payload
   (`attribution2` + `caveats` + `identity`), **ma dentro lo stesso JSON** e non
   in un secondo file come diceva la riga PHASES. Sostanza consegnata, forma no.
3. ✅ `npm test` 220 passed + 2 skipped (= baseline), `npx tsc --noEmit` pulito.
4. ⚠️ La riga PHASES chiedeva "run glmbench **exit 0**": impossibile in fase 1,
   perché exit 0 significa gate PASS e i gate passano solo in fase 6. La run è
   uscita **4** = report scritto + gate FAIL, che è l'esito corretto qui.
   Correzione della riga a docket item 4 (non presa nel loop).

**Il controfattuale a selezione registrata NON è stato costruito**: la
strumentazione con timestamp-query ha dato lo split cercato in modo diretto e
meno invasivo (nessuna modifica alla semantica del forward), quindi
l'esperimento più pesante non serviva. Scelta di metodo, dichiarata.

### Verifica

Fatta in proprio e meccanica: `npx tsc --noEmit` pulito; `npm test` 220 passed
+ 2 skipped (identico al baseline C2); identità della strumentazione chiusa allo
0.3%; headline non regredito (4.678 ≥ 4.640 di C2); overflow del ring 0;
`timestampQuery.used = true`. **Verifier indipendente NON lanciato**: la policy
di sessione vieta di spawnare subagent senza richiesta esplicita del PI — il
gate del protocollo resta formalmente scoperto per questa iterazione ed è
segnalato qui, non aggirato in silenzio.

### Prossimo passo

Fase 2 (spec) è **bloccata da un ruling**: la scelta del meccanismo dei sync ora
dipende da quale ramo della forbice si assume, e la sufficienza delle leve del
contratto è smentita. Docket item 4.
