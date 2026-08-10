# Journal — engine-fase-d

## it.0 (2026-08-10) — goal-setup

- Goal APERTO dopo il ruling PI 2026-08-10 (direction §7-ter) e la
  riapertura di q1 (docket q1 item 14): la generalizzazione senza parità di
  ottimizzazioni non è una generalizzazione.
- PHASES: 9 fasi, ordine del BLOCCO A dal ROI MISURATO in q1
  (gap-decomposition §4), non dall'ordine storico di GLM: parametrizzazione
  → slab pre-impacchettati → decode multi-step → prefill batched → policy
  MoE → checkpoint di misura. Blocco B (spec-dec MTP) sopra la parità.
- Regola di misura scritta NEL PIANO: micro-bench durante lo sviluppo,
  bench pieni solo alle fasi 6 e 8. È la regola violata in q1.
- Fase 1 = la più rischiosa (tocca residency/moe, il core indurito da
  C3a/b/c): paracadute = ktest bit-exact GLM, minuti non ore.
- Next: it.1 = fase 1 (parametrizzazione del core + test di non-duplicazione).

## it.1 (2026-08-10) — fase 1 slice A: moe.ts PARAMETRICO + gate strutturale

- `moe.ts` riscritto come meccanica UNICA con tre parti parametriche:
  (1) tabella di geometria dei formati quant (q4_0/q4_1/q8_0/q4_K/q5_K/q6_K:
      pesi e byte per blocco, forma del repack split vs monolitico) — non
      un `if` per famiglia;
  (2) `routerSelect(logits, bias, cfg)` con `RouterConfig`: GLM =
      sigmoid+bias+scale 1.8, qwen35moe = softmax puro senza bias né scale.
      Le due semantiche restano quelle verificate sulle rispettive fonti,
      ma il CODICE è uno solo (tie-break indice-minore condiviso);
  (3) `mkSlabLayout(id, gate, up, down)`: UNICO builder. Le classi GLM
      (SLAB_DOWN_Q4_0/Q4_1) ora sono DERIVATE da lui — e il test verifica
      che i byte storici siano invariati (5.308.416 / 5.505.024).
  `packExpertSlab` parametrico: legacy → 2 segmenti (qs+scales), K-quant →
  segmento unico paddato a word.
- Campi piatti del layout (gateQs/…/downScalesBytes) TENUTI come compat
  DERIVATA (una sola fonte di verità): i ~100 call site GLM non cambiano.
  Da rimuovere quando migrano — registrato a docket.
- `tests/engine-one-mechanism.test.ts` — GATE STRUTTURALE (pattern
  gpudevice.test): scansiona `src/engine/**` e vieta un secondo router, un
  secondo layout di slab, una seconda arena/LRU. Implementato come RATCHET:
  la lista delle duplicazioni note può solo ACCORCIARSI (rosso se cresce,
  rosso anche se sparisce senza aggiornare la lista). Stato oggi:
  router = [q35cpurefmodel (ECCEZIONE DICHIARATA PERMANENTE: il riferimento
  del differential testing DEVE essere indipendente), q35gpumodel],
  slab = [q35gpumodel], arena = [q35gpumodel]. Le tre voci q35gpumodel
  sono il lavoro delle slice B/C di fase 1.
- FINDING dal test: q4_0 e q4_K hanno gli STESSI byte/peso (0.5625) — il
  controllo di taglia del packer non li distingue; il tipo lo detta il
  layout, che viene dal GGUF validato. Annotato nel test.
- GATE: ktest **84/84**, GLM BIT-IDENTICO al pre-refactor (moe-ffn Σw
  1.800000, glm-model-2layer L2rel 2.07e-7, router-top4 0 flip,
  layer0-conformance 2.35e-7); suite 382 (+7); tsc pulito.
- Next: it.2 = fase 1 slice B — `residency.ts` parametrico (park/slot
  table/arena/eviction su config di modello invece che su G.*), poi la
  migrazione di q35gpumodel che fa sparire due voci del ratchet.

## it.2 (2026-08-10) — fase 1 slice B: residency.ts parametrica NEI DERIVATI (titolo corretto dopo il verifier: il MOTORE della cache è ancora GLM-shaped)

- `MoeModelConfig` (id, nLayer, denseLead, nExpert, nExpertUsed, classes,
  classOf(layer), layout(cls)) è ora la struttura da cui la residenza
  DERIVA tutto: chiavi (`expertKeyFor`), parco (`moeParkOf`), minimi
  (`minSlotsOf`), entry della slotTable (`slotTableEntriesOf`).
- `ExpertClass` da unione chiusa `"q4_0"|"q4_1"` a STRINGA: le classi le
  detta il modello, non il file. Fallout tipizzato dove serviva
  (`Record<ExpertClass, …>` nei record per-classe di glmmodel; il kernel
  legacy `gemvAccumFast` ha un narrowing ESPLICITO con errore parlante —
  esiste solo per q4_0/q4_1, e ora si vede).
- `MOE_CFG_GLM47` = il modello-tesi come CONFIGURAZIONE. I simboli storici
  (PARK_*, MIN_SLOTS, SLOT_TABLE_ENTRIES, expertKey, classOf statico)
  restano come compat DERIVATA — una sola fonte di verità.
- `ExpertCache` prende `opts.cfg` (default GLM): pins, slotTable, chiave
  inversa, minimo bindabile e `occupied` ora vengono dalla config, non da
  `G.*`. Aggiunti `classOfLayer()` e `keyOf()` di istanza.
- TEST: `engine-one-mechanism` esteso — i valori storici GLM (parco 2944 =
  2688+256, MIN_SLOTS, slotTable) sono verificati come DERIVATI, e una
  config qwen35moe (40 layer, 256 expert, top-8, classi q4_K/q6_K) produce
  dallo STESSO codice parco 10240 = 37×256 + 3×256 e minimi 8×37 / 8×3 —
  i numeri misurati in q1.
- GATE: ktest **84/84** con GLM BIT-IDENTICO (glm-model-2layer L2rel
  2.07e-7, arena-vs-slotrange "BIT-A-BIT identico", layer0 2.35e-7);
  suite 384 (+2); tsc pulito.
- Next: it.3 = migrazione di `q35gpumodel` a ExpertCache+mkSlabLayout+
  routerSelect ⇒ due voci del ratchet spariscono (arena e slab), il router
  resta solo nel cpuref (eccezione dichiarata).

### it.2 — correzioni dal verifier (stesso giorno, PRIMA di proseguire)

Il verifier ha dato PASS ma con tre rilievi che accolgo per intero: il titolo
"residency PARAMETRICA" era un OVERCLAIM — corretto sopra.

1. **Parametrizzazione a metà, e ora lo dice il codice.** Derivati (chiavi,
   parco, minimi, slotTable) sì; ma il motore della cache costruisce ancora
   gli stati di classe da `SLAB_DOWN_Q4_0/Q4_1`, e `expertSlots`,
   `arenaNeeds`, `ensure()`, `repinPass`, `stats`, `destroy` iterano la lista
   letterale delle classi GLM. `keyOf()`/`classOfLayer()` non avevano NEMMENO
   UN chiamante. Conseguenza reale: una cfg qwen sarebbe stata accettata in
   silenzio e sarebbe esplosa dopo con un TypeError. AGGIUNTO UN GUARD che
   la RIFIUTA rumorosamente citando la slice C. La parametrizzazione vera è
   il lavoro di it.3.
2. **Il gate strutturale era disonesto su un punto**: l'asserzione "il router
   vive solo in moe.ts" era già FALSA — `src/engine/cpuref.ts` (il cpuref
   GLM) ha un terzo router che la regex non prendeva. Sanato: l'eccezione è
   ora una CATEGORIA (`/cpuref/` = riferimenti indipendenti per contratto,
   sia GLM sia Qwen), non una lista di file.
3. **Limite noto del gate, registrato**: i predicati sono FIRME TESTUALI, non
   invarianti non aggirabili come il `requestDevice` di gpudevice.test — il
   verifier ha provato che una copia con spaziatura diversa sfugge. Da
   irrobustire PRIMA di dichiarare done la fase 1 (docket item 4).
4. Altri due blocker registrati a docket: i campi compat di `SlabLayout`
   fabbricano un offset `scales` finto sui K-quant (trappola per it.3), e
   `expertSlots` ripartisce il budget sul parco GLM (finché non è cfg-driven,
   q35 potrebbe passare solo da `slotsOverride`, cioè da un bypass).

Processo: digests.md era vuoto e il verifier è arrivato dopo DUE iterazioni
invece che dopo ognuna — entrambe le cose sanate qui.
