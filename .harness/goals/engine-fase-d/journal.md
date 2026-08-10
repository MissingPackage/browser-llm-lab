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

## it.3 (2026-08-10) — fase 1 slice C: il MOTORE della residenza è cfg-driven

Chiusi i quattro blocker del verifier (docket item 4 a/c/d + f; il gate
strutturale irrobustito (4b) resta a it.4).

- **(a) stati di classe dalla config**: il costruttore non conosce più
  `SLAB_DOWN_Q4_0/Q4_1` — cicla `cfg.classes` e chiede `cfg.layout(c)`.
- **(b) `expertSlots` cfg-driven**: la ripartizione del budget usa il parco
  DELLA CONFIG (`moeParkOf`) e i byte dei SUOI layout. Conseguenza che
  contava: q35 non deve più passare da `slotsOverride`, cioè da un bypass.
  Per GLM la formula è identica ⇒ numeri identici (verificato).
- **(c) niente più liste letterali**: `arenaNeeds`, `repinPass`, `stats`
  (occupied/slots/pinSlots/pinCap) e `destroy` iterano `cfg.classes` /
  `Object.entries(this.cls)`. I tipi pubblici delle stats sono diventati
  `Record<ExpertClass, number>`.
- **(d) path CALDO parametrico**: `ensure()`, `noteSelection()` e
  `debugMarkMiss()` usano `this.classOfLayer()` e `this.keyOf()` — i due
  metodi che a it.2 non avevano nemmeno un chiamante.
- **(e) GUARD RIMOSSO**: la config ora è onorata davvero, non serve più
  rifiutarla.
- **(f) campi compat**: `gateScales`/`upScales`/`downScales` sono diventati
  GETTER che FALLISCONO su un layout K-quant ("non esiste sul formato
  q4_K: usa la vista generica") invece di restituire un offset finto con
  binding di taglia 0 — la trappola che il verifier aveva individuato.
- TEST nuovi (3): `expertSlots` ripartisce sul parco della config qwen
  (37/40 vs 3/40) e GLM resta invariato; `arenaNeeds` dimensiona sulle
  classi della config (finestra ≥ slab della classe più grande); i campi
  compat falliscono sui K-quant e restano corretti sui legacy.
- GATE: ktest **84/84** GLM BIT-IDENTICO (2layer L2rel 2.07e-7,
  arena-vs-slotrange BIT-A-BIT, layer0 2.35e-7, moe-ffn Σw 1.800000);
  suite 387 (+3); tsc pulito.
- Next: it.4 = gate strutturale da firme testuali a invariante non
  aggirabile (docket 4b) — l'ultimo blocco prima di poter dichiarare la
  fase 1; poi la migrazione di q35gpumodel alla ExpertCache unica.

### it.3 — rilievi del verifier chiusi subito (it.4, prima di proseguire)

Verifier PASS con 5 rilievi; i 4 minori chiusi immediatamente:
- `flushSlotTable` usava la costante GLM `SLOT_TABLE_ENTRIES` per
  l'intervallo sporco: su una config con parco diverso riscriveva ~7k
  entry per flush invece di quelle sporche (dato corretto, banda sprecata).
  Ora usa `t.shadow.length` — la shadow è la verità.
- getter compat resi NON enumerabili: `{...layout}` e `JSON.stringify` su
  un layout K-quant lanciavano con un messaggio fuorviante (trappola
  latente per la telemetria delle fasi 2-5).
- `slotsOverride` con chiavi di un'altra config costruiva una cache a
  ZERO classi in silenzio: ora le chiavi si validano contro `cfg.classes`.
- docket item 4 annotato con lo stato reale (a/c/d CHIUSI, b unico residuo).
Il quinto (gate strutturale da irrobustire) È il lavoro di it.4.

## it.4 (2026-08-10) — fase 1: il gate strutturale è un INVARIANTE (docket 4b CHIUSO)

Il gate di it.1 era fatto di FIRME TESTUALI ricalcate sul testo degli
offender: il verifier ha dimostrato che una copia con spaziatura diversa
sfuggiva. Riscritto sul pattern vero di `gpudevice.test` — intercettare
qualcosa che NON SI PUÒ EVITARE, con allowlist motivata:

- **INVARIANTE A (arena/slab)**: per mettere un expert in VRAM servono per
  forza (i) i NOMI GGUF dei tensori expert (`ffn_{gate,up,down}_exps` —
  convenzione llama.cpp valida per OGNI famiglia MoE) e (ii) la creazione
  di buffer GPU. Chi fa entrambe DEVE importare moe.ts/residency.ts.
- **INVARIANTE B (router)**: un router MoE fedele DEVE applicare il clamp
  di `build_moe_ffn` (6.103515625e-5). Chi scrive il letterale invece di
  importare `WEIGHTS_SUM_CLAMP_MIN` sta riscrivendo il router.
- **ALLOWLIST CON RAZIONALE** (come gpudevice.test): ogni voce dice perché.
  Un test verifica che i razionali non siano vuoti e che le voci marcate
  "DEBITO NOTO" siano SOLO q35gpumodel — la fase 1 non chiude finché ci sono.
- **IL GATE PROVA SE STESSO** (anti-marciume): 3 test danno ai predicati
  offender SINTETICI e pretendono che scattino — inclusa la variante di
  spaziatura che sfuggiva a it.1 — e controesempi che NON devono scattare
  (chi importa la meccanica, chi nomina i tensori senza toccare la VRAM).
  Senza questo, un predicato può marcire in un no-op restando verde.
- Test aggiunti anche per i due fix di it.3 senza copertura: `slotsOverride`
  con chiavi di un'altra config FALLISCE; JSON/spread di un layout K-quant
  NON esplodono (getter non enumerabili).
- GATE: ktest **84/84** GLM bit-identico; suite **391** (+4); tsc pulito.
- STATO FASE 1: docket item 4 completamente CHIUSO (a/c/d in it.3, b qui).
  La fase NON è ancora done: resta la migrazione di `q35gpumodel` alla
  meccanica unica, che il gate ora ESIGE (le sue due voci di allowlist sono
  marcate DEBITO NOTO e vanno rimosse). È it.5, e chiude la fase 1.
