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

## it.5 (2026-08-10) — il gate diventa un invariante VERO (docket item 4 CHIUSO)

Il verifier ha bocciato la versione di it.4 ESEGUENDO tre evasioni in buona
fede, tutte passate verdi. Il difetto di fondo: avevo fatto una CONGIUNZIONE
di due indizi nello stesso file, mentre `gpudevice.test` intercetta UN ATTO
SINGOLO. Riscritto:

- **tre invarianti INDIPENDENTI, ognuno su un atto singolo**: (A) l'allocazione
  di memoria GPU (`createBuffer` — l'analogo esatto di `requestDevice`);
  (B) i NOMI dei tensori expert, con la regex che prende anche le forme
  costruite per parti (`ffn_${p}_exps`); (C) il clamp del router.
  Ciascuno con la propria allowlist motivata (8+8+5 voci).
- **niente più esenzione per import**: importare qualsiasi cosa da
  moe/residency non è più un lasciapassare (era l'evasione 3).
- **niente congiunzioni da spezzare**: nomi in un file e buffer in un altro
  fanno scattare due invarianti diversi (era l'evasione 2).
- **guard anti-scansione-vuota** (`SRC.length > 30`), che gpudevice.test ha e
  a it.4 mancava.
- **il ciclo delle asserzioni è esso stesso testato** (buco M2): `offendersOf`
  viene esercitato con predicati sintetici, quindi renderlo vacuo fa rosso.
- VERIFICA MIA, non solo del verifier: ho riprovato le tre evasioni creando
  file temporanei in `src/engine/` — **tutte e tre ora fanno ROSSO**, e
  rimosse tornano verdi con zero file sporchi.
- Processo: docket item 4 e HANDOFF allineati allo stato reale (il verifier
  li aveva trovati stale — stessa svista di it.3, questa volta chiusa nella
  stessa iterazione).
- GATE: suite **396** (correzione: avevo scritto 392, il verifier l'ha rilevato); tsc pulito; ktest invariato (nessun file di src toccato
  in questa iterazione: solo il test e i documenti).
- STATO FASE 1: docket item 4 CHIUSO per intero. Resta la migrazione di
  `q35gpumodel`, che il gate ora ESIGE (tre voci DEBITO NOTO). È it.6 e
  chiude la fase 1.

## it.6 (2026-08-10) — l'invariante si sposta nel SISTEMA DI TIPI (e la pretesa del gate si ridimensiona)

Il verifier ha bocciato ANCHE la versione di it.5, con **cinque** evasioni
eseguite. La più importante non è un'evasione: **N1 = un router Qwen**
(softmax puro, niente clamp, niente nomi di tensori, niente VRAM) è
invisibile a ogni impronta testuale — ed è codice che DEVE esistere dopo la
migrazione. Diagnosi, dopo tre tentativi sbagliati sullo stesso bersaglio:

> Fallisce perché sto cercando di rilevare un MECCANISMO duplicato tramite
> IMPRONTE TESTUALI: ogni impronta è o specifica della forma GLM (il clamp) o
> aggirabile con un refactoring ordinario. Mostrato da: le 5 evasioni.
> `gpudevice.test` funziona perché `requestDevice` è un nome della
> PIATTAFORMA — non esiste altra porta. Per la residenza expert quella porta
> non esisteva: va CREATA, non cercata.

Cosa ho fatto:
- **MARCHIO DI CONIO su `SlotRef`** (`residency.ts`): campo con
  `unique symbol` ⇒ uno SlotRef può essere CONIATO solo dentro residency.ts
  (un solo sito, marcato). Chi prova a fabbricarne uno per spacciare la
  propria arena come residenza **viene rifiutato da `tsc`** — il verifier
  l'ha confermato con 11 sonde ostili (letterale, interfaccia gemella,
  `class implements`, `Object.assign`, helper generico, cast diretto: tutti
  ROSSI). È l'evasione N2, chiusa dal compilatore invece che da una regex.
  **LIMITE, nella formulazione corretta dal verifier**: il marchio ferma la
  CONTRAFFAZIONE, non l'INDIFFERENZA — un'arena che semplicemente non usa
  SlotRef non viene sfiorata, ed è il caso di `q35gpumodel` finché la fase 1
  non chiude. Diventa portante con it.7. Bypass noti senza cast: inflow
  any-tipizzato e spread di uno SlotRef genuino (si sorveglia il conio, non
  la circolazione). Tutto scritto nei file, non solo qui.
- **`tests/types/slotref-brand.ts`**: test DI TIPO con `@ts-expect-error`.
  Se qualcuno togliesse il marchio, la direttiva diventerebbe inutilizzata e
  **tsc andrebbe rosso**. L'invariante si sorveglia da solo, in un gate che
  il progetto esegue già.
- **Scansione allargata e ancorata**: tutto `src/` (non solo engine), via
  `__dirname` (non la cwd), estensioni `.ts/.mts/.cts/.tsx`. Uccide N3 e N4.
  Ha subito trovato `src/microbench/runner.ts` — dichiarato con razionale.
- **`allocatesGpu`** su identificatore e non su token con parentesi: uccide
  N5 (`createBuffer.bind(dev)`).
- **PRETESA RIDIMENSIONATA, per iscritto nel file**: quel test NON garantisce
  "una meccanica, una implementazione" — non può, perché la differenza fra
  duplicazione e seconda famiglia legittima è semantica. È un RATCHET su
  impronte note che alza il costo di una deriva distratta e tiene il debito
  dichiarato. L'invariante vero è il marchio. Il `describe` è stato
  rinominato di conseguenza.
- Correzione: suite 392 → **396** nel journal di it.5 (numero stale rilevato
  dal verifier).
- GATE: suite **396 passed | 9 skipped** (misurata, non derivata: il verifier
  ha bocciato it.6 anche per un 397 dedotto per aritmetica — seconda
  occorrenza della stessa classe di errore, i conteggi si LEGGONO dall'output);
  tsc pulito (incluso il test di tipo); nessun tocco ai
  path GLM oltre il marchio, che è additivo.

**LEZIONE, e cambio di rotta**: ho speso tre iterazioni a costruire un
poliziotto quando la cosa che elimina davvero la duplicazione è LA
MIGRAZIONE. Il gate serve contro le derive future, non come precondizione.
Da qui si migra `q35gpumodel` (it.7) e la fase 1 si chiude lì.

## it.6b (2026-08-10) — sanatoria del FAIL del verifier su it.6

Due rilievi, entrambi fondati, entrambi sanati PRIMA di procedere:
1. **Numero dedotto invece che letto**: avevo scritto suite 397 per
   aritmetica (396 + il test di tipo, che però non è un caso vitest). Reale:
   **396 passed | 9 skipped**. Seconda occorrenza: da qui i conteggi si
   copiano dall'output, mai si calcolano.
2. **Sovravendita spostata, non eliminata**: avevo ridimensionato la pretesa
   del test a scansione e poi ri-gonfiata sul marchio, scrivendo che
   "un'arena parallela viene rifiutata da tsc". FALSO come scritto:
   `q35gpumodel.ts` gestisce oggi un'arena expert completa (byKey/LRU,
   `slotLoc`, bind group propri) SENZA toccare `SlotRef`, e tsc è verde.
   La frase vera è più stretta: il marchio ferma chi **contraffà** uno
   SlotRef, non chi lo **ignora**; diventa portante solo quando il binding
   expert passa da `SlotRef`/`slotBindRanges`, cioè con it.7. In più il
   verifier ha trovato due bypass NON dichiarati (inflow any-tipizzato;
   spread di uno SlotRef genuino, che conserva il marchio). Riqualificato
   in `residency.ts`, `tests/types/slotref-brand.ts`, docket item 4,
   digests, HANDOFF.

## it.7 (2026-08-10) — FASE 1 CHIUSA: q35 migrato alla meccanica unica

Il lavoro che elimina davvero la duplicazione (non il gate: la migrazione).

**Cosa è sparito da `q35gpumodel.ts`**: arena a chunk (`mkChunks`/`slotLoc`),
`byKey`+`lru` proprie, `ensure` proprio, `rpU8`, i bind group ricostruiti a
ogni miss, il calcolo a mano di `slotBytes`/`parkSlots`/`nSlots`, e il router
softmax+topK+clamp ricopiato dal cpuref. **Cosa c'è al posto**: la stessa
`ExpertCache` che serve GLM, con una `MoeModelConfig` dedotta dal GGUF, e
`routerSelect(logits, null, {...ROUTER_QWEN35MOE, nUsed: topK})`.

**Nuovo modulo `src/engine/q35expertstore.ts`** — gemello di `expertstore.ts`
(GLM): nomina i tensori expert, ne legge le fette per-expert, costruisce la
config. È così che le TRE voci DEBITO NOTO sono sparite DAVVERO invece di
essere rietichettate: il gate ora asserisce `debiti == []`.

**Aggiunte al meccanismo condiviso** (`residency.ts`), tutte generiche:
- `isResident(layer, expert)`: peek puro (niente LRU touch, niente stats). Il
  35B non sta in RAM: il forward guarda chi manca, `await`ta SOLO i miss e poi
  chiama `ensure` (sincrona) coi byte in mano. Senza, si leggerebbe sempre.
- `slotTensorRanges(slot)`: la vista a TRE range, valida per entrambe le
  famiglie. `slotBindRanges` (sei range) resta legacy e lancia sui K-quant.
- `maxBindRangeOf(layout)` + **BUG CORRETTO**: il controllo del limite di
  binding leggeva i campi compat (`qsBytes` = il GATE, più i due scale). Sui
  K-quant il segmento più grande è il DOWN (Q6_K 868.352 B contro 589.824):
  sfuggiva al controllo, e su un device stretto il bind sarebbe fallito a
  runtime invece che alla costruzione. Su GLM il valore non cambia.

**PROVE (tutte fresche, nessuna dedotta)**
1. `tests/q35-slab-parity.test.ts` **6/6** (CPU-only, permanente): offset e
   taglie identici all'aritmetica cancellata (gate@0, up@589.824,
   down@1.179.648; 1.769.472 / 2.048.000 B) e `packExpertSlab` == i tre
   repack a mano **parola per parola**. La migrazione non sposta un byte.
2. ktest GPU reale **84/84, 0 non-pass**. GLM BIT-IDENTICO: `glm-model-2layer`
   L2rel 2.07e-7 + argmax 6/6 (identici a it.3), `expert-arena-vs-slotrange`
   maxRel **0** "BIT-A-BIT identico", `moe-ffn` Σw=1.800000,
   `glm-layer0-conformance` 2.35e-7. `q35-model-4b-argmax` 6/6.
3. **Conformance smoke 35B sul path MIGRATO** (run di CORRETTEZZA, non bench —
   la piena da 2 h resta alla fase 6): **top1 5/5 = 100%**, e il confronto col
   run PRE-migrazione dello stesso golden è **identico nei numeri che contano**:

   | | pre (q1) | post (it.7) |
   |---|---|---|
   | top1 | 5/5 | 5/5 |
   | hits | 8846 | **8846** |
   | misses | 3314 | **3314** |
   | uploadedBytes | 5.916.950.528 | **5.916.950.528** |
   | nSlots q4k/q6k | 6657 / 539 | 6735 / 471 |

   Stesso routing, stessi miss, stessi byte caricati: la residenza si comporta
   in modo indistinguibile. L'unica differenza è la ripartizione del budget fra
   classi (il meccanismo condiviso riparte in proporzione al parco, il codice
   vecchio usava una `frac` unica) — non ha cambiato un solo miss qui, ma è una
   differenza REALE che va detta. Tempo prompt 54,4 s contro 59,3 s: NON lo
   chiamo speedup, è una misura singola non pinnata, e i tempi si misurano
   alla fase 6.
4. suite **402 passed | 9 skipped** (letta dall'output), `tsc --noEmit` pulito,
   gate 21/21 con `debiti == []`.

**Effetti collaterali reali**: i bind group ora nascono una volta per SLOT
(dipendono dall'indirizzo, non dall'expert che ci abita) invece che a ogni
miss; `destroy()` libera l'arena, che prima restava allocata.

**FASE 1 CHIUSA.** Done-when verificate una per una: core parametrico ✓,
arena q35-only rimossa ✓, test di non-duplicazione in `npm test` ✓ (con la
pretesa dichiarata onestamente: ratchet + marchio), ktest tutti PASS con GLM
bit-exact ✓. Prossimo: fase 2 (slab pre-impacchettati all'import).

## it.8 (2026-08-10) — fase 2: il repack costava 6x quello che doveva

**Come e' iniziata**: invece di scrivere subito il generatore di slab
all'import (la lettera del done-when), ho MISURATO. Micro-bench CPU sui due
slab reali del 35B: pack 2,94 ms (q4_K) e 2,49 ms (q6_K) per miss, cioe'
600-820 MB/s. Un memcpy fa 10 volte tanto: il numero diceva che il problema
non era DOVE stava il repack, ma COME era scritto.

**Poi la scomposizione vera**, esponendo la telemetria che `ExpertCache` ha
gia' (`timing: true` su q35 — 4 `performance.now()` per MISS, non per token e
non sugli hit): sullo smoke 35B, `packMs` **11.585 ms su 3314 miss = 3,50
ms/miss**, il **22% dei 53 s** di prompt. `uploadMs` 0,12 ms/miss.
CAVEAT ONESTO: `readMs` risulta ~0 ma NON e' l'I/O — la lettura Range avviene
nell'`await` prima di `ensure`, quindi cade fuori dalla finestra cronometrata.
Il costo di I/O resta non misurato (docket per la fase 5, dove serve).

**Due cause, entrambe reali**:
1. `repackKQuant` ricostruiva ogni parola a 32 bit con un `|=` per BYTE. Ma
   su little-endian `out[j>>2] |= src[j] << (j&3)*8` scrive esattamente il
   byte j al suo posto: e' una COPIA travestita da aritmetica. Con `set()`
   diventa memcpy. La forma scalare resta come DEFINIZIONE del risultato (e
   per big-endian, dove non e' una copia).
2. `packExpertSlab` allocava un `Uint32Array` temporaneo per tensore e poi lo
   ricopiava nello slab: ogni byte toccato TRE volte (zero-fill, copia nel
   temp, copia nello slab). Nato `repackKQuantInto`, che scrive dritto nello
   slab.

**RISULTATI**

Micro-bench CPU (20 ripetizioni, slab reali):

| | prima | dopo | fattore |
|---|---|---|---|
| pack q4_K (1.769.472 B) | 2,94 ms | 0,44 ms | **6,7x** |
| pack q6_K (2.048.000 B) | 2,49 ms | 0,58 ms | **4,3x** |

End-to-end, smoke 35B, stesso golden (`corpusHash` verificato uguale):

| | PRIMA | DOPO |
|---|---|---|
| top1 | 5/5 | 5/5 |
| hits / miss | 8846 / 3314 | **8846 / 3314** |
| packMs totale | 11.585 | **1.856** (6,24x) |
| pack per miss | 3,50 ms | **0,56 ms** |
| uploadMs per miss | 0,12 | 0,13 |
| prompt | 53,0 s | **42,5 s** |

I 9,7 s di pack risparmiati spiegano 9,7 dei 10,5 s di prompt in meno: il
delta e' attribuibile al meccanismo, non e' rumore di host. Resta una misura
singola per lato — il numero su cui mi impegno e' `packMs`, che e'
telemetria diretta, non orologio.

**Correttezza**: `tests/quant-repack-fast.test.ts` (NUOVO) confronta il
repack con un oracolo scalare INDIPENDENTE riscritto nel test, su blockBytes
144/176/210 e a due offset di sorgente, parola per parola; piu' un caso che
verifica che i 2 byte di padding di OGNI superblocco Q6_K restino ZERO con
sorgente tutta a 0xff (sbagliarlo non darebbe un errore, darebbe pesi
leggermente diversi — il bug che si vede solo come qualita' che cala).
Nota sulla catena di prove: l'oracolo di `q35-slab-parity` usa
`repackKQuant`, quindi da solo non varrebbe piu'; vale perche' il test nuovo
lega `repackKQuant` a un oracolo scalare indipendente.

**GATE**: suite **406 passed | 9 skipped**; tsc pulito; ktest **84/84,
0 non-pass**, con i valori IDENTICI cifra per cifra a it.7 (glm-model-2layer
maxRel 1.309250593300995e-4, arena-vs-slotrange maxRel 0,
q35-moe-block-real blk0 5.319934364985386e-6 e blk34 1.2826589096342132e-5):
il repack veloce e' bit-a-bit lo stesso repack.

**DONE-WHEN: obiettivo centrato, lettera no** — il repack NON e' uscito dal
path di miss, e' diventato ~6x piu' economico dentro. Spostarlo all'import
comprerebbe il residuo (~1,7 s su 45, ~4%) al prezzo di ~18 GB di slab su
disco. Non decido io un cambio di contratto: **docket item 5**, col mio
parere (non conviene) e coi numeri.

## it.10 (2026-08-10) — fase 3 sui DENSI [NUMERI CORRETTI IN it.12: il -15,0 era un ARTEFATTO DI MISURA, il valore onesto e' -3,3 — vedi it.12]

**Misurato prima, di nuovo.** Strumentata la scomposizione per token
(`Q35GpuModel.perf()`: embed CPU / readback / argmax CPU) e girata sul 4B:
`readbackMs` 44,6 ms/token. **NON l'ho letto come "il trasferimento di 604 KB
costa 44 ms"**: quell'attesa su `mapAsync` include tutto il lavoro GPU del
token. Il numero che serviva era un altro — quanto CPU e GPU si aspettano a
vicenda — e l'ho preso con un micro-bench nel ktest: gli stessi token
accodati con `read=false` e una sola attesa alla fine.

| 4B, 39 token | ms/token | **[CORRETTO in it.12]** |
|---|---|---|
| con sync per token (`step`) | ~~50,5~~ | **40,0 a caldo** (il 50,5 era la PRIMA passata dopo il load) |
| senza sync (accodati, una attesa) | 35,4 | 35,4 |
| **tetto recuperabile** | ~~15,0~~ | **4,58** |
| embed dequant CPU | 0,07-0,12 | idem |

**Fatto**: `Q35GpuModel.decodeBatch(tokens, posStart)` — K token
teacher-forced in UN submit, argmax su GPU (`argmaxStage1/2`, gli stessi
kernel di `gpuforward`), un readback di K·4 byte invece di K readback da
604 KB. Le righe di embedding e gli uniform dei K step si impacchettano in un
buffer e si copiano nello scratch DENTRO l'encoder: `queue.writeBuffer` e'
ordinata prima del submit, quindi scriverli in ciclo darebbe a tutti gli step
l'ultimo valore (stesso schema di `dbSlots` in gpuforward). Error scope
esplicito, come ogni encode nuovo.

**RISULTATO [CORRETTO in it.12]**: ~~35,5 contro 50,5 = -15,0 (-29,6%)~~ —
ARTEFATTO DI MISURA (braccio lento a freddo, braccio veloce a caldo). Valore
onesto, a caldo e interleavato: **36,69 contro 40,02 = -3,33 ms/token
(-8,3%)**, che sta 1,25 ms SOPRA il pavimento senza sync (ne recupera il
73%, non "praticamente tutto"). **Sotto il >= -5,1 che il done-when
chiedeva** → docket item 9.

**GATE SECCO (quello che il contratto chiede)**: nel ktest
`q35-model-4b-argmax`, gli argmax di `decodeBatch` sono confrontati uno a uno
con quelli del path a readback su tutti i 39 token → **IDENTICI**, ed e' una
condizione di `pass` (sono interi: o sono uguali o non lo sono). ktest
**84/84, 0 non-pass**.

**ERRORE MIO, CORRETTO IN CORSA**: avevo messo la conformance a batchare
TUTTI i token e il risultato era piu' LENTO (1,75 → 1,82 s). Motivo: li' il
prefill gira gia' con `read=false`, senza attesa — batcharlo non toglie
niente e AGGIUNGE l'argmax su GPU su token che non lo useranno. Corretto:
si batcha solo lo span di cui si legge l'argmax. Sullo smoke resta NEUTRO
(1,83 vs 1,75, dentro il rumore di una run da 1,8 s con 33 prefill su 39) e
lo dico: il guadagno non e' li'. Il corpus pieno del 4B e' 27.312 token di
prefill contro 1024 posizioni generate — 96% prefill. La conformance resta
sul path batch non per velocita' ma perche' cosi' il golden VALIDA
`decodeBatch`: argmax generati identici a PRIMA, verificato.

**IL MoE NON PUO' USARLO**, e `decodeBatch` e' `null` li' — non un'omissione,
una proprieta': la selezione degli expert legge i logits del router su CPU a
OGNI layer, quindi sul 35B sono **41 submit e 41 readback per token**. E' la
stessa situazione che GLM aveva prima del lavoro "47 → 2 sync/token" della
fase C, e si risolve con il resolve su GPU (slotTable + kernel d'arena +
router su GPU) — che `residency.ts` gia' offre (`arena: true`,
`slotTable: true`) ma che q35 non usa. **E' il pezzo piu' grosso rimasto del
goal** e non e' una riga della fase 3: docket item 8.

**GATE**: suite 410 | 9 skipped, tsc pulito, ktest 84/84, conformance 4B
smoke top1 6/6 con argmax identici al path precedente.

## it.11 (2026-08-10) — fase 3b, fetta 1: kernel K-quant con indirizzamento d'arena

`gemvQ4KWgsl`/`gemvQ6KWgsl` accettano ora un `KArenaOpts`
{nBuf, slabWords, slabsPerBuf, tensorWords}: il buffer di classe si binda
INTERO, lo slot arriva da `Sel` e il kernel ricava da solo (binding, base).
Riusata la testa di GLM (`SEL_STRUCT`, `MoeIdx`, `ld4`, `arenaSlotWgsl`), non
riscritta: `arenaHeadWgsl` ora accetta il sottoinsieme strutturale, perche' un
tensore K-quant e' UN segmento e non ha i campi scale di GLM. Un solo corpo
aritmetico per i due regimi: cambia solo da dove arrivano le parole (`blkw`).

GATE NUOVO sul 35B reale: `q35-arena-vs-slotrange-blk0-q4kdown` e
`blk34-q6kdown` — l'uscita degli 8 expert calcolata coi due regimi di
indirizzamento deve essere **BIT-A-BIT identica** (`Object.is` su tutti i
2048 f32). **PASS su entrambe le classi.** ktest 86/86.

NOTA OPERATIVA (costata mezz'ora): le modifiche a `ktest.worker.ts` sono
andate perse una prima volta perche' il verifier, ripulendo il working tree
dai suoi file di prova, l'ha riportato a HEAD. **Non lasciare lavoro non
committato mentre un verifier gira.**

## it.12 (2026-08-10) — IL -15,0 ms ERA UN ARTEFATTO DI MISURA. Valore vero: -3,3

Il verifier ha bocciato it.10 e ha ragione. L'errore era di METODO:
`msTokenConSync` era la **PRIMA passata dopo il load del modello**, a freddo;
`msTokenBatch` era la terza, a caldo. Circa 8 ms/token di warm-up stavano
tutti dentro il braccio lento, e il delta usciva gonfiato di ~5x. Il verifier
l'ha dimostrato cronometrando una quarta passata — path identico, bit per
bit — che costava 39,9 ms contro i 48 della prima.

**Micro-bench riscritto**: una passata di warm-up SCARTATA, poi i tre bracci
INTERLEAVATI e ripetuti 3 volte, e si riporta la MEDIANA con min-max. Non un
campione singolo.

| 4B, a caldo, 3 ripetizioni interleavate | ms/token | [min-max] |
|---|---|---|
| `step` con sync per token | **40,02** | 40,0-40,2 |
| `decodeBatch` | **36,69** | 36,7-36,8 |
| accodato senza sync (pavimento) | 35,44 | 35,4-35,5 |
| **delta batch** | **-3,33 (-8,3%)** | spread 0,19 / 0,10 |

**Cosa cade e cosa resta**:
- CADE "-15,0 ms/token (-29,6%)": era l'artefatto. Corretto in journal it.10,
  digests, HANDOFF, PHASES e docket.
- CADE "cade esattamente sul tetto, recupera praticamente tutta la
  serializzazione": il batch sta 1,25 ms SOPRA il pavimento senza sync. Ne
  recupera 3,33 su 4,58, cioe' il 73%. Il verifier ha osservato che in
  4 run su 4 il batch stava sopra il pavimento e che il mio 35,5-vs-35,4 era
  l'unico campione in cui coincidevano: aveva ragione anche li'.
- RESTA il gate secco: **argmax IDENTICO** su 39 token, dentro il `pass`.
- RESTA che la serializzazione vera, a caldo, e' 4,58 ms/token (40,02-35,44),
  non 15.

**IL DONE-WHEN DELLA FASE 3 NON E' SODDISFATTO**: chiedeva >= -5,1 ms/token,
misurati -3,33. Non lo assorbo e non riscrivo il contratto: **docket item 9**.

Regola nuova per l'harness (osservazione del verifier, vale oltre questa
fase): il primo passaggio dopo `createQ35GpuModel` non e' comparabile con
nulla. Ogni micro-bench futuro — incluso quello del prefill alla fase 4, che
e' formulato allo stesso modo — deve scartare una passata e interleavare i
bracci. Registrato a docket item 10.

## it.13 (2026-08-10) — fase 3b, fetta 2: router+resolve QWEN su GPU, fedele

`routerTopKWgsl` diventa parametrico sul GATING: `sigmoid` (GLM, col bias di
selezione) o `softmax` (qwen35moe). E' la trascrizione in WGSL dello stesso
`RouterConfig` che `moe.ts` porta su CPU, non una seconda verita'.

Scelta di forma che vale la pena registrare: il binding `bias` resta
dichiarato in ENTRAMBI i casi e chi non lo usa ci lega un buffer di ZERI.
`probs[i] + 0.0` e' esatto in floating point, ed e' letteralmente cio' che
`routerSelect` fa con `sel.set(probs)` quando `usesBias` e' false. Cosi' il
layout dei binding non dipende dalla famiglia e — verificato — con
`gating: "sigmoid"` il testo emesso resta BYTE-IDENTICO a prima.

Il softmax ha bisogno del massimo globale prima di esponenziare, quindi il
prefill parallelo mette via i logit grezzi e la normalizzazione la fa il
thread 0: tre passate su 256 expert, niente rispetto alle 256x8 della
selezione che segue.

**GATE NUOVO `q35-router-resolve-gpu-vs-cpu`** (64 estrazioni, 256 expert,
top-8), stessa metodologia del caso GLM — separazione come gate, non
tolleranza cieca:

| | |
|---|---|
| flip d'insieme (sep >= 1e-5) | **0** |
| flip d'ordine | **0** |
| resolve errati (slot/flag/peso in Sel) | **0** |
| errore relativo max sui pesi | **2,32e-7** (soglia 1e-5) |
| separazione minima retta | 1,92e-5 |

Il resolve si prova nello stesso caso: una slotTable finta mappa i
selezionati su slot noti e **uno di loro su MISS di proposito**, e si pretende
che `Sel` riporti slot e flag esatti — slot e flag sono interi, quindi il
confronto e' secco, non una tolleranza. E' l'unico punto in cui l'indirizzo
dell'expert smette di passare dalla CPU: se sbagliasse, il kernel leggerebbe
byte di un altro expert senza che nulla fallisca.

ktest **87/87, 0 non-pass**; GLM invariato (`router-top4-gpu-vs-cpu` maxRel
1.6400254653154082e-7, `router-top4-near-tie` tenuto fino a eps=1e-6).

Prossima fetta (3): cablare tutto in `q35gpumodel` — arena + slotTable +
router su GPU per layer, miss rilevato su GPU (`dirty`) e repair+replay dalla
CPU ⇒ 1 submit/token a residenza piena.

### Progetto della fetta 3 (cablaggio in `q35gpumodel`) — scritto PRIMA, per riprendere da disco

Ricognizione fatta su `glmmodel.ts`, che ha gia' tutto: la fetta 3 e' un
PORT, non un'invenzione. I pezzi, nell'ordine in cui vanno montati:

1. **Cache in regime arena**: `new ExpertCache(device, {..., arena: true,
   slotTable: true})`. Cambia la taglia dei buffer (bindati interi) e accende
   la tabella expertKey→slot in VRAM. `arenaGeometry(cls)` da' gia'
   {layout, nBuf, slabsPerBuf, slabWords, nSlots}: e' esattamente ciò che
   serve per costruire i `KArenaOpts` dei kernel di it.11.
2. **Bind group layout ESPLICITO** per le pipeline expert e per il router:
   `hasDynamicOffset: true` NON e' esprimibile con `layout: "auto"`
   (glmmodel.ts:556 e 1052 lo fanno cosi'). `moeIdx` viaggia come uniform a
   dynamic offset con `minBindingSize: MOE_IDX_BYTES` e `size` ESPLICITA nel
   binding (glmmodel.ts:1007 dice perche': con hasDynamicOffset la
   validazione la pretende).
   ⇒ 3 bind group per classe (gate, up, down) + 1 per il router, e l'offset
   dinamico seleziona la entry di Sel. NON 320 bind group.
3. **`Sel`**: nLayerMoE x topK entry da 16 B. `moeIdx = {selIdx, tableBase,
   moeLayer, pad}` con stride `MOE_IDX_STRIDE` (256, allineamento uniform).
4. **Router per layer**: `routerTopKWgsl({nExpert, nUsed, weightsScale: 1,
   clampMin, gating: "softmax", resolve: {..., dirty: true}})`, bindings
   [routerLogits, biasZeri, ids, wts, selBuf, slotTable, moeIdx, dirtyB].
5. **`dirty`**: [0] = primo layer MoE con miss (atomicMin, sentinella
   0xffffffff azzerata dalla CPU per token), [1] = conteggio miss (atomicAdd).
6. **Esecuzione**: un encoder per token — per ogni layer: segmento statico
   (attn + shexp + gemv del router) → routerTopK+resolve → topK x (gate, up,
   silu, down, axpy) che leggono Sel → add. Alla fine argmax. UN readback:
   dirty + argmax.
7. **Repair+replay**: se `dirty[1] > 0`, la CPU rilegge Sel, fa `ensure` sugli
   expert con flag MISS, `flushSlotTable()` e RIESEGUE dal layer `dirty[0]`.
   A residenza piena non scatta mai ⇒ 1 submit/token.

**Ordine di montaggio proposto** (ogni passo con un gate proprio, come le
fette 1 e 2): (3a) arena + Sel riempita dalla CPU, tutto il resto invariato —
gate: bit-identico al path attuale sul 35B; (3b) router su GPU in OMBRA (Sel
di produzione ancora dalla CPU, quella GPU in una regione parallela, e si
confrontano) — gate: selezione e pesi entro le tolleranze di it.13 sui layer
VERI; (3c) Sel di produzione dal router GPU + dirty + repair/replay — gate:
argmax identico e submit/token misurati.

Rischio principale identificato: il regime arena cambia la TAGLIA dei buffer
di classe (bindati interi ⇒ cappati anche da maxStorageBufferBindingSize, non
solo da maxBufferSize). Sul 35B con budget 12 GiB significa piu' buffer per
classe, e `nBuf` entra nei kernel come numero di binding: va verificato che
stia nel limite di binding per bind group del device (glmmodel.ts:551 ha
gia' un assert di questa forma).

## it.14 (2026-08-11) — fase 3b fetta 3a: q35 in regime d'ARENA (Sel dalla CPU)

**Ruling PI incassato prima di iniziare**: docket item 9 chiuso con l'opzione
(a) — la riga 3 di PHASES non chiede piu' `>= -5,1 ms` ma il delta MISURATO a
caldo con dispersione. La fase 3 e' chiusa sui densi anche per contratto.

**Cosa e' cambiato** (il PORT progettato in it.13, passo 3a):

1. `ExpertCache` di q35 nasce con `arena: true, slotTable: true`. Non cambia
   ne' quanti slot ci sono ne' quanta VRAM costano (`expertSlots` non guarda il
   regime): cambia che i buffer di classe si bindano INTERI.
2. Nascono `Sel` (40x8 entry da 16 B, tutto il token — non un layer: il layout
   non deve cambiare quando i 40 submit diventeranno uno) e `MoeIdx` statica
   (40x8 entry a stride 256, `{selIdx, tableBase, moeLayer}` note al load).
3. **Bind group layout ESPLICITO** per la catena expert: `hasDynamicOffset` non
   si esprime con `layout: "auto"`. Risultato: **3 bind group per classe**
   (gate, up, down) invece di 3 per slot — l'indirizzo non sta piu' nel bind
   group, sta in `Sel`, e l'offset dinamico sceglie quale entry leggere.
4. `runLayer` scrive `Sel` (id, `slot.idx`, peso) dopo gli `ensure` e chiama
   `flushSlotTable()`. L'indirizzo continua a venire da uno `SlotRef` coniato
   da `residency.ts`: cambia la RAPPRESENTAZIONE (l'indice globale dello slot
   invece di buffer+offset), non la provenienza — il marchio di conio resta
   sul path.
5. `moeLayerAbs[]`: il layer ASSOLUTO di ogni segmento MoE. Oggi coincide con
   l'indice del segmento (`denseLead: 0`), ma la chiave della slotTable e la
   classe dello slab si leggono col layer assoluto: tenerlo esplicito evita che
   un modello della famiglia con layer densi in testa indirizzi la classe
   sbagliata SENZA fallire.
6. `SEL_BYTES`/`MOE_IDX_BYTES`/`MOE_IDX_STRIDE` **si spostano in
   `kernels/wgsl.ts`**, accanto alle struct WGSL che descrivono, e glmmodel le
   importa da li'. Due famiglie che scrivono la stessa `Sel` con due costanti
   locali sono due ABI che nessun compilatore confronta.
7. `q35conf.worker` negozia i due limiti nuovi dell'arena (`arenaNeeds` con la
   config dedotta dal GGUF): buffer d'arena => `maxStorageBuffersPerShaderStage`,
   finestra => `maxStorageBufferBindingSize`. E il runner `q35-conf-run.mjs`
   guadagna `--arena-gib`, che HANDOFF gli attribuiva ma non aveva: il budget
   decide gli slot, quindi i miss — due bracci a budget diverso non sono
   confrontabili.

**IL RISCHIO DI it.13 ERA REALE E STA NEI NUMERI**: in regime arena la classe
q4k del 35B si spezza in 5-6 buffer, e ogni pipeline expert li binda TUTTI =
8-9 storage binding contro i 7 del path Qwen. Senza la negoziazione al punto 7
il device ne concede 8 di default e la pipeline sarebbe stata invalida. Il
controllo esplicito (`expertArenaBindings` > limite ⇒ errore parlante) c'e'.

**IL GATE** (bit-identita' end-to-end sul 35B vero, due bracci nella stessa
sessione, stesso host, stesso golden smoke, stesso budget):

| | prima (HEAD 863b641) | dopo (arena) | |
|---|---|---|---|
| top1 | 5/5 | 5/5 | identico |
| argmax generati | [248068, 271, 248069, 271, 29292] | idem | **identico** |
| routing (istogramma completo) | 3314 chiavi, 12160 selezioni | idem, **mappa uguale** | **identico** |
| hits / misses | 8846 / 3314 | 8846 / 3314 | identico |
| uploadedBytes | 5.916.950.528 | 5.916.950.528 | identico |
| nSlots q4k/q6k | 5613 / 393 | 5613 / 393 | identico |
| dispatch/token | 782 | 782 | identico |

Il numero che porta il peso e' **l'istogramma di routing**, non il top1 da 5
posizioni: sono 12.160 selezioni (38 token x 40 layer x 8) e ognuna dipende
dai logits del router, cioe' dallo stato nascosto prodotto dagli expert del
layer precedente. Un offset d'arena sbagliato darebbe un risultato plausibile
ma diverso, e la mappa divergerebbe al primo layer. E' uguale chiave per
chiave. Il braccio PRIMA e' girato da un `git worktree` su HEAD (niente `git
stash` con lavoro non committato: e' la landmine di it.11), con lo STESSO
runner copiato nei due bracci — cambia il motore, non l'harness.

**COSA IL GATE NON COPRE, detto**: con 6006 slot e 3314 expert distinti
toccati, i miss sono 3314 = il numero di chiavi di routing ⇒ ogni expert e'
stato caricato UNA volta e **non c'e' stata nessuna eviction**. Il riuso di
uno slot dopo eviction non e' esercitato qui. Attenuante strutturale: l'eviction
cambia i BYTE dentro uno slot, non il suo indirizzo, e l'aritmetica
slot -> (binding, base) e' gia' gateata bit-a-bit in it.11. Resta una casella
vuota da riempire alla fase 6 (dove il corpus pieno evince davvero).

**OOM E COSA HA INSEGNATO**: il primo tentativo, a budget 12 GiB (il default,
quello di it.7), e' morto con `VK_ERROR_OUT_OF_DEVICE_MEMORY` alla
costruzione dell'arena. Prima di toccare qualunque cosa ho preso l'osservazione
che DISCRIMINA: ho girato il braccio PRIMA — codice HEAD, non modificato —
allo stesso budget, ed e' morto allo stesso modo. Non e' la modifica: e' lo
stato VRAM dell'host (14,4 GiB liberi contro 12 di arena + densi + KV; it.7
girava con piu' margine). Entrambi i bracci sono quindi a **10 GiB DICHIARATI**.
Conferma aritmetica indipendente: `expertSlots` non guarda il regime e i buffer
di classe hanno la stessa taglia nei due regimi (min(maxBuffer, maxBinding) =
2.146.369.536 B ⇒ 1213 slab/buffer per q4k in entrambi), quindi l'arena non
chiede un byte in piu'. **Lezione per la fase 6**: il budget del 35B non e'
ctx-aware come quello di GLM (`slabBudgetCtxAware`) — e' un parametro fisso, e
a 12 GiB e' sopra il tetto di questo host. Va al docket (item 11).

**Tempi, e perche' NON sono un dato**: prompt 42 s (prima) contro 44 s (dopo).
E' UN campione per braccio, senza warm-up scartato e senza interleaving: per la
regola del docket item 10 non e' una misura, e non la chiamo ne' regressione ne'
rumore. Cio' che si puo' dire per costruzione: la fetta 3a AGGIUNGE due
writeBuffer per layer (Sel + flush della slotTable) e non toglie ancora niente,
perche' i 40 submit e i 40 readback del router sono ancora tutti li'. Il
guadagno e' la fetta 3c; questa fetta compra la PRECONDIZIONE.

**Gate permanenti**: `npx tsc --noEmit` pulito, `npm test` **410 passed | 9
skipped** (invariato), ktest GPU reale **87/87, 0 non-pass** — GLM compreso,
che con lo spostamento delle costanti e' il modo di verificare che l'ABI di
`Sel` non si sia mossa.

**Prossimo**: fetta 3b — il router GPU in OMBRA sui layer veri (Sel di
produzione ancora dalla CPU, quella GPU in una regione parallela dello stesso
buffer, e si confrontano). Il kernel c'e' gia' da it.13, la slotTable e'
popolata da questa fetta: manca il cablaggio e il confronto.

## it.15 (2026-08-11) — fase 3b fetta 3b: il router GPU in OMBRA sui layer veri

**Perche' questa fetta esiste anche se it.13 aveva gia' gateato lo stesso
kernel**: it.13 lo ha provato su 64 estrazioni SINTETICHE 256x8. La domanda
che la sintetica non poteva porre e' se la DISTRIBUZIONE vera dei logits del
35B produca margini sotto la risoluzione dell'f32 — la CPU sceglie in f64, la
GPU in f32, e dove due score sono piu' vicini della risoluzione le due
scelgono expert diversi. Il gate serve a rispondere con un numero.

**Cablaggio**: `Sel` RADDOPPIA — `[0, nSel)` produzione (la riempie la CPU),
`[nSel, 2nSel)` ombra (ci scrive il resolve GPU). Due regioni dello stesso
buffer, non due buffer: la fetta 3c cambiera' solo la entry di `MoeIdx` che
indirizza il kernel, non il kernel. `MoeIdx` guadagna una entry per layer col
`selIdx` gia' spostato nell'ombra. Il dispatch del router gira **nello STESSO
submit del segmento statico**, in coda al GEMV che ha appena scritto i logits
(dentro il `tail` dell'encoder: `steps[]` non sa fare offset dinamici e non
l'ho toccato). Il binding `bias` c'e' anche qui con un buffer di ZERI, come
deciso in it.13.

**Il confronto e' appaiato per EXPERT, non per posizione** — con un flip
d'ordine la posizione k porterebbe expert diversi, e il confronto per indice
darebbe un falso allarme sui pesi o, peggio, un falso via libera sugli slot.
I miss si prendono PRIMA degli `ensure` del layer: dopo, `isResident` direbbe
sempre si'.

**RISULTATI, smoke 35B, 38 token x 40 layer = 1520 confronti, 12.160 selezioni**:

| | misurato | soglia / atteso |
|---|---|---|
| flip d'insieme (top-8 diverso) | **0** | 0 |
| flip d'ordine | **0** | 0 |
| errore relativo max sui pesi | **3,80e-7** | < 1e-5 (it.13: 2,32e-7) |
| **margine minimo sui logit** | **1,0014e-5** | — e' il numero che qualifica il gate |
| slot risolti diversi dalla CPU | **0** | 0 |
| miss GPU / miss CPU / disaccordi | **3314 / 3314 / 0** | uguali |

**Il numero che conta e' il margine, non gli zeri.** 0 flip su 1520 layer non
dice niente se il caso peggiore era facile. Il caso peggiore aveva un margine
di **1,0e-5 fra l'ultimo expert preso e il primo scartato** (sui logit: la
softmax e' monotona, quindi e' li' che si decide l'ordinamento). Con logit
dell'ordine dell'unita', la risoluzione dell'f32 e' ~1e-6: il caso piu' stretto
che questo corpus ha prodotto sta **una decade sopra la risoluzione**, e la GPU
l'ha risolto come la CPU. **Cio' che NON e' provato, detto**: un margine sotto
~1e-6 sarebbe un testa-o-croce, e in questo run non ce n'e' stato nessuno.
Non si puo' concludere "il router GPU non flippera' mai": si puo' concludere
che su 12.160 selezioni vere non ha flippato e che il margine peggiore visto
e' 1e-5. Il gate va rifatto sul corpus pieno alla fase 6.

**Il resolve e' l'altra meta', e passa secca**: `slotMismatch = 0` significa
che per OGNI expert gia' residente la GPU ha letto dalla slotTable esattamente
lo slot che la CPU ha poi usato — cioe' che `tableBase` (layer assoluto x
nExpert) e il mantenimento della tabella dentro `ensure` sono giusti end-to-end,
non solo nel caso finto di it.13. E `missDisagree = 0` con 3314 miss per parte
significa che la residenza vista da GPU e quella vista dalla CPU sono la stessa,
miss per miss. L'ombra gira PRIMA degli `ensure` del suo layer, quindi un
expert che sta per essere caricato si risolve MISS: e' la residenza parziale
vista da GPU, ed e' anche la ragione per cui la fetta 3c pretende il
repair+replay invece di sperare.

**La produzione non si e' mossa**: routing, argmax generati e top1 sono
IDENTICI a quelli di it.14 (stesso confronto chiave per chiave). L'ombra e'
davvero un'ombra — un dispatch e una copia da 128 B per layer, spenta di
default (`routerShadow`, opt-in; `--shadow` nel runner).

**Gate permanenti**: tsc pulito, suite **410 passed | 9 skipped**, 0 gpu-error
nel run. ktest non ri-eseguito in questa iterazione: nessun kernel e' stato
toccato (il WGSL del router e' quello di it.13, generato con gli stessi
parametri) e il path GLM non e' stato sfiorato.

**Prossimo: fetta 3c** — la `Sel` di produzione la scrive il router GPU, entra
`dirty` (atomicMin sul primo layer con miss, atomicAdd sul conteggio) e il
repair+replay dalla CPU quando `dirty[1] > 0`. E' li' che cadono i 40 submit e
i 40 readback per token. Precondizione ora verificata: la selezione e il
resolve su GPU sono fedeli sui layer veri.

## it.16 (2026-08-11) — MISURA PRIMA DELLA 3c: quanti token sono sporchi?

**Perche' mi sono fermato a misurare invece di scrivere la 3c.** Leggendo il
path ottimistico di GLM per portarlo, ho trovato la sua PRECONDIZIONE:
`optimisticMinResidency` (default 0.8) — GLM RIFIUTA di costruirsi in modo
ottimistico sotto quella residenza, con un messaggio che dice perche': "in
regime di scarsita' ogni token e' sporco e il replay costa piu' del sync".
Il 35B a 10 GiB ha 6006 slot su un parco di 10.240 expert = **58,7%**. Sotto
la soglia di GLM. Se ogni token fosse sporco, la fetta 3c toglierebbe 40
submit e 40 readback per aggiungere N replay che ricalcolano il token da un
layer basso in giu': un cambio in perdita. Non e' una cosa da scoprire dopo
aver scritto 300 righe.

**Strumento** (opt-in, `--misstrace`): due passate SULLO STESSO prompt.
`resetState()` fra le due azzera lo stato ricorrente ma NON la cache expert,
quindi la passata 0 e' il caso freddo (peggiore) e la 1 e' il caso
perfettamente caldo (migliore). L'uso vero sta in mezzo e questi due numeri lo
delimitano. Smoke 35B, 39 token, 320 selezioni/token, budget 10 GiB:

| | pass 0 (freddo) | pass 1 (caldo) |
|---|---|---|
| token SPORCHI | **39 / 39** | **0 / 39** |
| miss/token, mediana | 68 | 0 |
| miss/token, primo token | 320 (tutto) | 0 |
| miss totali | 3341 | **0** |
| hit rate | 73,2% | **100,0%** |

**Le due conclusioni, che sono opposte e vanno tenute insieme**:

1. **A residenza raggiunta il decode ottimistico e' ESATTAMENTE cio' che
   serve**: zero token sporchi su 39 ⇒ zero replay ⇒ **1 submit/token**, che
   e' alla lettera il done-when (a) della fase 3b. E non e' una fortuna: i
   3341 expert distinti che questo prompt tocca stanno nei 6006 slot, quindi
   non c'e' nessuna eviction e la cache converge. Il caso "residenza piena"
   del contratto e' MISURABILE su questo stesso smoke, con la seconda passata.
2. **A cache fredda TUTTI i token sono sporchi**, mediana 68 miss su 320
   selezioni. Il repair+replay li' non e' un caso limite: e' il regime. Un
   token con 68 miss sparsi su 40 layer ha il primo layer sporco in basso, e
   il replay ricalcola quasi tutto il token — piu' volte (i round avanzano per
   prefisso). Il path ottimistico NUDO, applicato al prefill freddo, e' una
   regressione, non un guadagno.

**Conseguenza sul progetto della fetta 3c** (scritto qui PRIMA di iniziarla,
come per la fetta 3): la 3c non e' "porta il path ottimistico". E':

- (3c-i) **il path a submit unico** con `Sel` di produzione scritta dal router
  GPU, `dirty` (atomicMin sul primo layer MoE sporco + atomicAdd sul
  conteggio), `hiddenCkpt` (l'hidden di INGRESSO di ogni layer, copiato
  nell'encoder: e' l'input del replay) e repair+replay al confine di token.
  Pezzi noti: `clearBuffer` per azzerare `moeAcc` DENTRO l'encoder (oggi e'
  una `writeBuffer` per layer, che nell'encoder non si puo' fare); un axpy che
  legge il peso da `Sel.w` invece che da un buffer scritto dalla CPU (i pesi
  ora nascono su GPU); il guard `setInFlight` di `ExpertCache` fra submit e
  readback; la ricostruzione di `routing` dalla `Sel` letta in coda, perche'
  in questo modo NESSUNO sulla CPU ha visto la selezione del token.
- (3c-ii) **una POLICY di ingresso**, che e' la vera lezione di questa misura:
  il modo ottimistico si accende quando conviene e non per decreto. La forma
  piu' semplice che i numeri sostengono: si parte in modo sync (quello di
  oggi) e si passa a ottimistico quando i miss per token scendono sotto una
  soglia, con isteresi. La soglia si TARA sui numeri, non si inventa: il
  micro-bench deve dare il costo di un token sync (40 submit) e quello di un
  token ottimistico sporco (1 submit + R replay) sullo stesso host.
- **GATE della 3c**, con la metodologia di questa misura: due passate sullo
  stesso prompt, submit/token e readback/token riportati per PASSATA — freddo
  e caldo separati, mai mediati insieme (mediarli nasconderebbe esattamente il
  fenomeno che questa iterazione ha trovato). Piu' i gate secchi del contratto:
  argmax identico al path attuale sul campione, routing e conteggio miss
  identici.

**Nota sulla FEDELTA', che cambia rispetto alle fette precedenti**: dalla 3c i
pesi di mixing arrivano dal router GPU (f32) invece che dalla CPU (f64). it.15
ha misurato la differenza: 3,80e-7 relativo. Quindi la 3c **non e'
bit-identica per costruzione** e il suo gate non puo' essere la bit-identita':
e' l'argmax identico sul campione (done-when (b) del contratto), piu' routing
e miss invariati (done-when (d)). Va detto prima, non dopo aver visto i numeri.

**Gate di questa iterazione**: tsc pulito, suite 410|9, 0 gpu-error nel run,
JSON committato (`results/engine/q35-misstrace-35b-it16.json`). Lo strumento
resta: serve di nuovo alla 3c per il gate, e alla fase 5 per la policy.

### it.16 — CORREZIONE DI SCOPO, stesso giorno, prima di iniziare la 3c

Ho riletto il contratto invece di fidarmi di quello che avevo appena scritto, e
il pezzo (3c-ii) qui sopra E' FUORI POSTO. La riga 5 di PHASES si intitola
"**Decode ottimistico + policy sul MoE q35**" e dice, testualmente: "decode
ottimistico (1 submit/token, repair+replay) **attivo dove la residenza lo
consente**". La policy d'ingresso e' gia' assegnata, ed e' della fase 5.

E la riga 3b chiede solo: "(a) submit/token e readback/token MISURATI prima e
dopo nello stesso JSON, **con il caso a residenza piena a 1 submit/token**".
Il caso freddo NON e' un suo done-when.

Quindi la fetta 3c e' UNA cosa sola, non due:

- **3c** = il MECCANISMO: path a submit unico, `Sel` di produzione dal router
  GPU, `dirty`, `hiddenCkpt`, repair+replay al confine di token. **Opt-in**
  (come `select` in GLM), col path sync di oggi che resta il default: cosi' non
  puo' regredire niente mentre il meccanismo esiste ma non conviene ancora.
  Il gate misura ENTRAMBE le passate e le riporta separate — il caso caldo
  soddisfa il done-when, il caso freddo si riporta perche' e' vero, non perche'
  sia richiesto.
- **La soglia d'ingresso va alla FASE 5**, dov'era gia'. La misura di it.16
  (39/39 sporchi a freddo, 0/39 a caldo) e' il dato con cui la fase 5 la
  tarera': la lascio qui come input, non come lavoro da fare adesso.

Perche' scrivo la correzione invece di riscrivere il paragrafo: un pezzo di
fase 5 assorbito dentro la 3b sarebbe scope creep travestito da completezza —
la fase 3b avrebbe "finito" facendo anche altro, e la fase 5 sarebbe arrivata
con una decisione gia' presa altrove e senza i suoi numeri.

## it.17 (2026-08-11, fase 3b, fetta 3c) — il token intero in UN submit: 81 → 1

**Cosa e' cambiato.** La `Sel` di produzione non la scrive piu' la CPU: la
scrive il router su GPU, che risolve expert→slot dalla `slotTable` nello stesso
dispatch e marca `dirtyB` (atomicMin sul primo layer MoE con miss, atomicAdd sul
conteggio) quando trova un MISS. I 40 segmenti statici, i 40 router e i 320
dispatch expert stanno in UN encoder e UN submit; il routing lo ricostruisce la
CPU alla fine, dalla `Sel` copiata in coda, perche' mentre il token gira nessuno
sulla CPU vede la selezione.

**I kernel expert non sono cambiati di una riga.** E' il pagamento
dell'indirezione costruita nella fetta 3a: cambia CHI riempie `Sel`, non chi la
legge. L'unica differenza di cablaggio e' la entry di `MoeIdx` che l'offset
dinamico seleziona.

**I due pezzi nuovi, ed erano entrambi prevedibili dal progetto di it.16:**
- `clearBuffer(moeAcc)` DENTRO l'encoder. La `queue.writeBuffer` del path sync
  qui non funziona: e' ordinata prima dell'INTERO submit, quindi i 40 layer
  vedrebbero un solo azzeramento — l'ultimo. Questo non e' un dettaglio di
  stile: l'accumulatore sporco darebbe numeri plausibili e sbagliati.
- `axpySelWgsl`, l'axpy col peso di mixing preso da `Sel[selIdx].w`. Il peso ora
  NASCE su GPU; `axpyWgsl` lo legge da un buffer che nel path sync riempie la
  CPU dopo il readback del router, e qui non c'e' piu' nessuno a riempirlo. Sul
  MISS il contributo e' ZERO e non i byte dello slot 0 (che e' l'indirizzo di
  ripiego dei kernel d'arena): il token resta comunque sbagliato — un expert che
  manca cambia il risultato — ma sbagliato in modo DEFINITO.

**La contabilita' e' il pezzo che non si vede.** Nel path sync ogni selezione
passa da `ensure`, che conta hit/miss e TOCCA la LRU. Nel path a submit unico
`ensure` non viene chiamata mai: senza rimedio la recency degraderebbe
all'ordine di inserimento — la LRU diventerebbe una FIFO e le vittime
cambierebbero, cioe' il path nuovo misurerebbe miss diversi per un motivo che
non c'entra col meccanismo. Nasce `ExpertCache.noteResidentHit`, che al confine
di token conta l'hit e tocca la LRU esattamente come farebbe `ensure`, e LANCIA
se l'expert non e' residente (la `Sel` lo dava risolto: un disaccordo li' e'
eviction fra resolve e confine di token, bug strutturale).

**GATE, smoke 35B, 39 token, 320 selezioni/token, budget 10 GiB dichiarato.**
Tre regimi, riportati SEPARATI (mediarli nasconderebbe il fenomeno di it.16):

| | sync FREDDO | sync CALDO | ottimistico CALDO |
|---|---|---|---|
| submit/token | 81 | 81 | **1** |
| readback/token | 41 | 41 | **1** |
| miss | 3341 | 0 | **0** |
| hit | 9139 | 12480 | 12480 |
| ms/token | 1192,9 | **143,49** [142,51-145,94] | **71,50** [71,05-71,57] |

- **argmax IDENTICO 39/39** contro il path sync (e anche contro la passata
  fredda: 39/39).
- **routing IDENTICO chiave per chiave**: 3341 chiavi, 0 differenze.
- hit+miss = 12 480 = 39x320 in tutte e tre le passate: la contabilita' del
  confine di token conta esattamente quanto quella di `ensure`.

**Il numero di velocita' e' a PARITA' di residenza, e ci e' voluta una passata
in piu' per averlo.** Il primo giro del gate aveva due passate — fredda (sync) e
calda (ottimistica) — e dava "1192,9 → 71,5". Quel confronto e' freddo-contro-
caldo travestito da speedup: la passata fredda paga l'I/O di 3341 miss. Il
termine di paragone vero e' il path di OGGI sulla cache gia' calda, ed e' una
terza passata. Piu' il docket item 10: bracci INTERLEAVATI, 4 ripetizioni, prima
coppia scartata, mediana e dispersione. **−71,99 ms/token (−50,2%)**, con
dispersione 3,43 ms sul braccio sync e 0,52 su quello ottimistico: il delta non
e' rumore.

**Il done-when (a) e' soddisfatto alla lettera** ("submit/token e readback/token
misurati prima e dopo nello stesso JSON, col caso a residenza piena a 1
submit/token"), e con lui (b) e (d) sul campione caldo.

**Una correzione di fatto al docket item 8**: diceva "41 submit e 41 readback
per token". I readback erano giusti, i submit no — sono **81**: ogni layer MoE
ne fa due, uno per il segmento statico e uno per i dispatch dinamici. Il numero
non cambia nessuna decisione (l'ordine di grandezza e la conclusione erano
quelli), ma era una stima e adesso e' una misura.

**COSA NON E' COPERTO, DETTO PRIMA DELLA PROSSIMA FETTA.** Il path ottimistico
qui ALZA sul token sporco (degrado definito, invariante I2 di GLM: un token
sporco non si campiona). Serve dunque il repair+replay, ed e' la fetta
successiva. Portandolo ho trovato la sua precondizione, e non e' un dettaglio:

> **il replay di GLM non e' portabile su un modello RICORRENTE.** GLM rigioca i
> layer da `firstDirty` in giu' rientrando da `hiddenCkpt`, e questo e'
> idempotente perche' i suoi layer sono senza stato (il KV append riscrive la
> stessa posizione). Nel 35B **30 layer su 40 sono deltanet**, e
> `deltaNetConv`/`deltaNetCore` aggiornano `convSt` e `stateS` IN PLACE: un
> replay li applicherebbe DUE VOLTE. Un replay nudo non darebbe un errore —
> darebbe uno stato ricorrente sbagliato e numeri plausibili.

La via d'uscita c'e' ed e' economica, perche' ogni layer tocca il proprio stato
UNA volta per token: lo stato all'ingresso del layer E' lo stato all'ingresso
del token. Basta uno snapshot per token (nell'encoder, `copyBufferToBuffer`) e
un restore dei soli layer >= `firstDirty` all'inizio del replay — un solo
snapshot serve anche ai round successivi, che ripartono sempre piu' in basso.
Costo, calcolato non stimato: 30 layer x (conv 98 304 B + S 2 097 152 B) =
**62,8 MiB** di VRAM in piu' e una copia da 62,8 MiB per token dentro il submit
che c'e' gia'. Va MISURATO contro i 71,5 ms/token di qui, non assunto
trascurabile.

**Gate permanenti**: tsc pulito, suite 410|9, 0 gpu-error nel run, JSON
committato (`results/engine/q35-optimistic-35b-it17.json`).

## it.18 (2026-08-11, fase 3b) — repair+replay: la fase 3b si chiude, e il regime freddo dice una cosa che non mi aspettavo

**Cosa manca(va)**: la riga 3b chiede "miss rilevato su GPU con **repair+replay**
dalla CPU". it.17 rilevava il miss e ALZAVA. Questa iterazione ripara e rigioca.

**La precondizione, che GLM non aveva.** Il replay di GLM rientra da
`hiddenCkpt[firstDirty]` e rigioca i layer da li' in giu': idempotente, perche' i
suoi layer sono senza stato (il KV append riscrive la stessa posizione). Nel 35B
**30 layer su 40 sono deltanet** e aggiornano `convSt`/`stateS` IN PLACE:
rigiocarli applicherebbe l'aggiornamento DUE VOLTE — senza errore, con uno stato
ricorrente sbagliato e numeri plausibili. Lo snapshot costa poco perche' ogni
layer tocca il proprio stato UNA volta per token, quindi lo stato all'ingresso
del LAYER e' quello all'ingresso del TOKEN: **una copia sola a inizio token**
(62,8 MiB misurati: 30 x (conv 98.304 B + S 2.097.152 B)) serve a tutti i round,
e il restore tocca solo i layer >= `firstDirty`.

**IL BUG CHE IL GATE HA TROVATO, e come l'ho preso.** Primo giro a cache fredda:
morto al token 0 con "replay round 2 sporco allo STESSO layer MoE 0 — progresso
violato". Ipotesi PRIMA della correzione, dal codice: avevo scritto
`if (startLayer === 0) snapshot else rientro`, cioe' avevo usato
`startLayer === 0` come sinonimo di "primo giro". **Non lo e': a cache vuota il
primo layer sporco E' lo zero**, quindi il replay riparte proprio da li' e
finiva nel ramo dello snapshot — senza rientro, `x` conteneva l'uscita del giro
precedente invece dell'embedding, il router di layer 0 sceglieva 8 expert
diversi, quelli non erano residenti, e il round dopo ritrovava il layer 0 sporco.
Il sintomo osservato (stesso layer, round 2) e' esattamente cio' che quella causa
produce. Corretto separando i due concetti (`runPass(startLayer, first)`),
ri-verificato contro lo STESSO repro: 39/39 token completati.

**GATE, smoke 35B, 39 token, 320 selezioni/token, 10 GiB.** Il regime freddo
gira in un run a parte, perche' la cache fredda esiste una volta sola per
processo (`--opt-cold`):

| | ottimistico FREDDO | sync CALDO | ottimistico CALDO |
|---|---|---|---|
| submit/token | 3,79 | 81 | **1** |
| readback/token | 3,79 | 41 | **1** |
| token sporchi | **39 / 39** | 0 | 0 |
| replay | 109 (2,79/token) | 0 | 0 |
| miss | 3742 | 0 | 0 |
| ms/token | 738,6 | **141,18** [140,61-144,64] | **72,58** [72,45-72,73] |

- **argmax IDENTICO 39/39** in tutti i confronti: caldo-vs-caldo, e
  **freddo-ottimistico contro sync-caldo**. Quest'ultimo e' IL gate del replay:
  39 token con 2,79 round medi ciascuno, 109 restore dello stato ricorrente, e
  l'argmax non si sposta di una posizione. Se il restore fosse sbagliato la
  deriva sarebbe cumulativa (il deltanet e' ricorrente) e divergerebbe subito.
- routing IDENTICO chiave per chiave (3341 chiavi, 0 diff), miss 0 vs 0 a caldo.
- ms/token a caldo: **−68,60 (−48,6%)**, coerente coi −71,99 di it.17 (i due run
  distano ~3 ms, dentro la banda di dispersione).

**IL PREZZO DEL REGIME FREDDO, misurato.** Ogni replay rigioca in media **32,3
layer su 40 (80,7%)**: il primo layer sporco sta in basso e quasi tutto il token
si ricalcola. E il repair FETCHA PIU' DI QUANTO SERVA: 3742 miss contro i 3341
del path sync sullo stesso prompt (**+12,0%**), perche' un expert riparato puo'
non finire nella `Sel` definitiva — il replay a valle, con l'hidden corretto,
sceglie diversamente. Sono 910 selezioni contate come miss e mai usate. Il
fetch e' avvenuto davvero: contarlo e' onesto, nasconderlo no. Il tempo CPU del
repair e' **484,3 ms/token, il 65,6% del token freddo**.

**E QUI IL NUMERO SMENTISCE LA MIA PREVISIONE DI it.16.** Avevo scritto che a
cache fredda "il path ottimistico NUDO applicato al prefill freddo e' una
regressione". Misurato: 738,6 ms/token contro i 1192,9 del sync freddo di it.17.
**Non lo prendo per buono**: sono UN campione per braccio, in DUE run diversi,
su una passata dominata dall'I/O — per il docket item 10 non e' una misura. Ma
non e' nemmeno la conferma di quello che avevo previsto, e vale come segnale:
la soglia della fase 5 va tarata su una misura fatta apposta (bracci
interleavati nello stesso processo), non sulla mia intuizione di it.16 ne' su
questi due numeri.

**FASE 3b: DONE-WHEN VOCE PER VOCE.**
- (a) submit/token e readback/token misurati prima e dopo nello stesso JSON, col
  caso a residenza piena a 1 submit/token → **81 → 1**, misurati. ✓
- (b) argmax identico al path attuale su un campione del golden 35B → 39/39. ✓
- (c) ktest MoE q35 PASS invariati e GLM bit-identico → **87/87**. ✓
- (d) conteggio dei miss e routing invariati sullo stesso campione → a caldo
  identici (0 vs 0, routing chiave per chiave). **A FREDDO NO, e non puo'
  esserlo**: +12,0% di fetch e' inerente al meccanismo, non un difetto
  dell'implementazione. Il campione che il contratto nomina in (a) e' quello a
  residenza piena; li' (d) e' soddisfatta alla lettera. Il numero freddo si
  pubblica come fatto misurato, non si nasconde.

**Gate**: tsc pulito, suite 410|9, ktest 87/87, 0 gpu-error, due JSON committati
(`q35-optimistic-35b-it17.json` per il freddo-sync, `q35-optimistic-cold-35b-it18.json`
per il freddo-ottimistico).
