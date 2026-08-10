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

## it.10 (2026-08-10) — fase 3 sui DENSI: -15,0 ms/token. Sul MoE non si puo' (ancora)

**Misurato prima, di nuovo.** Strumentata la scomposizione per token
(`Q35GpuModel.perf()`: embed CPU / readback / argmax CPU) e girata sul 4B:
`readbackMs` 44,6 ms/token. **NON l'ho letto come "il trasferimento di 604 KB
costa 44 ms"**: quell'attesa su `mapAsync` include tutto il lavoro GPU del
token. Il numero che serviva era un altro — quanto CPU e GPU si aspettano a
vicenda — e l'ho preso con un micro-bench nel ktest: gli stessi token
accodati con `read=false` e una sola attesa alla fine.

| 4B, 39 token | ms/token |
|---|---|
| con sync per token (`step`) | **50,5** |
| senza sync (accodati, una attesa) | **35,4** |
| **tetto recuperabile** | **15,0** |
| embed dequant CPU | 0,07-0,12 |

**Fatto**: `Q35GpuModel.decodeBatch(tokens, posStart)` — K token
teacher-forced in UN submit, argmax su GPU (`argmaxStage1/2`, gli stessi
kernel di `gpuforward`), un readback di K·4 byte invece di K readback da
604 KB. Le righe di embedding e gli uniform dei K step si impacchettano in un
buffer e si copiano nello scratch DENTRO l'encoder: `queue.writeBuffer` e'
ordinata prima del submit, quindi scriverli in ciclo darebbe a tutti gli step
l'ultimo valore (stesso schema di `dbSlots` in gpuforward). Error scope
esplicito, come ogni encode nuovo.

**RISULTATO**: `decodeBatch` **35,5 ms/token contro 50,5** = **-15,0
(-29,6%)**, e cade esattamente sul tetto senza sync (35,4): recupera
praticamente tutta la serializzazione. Il done-when ne chiedeva >= -5,1.

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
