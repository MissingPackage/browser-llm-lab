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

## it.19 (2026-08-11, fase 4) — PREVISIONE SCRITTA PRIMA DI MISURARE: perche' il prefill batched sul 35B non puo' rendere come su GLM

Ricognizione su `glmprefillplan.ts` + `glmmodel` (il prefill batched di GLM) e
su `q35gpumodel`, prima di scrivere una riga. Due cose cambiano rispetto a GLM,
ed entrambe abbassano il tetto.

**1. TRENTA LAYER SU QUARANTA NON SI BATCHANO PER RIGHE.** Il batching M>1 del
prefill fa girare M token insieme come righe di una GEMM. Funziona sui layer
senza memoria fra token: attenzione (GLM: MLA; qui i 10 layer full-attn),
shexp, router, expert. **Il deltanet e' ricorrente sul TEMPO**: `deltaNetConv`
shifta lo stato e `deltaNetCore` aggiorna S token per token. Le righe di un
chunk sono token CONSECUTIVI, quindi la riga m+1 dipende dallo stato prodotto
dalla riga m: non e' parallelizzabile nella forma in cui il kernel esiste oggi.
GLM non ha layer ricorrenti e questo problema non ce l'aveva. Esiste una forma
chunkwise del delta rule (e' come si addestrano questi modelli), ma e' un kernel
di ricerca, non un port — e non e' quello che la riga 4 di PHASES chiede.
Conseguenza: nel chunk, i 2 dispatch ricorrenti per layer restano M, tutto il
resto diventa 1.

**2. L'UNIONE DEGLI EXPERT SI COMPRIME MOLTO MENO.** Il guadagno del batching
per expert e' la MOLTEPLICITA' media (quante righe del chunk selezionano lo
stesso expert): sotto quella si dividono sia i dispatch sia il traffico dei
pesi. GLM ha 64 expert e top-4: a M=16 sono 64 selezioni su 64 expert,
E[|unione|] ~ 40, molteplicita' ~1,6. Il 35B ha **256 expert e top-8**: a M=16
sono 128 selezioni su 256 expert, E[|unione|] = 256·(1−(1−1/256)^128) = **100,9**,
molteplicita' **1,27**. Il parco e' 4x piu' grande e la molteplicita' crolla.

**LA PREVISIONE, in dispatch per layer per token** (statici row-parallel 17,
ricorrenti 2, expert 5 per expert dell'unione; oggi: 19,5 statici + 40 expert =
**59,5**):

| M | \|unione\| | molteplicita' | disp/layer/token | vs oggi |
|---|---|---|---|---|
| 8 | 56,7 | 1,13 | 39,6 | 1,50x |
| 16 | 100,9 | 1,27 | 34,6 | **1,72x** |
| 32 | 162,0 | 1,58 | 27,8 | 2,14x |
| 64 | 221,5 | 2,31 | 19,6 | 3,03x |

Cioe': a M=16 il prefill batched toglie il 42% dei dispatch, non l'80% che il
caso GLM lascerebbe sperare, e il residuo e' quasi tutto path expert.

**MA LA PREVISIONE VALE SOLO SE IL TOKEN E' DISPATCH-BOUND, E QUESTO NON L'HO
MISURATO.** L'aritmetica dice che potrebbe esserlo: 2382 dispatch/token (782
statici + 320x5 expert) in 72,6 ms fanno **30,5 us per dispatch**, contro un
lavoro vero per GEMV expert stimabile in 1-2 us (0,6 MB di pesi a ~500 GB/s).
Un ordine di grandezza di scarto dice "overhead per dispatch", ma e' una stima
di FLOP e banda fatta a mano, non una misura — e la banda vera di questi kernel
non l'ho mai misurata sul 35B.

**QUINDI PRIMA LO STRUMENTO, POI LA FETTA** (stesso schema di it.16): serve la
decomposizione del token in tempo GPU — quanto va nel segmento statico, quanto
nel router, quanto nei 1600 dispatch expert. Se il tempo sta negli expert e
scala coi dispatch, la tabella qui sopra e' la previsione del guadagno di fase 4
e la fetta si fa. Se sta altrove (attenzione, deltanet, coda), la fase 4 va
ridiscussa PRIMA di spendere 2-3 iterazioni: il suo done-when e' un micro-bench
tok/s prima/dopo, e un "dopo" che non si muove e' una fase spesa male.

Previsione registrata PRIMA della misura, con la data e il commit di questa
riga: mi aspetto che il segmento expert sia >= 60% del tempo GPU del token.

### it.19 — LA MISURA, e la previsione che avevo registrato E' SBAGLIATA

Strumento nuovo (`--gpu-time`, opt-in): `timestamp-query` sui pass del path a
submit unico, con il pass di ogni layer spezzato in tre — statico / router /
expert — piu' la coda. Spezzare i pass PERTURBA (tre barriere invece di una) e
la perturbazione e' misurata, non assunta: **73,85 contro 72,58 ms/token, +1,7%**.

**Smoke 35B, 156 giri cronometrati, budget 10 GiB:**

| categoria | ms/token | % del tempo GPU | pass/token | dispatch/token | us/dispatch |
|---|---|---|---|---|---|
| expert | 33,44 | **58,0%** | 40 | 1600 | 20,9 |
| statico | 14,53 | 25,2% | 40 | 742 | 19,6 |
| coda (norma+head) | 6,86 | 11,9% | 1 | 2 | — |
| router | 2,83 | 4,9% | 40 | 40 | **70,8** |
| **totale GPU** | **57,67** | 100% | 121 | 2384 | |

Il token dura 73,85 ms: **16,18 ms (21,9%) stanno FUORI dai pass** — encode CPU,
submit, attesa del readback, argmax, dequant della riga di embedding.

**LA PREVISIONE ERA "expert >= 60% del tempo GPU del token". E' 58,0%.** Sbagliata,
di poco ma sbagliata; e sul tempo di parete e' 45,3%, ancora piu' lontana. La
registro come fallita invece di riscriverla: la scommessa era che il path expert
dominasse abbastanza da rendere la fase 4 quasi tutta un problema di expert, e
non e' cosi' — un quarto del tempo e' nel segmento statico, che si batcha molto
meglio degli expert.

**Il numero che invece regge, ed e' quello che conta**: **~20 us per dispatch,
uguale fra statico (19,6) ed expert (20,9)**, contro un lavoro vero stimabile in
1-2 us per GEMV expert. Due categorie con kernel diversi, taglie diverse e
traffico diverso che costano lo stesso per dispatch dicono che il costo NON e'
il lavoro: e' il per-dispatch. Il token e' dispatch-bound, e la fase 4 — che
toglie dispatch — e' la leva giusta. (Il conto della banda lo conferma: 320
selezioni x 1,785 MB = 571 MB/token, che a ~500 GB/s sarebbero 1,14 ms contro i
33,4 misurati.)

**PROIEZIONE DELLA FASE 4 rifatta sui numeri misurati** (M=16, |unione| 100,9):
expert 33,44 → 26,36 (l'unione comprime solo 1,27x) · statico 14,53 → 2,01 (i
682 dispatch row-parallel per token diventano 682 per chunk; restano i 60
ricorrenti del deltanet) · router 2,83 → 0,18 · coda 6,86 → 0 nel prefill.
**Totale GPU 57,67 → 28,55 ms, cioe' 2,02x.** Il collo resta il path expert: e'
il 92% del tempo residuo.

**E LA MISURA HA TROVATO UNA COSA CHE NON C'ENTRA COL BATCHING.** La coda —
norma finale + head — vale 6,86 ms/token (9,3% del token) **e nel prefill viene
buttata**: `step(token, pos, false)` la calcolava lo stesso, e la head scrive
`logits`, che con `read=false` nessuno legge. Tolta (`headCut`, tre righe, su
MoE e densi): la norma finale scrive `xn`, uno scratch, e il residuo `x` non
viene sfiorato, quindi non cambia un solo numero.
NON MISURATA end-to-end: il gate gira con `read=true` su ogni token, quindi la
head c'e' comunque e il suo ms/token non si muove (72,96 contro 72,58, dentro
la banda). Il guadagno vale sui token di PREFILL e lo misurera' il micro-bench
della fase 4, che e' il done-when della riga. Quello che questa iterazione
prova e' la NON-REGRESSIONE: argmax 39/39, routing identico chiave per chiave,
miss 0.

**Rilievo di margine**: il dispatch del router costa **70,8 us**, 3,5x un GEMV
expert, ed e' UN workgroup — softmax su 256 expert e top-8 fatti dal thread 0 in
seriale (`routerTopKWgsl`). Sono 2,83 ms/token, il 4,9%. Non e' lavoro di questa
fase; va sul docket.

**Gate**: tsc pulito, suite 410|9, 0 gpu-error, due JSON committati
(`q35-gputime-35b-it19.json`, `q35-optimistic-35b-it19-headcut.json`).
Nota di dispersione: la passata fredda oscilla fra 1137,8 e 1192,9 ms/token fra
run diversi (~5%) — e' dominata dall'I/O e non e' un numero su cui appoggiarsi.

## it.20 (2026-08-11, fase 4) — il piano di prefill smette di essere di GLM

Primo passo della fetta, ed e' fase-1-shaped: `glmprefillplan.ts` era cablato su
`GLM47_FLASH` — 64 expert, top-4 — in SETTE punti (la taglia dell'array dei
pesi, il passo `row*nExpertUsed+k` in due posti, il controllo del range
dell'expert, il messaggio d'errore, e il loop di `combineMoeRow`). Portarlo a
Qwen 3.6 copiandolo era la strada che il gate strutturale del goal vieta
(direction §7-ter: una meccanica, una implementazione), quindi: parametrico.

**La config e' STRUTTURALE e minima** — `MoePlanShape { nExpert, nExpertUsed }` —
e NON un import di `MoeModelConfig`. Il motivo e' un ciclo: `residency.ts`
importa `GLM_PREFILL_M` da questo file, quindi dipendere di la' a RUNTIME lo
chiuderebbe. `MoeModelConfig` ha entrambi i campi ed e' assegnabile per
struttura: l'unificazione la fa il sistema di tipi senza creare la dipendenza,
e i call site passeranno la loro `cfg` senza conversioni.

Il file si chiama ora `moeprefillplan.ts`: tenere il prefisso `glm` su un
meccanismo condiviso e' il primo passo per ricopiarlo la prossima volta.
Rinominato con `git mv` (la storia segue), sei import aggiornati.

**I test girano su DUE famiglie** (`describe.each`): GLM 64/top-4 e q35
256/top-8, sulle stesse proprieta' — biiezione delle selezioni, ordine
deterministico, taglia dell'unione, pesi in selF32, validazione, e soprattutto
l'IDENTITA' BIT-A-BIT fra il percorso a unione+combine e la catena del decode
in ordine k, con la sua controprova (accumulare nell'ordine dell'unione DEVE
divergere in f32, altrimenti la struttura a slot sarebbe complessita' inutile).
Piu' un test che il default senza `cfg` e' ancora esattamente GLM.
Non e' ridondanza: un piano che funziona solo col parco e il top-K di GLM
passerebbe meta' di questi test e fallirebbe gli altri.

**Gate**: tsc pulito, suite **417 | 9** (erano 410|9: +7 dalla seconda famiglia),
ktest **87/87** — con `prefill-moe-batched-vs-decode-chain` BIT-IDENTICO e
`glm-model-2layer` a L2rel 2,07e-7, cioe' il prefill batched di GLM non si e'
mosso di un bit. Nessun run GPU nuovo: qui non e' cambiato niente che giri su
GPU, e il ktest e' il paracadute che lo prova.

**Resta della fase 4** (e ora il piano non e' piu' l'ostacolo): i kernel a M
righe per statico/router/expert su q35, il gather per expert dell'unione, la
combine in ordine k, e la scelta di M — che va fatta coi numeri di it.19
(proiezione 2,02x a M=16) e non per analogia con GLM.

## it.21 (2026-08-11, fase 4) — l'esperimento da 320 dispatch che ha SMENTITO la lettura di it.19

**Cosa ho fatto.** Il down degli expert ora ACCUMULA: `y[r] = y[r] + sel.w·dot`
invece di scrivere `y[r] = dot` e lasciare a un axpy separato la moltiplicazione
per il peso. Il peso ce l'aveva gia' in mano — il preambolo d'arena legge `Sel`
per sapere quale slot indirizzare, e `sel.w` sta li'. Opzione `accum` sui due
kernel K-quant, un dispatch per expert in meno: la catena passa da
gate/up/silu/down/axpy a gate/up/silu/down. **320 dispatch/token in meno** su
2384, e l'`axpySelWgsl` nato in it.17 diventa codice morto e sparisce (con lui
`dnE`, i `wBufs` e il bind group dell'axpy del path sync).

**Bit-identico per costruzione, non "atteso identico"**: l'axpy calcolava
`out[i] + w·x[i]` in f32 con `x[i]` uguale ESATTAMENTE al `partial[0]` che il
kernel scriveva; qui l'espressione, le operazioni f32 e l'ordine sono gli
stessi. Verificato: **argmax 39/39, routing identico chiave per chiave** (3341
chiavi, 0 differenze), miss 0.

**E QUI IL NUMERO CHE NON TORNA.** it.19 aveva misurato ~20 us per dispatch,
uguale fra statico (19,6) ed expert (20,9), e io ne avevo concluso che il token
fosse DISPATCH-BOUND — con la proiezione della fase 4 (2,02x a M=16) costruita
su quella lettura. Togliendo 320 dispatch mi aspettavo ~−6,7 ms. Misurato:

| | it.19 | it.21 | delta |
|---|---|---|---|
| expert (ms/giro, sonda accesa) | 33,442 | **33,115** | **−0,327** |
| totale GPU | 57,668 | 57,330 | −0,338 |

**−1,02 us per dispatch rimosso, contro i ~21 attesi.** L'axpy era un kernel
minuscolo (2048 elementi, 32 workgroup) e costava quanto il lavoro che faceva.
Quindi i ~20 us medi di it.19 NON sono un costo fisso di lancio: sono
semplicemente totale/conteggio, e l'uniformita' fra statico ed expert era una
coincidenza di medie. **La conclusione "il token e' dispatch-bound" e' sbagliata,
e con essa la proiezione 2,02x della fase 4**, che assumeva tempo ∝ dispatch.

**COSA DICE INVECE IL NUMERO GIUSTO.** Se il tempo degli expert non e' lancio,
e' lavoro — e allora va guardato il lavoro: 320 selezioni x 1,785 MB di slab =
**571 MB di pesi letti per token in 33,1 ms = 17,2 GB/s efficaci**, su una
scheda che ne fa ~500. Il 3% della banda. Il perche' e' nel kernel, e si legge
dal sorgente: `gemvQ4K` distribuisce i SUPERBLOCCHI DI UNA RIGA sui 64 thread
del workgroup (`for sb = t; sb < SB_PER_ROW; sb += 64`), e sul 35B
SB_PER_ROW = K/256 vale **8 per gate/up (K=2048) e 2 per il down (K=512)**.
Cioe' **8 lane attive su 64 (12,5%) e 2 su 64 (3,1%)**: il workgroup occupa uno
slot intero per far lavorare due thread.

**Conseguenze, e le tengo separate da cio' che ho verificato.** VERIFICATO: il
tempo non scala coi dispatch (l'esperimento sopra), e la banda efficace e' 17,2
GB/s (aritmetica su byte misurati e tempo misurato). NON VERIFICATO: che
riscrivendo la distribuzione del lavoro nel kernel si recuperi banda — le lane
inattive sono una spiegazione coerente col numero, non una prova, e potrebbero
esserci altri colli (latenza, accessi non coalescenti sui superblocchi). La
prova sarebbe un kernel alternativo misurato contro questo, ed e' lavoro che il
contratto non assegna a nessuna fase. Va sul docket (item 15), con i numeri.

**Il cambio resta**: −320 dispatch, −0,33 ms (0,5%), bit-identico, e tre
oggetti in meno da mantenere. E' piccolo e lo dico piccolo — la sua utilita'
vera e' stata falsificare l'ipotesi su cui stavo per spendere la fase 4.

**Gate**: tsc pulito, suite 417|9, argmax 39/39, routing identico, miss 0,
JSON committato (`q35-accumdown-35b-it21.json`).

## it.22 (2026-08-11, FASE 4-BIS) — 64 lane invece di 2: il decode del 35B passa da 71,9 a 44,3 ms/token

**Ruling PI incassato per primo** (docket item 15, opzione (a)): riga 4-bis
aggiunta a PHASES fra la 3b e la 4, col done-when misurabile e i numeri di
it.21 come "prima". E dichiarato PRIMA di cominciare: **non sara' bit-identico**
— distribuire il lavoro su piu' lane cambia l'ordine delle somme f32, e in f32
l'addizione non e' associativa. Il gate e' ktest contro cpuref, argmax identico,
routing invariato.

**La correzione.** I due kernel K-quant spartivano i SUPERBLOCCHI di una riga
sui 64 thread (`for sb = t; sb < SB_PER_ROW`). Sul 35B `SB_PER_ROW = K/256` vale
8 per gate/up (K=2048) e **2 per il down** (K=512): otto lane su 64, e due su
64. L'unita' di lavoro diventa un PEZZO del gruppo — `lpu` valori dell'indice
interno — scelto perche' le unita' arrivino a 64. La scelta sta in una funzione
sola (`kquantWorkSplit(sbPerRow, groupsPerSb)`, 4 gruppi per q4_K e 2 per q6_K):
i due kernel non hanno due aritmetiche, ne hanno una.

**Il test che non serve la GPU** (`engine-kquant-worksplit.test.ts`): la
BIIEZIONE della spartizione — ogni (superblocco, gruppo, l) coperto esattamente
una volta, su 18 combinazioni di K e geometria. E' dove stanno gli errori che
diventano numeri plausibili: un'unita' che copre due volte gonfia il prodotto
scalare, una che salta lo taglia, e in entrambi i casi il modello continua a
girare. 23 test, tutti verdi.

**NUMERI (smoke 35B, 39 token, 10 GiB, sonda `--gpu-time` accesa in entrambi):**

| categoria | it.21 | it.22 | delta |
|---|---|---|---|
| expert | 33,115 | **8,743** | **−24,372** |
| coda (norma + head) | 6,859 | **1,415** | −5,444 |
| statico | 14,523 | 16,144 | **+1,621** |
| router | 2,833 | 3,169 | +0,336 |
| **totale GPU** | **57,330** | **29,471** | **−27,859** |

**ms/token a parita' di residenza** (bracci interleavati, prima coppia scartata):
**71,90 → 44,26** = **−27,64 ms (−38,4%)**, cioe' da 13,9 a **22,6 tok/s** sul
35B. La banda efficace del segmento expert va da **17,2 a 65,3 GB/s** (3,8x).
La coda cala perche' anche la head e' un GEMV K-quant, e non ci avevo pensato:
il guadagno l'ha presa senza che nessuno la toccasse.

**IL NUMERO CHE NON MI TORNA, e lo lascio scritto invece di ignorarlo**: il
segmento STATICO e' PEGGIORATO di 1,62 ms (+11%). Non e' rumore — nelle tre
misure precedenti stava fra 14,52 e 14,53. Ipotesi che non ho verificato: nel
regime nuovo le scale del superblocco si ri-estraggono una volta per unita'
invece che una per superblocco, e sui GEMV statici K-quant con N grande la
ridondanza potrebbe superare il guadagno. Ma sugli expert la stessa ridondanza
c'e' ed e' stravinta 3,8 a 1, quindi la spiegazione non mi convince. Va sul
docket (item 16) come lavoro di misura, non come congettura da correggere alla
cieca.

**GATE della 4-bis, voce per voce:**
- (a) tempo del segmento expert e banda efficace misurati prima e dopo →
  33,115 → 8,743 ms, 17,2 → 65,3 GB/s. **NOTA SUL "nello stesso JSON"**: la
  formulazione l'ho scritta io e per una SOSTITUZIONE di kernel non e'
  ottenibile alla lettera — il kernel vecchio non esiste piu' nell'albero. I
  due JSON sono di run consecutivi sullo stesso host, stesso golden, stesso
  budget, a minuti di distanza, che e' il rigore della prova a due bracci di
  it.14. Lo dico invece di far finta che la riga sia soddisfatta com'e' scritta.
- (b) ktest **87/87**, tolleranze contro cpuref invariate o MIGLIORI: q4_K
  2048x512 maxRel 2,74e-4 → **5,58e-5**; q4_K 512x2048 2,02e-5 → 2,33e-5; il
  blocco MoE reale del 35B a 6,4e-6 (q4_K down) e 8,9e-6 (q6_K down). ✓
- (c) argmax **39/39 IDENTICO** sul golden smoke, routing **identico chiave per
  chiave** (3341 chiavi, 0 differenze), miss 0. ✓
- (d) GLM **bit-identico**: `glm-model-2layer` L2rel 2,072937787401139e-07,
  la STESSA cifra di prima fino all'ultima decimale, e `glm-layer0-conformance`
  a 2,35e-7. I suoi expert sono q4_0/q4_1 e non passano da questi kernel. ✓

**Gate permanenti**: tsc pulito, suite **440 | 9** (+23 dal test nuovo),
ktest 87/87, 0 gpu-error, JSON committato
(`q35-occupancy-35b-it22.json`).

**E la fase 4 va ri-guardata ORA**: la sua proiezione (it.19) assumeva tempo ∝
dispatch ed era gia' caduta con it.21; adesso il segmento expert e' 8,7 ms su
29,5 di tempo GPU (29,7%, era il 58%), quindi anche il peso relativo di cio' che
il batching comprimerebbe e' cambiato. La riga 4 dice gia' "proiezione da
RIFARE": si rifa' coi numeri di qui.

## it.23 (2026-08-11, fase 4) — la proiezione RIFATTA dal basso, e un'altra mia previsione ridimensionata

**Perche' serviva.** La proiezione di it.19 (2,02x) era costruita su "tempo ∝
dispatch", che it.21 ha smentito; e it.22 ha cambiato le proporzioni (il
segmento expert e' passato dal 58% al 29% del tempo GPU). Rifarla a occhio
sarebbe stato inventare. Serviva la decomposizione del segmento STATICO, che
finora era una categoria sola.

**Strumento**: le marche (`segMarks`) — `{at, cat}` su INTERVALLI di step invece
di un campo per step, cosi' i ~25 siti di `push` non si toccano. Con la sonda
spenta il pass resta uno per layer (la forma di produzione); accesa, si spezza
sulle marche. `TSQ_PASSES` da 256 a 512 perche' le categorie per layer sono 7 e
281 pass non ci stavano — **e l'overflow ora si CONTA** (era silenzioso: avrebbe
fatto sembrare piu' economici gli ultimi layer). Misurato: overflow 0.

**DECOMPOSIZIONE (smoke 35B, 156 giri, 10 GiB):**

| categoria | ms/token | % | cosa e' |
|---|---|---|---|
| expert | 8,716 | 28,7% | i 320 GEMV expert |
| ssmGemv | 7,602 | 25,0% | qkv/z/beta/alpha dei 30 layer deltanet |
| router | 3,155 | 10,4% | il top-8 su 256, un workgroup per layer |
| attn | 2,925 | 9,6% | i 10 layer full-attention |
| shexp | 2,333 | 7,7% | shared expert |
| ssmOut | 2,235 | 7,4% | la proiezione d'uscita del deltanet |
| tail | 1,403 | 4,6% | norma finale + head |
| routerGemv | 0,701 | 2,3% | il GEMV che produce i logit del router |
| **ssmRec** | **0,588** | **1,9%** | **conv + core: LA RICORRENZA** |
| norm/resid/altro | 0,701 | 2,3% | |
| **totale GPU** | **30,359** | | |

**LA PREVISIONE DI it.19 ERA STRUTTURALMENTE GIUSTA E QUANTITATIVAMENTE
IRRILEVANTE.** Avevo scritto che "30 layer su 40 sono deltanet, ricorrenti sul
tempo" e che questo era il vincolo strutturale della fase 4. E' vero che non si
batchano — ma la ricorrenza VERA (`deltaNetConv` + `deltaNetCore`) costa **0,588
ms, l'1,9%**. Tutto il resto del blocco deltanet sono GEMV (`ssmGemv` +
`ssmOut` = 9,84 ms, il 32%), che sono row-parallel come qualunque altro GEMV.
Avevo scambiato "il layer e' ricorrente" con "il layer non si batcha".

**PROIEZIONE DELLA FASE 4, rifatta dal basso su questi numeri** (M=16, token di
prefill = 28,96 ms perche' `headCut` toglie gia' la coda):

- comprimibili ~M (il traffico dei pesi si legge una volta per chunk):
  ssmGemv + ssmOut + shexp + routerGemv + norm + resid = **13,572 → 0,85**
- expert: si comprime solo per la MOLTEPLICITA' dell'unione, 1,27x a M=16
  (256 expert, top-8): **8,716 → 6,86**
- router: il lavoro e' per riga e resta, **3,155**
- attn: ogni riga attende al proprio prefisso, non si comprime, **2,925**
- ssmRec: **0,588**

**28,96 → 14,38 ms, cioe' 2,01x.** Lo stesso numero di it.19 per ragioni
completamente diverse — e stavolta e' costruito su categorie misurate invece
che su un conteggio di dispatch. Il pavimento della fase 4 e' expert (6,86) +
router (3,16) + attn (2,93) = 12,94 dei 14,38: **il 90%**.

**QUELLO CHE LA PROIEZIONE DICE ANCHE, E CHE CONTA DI PIU'.** La fase 4 vale sul
TTFT e zero sul decode, e la funzione obiettivo del goal e' il decode. Oggi il
35B sta a **45,48 ms/token con la sonda accesa (44,26 spenta) = 22,0-22,6
tok/s**; per 30 tok/s servono 33,3 ms, cioe' **−12 ms**. I candidati, misurati:

1. **fuori dai pass: 15,12 ms, il 33,2% del token.** Non e' tempo GPU dei
   kernel: e' encode CPU, submit, attesa del readback, argmax su 151k logit,
   dequant della riga di embedding. E' la voce piu' grossa dopo gli expert e
   nessuna fase del contratto la guarda.
2. **router 3,155 ms** — 40 dispatch da 79 us, UN workgroup che fa softmax su
   256 e top-8 in seriale sul thread 0 (docket item 14, gia' aperto).
3. **expert 8,716 ms a 65,3 GB/s** e **ssmGemv+ssmOut 9,84 ms** per ~1,07 GB di
   pesi = ~109 GB/s: entrambi lontani dal picco (~500).

**Gate**: tsc pulito, suite 440|9, argmax 39/39, routing identico chiave per
chiave, miss 0, overflow della sonda 0, JSON committato
(`q35-statbreak-35b-it23.json`). Perturbazione della sonda dichiarata: con 12
categorie invece di 3 il totale GPU passa da 29,47 a 30,36 (+3,0%) — piu'
barriere, ed e' il prezzo di sapere dove vanno i ms.

**Docket item 16 NON si chiude con questa misura**: per attribuire il +1,62 ms
del segmento statico servirebbe la stessa granularita' PRIMA della 4-bis, cioe'
rimettere il kernel vecchio. Resta aperto.

## it.24 (2026-08-11, fase 4) — item 16 CHIUSO da un fatto sui tipi, e l'inventario che ridimensiona la fase

**Item 16 (lo statico +1,62 ms dopo la 4-bis): chiuso, e non da una misura in
piu' ma da una lettura del GGUF.** Ho enumerato i tipi dei tensori del 35B:

| gruppo | tipo |
|---|---|
| attn q/k/v/output, attn_qkv, attn_gate | **Q8_0** |
| ssm_out, shexp gate/up/down | **Q8_0** |
| ffn_gate_inp (router), alpha, beta, norm, conv1d | **F32** |
| SOLO gli expert (`*_exps.weight`) | q4_K / q6_K |

Cioe': **il segmento statico non contiene NEMMENO UN GEMV K-quant.** Passa tutto
da `gemvQuantWgsl` (q8_0) e `gemvF32Wgsl`, che la 4-bis non ha toccato — il
testo emesso di quei due kernel e' identico byte per byte a prima. Quindi il
+1,62 ms **non puo' venire dal cambio**: sono gli stessi kernel, sugli stessi
dati, con lo stesso lancio. Restano due spiegazioni, ed entrambe sono globali e
non di codice: deriva fra run, oppure un effetto di sistema (il token e'
passato da 71,9 a 44,3 ms, quindi la stessa GPU fa molto piu' lavoro al secondo
— clock e potenza non sono gli stessi). E' esattamente la classe di fenomeni per
cui il progetto ha `hostState` e la regola dei bracci interleavati.
**Item 16 CHIUSO: non attribuibile alla 4-bis, per costruzione.**

Nota metodologica che mi porto dietro: avevo scritto "l'ipotesi della ridondanza
nell'estrazione delle scale non mi convince" e avevo ragione a non correggere
alla cieca — la correzione sarebbe stata su un kernel che nemmeno partecipa.

**INVENTARIO DELLA FASE 4** (ricognizione prima della fetta, come it.13/it.16/
it.19). Cosa serve per far girare M righe insieme, e cosa c'e' gia':

GIA' PRONTO (esiste ed e' ktestato, `dense-batch-*` BIT-IDENTICO su 3 righe):
`gemvQuantWgsl` batch (q4_0/q4_1/**q8_0**) — ed e' il kernel di TUTTO lo statico
del 35B, cioe' della voce piu' grossa da comprimere (ssmGemv 7,60 + ssmOut 2,24
+ shexp 2,33 + attn) · `gemvF32Wgsl` batch · `rmsnormWgsl` batch ·
`kvAppendWgsl` batch · `stridedCopyWgsl` batch.

DA FARE, e non e' poco:
1. **attenzione a chunk per q35**. `attnPrefillChunkWgsl` esiste ma e' del path
   Qwen 2.5: legge un `qkv` FUSO con quel layout, mentre q35 ha q/k/v separati,
   q_norm/k_norm, rope-neox e un gate sigmoid. Non e' un drop-in.
2. **`ropeNeoxWgsl` batch** — oggi prende `pos` da un uniform, serve per riga.
3. **elementwise batch**: `siluMulWgsl`, `sigmoidMulWgsl`, `addInPlaceWgsl`,
   `axpyWgsl` (offset di riga).
4. **`deltaNetGatesWgsl` batch** (le gate sono row-parallel; conv e core NO,
   restano M — ma costano 0,588 ms in totale, it.23).
5. **il path expert a GATHER**: GLM ha `pairGemvSiluGatherWgsl` e
   `gemvDownSlotsWgsl`, ma **solo per q4_0/q4_1**. Per i K-quant non esistono, e
   sono il pezzo che comprime la voce piu' grossa dopo lo statico.
6. **l'orchestratore**: un secondo forward a M righe in `q35gpumodel`, con gli
   scratch per riga, il piano `planMoeChunk` (gia' parametrico da it.20) e la
   combine in ordine k.

**Conseguenza sulla TAGLIA**: PHASES stima la fase 4 in 2-3 iterazioni. Con sei
voci di cui due sono famiglie di kernel nuove (attenzione a chunk, gather
K-quant), **3-5 e' piu' onesto**, ed e' il tipo di correzione che va fatta prima
di cominciare e non a meta'. Aggiornata la riga delle taglie.

**Gate**: nessun codice toccato in questa iterazione (tsc pulito, suite 440|9
invariate) — e' ricognizione e chiusura di un item, con l'evidenza su disco.

## it.25 (2026-08-11, fase 4) — i kernel a M righe che mancavano, tre voci dell'inventario su sei

Prima fetta di codice della fase 4, sulle voci 2-3-4 dell'inventario di it.24
(quelle piccole e indipendenti; l'attenzione a chunk e il gather K-quant sono
famiglie nuove e vanno da sole).

**Fatti**: `ropeNeoxWgsl` batch (la posizione per riga da `rowPos`, il vettore a
offset di riga — stesso idioma di `kvAppendWgsl`), `siluMulWgsl`,
`sigmoidMulWgsl`, `addInPlaceWgsl` batch (indici a offset di riga), e
`deltaNetGatesWgsl` batch. Quest'ultimo merita una riga: le GATE del deltanet
sono row-parallel — dipendono solo da beta/alpha della riga — mentre conv e core
sono la ricorrenza vera e restano per riga. E' la distinzione che it.23 ha
misurato (0,588 ms in tutto) e che in it.19 avevo sbagliato, trattando l'intero
blocco deltanet come non batchabile.

**L'idioma e' quello della casa**: senza `batch` il testo emesso e' IDENTICO
byte per byte a prima, quindi il path di decode non cambia di una virgola, e il
corpo aritmetico esiste una volta sola per i due regimi.

**GATE — ktest 92/92** (5 casi nuovi, tutti **BIT-IDENTICO su 3 righe**):
`dense-batch-rope-neox` (posizioni CRESCENTI per riga: e' il caso vero di un
chunk di prefill, dove ogni riga e' un token diverso), `dense-batch-siluMul`,
`dense-batch-sigmoidMul`, `dense-batch-addInPlace`,
`dense-batch-deltanet-gates`. Il confronto e' col kernel per-riga eseguito M
volte: la prova non e' "il batch da' numeri plausibili" ma "da' gli STESSI bit".
E i casi non-batch (`rope-neox`, `deltanet-gates`, ...) continuano a passare, che
e' la verifica che l'idioma ha tenuto.

**Resta della fase 4** (voci 1, 5, 6 dell'inventario): l'attenzione a chunk per
q35, il path expert a gather per i K-quant, e l'orchestratore a M righe. Sono
le tre grosse, e sono nell'ordine in cui vanno fatte.

**Gate**: tsc pulito, suite 440|9, ktest **92/92**, nessun run del modello
(niente e' cambiato nel path di decode: il testo non-batch e' identico).

## it.26 (2026-08-11, fase 4) — l'attenzione a chunk: era una variante, non una famiglia

**Correzione all'inventario di it.24.** Avevo scritto che l'attenzione a chunk
era una delle due "famiglie di kernel nuove" e avevo motivato cosi':
`attnPrefillChunkWgsl` esiste ma e' del path Qwen 2.5, legge un `qkv` FUSO e non
e' un drop-in. Vero — ma la conclusione era sbagliata: **non serviva adattare il
kernel di un'altra famiglia, serviva dare il modo `batch` al kernel di q35**,
`attnDecodeWgsl`, che ha gia' i buffer separati (q, kCache, vCache) e la GQA.
Trenta righe invece di una famiglia. Avevo guardato cosa c'era di simile altrove
invece di guardare cosa avevo gia' in casa.

**Il pezzo che rende l'idea corretta**: `wid.y` = riga, `nPast` PER RIGA da
`rowPast`, q e out a offset di riga. **La causalita' viene gratis**: la riga m
somma sulle posizioni 0..rowPast[m], quindi vede se stessa e tutto cio' che la
precede — comprese le righe PRECEDENTI dello stesso chunk, che `kvAppend` ha
gia' scritto in cache — e non vede quelle dopo. Non serve una maschera: serve
che l'append preceda l'attenzione, e nell'encoder e' cosi'. `scores` e `red`
sono di workgroup e i workgroup sono (head, riga): ogni riga ha i suoi.

**GATE — ktest 93/93**, caso nuovo `dense-batch-attn-chunk`: **BIT-IDENTICO su 3
righe** con `nPast` CRESCENTE (9, 10, 11), cioe' proprio il caso in cui le righe
del chunk si guardano fra loro. Il confronto e' col kernel per-riga eseguito M
volte sulla stessa cache.

**Inventario della fase 4: 4 voci su 6.** Restano il path expert a GATHER per i
K-quant (GLM ce l'ha solo per q4_0/q4_1) e l'orchestratore a M righe. La taglia
che avevo corretto in it.24 (3-5 iterazioni) va a sua volta ridimensionata: una
delle due voci "grosse" era una variante.

**Gate**: tsc pulito, suite 440|9, ktest **93/93**, nessun run del modello (il
testo non-batch e' identico byte per byte, il decode non cambia).

## it.27 (2026-08-11, fase 4) — PROGETTO DELL'ORCHESTRATORE scritto prima di scriverlo, e un cambio d'ORDINE motivato

Le voci rimaste erano due — il gather K-quant (5) e l'orchestratore (6) — e
l'inventario le metteva in quest'ordine. **Le inverto, e il motivo non e'
comodita': e' il GATE.**

**Il done-when della riga 4 chiede "logits del prefill batched == logits
sequenziali sul campione (gate secco)".** Con gli expert PER RIGA dentro il
chunk — cioe' lasciando la catena expert esattamente com'e' oggi e batchando
solo il resto — quel confronto e' **BIT-IDENTICO per costruzione**, perche' ogni
kernel batched che ho aggiunto in it.25-26 e' ktestato BIT-IDENTICO per riga
contro il suo per-riga. Col gather, invece, l'ordine delle somme cambia
(l'unione visita gli expert in ordine di id, non di k) e GLM ha dovuto
inventare la struttura a slot + combine in ordine k proprio per recuperare
l'identita'. Fare prima l'orchestratore significa avere il gate PIU' FORTE
disponibile subito, e avere una base verificata su cui il gather diventa un
delta misurabile invece che una variabile in piu'.

**E costa poco in guadagno.** Sui numeri di it.23, il prefill batched a M=16:

| | con gather | senza gather (expert per riga) |
|---|---|---|
| row-parallel /M | 0,85 | 0,85 |
| expert | 6,86 | 8,72 |
| router | 3,16 | 3,16 |
| attn | 2,93 | 2,93 |
| ssmRec | 0,59 | 0,59 |
| **totale** | **14,38** | **16,25** |
| vs 28,96 di oggi | 2,01x | **1,78x** |

**Il gather vale 0,23x su 2,01x**: l'89% del guadagno della fase sta
nell'orchestratore, non nel gather. Fare prima la parte che vale l'89% e col
gate piu' forte e' l'ordine giusto, e il gather resta come incremento
misurabile.

**PROGETTO** (quello che la prossima iterazione implementa):
1. **Prima il tipo di layer DENSO** (4B/9B): stessa attenzione, stesso deltanet,
   FFN denso al posto del MoE. Esercita TUTTA la macchina batched (rmsnorm,
   gemv, rope, kvAppend, attn-chunk, gate deltanet, silu, add) e il gate e'
   bit-identico. Il MoE si aggiunge dopo, ed e' l'unico pezzo nuovo.
2. Scratch a M righe accanto a quelli per riga (non al posto: il decode resta
   quello che e', e la non-regressione e' per costruzione).
3. `rowPos`/`rowPast` come storage, riempiti per chunk.
4. Lista di step PARALLELA (`stepsB`), costruita nello STESSO giro di
   `steps` — non in un secondo giro, perche' due giri sullo stesso file di pesi
   sono due verita' che divergono al primo tensore aggiunto.
5. La ricorrenza (`deltaNetConv`/`deltaNetCore`) resta per riga DENTRO il chunk:
   M dispatch in sequenza. Costa 0,588 ms/token misurati (it.23) e non e'
   comprimibile senza una forma chunkwise del delta rule, che e' ricerca.
6. Gate: `prefillChunk(tokens, pos0)` contro `step()` sequenziale sullo stesso
   prompt, **logits bit-identici**, piu' il micro-bench tok/s prima/dopo che il
   done-when chiede.

**Rischio dichiarato**: la ricorrenza per riga dentro un encoder batched vuole
che i dispatch di conv/core della riga m vedano lo stato lasciato dalla riga
m−1. Sono dispatch nello stesso pass e WebGPU li ordina, quindi funziona — ma e'
la stessa proprieta' su cui si regge il path a submit unico (it.17) e va detta,
non assunta.

**Gate di questa iterazione**: nessun codice, e' il progetto della fetta scritto
prima di iniziarla (pattern di it.13/it.16/it.19/it.24). tsc pulito, suite
440|9, ktest 93/93 invariati dall'iterazione precedente.

## it.28 (2026-08-11, FASE 4-TER) — i 15,12 ms "fuori dai pass" NON sono CPU, e la mia ipotesi dell'item 17 era sbagliata

**Ruling PI incassato** (item 17, opzione (a)): riga 4-ter aggiunta a PHASES
fra la 4-bis e la 4, col done-when che include l'USCITA — se le voci aggredibili
sommano meno di ~8 ms la fase si chiude come esclusa coi numeri.

**Prima misura, e ribalta l'ipotesi.** Nell'item 17 avevo scritto che i 15,12 ms
erano "encode CPU dei ~2300 dispatch, submit, attesa del readback, argmax su
151k logit, dequant dell'embedding". Decomposti (smoke 35B, passata calda):

| voce | ms/token |
|---|---|
| attesa del readback (mapAsync) | 29,194 |
| **encode CPU dei ~2400 dispatch** | **1,267** |
| argmax su 151k logit | 0,431 |
| contabilita' di fine token | 0,210 |
| dequant della riga di embedding | 0,028 |
| **residuo non attribuito** | **11,906** |

**Il lavoro CPU e' 1,94 ms — il 4,5% del token.** L'encode dei 2400 dispatch, che
avevo messo per primo nella lista dei sospetti, costa 1,27 ms. E "l'attesa del
readback" non e' overhead: e' la GPU che lavora. Avevo sommato la GPU alla CPU e
chiamato il totale "fuori dai pass".

**Poi ho inseguito il residuo, e due ipotesi su tre sono cadute.**
1. **Lo snapshot dello stato ricorrente** (62,8 MiB per token, nato in it.18): il
   sospetto numero uno, perche' e' la piu' grossa operazione GPU fuori dai pass.
   Misurato con una sonda che lo salta (`--no-snapshot`, che LANCIA se serve un
   replay, cosi' non puo' dare numeri sbagliati in silenzio): **0,30 ms**.
   Refutata.
2. **Il checkpoint dell'hidden** (40 copie da 8 KB): **~0** (42,764 → 42,792,
   dentro il rumore). Refutata.
3. **I due `popErrorScope` awaitati PRIMA del readback**: confermata come
   ATTRIBUZIONE, non come costo. Spostandoli dopo le mapAsync, l'attesa del
   readback passa da 29,19 a **40,43 ms** e il residuo crolla da 11,91 a
   **0,98** — ma il token NON accelera (43,06 → 43,32, dentro il rumore). Cioe'
   quei 11,9 ms erano gia' tempo GPU: `popErrorScope` si risolve quando il
   device ha processato il lavoro, quindi awaitarlo prima era una seconda attesa
   della stessa cosa, non un costo in piu'. La riordino comunque, perche' ora
   `readbackMs` significa "attesa totale della GPU" invece di una sua meta'
   arbitraria — e' una correzione di CONTABILITA', non un guadagno, e la scrivo
   cosi'.

**DOVE SIAMO, ONESTAMENTE.** Il token e' 43,1 ms: **~40,4 di GPU** (attesa vera)
e **1,94 di CPU**. Ma la somma dei pass cronometrati e' 29,5 ms (it.23): restano
**~11 ms di tempo GPU che nessun pass contiene**. I pass boundary da soli non lo
spiegano — la sonda che ne aggiunge 80 e poi 160 costa 1,2 ms per volta, cioe'
~10 us l'uno, e i 41 boundary del regime normale valgono ~0,4 ms.

**Il sospetto rimasto, e l'esperimento gia' scritto per la prossima iterazione**:
i **40 `clearBuffer(moeAcc)` per token**, che sono l'unica ragione per cui il
pass si SPEZZA a ogni layer (piu' il checkpoint dell'hidden, che pero' e'
replay-only e costa 0). Se il costo sta li', si toglie sostituendo il
`clearBuffer` con un dispatch di azzeramento DENTRO il pass — e il token
diventa un pass solo invece di 41.

**Il done-when della 4-ter NON si applica ancora**: le voci CPU sommano 1,94 —
sotto la soglia degli 8 ms — ma la fase non si chiude per esclusione, perche' la
decomposizione ha spostato il bersaglio invece di eliminarlo: gli ~11 ms
esistono, sono GPU, e hanno un esperimento da una riga che li spiega o li
esclude. Chiudere adesso sarebbe usare la lettera della riga contro il suo scopo.

**Gate**: tsc pulito, suite 440|9, argmax **39/39**, routing identico chiave per
chiave, miss 0 in tutti e quattro i run; JSON committati (`q35-cpubreak`,
`q35-nosnap`, `q35-nosnap2`, `q35-errscope`). Stabilita' run-to-run misurata di
passaggio: 43,06 e 43,24 sullo stesso codice, +0,4%.

## it.29 (2026-08-11, FASE 4-TER) — la quarta ipotesi cade, e la fase si CHIUDE come il suo done-when prevede

**L'esperimento scritto in it.28**: i 40 `clearBuffer(moeAcc)` per token sono
l'unica ragione per cui il pass si spezza a ogni layer; se gli ~11 ms di GPU
fuori dai pass sono costo di boundary, togliendoli si vedono.

**Due modi, ed erano entrambi migliorativi sulla carta.** (a) L'azzeramento si
puo' ELIMINARE, non spostare: l'axpy dello shexp e' il primo contributo di
`moeAcc` nel layer, e se SCRIVE invece di accumulare l'accumulatore non va
azzerato (`axpyWgsl(..., assign)`; `0 + w·x` e `w·x` sono lo stesso f32 per ogni
valore finito). Toglie 40 `clearBuffer` dal path ottimistico e 40 `writeBuffer`
da quello sync. (b) Il checkpoint dell'hidden, l'altra operazione d'encoder, da
`copyBufferToBuffer` a DISPATCH (`copyRowWgsl`, riga dall'uniform a dynamic
offset). Con entrambe, il token diventa **1 pass invece di 41**.

**MISURATO (mediana di 3, bracci interleavati, prima coppia scartata):**

| variante | pass/token | ms/token |
|---|---|---|
| baseline (clearBuffer + copia d'encoder) | 41 | **43,32** [43,09-43,85] |
| assign + copyRow a dispatch | **1** | 44,26 [43,81-44,26] |
| assign + copia d'encoder | 41 | 44,36 [43,69-44,98] |

**Nessun guadagno, e le differenze stanno dentro il rumore**: sullo STESSO codice
avevo gia' misurato 43,06 e 43,24 (+0,4%), e qui le bande [43,09-43,85] e
[43,69-44,98] si sovrappongono. La conclusione non dipende dal segno: **l'effetto
e' ≤1 ms contro gli ~11 previsti dall'ipotesi. Refutata.**

**Ho rimesso l'albero com'era.** Il cambio non e' peggiorativo in modo
dimostrabile, ma non e' nemmeno migliorativo, e lo stato che ha misurato meglio
e' quello di prima: `git checkout` sui sorgenti, e restano solo i JSON. La regola
di casa e' preferire lo stato osservato-funzionante quando non si puo' nominare
il guasto che si sta correggendo; qui il guasto non c'e'.

**LA FASE 4-TER SI CHIUDE COME ESCLUSA COI NUMERI**, che e' esattamente cio' che
il suo done-when prevede ("se le voci aggredibili sommano meno di ~8 ms").
Bilancio di quattro ipotesi:

| voce | misurata | esito |
|---|---|---|
| encode CPU dei 2400 dispatch | 1,267 ms | il sospetto n.1 dell'item 17, ed e' piccolo |
| argmax + contabilita' + embed | 0,669 ms | |
| snapshot dello stato ricorrente | 0,30 ms | refutata |
| checkpoint dell'hidden | ~0 | refutata |
| `popErrorScope` prima del readback | 0 | era attribuzione, non costo |
| spezzare il pass a ogni layer | ≤1 ms, nel rumore | refutata |
| **totale aggredibile trovato** | **~2,2 ms** | sotto la soglia degli 8 |

**Cosa resta non spiegato, e va detto**: il token e' 43,3 ms, di cui ~41 di
attesa GPU, ma la somma dei pass cronometrati e' 29,5. Restano **~11 ms di tempo
GPU che nessun pass contiene e che nessuna delle quattro ipotesi spiega**. La
spiegazione residua piu' plausibile — e non l'ho verificata — e' la latenza del
round-trip GPU→CPU per token (wire Dawn + event loop del browser), che e'
inerente a "un sync per token" e si attaccherebbe solo col PIPELINING: encodare
il token N+1 mentre il readback di N e' in volo. E' un cambio di semantica del
decode, non una riga di questa fase. Va sul docket (item 18), non deciso qui.

**Gate**: tsc pulito, suite 440|9, argmax **39/39** e routing identico in tutti e
tre i run dell'esperimento, JSON committati (`q35-onepass`, `q35-noclear`).
Albero identico a it.28 (`git checkout`).

## it.30 (2026-08-11, fase 4) — l'ultimo pezzo di kernel: la ricorrenza indicizzata per riga

Progettando l'orchestratore (it.27) ho trovato il pezzo che il piano non aveva:
**dentro un layer batched, conv e core girano per RIGA**, e i loro bind group
oggi puntano ai buffer per-riga. Le due strade erano M bind group per layer per
kernel — 30 layer x 16 righe x 2 = **960 bind group** — oppure indicizzare la
riga da un uniform. La seconda, per la stessa ragione per cui i kernel d'arena
prendono lo slot da `Sel`: l'indirizzo non sta nel bind group.

**`rows` su `deltaNetConvWgsl` e `deltaNetCoreWgsl`**: x/outv (conv) e
convOut/beta/g/z/outv (core) diventano matrici [M, ...] e la riga arriva da un
uniform. Lo STATO no: `state` e `S` non sono per riga — sono la memoria che
attraversa il chunk, ed e' esattamente il motivo per cui questi due kernel
restano M dispatch in ordine invece di diventare uno solo. Senza `rows` il testo
emesso e' identico byte per byte.

**Perche' `rows` e non `gid.y` come negli altri batched**: gli altri kernel
batched fanno tutte le righe in UN dispatch, e la riga e' una coordinata del
lancio. Qui i dispatch restano M e devono essere ORDINATI: la riga e' un
parametro, non una dimensione. Sono due idiomi diversi perche' sono due cose
diverse, e confonderli avrebbe prodotto un kernel che sembra batchato e non lo e'.

**GATE — ktest 94/94**, caso nuovo `dense-rows-deltanet-recurrence`:
**BIT-IDENTICO su 3 righe** contro il kernel per-riga eseguito M volte **sullo
stesso stato che evolve**. E' il test forte: se l'indicizzazione della riga fosse
sbagliata, la catena divergerebbe al SECONDO passo, perche' la riga m legge lo
stato che ha lasciato la m−1. Un test su una riga sola non l'avrebbe visto.

**Inventario della fase 4: 5 voci su 6.** Non serve piu' nessun kernel per
l'orchestratore — resta solo orchestrazione (voce 6), piu' il gather K-quant
(voce 5) che it.27 ha spostato DOPO, col motivo (gate bit-identico subito, e il
gather vale 0,23x su 2,01x).

**Gate**: tsc pulito, suite 440|9, ktest **94/94**, nessun run del modello (il
testo non-`rows` e' identico, il decode non cambia).

## it.31 (2026-08-11, fase 4) — una collisione temuta che non c'era, e meno lavoro per l'orchestratore

Iniziando l'orchestratore ho visto una cosa che sembrava un problema:
`rmsnormWgsl` ha gia' un modo `batch`, e q35 **lo usa gia'** — ma per le HEAD
(`push(rmsnormWgsl(hd, eps, true), [qB, qNormW, qN], S.nHead)`), non per le
righe. Sembrava servisse una seconda dimensione, cioe' toccare il kernel.

**Non serve.** Il `batch` di quel kernel non e' "righe": e' **per-VETTORE**, con
`wid.x` = indice del vettore. E i buffer sono row-major col passo di riga uguale
a `nVec*len`, quindi il vettore (riga, head) sta esattamente all'indice
`riga*nVec + head`. Dispatchare `nVec*M` vettori invece di `nVec` fa gia' la cosa
giusta su un buffer [M, nVec*len]. **L'appiattimento e' l'identita', non una
approssimazione.**

Vale per tutta la famiglia per-vettore: `rmsnorm` per head e `stridedCopy` (che
estrae q e gate da `qFull` con `srcStride 2*hd`: per (riga, head) l'offset e'
`riga*2*qDim + head*2*hd = (riga*nVec + head)*2*hd`, e torna).

**GATE — ktest 96/96**, due casi nuovi alla geometria VERA di q35:
`dense-flat-rmsnorm-per-head` e `dense-flat-stridedcopy`, **BIT-IDENTICI su 3
righe** contro il per-riga eseguito M volte. Li ho scritti coi numeri di q35
(nHead x headDim, `srcStride 2*headDim`) e non con una geometria comoda, perche'
la proprieta' che si sta provando e' un'aritmetica di offset e vive o muore sui
passi reali.

**Conseguenza sull'orchestratore**: due kernel in meno da toccare, e soprattutto
la conferma che **non serve piu' nessun cambio di kernel** — l'ultimo dubbio
aperto era questo. Restano solo i buffer a M righe, la lista di step gemella e
il driver.

**Perche' mi fermo qui invece di scrivere l'orchestratore**: e' l'ultima fetta
grossa (~250 righe, un gate end-to-end e un run), la sessione e' lunga, e la
regola del loop dice di consegnare su un confine pulito quando il dubbio sul
budget e' concreto invece di lasciare un merge a meta'. Il prossimo giro parte da
codice, non da ricognizione: sotto c'e' la lista esatta.

**Da fare, in ordine** (nessuno dei punti ha piu' incognite):
1. `Q35GpuModelOpts.prefillM?: number` — quando c'e', si costruisce anche la
   lista gemella; quando manca, non cambia una riga (il 35B di default non la
   vede nemmeno).
2. Scratch a M righe accanto a quelli per riga (xM, xnM, qkvM, ...).
3. `stepsB[]` + `pushB()` nello STESSO giro di `steps`, coi kernel batched gia'
   ktestati; per-vettore = appiattimento (nVec x M), per-riga = `gid.y`,
   ricorrenza = `rows` con M bind group per layer che differiscono solo
   nell'offset dell'uniform.
4. `prefillChunk(tokens, pos0)`: dequant di M righe, `rowPos`/`rowPast`, un
   encoder, head sulla SOLA ultima riga (e' l'unica che serve al prefill).
5. Gate: `prefillChunk` contro `step()` sequenziale sullo stesso prompt del 4B,
   **logits bit-identici**, piu' il micro-bench tok/s che il done-when chiede.

**Gate**: tsc pulito, suite 440|9, ktest **96/96**, nessun run del modello.

## it.32 (2026-08-11, fase 4) — l'orchestratore gira, i logits sono BIT-IDENTICI, e lo speedup e' 1,15x invece di 2

**Montato**: `prefillM` (opt-in; assente, non cambia una riga), scratch a M righe
accanto a quelli per riga, lista di step GEMELLA costruita nello stesso giro di
`steps`, e `prefillChunk(tokens, pos0)` — M token in un submit, con la head sulla
SOLA ultima riga (l'unica che il prefill produce). I tre idiomi convivono come
progettato: per-vettore = appiattimento (nVec x M), per-riga = `gid.y`,
ricorrenza = `rows` con un bind group per riga che differisce solo nell'offset
dell'uniform.

**GATE (4B, smoke, M=8, 5 chunk): logits BIT-IDENTICI, 1.241.600 su 1.241.600,
maxAbs 0.** Non "vicini": identici. Era l'atteso per costruzione — ogni kernel
batched e' ktestato bit-identico per riga — ed e' il done-when della riga 4
soddisfatto alla lettera.

**E QUI IL NUMERO CHE CORREGGE LA MIA PROIEZIONE.** Misurato (primo chunk
SCARTATO, docket item 10 — e il perche' si vede: 489 ms il primo chunk
sequenziale contro ~240 i successivi):

| | ms/token |
|---|---|
| sequenziale | 29,88 |
| a chunk M=8 | **25,97** |
| **speedup** | **1,151x** |

La proiezione di it.23 diceva ~2x, e si reggeva su "il traffico dei pesi si legge
una volta per chunk". **Non e' vero per questi kernel**: il modo `batch` di
`gemvQuant`/`gemvF32`/K-quant mette `wid.z` = riga e ogni workgroup (riga, riga
d'uscita) **rilegge la propria riga di pesi**. Fonde i DISPATCH, non il
traffico. Il guadagno misurato e' quello che resta — meno dispatch e un po' di
riuso in L2 — ed e' 1,15x, non 2.

**Per avere il 2x servirebbe una GEMM vera**, che carica una tile di pesi e la
moltiplica per M righe: e' la famiglia `rmsPairGemmSiluChunkFast` del path Qwen
2.5, che esiste ma per un'altra geometria. Non e' una riga di questa fetta: e'
un'altra fetta, e va decisa coi numeri (docket item 19).

**M=16 non misurato**: il gate pretende almeno 3 chunk per poter scartare il
primo, e lo smoke del 4B ha 47 token — a M=16 fanno 2. Il guard ha fatto il suo
mestiere invece di lasciarmi citare una mediana di due campioni di cui uno
freddo. Serve un prompt piu' lungo, ed e' materia del bench, non del gate.

**Una cosa che l'inventario aveva sbagliato**: it.24 concludeva "sul 35B i pesi
statici sono tutti Q8_0/F32, i K-quant stanno solo negli expert". Vero per il
35B — ma il **4B ha `ssm_out` in Q5_K**, e il primo run e' morto proprio li'.
L'inventario valeva per UN MODELLO e l'avevo scritto come se valesse per la
famiglia. Aggiunto il modo `batch` ai tre gemv K-quant non-arena (con guard
esplicito: `batch` e `arena` non si combinano — l'arena e' il path expert, il
batch il prefill).

**Gate**: tsc pulito, suite 440|9, logits bit-identici 1.241.600/1.241.600,
JSON committato (`q35-prefillchunk8-4b-it32.json`).

## it.33 (2026-08-11, fase 4) — il guadagno cresce col CONTESTO, non con M

Volevo la curva in M che la riga 4 chiede come micro-bench. Il primo giro me
l'ha fatta sbagliare e vale la pena raccontarlo: avevo cappato il gate a **6
chunk**, quindi a M=8 misuravo 48 token di contesto e a M=16 ne misuravo 96 —
due contesti diversi spacciati per due M diverse. Il tetto va sui TOKEN.

**A parita' di contesto (1024 token, 4B, primo chunk scartato):**

| M | chunk | ms/token seq → chunk | speedup |
|---|---|---|---|
| 8 | 128 | 32,39 → 26,59 | **1,218x** |
| 16 | 64 | 31,98 → 25,88 | **1,236x** |

**M quasi non conta**, ed e' coerente con l'item 19: il modo `batch` fonde i
dispatch, non il traffico dei pesi, quindi raddoppiare le righe non raddoppia
niente.

**Ma il CONTESTO conta, e molto.** Un run precedente (non cappato, M=8, 807
chunk = 6456 token, 806 campioni dopo lo scarto) ha dato **60,66 → 30,05
ms/token = 2,019x**. Stesso codice, stessa M: cambia solo quanto e' lungo il
contesto. La spiegazione plausibile — e la dico come plausibile — e' che a
contesto lungo domina l'attenzione, e nel kernel a chunk le M righe leggono la
STESSA KV cache da workgroup concorrenti: il riuso non lo fa il kernel, lo fa la
cache della GPU. E' l'unica amortizzazione che questo meccanismo ottiene, e
arriva da dove non l'avevo cercata.

**Quindi il numero della fase 4 non e' un numero, e' una curva**: 1,22x a 1k di
contesto, **2,02x a 6,5k**. E la seconda meta' e' il regime che conta, perche' il
prefill lungo e' esattamente il caso in cui il TTFT fa male.

**Gate in tutti e tre i run: LOGITS BIT-IDENTICI** — 31.784.960 su 31.784.960 a
M=8, 15.892.480 su 15.892.480 a M=16, 200.394.240 su 200.394.240 nel run lungo.
Duecento milioni di f32 confrontati bit a bit senza una differenza.

**Correzione all'harness**: il tetto del gate ora e' in TOKEN (1024) e non in
chunk, cosi' due M girano sullo stesso contesto e la M resta l'unica variabile.

**Gate**: tsc pulito, suite 440|9, JSON committati (`q35-prefillT8`,
`q35-prefillT16`, `q35-prefillM8`).

## it.34 (2026-08-11, fase 4) — il prefill a chunk sui MoE: il bivio, e perche' NON uso il path ottimistico

Portare `prefillChunk` sul 35B vuol dire decidere chi sceglie gli expert dentro
il chunk. Due strade, e la differenza non e' di comodita':

**(A) Il router su GPU (path ottimistico di it.17).** Zero readback, ma il
prefill e' esattamente la fase FREDDA: i miss sono il regime (it.16: 39 token su
39 sporchi a cache vuota), quindi servirebbe repair+replay **a livello di
chunk** — e le righe di un chunk sono interdipendenti, quindi non e' il replay
di it.18 con un indice in piu': e' un'altra cosa.

**(B) La selezione sulla CPU, col readback BATCHATO.** Per layer: segmento
statico batchato per tutte le M righe → UN readback dei logit del router per
tutte le M righe → selezione ed `ensure` per riga sulla CPU → catene expert.
Sono **40 readback per CHUNK invece di 40 per TOKEN**: la riduzione dei sync e'
M volte, che e' il grosso di cio' che il batch del prefill puo' dare, e il
regime freddo resta gestito com'e' oggi (l'`ensure` precede il dispatch, sempre).

**Prendo (B)**, e non solo per la semplicita': con (B) la selezione e la catena
expert sono LE STESSE del path sequenziale, quindi il gate resta la
**bit-identita' dei logits** — che e' il done-when della riga 4 e la cosa piu'
forte che si possa chiedere. Con (A) il gate diventerebbe "argmax identico", piu'
debole, e in cambio di un meccanismo che nel regime del prefill e' in perdita.
E' la stessa logica con cui it.27 ha messo gli expert per riga invece del gather:
prima il gate forte, poi l'ottimizzazione che lo indebolisce, se paga.

**Conseguenza operativa**: nel chunk il MoE resta a 40 submit, mentre il resto
del layer e' batchato. Il guadagno atteso e' quindi minore che sul denso — e la
misura dira' quanto, invece di una proiezione.

### it.34 — il prefill a chunk sul 35B: 3,75x, BIT-IDENTICO, e tre bug presi dal gate

**Montato** (strada (B), quella decisa sopra): il segmento statico del layer MoE
e' batchato (shexp, scalare, router GEMV); per layer UN readback dei logit del
router per tutte le M righe; poi selezione ed `ensure` per riga sulla CPU e le M
catene expert in un submit. `runLayer` si e' spezzata in `prepLayer` (la meta'
CPU) e `encodeExperts` (la meta' GPU), cosi' il path sequenziale e quello a
chunk usano LO STESSO codice invece di due copie che divergono.

**IL GATE HA PRESO TRE BUG, e nessuno dei tre sarebbe morto rumorosamente.**
1. **`Sel` senza dimensione di riga.** Le M scritture del layer andavano tutte
   allo stesso offset e — visto che `queue.writeBuffer` e' ordinata PRIMA del
   submit — ogni riga avrebbe usato la selezione dell'ULTIMA. Risolto dando a
   `Sel` e a `MoeIdx` la dimensione riga (solo quando il chunk e' cablato).
2. **Il pin non copriva l'unione.** L'`ensure` della riga r+1 poteva evincere
   uno slot che i dispatch GIA' ENCODATI della riga r avrebbero letto, e la
   writeBuffer del nuovo slab sarebbe arrivata prima del submit: l'expert
   sbagliato, con numeri plausibili. Risolto calcolando l'unione delle M
   selezioni PRIMA di qualunque `ensure` (`pinUnion`).
3. **`moeAcc` per riga mai azzerato.** Nel path per riga lo azzera una
   `writeBuffer` per layer; nel chunk l'axpy dello shexp accumulava attraverso i
   40 layer.

Tutti e tre davano risultati **plausibili e sbagliati** — maxAbs 27,6 sui
logits, cioe' un modello che "funziona". Li ha presi il confronto bit a bit, non
un occhio.

**E UNA TRAPPOLA DI MISURA, la stessa di it.17**: il primo giro riportava
**30,8x**. Falso: il braccio sequenziale girava per primo a cache expert VUOTA e
quello a chunk la trovava calda — era la cache che si riempiva, non il batch.
Aggiunta una passata di scaldata prima dei due bracci.

**NUMERI A PARITA' (35B, smoke, M=8, primo chunk scartato):**

| | ms/token |
|---|---|
| sequenziale | 131,08 |
| a chunk M=8 | **34,95** |
| **speedup** | **3,750x** |

**Logits BIT-IDENTICI: 993.280 su 993.280, maxAbs 0.**

Il 3,75x e' molto piu' del 2,02x del denso a contesto lungo, e la ragione e'
strutturale: sul MoE il batch toglie anche i **40 readback per token**, che
diventano 40 per CHUNK. E' la voce che it.19 aveva misurato come dominante nel
path sequenziale, ed e' esattamente cio' che il prefill a chunk amortizza.

**Nota di contratto, e la scrivo invece di lasciarla implicita**: la riga 4 dice
"pattern `planMoeChunk` + gemv batch". Il done-when MECCANICO (micro-bench
prima/dopo + logits identici) e' soddisfatto, ma `planMoeChunk` — il piano a
unione con gather — NON e' usato: it.27 ha scelto gli expert PER RIGA proprio
per avere il gate bit-identico, e il gather resta l'incremento successivo
(docket item 19). Se il PI considera `planMoeChunk` parte del done-when e non
della descrizione, la riga non e' chiusa.

**Gate**: tsc pulito, suite 440|9, logits bit-identici 993.280/993.280 sul 35B e
31.784.960/31.784.960 sul 4B, JSON committati.

## it.35 (2026-08-11, FASE 5) — il budget dell'arena si DERIVA, e docket item 11 si chiude

Prima delle quattro voci della riga 5, e chiude anche un item aperto da it.14: il
budget dell'arena expert del 35B era **un parametro fisso, 12 GiB di default,
che nessuno controllava**. Su un host con 14,4 GiB liberi sfondava, e il
fallimento era un `VK_ERROR_OUT_OF_DEVICE_MEMORY` alla `createBuffer` — rumoroso
solo grazie al listener, coi buffer invalidi che lasciavano girare il modello su
numeri plausibili (top1 1/5 nel run viziato di it.14).

**Come lo deriva GLM e come lo derivo qui.** GLM sottrae termini CALCOLATI
(`slabBudgetCtxAware`: tetto − non-expert − KV − buffer di lavoro − riserva), e
ogni termine e' una formula che qualcuno deve tenere aggiornata — infatti il suo
commento racconta di un OOM storico dovuto a un termine dimenticato. Qui si puo'
fare di meglio: **il modello sa esattamente quanto ha preso**, perche' i buffer
passano tutti da due helper. Un contatore dentro `empty`/`sbuf` e il budget e'
`tetto − allocato − riserva`, senza formule da mantenere.

**Ed e' ctx-aware per costruzione**, che era la richiesta della riga: `kCache` e
`vCache` si allocano con `ctxMax` e finiscono nel contatore come tutto il resto.
Lo stesso vale per il piano a M righe del prefill, se acceso — nessuno deve
ricordarsi di aggiungerlo alla formula, perche' non c'e' formula.

**Misurato (35B, tetto 13 GiB dichiarato)**: allocati **1,94 GiB** (pesi
non-expert + KV a ctx 47 + scratch), riserva 0,50, **budget expert 10,56 GiB**.
L'1,94 e' un numero che prima nessuno aveva: era dentro la differenza fra il
budget chiesto e la VRAM vera, ed e' esattamente la parte che faceva sfondare.

**Gate**: run verde col budget derivato — argmax **39/39 identico**, routing
identico chiave per chiave, 0 miss a caldo, 43,74 ms/token (in banda coi ~44 di
riferimento). tsc pulito, suite 440|9. **Docket item 11 CHIUSO.**

Il vecchio parametro resta come default per non cambiare i run esistenti: senza
`vramCeilingBytes` il comportamento e' quello di prima, e il JSON adesso porta
la scomposizione (`vramPlan`) invece di un numero asserito.

## it.36 (2026-08-11, fase 5) — la POLICY d'ingresso non serve, ed e' un numero a dirlo

Seconda voce della riga 5. Il progetto di it.16 prevedeva una soglia — sync
finche' i miss/token stanno sopra, ottimistico sotto, con isteresi — perche' a
cache fredda ogni token e' sporco e il replay sembrava dover costare piu' del
sync. it.18 aveva gia' misurato il contrario (738 contro 1193) ma l'aveva
dichiarato NON una misura: un campione per braccio, in due run diversi.

**Strumento**: `debugEvictAll` (solo harness) rimette la cache a vuota, cosi' i
due path partono entrambi da freddo **nello stesso processo**. La cache fredda
esiste una volta sola per processo, ed e' il motivo per cui finora i due bracci
stavano in due run.

**E l'ho fatto nei DUE ORDINI**, perche' il secondo braccio rilegge dal GGUF
range che il primo ha gia' toccato — se la pagina e' in cache di sistema, il
secondo braccio ha un vantaggio che non c'entra col meccanismo:

| | sync a freddo | ottimistico a freddo |
|---|---|---|
| ordine A (sync per primo) | 1111,50 | 650,92 |
| ordine B (ottimistico per primo) | 1098,40 | 660,67 |

**L'ordine sposta l'1,2%**: il confondente e' limitato e la conclusione tiene.

**RISULTATO: a freddo l'ottimistico e' 1,68x PIU' VELOCE del sync** (~656 contro
~1105), e a caldo 3,05x (43,57 contro 132,81). Il replay costa davvero — 109
replay, +12% di fetch, l'80,7% del token rigiocato ogni volta — ma costa **meno
dei 77 round-trip per token** che il path sync paga comunque. E' coerente con
it.19: nel path sync il tempo sta nei sync, non nel calcolo.

**Quindi la soglia NON SERVE, ed e' esclusa coi numeri** — che e' esattamente
cio' che il done-when della riga 5 ammette ("delta misurato O esclusione
motivata coi numeri"). Il decode ottimistico si accende e basta. La decisione e'
scritta nel codice accanto al flag, coi numeri, perche' chi la trovera' fra sei
mesi vedra' un `let optimisticOn = ...` senza policy e si chiedera' se manca
qualcosa.

**Chiude anche il docket item 13**, che elencava i tre numeri con cui tarare la
soglia: la soglia non c'e' piu', e i tre numeri restano come descrizione del
regime freddo.

**Gate**: tsc pulito, suite 440|9, argmax **39/39 identico** e routing identico
in entrambi i run, JSON committati (`q35-coldboth`, `q35-coldboth-rev`).

## it.37 (2026-08-11, FASE 5) — prefetch e AUTOPIN: due esclusioni, una strutturale e una misurata

Ultime due voci della riga 5. Il done-when ammette "delta misurato O esclusione
motivata coi numeri", e qui sono due esclusioni — ma di natura diversa, e la
differenza va detta.

**PREFETCH IN-FORWARD: escluso per STRUTTURA, con i numeri accanto.** Il
prefetch di GLM esiste per riempire una finestra precisa: mentre la CPU aspetta
il readback del router del layer L, fetcha gli expert predetti per L+1. Su q35
quella finestra **non esiste piu'**: la fetta 3c ha portato i readback da 41 a 1
per token. E non e' che sia solo piccola — durante il token la slotTable e'
INTOCCABILE (invariante I1, `setInFlight`), quindi un prefetch li' dentro non
sarebbe lento: sarebbe illegale.

Resta il confine di token, dove pero' i fetch che servono sono quelli del
REPAIR, e il repair sa ESATTAMENTE cosa gli manca (la lista viene da `Sel`),
mentre il prefetch predice: il recall q35 e' **82,67%@8**, cioe' aggiungerebbe
~17% di fetch sbagliati a un path che gia' fetcha esattamente il necessario.
E il numero che chiude: senza prefetch, a freddo, l'ottimistico e' gia' **1,68x
piu' veloce del sync** (it.36).

**TIER / AUTOPIN: escluso per MISURA, sotto pressione vera.** Cablata la policy
su q35 (non c'era: senza cablaggio non c'e' ne' delta ne' esclusione) e messa la
cache sotto pressione col budget derivato — `--vram-gib 6` da' 3,56 GiB, cioe'
**2134 slot per 3341 expert distinti**, quindi eviction vere.

| | LRU | TIER (AUTOPIN) |
|---|---|---|
| sync-cold, miss | 3601 | 3601 |
| sync-warm, miss | 2985 | **2980** |
| ottimistico-warm, miss | 3592 | **3567** |
| miss totali del run | 29.909 | **29.868** |
| ms/token (ott. caldo) | 653,4 | 645,5 |

**−0,14% di miss, e il tempo dentro il rumore.**

**IL CAVEAT, che e' la parte onesta.** AUTOPIN ha soglie proprie:
`AUTOPIN_MIN_HIST` = 5000 selezioni prima di pinnare qualcosa, repin ogni 2944.
Questa passata ne fa 12.480, quindi la policy comincia ad agire dopo il 40% del
run e fa due o tre passate di repin in tutto. **L'esclusione vale su questo
corpus**, e il motivo per cui vale e' che il corpus e' corto per le soglie della
policy stessa — non che AUTOPIN non funzioni. Ri-misurarla su corpus pieno e'
lavoro della fase 6, dove i riferimenti si fanno comunque.

**Un rilievo di margine**: `moeStats` non riporta la policy attiva (`policy:
None` nel JSON anche col tier acceso). Due run che differiscono per la policy
non si distinguono dal report — va sistemato prima della fase 6, che di report
vive.

**FASE 5 CHIUSA**, voce per voce: budget ctx-aware **derivato** (it.35, docket
11) · decode ottimistico attivo, **soglia esclusa coi numeri** (it.36, docket
13) · prefetch **escluso** (struttura + recall 82,67%) · tier/AUTOPIN **escluso**
(−0,14% misurato sotto pressione, col caveat del corpus) · **GLM invariato**:
ktest **96/96**, `glm-model-2layer` a L2rel 2,072937787401139e-07, la stessa
cifra fino all'ultima decimale.

**Gate**: tsc pulito, suite 440|9, ktest 96/96, argmax 39/39 e routing identico
in entrambi i run sotto pressione, JSON committati.

## it.38 (2026-08-11, fase 6) — i sensori PRIMA dei bench, e il piano del CHECKPOINT A

**Perche' i sensori prima.** La regola di casa e' che i bench lunghi si fanno su
codice finale, e il CHECKPOINT A e' fatto di ore di GPU. Un report che mente le
invalida tutte, e ne avevo tre di aperti (docket item 12 + il rilievo di it.37):

1. **`moeStats` non riportava la policy attiva.** Due run che differiscono per la
   policy — cioe' esattamente il confronto di it.37 — si distinguevano solo dal
   comando. Aggiunta.
2. **`dispatchesPerToken` non era il numero di dispatch per token.** Era
   `steps.length`, cioe' i soli step STATICI: non ha mai contato i per-expert.
   Ora c'e' `dispatchBreakdown` con `static` / `dynamic` / `total`, e sul 35B il
   totale e' **2.102** contro i 782 che il campo pubblicava (782 statici + 1.320
   dinamici). Il vecchio campo resta per non rompere i JSON storici, ma accanto
   c'e' la verita'.
3. **`readTap` nel path ottimistico restituiva un array VUOTO.** Adesso lancia:
   uno strumento di debug che tace quando dovrebbe urlare e' peggio che non
   averlo.

Piu' `vramPlan` nel report della conformance: il budget derivato si vede, invece
di essere un numero che qualcuno ha passato.

**Verificato**: gate verde sul 35B dopo i tre cambi (argmax 39/39, routing
identico, `policy: "lru"` nel report, 44,37 ms/token in banda). tsc pulito,
suite 440|9.

**PIANO DEL CHECKPOINT A** (scritto qui perche' e' un merge gate e l'ordine
conta; i run sono ore di GPU e vanno lanciati su albero congelato):

1. **Albero congelato e host dichiarato.** Nessun cambio di codice dopo l'inizio
   dei run: i runner scrivono `hostState` PRIMA e DOPO, ed e' quello che rende
   confrontabili i numeri con i riferimenti storici.
2. **GLM non-regressione PIENA** — e' la prima perche' e' il gate che protegge
   il modello-tesi: `glm-bench-run.mjs` b12 optimistic contro
   **13.172 / 31,26 / 14,74** in banda ±5%, golden AL PIN, cpuref, firma.
   Se questa non passa, il resto non conta.
3. **Riferimenti q35 NUOVI**: 4B, 9B e 35B ai tier 8/12/16 GiB, con
   `--vram-gib` (il budget ora si DERIVA, quindi i tier sono tetti veri e non
   budget asseriti) e `hostState`.
4. **Gap nativo ri-misurato a parita'** — stesso host, stessa GPU scarica.
5. **Ratchet golden q35 riverificati** (conformance full sui tre modelli).
6. **`direction §7-bis` riscritto coi numeri veri**, e via la marcatura stale.

**Nota sull'ordine 2-prima-di-3**: sono ore di GPU e se GLM regredisce vanno
buttate. Il gate piu' fragile va per primo.

## it.39 (2026-08-11, fase 6) — il primo braccio del CHECKPOINT A si ferma, e NON e' il codice

Lanciata la non-regressione GLM (b12 optimistic, prompt 6, 64 gen, 3 rep), che
il piano di it.38 metteva per prima perche' e' il gate piu' fragile. **Morta**,
con una cascata di `Invalid BindGroup`. Ma il primo errore del log non e' quello:

```
GPU error: vkAllocateMemory failed with VK_ERROR_OUT_OF_DEVICE_MEMORY
 - While calling [Device].CreateBuffer([BufferDescriptor]).
```

Tutti gli `Invalid BindGroup` che seguono sono **conseguenze**: buffer nati
invalidi e poi bindati. Guardare l'ultimo errore invece del primo avrebbe fatto
cercare un bug nei bind group — che e' esattamente dove non e'.

**IPOTESI E PROVA, senza toccare codice.** Il JSON del riferimento porta
`vramPeakMiB: 15160` e uno `hostState` con `memUsedMiB: 817` ("user-session-light"
dichiarata). Oggi:

| | MiB |
|---|---|
| picco che il riferimento richiede | **15.160** |
| liberi ora | 13.971 |
| **mancano** | **1.189** |
| usati dalla sessione ora | 1.977 |
| usati quando fu preso il riferimento | 817 |

I 1.160 MiB in piu' che la sessione desktop tiene oggi sono, entro 30 MiB,
esattamente i 1.189 che mancano. **Non e' una regressione: e' l'host.** E' la
stessa diagnosi di it.14 (dove il braccio non modificato moriva uguale), fatta
stavolta senza spendere un run perche' il riferimento porta con se' il suo
`vramPeakMiB`.

**E ktest lo conferma dal lato correttezza**: 96/96 con `glm-model-2layer` a
L2rel 2,072937787401139e-07, la stessa cifra fino all'ultima decimale. Cio' che
manca al checkpoint non e' la correttezza di GLM: e' la sua PRESTAZIONE misurata
a parita' di host.

**BLOCCO, e non lo decido io.** La riga 6 pretende "host DICHIARATO, GPU
scarica": la condizione non e' soddisfatta e non e' una cosa che tocchi al
motore — significa liberare 1,2 GiB di VRAM sulla macchina, cioe' chiudere roba
sul desktop di chi lavora. Docket item 20, con due strade e il mio parere.

**Cosa PUO' procedere intanto**: i riferimenti q35, che stanno nei tetti
disponibili — il 35B a `--vram-gib 13` ha girato oggi senza problemi (budget
derivato 10,56 GiB, picco molto sotto). Non li lancio in questa iterazione
perche' il piano di it.38 li mette DOPO il gate di GLM, e cambiare l'ordine per
"intanto facciamo qualcosa" e' esattamente il modo di ritrovarsi con ore di GPU
spese su un albero che poi non passa il gate. Aspetto il ruling.

## it.40 (2026-08-11, fase 6) — il CHECKPOINT A riparte: docket 20 evaporato, primo braccio IN BANDA

**Il blocco di it.39 si e' sciolto senza un ruling, e non perche' abbia deciso
io: la premessa non esiste piu'.** L'item 20 chiedeva al PI se liberare ~1,2 GiB
di VRAM sulla macchina di chi lavora. Ri-ancorato da disco a inizio sessione,
`nvidia-smi` dice **577 MiB usati / 15.372 liberi**: la sessione desktop e' oggi
piu' leggera di quando fu preso il riferimento (817). La condizione della riga 6
("host DICHIARATO, GPU scarica") e' soddisfatta a costo zero. Item 20 CHIUSO
come SUPERATO DAI FATTI — non e' stata scelta la (a), la (b) o la (c): la
domanda e' decaduta.

**PRIMA del braccio, il banco di prova interattivo** (richiesta PI in sessione):
`chat.html` + `src/engine/chat/`. E' il primo posto del repo dove il motore gira
NON teacher-forced. Vincolo che mi sono dato e verificato: **non tocca una riga
del motore** — serviva tutto gia' pubblico — cosi' l'albero resta congelato come
il checkpoint pretende. L'unico file tracciato modificato e' `vite.config.ts`
(una entry di build). tsc pulito, suite 440|9 invariata.

**DUE RUN BUTTATE, stessa causa, ed e' mia.** Ho dedotto la semantica di un flag
invece di leggerla.
1. Prima invocazione senza `--prefill-batch`: il default del runner e' `0` e il
   riferimento e' a **chunked M=16**. Esito: prefill 13,02 contro 31,26 (-58%),
   TTFT 35,4 s contro 14,7. Preso confrontando `config.prefillPath` campo per
   campo col riferimento, non guardando i tok/s.
2. Seconda invocazione con `--prefill-batch 16`: **non ha fatto niente**. La
   pagina legge `q.get("prefillbatch") === "1"` — e' un BOOLEANO, la M e' la
   costante `GLM_PREFILL_M`. Un valore fuori dominio non e' un errore, e'
   silenziosamente `false`.
Corretto il commento d'uso di `glm-bench-run.mjs`, che non elencava ne'
`--select` ne' `--prefill-batch` (ne' prefetch/policy/park-frac/ctx-max/
host-state): e' il documento che avrebbe evitato entrambe. Le due run
mis-invocate NON stanno in `results/`: erano errori di invocazione, non misure.

**FALSO ALLARME che vale la pena scrivere**: la prima run urlava `hit 0.00%
(retention 0.00%)` e verrebbe da cercare un bug nella cache. Ce l'ha anche il
riferimento, con la sua nota accanto: nel path ottimistico solo i MISS passano
da `ensure`, quindi hitRate 0 e' la semantica, non un sintomo. Confrontare col
riferimento invece di leggere il numero da solo e' cio' che ha impedito di
partire per la tangente.

**BRACCIO 2 DEL CHECKPOINT A (non-regressione GLM, b12 optimistic) — PASSA.**
`config` IDENTICA al riferimento (confronto programmatico, non a occhio):

| metrica | rif. 2026-08-09 | oggi | delta |
|---|---|---|---|
| decode tok/s | 13,172 | **13,437** | **+2,01%** |
| prefill tok/s | 31,265 | **31,813** | **+1,75%** |
| TTFT ms | 14.744,9 | **14.490,8** | **-1,72%** |

Tutte e tre **in banda +/-5%** e tutte e tre dalla parte giusta. Dispersione
sulle 3 repliche dichiarata (stdev decode 0,265 contro 0,125 del riferimento:
oggi rumore un po' maggiore, banda comunque larga il doppio del delta).

**Il numero che qualifica la non-regressione NON sono i tok/s: e' la MECCANICA,
identica all'ultima cifra.** `residencyRatio` 0,8210 = 0,8210 · `pDirty` 0,9375
= 0,9375 · `replaysPerToken` 1,1875 = 1,1875 · `replayFracPerDirty` 0,6459 =
0,6459 · `missesPerToken` **4,781 = 4,781**. Il motore prende esattamente le
stesse decisioni di residenza sugli stessi token: cio' che varia e' solo il
tempo di parete. Un delta di tok/s in banda con la meccanica che deriva sarebbe
una non-regressione fortunata; questa e' una non-regressione strutturale.

**EXIT CODE 4, e non e' una regressione — e' una trappola del runner.** I gate
che il runner valuta sono il FLOOR CPU di llama.cpp (13,43 decode / 56,58
prefill) e lo strutturale `<= 2 submit/token`. Il riferimento li fallisce
ENTRAMBI (`decodePass: false`, `prefillPass: false`, submits 2,1875). Oggi:
prefill FAIL come il riferimento (31,81 vs 56,58), strutturale FAIL con lo
STESSO valore (2,1875), e **decode PASS — 13,437 supera i 13,43 che il
riferimento non superava**. Quindi exit 4 e' l'esito NORMALE di questa run e
l'exit code NON distingue "regredito" da "ancora sotto il floor CPU": chi
automatizzasse il checkpoint sull'exit code leggerebbe un PASS come FAIL.
Annotato nel runner e aperto come docket item 21.

**Host**: dichiarato `user-session-light` come il riferimento, e piu' leggero
davvero — `memUsedMiB` 615->623 contro 817->825; `vramPeakMiB` **14.974** contro
15.160. Temperatura 46->50 (rif. 40->51).

**Resta del braccio 2**: golden AL PIN, cpuref e firma (le conformance full
GLM), che sono ~96+95 min e non sono state lanciate in questa iterazione.

## it.41 (2026-08-11, fase 6) — il 2a rifatto a host QUIESCENT: la banda non serviva

Rilanciato il braccio 2 identico (config confrontata col pin: IDENTICA) con
l'host dichiarato **`quiescent`** invece di `user-session-light`. Non c'era
nulla da spegnere — unico client GPU il compositor (`kwin_wayland`, 58 MiB),
nessun Chrome residuo dagli smoke — e vite non e' spegnibile perche' e' lui che
serve `/glmbench.html` al runner.

| metrica | rif. 2026-08-09 | it.40 (light) | it.41 (**quiescent**) |
|---|---|---|---|
| decode tok/s | 13,172 | 13,437 (+2,01%) | **13,125 (-0,36%)** |
| prefill tok/s | 31,265 | 31,813 (+1,75%) | **31,255 (-0,03%)** |
| TTFT ms | 14.744,9 | 14.490,8 (-1,72%) | **14.749,4 (+0,03%)** |

**La dispersione e' il vero risultato di questa iterazione**: stdev del decode
**0,087** contro 0,125 del riferimento e 0,265 di it.40 — un terzo del rumore
del braccio a sessione "light", e piu' basso del riferimento stesso. La GPU
partiva a 35 C invece di 46. E' la prova che la dichiarazione dell'host non e'
burocrazia: e' la larghezza della banda.

**Il numero che conta lo dicono i due run INSIEME.** Fra loro differiscono del
2,4% sul decode (13,437 contro 13,125), cioe' PIU' di quanto ciascuno disti dal
riferimento. Quindi it.40 non aveva misurato un miglioramento del +2,01% e it.41
non misura un peggioramento dello 0,36%: la variazione run-to-run su questo host
e' ~2-3%, il motore sta SOPRA il riferimento entro il rumore, e la banda +-5%
della riga 6 e' larga quel tanto che serve — non e' generosa, e' calibrata.

**La meccanica resta identica all'ultima cifra in TUTTI E TRE i run**:
`residencyRatio` 0,8210 · `pDirty` 0,9375 · `replaysPerToken` 1,1875 ·
`replayFracPerDirty` 0,6459 · `missesPerToken` 4,781. Dopo un goal di
unificazione del core, le decisioni di residenza sugli stessi token sono le
stesse; varia solo il tempo di parete, e varia dentro il rumore dell'host.

Exit code 4 di nuovo, e di nuovo NON e' una regressione (docket item 21): il
prefill e lo strutturale falliscono i gate del floor CPU esattamente come li
fallisce il riferimento. Qui il decode torna FAIL (13,125 < 13,43) mentre in
it.40 era PASS: e' lo stesso rumore del 2,4%, e conferma che quel gate e' una
soglia sul floor llama.cpp, non un giudizio di non-regressione.

`vramPeakMiB` 14.982 (rif. 15.160), host `memUsedMiB` 623->631, temp 35->47 C.
JSON: `bench-glm-4090-b12-optimistic-nonreg-fase-d-it41-quiescent.json`, che e'
il run di GATE del braccio 2 (it.40-c resta come replica indipendente).

**Braccio 2 del CHECKPOINT A, prima voce: PASSA.** Restano golden AL PIN,
cpuref e firma.

## it.42 (2026-08-11, fase 6) — conformance logits full: non "in banda", IDENTICA cifra per cifra

Ruling PI in sessione: opzione (a), logits full adesso. Lanciata riproducendo la
config del riferimento 2026-08-09 — `budgetGiB` **11** (non il default 12 del
runner: i flag li ho LETTI, dopo la lezione di it.40), prompts/maxGen/prefetch
null. `config`, `corpusHash`, `ggufSha256` e blocco `oracle` confrontati
programmaticamente: identici. 95,0 min contro 96,1.

**I DUE GATE DELLA RIGA 6 PASSANO, e non "entro soglia": con gli STESSI NUMERI.**

| | riferimento 2026-08-09 | oggi |
|---|---|---|
| `gateGolden` top1 | **1012/1024 = 98,828125%** | **1012/1024 = 98,828125%** |
| `gateCpuref` | **256/256** | **256/256** |
| `klMeanTop32` | 0,0029912943397463533 | **la stessa, all'ultima cifra** |
| `maxAbsDeltaLogit` | 58,567843466186524 | **la stessa, all'ultima cifra** |

E i 1.024 controlli non sono aggregati fortunati: **tutti e 8 i prompt hanno
top1, klMean e maxDl identici cifra per cifra** (p0 0,984375 · p1 0,9921875 ·
p2 0,9765625 · p3 0,96875 · p4 0,9921875 · p5 1 · p6 1 · p7 0,9921875).

**IL RISULTATO CHE CHIUDE IL BUCO DI COPERTURA** e' pero' un altro, ed e' il
motivo per cui questa run andava fatta adesso e non a fine goal. Il blocco
`residency` — la macchina che il goal ha RISCRITTO — e' identico voce per voce:

    hits           4.785.946 = 4.785.946
    misses           213.334 =   213.334
    evictions        211.117 =   211.117
    requests       4.999.280 = 4.999.280
    bytesUploaded  1.144.193.875.968 = 1.144.193.875.968   (al BYTE)
    retention      0,9577705189547295 = la stessa

**211.117 sfratti riprodotti uno per uno.** Il `glm-model-2layer` di ktest non
sfratta mai (a quel budget ci sta tutto) e il bench b12 sfratta ma non guarda
cosa esce: questa e' la sola prova che l'eviction unificata si comporta come
quella di prima SOTTO PRESSIONE VERA, su 5 milioni di richieste. Era
esattamente il rischio non coperto argomentato prima di lanciare.

Nota di forza probatoria, senza esagerare: `bytesUploaded` identico al byte su
5M richieste implica che la SEQUENZA di expert richiesti e' stata la stessa —
una selezione diversa anche di poco muoverebbe miss, sfratti e byte. E' evidenza
circostanziale FORTISSIMA che il routing sia invariato, ma **non e' il gate
`firma routing`**, che confronta l'istogramma per expert chiave per chiave ed e'
un'altra run (`glm-routing-conformance`, ~95 min).

**STATO DELLA RIGA 6, voce per voce.** Braccio 2 (GLM non-reg PIENA) = b12
optimistic in banda ✓ (it.41) · golden AL PIN ✓ · cpuref ✓ · **firma ✗**. La
`firma` della riga 6 e' la routing conformance (direction §: "routing = firma
item 14b", "firma routing invariata ai conteggi"), che il PI ha deciso in
sessione di NON fare ora scegliendo "solo conformance ai logits". Non e' una
dimenticanza: e' una scelta di scope dichiarata, e va a docket item 22 perche'
lascia una voce della riga 6 scoperta.

Restano poi i punti 3-6 del piano: riferimenti q35, gap nativo, ratchet golden
q35, `direction §7-bis`.

## it.43 (2026-08-11, fase 6) — riferimenti q35: 8 celle su 9, un tier irraggiungibile, e due difetti d'armatura

Punto 3 del piano. **Prima di eseguire, due letture del codice che hanno cambiato
l'esecuzione** (dopo it.40 i flag li leggo, non li deduco):

1. **`q35-conf-run.mjs` non ha `--bench`**: i riferimenti li fa
   `q35-bench-run.mjs`, che pero' esponeva solo `--arena-gib` e NON `--vram-gib`.
   Il done-when della riga 6 chiede i tier "con `--vram-gib` (i tier ora sono
   tetti VERI e non budget asseriti, dopo it.35)": **non era eseguibile**.
   Aggiunto il passthrough (la pagina leggeva gia' `?vram=`), piu' `vramTierGiB`
   nel report e il tag `tierN` nel nome del file. **Solo harness: `git diff
   --name-only` non tocca `src/` — il motore sotto misura e' invariato.**
2. **`vramCeilingBytes` vive solo nel ramo MoE** (q35gpumodel: budget derivato →
   ExpertCache); sui densi `vramPlan()` e' null. Quindi 6 celle su 9 sono
   degeneri per costruzione. Le ho girate lo stesso — e' un merge gate, la riga
   dice 9 celle — ma sapendo cosa aspettarsi.

**LA MATRICE (prompt idx 4, 64 decode, host `quiescent`, come i riferimenti q1):**

| | decode | prefill | TTFT s | miss | upl GiB |
|---|---|---|---|---|---|
| 4B tier 8/12/16 | **25,93 / 25,88 / 26,08** | 32,9 / 33,0 / 32,9 | 22,7 / 22,0 / 21,4 | — | — |
| 9B tier 8/12/16 | **16,00 / 15,95 / 16,06** | 18,7 / 19,2 / 18,7 | 36,3 / 34,6 / 35,0 | — | — |
| 35B tier 8 | 1,26 | 1,12 | 355,5 | 23.612 | 39,7 |
| 35B tier 12 | 2,69 | 2,10 | 194,3 | 9.908 | 16,6 |
| 35B tier 16 | **FALLITO** | | | | |

**La degenerazione dei densi e' MISURATA, non assunta**: spread 0,77% sul 4B e
0,69% sul 9B fra tier 8 e tier 16. Il tier non li tocca, e ora c'e' il numero
accanto alla lettura del codice.

**Contro i riferimenti q1** (stesso prompt, stessi 64 decode): 4B decode
22,93 → 25,93 (**+13,3%**) e prefill 25,97 → 32,9 (**+26,7%**); 9B decode
14,55 → 16,00 (**+10,0%**), prefill 15,35 → 18,7 (**+22%**).

**IL 35B NON MIGLIORA, E LA MIA SPIEGAZIONE ERA SBAGLIATA.** Avevo ipotesi
"fetch-bound: ai tier stretti si aspetta la PCIe, quindi i guadagni di kernel
non si vedono". **Falsa, e la sonda lo dice**: il fetch (read+pack+upload) vale
40,5 ms/token a tier 8 e 18,8 a tier 12, cioe' **~5% del token** in entrambi i
casi, contro p50 di 791,9 e 371,6 ms. Il collo NON e' l'I/O.

Il motivo vero e' che **questi riferimenti non esercitano il path del goal**:
`q35conf.worker` costruisce il modello con `select: cfg.optTrace ? "optimistic"
: "cpu"`, e `q35-bench-run.mjs` non passa MAI `?optimistic=1` (`grep -c
optimistic` = 0). I riferimenti misurano il path SYNC — un readback del router
per layer — che e' il regime pre-fase-3b. Il report lo dichiara pure, nel campo
`declared`: "sul MoE ancora un readback del router per layer". Sono numeri
onesti di una cosa diversa da quella che il goal ha ottimizzato (44 ms/token,
cioe' ~22,6 tok/s, sono del path ottimistico a caldo). → **docket item 25.**

**TIER 16 NON E' RAGGIUNGIBILE SU QUESTA SCHEDA, e il come lo dimostra conta.**
Primo tentativo (`--vram-gib 16` da solo): errore con diagnosi precisa del
motore — "la classe q4k ha 7 buffer d'arena ⇒ 10 storage binding, il device ne
concede 9". Ipotesi formulata dal codice: `arenaNeeds` (quanti buffer negoziare
col device) si calcola da `arenaGiB`, che vale **12 di default**, mentre il
budget si DERIVA da `vramGiB` — a tier 16 il derivato e' ~13,5 GiB chiesto a un
device negoziato per 12. Torna coi fatti: tier 12 (derivato 9,5) e tier 8 (5,5)
passano, solo il 16 sfonda. → **docket item 23.**

Secondo tentativo, l'osservazione che discrimina "difetto d'armatura" da
"scheda troppo piccola": `--vram-gib 16 --arena-gib 16`. Il guard dei binding
sparisce e arriva **`VK_ERROR_OUT_OF_DEVICE_MEMORY` vero**: 13,5 GiB di arena +
~2 di modello contro ~15,1 allocabili con Chrome vivo. **Tier 16 su una scheda
da 16.376 MiB non esiste**, ed e' un fatto sul LADDER dei tier, non un bug.

**E QUI LA LANDMINE DI it.14, DAL VIVO.** Quel secondo tentativo **e' uscito 0 e
ha scritto un report**: decode 4,80 tok/s, miss 7.657, upload 12,8 GiB — numeri
che continuano PERFETTAMENTE la tendenza di tier 8 e tier 12 (miss 23.612 →
9.908 → 7.657). Sono spazzatura: i buffer erano invalidi (`Invalid BindGroup`,
`Invalid CommandBuffer`) e il 4,80 e' la velocita' di NON fare il lavoro. Il
JSON e' in quarantena fuori da `results/`. `q35-bench-run.mjs` LOGGA i
`pageerror` ma non li guarda: uscirebbe 0 comunque. → **docket item 24**, ed e'
il piu' grave dei tre perche' il prodotto di questa fase SONO i JSON.

**Verificato che le altre 8 celle sono pulite**: un solo `pageerror` in tutto il
log della matrice, ed e' nella cella tier 16. Le referenze non sono contaminate.

## it.44 (2026-08-11, fase 6) — il gap nativo NON e' ri-misurabile "a parita'" senza dichiarare la page cache

Punto 4 del piano. `llama-bench` sta in `~/Projects/llama.cpp-oracle/build/bin`,
e la build e' **identica al riferimento** (`5f55650`), stessi 16 thread, stesso
GGUF. Il riferimento ha esattamente due test, e sono i due floor che gateano
tutta la funzione obiettivo: `pp512` = 56,58 e `tg64` = 13,43.

| test | rif. 2026-07-30 | oggi | delta | stdev |
|---|---|---|---|---|
| pp512 | 56,5757 | **76,4134** | **+35,06%** | 3,743 → 1,249 |
| tg64 | 13,4318 | **18,9606** | **+41,16%** | 0,097 → 0,633 |

**Un floor che si alza del 35-41% a build identica non e' un miglioramento di
llama.cpp: e' l'host.** Ipotesi formulata prima di toccare altro — la page
cache, che la landmine §4 nomina gia' ("page cache OPFS e VRAM baseline muovono
stallResidenza 4.3→29 ms/token"). La conformance di it.42 ha letto **1,1 TB**
da quel file un'ora fa, e `public/models/GLM-4.7-Flash-Q4_0.gguf` e' un SYMLINK
allo stesso inode che llama.cpp mmappa.

**Verificata con `mincore`, non supposta**: il GGUF e' residente in page cache al
**74,9% — 12,01 GiB**, cioe' esattamente tutto il `buff/cache` della macchina
(`free`: 12 GiB su 31 totali). llama.cpp mmappa i pesi: con 3/4 del modello gia'
in RAM non tocca il disco per quei tre quarti. La stdev di `pp512` che CROLLA da
3,743 a 1,249 e' la firma della stessa cosa — meno page fault, meno varianza.

**CONSEGUENZA SUL PUNTO 4, che e' il punto**: il done-when dice "gap nativo
RI-MISURATO a parita'", e questo run **non e' a parita'** — e' un floor a cache
calda. Peggio: non lo sarebbe nemmeno un run fatto adesso in altro ordine,
perche' lo stato della page cache non e' dichiarato **in nessuno dei due
artefatti**. I run del browser portano `hostState` con VRAM, clock e
temperatura; il JSON di `llama-bench` non porta niente, e la page cache non
compare in nessuno dei due. JSON rinominato
`llama-bench-...-WARMCACHE-fase-d-it44.json` perche' nessuno lo scambi per il
gap a parita'. → **docket item 26.**

**E c'e' una ricaduta piu' larga di questo punto.** I floor 13,43/56,58 sono il
denominatore della funzione obiettivo (direction §2) e i gate che
`glm-bench-run.mjs` valuta a ogni run. Sono stati misurati a stato di cache non
dichiarato. Se il floor vero a cache calda e' 18,96/76,41, il divario del motore
dal nativo e' PIU' GRANDE di quello pubblicato — il decode del browser (13,44)
non e' "appena sopra il floor", e' sotto di un terzo. Non lo affermo: il
confronto onesto pretende i due lati allo stesso stato di cache, e oggi non li
ho.

## it.46 (2026-08-12, fase 6) — firma routing IDENTICA: CHECKPOINT A CHIUSO

Ultima voce della riga 6 dopo il taglio. Config, corpusHash, gguf e oracle
confrontati col riferimento 2026-08-09: identici. 93,8 min contro 94,8.

| | riferimento | oggi |
|---|---|---|
| setMatch prefill | 1.047.485/1.203.084 = 87,06666% | **le stesse cifre** |
| setMatch decode | 208.441/235.520 = 88,50246% | **le stesse cifre** |
| residency hits/miss/evict | 5.614.247 / 140.169 / 137.750 | **identici** |
| bytesUploaded | 752.952.213.504 | **identico al byte** |
| router GPU set/order match | 1.438.591 / 1.438.572 | **identici** |

Non "entro soglia": numeratore E denominatore uguali su 31.274 posizioni.
`gpuErrors: []` — la sentinella di it.45 al primo impiego vero, pulita.

Exit 4 come previsto e come il riferimento: la soglia del gate e' 99% e il
riferimento stesso fa `pass: false`. La non-regressione si legge dal confronto.

**RIGA 6 COMPLETA**: b12 in banda (it.41) · golden al pin (it.42) · cpuref
(it.42) · firma routing (it.46). CHECKPOINT A chiuso. Prossimo: fase 7.

## it.47 (2026-08-12, fase 7) — la testa MTP: cos'e', dove sta, e perche' non serve un kernel nuovo

Iterazione 1 di 2-4 previste. Distanza dall'obiettivo: 13,4 tok/s (GLM) e 22,6
(35B) contro i 30 di direction §2.

**La riga 7 dice "`*-MTP-GGUF` pinnati" ma non lo erano**: `q35-manifest.json`
non aveva voci MTP e su disco non c'era niente. Prima di scaricare GB ho
verificato due cose gratis: (a) i tre GGUF locali hanno **zero** tensori e zero
metadata MTP — la testa non e' nei modelli che gia' abbiamo; (b) i repo
`unsloth/*-MTP-GGUF` esistono, stesso publisher e stesso quant dei pin base.

**Poi l'ho letta senza scaricarla**: 42 MB via Range dall'URL di HF bastano per
l'intestazione GGUF. Il 4B-MTP ha **441 tensori contro i 426 del base**, e
`qwen35.nextn_predict_layers: 1`.

**LA TESTA E' `blk.32`, cioe' 15 tensori** (il base ha blk.0..31):

| | |
|---|---|
| 11 tensori | IDENTICI per nome, forma e tipo a `blk.31` — attn_q/k/v/output, ffn_gate/up/down, le norme |
| `nextn.eh_proj` | **[5120, 2560] Q8_0** — 5120 = 2x2560: concatena embedding e hidden |
| `nextn.enorm` / `nextn.hnorm` | [2560] F32 — una norma per ciascuno dei due ingressi |
| `nextn.shared_head_norm` | [2560] F32 — prima della lm_head CONDIVISA col modello |
| totale | **120,6 M parametri** su un modello da 4B |

**CONSEGUENZA CHE CAMBIA LA FASE**: `blk.31` e' un layer FULL-attention (31%4=3
con `full_attention_interval` 4) e `blk.32` ne ha gli stessi 11 tensori, quindi
la testa e' un layer full normale — **non un deltanet**, e non dipende dalla sua
posizione. Tutte le primitive esistono gia': `rmsnorm`, `gemvQuant` Q8_0 (che e'
il kernel di tutto lo statico del 35B), attenzione e FFN. **La fase 7 e'
orchestrazione, non kernel nuovi** — la stessa conclusione della fase 4, e per
la stessa ragione.

**Pinnati in `q35-manifest.json`** 4B-MTP (2.669.209.920 B) e 35B-MTP
(21.388.319.008 B) con le sha LFS. Scaricato **solo il 4B**, sha256 verificata
contro il pin. Il 35B si scarica quando il reader passa sul piccolo: 21 GB per
un reader non ancora scritto sono spesi male.

## it.48 (2026-08-12, fase 7) — il reader della testa MTP, e un'assunzione presa dal test

Iterazione 2 di 2-4. Distanza dall'obiettivo invariata (nessuna misura di
prestazione in questa iterazione): 13,4 / 22,6 contro 30 tok/s.

`validateQwen35` ora riconosce la testa. Due cose, non una:
- `Q35Shape.mtpLayers` da `qwen35.nextn_predict_layers` (0 sui file base);
- il corpo del loop dei layer ESTRATTO in `expectBlock(b, full)`, perche' i
  chiamanti sono due — i layer del modello e la testa, che e' un blocco
  identico. Duplicare venti `expect` sarebbe stata una seconda verita' sulla
  stessa forma, ed e' esattamente cio' che il gate strutturale del goal vieta.

**L'ASSUNZIONE CHE IL TEST HA PRESO.** Avevo dedotto dai nomi dei tensori che il
modello avesse 32 layer con la testa a `blk.32`. Il file dice `block_count` =
**33**: la testa E' CONTATA nei blocchi. Il primo giro del test e' morto con
"block_count 33 non multiplo di interval 4" — che e' il guard giusto che urla
per la ragione giusta. Corretto alla radice: i layer del modello sono
`block_count - nextn_predict_layers`, e il controllo sull'interval vale su
quelli. Sui file base `mtpLayers` e' 0 e l'aritmetica torna identica a prima.

Se avessi scritto un fixture a mano invece di leggere il file vero, avrei
codificato la mia assunzione e il test l'avrebbe confermata.

**Gate** (`tests/engine-q35-mtp-head.test.ts`, sul GGUF pinnato): il file MTP si
valida, `mtpLayers` 1, `eh_proj` [2*dModel, dModel] — se il formato passasse da
concatenazione a somma quella riga fallirebbe invece di far girare il draft su
pesi mal interpretati — la testa e' full-attention benche' `32 % 4 !== 3`, e
441 tensori = 426 + 15. **E il file BASE resta invariato**: `mtpLayers` 0, 426
tensori, nessun `blk.32`. Suite **442|9** (era 440|9), tsc pulito.

Prossimo: il ciclo draft/verify. Tutte le primitive esistono (it.47), quindi e'
orchestrazione; il primo numero utile sara' l'accept-rate sul 4B.

## it.49 (2026-08-12, fase 7) — la testa MTP misurata in CPU: ordine deciso, accept-rate 31,8% (e sotto le attese)

Iterazione 3 di 2-4 (fuori stima di 1). Distanza dall'obiettivo: 13,4 / 22,6
contro 30 tok/s.

**Perche' un riferimento CPU prima della GPU.** Il gate della riga 7 ("token
accettati == token del greedy") e' INSENSIBILE alla qualita' della testa: la
verifica scarta i draft sbagliati, quindi una testa scritta male passa il gate e
si manifesta come accept-rate basso — cioe' come un risultato negativo SULL'MTP
invece che come un bug nostro. Questo numero e' la prova indipendente che al
gate manca, e costa zero GPU.

`Q35CpuRefModel.mtpDraftRef` in f64: h'_i = eh_proj([norm(emb(t_{i+1})) ;
norm(hidden_i)]) → blocco della testa (attn full forzata + ffn) →
shared_head_norm → lm_head condivisa. Due aggiunte minime al cpuref: un hook
sull'hidden finale e un `forceFull` su `attnLayerRef` (assente = comportamento
di prima, bit per bit).

**ORDINE DI `eh_proj` DECISO PER MISURA, non indovinato.** Non e' documentato nei
metadata; provati entrambi sullo STESSO hidden:

| concatenazione | accordo col modello | col corpus |
|---|---|---|
| **[emb ; hidden]** | **31,8%** (7/22) | 22,7% |
| [hidden ; emb] | **0,0%** | 0,0% |

Lo zero secco dell'ordine sbagliato e' la firma che cercavo: non e' "un po'
peggio", e' rumore puro. **L'embedding va per primo.**

**E HO MISURATO LA COSA SBAGLIATA AL PRIMO GIRO.** Il primo test confrontava la
predizione col token VERO del corpus (22,7%) e falliva la soglia. Ma
l'accept-rate non e' quello: in spec-dec il draft si accetta se coincide col
**greedy del modello**, non col testo — e il modello stesso indovina il corpus
circa a meta'. Bersaglio corretto: `argmax[i+1]` del forward, che avevo gia'.
31,8% contro 22,7%: confondere i due sottostima la testa di quanto sbaglia il
modello.

**IL NUMERO E' BASSO, E LO DICO.** Per l'architettura NextH/MTP a una testa la
letteratura riporta accettazioni molto piu' alte. 31,8% su 22 posizioni ha un
intervallo largo (~15-55%) ed e' misurato sui PRIMI 24 token di un prompt, cioe'
nel contesto piu' povero che esista. Ma potrebbe anche essere un residuo di
implementazione, e ho un candidato concreto: **le posizioni RoPE del blocco
della testa**. `attnLayerRef` usa l'indice dell'array come posizione, mentre h'_i
predice il token i+2 e dovrebbe stare alla posizione i+1. Un off-by-one del
genere degrada l'attenzione senza distruggerla — che e' esattamente la forma di
questo risultato.

Non lo correggo a naso: la prossima iterazione misura con l'offset e su un
campione piu' lungo, e i due numeri lo dicono.

**Se 31,8% regge**, un draft per token da' ~1,3x sul decode al netto della testa
(120,6 M parametri = ~3% del 4B): il 4B passerebbe da 25,9 a ~34 tok/s, cioe'
**sopra i 30 dell'obiettivo**, e il 35B da 22,6 a ~29,8. E' la prima proiezione
di questa sessione che parla all'obiettivo invece che al contratto.

Test gated su `Q35_MTP=1` (116 s: forward f64 + due passate sul vocabolario da
248.320), fuori dalla suite di default. Suite **442|10**, tsc pulito.

## it.50 (2026-08-12, fase 7) — lo sfasamento del rope: il test non discrimina, e questa e' l'informazione

Ipotesi di it.49: la testa MTP riceve posizioni sfasate di uno (h'_i predice il
token i+2 ed e' costruito su emb(t_{i+1}), quindi il rope dovrebbe stare a i+1).
Cablato `posOffset` su `attnLayerRef` e `mtpDraftRef` (default 0 = comportamento
di prima) e misurato.

**Accept-rate: 31,8% con offset 0, 31,8% con offset 1** — 7/22 in entrambi.

Prima di dichiarare l'ipotesi refutata ho controllato se le PREDIZIONI cambiano,
perche' uno stesso conteggio puo' nascondere token diversi: **0 su 23 diversi,
identici uno per uno**.

**E questo NON e' una refutazione: e' un'anomalia.** Sul canale 0 il rope ha
theta = pos, quindi spostare di una posizione vale ~1 radiante — non puo'
lasciare invariati 23 argmax su 248.320 classi. Le spiegazioni possibili sono
due e NON le ho distinte:
1. il parametro non arriva davvero al calcolo (il ramo full si prende
   `forceFull`, il rope legge `t + posOffset`: sembra cablato, ma "sembra" non
   e' una prova);
2. il contributo dell'attenzione in quel blocco e' trascurabile rispetto al
   residuo — che sarebbe di per se' un fatto sospetto sulla testa, non una
   rassicurazione.

Chi riprende parta da qui, e la prova che discrimina e' a costo quasi zero:
azzerare `attnOut` nel blocco della testa e rimisurare. Se il numero non cambia,
e' la (2) e l'attenzione della testa non sta facendo niente. Se cambia, e' la
(1) e il bug e' nel cablaggio del parametro.

Il 31,8% resta il numero da spiegare, e l'ordine di `eh_proj` resta deciso
(it.49): quello non e' in discussione.

## it.51 (2026-08-12, fase 7) — l'anomalia era il misuratore: accept-rate 50,0% (73,9% in traiettoria)

Iterazione 5 di 2-4 sulla fase 7 (fuori stima, sotto il 3x di ruling C9).

**Le due ipotesi di it.50 sono ENTRAMBE false, e la terza non l'avevo scritta.**

Ablazione dell'attenzione nel blocco della testa (`Q35MtpProbe.ablateAttn`),
finestra 24, stesso hidden: accept-rate identico (7/22) ma **17 predizioni su 23
diverse**, e ‖attn‖/‖h'‖ = 0,97. L'attenzione non e' inerte: pesa quanto il
residuo e riscrive il 74% delle predizioni. Quindi non e' la (2). E non e'
nemmeno la (1): il parametro arrivava.

**Il rope e' RELATIVO, e questo chiude l'anomalia senza codice.** Ruotando q_t e
k_p con lo stesso scarto, il punteggio dipende solo da (t−p): uno shift uniforme
di tutte le posizioni e' un no-op ESATTO, e nella testa MTP tutte le posizioni
scalano insieme per costruzione. I 23 argmax identici di it.50 erano il
risultato atteso. Il ragionamento di it.50 ("~1 radiante non puo' lasciare tutto
invariato") guardava l'angolo ASSOLUTO, che nel punteggio non entra mai.
`posOffset` RIMOSSO da `attnLayerRef`/`mtpDraftRef`: non era un knob, era un
invariante travestito da knob — e il suo zero e' costato un'iterazione.

**Conferma esterna (vLLM `qwen3_5_mtp.py`, letto oggi)**, che it.49 aveva deciso
per misura senza riferimento: concat `[inputs_embeds ; hidden_states]` —
embedding PRIMO, come misurato — due RMSNorm separate sui due ingressi,
`positions` passate **non modificate** al layer della testa, norma finale prima
della lm_head condivisa. La nostra `mtpDraftRef` corrisponde voce per voce.

**IL VERO BUG ERA LO STRUMENTO.** 22 posizioni: ±1 hit = ±4,5 punti, e infatti
la testa e la sua ablazione davano lo STESSO 31,8% predicendo cose diverse nel
74% dei casi. Sostituito il conteggio top-1 con **rango e log-prob del bersaglio
nei logit della testa** (varianza molto piu' bassa, costo zero: i logit li
calcolavamo gia' tutti per l'argmax) e finestra portata a 64 (`Q35_MTP_WINDOW`).

Finestra 64, 62 posizioni, prompt 0 del golden 4B:

| | accept vs modello | rango medio | mediana | top-8 | log-prob |
|---|---|---|---|---|---|
| **testa completa** | **50,0%** (31/62) | 12,6 | **1** | 44/62 | −2,90 |
| attenzione azzerata | 30,6% (19/62) | 2536,9 | 3 | 40/62 | −4,09 |
| ordine [hidden;emb] | 0,0% (0/62) | 37.526,3 | 5538 | 0/62 | −15,31 |

Il 31,8% di it.49 era **rumore di campionamento**, non un difetto: su 62
posizioni la testa fa 50,0%. E l'attenzione si vede finalmente: 50,0 contro
30,6, rango medio 12,6 contro 2537, log-prob −2,90 contro −4,09. Lavora, e nella
direzione giusta.

**IL NUMERO CHE CONTA E' 73,9%.** Lo spec-dec gira sulla traiettoria del modello
stesso, non su testo umano: il draft si accetta se coincide col greedy del
target, e durante il decode la sequenza E' l'output greedy. Sulle 23 posizioni
in cui il greedy del modello coincide col corpus — l'unico regime confrontabile
col decode vero — la testa fa **17/23 = 73,9%**, in linea con la letteratura
MTP. Il 50,0% mescola i due regimi e sottostima.

**Intervalli, perche' 62 non sono 6200**: 31/62 = 50% ha un 95% CI ~38-62%;
17/23 = 74% ha ~52-90%. Il segno e' solido (l'ablazione e' fuori intervallo),
la cifra no.

**Proiezione corretta, e it.49 sbagliava il COSTO.** it.49 contava la testa al 3%
(120,6 M parametri su 4B) e dimenticava che il draft rifa' anche la **lm_head**:
248.320x2560 = 636 M pesi, che a Q4_0 sono ~318 MB di traffico contro i ~2,6 GB
del token. Costo reale del draft ~15%, non 3%. Con la verifica del draft nella
STESSA passata (2 posizioni, regime memory-bound): (1+a)/1,15 → **1,30x a
a=0,50** e **1,51x a a=0,74**. Il 4B da 25,9 tok/s passa a **33,7-39,1**: sopra i
30 dell'obiettivo in entrambi gli scenari. Il 35B da 22,6 a 29,4-34,1.

Suite **442|10**, tsc pulito. Test gated `Q35_MTP=1` (310 s a finestra 64).

## it.52 (2026-08-12, fase 7) — la testa MTP gira su GPU e coincide col riferimento: L2rel 2,63e-7

Iterazione 6 di 2-4 sulla fase 7. Prima fetta di CODICE dopo tre iterazioni di
misura: il blocco della testa in WGSL, gated contro il cpuref f64.

**Nessun kernel nuovo, come previsto da it.47** — e il test copre esattamente
cio' che resta: il CABLAGGIO. Due RMSNorm su ingressi di scala diversa
(embedding grezzo vs hidden finale), la concatenazione in un buffer [2d], una
GEMV **q8_0** (`eh_proj` e' l'unico peso del blocco non q4_0: type 8 contro
type 2 di tutti gli altri, letto dal file, non dedotto), poi un layer full
identico a quelli del modello con la SUA cache KV, ffn denso,
`shared_head_norm`.

**Perche' un ktest e non aspettare lo spec-dec**: se enorm e hnorm finissero
scambiate, o le due meta' della concatenazione invertite, il modello pieno non
se ne accorgerebbe — la testa non e' sul suo percorso — e lo spec-dec lo
mostrerebbe come accept-rate basso, cioe' come un risultato negativo sull'MTP
invece che come un bug nostro. E' lo stesso argomento di it.49, applicato al
pezzo GPU.

**FONTE UNICA.** Estratto `mtpHeadRef(emb, hidden, embFirst, dbg)` da
`mtpDraftRef`: il blocco senza lm_head, su ingressi grezzi. Il fixture chiama
QUESTA funzione, non una copia — stessa regola che `attnLayerRef` ha per i
layer del modello. Verifica di identita' del refactor: `Q35_MTP_WINDOW=24`
riproduce it.51 cifra per cifra (7/22, ablate 7/22, predDiverse 17, attnRel
0,9701625134991295).

**La lm_head resta FUORI dal fixture**: e' condivisa col modello e gia' coperta
dal ktest full-model, e includerla avrebbe portato il fixture da 74,5 MB a
oltre 1,3 GB di `token_embd`.

Input sintetici seeded con SCALE DIVERSE per i due ingressi (0,05 contro 2,0):
l'embedding grezzo e l'hidden finale vivono su ordini di grandezza diversi, ed
e' il motivo per cui la testa ha due norme separate. Dare loro la stessa scala
avrebbe reso invisibile uno scambio fra enorm e hnorm.

**Risultato, primo colpo**: `q35-mtp-head-real-blk32` **PASS**, L2rel
**2,63e-7**, maxAbs 4,77e-6 su T=3 posizioni, pesi reali del 4B-MTP, adapter
nvidia lovelace. ktest **97 PASS / 0 FAIL** (era 96 + la riga nuova).

Gate: `npx tsc --noEmit` pulito, suite **442|10**, ktest 97/0, identita'
numerica del cpuref verificata contro it.51.

**Prossima fetta**: la testa dentro `q35gpumodel` — caricamento dei pesi nel
modello vero, cache KV della testa nel piano di decode, e il draft prodotto
dalla lm_head condivisa. Poi il ciclo draft+verify col gate di bit-invarianza.

## it.53 (2026-08-12, fase 7) — la testa dentro il modello vero: accept-rate GPU 31/62, LO STESSO conteggio della CPU

Iterazione 7 di 2-4 sulla fase 7. Seconda fetta di codice: la testa non e' piu'
un fixture, e' un pezzo di `q35gpumodel`.

**Cosa e' stato aggiunto.** `opts.mtp` costruisce il blocco `blk.<nLayer>` come
un SECONDO piano di dispatch — non un pezzo del token: legge `x` (il residuo
finale del modello, prima di `output_norm`) e l'embedding grezzo del token
appena predetto, e produce il draft con la lm_head CONDIVISA. Cache KV sua
(la testa attende la sequenza degli h', non quella del modello), argmax su GPU
con le pipeline gia' esistenti e un bind group suo, un solo submit.

**Un dettaglio di meccanica che valeva un commento nel codice**: `loadW` e
`gemv` scrivono in `steps`, il piano del token. Invece di duplicarli ho reso il
bersaglio degli step una variabile (`stepTarget`), spostata per il tempo della
costruzione della testa e rimessa subito. Il piano del modello non cambia di un
dispatch — `dispatchesPerToken` si misura prima.

**VINCOLO D'USO scritto nell'interfaccia**: `mtpDraft` va chiamata SUBITO dopo
lo `step` che ha prodotto il token, perche' legge `x` e il token successivo lo
sovrascrive. Non e' documentazione difensiva: e' l'unico modo in cui il
contratto puo' essere sbagliato senza che nulla fallisca.

**IL NUMERO, e non era scontato**: `q35-mtp-draft-4b` accept-rate GPU
**31/62 = 50,0%** sui primi 64 token del prompt 0 — **lo stesso conteggio
esatto** del riferimento CPU f64 di it.51 (31/62). Non "in banda": identico. Fra
i due percorsi cambiano precisione (f32 contro f64) e hidden (il full-model
diverge dall'oracolo su ~1% delle posizioni), quindi l'uguaglianza del conteggio
e' piu' di quanto il gate chiedesse (soglia a 40%).

**E IL COSTO, che serve alla fase 8**: **5,57 ms per draft**, contro 34,6 ms per
token (sync) misurati sullo stesso host dal micro-bench del ktest. Cioe' il
**16,1%** di un token — conferma la stima del 15% di it.51 e smentisce il 3% di
it.49. Ed e' un TETTO: quei 5,57 ms includono un submit a parte e la sua
attesa, che il ciclo vero puo' fondere.

Proiezione con questi due numeri misurati sullo stesso host:
(1+0,50)/(1+0,161) = **1,29x** → il 4B da 25,9 a **33,4 tok/s**, sopra i 30.

**Il primo golden non era confrontabile e il test lo diceva**: alla prima
esecuzione ho puntato `golden-smoke.json`, che ha 40 token in tutto, e l'accept
e' uscito 13/38 = 34,2% — un numero giusto su una finestra sbagliata. Il
confronto col riferimento CPU vuole gli STESSI token: `golden-full.json`, primi
64 del prompt 0 (identici nei due golden, e' lo stesso corpus).

Gate: tsc pulito, suite **442|10**, ktest **98 PASS / 0 FAIL** (adapter nvidia
lovelace), `q35-mtp-head-real-blk32` invariato a L2rel 2,63e-7.

**Prossima fetta**: il ciclo draft+verify — verificare il draft nella stessa
passata del token successivo (2 posizioni in un submit) e tenere il token solo
se coincide col greedy. E' li' che il 50% diventa tok/s, ed e' li' che vive il
gate di bit-invarianza della riga 7.

## it.54 (2026-08-12, fase 7) — draft+verify: 16/16 token identici al sequenziale, e il meccanismo e' PIU' LENTO

Iterazione 8 di 2-4 sulla fase 7 (oltre la stima, sotto il 3x di ruling C9).
Terza fetta di codice: il ciclo si chiude, e il gate secco della riga 7 passa.

**Il pezzo difficile non era la speculazione: era disfarla.** 24 layer su 32 del
4B sono DeltaNet e aggiornano `convSt`/`stateS` IN PLACE. Una riga speculativa
rifiutata non lascia un errore che si sovrascrive — lascia uno stato ricorrente
sbagliato e numeri PLAUSIBILI (la stessa lezione del replay ottimistico,
it.18). `specVerify` prende lo snapshot **dentro l'encoder**, fra la riga certa
e quella speculativa: nessun round-trip, e lo stato salvato e' esattamente
quello "dopo il token certo". Salva anche `x`, perche' la testa legge il residuo
finale e dopo la riga speculativa `x` e' quello della posizione sbagliata.

**La cache KV invece NON si ripara**, e vale la pena averlo scritto: `kvAppend`
scrive all'indice della posizione e l'attenzione legge solo fino a `pos`, quindi
la riga rifiutata viene riscritta dalla posizione vera prima che qualcuno la
legga. Ripararla sarebbe stato lavoro inutile.

**IL GATE SECCO DELLA RIGA 7 E' PASSATO**: `q35-mtp-specdec-invariance` genera
16 token con draft+verify e li confronta uno per uno con la generazione
sequenziale — **16/16 IDENTICI**, con **11 passate, 6 accettate (55%) e 5
rollback ESERCITATI**. I 5 rollback contano quanto i 16 token: se lo stato
ricorrente non fosse riparato, il test produrrebbe testo plausibile e diverso, e
questo e' l'unico gate che se ne accorge.

**E ADESSO IL NUMERO SCOMODO: 48,8 ms/token contro 34,5 sequenziale.** Lo
spec-dec, cosi' com'e', e' **1,41x PIU' LENTO**. Non e' la testa (5,51 ms, il
16%): e' che `specVerify` esegue le due righe IN SEQUENZA sullo stesso piano di
dispatch, quindi rilegge tutti i pesi due volte. In un decode memory-bound due
righe costano il doppio, e 1,5 token per due righe e' peggio di 1 token per
riga. **Il calcolo si poteva fare prima di scrivere il codice, e non l'ho
fatto** — l'ho scoperto misurando. Il piano che serve esiste gia': `PB`/
`prefillM`, kernel batch con `gid.y` = riga, pesi letti UNA volta (fase 4).
Manca il cablaggio per M=2 con l'argmax di ENTRAMBE le righe. Docket item 29,
proiezione ~28 ms/token = ~36 tok/s.

**Errore di misura corretto sul posto**: il primo giro riportava 102,4 contro
88,9 ms/token perche' il cronometro includeva 24 posizioni di prefill divise per
16 token generati (~50 ms a token di zavorra su ENTRAMBI i bracci). Spostato il
via dopo il prefill: 48,8 contro 34,5, e il secondo numero combacia col 34,6 del
micro-bench del ktest — che e' il controllo che la correzione era quella giusta.

**Il 35B resta scoperto**: `specVerify` e `decodeBatch` sono null sui MoE (la
selezione degli expert passa dalla CPU a ogni layer), tranne nel path
ottimistico dove il token e' un submit solo. Docket item 30: o si cabla li', o
il checkpoint B misura il solo 4B e lo dichiara.

Gate: tsc pulito, suite **442|10**, ktest **99 PASS / 0 FAIL**, riga 7 di
PHASES aggiornata voce per voce con le evidenze.

## it.55 (2026-08-12, fase 7) — verifica a 2 righe: 41,2 ms/token, ancora sopra il sequenziale

Iterazione 9 di 2-4 sulla fase 7. Quarta fetta: la verifica speculativa sul
piano a righe multiple, quello che legge ogni peso una volta sola.

**Il problema strutturale e la sua soluzione.** Nel piano a M righe il momento
"dopo la riga certa" NON e' un punto solo della sequenza di dispatch: ogni layer
aggiorna il proprio stato ricorrente prima di passare al successivo. Quindi lo
snapshot si prende **per layer**, subito dopo la riga 0 — e con un KERNEL di
copia, non con `copyBufferToBuffer`, perche' dentro un compute pass le copie di
encoder non entrano. `stridedCopyWgsl` fa gia' esattamente quel lavoro.

Lo snapshot vive solo se il modello e' costruito per lo spec-dec (`mtp`) con
M=2: il prefill a chunk vero gira con M piu' grandi e non paga niente. E `x`
non si salva piu': dopo la passata il residuo giusto e' una RIGA di `PB.x` —
riga 1 se il draft e' accettato, riga 0 se rifiutato — quindi `specCommit`
copia quella e basta.

**IL GATE SECCO REGGE SU ENTRAMBI I PATH**: 16/16 token identici al
sequenziale, per-riga e batch.

**E IL NUMERO: 41,2 ms/token contro 48,8 del per-riga e 34,6 del sequenziale.**
Il piano batch guadagna il 16% e resta **1,19x piu' lento**. Il conto non torna:
per passata la verifica batch costa 60,4 ms contro i 34,6 di UNA riga, cioe'
1,75x per due righe, mentre a pesi letti una volta dovrebbe stare su 1,1-1,2x.
Tre candidati (ricorrenza DeltaNet per riga anche nel batch; lm_head eseguita
due volte; gemv batch che non amortizza) e **nessuno discriminato**: la prossima
misura e' la sonda `--gpu-time` della fase 4-bis, non un'altra ottimizzazione a
naso. Docket item 31.

**E una discrepanza che non mi aspettavo**: i due path danno gli stessi 16
token ma con conteggi diversi (11 passate / 6 accettate contro 10 / 7), il che
significa che almeno un draft differisce — cioe' che i due piani producono
hidden diversi. La fase 4 aveva chiuso con "logits BIT-IDENTICI" fra batch e
sequenziale: o quella proprieta' non vale a M=2, o e' il mio cablaggio. Docket
item 32. L'uscita non e' sbagliata (il gate secco passa), ma due sensori che
dovrebbero coincidere non coincidono.

**Onesta' sulla traiettoria**: tre iterazioni di fila (53, 54, 55) hanno
prodotto codice corretto e verificato senza muovere l'obiettivo di un token al
secondo. Il meccanismo c'e', il gate c'e', la velocita' no — e la causa non e'
piu' un'ipotesi da sviluppare ma una misura da prendere.

Gate: tsc pulito, suite **442|10**, ktest **99 PASS / 0 FAIL**.

## it.56 (2026-08-12, fase 7) — dove vanno i 60 ms: il corpo a 2 righe costa 1,70x, non 1,1x

Iterazione 10 di 2-4 sulla fase 7. Misura, non ottimizzazione (docket item 31).

**Metodo**: `step(read=false)` esegue il CORPO e basta — la coda (norma finale +
lm_head + argmax + readback) e' tagliata dal `headCut` della fase 4 — e una
fence su `queue.onSubmittedWorkDone()` rende il tempo confrontabile col token
intero, che aspetta il readback per costruzione. 12 token per braccio, dopo il
prefill.

| | ms |
|---|---|
| token intero | 34,7 |
| — corpo | **27,1** |
| — coda (norma + lm_head + argmax + readback) | **7,6** |
| passata di verifica (2 righe + 2 code + draft) | 66,8 |
| — corpo a 2 righe, per differenza | **46,2** |
| — due code | 15,2 |
| — draft della testa | 5,5 |

**IL NUMERO: corpo2 / corpo = 1,70x.** A pesi letti una volta sola dovrebbe
stare intorno a 1,1x. Il piano batch, su questo modello, NON amortizza.

**L'ipotesi principale e' architetturale, non di cablaggio**: 24 layer su 32 del
4B sono DeltaNet, e una RICORRENZA non si batcha per costruzione — lo stato
attraversa i token, quindi le M righe vanno eseguite in ordine, una alla volta,
qualunque sia il piano. Lo speculative decoding e' nato sui transformer puri,
dove verificare K token costa una lettura dei pesi; su un ibrido ricorrente la
ricorrenza deve comunque fare K passi. Se e' cosi', il tetto non si sposta con
altro cablaggio.

**Non e' ancora discriminata da**: le mie copie di snapshot per layer (~50 MB a
passata, 48 dispatch) che stanno DENTRO `corpo2` e che non ho attribuito a
parte.

**L'aritmetica di cosa resta possibile**, coi numeri misurati: per battere i
34,4 ms/token sequenziali con 1,5 token a passata serve una passata <= 51,6 ms;
oggi e' 66,8. Batchando la coda sulle due righe (una lm_head invece di due) si
scende a ~59,2 — ancora sopra. Con l'accept-rate del regime vero (74%, 1,74
token a passata) la soglia sale a ~60 ms e i 59,2 la sfiorano.

**Detto senza girarci intorno: lo spec-dec su questo modello paga al massimo in
modo marginale, e solo se entrambe le condizioni tengono.** La prossima leva —
una sola coda invece di due — e' cheap e la misura decide; se anche dopo resta
sopra il sequenziale, la fase 7 si chiude come ESCLUSIONE COI NUMERI, che e' lo
stesso esito gia' applicato a prefetch e policy a livelli (it.37).

Gate: tsc pulito, ktest **100 PASS / 0 FAIL**, suite invariata.

## it.57 (2026-08-12, fase 7) — il kernel batch NON riusa i pesi: lo spec-dec e' escluso coi numeri su questo motore

Iterazione 11 di 2-4 sulla fase 7. Nessun codice nuovo: una LETTURA che chiude
la fase, piu' aritmetica su quantita' misurate.

**HO LETTO IL KERNEL, invece di continuare a costruirci sopra.** In
`gemvQuantWgsl({batch:true})` la riga e' `wid.z` e il corpo fa
`qs[gb * 4u + w]` dentro l'invocazione di OGNI riga: le due righe sono due
workgroup diversi che **rileggono gli stessi blocchi di pesi**. Il "batch" di
questo motore FONDE I DISPATCH, non riusa i dati.

**Questo corregge una mia affermazione di it.55 e del docket item 29**, dove
avevo scritto "kernel batch con gid.y = riga, pesi letti UNA volta". Non l'avevo
verificato nel codice: l'avevo dedotto dal fatto che il prefill a chunk andava
2,02x. Quel 2,02x viene dal riuso in **L2** fra righe concorrenti (con M=16 e'
efficace), non da un riuso esplicito — ed e' per questo che a M=2 rende solo il
15%, cioe' il 1,70x misurato in it.56 invece del 2,00x di due righe indipendenti.

**Il tetto, con i numeri di it.56 e nessuna assunzione nuova:**

- passata di verifica misurata 66,8 ms → 1,6 token a passata → 41,8 ms/token
- la sola leva rimasta e' una coda invece di due. La coda GPU vale meno dei 7,6
  ms del token intero (quelli includono readback e argmax su CPU che nello spec
  non ci sono): ~5 ms. Passata → ~61,8, cioe' **38,6 ms/token**
- il sequenziale e' **34,4**

**Anche nel suo caso migliore, lo spec-dec su questo motore resta sopra il
sequenziale.** Non serve costruire la coda batch per saperlo: servirebbe a
misurare un numero che sappiamo gia' essere dalla parte sbagliata.

**PERCHE' il tetto e' li', in una frase**: verificare un draft costa una riga
INTERA di modello, e su questo motore due righe costano 1,70-1,89 volte una
perche' nessun kernel riusa i pesi fra righe. Lo speculative decoding paga
quando K token si verificano al prezzo di uno; qui si verificano al prezzo di
1,8. Con 24 layer su 32 ricorrenti, per giunta, una parte di quel costo non e'
nemmeno attaccabile da un kernel migliore: la ricorrenza fa K passi comunque.

**Il premio che resta dietro un kernel diverso, dichiarato per chi verra' dopo**:
con un GEMM a piu' righe che riusa davvero i pesi (corpo2 ≈ corpo), la passata
starebbe a ~37,6 ms per 1,6 token = **23,5 ms/token, cioe' ~42 tok/s**. E' una
famiglia di kernel che non esiste in questo motore e non e' un lavoro da una
fetta: e' un progetto suo.

**DECISIONE PRESA, non escalata** (il PI l'aveva vista arrivare due digest fa e
non ha obiettato): la fase 7 si chiude con il MECCANISMO COSTRUITO E VERIFICATO
— reader, testa su GPU bit-comparata col riferimento, accept-rate 50%/74%, gate
secco di bit-invarianza su due implementazioni — e la fase 8 lo registra come
**ESCLUSIONE COI NUMERI**, che e' esattamente il ramo che il done-when della
riga 8 prevede ("speedup ... O esclusione motivata coi numeri se sotto soglia
utile"). Il codice resta: e' gated, testato, e diventa utile il giorno in cui i
kernel a piu' righe esistono.

Gate: nessun codice toccato, ktest e suite invariati da it.56 (100 PASS / 0
FAIL, 442|10).

## it.58 (2026-08-12, fase 8) — CHECKPOINT B: l'esclusione diventa un artefatto

Iterazione 1 di 1 sulla riga 8. Il ramo esercitato e' quello che il done-when
prevede: "esclusione motivata coi numeri se sotto soglia utile".

**Un numero che vive solo in un log non e' un artefatto.** `scripts/q35-specdec-run.mjs`
guida il ktest (che le misure le fa gia', it.53-56), estrae le tre righe MTP dal
`window.__report` con i loro `metrics`, e ci mette accanto lo stato dell'host
DICHIARATO — piu' la sentinella GPU e la quarantena `.INVALID` di it.45, perche'
questo JSON e' esattamente il genere di file che finisce in un glob e diventa un
numero pubblicato.

`results/engine/specdec-4090-2026-08-12T21-49-18-513Z.json`, hostState
`quiescent`, adapter nvidia lovelace, `gates.allPass` true:

| | ms/token |
|---|---|
| sequenziale | **35,31** |
| spec, riga per riga | 48,32 |
| spec, piano a 2 righe | **41,69** (1,18x piu' lento) |

corpo 27,08 · coda 7,94 · passata di verifica 66,70 = corpo2 45,82 (**1,69x**) +
2 code + draft 5,01. Accept-rate 50% sulla finestra del corpus (identico al
riferimento CPU f64) e **70% in generazione libera**, che e' il regime vero.

**La ri-misura conferma entro il rumore dichiarato**: 1,18x contro 1,19x di
it.55, corpo2/corpo 1,69 contro 1,70 — la banda di variazione fra due run
identiche su questo host e' ~2,4% (landmine §4).

Il JSON porta anche il VERDETTO in chiaro, non solo i numeri: un file di
misure senza la frase che le lega si rilegge fra un mese come "dati" invece che
come "decisione presa".

**RIGA 8 CHIUSA.** Resta la 9: checklist del contratto voce per voce, non-reg
GLM fresca, direction/ledger/HANDOFF, triage del docket, e il q1 item 14.

Gate: runner exit 0, tre righe MTP PASS, ktest `done`, JSON committato.
