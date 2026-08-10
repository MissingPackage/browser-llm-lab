# Docket — engine-fase-d

1. **PLAN-CHECK — PRE-AUTORIZZATO (2026-08-10, PI in chat: "vai" sul
   contratto presentato).** PHASES.md è su disco al tag di apertura: 9 fasi
   sequenziali, authority delta = none ovunque, nessuna docket-born. Blocco A
   = fasi 1-6 (merge gate autonomo), blocco B = 7-8, chiusura = 9. Se vuoi
   rivederla, il punto giusto è PRIMA della fase 2 (la fase 1 è
   parametrizzazione pura, reversibile e coperta dal ktest bit-exact).
   REGOLA DI MISURA nel piano: bench pieni SOLO alle fasi 6 e 8.

2. **REGISTRAZIONE (2026-08-10, it.1)** — i campi piatti di `SlabLayout`
   (`gateQs`, `gateScales`, `upQs`, `upScales`, `downQs`, `downScales`,
   `qsBytes`, `gateScalesBytes`, `downScalesBytes`, `downKind`) restano come
   COMPAT DERIVATA per non toccare ~100 call site GLM in un colpo solo. Non
   sono una seconda fonte di verità (derivano dai tre `SlabTensorLayout`).
   Da rimuovere quando i call site migrano alla vista generica — lavoro di
   igiene interno al goal, non un debito verso l'esterno.

3. **ECCEZIONE DICHIARATA E PERMANENTE (2026-08-10, it.1)** — il router
   del CPUREF (`q35cpurefmodel.ts`) resta un'implementazione INDIPENDENTE:
   è il riferimento del differential testing, se usasse `routerSelect` un
   bug del router sarebbe invisibile al confronto. Il gate strutturale la
   elenca come eccezione, non come debito.

4. **DA CHIUDERE PRIMA DEL DONE DI FASE 1 (2026-08-10, verifier it.2)** —
   STATO al 2026-08-10 it.5: **(a) (c) (d) CHIUSI** in it.3 (verificati);
   **(b) CHIUSO in it.6, MA NON come previsto** — il gate è ora fatto di invarianti su ATTI SINGOLI
   (allocazione GPU; nomi dei tensori expert; clamp del router), ognuno con
   allowlist motivata, senza esenzioni per import e senza congiunzioni da
   spezzare. MA il verifier ha bocciato anche quella versione con 5 evasioni
   eseguite, di cui una (un router Qwen legittimo) NON è catturabile da
   nessuna scansione del sorgente: la differenza fra duplicazione e seconda
   famiglia è SEMANTICA. ESITO di it.6: l'invariante è stato spostato nel
   SISTEMA DI TIPI — marchio di conio su `SlotRef` (solo residency.ts può
   CONIARLO: 11 sonde ostili del verifier tutte rifiutate da tsc). ATTENZIONE
   alla formulazione (il verifier ha bocciato quella precedente): il marchio
   ferma chi CONTRAFFÀ uno SlotRef, NON chi lo ignora — q35gpumodel oggi
   gestisce un'arena completa senza SlotRef e tsc è verde. Diventa portante
   con it.7. Bypass noti senza cast: inflow any e spread di uno SlotRef
   genuino. Più il test di tipo
   `tests/types/slotref-brand.ts` che va rosso se il marchio sparisce. La
   scansione del sorgente resta come RATCHET su impronte note, con la
   pretesa ridimensionata PER ISCRITTO nel file. Evasioni N3/N4/N5 chiuse
   allargando la scansione a tutto src/ (ancorata a __dirname, estensioni
   incluse) e correggendo il predicato di allocazione. **Docket item 4
   CHIUSO.** Resta la migrazione di q35gpumodel (it.7), che è ciò che
   elimina DAVVERO la duplicazione: le tre voci DEBITO NOTO dell'allowlist
   spariscono lì. Rilievi minori del verifier it.3 chiusi in it.4: flushSlotTable
   dimensionata sulla shadow e non sulla costante GLM; getter compat non
   enumerabili (spread/JSON di un layout K-quant non esplodono);
   `slotsOverride` con chiavi validate contro le classi della config (prima
   costruiva una cache a zero classi in silenzio).

   (a) motore di `ExpertCache` cfg-driven (stati di classe, `expertSlots`,
   `arenaNeeds`, `ensure`, `repinPass`, `stats`, `destroy`): oggi c'è un
   GUARD che rifiuta le config non onorate, che è onesto ma non è la parità;
   (b) gate strutturale da firme TESTUALI a invariante non aggirabile
   (modello: il nome-API di `gpudevice.test`, che non si può eludere);
   (c) campi compat di `SlabLayout`: sui K-quant fabbricano un offset
   `scales` finto (bytes 0) — o si rimuovono o si fa fallire chi li usa su
   un layout K-quant, prima che q35 passi da `slotBindRanges`;
   (d) `expertSlots` ripartisce sul parco GLM: renderlo cfg-driven, altrimenti
   q35 girerebbe solo via `slotsOverride` = un bypass.

## item 5 — la fase 2 ha centrato l'OBIETTIVO ma non la LETTERA del done-when (PI)

**Cosa dice PHASES riga 2**: "il repack K-quant esce dal path on-miss
(import-time, come GLM `expertSlab`)".

**Cosa ho fatto invece**: ho reso il repack quasi gratuito DOVE STA. Misure
(micro-bench CPU sui due slab reali del 35B, 20 ripetizioni):

| | prima | dopo | fattore |
|---|---|---|---|
| pack q4_K | 2,94 ms | 0,44 ms | 6,7x |
| pack q6_K | 2,49 ms | 0,58 ms | 4,3x |

Due cause, entrambe reali: (a) `repackKQuant` ricostruiva le parole a 32 bit
con un `|=` per BYTE — ma su little-endian quell'aritmetica scrive
esattamente il byte al suo posto, cioe' e' una COPIA travestita; (b)
`packExpertSlab` allocava un array temporaneo per tensore e poi lo ricopiava
nello slab, toccando ogni byte TRE volte (zero-fill, copia, copia).

**La domanda per il PI**: il residuo e' ~0,5 ms/miss. Sullo smoke del 35B
sono ~1,7 s su ~45 di prompt (~4%). Spostarlo all'import costerebbe
**~18 GB di slab su disco** (il GGUF e' 20 GB) piu' una passata di import su
tutto il file. Il mio parere: NON conviene, e la riga 2 di PHASES andrebbe
riscritta come "il costo di pack per miss scende di >=4x, misurato" — che e'
l'obiettivo per cui la fase esisteva. Ma riscrivere un done-when e' un
cambio di contratto, non una decisione di meccanismo: la lascio qui.

**Se il PI dice "no, fallo all'import"**: si fa, e la fase 2 si riapre — il
lavoro fatto resta valido comunque (il pack veloce serve anche all'import).

Registrato it.8 (2026-08-10).

### RULING PI 2026-08-10: ACCOLTO

"Sono d'accordo con te su item 5. Hai fatto la scelta giusta."

Il done-when della fase 2 e' RISCRITTO in PHASES.md nella forma ratificata:
"il costo di pack per miss scende di >=4x, misurato PRIMA e DOPO nello stesso
JSON". Gli slab all'import NON si fanno: ~4% di residuo non vale ~18 GB su
disco. **Item 5 CHIUSO.** La fase 2 e' chiusa anche per contratto.


## item 6 — la telemetria read/pack/upload esiste su q35 ma non su GLM (io, fase 5/6)

Rilievo del verifier di it.8: `timing: true` è ora permanente sul path q35
(costa 4 `performance.now()` per MISS) ed è ciò che ha reso DECIDIBILE la
fase 2 — senza la scomposizione non avrei saputo che il pack era il 22%.
Su GLM la telemetria è opt-in dal chiamante e i worker non la accendono.
Alla fase 5 (policy) e alla 6 (checkpoint) serve lo stesso strumento sui due
path, altrimenti si confrontano due cose misurate diversamente.
NON è una decisione da PI: è mio, e lo faccio quando serve. Registrato qui
per non perderlo. it.9 (2026-08-10).

## item 7 — il costo di I/O per miss NON è misurato (io, fase 5)

`readMs` misura la finestra dentro `ensure`, dove per q35 l'I/O non c'è per
costruzione (la lettura Range sta nell'`await` prima). Il campo ora lo dice
nel tipo (it.9), ma il NUMERO manca: e la fase 5 deve decidere se il
prefetch si giustifica, il che dipende esattamente da quanto costa una
lettura. Va strumentato l'`await` in `runLayer` prima di attaccare la fase 5.
it.9 (2026-08-10).


## item 8 — il MoE fa 41 submit/token: serve il resolve su GPU (PI: e' una fase nuova?)

Emerso misurando la fase 3 (it.10). `decodeBatch` funziona sui densi
(-15,0 ms/token) ed e' `null` sul MoE, perche' la selezione degli expert
legge i logits del router su CPU a OGNI layer: sul 35B sono **41 submit e 41
readback per token**.

E' esattamente il problema che GLM ha risolto in fase C passando da 47 a 2
sync/token, e la meccanica c'e' gia' in `residency.ts` (`arena: true`,
`slotTable: true` + kernel d'arena che ricavano l'indirizzo dello slot da
soli). q35 oggi non la usa: binda sotto-range calcolati dalla CPU.

**Perche' e' PI-gated**: non e' una riga della fase 3, ed e' plausibilmente
il pezzo piu' redditizio rimasto nel goal — sul 35B il decode e' dominato da
quei 41 round-trip, non dal calcolo. Le opzioni:
(a) fase 3-bis nuova, subito dopo la 3, prima di prefill e policy;
(b) dentro la fase 5 (policy MoE), che gia' tocca la residenza;
(c) piu' avanti, accettando che il 35B resti sul path lento fino alla fase 5.

**Il mio parere: (a)**. Il prefill batched (fase 4) e la policy (fase 5) sul
MoE si misurano male finche' ogni token paga 41 round-trip: rischiamo di
ottimizzare sopra un collo di bottiglia che poi sparisce, cioe' di rifare le
misure due volte. Registrato it.10 (2026-08-10).
