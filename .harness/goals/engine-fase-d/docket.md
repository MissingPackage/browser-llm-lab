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
