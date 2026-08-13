# DOCKET — engine-ttft

## item 1 — plan-check (PI)

`PHASES.md` scritto in it.0. Goal di PRODOTTO ⇒ gate `plan-check`: il PI approva
la decomposizione prima dell'iterazione 1.

**Da guardare in particolare**, perché il test di fattibilità (C7) ha cambiato
il piano rispetto al contratto approvato in chat:

- **(C7-1)** il bench prefilla una posizione alla volta
  (`q35conf.worker.ts:207`): i 32,91 tok/s non misurano un chunking lento, ne
  misurano l'assenza. `prefillChunk` è già in albero e già bit-esatto. La riga 0
  lo instrada e rimisura: potrebbe muovere la metrica senza scrivere kernel.
- **(C7-2)** il done-when del contratto dice «riuso ≥ 4x a M=8». M=8 è
  **degenere per l'obiettivo**: con riuso perfetto e alla banda migliore mai
  misurata qui (300 GB/s) sono 5,94 s di sola lettura pesi, sopra il budget di
  3,96. In PHASES la riga 2 chiede **≥ 8x a M ≥ 16**. È uno scostamento dal
  testo del contratto: **serve il tuo sì**, o il contratto si aggiorna.
- **(C7-3)** la clausola (e2a) è sul path di **conformità 0.5B**, non sul 4B:
  `rmsPairGemmSiluChunkFast` è importato da `gpuforward.ts`, che assembla
  Qwen2.5-0.5B. Non tocca la metrica di questo goal. La (e2b) invece sì.
- **Conflitto strutturale fra due done-when**: alzare `mMax` per il riuso
  PEGGIORA il workgroup storage (`mMax·3856` B: 30.848 a M=8, 61.696 a M=16),
  che (e2a) vuole sotto 16.384. La riga 1 deve trovare una forma il cui shared
  non scali con M, o la (e2a) si dichiara debito. Registrato, non deciso.

**RULING (PI, 2026-08-13): PIANO APPROVATO.** Lo scostamento M=8 → M ≥ 16 è
ratificato. Il conflitto mMax-vs-shared è poi risultato in buona parte un
artefatto: v. item 6, il tetto è negoziabile.

## item 2 — lo spec-dec MTP non ha un obiettivo a cui appartenere (io → PI)

HANDOFF assegnava a questo goal «la resurrezione della predizione doppia, che
sopra quel kernel tornerebbe conveniente». Non l'ho messa in PHASES, e la
ragione è che **la predizione speculativa accelera il DECODE, non il prefill**:
non muove `prefill.ms + decode.firstMs`, che è la metrica di questo goal.

Ma non ha nemmeno un altro posto dove stare: il goal sul decode è CHIUSO e
superato (47,93 contro una soglia di 30), quindi non c'è una funzione obiettivo
che la giustifichi. Il codice resta in albero e gated, com'è oggi.

Le opzioni, senza mia preferenza forte: (a) resta gated e senza goal finché una
funzione obiettivo non la reclama; (b) diventa un goal suo con una soglia di
decode più alta di 30; (c) si rimuove. La (c) la sconsiglio: è costata misure e
la landmine dei 22 campioni dice che il suo accept-rate non è mai stato
stabilito bene.

**RULING:** _

## item 3 — `--out` assoluto/relativo: nove runner, due convenzioni opposte (io, ergonomia)

Eredita l'item 5 del docket di `engine-kernel-decode`, che nominava un runner
solo. Il censimento di oggi dice che sono nove e che il difetto è simmetrico —
**nessuno usa `isAbsolute`**:

- `join(ROOT, out)` ⇒ mangiano un path ASSOLUTO (`/tmp/x.json` →
  `<root>/tmp/x.json`, poi ENOENT in scrittura a run finita): `glm-bench-run:129`,
  `glm-prefill-run:61`, `glm-conf-run:86`, `glm-route-run:96`,
  `glm-instanton-run:81`, `webgpu-limits:105`.
- `out` grezzo ⇒ risolvono i path RELATIVI contro la CWD invece che contro la
  root: `q35-bench-run:101`, `q35-conf-run:108`, `vram-ceiling:259`.

Costo già pagato: una run GPU da ~20 minuti persa a goal precedente. Il lavoro è
due righe per file o un helper condiviso in `scripts/lib/`. **Lavoro mio, non un
ruling** — lo faccio quando la riga 0 tocca comunque i runner, così la modifica
viaggia con una verifica che la esercita.

## item 4 — `hostState.declared` non è verificato da nessun runner (io, metodo)

Eredita l'item 7 di `engine-kernel-decode`, con la sua correzione: l'episodio che
lo generò NON era contesa di host (era `--prefill-batch 1` mancante, it.9). Il
difetto resta comunque.

`scripts/lib/hoststate.mjs:36-42` mette `declared` nel report così com'è
arrivato dall'operatore. Campiona `before`/`after` da `nvidia-smi`, e il
campione **contiene già** `utilizationPct` e `memUsedMiB` — cioè esattamente ciò
che falsificherebbe un `"quiescent"` mentito. Nessun runner confronta i due.

Il controllo costa un `if` sul campione `before` che già esiste. La domanda che
NON decido io: se una dichiarazione smentita debba **fallire la run** (come la
sentinella sugli errori GPU, che esce non-zero e scrive fuori dal percorso
nominale) o solo **annotare il report**. La prima è coerente con la norma «le
metriche misurate non peggiorano mai» e con la landmine sul confronto fra
macchine; la seconda non butta via venti minuti di GPU per una soglia tarata
male.

**RULING:** _

## item 5 — due call-site GLM nel ktest non sono coperti dal freeze (ereditato)

Eredita l'item 6 di `engine-kernel-decode`, invariato. `ktest.worker.ts:1514` e
`:1516` sono siti GLM protetti SOLO dalla convenzione: il freeze sha256 copre il
down `scaledAccum` di riga 1519, non loro. Oggi nessuno passa `vec4Rows2` lì,
quindi sono conformi — ma la garanzia è "nessuno lo fa", non "il test lo
impedisce".

Caveat da tenere se un domani qualcuno volesse adottarli: i loro binding sono
SOTTO-RANGE di uno slab (offset+size), quindi la size andrebbe verificata
multipla di 16 B prima di poterli bindare come `array<vec4<u32>>`.

Lavoro mio, non un ruling. Non fatto ancora perché tocca il ktest di GLM e
questo goal non ha ragione di entrarci prima della riga 3.

## item 6 — il tetto di workgroup storage È GIÀ negoziabile, e cambia il senso della riga 4 (PI → io, verificato)

Spunto del PI (2026-08-13): rendere il cap di WebGPU configurabile, così che il
motore scelga da solo la migliore coppia (M, workgroup storage) in base al
device — alto su GPU potenti e memoria unificata, basso su telefoni — e da lì
sappia dire quale modello quel browser può reggere. Vincolo posto dal PI: solo
se fattibile **senza modifiche che non tutti possono eseguire** (es. permessi
Android).

**VERIFICATO SULLA SPEC** (`/gpuweb/gpuweb`, `GPUDeviceDescriptor`): è già così,
ed è il meccanismo standard.

- `requestDevice({ requiredLimits: {...} })` accetta
  `record<DOMString, GPUSize64>`. Testuale dal materiale della spec:
  «Developers should receive minimum limits by default and request higher ones
  if needed» e «Users must explicitly enable higher limits to obtain them on
  their device. Limits on the device itself will match the requested limits».
- Quindi **16.384 B non è un tetto: è ciò che ti danno se non chiedi.**
  `adapter.limits.maxComputeWorkgroupStorageSize` riporta il massimo vero del
  device (49.152 su questa scheda, misurato in
  `results/engine/webgpu-limits-4090laptop-2026-08-02.json`).
- Chiedere più del massimo dell'adapter ⇒ `requestDevice` **rigetta**.
- È **puro JavaScript**: nessun permesso, nessun flag, nessuna installazione.
  Su Chrome Android funziona identico. Il vincolo del PI è soddisfatto.

**E metà della macchina è già scritta.** `gpulimits.ts` ha `limitsFor` (riga
267: `requiredLimits = min(adapter, requisito)`, con `UnmetLimitError` che
nomina il consumatore), `negotiateLimits` (282) e `grantedLimits` (287). Manca
solo la **direzione inversa**: oggi `engineNeeds(o)` prende una configurazione e
calcola cosa il device deve concedere. Serve `bestConfigFor(adapter)`: dati i
limiti concessi, scegliere M e ctxMax.

**Conseguenza sul piano — e non è piccola.** Il conflitto che avevo registrato a
item 1 («alzare mMax peggiora (e2a)») è in buona parte un artefatto di aver
trattato i 16.384 come un pavimento universale invece che come un minimo
negoziabile. Il senso della riga 4 cambia: non «stare sotto 16.384 sempre», ma
**«dichiarare, negoziare, e degradare M con grazia quando il device concede
meno»**. Che è una clausola più difficile da soddisfare in un senso (serve un
percorso di degradazione vero, cioè lo stesso kernel generabile a M diversi) e
molto più facile nell'altro (su un device che concede 49.152, M=8 col kernel
fuso ci sta già).

**IL TRANELLO, che va misurato e non assunto**: chiedere il massimo NON è
gratis. Più workgroup storage per workgroup = meno workgroup residenti per
multiprocessore = meno occupancy = meno latenza nascosta. «Massimizzare il cap»
può risultare più LENTO di «chiederne meno e tenere più workgroup in volo». La
scelta automatica non è «prendi il massimo»: è **trovare il ginocchio**, per
classe di device. È una curva, e va misurata.

**Cosa ho fatto e cosa no.** Ho esteso il done-when (d) della riga 1 perché era
già una misura pianificata («workgroup storage di ogni variante, misurato non
dedotto») e questo la allarga di poco: aggiunge la curva throughput-vs-M a
limite concesso variabile, che è il dato che serve per decidere. **NON ho
toccato la riga 4**: cambiarne il done-when è must-docket.

**RULING RICHIESTO**: la riga 4 passa da «sotto 16.384 sempre» a «dichiarare,
negoziare, degradare»? E il selettore di modello per device — che è un pezzo di
PRODOTTO, non di motore — è un goal suo o entra qui? La mia raccomandazione:
sì alla riformulazione della riga 4 (è più onesta e sblocca il conflitto),
selettore di modello a un goal suo (ha bisogno del tetto di VRAM e della
paginazione, non solo dei limiti di workgroup — e quello è il goal sul load).

---

## item 9 — il warm-up del banco è tarato sui kernel corti, e sui lunghi mente (misurato, it.2)

**Cosa ho visto.** La sonda del picco fp32 (dispatch da 15-55 ms) misurava p50
**5,86 TFLOP/s con IQR 16,6%**, sotto il bordo della banda pre-registrata. I 40
campioni mostrano una **rampa di boost**: i primi 5-8 girano a 30-39 ms, il
regime è 15-16. `WARMUP_SAMPLES = 3` è tarato sui kernel da decimi di ms del
decode, dove basta; su un dispatch da decine di ms dopo un idle, tre scarti non
portano la GPU a clock stabile. Rifatta con le due shape **interleavate fra
loro** e 4 passate: 8,92 p50 / 9,26 in regime, IQR utile.

**Perché va a docket e non l'ho "sistemato" ovunque.** La stessa distorsione può
esserci sulle celle lente della fase 0 del goal precedente
(`kernel-decode-fase0-*.json`): `gemv base K2560xN248320` a 5,85 ms e
`attn-decode base n=6333` a 9,98 ms sono nella fascia dove la rampa morde, e
sono **celle di baseline già pubblicate e già usate come riferimento**. Se sono
sottostimate, i rapporti di quel goal sono SOVRA-stimati. Correggerle vuol dire
ri-misurare un artefatto chiuso e toccare numeri che il ruling di
non-regressione protegge: non è un ritocco, è una decisione.

**RULING RICHIESTO**: si ri-misura la fase 0 del goal precedente col warm-up
corretto (e si accetta che qualche rapporto pubblicato scenda), oppure la
correzione vale solo da qui in avanti e si annota il limite sui vecchi JSON?
Raccomandazione: **solo da qui in avanti**, con una riga nel memo di fase 0 —
quei numeri hanno già deciso ciò che dovevano decidere, e il verso dell'errore
(baseline lenta ⇒ rapporto generoso) non ha cambiato nessuna scelta.

## item 10 — l'occupancy è il termine che il piano non ha mai scritto (it.2)

Entrambe le sonde della riga 1 sono state decise dall'**occupancy**, e in
entrambe la pre-registrazione aveva scritto «traffico di memoria»:

- il moltiplicatore multi-riga: `splitk` batte `regs` di **2,13x** con lo stesso
  corpo e lo stesso workgroup storage, solo 576 workgroup invece di 144;
- l'attenzione a chunk: la fusione GQA taglia il traffico KV di **4x** ed è
  **più lenta**, perché scende da 256 a 64 workgroup su 76 SM.

Il modello mentale del goal («il prefill è memory-bound sui pesi, il riuso è la
leva») era giusto sulla forma ATTUALE e sbagliato su tutte le forme candidate:
appena il riuso c'è, il collo si sposta e la banda misurata crolla a **108 GB/s**
su un device che ne fa 300+.

**Non ho toccato niente**: le righe 2 e 3 hanno done-when scritti in termini di
byte per token, e quelli restano validi e verificabili. Ma se il piano volesse
un secondo termine, quello è «workgroup in volo per dispatch», e oggi non è
misurato da nessuna parte nel motore.

**Segnalazione, non ruling richiesto**: l'unica cosa che chiedo è che la riga 3
NON adotti la fusione GQA "perché sul decode ha funzionato" — sul prefill la
misura dice il contrario, ed è nel memo.

## item 11 — una leva intera mai provata: `packed_4x8_integer_dot_product` (it.2)

La sonda delle feature la dà **presente e corretta** su questo stack
(`dot4I8Packed((1,2,3,4),(5,6,7,8)) = 70`), e nessun kernel del motore la usa. Il
dequant q4_0 di oggi fa `f32(nibble) - 8.0` e poi `dot()` in floating point: la
via intera farebbe il prodotto scalare sui nibble impacchettati e applicherebbe
la scala una volta sola per blocco. Non è nello scope di questo goal e non l'ho
misurata. Va nel serbatoio del prossimo, insieme al fatto che `shader-f16`
resta assente qui (quindi la via f16 non è un'alternativa su questo stack).

## item 12 — quattro scostamenti trovati dal grade e non registrati dall'esecuzione (io, verifica di it.2)

Il grade indipendente ha dato verdetto **refuted** (composito: P3 e P5 hanno
fatto scattare le loro failCondition pre-registrate, non per lettura infedele —
ogni numero coincide con l'artefatto alla cifra). Ho rieseguito i gate da sola e
confermano: ktest 100 PASS / 0 FAIL, vitest 545 passed | 10 skipped, tsc pulito.
Ho anche ricalcolato le tre cifre di titolo dall'artefatto: 2,6225/0,0609 =
43,1x, 12,2993/1,8207 = 6,76x, 0,2581 ms per blocco FFN → 5.776 + 2.888 = 8.665
ms. Reggono.

Ma quattro cose il grade le ha viste e l'esecuzione non le ha messe a docket.

**(a) Il banco è stato CLONATO, non esteso.** Il WP diceva «estenderlo, non
riscriverlo» e nominava `kdRunner.ts / kdGemv.ts / kdAttn.ts / kdbench.html`.
L'esecuzione ha creato una famiglia parallela `tt-*` (`ttRunner.ts`, `ttGemm.ts`,
`ttAttn.ts`, `ttPage.ts`, `ttbench.html`, `scripts/tt-microbench-run.mjs`,
+1600 righe), toccando `kdRunner.ts` solo di sponda. **Da oggi ci sono due
famiglie di banco da mantenere**, con lo stesso schema e lo stesso pattern di
pagina. Lavoro mio, non un ruling: o si unificano, o si dichiara perché due.

**(b) Il memo dichiara 34 celle, il JSON ne ha 32.** `skipped: []` è corretto in
entrambi. Errore di contabilità nel memo, non nell'artefatto. Da correggere nel
memo — è esattamente il tipo di cifra che qualcuno ricontrollerà fra sei mesi.

**(c) La cella `coldw` non è stata girata sulla VINCITRICE.** Era
pre-registrata come discriminante della CAUSA di P3 — se il vantaggio evapora a
pesi freddi, è occupancy e non traffico. È stata girata su `regs` (0,1400 contro
0,1294: il vantaggio regge) e sulla forma attuale, **non su `splitk`**, che è la
forma che ha poi vinto. Quindi la causa della vincitrice resta indiscriminata, e
l'artefatto non lo dice. Costo per chiuderla: una cella. Da fare all'inizio
della riga 2, prima di costruirci sopra.

**(d) Il disegno di P1 è stato rivisto DOPO aver visto un numero che lo
refutava.** Prima sonda: 5,86 TFLOP/s, sotto il bordo della banda. Rivista con
12 warm-up, 40 campioni e shape interleavate: 8,92. La revisione è dichiarata
apertamente nel memo, resta dentro i minimi del §DISEGNO, e la giustificazione
(rampa di boost) è visibile nei campioni pubblicati — e il verso dell'errore
spinge il numero verso il bordo BASSO, quindi P1 reggeva anche col disegno
vecchio (8,07 a 4096³ è già dentro la banda). Ma è l'unico punto del WP in cui
il protocollo si è mosso a risultato noto, e va contato come grado di libertà
speso. Nessuna azione: registrato perché il prossimo che legge questi numeri
sappia quanto pesarli.

## item 13 — DUE INCARNAZIONI DELLA STESSA SESSIONE hanno lavorato in parallelo (it.5, metodo)

**Cosa ho osservato, non dedotto.** A metà iterazione la GPU era al 93% mentre
credevo l'host quiescente. `nvidia-smi --query-compute-apps` e `ps` hanno
mostrato il PID 324183: `node scripts/tt-microbench-run.mjs`, lanciato con un
`grep -E "idot|quant|splitk|SKIP|skipped"` **che in quel turno non ho scritto
io**, dentro una shell col mio stesso `CODEX_COMPANION_SESSION_ID`.

Sintomi che avevo già visto e attribuito a me stessa:
- `src/microbench/ttGemm.ts` con **due** implementazioni di
  `gemmQ4MultiRowSplitKIdotWgsl`, la seconda accompagnata da un
  `quantXQ8Wgsl` che non avevo scritto — e commenti che citano «P8/P9
  nell'addendum it.5», cioè la MIA pre-registrazione di dieci minuti prima;
- la seconda copia **sparita da sola** fra due comandi consecutivi;
- i system-reminder «il file è stato modificato dall'utente o da un linter».

**Il danno reale**: una run GPU sprecata (la mia è morta subito perché il
profilo Chrome era occupato), il profilo `/tmp/blab-e2e-profile` lasciato in
mano a dieci processi Chrome orfani con un `SingletonLock` da rimuovere a mano,
e — la parte cara — **dieci minuti passati a fare forensics su un file che
cambiava sotto di me**, credendo di aver duplicato io del codice.

**Perché va a docket e non è aneddotica**: la norma di questo progetto dice «i
bench GPU sono seriali ed esclusivi» e la landmine dice «due runner playwright
sullo stesso profilo si bloccano a vicenda». Nessuna delle due protegge da
questo caso, perché entrambe assumono **un solo agente**. Non c'è un lock: né
sul profilo, né sulla GPU, né sull'albero.

**Raccomandazione, non ruling** (è lavoro mio se la vuoi): un lock file su
`/tmp/blab-e2e-profile.lock` preso da ktest e dai runner playwright, che
fallisca subito con «un altro bench sta girando (PID N)» invece di dare
`Opening in existing browser session` dentro una eccezione non gestita. Costa
dieci righe in `scripts/lib/` e le usa chiunque.

## item 14 — ktest che CRASHA riporta 0 PASS e 0 FAIL, e un gate a grep lo legge come verde (it.5)

Trovato eseguendo, non ragionando. Con il profilo occupato, ktest muore con
un'eccezione non gestita **prima di eseguire un solo test**. Il mio gate era

    node .harness/tools/engine-ktest.mjs 2>&1 | grep -cE "\tFAIL\t"   # -> 0

e **zero FAIL è esattamente ciò che si vede quando va tutto bene**. L'ho preso
per verde una volta prima di accorgermene dal conteggio dei PASS a zero.

È la stessa classe delle sentinelle che questo progetto si è già dato altrove
(la landmine «i JSON in `results/` possono mentire in silenzio»): uno strumento
che tace è peggio che non averlo. Due difetti sommati:
1. il gate non guardava l'**exit code** (qui 1) né il numero di PASS attesi;
2. `engine-ktest.mjs` ha `BASE_URL` con default **5173** mentre questa sessione
   serve su **5199**, e `E2E_PROFILE` con default il profilo condiviso —
   entrambi flag da leggere prima di spenderci sopra, terza recidiva della
   landmine più cara del progetto.

**Lavoro mio, non un ruling**: il gate di ktest va scritto come «exit code 0 E
PASS >= N atteso E FAIL == 0», mai come «FAIL == 0» da solo. Da fare quando si
tocca il gate per altro; nel frattempo il comando corretto è
`BASE_URL=http://localhost:5199 node .harness/tools/engine-ktest.mjs` e si
guarda l'exit code.
