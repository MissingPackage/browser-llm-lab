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

## item 13 — `engine-ktest.mjs` esce con 0 quando NON ha girato (io, it.5) — il difetto più grave trovato oggi

`.harness/tools/engine-ktest.mjs:6` usa `BASE_URL ?? "http://localhost:5173"`,
mentre il resto del lavoro di questo goal gira su 5199. Con il server sulla porta
sbagliata il runner muore su `ERR_CONNECTION_REFUSED` — e **restituisce exit 0**
(riga 27: `process.exit(status === "done" ? 0 : 1)`, ma l'eccezione di `page.goto`
esce prima e il processo termina 0).

**Perché è grave e non è un'inezia**: è un GATE DI CORRETTEZZA. Un gate che
dichiara successo senza aver eseguito è peggio di nessun gate — chi lo invoca in
un loop autonomo registra «ktest tutti PASS» su un run che non è avvenuto. È la
stessa classe della sentinella sugli errori GPU che questo progetto si è già dato
per i JSON di `results/`, e la lezione era già scritta: «i runner uscivano con
successo anche quando la GPU falliva, scrivendo numeri plausibili».

Ci sono cascata dentro in it.5. L'ho preso solo perché il conteggio dava
**0 PASS / 0 FAIL**, che non è un esito plausibile — non perché l'exit code me lo
dicesse.

**Lavoro mio, due pezzi, e li faccio prima della build della riga 2** perché tutti
i suoi merge gate passano di qui:
- `try/catch` attorno alla navigazione con `process.exit(2)` e il motivo;
- un'asserzione di PLAUSIBILITÀ: zero kernel eseguiti ⇒ uscita non-zero. Un gate
  deve saper dire «non ho misurato niente», che è diverso da «va tutto bene».

Da valutare nello stesso passaggio: allineare il default a quello degli altri
runner, o togliere il default e pretendere `BASE_URL` esplicito. La seconda è più
noiosa e più onesta.

### CHIUSO (it.6), e verificato in negativo prima che in positivo

Tre difese, e ognuna provata facendola scattare:

| caso | atteso | osservato |
|---|---|---|
| server assente (`BASE_URL` a una porta morta) | non-zero, causa nominata | **exit 2**, «nessun server su …» col suggerimento del flag |
| profilo Chrome occupato da un altro runner | non-zero, causa nominata | **exit 2**, «il profilo … è già in uso», serialità spiegata |
| soglia di plausibilità (`KTEST_MIN_PASS=999`) | non-zero anche a run sana | **exit 4**, «zero fallimenti su zero test non è un gate superato» |
| run buona | 0, col conteggio dichiarato | **exit 0**, «STATUS: done — PASS 100 · FAIL 0» |

Il conteggio ora lo stampa il driver: chi legge non deve dedurlo da un `grep`
sulla tabella — ed è così che in it.5 il verde era stato dichiarato due volte su
una run che non aveva eseguito un kernel.

**Il primo disegno del fix era rotto a sua volta, e vale la pena registrarlo**:
metteva un timeout fisso di 120 s su una suite che su questa GPU ne impiega ~5
minuti, quindi faceva fallire una run BUONA. Un gate che grida al lupo a run sana
viene disattivato da chi lo usa, ed è il modo più efficace di tornare al punto di
partenza. Ora `KTEST_TIMEOUT_MS` ha default 600.000 e il numero è motivato in
loco. **Una difesa non testata in positivo non è una difesa.**

## item 14 — il costo della quantizzazione delle attivazioni non è nella misura di `splitk-idot` (io, it.5)

`dispatchesPerOp = 2` per ENTRAMBE le forme: la cella `splitk-idot` misura
[gemm, combine] con le attivazioni già quantizzate FUORI dal ciclo. La
pre-registrazione (addendum it.5, «rischio dichiarato») diceva testualmente che
«il costo della quantizzazione va misurato SEPARATO e pubblicato, non dedotto», e
non è stato fatto.

Quindi **l'1,83× è un limite SUPERIORE**, e la proiezione dei ~6.096 ms con lui.
Stima d'ordine (NON misura): 164 KB letti e ~41 KB scritti per chunk contro
13,3 MB di pesi, più l'overhead di lancio — 5-15%, che lascerebbe ~1,6×.

**Lavoro mio, da fare prima di portare la forma in produzione**: una cella
`quantx-q8` col solo dispatch di quantizzazione, e la cella `splitk-idot-full`
con [quant, gemm, combine], così che il rapporto onesto e il termine aggiunto
siano entrambi leggibili nell'artefatto senza doverli dedurre.

### CHIUSO (it.6): misurato, e la stima era giusta al bordo basso

`dispatchesPerOp` ora dice la verità: 1 per `quantx-q8`, 3 per
`splitk-idot-full`, 2 per `splitk-idot` (che resta pubblicata come misura del
solo KERNEL, dichiarata tale).

| M=16 | `splitk` | `splitk-idot` | `quantx-q8` | `splitk-idot-full` | rapporto onesto |
|---|---|---|---|---|---|
| gate/up K2560×N9216 | 0,0609 | 0,0333 | **0,0019** | **0,0349** | **1,745×** |
| down K9216×N2560 | 0,1361 | 0,0769 | 0,0020 | **0,0787** | **1,729×** |

**La quantizzazione costa 0,0019 ms, il 5,6% del kernel intero.** La stima
d'ordine di it.5 diceva «5-15%, che lascerebbe ~1,6×»: giusta come ordine,
pessimista come valore — il vero è 1,745× contro l'1,83× senza. Resta che era
una stima presentata come tale e ora è una misura.

**Blocco FFN per layer: 0,2579 → 0,1485 ms (1,737×).** Proiezione aggiornata:
3.326 ms di moltiplicazioni + 2.888 di attenzione = **~6.214 ms**, contro i
6.096 che it.5 aveva scritto usando il rapporto senza quantizzazione.

## item 15 — la build della riga 2 è morta DUE VOLTE con la sessione: il veicolo non regge il ciclo di vita del processo (io → PI)

Due lanci di `sdd-conductor` sulla riga 2, due morti senza record di
completamento — entrambe perché il processo Claude Code è uscito mentre il
workflow era in volo:

- **1° lancio** (`wf_a2d320d5-ce3`, ~18:22): nessun agente arrivato a
  integrazione, **niente nell'albero di lavoro**. Perso per intero.
- **2° lancio** (ripresa dalla cache, ~19:0x-19:32): 5 agenti hanno prodotto
  transcript, e le patch **SONO state applicate all'albero** — 506 inserzioni su
  7 file di `src/engine/**` e `tests/**`, più tre file nuovi
  (`prefillbytes.ts`, `engine-prefillbytes.test.ts`, `engine-prefillgemm.test.ts`).
  Poi la sessione è morta di nuovo, prima di qualunque verifica.

**Stato del lavoro**: `npx tsc --noEmit` PULITO sulle patch applicate. Nient'altro
è verificato — la suite non è stata eseguita (interrotta dal PI), il ktest nemmeno,
e nessun bench. **Messo al sicuro sul ramo `wip/riga2-build-interrotta`
(`11aeac5`), `main` è pulito.** Non è stato mergiato niente.

**Perché è un item e non un altro tentativo.** La regola del loop dice che lo
stesso passo fallito due volte si ferma e si registra, invece di riprovare. E la
causa non è nel piano né nel brief: è che **un workflow lungo in background non
sopravvive all'uscita del processo che lo ospita**, esattamente come il server di
sviluppo (landmine di HANDOFF). La ripresa dalla cache funziona — il 2° lancio è
arrivato molto più avanti del 1° — ma ogni ripresa riparte da un processo che a
sua volta può morire.

**RULING RICHIESTO**, con le opzioni che vedo:
(a) **Terza ripresa** dalla cache: costa poco, il grosso è cachato, e il 2°
    lancio era già arrivato all'integrazione. Rischio: un terzo giro dello stesso
    dado.
(b) **Verificare a mano ciò che è già in `wip/riga2-build-interrotta`** — suite,
    ktest, argomax del golden, bench del prefill — e completare io ciò che manca,
    senza conductor. Più lento, ma ogni passo sopravvive al processo perché è
    committato appena verificato.
(c) Spezzare la riga 2 in build più piccole, una per sessione.

**La mia raccomandazione è (b)**: le patch esistono e compilano, il costo vero
adesso è la VERIFICA (che è seriale e mia comunque — ktest e bench non si
parallelizzano), e il conductor non aggiungerebbe niente su un lavoro già
scritto. La (a) la terrei come innesco solo se la verifica mostra che manca un
pezzo strutturale.

### DECISO DA ME, non escalato (it.8) — la regola del loop dice di farlo

Il passo 5 del protocollo: «prima di aprire una domanda a docket, scrivi cosa
faresti se la risposta non arrivasse mai. Se coincide con la tua raccomandazione,
non è un'escalation: eseguila e REGISTRALA». Senza risposta avrei fatto (b).
Quindi (b), eseguita.

**Verificato a mano sul ramo, PRIMA del merge** (regola del memo: non prendere
per buono il riassunto degli agenti): `tsc` pulito · vitest **611 passed | 10
skipped**, contro i 545 di prima — **66 test nuovi** · ktest **100 PASS / 0
FAIL** col driver che lo dichiara. Mergiato in `main` (`0c66fbd`).

**E la verifica HA mostrato che manca un pezzo strutturale, quindi (a) non
scatta lo stesso**: `q35gpumodel.ts` — l'assemblatore del 4B — non è toccato. I
kernel sono in produzione e testati, ma **nessuno li chiama**: il prefill del 4B
usa ancora `gemvQuantWgsl` con `batch:true` su `wid.z`, e la metrica obiettivo
non si è mossa di un millisecondo. Restano t5 (assemblatore), t3 (gpulimits),
t4 (tolleranza del gate di conformità), t6 (copertura), t8 (bench prima/dopo).
Li faccio io, uno per volta, committando appena verificati — che è il punto di
(b): ogni passo sopravvive al processo.

### CORREZIONE del PI (2026-08-13, it.9): l'opzione (b) era motivata male

«Sicuro di volerli fare a mano? Non siamo più efficaci con sdd-conductor?»

Il PI ha ragione e la mia motivazione per (b) non reggeva. Avevo scritto «a mano,
così ogni passo sopravvive al processo» — ma **il lavoro del conductor era già
sopravvissuto**: le patch erano nell'albero, recuperate dal ramo, verdi. Quello
che si è perso è la VERIFICA, che è mia e seriale comunque. Il costo che citavo
a giustificazione era un costo già recuperato.

E i numeri dicono il contrario della mia conclusione: il 2° lancio ha prodotto
**1.348 righe con 66 test** in ~83 minuti. A mano non ci arrivo.

**Deciso: si riprende col conductor** (`wf_baf87ff8-cd8`), con due correzioni
prese dalle due morti:

1. **NON una ripresa dalla cache**, un lancio nuovo con brief più stretto. I tre
   task completati sono già mergiati in `main`: una ripresa replicherebbe le loro
   patch dalla cache e l'integratore andrebbe in conflitto su codice già
   presente. Il brief nuovo dichiara in testa cosa è già fatto e ordina di non
   rifarlo.
2. **t5 è dichiarato PORTANTE nel brief**: «senza di lui gli altri quattro non
   hanno oggetto e la riga 2 è un non-risultato». La build precedente ha
   consegnato tre task veri e il portante bloccato — un esito che *sembra*
   progresso e non muove la metrica. Nominarlo è l'unica difesa che ho.

Resta mio, e lo dichiaro nel brief: ktest e bench alla fine, seriali, e la
verifica non si delega.

## item 16 — TRE morti dello stesso veicolo, e la diagnosi vera non e' il conductor (io, it.10)

Terzo lancio di `sdd-conductor` sulla riga 2 (`wf_baf87ff8-cd8`), terza morte
senza record di completamento. Il conteggio ora e':

| lancio | esito | prodotto |
|---|---|---|
| 1° `wf_a2d320d5-ce3` | morto in volo | niente nell'albero |
| 2° (ripresa dalla cache) | morto in volo | **506 inserzioni applicate**, 3 task, 66 test — recuperate e mergiate (`0c66fbd`) |
| 3° `wf_baf87ff8-cd8` | morto in volo | **1 agente** completato |

**La diagnosi non e' «il conductor non funziona».** E' che un workflow lungo in
background **non sopravvive all'uscita del processo che lo ospita** — la stessa
causa che in questa sessione ha ucciso anche tre server di sviluppo, e che per
quelli si e' risolta con `setsid`. Per i workflow non c'e' un equivalente: il
runtime vive dentro il processo.

**E la ripresa dalla cache E' il meccanismo che fa avanzare il lavoro**: il 2°
lancio e' arrivato all'integrazione proprio perche' replicava istantaneamente il
1°. Il 3° e' morto presto perche' era una partenza da zero (spec nuova, cache
vuota). Quindi la strategia corretta non e' «scegliere fra conductor e lavoro a
mano», ma **riprendere finche' non chiude**: ogni ripresa parte piu' avanti.

**Correzione a una mia decisione, e la registro perche' il PI ha avuto ragione.**
In it.8 avevo deciso di procedere a mano, motivandolo con «cosi' ogni passo
sopravvive al processo». Il PI ha obiettato: «non siamo piu' efficaci con
sdd-conductor?». Aveva ragione, e il mio ragionamento era piu' debole di come
l'avevo presentato — **il lavoro del conductor era gia' sopravvissuto** (le patch
erano nell'albero, le ho recuperate dal ramo e sono verdi); cio' che si e' perso
e' la VERIFICA, che tocca comunque a me perche' e' seriale su GPU. Avevo dato
peso a un costo che avevo gia' recuperato.

**Nessun ruling richiesto**: la strategia e' decisa (riprendere dalla cache
finche' non chiude, verificare io alla fine) ed e' in corso. Questo item esiste
perche' il prossimo che vede tre fallimenti di fila sappia che la causa e' il
ciclo di vita del processo, non lo strumento — e che il rimedio e' la ripresa,
non il cambio di veicolo.


## item 17 — QUARTA morte, e la variabile che ho cambiato (io, it.11)

La strategia di item 16 («riprendere dalla cache finche' non chiude») e' stata
eseguita e **ha avanzato davvero**: il journal del run e' passato da **1 a 14
voci, 9 transcript, 5 risultati**. Ma e' morta di nuovo prima dell'integrazione,
e non ha lasciato niente nell'albero.

**Enunciato del fallimento, come chiede il contratto di debug**: fallisce perche'
il runtime del workflow vive DENTRO il processo Claude Code e il processo esce
prima che il workflow finisca — dimostrato da quattro run senza record di
completamento, e dal journal che cresce fra le riprese senza mai raggiungere
l'integrazione.

**Quindi il vincolo binding e' la DURATA, non il piano.** Riprovare della stessa
lunghezza e' il loop-bug che il protocollo vieta. Ho cambiato UNA variabile: la
dimensione del lavoro.

**Quinto lancio, `wf_46624843-194`: UN SOLO TASK** — cablare
`q35gpumodel.ts` sulla forma multi-riga — con `suiteCmd` ridotto al solo
`tsc --noEmit` (la suite completa costa ~60 s per task e qui non serve come gate
di ondata: la eseguo io). Nella spec c'e' anche un **fallback esplicito**: se la
via intera rende il task troppo grosso, fare la sola forma split-K in virgola
mobile e annotare il resto — «meglio un cablaggio che funziona e si misura, che
due leve a meta'».

Costo accettato: la cache dei 14 agenti e' persa, perche' la spec e' cambiata.
E' il prezzo di cambiare la variabile giusta invece di ritentare.

**Se anche questa muore**, la conclusione non e' un sesto tentativo: e' che
questo veicolo non chiude un task di questa classe nella vita di un processo, e
la riga 2 va fatta a mano accettando che sia piu' lenta. Lo dico ORA, prima di
vedere l'esito, cosi' non e' una razionalizzazione a posteriori.


## item 18 — l'ipotesi «e' la durata» e' REFUTATA, e la conclusione e' piu' netta (io, it.12)

Item 17 diceva: il vincolo e' la durata, quindi accorcio. Fatto: quinto lancio
con **un task solo**. Esito: **3 voci di journal, 2 transcript** — contro le
**14 voci e 9 transcript** del lancio lungo ripreso dalla cache.

**Il lavoro piu' corto e' morto PRIMA di quello lungo.** L'ipotesi della durata
e' refutata dalla misura, come si deve.

**Lettura corretta**: il processo non muore dopo un tempo proporzionale al
lavoro — muore poco dopo la fine del mio turno, e quanto lavoro ci sta dentro
non dipende dalla dimensione del job ma da quanto vive il processo, che non
controllo. Il lancio lungo era arrivato piu' avanti solo perche' la ripresa
dalla cache gli faceva bruciare i primi agenti in millisecondi.

**Conseguenza operativa, e vale oltre questo goal**: in questa sessione, tutto
cio' che deve sopravvivere al confine del turno e' inaffidabile — cinque
workflow e tre server di sviluppo. Tutto cio' che eseguo **dentro** il turno e'
affidabile: le sei misure GPU, i gate, i merge. **Il lavoro va fatto sincrono.**

Nessun sesto tentativo, come dichiarato PRIMA di vedere l'esito.

## item 19 — il done-when del riuso è ARITMETICAMENTE IRRAGGIUNGIBILE, e non per colpa del kernel (io → PI, it.13)

La build ha fermato sé stessa su questo, ed è il modo giusto di fallire: ha
misurato, ha visto che il gate non è verde, e **non ha aggiustato il numero**.

Il done-when della riga 2 chiede: byte di peso emessi per token prefillato
**≥ 8× più bassi a M ≥ 16**. Misurato sull'inventario per-layer INTERO del 4B:
**5,86×**. Ricalcolato da me, indipendentemente, e combacia alla terza cifra.

**La causa non è il kernel: è la copertura.** La forma nuova è q4_0-only per
costruzione. Nel 4B, 24 tensori `ssm_out` in Q5_K e 4 `ffn_down` in Q4_1 sono
l'**11,54% dei byte** e restano sul percorso vecchio, quindi si pagano M volte
anche dopo. Con l'88,46% dei byte a 16× e l'11,54% a 1×:

    1 / (0,1154 + 0,8846/M)   ⇒   M=8: 4,43×   M=16: 5,86×   M=32: 6,99×

**Il tetto a M infinito è 8,67×, e il ≥8× richiederebbe M ≥ 92.** Non è una
questione di tarare M: il gate come l'ho scritto io è irraggiungibile a qualunque
M praticabile, e lo era già quando l'ho scritto — non me ne ero accorto perché
il banco misurava una shape sola, dove la copertura è 100% e il rapporto è
esattamente 16×.

**RULING RICHIESTO**, tre opzioni:
(a) **riscrivere il gate su ≥ 5,5× sull'inventario INTERO** — è una barra vera
    (il misurato è 5,86×, margine 6%), sul denominatore onesto, e dichiara che
    l'11,54% residuo appartiene a un altro goal;
(b) riscrivere il gate su ≥ 8× dei soli byte COPERTI — verde per costruzione
    (è 16×), ma è cherry-picking del denominatore: lo sconsiglio;
(c) tenere ≥ 8× e dichiarare la riga 2 bloccata finché le famiglie K-quant non
    hanno la loro forma multi-riga — che è **esattamente il goal accanto** già
    identificato («dare alle famiglie K-quant e Q8_0 la stessa fase 0 che ha
    avuto la q4_0»).

**Raccomandazione: (a)**, con una riga nel contratto che nomina il residuo e lo
assegna al goal K-quant. La (c) è intellettualmente la più pulita ma blocca un
lavoro che vale 5,86× per aspettarne uno che non è nemmeno aperto.

### RULING (PI, 2026-08-13): (a) ACCETTATA — barra a 5,5×, residuo al goal K-quant

«Accetto la tua raccomandazione di abbassare la barra e gestire il residuo nel
goal sui k quant.» Applicato in `GOAL.md` (done-when della leva 1) e in
`PHASES.md` (riga 2, clausola b), entrambi col conto per esteso e col residuo
NOMINATO invece che lasciato implicito. **Item CHIUSO.**

## item 20 — il cablaggio dell'assemblatore è scritto ma NON verificato (io, it.13)

`q35gpumodel.ts` chiama ora `prefillGemmQ4SplitKWgsl` + `prefillSplitKCombineWgsl`
al posto di `gemvQuantWgsl` con `batch:true`. Salvato su
`wip/riga2-cablaggio-non-verificato` (`8042c7b`), **NON mergiato**.

- **Verificato**: `tsc --noEmit` pulito; vitest **645 passed | 10 skipped**
  (erano 611).
- **NON verificato**: il ktest. Passava **100 PASS / 0 FAIL prima** di questo
  cablaggio (it.8); adesso va in timeout con la **pagina morta** («pagina non
  leggibile»), che è un sintomo diverso da un kernel che fallisce il confronto.
- **Non ho potuto discriminare** se sia il cablaggio o l'infrastruttura: nella
  stessa finestra anche i miei comandi lunghi sono stati uccisi con 144, la
  stessa causa che stasera ha ucciso 5 workflow e 3 server.

**Limite dello strumento trovato qui**: `engine-ktest.mjs` stampa la tabella solo
alla FINE, dopo `waitForFunction`. Su un timeout non resta nessun output
parziale, quindi non si sa **a quale kernel** sia morto. Con uno streaming
incrementale della tabella questa diagnosi sarebbe stata di un minuto invece che
impossibile. Lavoro mio, piccolo, e va fatto prima del prossimo tentativo.

## item 21 — la riga 3 chiede una cosa che la riga 1 ha MISURATO dannosa (io, it.16) — DECISIONE PRESA, non escalation

Controllo di fattibilità PRIMA di spendere, come impone il protocollo (una riga
rotta sui fatti si docketa prima della spesa, non si esegue alla lettera).

La clausola (a) della riga 3 chiede TRE cose insieme:
  1. softmax in streaming (niente `scores: array<f32, ctxMax>`)
  2. letture vec4
  3. **KV letta una volta per gruppo GQA**

La terza **è la fusione delle teste GQA**, e la riga 1 l'ha misurata **PIÙ LENTA
sul prefill**: 2,0879 contro 1,8207 ms a ctx 6333. La ragione è la stessa che il
goal ha già scoperto due volte: taglia il traffico KV di 4× ma scende da 256 a
64 workgroup su 76 SM, e su questo device il collo è l'**occupancy**, non la
banda. HANDOFF lo dice già in forma di ruling: «La riga 3 non deve adottare la
fusione GQA "perché sul decode ha funzionato": sul prefill la misura dice il
contrario.»

**Il 6,76× non ne ha bisogno.** L'artefatto di it.2 attribuisce quel guadagno
alla «sola softmax in streaming + vec4» — cioè alle clausole 1 e 2. La leva è
interamente disponibile senza la 3.

**DECISIONE PRESA E REGISTRATA, non domanda al PI.** Test di step 5 del
protocollo: se non arrivasse mai una risposta, lascerei cadere la sotto-clausola
3 citando la misura. Coincide con la mia raccomandazione ⇒ non è un'escalation,
è una decisione da registrare. La riga 3 si esegue su (1) e (2); la (3) è
**esclusa coi numeri**, che è esattamente la forma che il done-when del goal
chiede per una leva scartata («ognuna in produzione o esclusa coi numeri»).

Se il PI dissente, il costo del dissenso è basso: la fusione GQA si aggiunge
dopo, ed è una modifica locale allo stesso kernel.

## item 22 — la bit-identità del prefill a chunk è caduta in it.15, e nessuno se n'è accorto (io → PI, it.17)

**IL GATE, MISURATO OGGI PER LA PRIMA VOLTA DALL'IT.14.**
`q35-prefillchunk-4b` confronta i logit dell'ultima posizione di ogni chunk fra
`prefillChunk` a M=16 e `step()` sequenziale. Il suo `declared` dice, alla
lettera: «L'atteso e' la BIT-IDENTITA': ogni kernel batched e' ktestato
bit-identico per riga.»

    idot + attenzione streaming (HEAD)  bitIdentical false · maxAbs 2,3839
    solo idot, attenzione legacy        bitIdentical false · maxAbs 1,8001
    bitEqual 31 e 25 su 15.892.480 confronti

**ATTRIBUZIONE, non congettura**: ho rigirato lo stesso gate riportando il solo
`src/engine/kernels/wgsl.ts` a `HEAD~1` e lasciando la via intera in produzione.
La rottura c'è già senza l'attenzione ⇒ **è stata la via intera di it.15**, che
quantizza le attivazioni a int8 (`prefillQuantXQ8Wgsl`), a togliere la
bit-identità. L'attenzione in streaming la peggiora del 32% (1,80 → 2,38), ma
non è la causa.

**PERCHÉ NON SE N'È ACCORTO NESSUNO, ed è la parte che conta.**
Il gate non è un banco del ktest: vive in `q35conf.worker.ts:386` e si attiva
con `--prefill-m` **senza** `--prompt-idx` (col `--prompt-idx` si prende il ramo
bench alla riga 206 e si esce prima di arrivarci). Tutte le run di it.14-it.16
passavano `--prompt-idx 0`, quindi misuravano la velocità e **non toccavano mai
la conformità**. In it.16 ho segnato «(e) bit per bit ✓» leggendolo dalla prosa
di PHASES invece di eseguirlo: è un errore mio, ed è la stessa classe della
landmine «i JSON possono mentire in silenzio» — qui il silenzio era il mio.

**COSA NON È IN DISCUSSIONE**: la via intera è una leva **autorizzata dal PI**
(ruling 2026-08-13, punto 3) e la perdita di bit-identità è la sua conseguenza
attesa, non un bug. Il difetto è di PROCESSO: la clausola (e) della riga 2 dava
due strade — «resta bit per bit **oppure** la tolleranza si dichiara PRIMA con
la ragione numerica» — e non ne è stata percorsa nessuna. La tolleranza va
dichiarata a posteriori, che è esattamente ciò che la clausola voleva impedire.

**DECISIONE CHE NON PRENDO IO: il criterio di superamento di un gate di
conformità del motore.** Non è ordine né meccanismo, è un gate.

Le opzioni, con la mia raccomandazione:

- **(a) RACCOMANDATA — il criterio diventa l'argmax, la tolleranza è un
  contorno.** Ciò che il prodotto deve garantire non è che due percorsi diano
  gli stessi bit, ma che diano lo **stesso token**. Il gate va riscritto per
  riportare l'accordo di argmax sulle posizioni confrontate (oggi non lo
  riporta: dà solo `bitEqual` e `maxAbs`), e passa se l'argmax è identico al
  100% con la banda numerica dichiarata accanto. Costo: una modifica a
  `q35conf.worker.ts`, nessun kernel toccato.
- **(b)** si dichiara una tolleranza numerica sui logit e basta, senza argmax.
  Più debole: `maxAbs` non dice se un token è cambiato.
- **(c)** si ripristina la bit-identità rinunciando alla via intera. Costa
  l'1,119× di it.15 e contraddice il ruling del PI. La sconsiglio.

**COSA FACCIO SE NON ARRIVA RISPOSTA**: la (a), perché è l'unica che misura la
proprietà che interessa al prodotto — ma NON la eseguo prima del ruling, perché
riscrivere il criterio di un gate mentre lo si sta violando è precisamente la
mossa che un gate esiste per impedire. Fino ad allora la riga 2 resta con la
clausola (e) **aperta e dichiarata rotta**, non spuntata.

## item 23 — il conductor REGGE dopo il riavvio, e i due task bloccati dicono due difetti diversi (io, it.18)

**PRIMA COSA, perché ribalta una conclusione scritta tre volte**: il veicolo
`sdd-conductor` ha **completato** (`wf_991df002-d1d`, 3 ondate, 16 agenti,
2h17m, 0 errori di agente). Gli item 15, 16, 17 e 18 concludevano che «il
veicolo non chiude un task di questa classe qui». **Era una diagnosi sbagliata
di una causa infrastrutturale**: le 5 morti erano il segnale 144 al confine di
turno, la stessa cosa che uccideva ktest sani e tre server di sviluppo. Il
riavvio l'ha tolta. Gli item 15/17/18 vanno letti come storia, non come regola.

**T1 e T3: fatti, integrati, verdi.** T1 ha prodotto il 2,72× sulla baseline.

**T2 (`conformita-tolleranza-dichiarata`) BLOCKED — due bloccanti indipendenti,
e solo il secondo è colpa del piano.**

1. **Difetto di canale, non di piano.** La patch è arrivata strutturalmente
   monca: l'hunk `@@ -2983,7 +3028,16 @@` su `ktest.worker.ts` dichiara 7 righe
   old / 16 new e ne consegna 6 / 15 — manca l'ultima riga di contesto (la
   graffa di chiusura). `git apply --check` esce 128 con «corrupt patch at
   line 154». L'integrator lo ha misurato invece di dedurlo: la trascrizione
   fedele fa 25.658 caratteri contro i 25.664 dichiarati nel brief, cioè
   **6 caratteri**, l'ordine di grandezza esatto della riga mancante. È la
   stessa famiglia del troncamento a 16 KB già corretto alla fonte, e va
   segnalata alla sessione harness.
2. **OWNERSHIP SOVRAPPOSTA FRA DUE TASK DELLA STESSA ONDATA — questo sì è un
   difetto del piano, ed è quello da ricordare.** T1 e T2 scrivevano
   **lo stesso blocco** di `ktest.worker.ts` con soglie **diverse**: T1 rel 1e-4
   / abs 1e-5 locali, giustificate da simulazione f32; T2 rel 5e-4 / abs 1e-4
   importate da un modulo nuovo `src/engine/attnchunktol.ts`. Il contratto §4
   pretende `owns` disgiunti e li ha validati in codice, ma la disgiunzione è
   **per file**, e due task possono possedere regioni diverse dello stesso file
   solo finché non si scoprono a scrivere la stessa. T1 è arrivato primo e ha
   vinto; T2 non applica più perché il contesto che cerca non esiste.

**Conseguenza sostanziale, da non nascondere dietro il BLOCKED**: la tolleranza
oggi vive come **due costanti locali** in `ktest.worker.ts:2987`, non in una
sede unica. L'AC3 chiedeva «dichiarata PRIMA, con la ragione numerica scritta»,
e T1 la soddisfa (dichiarata prima, ragione in loco, banda nel `note` del
banco). **Non** soddisfa il principio del progetto «una soglia, un posto», che è
esattamente il difetto che è costato la riga 2. Il modulo `attnchunktol.ts` di
T2 è la forma giusta e **non esiste in albero**.

**T4 (`assemblatore-e-copertura`) BLOCKED per dipendenza a monte**, non per un
suo difetto. È la clausola (d)/AC6: la copertura dei call-site con worklist.

**Nessuno dei due si ripara rilanciando il conductor**: T2 va rifatto a mano
sopra il codice di T1 (spostare le due costanti in `attnchunktol.ts` e importarle
nei due consumatori), T4 è un censimento che ho già fatto in it.17 e che va solo
scritto.

**RULING (PI, 2026-08-14):** «Sulla bit-identità capisco che non sia fattibile
in questa fase di transizione, ma dovrà esserci in futuro. Quando avremo migrato
tutto su int.»

**È UNA SOSPENSIONE, NON UN'ABOLIZIONE**, e la differenza è tutta nel
meccanismo che la fa finire. Forma meccanica applicata in it.20:

1. **Criterio di oggi: ARGMAX IDENTICO su ogni chunk**, col numero accanto —
   `maxAbs` e `maxRel` restano nel report, o chi legge fra sei mesi non ha modo
   di accorgersi che la divergenza sta crescendo. L'argmax è un criterio più
   debole e va detto: può coincidere anche con logit visibilmente diversi.
2. **La bit-identità resta MISURATA.** `bitIdentical` continua a comparire nel
   report anche se non decide più: il giorno in cui torna vera lo si vede senza
   rifare niente.
3. **LA CONDIZIONE CHE LA RIACCENDE non è «riportare il chunk in virgola
   mobile»** — sarebbe buttare via 1,745×. È l'opposto: **quando anche il
   percorso sequenziale (`step`) passerà sulla via intera**, i due bracci
   torneranno a fare la stessa aritmetica e la bit-identità tornerà esigibile.
   Il report lo dichiara in `bitIdentityReturnsWhen`.
4. **IL PROMEMORIA SUONA DA SOLO**: `tests/engine-bitidentity-debt.test.ts`.
   Un debito scritto in un docket lo paga solo chi si ricorda di rileggere il
   docket. Quel test invece **fallisce da solo** nell'istante in cui la
   condizione del ruling si avvera — il segnale è l'intrinseco `dot4I8Packed`
   che compare nel GEMV sequenziale — e il messaggio di errore cita il ruling e
   dice cosa fare. Contiene anche la prova che il segnale discrimina (il kernel
   intero lo ha, quello f32 no): senza, sarebbe un test che passa comunque.

**Item 22 CHIUSO.** Il debito non è chiuso: è armato.

## item 24 — il criterio di transizione NON passa, e il margine dice perché (io → PI, it.20)

**MISURATO col criterio nuovo** (`results/engine/q35-prefillchunk-4b-it20-margine-2026-08-14.json`):

    argmaxSame 63 / 64        maxAbs 2,3839   maxRel 2,0000
    bitEqual   31 / 15.892.480

**Il disaccordo è UNO, ed è al chunk 0:**

    seqTok 11 · chunkTok 248046
    seqTop2Gap  0,0179     il sequenziale preferiva il proprio vincitore di 0,018
    swapMargin  0,0179     identico ⇒ il vincitore del chunk ERA il suo secondo
    chunkTop2Gap 0,0814

**LA TENTAZIONE ERA DICHIARARLO PAREGGIO, E I NUMERI NON LO CONSENTONO.**
La casa ha già l'idioma giusto per i quasi-pareggi — il banco
`router-top4-near-tie` conta i flip solo sopra una separazione dichiarata di
**1e-5**. Applicando quella soglia, che è pre-esistente e non scelta da me su
questi dati, il flip **conta lo stesso**: 0,0179 è **1.800×** la soglia di casa.
Non è un pareggio, è un disaccordo con margine piccolo ma reale.

**E il rapporto fra i due numeri è la cosa che conta davvero.** La
quantizzazione int8 delle attivazioni perturba i logit fino a **2,38**, mentre
il distacco fra primo e secondo qui è **0,018**: la perturbazione è ~130× il
margine. In quel regime **qualunque** posizione con distacco sotto ~2,4 può
ribaltarsi, e che 63 chunk su 64 abbiano tenuto dice più sulla distribuzione dei
distacchi che sulla robustezza del kernel. Il campione è 64 chunk — e la
landmine di questo progetto sul campione da 22 posizioni avverte esattamente di
non concludere da un conteggio così.

**NON DICHIARO IO LA SOGLIA.** Ho già visto il numero: sceglierla adesso sarebbe
il difetto che l'AC3 della riga 3 esiste per impedire («dichiarare la tolleranza
DOPO aver visto il numero»). E una soglia di gate è materia del PI.

**Le opzioni, con la mia raccomandazione per prima:**

(a) **Accettare 63/64 come stato di transizione dichiarato**, col margine nel
    report, e trattare l'argmax come sorvegliato-non-vincolante finché la
    migrazione a intero non è completa — che è il momento in cui il ruling
    riaccende la bit-identità. È coerente con lo spirito del ruling: la fase di
    transizione ha una correttezza più debole, dichiarata, e con la data di
    scadenza già scritta nel test che suona da solo.
(b) **Alzare l'asticella**: zero flip, e allora il gate è ROSSO oggi e la via
    intera non è mergiabile com'è. Costa 1,745× sul moltiplicatore.
(c) **Misurare prima di decidere**: rigirare il gate su più prompt (oggi è un
    prompt, 64 chunk) per sapere se 63/64 è tipico o fortunato. È l'unica
    opzione che aggiunge informazione invece di scegliere una soglia, e costa
    una run.

**RULING:** _

### OPZIONE (c) ESEGUITA (it.21) — 63/64 non era tipico, era il caso buono

Prima serviva togliere un difetto: **il gate era cablato su `golden.prompts[0]`**,
quindi `--prompts N` non cambiava ciò che misurava e chi lo passava credeva di
aver variato qualcosa. Ora prende il primo dei prompt selezionati (default
invariato).

| prompt | argmax | maxAbs | distacchi dei disaccordi |
|---|---|---|---|
| 0 | 63/64 | 2,384 | 0,0179 |
| 1 | **61/64** | **10,139** | 0,0525 · **0,2652** · 0,0056 |
| 2 | 62/64 | 1,470 | 0,0302 · **0,3574** |
| 3 | **64/64** | 0,851 | — |

**Totale 250/256 = 97,66%.** Tre cose che il campione singolo non poteva dire:

1. **63/64 stava sopra la media, non sulla media.** L'intervallo vero è 61-64, e
   il peggiore è 95,3%. Concludere da un prompt sarebbe stato l'errore che la
   landmine del campione da 22 posizioni descrive.
2. **La perturbazione varia di 12×fra i prompt** (maxAbs 0,851 → 10,139). Sul
   prompt 1 la quantizzazione int8 sposta un logit di **oltre 10**. Non è una
   proprietà del kernel sola: dipende dai dati che ci passano.
3. **E QUESTO CHIUDE LA QUESTIONE DEL «PAREGGIO».** I distacchi più grandi che
   si sono ribaltati sono **0,2652** e **0,3574** — rispettivamente 26.500× e
   35.700× la soglia di quasi-pareggio che la casa usa nel banco
   `router-top4-near-tie` (1e-5). Non c'è nessuna soglia onesta sotto la quale
   questi sei disaccordi diventino rumore: **sono cambi di predizione veri**.

**Cosa cambia per le tre opzioni.** La (b) resta cara com'era. La (a) resta
difendibile — è una fase di transizione dichiarata, con la scadenza già armata
nel test — ma il numero onesto da scriverci accanto non è «63/64»: è **250/256,
peggior prompt 61/64, con ribaltamenti fino a 0,36 di distacco**. Chi accetta
la (a) accetta questo, non un arrotondamento.

### LA RISPOSTA PRATICA (it.21) — l'effetto è ZERO, misurato nella metrica di casa

Domanda del PI: «non so se nella pratica l'effetto è trascurabile o
problematico». Il gate confronta i due bracci **fra loro**; non poteva
rispondere. Serviva misurare contro l'**oracolo**, e non si poteva: la
conformance golden prefillava sempre con `step()`. Aggiunto `--conf-prefill-m`
(flag separato da `--prefill-m`, che attiva il gate ed esce prima del replay).

**Due bracci full-corpus, stesso codice, stessa sessione:**

| | top-1 vs oracolo | sequenza generata |
|---|---|---|
| prefill sequenziale | **1012/1024 = 98,828%** | — |
| prefill a chunk M=16, via intera | **1012/1024 = 98,828%** | **identica su 8/8 prompt** |

Non «equivalente entro tolleranza»: gli argmax generati sono gli **stessi
token**, su tutti e 1024 le posizioni, su tutti e otto i prompt. E il
sequenziale riproduce esattamente il ratchet storico 1012/1024, quindi il
confronto è ancorato.

**PERCHÉ I SEI DISACCORDI DEL GATE NON SI VEDONO.** Sono ai **confini dei
chunk**, e quei logit **in produzione non li legge nessuno**: il prefill serve a
riempire la KV cache, e l'unica cosa che esce è il primo token. La differenza
nella cache c'è — ma non basta a spostare un solo token generato.

**E IL SEGNO DELLA DOMANDA ERA ROVESCIATO.** L'oracolo **è llama.cpp**
(`golden.oracle.impl = "llama.cpp-oracle"`), e llama.cpp quantizza le
attivazioni a Q8_0 per **ogni** matmul q4_0, su ogni token, prefill e decode:

    ggml/src/ggml-cpu/ggml-cpu.c
    [GGML_TYPE_Q4_0] = { .vec_dot = ggml_vec_dot_q4_0_q8_0,
                         .vec_dot_type = GGML_TYPE_Q8_0, ... }

La nostra via intera fa **quello che fa il riferimento**. È il nostro percorso
sequenziale in virgola mobile a essere *più preciso dell'oracolo* contro cui ci
misuriamo. Chiamare «regressione» l'avvicinamento all'implementazione di
riferimento era il verso sbagliato.

**Conseguenza per l'item 24**: la scelta non è più fra tre opzioni al buio. La
(a) è sostenuta da una misura diretta, e il numero da citare non è né 63/64 né
250/256 — è **1012/1024 top-1 e 8/8 sequenze identiche**. Resta vero che i sei
disaccordi non sono pareggi; è vero anche che non arrivano all'uscita.
Il gate `q35-prefillchunk-4b` misura una quantità che **non è quella di
prodotto**, e questo andrebbe scritto nel suo `declared`.

## item 25 — (e2a): il debito è dichiarato, ma il done-when del CONTRATTO va deciso (io → PI, it.22)

La riga 4 dice: `rmsPairGemmSiluChunkFast` «o scende sotto 16.384 B, o la
clausola si dichiara **debito del path 0.5B** con la ragione, **e il done-when
del contratto va a docket per un ruling**». Ho fatto la seconda: il debito è
dichiarato in `gpulimits.ts` con l'aritmetica, la ragione e la via d'uscita, ed
è ora **falsificabile** — `tests/gpulimits.test.ts` fallisce se qualcuno cabla
il kernel fuso nell'assemblatore del 4B, perché in quel caso (C7-3) diventa
falsa e il debito smette di essere «solo 0.5B».

**Quello che resta è il testo del contratto, e non lo tocco io.** Il DONE WHEN
del goal elenca la portabilità fra le sue voci. Con (e2a) dichiarata debito, il
goal può chiudere lasciando **un termine sopra la garanzia WebGPU di 16.384 B** —
l'unico del motore. Su un device che concede solo il minimo di spec, la pipeline
del path di conformità 0.5B non si crea.

**Perché non lo decido io** (test dello step 5): se non arrivasse mai risposta
dichiarerei il debito e chiuderei — che è ciò che ho fatto. Ma **cosa il DONE
WHEN del goal promette** è la funzione obiettivo, non il meccanismo: è la sola
cosa che il protocollo mi vieta di decidere.

**Le opzioni, con la mia raccomandazione per prima:**

(a) **Il DONE WHEN della portabilità si legge come soddisfatto**, perché
    (C7-3) — ratificata nel plan-check, item 1 — scopa quella clausola fuori dal
    percorso di prodotto del 4B. Il debito resta scritto, falsificabile, e
    diventa scope del goal che porterà il path 0.5B alla forma multi-riga.
(b) Il goal **non chiude** finché il termine non scende sotto 16.384. Costa un
    lavoro che (C7-3) dice non muovere la metrica di questo goal.

**Il conto, per decidere con un numero e non a sensazione**: il termine è
`4·K·mMax` con K = 896 fisso ⇒ sotto 16.384 servirebbe **mMax ≤ 4**. Alzare mMax
è la leva del prefill, quindi le due tirano in direzioni opposte (a mMax 16
sarebbero 61.696 B). La strada vera non è stringere il buffer ma **dare al path
0.5B la forma multi-riga del 4B, il cui workgroup storage non scala con M**
(1.152 B via idot, 4.096 via f32 a M=16). Quella forma esiste già: manca solo
portarcela.

**RULING:** _

**RULING (PI, 2026-08-14) sull'item 24 — il `declared` NON si tocca:** «Ottimo
che l'oracolo confermi. Per la questione del declared bit-identico, io non la
cambierei per ora se poi torneremo alla bit-identità per tutti. Siamo in una
work in progress.»

**Applicato: nessuna modifica.** Il testo del gate resta quello che promette la
bit-identità. La ragione è la stessa che regge il promemoria di it.20: il ruling
sulla bit-identità è una **sospensione con data di scadenza**, non un cambio di
rotta, e riscrivere il `declared` oggi vorrebbe dire cancellare la promessa che
si intende mantenere. Un gate che promette più di quanto oggi ottenga, **dentro
una transizione dichiarata e con la sua scadenza armata in un test**, è più
onesto di un gate riscritto attorno allo stato provvisorio.

Restano scritti nell'artefatto i numeri veri (`argmaxSame`, `bitIdentical`,
`maxAbs`, `maxRel`, `argmaxDiffs` col margine), e resta armato
`tests/engine-bitidentity-debt.test.ts`, che suona quando il percorso
sequenziale passerà anch'esso su intero. **Item 24 CHIUSO.**

## item 26 — «l'unico termine che sfora» era FALSO: sono quattro, e il numero era giusto per coincidenza (io, it.24)

Il PI ha bocciato la mia scelta di dichiarare (e2a) come debito invece di
sistemarla: «stai aggiungendo regole, check, controlli ed eccezioni varie che
sporcano il codice ... Piuttosto dovresti lanciare uno o più subagent per
adeguare anche il modello 0.5B. Questo è il tipo di comportamento che mi aspetto
da te. Che migliori le cose che incontri strada facendo.» Ruling accettato e
scritto in memoria (`fix-dont-fence`).

**Il subagent si è fermato PRIMA di scrivere codice, ed è stata la cosa giusta**:
la premessa del lavoro era falsa. Verificato da me generando il WGSL di ogni
pipeline che `gpuforward.ts` crea e sommando le `var<workgroup>`:

| B | pipeline | era dichiarato? |
|---|---|---|
| 30.848 | `rmsPairGemmSiluChunkFast` K=896 m=8 | **sì**, era `QWEN_WORKGROUP_STORAGE_BYTES` |
| 30.720 | `rmsGemmQkvChunkFast` K=896 m=8 | no |
| 30.720 | `gemmResidChunkFast` K=896 m=8 | no |
| 19.712 | `gemvResidualFast` K=4864 M=1 | no — ed è il **DECODE** |

**IL PERICOLO NON ERA IL VALORE, ERA LA DERIVAZIONE.** I 30.848 erano il massimo
dei quattro, per caso. `limitsFor` usa quella costante come `requiredLimits`,
cioè come **tetto** del device: bastava portare `pairSilu` alla forma multi-riga
perché la costante scendesse a ~4.096 mentre `rmsQkv` continuava a chiederne
30.720, e `createComputePipeline` sarebbe fallito in validazione **su ogni
device, 4090 compreso**. Cioè: il lavoro che stavo per commissionare avrebbe
rotto il motore, e il recinto che avevo costruito non l'avrebbe visto — perché
sorvegliava la cosa sbagliata.

**FATTO SUBITO** (`it.24`): la costante non è più scritta a mano, è un
`Math.max` **calcolato** dalle formule che vivono accanto ai kernel
(`qwenFusedChunkWorkgroupStorageBytes`, `qwenGemvResidualWorkgroupStorageBytes`).
Stesso valore, 30.848, ma per costruzione invece che per fortuna: un massimo
calcolato non può scendere sotto un consumatore vivo. Tolto il test-recinto
`(e2a)`, corretta l'asserzione **falsa** di `gpulimits.test.ts` che diceva «ciò
che resta sopra i 16 KB è UN consumatore solo».

**IL LAVORO VERO È GOAL-SIZED, e va nominato invece che stipato in una riga.**
Sono quattro porting, non uno, e uno è sul path caldo del decode:

1. `pairSiluC` → rms di chunk + quantX + 2 split-K + 2 combine + siluMul
2. `rmsQkvC` → rms di chunk + quantX + split-K + combine + bias
3. `oResidC` → split-K + combine + residual **oppure** una riga sola: la soglia
   `useShared = mMax*K*4 <= 28672` (`wgsl.ts`) è centrata **esattamente** su
   8·896·4 = 28.672, e abbassarla manda `x` in storage facendo crollare il
   termine a 2.048 B
4. `gemvResidualFast` (decode): mette `x` in shared **senza soglia**. Non è
   raggiungibile dalla forma multi-riga — gira a M=1 e non è un GEMM di
   prefill. **Finché c'è lui, nessun porting del prefill porta il 0.5B sotto i
   16.384 B.**

Tutti e quattro spostano kernel misurati (TTFT del prefill, tok/s del decode) e
cambiano `dispatchesPerToken`: sotto il ruling di non-regressione vogliono una
passata GPU + conformance prima del merge. **Proposta: goal suo**, accanto a
quello sul tempo al primo token, non una coda di questa riga.

**Tre difetti minori visti e NON sistemati** (perché fuori dal percorso, e
inseguirli sarebbe la deriva che il protocollo vieta): `gemmResidChunkFast` sta
esattamente sul confine della sua soglia shared (28.672 = 28.672), quindi mMax=9
o un K diverso cambierebbero ramo in silenzio; il termine del decode non è
nominato in nessun documento; e il test che scansiona il WGSL guarda i kernel
d'attenzione ma mai la famiglia dei GEMM di chunk, che è il motivo per cui la
svista non poteva cadere da sola.

## item 27 — TRIAGE DI CHIUSURA: cosa resta aperto, e a chi (io → PI, it.26)

Consuntivo in `docs/engine/ttft-consuntivo-2026-08-14.md`.

**CHIUSI nel goal**: 1 (plan-check), 6 (tetto negoziabile: non e' una leva), 11
(dot product intero: autorizzato, montato, 1,745x), 13 (il ktest usciva 0 senza
girare), 14 (costo della quantizzazione delle attivazioni), 19 (barra del riuso),
20 (cablaggio verificato), 21 (fusione GQA esclusa coi numeri), 22 (criterio del
gate: sospensione con scadenza armata), 23 (il conductor regge), 24 (la costante
era giusta per coincidenza).

**SUPERATI DAI FATTI**: 15, 16, 17, 18 — le quattro diagnosi «il veicolo
`sdd-conductor` non chiude un task di questa classe qui». Erano una diagnosi
sbagliata di una causa infrastrutturale: le morti erano il segnale 144 al confine
di turno, e dopo il riavvio il veicolo ha completato al primo tentativo (item
23). **Non cancellati**: restano come storia di un'attribuzione sbagliata, che e'
il loro valore.

**APERTI, e sono tutti e tre del PI:**

1. **La chiusura formale del goal con la barra mancata.** 32.127 ms contro
   21.905. Dieci clausole su dodici sono soddisfatte, la causa delle due che
   cadono e' misurata ed e' fuori dalla portata di questo goal (v. sotto). Non
   la decido io: «cosa il DONE WHEN promette» e' funzione obiettivo, ed e' la
   sola classe che il protocollo mi vieta.
2. **item 25** — cosa promette il DONE WHEN sulla portabilita', ora informato
   dall'item 26.
3. **item 26** — il porting del path 0.5B come goal suo (quattro kernel, uno non
   raggiungibile dalla forma multi-riga).

**E UNA COSA CHE IL PI DEVE SAPERE PRIMA DI PRIORITIZZARE**, perche' ribalta un
suo ruling precedente: l'11,54% dei byte che l'item 19 aveva assegnato al goal
K-quant **come coda** e' il **37,9% del tempo** del prefill. La quota di byte
sottostimava il peso perche' la forma legacy rilegge i pesi M volte. Il goal
K-quant non e' un completamento: e' **la leva piu' grande rimasta sul tempo al
primo token**.

**RULING:** _

---

**RULING (PI, 2026-08-14) — CHIUSURA DEL GOAL, e i due successivi:** «Ok
chiudere il goal. Abbiamo fatto un ottimo lavoro. I prossimi sono i k quant e il
0.5B (non so ancora in che ordine). Per il 0.5B migreremo il possibile.»

**Applicato. Chiude tre item in un colpo:**

- **item 27 — chiusura formale con la barra mancata: SI'.** Il goal chiude a
  32.127 ms contro una barra di 21.905, con dieci clausole su dodici soddisfatte
  e la causa delle due mancate misurata e fuori dalla sua portata. `GOAL.md`
  porta `STATUS: CLOSED` con la citazione.
- **item 26 — il porting 0.5B diventa un goal: SI'**, ed e' uno dei due
  successivi. **Scope fissato dal PI: «migreremo il possibile»** — cioe' i tre
  siti che una forma esistente puo' accogliere, con il quarto
  (`gemvResidualFast`, down-proj del decode a M=1) **dichiarato non-migrabile**,
  non contato come buco. E' esattamente la forma che il workflow
  `pattern-migration` ora produce (`nonMigrable` + `reading`, harness `5180ff0`).
- **item 25 — cosa promette il DONE WHEN sulla portabilita': RISOLTO PER
  CONSEGUENZA**, e lo registro come mia lettura del ruling invece che come
  risposta separata: il goal chiude con (e2a) dichiarata debito, e quel debito
  **diventa il goal 0.5B**. Non resta implicito e non resta orfano.

**APERTO, e il PI lo dice esplicitamente: l'ORDINE fra K-quant e 0.5B.** Non lo
decido io. Il dato che gli serve, gia' misurato: **il K-quant vale il 37,9% del
tempo del prefill** (`ssm_out` Q5_K sul percorso vecchio, riletto 16 volte per
chunk, 89,9 GB/s su un motore che ne fa ~300), mentre il **0.5B non muove la
metrica di prodotto** — e' il path di conformita', e il suo valore e' la
portabilita' su device che concedono il minimo di spec. Due assi diversi:
velocita' contro raggiungibilita'.
