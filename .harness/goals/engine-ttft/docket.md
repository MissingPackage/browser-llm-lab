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
