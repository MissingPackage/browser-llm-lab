# DIGESTS — engine-kquant

Tre-sei righe per iterazione. Chi legge solo questo file deve sapere dove sta
il goal e cosa e' in dubbio.

## it.0 — la spina (2026-08-14)

- Contratto chartered: doppia barra sul TTFT (**< 22.500 obbligatoria**,
  **< 18.000 nice to have**), segmenti `gemm:deltanet-out` e `gemm:ffn-down`
  entrambi **≤ 2.000 ms**, copertura del piano da 5,86× a **≥ 15,5×**.
- `PHASES.md`: sette righe. Fase 0 al banco (riga 1) → Q5_K in produzione
  (riga 2, −11,4 s proiettati) → Q4_1 (riga 3, −3,3 s) → le tre forme del 35B
  misurate ma NON cablate (riga 4) → misura di chiusura → gate → consuntivo.
- Quattro correzioni di fattibilita' trovate PRIMA di scrivere la tabella: lo
  split-K conta in blocchi da 32 e i K-quant hanno superblocchi da 256; la
  quantizzazione delle attivazioni si riusa tale e quale; serve un termine Σx
  che la q4_0 non ha; una cella del banco (35B down, K=512) sarebbe degenere a
  4 fette.
- Deciso da me: la famiglia di kernel si fa intera qui, il **cablaggio** del
  35B e' il goal successivo — non ha un byte di q4_0, ma il suo collo e' la
  residency, non il kernel.
- In dubbio: niente, per ora. Il plan-check e' registrato come approvato in
  modo implicito («procedi in loop»), e il primo digest e' il momento per
  cambiarlo.

## it.1 — la fase 0 autorizza entrambe le famiglie (2026-08-14)

- **Metrica obiettivo ferma a 32.127 ms per costruzione**: la riga 1 non tocca
  il motore. Ha pero' potere di veto, e non l'ha esercitato.
- **Q5_K 28,07x · Q4_1 22,58x** sulla forma di produzione, a M=16, con zero
  celle scartate dal gate del banco. La regola di stop chiedeva 1,5x.
- **Il banco riproduce il segmento vero in millisecondi**: `24 x 395 x 1,2700 =
  12.039` contro i **12.169 ms** misurati in produzione, 1,1% di scarto. Il
  termine di paragone e' il percorso vecchio, non una sua imitazione.
- Proiezione **−15,2 s ⇒ TTFT ~16,9 s**, sotto anche la barra nice-to-have. E'
  una proiezione da microbench: la conferma sta nella riga 5.
- **In dubbio, e ora ha un'azione**: la quota Q4_1 del segmento `gemm:ffn-down`
  e' DEDOTTA, non misurata (il mio confronto era banco contro banco). La riga 3
  non chiude senza una categoria di misura propria per quei quattro siti.
- Verificatore indipendente: **PASS** con sei correzioni, tutte applicate —
  fra cui `396 → 395` chunk, che stava anche nel contratto.

## it.2 — riga 1 CHIUSA (2026-08-14)

- **Metrica obiettivo ancora 32.127 ms**: la fase 0 non tocca il motore. Da
  qui in poi lo tocca: la riga 2 e' la prima che muove il numero.
- **Cinque famiglie misurate, 26 celle, zero scartate.** Tutte superano la
  regola di stop (1,5x): Q8_0 **34,65x** · Q5_K **28,03x** · Q4_1 **22,57x** ·
  Q6_K **5,36x** · Q4_K **4,16-5,20x**.
- **Tre previsioni su cinque sono cadute**, e dicono tutte la stessa cosa:
  avevo attribuito il guadagno al FORMATO, mentre e' una proprieta' della
  SHAPE. Conta quanto costa rileggere la matrice M volte — i tensori grandi
  rendono 22-35x, quelli piccoli degli expert (0,6 MB, che stanno in cache)
  4-5x.
- **Nota che il goal 35B deve ereditare**: quel 4-5x e' un LIMITE INFERIORE. Il
  banco misura un tensore in isolamento, caldo in cache; in produzione i 17,67
  GB di expert non ci stanno, e li' la rilettura costa il prezzo pieno.
- Prossimo: riga 2 — Q5_K in produzione, la prima riga che muove il TTFT
  (−11,7 s proiettati). Veicolo dichiarato: `sdd-conductor`.

## it.3 — riga 1 chiusa sul done-when SCRITTO (2026-08-14)

- **Il verificatore ha bocciato it.2**, e sul punto giusto: avevo chiuso la
  riga 1 restringendo io il done-when (niente fallback f32 sulle tre famiglie
  del 35B, e solo M=16 invece di M=1,8,16). Una delle due restrizioni violava
  un CONSTRAINT esplicito del contratto. **Non ho escalato: ho eseguito** —
  chiedere il permesso di saltare tre kernel costava piu' che scriverli.
- **54 celle, zero scartate**, tutte le famiglie a M = 1, 8, 16 con entrambe le
  vie. A M=16: Q8_0 **35,20x** · Q5_K **28,10x** · Q4_1 **22,57x** · Q6_K
  **6,13x** · Q4_K **4,16-5,23x**.
- **Il reperto che i fallback hanno comprato**: sul Q4_K la via f32 **non passa
  la regola di stop** (1,14x sulla shape gate/up, barra 1,5x). Su un device
  senza il prodotto scalare intero, li', la forma multi-riga non vale la pena.
  Non l'avrei mai saputo saltando quei kernel.
- **RITRATTATA** l'affermazione «il 4-5x del 35B e' un limite inferiore»: per
  Q4_K e Q6_K il braccio legacy che ho misurato **non e' il percorso che il
  motore esegue** (gli expert vanno in regime d'arena, dove il `batch` e'
  vietato per costruzione). Quel numero vale per una shape su un braccio
  ipotetico, e la scheda di consegna lo dira' cosi'.
- Metrica obiettivo ancora **32.127 ms**. La riga 2 e' la prima che la muove.

## it.4 — riga 2 in produzione: il Q5_K non rilegge piu' i pesi (2026-08-15)

- **Copertura del piano da 5,86x a 10,94x** sull'inventario per-layer intero del
  4B: 196/248 siti, **96,9% dei byte** ora sulla forma multi-riga.
- **Gate eseguiti da me, non dedotti dall'autoreport del workflow**: ktest
  **103 PASS / 0 FAIL** (erano 101) coi due casi nuovi a maxRel 2,6e-7 e 4,3e-7,
  vitest **779**, tsc pulito, e conformita' col golden llama.cpp sul prompt da
  6333 token a chunk: **top1 62/64**.
- **La trappola era reale**: `ssm_out` non passa dal bivio che sembrava ovvio
  cablare. Chi avesse cablato `gemvB` non avrebbe cambiato niente, coi test di
  copertura verdi lo stesso.
- **Non ancora verificato, e dichiarato**: che il segmento sia sceso davvero. Il
  piano non e' il cronometro — quella misura e' la riga 5, su codice finale.
- Riga 3 (Q4_1) lanciata: `[6c]` deve salire a **≥ 15,5x**, e porta con se' una
  categoria di misura propria per i quattro siti Q4_1, cosi' la riga 5
  attribuisce quel tempo invece di dedurlo.

## it.5-it.6 — riga 3 cablata, gate GPU indisponibile (2026-08-15)

- **Copertura 10,94x → 15,5247x**: 200/248 siti, **99,796% dei byte** del
  prefill del 4B ora sulla forma multi-riga. Resta un solo kind legacy (48 siti
  Q8_0, 0,204%, esclusi coi numeri).
- **La guardia difensiva della riga 2 ha pagato**: quando il piano ha accettato
  il q4_1 prima che il motore avesse i suoi kernel, ha fatto ricadere quei
  tensori sulla legacy invece di leggerli col kernel del q4_0 — logit storti in
  silenzio, senza nessun errore WebGPU.
- Banco ktest del q4_1 scritto, tolleranze **derivate** dal pavimento (11,8x
  sopra) e discriminante verificato: togliere il termine `m*Sigma(x)` fa
  esplodere l'errore di quattro ordini.
- **IL GATE SU GPU E' INDISPONIBILE** da ~01:00: cinque fallimenti, tre sintomi,
  e fallisce anche col banco nuovo disattivato ⇒ **non e' il kernel**. Docket
  item 2, con l'attribuzione per esclusione.
- **Suite senza GPU: 836 passed | 10 skipped**, tsc pulito.
- **STOP BY DESIGN**: tutto il resto del goal passa da quella infrastruttura.

## it.7 — riga 3 CHIUSA, il gate GPU era l'ambiente (2026-08-15)

- **ktest 105 PASS / 0 FAIL** su macchina riavviata, **senza toccare una riga di
  codice** fra l'ultimo fallimento e questa run. L'ipotesi «stato accumulato
  dell'host» era quella giusta; il banco `q35-mtp-head-real-blk32` e' scagionato
  e non diventa lavoro suo. Docket item 2 chiuso.
- **Q4_1 verificato su GPU vera**: `prefill-gemm-q41-multirow-idot` maxRel
  **1,73e-5**, `-f32` **1,51e-5**, contro pavimenti derivati 1,693e-5 / 1,715e-5
  e tolleranza 2e-4. **L'errore misurato sta sul pavimento**: il margine 11,8x
  e' aritmetica del formato, non slack — e derivare la tolleranza invece di
  sceglierla e' cio' che permette di dirlo.
- **Riga 3 CHIUSA** su tutte e cinque le clausole: copertura 15,5247x, ktest
  PASS, floor test, gpulimits, `tsc` exit 0 e `vitest` **836 passed | 10
  skipped** exit 0.
- **Lezione di it.6 registrata**: quando disattivare il codice nuovo non cambia
  l'esito, l'attribuzione all'ambiente e' gia' completa — i due ritentativi in
  piu' erano speranza, non misura.
- **Difetto tolto**: il comando di ripresa in HANDOFF.md era senza `BASE_URL` e
  puntava alla 5173 di default. Fallito al primo colpo, corretto.

## it.8 — riga 4 CHIUSA: le tre forme del 35B verificate su GPU, e non cablate (2026-08-15)

- **ktest 111 PASS / 0 FAIL**: sei bracci nuovi (q4_K, q6_K, q8_0 × idot/f32),
  tutti **sul pavimento derivato o appena sotto**. Terza volta di fila che i
  pavimenti predicono il silicio, ora su formati mai girati in produzione.
- **Conflitto di fattibilita' risolto prima di spendere**: il piano derivava i
  formati ammessi dall'elenco dei kernel, quindi «portato ma non instradato» non
  era esprimibile — e `prefillGemmCheck` non guarda N, per cui il solo q8_0
  avrebbe instradato in silenzio i 48 siti del 4B a N=32. Flag `wired` +
  `wiredWhy`, una sede sola, rifiuto con frase propria.
- **Il port e' provato port**: `PREFILL_GEMM_PORT_DIFFS` resta vuoto con sei
  kernel in piu' (14 coppie, due direzioni). Produzione del 4B invariata:
  `[6c]` 15,5247x, 200/248 siti, zero dispatch di differenza.
- **Refuso corretto nel contratto**: il prefill del 35B NON gira su
  `moeprefillplan.ts` (unico consumatore: il GLM). Il lavoro del goal successivo
  e' il ramo `moe` di `q35gpumodel.ts` — 40 readback per chunk, 512 dispatch per
  layer. Il piano CPU-side e' gia' parametrico e non e' il lavoro.
- **Suite: 998 passed | 10 skipped** (erano 836), tsc exit 0.
- **Trappola consegnata scritta**: `wired` e' per formato, non per shape. Quando
  il 35B accendera' q8_0, i 48 siti del 4B entrano nello stesso istante.

## it.9 — riga 5 CHIUSA: 1,873x misurato, barra e nice-to-have passati (2026-08-15)

- **TTFT a caldo 32.127 -> 17.153 ms.** Sotto la barra (22.500) E sotto il
  nice-to-have (18.000). prefill 197,25 -> **369,72 tok/s**. Host quiescent,
  zero errori GPU. **Prima misura di tempo dell'intero goal.**
- **La proiezione del banco (~16,9 s) ha sbagliato dell'1,5%**, e la
  decomposizione predetta nel contratto pure: 9.167 fuori dai pass + 7.959
  dentro = 17.126, contro «9.350 + 8.050» scritti prima di iniziare.
- **Segmenti**: `gemm:deltanet-out` 12.169 -> **572,8 ms**; `gemm:ffn-down`
  4.971 -> **1.413,4** (1.187,6 + 225,8 della categoria q41 scorporata: il
  confronto onesto e' la somma).
- **Termine PRIMO dopo la leva, misurato**: `deltanet:recurrence` 1.732 ms
  (10,11% del prefill). E' la voce che la riga 7 deve nominare.
- **Una run di GPU buttata**: la riga EVIDENCE del contratto ometteva
  `--prefill-m 16` (default `null` ⇒ prefill sequenziale, 91.230 ms).
  Intercettata perche' il runner DICHIARA `prefillPath` invece di lasciarlo
  dedurre. Secondo comando incompleto trovato oggi, dopo il `BASE_URL`.
- **Il debito del checkpoint era peggio del dichiarato**: l'inventario dei siti
  viveva dentro un file di TEST (per questo i byte erano ricopiati), e il
  checkpoint del 14 agosto pubblicava DUE numeri falsi — `gemm:deltanet-out`
  ancora `legacy` (16x i byte veri) e `gemm:qkv` contato su 80 tensori invece di
  24 (**738 GB/s erano 5x**). Ora byte e forma sono derivati, con due guardie
  nuove e il test che li sorveglia.
- **Trovato nel blocco accanto**: baseline e barra di engine-**ttft** ancora
  incise nel builder — avrebbero pubblicato 5,108x invece di 1,873x. Ora
  arrivano dal contratto, obbligatorie, con un test che impedisce di rimetterle.
- **Tre campi del ratchet dichiarati "NON MISURATO"** invece di ricopiati: il
  top-1 pieno viene da un albero precedente. Sono gate della riga 6.
- Gate: tsc exit 0 · vitest **1017 passed | 10 skipped** · **ktest 111 PASS / 0
  FAIL** sull'albero finale.

## it.10 — riga 6 CHIUSA: il gate di merge passa (2026-08-15)

- **Otto gate, tutti misurati su QUESTO albero** (era il difetto che il ratchet
  denunciava da solo): ktest 111/0 · top-1 **1012/1024** su ENTRAMBI i bracci ·
  **sequenze generate identiche 8/8, zero token di differenza** · decode 47,06
  (barra 45,5) · vitest 1017 passed | 10 skipped · tsc exit 0.
- I due bracci coincidono anche nella ripartizione per prompt, non solo nel
  totale: il prefill a chunk non cambia cosa il modello scrive.
- **Una clausola interpretata e dichiarata**: il GLM b12 sta FUORI dalla banda
  +-5% da tutte e tre le parti, **in meglio** (decode +16,4%, prefill +20,1%,
  TTFT -16,7%). La leggo come banda di rumore per la non-regressione, non come
  requisito a due code. **E non la spaccio per una vittoria**: host diverso fra
  riferimento e run, e il decode +16,4% NON e' spiegabile da questo goal (a M=1
  la forma multi-riga perde ed e' esclusa dal decode).
- **Ratchet e checkpoint ricostruiti**: niente piu' `NON MISURATO`, e la banda
  per segmento e' derivata invece che ricopiata.
- **GPU nei pass 46,5%, fuori 53,5%**: piu' della meta' del prefill non e' piu'
  calcolo. Da consegnare alla riga 7 insieme a `deltanet:recurrence`.
