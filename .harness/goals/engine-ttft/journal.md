# JOURNAL — engine-ttft

## it.0 (2026-08-13) — decomposizione, e tre correzioni trovate verificando la fattibilità

Goal aperto dopo la chiusura di `engine-kernel-decode` (47,93 tok/s a ctx 6333,
4,82x). Ruling A del PI in chat: i 4 secondi valgono sul TTFT **a modello
caldo** (`prefill.ms + decode.firstMs`); il load ha una soglia sua, separata.

**Il contratto nasce già da una correzione.** Il numero «TTFT 22,7 s su un
prompt da 6k» che HANDOFF portava è sbagliato in due punti, entrambi verificati
sull'artefatto `results/engine/q35-bench-4b-tier8-fase-d-it43.json`:

1. il prompt è il **prompt-idx 4 = 388 token**, non 6k. I 6333 sono il
   prompt-idx 0, usato per il DECODE del goal precedente. Due run su prompt
   diversi, appaiate per errore nel consuntivo.
2. `ttftMs = loadMs + prefillMs + firstMs` (`q35conf.worker.ts:246`): dei
   22.695 ms, **10.890 sono il LOAD** e 11.760 il prefill di 387 token. Il primo
   token vero costa 46 ms.

Il TTFT su un prompt da 6k **non è mai stato misurato**. Da qui il primo
done-when del contratto e la riga 0 di PHASES.

**Poi il test-fit dei done-when (ruling C7) ne ha cambiati altri tre.** Sono in
`PHASES.md` per esteso e a docket item 1 per il ruling; qui il sunto:

- **(C7-1)** il bench prefilla **una posizione alla volta**
  (`q35conf.worker.ts:207`, «prefill sequenziale»). I 32,91 tok/s non misurano
  un chunking che rende poco: ne misurano l'assenza. `prefillChunk` esiste
  (`q35gpumodel.ts:1281`), è gated bit per bit, e non è sul percorso del bench.
- **(C7-2)** M=8 è **degenere per l'obiettivo**: con riuso perfetto e a 300 GB/s
  (la banda migliore mai misurata qui) sono 5,94 s di sola lettura pesi, sopra
  il budget di 3,96. Il done-when del contratto dice «≥ 4x a M=8»; PHASES scrive
  **≥ 8x a M ≥ 16**. Scostamento dichiarato, ruling richiesto.
- **(C7-3)** la clausola (e2a) è sul path di conformità **0.5B**:
  `rmsPairGemmSiluChunkFast` è importato da `gpuforward.ts`, che assembla
  Qwen2.5-0.5B. Non tocca la metrica di questo goal. La (e2b) sì.

**E un conflitto strutturale fra due done-when del contratto**, che nessuno dei
due nominava: alzare `mMax` per il riuso peggiora il workgroup storage
(`mMax·3856` B — 30.848 a M=8, 61.696 a M=16), che (e2a) vuole sotto 16.384. Le
due leve tirano in direzioni opposte. È la riga 1 a dover trovare una forma il
cui shared non scali con M.

**Perché la forma attuale non riusa niente, per costruzione**: il prefill del 4B
non usa una GEMM. Usa `gemvQuantWgsl` con `batch: true` dispatchata su
`[gx, gy, M_MAX]` (`q35gpumodel.ts:720`) — cioè **M GEMV replicate sull'asse z**,
ognuna che rilegge la riga di pesi per intero. E il GEMV veloce nuovo del goal
precedente **rifiuta esplicitamente `batch`** (`wgsl.ts:216-249`): il prefill non
ha ricevuto nemmeno l'1,77x dei moltiplicatori. Due fatti che spiegano perché
oggi il prefill (32,91 tok/s) è più LENTO del decode (47,93).

**Fabbisogno di calcolo, per sapere se il bersaglio esiste**: 2·P·T con P=4e9 e
T=6333 ≈ **50,7 TFLOP**, cioè **12,8 TFLOP/s sostenuti** in 3,96 s. Il picco
fp32 di questo device in WebGPU non è mai stato misurato: è la sonda (a) della
riga 1, ed è ciò che decide se la riga 5 chiude col bersaglio o con
l'esclusione. [proiezione, non promessa]

**STOP by design**: goal di prodotto ⇒ gate `plan-check`. PHASES.md aspetta
l'approvazione del PI prima dell'iterazione 1.

## it.1 (2026-08-13, riga 0) — BASELINE MISURATA, e l'ipotesi della riga è refutata

Due bracci sul prompt-idx 0 (6333 token), host `quiescent` verificato a mano
prima di ogni run (GPU 0%, 572/581 MiB), tre termini scomposti in entrambi.
Artefatto unico: `results/engine/ttft-baseline-4b-prompt0-2026-08-13.json`.

| | load | prefill | tok/s prefill | firstMs | **TTFT a caldo** | decode |
|---|---|---|---|---|---|---|
| (a) sequenziale | 10.892 | 87.582 | **72,30** | 36 | **87.618 ms** | 47,79 |
| (b) chunked M=16 | 10.366 | 184.283 | 34,36 | 27 | 184.310 ms | 44,81 |

**IL CHUNKING È 2,10× PIÙ LENTO.** L'ipotesi che la riga 0 doveva testare —
«instradare il bench su `prefillChunk` potrebbe muovere la metrica senza
scrivere un kernel» — è **refutata dalla misura**. Va nella direzione opposta.

**La causa combacia col censimento di it.0, e questo è il valore del risultato.**
`prefillChunk` instrada su `gemvQuantWgsl` con `batch: true`, cioè M GEMV
replicate sull'asse z con riuso dei pesi ZERO; e il GEMV veloce del goal
precedente (vec4 + 2 righe/workgroup + `subgroupAdd`) **rifiuta `batch` per
costruzione** (`wgsl.ts:216-249`). Il path sequenziale usa `step()`, che quei
kernel veloci li ha. Quindi oggi il chunking paga M volte la lettura dei pesi
**col kernel lento**, e il sequenziale la paga una volta per token **col kernel
veloce** — e vince. Non è un difetto del chunking: è che il chunking non ha mai
ricevuto le ottimizzazioni del decode.

**Conseguenza per la riga 2**: non basta "usare `prefillChunk`". La leva 1 deve
portare il riuso vero AL kernel veloce, non instradare su quello lento.

**Correzione a un numero mio di it.0**: avevo proiettato ~192 s di prefill dai
32,91 tok/s del contratto. Quel 32,91 veniva da `q35-bench-4b-tier8-fase-d-it43`
— codice vecchio, prompt-idx 4 da 388 token. Sul codice di oggi il prefill
sequenziale rende **72,30 tok/s**: l'attenzione riscritta del goal precedente ha
già dato 2,20× anche al prefill, perché il prefill sequenziale *è* il path del
decode. Il prefill misurato è **87,6 s**, non 192.

**LA BASELINE DEL GOAL**: TTFT a caldo **87.618 ms** contro un bersaglio di
4.000. Serve **21,9×**, non i 48× che la proiezione sbagliata suggeriva.

**Primo bench end-to-end del 4B sul codice di oggi**: decode 47,79 tok/s a ctx
6333-6397, che conferma il 47,93 del checkpoint di micro-bench e soddisfa il
gate di non-regressione (≥ 45,53).

**Osservazione, non regressione**: nel braccio (b) il decode rende 44,81 tok/s,
−6,2% dal riferimento e fuori banda. Il codice committato non è toccato (senza
`--prefill-m` il modello non si costruisce con `prefillM`), ma costruire il
modello con `prefillM: 16` sembra costare ~6% al decode. n=1 per braccio:
registrato, non concluso.

**Codice toccato** (dentro l'authority della riga 0: `src/engine/q35conf/**`,
`scripts/**`): il braccio del prefill nel bench, con `prefillPath` DICHIARATO
nel report — è l'applicazione diretta della lezione di it.9 del goal precedente,
dove due JSON incompatibili furono riconciliati confrontando quel campo e non i
numeri. E il fix di `--out` assoluto/relativo (docket item 3) su
`q35-bench-run.mjs`, fatto qui perché la riga toccava comunque il runner e la
modifica viaggia con due run che la esercitano.

**Gate**: ktest 100 PASS / 0 FAIL · vitest 531 passed | 10 skipped ·
`tsc --noEmit` pulito · non-reg decode PASS.

---

## it.2 — riga 1: le leve esistono (43x), il bersaglio no (~8,7 s proiettati)

Fase di sole misure con potere di chiudere il goal. Pre-registrazione committata
**prima** di eseguire (`79cc3bd`), banco dopo (`d29eb0d`), risultati dopo ancora:
l'ordine dei commit è la cosa che rende falsificabile tutto il resto.

**LA REGOLA DI STOP NON SCATTA, e non ci va vicino.** A M=16 su K2560×N9216 la
forma vincente fa **43,1x** la forma attuale (262.881 contro 6.101 token/s), con
riuso dei pesi **16,0x** — il massimo teorico, e il doppio degli 8x che il
done-when della riga 2 chiede. La riga 2 e la riga 3 si fanno.

**Ma la vincitrice non è quella predetta, e il motivo conta.** P3 diceva `regs`
(pesi in registri): vince invece `splitk`, che è `regs` col contesto K spezzato
su 4 workgroup — stesso corpo, stesso workgroup storage (4.096 B), **2,13x** più
veloce. La ragione la dicono le bande: `regs` legge i suoi 13,3 MB unici a soli
**108 GB/s** su un device che ne fa 300+. Non era memory-bound: era
**occupancy-bound** (144 workgroup su 76 SM). La pre-registrazione aveva
modellato il problema come traffico di memoria, e per quello ha nominato la
forma sbagliata. La cella `coldw` chiude l'obiezione simmetrica: con i pesi a
106 MB, fuori dalla L2, il vantaggio passa da 20,3x a **18,7x** — non è un
artefatto di cache.

**Stessa storia sull'attenzione, e stessa sorpresa.** Il legacy misura 12,30 ms
per chunk a ctx 6333 (predetto 8–20, punto 12: centrato). La softmax in
streaming + vec4 rende **6,76x** — ma le due varianti che tagliano il traffico
KV di 4x e 8x (fusione GQA, righe fuse) sono **più lente**, perché scendono a 64
e 32 workgroup. E a ctx 388 il guadagno è **5,42x**, non il «< 1,5x» predetto:
il legacy paga `scores[ctxMax]` = 25.856 B di workgroup memory **anche quando il
contesto è 388**, e quello strozza l'occupancy sempre. La riga 3 va riscritta:
streaming sì, fusione GQA no.

**IL BERSAGLIO DEI 4 s VA ESCLUSO COI NUMERI.** Picco fp32 sostenuto misurato in
WebGPU: **9,26 TFLOP/s** (GEMM densa sulla shape reale del prefill) — i 12,8
TFLOP/s richiesti sono il **138%**, non il 40%. Il conto sui tempi misurati
(blocco FFN a 8,78 TFLOP/s effettivi ⇒ 5.776 ms di matmul; 8 layer full
attention ⇒ 2.888 ms) proietta **8.665 ms**, dentro la banda P2 pre-registrata
(5.000–15.000, punto 8.000). È **10,1x** sulla baseline e **2,2x** dal bersaglio,
e non conta ancora i 24 layer DeltaNet, le norm, la RoPE e i dispatch. La riga 5
chiuderà con `decision: "excluded-by-numbers"`.

**Il tetto negoziabile non è una leva** (docket item 6 risolto): spazzando
{16.384, 24.576, 32.768, 49.152} il throughput è piatto entro **0,1–2,3%** su
tutte e tre le forme. E il **conflitto di docket item 1 non esiste**: la forma
vincente spende 4.096 B a M=16, sotto il pavimento di spec. Verificato non
dedotto — il legacy dell'attenzione, chiedendo esattamente 16.384 B, **fallisce
in creazione della pipeline** («total use of workgroup storage (25856 bytes) is
larger than the maximum allowed (16384)»), e fallisce anche a 24.576.

**Correzione di metodo, dichiarata nel memo.** La prima misura del picco fp32
dava 5,86 TFLOP/s con IQR 16,6% — sotto il bordo della banda P1 per un artefatto:
i campioni mostrano una rampa di boost, e 3 warm-up non bastano per dispatch da
decine di ms dopo un idle. Rifatta con le due shape interleavate fra loro e 4
passate (40 campioni): 8,92 p50, 9,26 in regime. Il codice pubblicato è quello
che ha prodotto il JSON pubblicato. Il banco ha guadagnato per questo un
parametro `passes` su `measureInterleaved`.

**Codice**: nessun file di `src/engine/**` toccato. `kdRunner.ts` esporta la sua
meccanica di misura e guadagna la terza dimensione di griglia (la forma attuale
mette le M righe del chunk su `wid.z`); il resto è nuovo sotto
`src/microbench/tt*.ts`. La spazzata dei limiti vive nel driver: chiede device
distinti con `requiredLimits` espliciti, e il punto unico di `src/` negozia
sempre ≥ 30.848 B.

**Artefatti**: `results/microbench/ttft-riga1-4090-linux-2026-08-13T13-06-23-120Z.json`
(34 celle, **0 skipped**, `timestamp-query` su tutte, checksum entro 2·10⁻⁸ dalla
forma attuale contro una tolleranza di 10⁻³), memo
`docs/deep-dive/ttft-riga1-memo-2026-08-13.md`.

**Gate**: ktest 100 PASS / 0 FAIL · vitest 545 passed | 10 skipped ·
`tsc --noEmit` pulito. Zero run di modello, come da vincolo.

## it.3 (2026-08-13) — i tre ruling del PI, e il contratto riscritto attorno a «il più in basso possibile»

Il PI, dopo aver letto l'esito della riga 1: (1) via il bersaglio dei 4 secondi,
«scendiamo il più possibile»; (2) va bene non sfruttare tutti i 49.152 B, «il 4B
evidentemente non li sfrutta, ma modelli più grossi lo farebbero»; (3)
`packed_4x8_integer_dot_product` autorizzato.

**Il problema di forma, e come l'ho risolto.** «Il più in basso possibile» non è
graduabile da un verificatore: un goal senza soglia si chiude quando qualcuno si
stanca. L'ho reso meccanico in tre clausole invece che in una soglia:

- **TTFT a caldo < 21.905 ms**, un quarto della baseline. Non è il bersaglio: è
  la barra che dice «c'è stato un salto d'ordine di grandezza». Sta comodamente
  sopra il pavimento proiettato (8.665 ms), quindi fallirla significherebbe che
  qualcosa è andato storto, non che la macchina non ce la fa.
- **ESAURIMENTO DELLE LEVE**: ogni leva nominata dalla riga 1 in produzione e
  misurata prima/dopo, oppure esclusa coi numeri. Una leva lasciata cadere in
  silenzio è un done-when mancato. È questa la clausola che sostituisce davvero
  il bersaglio.
- **CONTABILITÀ DEL TETTO RESIDUO** per segmento, coi workgroup in volo per
  dispatch. È ciò che rende «il più in basso possibile» un'affermazione
  verificabile invece che una resa.

Conseguenza sulla riga 5: non sceglie più fra «raggiunto» ed «escluso»,
contabilizza. E il ramo `excluded-by-numbers` non serve più — l'esclusione era
relativa a una soglia che non c'è più.

**Sul dot product intero ho verificato una cosa prima di lasciarla al prossimo
ciclo: NON è un drop-in** (docket item 11, esteso). `dot4I8Packed` vuole interi
su ENTRAMBI i lati, mentre oggi dequantizziamo i nibble a f32 e moltiplichiamo
contro attivazioni f32. La via intera è quella q4_0 × q8_0 di llama.cpp e
richiede di quantizzare le ATTIVAZIONI a i8 per blocco, accumulare in i32, e
applicare `sc_w · sc_x` una volta per blocco. Il risultato non è bit-identico:
cambia l'aritmetica, non l'ordine delle somme, e la tolleranza va dichiarata
prima di misurare. Due termini di segno opposto da misurare separati: sparisce
il dequant dal ciclo interno, ma si aggiunge una passata sulle attivazioni.

**Stato**: righe 0 e 1 chiuse e verificate. La riga 2 apre con due celle di fase
0 residua — `splitk-coldw` (la vincitrice a pesi freddi, che discrimina se il
suo vantaggio è traffico o occupancy: mai girata, docket item 12c) e
`splitk-idot`. Nessuna riscrittura del motore prima di quelle due misure.

## it.4 (2026-08-13, riga 2 cella a0) — P7 CONFERMATA: il 43× non era la cache

La cella che la riga 1 aveva pre-registrato come discriminante della CAUSA e poi
girato su tutte le forme tranne quella che ha vinto (docket item 12c). P7
scritta e committata PRIMA di misurare: `splitk` a pesi freddi ≤ 0,0700 ms.

| | caldo | freddo (8 copie, 106 MiB) | degrado |
|---|---|---|---|
| `splitk` | 0,0608 ms | **0,0687 ms** | **+12,9%** |
| forma attuale | 2,6230 | 2,6083 | −0,6% |

**P7 CONFERMATA** (0,0687 ≤ 0,0700), soglia d'allarme dei 2× non sfiorata. Il
vantaggio della vincitrice **non è L2-residenza del banco**: è occupancy, e
sopravvive quando i pesi streammano davvero. Coerente col precedente misurato
sulla stessa sessione (`regs`: +8,2%) e con la causa scritta nel docket item 10.

**Il numero che conta per il motore è 38,0×, non 43,1×** — il rapporto va preso
fra le due celle FREDDE (2,6083 / 0,0687), perché nel motore i pesi sono 2,25 GB
e streammano dalla VRAM. La forma attuale è indifferente alla cache (rilegge
tutto M volte: è già limitata da altro), quindi tutto il degrado sta dalla parte
della vincitrice. Riproducibilità: `splitk` caldo 0,0609 in it.2, 0,0608 qui.

**Proiezione aggiornata coi tempi FREDDI**: il blocco FFN per layer passa da
0,2581 a ~0,2913 ms [gate e up misurati freddi, il down scalato del +12,9%
perché la sua cella fredda non è stata girata], quindi i 5.776 ms di
moltiplicazioni diventano ~6.521 e la somma col termine di attenzione ~**9.409
ms** invece di 8.665. Non cambia nessuna conclusione: resta un pavimento e resta
sotto la barra dei 21.905.

**UN'OSSERVAZIONE CHE INDEBOLISCE P6, e la registro perché va nell'altro verso
rispetto a ciò che fa comodo.** Rigirando lo sweep sui tetti concessi, lo spread
è **7,6% su `regs` e 8,6% su `splitk`**, non «entro ±5%» come la riga 1 aveva
concluso confermando P6. Solo `shared` sta a 1,1%. E il verso è che con PIÙ
memoria concessa il kernel va lievemente più VELOCE (splitk 0,0930 a 16.384
contro 0,0870 a 32.768), non più lento come la tesi dell'occupancy suggerirebbe.
Non cambia il ruling del PI — non sfruttare i 49K resta accettabile, e i numeri
dello sweep (0,087-0,093) vivono in un contesto di misura diverso da quello
della cella principale (0,0608) — ma **«piatto entro ±5%» è una conclusione più
debole di come la riga 1 l'ha scritta**, e chi la citerà deve saperlo.

**Cosa NON ho fatto in questo ciclo**: la cella `splitk-idot`. Richiede un
binding nuovo (attivazioni quantizzate a i8 impacchettate + le loro scale per
blocco) e una passata di quantizzazione, cioè più codice del semplice riuso di
una variante esistente. Il disegno è scritto per intero nel docket item 11 e la
tolleranza va dichiarata prima di misurare: non è bit-identica per costruzione.

**Gate**: ktest 100 PASS / 0 FAIL · vitest 545 passed | 10 skipped · tsc pulito.

## it.5 (2026-08-13, riga 2 cella a0) — P8 e P9 CONFERMATE: la via intera vince 1,83×

Enunciati committati PRIMA di scrivere il kernel (addendum it.5 della
pre-registrazione). `splitk-idot`: via q4_0 × q8_0 — attivazioni quantizzate a
i8 per blocco da 32, `dot4I8Packed`, accumulo i32, scala una volta per blocco.

| M=16, K2560×N9216 | p50 | vs `splitk` | vs forma attuale |
|---|---|---|---|
| forma attuale | 2,6643 ms | — | 1× |
| `splitk` | 0,0609 ms | 1× | 43,7× |
| **`splitk-idot`** | **0,0333 ms** | **1,83×** | **80,0×** |

**P9 CONFERMATA** (predetto < 0,0608; misurato 0,0333). **P8 CONFERMATA**:
`checksumRelDiff` 2,97e-3, dentro la banda ≤ 2,0e-2 ricavata dal conto
sull'errore di quantizzazione — e sopra 1e-3, che è il segnale che il kernel sta
davvero quantizzando invece di fare finta.

**La mia stessa nota di rischio è refutata, ed è la parte informativa.** Avevo
scritto: «se il kernel è davvero latency-bound sui workgroup in volo, P9 cade,
e sarebbe la terza conferma della stessa causa». Non è caduta. Quindi il collo
non era *solo* occupancy: dopo lo split-K c'era ancora testa di calcolo, e il
lavoro aritmetico per elemento era materiale. Il modello del docket item 10 va
raffinato, non buttato — l'occupancy decide *fra le forme*, il costo aritmetico
decide *dentro* la forma vincente.

**Proiezione aggiornata**: blocco FFN da 0,2579 a **0,1433 ms** (1,80×), quindi
le moltiplicazioni scendono da 5.776 a **3.209 ms** e la somma col termine di
attenzione fa **~6.097 ms** invece di 8.665. **L'attenzione è ora il 47% del
totale**: dopo la riga 2 il peso si sposta sulla riga 3.

**Fuori misura, dichiarato**: il costo della passata che quantizza le
attivazioni. Nel banco arrivano già quantizzate. Su M=16 sono ~0,16 MB contro
13,3 MB di pesi, quindi il termine dovrebbe essere piccolo — e «dovrebbe» in
questa riga ha già sbagliato due volte. Va misurato prima della produzione.

**Un errore mio, di dieci minuti di GPU**: ho scritto `enable
packed_4x8_integer_dot_product;` in testa al kernel. `dot4I8Packed` è una
LANGUAGE FEATURE, non un'estensione da abilitare: la compilazione è fallita con
«expected extension» e tutte e cinque le celle sono uscite `skipped`. La sonda
già in albero (`kdGemv.ts:168`) lo usa nudo e me lo diceva: non l'ho copiata.

**Due difetti trovati eseguendo, entrambi a docket**: una seconda incarnazione
di questa sessione girava comandi in parallelo, occupando GPU e profilo Chrome e
riscrivendo i sorgenti sotto di me (item 13); e ktest che crasha riporta 0 PASS e
0 FAIL, che un gate scritto su `grep -c FAIL` legge come verde (item 14).

**Gate** (rieseguiti col comando giusto, exit code guardato): ktest exit 0, 100
PASS / 0 FAIL · vitest 545 passed | 10 skipped · `tsc --noEmit` pulito.

## it.5 (2026-08-13, riga 2 cella a0) — la via intera vince, e due difetti del banco trovati per strada

**P8 CONFERMATA.** `checksumRelDiff` = 2,97e-03 (gate/up) e 2,47e-03 (down),
contro la tolleranza pre-registrata di 2e-2 e vicinissimi all'ordine 4e-3 che il
conto sull'errore di quantizzazione prevedeva. Zero celle saltate.

**P9: la failCondition non scatta, ma la banda è superata** — stessa forma di P0
in it.2. Predetto 1,15-1,8× (punto 1,35), fallisce sotto 1,15. Misurato:

| shape | `splitk` | `splitk-idot` | |
|---|---|---|---|
| gate/up K2560×N9216 | 0,0609 ms | **0,0333** | **1,83×** |
| down K9216×N2560 | 0,1362 | **0,0767** | **1,78×** |

E la memoria di gruppo **scende 4×** (4.096 → 1.152 B a M=16): gli i8 in shared
occupano un quarto degli f32. La forma vincente diventa ancora più portabile.

**22,69 TFLOP/s** di aritmetica utile a M=16 — contro gli 8,92 della sonda del
picco su GEMM densa fp32. Conferma quanto già notato in it.2: **quella sonda è
un pavimento del raggiungibile, non il tetto della macchina.**

**Proiezione**: il blocco FFN per layer passa da 0,2579 a **0,1432 ms** (1,80×),
quindi i 5.776 ms di moltiplicazioni scendono a ~3.208 e la somma col termine di
attenzione a **~6.096 ms** (era 8.665 con `splitk`, 9.409 coi tempi freddi).

**IL LIMITE DI QUESTO NUMERO, ed è quello che la pre-registrazione chiedeva di
misurare: `dispatchesPerOp = 2` per ENTRAMBE le forme.** La cella `splitk-idot`
misura [gemm, combine] con le attivazioni già quantizzate fuori dal ciclo: **la
passata di quantizzazione NON è dentro la misura**. Il prereg diceva testualmente
«il costo della quantizzazione va misurato SEPARATO e pubblicato, non dedotto», e
non è stato fatto. L'1,83× è quindi un limite SUPERIORE. Stima d'ordine, non
misura: 164 KB letti e ~41 KB scritti per chunk contro 13,3 MB di pesi, più
l'overhead di lancio di un dispatch — probabilmente 5-15%, che lascerebbe
~1,6×. **Va misurato prima di portare la forma in produzione.**

### Due difetti del banco, trovati mentre verificavo

**(1) Il kernel `splitk-idot` era già in albero e non compilava.** Scritto in un
ciclo precedente, emetteva `enable packed_4x8_integer_dot_product;` — ma quella è
una LANGUAGE FEATURE del WGSL (`navigator.gpu.wgslLanguageFeatures`), non
un'estensione da abilitare: Tint risponde «expected extension» e **tutte e
cinque le celle erano finite in `skipped`**. Il fix era già in albero anche lui,
con il commento che documenta la causa; l'artefatto delle 17:42 è la run rotta,
precedente al fix, e non va usato. Io ci ho appeso sopra un duplicato del kernel
per non aver letto il file prima di scrivere: rimosso.

**(2) Il gate del checksum era fisso a 1e-3 per tutte le varianti**, quindi
avrebbe scartato `splitk-idot` anche dopo la compilazione. Reso **per variante**:
1e-3 a chi cambia solo l'ordine delle somme, 2e-2 a chi cambia l'aritmetica, col
valore preso dalla pre-registrazione e non da ciò che è uscito. Resta un gate.

### E un difetto del GATE, che è la cosa più grave della giornata

**`.harness/tools/engine-ktest.mjs` punta a `localhost:5173` e ESCE CON 0 anche
quando non ha girato affatto.** Con il server su 5199 il runner è morto su
`ERR_CONNECTION_REFUSED` e ha restituito **exit 0**: un gate che dichiara
successo senza aver eseguito. Ci sono cascata dentro in questo ciclo — l'ho preso
solo perché il conteggio dava `0 PASS / 0 FAIL`, che non è un esito plausibile.
È la stessa classe della sentinella sui JSON di `results/`, e va chiusa allo
stesso modo (docket).

**Gate, eseguiti sul BASE_URL giusto e con `STATUS: done` verificato**:
ktest **100 PASS / 0 FAIL** · vitest **545 passed | 10 skipped** · tsc pulito.

## it.6 (2026-08-13) — il gate che non poteva fallire ora fallisce, e il rapporto della via intera è misurato

**Chiuso il difetto del gate (docket item 14).** `engine-ktest.mjs` usciva 0/1
solo sullo `status` della pagina: ogni errore PRIMA — profilo occupato, porta
sbagliata, timeout — era un'eccezione nuda, nessun test eseguito, tabella vuota,
e `grep -c FAIL` restituiva **0**, che è esattamente ciò che si vede quando va
tutto bene. In it.5 l'ho dichiarato verde due volte su una run che non aveva
eseguito un solo kernel.

Ora ha tre difese: ogni fallimento è catturato e stampato come causa con exit
non-zero; il driver **conta lui** i PASS/FAIL invece di lasciarli dedurre a un
grep; e c'è una **asserzione di plausibilità** (`KTEST_MIN_PASS`, default 90) —
zero fallimenti su zero test non è un gate superato.

**Verificato in negativo, che è l'unico modo di sapere che un gate esiste**:
porta sbagliata → exit 2 col messaggio che nomina `BASE_URL`; soglia impossibile
→ exit 4 con «ZERO FALLIMENTI SU ZERO TEST NON E' UN GATE SUPERATO»; run buona →
exit 0, 100 PASS. Prima di questo, la prova negativa non era mai stata fatta.

**Chiuso il «fuori misura» di it.5 (docket item 12c per la via intera).** Due
celle nuove: `quantx-q8` (la sola passata di quantizzazione) e
`splitk-idot-full` (quantizzazione + moltiplicatore + combinazione nello stesso
campione).

| M=16, K2560×N9216 | p50 |
|---|---|
| `splitk` (f32) | 0,0609 ms |
| `quantx-q8` | 0,0019 ms |
| `splitk-idot` (solo kernel) | 0,0333 ms |
| **`splitk-idot-full` (la LEVA)** | **0,0349 ms** |

**Il rapporto onesto è 1,74×, non 1,83×.** La quantizzazione costa il 5,3% del
totale. Avevo scritto in it.5 che «dovrebbe essere piccolo»: lo è — ma adesso è
nell'artefatto invece che nel mio giudizio, ed è la differenza fra le due cose
che conta, non il numero.

**Proiezione aggiornata**: blocco FFN 0,2578 → 0,1482 ms (1,74×), matmul 3.320
ms, totale con l'attenzione **~6.208 ms** (era 8.665 con `splitk` in virgola
mobile, 6.097 col kernel intero contato senza la sua quantizzazione).

**Due difetti di misura trovati facendo, e sistemati**: il gate di checksum a
1e-3 bocciava `splitk-idot-full`, che fa la STESSA aritmetica di `splitk-idot` e
ha la stessa tolleranza pre-registrata in P8 (2e-2); e `quantx-q8` non scrive
`y`, quindi il suo checksum era quello lasciato dalla variante precedente e non
diceva niente — ora è esente, con la ragione nel JSON invece che con un numero
finto.

**Da tenere presente leggendo l'artefatto**: le colonne `tflops` e `tokensPerSecond`
della cella `quantx-q8` sono prive di senso (406 TFLOP/s a M=16): la formula
assume una GEMM e quella cella non lo è. Il numero da guardare è solo `p50`.

**La contesa dell'item 13 è costata due run.** Due tentativi morti su «profilo
già in uso» perché l'altra incarnazione girava `engine-ktest`. Ho aspettato che
finisse e poi usato un profilo dedicato — che è il rimedio, non la soluzione.

**Gate** (col ktest nuovo, exit code guardato): ktest exit 0 · 100 PASS / 0 FAIL
· vitest 545 passed | 10 skipped · `tsc --noEmit` pulito.

## it.6 (2026-08-13) — i due item chiusi: il gate che mentiva e la misura che mancava

**Item 13 — `engine-ktest.mjs`.** Tre difese, ognuna provata FACENDOLA SCATTARE
prima di fidarmene: server assente → exit 2 con la causa; profilo Chrome
occupato → exit 2 con la serialità spiegata; `KTEST_MIN_PASS=999` → exit 4 con
«zero fallimenti su zero test non è un gate superato»; run buona → exit 0 con
«PASS 100 · FAIL 0» stampato dal driver.

Il primo disegno del fix era rotto a sua volta: **timeout fisso di 120 s su una
suite che ne impiega ~5 minuti**, quindi faceva fallire una run BUONA. Un gate
che grida al lupo a run sana viene disattivato da chi lo usa, ed è il modo più
efficace di tornare al punto di partenza. `KTEST_TIMEOUT_MS` default 600.000.
**Una difesa non testata in positivo non è una difesa.** E, mentre lo provavo,
il gate ha diagnosticato da solo un Chrome rimasto appeso dal proprio timeout
precedente — invece di uscire 0. È esattamente il caso per cui esiste.

**Item 14 — il costo della quantizzazione, che it.5 aveva dedotto.** Ora è
misurato: `quantx-q8` da sola e `splitk-idot-full` = [quant, gemm, combine],
con `dispatchesPerOp` 1 e 3 contro i 2 di `splitk-idot`.

| M=16 | `splitk` | `quantx-q8` | `splitk-idot-full` | onesto |
|---|---|---|---|---|
| gate/up | 0,0609 | **0,0019** | **0,0349** | **1,745×** |
| down | 0,1361 | 0,0020 | **0,0787** | **1,729×** |

La quantizzazione costa il **5,6%** del kernel intero. La stima di it.5 («5-15%,
lascerebbe ~1,6×») era giusta come ordine e pessimista come valore: il vero è
1,745× contro l'1,83× senza. **Blocco FFN 0,2579 → 0,1485 ms (1,737×)**,
proiezione **~6.214 ms**.

**Dove siamo rispetto al contratto riscritto**: baseline 87.618 ms, barra
meccanica < 21.905, proiezione con tutte le leve ~6.214 (pavimento: non conta i
24 layer DeltaNet, norm, RoPE, dispatch). Le tre leve sono tutte MISURATE e
nessuna esclusa: `splitk` 38,0× a pesi freddi, via intera 1,74× sopra di lei,
attenzione in streaming 6,76×. Resta da portarle in produzione — la riga 2.

**Gate**: ktest 100 PASS / 0 FAIL (col driver che lo dichiara) · vitest 545
passed | 10 skipped · tsc pulito.

## it.8 (2026-08-13, riga 2) — il lavoro salvato è verde e mergiato, ma la metrica non si è mossa

L'item 15 chiedeva un ruling fra tre opzioni. **Non era un'escalation**: il passo
5 del protocollo dice che se ciò che faresti senza risposta coincide con la tua
raccomandazione, si esegue e si registra. Avrei fatto (b). Fatta (b).

**Verificato a mano sul ramo prima del merge**, non sulla parola degli agenti:
`tsc --noEmit` pulito · vitest **611 passed | 10 skipped** contro i 545 di prima
(**+66 test**) · ktest **100 PASS / 0 FAIL**, col driver che lo dichiara.
Mergiato in `main` (`0c66fbd`).

**Cosa è entrato** (3 task su 8): i kernel vincenti della fase di sonde portati
in `src/engine/kernels/wgsl.ts` (+385 righe), `PREFILL_M` da 8 a 16, la
contabilità dei byte di peso per token in `prefillbytes.ts`.

**Il conflitto di it.0 è risolto come si deve.** `PREFILL_M = 16` è la
CONVENZIONE; il path denso 0.5B ha un'**eccezione pinnata** a 8 con la ragione
NUMERICA e non storica — `rmsPairGemmSiluChunkFast` chiede `4·K·m + 256·m + 16·m`
con K=896, cioè 30.848 B a m=8 e **61.696 a m=16**, sopra il cap negoziato — e il
conto vive come aritmetica ESEGUIBILE in `tests/engine-chunking.test.ts`: se
cambia, il test lo dice. È la forma giusta per un'eccezione: non un commento, un
test.

**COSA NON È ENTRATO, ed è perché la riga 2 NON è chiusa.** `q35gpumodel.ts` —
l'assemblatore del 4B — **non è toccato**. Il prefill del 4B usa ancora
`gemvQuantWgsl` con `batch:true` su `wid.z`. **I kernel sono in produzione,
testati, e non li chiama nessuno: la metrica obiettivo è ferma a 87.618 ms.**

Restano t5 (assemblatore), t3 (gpulimits), t4 (tolleranza del gate di
conformità), t6 (copertura della convenzione), t8 (bench prima/dopo). Li faccio
io uno per volta, committando appena verificati — che è il punto dell'opzione
(b): ogni passo sopravvive alla morte del processo, che in questa sessione ha
già ucciso due build e tre server.

## it.10 (2026-08-13, riga 2) — terza morte, e la diagnosi giusta

Il terzo lancio del conductor e' morto con **un solo agente** completato. Contando
i tre: il 1° non ha prodotto niente, il 2° (ripresa dalla cache) e' arrivato
all'integrazione con 506 inserzioni e 66 test — quelli mergiati in `0c66fbd` — e
il 3°, partenza da zero con spec nuova, e' morto subito.

**La causa non e' il conductor: e' che un workflow in background non sopravvive
all'uscita del processo che lo ospita.** Stessa causa dei tre server di sviluppo
morti stasera; per quelli `setsid` ha risolto, per i workflow non esiste
l'equivalente perche' il runtime vive dentro il processo.

**E la ripresa dalla cache e' il meccanismo che fa avanzare il lavoro**: il 2°
lancio e' arrivato lontano proprio perche' replicava il 1° istantaneamente. La
strategia corretta non e' scegliere fra conductor e lavoro a mano, ma
**riprendere finche' non chiude**. Ripresa lanciata (`wp3x2as00`).

**Il PI aveva ragione e la mia decisione di it.8 era mal motivata**: avevo scelto
il lavoro a mano dicendo «cosi' ogni passo sopravvive al processo», ma il lavoro
del conductor era gia' sopravvissuto — le patch erano nell'albero e sono risultate
verdi. Cio' che si perde a ogni morte e' la VERIFICA, che tocca a me comunque
perche' e' seriale su GPU. Avevo pesato un costo gia' recuperato. Registrato a
docket item 16.

**Stato invariato della metrica**: TTFT a caldo fermo a **87.618 ms**. I kernel
sono in produzione e testati; `q35gpumodel.ts` non e' ancora cablato.

## it.11 (2026-08-13, riga 2) — quarta morte: la ripresa avanza ma non chiude, quindi accorcio

La strategia di it.10 era «riprendere finche' non chiude». Eseguita, e **ha
avanzato**: il journal e' passato da 1 a 14 voci (9 transcript, 5 risultati).
Ma e' morta di nuovo prima dell'integrazione, senza lasciare niente nell'albero.

**Il vincolo binding e' la DURATA, non il piano** — e riprovare della stessa
lunghezza sarebbe il loop-bug che il protocollo vieta. Ho cambiato una variabile
sola: la dimensione. Quinto lancio con **UN SOLO TASK** (cablare
`q35gpumodel.ts`), `suiteCmd` ridotto a `tsc --noEmit`, e un fallback scritto
nella spec — se la via intera e' troppo grossa, fare la sola forma split-K in
virgola mobile e annotare il resto.

**Dichiarato PRIMA di vedere l'esito**, per non razionalizzare dopo: se anche
questa muore, non ci sara' un sesto tentativo. La conclusione sara' che il
veicolo non chiude un task di questa classe nella vita di un processo, e la riga
2 si fa a mano accettando che sia piu' lenta.

**Metrica invariata**: TTFT a caldo **87.618 ms**. Kernel in produzione e
testati, `q35gpumodel.ts` non ancora cablato.

## it.12 (2026-08-13) — l'ipotesi della durata e' refutata; il piano esatto per il cablaggio a mano

Quinto lancio, un task solo: **3 voci di journal contro le 14** del lancio lungo
ripreso. **Il lavoro piu' corto e' morto prima.** L'ipotesi di it.11 e' refutata.

La lettura giusta: il processo muore poco dopo la fine del turno, e la
dimensione del job non c'entra — il lancio lungo era andato piu' avanti solo
perche' la cache gli bruciava i primi agenti in millisecondi. **Tutto cio' che
deve sopravvivere al confine del turno e' inaffidabile qui (5 workflow, 3
server); tutto cio' che eseguo dentro il turno regge (6 misure GPU, tutti i
gate, i merge).** Il lavoro va fatto sincrono. Docket item 18.

### Il piano del cablaggio, scritto ora che ho letto il codice

I kernel portati NON sono nudi: hanno gia' i loro helper in
`src/engine/kernels/wgsl.ts`, e questo riduce il lavoro a plumbing.

- `prefillGemmQ4SplitKWgsl(o)` — forma split-K in virgola mobile
- `prefillGemmQ4SplitKIdotWgsl(o)` — via intera q4_0 x q8_0
- `prefillQuantXQ8Wgsl({K, M})` — quantizzazione delle attivazioni
- `prefillSplitKCombineWgsl({N, M, splits})` — somma dei parziali
- `prefillGemmGrid(o)` — la griglia, gia' calcolata (niente terzo posto che
  decide le righe-per-workgroup: e' la lezione di it.7 del goal precedente)
- `prefillPartialFloats(o)` — quanti float serve il buffer dei parziali
- scelta degli splits a riga ~3965: `PREFILL_SPLITS_MEASURED = 4` se i blocchi
  sono divisibili, altrimenti `PREFILL_SPLITS_UNSPLIT = 1`
- `prefillGemmCheck` LANCIA se `kind != "q4_0"` o se `K % 64 != 0`: la via
  veloce e' q4_0-only per costruzione, come il GEMV del goal precedente

**Il sito da cambiare e' `gemvB` in `q35gpumodel.ts` (~riga 709-720)**, che oggi
fa `pushB(gemvQuantWgsl({...batch: true}), [qs, scales, src, dst], [gx, gy, M_MAX])`.

**Cosa serve in piu' rispetto a un dispatch solo:**
1. un buffer dei PARZIALI, dimensionato da `prefillPartialFloats` sul piu' grande
   (K, N) del modello, allocato UNA VOLTA al load come gli altri;
2. per la via intera, i buffer delle attivazioni quantizzate (`xq` u32
   impacchettati + `xsc` scale per blocco) e il dispatch di quantizzazione
   PRIMA del gemm;
3. `gemvB` diventa 2 dispatch (gemm + combine) o 3 (con quantizzazione), quindi
   il piano batch deve accettare piu' step per chiamata.

**Ordine consigliato**: prima la sola forma split-K in VIRGOLA MOBILE (2
dispatch, nessun buffer di quantizzazione, bit-identita' del gate di conformita'
INTATTA), misurare, committare. Poi la via intera come secondo passo, con la
tolleranza 2e-2 gia' pre-registrata. Cosi' il primo movimento della metrica
arriva senza toccare il gate di correttezza, e la via intera si misura contro un
riferimento gia' aggiornato invece che contro la baseline vecchia.

**Metrica invariata**: TTFT a caldo **87.618 ms**.

## it.13 (2026-08-13, riga 2) — il cablaggio E' SCRITTO, il gate GPU NO

Primo lavoro sincrono dopo la decisione di it.12. `gemvB` in `q35gpumodel.ts`
instrada sulla forma split-K quando `kind` e' q4_0 e K e' multiplo di 64 —
condizioni VERIFICATE nel codice, non assunte — con griglia da
`prefillGemmGrid` e splits da `prefillGemmSplitsFor`. Fuori da quelle, resta la
forma di prima, corretta e solo lenta.

Aggiunto il buffer dei parziali, allocato al load e dimensionato sul **massimo
di tutte le `n` che `gemvB` puo' ricevere**, non su quella che capita per prima:
un buffer corto darebbe scritture fuori range su una shape piu' grande
incontrata dopo, e **la validazione WebGPU non la vedrebbe**, perche' il binding
e' l'intero buffer. E' il tipo di errore che produce numeri plausibili.

**Verificato**: `tsc --noEmit` pulito · vitest **645 passed | 10 skipped**
(erano 611 su main: i test della build coprono il path nuovo).

**NON verificato, e per questo il codice sta su `wip/riga2-cablaggio-splitk` e
NON su main**: il ktest. Una run e' andata in **TIMEOUT a 600 s con «pagina non
leggibile»**, poi tre tentativi sono stati uccisi dall'ambiente con exit 144 —
lo stesso segnale che stasera ha ucciso cinque workflow e tre server.

**Non so attribuire quel timeout**, e non voglio indovinare: puo' essere
l'infrastruttura che uccide anche i comandi lunghi in primo piano, oppure questo
kernel che pianta la GPU. Sono due cause con rimedi opposti, e dichiararne una
senza prove sarebbe esattamente l'errore che questo progetto si e' gia' fatto
costare piu' volte. **La prossima iterazione deve discriminare**: rieseguire il
ktest su `main` (senza il cablaggio) e sul ramo, e confrontare. Se `main` va in
timeout uguale, e' infrastruttura; se solo il ramo, e' il kernel.

**Metrica invariata**: TTFT a caldo **87.618 ms**. Il cablaggio esiste ma non e'
misurato, e finche' non passa il gate non conta.

## it.14 (2026-08-13) — ruling sul riuso applicato, e il riavvio è l'esperimento discriminante

**Ruling del PI**: barra del riuso da ≥ 8× a **≥ 5,5× sull'inventario per-layer
INTERO**, residuo assegnato al goal K-quant. Applicato in `GOAL.md` (done-when
della leva 1) e `PHASES.md` (riga 2, clausola b), entrambi col conto per esteso.
Docket item 19 CHIUSO.

**L'errore era mio e vale la pena ricordare da dove veniva**: il banco misurava
UNA shape, dove la copertura q4_0 è totale e il rapporto è esattamente 16×. Ho
scritto il done-when su quel numero. Sull'inventario vero del modello la
copertura è 88,46% e il resto — 24 `ssm_out` Q5_K + 4 `ffn_down` Q4_1 — si paga
M volte anche dopo, quindi il tetto è 8,67× e il ≥ 8× richiedeva M ≥ 92. **Una
misura su una shape non autorizza un gate sull'intero modello**, ed è la stessa
classe della landmine «nessun tok/s senza il suo contesto».

**IL RIAVVIO DELLA MACCHINA È L'ESPERIMENTO CHE MI MANCAVA.** Il cablaggio su
`wip/riga2-cablaggio-non-verificato` (`8042c7b`) ha tsc pulito e vitest 645
verde, ma il ktest — che passava **100/0 prima** del cablaggio — va in timeout
con la pagina morta. Non ho potuto discriminare fra cablaggio e infrastruttura
perché nella stessa finestra anche i miei comandi venivano uccisi con 144.

Dopo il riavvio la discriminazione è di una riga:
- ktest **verde** sul ramo ⇒ la causa era l'infrastruttura, il cablaggio è sano,
  si merga e si misura il prefill;
- ktest **ancora morto** ⇒ la causa è il cablaggio, e va cercata lì. In quel caso
  la prima cosa da fare è lo streaming incrementale della tabella (docket item
  20), senza il quale non si sa nemmeno a quale kernel muore.

**Metrica invariata: TTFT a caldo 87.618 ms.** Le tre leve sono misurate
(38,0× · 1,745× · 6,76×), i kernel sono in produzione e testati, il cablaggio è
scritto e non verificato. Proiezione a leve montate ~6.214 ms; barra del
contratto < 21.905.

---

## it.14 (2026-08-14) — il riavvio ha risposto: era INFRASTRUTTURA. Il cablaggio è sano, misurato, mergiato

**L'esperimento discriminante è costato tre comandi e ha dato una risposta
secca.** Server avviato staccato su 5199, `git checkout
wip/riga2-cablaggio-non-verificato`, ktest: **100 PASS · 0 FAIL · exit 0**. Lo
stesso ramo, lo stesso comando, la stessa GPU che ieri dava «pagina non
leggibile». La causa era il reaping al confine di turno, non il cablaggio — e
l'ipotesi «il conductor non chiude un task di questa classe qui» va riletta alla
sua luce: 5 morti su 5 erano lo stesso segnale 144, non un limite del veicolo.

**La misura, comando identico alla baseline di it.1** (`--prompt-idx 0
--n-decode 64 --vram-gib 8 --declared quiescent --prefill-m 16`):

| | baseline it.1 | it.14 | rapporto |
|---|---|---|---|
| prefill, braccio chunked M=16 | 34,36 tok/s | **111,16 tok/s** | **3,24×** |
| prefill, path migliore di allora (seq) | 72,30 tok/s | 111,16 tok/s | 1,54× |
| `prefill.ms` | 87.582 (seq) | **56.961** | |
| TTFT a caldo | 87.618 ms | **56.984 ms** | **1,54×** |
| decode a ctx 6333 | 47,79 tok/s | 47,17 tok/s | −1,30%, dentro banda |

Artefatto `results/engine/ttft-riga2-4b-splitk-m16-prompt0-2026-08-14.json`.
Mergiato in main (`7bc6b55`). **Barra del contratto 21.905 ms: manca 2,60×.**

**IL BUCO CHE HO TROVATO GUARDANDO IL CALL-SITE, e che è la cosa che conta di
questa iterazione.** Il done-when (a0) della riga 2 dice, alla lettera: «Se
`idot` vince, è LEI la forma che va in produzione». `idot` ha vinto — 1,745×
sopra la f32, misurato in it.6. Ma `q35gpumodel.ts:746` chiama
`prefillGemmQ4SplitKWgsl`, cioè la via **f32**, che il commento del kernel stesso
(`wgsl.ts:3795`) dichiara «FALLBACK DICHIARATO ... si porta per i device senza
`packed_4x8_integer_dot_product`, **non come alternativa preferibile**».
`prefillGemmQ4SplitKIdotWgsl` esiste (`wgsl.ts:3718`), è portata riga-per-riga
dal banco, ha il suo test di divergenza testuale — e **nessun call-site la
raggiunge**. `gpulimits.ts` conosce già il flag (`prefillGemmIdot`), quindi il
punto di innesto c'è: manca la scelta a runtime sulla language feature e il
ramo che la usa.

Il cablaggio non è quindi «la riga 2 fatta al 90%»: è la riga 2 fatta **con la
forma perdente**. È anche parte della spiegazione del divario fra i 43,1× del
banco sul kernel e gli 1,54× end-to-end.

**Contabilità del residuo, perché è il done-when della riga 5.** A 56.984 ms
mancano 2,60× per la barra. Leve misurate e NON ancora in produzione:
`idot` **1,745×** sul segmento moltiplicazioni · attenzione del prefill in
streaming **6,76×** sul segmento attenzione (riga 3, `attnDecodeWgsl` con
`batch` ancora sul legacy) · l'11,54% dei byte fuori dal percorso q4_0, che è
scope del goal K-quant. Il divario 43,1× → 3,24× dice che dopo il cablaggio il
moltiplicatore non è più il termine dominante: **l'attenzione lo è**, ed è la
riga 3.

## it.15 (2026-08-14) — la via intera cablata: 1,70× totale, e la causa del buco

Il riavvio ha risolto: `ktest` **100 PASS / 0 FAIL** sullo stesso codice che ieri
faceva morire la pagina. Era infrastruttura, non cablaggio.

**Misura indipendente che corrobora**: it.14 (altra iterazione) ha misurato TTFT
a caldo 56.984 ms, io 57.485 sulla stessa forma. **0,9% di scarto su due run
separate**, dentro il rumore di ~2,4% di questa macchina.

| | prefill | TTFT a caldo | decode |
|---|---|---|---|
| baseline sequenziale | 72,30 tok/s | 87.618 ms | 47,79 |
| split-K f32 | 110,19 | 57.485 | 49,59 |
| **via INTERA** | **123,26** | **51.392** | 48,00 |

**Totale 1,70× sulla baseline.** Barra del contratto < 21.905: manca 2,35×.

**LA CAUSA DEL BUCO VALE PIÙ DEL GUADAGNO.** `prefillgemmplan.ts` esisteva già —
completo, con `planPrefillGemm`, `prefillGemmCapsFor`, `prefillGemmScratchFor` e
i suoi test — e `q35gpumodel.ts` **non lo importava: zero riferimenti**. Il sito
ri-derivava la condizione a mano (`if (kk === "q4_0" && w.k % 64 === 0)`) e
finiva sempre sulla via f32, che il kernel stesso documenta come fallback.

È la stessa forma del difetto trovato in it.7 del goal precedente — «un terzo
posto che decide le righe-per-workgroup» — e il commento che la nomina era
**tre righe sotto** il codice che la ripeteva. Un piano non collegato non è un
piano: è documentazione.

**Il guadagno end-to-end della sola via intera è 1,119×, contro l'1,745× del
micro-bench.** Non è una delusione ed era prevedibile: i moltiplicatori sono
solo una parte del prefill, e l'attenzione — 6,76× misurata — non è ancora
cablata. È la riga 3.

**Gate**: tsc pulito · vitest 645 passed | 10 skipped · ktest 100 PASS / 0 FAIL ·
non-reg decode 48,00 ≥ 45,53 · guardia prefill > decode 123,3 > 48,0.

---

## it.16 (2026-08-14) — un agente fantasma nel mio working tree, e la miccia era un file in git

**Non è l'iterazione che avevo in programma.** Stavo per lanciare `sdd-conductor`
sulla riga 2 quando ho letto `q35gpumodel.ts` e ci ho trovato il cablaggio
`idot` — completo, con un commento che citava «(it.14)», cioè la MIA diagnosi.
Non l'avevo scritto io. `git status`: file modificato, non committato. Un JSON
di misura untracked, scritto 24 secondi prima.

**`ListAgents`: una sessione peer, `browser-llm-lab-89`, avviata 13 minuti
prima, viva, che scriveva nel mio stesso albero e usava la mia stessa GPU.**

Mi sono fermato prima di scrivere qualunque cosa e le ho mandato un messaggio
chiedendo chi fosse, cosa possedesse e chi dovesse continuare. Ha risposto in
modo esemplare: loop rianimato dal watchdog dalla sessione `0348f2c0`, che si
era fermata DELIBERATAMENTE con `ScheduleWakeup{stop:true}` prima del riavvio.
Ha committato e pushato tutto, si è tolta di mezzo, e ha lasciato un passaggio
di consegne con la trappola GQA e due note operative sui profili Chrome.

**LA CAUSA, ricostruita dal reflog e dal log del watchdog, e combacia al
secondo:**

    00:00:51  io: git checkout wip/riga2-cablaggio-non-verificato
              — il PRIMO comando che il mio stesso HANDOFF prescriveva
    quel ramo porta .harness/loop-state.json con status "active",
    next_wake_epoch 1786655564 = 2026-08-13T23:12:44
    00:01:34  il watchdog polla (timer ogni 2 min), legge il file che il mio
              checkout aveva appena riscritto, vede un risveglio scaduto da
              48 minuti, e rianima la sessione 0348f2c0

Il log dice «scaduto da 48 min»: 00:01:34 − 48 = 23:13, cioè il `next_wake` del
RAMO — non quello di main (23:19:17), non lo `stopped` di main. Non è
un'inferenza: è lo stesso numero.

**Il watchdog non legge male: legge `active` perché glielo ha appena scritto
git.** Il difetto è che un file di RUNTIME stesse sotto controllo di versione.
Corretto (`c21648e`): `.gitignore` + `git rm --cached`.

**Il residuo è più grande dell'incidente: 19 rami portano ancora
`status: "active"`** — tutti i `wip/riga2-*` e i `worktree-wf_*`. Toglierlo da
main impedisce a main di ri-armare la miccia, NON impedisce a un checkout di
quei rami di riarmarla. La difesa robusta sta in
`~/Projects/harness/tools/loop-watchdog.sh` — altro repo, non toccato, riportato
al PI. Due difetti da correggere lì: rifiutare uno stato scaduto da troppi poll
(48 minuti su un timer da 2 significa ~24 poll mancati: è staleness, non
ritardo), e non saltare il controllo «transcript fresco ⇒ sta lavorando» sul
ramo `headless_dead`, che è il motivo per cui è stata rianimata dentro una
directory in cui qualcuno stava attivamente scrivendo.

**Verifica indipendente del lavoro del peer** (non ho preso per buono il suo
riassunto — è la regola che mi sono dato): `HEAD == origin/main == 719b229`,
albero pulito, i tre commit ci sono. Poi i gate rifatti da me: **tsc pulito ·
vitest 645 passed | 10 skipped · ktest 100 PASS / 0 FAIL**. Regge.

**Stato della metrica, che nel frattempo si è mossa davvero:**

| | prefill | TTFT a caldo |
|---|---|---|
| baseline (it.1) | 72,30 tok/s | 87.618 ms |
| split-K f32 (it.14, mia) | 111,16 | 56.984 |
| **via intera (it.15, del peer)** | **123,26** | **51.392** |

**1,70× sulla baseline. Barra 21.905 ⇒ manca 2,35×.** Le due misure indipendenti
della forma f32 (56.984 e 57.485) distano lo 0,9%, dentro il rumore ~2,4% di
questa macchina: si confermano a vicenda.

**Riga 3 lanciata via `sdd-conductor`** (`wf_991df002-d1d`), che è il veicolo
dichiarato per quella riga. È anche il test della conclusione sospesa: se regge
dopo il riavvio, l'item 15/17/18 («il veicolo non chiude un task di questa
classe qui») era una diagnosi sbagliata di una causa infrastrutturale.

**Controllo di fattibilità prima di spendere ⇒ docket item 21.** La clausola (a)
della riga 3 chiede tre cose, e la terza — «KV letta una volta per gruppo GQA» —
**è la fusione GQA, che la riga 1 ha misurato più LENTA sul prefill** (2,0879
contro 1,8207 ms). Il 6,76× non ne ha bisogno: l'artefatto lo attribuisce alla
sola softmax in streaming + vec4. Decisione presa e registrata, non escalata
(test dello step 5: se non arrivasse mai risposta farei esattamente questo).
Il vincolo negativo è scritto nella spec del conductor, col numero, e il review
ha istruzione di bocciare qualunque task che proponga la fusione.

---

## it.17 (2026-08-14) — la riga 3 rende 1,59×, e il gate di conformità era rotto da due iterazioni senza che nessuno lo sapesse

**Il conductor è arrivato quasi in fondo, e il veicolo REGGE.** `wf_991df002-d1d`
ha prodotto il grafo, implementato T1 e T2, chiuso due round di review avversaria
e integrato T1 con `git apply` pulito (nessun fuzz, nessun `--3way`). Poi il mio
processo è uscito e il workflow è morto con lui — ma **non al confine di turno e
non per la causa di ieri**: aveva già fatto il lavoro. La conclusione degli item
15/17/18 («il veicolo non chiude un task di questa classe qui») è **da
considerare confutata**: dopo il riavvio ha chiuso.

**T2 NON è entrato**, per due bloccanti indipendenti che l'integrator ha
diagnosticato bene: (1) la patch è arrivata monca di una riga di contesto —
25.658 caratteri contro i 25.664 dichiarati, cioè lo scarto di una riga sola:
difetto di canale, non di piano; (2) conflitto reale, perché T1 aveva **già**
convertito lo stesso blocco di `ktest.worker.ts` a tolleranza, ma con costanti
LOCALI (`ATTN_CHUNK_REL_TOL = 1e-4`) invece del modulo unico che T2 creava.
Quindi la soglia oggi è dichiarata, ma **non ha una sede sola** — che era
l'oggetto di T2.

**LA MISURA, ed è la seconda volta in una notte che la metrica si muove davvero:**

| | prefill | TTFT a caldo |
|---|---|---|
| baseline (it.1) | 72,30 tok/s | 87.618 ms |
| split-K f32 (it.14) | 111,16 | 56.984 |
| via intera (it.15) | 123,26 | 51.392 |
| **+ attenzione streaming (it.17)** | **196,41** | **32.265** |

**2,72× sulla baseline. Barra 21.905 ⇒ manca 1,47×.** Decode 47,89 tok/s, sopra
il gate di 45,5. Gate: tsc pulito · vitest 660|10 (erano 645) · ktest **101 PASS
/ 0 FAIL**, incluso il banco nuovo `dense-batch-attn-chunk-multitile` che
attraversa il rescale online (maxRel 8,44e-5 contro banda dichiarata 1e-4).

**IL RITROVAMENTO CHE VALE PIÙ DELLA MISURA — e nasce da un mio errore.**
Il review del conductor aveva segnalato che dopo T1 la prosa di PHASES sulla
bit-identità sarebbe diventata falsa. Sono andato a verificare invece di
correggerla d'ufficio, e ho scoperto due cose peggiori:

1. **`q35-prefillchunk-4b` non è un banco del ktest.** Vive in
   `q35conf.worker.ts:386` e si attiva con `--prefill-m` **senza**
   `--prompt-idx` — col `--prompt-idx` si prende il ramo bench alla riga 206 e
   si esce prima. Tutte le run di it.14-it.16 passavano `--prompt-idx 0`:
   misuravano la velocità e non toccavano mai la conformità.
2. **In it.16 avevo scritto «(e) bit per bit ✓» senza eseguirlo**, leggendolo
   dalla prosa di PHASES. Girandolo oggi: `bitIdentical: false`, bitEqual
   **31 su 15.892.480**, maxAbs 2,38.

**Non ho attribuito a naso.** Ho rigirato lo stesso gate riportando il solo
`wgsl.ts` a `HEAD~1` — attenzione legacy, via intera in produzione:

    idot + attenzione streaming   maxAbs 2,3839
    solo idot                     maxAbs 1,8001

**La rottura è di it.15**, la via intera, che quantizza le attivazioni a int8.
L'attenzione la peggiora del 32%, non la causa. Ed era prevedibile: quantizzare
le attivazioni NON può preservare i bit, e il ruling del PI che autorizza la via
intera lo implicava. La clausola (e) della riga 2 dava due strade — «resta bit
per bit **oppure** la tolleranza si dichiara PRIMA con la ragione numerica» — e
**non ne è stata percorsa nessuna**.

**Docket item 22, e non lo decido io**: il criterio di superamento di un gate di
conformità è un gate, non ordine né meccanismo. La mia raccomandazione è che il
criterio diventi l'**argmax** (stesso token, non stessi bit) con la banda
numerica accanto — oggi il gate non riporta nemmeno l'argmax. Ma non la eseguo
prima del ruling: riscrivere il criterio di un gate mentre lo si sta violando è
esattamente la mossa che un gate esiste per impedire.

**Clausola (d) della riga 2, censimento chiuso**: `gemvB` è l'unico imbuto dei
GEMM quantizzati del prefill e chiede la rotta a `planPrefillGemm`. Eccezioni
motivate: `gemvF32Wgsl` (`q35gpumodel.ts:448` e `960`, pesi f32) e i K-quant
(`468-469`), fuori dalla via veloce perché è q4_0-only per costruzione. Coerente
col 172/248 siti = 88,46% dei byte già pinnato nel test `[6c]`.

---

## it.18 (2026-08-14) — il conductor regge, la riga 3 rende 2,72×, e il loop non è più eseguibile su questa macchina

**IL NUMERO.** TTFT a caldo **87.618 → 32.265 ms = 2,72× sulla baseline**.
Barra del contratto 21.905 ⇒ **manca 1,47×**. Il prefill è passato da 72,30 a
**196,41 tok/s**. Verificato da me su questo albero, non ereditato: tsc pulito ·
vitest **671 passed | 10 skipped** (erano 645) · ktest **101 PASS / 0 FAIL**
(era 100: +1, il banco dell'attenzione in streaming con la tolleranza
dichiarata) · decode 47,89 tok/s a ctx 6333 (−0,2%, in banda).

**IL CONDUCTOR HA COMPLETATO** — 3 ondate, 16 agenti, 2h17m, zero errori di
agente. Gli item 15/16/17/18 concludevano, con crescente sicurezza, che «il
veicolo non chiude un task di questa classe qui». **Era una diagnosi sbagliata
di una causa infrastrutturale.** Le 5 morti erano il segnale 144 al confine di
turno — la stessa cosa che uccideva ktest sani e tre server. Il riavvio l'ha
tolta, e il veicolo ha chiuso al primo tentativo. Registrato a item 23.

**T2 e T4 BLOCKED, e i due bloccanti dicono cose diverse** (item 23). Il primo è
un difetto di canale: la patch è arrivata monca di una riga di contesto, e
l'integrator l'ha *misurato* — 25.658 caratteri contro i 25.664 dichiarati, cioè
6, l'ordine di grandezza esatto della riga mancante. Il secondo è un difetto del
piano, ed è quello da ricordare: **T1 e T2 possedevano lo stesso blocco di
`ktest.worker.ts` nella stessa ondata**, con soglie diverse. `owns` disgiunti
valida **per file**; due task possono possedere regioni diverse dello stesso
file finché non si scoprono a scrivere la stessa. T1 è arrivato primo.
Conseguenza sostanziale da non nascondere dietro il BLOCKED: la tolleranza vive
come due costanti locali, non in una sede unica — il principio «una soglia, un
posto» è violato, ed è lo stesso difetto che è costato la riga 2.

**IL MIO ERRORE DI IT.16, attribuito dal duplicato e confermato da me.** Avevo
segnato «(e) `q35-prefillchunk-4b` bit per bit ✓» **leggendolo dalla prosa di
PHASES invece di eseguirlo**. Il gate era rotto da due iterazioni: la via intera
di it.15 quantizza le attivazioni a int8 e la bit-identità è caduta lì. Nessuno
se n'era accorto perché il gate si attiva con `--prefill-m` **senza**
`--prompt-idx`, e tutte le run di it.14-16 passavano `--prompt-idx 0`. È la
landmine «i JSON possono mentire in silenzio», e qui il silenzio era il mio.
Item 22, al PI.

**QUATTRO SESSIONI FANTASMA IN TRE ORE, E LA QUARTA ERO IO.** Cronologia dal log
del watchdog e dal reflog:

    00:01:34  rianimata 0348f2c0 (loop chiuso apposta) — miccia: loop-state.json
              versionato, riscritto da un mio `git checkout`
    00:56:05  duplicata ae3ad6a9 (la MIA sessione) — miccia: transcript fermo
              perché aspettavo un workflow in background
    02:03:28  duplicata di nuovo ae3ad6a9 — stessa miccia, nonostante avessi
              allungato il risveglio a 60 minuti apposta

**La mitigazione non funziona, e ora so perché.** Il watchdog usa il silenzio
del transcript come test di morte. Una sessione che aspetta un workflow è
silenziosa per costruzione: il workflow non scrive sul transcript di chi l'ha
lanciato. Quindi **più il lavoro dura, più è certo che venga duplicato**. E il
ramo `headless_dead` salta del tutto il controllo «transcript fresco ⇒ sta
lavorando», che è l'unica difesa prevista. Allungare il risveglio sposta
l'istante, non toglie la condizione.

**Perché fermo il loop invece di schedulare.** Non è prudenza: schedulare un
risveglio È l'azione che arma la duplicazione successiva. I duplicati hanno
lavorato bene — l'integrazione di it.17 è pulita e l'attribuzione del gate rotto
è fatta con un esperimento, non a naso — ma sono fuori dal mio controllo, non
posso verificarli mentre corrono, e lavorano sullo stesso albero e sulla stessa
GPU. Due delle tre condizioni di stop del protocollo sono soddisfatte: il
residuo del goal richiede un ruling (item 22 blocca la riga 5), e l'ambiente
rende l'iterazione non eseguibile in sicurezza.

**Cosa resta, per chi riprende**: la tolleranza da spostare in
`attnchunktol.ts` (T2 a mano sopra il codice di T1), la clausola (d) della riga
2 da scrivere (censimento già fatto in it.17), le righe 4, 5, 6. E il ruling
sull'item 22, senza il quale la riga 5 non può dichiarare i ratchet intatti.

---

## it.19 (2026-08-14) — il watchdog è corretto, e i due residui delle righe 2 e 3 sono chiusi

**IL BLOCCO AMBIENTALE È CADUTO.** La sessione harness ha corretto tutti e tre i
difetti che le avevo riportato e ha committato in `2679491`. **Verificato da me
leggendo il codice, non il suo riassunto**: `sidecar_mtime` con
`act = max(tm, sc)` (il polso non è più il solo transcript, che è il difetto per
cui aspettare un workflow garantiva la duplicazione); `headless_dead` legato al
`session_id` e non più a `prev_unit`, con `idle_needed` che non salta il
controllo di vita in nessun ramo; e le due sentinelle **prima** di tutto il
resto, nell'ordine giusto — copia (`fmtime - upd > COPY_TOL_S`) e poi staleness
(6h). Che la rilevazione della copia venga per prima è la cosa giusta: una copia
fresca di uno stato vecchio non è staleness, ed è la firma esatta dei miei 19
rami. `--dry-run --verbose` gira pulito. **Loop rimesso in moto.**

**RESIDUO 1 CHIUSO — la tolleranza ha una sede unica** (`5ce2e2f`). T2 rifatto a
mano sopra il codice di T1, che è quello in produzione: `attnchunktol.ts` con la
derivazione per esteso, stessa forma di `KQUANT_FAST_Q5K_PAIR_REL_TOL`.
**Spostate, non ritarate** — cambiare un numero è un'altra decisione e vuole
un'altra misura. Il test non legge più i due valori dal sorgente con una regex:
era più debole di quanto sembrasse, verificava che nel testo ci fosse *una*
costante, non che il banco usasse *quella*.

**E misurando ho trovato una cosa che la derivazione non diceva.** Sul caso
multitile il device è **2,14× peggiore della simulazione CPU** da cui la banda
era stata derivata: maxRel **8,44e-5** contro 3,95e-5 simulato. La banda
relativa 1e-4 lascia il **18% di margine**, non il 2,5× che la derivazione
lasciava sperare — il gate passa per la banda **assoluta**, larga ~565×. È
scritto nella sede unica, che ora è il posto dove serve saperlo: chi stringe la
relativa senza rimisurare la fa diventare rossa su un cambio di driver.

**RESIDUO 2 CHIUSO — la clausola (d) della riga 2**, come test `[6f]` e non come
prosa, così non marcisce. La copertura per **call-site**, che è una domanda
diversa da quella per byte del test `[6c]`: un solo sito dimenticato che
ri-derivi la condizione a mano vale zero byte in quel conto e vale la riga
intera — è precisamente ciò che era successo in it.14.

**E il test mi ha corretto un errore del mio censimento di it.17.** Avevo
scritto che i kind legacy erano due (q4_1, K-quant): sono **tre**, perché avevo
dimenticato i **24 `ssm` Q8_0**. La distinzione che avevo confuso e che il test
ora tiene ferma: il Q8_0 **passa dall'imbuto** `gemvB` e chiede la rotta al
piano, come si deve — è il piano a rispondergli «legacy», perché la via veloce è
q4_0-only per costruzione. **Coperto dalla convenzione ≠ instradato sulla via
veloce.** Due test scritti su una previsione sono andati rossi al primo giro, ed
è esattamente il loro mestiere.

**RIGA 2 COMPLETA. RIGA 3 COMPLETA.** Gate su questo albero: tsc pulito ·
vitest **676 passed | 10 skipped** · ktest **101 PASS / 0 FAIL**. Metrica
invariata (nessuna delle due modifiche tocca un kernel): TTFT a caldo
**32.265 ms**, 2,72× sulla baseline, alla barra manca 1,47×.

**Cosa resta**: la riga 4 è quasi fatta — (e2b) l'ha chiusa T1, che ha reso il
ramo batch dell'attenzione il quinto consumatore dichiarato in `engineNeeds`,
costante in ctxMax; resta (e2a), che per la C7-3 non tocca la metrica di questo
goal e si dichiara debito. La riga 5 è il lavoro vero e **richiede strumentazione
nuova**: i workgroup in volo per dispatch il motore non li misura da nessuna
parte. E la riga 5 non può dichiarare i ratchet intatti finché non c'è il ruling
sull'**item 22**.

---

## it.22 (2026-08-14, riga 4) — la portabilità era già mezza fatta, e l'altra metà è un debito reso falsificabile

**Riga 4 CHIUSA. Metrica invariata e doveva esserlo**: questa riga non tocca
kernel. TTFT a caldo **32.265 ms**, 2,72× sulla baseline.

**(e2b) l'aveva già chiusa T1**, e verificarlo è costato meno che rifarlo — è la
ragione per cui il primo passo di una riga è leggere, non scrivere. Il ramo
`batch` dell'attenzione è il **quinto consumatore dichiarato** in `engineNeeds`,
col fabbisogno **costante in ctxMax** e uguale a quello del decode. Il pezzo che
mi ha convinto è il `describe` in fondo a `gpulimits.test.ts`: si chiama
«garanzia: il ramo batch dell'attenzione e' contato e non alza il requisito» ed
è **letteralmente il vecchio sensore del debito, rovesciato**. Prima provava che
il fabbisogno cresceva col contesto — cioè che la frase «path Qwen indipendente
dal contesto» era falsa; ora prova il contrario. Il test non è stato cancellato:
è stato girato, e resta a guardia del fatto nuovo.

**(e2a) dichiarata debito, che è la seconda delle due strade che la riga
ammette.** Prima ho ri-verificato (C7-3) **sul codice di oggi invece di
ereditarla da it.0**: `q35gpumodel.ts` non importa né `rmsPairGemmSiluChunkFast`
né `prefillplan.ts`. Il consumatore è `gpuforward.ts`, l'assemblatore di
Qwen2.5-0.5B — path di conformità, non di prodotto.

Il debito è scritto con l'aritmetica invece che con l'aggettivo: il termine è
`4·K·mMax + …` con K = 896 **fisso**, quindi per scendere sotto i 16.384 B
garantiti da WebGPU servirebbe **mMax ≤ 4** — e alzare mMax è la leva che il
prefill vuole. Le due tirano in direzioni opposte (a mMax 16 sarebbero 61.696
B). La via d'uscita vera non è stringere il buffer: è **dare al path 0.5B la
forma multi-riga del 4B**, il cui workgroup storage non scala con M (1.152 B via
idot, 4.096 via f32 a M=16). Quella forma la riga 1 l'ha già trovata.

**E QUI STA IL PEZZO CHE VALE PIÙ DELLA RIGA.** Una dichiarazione di debito è
una frase, e una frase invecchia in silenzio. (C7-3) è vera **oggi**; se domani
qualcuno cabla il kernel fuso nell'assemblatore del 4B, il debito smette di
essere «solo 0.5B», lo scoping dell'intera riga 4 diventa falso, e **nessuno se
ne accorge finché un utente non apre la pagina su un device che concede i 16 KB
di spec**. Ora c'è un sensore: il test `(e2a)` legge `q35gpumodel.ts` e fallisce
se ci trova quel kernel. È la stessa forma del promemoria della bit-identità di
it.20 — un debito che suona da solo invece di aspettare che qualcuno rilegga il
docket.

**Item 25 al PI**, ed è l'unica cosa che non ho deciso: non *se* dichiarare il
debito (l'ho fatto, ed è la strada che la riga stessa ammette), ma **cosa
promette il DONE WHEN del contratto** sulla portabilità. Con (e2a) a debito il
goal chiuderebbe lasciando un termine sopra la garanzia WebGPU — l'unico del
motore. Quello è funzione obiettivo, non meccanismo, ed è la sola classe che il
protocollo mi vieta di decidere.

**Gate**: tsc pulito · vitest **681 passed | 10 skipped** (erano 676) ·
`gpulimits.test.ts` 32/32. **ktest NON rieseguito, e lo dichiaro**: questa riga
ha toccato solo commenti e un test, nessun kernel e nessun worker.

**Resta la riga 5**, che è il lavoro vero e non è burocrazia: serve
strumentazione **nuova** — banda per segmento, workgroup in volo per dispatch
(il motore non li misura da nessuna parte), TFLOP/s sostenuti contro il picco
9,26 misurato in riga 1. È la misura che deve spiegare **perché siamo a 32.265
ms mentre la proiezione della riga 1 dava un pavimento di ~8.665**: o esiste un
quarto collo che nessuno ha nominato, o quella proiezione era ottimistica.

---

## it.23 (2026-08-14, riga 5) — IL QUARTO COLLO HA UN NOME: la ricorrenza DeltaNet, 47% del piano

**La domanda della riga 5 era: siamo a 32.239 ms di prefill mentre la proiezione
della riga 1 dava un pavimento di ~8.665. Quarto collo mai nominato, o proiezione
ottimistica?** Risposta: **né l'uno né l'altro — la proiezione era INCOMPLETA PER
COSTRUZIONE, e lo diceva.** Il suo stesso testo escludeva «i 24 layer DeltaNet,
le norm, il RoPE, i dispatch». Il termine mancante non era nascosto: era
dichiarato e mai misurato.

**COSTRUITO L'INVENTARIO DEL PIANO**, che è il primo dei tre numeri che la riga
5 chiede e quello che il motore non misurava da nessuna parte: i **workgroup in
volo per dispatch**. Non è una misura GPU — è una proprietà **statica** del
piano, già decisa quando i bind group sono stati costruiti, quindi si legge
senza spendere un submit. Meccanismo: una categoria corrente (`pbCat`) che il
costruttore muove ai confini dei segmenti, e che `pushB` timbra su ogni step.
Sedici punti di assegnazione invece di 39 call-site toccati.

**Il piano di UN chunk da 16 righe: 1.624 dispatch. Sul prompt da 6333 token
sono 395 chunk ⇒ 641.480 dispatch.**

| categoria | disp | % piano | wg/disp | min |
|---|---|---|---|---|
| **deltanet:recurrence** | **768** | **47,3** | 80 | 32 |
| deltanet:gemm | 192 | 11,8 | 613 | 20 |
| gemm:ffn | 192 | 11,8 | 967 | 20 |
| gemm:ffn-down | 120 | 7,4 | 1.739 | 72 |
| gemm:qkv | 72 | 4,4 | 362 | 20 |
| norm:attn · norm:ffn | 64 | 4,0 | **16** | 16 |
| attn:core | 16 | 1,0 | 640 | 256 |

**LA RICORRENZA È QUASI META' DEL PIANO, e non è comprimibile col batch.** Emette
`2·M` dispatch **per layer** — 32 a M=16 — e sono seriali **per definizione**:
ognuno legge lo stato che il precedente ha scritto. Su 24 layer DeltaNet fanno
768 dispatch. Il commento in loco lo diceva già da it.30 («la ricorrenza non si
batcha, quindi restano M dispatch IN ORDINE»), ma nessuno aveva mai messo quel
fatto accanto a un conteggio.

**E l'occupancy la inchioda al pavimento**: 80 workgroup per dispatch di media,
minimo 32, su una scheda con 76 processori. Un dispatch che mette in volo ~80
workgroup riempie la scheda per **una sola ondata** e poi la svuota — e la riga
1 aveva già stabilito che su questo device il collo è l'occupancy, non la banda
(`splitk` batte `regs` di 2,13× solo per 576 workgroup invece di 144; la fusione
GQA taglia il traffico di 4× ed è più LENTA perché scende a 64 workgroup).
La ricorrenza sta esattamente nella zona che quella misura dichiara tossica,
per il 47% dei dispatch del prefill.

**Prima categoria separata di proposito**: `deltanet:recurrence` stava dentro
`deltanet:gates` nella prima passata, e la tabella diceva «792 dispatch di
gates», che è vero e inutile. Separarla è ciò che ha reso visibile il termine.

**Due sotto-occupazioni minori, nominate perché non si perdano**: `norm:attn` e
`norm:ffn` mettono in volo **16 workgroup** — il 21% dei 76 processori — per 64
dispatch a chunk. Piccoli in numero, ma sono il tipo di dispatch che costa
quanto il suo overhead e nient'altro.

**`sm` resta `null` nell'artefatto, di proposito**: WebGPU non espone il numero
di processori, e pinnare 76 dentro il motore significherebbe scrivere il valore
di QUESTA scheda in un modulo che gira ovunque. Chi confronta lo porta da fuori,
dichiarato.

**Gate**: tsc pulito · vitest **681 | 10** · ktest **101 PASS / 0 FAIL**.
Metrica invariata (32.265 ms) e doveva esserlo: questa iterazione misura, non
ottimizza.

**Cosa manca alla riga 5**, e non è poco: il tempo **per segmento**. L'inventario
dice quanti dispatch e quanto sotto-occupati, non quanti millisecondi. Serve
`timestamp-query` sul piano a chunk, raggruppando gli step per categoria — la
sonda esistente (`telemetryGpu`) è solo per il decode e spezza i pass per layer,
non per segmento di prefill. Con quello si chiudono anche gli altri due numeri
del done-when: banda efficace per segmento e TFLOP/s sostenuti contro il picco
9,26 misurato in riga 1.

---

## it.25 (2026-08-14, riga 5) — IL CRONOMETRO RIBALTA IL CONTEGGIO: non è la ricorrenza, è un Q5_K sul percorso vecchio

**RIGA 5 CHIUSA.** TTFT a caldo **32.127 ms**, discesa **2,727×** sulla
baseline, **1,467× dalla barra**. Artefatto
`results/engine/q35-ttft-kernel-checkpoint-4b-2026-08-14.json`.

**E DEVO CORREGGERE ME STESSO DI DUE ITERAZIONI FA.** In it.23 avevo scritto «il
quarto collo ha un nome: la ricorrenza DeltaNet, 47% del piano». Era il
**conteggio dei dispatch**, non il tempo. Col cronometro la ricorrenza è il
**5,0%** del prefill. Contare i dispatch non è misurare il tempo, e l'avevo
presentato come se lo fosse.

**LA VERITÀ, cronometrata per segmento** (sonda `timestamp-query` sul piano a
chunk, un pass per categoria, 64 chunk, zero overflow):

| segmento | ms totali | % prefill | disp | wg/disp |
|---|---|---|---|---|
| **`gemm:deltanet-out`** | **12.169** | **37,9** | 24 | 40.960 |
| `gemm:ffn-down` | 4.971 | 15,5 | 120 | 1.739 |
| `deltanet:recurrence` | 1.602 | 5,0 | 768 | 80 |
| `gemm:ffn` | 1.234 | 3,8 | 192 | 967 |

**24 dispatch su 24 layer DeltaNet fa UNO per layer — e la via veloce ne emette
tre.** Da lì la diagnosi: quel sito **cade sul fallback legacy**. Il perché è
scritto nel piano da sempre: `ssm_out` è **Q5_K**, la via veloce è q4_0-only per
costruzione, quindi `planPrefillGemm` risponde «legacy» — M gemv replicate
sull'asse z, **riuso dei pesi ZERO**, i pesi riletti 16 volte per chunk.

**E QUESTO RI-INQUADRA IL RULING DEL RIUSO.** L'item 19 aveva abbassato la barra
a 5,5× perché «l'11,54% dei byte resta sul percorso vecchio», e quel residuo era
stato assegnato al goal K-quant come una coda. **L'11,54% dei byte è il 37,9%
del tempo.** La quota di byte sottostimava massicciamente il peso, proprio
perché la forma legacy rilegge i pesi M volte: è la stessa asimmetria che rende
il riuso una leva. Il residuo non è una coda: è **la leva più grande rimasta sul
tempo al primo token**.

**LA CONTABILITÀ DEL TETTO, che è ciò che la riga 5 chiede:**

- prefill **32.101 ms**, di cui **22.751 dentro i pass GPU (70,9%)** e
  **9.350 fuori (29,1%)**: encode CPU, submit, e i buchi fra un submit e il
  successivo. Non l'ho attribuito più finemente e lo dichiaro — servirebbe una
  sonda CPU per dispatch che oggi non esiste.
- **1,578 TFLOP/s sostenuti contro il picco fp32 misurato di 9,26 = 17%.** La
  quota NON è un'efficienza: il prefill gira su pesi quantizzati e quel tetto
  non lo tocca per costruzione. Serve a dire una cosa sola, ed è quella che
  conta: **il collo non è l'ALU.**
- Perturbazione della sonda dichiarata invece che assunta: 57,6 ms/chunk a sonda
  accesa contro 73,0 a sonda spenta. La sonda somma il solo tempo *dentro* i
  pass, il totale include CPU e buchi — i due numeri misurano cose diverse, ed è
  esattamente la differenza che la riga 5 voleva vedere.

**Perché la proiezione di riga 1 dava 8.665 ms**: contava le moltiplicazioni
q4_0 e l'attenzione. Non contava — e lo dichiarava — i layer DeltaNet, e
soprattutto non contava che il loro `ssm_out` **non è q4_0** e quindi non
beneficia di nessuna delle leve montate. Non era ottimistica sul lavoro fatto:
era cieca su un termine che il lavoro non poteva toccare.

**Ratchet di correttezza, riportati coi numeri** (item 24 chiuso dal ruling):
ktest 101/0 · vitest 680|10 · tsc pulito · top-1 contro l'oracolo **1012/1024 =
98,828% su ENTRAMBI i bracci** · sequenze generate **identiche 8/8** · decode
48,15 tok/s (≥ 45,5). Il bench è stato rilanciato **sul codice finale** dopo la
sonda: 32.127 contro 32.265, −0,4%, dentro la banda.

**Lo strumento del checkpoint è uno script** (`scripts/build-ttft-checkpoint.mjs`)
e non un comando incollato qui: il checkpoint si rifà a ogni cambio di kernel, e
un conto ricopiato a mano è un conto che la prossima volta nessuno ripete
uguale. Ha la sentinella sul `kind` dei due artefatti che unisce — la landmine
di HANDOFF nasce da un file il cui nome diceva una cosa e il contenuto un'altra.

---

## it.26 (2026-08-14, riga 6) — chiusura: dieci clausole su dodici, e la barra mancata ha una causa misurata

**RIGA 6 CHIUSA. Il goal ha il consuntivo pronto e la chiusura formale è del
PI.** Consuntivo in `docs/engine/ttft-consuntivo-2026-08-14.md`, DONE WHEN voce
per voce con l'artefatto accanto a ognuna — non a memoria.

**TTFT a caldo 87.618 → 32.127 ms = 2,727×. Barra < 21.905: NON raggiunta,
manca 1,467×.**

**Dieci clausole su dodici soddisfatte. Le due che cadono sono la stessa** — la
barra, e la sua gemella `prefill.tokS > 289` (misurato 197,25), che è la stessa
clausola vista dal lato del rate.

**Non la chiamo un fallimento e non la chiamo un successo: la chiamo col suo
numero, e dico perché.** Il primo termine del prefill è `gemm:deltanet-out`,
37,9% del tempo con 24 dispatch, e cade sul fallback legacy perché `ssm_out` è
Q5_K mentre tutte le leve di questo goal sono **q4_0-only per costruzione** —
scritto nel contratto fra i vincoli, non scoperto adesso. Il goal ha esaurito le
leve che poteva usare; quello che resta non era suo.

**Ultima voce del done-when chiusa in questa iterazione: la banda per segmento.**
`gemm:deltanet-out` muove **1.093 GB a 89,9 GB/s** su un motore che ne ha
dimostrati ~300 — lento *e* con 16× i byte che servirebbero. `gemm:qkv` sta a
**738 GB/s**, che è sopra la DRAM di questo device: non è traffico verso la
memoria, è la cache che serve i pesi fra un chunk e il successivo. **Il numero
dice che il riuso funziona, non che la memoria vada così**, e l'ho scritto
nell'artefatto perché chi lo cita non lo prenda per una banda.
L'ho attribuita **solo** ai due segmenti i cui byte si legano all'inventario
pinnato: un GB/s calcolato su una stima di byte direbbe una cosa per un'altra.

**TRIAGE DEL DOCKET** (item 27). Undici item chiusi. Quattro — le diagnosi
«il conductor non regge» — marcati **superati dai fatti** e non cancellati:
restano come storia di un'attribuzione sbagliata, che è il loro valore. Tre
aperti, tutti del PI: la chiusura formale con la barra mancata, cosa promette il
DONE WHEN sulla portabilità, e il porting del 0.5B come goal suo.

**E una cosa che il PI deve sapere prima di prioritizzare**, perché ribalta un
suo ruling: l'11,54% dei byte che l'item 19 aveva assegnato al goal K-quant
**come coda** è il **37,9% del tempo**. La quota di byte sottostimava il peso
perché la forma legacy rilegge i pesi M volte — la stessa asimmetria che rende
il riuso una leva. Il goal K-quant non è un completamento: è la leva più grande
rimasta sul tempo al primo token.

**HANDOFF §1 riscritto e §2 corretta**: la Mappa era ferma a 57,0 s di it.14 ed
è la sezione che qualcuno legge per prima.

**Il loop si ferma qui, per esaurimento del lavoro decidibile**: tutto ciò che
resta è una decisione del PI, e le tre sono in chat. Non invento scope per
tenerlo vivo.
