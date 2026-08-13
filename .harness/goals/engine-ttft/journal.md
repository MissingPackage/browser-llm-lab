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
