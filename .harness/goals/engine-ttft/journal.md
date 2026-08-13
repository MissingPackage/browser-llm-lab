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
