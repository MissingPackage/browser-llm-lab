# Docket — engine-fase-c3a (decisioni PI pendenti)

13. **FINDING registrato, non una decisione PI** (2026-08-03, it.12, dalle due
    review avversarie). Con lo split MLA in produzione, il termine
    `mlaWorkgroupStorageBytes(ctxMax)` in `gpulimits.ts` ha come consumatori
    REALI solo glmforward/glmroute e ktest (il monolitico), non più glmmodel:
    il tetto di contesto del forward di produzione non è più la shared memory,
    è la VRAM della KV. Rimuovere o condizionare quel termine è MECCANISMO
    (ruling 2026-08-03 "decidere, non escalare": non si escala), ma cambia i
    limiti negoziati ⇒ si fa quando un consumatore lo chiede davvero (C3b/ctx
    lunghi), con ri-bench dichiarato, non come ritocco di passaggio. Nel
    frattempo il comportamento resta identico al pre-it.12 (verificato: pin di
    tests/gpulimits.test.ts invariati).

12. ~~**RULING — la fase 4b parte subito, portando la famiglia fusa di Qwen**~~
    **RISOLTO in apertura (2026-08-02, PI: "porta su glm tutte le ottimizzazioni
    che ci hanno dato tante soddisfazioni su qwen")** — registrato qui perche'
    **modifica l'ordine di PHASES**: la 4b era `blocked-by-4`, dimensionata dalla
    ri-misura dei clock che chiude la fase 4. Ora parte prima.

    **Perche' il ruling e' ben fondato** (state-2026-08-02 §2, non opinione): a
    parita' di device il path GLM rende 29.3 GB/s utili contro 155.6 del path
    Qwen — 5.1% del picco contro 27.0%. Le 18 fusioni WGSL esistenti sono tutte
    solo-Qwen. La 4b non inventa ottimizzazioni: ne porta di gia' validate, che
    e' il profilo di rischio piu' basso del goal.

    **Cosa NON cambia**: il gate 13.43, i gate di correttezza, e l'obbligo della
    fase 4 di ripresentare l'item 2. La ri-misura dei clock che la fase 4 doveva
    fare per dimensionare la 4b resta dovuta — cambia solo che ora servira' a
    spiegare quanto del guadagno viene dai kernel e quanto dai clock, che e'
    esattamente l'attribuzione richiesta dal done-when della 4b.

    **Conseguenza registrata**: la riga 4b di PHASES passa da `blocked-by-4` a
    `in corso`, e il suo primo task di spec ("prima si misura DOVE, poi si
    fonde") e' stato assolto dall'analisi di it.8, che ha misurato il DOVE a
    livello di path invece che di categoria di pass.

1. ~~**RULING — split della fase C3 e formulazione del gate prefill**~~ RISOLTO
   (2026-08-01, ruling PI in chat, sessione 18 — /goal-brief). Due decisioni
   prese insieme al contratto, registrate qui perche' modificano il perimetro
   ereditato da direction §7 e sciolgono una domanda lasciata aperta in C2.

   **(a) Split C3 → C3a + C3b.** PI: "ok lo split in 2 goal". Motivazione
   accettata: l'attribuzione misurata in C2 (docket C2 item 8) separa 158.9
   ms/token di STRUTTURA (pack CPU nel path caldo, 46 sync router/token,
   prefill sequenziale) da 56.1 ms/token di RESIDENZA. Le due masse si
   aggrediscono con leve diverse e chiudono su condizioni diverse; un goal
   unico "slab+tier+AUTOPIN+PILOT-real + gate tok/s" poteva chiudere a meta'
   su uno dei due assi. Perimetri:
   - **C3a** (questo goal): repack all'import, eliminazione/batching dei sync
     router, prefill batched M>1. Gate di chiusura = floor C1 13.43/56.58.
   - **C3b** (chartered, non avviato): budget slab ctx-aware, tier.h, AUTOPIN,
     PILOT-real/prefetch, modello di banda, instant-on, WP banda fredda browser.
   direction §7 emendata con lo split. Nomi c3a/c3b confermati dal PI
   ("ok sui nomi") per non rompere i riferimenti "C3" gia' scritti nei doc.

   **(b) Gate prefill: velocita', non sola capacita' — ma misurata sull'UX.**
   Il docket C2 item 6 chiudeva con una domanda esplicita non sciolta dal
   ruling 6a: "il PI decida se il gate prefill vada riformulato in C3 come
   requisito di capacita' e non di velocita'". Ruling PI: "su capacita' vs
   velocita' e' stato il dilemma iniziale del progetto. alla fine abbiamo
   deciso di puntare alla massima intelligenza possibile con una velocita'
   minima accettabile. 30+ tk/s come soglia normale e di piu' se thinking. per
   garantire l'esperienza utente ottimale".

   Traduzione nel contratto, in due livelli (entrambi scritti in GOAL.md):
   - **Gate di chiusura** (meccanicamente verificabile oggi): decode >= 13.43,
     prefill >= 56.58 tok/s — il floor oracolo CPU C1, come da ruling C2 6a.
   - **Target di programma, riportato a ogni bench e NON gateato**: decode
     >= 30 tok/s (>= 60 in regime thinking — i token di ragionamento sono
     latenza pura prima della risposta visibile) e **TTFT <= 4 s**.
   Il floor prefill 56.58 e' un `pp512` di llama-bench CPU: su p6 (461 token)
   vale 8.15 s di time-to-first-token, cioe' soddisfa il gate d'ingresso e non
   soddisfa nessuna esperienza utente. Lo stato attuale e' 5.221 tok/s = 88 s
   di TTFT. Da qui la doppia scrittura: C3a non puo' chiudere sotto il floor
   (era la deroga C2), ma non deve poter chiudere dichiarandosi arrivato.
   L'assenza dei numeri di gap dal report e' un FAIL della checklist.

   **Budget TTFT = 4 s** (non i 2 s proposti nel draft). PI: "il ttft va bene
   anche minore di 4 secondi. a 2 rischiamo di essere troppo stringenti e non
   chiudere mai. alla fine 4 secondi e' accettabile sul mio hardware che non e'
   top". Su p6 (461 token) 4 s equivalgono a ~115 tok/s di prefill, ~2x il
   floor CPU.

   **Soglie [ASSUMED] del draft confermate senza correzioni** ("ok, soglie
   confermate"): sync/token <= 2; pack < 1.0 ms/token nel path caldo; 60 tok/s
   come soglia UX in regime thinking; errore del modello di banda +/-15% (C3b);
   budget slab di prova 50% e 25% del parco (C3b).

11. ~~**RULING RICHIESTO — conformance full-corpus della fase 3**~~
    **RISOLTO (2026-08-02, PI: "ok item 11 e 8. ok sulla sostituzione")** —
    opzione (a): la sostituzione byte-identica è accettata come verifica della
    fase 3, e la conformance full-corpus resta il gate di chiusura della fase 6.
    Non è stata scelta la (c) (campione ratificato nell'harness): resta come
    debito registrato in `docs/engine/state-2026-08-02.md` §7 punto 6, insieme
    al fatto che il 256/256 non esiste come campo in nessun JSON. Se la fase 6
    dovesse ripresentare il problema "un gate da 5 ore non si esegue", la (c)
    torna in tavola come item nuovo.
    Testo originale (2026-08-02,
    it.7). Non blocca: si può fare in fase 6, dove serve comunque.

    Il done-when della fase 3 chiede «correttezza invariata (argmax ≡ cpuref-f64
    256/256 sul campione ratificato)». **Non l'ho eseguita**: il run C2 sul
    corpus intero ha richiesto **4,9 ore** e l'harness non ha un'opzione per il
    solo campione ratificato (`--prompts` seleziona i prompt, ma il campione
    2/8 di C2 non è documentato con gli indici).

    Al suo posto ho verificato la catena completa in modo più stretto e in
    secondi: (a) test unitario che carica gli stessi expert per **entrambi** i
    percorsi (raw+pack e slab) e confronta **byte per byte** ciò che finisce in
    VRAM; (b) test che confronta gli slab **sul disco** con `packExpertSlab`
    sui byte grezzi del GGUF — 7 campioni su entrambe le size-class, estremi e
    centro, tutti identici. Se i byte in VRAM sono identici, il calcolo lo è.

    **Opzioni**: (a) [RACCOMANDATA] accettare la sostituzione e lasciare la
    conformance alla fase 6, dove è un gate di chiusura; (b) eseguirla ora
    (~5 h di macchina); (c) aggiungere all'harness l'opzione per il campione
    ratificato — utile a prescindere, perché un gate che costa 5 ore non si
    esegue a ogni fase, e infatti non è stato eseguito.

10. ~~**RULING RICHIESTO — allineamento del path Qwen e ri-baseline del gate**~~
    **SUPERATO dal ruling sui limiti derivati (2026-08-02, it.6).** Con
    `min(adapter, requisito)` non si alza più nulla oltre il necessario: il
    path Qwen chiede già oggi 256 MiB di binding e 32 KiB di workgroup storage,
    valori compatibili coi requisiti calcolati (250,5 MiB e 30 848 B). Resta
    **una cosa da fare, non una da decidere**: allineare `gpuforward.ts:101-107`
    alla stessa derivazione, così ktest smette di essere più permissivo del
    motore che valida. Senza ri-bench, perché i valori non cambiano.
    Testo originale:

    (2026-08-02, it.5, da review avversaria). **Non blocca la fase 3**, ma va
    deciso prima di far girare il gate della fase 6.

    **(a) Inversione di permissività fra harness e motore.** I limiti sono
    negoziati sui 4 worker GLM ma NON su `gpuforward.ts:99` (`createEngine`,
    il motore Qwen usato da bench/conformance/e2e) né su altri 4 siti minori
    (`engine.worker.ts:513`/`:705` kernel-diag e attn-bench,
    `microbench.worker.ts:31`). Scenario concreto: la fase 4b riscrive un GEMV
    con `workgroup_size(1024)` — è dichiaratamente lo scopo per cui si negozia
    1024. Il device di **ktest** concede 1024 e valida il kernel: PASS. Il
    device del motore Qwen concede 256 e la pipeline fallisce. **L'harness di
    correttezza diventa più permissivo del motore che deve validare**, che è
    l'inversione che lo rende inutile.
    Non l'ho fatto di mia iniziativa perché toccare il device del path Qwen
    obbliga a ri-benchmarkare: quel path ha il gate di non-regressione HARD
    contro il baseline quiescente 2026-08-01.

    **(b) Il baseline del gate GLM non è più confrontabile alla lettera.**
    `results/engine/bench-glm-4090-b12-quiesced-2026-08-01.json` è stato
    prodotto con 4 limiti diversi da quelli di adesso e **non contiene** il
    campo `deviceLimits` (nasce ora). Se il prossimo decode scende, non c'è
    modo di distinguere rumore da effetto dei limiti. La spec WebGPU §3.6.2
    avverte esplicitamente che chiedere limiti migliori *può* avere un impatto
    prestazionale, e nessuno shader committato oggi ha bisogno di >256
    invocazioni o >8 storage buffer (inventario: max `workgroup_size` 256, max
    workgroup storage 27008 B, max storage buffer per stage 7).

    **Opzioni**:
    (a) [RACCOMANDATA] **negoziare ovunque + ri-baseline dichiarata**: si
        applica `negotiateLimits` anche a `gpuforward.ts` e ai siti minori, si
        ri-esegue il bench GLM e quello Qwen a macchina quiescente, e i due
        report nuovi (che ora contengono `deviceLimits`) diventano il baseline
        di riferimento, con la sostituzione registrata qui. Costo: due run di
        bench. Beneficio: un solo regime di limiti in tutto il repo, e da qui
        in avanti ogni confronto è falsificabile.
    (b) **restringere la negoziazione a ciò che ha un consumatore oggi**:
        si torna ai limiti vecchi e si alza solo quello che una fase usa
        davvero, quando lo usa. Più conservativo, ma contraddice il ruling
        "negoziali subito" e sposta il problema in avanti.
    (c) **lasciare com'è**: i 4 worker GLM negoziano, il resto no. Sconsigliata:
        è la configurazione che produce l'inversione descritta in (a).

9. **FINDING registrato, non una decisione** (2026-08-02, it.3, ruling PI
   "annota questi 2 findings"). Due cose emerse dal probe e dalla ricerca che
   serviranno oltre C3a:

   (a) **`subgroup-matrix` NON è più assente dai browser.** L'adapter espone
   `chromium-experimental-subgroup-matrix`, `subgroups`, `subgroup-size-control`
   e `chromium-experimental-timestamp-query-inside-passes`. direction §8
   rischio 1 corretto; ideas-ledger §B guadagna la riga "GEMV con subgroup ops".
   **Perimetro**: visibili solo con `--enable-unsafe-webgpu` ⇒ valgono per il
   ceiling del motore, non per i confronti pubblici, dove la clausola originale
   resta valida e va dichiarata (direction §8.4).

   (b) **Il readback per layer preclude per costruzione il record/replay del
   grafo di comandi.** Il graph capture di ORT richiede shape statiche *e
   nessun kernel su CPU*. Quindi il drain MoE non costa solo latenza: chiude
   un'intera famiglia di ottimizzazioni future. Rende l'item 8 (residenza
   totale) più pesante di quanto sembri guardando solo i ms/token.
   Registrato in ideas-ledger §B.

8. ~~**RULING RICHIESTO — residenza totale: pagare 0.67 GiB di qualità per
   eliminare il drain?**~~ **RISOLTO (2026-08-02, PI: "ok item 11 e 8")** — si
   paga. La leva 2 si progetta **a residenza totale**: binding fissi + offset
   aritmetico, che è il pattern provato da ORT/ggml/MLC, e senza miss il
   readback del router sparisce del tutto (non "si riduce": non serve più).

   **Conseguenze operative del ruling** (registrate qui, non decise nel loop):
   - Serve una **eval di perdita** prima di ratificare quali expert degradare:
     il ruling autorizza la spesa, non esonera dal misurarla. Il gate resta
     top-1 vs golden ≥ 98.83% full-corpus e argmax ≡ cpuref-f64 sul campione.
   - ~~Emendamento a PHASES da approvare~~ **FATTO (2026-08-03, PI: "fai
     l'emendamento e chiudiamo")**: `GOAL.md` emendamento 4 + riga **4c** in
     PHASES (`blocked-by-4`, la fase 6 ora è `blocked-by-3,4,4b,4c,5`). È
     l'unica fase del goal con un authority delta: quant asimmetrica e nuova
     versione di layout dello slab escono dal must-docket. La scelta di QUALI
     expert degradare resta vincolata alla eval di perdita — il ruling
     autorizza la spesa, non esonera dal misurarla.
   - Il deficit a ctx 4096 è 1.03 GiB, non 0.67: la scelta di quali expert
     degradare va dimensionata sul contesto che C3b vuole servire, non su 525.

   Testo originale (2026-08-02, it.3). Non blocca le fasi 3-4; serve
   prima di decidere la forma finale della leva 2.

   **Il fatto.** Il drain della coda (46 `mapAsync` per token, ognuno una
   barriera che aspetta TUTTO il lavoro accodato) sparisce solo se la selezione
   dell'expert non richiede mai la CPU — cioè solo se **ogni** expert è
   residente, perché solo la CPU può leggere da OPFS. Conto esatto su questo
   device:

   | voce | GiB |
   |---|---|
   | parco expert COMPLETO | 14.60 |
   | pesi non-expert | 1.26 |
   | KV a ctx 525 | 0.05 |
   | **richiesto** | **15.91** |
   | VRAM usabile (16376 MiB − 763 di desktop) | 15.25 |
   | **deficit** | **0.67** |

   **Mancano 135 expert su 2944 — il 4.6% del parco.** A ctx 4096 il deficit
   sale a 1.03 GiB.

   **Perché è una decisione tua e non mia**: cade esattamente sulla funzione
   obiettivo (direction §2, due assi: capienza a parità di spazio, velocità a
   parità di modello). Qui i due assi si toccano — si spende qualità per
   comprare velocità. Vie: quant più aggressiva sul 5-10% di expert più freddi
   (la matrice usage di C1 li identifica; lineage ds4 = quant asimmetrica), KV
   quantizzato (risolve soprattutto a ctx lunghi), head Q6_K→Q5_K (−40 MB, non
   basta da sola).

   **Costo di non pagarlo**: il drain resta e la leva 2 si riduce alla sola
   sovrapposizione CPU/GPU via prefetch (~54 ms/token, comunque il guadagno
   singolo più grosso identificato finora). Non è un disastro: è metà del
   valore.

   **Serve una eval di perdita** prima di decidere: quantizzare gli expert
   freddi cambia la qualità, ed è esattamente il tipo di misura che direction
   dichiara parte del progetto. Proposta: la fase 4 misura prima la
   sovrapposizione (che non costa qualità), e la residenza totale diventa un
   item separato con la sua eval.

7. ~~**RULING RICHIESTO — prefetch e limiti WebGPU**~~ **RISOLTO (2026-08-02,
   ruling PI)**: (a) prefetch LOOKA **ammesso nel perimetro C3a** ("come la
   quarta leva: la misura dimostra che senza non si prende il gate");
   (b) i tre limiti WebGPU non negoziati **si negoziano subito**, in fase 3.
   Recepito in GOAL emendamento 2 e in spec §3.1 / §3.2-bis.

   Contesto della richiesta: CPU e GPU non si sovrappongono mai (per layer:
   1.7 ms GPU con CPU ferma, 1.5 di drain, 1.18 di `ensure` con GPU ferma);
   il prefetch a recall 92% @K=8 sposta ~54 ms/token dietro il lavoro GPU.
   Il probe ha misurato che il device prende i default di spec su
   `maxStorageBuffersPerShaderStage` (8 su 16 disponibili),
   `maxComputeInvocationsPerWorkgroup` (256 su 1024) e `maxBufferSize`
   (clampato a 2 GiB su 4).

6. ~~**RULING RICHIESTO — spec C3a**~~ **RISOLTO (2026-08-02, PI: "ok andiamo
   avanti")**, dopo che le 5 decisioni erano state esposte con raccomandazione
   esplicita per ciascuna. Interpretazione registrata per essere corretta se
   sbagliata: approvazione della spec **nella forma raccomandata** —
   (1) repack come secondo file OPFS da 15.68 GB col GGUF mantenuto;
   (2) criterio "minimizzare i drain della coda", con lo scarto esplicito del
   pipelining nel decode; (3) leva 4: prima misurare dove va `gpuBusy`, poi
   fondere, attribuendo separatamente kernel e clock; (4) prefill M=16 iniziale
   con identità sull'argmax; (5) gate 13.43/56.58 confermati, TTFT ≤4 s
   riportato-non-gateato. **Fasi 3-6 SBLOCCATE.**
   **NON coperto da questo ruling**: l'item 8 (residenza totale, 0.67 GiB di
   qualità per eliminare il drain) è emerso DOPO ed è una decisione a sé.

   ⚠️ **La §3 è stata riscritta in it.3**: il meccanismo attribuito nella
   v1 ("latenza dei submit") era SBAGLIATO — il costo API di una submit è ~13 µs,
   quindi 47 submit valgono 0.6 ms/token, non 75. Il meccanismo vero è che
   `mapAsync` è una **barriera** che drena la coda: la GPU è idle esattamente
   quando non è dentro un pass (`gpuBusy`/wall 36.4% ≈ utilizzo campionato
   34.6%). Le decisioni 2 e 3 di §8 vanno lette sulla versione riscritta. Documento: `docs/superpowers/specs/2026-08-01-engine-fase-c3a-design.md`.

   Cinque decisioni, in §8 della spec:
   1. **Repack (§2)**: secondo file OPFS `*.slabs.bin` da **15.68 GB** accanto
      al GGUF (che resta, perché conformance e routing lo leggono) ⇒ 32.89 GB
      in OPFS. Spazio verificato (1.2 TB liberi). Header con magic + versione
      di layout + SHA del sorgente, rigenerazione su mismatch, scrittura su
      temporaneo + rename. One-shot ~28 s di CPU.
   2. **Criterio della leva 2 (§3)**: la misura dice che i readback valgono
      solo 7.6 degli 83 ms/token — il resto è **frammentazione dei submit**.
      Quindi il criterio proposto è **"minimizzare i submit per token, non i
      readback"**, con raccomandazione (A) *selezione su GPU con parco bindato
      per size-class*, **condizionata a un probe dei limiti WebGPU reali**
      (`maxStorageBufferBindingSize`/`maxBufferSize` non sono mai stati letti —
      marcati [VERIFY], probe da ~20 righe come primo task della fase 4).
      Si chiede anche di ratificare lo **scarto esplicito del pipelining**
      (era fra le vie note del contratto, ma nel decode il token t+1 dipende da
      t e il layer l+1 da l: non è applicabile).
   3. **Metodo della leva 4 (§4)**: prima **spezzare `gpuBusy` per categoria**
      (attention / router / shexp / catene expert / head) riusando i timestamp
      già raccolti, poi fondere. E obbligo di **attribuire separatamente**
      quanto del guadagno viene dai kernel e quanto dai clock che salgono.
   4. **Prefill (§5)**: M=16 iniziale [ASSUMED]; condizione di identità =
      **argmax identico su tutte le posizioni** M=1 vs M>1 (`maxAbsDeltaLogit`
      resta metrica di scala, mai gate — landmine C2).
   5. Conferma che **i gate restano 13.43 / 56.58** e che il TTFT ≤4 s resta
      riportato-non-gateato.

   Nota che emerge dalla spec e vale la pena leggere: con M=16 il TTFT
   scenderebbe verso ~5-6 s, **ancora sopra i 4 s** — il prefill ha bisogno
   anche delle leve 1/2/4, non solo del batching.

2. **RULING RICHIESTO — clausola di fallback per la fase 4 (sync router)**
   ⏸ **RIMANDATO dal PI (2026-08-01, opzione c): "decidere dopo la fase 4"** —
   la clausola si scrive quando si vedrà l'effetto reale della rimozione dei
   sync sui clock GPU, cioè quando la forbice 10.2-15.6 sarà sciolta da una
   misura. **Da ripresentare al PI a fine fase 4, prima della fase 6.**
   Testo originale:

   (2026-08-01, iteration 0). Non blocca: le fasi 1-2 girano comunque; serve
   una risposta prima della fase 6.

   C1 aveva una clausola pre-negoziata (se il tap hidden sfora il timebox, il
   goal chiude in modalità ridotta e la via alternativa va a docket). C3a non
   ne ha. Il rischio è concreto e quantificato: la proiezione C2 dice che a
   residenza perfetta il decode arriva a **6.3-7.0 tok/s**, quindi tutto il
   resto del cammino verso 13.43 deve venire dall'eliminazione dei 46 sync,
   la cui resa non è mai stata misurata isolatamente (i 158.9 ms/token sono
   sync *più* 1.816 dispatch, senza split noto — è esattamente ciò che la
   fase 1 va a misurare). Senza clausola, l'esito "sotto il floor" produce una
   seconda deroga come in C2, cioè una decisione presa sotto pressione a fine
   goal invece che a mente fredda adesso.

   **Opzioni**:
   (a) [RACCOMANDATA] **clausola simmetrica a C1**: se dopo le fasi 3-5,
       ri-misurato a macchina quiescente, il decode resta sotto 13.43, C3a
       chiude con la misura, l'attribuzione aggiornata e la leva residua
       identificata a docket — senza scorciatoie sul path e senza auto-deroga;
       la decisione sul da farsi resta PI.
   (b) **nessuna clausola**: sotto il floor = FAIL del goal e ri-scope.
   (c) **clausola condizionata al numero di fase 1**: si decide dopo aver
       visto lo split sync/dispatch misurato (rimanda la decisione di
       un'iterazione, con l'informazione in mano).

3. ~~**PLAN-CHECK**~~ PRE-AUTORIZZATO (2026-08-01, PI in chat: "vai con goal
   setup e poi parti col loop"). PHASES.md è scritto e il loop parte senza
   attendere approvazione esplicita: 7 fasi, sequenziali (owns sovrapposti sul
   path caldo), fase 1 ready subito, fasi 3-6 gated dal ruling di spec (fase 2),
   fase 4 = rischio dichiarato del goal. Se leggendo PHASES.md la
   decomposizione non ti convince, la modifica passa da qui (item nuovo), non
   dal loop.

4. ~~**RULING RICHIESTO — le tre leve del contratto non raggiungono il gate**~~
   **RISOLTO (2026-08-01, ruling PI: opzione (b) — "ammettere subito la quarta
   leva")**. La granularità/fusione dei dispatch entra nel perimetro C3a; il
   gate 13.43 resta invariato. Conseguenze recepite:
   - **GOAL.md emendamento 1**: quarta leva nel perimetro, nuovo DONE WHEN
     `gpuBusy ≤ 54.5 ms/token` (soglia DERIVATA, non arbitraria: budget del
     gate 74.46 − stallo post-repack 12.4 − floor sync misurato 7.6); la voce
     "fusione dei dispatch" esce dal must-docket.
   - **PHASES.md**: nuova fase **4b** (guerra ai dispatch), e la fase 4 chiude
     con una **ri-misura di `gpuBusy` e dei clock** — che è il pezzo utile
     dell'opzione (a) conservato dentro la (b): dimensiona 4b con un fatto
     invece che con l'assunzione ottimistica.
   - Fase 2 (spec) SBLOCCATA.
   Testo originale:

   (2026-08-01, it.1 fase 1). **BLOCCA la fase 2**
   (la spec deve scegliere il meccanismo dei sync, e la scelta dipende da
   questo). Numeri in `results/engine/bench-glm-4090-b12-attrib-2026-08-01.json`,
   journal it.1.

   **Misura.** Wall decode 215.0 ms/token si scompone in: `gpuBusy` **78.2**
   (36.4%) + stallo residenza **53.8** + sync/CPU **83.0** ⇒ **63.6% fuori
   dalla GPU**. Il probe indipendente dice che i 46 readback costano solo
   **7.6 ms/token** di floor irriducibile: meno del 10% degli 83 ms di
   sync/CPU è readback vero, il resto è latenza di submit e bolle.

   **Conseguenza.** Con le leve 1 e 2 del contratto al 100% di efficacia
   (repack che azzera il pack CPU −41.4; sync ridotti al floor −75.4):
   **98.2 ms/token = 10.18 tok/s**, contro un gate 13.43 che vuole ≤74.46
   ms/token. **`gpuBusy` da solo (78.2) eccede il budget del gate.** Le leve
   del contratto non bastano per costruzione — non è congettura, è il wall
   misurato meno i due termini che le leve rimuovono.

   **Osservazione discriminante già acquisita** (per questo ho campionato i
   clock durante la run): la GPU gira a **1746 MHz medi su 3105 di cap** con
   utilizzo **34.6%** — è sotto-clockata *dalle bolle che la leva 2 va a
   togliere*. Se `gpuBusy` scalasse col clock SM (limite ottimistico: i GEMV
   sono in parte memory-bound e il clock memoria è già al massimo) diventerebbe
   44.0 ms ⇒ **64.0 ms/token = 15.63 tok/s, gate PASS**.
   **La forbice vera è 10.2 – 15.6 tok/s, col gate 13.43 dentro.** La leva 2 ha
   un payoff di secondo ordine (far salire i clock) che nessuno aveva
   dimensionato e che può valere quanto quello di primo ordine.

   **Terzo dato, indipendente**: 2.22 GB di pesi letti per token contro 576
   GB/s di banda reale del device ⇒ floor memory-bound **3.85 ms/token**;
   `gpuBusy` è **20× sopra**. A 1816 dispatch/token siamo a 43 µs per dispatch
   su GEMV che ne dovrebbero costare ~2. Esiste una **quarta leva**
   (granularità/fusione dei dispatch) con margine enorme, non elencata nel
   contratto e messa per ultima da direction §2.

   **Opzioni**:
   (a) [RACCOMANDATA] **spec a due stadi, gate invariato**: la fase 4 attacca i
       sync e **ri-misura `gpuBusy` e i clock subito dopo**; il numero decide se
       serve la quarta leva. Costa un ciclo di misura, non un ciclo di
       implementazione, e scioglie la forbice con un fatto invece che con
       un'assunzione. Se dopo la leva 2 il gate resta irraggiungibile, la
       quarta leva entra in C3a via emendamento (item nuovo), non di soppiatto.
   (b) **ammettere subito la quarta leva** (fusione/granularità dei dispatch)
       nel perimetro C3a: massimizza la probabilità di prendere il gate, ma
       allarga lo scope a kernel engineering — che direction §2 deprioritizza —
       e allunga il goal di parecchie iterazioni.
   (c) **abbassare il gate di C3a** a un valore dentro la forbice pessimistica
       (es. 10 tok/s) e spostare 13.43 a un goal successivo: onesto, ma ripete
       la dinamica della deroga C2 prima ancora di provarci.
   (d) **chiudere C3a sulle sole leve 1-3 con misura** e aprire un goal
       dedicato alla guerra ai dispatch.

   Collegato all'**item 2** (clausola di fallback): se il PI sceglie (a), la
   clausola diventa più urgente, perché il ramo pessimistico della forbice è
   ora quantificato e non più ipotetico.

5. ~~**CORREZIONE PHASES**~~ **RISOLTO (2026-08-01, ruling PI: "correggere la
   riga")**. PHASES fase 1 emendata: done-when "run glmbench che scrive il
   report (exit 0 o 4)" e attribuzione dentro il report di bench. Testo
   originale:

   (2026-08-01, it.1). La riga
   della fase 1 chiede "run glmbench **exit 0**": è insoddisfacibile in fase 1,
   perché exit 0 = gate PASS e i gate passano solo in fase 6. L'esito corretto
   della fase 1 è **exit 4** (report scritto + gate FAIL). Chiedo di emendare la
   riga in "run glmbench che scrive il report (exit 0 o 4)". Stessa riga: il
   secondo JSON di attribuzione è stato consegnato **dentro** il report di
   bench (`attribution2`) invece che come file separato — sostanza consegnata,
   forma no; proposta: allineare il testo alla forma consegnata.
