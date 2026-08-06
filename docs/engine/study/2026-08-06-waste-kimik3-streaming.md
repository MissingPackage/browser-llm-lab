# Studio: WASTE e kimi-k3-in-c — streaming di modelli oltre-memoria (2026-08-06)

Richiesto dal PI (ruling 2026-08-06, direction §2-bis punto 3): i due repo
che eseguono Kimi K3 (2.78T parametri) su hardware consumer, come catalogo
di strategie per il nostro regime browser. Cloni shallow in
`~/study-repos/{waste,kimi-k3-in-c}` (fuori repo). Ogni numero citato ha la
sua fonte nei doc dei repo; niente è riprodotto da noi.

## 0. I due regimi, e il nostro

| | kimi-k3-in-c | WASTE | noi (GLM-4.7-Flash) |
|---|---|---|---|
| modello | K3 2.78T MoE | K3 2.78T / Kimi-Linear 48B | 30B-A3B MoE |
| compute | CPU AVX2 (EPYC 124c) | CPU (M5 Pro) | WebGPU (4090M) |
| memoria/modello | 8.24 GB / 1.56 TB = **0.5%** | 46 GB / 982 GB = 4.7% | 15.5 / 15.9 GiB = **97%** |
| decode | 0.03-0.09 tok/s | 0.45-0.63 (K3), 10.65 (48B) | 5.17 tok/s |
| collo | disco (25.8 GB/token) | disco (10.5 GB/token) | **struttura** (sync/kernel) |

Il punto di mappa: loro sono profondamente disk-bound (ogni token attraversa
il disco), noi siamo al 97% residenti e il disco tocca ~4.5 miss/token.
Le loro *strategie di scheduling* trasferiscono; i loro *numeri* no. Il
datapoint "usabilità": Kimi-Linear 48B a 10.65 tok/s su CPU con floor
1.28 GB — un MoE della classe 48B è usabile in streaming puro. Il nostro
30B su GPU parte con un vantaggio strutturale di banda che oggi non
incassiamo (state-2026-08-02 §2).

## 1. Convergenze indipendenti (le nostre scommesse, validate altrove)

1. **Lookahead router = LOOKA, terza replica indipendente.** WASTE esegue
   il router di L+1 sull'hidden di L e prefetcha i top-6: precision 92.2%
   rank-1, 81.4% top-6 (TECHNICAL.md). Noi: recall 92% @K=8 (C1); colibri:
   71.6% su GLM-5.2. Regola loro che adottiamo com'è: «la predizione cambia
   solo QUANDO si muovono i byte, mai il risultato» — il router vero decide
   sempre. Bit-identity come criterio di ship per ogni meccanismo di timing
   (cache, read-ahead, lookahead): è la nostra conformance, promossa a gate
   esplicito per-meccanismo.
2. **Chunked prefill = la nostra fase 5, misurata da altri.** Dedup degli
   expert sul chunk: 3.3× meno letture, logits identici a rumore float;
   toglie il 70-76% dell'I/O e lo 0% del compute (EFFICIENCY §1). Per noi
   il guadagno è doppio: il dedup I/O vale per i miss, e — a differenza del
   loro caso CPU — il batching GEMV→GEMM su GPU taglia ANCHE il compute.
3. **Overlap I/O/compute: «il valore del prefetch non è la banda, è che la
   read smette di bloccare l'aritmetica»** (EFFICIENCY §2: il disco è
   quasi saturo a queue depth 1; §4A: pipeline = 1.6× misurato, "turns a
   sum into a maximum"). È l'emendamento 2 di C3a detto con altre parole.
4. **Budget: somma tutto prima di allocare, rifiuta sotto il floor, il
   consumo misurato è l'unico numero da citare** (kimi-k3-in-c §11,
   WASTE ENGINE §3). Già nostra postura (precondizione con rifiuto, R7);
   loro aggiungono: budget della cache SOLO in multipli interi del working
   set di un token — sotto un multiplo la cache non tiene vivo niente.
5. **Prefill lento = sessioni salvate.** A velocità da streaming il
   re-prefill costa minuti, il restore è una read (WASTE session state).
   È la prefix-cache OPFS di direction fase B, con una ragione in più.

## 2. Risultati negativi loro che ci risparmiano lavoro

1. **«Volatile memory is memory you have given away»** (EFFICIENCY §4B,
   LEARNED §24/§30-32). Purgeable/mlock/pressione: al budget che funziona,
   la memoria oltre quel che la macchina lascia residente NON si può
   comprare a nessun prezzo; il knee lo fissa la capacità, la policy decide
   solo cosa viene distrutto. È ESATTAMENTE il nostro probe it.19 (la LRU
   del driver che retrocede 1.25 GiB a 8 GB/s): due stack, stessa fisica.
   Chiude in via definitiva ogni tentazione di "spremere il brim" della
   VRAM: si sta SOTTO il tetto, con margine dichiarato.
2. **Allocazione batte capacità, e l'hit-rate è una metrica che mente**
   (kimi-k3-in-c Parte IV): la config più veloce legge il 79% di byte in
   PIÙ della più lenta (trunk pinnato > cache expert); 28× la memoria =
   1.70× la velocità. E "hit rate" ha tre definizioni di cui solo la
   RETENTION (1 − evicted/requests) concorda coi byte letti — il prefetch
   gonfia le altre due. → requisito diretto per la telemetria unica della
   fase 4d: contatori separati per hit-residente / hit-da-prefetch /
   retention, mai un "hit rate" solo.
3. **LRU su scansione ciclica = hit rate zero esatto** (kimi-k3-in-c §11):
   90 slot su un ciclo di 93 layer evicta sempre il prossimo richiesto;
   prefix pinnato dà N/93 deterministico. Per C3b: gli accessi
   layer-sequenziali (trunk, KV) vogliono pin+ring, MAI cache LRU; la LRU
   va bene solo sugli accessi random (expert). Il loro victim-picker a tre
   stati (EMPTY subito, INFLIGHT mai, pinned dopo) è il nostro ExpertCache
   con una macchina a stati più onesta sotto concorrenza.
4. **Quant asimmetrica per-expert: leva inesistente, seconda conferma**
   (TECHNICAL "failed non-uniform expert allocation", LEARNED §20/§23):
   dopo lo scaling per-canale gli expert differiscono in cosa calcolano,
   non in quanto sono difficili da quantizzare (allocatore ottimo ≈
   random, 1.01-1.15×); demote i freddi = risparmi disco e ~zero I/O,
   demote i caldi = quality loss grande (25.6% I/O per +11pp errore). La
   nostra ladder Q3_K (it.18) ha misurato lo stesso verdetto sulla stessa
   classe di idea. Due modelli, stessa conclusione: NON riaprire.
5. **Trunk a 3 bit: collassa** (Q3G: logits a 36% dal Q4G, output in
   punteggiatura — TECHNICAL). Il muro di qualità sta davanti al muro di
   velocità. Conferma il nostro pin 98.83 come vincolo, non pignoleria.
6. **Il lookahead in prefill NON paga** (LEARNED §36): un chunk da 64
   token tocca ~550 expert distinti/layer, gli speculativi vengono
   evictati prima dell'uso (+6.9% read, zero wall). Per la fase 5: niente
   prefetch speculativo nel path M>1, il dedup del chunk basta.
7. **La cache che «non partecipava»** (kimi-k3-in-c Parte IV): il Quantile
   Balancing di K3 appiattisce l'usage by design e uccide la LRU sotto i
   36 GB di arena. GLM ha la stessa firma non-Zipf (C1: mediana 0.0257%
   vs 0.034% uniforme) MA il nostro rapporto residenza/parco è 97% contro
   il loro 1-4%: la loro lezione morde solo il regime telefono di C3b,
   dove budget piccoli senza lookahead possono comprare ~niente.
   → la simulazione WP-0 deve includere il ramo "budget 25-50%".

## 3. Cosa NON fanno (e resta nostro)

Nessuno dei due ha il problema del drain: su CPU il risultato del router è
disponibile in-thread, la selezione non attraversa mai un confine
host↔device. Il **decode ottimistico con riparazione esatta** (proposta
2026-08-06, tradeoff replay ratificato dal PI) non ha equivalente nei due
repo: è la risposta a un problema specificamente GPU/WebGPU. I loro
meccanismi sono complementari: lookahead (validato 3×) alimenta il
prefetch; l'ottimismo+repair elimina il sync. Nota di metodo loro che
adottiamo per il repair: bit-identity check "repair on vs off" su un run
senza miss forzati, e contatori deterministici separati dal wall.

## 4. Aggiornamenti operativi derivati (nessuno preso senza il suo ruling)

- **fase 4d (telemetria)**: schema contatori con retention + fonte
  dell'hit (residente/prefetch) — da §2.2. Già nel perimetro em.6.
- **WP-0 (simulazione)**: aggiungere Belady come ceiling accanto a LRU
  (kimi-k3-in-c §13: «la flatness appartiene alla policy o al workload?»),
  ramo budget 25-50% per il regime telefono, e il caveat loro sulla
  contaminazione prefill/decode nelle tracce (la nostra traccia C1 separa
  già decode da full — usare il ramo decode per lo steady-state).
- **fase 5**: niente lookahead speculativo nel chunk (§2.6); il test di
  identità M=1/M>1 resta il gate giusto (loro: max abs 6.7e-6 = rumore).
- **C3b charter**: pin+ring per accessi sequenziali vs LRU per random
  (§2.3); budget in multipli del working set; «sotto il tetto con margine»
  come principio (§2.1).
- Meta: WASTE dichiara «ideas/hypotheses/priorities/tests/decisions are
  human, the code is written by LLMs» — lo stesso modello di sviluppo di
  questo repo, con la stessa disciplina (LEARNED con risultati negativi
  datati e correzioni a fianco ≈ il nostro journal+docket).
