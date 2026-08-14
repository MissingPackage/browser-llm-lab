# Pre-registrazione — fase 0 di `engine-kquant` (riga 1), famiglie del 35B

**Scritta PRIMA della run**, come la sua gemella
`ttft-riga1-prereg-2026-08-13.md`. Serve a una cosa sola: rendere possibile
sbagliare. Un numero commentato dopo averlo visto non e' una previsione, e la
fase 0 di questo progetto ha gia' pagato una volta il prezzo di concludere da
un conteggio invece che da un cronometro.

## Cosa e' gia' misurato (it.1, e non si rimisura qui)

| famiglia | shape | legacy | multi-riga intera | rapporto |
|---|---|---|---|---|
| Q5_K | `[4096, 2560]` | 1,2700 ms | 0,0452 ms | **28,07×** |
| Q4_1 | `[9216, 2560]` | 2,3483 ms | 0,1040 ms | **22,58×** |

Entrambe superano la regola di stop (≥ 1,5×) e **si cablano**. Questa
pre-registrazione riguarda le **tre famiglie che questo goal NON cabla** e che
servono al goal successivo (il 35B).

## Le shape, e da dove vengono

Header dump `results/engine/q35-header-dump-2026-08-10.json`, entry
`Qwen3.6-35B-A3B-UD-Q4_K_S` (dModel 2048, dFfnExpert 512, nExpert 256, 40
layer). Non sono numeri tondi scelti da me:

| famiglia | shape | tensore vero | byte nel 35B |
|---|---|---|---|
| Q4_K | `[2048, 512]` | expert gate/up (117 tensori con la down) | 17,67 GB |
| Q4_K | `[512, 2048]` | expert down — **due superblocchi per riga** | (idem) |
| Q6_K | `[512, 2048]` | expert down di 3 layer (860.160 B/expert, verificato) | 0,66 GB |
| Q8_0 | `[2048, 4096]` | attn q-proj | 1,09 GB |

## DECISIONE REGISTRATA, non escalata: solo la via intera per queste tre

Le tre famiglie che **non** vengono cablate da questo goal si misurano **solo
sulla via intera**; il fallback f32 si scrive quando avranno un consumatore.
Ragione: una forma senza consumatore misurata oggi e' una forma che nessuno
esegue per mesi, e il costo di riscriverla al momento del cablaggio e' minore
del costo di tenerla in vita nel frattempo. Le due famiglie CABLATE (Q5_K,
Q4_1) hanno entrambe le vie, perche' il fallback lo eseguono davvero i device
senza `packed_4x8_integer_dot_product`.

Se non arrivasse mai un ruling su questo punto, farei esattamente cosi' ⇒ non
e' un'escalation, e' una decisione registrata (§5 del protocollo).

## Le previsioni

**P1 — la regola di stop passa su tutte e tre a M=16, con rapporto ≥ 10×.**
Base: le due famiglie gia' misurate stanno a 22-28×, e il meccanismo del
guadagno non e' l'aritmetica del formato ma la **rilettura M volte** che la
forma legacy fa per costruzione. Un rapporto sotto 10 direbbe che il formato,
non la forma, e' il collo — e sarebbe la scoperta piu' interessante della run.

**P2 — il Q8_0 e' la piu' veloce delle tre in byte al secondo, ma la meno
migliorata in rapporto.** Non ha unpack (i pesi sono gia' i8: il ciclo interno
e' `dot4I8Packed` nudo), quindi la sua forma legacy e' gia' relativamente
efficiente e c'e' meno da recuperare.

**P3 — le shape con K=512 rendono meno di quelle con K=2048, a parita' di
famiglia.** Due superblocchi per riga significano due fette invece di quattro,
cioe' meta' dei workgroup in volo su un device che questo goal ha gia' visto
essere limitato dall'occupancy (la fusione GQA e' stata esclusa proprio cosi').
Se questa previsione cade, il modello «qui il collo e' l'occupancy» va rivisto.

**P4 — nessuna cella verra' scartata dal gate di checksum.** Le tre forme si
confrontano col GEMV di produzione della loro famiglia, e l'unica differenza
aritmetica dichiarata e' la quantizzazione delle attivazioni a 8 bit
(tolleranza 2e-2, gia' usata in riga 1). Una cella scartata significa un bug
nell'unpack, non imprecisione — e in particolare:
  - **Q4_K** e' il Q5_K senza il piano del 5º bit: se sbaglia, l'errore e'
    negli offset (qs a byte 16 invece di 48, superblocco da 36 parole);
  - **Q6_K** e' l'unica con sotto-blocchi da **16** e non da 32, e con i pesi
    centrati su −32. Il termine `−32·Σx` per mezzo sotto-blocco e' il posto
    dove mi aspetto di sbagliare per primo;
  - **Q8_0** e' la piu' semplice: se sbaglia lei, sbaglia il banco.

**P5 — il fabbisogno di memoria di gruppo resta sotto i 16.384 B garantiti da
WebGPU su tutte le forme, a ogni M ≤ 16.** E' gia' verificato in CI per le due
famiglie di it.1; qui vale come previsione perche' il Q6_K aggiunge le somme
per mezzo sotto-blocco, cioe' un array in piu'.

## Regola di stop, per famiglia

Nessuna variante ≥ 1,5× sulla legacy ⇒ quella famiglia si chiude col numero e
**non** si cabla (e per queste tre, che questo goal non cabla comunque, il
numero e' cio' che eredita il goal 35B: una forma misurata, non una da
inventare).
