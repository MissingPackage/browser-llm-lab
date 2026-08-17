# Come misuriamo questo motore

**A chi serve.** A chiunque debba credere — o non credere — a un numero prodotto
da questo progetto. Non presuppone di conoscere il motore né la quantizzazione:
ogni strumento è spiegato con *la domanda a cui risponde*, *cosa non può dire*, e
*il comando per rieseguirlo*.

**Perché è un documento a sé.** Un motore che gira nel browser produce numeri
facili da sbagliare e difficili da smentire: la GPU è condivisa, il modello non
ci sta in memoria, la generazione è stocastica, e un difetto numerico non fa
crashare niente — produce **risposte plausibili e sbagliate**. Quasi tutti gli
strumenti qui dentro esistono perché un numero ci ha ingannato almeno una volta,
e la sezione [§7](#7-le-trappole-che-abbiamo-pagato) elenca quelle volte.

---

## Indice

1. [Il principio: misurare, non dedurre](#1-il-principio-misurare-non-dedurre)
2. [Le quattro domande, e chi risponde a quale](#2-le-quattro-domande-e-chi-risponde-a-quale)
3. [Velocità](#3-velocità)
4. [Intelligenza](#4-intelligenza)
5. [Correttezza del motore](#5-correttezza-del-motore)
6. [Capienza](#6-capienza)
7. [Le trappole che abbiamo pagato](#7-le-trappole-che-abbiamo-pagato)
8. [I principi di metodo, e cosa ci sono costati](#8-i-principi-di-metodo-e-cosa-ci-sono-costati)
9. [Riprodurre tutto](#9-riprodurre-tutto)
10. [Cosa NON misuriamo ancora](#10-cosa-non-misuriamo-ancora)

---

## 1. Il principio: misurare, non dedurre

Tre regole che valgono su ogni numero di questo repo.

**Ogni numero ha un artefatto.** Non si cita un tok/s che non stia in un JSON in
`results/`, con dentro: il file del modello e il suo SHA-256, i parametri di
carico, lo stato dell'host, il contesto, e quali leve erano accese. Un numero
senza queste cose non dice *cosa* ha misurato.

**Un numero dedotto non è un numero misurato.** Il caso che ha fissato la regola:
per giorni «il token pulito costa 21,1 ms» è stato un pilastro di un contratto —
era ottenuto per sottrazione fra due aggregati. Misurato direttamente: **43,6
ms**, cioè il doppio, e la conclusione che ne dipendeva era rovesciata.

**Si dichiara ciò che non si è verificato.** Ogni consuntivo ha una sezione
«NON VERIFICATO». Non è modestia: è che il lettore successivo non ha modo di
distinguere «misurato» da «plausibile» se non glielo diciamo noi.

---

## 2. Le quattro domande, e chi risponde a quale

Il pericolo più comune è usare lo strumento di una domanda per rispondere a
un'altra. Questa tabella esiste per impedirlo.

| domanda | strumento | risposta tipica |
|---|---|---|
| **quanto è veloce?** | chat a 10 turni · banco A/B · sonda GPU | 34,97 tok/s di regime |
| **quanto è intelligente?** | bits/token appaiato · log-prob del bersaglio | +0,13 bit/token contro il quant migliore |
| **il motore è corretto?** | ktest · conformance vs oracolo · audit del trascritto | top-1 99,04%, rango massimo 2 |
| **ci sta in memoria?** | dump dell'header · aritmetica dell'arena | parco 10,391 GiB su 11,17 di arena |

> **La confusione più costosa che abbiamo fatto**: usare un *confronto di
> conversazioni generate* per giudicare la fedeltà del motore. Due decodifiche
> greedy divergono dopo ~15 token anche a implementazioni corrette: da lì in poi
> sono due conversazioni diverse, e confrontarle misura le **traiettorie**, non i
> motori. Per i motori serve il teacher-forcing (§5).

---

## 3. Velocità

### 3.1 La chat a dieci turni — `scripts/chat-smoke.mjs`

**La domanda**: quanti token al secondo vede un utente vero?

**Come funziona.** Pilota la pagina di chat con un browser reale (Playwright),
carica il modello sulla GPU, e conduce **una conversazione di dieci turni**: una
domanda tecnica e nove follow-up che approfondiscono, con un turno di riassunto
che obbliga il modello a rileggere tutto il contesto. Decodifica deterministica
(temperatura 0). Esporta un JSON con, per ogni turno, tok/s, TTFT, token
generati, posizione nel contesto, e i contatori di residenza.

**Perché dieci e non due.** In un modello Mixture-of-Experts i primi turni non
misurano il regime: misurano l'**arena che si riempie**. Il caso reale:

```
turno       1      2      3      4      5      6      7      8      9     10
tok/s   11,51  14,31  33,33  33,47  35,00  35,06  34,98  34,65  34,98  35,17
miss     7012   1413    296    271     67     43     69     41     38     17
```

Con due turni avremmo riportato **1,5×**; il valore vero è **2,95×**. La regola
è: si legge il **regime** (media degli ultimi cinque turni), non la media di
tutti, e si guarda la **curva dei miss** — se tende a zero il modello ci sta in
memoria, se oscilla non ci sta.

**Cosa non dice**: niente sulla qualità delle risposte, e niente su un host
diverso dal nostro. Una run sola non ha repliche: la banda di rumore misurata su
questa macchina è ~2,4%.

### 3.2 Il banco A/B — `scripts/q35-bench-run.mjs`

**La domanda**: quanto vale *questa* leva, isolata da tutto il resto?

Accende e spegne una singola leva **a caldo, nello stesso processo** (`--kfan`,
`--splitk`), così i due bracci condividono cache, driver e stato termico. È il
solo modo di attribuire un guadagno a una causa: due run separate differiscono
anche per cose che non abbiamo cambiato.

**Cosa non dice**: il banco gira in un regime a zero miss che una chat vera non
raggiunge. Il suo 40,06 tok/s e i 34,97 della chat sono entrambi veri e
rispondono a domande diverse.

### 3.3 La sonda per categoria — `--gpu-time`

Ripartisce il tempo del token per categoria di kernel usando i timestamp della
GPU. Serve a rispondere «*dove* va il tempo», e ha già smentito due volte una
diagnosi basata sul conteggio dei dispatch: **contare i dispatch non è misurare
il tempo** — la ricorrenza DeltaNet è il 47,3% dei dispatch e il 5,0% del tempo.

---

## 4. Intelligenza

Qui la parola «intelligenza» va presa alla lettera *e con le pinze*: misuriamo
**quanto il modello è sorpreso dal testo giusto**, che è un proxy — buono, ma un
proxy.

### 4.1 Bits per token, appaiato — `scripts/quant-quality.py`

**La domanda**: quanto costa, in fedeltà, una quantizzazione più aggressiva?

**Come funziona.** Il modello non genera mai: gli si fa leggere un testo fissato
e si misura, posizione per posizione, quanti bit di sorpresa gli costa il token
successivo (`−log₂ P(token)`). Deterministico, nessun campionamento, nessuna
varianza da repliche.

**Perché appaiato.** I due modelli vedono **gli stessi token nelle stesse
posizioni**, e si riporta la differenza *per posizione*. La difficoltà del
singolo token — che è la fonte di rumore dominante — si cancella nella
differenza. Confrontare due medie separate sarebbe molto più rumoroso.

**Perché il bootstrap a blocchi.** I token di un testo non sono indipendenti:
sbagliare una parola rende più probabile sbagliare la successiva. Un errore
standard classico (`σ/√n`) sarebbe ottimista di un fattore ignoto. Il bootstrap
ricampiona **segmenti contigui**, così conserva la correlazione invece di negarla.
Un caso di test lo dimostra: su una serie autocorrelata l'intervallo a blocchi è
più del doppio di quello ingenuo.

**Il corpus** (`scripts/quant-eval-corpus.mjs`) ha quattro domini, e non è
decorazione: il danno di una quantizzazione **non è uniforme**.

| sezione | perché c'è |
|---|---|
| italiano tecnico | è il regime d'uso vero di questo progetto |
| codice | distribuzione molto piccata: mezzo bit sposta l'argmax |
| conversazione | turni, marche del template, contesto che cresce |
| wikitext-2 (en) | il solo numero che qualcuno fuori dal progetto può confrontare |

### 4.2 Log-prob e rango del bersaglio, sui compiti

**La domanda**: il modello sa ancora rispondere?

200 item a risposta verificabile (GSM8K, MMLU). **Non si contano le risposte
giuste**: si misura la log-probabilità e il **rango** della risposta corretta.
Stessa informazione, varianza molto più bassa — e a costo zero, perché è
teacher-forced.

*Il perché è una lezione pagata: con un campione da 22 posizioni, ±1 successo
vale ±4,5 punti percentuali, e due iterazioni si sono perse a inseguire un
rumore.*

### 4.3 Il giudizio in cieco

Tre sistemi rispondono alla stessa conversazione; un agente giudice legge le
risposte **con etichette anonime** (non sa quale sia quale) e rifà i conti a
mano — l'entropia dichiarata, l'albero di Huffman, `log₂6 = 2,585`, «cinque
punti» quando ne sono stati chiesti cinque.

**Cosa non dice, ed è cruciale**: i sistemi condividono il contesto solo al primo
turno. Da lì divergono, e ogni sistema legge **le proprie** risposte precedenti.
Il giudizio classifica **traiettorie**, non implementazioni. È informativo su
«come va una conversazione vera», e non sostituisce §4.1.

---

## 5. Correttezza del motore

### 5.1 Il ktest — `.harness/tools/engine-ktest.mjs`

**La domanda**: questo kernel calcola la cosa giusta?

115 casi che girano su GPU vera e confrontano ogni kernel con un **riferimento
CPU indipendente** (`src/engine/quant.ts`, `src/engine/cpuref.ts`), che a sua
volta è verificato byte per byte contro `llama.cpp`. Un kernel nuovo non è
consegnato finché non ha il suo caso.

### 5.2 La conformance contro l'oracolo — `scripts/q35-conf-run.mjs`

**La domanda**: il motore intero riproduce il modello?

Teacher-forced su una sequenza fissata, confronta il token più probabile con
quello dell'oracolo `llama.cpp`. Soglia: **≥ 1012/1024 = 98,83%**.

**Perché non il 100%.** Aritmetica in virgola mobile con ordini di somma diversi
su hardware diverso: una differenza al quindicesimo bit può ribaltare due
candidati quasi pari. La soglia non è indulgenza, è la constatazione che
l'identità bit-per-bit fra GPU e CPU non è ottenibile.

### 5.3 L'audit del trascritto — `scripts/transcript-audit.py`

**La domanda**: ciò che il *nostro* motore ha scritto, quanto era improbabile per
il riferimento?

Si prende il testo che il motore ha generato e lo si fa **rileggere** a
`llama.cpp` teacher-forced, chiedendo per ogni token quale **rango** gli dava.

```
rango 1        il riferimento avrebbe scritto lo stesso token
rango basso    scelta legittima, era fra le sue candidate
rango enorme   abbiamo scritto qualcosa che il riferimento non contemplava
```

È lo strumento che ha sciolto il dubbio più serio della sessione. Il motore
chiudeva i turni tardivi dopo 64-135 token mentre il riferimento arrivava a 400,
e sembrava un difetto. Misurato:

```
2.177 token · top-1 99,04% · rango mediano 1 · RANGO MASSIMO 2
il token di chiusura <|im_end|>: rango 1 in 6 casi su 7, probabilità 41-85%
```

**Chiudere era la scelta del modello, non un difetto del motore.**

### 5.4 Le sequenze identiche e la bit-identità

Dove l'identità *è* ottenibile, si pretende esatta: due percorsi interni allo
stesso motore devono dare gli stessi byte. Esempi in uso: le sequenze generate
identiche 8/8 fra due configurazioni; la bit-identità del prefill MoE col
percorso sequenziale; e il gate della rifattorizzazione dei kernel K-quant —
**il WGSL emesso dev'essere identico carattere per carattere** a quello di prima,
con le stringhe di riferimento catturate *prima* della modifica (e verificate
rigenerandole da un worktree del commit precedente, perché uno snapshot
rigenerato dal codice nuovo sarebbe un test vuoto).

---

## 6. Capienza

### `scripts/q35-header-dump.mjs`

**La domanda**: questo modello ci sta, e che kernel richiede?

Legge l'header di un GGUF e produce l'istogramma dei tipi **per categoria di
tensore**, separando il **parco expert** — l'unica parte che si contende gli slot
in VRAM — da tutto il resto.

Due proprietà che lo rendono utile:

- **legge un URL**, non solo un file: l'header sta nei primi MB, quindi valutare
  un modello candidato costa **64 MB invece di 15 GB**;
- **pesa anche i formati che il motore non sa leggere**, e li marca. È così che
  abbiamo scoperto che un file chiamato `Q2_K_XL` non contiene un solo tensore
  `Q2_K` fra gli expert: sono i-quant a codebook, cioè un'altra famiglia di
  kernel. *Il nome del file mente, l'header no.*

La tabella di ripiego dei formati sconosciuti **si verifica da sola**: la somma
dei byte di tutti i tensori contro la taglia vera del file. Scarto osservato:
0,06-0,09% (header e allineamenti).

---

## 7. Le trappole che abbiamo pagato

Ognuna ha prodotto numeri credibili e sbagliati.

**Metriche plausibili su testo spazzatura.** Il runner di riferimento leggeva i
logit da un buffer che la libreria, in quella configurazione, *non scrive
affatto*. Risultato: 400 token di spazzatura con **12,14 tok/s perfettamente
plausibili**. Ora legge il puntatore vero, e c'è una guardia sulla varietà
lessicale che fa fallire subito una risposta degenere.

**Due turni invece di dieci.** Vedi §3.1: 1,5× riportato dove il vero era 2,95×.

**Il corpus vivo.** Una sezione del corpus pescava da un file del repo che
continuavamo a modificare: fra due giornate il corpus era cambiato sotto di noi.
Lo SHA lo registra e il comparatore **rifiuta** corpus diversi — nessun confronto
sbagliato è possibile — ma un corpus che deve reggere confronti fra giornate va
congelato.

**Il trascritto auto-generato.** La sezione «conversazione» del corpus è output
del braccio di riferimento: misurarlo su sé stesso dà una sorpresa bassissima
(0,77 bit/token) e potrebbe sbilanciare il confronto a suo favore. Il delta è
atterrato dove sono atterrate le sezioni non sbilanciate, quindi la conclusione
regge — ma la prossima versione userà un trascritto che nessuno dei bracci ha
prodotto.

**Il flag dedotto invece che letto.** Tre volte un runner è stato invocato con
parametri supposti: una run intera buttata (prefill sequenziale invece che a
chunk, 91 s contro 17), e un `--help` dato a un runner che non lo conosce, che ha
eseguito il bench coi default **uccidendo una misura in corso**. Regola: *i flag
di un runner si leggono dal sorgente*.

**Il comando lasciato per la ripresa e mai eseguito.** Un comando documentato in
un journal non partiva affatto (`ERR_MODULE_NOT_FOUND`). Regola: *un comando che
si lascia scritto si esegue come sta scritto, prima di lasciarlo*.

**`pgrep` che trova sé stesso.** Verificare che il server di sviluppo sia vivo
con `pgrep -f "vite --port 5199"` dà sempre esito positivo, perché la riga di
comando del `pgrep` contiene la stringa cercata. Si verifica con `curl`.

---

## 8. I principi di metodo, e cosa ci sono costati

| principio | l'errore che l'ha generato |
|---|---|
| **Appaiare, non confrontare medie** | la varianza fra token dominava il segnale |
| **Bootstrap a blocchi** | i token sono correlati; `σ/√n` mente |
| **Rango e log-prob, non conteggi** | ±1 successo su 22 valeva ±4,5 punti |
| **Dieci turni** | due misurano il transitorio, non il regime |
| **Dichiarare il fallback nell'artefatto** | due leve misurate erano spente in produzione, e niente lo diceva |
| **Mai troncare in silenzio** | una patch tagliata a metà sembrava un difetto di progettazione |
| **Ogni tabella si verifica da sola** | una taglia di blocco sbagliata si vede solo confrontando la somma col file |
| **Un allow-list che si ferma rumorosamente** | un tipo sconosciuto letto come noto darebbe pesi plausibili e sbagliati |

---

## 9. Riprodurre tutto

```bash
# server (staccato: muore al confine di turno se lanciato normalmente)
setsid nohup npx vite --port 5199 > /tmp/vite-5199.log 2>&1 < /dev/null &
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5199/chat.html   # 200, non pgrep

# correttezza (GPU)
BASE_URL=http://localhost:5199 node .harness/tools/engine-ktest.mjs

# velocità: la chat a dieci turni
BASE_URL=http://localhost:5199 node scripts/chat-smoke.mjs \
  --model 35b-q2k --ctx 8192 --vram 13 --maxnew 400 --turns 10 \
  --out results/chat/<nome>.json

# intelligenza: corpus, misura per modello, confronto appaiato
node scripts/quant-eval-corpus.mjs --out results/eval/quant-corpus.json
uv run scripts/quant-quality.py --model <A.gguf> --corpus results/eval/quant-corpus.json --out A.json
uv run scripts/quant-quality.py --model <B.gguf> --corpus results/eval/quant-corpus.json --out B.json
node scripts/quant-quality-compare.mjs A.json B.json --out confronto.json

# riferimento: la stessa conversazione su llama.cpp, e l'audit del trascritto
uv run scripts/chat-llamacpp.py --model <gguf> --out results/chat/<nome>.json
uv run scripts/transcript-audit.py --model <gguf> --chat results/chat/<nostro>.json --out audit.json

# capienza: un candidato remoto costa 64 MB
node scripts/q35-header-dump.mjs https://huggingface.co/<repo>/resolve/main/<file>.gguf
```

**Due vincoli di esecuzione**, entrambi pagati: mai due runner GPU insieme (la
scheda è una), e mai un bench mentre gira un workflow di agenti (contesa).

---

## 10. Cosa NON misuriamo ancora

Sezione deliberata: la mappa dei buchi vale quanto quella degli strumenti.

- **Un solo dispositivo.** Tutti i numeri vengono da una macchina (una GPU
  desktop, Chrome, Linux). Non esiste una matrice di dispositivi: cosa succede
  con 8 GB di VRAM, su GPU integrata, su Metal, su un browser diverso.
- **Nessuna replica sulle misure di chat.** Una run per braccio. La banda di
  rumore è nota (~2,4%) e i confronti riportati la superano largamente, ma
  «tre repliche» non è ancora una pratica.
- **Il contesto lungo è misurato fino a ~8k token**, mentre il modello ne
  dichiara 262k. Nessuna affermazione oltre ciò che è stato misurato.
- **La qualità è misurata su un modello e due quantizzazioni.** Nessun confronto
  con altri modelli della stessa taglia.
- **Non misuriamo il tempo di primo utilizzo reale**: il download dei pesi
  (13-21 GB) sta fuori da ogni cronometro di questo repo.
