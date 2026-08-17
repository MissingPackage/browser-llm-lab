# Architettura del motore — diagramma

`motore-architettura.excalidraw` — apribile su [excalidraw.com](https://excalidraw.com)
(*Apri* → scegli il file) o con l'estensione Excalidraw di VS Code. Tutti gli
elementi sono modificabili: è un disegno, non un'immagine.

`motore-architettura.svg` — la stessa cosa in sola lettura, per guardarla senza
aprire niente.

## Il file che va letto PRIMA di progettare

`MECCANISMI.md` — la tabella **meccanismi × path**: cosa esiste, chi lo usa,
qual e' il default, cos'e' gia' misurato. Serve a rendere visibili le **celle
vuote**, perche' un grep dice cosa c'e' e non cosa manca. E' tenuto onesto da
`tests/engine-meccanismi.test.ts`, che confronta le sue affermazioni strutturali
col sorgente: se qualcuno adotta un meccanismo e non aggiorna la riga, il test
fallisce.

Il diagramma qui sotto mostra come i pezzi si parlano; la tabella mostra chi non
li usa. Servono a domande diverse.

## Il file da leggere prima di scegliere un modello o scrivere un kernel

`QUANTIZZAZIONE.md` — la spiegazione per intero di come sono fatti i pesi
quantizzati: le tre famiglie (legacy, K-quant, i-quant), i layout di blocco
verificati sul nostro `quant.ts`, la differenza fra un **formato** e una
**ricetta** (`_S`/`_M`/`_L`/`_XL`/`UD`), la tabella dei bit per peso, e cosa
cambia per un motore che dequantizza dentro lo shader. Non presuppone di sapere
cosa sia un quant.

Serve a due decisioni ricorrenti: **quale file scaricare** (la taglia del file
non dice quanto pesa il parco expert, che è l'unica parte che si contende la
VRAM) e **quanto costa un formato nuovo** (dequant di riferimento, kernel, layout
di slab, caso di conformance). Chiude con la sezione «cosa NON è verificato in
casa», che separa ciò che abbiamo misurato da ciò che abbiamo letto.

## Come si legge

Sette bande, dall'alto: chi entra (**A**), come si leggono i byte (**B**), i due
modelli e i kernel condivisi (**C**), come si costruisce il piano dei dispatch
(**D**), la catena di un token di decode coi ms misurati (**E**), la residenza
degli expert (**F**), il ciclo ottimistico (**G**).

**I numeri sono misurati, non stimati**, e vengono dagli artefatti in
`results/engine/`: il profilo per categoria da `q35-splitk-gputime-2026-08-16.json`,
il token e il gate da `q35-splitk-floor-2026-08-16.json`, la banda del canale da
`q35-io-locality-2026-08-16.json`.

## Cosa NON è nel disegno, di proposito

- **Il prefill.** Ha un piano gemello (`pushB` → `gemvB`, M=16) e una sua
  decomposizione per segmento. Metterlo accanto al decode avrebbe raddoppiato le
  bande per una parte che questo goal ha dichiarato fuori scope.
- **La tokenizzazione, il template di chat, la KV cache.** Sono a monte e a
  valle del pezzo che decide i tok/s.
- **Le famiglie 4B e 9B in dettaglio.** Hanno la stessa forma del 35B senza il
  blocco MoE (niente banda F, niente `router`/`expert` nella catena).

## Rigenerarlo

Il file è generato: se i numeri cambiano, si rigenera invece di editarlo a mano
(lo script vive nella cronologia della sessione che l'ha prodotto). Il generatore
**asserisce che nessun riquadro si sovrapponga** — un diagramma che si genera da
sé e si controlla da sé non marcisce in silenzio come un'immagine.
