# Architettura del motore — diagramma

`motore-architettura.excalidraw` — apribile su [excalidraw.com](https://excalidraw.com)
(*Apri* → scegli il file) o con l'estensione Excalidraw di VS Code. Tutti gli
elementi sono modificabili: è un disegno, non un'immagine.

`motore-architettura.svg` — la stessa cosa in sola lettura, per guardarla senza
aprire niente.

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
