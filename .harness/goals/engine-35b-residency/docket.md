# DOCKET — engine-35b-residency

Le decisioni REGISTRATE E NON PRESE, in attesa di ruling del PI. Ordine e
meccanismo non stanno qui: quelli li decido io e finiscono nel journal.

## item 1 — host non quiescente: la baseline non è eseguibile adesso (io → PI, it.1)

**La riga 1 è ferma sulla sua seconda metà, e non è un problema di codice.**
La strumentazione è in albero e verde (vitest 1025 passed, tsc exit 0). Ciò che
manca è la misura, e l'host non è nello stato in cui una misura vale.

**Osservato**, `pgrep -af`: quattro istanze di `@playwright/mcp` con Chrome
vivo, fra cui un `--type=gpu-process` con `--enable-unsafe-swiftshader`. Più
`free -g`: 9 GB di RAM usati su 31, 21 disponibili. Il 35B chiede ~13 GiB di
VRAM e l'arena si dimensiona su ciò che resta.

**Perché non parto lo stesso.** Due ragioni, e la seconda è quella che pesa:
1. Il precedente esiste ed è documentato: `.harness/goals/engine-kquant/docket.md`
   item 2 registra cinque fallimenti consecutivi con tre sintomi diversi
   (`Failed to fetch`, `Target crashed`, device WebGPU perso) su un ambiente
   sporco, costati un gate intero.
2. **Una baseline presa su host non quiescente non è una baseline.** Il
   contratto pretende `hostState.declared = "quiescent"`, e la regola del
   progetto è che ogni misura dichiari il suo host. Prenderla adesso vorrebbe
   dire scrivere `"declared": "non quiescente"` su un artefatto che tutte le
   righe successive useranno come termine di paragone.

**Cosa serve, ed è una riga di comando che non tocco io**: chiudere le sessioni
Chrome/MCP di Playwright. Sono sessioni del PI, non mie, e non le termino senza
che me lo dica.

**Cosa NON è bloccato nel frattempo**: nulla di utile. La riga 2 dipende da
C0-4, C0-4 dipende dalla misura, la misura dipende da questo item. È uno
stop-by-design, non una pausa.

**RULING:** _

## item 2 — `blankNonCode` è duplicato in sette file di test (io → PI, it.1)

Trovato scrivendo `tests/engine-35b-repair-counters.test.ts`. La funzione che
bianca commenti/stringhe/template prima di scansionare il sorgente — ~90 righe —
è **copiata identica in sette file**: `engine-q35attnwiring`,
`engine-ktest-q41-wiring`, `engine-subgroups-feature`, `engine-ktest-q5k-wiring`,
`engine-prefillwiring-q5k`, `engine-prefillpbcat-q41`,
`engine-ktest-kquant35b-wiring`. `tests/helpers/` esiste già e ospita due
fixture, quindi la sede c'è.

**Non l'ho fatto**, e la ragione è la regola sui bench: un'estrazione tocca
sette file di test in un commit che deve restare leggibile come «la
strumentazione del repair». Nel mio test ho evitato la duplicazione con uno
stripper di soli commenti (15 righe), sufficiente lì perché gli identificatori
cercati non compaiono in letterali — ma è un'ottava variante, non una soluzione.

**Registrato secondo il ruling «sistemare, non recintare» (2026-08-14)**: il
difetto va tolto, non sorvegliato. Chiedo solo se toglierlo QUI (fuori brief,
sette file) o come item d'igiene a sé.

**RULING:** _
