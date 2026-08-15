# JOURNAL — engine-35b-residency

## it.0 — charter (2026-08-15)

Goal chartato dai contatori di `results/chat/chat-35b-2026-08-15T16-25-12.json`.
`GOAL.md` + `PHASES.md` scritti. Sette righe, barra decode 35B ≥ 30 tok/s.
Tre correzioni di fattibilità nel test-fit (C0-1, C0-2, C0-3).

## it.1 — la riga 1, prima metà: i contatori (2026-08-15)

**Fatto.** Sette contatori nuovi in `q35gpumodel.ts`, propagati a
`chat.worker.ts` e `q35conf.worker.ts`. Commento di `perf()` corretto (—84%).
`tests/engine-35b-repair-counters.test.ts`, 6 casi. vitest 1025 passed, tsc
exit 0. Commit `2f1637b`.

**Ritrattato nello stesso ciclo (C0-4).** «Token pulito = 21,1 ms» non regge:
`readbackMs`/pass 37,4 ms > 21,8 ms di porzione non-tail, e `q35-vramplan-35b-it35`
misura 43,74 ms su un token con 0 miss. Commit `967f928`. La riga 2 non parte
finché il token pulito non è misurato.

**Fermo sul docket item 1**: host non quiescente, quattro Chrome/MCP di
Playwright vivi. Sono della sessione `personal-site-47`; il PI ha chiesto di
accordarmi con lei invece di chiudere. Messaggio inviato, in attesa.

---

## PRONTO PER LA FINESTRA — i due comandi, verificati dal sorgente

Preparati mentre aspetto, così la finestra si spende eseguendo. **Nessuno dei
due è stato eseguito.** Flag letti da `scripts/q35-conf-run.mjs:11-40` e da
`src/engine/q35conf/q35conf.worker.ts:575-637` — mai chiedendo `--help` al
runner, che non lo conosce e partirebbe coi default (landmine di `HANDOFF.md:62`).

**Precondizione**: server già vivo e verificato con `curl` (200 su
`http://localhost:5199/`), mai con `pgrep`. Un runner GPU alla volta: A poi B,
mai insieme.

### Run A — il discriminante di C0-4 (~2 min)

    node scripts/q35-conf-run.mjs --model 35b --golden-kind smoke \
      --optimistic --arena-gib 12 \
      --out results/engine/q35-optimistic-35b-cleantoken-2026-08-15.json

Riproduce la forma di `q35-vramplan-35b-it35.json` sull'albero di oggi. **39
token** (34 prompt + 5 generati, golden `smoke` — è da lì che vengono i 39 di
it.35, verificato: il `full` ne ha 6.461). Arena 12 GiB **esplicita**: il report
scrive `cfg.arenaGiB ?? 12`, quindi senza il flag dichiarerebbe 12 anche se il
motore ne usasse un altro — sarebbe una dichiarazione non misurata.

Protocollo del runner, già di grado riferimento: cold sync che scalda, poi
REPS=4 coppie interleavate sync/optimistic sulla stessa cache, **prima coppia
scartata**, mediana e dispersione riportate (`q35conf.worker.ts:592-605`).

**Cosa leggo**: `passes[optimistic-warm].cpu.tokenMs`, con `misses: 0` e
`dirtyTokens: 0` a confermare che è un token pulito.
- **~43 ms** ⇒ C0-4 è confermata, il contratto si riapre sulla quarta leva
  (`readbackWait` era il 94%) PRIMA della riga 2.
- **~21 ms** ⇒ le righe 2 e 3 bastano, si procede come chartato.

### Run B — validare i contatori nuovi su GPU vera (~2 min)

    node scripts/q35-conf-run.mjs --model 35b --golden-kind smoke \
      --optimistic --arena-gib 4 \
      --out results/engine/q35-optimistic-35b-arena4-2026-08-15.json

Stessa forma, **arena strozzata a 4 GiB**: con 39 token e un'arena che non
contiene il working set i miss ricompaiono anche su una run corta, che è il modo
economico di far girare il path di repair. Serve perché oggi la strumentazione è
scritta e **non ha mai girato una volta su hardware**: i test sono statici e non
si accorgerebbero di un contatore piazzato nel posto sbagliato.

**Cosa leggo**: `passes[*].repair` — `namedFrac` (≥ 0,95 è il done-when),
`msPerFetch` (oggi 5,98 dai contatori della chat), `accountingMs`,
`fetchRepairCalls` contro `misses` del blocco `moe` (devono coincidere: un
`readExpert` per miss).

### Cosa NON è in questi due comandi, e perché

La **baseline nel regime sporco vero** (89,7% di token sporchi) vuole una
generazione lunga, non 39 token: al `full` sono 6.461 token e il solo pass
sync-cold sarebbe ~2 ore. Non entra in una finestra di 5 minuti e non serve per
C0-4. Va pianificata a parte, con `--max-gen` scelto sui numeri di A e B.
