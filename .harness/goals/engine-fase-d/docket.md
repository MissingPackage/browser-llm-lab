# Docket — engine-fase-d

1. **PLAN-CHECK — PRE-AUTORIZZATO (2026-08-10, PI in chat: "vai" sul
   contratto presentato).** PHASES.md è su disco al tag di apertura: 9 fasi
   sequenziali, authority delta = none ovunque, nessuna docket-born. Blocco A
   = fasi 1-6 (merge gate autonomo), blocco B = 7-8, chiusura = 9. Se vuoi
   rivederla, il punto giusto è PRIMA della fase 2 (la fase 1 è
   parametrizzazione pura, reversibile e coperta dal ktest bit-exact).
   REGOLA DI MISURA nel piano: bench pieni SOLO alle fasi 6 e 8.

2. **REGISTRAZIONE (2026-08-10, it.1)** — i campi piatti di `SlabLayout`
   (`gateQs`, `gateScales`, `upQs`, `upScales`, `downQs`, `downScales`,
   `qsBytes`, `gateScalesBytes`, `downScalesBytes`, `downKind`) restano come
   COMPAT DERIVATA per non toccare ~100 call site GLM in un colpo solo. Non
   sono una seconda fonte di verità (derivano dai tre `SlabTensorLayout`).
   Da rimuovere quando i call site migrano alla vista generica — lavoro di
   igiene interno al goal, non un debito verso l'esterno.

3. **ECCEZIONE DICHIARATA E PERMANENTE (2026-08-10, it.1)** — il router
   del CPUREF (`q35cpurefmodel.ts`) resta un'implementazione INDIPENDENTE:
   è il riferimento del differential testing, se usasse `routerSelect` un
   bug del router sarebbe invisibile al confronto. Il gate strutturale la
   elenca come eccezione, non come debito.
