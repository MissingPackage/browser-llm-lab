# Docket — fase-2-deep-dive

Decisioni PI-gated e questioni aperte. Append-only; le decisioni le prende Cristiano.

1. **plan-check** (aperto 2026-07-27, iterazione 0). Goal di prodotto → PHASES.md va
   approvato dal PI prima dell'iterazione 1. **Nota**: il PI ha pre-autorizzato in chat
   l'avvio del product-loop subito dopo il setup ("facciamo goal per fare il setup e una
   volta ottenuto tutto parti con il product-loop", 2026-07-27) — la tabella fasi è stata
   mostrata in chat nel digest di setup. Trattato come approvazione condizionale: il loop
   parte, ma qualunque obiezione del PI alla tabella riapre questa voce e la decomposizione
   si corregge via ruling prima di proseguire.

2. **Candidato esperimento di fattibilità: swap `q4f32_1` → `q4f16_1` su S22** (aperto
   2026-07-27, iterazione 1, emerso dal test GREEN della skill). Il probe S22 espone
   `shader-f16` ma la config bench fissa `q4f32_1` ovunque (`scripts/seq-bench.mjs:20`,
   `src/conformance/page.ts:26`). Costo quasi zero (un modelId), nessun tocco a
   `webllm.ts`; chiuderebbe il buco "quanto del gap 4090→S22 è banda intrinseca e quanto
   percorso f32 non necessario". Consuma 1 dei 2 slot esperimento del goal e il run S22 è
   manuale (mani di Cristiano) → decisione PI: approvare questo come esperimento #1?
   (La sezione completa con l'analisi è in `baseline/run-C-green-dequant.md`.)
