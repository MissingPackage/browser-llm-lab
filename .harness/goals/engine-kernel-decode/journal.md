# JOURNAL — engine-kernel-decode

## it.1 (2026-08-13, fase 0 + fase 1 prep) — la fase 0 gira; intanto chiudo la trappola dei limiti

Fase 0 lanciata col veicolo dichiarato (`research-campaign`, prediction-gated:
pre-registra → esegui → grade indipendente → memo). Gira in background sul
micro-bench isolato, zero run di modello.

**Intanto, lavoro della riga 1 che NON dipende da quale variante vince**: il
modulo dei limiti diceva il falso sul path Qwen.

**E la mia formulazione dell'item 2 era più grave del vero.** Verificando prima
di scrivere il fix: il path q35 di produzione non passa mai `mlaAttention:
false`, quindi il limite giusto lo otteneva già — ma attraverso un campo che
porta il nome dell'attenzione di GLM, mentre il proprio consumatore restava
invisibile. Non un bug vivo: una trappola, e l'invito a caderci era scritto nel
commento ("un consumatore che quel modello non ha", falso).

**Fix, tre pezzi.** `attnDecodeWorkgroupStorageBytes(ctxMax)` esportata dal file
del KERNEL e non ricopiata nei limiti — una formula sola, dove sta il
consumatore, perché il difetto era esattamente due posti che divergevano. Il
modulo dei limiti la conta SEMPRE, indipendentemente da `mlaAttention`. E un
test che chiude la trappola: spegnere l'MLA non può più far sparire il
fabbisogno di Qwen, a nessun ctxMax, e il valore deve crescere col contesto —
cioè la frase "path Qwen indipendente dal contesto" è ora falsificabile da un
test invece che da una lettura attenta.

Gate: `npx tsc --noEmit` pulito, suite **443|10** (era 442|10: +1 test nuovo),
`tests/gpulimits.test.ts` 19/19.

Nota di traiettoria: questa iterazione NON muove la metrica obiettivo (9,95
tok/s a ctx 6333) ed è dichiarata come tale — è un done-when della riga 1
(portabilità) che si poteva chiudere in parallelo alla fase 0 invece di
aspettarla.
