---
name: bottleneck-brainstorm
description: Use when writing the "Bottleneck & vie d'uscita" section of a docs/deep-dive/ document, or whenever a technical limit (buffer cap, memory ceiling, thermal/DVFS variance, WASM wall) needs workaround ideas evaluated for the browser-LLM case.
---

# Bottleneck Brainstorm

## Overview

Trasforma un limite tecnico osservato in idee di aggiramento **rubabili e adattabili** al
caso browser — non un censimento di "chi fa cosa". Il valore sta nell'adattamento: cosa
cambierebbe portando quella tecnica dentro i vincoli di WebGPU/WASM/OPFS.

## Processo

1. **Inchioda il bottleneck ai numeri.** Prima di ideare: il limite citato con un run reale
   (`results/*.json`) e/o una riga di codice (bundle con versione pacchetto). Se il numero
   "noto" non è verificato nel repo, verificalo ora — la baseline di fase 2 ha già mostrato
   che il valore di targa può mentire (probe ~2 GiB vs richiesta hardcoded 1 GiB).
2. **Sweep prior-art su almeno 3 famiglie diverse**, non solo la più ovvia:
   - motori nativi: llama.cpp, vLLM, ExecuTorch;
   - motori browser: ONNX Runtime Web, WebNN, MediaPipe;
   - sistemi fuori dominio: `antirez/ds4` (SSD streaming), FlexGen, DeepSpeed-Inference,
     tecniche OS (paging, prefetch, DVFS warm-up).
   Docs correnti (context7/repo), mai a memoria.
3. **Genera includendo almeno un'idea fuori dagli schemi** — un trasferimento da un dominio
   non-LLM o una combinazione non vista nello sweep. Può essere scartata, ma deve esistere
   ed essere valutata come le altre.
4. **Valuta e instrada** ogni idea secondo il contratto di output qui sotto.

## Contratto di output (la sezione È questa forma)

1. Sottosezioni `###` per ogni bottleneck distinto, con i numeri inchiodati (passo 1).
2. **Tabella unica** delle idee, colonne esatte:
   `| Idea | Prior art | Fattibilità / costo | Rischio | Instradamento |`
3. `Instradamento` ha uno di tre valori: `esperimento` (candidato per i max 2 esperimenti
   di fattibilità del goal — diventa voce docket, mai eseguito d'impulso), `engine-notes`
   (richiede rework del motore: destinazione `engine-design-notes.md`), `scartata` (con la
   ragione nella cella).
4. Paragrafo di chiusura: quale idea proporresti come esperimento e perché lei — una sola
   raccomandazione, motivata.
5. Ogni claim citato (codice+versione, run, fonte upstream) o marcato `[VERIFY]`.

## Errori comuni

- Idea valutata su un numero di targa non verificato nel repo (passo 1 saltato).
- Sweep monofamiglia (solo llama.cpp) — le idee migliori della baseline venivano dai
  trasferimenti (vLLM PagedAttention, ds4, prassi mobile DVFS).
- Eseguire un esperimento direttamente dalla sezione: l'instradamento produce una voce
  docket, il tetto (2 per goal) vive nel contratto del goal.
