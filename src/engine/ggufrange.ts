// LA LETTURA A RANGE DEL GGUF, in un posto solo.
//
// PERCHE' ESISTE (goal engine-velocita-decode, riga 2b). Il PI ha chiesto «il
// raggruppamento delle richieste http, se puo' dare un boost globale al
// motore», e il contratto della riga dice che la leva va messa nel path di I/O
// **condiviso** e non nei due call-site del 35B, «altrimenti e' la piccolezza
// specifica che il ruling vieta».
//
// Il path condiviso NON ESISTEVA: la stessa funzione da sette righe era copiata
// in CINQUE posti — `chat.worker.ts`, `q35conf.worker.ts` e tre volte in
// `ktest.worker.ts` — identica a meno del prefisso del messaggio d'errore.
// Aggiungere li' una finestra di concorrenza avrebbe voluto dire scriverla
// cinque volte, che e' il difetto che la regola di riuso di questo progetto
// nomina per primo: alla seconda copia e' una domanda, alla terza non lo e' piu'.
//
// E LA QUINTA COPIA AVEVA GIA' DIVERGITO: `q35-moe-block-real` faceva
//     return new Uint8Array(await rr.arrayBuffer());
// senza il controllo sulla lunghezza che le altre quattro hanno. Un Range corto
// — un server che tronca, una richiesta oltre la fine del file — sarebbe
// passato in silenzio e avrebbe prodotto pesi troncati, cioe' numeri plausibili
// e sbagliati. Non e' teorico: e' la stessa classe del difetto che it.19 ha
// pagato con mezz'ora di GPU. Qui il controllo c'e' per tutti, per costruzione.
//
// DOVE ANDRA' LA LEVA: la finestra di concorrenza e il coalescing delle Range
// adiacenti si mettono QUI, e i cinque chiamanti li ereditano senza toccarli.
// Misurato in it.2 e non assunto: 6,90 ms/fetch con 24 richieste concorrenti
// (`prepLayer`) contro 3,27 ms con qualche centinaio (il repair) — stessi byte,
// stesso server, stesso lettore: cambia solo il raggruppamento.

/** Legge `len` byte dall'offset `off` del GGUF. */
export type GgufRangeReader = (off: number, len: number) => Promise<Uint8Array>;

/**
 * Il lettore per un GGUF servito via HTTP Range.
 *
 * `urlOf` e' una FUNZIONE e non una stringa perche' i worker riassegnano
 * l'URL quando cambiano modello (`URL_GGUF = M.url` dentro `main`), e un
 * lettore costruito a modulo caricato catturerebbe il valore sbagliato.
 *
 * `label` compare nei messaggi d'errore: e' l'unica cosa che distingueva le
 * cinque copie, e resta un parametro perche' su un Range corto sapere QUALE
 * lettore ha fallito e' meta' della diagnosi.
 */
export const ggufRangeReader = (urlOf: () => string, label: string): GgufRangeReader =>
  async (off, len) => {
    const rr = await fetch(urlOf(), { headers: { Range: `bytes=${off}-${off + len - 1}` } });
    if (rr.status !== 206) throw new Error(`${label}: Range non onorato (${rr.status})`);
    const ab = await rr.arrayBuffer();
    // IL CONTROLLO CHE UNA DELLE CINQUE COPIE NON AVEVA. Un Range corto qui
    // lancia; senza, i byte mancanti diventano zeri nel tensore e il modello
    // produce numeri plausibili invece di un errore.
    if (ab.byteLength !== len) throw new Error(`${label}: Range corto ${ab.byteLength}/${len}`);
    return new Uint8Array(ab);
  };
