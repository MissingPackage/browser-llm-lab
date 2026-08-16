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
 * CONTATORI DEL LETTORE (riga 2b, it.34), e servono a una domanda precisa.
 *
 * it.32 e it.33 hanno ESCLUSO il trasporto con due esperimenti controllati: il
 * canale da' 460-740 MB/s in ogni configurazione provata, mentre il `prep` del
 * motore ne ottiene 250. Il deficit e' dentro il path, e per trovarlo servono
 * due numeri che finora ho solo stimato:
 *
 *   - **quante richieste sono DAVVERO in volo** (avevo scritto 24 leggendo il
 *     caso peggiore del contratto; il conto sui contatori dice ~7, e nessuno
 *     dei due e' una misura);
 *   - **il parallelismo EFFETTIVO** = somma delle durate / durata di parete.
 *     Vale ~1 se le letture sono di fatto serializzate, ~N se sono parallele.
 *     E' la sola cifra che distingue «il canale e' lento per noi» da «le nostre
 *     letture non sono parallele come crediamo».
 *
 * Stanno a livello di modulo perche' il lettore e' uno solo (it.30) e i suoi
 * cinque chiamanti non devono saperne niente. Il costo e' qualche incremento
 * per range da 594 KB: sotto il rumore.
 */
export interface GgufRangeStats {
  calls: number; bytes: number; fails: number;
  /** somma delle durate delle singole letture */
  sumMs: number;
  /** la piu' lunga: con `Promise.all` e' lei a decidere la parete della raffica */
  maxMs: number;
  inFlight: number; maxInFlight: number;
  /** inizio della prima lettura e fine dell'ultima, per la durata di parete */
  firstStart: number; lastEnd: number;
}

const zero = (): GgufRangeStats => ({
  calls: 0, bytes: 0, fails: 0, sumMs: 0, maxMs: 0,
  inFlight: 0, maxInFlight: 0, firstStart: 0, lastEnd: 0,
});

export const ggufRangeStats: GgufRangeStats = zero();

/** Azzera i contatori e restituisce quelli di prima (per finestre di misura). */
export const takeGgufRangeStats = (): GgufRangeStats => {
  const snap = { ...ggufRangeStats };
  Object.assign(ggufRangeStats, zero());
  return snap;
};

/**
 * Parallelismo effettivo di una finestra: somma delle durate diviso la parete.
 * `null` senza letture o senza parete misurabile — mai un 1 finto su zero dati.
 */
export const effectiveParallelism = (s: GgufRangeStats): number | null => {
  const wall = s.lastEnd - s.firstStart;
  return s.calls > 0 && wall > 0 ? s.sumMs / wall : null;
};

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
    const t0 = performance.now();
    if (ggufRangeStats.firstStart === 0) ggufRangeStats.firstStart = t0;
    ggufRangeStats.inFlight++;
    if (ggufRangeStats.inFlight > ggufRangeStats.maxInFlight) {
      ggufRangeStats.maxInFlight = ggufRangeStats.inFlight;
    }
    const done = (ok: boolean, bytes: number): void => {
      const dt = performance.now() - t0;
      ggufRangeStats.inFlight--;
      ggufRangeStats.calls++;
      ggufRangeStats.sumMs += dt;
      if (dt > ggufRangeStats.maxMs) ggufRangeStats.maxMs = dt;
      ggufRangeStats.bytes += bytes;
      if (!ok) ggufRangeStats.fails++;
      ggufRangeStats.lastEnd = performance.now();
    };
    let rr: Response;
    try { rr = await fetch(urlOf(), { headers: { Range: `bytes=${off}-${off + len - 1}` } }); }
    catch (e) { done(false, 0); throw e; }
    if (rr.status !== 206) { done(false, 0); throw new Error(`${label}: Range non onorato (${rr.status})`); }
    const ab = await rr.arrayBuffer();
    done(ab.byteLength === len, ab.byteLength);
    // IL CONTROLLO CHE UNA DELLE CINQUE COPIE NON AVEVA. Un Range corto qui
    // lancia; senza, i byte mancanti diventano zeri nel tensore e il modello
    // produce numeri plausibili invece di un errore.
    if (ab.byteLength !== len) throw new Error(`${label}: Range corto ${ab.byteLength}/${len}`);
    return new Uint8Array(ab);
  };
