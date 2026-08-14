/**
 * LA BANDA DEL BANCO `dense-batch-attn-chunk` — una soglia, un posto.
 *
 * Il ramo `batch` di `attnDecodeWgsl` (l'attenzione a chunk del prefill) fa la
 * softmax in STREAMING a tile di 64 posizioni, quindi somma in un ordine diverso
 * dal riferimento per-riga legacy: **non è bit-identico per costruzione**, ed è
 * la stessa situazione già accettata sulla riga 1 di `engine-kernel-decode`.
 * Il confronto è a tolleranza, e la tolleranza è DICHIARATA PRIMA di guardare i
 * numeri del device (AC3 della riga 3 di `engine-ttft`).
 *
 * DERIVAZIONE — simulazione f32 in lock-step sugli STESSI due casi del banco,
 * cioè i numeri che giustificano queste due costanti e non altre:
 *
 *     un tile      (n = 10..12):   max assoluto 2,98e-8 · max relativo 4,26e-6
 *     cinque tile  (n = 301..303): max assoluto 1,68e-8 · max relativo 3,95e-5
 *
 * `compare` passa se UNA delle due bande regge, e quella assoluta è lì per le
 * componenti vicine a zero, dove il relativo esplode senza significare niente.
 *
 * ⚠ IL MARGINE VERO È MOLTO PIÙ STRETTO DI QUELLO SIMULATO, e va saputo prima
 * di ritoccare questi numeri. La derivazione sopra è una simulazione su CPU,
 * che NON modella la fma del device. Sul 4090, ktest del 2026-08-14:
 *
 *     un tile      maxAbs 1,49e-8 · maxRel 4,91e-6   (simulato 4,26e-6)
 *     cinque tile  maxAbs 1,77e-8 · maxRel 8,44e-5   (simulato 3,95e-5)
 *
 * Sul caso multitile il device è **2,14× peggiore della simulazione**, e la
 * banda relativa 1e-4 gli lascia solo il **18% di margine** — non il 2,5×
 * che la simulazione lasciava sperare. La banda assoluta invece è larghissima
 * (~565×) e regge da sola. Conseguenza pratica: questo gate è passato per la
 * banda assoluta, e chi stringesse la relativa senza rimisurare lo farebbe
 * diventare rosso su un cambio di driver. Chi la allarga, invece, deve dire di
 * quanto e perché.
 *
 * PERCHÉ VIVONO QUI E NON NEL BANCO. Il task T1 le aveva lasciate come due
 * costanti locali in `ktest.worker.ts`, e T2 — che doveva dar loro una sede
 * unica — è finito BLOCKED per ownership sovrapposta (docket item 23): i due
 * task scrivevano lo stesso blocco nella stessa ondata. Una soglia in due posti
 * è il difetto che è già costato la riga 2 di questo goal, dove
 * `prefillgemmplan.ts` esisteva e nessuno lo importava. La forma è quella già
 * usata in casa da `KQUANT_FAST_Q5K_PAIR_REL_TOL` (`kquantfast.ts`).
 *
 * SPOSTATE, NON RITARATE: i valori sono quelli di T1, con la loro derivazione.
 * Cambiare un numero è un'altra decisione e vuole un'altra misura.
 */
export const ATTN_CHUNK_REL_TOL = 1e-4;
export const ATTN_CHUNK_ABS_TOL = 1e-5;
