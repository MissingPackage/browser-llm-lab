// Residenza minima expert (goal C2 fase 5, spec §5): cache VRAM a slot con
// policy LRU PURA, due size-class esatte (down Q4_0 blk.5-46 / down Q4_1
// blk.1-4). Il MINIMO che fa girare 17 GB in 16 GiB: miss ⇒ read OPFS →
// packExpertSlab → writeBuffer allo slot LRU-vittima, SINCRONO nel forward
// (stallo dichiarato e misurato: è il numero che C3 deve battere). Niente
// prefetch, niente pin, niente tier — quelli sono C3; il simulatore C1 dà
// 96.4% decode hit a budget 87% con LRU pura.
//
// Slot indirizzato (classe, buffer, offset): N buffer GPU per classe con
// taglia ≤ maxBufferSize; offset slab multipli di 256
// (minStorageBufferOffsetAlignment — garantito dal layout in moe.ts).
//
// CORREZIONE C3a it.5: fino a qui la taglia del buffer era cappata con
// `min(maxStorageBufferBindingSize, maxBufferSize)` "per difesa". È sbagliato:
// i due limiti misurano cose diverse — `maxStorageBufferBindingSize` è la
// taglia massima di un GPUBufferBinding (il SOTTO-RANGE che si binda, qui
// ~1.5 MB), `maxBufferSize` è la taglia massima del buffer. Il min buttava via
// il limite più grande e su NVIDIA (binding clampato a 2 GiB−4 da Dawn per un
// bug driver) teneva i buffer a metà di quanto potessero essere: 7 buffer
// invece di 4 a budget 12 GiB. Ora si cappa col limite giusto e si ASSERTA
// che il sotto-range più grande stia nel limite di binding.
//
// C3a fase 4 (strato 1), regime ARENA: col binding fisso il buffer È il
// binding, quindi il `min` di cui sopra torna — non come difesa ma come
// requisito. Il regime si accende con `arena: true` e cambia SOLO la taglia dei
// buffer (e di conseguenza quanti sono): slot, LRU, telemetria e VRAM totale
// restano quelli.
import { GLM47_FLASH as G } from "./shape";
import { packExpertSlab, SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, type SlabLayout, type SlabTensorLayout } from "./moe";
import { downIsQ4_1 } from "./expertstore";
import { mlaPartialsLen } from "./mlasplit";
import { GLM_PREFILL_M } from "./moeprefillplan";

/**
 * Classe di slab: l'ID della combinazione di formati (GLM: "q4_0"/"q4_1" dal
 * tipo del down; qwen35moe: "q4_K"/"q6_K"). Stringa e non unione chiusa dal
 * goal fase-D: le classi le detta il MODELLO, non questo file.
 */
export type ExpertClass = string;

/**
 * CONFIG DI MODELLO MoE (goal engine-fase-d fase 1, ruling direction §7-ter:
 * una meccanica, una implementazione). Tutto ciò che in questo file era
 * cablato su GLM — chiavi, parco, classi, minimi, slotTable — si DERIVA da
 * qui. GLM e Qwen 3.5/3.6 sono due configurazioni della stessa residenza.
 */
export interface MoeModelConfig {
  id: string;
  /** layer totali del modello (i primi `denseLead` sono densi, senza expert) */
  nLayer: number;
  denseLead: number;
  /** expert per layer MoE */
  nExpert: number;
  /** expert attivi per token (top-K): il minimo bindabile in un forward */
  nExpertUsed: number;
  /** id delle classi di slab presenti nel modello */
  classes: readonly ExpertClass[];
  /** classe del layer (chiamata solo sui layer MoE) */
  classOf(layer: number): ExpertClass;
  /** layout della classe (dal builder unico di moe.ts) */
  layout(cls: ExpertClass): SlabLayout;
}

/** Chiave globale expert → intero, per config. */
export const expertKeyFor = (cfg: MoeModelConfig, layer: number, expert: number): number =>
  layer * cfg.nExpert + expert;

/** Park per classe, opzionalmente troncato a `nLayer` (test di modelli parziali). */
export function moeParkOf(cfg: MoeModelConfig, nLayer: number = cfg.nLayer): Record<ExpertClass, number> {
  const park: Record<ExpertClass, number> = {};
  for (const c of cfg.classes) park[c] = 0;
  for (let l = cfg.denseLead; l < nLayer; l++) park[cfg.classOf(l)] += cfg.nExpert;
  return park;
}

/** Slot minimi per classe: un token deve poter bindare top-K expert per layer. */
export function minSlotsOf(cfg: MoeModelConfig): Record<ExpertClass, number> {
  const park = moeParkOf(cfg);
  const out: Record<ExpertClass, number> = {};
  for (const c of cfg.classes) out[c] = cfg.nExpertUsed * (park[c] / cfg.nExpert);
  return out;
}

/** Entry della slotTable: tutto il parco, layer assoluti. */
export const slotTableEntriesOf = (cfg: MoeModelConfig): number => cfg.nLayer * cfg.nExpert;

// Parco expert per classe (spec §1): blk.1-4 → 256 expert down-Q4_1,
// blk.5-46 → 2.688 down-Q4_0. blk.0 è denso.
/** CONFIG del modello-tesi: GLM-4.7-Flash è una configurazione, non il default cablato. */
export const MOE_CFG_GLM47: MoeModelConfig = {
  id: "glm-4.7-flash",
  nLayer: G.nLayer,
  denseLead: G.denseLead,
  nExpert: G.nExpert,
  nExpertUsed: G.nExpertUsed,
  classes: ["q4_0", "q4_1"],
  classOf: (layer) => (downIsQ4_1(layer) ? "q4_1" : "q4_0"),
  layout: (cls) => (cls === "q4_1" ? SLAB_DOWN_Q4_1 : SLAB_DOWN_Q4_0),
};

// Park GLM — ora DERIVATI dalla config (i valori storici 256 / 2.688 restano,
// e il test li verifica).
export const PARK_Q4_1 = 4 * G.nExpert;                          // 256
export const PARK_Q4_0 = (G.nLayer - G.denseLead - 4) * G.nExpert; // 2.688

export const expertKey = (layer: number, expert: number): number => layer * G.nExpert + expert;

export interface ExpertRawBytes { gate: Uint8Array; up: Uint8Array; down: Uint8Array }

/**
 * Come la cache ottiene i byte di un expert. La forma a funzione e' quella
 * storica (byte GGUF grezzi, da impacchettare); la forma a oggetto permette
 * alla sorgente di offrire lo slab GIA' impacchettato (`slab`), tenendo `raw`
 * come fallback quando il file slab non c'e'.
 */
export type ExpertReader =
  | ((layer: number, expert: number) => ExpertRawBytes)
  | {
      raw: (layer: number, expert: number) => ExpertRawBytes;
      slab?: (layer: number, expert: number) => Uint8Array;
    };

/**
 * MARCHIO DI CONIO (goal fase-D it.6). Il campo privato qui sotto fa sì che
 * un `SlotRef` possa essere CONIATO solo dentro questo modulo: chi prova a
 * fabbricarne uno per spacciare la propria arena come residenza viene
 * rifiutato da `tsc` (letterale, interfaccia gemella, `class implements`,
 * `Object.assign`, helper generico, perfino il cast diretto `as SlotRef`).
 *
 * COSA NON GARANTISCE (correzione del verifier di it.6, che ha bocciato la
 * prima formulazione): il marchio ferma la CONTRAFFAZIONE, non l'INDIFFERENZA.
 * Un'arena che semplicemente non usa `SlotRef` non viene sfiorata — ed è il
 * caso di `q35gpumodel.ts` fino alla chiusura della fase 1. Il marchio
 * diventa portante quando i kernel expert accettano solo `SlotRef`, cioè
 * quando ANCHE q35 passa da qui. Restano bypassabili senza cast: l'inflow
 * any-tipizzato e lo spread di uno `SlotRef` genuino (`{...vero, idx: n}`),
 * che ne conserva il marchio — si sorveglia il conio, non la circolazione.
 *
 * PERCHÉ COSÌ (lezione di it.4-5, tre gate bocciati): un test che scansiona
 * il sorgente NON può distinguere una duplicazione da una seconda famiglia
 * legittima — è una distinzione semantica, e ogni impronta testuale è
 * aggirabile con un refactoring ordinario (il verifier l'ha dimostrato con
 * cinque evasioni eseguite). `gpudevice.test` funziona perché `requestDevice`
 * è un nome della PIATTAFORMA: non esiste altra porta. Per la residenza degli
 * expert quella porta non esisteva, e va CREATA — qui, nel sistema di tipi,
 * dove il compilatore la sorveglia invece di una regex.
 */
declare const SLOT_REF_BRAND: unique symbol;

export interface SlotRef {
  /** conio: solo `residency.ts` può produrlo (vedi il commento sopra) */
  readonly [SLOT_REF_BRAND]: true;
  cls: ExpertClass;
  layout: SlabLayout;
  buffer: GPUBuffer;
  offset: number; // byte, inizio slab nel buffer di classe
  /**
   * Indice GLOBALE dello slot nella classe (0..nSlots-1). E' il numero che va
   * in `Sel.slot`: nel regime arena i kernel ne ricavano da soli buffer e base
   * (`slot / SLABS_PER_BUF`, `(slot % SLABS_PER_BUF)·SLAB_W`), quindi `buffer`
   * e `offset` restano solo per il regime a sotto-range e per i test.
   */
  idx: number;
}

export interface BindRange { buffer: GPUBuffer; offset: number; size: number }

// I sei sotto-range dello slab nell'ordine dei binding dei kernel GEMV.
export function slotBindRanges(s: SlotRef): {
  gateQs: BindRange; gateScales: BindRange; upQs: BindRange; upScales: BindRange;
  downQs: BindRange; downScales: BindRange;
} {
  const l = s.layout;
  const r = (off: number, size: number): BindRange => ({ buffer: s.buffer, offset: s.offset + off, size });
  return {
    gateQs: r(l.gateQs, l.qsBytes), gateScales: r(l.gateScales, l.gateScalesBytes),
    upQs: r(l.upQs, l.qsBytes), upScales: r(l.upScales, l.gateScalesBytes),
    downQs: r(l.downQs, l.qsBytes), downScales: r(l.downScales, l.downScalesBytes),
  };
}

/**
 * I TRE sotto-range del generico slab, dalla vista `layout.gate/up/down`.
 * `slotBindRanges` sopra è la vista LEGACY a sei range: esiste solo per i
 * formati a due segmenti (q4_0/q4_1/q8_0) e LANCIA sui K-quant, dove un
 * segmento scale separato non esiste. Questa è la forma che vale per
 * entrambe le famiglie, ed è quella che i call site nuovi devono usare.
 */
export function slotTensorRanges(s: SlotRef): { gate: BindRange; up: BindRange; down: BindRange } {
  const r = (t: SlabTensorLayout): BindRange => ({ buffer: s.buffer, offset: s.offset + t.data, size: t.dataBytes });
  return { gate: r(s.layout.gate), up: r(s.layout.up), down: r(s.layout.down) };
}

/** Il sotto-range PIÙ GRANDE che i kernel bindano su una classe di slab. */
export function maxBindRangeOf(layout: SlabLayout): number {
  let m = 0;
  for (const t of [layout.gate, layout.up, layout.down]) m = Math.max(m, t.dataBytes, t.scalesBytes);
  return m;
}

export interface ExpertCacheOpts {
  budgetBytes: number;            // budget VRAM residuo per gli slab expert
  maxBindingBytes: number;        // maxStorageBufferBindingSize negoziato
  maxBufferBytes: number;         // maxBufferSize negoziato
  // override per i test (budget ripartito ignorato se presente)
  slotsOverride?: Record<ExpertClass, number>;
  /** config del modello: default il modello-tesi (GLM). Qwen passa la sua. */
  cfg?: MoeModelConfig;
  timing?: boolean;               // telemetria liv.1: performance.now SOLO se true
  /**
   * Regime ARENA (C3a fase 4, strato 1): i buffer di classe si dimensionano per
   * essere bindati INTERI, non a sotto-range. Cambia solo la taglia dei buffer
   * (e quindi quanti sono): nSlots, LRU, contatori e VRAM totale sono identici.
   */
  arena?: boolean;
  /**
   * Tabella expertKey → slot in VRAM (C3a fase 4, slice B): la SOLA struttura
   * che permette al resolve GPU di tradurre un expert in indirizzo senza
   * chiedere niente alla CPU. Costa 12 032 B e una writeBuffer per layer con
   * miss; da spenta la cache e' identica a prima (nessun campo tocca il path).
   */
  slotTable?: boolean;
  /**
   * Policy di residenza (C3c fase 5, spec §4). "lru" (default) = il
   * comportamento storico, zero overhead. "tier" = LRU + AUTOPIN (pin dei
   * top-eusage con confidenza, cap HARD 12.5% degli slot per classe, assert)
   * + REPIN LFRU (score heat<<8|recency, isteresi 25%+4, max 4 swap/passata,
   * decay del calore). Il pin protegge dall'eviction; l'eviction resta LRU
   * fra i non pinnati. La selezione si registra con `noteSelection` (chiamata
   * dal forward dopo routerSelect — no-op in "lru").
   */
  policy?: "lru" | "tier";
}

export interface ExpertCacheStats {
  hits: number; misses: number; evictions: number;
  /**
   * Fonte dell'hit, separata (fase 4d, lezione kimi-k3-in-c: "hit rate" ha
   * tre definizioni e solo la retention concorda coi byte letti — il prefetch
   * gonfia le altre). Oggi hitsPrefetch e' strutturalmente 0: il prefetch non
   * esiste (e' C3b); il contatore c'e' perche' lo schema non cambi quando
   * arriva. hits === hitsResident + hitsPrefetch, sempre.
   */
  hitsResident: number; hitsPrefetch: number;
  /** hits + misses: il denominatore di retention, esplicito nel report. */
  requests: number;
  /** 1 − evictions/requests — null senza richieste (mai NaN nel JSON). */
  retention: number | null;
  bytesRead: number; bytesUploaded: number;
  /**
   * ATTENZIONE a `readMs` (rilievo del verifier, fase-D it.8): misura la
   * chiamata a `readRaw` DENTRO `ensure`, che è sincrona. Le sorgenti che
   * leggono in modo asincrono (q35: fetch Range sul GGUF) fanno l'I/O
   * PRIMA di chiamare `ensure` e consegnano byte già in memoria — per loro
   * `readMs` è ~0 e NON significa "I/O gratuito": significa che l'I/O sta
   * fuori da questa finestra e non è misurato qui.
   */
  readMs: number; packMs: number; uploadMs: number;
  occupied: Record<ExpertClass, number>;
  slots: Record<ExpertClass, number>;
  /** policy tier (C3c fase 5): pin correnti, cap 12.5%, selezioni, repin.
   *  null con policy "lru" (schema unico, null contagioso). */
  policy: {
    pinSlots: Record<ExpertClass, number>;
    pinCap: Record<ExpertClass, number>;
    selections: number;
    repinPasses: number;
    repinSwaps: number;
  } | null;
}

/**
 * Tetto di banda del DISPOSITIVO su cui l'OPFS vive, in GiB/s. Sopra questo
 * valore i byte non sono arrivati dal disco: li ha serviti la page cache del
 * sistema operativo.
 *
 * PROVENIENZA — misurata, e dagli artefatti stessi del bench GLM, non da una
 * scheda tecnica. Le passate di WARM-UP (che leggono l'intero parco expert per
 * la prima volta) danno la banda del dispositivo:
 *
 *     bench-glm-4090-b12-riga6-2026-08-15  prefill 19,10 GiB / 6,414 s = 2,98 GiB/s
 *                                          decode   1,51 GiB / 0,460 s = 3,29 GiB/s
 *     bench-glm-4090-b12-BASELINE-2026-08-16 prefill 19,10 / 8,277 s = 2,31 GiB/s
 *                                            decode   1,51 / 1,117 s = 1,35 GiB/s
 *
 * 4 GiB/s sta sopra tutte e quattro con margine: chi lo supera non ha letto.
 *
 * PERCHE' ESISTE (goal engine-velocita-decode, it.20). Il riferimento del
 * 2026-08-15 dichiarava 15,330 tok/s di decode sul GLM. Le sue REPLICHE
 * leggevano a **9,55 GiB/s** — cioe' 2,9 volte piu' veloce della sua stessa
 * passata di warm-up, sullo stesso file e a minuti di distanza. Non era una
 * proprieta' del motore: era la page cache. Rimisurato oggi a cache fredda, con
 * il path di lettura BYTE-IDENTICO e gli stessi `bytesRead` cifra per cifra, lo
 * stesso bench da 11,35 tok/s. Il numero e' oscillato del 26% senza che una
 * riga di codice cambiasse, e nessun campo dell'artefatto lo diceva.
 *
 * Un riferimento che non dichiara il suo regime di lettura non e' un
 * riferimento: e' una fotografia della RAM libera di quel pomeriggio.
 */
export const OPFS_DEVICE_CEILING_GIBS = 4;

/** Regime da cui sono arrivati i byte di una finestra di lettura. */
export type ReadRegime = "disk" | "os-cache" | "non-misurato";

/**
 * Banda implicita e regime di una finestra di lettura. `non-misurato` quando la
 * finestra non ha letto niente o il tempo non e' stato preso — mai NaN nel
 * JSON, e mai un `regime` inventato su zero byte.
 */
export function readBandwidth(bytesRead: number, readMs: number): { gibs: number | null; regime: ReadRegime } {
  if (!(bytesRead > 0) || !(readMs > 0)) return { gibs: null, regime: "non-misurato" };
  const gibs = bytesRead / (1 << 30) / (readMs / 1000);
  return { gibs, regime: gibs > OPFS_DEVICE_CEILING_GIBS ? "os-cache" : "disk" };
}

interface ClassState {
  layout: SlabLayout;
  buffers: GPUBuffer[];
  slabsPerBuffer: number;
  nSlots: number;
  free: number[];               // indici slot liberi (LIFO)
  lru: Map<number, number>;     // expertKey → slotIdx, ordine di inserimento = LRU
}

/** Geometria d'arena di UNA classe: quello che serve a generare i kernel. */
export interface ArenaGeometry {
  layout: SlabLayout;
  nBuf: number;         // buffer della classe = binding d'arena della pipeline
  slabsPerBuf: number;  // capienza di un buffer, in slab
  slabWords: number;    // layout.bytes / 4
  nSlots: number;
}

/** Riparto del budget fra le due classi (spec §5). Usato dal costruttore, da
 *  `arenaNeeds` e dalla precondizione di residenza totale dello slice C: tutti
 *  devono poterlo calcolare PRIMA che la cache (e quindi la VRAM) esista. */
export function expertSlots(
  o: { budgetBytes: number; slotsOverride?: Record<ExpertClass, number>; cfg?: MoeModelConfig },
): Record<ExpertClass, number> {
  if (o.slotsOverride) {
    // le chiavi dell'override devono essere le classi DELLA CONFIG: senza
    // questo controllo una cache si costruisce con zero classi, in silenzio
    // (degradazione trovata dal verifier it.3).
    const cfgO = o.cfg ?? MOE_CFG_GLM47;
    for (const c of cfgO.classes) {
      if (!(c in o.slotsOverride)) {
        throw new Error(`expertSlots: slotsOverride senza la classe "${c}" della config "${cfgO.id}"`);
      }
    }
    return o.slotsOverride;
  }
  const cfg = o.cfg ?? MOE_CFG_GLM47;
  const park = moeParkOf(cfg);
  let total = 0;
  for (const c of cfg.classes) total += park[c];
  const out: Record<ExpertClass, number> = {};
  for (const c of cfg.classes) {
    out[c] = Math.min(Math.floor((o.budgetBytes * park[c] / total) / cfg.layout(c).bytes), park[c]);
  }
  return out;
}

/**
 * Il parco expert per classe di un modello di `nLayer` layer: quanti expert
 * DEVONO stare in VRAM perché la residenza sia TOTALE (C3a fase 4 slice C).
 * Non è `PARK_Q4_0`/`PARK_Q4_1`, che sono il parco del modello INTERO: i
 * mini-modelli dei ktest hanno 2 layer, e la residenza totale per loro è 64
 * expert, non 2 944. Il conto è sui layer che ci sono davvero, con la stessa
 * `classOf` che decide dove finisce lo slab.
 */
/** COMPAT GLM (i call site storici passano nLayer): park del modello-tesi. */
export function modelExpertPark(nLayer: number): Record<ExpertClass, number> {
  const park: Record<ExpertClass, number> = { q4_0: 0, q4_1: 0 };
  for (let l = G.denseLead; l < nLayer; l++) park[downIsQ4_1(l) ? "q4_1" : "q4_0"] += G.nExpert;
  return park;
}

// ---------------------------------------------------------------------------
// Budget slab ctx-aware (C3c fase 3, spec 2026-08-08 §2): il budget non è più
// una costante di config ma una funzione di (tetto allocabile MISURATO, ctx).
// A ctx 6k il KV (665 MB) + partials sfondava il budget fisso ⇒ OOM.

/** Non-expert residente, MISURATO (probe vram-ceiling it.19 c3a,
 *  `required.nonExpertBytes`): attention + denso + shexp + router + head +
 *  norm. Si aggiorna solo con una rimisura, non a mano. */
export const NON_EXPERT_BYTES = 1_354_078_720;

/** KV f32 per token: nLayer × keyLen × 4 (glmmodel alloca
 *  `storage(ctxMax·keyLen·4)` PER LAYER). = 108 288 B/token — il "54 KB" citato
 *  fino a c3c it.2 era il conto f16, stale di 2× (docket c3c item 4). */
export const KV_PER_TOKEN_BYTES = G.nLayer * G.keyLen * 4;

/** Riserva driver/frammentazione, TARATA sul punto OOM osservato (spec §2:
 *  partiva [ASSUMED 256 MiB]). Evidenza della taratura (c3c it.3, due run):
 *  a sessione utente viva con free nvidia-smi 15 139 MiB la build va OOM con
 *  domanda ~14 890 ⇒ slack reale > 247 MiB — il probe it.19 (sessione minima
 *  post-riavvio) misurava ~160, ma la sessione viva aggiunge lo staging ring
 *  di Dawn durante il preload (~13 GB di writeBuffer) e le fluttuazioni del
 *  compositor. 512 MiB copre entrambe le osservazioni; copre anche i buffer
 *  fissi piccoli (staging logits, row576/q576, Sel, dirtyB: < 10 MiB). */
export const SLAB_RESERVE_BYTES = 512 * 2 ** 20;

/** Minimo di slot per classe = pin-for-replay del decode ottimistico (c3b I3:
 *  fino a 4 expert × layer MoE della classe devono poter restare pinnati
 *  durante un repair). q4_0: 4×42 layer, q4_1: 4×4 layer. */
export const MIN_SLOTS = minSlotsOf(MOE_CFG_GLM47);

export interface SlabBudgetInputs {
  /** Tetto di allocazione VRAM MISURATO (nvidia-smi total−used−reserved al
   *  lancio, o probe vram-ceiling): mai una costante inventata. La sessione
   *  host va dichiarata nel report (pattern hostState). */
  allocCeilingBytes: number;
  ctxMax: number;
  /** override per test/rimisure; default le costanti misurate qui sopra */
  nonExpertBytes?: number;
  reserveBytes?: number;
}

export interface SlabBudget {
  budgetBytes: number;
  slots: Record<ExpertClass, number>;
  /** addendi della sottrazione, per il report (spec §2: il JSON mostra il
   *  budget CALCOLATO, non asserito) */
  allocCeilingBytes: number;
  nonExpertBytes: number;
  kvBytes: number;
  workBytes: number;
  reserveBytes: number;
  ctxMax: number;
}

/** I buffer di lavoro ctx-dipendenti, derivati dagli stessi consumatori di
 *  engineNeeds (spec §2: NON stimati a mano). I termini che crescono con ctx
 *  sono i partials dello split MLA: quello del decode (×1) E quello del
 *  prefill chunked `attnPartialsM` (×GLM_PREFILL_M — glmmodel.ts:1125, già
 *  colpevole di un OOM storico a ctx 6688, commento a riga 641; ri-scoperto
 *  dall'OOM della prima run ctx-aware, c3c it.3). hiddenCkpt è fisso e sta
 *  qui perché è il più grosso dei "fissi" (385 KB). */
export const slabWorkBytes = (ctxMax: number): number =>
  (1 + GLM_PREFILL_M) * mlaPartialsLen(G.nHead, G.kvLora, ctxMax) * 4 + G.nLayer * G.dModel * 4;

export function slabBudgetCtxAware(o: SlabBudgetInputs): SlabBudget {
  const nonExpertBytes = o.nonExpertBytes ?? NON_EXPERT_BYTES;
  const reserveBytes = o.reserveBytes ?? SLAB_RESERVE_BYTES;
  const kvBytes = o.ctxMax * KV_PER_TOKEN_BYTES;
  const workBytes = slabWorkBytes(o.ctxMax);
  const budgetBytes = o.allocCeilingBytes - nonExpertBytes - kvBytes - workBytes - reserveBytes;
  if (budgetBytes <= 0) {
    throw new Error(
      `slab ctx-aware: budget ${budgetBytes} B <= 0 (ceiling ${o.allocCeilingBytes} − ` +
      `nonExpert ${nonExpertBytes} − kv ${kvBytes} @ctx${o.ctxMax} − work ${workBytes} − riserva ${reserveBytes})`);
  }
  const slots = expertSlots({ budgetBytes });
  for (const cls of MOE_CFG_GLM47.classes) {
    if (slots[cls] < MIN_SLOTS[cls]) {
      throw new Error(
        `slab ctx-aware: ${slots[cls]} slot ${cls} < minimo ${MIN_SLOTS[cls]} (pin-for-replay c3b I3) ` +
        `a budget ${budgetBytes} B (ceiling ${o.allocCeilingBytes}, ctx ${o.ctxMax}) — ` +
        `niente degradazione silenziosa (emendamento 5 c3a): servono più VRAM o meno contesto`);
    }
  }
  return { budgetBytes, slots, allocCeilingBytes: o.allocCeilingBytes, nonExpertBytes, kvBytes, workBytes, reserveBytes, ctxMax: o.ctxMax };
}

/** Slab per buffer nel regime arena: il buffer È il binding, quindi lo cappano
 *  ENTRAMBI i limiti (vedi il commento sull'inversione della correzione it.5). */
const arenaSlabsPerBuffer = (layout: SlabLayout, maxBufferBytes: number, maxBindingBytes: number): number =>
  Math.floor(Math.min(maxBufferBytes, maxBindingBytes) / layout.bytes);

/**
 * I due requisiti di limite che il regime arena aggiunge, calcolati dove il
 * device non c'è ancora (i worker li passano a `negotiateLimits`). Sta QUI e non
 * in gpulimits perché è la stessa aritmetica del costruttore: ricopiarla di là
 * significherebbe due verità sul numero di buffer.
 * `maxBufferBytes`/`maxBindingBytes` sono i tetti dell'ADAPTER: la finestra che
 * si chiede è il buffer più grande che la cache creerebbe potendo.
 */
export function arenaNeeds(o: {
  budgetBytes: number;
  slotsOverride?: Record<ExpertClass, number>;
  maxBufferBytes: number;
  maxBindingBytes: number;
  cfg?: MoeModelConfig;
}): { arenaBuffers: number; arenaWindowBytes: number } {
  const slots = expertSlots(o);
  const cfg = o.cfg ?? MOE_CFG_GLM47;
  const classes = cfg.classes.map((c) => [c, cfg.layout(c)] as const);
  // 1) la finestra: il buffer più grande che la cache creerebbe ai tetti dati.
  let arenaWindowBytes = 0;
  for (const [cls, layout] of classes) {
    // Una classe a zero slot non è un caso limite innocuo: `ceil(0/0)` è NaN, e
    // un NaN dentro `requiredLimits` passa silenziosamente ogni confronto per
    // finire nella richiesta al device. Qui si ferma, col consumatore in chiaro.
    if (!Number.isInteger(slots[cls]) || slots[cls] < 1) {
      throw new Error(
        `residency arena: la classe ${cls} ha ${slots[cls]} slot (budget ${o.budgetBytes} B, ` +
        `slab ${layout.bytes} B) — il modello ne binda ${cfg.nExpertUsed} per token`);
    }
    const per = arenaSlabsPerBuffer(layout, o.maxBufferBytes, o.maxBindingBytes);
    if (per < 1) {
      throw new Error(`residency: uno slab ${cls} (${layout.bytes} B) non sta in un binding (${o.maxBindingBytes} B)`);
    }
    arenaWindowBytes = Math.max(arenaWindowBytes, Math.min(slots[cls], per) * layout.bytes);
  }
  // 2) i buffer si contano sulla finestra CHE SI STA CHIEDENDO, non sui tetti
  //    dell'adapter: il device concederà quella, e una classe con slab più
  //    grandi ce ne farebbe stare uno di meno. Contarli sul tetto darebbe una
  //    stima ottimista di `arenaBuffers`, cioè il limite sbagliato — e
  //    l'errore uscirebbe a createComputePipeline, non qui.
  //    Il device concede sempre >= di quanto chiesto ⇒ i buffer VERI non
  //    possono essere più di questi.
  let arenaBuffers = 1;
  for (const [cls, layout] of classes) {
    const per = Math.floor(arenaWindowBytes / layout.bytes);
    if (per < 1) {
      throw new Error(
        `residency arena: la finestra chiesta (${arenaWindowBytes} B) non regge uno slab ${cls} ` +
        `(${layout.bytes} B) — le due classi non stanno nella stessa finestra`);
    }
    arenaBuffers = Math.max(arenaBuffers, Math.ceil(slots[cls] / per));
  }
  if (!Number.isInteger(arenaBuffers) || arenaBuffers < 1) {
    throw new Error(`residency arena: arenaBuffers = ${arenaBuffers} (non un conteggio)`);
  }
  return { arenaBuffers, arenaWindowBytes };
}

/** Entry della slotTable: nessuno slot pubblicato (design §2.1, `Sel.slot`). */
export const SLOT_TABLE_MISS = 0xffffffff;
/** Chiavi della slotTable: TUTTO il parco, layer assoluti (come `expertKey`). */
export const SLOT_TABLE_ENTRIES = slotTableEntriesOf(MOE_CFG_GLM47);

// ---- policy tier.h + AUTOPIN (C3c fase 5, spec §4 — colibri §2 tradotto) ----
// Costanti [ASSUMED spec §4], taratura eventualmente in fase 5 coi numeri:
/** storia minima prima che l'AUTOPIN pinni qualcosa (colibri: 5 000) */
export const AUTOPIN_MIN_HIST = 5000;
/** confidenza = selezioni/200k, cap 1 (colibri conf=hist/200000) */
export const AUTOPIN_CONF_DIVISOR = 200_000;
/** cap HARD del pin: ≤ 12.5% degli slot della classe (ruling C1; assert) */
export const PIN_CAP_FRAC = 0.125;
/** passata di repin ogni N selezioni di layer (≈ ogni 64 posizioni × nMoe) */
export const REPIN_EVERY_SEL = 64 * 46;
/** isteresi anti ping-pong: swap solo se score_cand > score_pin×1.25 + 4 */
export const REPIN_HYST_MULT = 1.25;
export const REPIN_HYST_ADD = 4;
export const REPIN_MAX_SWAPS = 4;

export class ExpertCache {
  private device: GPUDevice;
  /** config del modello: rende parametrici chiavi, parco, classi, minimi. */
  readonly cfg: MoeModelConfig;
  private cls: Record<ExpertClass, ClassState> = {};
  private timing: boolean;
  private s = { hits: 0, hitsResident: 0, hitsPrefetch: 0, misses: 0, evictions: 0, bytesRead: 0, bytesUploaded: 0, readMs: 0, packMs: 0, uploadMs: 0 };
  // ---- stato della policy "tier" (null con policy "lru": zero overhead) ----
  private policy: "lru" | "tier";
  private eusage: Uint32Array | null = null; // storia per expertKey (persistibile, additiva)
  private eheat: Uint16Array | null = null;  // calore di sessione (decade >>1 al repin)
  private erec: Uint32Array | null = null;   // clock dell'ultima selezione (recency LFRU)
  private selClock = 0;                      // selezioni totali (confidenza AUTOPIN)
  private pins: Record<ExpertClass, Set<number>> = {};
  private repinCountdown = REPIN_EVERY_SEL;
  private pol = { repinSwaps: 0, repinPasses: 0 };
  // slotTable (slice B): ombra CPU + intervallo sporco. La GPU la vede solo
  // quando il chiamante chiama `flushSlotTable`, cioe' una volta per layer.
  private table: { buf: GPUBuffer; shadow: Uint32Array; lo: number; hi: number } | null = null;
  // I1 (C3b): true fra submit e readback di coda in modo "optimistic"
  private inFlight = false;

  constructor(device: GPUDevice, opts: ExpertCacheOpts) {
    this.device = device;
    this.cfg = opts.cfg ?? MOE_CFG_GLM47;
    for (const c of this.cfg.classes) this.pins[c] = new Set();
    this.timing = opts.timing ?? false;
    this.policy = opts.policy ?? "lru";
    if (this.policy === "tier") {
      const entries = slotTableEntriesOf(this.cfg);
      this.eusage = new Uint32Array(entries);
      this.eheat = new Uint16Array(entries);
      this.erec = new Uint32Array(entries);
    }
    const arena = opts.arena === true;
    const mk = (layout: SlabLayout, nSlots: number): ClassState => {
      if (nSlots < this.cfg.nExpertUsed) throw new Error(`residency: ${nSlots} slot < ${this.cfg.nExpertUsed} (un token deve poter bindare top-K expert)`);
      // Il sotto-range più grande che i kernel bindano è `qsBytes` (~1.5 MB):
      // deve stare nel limite di BINDING. Se non ci sta, il layout è
      // incompatibile col device e va detto subito, non scoperto al primo bind.
      //
      // ARENA (C3a fase 4 strato 1): qui la correzione di it.5 si INVERTE. Con i
      // sotto-range il binding è ~1.5 MB e la taglia del buffer la decide solo
      // `maxBufferSize`; con l'arena il binding È il buffer intero, quindi il
      // buffer va cappato da ENTRAMBI i limiti — il min che it.5 ha tolto torna,
      // per la ragione opposta a quella per cui era stato messo.
      if (arena) {
        const perBuf = arenaSlabsPerBuffer(layout, opts.maxBufferBytes, opts.maxBindingBytes);
        if (perBuf < 1) {
          throw new Error(
            `residency arena: uno slab (${layout.bytes} B) non sta in un binding ` +
            `(min(maxBufferSize ${opts.maxBufferBytes}, maxStorageBufferBindingSize ${opts.maxBindingBytes}))`);
        }
      } else {
        // Il max si prende dalla VISTA GENERICA, non dai campi compat: quelli
        // descrivono il gate (`qsBytes`) e i due segmenti scale, e sui K-quant
        // il segmento più grande è il DOWN (Q6_K: 868 352 B contro 589 824 del
        // gate) — con i campi compat sarebbe sfuggito al controllo e il bind
        // sarebbe fallito a runtime. Su GLM il valore non cambia (i tre
        // tensori hanno la stessa taglia).
        const maxRange = maxBindRangeOf(layout);
        if (maxRange > opts.maxBindingBytes) {
          throw new Error(
            `residency: sotto-range ${maxRange} B > maxStorageBufferBindingSize ${opts.maxBindingBytes}`);
        }
      }
      const slabsPerBuffer = arena
        ? arenaSlabsPerBuffer(layout, opts.maxBufferBytes, opts.maxBindingBytes)
        : Math.max(1, Math.floor(opts.maxBufferBytes / layout.bytes));
      const buffers: GPUBuffer[] = [];
      for (let left = nSlots; left > 0; left -= slabsPerBuffer) {
        buffers.push(device.createBuffer({
          size: Math.min(left, slabsPerBuffer) * layout.bytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }));
      }
      return {
        layout, buffers, slabsPerBuffer, nSlots,
        free: Array.from({ length: nSlots }, (_, i) => nSlots - 1 - i),
        lru: new Map(),
      };
    };
    // riparto del budget tra le classi in proporzione al parco (spec §5),
    // PARAMETRICO: le classi e i loro layout vengono dalla config del modello.
    const slots = expertSlots({ ...opts, cfg: this.cfg });
    this.cls = {};
    for (const c of this.cfg.classes) this.cls[c] = mk(this.cfg.layout(c), slots[c]);
    if (opts.slotTable === true) {
      // Dimensionata sul parco INTERO (47×64), non sui layer del modello: la
      // chiave e' `expertKey`, che usa il layer assoluto perche' l'eviction non
      // rispetta i layer. Sono 12 032 B: tenerla piena costa meno che avere due
      // convenzioni di chiave.
      const shadow = new Uint32Array(slotTableEntriesOf(this.cfg)).fill(SLOT_TABLE_MISS);
      const buf = device.createBuffer({
        size: shadow.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      // stato iniziale ESPLICITO: senza questa scrittura il buffer sarebbe
      // zero, cioe' "tutti gli expert nello slot 0" — un miss che si legge come
      // hit sull'indirizzo sbagliato.
      device.queue.writeBuffer(buf, 0, shadow as unknown as BufferSource);
      this.table = { buf, shadow, lo: shadow.length, hi: -1 };
    }
  }

  /** Il buffer della slotTable (solo con `slotTable: true`). */
  slotTableBuffer(): GPUBuffer {
    if (!this.table) throw new Error("residency: slotTable non abilitata (ExpertCacheOpts.slotTable)");
    return this.table.buf;
  }

  private tableSet(key: number, slot: number): void {
    const t = this.table;
    if (!t || t.shadow[key] === slot) return;
    t.shadow[key] = slot;
    if (key < t.lo) t.lo = key;
    if (key > t.hi) t.hi = key;
  }

  /**
   * Pubblica sulla GPU le variazioni accumulate dagli `ensure`. Va chiamata UNA
   * volta per layer, DOPO gli ensure di quel layer e prima che qualcuno legga la
   * tabella: le writeBuffer degli slab sono gia' in coda, quindi la tabella
   * arriva dopo il dato che indirizza (R5 del design — l'ordine inverso
   * pubblicherebbe uno slot ancora vuoto). Un solo intervallo [lo,hi]: gli
   * expert di un layer sono contigui per costruzione della chiave, le vittime no
   * — nel caso peggiore si riscrive qualche KB, che e' meno di quanto costi
   * tenere una lista di intervalli.
   */
  flushSlotTable(): void {
    this.assertNotInFlight("flushSlotTable");
    const t = this.table;
    if (!t || t.hi < t.lo) return;
    this.device.queue.writeBuffer(t.buf, t.lo * 4, t.shadow, t.lo, t.hi - t.lo + 1);
    t.lo = t.shadow.length; // (fase-D it.4) NON la costante GLM: la shadow è la verità
    t.hi = -1;
  }

  /**
   * I1 del decode ottimistico (C3b spec §1): fra il submit di un token e il
   * suo readback di coda la slotTable e' INTOCCABILE — insert/evict avvengono
   * SOLO al confine di token. Il guard si arma da glmmodel nel solo modo
   * "optimistic": nel path sync l'ensure fra submit e mapAsync E' il design.
   */
  setInFlight(v: boolean): void {
    this.inFlight = v;
  }

  private assertNotInFlight(op: string): void {
    if (this.inFlight) {
      throw new Error(
        `residency: ${op} con token in volo — la slotTable e' intoccabile fra submit e readback ` +
        "(I1, C3b decode ottimistico): insert/evict solo al confine di token");
    }
  }

  /**
   * SOLO HARNESS (C3b, ktest): EVIZIONE deterministica delle chiavi date —
   * rimosse dalla LRU, slot restituiti alla free list, entry di slotTable a
   * MISS, flush immediato. Serve a forzare miss veri (expert genuinamente
   * non residente, come un miss di capacita') nel resolve del modo
   * "optimistic": cosi' il repair della fase 3 fa un fetch reale. Le
   * statistiche NON si toccano (l'evizione e' iniettata, non del regime).
   */
  debugMarkMiss(keys: number[]): void {
    if (!this.table) throw new Error("residency: debugMarkMiss senza slotTable (ExpertCacheOpts.slotTable)");
    this.assertNotInFlight("debugMarkMiss");
    for (const k of keys) {
      const layer = Math.floor(k / this.cfg.nExpert);
      const c = this.cls[this.classOfLayer(layer)];
      const found = c.lru.get(k);
      if (found !== undefined) {
        c.lru.delete(k);
        c.free.push(found);
      }
      this.tableSet(k, SLOT_TABLE_MISS);
    }
    this.flushSlotTable();
  }

  /** COMPAT GLM: la classe del layer nel modello-tesi. */
  static classOf(layer: number): ExpertClass {
    return MOE_CFG_GLM47.classOf(layer);
  }

  /** La classe del layer SECONDO LA CONFIG di questa cache (parametrica). */
  classOfLayer(layer: number): ExpertClass {
    return this.cfg.classOf(layer);
  }

  /**
   * L'expert è già in VRAM? Peek PURO: non tocca la LRU, non conta hit/miss,
   * non alloca. Serve a chi legge i pesi in modo ASINCRONO (il 35B non sta in
   * RAM: i byte arrivano da fetch Range): il forward guarda quali dei top-K
   * mancano, `await`ta SOLO quelli, e poi chiama `ensure` con i byte in mano.
   * Senza questo si dovrebbe leggere anche sugli hit — cioè sempre.
   */
  isResident(layer: number, expert: number): boolean {
    return this.cls[this.classOfLayer(layer)].lru.has(this.keyOf(layer, expert));
  }

  /**
   * HIT del path a submit unico (fase-D fase 3b, fetta 3c): l'expert era già
   * residente quando il resolve su GPU l'ha risolto, quindi NESSUNO ha chiamato
   * `ensure` — la CPU non ha visto la selezione mentre il token girava. Questa
   * è la contabilità di quell'hit, fatta al confine di token dalla `Sel` letta
   * in coda: conta l'hit e TOCCA la LRU esattamente come farebbe `ensure`.
   *
   * Serve perché senza il touch la recency degraderebbe all'ordine di
   * inserimento — la LRU diventerebbe una FIFO e le vittime cambierebbero:
   * il path ottimistico misurerebbe miss diversi dal path sync per un motivo
   * che non ha niente a che vedere col meccanismo che sta provando.
   *
   * LANCIA se l'expert non è residente: chi chiama sta rileggendo una `Sel` che
   * dichiarava HIT (flag miss a 0), e un disaccordo lì è un bug strutturale
   * (eviction fra resolve e confine di token), non un dato da interpretare.
   */
  noteResidentHit(layer: number, expert: number): void {
    const cls = this.classOfLayer(layer);
    const c = this.cls[cls];
    const key = this.keyOf(layer, expert);
    const found = c.lru.get(key);
    if (found === undefined) {
      throw new Error(
        `residency: noteResidentHit su expert non residente (layer ${layer}, expert ${expert}, ` +
        `chiave ${key}) — la Sel del token lo dava risolto: eviction fra il resolve e il confine ` +
        "di token, bug strutturale");
    }
    c.lru.delete(key);
    c.lru.set(key, found);
    this.s.hits++;
    this.s.hitsResident++;
  }

  /** Chiave globale dell'expert secondo la config di questa cache. */
  keyOf(layer: number, expert: number): number {
    return expertKeyFor(this.cfg, layer, expert);
  }

  /** I buffer d'arena di una classe, nell'ordine dei binding della pipeline. */
  arenaBuffers(cls: ExpertClass): GPUBuffer[] {
    return this.cls[cls].buffers;
  }

  /** La geometria con cui si generano i kernel d'arena della classe. */
  arenaGeometry(cls: ExpertClass): ArenaGeometry {
    const c = this.cls[cls];
    return {
      layout: c.layout, nBuf: c.buffers.length, slabsPerBuf: c.slabsPerBuffer,
      slabWords: c.layout.bytes / 4, nSlots: c.nSlots,
    };
  }

  /**
   * L'indirizzo di uno slot senza passare da `ensure`. Serve a verificare che
   * l'aritmetica dei kernel d'arena (slot → binding + base) cada esattamente
   * sui sotto-range che il regime a binding mobile bindava, per OGNI slot.
   */
  slotAt(cls: ExpertClass, idx: number): SlotRef {
    const c = this.cls[cls];
    if (idx < 0 || idx >= c.nSlots) throw new Error(`residency: slot ${idx} fuori da [0, ${c.nSlots})`);
    return this.slotRef(c, cls, idx);
  }

  /** L'UNICO sito di conio di uno `SlotRef` in tutto il motore. */
  private slotRef(c: ClassState, cls: ExpertClass, idx: number): SlotRef {
    return {
      cls, layout: c.layout,
      buffer: c.buffers[Math.floor(idx / c.slabsPerBuffer)],
      offset: (idx % c.slabsPerBuffer) * c.layout.bytes,
      idx,
    } as SlotRef; // il marchio si applica QUI e in nessun altro posto
  }

  // Garantisce l'expert residente e ritorna lo slot. `readRaw` viene chiamata
  // SOLO su miss (read OPFS: il costo entra in readMs). `pinned` = slot che il
  // token corrente sta per bindare: mai scelti come vittima (i top-4 di un
  // layer devono coesistere).
  ensure(layer: number, expert: number, readRaw: ExpertReader, pinned?: Set<number>): { slot: SlotRef; hit: boolean } {
    this.assertNotInFlight("ensure");
    const cls = this.classOfLayer(layer);
    const c = this.cls[cls];
    const key = this.keyOf(layer, expert);
    const found = c.lru.get(key);
    if (found !== undefined) {
      c.lru.delete(key);
      c.lru.set(key, found); // touch: rientra in coda alla LRU
      this.s.hits++;
      this.s.hitsResident++; // unica fonte oggi: il prefetch e' C3b
      return { slot: this.slotRef(c, cls, found), hit: true };
    }
    this.s.misses++;
    let idx: number;
    if (c.free.length > 0) {
      idx = c.free.pop()!;
    } else {
      // vittima = primo della mappa (least recently used) non pinnato — né dal
      // chiamante (pin-for-replay/ensure) né dalla policy (AUTOPIN, fase 5)
      const polPins = this.pins[cls];
      let victim: [number, number] | undefined;
      for (const e of c.lru) {
        if (!pinned?.has(e[0]) && !polPins.has(e[0])) { victim = e; break; }
      }
      if (!victim) throw new Error(`residency: nessuna vittima evincibile (pinned caller ${pinned?.size ?? 0} + policy ${polPins.size} su ${c.lru.size} residenti)`);
      c.lru.delete(victim[0]);
      idx = victim[1];
      this.s.evictions++;
      // lo slot cambia proprietario: la vittima torna MISS PRIMA che il nuovo
      // expert lo pubblichi (fra le due scritture nessuno legge — il flush e'
      // uno solo, a fine layer)
      this.tableSet(victim[0], SLOT_TABLE_MISS);
    }
    // Percorso RAPIDO (C3a fase 3): se la sorgente sa dare lo slab gia'
    // impacchettato (file slab generato all'import), il pack CPU sparisce dal
    // path caldo — erano 9,5 ms per miss, 41,4 ms/token al bench.
    const t0 = this.timing ? performance.now() : 0;
    let slab: Uint8Array;
    let rawBytes: number;
    let t1: number, t2: number;
    if (typeof readRaw === "object" && readRaw.slab) {
      slab = readRaw.slab(layer, expert);
      t1 = t2 = this.timing ? performance.now() : 0; // niente pack: read == tutto
      rawBytes = slab.length;
    } else {
      const read = typeof readRaw === "function" ? readRaw : readRaw.raw;
      const raw = read(layer, expert);
      t1 = this.timing ? performance.now() : 0;
      slab = packExpertSlab(raw.gate, raw.up, raw.down, c.layout);
      t2 = this.timing ? performance.now() : 0;
      rawBytes = raw.gate.length + raw.up.length + raw.down.length;
    }
    const slot = this.slotRef(c, cls, idx);
    this.device.queue.writeBuffer(slot.buffer, slot.offset, slab as unknown as BufferSource);
    if (this.timing) {
      const t3 = performance.now();
      this.s.readMs += t1 - t0;
      this.s.packMs += t2 - t1;
      this.s.uploadMs += t3 - t2;
    }
    this.s.bytesRead += rawBytes;
    this.s.bytesUploaded += slab.length;
    c.lru.set(key, idx);
    this.tableSet(key, idx);
    return { slot, hit: false };
  }

  // ---- policy tier: selezione, repin, persistenza (C3c fase 5, spec §4) ----

  /**
   * Registra la selezione del router per un layer (chiamata dal forward dopo
   * routerSelect; in policy "lru" è un no-op). Fa scattare la passata di
   * repin ogni REPIN_EVERY_SEL selezioni.
   */
  noteSelection(layer: number, experts: ArrayLike<number>): void {
    if (this.policy !== "tier") return;
    for (let k = 0; k < experts.length; k++) {
      const key = this.keyOf(layer, experts[k] as number);
      this.eusage![key]++;
      if (this.eheat![key] < 0xffff) this.eheat![key]++;
      this.erec![key] = ++this.selClock;
    }
    this.repinCountdown -= experts.length; // la cadenza e' in SELEZIONI, non in chiamate
    if (this.repinCountdown <= 0) {
      this.repinCountdown = REPIN_EVERY_SEL;
      this.repinPass();
    }
  }

  /** score LFRU di colibri tier.h: frequenza primaria, recency a spareggio */
  private lfruScore(key: number): number {
    const age = this.selClock - this.erec![key];
    const rec = Math.max(0, 255 - Math.min(255, age >> 6));
    return this.eheat![key] * 256 + rec;
  }

  /**
   * Passata di repin (colibri repin_pass_limit): per classe — (1) budget
   * AUTOPIN = cap 12.5% × confidenza (storia/200k, cap 1; niente pin sotto
   * AUTOPIN_MIN_HIST selezioni); (2) riempi il budget coi top-eusage
   * RESIDENTI; (3) fino a REPIN_MAX_SWAPS scambi pin-freddo ↔ residente-caldo
   * con isteresi (score_cand > score_pin×1.25 + 4); (4) decay del calore
   * (>>1). Solo metadata: nessun load qui — la protezione dall'eviction
   * plasma il set residente nel tempo (deviazione dichiarata da colibri, che
   * nel repin carica ~20 MB a swap).
   */
  private repinPass(): void {
    this.pol.repinPasses++;
    for (const cls of this.cfg.classes) {
      const c = this.cls[cls];
      const pins = this.pins[cls];
      const cap = Math.floor(PIN_CAP_FRAC * c.nSlots);
      const conf = Math.min(1, this.selClock / AUTOPIN_CONF_DIVISOR);
      // sopra la storia minima si parte da 1 pin: floor(cap×conf) a classi
      // piccole resterebbe 0 fino a conf 0.5 e l'AUTOPIN non partirebbe mai
      const budget = this.selClock < AUTOPIN_MIN_HIST ? 0 : Math.min(cap, Math.max(1, Math.floor(cap * conf)));
      // sopra budget (confidenza scesa? cap piu' piccolo?): spinna i piu' freddi
      while (pins.size > budget) {
        let coldest = -1, coldScore = Infinity;
        for (const k of pins) { const s = this.lfruScore(k); if (s < coldScore) { coldScore = s; coldest = k; } }
        pins.delete(coldest);
      }
      if (budget === 0) continue;
      // candidati = residenti non pinnati, per eusage (l'AUTOPIN pinna la storia)
      const cand: number[] = [];
      for (const k of c.lru.keys()) if (!pins.has(k)) cand.push(k);
      cand.sort((a, b) => this.eusage![b] - this.eusage![a]);
      let ci = 0;
      while (pins.size < budget && ci < cand.length) pins.add(cand[ci++]);
      // swap con isteresi: il residente più caldo scalza il pin più freddo
      for (let sw = 0; sw < REPIN_MAX_SWAPS && ci < cand.length; sw++) {
        const cand2 = cand.slice(ci).sort((a, b) => this.lfruScore(b) - this.lfruScore(a));
        const hot = cand2[0];
        if (hot === undefined) break;
        let coldest = -1, coldScore = Infinity;
        for (const k of pins) { const s = this.lfruScore(k); if (s < coldScore) { coldScore = s; coldest = k; } }
        if (coldest < 0 || this.lfruScore(hot) <= coldScore * REPIN_HYST_MULT + REPIN_HYST_ADD) break;
        pins.delete(coldest); pins.add(hot);
        cand.splice(cand.indexOf(hot), 1);
        this.pol.repinSwaps++;
      }
      // il cap e' un contratto (ruling C1), non una speranza
      if (pins.size > cap) {
        throw new Error(`residency tier: pin ${pins.size} > cap ${cap} (12.5% di ${c.nSlots} slot ${cls})`);
      }
    }
    if (this.eheat) for (let i = 0; i < this.eheat.length; i++) this.eheat[i] >>= 1;
  }

  /** snapshot della storia eusage (per la persistenza OPFS, write atomica del
   *  chiamante via createWritable — commit-on-close). */
  usageSnapshot(): Uint8Array {
    if (!this.eusage) throw new Error("residency: usageSnapshot richiede policy tier");
    return new Uint8Array(this.eusage.buffer.slice(0));
  }

  /** carica la storia ADDITIVAMENTE (colibri usage_load: eusage[k] += cnt). */
  loadUsage(bytes: Uint8Array): void {
    if (!this.eusage) throw new Error("residency: loadUsage richiede policy tier");
    if (bytes.byteLength !== this.eusage.byteLength) {
      throw new Error(`residency: eusage ${bytes.byteLength} B, attesi ${this.eusage.byteLength}`);
    }
    const inc = new Uint32Array(bytes.buffer, bytes.byteOffset, this.eusage.length);
    for (let i = 0; i < this.eusage.length; i++) this.eusage[i] += inc[i];
  }

  stats(): ExpertCacheStats {
    const requests = this.s.hits + this.s.misses;
    return {
      ...this.s,
      requests,
      retention: requests > 0 ? 1 - this.s.evictions / requests : null,
      occupied: Object.fromEntries(Object.entries(this.cls).map(([c, st]) => [c, st.lru.size])),
      slots: Object.fromEntries(Object.entries(this.cls).map(([c, st]) => [c, st.nSlots])),
      // policy tier (C3c fase 5): null con policy "lru" (schema unico)
      policy: this.policy === "tier" ? {
        pinSlots: Object.fromEntries(this.cfg.classes.map((c) => [c, this.pins[c].size])),
        pinCap: Object.fromEntries(this.cfg.classes.map((c) => [c, Math.floor(PIN_CAP_FRAC * this.cls[c].nSlots)])),
        selections: this.selClock,
        repinPasses: this.pol.repinPasses,
        repinSwaps: this.pol.repinSwaps,
      } : null,
    };
  }

  resetStats(): void {
    this.s = { hits: 0, hitsResident: 0, hitsPrefetch: 0, misses: 0, evictions: 0, bytesRead: 0, bytesUploaded: 0, readMs: 0, packMs: 0, uploadMs: 0 };
  }

  destroy(): void {
    for (const st of Object.values(this.cls)) for (const b of st.buffers) b.destroy();
    this.table?.buf.destroy();
  }
}
