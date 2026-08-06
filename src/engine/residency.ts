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
import { packExpertSlab, SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, type SlabLayout } from "./moe";
import { downIsQ4_1 } from "./expertstore";

export type ExpertClass = "q4_0" | "q4_1";

// Parco expert per classe (spec §1): blk.1-4 → 256 expert down-Q4_1,
// blk.5-46 → 2.688 down-Q4_0. blk.0 è denso.
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

export interface SlotRef {
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

export interface ExpertCacheOpts {
  budgetBytes: number;            // budget VRAM residuo per gli slab expert
  maxBindingBytes: number;        // maxStorageBufferBindingSize negoziato
  maxBufferBytes: number;         // maxBufferSize negoziato
  // override per i test (budget ripartito ignorato se presente)
  slotsOverride?: { q4_0: number; q4_1: number };
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
  readMs: number; packMs: number; uploadMs: number;
  occupied: { q4_0: number; q4_1: number };
  slots: { q4_0: number; q4_1: number };
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
export function expertSlots(o: { budgetBytes: number; slotsOverride?: { q4_0: number; q4_1: number } }): { q4_0: number; q4_1: number } {
  if (o.slotsOverride) return o.slotsOverride;
  const park = PARK_Q4_0 + PARK_Q4_1;
  return {
    q4_0: Math.min(Math.floor((o.budgetBytes * PARK_Q4_0 / park) / SLAB_DOWN_Q4_0.bytes), PARK_Q4_0),
    q4_1: Math.min(Math.floor((o.budgetBytes * PARK_Q4_1 / park) / SLAB_DOWN_Q4_1.bytes), PARK_Q4_1),
  };
}

/**
 * Il parco expert per classe di un modello di `nLayer` layer: quanti expert
 * DEVONO stare in VRAM perché la residenza sia TOTALE (C3a fase 4 slice C).
 * Non è `PARK_Q4_0`/`PARK_Q4_1`, che sono il parco del modello INTERO: i
 * mini-modelli dei ktest hanno 2 layer, e la residenza totale per loro è 64
 * expert, non 2 944. Il conto è sui layer che ci sono davvero, con la stessa
 * `classOf` che decide dove finisce lo slab.
 */
export function modelExpertPark(nLayer: number): Record<ExpertClass, number> {
  const park: Record<ExpertClass, number> = { q4_0: 0, q4_1: 0 };
  for (let l = G.denseLead; l < nLayer; l++) park[downIsQ4_1(l) ? "q4_1" : "q4_0"] += G.nExpert;
  return park;
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
  slotsOverride?: { q4_0: number; q4_1: number };
  maxBufferBytes: number;
  maxBindingBytes: number;
}): { arenaBuffers: number; arenaWindowBytes: number } {
  const slots = expertSlots(o);
  const classes = [["q4_0", SLAB_DOWN_Q4_0], ["q4_1", SLAB_DOWN_Q4_1]] as const;
  // 1) la finestra: il buffer più grande che la cache creerebbe ai tetti dati.
  let arenaWindowBytes = 0;
  for (const [cls, layout] of classes) {
    // Una classe a zero slot non è un caso limite innocuo: `ceil(0/0)` è NaN, e
    // un NaN dentro `requiredLimits` passa silenziosamente ogni confronto per
    // finire nella richiesta al device. Qui si ferma, col consumatore in chiaro.
    if (!Number.isInteger(slots[cls]) || slots[cls] < 1) {
      throw new Error(
        `residency arena: la classe ${cls} ha ${slots[cls]} slot (budget ${o.budgetBytes} B, ` +
        `slab ${layout.bytes} B) — il modello ne binda ${G.nExpertUsed} per token`);
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
export const SLOT_TABLE_ENTRIES = G.nLayer * G.nExpert;

export class ExpertCache {
  private device: GPUDevice;
  private cls: Record<ExpertClass, ClassState>;
  private timing: boolean;
  private s = { hits: 0, hitsResident: 0, hitsPrefetch: 0, misses: 0, evictions: 0, bytesRead: 0, bytesUploaded: 0, readMs: 0, packMs: 0, uploadMs: 0 };
  // slotTable (slice B): ombra CPU + intervallo sporco. La GPU la vede solo
  // quando il chiamante chiama `flushSlotTable`, cioe' una volta per layer.
  private table: { buf: GPUBuffer; shadow: Uint32Array; lo: number; hi: number } | null = null;

  constructor(device: GPUDevice, opts: ExpertCacheOpts) {
    this.device = device;
    this.timing = opts.timing ?? false;
    const arena = opts.arena === true;
    const mk = (layout: SlabLayout, nSlots: number): ClassState => {
      if (nSlots < G.nExpertUsed) throw new Error(`residency: ${nSlots} slot < ${G.nExpertUsed} (un token deve poter bindare 4 expert)`);
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
        const maxRange = Math.max(layout.qsBytes, layout.gateScalesBytes, layout.downScalesBytes);
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
    // riparto del budget tra le classi in proporzione al parco (spec §5)
    const { q4_0: n40, q4_1: n41 } = expertSlots(opts);
    this.cls = { q4_0: mk(SLAB_DOWN_Q4_0, n40), q4_1: mk(SLAB_DOWN_Q4_1, n41) };
    if (opts.slotTable === true) {
      // Dimensionata sul parco INTERO (47×64), non sui layer del modello: la
      // chiave e' `expertKey`, che usa il layer assoluto perche' l'eviction non
      // rispetta i layer. Sono 12 032 B: tenerla piena costa meno che avere due
      // convenzioni di chiave.
      const shadow = new Uint32Array(SLOT_TABLE_ENTRIES).fill(SLOT_TABLE_MISS);
      const buf = device.createBuffer({
        size: shadow.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      // stato iniziale ESPLICITO: senza questa scrittura il buffer sarebbe
      // zero, cioe' "tutti gli expert nello slot 0" — un miss che si legge come
      // hit sull'indirizzo sbagliato.
      device.queue.writeBuffer(buf, 0, shadow as unknown as BufferSource);
      this.table = { buf, shadow, lo: SLOT_TABLE_ENTRIES, hi: -1 };
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
    const t = this.table;
    if (!t || t.hi < t.lo) return;
    this.device.queue.writeBuffer(t.buf, t.lo * 4, t.shadow, t.lo, t.hi - t.lo + 1);
    t.lo = SLOT_TABLE_ENTRIES;
    t.hi = -1;
  }

  static classOf(layer: number): ExpertClass {
    return downIsQ4_1(layer) ? "q4_1" : "q4_0";
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

  private slotRef(c: ClassState, cls: ExpertClass, idx: number): SlotRef {
    return {
      cls, layout: c.layout,
      buffer: c.buffers[Math.floor(idx / c.slabsPerBuffer)],
      offset: (idx % c.slabsPerBuffer) * c.layout.bytes,
      idx,
    };
  }

  // Garantisce l'expert residente e ritorna lo slot. `readRaw` viene chiamata
  // SOLO su miss (read OPFS: il costo entra in readMs). `pinned` = slot che il
  // token corrente sta per bindare: mai scelti come vittima (i top-4 di un
  // layer devono coesistere).
  ensure(layer: number, expert: number, readRaw: ExpertReader, pinned?: Set<number>): { slot: SlotRef; hit: boolean } {
    const cls = ExpertCache.classOf(layer);
    const c = this.cls[cls];
    const key = expertKey(layer, expert);
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
      // vittima = primo della mappa (least recently used) non pinnato
      let victim: [number, number] | undefined;
      for (const e of c.lru) {
        if (!pinned?.has(e[0])) { victim = e; break; }
      }
      if (!victim) throw new Error("residency: nessuna vittima evincibile (tutti gli slot pinnati)");
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

  stats(): ExpertCacheStats {
    const requests = this.s.hits + this.s.misses;
    return {
      ...this.s,
      requests,
      retention: requests > 0 ? 1 - this.s.evictions / requests : null,
      occupied: { q4_0: this.cls.q4_0.lru.size, q4_1: this.cls.q4_1.lru.size },
      slots: { q4_0: this.cls.q4_0.nSlots, q4_1: this.cls.q4_1.nSlots },
    };
  }

  resetStats(): void {
    this.s = { hits: 0, hitsResident: 0, hitsPrefetch: 0, misses: 0, evictions: 0, bytesRead: 0, bytesUploaded: 0, readMs: 0, packMs: 0, uploadMs: 0 };
  }

  destroy(): void {
    for (const cls of [this.cls.q4_0, this.cls.q4_1]) for (const b of cls.buffers) b.destroy();
    this.table?.buf.destroy();
  }
}
