// Residenza minima expert (goal C2 fase 5, spec §5): cache VRAM a slot con
// policy LRU PURA, due size-class esatte (down Q4_0 blk.5-46 / down Q4_1
// blk.1-4). Il MINIMO che fa girare 17 GB in 16 GiB: miss ⇒ read OPFS →
// packExpertSlab → writeBuffer allo slot LRU-vittima, SINCRONO nel forward
// (stallo dichiarato e misurato: è il numero che C3 deve battere). Niente
// prefetch, niente pin, niente tier — quelli sono C3; il simulatore C1 dà
// 96.4% decode hit a budget 87% con LRU pura.
//
// Slot indirizzato (classe, buffer, offset): N buffer GPU per classe con
// taglia ≤ maxStorageBufferBindingSize negoziato (difensivo: i bind sono
// comunque sotto-range da ~1.5-2 MB); offset slab multipli di 256
// (minStorageBufferOffsetAlignment — garantito dal layout in moe.ts).
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

export interface SlotRef {
  cls: ExpertClass;
  layout: SlabLayout;
  buffer: GPUBuffer;
  offset: number; // byte, inizio slab nel buffer di classe
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
}

export interface ExpertCacheStats {
  hits: number; misses: number; evictions: number;
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

export class ExpertCache {
  private device: GPUDevice;
  private cls: Record<ExpertClass, ClassState>;
  private timing: boolean;
  private s = { hits: 0, misses: 0, evictions: 0, bytesRead: 0, bytesUploaded: 0, readMs: 0, packMs: 0, uploadMs: 0 };

  constructor(device: GPUDevice, opts: ExpertCacheOpts) {
    this.device = device;
    this.timing = opts.timing ?? false;
    const mk = (layout: SlabLayout, nSlots: number): ClassState => {
      if (nSlots < G.nExpertUsed) throw new Error(`residency: ${nSlots} slot < ${G.nExpertUsed} (un token deve poter bindare 4 expert)`);
      const cap = Math.min(opts.maxBindingBytes, opts.maxBufferBytes);
      const slabsPerBuffer = Math.max(1, Math.floor(cap / layout.bytes));
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
    let n40: number, n41: number;
    if (opts.slotsOverride) {
      ({ q4_0: n40, q4_1: n41 } = opts.slotsOverride);
    } else {
      // riparto del budget tra le classi in proporzione al parco (spec §5)
      const park = PARK_Q4_0 + PARK_Q4_1;
      n40 = Math.min(Math.floor((opts.budgetBytes * PARK_Q4_0 / park) / SLAB_DOWN_Q4_0.bytes), PARK_Q4_0);
      n41 = Math.min(Math.floor((opts.budgetBytes * PARK_Q4_1 / park) / SLAB_DOWN_Q4_1.bytes), PARK_Q4_1);
    }
    this.cls = { q4_0: mk(SLAB_DOWN_Q4_0, n40), q4_1: mk(SLAB_DOWN_Q4_1, n41) };
  }

  static classOf(layer: number): ExpertClass {
    return downIsQ4_1(layer) ? "q4_1" : "q4_0";
  }

  private slotRef(c: ClassState, cls: ExpertClass, idx: number): SlotRef {
    return {
      cls, layout: c.layout,
      buffer: c.buffers[Math.floor(idx / c.slabsPerBuffer)],
      offset: (idx % c.slabsPerBuffer) * c.layout.bytes,
    };
  }

  // Garantisce l'expert residente e ritorna lo slot. `readRaw` viene chiamata
  // SOLO su miss (read OPFS: il costo entra in readMs). `pinned` = slot che il
  // token corrente sta per bindare: mai scelti come vittima (i top-4 di un
  // layer devono coesistere).
  ensure(layer: number, expert: number, readRaw: (layer: number, expert: number) => ExpertRawBytes, pinned?: Set<number>): { slot: SlotRef; hit: boolean } {
    const cls = ExpertCache.classOf(layer);
    const c = this.cls[cls];
    const key = expertKey(layer, expert);
    const found = c.lru.get(key);
    if (found !== undefined) {
      c.lru.delete(key);
      c.lru.set(key, found); // touch: rientra in coda alla LRU
      this.s.hits++;
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
    }
    const t0 = this.timing ? performance.now() : 0;
    const raw = readRaw(layer, expert);
    const t1 = this.timing ? performance.now() : 0;
    const slab = packExpertSlab(raw.gate, raw.up, raw.down, c.layout);
    const t2 = this.timing ? performance.now() : 0;
    const slot = this.slotRef(c, cls, idx);
    this.device.queue.writeBuffer(slot.buffer, slot.offset, slab as unknown as BufferSource);
    if (this.timing) {
      const t3 = performance.now();
      this.s.readMs += t1 - t0;
      this.s.packMs += t2 - t1;
      this.s.uploadMs += t3 - t2;
    }
    this.s.bytesRead += raw.gate.length + raw.up.length + raw.down.length;
    this.s.bytesUploaded += slab.length;
    c.lru.set(key, idx);
    return { slot, hit: false };
  }

  stats(): ExpertCacheStats {
    return {
      ...this.s,
      occupied: { q4_0: this.cls.q4_0.lru.size, q4_1: this.cls.q4_1.lru.size },
      slots: { q4_0: this.cls.q4_0.nSlots, q4_1: this.cls.q4_1.nSlots },
    };
  }

  resetStats(): void {
    this.s = { hits: 0, misses: 0, evictions: 0, bytesRead: 0, bytesUploaded: 0, readMs: 0, packMs: 0, uploadMs: 0 };
  }

  destroy(): void {
    for (const cls of [this.cls.q4_0, this.cls.q4_1]) for (const b of cls.buffers) b.destroy();
  }
}
