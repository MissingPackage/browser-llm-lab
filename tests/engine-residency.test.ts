// Residenza minima (goal C2 fase 5 slice 2): Sha256Stream vs webcrypto,
// indice expert→range sul GGUF reale (aritmetica esatta vs tensorByteSize),
// ExpertCache (riparto budget, LRU pura, classi separate, pinned, telemetria)
// con device GPU mock. Il roundtrip OPFS→pack→upload→readback con byte veri
// sta nel ktest (serve browser+GPU).
import { describe, it, expect, beforeAll } from "vitest";
import { webcrypto } from "node:crypto";
import { existsSync, openSync, readSync, closeSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseGguf, tensorByteSize } from "../src/engine/gguf";
import { validateGlm47Flash, GLM47_FLASH as G, GLM47_DOWN_EXPS_Q4_1_LAST } from "../src/engine/shape";
import {
  Sha256Stream, GgufExpertIndex, downIsQ4_1,
  EXPERT_GATE_UP_BYTES, EXPERT_DOWN_Q4_0_BYTES, EXPERT_DOWN_Q4_1_BYTES,
} from "../src/engine/expertstore";
import {
  ExpertCache, arenaNeeds, expertKey, expertSlots, modelExpertPark, slotBindRanges,
  PARK_Q4_0, PARK_Q4_1, SLOT_TABLE_ENTRIES, SLOT_TABLE_MISS, type ExpertClass,
} from "../src/engine/residency";
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, packExpertSlab } from "../src/engine/moe";
import { pairGemvSiluFastWgsl, gemvAccumFastWgsl, type ArenaOpts } from "../src/engine/kernels/wgsl";

const GGUF_PATH = join(homedir(), ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf");

describe("Sha256Stream (incrementale, FIPS 180-4)", () => {
  const subtle = async (data: Uint8Array): Promise<string> =>
    [...new Uint8Array(await webcrypto.subtle.digest("SHA-256", data as unknown as NodeJS.BufferSource))]
      .map((b) => b.toString(16).padStart(2, "0")).join("");
  const rand = (n: number, seed: number): Uint8Array => {
    let s = seed >>> 0;
    return Uint8Array.from({ length: n }, () => ((s = (s * 1103515245 + 12345) >>> 0) & 0xff));
  };

  it("coincide con crypto.subtle sulle taglie limite del padding", async () => {
    for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 128, 1000, 65537]) {
      const data = rand(n, 100 + n);
      const s = new Sha256Stream();
      s.update(data);
      expect(s.hex(), `n=${n}`).toBe(await subtle(data));
    }
  });

  it("update a chunk arbitrari = update unico; hex() non muta lo stato", async () => {
    const data = rand(200_001, 7);
    const s = new Sha256Stream();
    let off = 0;
    for (const step of [1, 63, 64, 65, 4096, 100_000]) {
      s.update(data.subarray(off, off + step));
      off += step;
    }
    const mid = s.hex();
    expect(s.hex()).toBe(mid); // idempotente
    s.update(data.subarray(off));
    expect(s.hex()).toBe(await subtle(data));
  });
});

describe.skipIf(!existsSync(GGUF_PATH))("GgufExpertIndex sul GGUF reale", () => {
  let idx: GgufExpertIndex;
  let byName: Map<string, { offset: number; dims: number[]; type: number; name: string }>;
  let dataOffset = 0;
  let fileSize = 0;

  beforeAll(() => {
    const HEADER_BYTES = 64 * 1024 * 1024;
    const fd = openSync(GGUF_PATH, "r");
    const buf = Buffer.alloc(HEADER_BYTES);
    readSync(fd, buf, 0, HEADER_BYTES, 0);
    closeSync(fd);
    const f = parseGguf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + HEADER_BYTES));
    byName = validateGlm47Flash(f);
    dataOffset = f.dataOffset;
    fileSize = statSync(GGUF_PATH).size;
    idx = new GgufExpertIndex(byName, dataOffset);
  });

  it("identità aritmetica: fine dell'expert 63 = base + tensorByteSize, per ogni layer MoE", () => {
    for (let l = G.denseLead; l < G.nLayer; l++) {
      for (const t of ["gate", "up", "down"] as const) {
        const info = byName.get(`blk.${l}.ffn_${t}_exps.weight`)!;
        const first = idx.ranges(l, 0)[t];
        const last = idx.ranges(l, G.nExpert - 1)[t];
        expect(first.offset, `blk.${l} ${t}`).toBe(dataOffset + info.offset);
        expect(last.offset + last.bytes - first.offset, `blk.${l} ${t}`).toBe(tensorByteSize(info as never));
        expect(last.offset + last.bytes).toBeLessThanOrEqual(fileSize);
      }
    }
  });

  it("size-class per layer: down 1.966.080 B su blk.1-4, 1.769.472 altrove", () => {
    for (let l = G.denseLead; l < G.nLayer; l++) {
      const d = idx.ranges(l, 17).down.bytes;
      expect(d, `blk.${l}`).toBe(downIsQ4_1(l) ? EXPERT_DOWN_Q4_1_BYTES : EXPERT_DOWN_Q4_0_BYTES);
      expect(idx.ranges(l, 17).gate.bytes).toBe(EXPERT_GATE_UP_BYTES);
    }
  });

  it("validazione hard: layer denso ed expert fuori range → throw", () => {
    expect(() => idx.ranges(0, 0)).toThrow();
    expect(() => idx.ranges(5, 64)).toThrow();
    expect(() => idx.ranges(5, -1)).toThrow();
  });
});

// ------------------------- ExpertCache (device mock) -------------------------

interface MockWrite { buf: object; offset: number; bytes: number }

function mkDevice(): { device: GPUDevice; writes: MockWrite[]; buffers: Array<{ size: number }> } {
  const writes: MockWrite[] = [];
  const buffers: Array<{ size: number }> = [];
  const device = {
    createBuffer: (d: { size: number }) => {
      const b = { size: d.size, destroy() { /* mock */ } };
      buffers.push(b);
      return b;
    },
    queue: {
      writeBuffer: (buf: object, offset: number, data: ArrayBufferView) =>
        writes.push({ buf, offset, bytes: data.byteLength }),
    },
  } as unknown as GPUDevice;
  return { device, writes, buffers };
}

const rawFor = (layer: number) => ({
  gate: new Uint8Array(EXPERT_GATE_UP_BYTES),
  up: new Uint8Array(EXPERT_GATE_UP_BYTES),
  down: new Uint8Array(downIsQ4_1(layer) ? EXPERT_DOWN_Q4_1_BYTES : EXPERT_DOWN_Q4_0_BYTES),
});

describe("ExpertCache — percorso slab pre-impacchettato (C3a fase 3)", () => {
  beforeAll(() => {
    (globalThis as Record<string, unknown>).GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 8, COPY_SRC: 4 };
  });
  const GIB = 1 << 30;

  // device mock che CATTURA i byte scritti (quello sopra tiene solo la taglia)
  const mkCapturingDevice = () => {
    const uploads: Uint8Array[] = [];
    const device = {
      createBuffer: (d: { size: number }) => ({ size: d.size, destroy() { /* mock */ } }),
      queue: {
        writeBuffer: (_b: object, _o: number, data: ArrayBufferView) =>
          uploads.push(new Uint8Array(data.buffer as ArrayBuffer, data.byteOffset, data.byteLength).slice()),
      },
    } as unknown as GPUDevice;
    return { device, uploads };
  };

  // byte pseudo-casuali deterministici: se il pack sbagliasse offset o ordine,
  // il confronto byte-per-byte con il percorso lento lo vedrebbe
  // loop diretto: Uint8Array.from con lambda su 1,7 MB e' troppo lento per un test
  const rnd = (n: number, seed: number): Uint8Array => {
    const out = new Uint8Array(n);
    let s = seed >>> 0;
    for (let i = 0; i < n; i++) out[i] = ((s = (s * 1103515245 + 12345) >>> 0) >>> 24) & 0xff;
    return out;
  };
  const rawSeeded = (layer: number, expert: number) => ({
    gate: rnd(EXPERT_GATE_UP_BYTES, 1000 + layer * 64 + expert),
    up: rnd(EXPERT_GATE_UP_BYTES, 5000 + layer * 64 + expert),
    down: rnd(downIsQ4_1(layer) ? EXPERT_DOWN_Q4_1_BYTES : EXPERT_DOWN_Q4_0_BYTES, 9000 + layer * 64 + expert),
  });

  it("il percorso slab carica byte IDENTICI al percorso raw+pack", () => {
    const opts = { budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 }, maxBindingBytes: GIB, maxBufferBytes: GIB, timing: true };
    // percorso lento: raw + packExpertSlab dentro ensure
    const slow = mkCapturingDevice();
    const cSlow = new ExpertCache(slow.device, opts);
    for (const [l, e] of [[5, 0], [2, 1]] as Array<[number, number]>) {
      cSlow.ensure(l, e, (ll, ee) => rawSeeded(ll, ee));
    }
    // percorso rapido: slab gia' impacchettato fuori
    const fast = mkCapturingDevice();
    const cFast = new ExpertCache(fast.device, opts);
    const slabOf = (l: number, e: number) => {
      const r = rawSeeded(l, e);
      return packExpertSlab(r.gate, r.up, r.down, downIsQ4_1(l) ? SLAB_DOWN_Q4_1 : SLAB_DOWN_Q4_0);
    };
    for (const [l, e] of [[5, 0], [2, 1]] as Array<[number, number]>) {
      cFast.ensure(l, e, { raw: (ll, ee) => rawSeeded(ll, ee), slab: slabOf });
    }
    expect(fast.uploads.length).toBe(slow.uploads.length);
    // confronto a mano: toEqual su 5,3 MB di Uint8Array e' O(n) ma con overhead
    // di deep-equality tale da sforare il timeout del test
    for (let i = 0; i < slow.uploads.length; i++) {
      const a = slow.uploads[i], b = fast.uploads[i];
      expect(b.length, `upload ${i}: taglia`).toBe(a.length);
      let diff = -1;
      for (let j = 0; j < a.length; j++) if (a[j] !== b[j]) { diff = j; break; }
      expect(diff, `upload ${i}: primo byte diverso`).toBe(-1);
    }
  });

  it("col percorso slab il pack CPU sparisce dalla telemetria", () => {
    const opts = { budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 }, maxBindingBytes: GIB, maxBufferBytes: GIB, timing: true };
    const { device } = mkCapturingDevice();
    const c = new ExpertCache(device, opts);
    const slabOf = (l: number, e: number) => {
      const r = rawSeeded(l, e);
      return packExpertSlab(r.gate, r.up, r.down, downIsQ4_1(l) ? SLAB_DOWN_Q4_1 : SLAB_DOWN_Q4_0);
    };
    c.ensure(5, 0, { raw: (ll, ee) => rawSeeded(ll, ee), slab: slabOf });
    expect(c.stats().packMs).toBe(0); // era 9,5 ms per miss nel path caldo
    expect(c.stats().misses).toBe(1);
  });

  it("senza `slab` la cache ripiega sul percorso raw (sorgenti senza file slab)", () => {
    const opts = { budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 }, maxBindingBytes: GIB, maxBufferBytes: GIB, timing: true };
    const { device, uploads } = mkCapturingDevice();
    const c = new ExpertCache(device, opts);
    c.ensure(5, 0, { raw: (ll, ee) => rawSeeded(ll, ee) });
    expect(uploads.length).toBe(1);
    expect(uploads[0].length).toBe(SLAB_DOWN_Q4_0.bytes);
  });
});

describe("ExpertCache (LRU due size-class)", () => {
  beforeAll(() => {
    (globalThis as Record<string, unknown>).GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 8, COPY_SRC: 4 };
  });
  const GIB = 1 << 30;
  const baseOpts = { maxBindingBytes: GIB, maxBufferBytes: GIB };

  it("riparto budget proporzionale al parco, cap al parco, byte entro budget", () => {
    const { device } = mkDevice();
    const B = 100 * 5_325_512; // ~100 slab in media pesata
    const c = new ExpertCache(device, { ...baseOpts, budgetBytes: B });
    const s = c.stats().slots;
    expect(s.q4_0 * SLAB_DOWN_Q4_0.bytes + s.q4_1 * SLAB_DOWN_Q4_1.bytes).toBeLessThanOrEqual(B);
    // proporzione ~ parco (2688:256), entro il rounding
    expect(s.q4_0 / s.q4_1).toBeGreaterThan(PARK_Q4_0 / PARK_Q4_1 * 0.8);
    expect(s.q4_0 / s.q4_1).toBeLessThan(PARK_Q4_0 / PARK_Q4_1 * 1.2);
    // budget enorme ⇒ cap al parco intero
    const cap = new ExpertCache(device, { ...baseOpts, budgetBytes: 64 * GIB });
    expect(cap.stats().slots).toEqual({ q4_0: PARK_Q4_0, q4_1: PARK_Q4_1 });
  });

  it("meno di 4 slot in una classe → throw (un token binda 4 expert)", () => {
    const { device } = mkDevice();
    expect(() => new ExpertCache(device, { ...baseOpts, budgetBytes: GIB, slotsOverride: { q4_0: 3, q4_1: 4 } })).toThrow();
  });

  it("LRU pura: hit fa touch, la vittima è il least-recently-used della classe", () => {
    const { device, writes } = mkDevice();
    const c = new ExpertCache(device, { ...baseOpts, budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 } });
    let reads = 0;
    const rd = (l: number) => { reads++; return rawFor(l); };
    const L = 5; // classe q4_0
    for (const e of [0, 1, 2, 3]) expect(c.ensure(L, e, rd).hit).toBe(false);
    expect(reads).toBe(4);
    expect(c.ensure(L, 0, rd).hit).toBe(true);   // touch di 0: ora LRU = 1
    expect(reads).toBe(4);                        // hit non legge
    expect(c.ensure(L, 9, rd).hit).toBe(false);  // evince 1
    expect(c.stats().evictions).toBe(1);
    expect(c.ensure(L, 0, rd).hit).toBe(true);   // 0 ancora residente
    expect(c.ensure(L, 1, rd).hit).toBe(false);  // 1 era la vittima
    const st = c.stats();
    expect(st.hits).toBe(2);
    expect(st.misses).toBe(6);
    expect(st.occupied.q4_0).toBe(4);
    expect(st.bytesUploaded).toBe(6 * SLAB_DOWN_Q4_0.bytes);
    expect(st.bytesRead).toBe(6 * (2 * EXPERT_GATE_UP_BYTES + EXPERT_DOWN_Q4_0_BYTES));
    expect(writes.length).toBe(6);
  });

  it("retention e fonte dell'hit (4d): requests esplicito, 1 − evictions/requests, hit split", () => {
    const { device } = mkDevice();
    const c = new ExpertCache(device, { ...baseOpts, budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 } });
    // cache vergine: retention null (mai NaN nel JSON), non 1.0
    expect(c.stats().retention).toBeNull();
    expect(c.stats().requests).toBe(0);
    const rd = (l: number) => rawFor(l);
    const L = 5;
    for (const e of [0, 1, 2, 3]) c.ensure(L, e, rd); // 4 miss, 0 eviction
    c.ensure(L, 0, rd); c.ensure(L, 1, rd);           // 2 hit residenti
    c.ensure(L, 8, rd); c.ensure(L, 9, rd);           // 2 miss con eviction
    const st = c.stats();
    expect(st.requests).toBe(8);
    expect(st.evictions).toBe(2);
    expect(st.retention).toBeCloseTo(1 - 2 / 8, 12);
    // la fonte dell'hit e' separata (lezione kimi-k3-in-c): oggi tutto
    // residente, il contatore prefetch esiste per lo schema (C3b) e resta 0
    expect(st.hitsResident).toBe(2);
    expect(st.hitsPrefetch).toBe(0);
    expect(st.hits).toBe(st.hitsResident + st.hitsPrefetch);
  });

  it("pinned: i top-4 del token corrente non sono mai vittime", () => {
    const { device } = mkDevice();
    const c = new ExpertCache(device, { ...baseOpts, budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 } });
    const rd = (l: number) => rawFor(l);
    const L = 7;
    for (const e of [0, 1, 2, 3]) c.ensure(L, e, rd);
    const pinAll = new Set([0, 1, 2, 3].map((e) => expertKey(L, e)));
    expect(() => c.ensure(L, 9, rd, pinAll)).toThrow();
    const pin3 = new Set([0, 1, 2].map((e) => expertKey(L, e)));
    c.ensure(L, 9, rd, pin3);                    // evince 3 (unico non pinnato)
    expect(c.ensure(L, 0, rd, pin3).hit).toBe(true);
    expect(c.ensure(L, 3, rd).hit).toBe(false);  // 3 era la vittima
  });

  it("le classi sono indipendenti (un miss q4_1 non evince q4_0)", () => {
    const { device } = mkDevice();
    const c = new ExpertCache(device, { ...baseOpts, budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 } });
    const rd = (l: number) => rawFor(l);
    for (const e of [0, 1, 2, 3]) c.ensure(6, e, rd);   // riempi q4_0
    for (const e of [0, 1, 2, 3, 4, 5]) c.ensure(2, e, rd); // q4_1: 2 eviction interne
    expect(c.stats().evictions).toBe(2);
    for (const e of [0, 1, 2, 3]) expect(c.ensure(6, e, rd).hit).toBe(true); // q4_0 intatta
  });

  it("rifiuta un device il cui limite di BINDING non regge il sotto-range piu' grande", () => {
    // Invariante nuovo (C3a it.5): la taglia del buffer la decide maxBufferSize,
    // ma il sotto-range bindato (qs, ~1.5 MB) deve stare in
    // maxStorageBufferBindingSize. Se non ci sta va detto al costruttore, non
    // scoperto al primo setBindGroup.
    const { device } = mkDevice();
    expect(() => new ExpertCache(device, {
      budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 },
      maxBindingBytes: SLAB_DOWN_Q4_0.qsBytes - 1, maxBufferBytes: GIB,
    })).toThrow(/maxStorageBufferBindingSize/);
    // al limite esatto invece passa
    expect(() => new ExpertCache(device, {
      budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 },
      maxBindingBytes: SLAB_DOWN_Q4_0.qsBytes, maxBufferBytes: GIB,
    })).not.toThrow();
  });

  it("slot (classe, buffer, offset): partizione su più buffer e bind range dei 6 segmenti", () => {
    const { device, buffers } = mkDevice();
    // maxBUFFER = 2 slab e mezzo ⇒ 2 slab/buffer ⇒ 5 slot su 3 buffer (2+2+1).
    // C3a it.5: la leva era `maxBindingBytes`, ma la taglia del BUFFER è
    // limitata da `maxBufferSize`; il binding size limita il SOTTO-RANGE
    // bindato (~1.5 MB). Cambiata la leva, non l'intento del test.
    const c = new ExpertCache(device, {
      budgetBytes: 0, slotsOverride: { q4_0: 5, q4_1: 4 },
      maxBindingBytes: GIB, maxBufferBytes: Math.floor(2.5 * SLAB_DOWN_Q4_0.bytes),
    });
    const q40 = buffers.filter((b) => b.size % SLAB_DOWN_Q4_0.bytes === 0 && b.size <= 2 * SLAB_DOWN_Q4_0.bytes);
    expect(q40.map((b) => b.size / SLAB_DOWN_Q4_0.bytes)).toEqual([2, 2, 1]);
    const rd = (l: number) => rawFor(l);
    const slots = [0, 1, 2].map((e) => c.ensure(10, e, rd).slot);
    expect(slots[0].offset).toBe(0);
    expect(slots[1].offset).toBe(SLAB_DOWN_Q4_0.bytes);
    expect(slots[1].buffer).toBe(slots[0].buffer);
    expect(slots[2].offset).toBe(0);
    expect(slots[2].buffer).not.toBe(slots[0].buffer);
    const r = slotBindRanges(slots[1]);
    expect(r.gateQs).toEqual({ buffer: slots[1].buffer, offset: SLAB_DOWN_Q4_0.bytes, size: SLAB_DOWN_Q4_0.qsBytes });
    expect(r.downScales.offset).toBe(SLAB_DOWN_Q4_0.bytes + SLAB_DOWN_Q4_0.downScales);
    expect(r.downScales.size).toBe(SLAB_DOWN_Q4_0.downScalesBytes);
    for (const k of Object.values(r)) expect(k.offset % 256).toBe(0);
  });
});

// --------------------------- regime ARENA (C3a fase 4) ----------------------
// Lo slot smette di essere un binding e diventa un indirizzo: qui si verifica
// che l'indirizzo cada ESATTAMENTE dove il regime a sotto-range bindava, per
// OGNI slot delle due classi. È il gate 3 dello slice A: se l'aritmetica del
// WGSL e i BindRange divergessero anche di un solo slot, il kernel leggerebbe
// pesi di un altro expert senza che niente esploda.
describe("ExpertCache — arena a binding fisso", () => {
  beforeAll(() => {
    (globalThis as Record<string, unknown>).GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 8, COPY_SRC: 4 };
  });
  // limiti veri del 4090 Laptop (results/engine/webgpu-limits-4090laptop-2026-08-02)
  const MAX_BUFFER = 4_294_967_292;
  const MAX_BINDING = 2_147_483_644;
  // riparto a budget 12 GiB (design §0): 2.216 + 203 slot
  const SLOTS = { q4_0: 2216, q4_1: 203 };
  const mkArena = () => {
    const { device, buffers } = mkDevice();
    const c = new ExpertCache(device, {
      budgetBytes: 0, slotsOverride: SLOTS, arena: true,
      maxBufferBytes: MAX_BUFFER, maxBindingBytes: MAX_BINDING,
    });
    return { c, buffers };
  };

  it("il buffer È il binding: la finestra la cappa il min dei due limiti", () => {
    const { c } = mkArena();
    // 2 GiB / 5.308.416 = 404 slab (q4_0), / 5.505.024 = 390 (q4_1): sono i
    // numeri del design. Col cap del solo maxBufferSize sarebbero 809 e 780.
    expect(c.arenaGeometry("q4_0").slabsPerBuf).toBe(404);
    expect(c.arenaGeometry("q4_1").slabsPerBuf).toBe(390);
    expect(c.arenaGeometry("q4_0").nBuf).toBe(6);   // ceil(2216/404)
    expect(c.arenaGeometry("q4_1").nBuf).toBe(1);   // 203 < 390
    // fuori dal regime arena il cap resta quello di it.5 (solo maxBufferSize)
    const { device } = mkDevice();
    const plain = new ExpertCache(device, {
      budgetBytes: 0, slotsOverride: SLOTS,
      maxBufferBytes: MAX_BUFFER, maxBindingBytes: MAX_BINDING,
    });
    expect(plain.arenaGeometry("q4_0").slabsPerBuf).toBe(809);
  });

  it("SLAB_W e la geometria vengono dal layout, non da costanti riscritte", () => {
    const { c } = mkArena();
    expect(c.arenaGeometry("q4_0").slabWords).toBe(SLAB_DOWN_Q4_0.bytes / 4);
    expect(c.arenaGeometry("q4_1").slabWords).toBe(SLAB_DOWN_Q4_1.bytes / 4);
    expect(c.arenaGeometry("q4_0").slabWords).toBe(1_327_104);
    expect(c.arenaGeometry("q4_1").slabWords).toBe(1_376_256);
  });

  /**
   * L'aritmetica dello slot LETTA DAL WGSL GENERATO, non riscritta in TS.
   * Senza questa estrazione il round-trip qui sotto confronterebbe TypeScript
   * con TypeScript: una mutazione dentro il generatore (p.es. `bi = (slot + 1u)
   * / SLABS_PER_BUF`) lo lascerebbe verde mentre il kernel legge lo slab
   * sbagliato. Le regex sono ANCORATE alla forma emessa: se la forma cambia, il
   * test cade e va riletto — che è il punto.
   */
  const addressingFromWgsl = (code: string) => {
    const num = (re: RegExp, what: string): number => {
      const m = re.exec(code);
      if (!m) throw new Error(`WGSL: nessun ${what} nella forma attesa`);
      return Number(m[1]);
    };
    // `let bi = slot / SLABS_PER_BUF;` e `let base = (slot % SLABS_PER_BUF) * SLAB_W;`
    const bi = /^ {2}let bi = (\w+) \/ (\w+);$/m.exec(code);
    const base = /^ {2}let base = \((\w+) % (\w+)\) \* (\w+);$/m.exec(code);
    if (!bi || !base) throw new Error("WGSL: aritmetica dello slot non nella forma attesa");
    if (bi[1] !== "slot" || bi[2] !== "SLABS_PER_BUF") throw new Error(`WGSL: bi = ${bi[1]} / ${bi[2]}`);
    if (base[1] !== "slot" || base[2] !== "SLABS_PER_BUF" || base[3] !== "SLAB_W") {
      throw new Error(`WGSL: base = (${base[1]} % ${base[2]}) * ${base[3]}`);
    }
    // `let slot = select(0u, sel.slot, ok);` con `ok = sel.slot != 0xffffffffu`
    if (!/^ {2}let slot = select\(0u, sel\.slot, ok\);$/m.test(code)) throw new Error("WGSL: slot non da Sel");
    return {
      slabWords: num(/^const SLAB_W = (\d+)u;$/m, "SLAB_W"),
      slabsPerBuf: num(/^const SLABS_PER_BUF = (\d+)u;$/m, "SLABS_PER_BUF"),
      // gli offset di sezione, sempre in word, come li emette il generatore
      gateQs: num(/^ {2}let gQ4 = \(base \+ (\d+)u\) >> 2u;$/m, "gateQs"),
      gateSc: num(/^ {2}let gSc = base \+ (\d+)u;$/m, "gateScales"),
      upQs: num(/^ {2}let uQ4 = \(base \+ (\d+)u\) >> 2u;$/m, "upQs"),
      upSc: num(/^ {2}let uSc = base \+ (\d+)u;$/m, "upScales"),
    };
  };
  const downAddressingFromWgsl = (code: string) => ({
    qs: Number(/^ {2}let qsW4 = \(base \+ (\d+)u\) >> 2u;$/m.exec(code)![1]),
    sc: Number(/^ {2}let scW = base \+ (\d+)u;$/m.exec(code)![1]),
  });
  const arenaOptsOf = (geo: { nBuf: number; slabWords: number; slabsPerBuf: number; layout: typeof SLAB_DOWN_Q4_0 }): ArenaOpts => ({
    nBuf: geo.nBuf, slabWords: geo.slabWords, slabsPerBuf: geo.slabsPerBuf,
    qsWords: geo.layout.downQs / 4, scalesWords: geo.layout.downScales / 4,
    gateQsWords: geo.layout.gateQs / 4, gateScWords: geo.layout.gateScales / 4,
    upQsWords: geo.layout.upQs / 4, upScWords: geo.layout.upScales / 4,
  });

  it("round-trip idx → (binding, base) vs slotBindRanges, per OGNI slot", () => {
    const { c, buffers } = mkArena();
    for (const cls of ["q4_0", "q4_1"] as ExpertClass[]) {
      const geo = c.arenaGeometry(cls);
      const l = geo.layout;
      const arenaBufs = c.arenaBuffers(cls);
      expect(arenaBufs.length).toBe(geo.nBuf);
      // le costanti e gli offset con cui il round-trip gira arrivano dal TESTO
      // del kernel che girerà davvero, non dalla geometria TS
      const a = arenaOptsOf(geo);
      const wg = addressingFromWgsl(pairGemvSiluFastWgsl({ K: G.dModel, N: G.dFfnExpert, arena: a }));
      const wgDown = downAddressingFromWgsl(gemvAccumFastWgsl({ kind: cls as "q4_0" | "q4_1", K: G.dFfnExpert, N: G.dModel, arena: a }));
      expect(wg.slabWords, `${cls}: SLAB_W nel WGSL`).toBe(geo.slabWords);
      expect(wg.slabsPerBuf, `${cls}: SLABS_PER_BUF nel WGSL`).toBe(geo.slabsPerBuf);
      expect([wg.gateQs * 4, wg.gateSc * 4, wg.upQs * 4, wg.upSc * 4, wgDown.qs * 4, wgDown.sc * 4])
        .toEqual([l.gateQs, l.gateScales, l.upQs, l.upScales, l.downQs, l.downScales]);
      for (let idx = 0; idx < geo.nSlots; idx++) {
        // ESATTAMENTE l'aritmetica del WGSL (in word u32, non in byte),
        // con le costanti estratte dal WGSL
        const bi = Math.floor(idx / wg.slabsPerBuf);
        const baseW = (idx % wg.slabsPerBuf) * wg.slabWords;
        const r = slotBindRanges(c.slotAt(cls, idx));
        expect(arenaBufs[bi], `${cls} slot ${idx}: binding`).toBe(r.gateQs.buffer);
        const seg: Array<[number, number]> = [
          [baseW * 4 + l.gateQs, r.gateQs.offset], [baseW * 4 + l.gateScales, r.gateScales.offset],
          [baseW * 4 + l.upQs, r.upQs.offset], [baseW * 4 + l.upScales, r.upScales.offset],
          [baseW * 4 + l.downQs, r.downQs.offset], [baseW * 4 + l.downScales, r.downScales.offset],
        ];
        for (const [got, want] of seg) expect(got, `${cls} slot ${idx}`).toBe(want);
        // l'indice in word dell'ultimo byte dello slab sta nel buffer bindato
        const size = (buffers.find((b) => b === (arenaBufs[bi] as unknown as { size: number }))!).size;
        expect(baseW * 4 + l.bytes, `${cls} slot ${idx}: dentro il buffer`).toBeLessThanOrEqual(size);
        // gli indici vec4 restano ben dentro i 2^30 dichiarati nel design
        expect((baseW + l.downQs / 4) / 4 + (G.dFfnExpert / 32) * G.dModel).toBeLessThan(2 ** 30);
      }
    }
  });

  it("arenaNeeds dà i requisiti che i kernel poi consumano davvero", () => {
    const { c } = mkArena();
    const need = arenaNeeds({
      budgetBytes: 0, slotsOverride: SLOTS, maxBufferBytes: MAX_BUFFER, maxBindingBytes: MAX_BINDING,
    });
    // il buffer più grande che la cache crea DAVVERO = la finestra chiesta
    expect(need.arenaWindowBytes).toBe(404 * SLAB_DOWN_Q4_0.bytes);
    expect(need.arenaWindowBytes).toBeLessThanOrEqual(MAX_BINDING);
    expect(need.arenaBuffers).toBe(Math.max(c.arenaGeometry("q4_0").nBuf, c.arenaGeometry("q4_1").nBuf));
    // a parco completo (fase 4c, residenza totale) i buffer diventano 7
    expect(arenaNeeds({
      budgetBytes: 0, slotsOverride: { q4_0: PARK_Q4_0, q4_1: PARK_Q4_1 },
      maxBufferBytes: MAX_BUFFER, maxBindingBytes: MAX_BINDING,
    }).arenaBuffers).toBe(7);
  });

  it("arenaBuffers conta sulla finestra chiesta, non sul tetto dell'adapter", () => {
    // Caso asimmetrico: la classe q4_1 (slab più grandi) fissa la finestra e la
    // q4_0 deve starci dentro. Contando sul tetto si otterrebbe un nBuf più
    // piccolo del vero, cioè un `maxStorageBuffersPerShaderStage` insufficiente
    // scoperto solo a createComputePipeline.
    const window3 = 3 * SLAB_DOWN_Q4_1.bytes;
    const need = arenaNeeds({
      budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 7 },
      maxBufferBytes: window3, maxBindingBytes: window3,
    });
    expect(need.arenaWindowBytes).toBe(window3);
    expect(need.arenaBuffers).toBe(3); // ceil(7/3); q4_0: ceil(4/3) = 2
    // e i buffer VERI della cache non superano mai quel numero
    const { device } = mkDevice();
    const c = new ExpertCache(device, {
      budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 7 }, arena: true,
      maxBufferBytes: window3, maxBindingBytes: window3,
    });
    for (const cls of ["q4_0", "q4_1"] as ExpertClass[]) {
      expect(c.arenaGeometry(cls).nBuf).toBeLessThanOrEqual(need.arenaBuffers);
    }
    expect(c.arenaGeometry("q4_1").nBuf).toBe(3);
  });

  it("input degeneri: errore esplicito, mai un NaN dentro requiredLimits", () => {
    const base = { maxBufferBytes: MAX_BUFFER, maxBindingBytes: MAX_BINDING };
    // budget zero senza override ⇒ 0 slot per classe: prima dava ceil(0/0) = NaN,
    // e NaN passa QUALUNQUE confronto (anche `> ARENA_BUFFERS_MAX`) fino ad
    // arrivare in requiredLimits.
    expect(() => arenaNeeds({ ...base, budgetBytes: 0 })).toThrow(/slot/);
    expect(() => arenaNeeds({ ...base, budgetBytes: 0, slotsOverride: { q4_0: 0, q4_1: 4 } })).toThrow(/q4_0/);
    expect(() => arenaNeeds({ ...base, budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 0 } })).toThrow(/q4_1/);
    // budget che non basta a una classe sola (il riparto è proporzionale al parco)
    expect(() => arenaNeeds({ ...base, budgetBytes: 10 * SLAB_DOWN_Q4_0.bytes })).toThrow(/q4_1/);
    // e quando NON solleva, i due numeri sono conteggi interi e positivi
    const ok = arenaNeeds({ ...base, budgetBytes: 12 * (1 << 30) });
    expect(Number.isInteger(ok.arenaBuffers) && ok.arenaBuffers >= 1).toBe(true);
    expect(Number.isInteger(ok.arenaWindowBytes) && ok.arenaWindowBytes >= 1).toBe(true);
  });

  it("uno slab che non sta in un binding è un errore al costruttore", () => {
    const { device } = mkDevice();
    expect(() => new ExpertCache(device, {
      budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 }, arena: true,
      maxBufferBytes: MAX_BUFFER, maxBindingBytes: SLAB_DOWN_Q4_1.bytes - 1,
    })).toThrow(/arena/);
    expect(() => arenaNeeds({
      budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 },
      maxBufferBytes: MAX_BUFFER, maxBindingBytes: SLAB_DOWN_Q4_1.bytes - 1,
    })).toThrow();
  });
});

// --------------------- slotTable (C3a fase 4, slice B) ----------------------
// La tabella expertKey → slot è ciò che permette al resolve GPU di tradurre un
// expert in indirizzo senza chiedere niente alla CPU. Qui si verifica la
// MANUTENZIONE (chi la sporca, quando arriva sulla GPU, cosa succede a una
// vittima); che il WGSL la legga con la base giusta è il ktest
// `router-resolve-slottable`, che ha una GPU vera.
describe("ExpertCache — slotTable", () => {
  beforeAll(() => {
    (globalThis as Record<string, unknown>).GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 8, COPY_SRC: 4 };
  });
  const GIB = 1 << 30;
  const baseOpts = { budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 }, maxBindingBytes: GIB, maxBufferBytes: GIB };

  // device mock che APPLICA le scritture: della tabella conta il contenuto, non
  // il numero di byte. Tiene anche l'ordine, che è l'invariante R5 del design.
  const mkTableDevice = () => {
    const order: string[] = [];
    let table: { buf: object; data: Uint32Array } | null = null;
    const device = {
      createBuffer: (d: { size: number }) => {
        const b = { size: d.size, destroy() { /* mock */ } };
        // il primo buffer da 12 032 B è la slotTable (gli slab sono ~5 MB)
        if (d.size === SLOT_TABLE_ENTRIES * 4 && !table) table = { buf: b, data: new Uint32Array(SLOT_TABLE_ENTRIES) };
        return b;
      },
      queue: {
        writeBuffer: (buf: object, offset: number, data: ArrayBufferView, dataOffset = 0, size?: number) => {
          if (table && buf === table.buf) {
            const src = data as Uint32Array;
            const n = size ?? src.length - dataOffset;
            for (let i = 0; i < n; i++) table.data[offset / 4 + i] = src[dataOffset + i];
            order.push(`table[${offset / 4}..${offset / 4 + n - 1}]`);
          } else {
            order.push(`slab@${offset}`);
          }
        },
      },
    } as unknown as GPUDevice;
    return { device, order, read: (): Uint32Array => table!.data };
  };
  const rd = (l: number) => rawFor(l);

  it("nasce tutta MISS, e senza l'opzione non esiste", () => {
    const { device, read, order } = mkTableDevice();
    const c = new ExpertCache(device, { ...baseOpts, slotTable: true });
    expect(read().every((v) => v === SLOT_TABLE_MISS)).toBe(true);
    expect(order).toEqual([`table[0..${SLOT_TABLE_ENTRIES - 1}]`]);
    expect(c.slotTableBuffer()).toBeDefined();
    expect(SLOT_TABLE_ENTRIES * 4).toBe(12_032); // design §2.5
    const { device: d2 } = mkDevice();
    expect(() => new ExpertCache(d2, baseOpts).slotTableBuffer()).toThrow(/slotTable/);
  });

  it("l'ensure pubblica lo slot, ma solo dopo il flush — e SEMPRE dopo lo slab", () => {
    const { device, read, order } = mkTableDevice();
    const c = new ExpertCache(device, { ...baseOpts, slotTable: true });
    order.length = 0;
    const s = c.ensure(5, 7, rd).slot;
    // prima del flush la GPU non sa niente: c'è solo lo slab in coda
    expect(order).toEqual(["slab@0"]);
    expect(read()[expertKey(5, 7)]).toBe(SLOT_TABLE_MISS);
    c.flushSlotTable();
    // ORDINE (R5): lo slab è già in coda quando la tabella lo indirizza
    expect(order).toEqual(["slab@0", `table[${expertKey(5, 7)}..${expertKey(5, 7)}]`]);
    expect(read()[expertKey(5, 7)]).toBe(s.idx);
    // un flush senza modifiche non scrive niente
    order.length = 0;
    c.flushSlotTable();
    expect(order).toEqual([]);
    // e un hit non sporca la tabella
    c.ensure(5, 7, rd);
    c.flushSlotTable();
    expect(order).toEqual([]);
  });

  it("un flush per layer copre l'intervallo dei 4 ensure, non 4 scritture", () => {
    const { device, read, order } = mkTableDevice();
    const c = new ExpertCache(device, { ...baseOpts, slotTable: true });
    order.length = 0;
    for (const e of [9, 2, 30, 11]) c.ensure(6, e, rd);
    c.flushSlotTable();
    expect(order.filter((o) => o.startsWith("table"))).toEqual([`table[${expertKey(6, 2)}..${expertKey(6, 30)}]`]);
    for (const e of [9, 2, 30, 11]) expect(read()[expertKey(6, e)]).toBeLessThan(4);
    // slot distinti: quattro expert non possono condividere un indirizzo
    expect(new Set([9, 2, 30, 11].map((e) => read()[expertKey(6, e)])).size).toBe(4);
  });

  it("la vittima dell'eviction torna MISS (altrimenti indirizza lo slab di un altro)", () => {
    const { device, read } = mkTableDevice();
    const c = new ExpertCache(device, { ...baseOpts, slotTable: true });
    const L = 5; // classe q4_0, 4 slot
    for (const e of [0, 1, 2, 3]) c.ensure(L, e, rd);
    c.flushSlotTable();
    const victimSlot = read()[expertKey(L, 0)];
    c.ensure(L, 9, rd); // evince 0 (LRU)
    c.flushSlotTable();
    expect(read()[expertKey(L, 0)]).toBe(SLOT_TABLE_MISS);
    expect(read()[expertKey(L, 9)]).toBe(victimSlot); // lo slot è passato di mano
    // gli altri tre non si sono mossi
    for (const e of [1, 2, 3]) expect(read()[expertKey(L, e)]).not.toBe(SLOT_TABLE_MISS);
  });

  it("le due classi vivono nella stessa tabella senza pestarsi (chiave = layer assoluto)", () => {
    const { device, read } = mkTableDevice();
    const c = new ExpertCache(device, { ...baseOpts, slotTable: true });
    c.ensure(2, 5, rd);  // q4_1
    c.ensure(6, 5, rd);  // q4_0, stesso expert, layer diverso
    c.flushSlotTable();
    expect(read()[expertKey(2, 5)]).toBe(0);
    expect(read()[expertKey(6, 5)]).toBe(0); // stesso indice slot, classi diverse
    expect(expertKey(2, 5)).not.toBe(expertKey(6, 5));
  });
});

// Le due funzioni su cui poggia la precondizione di residenza totale del modo
// `select:"gpu"` (C3a fase 4 slice C). Sono l'unica cosa che sta fra "il modo gpu
// parte" e la corruzione silenziosa, e girano PRIMA che esista una cache: qui si
// verificano senza device.
describe("precondizione di residenza totale (slice C)", () => {
  it("modelExpertPark conta il parco dei layer che ci sono, non del modello intero", () => {
    // mini-modello del ktest: blk.0 denso + blk.1 MoE ⇒ solo q4_1
    expect(modelExpertPark(2)).toEqual({ q4_0: 0, q4_1: G.nExpert });
    // il modello intero torna il parco documentato (le due costanti storiche)
    expect(modelExpertPark(G.nLayer)).toEqual({ q4_0: PARK_Q4_0, q4_1: PARK_Q4_1 });
    expect(modelExpertPark(G.denseLead)).toEqual({ q4_0: 0, q4_1: 0 }); // nessun MoE
  });

  it("il confine q4_1→q4_0 cade fra il layer 4 e il 5 (GLM47_DOWN_EXPS_Q4_1_LAST)", () => {
    const L = GLM47_DOWN_EXPS_Q4_1_LAST; // 4
    // fino a L compreso: tutto q4_1, un layer alla volta
    expect(modelExpertPark(L + 1)).toEqual({ q4_0: 0, q4_1: (L + 1 - G.denseLead) * G.nExpert });
    // il layer L+1 e' il primo q4_0: e' l'unico delta fra le due chiamate
    expect(modelExpertPark(L + 2)).toEqual({ q4_0: G.nExpert, q4_1: (L + 1 - G.denseLead) * G.nExpert });
    expect(downIsQ4_1(L)).toBe(true);
    expect(downIsQ4_1(L + 1)).toBe(false);
  });

  it("expertSlots: l'override vince sul budget, altrimenti riparto proporzionale", () => {
    expect(expertSlots({ budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 64 } }))
      .toEqual({ q4_0: 4, q4_1: 64 });
    // a budget enorme si cappa al parco: e' la condizione di residenza totale vera
    expect(expertSlots({ budgetBytes: 64 * 2 ** 30 })).toEqual({ q4_0: PARK_Q4_0, q4_1: PARK_Q4_1 });
  });
});

// ------------------- I1 del decode ottimistico (C3b fase 3) -------------------
// Fra submit e readback di coda la slotTable e' intoccabile: ensure, flush e
// il debugMarkMiss dell'harness devono ALZARE col guard armato, e tornare a
// funzionare al confine di token. Il guard si arma solo in "optimistic"
// (glmmodel): qui si testa il meccanismo in ExpertCache, non chi lo arma.
describe("ExpertCache — I1: slotTable congelata a token in volo (C3b)", () => {
  const GIB = 1 << 30;
  it("ensure/flush/debugMarkMiss alzano in volo, tornano a funzionare al confine", () => {
    const { device } = mkDevice();
    const opts = { budgetBytes: 0, slotsOverride: { q4_0: 4, q4_1: 4 }, maxBindingBytes: GIB, maxBufferBytes: GIB, slotTable: true };
    const c = new ExpertCache(device, opts);
    const reader = (ll: number) => rawFor(ll);
    c.ensure(1, 0, reader); // fuori volo: legale
    c.setInFlight(true);
    expect(() => c.ensure(1, 1, reader)).toThrow(/token in volo/);
    expect(() => c.flushSlotTable()).toThrow(/token in volo/);
    expect(() => c.debugMarkMiss([64 + 0])).toThrow(/token in volo/);
    c.setInFlight(false);
    expect(() => c.ensure(1, 1, reader)).not.toThrow();
    c.flushSlotTable();
    // il debugMarkMiss EVINCE davvero: l'expert torna miss e rioccupa uno slot
    const before = c.stats().occupied.q4_1;
    c.debugMarkMiss([64 + 1]); // chiave = layer*64 + expert (layer 1: classe q4_1)
    expect(c.stats().occupied.q4_1).toBe(before - 1);
    const again = c.ensure(1, 1, reader);
    expect(again.hit).toBe(false); // fetch reale, come un miss di capacita'
  });
});
