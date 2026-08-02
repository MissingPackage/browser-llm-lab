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
import { validateGlm47Flash, GLM47_FLASH as G } from "../src/engine/shape";
import {
  Sha256Stream, GgufExpertIndex, downIsQ4_1,
  EXPERT_GATE_UP_BYTES, EXPERT_DOWN_Q4_0_BYTES, EXPERT_DOWN_Q4_1_BYTES,
} from "../src/engine/expertstore";
import { ExpertCache, expertKey, slotBindRanges, PARK_Q4_0, PARK_Q4_1 } from "../src/engine/residency";
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "../src/engine/moe";

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
