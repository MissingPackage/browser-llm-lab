import { describe, it, expect, beforeAll } from "vitest";
import {
  ExpertCache, expertKey,
  AUTOPIN_MIN_HIST, PIN_CAP_FRAC, REPIN_EVERY_SEL,
} from "../src/engine/residency";
import { GLM47_FLASH as G } from "../src/engine/shape";

// Unit CPU-side della policy tier.h + AUTOPIN (C3c fase 5, spec §4) sul mock
// device del pattern engine-residency: qui si verifica la MECCANICA (cap HARD,
// pin mai evinti, accumulo/persistenza additiva, no-op in lru); il delta di
// hit-rate vs LRU è la misura di fase 5 sull'harness routing, non questa suite.

const mkDevice = () =>
  ({
    createBuffer: (d: { size: number }) => ({ size: d.size, destroy() { /* mock */ } }),
    queue: { writeBuffer: () => { /* mock */ } },
  }) as unknown as GPUDevice;

// reader in forma-oggetto con slab già impacchettato (il path veloce): la
// taglia giusta la conosce la cache dal layout — basta un buffer della classe
const slabReader = {
  raw: () => { throw new Error("slab path atteso"); },
  slab: (layer: number) => new Uint8Array(layer >= 1 && layer <= 4 ? 5_505_024 : 5_308_416),
};

const mkCache = (policy: "lru" | "tier", slots = { q4_0: 64, q4_1: 16 }) =>
  new ExpertCache(mkDevice(), {
    budgetBytes: 0, slotsOverride: slots,
    maxBindingBytes: 2 ** 31 - 4, maxBufferBytes: 2 ** 32 - 4,
    policy,
  });

beforeAll(() => {
  (globalThis as Record<string, unknown>).GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 8, COPY_SRC: 4 };
});

describe("policy tier — meccanica (spec §4)", () => {
  it("in lru noteSelection è un no-op e stats.policy è null", () => {
    const c = mkCache("lru");
    c.noteSelection(5, [1, 2, 3, 4]);
    expect(c.stats().policy).toBeNull();
  });

  it("niente pin sotto AUTOPIN_MIN_HIST; poi pinna e il cap 12.5% regge", () => {
    const c = mkCache("tier");
    // carica un working set nella classe q4_0 (layer 5)
    for (let e = 0; e < 32; e++) c.ensure(5, e, slabReader);
    // selezioni concentrate sui primi 8 expert, abbastanza da superare la
    // storia minima e far scattare più passate di repin
    const rounds = Math.ceil((AUTOPIN_MIN_HIST + 2 * REPIN_EVERY_SEL) / 4);
    for (let i = 0; i < rounds; i++) c.noteSelection(5, [i % 8, (i + 1) % 8, (i + 2) % 8, (i + 3) % 8]);
    const st = c.stats();
    expect(st.policy).not.toBeNull();
    expect(st.policy!.selections).toBeGreaterThan(AUTOPIN_MIN_HIST);
    expect(st.policy!.repinPasses).toBeGreaterThan(0);
    // cap HARD: mai oltre il 12.5% degli slot della classe
    expect(st.policy!.pinSlots.q4_0).toBeLessThanOrEqual(Math.floor(PIN_CAP_FRAC * 64));
    expect(st.policy!.pinSlots.q4_1).toBeLessThanOrEqual(Math.floor(PIN_CAP_FRAC * 16));
  });

  it("gli slot pinnati non vengono MAI evinti (churn completo della classe)", () => {
    const c = mkCache("tier", { q4_0: 16, q4_1: 8 });
    for (let e = 0; e < 16; e++) c.ensure(5, e, slabReader);
    // storia fortissima sui primi 2 expert → pin (cap 12.5%×16 = 2)
    const rounds = Math.ceil((AUTOPIN_MIN_HIST + 2 * REPIN_EVERY_SEL) / 4);
    for (let i = 0; i < rounds; i++) c.noteSelection(5, [0, 1, 0, 1]);
    const pinned = c.stats().policy!.pinSlots.q4_0;
    expect(pinned).toBeGreaterThan(0);
    // churn: 64 expert nuovi passano dalla classe — il pinnato deve restare.
    // A questa confidenza il budget è 1: il pin è il top-eusage (5,0) — (5,1),
    // a pari conteggio, perde per ordine di residenza e PUÒ essere evinto.
    for (let e = 16; e < 80; e++) c.ensure(6, e % G.nExpert, slabReader);
    expect(c.ensure(5, 0, slabReader).hit).toBe(true);
  });

  it("eusage: snapshot/load ADDITIVO (colibri usage_load)", () => {
    const a = mkCache("tier");
    a.noteSelection(5, [1, 2]);
    a.noteSelection(5, [1, 3]);
    const snap = a.usageSnapshot();
    const b = mkCache("tier");
    b.noteSelection(5, [1, 9]);
    b.loadUsage(snap);
    b.loadUsage(snap); // additivo: due load = doppia storia
    const u = new Uint32Array(b.usageSnapshot().buffer);
    expect(u[expertKey(5, 1)]).toBe(1 + 2 * 2); // 1 live + 2×2 dai load
    expect(u[expertKey(5, 2)]).toBe(2);
    expect(u[expertKey(5, 9)]).toBe(1);
    // taglia sbagliata ⇒ throw esplicito
    expect(() => b.loadUsage(new Uint8Array(8))).toThrow(/eusage/);
    // in lru: API rifiutata, non silente
    expect(() => mkCache("lru").usageSnapshot()).toThrow(/tier/);
  });

  it("tutti-pinnati + caller-pinned: l'eviction rifiuta con messaggio, non degrada", () => {
    const c = mkCache("tier", { q4_0: 8, q4_1: 8 });
    for (let e = 0; e < 8; e++) c.ensure(5, e, slabReader);
    // il caller pinna TUTTI i residenti: nessuna vittima possibile
    const pinned = new Set<number>();
    for (let e = 0; e < 8; e++) pinned.add(expertKey(5, e));
    expect(() => c.ensure(5, 60, slabReader, pinned)).toThrow(/vittima/);
  });
});
