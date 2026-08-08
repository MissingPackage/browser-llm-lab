import { describe, it, expect } from "vitest";
import {
  slabBudgetCtxAware, slabWorkBytes, expertSlots,
  KV_PER_TOKEN_BYTES, NON_EXPERT_BYTES, SLAB_RESERVE_BYTES, MIN_SLOTS,
  PARK_Q4_0, PARK_Q4_1,
} from "../src/engine/residency";
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "../src/engine/moe";
import { GLM47_FLASH as G } from "../src/engine/shape";

// Unit CPU-side sulla formula del budget slab ctx-aware (C3c fase 3, spec
// 2026-08-08 §2): la formula è pura — monotonia, clamp al parco, MIN_SLOTS,
// riproduzione dei punti noti dai loro input misurati. La prova su device è
// la run glmbench di fase 3 (ctx 6k senza OOM), non questa suite.

const MiB = 2 ** 20;
const GiB = 2 ** 30;

describe("slabBudgetCtxAware — aritmetica", () => {
  it("KV per token = nLayer × keyLen × f32 (il 108 288 del probe, non il 54 KB stale)", () => {
    expect(KV_PER_TOKEN_BYTES).toBe(G.nLayer * G.keyLen * 4);
    expect(KV_PER_TOKEN_BYTES).toBe(108_288);
    // probe vram-ceiling it.19: required.kvBytes = 56 851 200 @ctx525
    expect(525 * KV_PER_TOKEN_BYTES).toBe(56_851_200);
  });

  it("il budget è la sottrazione dichiarata, addendo per addendo", () => {
    const ceiling = 15 * GiB;
    const b = slabBudgetCtxAware({ allocCeilingBytes: ceiling, ctxMax: 1024 });
    expect(b.kvBytes).toBe(1024 * KV_PER_TOKEN_BYTES);
    expect(b.workBytes).toBe(slabWorkBytes(1024));
    expect(b.budgetBytes).toBe(ceiling - NON_EXPERT_BYTES - b.kvBytes - b.workBytes - SLAB_RESERVE_BYTES);
    expect(b.slots).toEqual(expertSlots({ budgetBytes: b.budgetBytes }));
  });

  it("monotonia: più contesto ⇒ budget minore, mai maggiore", () => {
    const ceiling = 15 * GiB;
    let prev = Infinity;
    for (const ctx of [525, 2048, 4096, 6144, 8192]) {
      const b = slabBudgetCtxAware({ allocCeilingBytes: ceiling, ctxMax: ctx });
      expect(b.budgetBytes).toBeLessThan(prev);
      prev = b.budgetBytes;
    }
  });

  it("clamp al parco: un ceiling enorme non inventa slot oltre i 2944", () => {
    const b = slabBudgetCtxAware({ allocCeilingBytes: 64 * GiB, ctxMax: 525 });
    expect(b.slots.q4_0).toBe(PARK_Q4_0);
    expect(b.slots.q4_1).toBe(PARK_Q4_1);
  });

  it("MIN_SLOTS (pin-for-replay c3b I3): sotto il minimo si RIFIUTA, non si degrada", () => {
    // budget appena sotto il fabbisogno dei minimi per classe
    const minBudget = MIN_SLOTS.q4_0 * SLAB_DOWN_Q4_0.bytes + MIN_SLOTS.q4_1 * SLAB_DOWN_Q4_1.bytes;
    const ceilingTooSmall = minBudget * 0.9 + NON_EXPERT_BYTES + 525 * KV_PER_TOKEN_BYTES
      + slabWorkBytes(525) + SLAB_RESERVE_BYTES;
    expect(() => slabBudgetCtxAware({ allocCeilingBytes: Math.floor(ceilingTooSmall), ctxMax: 525 }))
      .toThrow(/pin-for-replay/);
    // budget <= 0: messaggio esplicito con gli addendi
    expect(() => slabBudgetCtxAware({ allocCeilingBytes: 1 * GiB, ctxMax: 525 }))
      .toThrow(/budget .* <= 0/);
  });

  it("riproduce il regime attuale dai suoi input misurati (probe sessione minima)", () => {
    // probe it.19, sessione minima: total 16 376 − used 1 067 − reserved 429
    // = 14 880 MiB liberi. La formula deve dare ALMENO il regime b12 (12 GiB,
    // il punto di produzione) e MAI più del tetto empirico OOM-free di c3b
    // (12.88 GiB): la riserva tarata a 512 MiB (due punti OOM, c3c it.3) la
    // rende conservativa di ~180 MiB rispetto al tetto — dichiarato, non un bug.
    const ceiling = (16_376 - 1_067 - 429) * MiB;
    const b = slabBudgetCtxAware({ allocCeilingBytes: ceiling, ctxMax: 525 });
    expect(b.budgetBytes).toBeGreaterThanOrEqual(12 * GiB);
    expect(b.budgetBytes).toBeLessThanOrEqual(Math.floor(12.88 * GiB));
    // e a ctx 6144 il budget cede ~esattamente il delta KV+work (665 MB + partials)
    const b6k = slabBudgetCtxAware({ allocCeilingBytes: ceiling, ctxMax: 6144 });
    expect(b.budgetBytes - b6k.budgetBytes)
      .toBe((6144 - 525) * KV_PER_TOKEN_BYTES + slabWorkBytes(6144) - slabWorkBytes(525));
  });
});
