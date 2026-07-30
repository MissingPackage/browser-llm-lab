import { describe, it, expect } from "vitest";
import { planDecodeBatch, trimAtEos, DECODE_K_MAX, DECODE_SLOT_STRIDE } from "../src/engine/decodebatch";

// Unit CPU-side sul piano del decode loop multi-step (spec B2 §Decode loop
// multi-step e §Unit) — convenzione CI-senza-GPU: geometria del batch (posizioni
// consecutive, slot uniform, bound) e semantica EOS/crop; l'identità numerica
// K>1 ≡ K=1 la verifica il gate token-identity (run GPU).

describe("planDecodeBatch", () => {
  it("k step a posizioni consecutive, slot a stride 256", () => {
    const plan = planDecodeBatch(468, 8, 1024);
    expect(plan.length).toBe(8);
    plan.forEach((s, i) => {
      expect(s.pos).toBe(468 + i);
      expect(s.slotOffset).toBe(i * DECODE_SLOT_STRIDE);
    });
  });

  it("k fuori range o oltre ctxMax ⇒ throw (contratto hard, postura ds4)", () => {
    expect(() => planDecodeBatch(0, 0, 1024)).toThrow();
    expect(() => planDecodeBatch(0, DECODE_K_MAX + 1, 1024)).toThrow();
    expect(() => planDecodeBatch(0, 2.5 as unknown as number, 1024)).toThrow();
    expect(() => planDecodeBatch(1020, 8, 1024)).toThrow();
    expect(planDecodeBatch(1016, 8, 1024).length).toBe(8); // limite esatto ok
  });
});

describe("trimAtEos — semantica EOS mid-batch (crop esatto da B1)", () => {
  it("niente EOS: tutto tenuto, cropTo = posStart + k (nessun crop necessario)", () => {
    const r = trimAtEos([5, 6, 7, 8], 100, 999);
    expect(r.kept).toEqual([5, 6, 7, 8]);
    expect(r.stop).toBe(false);
    expect(r.cropTo).toBe(104);
  });

  it("EOS a metà batch: tenuto FINO all'EOS incluso, cropTo taglia il resto", () => {
    const r = trimAtEos([5, 999, 7, 8], 100, 999);
    expect(r.kept).toEqual([5, 999]);
    expect(r.stop).toBe(true);
    expect(r.cropTo).toBe(102); // kvLen dopo il crop: le righe 102-103 sono garbage mai letto
  });

  it("EOS al primo e all'ultimo slot", () => {
    expect(trimAtEos([999, 1, 2], 50, 999)).toEqual({ kept: [999], stop: true, cropTo: 51 });
    expect(trimAtEos([1, 2, 999], 50, 999)).toEqual({ kept: [1, 2, 999], stop: true, cropTo: 53 });
  });
});
