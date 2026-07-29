import { describe, it, expect } from "vitest";
import {
  newCounters, wrapTimed, wrapCounted, installGpuProfPatch, measureClockQuantum,
  type GpuGlobals,
} from "../src/prof/profiler";

// Clock finto: avanza di `step` ms a ogni lettura — deterministico, niente GPU/DOM.
function fakeNow(step: number): () => number {
  let t = 0;
  return () => (t += step);
}

describe("prof counters", () => {
  it("newCounters: tutto a zero", () => {
    const c = newCounters();
    expect(Object.values(c).every((v) => v === 0)).toBe(true);
  });

  it("wrapTimed: conta, cumula il tempo, preserva this/args/return", () => {
    const c = newCounters();
    const proto = {
      dispatchWorkgroups(this: { tag: string }, x: number, y: number) {
        return `${this.tag}:${x},${y}`;
      },
    };
    expect(wrapTimed(proto, "dispatchWorkgroups", c, "dispatch", "tDispatch", fakeNow(2))).toBe(true);
    const obj = Object.assign(Object.create(proto) as typeof proto & { tag: string }, { tag: "enc" });
    expect(obj.dispatchWorkgroups(3, 4)).toBe("enc:3,4");
    expect(obj.dispatchWorkgroups(5, 6)).toBe("enc:5,6");
    expect(c.dispatch).toBe(2);
    expect(c.tDispatch).toBe(4); // 2 chiamate × (t1-t0)=2 ms col clock finto
  });

  it("wrapTimed: metodo assente -> false, contatori intatti", () => {
    const c = newCounters();
    expect(wrapTimed({}, "submit", c, "submit", "tSubmit", fakeNow(1))).toBe(false);
    expect(wrapTimed(undefined, "submit", c, "submit", "tSubmit", fakeNow(1))).toBe(false);
    expect(c.submit).toBe(0);
  });

  it("wrapCounted: conta senza timer e preserva il return", () => {
    const c = newCounters();
    const proto = { mapAsync: () => Promise.resolve("ok") };
    expect(wrapCounted(proto, "mapAsync", c, "mapAsync")).toBe(true);
    void proto.mapAsync();
    expect(c.mapAsync).toBe(1);
  });

  it("installGpuProfPatch: set completo -> nessun mancante, wrap attive", () => {
    const c = newCounters();
    const g: GpuGlobals = {
      GPUComputePassEncoder: { prototype: { dispatchWorkgroups: () => 0, end: () => 0 } },
      GPUDevice: { prototype: { createBindGroup: () => 0, createCommandEncoder: () => 0 } },
      GPUCommandEncoder: { prototype: { beginComputePass: () => 0 } },
      GPUQueue: { prototype: { writeBuffer: () => 0, submit: () => 0, onSubmittedWorkDone: () => 0 } },
      GPUBuffer: { prototype: { mapAsync: () => 0 } },
    };
    expect(installGpuProfPatch(g, c, fakeNow(1))).toEqual([]);
    (g.GPUComputePassEncoder!.prototype as { dispatchWorkgroups: () => unknown }).dispatchWorkgroups();
    (g.GPUQueue!.prototype as { submit: () => unknown }).submit();
    expect(c.dispatch).toBe(1);
    expect(c.submit).toBe(1);
  });

  it("installGpuProfPatch: global assenti -> tutti i 9 metodi fra i mancanti", () => {
    const missing = installGpuProfPatch({}, newCounters(), fakeNow(1));
    expect(missing).toHaveLength(9);
    expect(missing).toContain("dispatchWorkgroups");
    expect(missing).toContain("onSubmittedWorkDone");
  });

  it("measureClockQuantum: clock che avanza -> il passo; clock fermo -> null", () => {
    expect(measureClockQuantum(fakeNow(0.005))).toBeCloseTo(0.005, 10);
    expect(measureClockQuantum(() => 42)).toBeNull();
  });
});
