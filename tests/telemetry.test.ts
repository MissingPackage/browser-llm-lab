// Test di `src/engine/telemetry.ts` (C3a fase 4d): il nucleo unico dei
// contatori dei due path. Il contratto che conta: cumulativi diffabili e
// null contagioso su gpuBusyMs (null = "non misurato", mai 0, e una finestra
// con un estremo non misurato non e' una misura).
import { describe, expect, it } from "vitest";
import { diffCounters, zeroCounters, type CoreCounters } from "../src/engine/telemetry";
import type { GlmTelemetry } from "../src/engine/glmmodel";
import type { EngineTelemetry } from "../src/engine/gpuforward";

const at = (over: Partial<CoreCounters>): CoreCounters => ({ ...zeroCounters(), ...over });

describe("diffCounters", () => {
  it("finestra = after − before su ogni contatore", () => {
    const before = at({ forwards: 10, encodeCpuMs: 5, submits: 12, dispatches: 1400, gpuBusyMs: 100, gpuPasses: 40 });
    const after = at({ forwards: 74, encodeCpuMs: 21.5, submits: 76, dispatches: 91_300, gpuBusyMs: 3300, gpuPasses: 2960 });
    expect(diffCounters(after, before)).toEqual({
      forwards: 64, encodeCpuMs: 16.5, submits: 64, dispatches: 89_900, gpuBusyMs: 3200, gpuPasses: 2920,
    });
  });

  it("gpuBusyMs: null contagia la finestra (un estremo non misurato ⇒ non misurata)", () => {
    const on = at({ gpuBusyMs: 50, gpuPasses: 10 });
    const off = at({ gpuBusyMs: null });
    expect(diffCounters(on, off).gpuBusyMs).toBeNull();
    expect(diffCounters(off, on).gpuBusyMs).toBeNull();
    expect(diffCounters(off, off).gpuBusyMs).toBeNull();
    expect(diffCounters(on, on).gpuBusyMs).toBe(0); // misurato e fermo ≠ non misurato
  });

  it("zeroCounters: neutro sui contatori sempre-contati, contagioso su gpuBusyMs", () => {
    const t = at({ forwards: 3, encodeCpuMs: 1, submits: 3, dispatches: 9, gpuBusyMs: 2, gpuPasses: 3 });
    // zeroCounters parte NON misurato (gpuBusyMs null): una finestra ancorata
    // li' e' non misurata per contratto, anche se l'estremo destro misura
    expect(diffCounters(t, zeroCounters())).toEqual({ ...t, gpuBusyMs: null });
    expect(diffCounters(t, at({ gpuBusyMs: 0 }))).toEqual(t); // baseline MISURATA a 0: neutra davvero
  });
});

describe("aderenza dei due path al nucleo (compile-time, esercitata)", () => {
  // Se GlmTelemetry o EngineTelemetry smettessero di estendere CoreCounters
  // queste assegnazioni non compilano piu' — il test rende l'aderenza un
  // fatto del build, non una convenzione.
  it("GlmTelemetry ed EngineTelemetry sono CoreCounters", () => {
    const glm: GlmTelemetry = {
      ...zeroCounters(),
      on: false, ensureMs: 0, routerWaitMs: 0, tailWaitMs: 0, routerSyncs: 0,
      selMiss: 0, dirtyTokens: 0, replays: 0, replayLayers: 0, repairMs: 0,
      gpuPassOverflow: 0, gpuByCatMs: null, gpuByCatPasses: null,
      prefetch: null, // C3c fase 4: null = spento (schema unico)
    };
    const qwen: EngineTelemetry = {
      ...zeroCounters(),
      encodeCpuMsPerToken: 0, gpuMsPerToken: null, timestampNote: "",
    };
    const asCore = (c: CoreCounters): CoreCounters => c;
    expect(asCore(glm).dispatches).toBe(0);
    expect(asCore(qwen).submits).toBe(0);
  });
});
