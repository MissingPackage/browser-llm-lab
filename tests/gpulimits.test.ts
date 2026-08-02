// Test di `src/engine/gpulimits.ts` (C3a fase 3) + dell'invariante di
// dimensionamento dei buffer slab in `residency.ts`.
//
// Nascono da una review avversaria che ha osservato: il modulo aveva copertura
// ZERO e nessun test sarebbe fallito se `negotiateLimits` avesse restituito
// `{}`. Ognuno di questi casi discrimina un'implementazione sbagliata reale.
import { describe, expect, it } from "vitest";
import { negotiateLimits } from "../src/engine/gpulimits";
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "../src/engine/moe";

// Limiti misurati sul 4090 Laptop (results/engine/webgpu-limits-4090laptop-2026-08-02.json)
const ADAPTER_4090 = {
  maxBufferSize: 4294967292,
  maxStorageBufferBindingSize: 2147483644,
  maxStorageBuffersPerShaderStage: 16,
  maxComputeWorkgroupStorageSize: 49152,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
};
const fakeAdapter = (limits: Record<string, number>): GPUAdapter =>
  ({ limits }) as unknown as GPUAdapter;

describe("negotiateLimits", () => {
  it("chiede il massimo che l'adapter concede, non una costante", () => {
    const got = negotiateLimits(fakeAdapter(ADAPTER_4090));
    expect(got.maxStorageBuffersPerShaderStage).toBe(16);
    expect(got.maxComputeInvocationsPerWorkgroup).toBe(1024);
    expect(got.maxBufferSize).toBe(4294967292);
  });

  it("non chiede MAI piu' del massimo dell'adapter (requestDevice rifiuterebbe)", () => {
    const lim = ADAPTER_4090 as Record<string, number>;
    for (const [k, v] of Object.entries(negotiateLimits(fakeAdapter(lim)))) {
      expect(v).toBeLessThanOrEqual(lim[k]);
    }
  });

  it("`caps` applica un minimo, non sostituisce il valore", () => {
    // cap SOPRA il valore dell'adapter: deve restare quello dell'adapter
    const alto = negotiateLimits(fakeAdapter(ADAPTER_4090), { maxBufferSize: 8e9 });
    expect(alto.maxBufferSize).toBe(4294967292);
    // cap SOTTO: vince il cap (e' il caso di ktest, che tiene i buffer piccoli)
    const basso = negotiateLimits(fakeAdapter(ADAPTER_4090), { maxBufferSize: 1 << 30 });
    expect(basso.maxBufferSize).toBe(1 << 30);
  });

  it("salta i limiti che l'adapter non espone invece di chiedere 0", () => {
    // chiedere 0 sarebbe 'peggio del default': legale per spec, disastroso
    const got = negotiateLimits(fakeAdapter({ maxBufferSize: 1024 }));
    expect(got.maxBufferSize).toBe(1024);
    expect("maxComputeInvocationsPerWorkgroup" in got).toBe(false);
    expect(Object.values(got).every((v) => v > 0)).toBe(true);
  });

  it("riallinea le coppie vincolate dalla spec (X <= invocations)", () => {
    // cappare solo le invocazioni lascerebbe X=1024 con invocations=256: il
    // device nasce valido e poi ogni workgroup_size(1024) fallisce a runtime
    const got = negotiateLimits(fakeAdapter(ADAPTER_4090), { maxComputeInvocationsPerWorkgroup: 256 });
    expect(got.maxComputeInvocationsPerWorkgroup).toBe(256);
    expect(got.maxComputeWorkgroupSizeX).toBeLessThanOrEqual(256);
  });

  it("cappare maxBufferSize cappa anche il binding size", () => {
    const got = negotiateLimits(fakeAdapter(ADAPTER_4090), { maxBufferSize: 1 << 30 });
    expect(got.maxStorageBufferBindingSize).toBeLessThanOrEqual(1 << 30);
  });
});

describe("dimensionamento dei buffer slab (invariante di residency)", () => {
  // Regressione: il cap era `min(maxBindingBytes, maxBufferBytes)`, che su
  // NVIDIA (binding 2 GiB-4, buffer 4 GiB-4) teneva i buffer a meta'.
  const slabsPerBuffer = (maxBufferBytes: number, layout: { bytes: number }) =>
    Math.max(1, Math.floor(maxBufferBytes / layout.bytes));

  it("il sotto-range bindato e' ordini di grandezza sotto il limite di binding", () => {
    // e' il motivo per cui cappare il BUFFER col limite del BINDING e' sbagliato
    for (const l of [SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1]) {
      const maxRange = Math.max(l.qsBytes, l.gateScalesBytes, l.downScalesBytes);
      expect(maxRange).toBeLessThan(2 * 1024 * 1024);
      expect(maxRange).toBeLessThan(ADAPTER_4090.maxStorageBufferBindingSize);
    }
  });

  it("con maxBufferSize a 4 GiB entrano piu' slab per buffer che a 2 GiB", () => {
    const a2 = slabsPerBuffer(2147483644, SLAB_DOWN_Q4_0);
    const a4 = slabsPerBuffer(ADAPTER_4090.maxBufferSize, SLAB_DOWN_Q4_0);
    expect(a2).toBe(404);
    expect(a4).toBe(809); // 809 x 5.308.416 = 4.294.508.544 <= 4.294.967.292
    // 2216 slot q4_0 a budget 12 GiB: 6 buffer prima, 3 dopo
    expect(Math.ceil(2216 / a2)).toBe(6);
    expect(Math.ceil(2216 / a4)).toBe(3);
  });
});
