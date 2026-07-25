import { describe, it, expect } from "vitest";
import { probeWebGPU } from "../src/probe";

const fakeAdapter = {
  info: { vendor: "nvidia", architecture: "ada", device: "", description: "" },
  limits: { maxBufferSize: 4294967296, maxStorageBufferBindingSize: 2147483644 },
};

describe("probeWebGPU", () => {
  it("reports absence of webgpu", async () => {
    const p = await probeWebGPU(undefined, { userAgent: "ua" });
    expect(p.webgpu).toBe(false);
    expect(p.adapterInfo).toBeNull();
  });

  it("extracts adapter info and limits", async () => {
    const gpu = { requestAdapter: async () => fakeAdapter } as unknown as GPU;
    const p = await probeWebGPU(gpu, { userAgent: "ua", deviceMemory: 32 });
    expect(p.webgpu).toBe(true);
    expect(p.adapterInfo?.vendor).toBe("nvidia");
    expect(p.limits?.maxStorageBufferBindingSize).toBe(2147483644);
    expect(p.deviceMemoryGB).toBe(32);
  });

  it("handles requestAdapter returning null", async () => {
    const gpu = { requestAdapter: async () => null } as unknown as GPU;
    const p = await probeWebGPU(gpu, { userAgent: "ua" });
    expect(p.webgpu).toBe(false);
  });
});
