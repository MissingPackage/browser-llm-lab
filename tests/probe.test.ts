import { describe, it, expect } from "vitest";
import { probeWebGPU, parseBrowser } from "../src/probe";

const fakeAdapter = {
  info: { vendor: "nvidia", architecture: "ada", device: "", description: "" },
  limits: { maxBufferSize: 4294967296, maxStorageBufferBindingSize: 2147483644 },
  features: new Set(["shader-f16", "timestamp-query"]),
};

describe("parseBrowser", () => {
  it("recognizes firefox", () => {
    expect(parseBrowser("Mozilla/5.0 (X11; Linux x86_64; rv:152.0) Gecko/20100101 Firefox/152.0")).toEqual({
      name: "firefox",
      version: "152.0",
    });
  });

  it("recognizes chrome (not edge)", () => {
    expect(
      parseBrowser("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"),
    ).toEqual({ name: "chrome", version: "150.0.0.0" });
  });

  it("recognizes edge, not chrome", () => {
    expect(
      parseBrowser(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0",
      ),
    ).toEqual({ name: "edge", version: "150.0.0.0" });
  });

  it("falls back to unknown", () => {
    expect(parseBrowser("some-weird-ua")).toEqual({ name: "unknown", version: "" });
  });
});

describe("probeWebGPU", () => {
  it("reports absence of webgpu", async () => {
    const p = await probeWebGPU(undefined, { userAgent: "ua" });
    expect(p.webgpu).toBe(false);
    expect(p.adapterInfo).toBeNull();
    expect(p.features).toEqual([]);
    expect(p.anomalies).toEqual([]);
  });

  it("extracts adapter info, limits and features", async () => {
    const gpu = { requestAdapter: async () => fakeAdapter } as unknown as GPU;
    const p = await probeWebGPU(gpu, { userAgent: "Mozilla/5.0 Chrome/150.0.0.0 Safari/537.36", deviceMemory: 32 });
    expect(p.webgpu).toBe(true);
    expect(p.adapterInfo?.vendor).toBe("nvidia");
    expect(p.limits?.maxStorageBufferBindingSize).toBe(2147483644);
    expect(p.deviceMemoryGB).toBe(32);
    expect(p.features.sort()).toEqual(["shader-f16", "timestamp-query"]);
    expect(p.browser).toEqual({ name: "chrome", version: "150.0.0.0" });
    expect(p.anomalies).toEqual([]);
  });

  it("handles requestAdapter returning null", async () => {
    const gpu = { requestAdapter: async () => null } as unknown as GPU;
    const p = await probeWebGPU(gpu, { userAgent: "ua" });
    expect(p.webgpu).toBe(false);
  });

  it("handles requestAdapter rejection", async () => {
    const gpu = {
      requestAdapter: async () => {
        throw new Error("gpu lost");
      },
    } as unknown as GPU;
    const p = await probeWebGPU(gpu, { userAgent: "ua" });
    expect(p.webgpu).toBe(false);
  });

  it("degrades to null info when legacy requestAdapterInfo rejects", async () => {
    const adapter = {
      requestAdapterInfo: async () => {
        throw new Error("denied");
      },
      limits: { maxBufferSize: 1024 },
      features: new Set<string>(),
    };
    const gpu = { requestAdapter: async () => adapter } as unknown as GPU;
    const p = await probeWebGPU(gpu, { userAgent: "ua" });
    expect(p.webgpu).toBe(true);
    expect(p.adapterInfo).toBeNull();
    expect(p.limits?.maxBufferSize).toBe(1024);
  });

  it("flags software-adapter: empty vendor + maxBufferSize<=128MiB (llvmpipe/SwiftShader signature)", async () => {
    const softwareAdapter = {
      info: { vendor: "", architecture: "", device: "", description: "Software Rasterizer" },
      limits: { maxBufferSize: 134217728, maxStorageBufferBindingSize: 134217728 },
      features: new Set<string>(["shader-f16"]),
    };
    const gpu = { requestAdapter: async () => softwareAdapter } as unknown as GPU;
    const p = await probeWebGPU(gpu, { userAgent: "Mozilla/5.0 Firefox/152.0" });
    expect(p.webgpu).toBe(true);
    expect(p.anomalies).toHaveLength(1);
    expect(p.anomalies[0]).toContain("software-adapter");
  });

  it("does not flag a real GPU with empty vendor but large maxBufferSize", async () => {
    const partialInfoRealGpu = {
      info: { vendor: "", architecture: "", device: "", description: "" },
      limits: { maxBufferSize: 4294967296 },
      features: new Set<string>(),
    };
    const gpu = { requestAdapter: async () => partialInfoRealGpu } as unknown as GPU;
    const p = await probeWebGPU(gpu, { userAgent: "ua" });
    expect(p.anomalies).toEqual([]);
  });
});
