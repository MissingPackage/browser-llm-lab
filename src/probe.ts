import type { DeviceProbe } from "./schema";

const LIMIT_KEYS = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxStorageBuffersPerShaderStage",
] as const;

export async function probeWebGPU(
  gpu: GPU | undefined,
  nav: { userAgent: string; deviceMemory?: number },
): Promise<DeviceProbe> {
  const base: DeviceProbe = {
    webgpu: false,
    adapterInfo: null,
    limits: null,
    userAgent: nav.userAgent,
    deviceMemoryGB: nav.deviceMemory ?? null,
  };
  if (!gpu) return base;

  const adapter = await gpu.requestAdapter().catch(() => null);
  if (!adapter) return base;

  // Chrome recente: adapter.info; fallback legacy: requestAdapterInfo()
  const a = adapter as GPUAdapter & { requestAdapterInfo?: () => Promise<GPUAdapterInfo> };
  const info = a.info ?? (a.requestAdapterInfo ? await a.requestAdapterInfo() : null);

  const limits: Record<string, number> = {};
  for (const k of LIMIT_KEYS) {
    const v = (adapter.limits as unknown as Record<string, number>)[k];
    if (typeof v === "number") limits[k] = v;
  }

  return {
    ...base,
    webgpu: true,
    adapterInfo: info
      ? {
          vendor: info.vendor ?? "",
          architecture: info.architecture ?? "",
          device: info.device ?? "",
          description: info.description ?? "",
        }
      : null,
    limits,
  };
}
