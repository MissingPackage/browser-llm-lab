import type { DeviceProbe } from "./schema";

const LIMIT_KEYS = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxStorageBuffersPerShaderStage",
] as const;

// 128 MiB: cap osservato su llvmpipe (Firefox 152 silent-fallback a CPU, verificato via
// about:support). Confermato in README §"Cosa abbiamo verificato dal vivo".
const SOFTWARE_ADAPTER_MAX_BUFFER_BYTES = 134217728;

export function parseBrowser(userAgent: string): { name: string; version: string } {
  const firefox = userAgent.match(/Firefox\/([\d.]+)/);
  if (firefox) return { name: "firefox", version: firefox[1] };
  const edge = userAgent.match(/Edg\/([\d.]+)/);
  if (edge) return { name: "edge", version: edge[1] };
  const chrome = userAgent.match(/Chrome\/([\d.]+)/);
  if (chrome) return { name: "chrome", version: chrome[1] };
  const safari = userAgent.match(/Version\/([\d.]+).*Safari/);
  if (safari) return { name: "safari", version: safari[1] };
  return { name: "unknown", version: "" };
}

function detectSoftwareAdapter(vendor: string, architecture: string, limits: Record<string, number>): string[] {
  const maxBuffer = limits.maxBufferSize;
  if (vendor === "" && architecture === "" && typeof maxBuffer === "number" && maxBuffer <= SOFTWARE_ADAPTER_MAX_BUFFER_BYTES) {
    return [
      `software-adapter: vendor vuoto + maxBufferSize<=128MiB (${maxBuffer} B) — probabile fallback CPU (llvmpipe/SwiftShader)`,
    ];
  }
  return [];
}

export async function probeWebGPU(
  gpu: GPU | undefined,
  nav: { userAgent: string; deviceMemory?: number },
): Promise<DeviceProbe> {
  const base: DeviceProbe = {
    webgpu: false,
    adapterInfo: null,
    limits: null,
    features: [],
    userAgent: nav.userAgent,
    deviceMemoryGB: nav.deviceMemory ?? null,
    browser: parseBrowser(nav.userAgent),
    anomalies: [],
  };
  if (!gpu) return base;

  const adapter = await gpu.requestAdapter().catch(() => null);
  if (!adapter) return base;

  // Chrome recente: adapter.info; fallback legacy: requestAdapterInfo()
  const a = adapter as GPUAdapter & { requestAdapterInfo?: () => Promise<GPUAdapterInfo> };
  const info = a.info ?? (a.requestAdapterInfo ? await a.requestAdapterInfo().catch(() => null) : null);

  const limits: Record<string, number> = {};
  for (const k of LIMIT_KEYS) {
    const v = (adapter.limits as unknown as Record<string, number>)[k];
    if (typeof v === "number") limits[k] = v;
  }

  const features = adapter.features ? Array.from(adapter.features as unknown as Iterable<string>) : [];
  const vendor = info?.vendor ?? "";
  const architecture = info?.architecture ?? "";

  return {
    ...base,
    webgpu: true,
    adapterInfo: info
      ? { vendor, architecture, device: info.device ?? "", description: info.description ?? "" }
      : null,
    limits,
    features,
    anomalies: detectSoftwareAdapter(vendor, architecture, limits),
  };
}
