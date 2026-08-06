// Test di `src/engine/gpudevice.ts` (C3a fase 4d, emendamento 6).
//
// Due gruppi. Il primo è STRUTTURALE e vale per costruzione: scansiona TUTTO
// src/ e pretende che `requestAdapter`/`requestDevice` vivano solo nel punto
// unico (più l'allowlist motivata). È il test che il done-when della 4d
// chiede — "listener + negoziazione su OGNI creatore di device": invece di
// verificare ogni sito uno per uno (lista che marcisce), vieta l'esistenza di
// siti fuori dal helper, e il secondo gruppo verifica che il helper faccia
// entrambe le cose. Un nuovo harness che chiama `requestDevice` a mano fa
// diventare rosso questo file, non un lint.
import { describe, expect, it, vi } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { createEngineDevice } from "../src/engine/gpudevice";
import { QWEN_WORKGROUP_STORAGE_BYTES } from "../src/engine/gpulimits";

const SRC = join(__dirname, "../src");

// Allowlist con RAZIONALE — ogni riga deve dire perché il sito non passa dal
// punto unico. Aggiungere qui senza motivo = reintrodurre la deriva a mano.
const ALLOWED: Record<string, string> = {
  "engine/gpudevice.ts": "il punto unico stesso",
  "probe.ts": "capability report: solo requestAdapter, MAI requestDevice (verificato sotto)",
};
// NOTA (fuori dalla scansione, che copre src/): scripts/vram-ceiling.mjs e
// scripts/webgpu-limits.mjs creano device DIRETTAMENTE in page-context — sono
// probe del device grezzo (tetto allocabile, limiti adapter) e la negoziazione
// ne falserebbe la misura. Restano fuori di proposito.

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (p.endsWith(".ts")) out.push(p);
  }
  return out;
}

describe("punto unico di creazione device (scansione di src/)", () => {
  const files = walk(SRC);

  it("la scansione vede il repo vero", () => {
    expect(files.length).toBeGreaterThan(30);
  });

  it("requestAdapter/requestDevice esistono SOLO nel helper e nell'allowlist", () => {
    const offenders = files
      .filter((p) => /request(Adapter|Device)\s*\(/.test(readFileSync(p, "utf8")))
      .map((p) => relative(SRC, p))
      .filter((rel) => !(rel in ALLOWED));
    expect(offenders).toEqual([]);
  });

  it("probe.ts resta adapter-only (il razionale della sua riga di allowlist)", () => {
    const src = readFileSync(join(SRC, "probe.ts"), "utf8");
    expect(src).not.toMatch(/requestDevice\s*\(/);
  });
});

// -------------------------------------------------------------------------
// Il helper fa DAVVERO le due cose che la scansione dà per scontate:
// negoziazione (via gpulimits) e listener uncapturederror.
// -------------------------------------------------------------------------

// Limiti misurati sul 4090 Laptop (results/engine/webgpu-limits-4090laptop-2026-08-02.json)
const LIMITS_4090 = {
  maxBufferSize: 4294967292,
  maxStorageBufferBindingSize: 2147483644,
  maxStorageBuffersPerShaderStage: 16,
  maxComputeWorkgroupStorageSize: 49152,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
};

interface FakeWorld {
  gpu: GPU;
  adapter: { features: Set<string>; limits: typeof LIMITS_4090 };
  requested: GPUDeviceDescriptor | undefined;
  listeners: Array<{ type: string; fn: (e: unknown) => void }>;
  deviceFeatures: Set<string>;
}

function fakeWorld(adapterFeatures: string[]): FakeWorld {
  const w = {} as FakeWorld;
  w.requested = undefined;
  w.listeners = [];
  // il device concede le feature CHIESTE (il filtro è del helper, non del fake)
  w.deviceFeatures = new Set();
  const device = {
    features: w.deviceFeatures,
    limits: LIMITS_4090,
    addEventListener: (type: string, fn: (e: unknown) => void) => w.listeners.push({ type, fn }),
  };
  w.adapter = {
    features: new Set(adapterFeatures),
    limits: LIMITS_4090,
  };
  (w.adapter as unknown as GPUAdapter & { requestDevice: unknown }).requestDevice =
    async (desc: GPUDeviceDescriptor) => {
      w.requested = desc;
      for (const f of (desc.requiredFeatures ?? []) as string[]) w.deviceFeatures.add(f);
      return device as unknown as GPUDevice;
    };
  w.gpu = { requestAdapter: async () => w.adapter } as unknown as GPU;
  return w;
}

const NEEDS = { ctxMax: 64, mlaAttention: false as const, kvBytesPerLayer: 0 };

describe("createEngineDevice", () => {
  it("adapter assente ⇒ errore con la label dell'harness", async () => {
    const gpu = { requestAdapter: async () => null } as unknown as GPU;
    await expect(createEngineDevice({ label: "x", needs: NEEDS }, gpu))
      .rejects.toThrow(/\[x\].*requestAdapter null/);
    await expect(createEngineDevice({ label: "x", needs: NEEDS }, undefined))
      .rejects.toThrow(/\[x\]/);
  });

  it("i limiti passati a requestDevice sono quelli NEGOZIATI da gpulimits", async () => {
    const w = fakeWorld([]);
    await createEngineDevice({ label: "t", needs: NEEDS }, w.gpu);
    const lim = w.requested?.requiredLimits as Record<string, number>;
    // spot-check sul termine Qwen: derivato, non il massimo dell'adapter
    expect(lim.maxComputeWorkgroupStorageSize).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    expect(lim.maxComputeWorkgroupStorageSize).toBeLessThan(LIMITS_4090.maxComputeWorkgroupStorageSize);
  });

  it("needs come funzione riceve L'adapter (per arenaNeeds & co.)", async () => {
    const w = fakeWorld([]);
    let seen: unknown;
    await createEngineDevice({
      label: "t",
      needs: (adapter) => { seen = adapter; return NEEDS; },
    }, w.gpu);
    expect(seen).toBe(w.adapter);
  });

  it("optionalFeatures: chieste solo se l'adapter le espone, e has() legge il device", async () => {
    const w = fakeWorld(["timestamp-query"]); // niente shader-f16
    const eng = await createEngineDevice({
      label: "t", needs: NEEDS,
      optionalFeatures: ["timestamp-query", "shader-f16"],
    }, w.gpu);
    expect([...(w.requested?.requiredFeatures ?? [])]).toEqual(["timestamp-query"]);
    expect(eng.has("timestamp-query")).toBe(true);
    expect(eng.has("shader-f16")).toBe(false);
  });

  it("uncapturederror: listener agganciato SEMPRE, e URLA con la label", async () => {
    const w = fakeWorld([]);
    await createEngineDevice({ label: "glmx", needs: NEEDS }, w.gpu);
    const uncap = w.listeners.filter((l) => l.type === "uncapturederror");
    expect(uncap).toHaveLength(1);
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      expect(() => uncap[0].fn({ error: { message: "binding size 999 exceeds limit" } }))
        .toThrow(/GPU error \[glmx\].*binding size 999/);
      expect(spy).toHaveBeenCalledWith("[glmx][gpu-error]", expect.stringContaining("binding size 999"));
    } finally {
      spy.mockRestore();
    }
  });
});
