// Test di `src/engine/gpulimits.ts` (C3a fase 3/it.6).
//
// Il gruppo "derivazione vs parco kernel" è il cuore: ri-deriva le costanti
// SCANSIONANDO il WGSL vero invece di fidarsi di un numero scritto a mano.
// È l'unica cosa che tiene insieme un limite e il codice che lo consuma — il
// motore ha già sbagliato due volte proprio perché i due vivevano in file
// diversi senza niente in mezzo (prima costanti difensive inventate, poi il
// massimo dell'adapter chiesto senza consumatore).
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  engineNeeds, limitsFor, negotiateLimits, grantedLimits, UnmetLimitError,
  MAX_WORKGROUP_SIZE, MAX_STORAGE_BINDINGS_PER_STAGE, QWEN_WORKGROUP_STORAGE_BYTES,
  mlaWorkgroupStorageBytes, q6kHeadBytes, glmKvBytesPerLayer,
} from "../src/engine/gpulimits";
import { GLM47_FLASH as G } from "../src/engine/shape";

// Limiti misurati sul 4090 Laptop (results/engine/webgpu-limits-4090laptop-2026-08-02.json)
const ADAPTER_4090 = {
  maxBufferSize: 4294967292,
  maxStorageBufferBindingSize: 2147483644,
  maxStorageBuffersPerShaderStage: 16,
  maxComputeWorkgroupStorageSize: 49152,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
};
// I default di spec WebGPU, per verificare quali requisiti li sfondano davvero
const SPEC_DEFAULTS = {
  maxBufferSize: 268435456,
  maxStorageBufferBindingSize: 134217728,
  maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
};
const fakeAdapter = (limits: Record<string, number>): GPUAdapter =>
  ({ limits }) as unknown as GPUAdapter;

const GLM_NEEDS = { ctxMax: 525, head: { vocab: G.vocab, dModel: G.dModel } };

describe("derivazione vs parco kernel (scansione del WGSL vero)", () => {
  const src = readFileSync(join(__dirname, "../src/engine/kernels/wgsl.ts"), "utf8");

  it("MAX_WORKGROUP_SIZE copre ogni @workgroup_size del repo", () => {
    const sizes = [...src.matchAll(/@compute\s+@workgroup_size\((\d+)/g)].map((m) => Number(m[1]));
    expect(sizes.length).toBeGreaterThan(20); // sanity: la scansione ha trovato i kernel
    expect(Math.max(...sizes)).toBe(MAX_WORKGROUP_SIZE);
  });

  it("MAX_STORAGE_BINDINGS_PER_STAGE copre il bind group più affollato", () => {
    // conta i binding `var<storage>` dentro ogni generatore (blocchi separati
    // da `export function`), perché gli indici ripartono da 0 a ogni kernel
    const blocks = src.split(/^export function /m).slice(1);
    const perKernel = blocks.map((b) => (b.match(/@group\(0\)\s*@binding\(\d+\)\s*var<storage/g) ?? []).length);
    expect(blocks.length).toBeGreaterThan(20);
    expect(Math.max(...perKernel)).toBe(MAX_STORAGE_BINDINGS_PER_STAGE);
  });

  it("nessun kernel dichiara più binding di quanti il limite ne conceda", () => {
    const blocks = src.split(/^export function /m).slice(1);
    for (const b of blocks) {
      const n = (b.match(/@group\(0\)\s*@binding\(\d+\)\s*var<storage/g) ?? []).length;
      expect(n).toBeLessThanOrEqual(MAX_STORAGE_BINDINGS_PER_STAGE);
    }
  });
});

describe("requisiti derivati", () => {
  it("la testa Q6_K sfonda il default di spec sul binding size", () => {
    const head = q6kHeadBytes(G.vocab, G.dModel);
    expect(head).toBe(262_676_480); // 250,5 MiB
    expect(head).toBeGreaterThan(SPEC_DEFAULTS.maxStorageBufferBindingSize);
    // ...e sta appena SOTTO il default sul buffer size (margine 2,1%)
    expect(head).toBeLessThan(SPEC_DEFAULTS.maxBufferSize);
  });

  it("il workgroup storage e' il max fra path Qwen fuso e mlaAttnDecode(ctxMax)", () => {
    // a ctx corto vince Qwen (costante), a ctx lungo vince MLA
    const short = engineNeeds({ ctxMax: 525 }).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
    expect(short.value).toBe(QWEN_WORKGROUP_STORAGE_BYTES);
    const long = engineNeeds({ ctxMax: 8192 }).find((n) => n.limit === "maxComputeWorkgroupStorageSize")!;
    expect(long.value).toBe(mlaWorkgroupStorageBytes(8192));
    expect(long.value).toBe(33_024);
    // il vecchio cap a mano di 32768 tagliava il contesto a 8128 senza dirlo
    expect(mlaWorkgroupStorageBytes(8128)).toBeLessThanOrEqual(32768);
    expect(mlaWorkgroupStorageBytes(8129)).toBeGreaterThan(32768);
  });

  it("ogni requisito porta il suo consumatore", () => {
    for (const n of engineNeeds(GLM_NEEDS)) {
      expect(n.consumer.length).toBeGreaterThan(10);
      expect(n.value).toBeGreaterThan(0);
    }
  });
});

describe("limitsFor", () => {
  it("chiede il REQUISITO, non il massimo dell'adapter", () => {
    const got = negotiateLimits(fakeAdapter(ADAPTER_4090), GLM_NEEDS);
    // l'adapter concede 1024, ma il parco kernel ne usa 256: si chiede 256
    expect(got.maxComputeInvocationsPerWorkgroup).toBe(256);
    expect(got.maxComputeWorkgroupSizeX).toBe(256);
    // l'adapter concede 16 storage buffer, ne servono 7
    expect(got.maxStorageBuffersPerShaderStage).toBe(7);
    // qui invece il requisito e' reale e sopra il default
    expect(got.maxStorageBufferBindingSize).toBe(262_676_480);
  });

  it("non chiede MAI piu' del disponibile", () => {
    const lim = ADAPTER_4090 as Record<string, number>;
    for (const [k, v] of Object.entries(negotiateLimits(fakeAdapter(lim), GLM_NEEDS))) {
      expect(v).toBeLessThanOrEqual(lim[k]);
    }
  });

  it("un requisito HARD non servibile fallisce subito, col consumatore nel messaggio", () => {
    const povero = { ...ADAPTER_4090, maxStorageBufferBindingSize: SPEC_DEFAULTS.maxStorageBufferBindingSize };
    let err: unknown;
    try { negotiateLimits(fakeAdapter(povero), GLM_NEEDS); } catch (e) { err = e; }
    expect(err).toBeInstanceOf(UnmetLimitError);
    expect((err as UnmetLimitError).unmet[0].limit).toBe("maxStorageBufferBindingSize");
    expect((err as Error).message).toMatch(/output\.weight Q6_K/);
  });

  it("un requisito SOFT (packing) viene troncato, non fa fallire", () => {
    const needs = engineNeeds({ ...GLM_NEEDS, slabClassBytes: 99e9 });
    const got = limitsFor(fakeAdapter(ADAPTER_4090), needs);
    expect(got.maxBufferSize).toBe(ADAPTER_4090.maxBufferSize);
  });

  it("senza slabClassBytes il buffer resta al requisito hard, non al massimo", () => {
    const got = negotiateLimits(fakeAdapter(ADAPTER_4090), GLM_NEEDS);
    expect(got.maxBufferSize).toBe(262_676_480);
    expect(got.maxBufferSize).toBeLessThan(ADAPTER_4090.maxBufferSize);
  });

  it("un limite non esposto dall'adapter non viene chiesto a 0", () => {
    let err: unknown;
    try { limitsFor(fakeAdapter({ maxBufferSize: 1024 }), engineNeeds(GLM_NEEDS)); } catch (e) { err = e; }
    expect(err).toBeInstanceOf(UnmetLimitError); // tutti i hard non serviti
  });

  it("la KV cache diventa il vincolo solo a contesti irraggiungibili", () => {
    const head = q6kHeadBytes(G.vocab, G.dModel);
    // il workgroup storage taglia molto prima: 49152 B => ctxMax <= 12224
    expect(glmKvBytesPerLayer(12224)).toBeLessThan(head);
    expect(glmKvBytesPerLayer(114009)).toBeGreaterThan(head);
  });
});

describe("grantedLimits", () => {
  it("riporta i valori del DEVICE, non dell'adapter", () => {
    const dev = { limits: { ...SPEC_DEFAULTS, maxComputeWorkgroupsPerDimension: 65535 } } as unknown as GPUDevice;
    const g = grantedLimits(dev);
    expect(g.maxComputeInvocationsPerWorkgroup).toBe(256);
    expect(g.maxStorageBufferBindingSize).toBe(SPEC_DEFAULTS.maxStorageBufferBindingSize);
  });
});
