# Fase 1a — Scheletro harness: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SPA Vite+TS con bench-worker, adapter WebLLM, probe WebGPU e un modello (Qwen2.5-0.5B) end-to-end sulla 4090, con export JSON conforme allo schema versionato.

**Architecture:** Tutto il compute vive in un Web Worker nostro (`bench.worker.ts`), che contiene l'adapter WebLLM (via `CreateMLCEngine`, NON il `WebWorkerMLCEngine` di web-llm), il probe e il timing; il main thread è solo UI + protocollo messaggi tipizzato. Le metriche sono funzioni pure testate in CI; i run reali restano manuali.

**Tech Stack:** Vite + TypeScript, Vitest, `@mlc-ai/web-llm`, `@webgpu/types`.

## Global Constraints

- Spec di riferimento: `docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md` (Fase 1a soltanto: niente Transformers.js/wllama, niente qualità, niente S22/M4 in questo piano).
- `SCHEMA_VERSION = 1`; ogni file risultato lo dichiara.
- Decoding: `temperature: 0` (greedy), `max_tokens: 256`; decode tok/s = steady-state, primo token escluso.
- Prompt bench fisso e versionato (`bench-512-v1`), ~512 token; i token effettivi si riportano da `usage.prompt_tokens`.
- Device label manuale: `4090-linux`. Zero fingerprinting oltre al probe dichiarato.
- Dev server con COOP/COEP (crossOriginIsolated) — serve a `performance.measureUserAgentSpecificMemory()` e, in 1b, ai thread wllama. Contingenza: se il download shard da HF fallisce sotto `require-corp`, passare a `Cross-Origin-Embedder-Policy: credentialless` e annotarlo nel README.
- Commit frequenti, messaggi convenzionali, **nessuna attribuzione AI**.
- Nessun backend: tutto statico.

---

## File Structure

```
browser-llm-lab/
  index.html                  — shell UI minima
  vite.config.ts              — headers COOP/COEP dev+preview
  package.json / tsconfig.json / .gitignore
  src/
    main.ts                   — wiring UI ↔ worker (nessuna logica di calcolo)
    render.ts                 — renderResultsTable(): pure string builder (testabile)
    schema.ts                 — SCHEMA_VERSION, tipi risultato, newRunFile/addCell
    metrics.ts                — computeGenMetrics(): pure timing math
    protocol.ts               — messaggi MainToWorker/WorkerToMain (discriminated unions)
    probe.ts                  — probeWebGPU(gpu): limiti adapter + device memory
    promptset.ts              — PROMPT_512 deterministico, id 'bench-512-v1'
    adapters/
      types.ts                — InferenceAdapter, AdapterCapabilities, LoadReport, GenTimeline
      webllm.ts               — WebLLMAdapter (engine factory iniettabile per i test)
    benchServer.ts            — BenchServer: dispatch messaggi, orchestrazione (testabile)
    bench.worker.ts           — entry worker: 4 righe, istanzia BenchServer
  tests/
    schema.test.ts / metrics.test.ts / protocol.test.ts / probe.test.ts
    webllm-adapter.test.ts / benchServer.test.ts / render.test.ts
  results/                    — JSON esportati dai run (committati: sono i dati)
```

---

### Task 1: Scaffold Vite + TS + Vitest

**Files:**
- Create: `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.ts` (stub), `.gitignore`, `tests/smoke.test.ts`

**Interfaces:**
- Produces: progetto compilante; `npm test` e `npm run dev` funzionanti; header COOP/COEP attivi.

- [ ] **Step 1: Scaffold e dipendenze**

```bash
cd ~/Projects/browser-llm-lab
npm create vite@latest . -- --template vanilla-ts   # accetta di procedere in dir non vuota SENZA sovrascrivere docs/
npm i @mlc-ai/web-llm
npm i -D vitest @webgpu/types
```

Se lo scaffolder si rifiuta per dir non vuota: crearlo in `/tmp/scaffold` e copiare dentro tutto tranne `.git/` e `docs/`.

- [ ] **Step 2: Config**

`vite.config.ts`:
```typescript
import { defineConfig } from "vite";

const coopCoep = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  server: { headers: coopCoep },
  preview: { headers: coopCoep },
});
```

In `tsconfig.json` assicurare: `"lib": ["ES2022", "DOM", "DOM.Iterable", "WebWorker"]` e `"types": ["@webgpu/types", "vite/client"]`.

In `package.json` scripts: `"test": "vitest run", "test:watch": "vitest"`.

`.gitignore`: `node_modules/`, `dist/`.

Svuotare `src/main.ts` a stub: `console.log("browser-llm-lab");` e ripulire `index.html` dal boilerplate Vite (titolo `browser-llm-lab`, un `<div id="app"></div>`, lo script module).

- [ ] **Step 3: Smoke test**

`tests/smoke.test.ts`:
```typescript
import { describe, it, expect } from "vitest";

describe("toolchain", () => {
  it("runs", () => {
    expect(1 + 1).toBe(2);
  });
});
```

Run: `npm test` → Expected: 1 passed.

- [ ] **Step 4: Verifica dev server + crossOriginIsolated**

Run: `npm run dev` → aprire http://localhost:5173, in console: `crossOriginIsolated` → Expected: `true`. Chiudere.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: scaffold vite+ts+vitest, COOP/COEP headers"
```

---

### Task 2: Schema risultati (`schema.ts`)

**Files:**
- Create: `src/schema.ts`, `tests/schema.test.ts`

**Interfaces:**
- Produces: `SCHEMA_VERSION: 1`; tipi `DeviceProbe`, `LoadReport`, `GenMetrics`, `BenchCell`, `RunFile`; `newRunFile(deviceLabel: string, probe: DeviceProbe, ts: string): RunFile`; `addCell(run: RunFile, cell: BenchCell): RunFile` (immutabile).

- [ ] **Step 1: Test fallente**

`tests/schema.test.ts`:
```typescript
import { describe, it, expect } from "vitest";
import { SCHEMA_VERSION, newRunFile, addCell, type DeviceProbe, type BenchCell } from "../src/schema";

const probe: DeviceProbe = {
  webgpu: true,
  adapterInfo: { vendor: "nvidia", architecture: "", device: "", description: "" },
  limits: { maxStorageBufferBindingSize: 2147483644 },
  userAgent: "test-ua",
  deviceMemoryGB: 8,
};

const cell: BenchCell = {
  stack: "webllm",
  modelId: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  quant: "q4f16_1",
  promptId: "bench-512-v1",
  load: { loadMs: 1234, cacheState: "cold" },
  gen: { ttftMs: 100, decodeToksPerSec: 42.5, totalMs: 6000, promptTokens: 512, completionTokens: 256 },
};

describe("schema", () => {
  it("builds a versioned run file", () => {
    const run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    expect(run.schemaVersion).toBe(SCHEMA_VERSION);
    expect(run.deviceLabel).toBe("4090-linux");
    expect(run.cells).toEqual([]);
  });

  it("addCell is immutable", () => {
    const run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    const run2 = addCell(run, cell);
    expect(run.cells.length).toBe(0);
    expect(run2.cells.length).toBe(1);
    expect(run2.cells[0].gen.decodeToksPerSec).toBe(42.5);
  });
});
```

- [ ] **Step 2: Run → FAIL** (`npm test` — modulo inesistente)

- [ ] **Step 3: Implementazione**

`src/schema.ts`:
```typescript
export const SCHEMA_VERSION = 1 as const;

export interface DeviceProbe {
  webgpu: boolean;
  adapterInfo: { vendor: string; architecture: string; device: string; description: string } | null;
  limits: Record<string, number> | null;
  userAgent: string;
  deviceMemoryGB: number | null;
}

export interface LoadReport {
  loadMs: number;
  cacheState: "cold" | "warm" | "unknown";
}

export interface GenMetrics {
  ttftMs: number;
  decodeToksPerSec: number | null; // null se <2 chunk o token count assente
  totalMs: number;
  promptTokens: number | null;
  completionTokens: number | null;
}

export interface BenchCell {
  stack: "webllm"; // union estesa in 1b
  modelId: string;
  quant: string;
  promptId: string;
  load: LoadReport;
  gen: GenMetrics;
}

export interface RunFile {
  schemaVersion: typeof SCHEMA_VERSION;
  deviceLabel: string;
  ts: string; // ISO, momento di creazione run
  probe: DeviceProbe;
  cells: BenchCell[];
}

export function newRunFile(deviceLabel: string, probe: DeviceProbe, ts: string): RunFile {
  return { schemaVersion: SCHEMA_VERSION, deviceLabel, ts, probe, cells: [] };
}

export function addCell(run: RunFile, cell: BenchCell): RunFile {
  return { ...run, cells: [...run.cells, cell] };
}
```

- [ ] **Step 4: Run → PASS** (`npm test`)

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: versioned result schema"`

---

### Task 3: Metriche (`metrics.ts`)

**Files:**
- Create: `src/metrics.ts`, `tests/metrics.test.ts`

**Interfaces:**
- Consumes: `GenMetrics` da `schema.ts`.
- Produces: `interface GenTimeline { tRequestStart: number; chunkTimestamps: number[]; promptTokens: number | null; completionTokens: number | null }`; `computeGenMetrics(t: GenTimeline): GenMetrics`.

Definizioni (dalla spec, qui operative):
- `ttftMs` = primo chunk − `tRequestStart`.
- `decodeToksPerSec` = `(completionTokens − 1) / (ultimoChunk − primoChunk) * 1000` (steady-state: primo token escluso). Fallback token count: `chunkTimestamps.length` se `completionTokens` è null. `null` se chunk < 2, intervallo = 0, o token generati < 2 (steady-state non misurabile).
- `totalMs` = ultimo chunk − `tRequestStart`.

- [ ] **Step 1: Test fallente**

`tests/metrics.test.ts`:
```typescript
import { describe, it, expect } from "vitest";
import { computeGenMetrics, type GenTimeline } from "../src/metrics";

describe("computeGenMetrics", () => {
  it("computes ttft, steady-state decode rate, total", () => {
    // 5 chunk: primo a t=1100 (ttft 100ms), poi ogni 50ms → 4 token in 200ms = 20 tok/s
    const t: GenTimeline = {
      tRequestStart: 1000,
      chunkTimestamps: [1100, 1150, 1200, 1250, 1300],
      promptTokens: 512,
      completionTokens: 5,
    };
    const m = computeGenMetrics(t);
    expect(m.ttftMs).toBe(100);
    expect(m.decodeToksPerSec).toBeCloseTo(20, 5);
    expect(m.totalMs).toBe(300);
    expect(m.promptTokens).toBe(512);
    expect(m.completionTokens).toBe(5);
  });

  it("falls back to chunk count when completionTokens is null", () => {
    const m = computeGenMetrics({
      tRequestStart: 0,
      chunkTimestamps: [100, 200, 300],
      promptTokens: null,
      completionTokens: null,
    });
    expect(m.completionTokens).toBe(3);
    expect(m.decodeToksPerSec).toBeCloseTo(10, 5); // (3-1)/200ms
  });

  it("returns null rate when fewer than 2 chunks", () => {
    const m = computeGenMetrics({ tRequestStart: 0, chunkTimestamps: [50], promptTokens: null, completionTokens: 1 });
    expect(m.ttftMs).toBe(50);
    expect(m.decodeToksPerSec).toBeNull();
  });

  it("throws on empty timeline", () => {
    expect(() => computeGenMetrics({ tRequestStart: 0, chunkTimestamps: [], promptTokens: null, completionTokens: null })).toThrow();
  });
});
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implementazione**

`src/metrics.ts`:
```typescript
import type { GenMetrics } from "./schema";

export interface GenTimeline {
  tRequestStart: number;
  chunkTimestamps: number[]; // performance.now() di ogni chunk ricevuto, in-worker
  promptTokens: number | null;
  completionTokens: number | null;
}

export function computeGenMetrics(t: GenTimeline): GenMetrics {
  if (t.chunkTimestamps.length === 0) throw new Error("empty timeline: no chunks received");
  const first = t.chunkTimestamps[0];
  const last = t.chunkTimestamps[t.chunkTimestamps.length - 1];
  const completionTokens = t.completionTokens ?? t.chunkTimestamps.length;
  const span = last - first;
  const decodeToksPerSec =
    t.chunkTimestamps.length >= 2 && span > 0 && completionTokens >= 2
      ? ((completionTokens - 1) / span) * 1000
      : null;
  return {
    ttftMs: first - t.tRequestStart,
    decodeToksPerSec,
    totalMs: last - t.tRequestStart,
    promptTokens: t.promptTokens,
    completionTokens,
  };
}
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit** — `git commit -am "feat: pure generation metrics (ttft, steady-state tok/s)"`

---

### Task 4: Contratto adapter + protocollo messaggi

**Files:**
- Create: `src/adapters/types.ts`, `src/protocol.ts`, `tests/protocol.test.ts`

**Interfaces:**
- Consumes: `GenTimeline` (metrics), `DeviceProbe`, `LoadReport`, `BenchCell` (schema).
- Produces:
  - `interface AdapterCapabilities { logprobs: boolean; streaming: boolean; seed: boolean }`
  - `interface GenerateRequest { prompt: string; maxTokens: number }`
  - `interface InferenceAdapter { readonly id: "webllm"; capabilities(): AdapterCapabilities; load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport>; generate(req: GenerateRequest): Promise<GenTimeline>; dispose(): Promise<void> }`
  - `type MainToWorker = { type: "probe" } | { type: "bench"; modelId: string; quant: string }`
  - `type WorkerToMain = { type: "probe:result"; probe: DeviceProbe } | { type: "progress"; text: string; progress: number } | { type: "bench:result"; cell: BenchCell } | { type: "error"; message: string }`
  - `function isMainToWorker(x: unknown): x is MainToWorker`

- [ ] **Step 1: Test fallente**

`tests/protocol.test.ts`:
```typescript
import { describe, it, expect } from "vitest";
import { isMainToWorker } from "../src/protocol";

describe("protocol guards", () => {
  it("accepts valid messages", () => {
    expect(isMainToWorker({ type: "probe" })).toBe(true);
    expect(isMainToWorker({ type: "bench", modelId: "m", quant: "q4f16_1" })).toBe(true);
  });
  it("rejects invalid messages", () => {
    expect(isMainToWorker(null)).toBe(false);
    expect(isMainToWorker({ type: "bench" })).toBe(false); // manca modelId
    expect(isMainToWorker({ type: "nope" })).toBe(false);
  });
});
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implementazione**

`src/adapters/types.ts`:
```typescript
import type { LoadReport } from "../schema";
import type { GenTimeline } from "../metrics";

export interface AdapterCapabilities {
  logprobs: boolean; // 1a: false; usato dal modulo qualità in 1b
  streaming: boolean;
  seed: boolean;
}

export interface GenerateRequest {
  prompt: string;
  maxTokens: number;
}

export interface InferenceAdapter {
  readonly id: "webllm"; // union estesa in 1b
  capabilities(): AdapterCapabilities;
  load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport>;
  generate(req: GenerateRequest): Promise<GenTimeline>;
  dispose(): Promise<void>;
}
```

`src/protocol.ts`:
```typescript
import type { DeviceProbe, BenchCell } from "./schema";

export type MainToWorker =
  | { type: "probe" }
  | { type: "bench"; modelId: string; quant: string };

export type WorkerToMain =
  | { type: "probe:result"; probe: DeviceProbe }
  | { type: "progress"; text: string; progress: number }
  | { type: "bench:result"; cell: BenchCell }
  | { type: "error"; message: string };

export function isMainToWorker(x: unknown): x is MainToWorker {
  if (typeof x !== "object" || x === null || !("type" in x)) return false;
  const m = x as Record<string, unknown>;
  if (m.type === "probe") return true;
  if (m.type === "bench") return typeof m.modelId === "string" && typeof m.quant === "string";
  return false;
}
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit** — `git commit -am "feat: adapter contract + typed worker protocol"`

---

### Task 5: Probe WebGPU (`probe.ts`)

**Files:**
- Create: `src/probe.ts`, `tests/probe.test.ts`

**Interfaces:**
- Consumes: `DeviceProbe` (schema).
- Produces: `probeWebGPU(gpu: GPU | undefined, nav: { userAgent: string; deviceMemory?: number }): Promise<DeviceProbe>` — iniettabile per i test; nel worker si chiama con `navigator.gpu, navigator`.

Nota: `adapter.limits` non è enumerabile con `Object.entries` su tutti i browser — si copia una lista esplicita di chiavi. `GPUAdapterInfo` è esposto come `adapter.info` (property, non più `requestAdapterInfo()` nelle versioni recenti di Chrome); si supportano entrambe le forme.

- [ ] **Step 1: Test fallente**

`tests/probe.test.ts`:
```typescript
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
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implementazione**

`src/probe.ts`:
```typescript
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
  const info = a.info ?? (a.requestAdapterInfo ? await a.requestAdapterInfo().catch(() => null) : null);

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
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit** — `git commit -am "feat: webgpu device probe (adapter info + limits)"`

---

### Task 6: Prompt set (`promptset.ts`) + adapter WebLLM (`adapters/webllm.ts`)

**Files:**
- Create: `src/promptset.ts`, `src/adapters/webllm.ts`, `tests/webllm-adapter.test.ts`

**Interfaces:**
- Consumes: `InferenceAdapter`, `GenerateRequest` (types), `GenTimeline` (metrics), `LoadReport` (schema).
- Produces: `PROMPT_512: { id: "bench-512-v1"; text: string }`; `class WebLLMAdapter implements InferenceAdapter` con `constructor(deps?: { engineFactory?: EngineFactory; hasCache?: (modelId: string) => Promise<boolean>; now?: () => number })` — default ai moduli reali di `@mlc-ai/web-llm`, iniettabili nei test.

Note verificate su docs (context7, 2026-07-25): `CreateMLCEngine(modelId, { initProgressCallback })`; streaming = AsyncGenerator con `stream: true, stream_options: { include_usage: true }` (ultimo chunk porta `usage`); util di cache per "model is cached" presente nel package — **confermare il nome esatto dell'export (`hasModelInCache`) dai typings del package installato**; ID modello da confermare filtrando `prebuiltAppConfig.model_list` (Step 4).

- [ ] **Step 1: Prompt set**

`src/promptset.ts`:
```typescript
// Prompt deterministico ~512 token: paragrafo fisso ripetuto.
// NON MODIFICARE il testo senza incrementare l'id (i risultati citano promptId).
const PARA =
  "The city library opened its doors at nine in the morning, and the archivist began sorting the day's returns. " +
  "Each volume carried a small paper slip noting the date, the borrower's initials, and the shelf it belonged to. " +
  "Outside, the market square filled slowly with vendors arranging crates of apples, bread, and winter vegetables. " +
  "A tram passed every twelve minutes, and its bell could be heard clearly through the reading room windows. ";

export const PROMPT_512 = {
  id: "bench-512-v1" as const,
  text:
    "Read the following passage carefully, then continue the story in the same style.\n\n" +
    PARA.repeat(5) +
    "\n\nContinue the story:",
};
```

- [ ] **Step 2: Test fallente per l'adapter (fake engine)**

`tests/webllm-adapter.test.ts`:
```typescript
import { describe, it, expect } from "vitest";
import { WebLLMAdapter } from "../src/adapters/webllm";

function fakeEngine() {
  return {
    chat: {
      completions: {
        create: async function* mock() {
          yield { choices: [{ delta: { content: "a" } }], usage: null };
          yield { choices: [{ delta: { content: "b" } }], usage: null };
          yield { choices: [], usage: { prompt_tokens: 512, completion_tokens: 2 } };
        },
      },
    },
    unload: async () => {},
  };
}

describe("WebLLMAdapter", () => {
  it("load reports cold/warm from cache probe and measures loadMs", async () => {
    let t = 0;
    const a = new WebLLMAdapter({
      engineFactory: async () => fakeEngine() as never,
      hasCache: async () => false,
      now: () => (t += 500), // load: t0=500, t1=1000 → 500ms
    });
    const r = await a.load("test-model", () => {});
    expect(r.cacheState).toBe("cold");
    expect(r.loadMs).toBe(500);
  });

  it("generate builds a timeline with usage tokens", async () => {
    let t = 0;
    const a = new WebLLMAdapter({
      engineFactory: async () => fakeEngine() as never,
      hasCache: async () => true,
      now: () => (t += 100),
    });
    await a.load("test-model", () => {});
    const tl = await a.generate({ prompt: "hi", maxTokens: 8 });
    expect(tl.chunkTimestamps.length).toBe(2); // 2 chunk con contenuto; il chunk usage-only non conta
    expect(tl.promptTokens).toBe(512);
    expect(tl.completionTokens).toBe(2);
  });

  it("generate before load throws", async () => {
    const a = new WebLLMAdapter({ engineFactory: async () => fakeEngine() as never });
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow("not loaded");
  });
});
```

- [ ] **Step 3: Run → FAIL**

- [ ] **Step 4: Implementazione**

Prima di scrivere l'ID modello nel codice di Task 8, verificare l'ID esatto (una tantum, in console browser o node):
```typescript
import { prebuiltAppConfig } from "@mlc-ai/web-llm";
console.log(prebuiltAppConfig.model_list.map(m => m.model_id).filter(id => id.includes("Qwen2.5-0.5B")));
// atteso qualcosa come: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
```

`src/adapters/webllm.ts`:
```typescript
import {
  CreateMLCEngine,
  hasModelInCache, // verificare export nei typings del package installato
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";
import type { InferenceAdapter, AdapterCapabilities, GenerateRequest } from "./types";
import type { LoadReport } from "../schema";
import type { GenTimeline } from "../metrics";

type EngineFactory = (
  modelId: string,
  onProgress: (text: string, progress: number) => void,
) => Promise<MLCEngineInterface>;

const defaultFactory: EngineFactory = (modelId, onProgress) =>
  CreateMLCEngine(modelId, {
    initProgressCallback: (p) => onProgress(p.text, p.progress),
  });

export class WebLLMAdapter implements InferenceAdapter {
  readonly id = "webllm" as const;
  private engine: MLCEngineInterface | null = null;
  private engineFactory: EngineFactory;
  private hasCache: (modelId: string) => Promise<boolean>;
  private now: () => number;

  constructor(deps?: {
    engineFactory?: EngineFactory;
    hasCache?: (modelId: string) => Promise<boolean>;
    now?: () => number;
  }) {
    this.engineFactory = deps?.engineFactory ?? defaultFactory;
    this.hasCache = deps?.hasCache ?? ((id) => hasModelInCache(id).catch(() => false));
    this.now = deps?.now ?? (() => performance.now());
  }

  capabilities(): AdapterCapabilities {
    return { logprobs: false, streaming: true, seed: false }; // logprobs: rivalutare in 1b
  }

  async load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport> {
    const cached = await this.hasCache(modelId);
    const t0 = this.now();
    this.engine = await this.engineFactory(modelId, onProgress);
    return { loadMs: this.now() - t0, cacheState: cached ? "warm" : "cold" };
  }

  async generate(req: GenerateRequest): Promise<GenTimeline> {
    if (!this.engine) throw new Error("not loaded");
    const tRequestStart = this.now();
    const chunkTimestamps: number[] = [];
    let promptTokens: number | null = null;
    let completionTokens: number | null = null;

    const stream = await this.engine.chat.completions.create({
      messages: [{ role: "user", content: req.prompt }],
      temperature: 0,
      max_tokens: req.maxTokens,
      stream: true,
      stream_options: { include_usage: true },
    });

    for await (const chunk of stream) {
      if (chunk.choices[0]?.delta?.content) chunkTimestamps.push(this.now());
      if (chunk.usage) {
        promptTokens = chunk.usage.prompt_tokens ?? null;
        completionTokens = chunk.usage.completion_tokens ?? null;
      }
    }
    return { tRequestStart, chunkTimestamps, promptTokens, completionTokens };
  }

  async dispose(): Promise<void> {
    await this.engine?.unload();
    this.engine = null;
  }
}
```

Se i typings del package divergono (nome `hasModelInCache`, forma di `stream_options`, firma `initProgressCallback`): adeguare l'adapter ai typings reali, NON forzare con `as any`; annotare la differenza nel commit message.

- [ ] **Step 5: Run → PASS** (`npm test`)

- [ ] **Step 6: Commit** — `git commit -am "feat: webllm adapter with injectable engine, fixed 512-tok prompt"`

---

### Task 7: BenchServer + worker entry

**Files:**
- Create: `src/benchServer.ts`, `src/bench.worker.ts`, `tests/benchServer.test.ts`

**Interfaces:**
- Consumes: `InferenceAdapter` (types), `probeWebGPU`, `computeGenMetrics`, `PROMPT_512`, `protocol.ts`, `BenchCell` (schema).
- Produces: `class BenchServer { constructor(deps: { adapterFactory: () => InferenceAdapter; probe: () => Promise<DeviceProbe>; post: (m: WorkerToMain) => void }); handle(msg: unknown): Promise<void> }`. Sequenza bench: load → progress → generate → `bench:result` con cell completa; errori → `{ type: "error" }`, mai throw non gestito.

- [ ] **Step 1: Test fallente**

`tests/benchServer.test.ts`:
```typescript
import { describe, it, expect } from "vitest";
import { BenchServer } from "../src/benchServer";
import type { WorkerToMain } from "../src/protocol";
import type { InferenceAdapter } from "../src/adapters/types";

const fakeProbe = async () => ({
  webgpu: true, adapterInfo: null, limits: null, userAgent: "ua", deviceMemoryGB: null,
});

function fakeAdapter(): InferenceAdapter {
  return {
    id: "webllm",
    capabilities: () => ({ logprobs: false, streaming: true, seed: false }),
    load: async (_m, onProgress) => { onProgress("loading", 0.5); return { loadMs: 100, cacheState: "cold" as const }; },
    generate: async () => ({ tRequestStart: 0, chunkTimestamps: [10, 20, 30], promptTokens: 512, completionTokens: 3 }),
    dispose: async () => {},
  };
}

describe("BenchServer", () => {
  it("probe message → probe:result", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapterFactory: fakeAdapter, probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "probe" });
    expect(out[0].type).toBe("probe:result");
  });

  it("bench message → progress then bench:result with metrics", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapterFactory: fakeAdapter, probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "bench", modelId: "test-model", quant: "q4f16_1" });
    const types = out.map((m) => m.type);
    expect(types).toContain("progress");
    const result = out.find((m) => m.type === "bench:result");
    expect(result).toBeDefined();
    if (result?.type === "bench:result") {
      expect(result.cell.modelId).toBe("test-model");
      expect(result.cell.gen.ttftMs).toBe(10);
      expect(result.cell.load.cacheState).toBe("cold");
      expect(result.cell.promptId).toBe("bench-512-v1");
    }
  });

  it("adapter failure → error message, no throw", async () => {
    const broken = { ...fakeAdapter(), load: async () => { throw new Error("boom"); } };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapterFactory: () => broken, probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "bench", modelId: "m", quant: "q" });
    expect(out.some((m) => m.type === "error" && m.message.includes("boom"))).toBe(true);
  });

  it("invalid message → error", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapterFactory: fakeAdapter, probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ garbage: true });
    expect(out[0].type).toBe("error");
  });
});
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implementazione**

`src/benchServer.ts`:
```typescript
import { isMainToWorker, type WorkerToMain } from "./protocol";
import type { InferenceAdapter } from "./adapters/types";
import type { DeviceProbe, BenchCell } from "./schema";
import { computeGenMetrics } from "./metrics";
import { PROMPT_512 } from "./promptset";

export class BenchServer {
  private deps: {
    adapterFactory: () => InferenceAdapter;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
  };

  constructor(deps: {
    adapterFactory: () => InferenceAdapter;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
  }) {
    this.deps = deps;
  }

  async handle(msg: unknown): Promise<void> {
    if (!isMainToWorker(msg)) {
      this.deps.post({ type: "error", message: "invalid message" });
      return;
    }
    try {
      if (msg.type === "probe") {
        this.deps.post({ type: "probe:result", probe: await this.deps.probe() });
      } else if (msg.type === "bench") {
        const adapter = this.deps.adapterFactory();
        try {
          const load = await adapter.load(msg.modelId, (text, progress) =>
            this.deps.post({ type: "progress", text, progress }),
          );
          this.deps.post({ type: "progress", text: "generating…", progress: 1 });
          const timeline = await adapter.generate({ prompt: PROMPT_512.text, maxTokens: 256 });
          const cell: BenchCell = {
            stack: adapter.id,
            modelId: msg.modelId,
            quant: msg.quant,
            promptId: PROMPT_512.id,
            load,
            gen: computeGenMetrics(timeline),
          };
          this.deps.post({ type: "bench:result", cell });
        } finally {
          await adapter.dispose().catch(() => {});
        }
      }
    } catch (e) {
      this.deps.post({ type: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }
}
```

`src/bench.worker.ts`:
```typescript
import { BenchServer } from "./benchServer";
import { WebLLMAdapter } from "./adapters/webllm";
import { probeWebGPU } from "./probe";

const server = new BenchServer({
  adapterFactory: () => new WebLLMAdapter(),
  probe: () => probeWebGPU(navigator.gpu, navigator as { userAgent: string; deviceMemory?: number }),
  post: (m) => self.postMessage(m),
});

self.onmessage = (e: MessageEvent) => void server.handle(e.data);
```

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit** — `git commit -am "feat: bench server orchestration + worker entry"`

---

### Task 8: UI minima + export JSON

**Files:**
- Create: `src/render.ts`, `tests/render.test.ts`
- Modify: `src/main.ts`, `index.html`

**Interfaces:**
- Consumes: `protocol.ts`, `schema.ts` (`newRunFile`, `addCell`, `RunFile`).
- Produces: `renderResultsTable(run: RunFile): string` (HTML string, pure); UI con: probe box, select modello, bottone Run, progress line, tabella risultati, bottone "Export JSON" che scarica `<deviceLabel>-<ts>.json`.

- [ ] **Step 1: Test fallente per il renderer**

`tests/render.test.ts`:
```typescript
import { describe, it, expect } from "vitest";
import { renderResultsTable } from "../src/render";
import { newRunFile, addCell } from "../src/schema";

const probe = { webgpu: true, adapterInfo: null, limits: null, userAgent: "ua", deviceMemoryGB: null };

describe("renderResultsTable", () => {
  it("renders one row per cell with key metrics", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, {
      stack: "webllm", modelId: "M", quant: "q4f16_1", promptId: "bench-512-v1",
      load: { loadMs: 1500, cacheState: "cold" },
      gen: { ttftMs: 123.4, decodeToksPerSec: 41.7, totalMs: 6000, promptTokens: 512, completionTokens: 256 },
    });
    const html = renderResultsTable(run);
    expect(html).toContain("M");
    expect(html).toContain("41.7");
    expect(html).toContain("cold");
  });

  it("renders empty state", () => {
    const html = renderResultsTable(newRunFile("x", probe, "2026-07-25T12:00:00Z"));
    expect(html).toContain("Nessun risultato");
  });
});
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implementazione renderer**

`src/render.ts`:
```typescript
import type { RunFile } from "./schema";

const fmt = (n: number | null, digits = 1) => (n === null ? "—" : n.toFixed(digits));

export function renderResultsTable(run: RunFile): string {
  if (run.cells.length === 0) return "<p>Nessun risultato ancora.</p>";
  const rows = run.cells
    .map(
      (c) => `<tr>
        <td>${c.stack}</td><td>${c.modelId}</td><td>${c.quant}</td>
        <td>${fmt(c.load.loadMs, 0)}</td><td>${c.load.cacheState}</td>
        <td>${fmt(c.gen.ttftMs, 0)}</td><td>${fmt(c.gen.decodeToksPerSec)}</td>
        <td>${c.gen.promptTokens ?? "—"}/${c.gen.completionTokens ?? "—"}</td>
      </tr>`,
    )
    .join("");
  return `<table>
    <thead><tr><th>stack</th><th>model</th><th>quant</th><th>load ms</th><th>cache</th><th>TTFT ms</th><th>tok/s</th><th>tok in/out</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}
```

- [ ] **Step 4: Run → PASS, poi wiring UI**

`index.html` (body):
```html
<body>
  <h1>browser-llm-lab</h1>
  <section id="probe-box"><em>probing…</em></section>
  <section>
    <label>Model:
      <select id="model">
        <option value="Qwen2.5-0.5B-Instruct-q4f16_1-MLC" data-quant="q4f16_1">Qwen2.5-0.5B q4f16_1</option>
      </select>
    </label>
    <button id="run">Run bench</button>
    <button id="export" disabled>Export JSON</button>
  </section>
  <p id="status"></p>
  <section id="results"></section>
  <script type="module" src="/src/main.ts"></script>
</body>
```
(l'ID modello nell'`<option>` va allineato all'esito della verifica `prebuiltAppConfig` del Task 6 Step 4)

`src/main.ts`:
```typescript
import type { WorkerToMain, MainToWorker } from "./protocol";
import { newRunFile, addCell, type RunFile } from "./schema";
import { renderResultsTable } from "./render";

const worker = new Worker(new URL("./bench.worker.ts", import.meta.url), { type: "module" });
const $ = (id: string) => document.getElementById(id)!;

let run: RunFile | null = null;
const send = (m: MainToWorker) => worker.postMessage(m);

worker.onmessage = (e: MessageEvent<WorkerToMain>) => {
  const m = e.data;
  if (m.type === "probe:result") {
    run = newRunFile("4090-linux", m.probe, new Date().toISOString());
    $("probe-box").innerHTML = `<pre>${JSON.stringify(m.probe, null, 2)}</pre>`;
  } else if (m.type === "progress") {
    $("status").textContent = `${Math.round(m.progress * 100)}% — ${m.text}`;
  } else if (m.type === "bench:result") {
    if (run) run = addCell(run, m.cell);
    $("status").textContent = "done";
    $("results").innerHTML = run ? renderResultsTable(run) : "";
    ($("export") as HTMLButtonElement).disabled = false;
    ($("run") as HTMLButtonElement).disabled = false;
  } else if (m.type === "error") {
    $("status").textContent = `ERROR: ${m.message}`;
    ($("run") as HTMLButtonElement).disabled = false;
  }
};

$("run").addEventListener("click", () => {
  const sel = $("model") as HTMLSelectElement;
  const quant = sel.selectedOptions[0].dataset.quant ?? "unknown";
  ($("run") as HTMLButtonElement).disabled = true;
  send({ type: "bench", modelId: sel.value, quant });
});

$("export").addEventListener("click", () => {
  if (!run) return;
  const blob = new Blob([JSON.stringify(run, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${run.deviceLabel}-${run.ts.replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

send({ type: "probe" });
```

Run: `npm test` → Expected: tutti PASS. `npx tsc --noEmit` → Expected: 0 errori.

- [ ] **Step 5: Commit** — `git commit -am "feat: minimal bench UI with probe box and json export"`

---

### Task 9: Run end-to-end sulla 4090 + README

**Files:**
- Create: `README.md`, `results/<file esportato>.json`

Questo task è manuale (serve la GPU fisica) — lo esegue Cristiano o la sessione con accesso al browser.

- [ ] **Step 1: Verifica WebGPU nel browser**

Aprire `chrome://gpu` → sezione WebGPU: Expected "Hardware accelerated". Se disabilitato su Linux/NVIDIA: rilanciare con `google-chrome --enable-features=Vulkan` (o abilitare `chrome://flags/#enable-vulkan`); ultima spiaggia `--enable-unsafe-webgpu`. Annotare nel README cosa è servito davvero.

- [ ] **Step 2: Run cold**

`npm run dev` → aprire la pagina, verificare probe box (deve mostrare vendor nvidia e i limits — **annotare `maxStorageBufferBindingSize`**: è il dato-finding #0). DevTools → Application → Storage → "Clear site data" per garantire cold. Run bench. Expected: progress di download, poi riga risultati con `cache: cold`.

- [ ] **Step 3: Run warm**

Ricaricare pagina (senza clear), Run bench. Expected: `cache: warm`, loadMs molto più basso. Export JSON → salvare in `results/`.

- [ ] **Step 4: README**

`README.md` con: cos'è il progetto (2 righe + link alla spec), prerequisiti (Chrome recente, flag Linux/NVIDIA se servite allo Step 1), `npm install && npm run dev`, come si esegue un bench e si esporta, dove vivono i risultati, nota COOP/COEP (e l'eventuale fallback `credentialless` se serve per gli shard HF).

- [ ] **Step 5: Commit finale**

```bash
git add -A && git commit -m "feat: first end-to-end bench run on 4090-linux + README"
```

---

## Self-review (fatta in scrittura)

1. **Spec coverage (1a)**: SPA ✓ (T1), worker ✓ (T7), adapter WebLLM ✓ (T6), probe ✓ (T5), schema versionato ✓ (T2), metriche TTFT/tok-s/load cold-warm ✓ (T3, T9), export JSON ✓ (T8), run 4090 ✓ (T9). Fuori scope dichiarato: altri adapter, qualità, memoria stimata (richiede più lavoro su measureUserAgentSpecificMemory — 1b), long-task observer (1b), altri device.
2. **Placeholder**: nessun TBD; i due punti aperti (nome export cache util; ID modello esatto) hanno step di verifica espliciti con procedura.
3. **Coerenza tipi**: `GenTimeline` (metrics) ↔ adapter ↔ BenchServer; `LoadReport`/`GenMetrics`/`BenchCell` (schema) ↔ protocol ↔ render: nomi e firme identici tra i task.
