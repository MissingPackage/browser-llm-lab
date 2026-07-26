# Fase 1b — Matrice — Adapter Transformers.js Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TransformersJsAdapter` (Transformers.js / ONNX Runtime Web, WebGPU) implementing the existing `InferenceAdapter` interface, wire it into the bench worker behind a new UI stack selector, and produce one real (non-SwiftShader) benchmark run on the local 4090 as proof it works end-to-end.

**Architecture:** `TransformersJsAdapter` follows the exact dependency-injection shape already used by `WebLLMAdapter` (`src/adapters/webllm.ts`): an injectable factory that produces a minimal engine object (`generate()`/`dispose()`), an injectable cache-check, and an injectable clock — so unit tests never touch the real network/GPU/library internals. The only place that touches the real `@huggingface/transformers` API is the default factory, exercised for real only by the existing Playwright e2e driver on physical hardware. `BenchServer` moves from a single hardcoded adapter to a `Record<StackId, () => InferenceAdapter>` registry, dispatched by a new `stack` field on the `bench` protocol message.

**Tech Stack:** TypeScript, Vite, Vitest, `@huggingface/transformers` (Transformers.js), Playwright (existing e2e driver).

## Global Constraints

- No AI attribution in any commit message.
- `erasableSyntaxOnly` is on in `tsconfig.json`: no parameter properties in classes (e.g. `constructor(private x: T)` is forbidden — assign fields in the constructor body instead, as `WebLLMAdapter` already does).
- The existing WebLLM adapter and its tests must keep passing throughout — baseline before this plan is 39/39 tests green, `tsc --noEmit` clean, `npm run build` ok.
- This plan extends `StackId` with `"transformersjs"` only. `"wllama"` is a later phase (`fase-1b-matrice` Phase 2) — do not add it here.
- `qualityScore` / schema v3 bump is out of scope here (Phase 3 of the same goal) — only the `stack`/`id` union changes in this plan.
- **Version note**: the approved design doc (`docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md`) refers to "Transformers.js v3". Checked against the npm registry on 2026-07-26: the latest published version is `4.2.0` (v3 branch stopped at `3.8.1`). Same `pipeline()`/`TextStreamer` API shape applies per current official docs — this is a version-label correction, not a scope or API change. Use `^4.2.0` in `package.json`.
- Model for this plan: `onnx-community/Qwen2.5-0.5B-Instruct` at `dtype: "q4"`, `device: "webgpu"` — confirmed available on the Hub (2026-07-26 research), same base model already used for the WebLLM smoke run, keeping stacks apples-to-apples. Wiring Small/Mid tier models later is just a different `modelId`/`quant` pair through the same adapter code path — no new task needed for that.

---

### Task 1: `TransformersJsAdapter`

**Files:**
- Modify: `src/schema.ts` (add `StackId` type, change `BenchCell.stack` to use it)
- Modify: `src/adapters/types.ts` (change `InferenceAdapter.id` to use `StackId`)
- Create: `src/adapters/transformersjs.ts`
- Create: `tests/transformersjs-adapter.test.ts`
- Modify: `package.json` (add `@huggingface/transformers`)

**Interfaces:**
- Consumes: `InferenceAdapter`, `AdapterCapabilities`, `GenerateRequest` from `./types`; `LoadReport` from `../schema`; `GenTimeline` from `../metrics` (all unchanged).
- Produces: `StackId` (exported from `src/schema.ts`, union `"webllm" | "transformersjs"`) — Task 2 imports this for `protocol.ts` and `benchServer.ts`. `TransformersJsAdapter` class (exported from `src/adapters/transformersjs.ts`) — Task 2 imports this for `bench.worker.ts`.

- [ ] **Step 1: Add `StackId` and thread it through the existing types**

In `src/schema.ts`, add near the top (after the imports, before `DeviceProbe`):

```ts
export type StackId = "webllm" | "transformersjs";
```

Then change the `BenchCell` field:

```ts
export interface BenchCell {
  stack: StackId;
  modelId: string;
  quant: string;
  promptId: string;
  load: LoadReport;
  gen: GenMetricsAgg;
  replicates: GenMetrics[];
  anomalies: string[]; // es. "high-variance"
}
```

(This replaces the old `stack: "webllm"; // union estesa nel piano "1b — matrice"` line.)

In `src/adapters/types.ts`, add the import and change the field:

```ts
import type { LoadReport, StackId } from "../schema";
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
  readonly id: StackId;
  capabilities(): AdapterCapabilities;
  load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport>;
  generate(req: GenerateRequest): Promise<GenTimeline>;
  dispose(): Promise<void>;
}
```

- [ ] **Step 2: Run the existing suite to confirm nothing broke**

Run: `npm test`
Expected: PASS, same count as baseline (39/39) — `WebLLMAdapter.id = "webllm" as const` is still assignable to `StackId`, so no other file needs to change yet.

- [ ] **Step 3: Install the new dependency**

Run: `npm install @huggingface/transformers@^4.2.0`
Expected: `package.json`/`package-lock.json` updated, install succeeds.

- [ ] **Step 4: Write the failing tests for `TransformersJsAdapter`**

Create `tests/transformersjs-adapter.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { TransformersJsAdapter } from "../src/adapters/transformersjs";

function fakeEngine(tokenCount: number) {
  let disposed = false;
  return {
    engine: {
      generate: async (_messages: unknown, _maxTokens: number, onToken: () => void) => {
        for (let i = 0; i < tokenCount; i++) onToken();
      },
      dispose: async () => {
        disposed = true;
      },
    },
    wasDisposed: () => disposed,
  };
}

describe("TransformersJsAdapter", () => {
  it("load reports cold/warm from cache probe and measures loadMs", async () => {
    let t = 0;
    const { engine } = fakeEngine(3);
    const a = new TransformersJsAdapter({
      engineFactory: async () => engine,
      isCached: async () => false,
      now: () => (t += 500), // load: t0=500, t1=1000 → 500ms
    });
    const r = await a.load("test-model", () => {});
    expect(r.cacheState).toBe("cold");
    expect(r.loadMs).toBe(500);
  });

  it("generate builds a timeline with one timestamp per generated token", async () => {
    let t = 0;
    const { engine } = fakeEngine(3);
    const a = new TransformersJsAdapter({
      engineFactory: async () => engine,
      isCached: async () => true,
      now: () => (t += 100),
    });
    const r = await a.load("test-model", () => {});
    expect(r.cacheState).toBe("warm");
    const tl = await a.generate({ prompt: "hi", maxTokens: 8 });
    expect(tl.chunkTimestamps.length).toBe(3);
    expect(tl.promptTokens).toBeNull();
  });

  it("generate before load throws", async () => {
    const { engine } = fakeEngine(1);
    const a = new TransformersJsAdapter({ engineFactory: async () => engine });
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow("not loaded");
  });

  it("passes greedy decode params and the prompt as a chat message to the engine", async () => {
    let captured: unknown[] = [];
    const engine = {
      generate: async (messages: unknown, maxTokens: number, onToken: () => void) => {
        captured = [messages, maxTokens];
        onToken();
      },
      dispose: async () => {},
    };
    const a = new TransformersJsAdapter({ engineFactory: async () => engine, isCached: async () => false });
    await a.load("test-model", () => {});
    await a.generate({ prompt: "hi", maxTokens: 8 });
    expect(captured).toEqual([[{ role: "user", content: "hi" }], 8]);
  });

  it("dispose disposes the engine; generate after dispose throws not loaded", async () => {
    const { engine, wasDisposed } = fakeEngine(1);
    const a = new TransformersJsAdapter({ engineFactory: async () => engine, isCached: async () => false });
    await a.load("test-model", () => {});
    await a.dispose();
    expect(wasDisposed()).toBe(true);
    await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow("not loaded");
  });
});
```

- [ ] **Step 5: Run the test file to verify it fails**

Run: `npx vitest run tests/transformersjs-adapter.test.ts`
Expected: FAIL — `Cannot find module '../src/adapters/transformersjs'`.

- [ ] **Step 6: Implement `TransformersJsAdapter`**

Create `src/adapters/transformersjs.ts`:

```ts
import { pipeline, TextStreamer, ModelRegistry } from "@huggingface/transformers";
import type { InferenceAdapter, AdapterCapabilities, GenerateRequest } from "./types";
import type { LoadReport } from "../schema";
import type { GenTimeline } from "../metrics";

const DTYPE = "q4";
const DEVICE = "webgpu";
const TASK = "text-generation";

// Contratto minimo che usiamo davvero dell'oggetto restituito da pipeline():
// chiamabile, `.tokenizer` (passato a TextStreamer), `.dispose()`. Il tipo reale
// di pipeline() dipende dal task a runtime e non è utilmente esprimibile qui.
type RawPipeline = {
  tokenizer: ConstructorParameters<typeof TextStreamer>[0];
  dispose(): Promise<void>;
  (input: Array<{ role: string; content: string }>, options: Record<string, unknown>): Promise<unknown>;
};

interface TextGenerationEngine {
  generate(
    messages: Array<{ role: string; content: string }>,
    maxTokens: number,
    onToken: () => void,
  ): Promise<void>;
  dispose(): Promise<void>;
}

type EngineFactory = (
  modelId: string,
  onProgress: (text: string, progress: number) => void,
) => Promise<TextGenerationEngine>;

const defaultEngineFactory: EngineFactory = async (modelId, onProgress) => {
  const pipe = (await pipeline(TASK, modelId, {
    dtype: DTYPE,
    device: DEVICE,
    progress_callback: (info: { status: string; file?: string; progress?: number }) => {
      const pct = typeof info.progress === "number" ? info.progress / 100 : 0;
      onProgress(info.file ? `${info.status}: ${info.file}` : info.status, pct);
    },
  })) as unknown as RawPipeline;
  return {
    generate: async (messages, maxTokens, onToken) => {
      const streamer = new TextStreamer(pipe.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: () => onToken(),
      });
      await pipe(messages, { max_new_tokens: maxTokens, do_sample: false, streamer });
    },
    dispose: () => pipe.dispose(),
  };
};

const defaultIsCached = (modelId: string) =>
  ModelRegistry.is_pipeline_cached(TASK, modelId, { dtype: DTYPE }).catch(() => false);

export class TransformersJsAdapter implements InferenceAdapter {
  readonly id = "transformersjs" as const;
  private engine: TextGenerationEngine | null = null;
  private engineFactory: EngineFactory;
  private isCached: (modelId: string) => Promise<boolean>;
  private now: () => number;

  constructor(deps?: {
    engineFactory?: EngineFactory;
    isCached?: (modelId: string) => Promise<boolean>;
    now?: () => number;
  }) {
    this.engineFactory = deps?.engineFactory ?? defaultEngineFactory;
    this.isCached = deps?.isCached ?? defaultIsCached;
    this.now = deps?.now ?? (() => performance.now());
  }

  capabilities(): AdapterCapabilities {
    return { logprobs: false, streaming: true, seed: false }; // logprobs: rivalutare nel modulo qualità (Fase 3)
  }

  async load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport> {
    const cached = await this.isCached(modelId);
    const t0 = this.now();
    this.engine = await this.engineFactory(modelId, onProgress);
    return { loadMs: this.now() - t0, cacheState: cached ? "warm" : "cold" };
  }

  async generate(req: GenerateRequest): Promise<GenTimeline> {
    if (!this.engine) throw new Error("not loaded");
    const tRequestStart = this.now();
    const chunkTimestamps: number[] = [];
    await this.engine.generate([{ role: "user", content: req.prompt }], req.maxTokens, () => {
      chunkTimestamps.push(this.now());
    });
    return { tRequestStart, chunkTimestamps, promptTokens: null, completionTokens: null };
  }

  async dispose(): Promise<void> {
    await this.engine?.dispose();
    this.engine = null;
  }
}
```

**If `tsc` reports a mismatch** on the `progress_callback` parameter shape (i.e. the real `PretrainedOptions` type doesn't have `status`/`file`/`progress` under those names): check `node_modules/@huggingface/transformers/types/**/*.d.ts` for the actual declared type and adjust the inline annotation to match — the shape above is taken from the official Transformers.js docs (skill `huggingface-skills:transformers-js`, `TEXT_GENERATION.md`/progress-callback reference) but could drift between versions.

- [ ] **Step 7: Run the test file to verify it passes**

Run: `npx vitest run tests/transformersjs-adapter.test.ts`
Expected: PASS (5/5).

- [ ] **Step 8: Run the full suite and typecheck**

Run: `npm test && npx tsc --noEmit`
Expected: PASS, count is baseline + 5 (44/44 if baseline was 39/39), `tsc` clean.

- [ ] **Step 9: Commit**

```bash
git add src/schema.ts src/adapters/types.ts src/adapters/transformersjs.ts tests/transformersjs-adapter.test.ts package.json package-lock.json
git commit -m "feat: add TransformersJsAdapter behind InferenceAdapter"
```

---

### Task 2: `BenchServer` multi-stack routing

**Files:**
- Modify: `src/protocol.ts`
- Modify: `tests/protocol.test.ts`
- Modify: `src/benchServer.ts`
- Modify: `tests/benchServer.test.ts`
- Modify: `src/bench.worker.ts`

**Interfaces:**
- Consumes: `StackId` from `./schema` (Task 1); `TransformersJsAdapter` from `./adapters/transformersjs` (Task 1); `WebLLMAdapter` from `./adapters/webllm` (unchanged).
- Produces: `MainToWorker`'s `bench` variant now carries `stack: StackId` — Task 3 (`main.ts`) must send it on every `bench` message.

- [ ] **Step 1: Write the failing protocol tests**

Replace the contents of `tests/protocol.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { isMainToWorker } from "../src/protocol";

describe("protocol guards", () => {
  it("accepts valid messages", () => {
    expect(isMainToWorker({ type: "probe" })).toBe(true);
    expect(isMainToWorker({ type: "bench", stack: "webllm", modelId: "m", quant: "q4f16_1" })).toBe(true);
    expect(isMainToWorker({ type: "bench", stack: "transformersjs", modelId: "m", quant: "q4" })).toBe(true);
  });
  it("rejects invalid messages", () => {
    expect(isMainToWorker(null)).toBe(false);
    expect(isMainToWorker({ type: "bench" })).toBe(false); // manca modelId/quant/stack
    expect(isMainToWorker({ type: "bench", modelId: "m", quant: "q" })).toBe(false); // manca stack
    expect(isMainToWorker({ type: "bench", stack: "not-a-stack", modelId: "m", quant: "q" })).toBe(false);
    expect(isMainToWorker({ type: "nope" })).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run tests/protocol.test.ts`
Expected: FAIL — current `isMainToWorker` accepts `{ type: "bench", modelId, quant }` without `stack`, so the "manca stack" and "not-a-stack" assertions fail (they currently return `true` instead of `false`).

- [ ] **Step 3: Update `protocol.ts`**

Replace `src/protocol.ts`:

```ts
import type { DeviceProbe, BenchCell, StackId } from "./schema";

export type MainToWorker =
  | { type: "probe" }
  | { type: "bench"; stack: StackId; modelId: string; quant: string };

export type WorkerToMain =
  | { type: "probe:result"; probe: DeviceProbe }
  | { type: "progress"; text: string; progress: number }
  | { type: "bench:result"; cell: BenchCell }
  | { type: "error"; message: string };

const STACK_IDS: readonly StackId[] = ["webllm", "transformersjs"];

export function isMainToWorker(x: unknown): x is MainToWorker {
  if (typeof x !== "object" || x === null || !("type" in x)) return false;
  const m = x as Record<string, unknown>;
  if (m.type === "probe") return true;
  if (m.type === "bench") {
    return (
      typeof m.modelId === "string" &&
      typeof m.quant === "string" &&
      typeof m.stack === "string" &&
      (STACK_IDS as string[]).includes(m.stack)
    );
  }
  return false;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run tests/protocol.test.ts`
Expected: PASS (2/2).

- [ ] **Step 5: Write the failing `BenchServer` tests**

Replace `tests/benchServer.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { BenchServer } from "../src/benchServer";
import type { WorkerToMain } from "../src/protocol";
import type { InferenceAdapter } from "../src/adapters/types";

const fakeProbe = async () => ({
  webgpu: true,
  adapterInfo: null,
  limits: null,
  features: [],
  userAgent: "ua",
  deviceMemoryGB: null,
  browser: { name: "chrome", version: "150.0" },
  anomalies: [],
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

const fakeAdapters = () => ({ webllm: fakeAdapter, transformersjs: fakeAdapter });

describe("BenchServer", () => {
  it("probe message → probe:result", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapters: fakeAdapters(), probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ type: "probe" });
    expect(out[0].type).toBe("probe:result");
  });

  it("bench message → progress then bench:result with aggregated metrics over replicates", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapters: fakeAdapters(), probe: fakeProbe, post: (m) => out.push(m), replicateCount: 2 });
    await s.handle({ type: "bench", stack: "webllm", modelId: "test-model", quant: "q4f16_1" });
    const types = out.map((m) => m.type);
    expect(types.filter((t) => t === "progress").length).toBeGreaterThanOrEqual(2);
    const result = out.find((m) => m.type === "bench:result");
    expect(result).toBeDefined();
    if (result?.type === "bench:result") {
      expect(result.cell.modelId).toBe("test-model");
      expect(result.cell.stack).toBe("webllm");
      expect(result.cell.replicates.length).toBe(2);
      expect(result.cell.gen.ttftMs.mean).toBe(10);
      expect(result.cell.load.cacheState).toBe("cold");
      expect(result.cell.promptId).toBe("bench-512-v1");
      expect(result.cell.anomalies).toEqual([]);
    }
  });

  it("dispatches to the adapter registered for the requested stack", async () => {
    const out: WorkerToMain[] = [];
    const calls: string[] = [];
    const adapters = {
      webllm: () => { calls.push("webllm"); return fakeAdapter(); },
      transformersjs: () => { calls.push("transformersjs"); return { ...fakeAdapter(), id: "transformersjs" as const }; },
    };
    const s = new BenchServer({ adapters, probe: fakeProbe, post: (m) => out.push(m), replicateCount: 1 });
    await s.handle({ type: "bench", stack: "transformersjs", modelId: "m", quant: "q4" });
    expect(calls).toEqual(["transformersjs"]);
    const result = out.find((m) => m.type === "bench:result");
    if (result?.type === "bench:result") expect(result.cell.stack).toBe("transformersjs");
  });

  it("flags high-variance when decode rate spreads across replicates beyond threshold", async () => {
    let call = 0;
    const varyingAdapter: InferenceAdapter = {
      ...fakeAdapter(),
      generate: async () => {
        call++;
        const chunks = call === 1 ? [0, 100] : [0, 25];
        return { tRequestStart: 0, chunkTimestamps: chunks, promptTokens: 512, completionTokens: 5 };
      },
    };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => varyingAdapter, transformersjs: () => varyingAdapter },
      probe: fakeProbe,
      post: (m) => out.push(m),
      replicateCount: 2,
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q" });
    const result = out.find((m) => m.type === "bench:result");
    if (result?.type === "bench:result") {
      expect(result.cell.anomalies.some((a) => a.includes("high-variance"))).toBe(true);
    } else {
      throw new Error("expected bench:result");
    }
  });

  it("adapter failure → error message, no throw", async () => {
    const broken = { ...fakeAdapter(), load: async () => { throw new Error("boom"); } };
    const out: WorkerToMain[] = [];
    const s = new BenchServer({
      adapters: { webllm: () => broken, transformersjs: () => broken },
      probe: fakeProbe,
      post: (m) => out.push(m),
    });
    await s.handle({ type: "bench", stack: "webllm", modelId: "m", quant: "q" });
    expect(out.some((m) => m.type === "error" && m.message.includes("boom"))).toBe(true);
  });

  it("invalid message → error", async () => {
    const out: WorkerToMain[] = [];
    const s = new BenchServer({ adapters: fakeAdapters(), probe: fakeProbe, post: (m) => out.push(m) });
    await s.handle({ garbage: true });
    expect(out[0].type).toBe("error");
  });
});
```

- [ ] **Step 6: Run to verify it fails**

Run: `npx vitest run tests/benchServer.test.ts`
Expected: FAIL to compile/run — `BenchServer`'s constructor still expects `adapterFactory`, not `adapters`.

- [ ] **Step 7: Update `benchServer.ts`**

Replace `src/benchServer.ts`:

```ts
import { isMainToWorker, type WorkerToMain } from "./protocol";
import type { InferenceAdapter } from "./adapters/types";
import type { DeviceProbe, BenchCell, StackId } from "./schema";
import { computeGenMetrics, aggregateReplicates } from "./metrics";
import { PROMPT_512 } from "./promptset";

const DEFAULT_REPLICATE_COUNT = 3;
const HIGH_VARIANCE_THRESHOLD = 0.15; // stdev/mean sul tok/s aggregato

export class BenchServer {
  private deps: {
    adapters: Record<StackId, () => InferenceAdapter>;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
    replicateCount?: number;
  };

  constructor(deps: {
    adapters: Record<StackId, () => InferenceAdapter>;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
    replicateCount?: number;
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
        const adapter = this.deps.adapters[msg.stack]();
        const replicateCount = this.deps.replicateCount ?? DEFAULT_REPLICATE_COUNT;
        try {
          const load = await adapter.load(msg.modelId, (text, progress) =>
            this.deps.post({ type: "progress", text, progress }),
          );
          const replicates = [];
          for (let i = 0; i < replicateCount; i++) {
            this.deps.post({
              type: "progress",
              text: `generating (replica ${i + 1}/${replicateCount})…`,
              progress: (i + 1) / replicateCount,
            });
            const timeline = await adapter.generate({ prompt: PROMPT_512.text, maxTokens: 256 });
            replicates.push(computeGenMetrics(timeline));
          }
          const gen = aggregateReplicates(replicates);
          const anomalies: string[] = [];
          if (
            gen.decodeToksPerSec &&
            gen.decodeToksPerSec.mean > 0 &&
            gen.decodeToksPerSec.stdev / gen.decodeToksPerSec.mean > HIGH_VARIANCE_THRESHOLD
          ) {
            anomalies.push(
              `high-variance: decodeToksPerSec stdev/mean=${(gen.decodeToksPerSec.stdev / gen.decodeToksPerSec.mean).toFixed(2)} > ${HIGH_VARIANCE_THRESHOLD}`,
            );
          }
          const cell: BenchCell = {
            stack: adapter.id,
            modelId: msg.modelId,
            quant: msg.quant,
            promptId: PROMPT_512.id,
            load,
            gen,
            replicates,
            anomalies,
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

(Only the `deps`/constructor types changed from `adapterFactory: () => InferenceAdapter` to `adapters: Record<StackId, () => InferenceAdapter>`, and the dispatch line `const adapter = this.deps.adapters[msg.stack]();` replaces `const adapter = this.deps.adapterFactory();`. The rest of the method body is unchanged.)

- [ ] **Step 8: Run to verify it passes**

Run: `npx vitest run tests/benchServer.test.ts`
Expected: PASS (6/6).

- [ ] **Step 9: Wire both adapters into the worker**

Replace `src/bench.worker.ts`:

```ts
import { BenchServer } from "./benchServer";
import { WebLLMAdapter } from "./adapters/webllm";
import { TransformersJsAdapter } from "./adapters/transformersjs";
import { probeWebGPU } from "./probe";

const server = new BenchServer({
  adapters: {
    webllm: () => new WebLLMAdapter(),
    transformersjs: () => new TransformersJsAdapter(),
  },
  probe: () => probeWebGPU(navigator.gpu, navigator as { userAgent: string; deviceMemory?: number }),
  post: (m) => self.postMessage(m),
});

self.onmessage = (e: MessageEvent) => void server.handle(e.data);
```

- [ ] **Step 10: Run the full suite and typecheck**

Run: `npm test && npx tsc --noEmit`
Expected: PASS, `tsc` clean. (`webllm-adapter.test.ts` is untouched and still green — it doesn't go through `BenchServer`.)

- [ ] **Step 11: Commit**

```bash
git add src/protocol.ts tests/protocol.test.ts src/benchServer.ts tests/benchServer.test.ts src/bench.worker.ts
git commit -m "feat: route bench requests to a per-stack adapter registry"
```

---

### Task 3: UI stack selector

**Files:**
- Modify: `index.html`
- Modify: `src/main.ts`

**Interfaces:**
- Consumes: `StackId` from `./schema` (Task 1); `MainToWorker`'s `bench` variant now requires `stack` (Task 2).
- Produces: none consumed by later tasks — Task 4's e2e driver interacts with the `#stack`/`#model` DOM directly, not with any TS-level export.

There is no unit test for `main.ts` in this codebase (it's DOM-driven and already verified manually via Playwright per existing project convention — see `HANDOFF.md` §5). This task is verified manually with the dev server, same as prior UI tasks.

- [ ] **Step 1: Add the stack selector and tag existing model options**

In `index.html`, replace the `<section>` containing the model selector:

```html
    <section>
      <label>Stack:
        <select id="stack">
          <option value="webllm">WebLLM (MLC)</option>
          <option value="transformersjs">Transformers.js</option>
        </select>
      </label>
      <label>Model:
        <select id="model">
          <option value="Qwen2.5-0.5B-Instruct-q4f32_1-MLC" data-quant="q4f32_1" data-stack="webllm">Qwen2.5-0.5B q4f32_1 (Chrome/Linux: niente shader-f16)</option>
          <option value="Qwen2.5-0.5B-Instruct-q4f16_1-MLC" data-quant="q4f16_1" data-stack="webllm">Qwen2.5-0.5B q4f16_1 (richiede shader-f16)</option>
          <option value="onnx-community/Qwen2.5-0.5B-Instruct" data-quant="q4" data-stack="transformersjs">Qwen2.5-0.5B q4 (Transformers.js/WebGPU)</option>
        </select>
      </label>
      <button id="run">Run bench</button>
      <button id="export" disabled>Export JSON</button>
    </section>
```

- [ ] **Step 2: Filter the model dropdown by selected stack and send `stack` on run**

Replace `src/main.ts`:

```ts
import type { WorkerToMain, MainToWorker } from "./protocol";
import type { StackId } from "./schema";
import { newRunFile, addCell, type RunFile } from "./schema";
import { renderResultsTable } from "./render";

const worker = new Worker(new URL("./bench.worker.ts", import.meta.url), { type: "module" });
const $ = (id: string) => document.getElementById(id)!;

let run: RunFile | null = null;
const send = (m: MainToWorker) => worker.postMessage(m);

function applyStackFilter(): void {
  const stack = ($("stack") as HTMLSelectElement).value;
  const modelSel = $("model") as HTMLSelectElement;
  for (const opt of Array.from(modelSel.options)) {
    const compatible = opt.dataset.stack === stack;
    opt.hidden = !compatible;
    opt.disabled = !compatible;
  }
  const selected = modelSel.selectedOptions[0];
  if (!selected || selected.hidden) {
    const firstVisible = Array.from(modelSel.options).find((o) => !o.hidden);
    if (firstVisible) modelSel.value = firstVisible.value;
  }
}

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
    ($("export") as HTMLButtonElement).disabled = run === null || run.cells.length === 0;
  } else {
    const _exhaustive: never = m;
    throw new Error(`unhandled WorkerToMain variant: ${JSON.stringify(_exhaustive)}`);
  }
};

$("stack").addEventListener("change", applyStackFilter);
applyStackFilter();

$("run").addEventListener("click", () => {
  const sel = $("model") as HTMLSelectElement;
  const stackSel = $("stack") as HTMLSelectElement;
  const quant = sel.selectedOptions[0].dataset.quant ?? "unknown";
  ($("run") as HTMLButtonElement).disabled = true;
  ($("export") as HTMLButtonElement).disabled = true;
  send({ type: "bench", stack: stackSel.value as StackId, modelId: sel.value, quant });
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

- [ ] **Step 3: Run the full suite and typecheck**

Run: `npm test && npx tsc --noEmit`
Expected: PASS, `tsc` clean (no test exercises `main.ts` directly, so this just guards against a typo breaking the build).

- [ ] **Step 4: Manual verification with the dev server**

Run: `npm run dev`, open `http://localhost:5173`.
Expected: the "Stack" dropdown shows "WebLLM (MLC)" and "Transformers.js"; switching to "Transformers.js" filters the Model dropdown down to the single `onnx-community/Qwen2.5-0.5B-Instruct` option; switching back to "WebLLM (MLC)" restores the two MLC options. Take a screenshot or note the observed behavior in the task's PR/commit description.

- [ ] **Step 5: Commit**

```bash
git add index.html src/main.ts
git commit -m "feat: add stack selector to the UI, filtered by compatible models"
```

---

### Task 4: e2e driver stack support + real 4090 run

**Files:**
- Modify: `scripts/e2e-bench.mjs`
- Create: one new file under `results/` (the real run output)

**Interfaces:**
- Consumes: `#stack`/`#model` DOM elements from Task 3.
- Produces: a committed `results/*.json` file with `stack: "transformersjs"` — this is the goal's own "at least one real run" done-when condition, nothing downstream in this plan consumes it programmatically.

- [ ] **Step 1: Add `STACK` env override to the driver**

In `scripts/e2e-bench.mjs`, insert this block right before the existing `if (process.env.MODEL_ID) { ... }` block:

```js
// Override stack via env (default: primo stack nel select, cioè "webllm")
if (process.env.STACK) {
  await page.evaluate((stack) => {
    const stackSel = document.querySelector("#stack");
    stackSel.value = stack;
    stackSel.dispatchEvent(new Event("change"));
  }, process.env.STACK);
  console.log(`[e2e] stack override: ${process.env.STACK}`);
}
```

Then update the existing `MODEL_ID` override block to also tag the injected option with the current stack, so it survives the stack filter:

```js
if (process.env.MODEL_ID) {
  await page.evaluate(
    ([id, q, stack]) => {
      const sel = document.querySelector("#model");
      const o = document.createElement("option");
      o.value = id;
      o.dataset.quant = q;
      o.dataset.stack = stack;
      o.textContent = id;
      sel.appendChild(o);
      sel.value = id;
    },
    [process.env.MODEL_ID, process.env.QUANT ?? "unknown", process.env.STACK ?? "webllm"],
  );
  console.log(`[e2e] modello override: ${process.env.MODEL_ID}`);
}
```

- [ ] **Step 2: Run the real bench on the 4090**

Run: `HEADED=1 STACK=transformersjs node scripts/e2e-bench.mjs`

This uses the `onnx-community/Qwen2.5-0.5B-Instruct` option already in `index.html` (Task 3) — no `MODEL_ID` override needed.

Expected: console shows `[e2e] PROBE: ...` with `adapterInfo.vendor` = `"nvidia"`, `[e2e] GPU reale (nvidia, non-software)? true`, `[e2e] stack override: transformersjs`, then (after model download on first run) `[e2e] STATUS: done`, then `[e2e] EXPORT salvato: /tmp/<filename>.json`.

If it fails with a real error (not a flaky timeout), stop and diagnose per the `root-cause` protocol before touching code — do not loosen the driver's GPU-reality checks to make it "pass".

- [ ] **Step 3: Verify the exported file is schema-valid and copy it into `results/`**

Run:
```bash
LATEST=$(ls -t /tmp/4090-linux-*.json | head -1)
node -e "
const d = JSON.parse(require('fs').readFileSync(process.argv[1], 'utf8'));
if (d.schemaVersion !== 2) throw new Error('unexpected schemaVersion: ' + d.schemaVersion);
if (!d.cells.some((c) => c.stack === 'transformersjs')) throw new Error('no transformersjs cell in run file');
console.log('OK — schemaVersion', d.schemaVersion, '— stacks:', d.cells.map((c) => c.stack));
" "$LATEST"
cp "$LATEST" results/
```
Expected: `OK — schemaVersion 2 — stacks: [ 'transformersjs' ]` (or more, if you ran WebLLM cells earlier in the same session), then the file appears under `results/`.

- [ ] **Step 4: Final gate for the whole phase**

Run: `npm test && npx tsc --noEmit && npm run build`
Expected: full suite green (44/44 from Task 1 + any from Task 2's new assertions — check the actual count printed), `tsc` clean, build succeeds.

- [ ] **Step 5: Commit**

```bash
git add scripts/e2e-bench.mjs results/
git commit -m "feat: e2e driver stack override + real transformersjs run on 4090"
```

---

## Self-review notes (writing-plans, run before handoff)

- **Spec coverage**: every design-doc item assigned to Phase 1 of `fase-1b-matrice` (adapter behind the existing interface, minimal stack UI, real 4090 run) has a task. `qualityScore`/schema-v3 and the `wllama` adapter are explicitly out of scope here (Phase 2/3 of the same goal) — not gaps, by design.
- **Placeholder scan**: no TBD/TODO; the one open contingency (`progress_callback` field-name drift) is called out explicitly with a concrete fallback action, not left vague.
- **Type consistency**: `StackId` introduced once in `schema.ts`, consumed identically by `adapters/types.ts`, `protocol.ts`, `benchServer.ts`, `bench.worker.ts`, `main.ts` — no renamed duplicates. `TextGenerationEngine.generate(messages, maxTokens, onToken)` signature is identical between the adapter implementation and every test's fake engine.
