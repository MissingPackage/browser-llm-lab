# Fase 1b — Fondamenta: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chiudere i FIX-IN-1B della final review 1a, estendere probe e schema a v2 (`browser`, `features`, `anomalies`) con rilevazione del software-adapter, e introdurre repliche multiple per cella con aggregazione statistica — le fondamenta su cui poggerà la "matrice piena" (adapter Transformers.js/wllama, sweep multi-device, modulo qualità), che resta un piano separato successivo.

**Architecture:** Nessun nuovo modulo strutturale: si estendono i moduli esistenti (`schema.ts`, `probe.ts`, `metrics.ts`, `benchServer.ts`, `render.ts`, `main.ts`) mantenendo l'architettura 1a (compute in worker, metriche pure, adapter dietro `InferenceAdapter`). `SCHEMA_VERSION` passa da 1 a 2: è un breaking change di formato, dichiarato e non retro-compatibile con i JSON in `results/` (che restano validi come archivio storico v1, non vengono riscritti).

**Tech Stack:** stesso di 1a — Vite + TypeScript, Vitest, `@mlc-ai/web-llm`, `@webgpu/types`. Nessuna nuova dipendenza.

## Global Constraints

- Spec di riferimento: `docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md` (§Fasatura, 1b) + `HANDOFF.md` §1 (next-decidable) + `.superpowers/sdd/progress.md` (FIX-IN-1B, elencati per task).
- **Fuori scope in questo piano** (restano per il piano successivo "1b — matrice"): adapter `TransformersJsAdapter`/`WllamaAdapter`, sweep multi-device (M4/S22), modulo qualità-leggera (perplexity/task-scoring), espansione set modelli. Questo piano copre solo le fondamenta elencate in HANDOFF §1 "primo passo concreto".
- `SCHEMA_VERSION = 2`; ogni file risultato lo dichiara. I run precedenti in `results/*.json` restano `schemaVersion: 1` e non vengono migrati.
- Decoding invariato da 1a: `temperature: 0` (greedy), `max_tokens: 256`, prompt `bench-512-v1`.
- Device label manuale invariato: nessun fingerprinting oltre al probe dichiarato.
- Soglia euristica software-adapter: vendor `""` **e** `architecture` `""` **e** `limits.maxBufferSize <= 134217728` (128 MiB) — valore confermato dal finding Firefox 152/llvmpipe (README, sezione "Cosa abbiamo verificato dal vivo").
- Repliche per cella: `REPLICATE_COUNT = 3` di default (costante iniettabile per i test); ogni replica è un `generate()` sullo stesso engine già caricato (non si ricarica il modello tra repliche).
- Commit frequenti, messaggi convenzionali, **nessuna attribuzione AI**.
- `docs/superpowers/` è gitignored dal commit `347a7fe` (2026-07-25): questo piano vive fuori dal controllo versione salvo `git add -f` esplicito — non forzare il tracking senza una decisione esplicita dell'utente (vedi HANDOFF, item di docket aperto da questa sessione).

---

## File Structure

```
browser-llm-lab/
  src/
    schema.ts        — MODIFY: SCHEMA_VERSION→2, DeviceProbe.{browser,features,anomalies}, BenchCell.{gen→aggregato,replicates,anomalies}
    probe.ts          — MODIFY: parseBrowser(), enumerazione adapter.features, detectSoftwareAdapter()
    metrics.ts        — MODIFY: aggregateReplicates()
    benchServer.ts     — MODIFY: loop repliche, REPLICATE_COUNT iniettabile, flag high-variance
    render.ts          — MODIFY: escapeHtml(), colonne mean±stdev, badge anomalie
    main.ts            — MODIFY: exhaustiveness guard su WorkerToMain, export disabilitato durante bench
    adapters/webllm.ts — invariato in questo piano
  tests/
    render.test.ts      — MODIFY: escaping, mean±stdev, anomalie
    webllm-adapter.test.ts — MODIFY: + test dispose()
    schema.test.ts       — MODIFY: fixture DeviceProbe/BenchCell aggiornate a v2
    probe.test.ts        — MODIFY: features, browser, software-adapter
    benchServer.test.ts  — MODIFY: fixture probe v2, repliche
  README.md           — MODIFY: schema v2, metodologia repliche, euristica software-adapter
```

---

### Task 1: FIX — escaping HTML in `render.ts`

**Files:**
- Modify: `src/render.ts`
- Modify: `tests/render.test.ts`

**Interfaces:**
- Consumes: `RunFile` (schema.ts, invariato in questo task).
- Produces: `escapeHtml(s: string): string` (non esportata, uso interno a `render.ts`); `renderResultsTable` invariata nella firma.

Riferimento: `.superpowers/sdd/progress.md` Task 8 minor — "innerHTML senza escaping (inerte oggi, rischio se model list diventa dinamica)". Il rischio diventa concreto quando `modelId`/`stack` provengono da fonti esterne (sweep multi-modello, piano successivo).

- [x] **Step 1: Test fallente**

Aggiungere a `tests/render.test.ts`:
```typescript
it("escapes html-unsafe characters in modelId", () => {
  let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
  run = addCell(run, {
    stack: "webllm",
    modelId: "<img src=x onerror=alert(1)>",
    quant: "q4f16_1",
    promptId: "bench-512-v1",
    load: { loadMs: 1, cacheState: "cold" },
    gen: { ttftMs: 1, decodeToksPerSec: 1, totalMs: 1, promptTokens: 1, completionTokens: 1 },
  });
  const html = renderResultsTable(run);
  expect(html).not.toContain("<img src=x");
  expect(html).toContain("&lt;img src=x onerror=alert(1)&gt;");
});
```

- [x] **Step 2: Run → FAIL** (`npm test` — il markup non escapato contiene `<img src=x` letterale)

- [x] **Step 3: Implementazione**

`src/render.ts`:
```typescript
import type { RunFile } from "./schema";

const fmt = (n: number | null, digits = 1) => (n === null ? "—" : n.toFixed(digits));

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function renderResultsTable(run: RunFile): string {
  if (run.cells.length === 0) return "<p>Nessun risultato ancora.</p>";
  const rows = run.cells
    .map(
      (c) => `<tr>
        <td>${escapeHtml(c.stack)}</td><td>${escapeHtml(c.modelId)}</td><td>${escapeHtml(c.quant)}</td>
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

- [x] **Step 4: Run → PASS** (`npm test`)

- [x] **Step 5: Commit**

```bash
git add src/render.ts tests/render.test.ts
git commit -m "fix: escape html-unsafe chars in results table (FIX-IN-1B)"
```

---

### Task 2: FIX — exhaustiveness guard su `WorkerToMain` in `main.ts`

**Files:**
- Modify: `src/main.ts`

**Interfaces:**
- Consumes: `WorkerToMain` (protocol.ts, invariato in questo task).
- Nessuna nuova interfaccia esportata: il guard è un pattern interno al dispatcher.

Riferimento: `.superpowers/sdd/progress.md` Task 8 minor — "niente exhaustiveness guard su WorkerToMain". Senza il guard, un domani (Task 6 di questo piano estende `WorkerToMain`? No — questo piano non tocca `protocol.ts`; ma il piano "matrice piena" lo farà per nuovi adapter/messaggi di replica) l'aggiunta di una variante silenziosamente non gestita non produce errore di compilazione.

Nessun test dedicato: `main.ts` non ha copertura unitaria in questo progetto (wiring DOM/worker, non logica pura) — il guard stesso è verificato da `tsc`, che fallisce a compile-time se una variante di `WorkerToMain` non è coperta da nessun branch.

- [x] **Step 1: Implementazione**

In `src/main.ts`, sostituire il dispatcher `worker.onmessage` con:
```typescript
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
  } else {
    const _exhaustive: never = m;
    throw new Error(`unhandled WorkerToMain variant: ${JSON.stringify(_exhaustive)}`);
  }
};
```

- [x] **Step 2: Verifica compilazione**

Run: `npx tsc --noEmit` → Expected: 0 errori (il branch `else` è irraggiungibile oggi con le 4 varianti correnti, quindi `_exhaustive` è tipizzato `never` e il codice compila pulito).

- [x] **Step 3: Verifica manuale del guard** (una tantum, non ripetuta ad ogni test run)

Aggiungere temporaneamente una variante fittizia a `WorkerToMain` in `protocol.ts` (es. `{ type: "ping" }`), rilanciare `npx tsc --noEmit` → Expected: errore `Type '{ type: "ping"; }' is not assignable to type 'never'` sulla riga del guard. Rimuovere la variante fittizia subito dopo (non commitare la modifica di prova).

- [x] **Step 4: Commit**

```bash
git add src/main.ts
git commit -m "fix: exhaustiveness guard on WorkerToMain dispatch (FIX-IN-1B)"
```

---

### Task 3: FIX — copertura test per `dispose()`

**Files:**
- Modify: `tests/webllm-adapter.test.ts`

**Interfaces:**
- Consumes: `WebLLMAdapter` (invariato in questo task — chiude solo un gap di test).

Riferimento: `.superpowers/sdd/progress.md` Task 6 minor — "dispose() senza test".

- [x] **Step 1: Test (atteso già verde, chiude il gap di copertura)**

Aggiungere a `tests/webllm-adapter.test.ts`:
```typescript
it("dispose unloads the engine; generate after dispose throws not loaded", async () => {
  let unloaded = false;
  const engine = {
    chat: {
      completions: {
        create: async function* () {
          yield { choices: [{ delta: { content: "x" } }], usage: null };
        },
      },
    },
    unload: async () => {
      unloaded = true;
    },
  };
  const a = new WebLLMAdapter({ engineFactory: async () => engine as never, hasCache: async () => false });
  await a.load("test-model", () => {});
  await a.dispose();
  expect(unloaded).toBe(true);
  await expect(a.generate({ prompt: "x", maxTokens: 1 })).rejects.toThrow("not loaded");
});
```

- [x] **Step 2: Run → PASS** (`npm test` — nessuna modifica a `src/adapters/webllm.ts` necessaria; se fallisce, il gap era reale e va corretto `dispose()` prima di procedere)

- [x] **Step 3: Commit**

```bash
git add tests/webllm-adapter.test.ts
git commit -m "test: cover WebLLMAdapter.dispose() unload + post-dispose generate (FIX-IN-1B)"
```

---

### Task 4: FIX — disabilitare "Export" durante un bench in corso

**Files:**
- Modify: `src/main.ts`

**Interfaces:**
- Nessuna nuova interfaccia: solo wiring DOM.

Riferimento: `.superpowers/sdd/progress.md` Task 8 minor — "export non ri-disabilitato durante bench". Da run 2 in poi, il bottone Export resta abilitato (da un run precedente) mentre un nuovo bench è in volo: cliccarlo esporta il run vecchio, non l'errore in sé ma una fonte di confusione (l'export sembra "fresco" ma non lo è).

Nessun test dedicato (stesso motivo del Task 2: `main.ts` non ha copertura unitaria). Verifica manuale allo Step 2.

- [x] **Step 1: Implementazione**

In `src/main.ts`, nel listener del bottone "run":
```typescript
$("run").addEventListener("click", () => {
  const sel = $("model") as HTMLSelectElement;
  const quant = sel.selectedOptions[0].dataset.quant ?? "unknown";
  ($("run") as HTMLButtonElement).disabled = true;
  ($("export") as HTMLButtonElement).disabled = true;
  send({ type: "bench", modelId: sel.value, quant });
});
```
(Il ramo `bench:result` già in `main.ts` riabilita `export` a fine bench; il ramo `error` va esteso a fare lo stesso, altrimenti un fallimento lascia Export bloccato anche se un run precedente aveva prodotto dati esportabili):
```typescript
} else if (m.type === "error") {
  $("status").textContent = `ERROR: ${m.message}`;
  ($("run") as HTMLButtonElement).disabled = false;
  ($("export") as HTMLButtonElement).disabled = run === null || run.cells.length === 0;
}
```

- [x] **Step 2: Verifica manuale**

Run: `npm run dev` → aprire la pagina, eseguire un bench con successo (Export si abilita), rilanciare "Run bench" → Expected: Export si disabilita immediatamente al click e si riabilita solo a bench concluso (successo o errore, quest'ultimo solo se esiste già un run con celle).

- [x] **Step 3: Verifica compilazione**

Run: `npx tsc --noEmit` → Expected: 0 errori.

- [x] **Step 4: Commit**

```bash
git add src/main.ts
git commit -m "fix: disable Export while a bench run is in flight (FIX-IN-1B)"
```

---

### Task 5: Schema v2 — `DeviceProbe.{browser,features,anomalies}` + probe esteso

**Files:**
- Modify: `src/schema.ts`
- Modify: `src/probe.ts`
- Modify: `tests/probe.test.ts`
- Modify: `tests/schema.test.ts`
- Modify: `tests/benchServer.test.ts` (fixture `fakeProbe` aggiornata)

**Interfaces:**
- Produces (schema.ts): `SCHEMA_VERSION = 2`; `DeviceProbe` guadagna `browser: { name: string; version: string }`, `features: string[]`, `anomalies: string[]` (tutti campi richiesti, non opzionali).
- Produces (probe.ts): `parseBrowser(userAgent: string): { name: string; version: string }`; `probeWebGPU` invariata nella firma, popola i tre campi nuovi.
- Consumes: nessuna dipendenza da Task 1–4.

Riferimento: HANDOFF §1 ("probe esteso con `adapter.features` + rilevazione software-adapter... + campi `browser`/`anomalies` nello schema (v2)") + README "Cosa abbiamo verificato dal vivo" (soglia 128 MiB confermata dal caso Firefox 152/llvmpipe) + `.superpowers/sdd/progress.md` Task 9 ("Gap probe: non registra adapter.features (fix in 1b)").

- [x] **Step 1: Test fallenti — schema**

Aggiornare la fixture `probe` in `tests/schema.test.ts`:
```typescript
const probe: DeviceProbe = {
  webgpu: true,
  adapterInfo: { vendor: "nvidia", architecture: "", device: "", description: "" },
  limits: { maxStorageBufferBindingSize: 2147483644 },
  features: ["shader-f16"],
  userAgent: "test-ua",
  deviceMemoryGB: 8,
  browser: { name: "chrome", version: "150.0" },
  anomalies: [],
};
```
(il resto del file invariato in questo task; il tipo `BenchCell`/`gen` cambia solo nel Task 6)

- [x] **Step 2: Run → FAIL** (`npm test` — `DeviceProbe` non ha ancora `browser`/`features`/`anomalies`, errore di tipo)

- [x] **Step 3: Implementazione — `schema.ts`**

In `src/schema.ts`, sostituire:
```typescript
export const SCHEMA_VERSION = 1 as const;
```
con:
```typescript
export const SCHEMA_VERSION = 2 as const;
```
e sostituire `DeviceProbe`:
```typescript
export interface DeviceProbe {
  webgpu: boolean;
  adapterInfo: { vendor: string; architecture: string; device: string; description: string } | null;
  limits: Record<string, number> | null;
  features: string[]; // v2: GPUSupportedFeatures enumerate (es. "shader-f16")
  userAgent: string;
  deviceMemoryGB: number | null;
  browser: { name: string; version: string }; // v2: parsed da userAgent
  anomalies: string[]; // v2: es. rilevazione software-adapter
}
```
(`BenchCell`/`RunFile`/`newRunFile`/`addCell` invariati in questo task)

- [x] **Step 4: Test fallenti — probe**

Sostituire `tests/probe.test.ts` con:
```typescript
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
```

- [x] **Step 5: Run → FAIL** (`npm test` — `parseBrowser` non esiste, `probeWebGPU` non popola i campi nuovi)

- [x] **Step 6: Implementazione — `probe.ts`**

Sostituire `src/probe.ts` con:
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
```

- [x] **Step 7: Run → PASS** (`npm test`)

- [x] **Step 8: Aggiornare la fixture `fakeProbe` in `tests/benchServer.test.ts`**

```typescript
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
```

- [x] **Step 9: Run → PASS** (`npm test` — l'intera suite, incluso `benchServer.test.ts`)

- [x] **Step 10: Verifica compilazione**

Run: `npx tsc --noEmit` → Expected: 0 errori.

- [x] **Step 11: Commit**

```bash
git add src/schema.ts src/probe.ts tests/probe.test.ts tests/schema.test.ts tests/benchServer.test.ts
git commit -m "feat: schema v2 (browser/features/anomalies) + software-adapter detection in probe"
```

---

### Task 6: Repliche multiple per cella — aggregazione statistica

**Files:**
- Modify: `src/metrics.ts`
- Modify: `src/schema.ts`
- Modify: `src/benchServer.ts`
- Modify: `tests/schema.test.ts`
- Modify: `tests/benchServer.test.ts`

**Interfaces:**
- Produces (metrics.ts): `interface MetricAggregate { mean: number; stdev: number; samples: number[] }`; `interface GenMetricsAgg { ttftMs: MetricAggregate; decodeToksPerSec: MetricAggregate | null; totalMs: MetricAggregate; promptTokens: number | null; completionTokens: number | null }`; `aggregateReplicates(reps: GenMetrics[]): GenMetricsAgg`.
- Produces (schema.ts): `BenchCell.gen: GenMetricsAgg` (era `GenMetrics`); `BenchCell.replicates: GenMetrics[]` (le repliche grezze); `BenchCell.anomalies: string[]` (es. `"high-variance"`).
- Produces (benchServer.ts): `BenchServer` accetta `deps.replicateCount?: number` (default 3); esegue `generate()` N volte riusando lo stesso adapter/engine caricato una volta sola.
- Consumes: `GenMetrics`/`computeGenMetrics` (metrics.ts, invariati); `DeviceProbe` (Task 5).

Riferimento: HANDOFF §1 ("repliche multiple per cella") + §3 ("Varianza tok/s run-to-run ~10% → 1b deve introdurre repliche multiple per cella") + README ("varianza run-to-run (~10-25%) conferma la necessità di repliche multiple in 1b").

- [x] **Step 1: Test fallente — `aggregateReplicates`**

Aggiungere a `tests/metrics.test.ts`:
```typescript
import { aggregateReplicates } from "../src/metrics";

describe("aggregateReplicates", () => {
  it("computes mean/stdev over replicate GenMetrics", () => {
    const reps = [
      { ttftMs: 100, decodeToksPerSec: 40, totalMs: 6000, promptTokens: 512, completionTokens: 256 },
      { ttftMs: 110, decodeToksPerSec: 44, totalMs: 6100, promptTokens: 512, completionTokens: 256 },
      { ttftMs: 90, decodeToksPerSec: 42, totalMs: 5900, promptTokens: 512, completionTokens: 256 },
    ];
    const agg = aggregateReplicates(reps);
    expect(agg.ttftMs.mean).toBeCloseTo(100, 5);
    expect(agg.decodeToksPerSec?.mean).toBeCloseTo(42, 5);
    expect(agg.decodeToksPerSec?.samples).toEqual([40, 44, 42]);
    expect(agg.promptTokens).toBe(512);
    expect(agg.completionTokens).toBe(256);
  });

  it("returns decodeToksPerSec: null when every replicate rate is null", () => {
    const reps = [
      { ttftMs: 50, decodeToksPerSec: null, totalMs: 50, promptTokens: null, completionTokens: 1 },
      { ttftMs: 55, decodeToksPerSec: null, totalMs: 55, promptTokens: null, completionTokens: 1 },
    ];
    const agg = aggregateReplicates(reps);
    expect(agg.decodeToksPerSec).toBeNull();
  });

  it("throws on empty replicate list", () => {
    expect(() => aggregateReplicates([])).toThrow();
  });
});
```

- [x] **Step 2: Run → FAIL**

- [x] **Step 3: Implementazione — `metrics.ts`**

Aggiungere in coda a `src/metrics.ts` (dopo `computeGenMetrics`):
```typescript
export interface MetricAggregate {
  mean: number;
  stdev: number;
  samples: number[];
}

export interface GenMetricsAgg {
  ttftMs: MetricAggregate;
  decodeToksPerSec: MetricAggregate | null;
  totalMs: MetricAggregate;
  promptTokens: number | null;
  completionTokens: number | null;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function stdev(xs: number[], m: number): number {
  if (xs.length < 2) return 0;
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
}

function aggregate(xs: number[]): MetricAggregate {
  const m = mean(xs);
  return { mean: m, stdev: stdev(xs, m), samples: xs };
}

export function aggregateReplicates(reps: GenMetrics[]): GenMetricsAgg {
  if (reps.length === 0) throw new Error("aggregateReplicates: empty replicate list");
  const rates = reps.map((r) => r.decodeToksPerSec).filter((r): r is number => r !== null);
  return {
    ttftMs: aggregate(reps.map((r) => r.ttftMs)),
    decodeToksPerSec: rates.length > 0 ? aggregate(rates) : null,
    totalMs: aggregate(reps.map((r) => r.totalMs)),
    promptTokens: reps[0].promptTokens,
    completionTokens: reps[0].completionTokens,
  };
}
```

- [x] **Step 4: Run → PASS** (`npm test`)

- [x] **Step 5: Aggiornare `schema.ts`**

In `src/schema.ts`, import in testa:
```typescript
import type { GenMetricsAgg } from "./metrics";
```
Sostituire `BenchCell`:
```typescript
export interface BenchCell {
  stack: "webllm"; // union estesa nel piano "1b — matrice"
  modelId: string;
  quant: string;
  promptId: string;
  load: LoadReport;
  gen: GenMetricsAgg;
  replicates: GenMetrics[];
  anomalies: string[]; // es. "high-variance"
}
```

- [x] **Step 6: Test fallente — fixture `schema.test.ts`**

Sostituire la fixture `cell` in `tests/schema.test.ts`:
```typescript
const cell: BenchCell = {
  stack: "webllm",
  modelId: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
  quant: "q4f16_1",
  promptId: "bench-512-v1",
  load: { loadMs: 1234, cacheState: "cold" },
  gen: {
    ttftMs: { mean: 100, stdev: 5, samples: [95, 100, 105] },
    decodeToksPerSec: { mean: 42.5, stdev: 1.2, samples: [41, 42.5, 44] },
    totalMs: { mean: 6000, stdev: 100, samples: [5900, 6000, 6100] },
    promptTokens: 512,
    completionTokens: 256,
  },
  replicates: [
    { ttftMs: 95, decodeToksPerSec: 41, totalMs: 5900, promptTokens: 512, completionTokens: 256 },
    { ttftMs: 100, decodeToksPerSec: 42.5, totalMs: 6000, promptTokens: 512, completionTokens: 256 },
    { ttftMs: 105, decodeToksPerSec: 44, totalMs: 6100, promptTokens: 512, completionTokens: 256 },
  ],
  anomalies: [],
};
```
e l'assertion che leggeva il vecchio shape:
```typescript
expect(run2.cells[0].gen.decodeToksPerSec?.mean).toBe(42.5);
```
(`GenMetrics` resta definita in `schema.ts` come in 1a — nessun nuovo import necessario in `tests/schema.test.ts` per tipizzare `replicates`)

- [x] **Step 7: Run → FAIL poi PASS** dopo l'aggiustamento fixture (`npm test`)

- [x] **Step 8: Wiring `benchServer.ts` — repliche + flag high-variance**

Sostituire `src/benchServer.ts`:
```typescript
import { isMainToWorker, type WorkerToMain } from "./protocol";
import type { InferenceAdapter } from "./adapters/types";
import type { DeviceProbe, BenchCell } from "./schema";
import { computeGenMetrics, aggregateReplicates } from "./metrics";
import { PROMPT_512 } from "./promptset";

const DEFAULT_REPLICATE_COUNT = 3;
const HIGH_VARIANCE_THRESHOLD = 0.15; // stdev/mean sul tok/s aggregato

export class BenchServer {
  private deps: {
    adapterFactory: () => InferenceAdapter;
    probe: () => Promise<DeviceProbe>;
    post: (m: WorkerToMain) => void;
    replicateCount?: number;
  };

  constructor(deps: {
    adapterFactory: () => InferenceAdapter;
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
        const adapter = this.deps.adapterFactory();
        const replicateCount = this.deps.replicateCount ?? DEFAULT_REPLICATE_COUNT;
        try {
          const load = await adapter.load(msg.modelId, (text, progress) =>
            this.deps.post({ type: "progress", text, progress }),
          );
          const replicates = [];
          for (let i = 0; i < replicateCount; i++) {
            this.deps.post({ type: "progress", text: `generating (replica ${i + 1}/${replicateCount})…`, progress: (i + 1) / replicateCount });
            const timeline = await adapter.generate({ prompt: PROMPT_512.text, maxTokens: 256 });
            replicates.push(computeGenMetrics(timeline));
          }
          const gen = aggregateReplicates(replicates);
          const anomalies: string[] = [];
          if (gen.decodeToksPerSec && gen.decodeToksPerSec.mean > 0 && gen.decodeToksPerSec.stdev / gen.decodeToksPerSec.mean > HIGH_VARIANCE_THRESHOLD) {
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

- [x] **Step 9: Test fallenti/aggiornati — `benchServer.test.ts`**

Sostituire il test "bench message → progress then bench:result with metrics" in `tests/benchServer.test.ts` (usare `replicateCount: 2` per velocità):
```typescript
it("bench message → progress then bench:result with aggregated metrics over replicates", async () => {
  const out: WorkerToMain[] = [];
  const s = new BenchServer({ adapterFactory: fakeAdapter, probe: fakeProbe, post: (m) => out.push(m), replicateCount: 2 });
  await s.handle({ type: "bench", modelId: "test-model", quant: "q4f16_1" });
  const types = out.map((m) => m.type);
  expect(types.filter((t) => t === "progress").length).toBeGreaterThanOrEqual(2);
  const result = out.find((m) => m.type === "bench:result");
  expect(result).toBeDefined();
  if (result?.type === "bench:result") {
    expect(result.cell.modelId).toBe("test-model");
    expect(result.cell.replicates.length).toBe(2);
    expect(result.cell.gen.ttftMs.mean).toBe(10);
    expect(result.cell.load.cacheState).toBe("cold");
    expect(result.cell.promptId).toBe("bench-512-v1");
    expect(result.cell.anomalies).toEqual([]);
  }
});

it("flags high-variance when decode rate spreads across replicates beyond threshold", async () => {
  let call = 0;
  const varyingAdapter: InferenceAdapter = {
    ...fakeAdapter(),
    generate: async () => {
      call++;
      // completionTokens=5, span 100ms vs 25ms → decode rate 40 tok/s vs 160 tok/s: stdev/mean ben oltre 0.15
      const chunks = call === 1 ? [0, 100] : [0, 25];
      return { tRequestStart: 0, chunkTimestamps: chunks, promptTokens: 512, completionTokens: 5 };
    },
  };
  const out: WorkerToMain[] = [];
  const s = new BenchServer({ adapterFactory: () => varyingAdapter, probe: fakeProbe, post: (m) => out.push(m), replicateCount: 2 });
  await s.handle({ type: "bench", modelId: "m", quant: "q" });
  const result = out.find((m) => m.type === "bench:result");
  if (result?.type === "bench:result") {
    expect(result.cell.anomalies.some((a) => a.includes("high-variance"))).toBe(true);
  } else {
    throw new Error("expected bench:result");
  }
});
```

- [x] **Step 10: Run → PASS** (`npm test`)

- [x] **Step 11: Verifica compilazione**

Run: `npx tsc --noEmit` → Expected: 0 errori.

- [x] **Step 12: Commit**

```bash
git add src/metrics.ts src/schema.ts src/benchServer.ts tests/metrics.test.ts tests/schema.test.ts tests/benchServer.test.ts
git commit -m "feat: multi-replicate bench cells with mean/stdev aggregation + high-variance anomaly flag"
```

---

### Task 7: Render + UI — aggregati, browser, anomalie

**Files:**
- Modify: `src/render.ts`
- Modify: `tests/render.test.ts`
- Modify: `src/main.ts` (probe-box mostra `browser`/`anomalies`)

**Interfaces:**
- Consumes: `RunFile`/`BenchCell` (Task 6 — `gen` è ora `GenMetricsAgg`), `DeviceProbe.{browser,anomalies}` (Task 5).
- Produces: `renderResultsTable` invariata nella firma, output aggiornato (colonna tok/s mostra `mean ± stdev`; colonna anomalie).

- [ ] **Step 1: Test fallente**

Sostituire `tests/render.test.ts` con:
```typescript
import { describe, it, expect } from "vitest";
import { renderResultsTable } from "../src/render";
import { newRunFile, addCell } from "../src/schema";

const probe = {
  webgpu: true,
  adapterInfo: null,
  limits: null,
  features: [],
  userAgent: "ua",
  deviceMemoryGB: null,
  browser: { name: "chrome", version: "150.0" },
  anomalies: [],
};

function cellWith(overrides: { modelId?: string; anomalies?: string[] } = {}) {
  return {
    stack: "webllm" as const,
    modelId: overrides.modelId ?? "M",
    quant: "q4f16_1",
    promptId: "bench-512-v1",
    load: { loadMs: 1500, cacheState: "cold" as const },
    gen: {
      ttftMs: { mean: 123.4, stdev: 3, samples: [120, 123.4, 127] },
      decodeToksPerSec: { mean: 41.7, stdev: 2.1, samples: [40, 41.7, 43.4] },
      totalMs: { mean: 6000, stdev: 50, samples: [5950, 6000, 6050] },
      promptTokens: 512,
      completionTokens: 256,
    },
    replicates: [],
    anomalies: overrides.anomalies ?? [],
  };
}

describe("renderResultsTable", () => {
  it("renders one row per cell with mean±stdev tok/s", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, cellWith());
    const html = renderResultsTable(run);
    expect(html).toContain("M");
    expect(html).toContain("41.7");
    expect(html).toContain("±2.1");
    expect(html).toContain("cold");
  });

  it("renders anomaly badges when present", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, cellWith({ anomalies: ["high-variance: stdev/mean=0.20 > 0.15"] }));
    const html = renderResultsTable(run);
    expect(html).toContain("high-variance");
  });

  it("renders empty state", () => {
    const html = renderResultsTable(newRunFile("x", probe, "2026-07-25T12:00:00Z"));
    expect(html).toContain("Nessun risultato");
  });

  it("escapes html-unsafe characters in modelId", () => {
    let run = newRunFile("4090-linux", probe, "2026-07-25T12:00:00Z");
    run = addCell(run, cellWith({ modelId: "<img src=x onerror=alert(1)>" }));
    const html = renderResultsTable(run);
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("&lt;img src=x onerror=alert(1)&gt;");
  });
});
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implementazione — `render.ts`**

```typescript
import type { RunFile } from "./schema";

const fmt = (n: number, digits = 1) => n.toFixed(digits);

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtRate(agg: { mean: number; stdev: number } | null): string {
  return agg === null ? "—" : `${fmt(agg.mean)} ±${fmt(agg.stdev)}`;
}

export function renderResultsTable(run: RunFile): string {
  if (run.cells.length === 0) return "<p>Nessun risultato ancora.</p>";
  const rows = run.cells
    .map(
      (c) => `<tr>
        <td>${escapeHtml(c.stack)}</td><td>${escapeHtml(c.modelId)}</td><td>${escapeHtml(c.quant)}</td>
        <td>${fmt(c.load.loadMs, 0)}</td><td>${c.load.cacheState}</td>
        <td>${fmt(c.gen.ttftMs.mean, 0)}</td><td>${fmtRate(c.gen.decodeToksPerSec)}</td>
        <td>${c.gen.promptTokens ?? "—"}/${c.gen.completionTokens ?? "—"}</td>
        <td>${c.anomalies.map((a) => `<span class="anomaly">${escapeHtml(a)}</span>`).join(" ")}</td>
      </tr>`,
    )
    .join("");
  return `<table>
    <thead><tr><th>stack</th><th>model</th><th>quant</th><th>load ms</th><th>cache</th><th>TTFT ms</th><th>tok/s (mean±stdev)</th><th>tok in/out</th><th>anomalie</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}
```

- [ ] **Step 4: Run → PASS** (`npm test`)

- [ ] **Step 5: `main.ts` — mostrare browser/anomalie nel probe-box**

In `src/main.ts`, il ramo `probe:result` già serializza l'intero oggetto `m.probe` via `JSON.stringify` — `browser`/`features`/`anomalies` compaiono automaticamente senza modifiche di codice (il probe box mostra il JSON raw). Nessuno step di codice qui: verificare manualmente allo Step 6.

- [ ] **Step 6: Verifica manuale**

Run: `npm run dev` → aprire la pagina → Expected: il probe-box mostra `browser: { name, version }` e `anomalies: []` (o popolato, se il browser di test è in fallback software) nel JSON.

- [ ] **Step 7: Verifica compilazione**

Run: `npx tsc --noEmit` → Expected: 0 errori.

- [ ] **Step 8: Commit**

```bash
git add src/render.ts tests/render.test.ts
git commit -m "feat: render mean±stdev tok/s and anomaly badges in results table"
```

---

### Task 8: README — schema v2, repliche, euristica software-adapter

**Files:**
- Modify: `README.md`

Questo task è documentazione pura (docs procedure): nessun test, verifica per lettura.

- [ ] **Step 1: Aggiornare README**

Aggiungere una sezione dopo "Cosa abbiamo verificato dal vivo" (prima di "## Note"):
```markdown
## Fase 1b — fondamenta (schema v2)

- **Schema v2** (`SCHEMA_VERSION = 2`, non retro-compatibile con i file v1 in `results/`,
  che restano storici): `DeviceProbe` guadagna `browser` (nome/versione parsati dalla UA),
  `features` (elenco `adapter.features`, es. `shader-f16`), `anomalies` (flag rilevati dal probe).
  `BenchCell` guadagna `replicates` (le repliche grezze) e `anomalies` (flag per-cella);
  `gen` non è più una singola misura ma un aggregato `{ mean, stdev, samples }` per metrica.
- **Repliche multiple**: ogni cella esegue 3 `generate()` sullo stesso modello già caricato
  (nessun ricaricamento tra repliche) e aggrega tok/s, TTFT, tempo totale con media e
  deviazione standard. Una cella con `stdev/mean > 0.15` sul tok/s riceve l'anomalia
  `high-variance` — risponde al finding "varianza run-to-run ~10-25%" osservato in 1a.
- **Rilevazione software-adapter**: il probe marca `anomalies: ["software-adapter: ..."]`
  quando l'adapter dichiara `vendor` vuoto **e** `maxBufferSize <= 128 MiB` — la firma
  osservata su Firefox 152 in silent-fallback a llvmpipe (vedi sopra). Un run con questa
  anomalia è un datapoint CPU, non GPU: va escluso da confronti tok/s cross-device.
- **Fuori scope qui** (piano successivo "1b — matrice"): adapter Transformers.js/wllama,
  sweep sui 3 device, modulo qualità-leggera.
```
Aggiornare anche la riga "**Export JSON** ... schema v1" in "## Quick start" in "schema v2".

- [ ] **Step 2: Verifica per lettura**

Rileggere il README aggiornato end-to-end → Expected: nessuna contraddizione con `docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md` §Fasatura né con `HANDOFF.md`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document schema v2, replicate methodology, software-adapter heuristic"
```

---

## Self-review (fatta in scrittura)

1. **Copertura HANDOFF §1 "primo passo concreto"**: FIX-IN-1B (5 item) ✓ Task 1–4 (il 5° item, "probe features", ✓ Task 5); probe esteso con `adapter.features` ✓ Task 5; rilevazione software-adapter ✓ Task 5; repliche multiple per cella ✓ Task 6; campi `browser`/`anomalies` in schema v2 ✓ Task 5–6. Dichiarato fuori scope: adapter Transformers.js/wllama, sweep multi-device, modulo qualità — riservati al piano "1b — matrice" successivo.
2. **Placeholder**: nessun TBD; ogni step ha codice completo o comando+esito atteso.
3. **Coerenza tipi**: `GenMetricsAgg` (metrics.ts, Task 6) ↔ `BenchCell.gen` (schema.ts, Task 6) ↔ `render.ts` (Task 7, `fmtRate`) — stessi nomi di campo (`mean`, `stdev`, `samples`) in ogni task che li usa. `DeviceProbe.{browser,features,anomalies}` (Task 5) ↔ fixture `probe.test.ts`/`schema.test.ts`/`benchServer.test.ts` (stesso task, nessun task successivo li ridefinisce).
4. **Ordine delle dipendenze**: Task 1–4 sono indipendenti tra loro e possono eseguire in qualsiasi ordine (o in parallelo, se subagent-driven). Task 5 precede Task 6 (Task 6 estende `DeviceProbe`-adiacenti solo indirettamente via fixture, ma soprattutto introduce `GenMetricsAgg` che Task 7 consuma). Task 7 dipende da Task 5+6. Task 8 è indipendente da codice, va per ultimo per documentare lo stato finale.
5. **Rischio non coperto da test automatico**: Task 2 (exhaustiveness) e Task 4 (export disable) restano verificati solo da `tsc`/manuale, coerente con l'assenza di test unitari su `main.ts` già in 1a (wiring DOM, non logica pura) — nessuna nuova incoerenza introdotta rispetto al pattern esistente.
