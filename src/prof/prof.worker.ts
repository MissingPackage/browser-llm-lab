// Worker del dispatch profiler (prof.html): identico a bench.worker.ts nel percorso
// misurato — stesso BenchServer, stessi adapter — con in più il patch dei prototype
// WebGPU installato PRIMA di costruire il server, e un sampler dei contatori avviato
// al primo messaggio "bench". Il patch è a livello di prototype: conta le chiamate di
// qualunque device creato dopo (e anche prima) dell'installazione.
import { BenchServer } from "../benchServer";
import { WebLLMAdapter } from "../adapters/webllm";
import { TransformersJsAdapter } from "../adapters/transformersjs";
import { WllamaAdapter } from "../adapters/wllama";
import { probeWebGPU } from "../probe";
import type { WorkerToMain } from "../protocol";
import {
  newCounters, installGpuProfPatch, measureClockQuantum,
  type GpuGlobals, type ProfCounters,
} from "./profiler";
import type { ProfSample } from "./profSchema";

export const SAMPLE_MS = 250;

export type ProfWorkerOut =
  | WorkerToMain
  | {
      type: "prof:data";
      totals: ProfCounters;
      samples: ProfSample[];
      clockMinDeltaMs: number | null;
      sampleMs: number;
      missingApis: string[];
    };

const now = (): number => performance.now();

// I global WebGPU raccolti esplicitamente: su browser senza WebGPU (o contesti
// non-secure) i costruttori non esistono e `typeof` è l'unico accesso sicuro.
const gpuGlobals: GpuGlobals = {
  GPUComputePassEncoder: typeof GPUComputePassEncoder !== "undefined" ? GPUComputePassEncoder : undefined,
  GPUDevice: typeof GPUDevice !== "undefined" ? GPUDevice : undefined,
  GPUCommandEncoder: typeof GPUCommandEncoder !== "undefined" ? GPUCommandEncoder : undefined,
  GPUQueue: typeof GPUQueue !== "undefined" ? GPUQueue : undefined,
  GPUBuffer: typeof GPUBuffer !== "undefined" ? GPUBuffer : undefined,
};

const counters = newCounters();
const missingApis = installGpuProfPatch(gpuGlobals, counters, now);
const clockMinDeltaMs = measureClockQuantum(now);

let phase = "idle";
const samples: ProfSample[] = [];
let samplerStarted = false;
function startSampler(): void {
  if (samplerStarted) return;
  samplerStarted = true;
  const t0 = now();
  // Il sampler respira fra un await e l'altro del decode (stesso vincolo — e stessa
  // evidenza di funzionamento — del campionamento esterno del tool Playwright).
  setInterval(() => {
    samples.push({ tMs: now() - t0, phase, ...counters });
  }, SAMPLE_MS);
}

const server = new BenchServer({
  adapters: {
    webllm: () => new WebLLMAdapter(),
    transformersjs: () => new TransformersJsAdapter(),
    wllama: () => new WllamaAdapter(),
  },
  probe: () => probeWebGPU(navigator.gpu, navigator as { userAgent: string; deviceMemory?: number }),
  post: (m: WorkerToMain) => {
    // La fase tagga i campioni: l'analisi offline individua la finestra di decode
    // dai progress "generating (replica …)".
    if (m.type === "progress") phase = m.text;
    else if (m.type === "bench:result") phase = "done";
    else if (m.type === "error") phase = "error";
    self.postMessage(m);
  },
});

self.onmessage = (e: MessageEvent) => {
  const d = e.data as { type?: unknown };
  if (d?.type === "prof:dump") {
    const out: ProfWorkerOut = {
      type: "prof:data",
      totals: { ...counters },
      samples,
      clockMinDeltaMs,
      sampleMs: SAMPLE_MS,
      missingApis,
    };
    self.postMessage(out);
    return;
  }
  if (d?.type === "bench") startSampler();
  void server.handle(e.data);
};
