// Punto UNICO di creazione del device WebGPU (goal C3a fase 4d, emendamento 6).
//
// PERCHÉ ESISTE. Prima della fase 4d il repo aveva OTTO siti che chiamavano
// `requestAdapter`/`requestDevice`, e si erano già divisi in tre famiglie:
//   1. limiti negoziati + listener `uncapturederror` (gpuforward): corretto;
//   2. limiti negoziati SENZA listener (glmroute, glmconf, glmbench, ktest):
//      ogni errore di validazione WebGPU è silenzioso by default e trasforma
//      il submit in un no-op — la landmine vista dal vivo al bring-up
//      (grid > 65535 ⇒ top-1 0.2% muto);
//   3. limiti ad-hoc con o senza listener (engine.worker diag, microbench):
//      il difetto che gpulimits.ts documenta come già commesso due volte.
// La deriva non è un incidente: senza un punto unico OGNI nuovo harness
// ricomincia da capo e sceglie la sua famiglia. Questo modulo è il punto
// unico; `tests/gpudevice.test.ts` vieta per costruzione ogni altro sito
// (scansione del sorgente: `requestDevice` fuori da qui = test rosso).
//
// FUORI SCOPE, di proposito: gli script probe (`scripts/vram-ceiling.mjs`,
// `scripts/webgpu-limits.mjs`) misurano il device GREZZO — tetto allocabile
// e limiti dell'adapter. La negoziazione falserebbe la misura: restano
// creatori diretti, in page-context, fuori da src/.

import { negotiateLimits, type EngineNeedsOpts } from "./gpulimits";

export interface EngineDeviceOpts {
  /** Etichetta dell'harness nei messaggi d'errore: "[glmbench][gpu-error] …". */
  label: string;
  /**
   * Requisiti dei consumatori (gpulimits). Funzione quando i requisiti
   * dipendono dall'adapter (es. `arenaNeeds` legge maxBufferSize/binding
   * per dimensionare la finestra d'arena).
   */
  needs: EngineNeedsOpts | ((adapter: GPUAdapter) => EngineNeedsOpts);
  /**
   * Feature chieste SOLO se l'adapter le espone (timestamp-query,
   * shader-f16): il chiamante controlla poi `has()` e degrada dichiarando
   * (es. glmbench: `gpuBusyMs: null` nel report).
   */
  optionalFeatures?: GPUFeatureName[];
}

export interface EngineDevice {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** La feature opzionale è stata CONCESSA? (sul device, non sull'adapter) */
  has: (f: GPUFeatureName) => boolean;
}

/**
 * adapter + limiti negoziati (`min(adapter, requisito derivato)`, gpulimits) +
 * listener `uncapturederror` che URLA invece di lasciare il no-op silenzioso.
 * `gpu` è iniettabile solo per i test (vitest non ha navigator.gpu).
 */
export async function createEngineDevice(
  opts: EngineDeviceOpts,
  gpu: GPU | undefined = typeof navigator !== "undefined" ? navigator.gpu : undefined,
): Promise<EngineDevice> {
  const adapter = await gpu?.requestAdapter();
  if (!adapter) throw new Error(`[${opts.label}] WebGPU non disponibile (requestAdapter null)`);
  const requiredFeatures = (opts.optionalFeatures ?? []).filter((f) => adapter.features.has(f));
  const needs = typeof opts.needs === "function" ? opts.needs(adapter) : opts.needs;
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: negotiateLimits(adapter, needs),
  });
  device.addEventListener("uncapturederror", (e) => {
    const msg = (e as GPUUncapturedErrorEvent).error.message;
    console.error(`[${opts.label}][gpu-error]`, msg.slice(0, 400));
    throw new Error(`GPU error [${opts.label}]: ${msg.slice(0, 200)}`);
  });
  return { adapter, device, has: (f) => device.features.has(f) };
}
