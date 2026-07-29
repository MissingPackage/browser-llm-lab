// Dispatch profiler — porting in-SPA del patch di .harness/tools/dispatch-profile.mjs
// (sessione di stima 2026-07-28, docket #11: run manuali su M4/S22 senza Playwright/adb).
//
// Stessi nomi di contatore del tool: i JSON esportati dalla pagina prof.html e quelli
// scritti dal tool sul 4090 (results/dispatch-profile/) sono confrontabili campo a campo.
//
// Modulo puro: il wrapping lavora su oggetti-prototype passati dal chiamante, così i
// test unit girano in CI senza GPU (stesso principio di metrics.ts/quality.ts).

export interface ProfCounters {
  dispatch: number;
  bindGroup: number;
  computePass: number;
  writeBuffer: number;
  submit: number;
  commandEncoder: number;
  passEnd: number;
  mapAsync: number;
  onSubmittedWorkDone: number;
  tDispatch: number;
  tBindGroup: number;
  tComputePass: number;
  tWriteBuffer: number;
  tSubmit: number;
  tCommandEncoder: number;
  tPassEnd: number;
}

export type Now = () => number;

export function newCounters(): ProfCounters {
  return {
    dispatch: 0, bindGroup: 0, computePass: 0, writeBuffer: 0,
    submit: 0, commandEncoder: 0, passEnd: 0, mapAsync: 0, onSubmittedWorkDone: 0,
    tDispatch: 0, tBindGroup: 0, tComputePass: 0, tWriteBuffer: 0, tSubmit: 0,
    tCommandEncoder: 0, tPassEnd: 0,
  };
}

// Wrappa un metodo del prototype: conteggio + tempo CPU cumulato (encode lato CPU).
// Ritorna false se il metodo non esiste (WebGPU assente o parziale): non è un errore
// qui — il chiamante decide se e come segnalarlo.
export function wrapTimed(
  proto: object | undefined,
  name: string,
  c: ProfCounters,
  counterKey: keyof ProfCounters,
  timerKey: keyof ProfCounters,
  now: Now,
): boolean {
  if (!proto) return false;
  const p = proto as Record<string, unknown>;
  const orig = p[name];
  if (typeof orig !== "function") return false;
  p[name] = function (this: unknown, ...a: unknown[]) {
    const t0 = now();
    const r = (orig as (...args: unknown[]) => unknown).apply(this, a);
    c[timerKey] += now() - t0;
    c[counterKey]++;
    return r;
  };
  return true;
}

// Solo conteggio, nessun timer: per i metodi async (mapAsync, onSubmittedWorkDone)
// il tempo sincrono di invocazione non è la grandezza interessante.
export function wrapCounted(
  proto: object | undefined,
  name: string,
  c: ProfCounters,
  counterKey: keyof ProfCounters,
): boolean {
  if (!proto) return false;
  const p = proto as Record<string, unknown>;
  const orig = p[name];
  if (typeof orig !== "function") return false;
  p[name] = function (this: unknown, ...a: unknown[]) {
    c[counterKey]++;
    return (orig as (...args: unknown[]) => unknown).apply(this, a);
  };
  return true;
}

// Risoluzione del clock nel contesto corrente (Chrome coarsening — landmine nota:
// nel worker il quanto è ~5 µs). Misurata, non assunta: i timer aggregati vanno letti
// sapendo il quanto.
export function measureClockQuantum(now: Now): number | null {
  const deltas: number[] = [];
  for (let i = 0; i < 5000 && deltas.length < 50; i++) {
    const a = now();
    const b = now();
    if (b > a) deltas.push(b - a);
  }
  return deltas.length ? Math.min(...deltas) : null;
}

// Sottoinsieme dei costruttori WebGPU globali che il patch tocca. Opzionali perché
// su browser senza WebGPU (o contesti non-secure) i global non esistono affatto.
export interface GpuGlobals {
  GPUComputePassEncoder?: { prototype: object };
  GPUDevice?: { prototype: object };
  GPUCommandEncoder?: { prototype: object };
  GPUQueue?: { prototype: object };
  GPUBuffer?: { prototype: object };
}

// Installa tutte le wrap; ritorna i nomi dei metodi mancanti (diagnostica nel JSON,
// non un errore: un browser parziale produce comunque i contatori che può).
export function installGpuProfPatch(g: GpuGlobals, c: ProfCounters, now: Now): string[] {
  const missing: string[] = [];
  const timed = (
    proto: object | undefined, name: string,
    counterKey: keyof ProfCounters, timerKey: keyof ProfCounters,
  ): void => {
    if (!wrapTimed(proto, name, c, counterKey, timerKey, now)) missing.push(name);
  };
  timed(g.GPUComputePassEncoder?.prototype, "dispatchWorkgroups", "dispatch", "tDispatch");
  timed(g.GPUDevice?.prototype, "createBindGroup", "bindGroup", "tBindGroup");
  timed(g.GPUCommandEncoder?.prototype, "beginComputePass", "computePass", "tComputePass");
  timed(g.GPUQueue?.prototype, "writeBuffer", "writeBuffer", "tWriteBuffer");
  timed(g.GPUQueue?.prototype, "submit", "submit", "tSubmit");
  timed(g.GPUDevice?.prototype, "createCommandEncoder", "commandEncoder", "tCommandEncoder");
  timed(g.GPUComputePassEncoder?.prototype, "end", "passEnd", "tPassEnd");
  if (!wrapCounted(g.GPUBuffer?.prototype, "mapAsync", c, "mapAsync")) missing.push("mapAsync");
  if (!wrapCounted(g.GPUQueue?.prototype, "onSubmittedWorkDone", c, "onSubmittedWorkDone")) {
    missing.push("onSubmittedWorkDone");
  }
  return missing;
}
