// Negoziazione dei limiti del device (goal C3a fase 3, ruling PI 2026-08-02).
//
// PERCHÉ ESISTE QUESTO FILE. Un limite esposto dall'`adapter` è solo una
// promessa: il `device` riceve il DEFAULT DI SPEC per ogni limite che non
// compare in `requiredLimits`. Il motore chiedeva 3 limiti e si portava a casa
// i default su tutti gli altri — il probe di C3a it.3
// (`results/engine/webgpu-limits-4090laptop-2026-08-02.json`) ha misurato su
// 4090 Laptop / Chrome / Linux:
//
//   limite                              adapter      device che creavamo
//   maxStorageBuffersPerShaderStage          16                        8
//   maxComputeInvocationsPerWorkgroup      1024                      256
//   maxBufferSize                     4294967292   clampato a 2 GiB dal codice
//   maxComputeWorkgroupStorageSize        49152           32768 richiesti
//
// I primi due erano capacità regalate indietro; contano per la guerra ai
// dispatch (fase 4b: il GEMV usa workgroup_size(64) con riduzione in shared
// memory, un pattern che 1024 invocazioni e le subgroup ops rendono obsoleto)
// e per il binding fisso del parco expert (fase 4: servono ~6-8 storage buffer
// per coprire il residente, e con 8 non ci si sta).
//
// `maxStorageBufferBindingSize` invece NON è nostro da alzare: su NVIDIA Dawn
// lo clampa a 2 GiB−4 per un bug driver su `OpArrayLength`. Il cap che il
// codice applicava lì era corretto e resta.
//
// REGOLA: si chiede sempre il massimo che l'adapter concede, mai una costante
// scritta a mano. Un device diverso concede meno e il motore si adatta invece
// di fallire — ma il valore ottenuto va SEMPRE riportato, perché una
// prestazione misurata su limiti diversi non è confrontabile (lezione B2).

export interface NegotiatedLimits {
  required: Record<string, number>;
  /** Ciò che il device ha davvero concesso, per il report. */
  granted(device: GPUDevice): Record<string, number>;
}

// I limiti che ci interessano davvero, con il motivo per cui li chiediamo.
const WANTED = [
  "maxBufferSize",                     // taglia dei buffer slab (meno buffer = meno bind group)
  "maxStorageBufferBindingSize",       // taglia del sotto-range bindato
  "maxStorageBuffersPerShaderStage",   // quanti binding di parco in un dispatch (fase 4)
  "maxComputeWorkgroupStorageSize",    // attention: scores[ctxMax] in shared memory
  "maxComputeInvocationsPerWorkgroup", // forma dei GEMV (fase 4b)
  // `maxComputeWorkgroupSizeX` deve restare accoppiato a
  // maxComputeInvocationsPerWorkgroup (vincolo di spec: X <= invocations),
  // quindi lo si chiede insieme e la coppia si valida in negotiateLimits.
  "maxComputeWorkgroupSizeX",
] as const;

// Coppie che la spec vincola: se si cappa il primo, il secondo va cappato con
// lui, altrimenti si ottiene un device con limiti incoerenti che passa la
// validazione di requestDevice e poi fallisce alla creazione della pipeline.
const COUPLED: Array<[(typeof WANTED)[number], (typeof WANTED)[number]]> = [
  ["maxComputeInvocationsPerWorkgroup", "maxComputeWorkgroupSizeX"],
  ["maxBufferSize", "maxStorageBufferBindingSize"],
];

/**
 * Costruisce `requiredLimits` chiedendo all'adapter il massimo che concede su
 * ogni limite che ci serve. `caps` permette di tenere un tetto dove serve
 * (es. buffer piu' piccoli in un test) senza reintrodurre costanti nascoste.
 */
export function negotiateLimits(
  adapter: GPUAdapter,
  caps: Partial<Record<(typeof WANTED)[number], number>> = {},
): Record<string, number> {
  const lim = adapter.limits as unknown as Record<string, number>;
  const out: Record<string, number> = {};
  for (const k of WANTED) {
    const have = Number(lim[k] ?? 0);
    if (!have) continue; // limite non esposto: si lascia il default di spec
    const cap = caps[k];
    out[k] = cap === undefined ? have : Math.min(have, cap);
  }
  // Riallinea le coppie vincolate dalla spec: il secondo non puo' superare il
  // primo (es. workgroupSizeX <= invocationsPerWorkgroup).
  for (const [a, b] of COUPLED) {
    if (out[a] !== undefined && out[b] !== undefined && out[b] > out[a]) out[b] = out[a];
  }
  return out;
}

/** I limiti CONCESSI dal device, da mettere nel report accanto ai numeri. */
export function grantedLimits(device: GPUDevice): Record<string, number> {
  const lim = device.limits as unknown as Record<string, number>;
  const out: Record<string, number> = {};
  for (const k of WANTED) out[k] = Number(lim[k] ?? 0);
  return out;
}

/**
 * Il cap che la residenza deve usare per dimensionare i buffer slab: il minore
 * fra buffer size e binding size CONCESSI (non richiesti, non sperati).
 */
export function slabBufferCap(device: GPUDevice): { maxBindingBytes: number; maxBufferBytes: number } {
  const lim = device.limits;
  return { maxBindingBytes: lim.maxStorageBufferBindingSize, maxBufferBytes: lim.maxBufferSize };
}
