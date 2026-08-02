// Limiti del device DERIVATI dai consumatori reali (goal C3a fase 3).
//
// PERCHÉ ESISTE QUESTO FILE, E PERCHÉ È ALLA SECONDA STESURA.
// Il motore ha sbagliato due volte, nello stesso modo:
//   1. all'inizio chiedeva costanti difensive (`min(lim.X, 2 GiB)`, `32768`)
//      scelte senza un consumatore dichiarato ⇒ nessuno poteva più distinguere
//      un requisito da una supposizione, e i cap sono stati rivisti a mano;
//   2. la prima versione di questo file le ha sostituite col MASSIMO
//      dell'adapter ⇒ stesso identico difetto (nessun consumatore dichiarato),
//      solo nell'altra direzione: si chiedevano 1024 invocazioni per workgroup
//      mentre il kernel più largo del repo ne usa 256.
// La spec WebGPU §3.6.2 avverte che chiedere limiti migliori del necessario
// *può* costare prestazioni: "applications should generally only request limits
// better than the defaults if they may actually require them".
//
// REGOLA ORA: ogni limite si chiede come `min(adapter, requisito_derivato)`.
// L'adapter è il TETTO, non il target. Ogni requisito porta con sé il suo
// consumatore, e `tests/gpulimits.test.ts` ri-deriva le costanti scansionando
// il WGSL vero: se qualcuno aggiunge un kernel `workgroup_size(512)` senza
// aggiornare la derivazione, il test cade. È l'unica cosa che tiene insieme il
// limite e il codice che lo consuma — finora vivevano in file diversi senza
// niente in mezzo, ed è per questo che nessuno dei due si accorgeva dell'altro.

// ---------------------------------------------------------------------------
// Costanti derivate dall'inventario del parco kernel (C3a it.6).
// Ognuna è ri-verificata dal test scansionando src/engine/kernels/wgsl.ts.
// ---------------------------------------------------------------------------

/** Il `workgroup_size` più grande del parco: rmsnormWgsl e argmaxStage1/2. */
export const MAX_WORKGROUP_SIZE = 256;

/** Storage binding nel bind group più affollato: rmsPairGemvSiluFast (Qwen). */
export const MAX_STORAGE_BINDINGS_PER_STAGE = 7;

/**
 * Workgroup storage del path Qwen FUSO, indipendente dal contesto:
 * rmsPairGemmSiluChunkFast con K = dModel 896 e mMax = PREFILL_M 8
 *   4·K·mMax + 256·mMax + 16·mMax = 30 848 B.
 * NOTA: il commento storico in gpuforward.ts cita 19,7 KB (il down-proj), ma
 * il consumatore massimo è questo — i 32 768 richiesti a mano lasciavano
 * appena 1 920 B di margine senza che nessuno lo sapesse.
 */
export const QWEN_WORKGROUP_STORAGE_BYTES = 30_848;

/** mlaAttnDecode: scores[ctxMax] + red[64]. È il `wgNeed` del fail-fast GLM. */
export const mlaWorkgroupStorageBytes = (ctxMax: number): number => 4 * ctxMax + 256;

/** KV cache GLM per layer: ctxMax × keyLen(576) × 4 B. */
export const glmKvBytesPerLayer = (ctxMax: number): number => ctxMax * 2304;

/**
 * `output.weight` Q6_K dopo repackKQuant: 210 B/blocco paddati a 53 u32 = 212.
 * È il binding singolo più grande dell'intero motore (250,5 MiB a vocab
 * 154 880) e sfonda il default di spec (128 MiB) di quasi 2×.
 */
export const q6kHeadBytes = (vocab: number, dModel: number): number =>
  Math.ceil((vocab * dModel) / 256) * Math.ceil(210 / 4) * 4;

// ---------------------------------------------------------------------------

/** Un requisito, con il consumatore che lo determina (per i messaggi d'errore). */
export interface LimitNeed {
  limit: string;
  value: number;
  consumer: string;
  /** false ⇒ è un'ottimizzazione di packing, non un requisito di correttezza. */
  hard: boolean;
}

export interface EngineNeedsOpts {
  /** contesto massimo che il modello dovrà reggere */
  ctxMax: number;
  /** vocab e dModel della testa, se il modello ha l'output head */
  head?: { vocab: number; dModel: number };
  /**
   * Byte totali che ExpertCache vorrebbe in UN buffer di classe. Non è un
   * requisito di correttezza: la cache si adatta a qualunque valore spezzando
   * in più buffer. Meno buffer però = meno binding da coprire nella fase 4.
   */
  slabClassBytes?: number;
  /**
   * Binding che non vengono dal modello di produzione ma dall'harness che lo
   * ospita (es. ktest binda i pesi densi veri di blk.0). Ognuno dichiara il
   * proprio consumatore, come tutti gli altri requisiti.
   */
  extraBindings?: Array<{ bytes: number; consumer: string }>;
}

/** I requisiti del motore, ciascuno col suo consumatore. */
export function engineNeeds(o: EngineNeedsOpts): LimitNeed[] {
  const needs: LimitNeed[] = [
    {
      limit: "maxComputeInvocationsPerWorkgroup", value: MAX_WORKGROUP_SIZE, hard: true,
      consumer: "rmsnormWgsl / argmaxStage1-2 (workgroup_size 256)",
    },
    {
      limit: "maxComputeWorkgroupSizeX", value: MAX_WORKGROUP_SIZE, hard: true,
      consumer: "rmsnormWgsl / argmaxStage1-2 (workgroup_size 256)",
    },
    {
      limit: "maxStorageBuffersPerShaderStage", value: MAX_STORAGE_BINDINGS_PER_STAGE, hard: true,
      consumer: "rmsPairGemvSiluFastWgsl / rmsPairGemmSiluChunkFastWgsl (7 storage)",
    },
    {
      limit: "maxComputeWorkgroupStorageSize",
      value: Math.max(QWEN_WORKGROUP_STORAGE_BYTES, mlaWorkgroupStorageBytes(o.ctxMax)),
      hard: true,
      consumer: `max(rmsPairGemmSiluChunkFast ${QWEN_WORKGROUP_STORAGE_BYTES} B, mlaAttnDecode 4·ctxMax+256 = ${mlaWorkgroupStorageBytes(o.ctxMax)} B a ctxMax ${o.ctxMax})`,
    },
  ];
  // Il binding singolo più grande: testa Q6_K, la KV di un layer, o un binding
  // dichiarato dall'harness.
  const candidates: Array<{ bytes: number; consumer: string }> = [
    { bytes: glmKvBytesPerLayer(o.ctxMax), consumer: `KV cache di un layer a ctxMax ${o.ctxMax}` },
    ...(o.head ? [{ bytes: q6kHeadBytes(o.head.vocab, o.head.dModel), consumer: "output.weight Q6_K repacked" }] : []),
    ...(o.extraBindings ?? []),
  ];
  const biggest = candidates.reduce((a, b) => (b.bytes > a.bytes ? b : a));
  needs.push({
    limit: "maxStorageBufferBindingSize", value: biggest.bytes, hard: true,
    consumer: `${biggest.consumer} (${biggest.bytes} B)`,
  });
  // Requisito HARD anche sul buffer: un binding di N byte vive in un buffer di
  // almeno N byte. Il packing degli slab lo può alzare, ma non è correttezza.
  const hardBuffer = biggest.bytes;
  needs.push({
    limit: "maxBufferSize", value: hardBuffer, hard: true,
    consumer: `stesso buffer del binding più grande (${hardBuffer} B)`,
  });
  if (o.slabClassBytes && o.slabClassBytes > hardBuffer) {
    needs.push({
      limit: "maxBufferSize", value: o.slabClassBytes, hard: false,
      consumer: `packing ExpertCache: una classe intera in un buffer (${o.slabClassBytes} B)`,
    });
  }
  return needs;
}

export class UnmetLimitError extends Error {
  readonly unmet: Array<LimitNeed & { available: number }>;
  constructor(unmet: Array<LimitNeed & { available: number }>) {
    super(
      "device WebGPU insufficiente:\n" +
      unmet.map((u) => `  - ${u.limit}: servono ${u.value} B, l'adapter ne offre ${u.available} — consumatore: ${u.consumer}`).join("\n"));
    this.name = "UnmetLimitError";
    this.unmet = unmet;
  }
}

/**
 * `requiredLimits` = min(adapter, requisito). Fallisce SUBITO, con il
 * consumatore nel messaggio, se un requisito HARD non è servibile: meglio qui
 * che a runtime con un errore di validazione criptico.
 * I requisiti soft (packing) vengono semplicemente troncati al disponibile.
 */
export function limitsFor(adapter: GPUAdapter, needs: LimitNeed[]): Record<string, number> {
  const avail = adapter.limits as unknown as Record<string, number>;
  const out: Record<string, number> = {};
  const unmet: Array<LimitNeed & { available: number }> = [];
  for (const n of needs) {
    const have = Number(avail[n.limit] ?? 0);
    if (n.hard && have < n.value) { unmet.push({ ...n, available: have }); continue; }
    // più requisiti sullo stesso limite: vince il maggiore servibile
    out[n.limit] = Math.max(out[n.limit] ?? 0, Math.min(have, n.value));
  }
  if (unmet.length) throw new UnmetLimitError(unmet);
  return out;
}

/** Scorciatoia: requisiti del motore + negoziazione, in un colpo. */
export function negotiateLimits(adapter: GPUAdapter, o: EngineNeedsOpts): Record<string, number> {
  return limitsFor(adapter, engineNeeds(o));
}

/** I limiti CONCESSI dal device, da mettere nel report accanto ai numeri. */
export function grantedLimits(device: GPUDevice): Record<string, number> {
  const lim = device.limits as unknown as Record<string, number>;
  const keys = [
    "maxBufferSize", "maxStorageBufferBindingSize", "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX", "maxComputeWorkgroupsPerDimension",
  ];
  const out: Record<string, number> = {};
  for (const k of keys) out[k] = Number(lim[k] ?? 0);
  return out;
}

/** Cap per il dimensionamento dei buffer slab: valori CONCESSI, non sperati. */
export function slabBufferCap(device: GPUDevice): { maxBindingBytes: number; maxBufferBytes: number } {
  return {
    maxBindingBytes: device.limits.maxStorageBufferBindingSize,
    maxBufferBytes: device.limits.maxBufferSize,
  };
}
