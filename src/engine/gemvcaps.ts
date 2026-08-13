// Capability gate del percorso subgroup di gemv (T1-gemv-caps).
//
// Perché questo file esiste, in una riga: la feature "subgroups" NON basta.
//
// In fase 0 il microbench dei kernel di decode (src/microbench/kdRunner.ts,
// cella `vec4-rows2-sg`) ha misurato che la forma vincente mette DUE righe in
// un workgroup da 64 thread, assumendo UNA riga per subgroup da 32 lane. Quella
// mappatura riga->subgroup è un'assunzione sulla dimensione del subgroup, non
// sulla presenza della feature: se l'hardware usa subgroup da 16 o da 64 lane,
// due righe finiscono a condividere (o a spezzarsi su) subgroup sbagliati e la
// riduzione somma le lane sbagliate. Il risultato è numericamente sbagliato e
// NON solleva alcun errore: WebGPU non valida la dimensione del subgroup, il
// kernel compila e gira. È una rottura SILENZIOSA — l'unica difesa è rifiutare
// il percorso a monte, dove si legge adapterInfo.
//
// Da qui la regola: sg=true SOLO con feature presente E dimensione del subgroup
// fissa a 32 (subgroupMinSize === subgroupMaxSize === 32). Entrambi i limiti,
// non uno: un device wave32/wave64 (min 32, max 64) può schedulare subgroup da
// 64 lane pur avendo min=32, e romperebbe la mappatura esattamente allo stesso
// modo. Ogni altro caso, e in particolare "dimensione non leggibile", va a
// false: non sapere quanto vale il subgroup è indistinguibile dal saperlo
// sbagliato.
//
// Questo modulo non genera WGSL e non importa src/engine/kernels/wgsl.ts —
// serve a decidere PRIMA di generarlo, e un import qui creerebbe un ciclo.

// @webgpu/types non elenca ancora "subgroups" fra i GPUFeatureName: il cast è
// obbligatorio (stesso precedente in src/microbench/kdRunner.ts:275).
export const GEMV_SG_FEATURE = "subgroups" as GPUFeatureName;

/** La dimensione di subgroup che la forma `vec4-rows2-sg` assume (una riga per subgroup). */
export const GEMV_SG_REQUIRED_SIZE = 32;

export interface GemvCaps {
  /** true = il percorso subgroup di gemv è sicuro su questo device. */
  sg: boolean;
  /** Sempre non vuota: quale condizione ha deciso l'esito. Va nella telemetria. */
  why: string;
}

/**
 * Il minimo che serve per decidere. Un GPUDevice reale lo soddisfa per
 * struttura (`device.features`, `device.adapterInfo`): i chiamanti passano il
 * device nudo, senza adattatori.
 */
export interface GemvCapsSource {
  features: { has(f: string): boolean };
  adapterInfo?: { subgroupMinSize?: number; subgroupMaxSize?: number };
}

/** Decide se abilitare il percorso subgroup di gemv, e dice sempre perché. */
export function gemvCapsFor(d: GemvCapsSource): GemvCaps {
  if (!d.features.has(GEMV_SG_FEATURE)) {
    return { sg: false, why: `feature "${GEMV_SG_FEATURE}" assente sul device` };
  }

  const info = d.adapterInfo;
  if (!info) {
    return {
      sg: false,
      why: `adapterInfo non esposto: la dimensione del subgroup è ignota, non verificabile === ${GEMV_SG_REQUIRED_SIZE}`,
    };
  }

  const min = info.subgroupMinSize;
  const max = info.subgroupMaxSize;
  if (typeof min !== "number" || typeof max !== "number") {
    // Terza condizione, variante: adapterInfo c'è ma i campi no. Va detto
    // così, non come "dimensione diversa da 32": in telemetria una dimensione
    // mai letta non deve sembrare misurata.
    return {
      sg: false,
      why: `adapterInfo non espone subgroupMinSize/subgroupMaxSize (letti: ${fieldsRead(min, max)}): la dimensione del subgroup resta ignota, non verificabile === ${GEMV_SG_REQUIRED_SIZE}`,
    };
  }

  if (min !== GEMV_SG_REQUIRED_SIZE || max !== GEMV_SG_REQUIRED_SIZE) {
    return {
      sg: false,
      why: `dimensione subgroup non fissa a ${GEMV_SG_REQUIRED_SIZE} (min=${min}, max=${max}): la mappatura riga->subgroup di vec4-rows2-sg si romperebbe in silenzio`,
    };
  }

  return {
    sg: true,
    why: `feature "${GEMV_SG_FEATURE}" presente e subgroup fisso a ${GEMV_SG_REQUIRED_SIZE} (min=max=${GEMV_SG_REQUIRED_SIZE})`,
  };
}

/** Quali dei due campi erano leggibili, per il `why` del caso "campi assenti". */
function fieldsRead(min: number | undefined, max: number | undefined): string {
  const got: string[] = [];
  if (typeof min === "number") got.push(`subgroupMinSize=${min}`);
  if (typeof max === "number") got.push(`subgroupMaxSize=${max}`);
  return got.length > 0 ? got.join(", ") : "nessuno dei due";
}
