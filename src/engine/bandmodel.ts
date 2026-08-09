// Modello di banda del regime sync in scarsità (C3c fase 6, spec §5).
//
// tok/s = f(hit-rate d'uso, banda di lettura, budget slab implicito in h).
// La formula è CHIUSA e i coefficienti vengono da artefatti INDIPENDENTI dai
// punti che il modello deve predire (regola spec §5: niente fit a posteriori
// sul punto predetto):
//   - BASE_SYNC_MS_PER_TOKEN: bench sync b12 di c3b (bench-glm-4090-b12-sync-
//     nonreg-2026-08-07.json): msPerToken 188.705 − stallo 21.274 = 167.43.
//     È il costo strutturale del path sync (gpuBusy + 47 sync + encode) a
//     miss zero — NON dipende dal budget slab (la struttura non cambia).
//   - MISS_FIXED_MS: dallo stesso artefatto, costo per miss NON di banda:
//     stallo/miss 4.83 ms − componente di banda a cache calda
//     (5 308 416 B / 11.41 GB/s = 0.47 ms, banda warm dal WP fase 1)
//     = 4.36 ms (upload writeBuffer 3.70 + overhead API di lettura 0.66).
//   - la BANDA è un PARAMETRO: warm ~11.4e9 (WP fase 1 randExpertWarm),
//     fredda 1.79-1.94e9 (WP fase 1 randExpertCold) — il punto freddo lo
//     valida la fase 7 (instant-on), qui si validano i punti warm.
//
// Il TTFT a freddo (spec §5, seconda formula) si predice come delta sul TTFT
// a caldo: nonExpert in streaming + il premio freddo sui byte unici letti.
import { GLM47_FLASH as G } from "./shape";
import { SLAB_DOWN_Q4_0 } from "./moe";
import { NON_EXPERT_BYTES } from "./residency";

/** costo strutturale del path sync a miss zero (b12 sync c3b, v. header) */
export const BASE_SYNC_MS_PER_TOKEN = 188.705 - 21.274; // 167.43
/** costo per miss non-di-banda: upload + overhead API (v. header) */
export const MISS_FIXED_MS = 4.83 - (SLAB_DOWN_Q4_0.bytes / 11.41e9) * 1000; // ≈4.36
/** banda warm/fredda misurate dal WP fase 1 (B/s) — parametri, non costanti */
export const BAND_WARM_BPS = 11.41e9;
export const BAND_COLD_BPS = 1.79e9;
export const BAND_COLD_SEQ_BPS = 3.51e9;

/** ms per miss BLOCCANTE a una data banda di lettura */
export const missMs = (bandBps: number): number =>
  (SLAB_DOWN_Q4_0.bytes / bandBps) * 1000 + MISS_FIXED_MS;

// ---- coefficienti del costo per fetch nel path sync coi meccanismi C3c ----
// La misura di fase 6 (bench sync+tier+prefetch a b12/1472/736) mostra che il
// wall NON è BASE + stallo: l'I/O del prefetch e l'overhead CPU/GC dei fetch
// (5.3 MB l'uno) sbordano nel wall, e gpuBusy SALTA 48→84 ms/token quando i
// fetch/token superano ~20 (interleave copy/compute — misurato, b1472/b736).
// Scomposizione del costo per fetch: componente di BANDA (parametrica, dal WP
// fase 1) + componente FISSA (CPU/GC/upload). Fit dichiarato: C e STEP sui
// punti {1472, 736}; il punto b12 resta FUORI dal fit ed è la predizione
// (misurato −1.0%). A banda fredda la componente di banda cresce da sé.
/** costo per fetch alla banda warm di riferimento (fit {1472,736}) */
export const FETCH_COST_WARM_MS = 2.426;
/** quota FISSA del costo per fetch (CPU/GC/upload, non di banda) */
export const FETCH_FIXED_MS = FETCH_COST_WARM_MS - (SLAB_DOWN_Q4_0.bytes / BAND_WARM_BPS) * 1000; // ≈1.96
/** salto di gpuBusy oltre la soglia di fetch (misurato 48→84 ms/token) */
export const GPU_STEP_MS = 95.0;
export const GPU_STEP_FETCH_THRESHOLD = 20;

/** fetch totali per token (uso + prefetch): con prefetch acceso le richieste
 *  alla cache raddoppiano (184 ensure d'uso + 184 predizioni). */
export const fetchesPerToken = (hAll: number, prefetchOn = true): number =>
  (1 - hAll) * (G.nLayer - G.denseLead) * G.nExpertUsed * (prefetchOn ? 2 : 1);

/**
 * ms/token del decode sync in scarsità (meccanismi C3c accesi). `hAll` è
 * l'hit-rate AGGREGATO della cache (uso + prefetch — il numero del report
 * bench); la banda è un parametro: warm per validare sui punti misurati,
 * fredda (WP fase 1) per il regime disk-bound/instant-on.
 */
export function syncMsPerToken(hAll: number, bandBps: number): number {
  if (hAll < 0 || hAll > 1) throw new Error(`bandmodel: hAll ${hAll} fuori [0,1]`);
  const f = fetchesPerToken(hAll);
  const perFetch = (SLAB_DOWN_Q4_0.bytes / bandBps) * 1000 + FETCH_FIXED_MS;
  return BASE_SYNC_MS_PER_TOKEN + f * perFetch + (f > GPU_STEP_FETCH_THRESHOLD ? GPU_STEP_MS : 0);
}

export const syncToksPerSec = (hAll: number, bandBps: number): number =>
  1000 / syncMsPerToken(hAll, bandBps);

/**
 * TTFT a freddo predetto (spec §5): TTFT a caldo + streaming del non-expert
 * + premio freddo sui byte UNICI letti dal prefill (prima lettura fredda,
 * riletture calde) − la quota nascosta dall'overlap (0 = nessun overlap,
 * conservativo; l'overlap del prefill chunked è materia fase 7).
 */
export function coldTtftMs(o: {
  warmTtftMs: number;
  uniqueBytesRead: number;   // byte unici toccati dal prefill (≤ parco)
  coldBps?: number;          // default streaming seq (il layout slab è sequenziale)
  warmBps?: number;
  overlapMs?: number;        // quota di I/O nascosta dietro il compute (fase 7)
}): number {
  const cold = o.coldBps ?? BAND_COLD_SEQ_BPS;
  const warm = o.warmBps ?? BAND_WARM_BPS;
  const nonExpertMs = (NON_EXPERT_BYTES / cold) * 1000;
  const premiumMs = o.uniqueBytesRead * (1 / cold - 1 / warm) * 1000;
  return o.warmTtftMs + nonExpertMs + Math.max(0, premiumMs - (o.overlapMs ?? 0));
}
