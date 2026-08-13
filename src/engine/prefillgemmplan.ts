// PIANO del GEMM di prefill (ondata TTFT, riga 2): per ogni sito di
// moltiplicazione decide QUALE VIA prende, con quante fette e con quanto
// scratch — e lo decide PRIMA di avere un device, in aritmetica pura.
//
// MODULO PURO come `gemvcaps.ts` e `prefillbytes.ts`: nessun `GPUDevice`,
// nessun WGSL generato qui dentro, nessuna allocazione. Gira in vitest su
// qualunque runner, quindi la scelta della via e la taglia dei buffer sono
// verificabili in CI e non solo davanti a una GPU.
//
// LA REGOLA CHE REGGE TUTTO IL FILE: quello che sta accanto al kernel si
// CHIEDE al kernel, non si ricopia. Fette (`prefillGemmSplitsFor`), parziali
// (`prefillPartialFloats`), workgroup storage
// (`prefillGemmWorkgroupStorageBytes`) e — dalla review di it.7 in poi — anche
// il PREDICATO DI AMMISSIBILITA' (kind q4_0-only, K multiplo di 64) arrivano
// da `kernels/wgsl.ts`: qui non c'e' una seconda formula ne' una seconda
// soglia. Un secondo posto che decide gli stessi numeri e' esattamente il bug
// di it.7 — due posti che stabilivano le righe-per-workgroup, e le stabilivano
// diverse.
//
// L'unica geometria che questo file deve dire da se' e' la taglia delle
// attivazioni quantizzate (`xq`/`xsc`), perche' accanto al kernel non esiste
// una funzione che la esporti: e' presa alla lettera dal TESTO di
// `prefillQuantXQ8Wgsl` (`const BLOCKS = M*bpr`, `xq[b * 8u + i]`, `xsc[b]`) e
// il test la riverifica CONTRO quel testo, non contro se stessa.
import {
  prefillGemmSplitsFor, prefillPartialFloats, prefillGemmWorkgroupStorageBytes,
  PREFILL_SPLITS_UNSPLIT,
} from "./kernels/wgsl";
import type { PrefillDispatch, PrefillQuantKind } from "./prefillbytes";

/**
 * La LANGUAGE FEATURE che abilita `dot4I8Packed`, cioe' la via intera.
 *
 * Si legge da `navigator.gpu.wgslLanguageFeatures`, NON da `device.features`, e
 * nel WGSL non si scrive nessun `enable`: e' una language feature, non
 * un'estensione, e scriverla fa fallire la compilazione con «expected
 * extension» — costato una run intera in it.5.
 */
export const PREFILL_IDOT_FEATURE = "packed_4x8_integer_dot_product";

/**
 * Il minimo che serve per decidere. `navigator.gpu` lo soddisfa per struttura:
 * i chiamanti passano l'oggetto nudo, senza adattatori. Il campo e' opzionale
 * perche' un runtime che non lo espone e' un caso REALE da trattare, non un
 * errore di tipo da nascondere.
 */
export interface PrefillGemmCapsSource {
  wgslLanguageFeatures?: { has(f: string): boolean };
}

export interface PrefillGemmCaps {
  /** true = la via intera q4_0 × q8_0 e' disponibile su questo runtime. */
  idot: boolean;
  /** Sempre non vuota: quale condizione ha deciso l'esito. Va in telemetria. */
  why: string;
}

/**
 * Decide se la via intera e' disponibile, e dice sempre perche' (postura di
 * `gemvCapsFor`: un booleano senza motivo non e' diagnosticabile a posteriori).
 *
 * Assenza della feature NON e' un guasto: e' il caso in cui si prende il
 * fallback f32, che e' la forma su cui la via intera ha misurato il suo 1,745x.
 */
export function prefillGemmCapsFor(s: PrefillGemmCapsSource): PrefillGemmCaps {
  const f = s.wgslLanguageFeatures;
  if (!f) {
    return {
      idot: false,
      why: `wgslLanguageFeatures non esposto dal runtime: "${PREFILL_IDOT_FEATURE}" non e' verificabile, `
        + "e non sapere e' indistinguibile dal non averla ⇒ fallback f32",
    };
  }
  if (!f.has(PREFILL_IDOT_FEATURE)) {
    return {
      idot: false,
      why: `language feature "${PREFILL_IDOT_FEATURE}" assente ⇒ fallback DICHIARATO sulla via f32 `
        + "(stessa forma multi-riga, dequant in virgola mobile nel ciclo interno)",
    };
  }
  return {
    idot: true,
    why: `language feature "${PREFILL_IDOT_FEATURE}" presente ⇒ via intera q4_0 × q8_0 con dot4I8Packed`,
  };
}

/**
 * Le tre vie del GEMM di prefill.
 *   idot   = q4_0 × q8_0 intera, la forma VINCENTE della riga 1;
 *   f32    = stessa forma multi-riga, dequant in virgola mobile — FALLBACK
 *            DICHIARATO per i device senza la language feature, non
 *            un'alternativa preferibile;
 *   legacy = M gemv replicate su `wid.z`, riuso dei pesi ZERO. E' cio' che il
 *            motore emette oggi, e resta l'unica via possibile dove il kernel
 *            veloce non c'e' (kind ≠ q4_0) o non si applica (K % 64 ≠ 0).
 */
export type PrefillGemmVia = "idot" | "f32" | "legacy";

export interface PrefillGemmRoute {
  via: PrefillGemmVia;
  /** Sempre non vuota, su TUTTE le vie: perche' questo sito prende questa via. */
  reason: string;
  /** Fette di K dello split-K. 0 sulla via legacy: li' lo split-K non esiste. */
  splits: number;
  /** f32 del buffer dei parziali `part[(s*M + m)*N + r]`. 0 su legacy. */
  partialFloats: number;
  /** Workgroup storage del moltiplicatore. 0 su legacy. */
  wgStorageBytes: number;
  /** u32 delle attivazioni quantizzate. 0 sulle vie NON intere. */
  xqU32: number;
  /** f32 delle scale delle attivazioni. 0 sulle vie NON intere. */
  xscF32: number;
}

/** Un sito di moltiplicazione del prefill: il nome serve alla diagnostica. */
export interface PrefillSite {
  site: string;
  kind: PrefillQuantKind;
  K: number;
  N: number;
}

function checkGeom(o: { K: number; N: number; M: number }, who: string): void {
  for (const [name, v] of [["K", o.K], ["N", o.N], ["M", o.M]] as const) {
    if (!Number.isInteger(v) || v < 1) throw new Error(`${who}: ${name} non valido (${v})`);
  }
}

/**
 * IL PREDICATO DI AMMISSIBILITA' NON VIVE QUI: si SONDA il kernel.
 *
 * `prefillPartialFloats` passa per `prefillGemmCheck` (kernels/wgsl.ts), che e'
 * l'unico posto dove sono scritti sia il rifiuto dei kind ≠ q4_0 sia la soglia
 * `K % 64` — nell'ordine giusto, prima il formato e poi la geometria. Sondarlo
 * invece di ricopiare `K % 64` significa che se domani il kernel accetta un
 * altro kind o un altro passo, il piano lo segue senza toccare questo file; e
 * che non esiste una seconda soglia che possa divergere in silenzio.
 *
 * La sonda usa `PREFILL_SPLITS_UNSPLIT` (= 1 fetta) perche' e' il valore che
 * NON aggiunge condizioni proprie: con una fetta sola il controllo di
 * divisibilita' dei blocchi (`bpr % (splits*2)`) coincide con `K % 64`, quindi
 * la sonda misura esattamente l'ammissibilita' e non la scelta delle fette —
 * quelle le decide `prefillGemmSplitsFor` subito dopo, col suo ripiego.
 *
 * Il messaggio del kernel viene riportato COME STA dentro la ragione: e' gia'
 * scritto per essere letto da un umano, e riscriverlo qui sarebbe la stessa
 * duplicazione, spostata dalle soglie alle parole.
 */
function kernelVerdict(o: {
  kind: PrefillQuantKind; K: number; N: number; M: number;
}): { splits: number } | { rejected: string } {
  try {
    prefillPartialFloats({
      kind: o.kind as "q4_0", K: o.K, N: o.N, M: o.M, splits: PREFILL_SPLITS_UNSPLIT,
    });
  } catch (e) {
    return { rejected: e instanceof Error ? e.message : String(e) };
  }
  return { splits: prefillGemmSplitsFor(o.K, o.N) };
}

/**
 * LA DECISIONE, per un sito solo.
 *
 * Su legacy TUTTI i numeri di scratch sono 0: quella via non accende la
 * pipeline veloce, e un numero non nullo qui gonfierebbe lo scratch CONDIVISO
 * (che e' un max) per buffer che nessun dispatch legge.
 */
export function planPrefillGemm(o: {
  kind: PrefillQuantKind; K: number; N: number; M: number; idot: boolean;
}): PrefillGemmRoute {
  checkGeom(o, "planPrefillGemm");

  const verdict = kernelVerdict(o);
  if ("rejected" in verdict) {
    return {
      via: "legacy",
      reason: `il moltiplicatore multi-riga di prefill NON accetta questa shape ⇒ si resta su legacy `
        + `(M gemv replicate, riuso pesi zero) invece di inventare una forma non misurata. `
        + `Il rifiuto arriva dal contorno del kernel — kernels/wgsl.ts, prefillGemmCheck, `
        + `l'unico posto dove kind e geometria sono decisi: «${verdict.rejected}»`,
      splits: 0, partialFloats: 0, wgStorageBytes: 0, xqU32: 0, xscF32: 0,
    };
  }

  // Da qui in poi i numeri NON si decidono qui: si chiedono al kernel.
  const { splits } = verdict;
  const opts = { kind: "q4_0" as const, K: o.K, N: o.N, M: o.M, splits };
  const partialFloats = prefillPartialFloats(opts);
  const blocks = o.M * (o.K / 32);   // `const BLOCKS` di prefillQuantXQ8Wgsl

  if (o.idot) {
    return {
      via: "idot",
      reason: `q4_0 K=${o.K} accettato dal kernel e "${PREFILL_IDOT_FEATURE}" disponibile: via intera `
        + `q4_0 × q8_0 (dot4I8Packed), ${splits} fette di K`,
      splits,
      partialFloats,
      wgStorageBytes: prefillGemmWorkgroupStorageBytes(opts, "idot"),
      // layout di `prefillQuantXQ8Wgsl`: 8 u32 per blocco (`xq[b * 8u + i]`) e
      // una scala per blocco (`xsc[b]`), su M·(K/32) blocchi.
      xqU32: blocks * 8,
      xscF32: blocks,
    };
  }
  return {
    via: "f32",
    reason: `q4_0 K=${o.K} accettato dal kernel ma "${PREFILL_IDOT_FEATURE}" non disponibile: `
      + `FALLBACK DICHIARATO sulla via f32, stessa forma multi-riga con dequant in virgola mobile, `
      + `${splits} fette di K`,
    splits,
    partialFloats,
    wgStorageBytes: prefillGemmWorkgroupStorageBytes(opts, "f32"),
    xqU32: 0,   // la via f32 legge le attivazioni in virgola mobile: niente xq
    xscF32: 0,
  };
}

/**
 * Scratch per l'INTERO piano: UN solo set di buffer condivisi fra i siti,
 * quindi il MAX shape per shape — non la somma.
 *
 * I dispatch del prefill sono seriali dentro lo stesso comando e ogni sito
 * consuma i parziali (con la `combine`) prima che il sito dopo li riscriva:
 * sommare vorrebbe dire prenotare 248 buffer per un solo utilizzatore alla
 * volta (sul 4B ≈ 300 MB invece di 2,36).
 */
export function prefillGemmScratchFor(o: {
  sites: PrefillSite[]; M: number; idot: boolean;
}): { partialFloats: number; xqU32: number; xscF32: number } {
  let partialFloats = 0, xqU32 = 0, xscF32 = 0;
  for (const s of o.sites) {
    const r = planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M: o.M, idot: o.idot });
    partialFloats = Math.max(partialFloats, r.partialFloats);
    xqU32 = Math.max(xqU32, r.xqU32);
    xscF32 = Math.max(xscF32, r.xscF32);
  }
  return { partialFloats, xqU32, xscF32 };
}

/**
 * Il piano come SEQUENZA DI DISPATCH, col suo controfattuale.
 *
 * `dispatches` = cio' che il motore emetterebbe col piano nuovo (multirow dove
 * la via veloce si applica, legacy dove no). `legacy` = LE STESSE shape nello
 * stesso ordine, tutte contate a forma legacy: e' il "prima" contro cui si
 * misura il "dopo", e vive qui — non in una formula a parte — perche'
 * l'accettazione del goal chiede un rapporto fra due piani, non fra un piano e
 * una stima.
 *
 * `exceptions` elenca i siti rimasti legacy CON la loro ragione: e' la lista
 * che dice dove il guadagno non arriva, ed e' l'unico modo perche' un sito
 * dimenticato si veda invece di sparire nella media.
 */
export function prefillPlanDispatches(o: {
  sites: PrefillSite[]; M: number; idot: boolean;
}): {
  dispatches: PrefillDispatch[];
  legacy: PrefillDispatch[];
  exceptions: { site: string; kind: string; K: number; N: number; reason: string }[];
} {
  const dispatches: PrefillDispatch[] = [];
  const legacy: PrefillDispatch[] = [];
  const exceptions: { site: string; kind: string; K: number; N: number; reason: string }[] = [];
  for (const s of o.sites) {
    const r = planPrefillGemm({ kind: s.kind, K: s.K, N: s.N, M: o.M, idot: o.idot });
    dispatches.push({
      form: r.via === "legacy" ? "legacy" : "multirow",
      kind: s.kind, K: s.K, N: s.N, M: o.M,
    });
    legacy.push({ form: "legacy", kind: s.kind, K: s.K, N: s.N, M: o.M });
    if (r.via === "legacy") {
      exceptions.push({ site: s.site, kind: s.kind, K: s.K, N: s.N, reason: r.reason });
    }
  }
  return { dispatches, legacy, exceptions };
}
