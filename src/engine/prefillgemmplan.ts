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
// il PREDICATO DI AMMISSIBILITA' (quali kind, e per ognuno la sua unita' di
// taglio: 64 sul q4_0, il superblocco da 256 sul q5_K) arrivano da
// `kernels/wgsl.ts`: qui non c'e' una seconda formula ne' una seconda
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
  PREFILL_SPLITS_UNSPLIT, PREFILL_GEMM_KINDS, isPrefillGemmKind, type PrefillGemmKind,
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
  /** true = la via intera (pesi × q8_0 con `dot4I8Packed`) gira su questo runtime. */
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
    why: `language feature "${PREFILL_IDOT_FEATURE}" presente ⇒ via intera (pesi × q8_0) con dot4I8Packed`,
  };
}

/**
 * Le tre vie del GEMM di prefill.
 *   idot   = pesi × q8_0 intera, la forma VINCENTE (riga 1 sul q4_0, riga 2 del
 *            goal K-quant sul q5_K);
 *   f32    = stessa forma multi-riga, dequant in virgola mobile — FALLBACK
 *            DICHIARATO per i device senza la language feature, non
 *            un'alternativa preferibile;
 *   legacy = M gemv replicate su `wid.z`, riuso dei pesi ZERO. E' cio' che il
 *            motore emette oggi, e resta l'unica via possibile dove il kernel
 *            veloce non c'e' (kind fuori da `PREFILL_GEMM_KINDS`), non si
 *            applica (K non multiplo dell'unita' del formato), o dove il
 *            contratto lo impone (M=1: vedi `PREFILL_M1_LEGACY`).
 */
export type PrefillGemmVia = "idot" | "f32" | "legacy";

/**
 * M=1 ⇒ LEGACY SU OGNI KIND — clausola di CONTRATTO, e va letta per quello che
 * e', perche' le misure non la sostengono per intero.
 *
 * COSA DICE IL BANCO A M=1 (p50, RTX 4090, `base-batch-z` = la forma legacy):
 *   q4_0 K2560xN9216   0,1648   -> splitk-idot-full 0,019072 =  8,64x PIU' VELOCE
 *   q5_K K4096xN2560   0,08288  -> splitk-idot      0,023872 =  3,47x PIU' VELOCE
 *   q5_K K4096xN2560   0,08288  -> splitk-f32       0,148992 =  0,56x PIU' LENTA
 * (results/microbench/ttft-riga1-4090-linux-2026-08-14T18-54-05-813Z.json e
 *  kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json.)
 *
 * Cioe': **l'unica cella in cui la forma multi-riga perde a M=1 e' il q5_K
 * sulla via f32**. Il 0,56x NON e' una proprieta' di M=1 in generale, e questa
 * costante esiste perche' quel numero non finisca in telemetria travestito da
 * regola generale — era esattamente cosi' nel primo giro di questa riga.
 *
 * PERCHE' ALLORA LA CLAUSOLA RESTA COM'E': e' scritta nel contratto della riga
 * 2 (done-when 7) e nell'interfaccia congelata del task, ed e' CONSERVATIVA —
 * rende legacy, cioe' la via che il motore emette oggi e che non ha mai
 * sbagliato un risultato. Oggi e' anche INERTE: l'unico consumatore del piano
 * gira a `PREFILL_M` = 16. Restringerla alla cella misurata (q5_K + f32) e'
 * una decisione del PI, non dell'implementatore: sta nel consuntivo della riga
 * come divergenza aperta, non risolta qui in silenzio.
 */
export const PREFILL_M1_LEGACY = {
  /** L'M su cui la clausola scatta. */
  M: 1,
  /** La sola cella di banco in cui la forma multi-riga perde davvero a M=1. */
  measured: { kind: "q5_K", via: "f32", legacyMs: 0.08288, fastMs: 0.148992, ratio: "0,56x" },
  reasonFor(o: { kind: string; K: number }): string {
    return `M=1: la forma multi-riga non si instrada a una riga sola — clausola del `
      + `CONTRATTO della riga 2 (done-when 7), non una misura generale. Al banco a M=1 `
      + `l'unica cella piu' lenta della legacy e' q5_K sulla via f32 (0,08288 -> 0,148992 ms `
      + `= 0,56x), mentre q5_K/idot misura 3,47x e q4_0/idot 8,64x PIU' VELOCI della legacy: `
      + `la clausola e' quindi CONSERVATIVA, non derivata. La ragione strutturale che la `
      + `motiva e' che a M=1 non c'e' riuso dei pesi da ammortizzare (e' l'intero punto `
      + `della forma) e restano i due dispatch in piu' (quantizzazione + combine). Vale su `
      + `ogni kind, "${o.kind}" compreso: il rifiuto e' di M=1, non della shape K=${o.K}`;
  },
} as const;

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
 * l'unico posto dove sono scritti sia il rifiuto dei kind fuori elenco sia la
 * soglia geometrica di ciascun formato — nell'ordine giusto, prima il formato e
 * poi la geometria. Sondarlo invece di ricopiare la soglia significa che se
 * domani il kernel accetta un altro kind o un'altra unita' di taglio, il piano
 * lo segue senza toccare questo file; e che non esiste una seconda soglia che
 * possa divergere in silenzio.
 *
 * La sonda usa `PREFILL_SPLITS_UNSPLIT` (= 1 fetta) perche' e' il valore che
 * NON aggiunge condizioni proprie: con una fetta sola il controllo di
 * divisibilita' delle unita' coincide con la sola soglia su K, quindi la sonda
 * misura esattamente l'ammissibilita' e non la scelta delle fette — quelle le
 * decide `prefillGemmSplitsFor` subito dopo, col suo ripiego.
 *
 * Il messaggio del kernel viene riportato COME STA dentro la ragione: e' gia'
 * scritto per essere letto da un umano, e riscriverlo qui sarebbe la stessa
 * duplicazione, spostata dalle soglie alle parole.
 */
function kernelVerdict(o: {
  kind: PrefillQuantKind; K: number; N: number; M: number;
}): { splits: number; kind: PrefillGemmKind } | { rejected: string } {
  // IL KIND CHE ARRIVA AL KERNEL E' QUELLO VERO. Prima qui c'era `o.kind as
  // "q4_0"`: una bugia innocua finche' il kernel accettava un formato solo, ma
  // appena ne accetta due quel cast fa contare blocchi da 32 dove l'unita' e'
  // il superblocco da 256 — fette sbagliate, storage sbagliato, nessun errore.
  // Il cast qui sotto serve solo a far entrare nella firma un kind che il
  // kernel POTREBBE rifiutare: e' proprio quel rifiuto che si vuole sentire.
  try {
    prefillPartialFloats({
      kind: o.kind as PrefillGemmKind, K: o.K, N: o.N, M: o.M, splits: PREFILL_SPLITS_UNSPLIT,
    });
  } catch (e) {
    return { rejected: e instanceof Error ? e.message : String(e) };
  }
  // Il kernel NON ha rifiutato ⇒ il kind e' uno dei suoi. Il type guard non e'
  // cerimonia: e' cio' che permette di passare il kind REALE a
  // `prefillGemmSplitsFor` senza un secondo cast, e interroga l'elenco
  // ESPORTATO dal kernel, non una copia locale.
  if (!isPrefillGemmKind(o.kind)) {
    return {
      rejected: `kind "${o.kind}" accettato dal contorno del kernel ma assente da `
        + `PREFILL_GEMM_KINDS (${PREFILL_GEMM_KINDS.join(", ")}): il predicato e l'elenco `
        + "sono usciti dal passo, e instradare qui vorrebbe dire indovinare la geometria",
    };
  }
  return { splits: prefillGemmSplitsFor(o.K, o.N, o.kind), kind: o.kind };
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

  // M=1 prima di qualunque domanda al kernel: non e' la shape a non andare, e'
  // l'M. La costante porta con se' i numeri veri e il perche' resta com'e'.
  if (o.M === PREFILL_M1_LEGACY.M) {
    return {
      via: "legacy",
      reason: PREFILL_M1_LEGACY.reasonFor(o),
      splits: 0, partialFloats: 0, wgStorageBytes: 0, xqU32: 0, xscF32: 0,
    };
  }

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

  // Da qui in poi i numeri NON si decidono qui: si chiedono al kernel, e col
  // kind REALE — quello che il verdetto ha restituito, non un letterale.
  const { splits, kind } = verdict;
  const opts = { kind, K: o.K, N: o.N, M: o.M, splits };
  const partialFloats = prefillPartialFloats(opts);
  // `const BLOCKS` di prefillQuantXQ8Wgsl. Il quantizzatore delle attivazioni e'
  // lo STESSO su entrambi i kind: i sotto-blocchi K-quant sono anch'essi da 32,
  // otto per superblocco, quindi la via intera q5_K riusa `xq`/`xsc` tali e
  // quali — nessun secondo quantizzatore, nessun dispatch in piu'.
  const blocks = o.M * (o.K / 32);

  if (o.idot) {
    return {
      via: "idot",
      reason: `${kind} K=${o.K} accettato dal kernel e "${PREFILL_IDOT_FEATURE}" disponibile: via intera `
        + `${kind} × q8_0 (dot4I8Packed), ${splits} fette di K`,
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
    reason: `${kind} K=${o.K} accettato dal kernel ma "${PREFILL_IDOT_FEATURE}" non disponibile: `
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
