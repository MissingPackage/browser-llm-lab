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
  PREFILL_GEMM_ROWS_PER_WG,
  prefillGemmSplitsFor, prefillPartialFloats, prefillGemmWorkgroupStorageBytes,
  PREFILL_SPLITS_UNSPLIT, PREFILL_GEMM_KINDS, isPrefillGemmKind, prefillGemmWiring,
  type PrefillGemmKind,
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
 *            applica (K non multiplo dell'unita' del formato), NON E' CABLATO
 *            (kind con `wired: false`: il kernel esiste ed e' misurato, ma
 *            nessun sito di produzione ci passa), o dove il contratto lo impone
 *            (M=1: vedi `PREFILL_M1_LEGACY`).
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
/**
 * PAVIMENTO SU N DELLA ROTTA NEL DECODE — misurato, non scelto (it.38).
 *
 * Il predicato generale ammette `N >= PREFILL_GEMM_ROWS_PER_WG` (64), che e' la
 * soglia sotto cui il kernel lavorerebbe a meno di mezzo workgroup. Per il
 * PREFILL va bene: a M righe c'e' riuso dei pesi da ammortizzare e la forma
 * vince comunque. Per il DECODE no. Al banco, a M=1, sulle shape vere e contro
 * il kernel che il decode emette davvero:
 *
 *     q8_0  K=2048  N= 512   0,92x   PERDE    (ffn_gate/up_shexp del 35B)
 *     q8_0  K= 512  N=2048   2,45x   vince    (ffn_down_shexp)
 *     q8_0  K=2048  N=4096   3,25x   vince
 *     q8_0  K=2048  N=8192   3,80x   vince
 *
 * COSA HA TROVATO: la riga da 0,92x **era instradata** dalla rotta e nessuno se
 * n'era accorto, perche' il segmento `shexp` migliora lo stesso (−0,174
 * ms/token) — il guadagno sul `down` copre la perdita su `gate` e `up`.
 * Escluderli restituisce ~0,048 ms/token: poco, e sotto il rumore dell'A/B. Ma
 * un predicato che instrada una forma che il banco dichiara perdente e'
 * sbagliato anche quando il totale torna, e il giorno che qualcuno estende la
 * rotta ai K-quant — dove alle stesse shape si misura 0,20-0,57x — quel
 * predicato diventerebbe una regressione vera.
 *
 * PERCHE' 2048 E NON 1024: fra 512 e 2048 non ho misure. Mettere la soglia dove
 * non ho guardato sarebbe scegliere un numero invece di misurarlo; il valore sta
 * sul primo punto MISURATO che vince.
 */
export const DEC_SPLITK_MIN_N = 2048;

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
 * IL PREDICATO SULLA SHAPE — la sede unica, dichiarata, dell'ammissibilita' AL
 * PIANO, e la cosa che il flag `wired` NON sa fare.
 *
 * PERCHE' ESISTE, e va scritto per esteso perche' e' una trappola registrata:
 * `wired` e' un flag PER FORMATO. Accendere un formato lo accende su TUTTI i
 * suoi siti nello stesso istante — compresi i 48 `ssm_alpha`/`ssm_beta` del 4B,
 * che sono q8_0 con N=32. La forma split-K produce `PREFILL_GEMM_ROWS_PER_WG`
 * (64) righe di uscita per workgroup: a N=32 ogni dispatch lavorerebbe a mezzo
 * workgroup, su una shape dove la forma non e' mai stata misurata. Il q8_0 e'
 * stato cablato (2026-08-15) SOLO perche' questo controllo esisteva gia'; q2_K
 * e q3_K arriveranno con lo stesso vincolo. Sul 35B gli stessi tensori di attn
 * hanno N=4096 e devono passare: l'esclusione giusta e' sulla SHAPE, non sulla
 * famiglia.
 *
 * PERCHE' NEL PIANO E NON ACCANTO AL KERNEL. Il contorno del kernel risponde a
 * «questa forma si genera?», e su N la risposta e' SI': il kernel guarda
 * `r < N` e produce il valore giusto anche a N=32 (lo dice il commento di
 * `PREFILL_GEMM_ROWS_PER_WG` in kernels/wgsl.ts, che rimanda proprio qui).
 * Quello che N decide e' se la forma CONVIENE, ed e' una domanda del piano.
 * Metterlo nel kernel romperebbe anche le query di dimensionamento, che si
 * interrogano a N=1 apposta.
 *
 * PERCHE' `kind` E `K` STANNO NELLA FIRMA se la decisione guarda solo N: perche'
 * la ragione deve poter NOMINARE il sito che rifiuta — «N=32» da solo non dice
 * su quale formato e con quale K si e' fermato — e perche' il giorno in cui un
 * formato avesse un pavimento suo (un `DEC_SPLITK_MIN_N` per famiglia), questa
 * e' la sede in cui aggiungerlo senza aprirne una seconda. Cio' che qui NON si
 * decide e' la geometria su K: quella e' del kernel (`prefillGemmCheck`), e
 * ricopiarla qui sarebbe il difetto di it.7 in forma nuova.
 *
 * SI LEGGE IN UN POSTO SOLO — `kernelVerdict`, qui sotto, accanto al flag
 * `wired`. Il gate strutturale sta in tests/engine-prefill-q2k-q3k.test.ts
 * ([s4]), ed e' la stessa postura del gate [w4] su `prefillGemmWiring`.
 */
export function prefillGemmShapeOk(o: { kind: PrefillGemmKind; K: number; N: number }):
  { ok: boolean; why: string } {
  if (o.N < PREFILL_GEMM_ROWS_PER_WG) {
    return {
      ok: false,
      // LA RAGIONE NOMINA N. E' il requisito che la rende utile: in telemetria
      // «shape non ammessa» su 48 siti non direbbe QUALE dimensione li esclude.
      why: `N=${o.N} sotto le ${PREFILL_GEMM_ROWS_PER_WG} righe di uscita per workgroup della `
        + `forma split-K: il dispatch lavorerebbe a meno di meta' workgroup e la forma non e' `
        + `stata misurata a questa shape (${o.kind}, K=${o.K}). Il sito resta sulla via legacy, `
        + `che e' lenta e giusta`,
    };
  }
  return {
    ok: true,
    // Anche l'AMMISSIONE porta la sua ragione (postura di `GemvCaps.why`): un
    // booleano nudo non e' diagnosticabile a posteriori, e chi legge deve poter
    // capire che un sito rimasto legacy lo e' per un ALTRO motivo.
    why: `N=${o.N} >= ${PREFILL_GEMM_ROWS_PER_WG} righe di uscita per workgroup: la forma split-K `
      + `riempie almeno un workgroup intero, quindi la shape ${o.kind} K=${o.K} N=${o.N} e' `
      + `instradabile per geometria di uscita. Se questo sito resta legacy lo decide qualcos'altro `
      + `— il cablaggio del formato, o il contorno del kernel su K — non il suo N`,
  };
}

/**
 * IL PREDICATO DI AMMISSIBILITA' DEL KERNEL NON VIVE QUI: si SONDA il kernel.
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
 *
 * TRE DOMANDE, NON UNA. «Esiste un kernel per questo formato?», «quel kernel e'
 * INSTRADATO in produzione?» e «su QUESTA shape conviene?» erano la stessa
 * domanda finche' ogni forma portata veniva anche cablata e finche' ogni sito di
 * un formato aveva la stessa geometria. Non lo sono piu':
 *   - il CABLAGGIO lo dice il flag `wired` di `PREFILL_GEMM_SPEC`, che si legge
 *     QUI e in nessun altro posto (i kernel q4_K, q6_K, q2_K e q3_K esistono e
 *     sono portati, e nessun sito ci passa);
 *   - la SHAPE la dice `prefillGemmShapeOk`, qui sopra, e serve perche' `wired`
 *     e' per FAMIGLIA: accendere un formato accenderebbe anche i suoi siti a
 *     N=32 (i 48 `ssm_alpha`/`ssm_beta` del 4B) se non ci fosse.
 *
 * L'ORDINE DELLA COLPA NON E' INDIFFERENTE, e resta quello di prima: PRIMA il
 * cablaggio, POI la shape. Su un formato non cablato la ragione STRUTTURALE e'
 * il cablaggio — con N buono quel sito resterebbe legacy comunque, quindi
 * accusare la shape sarebbe una risposta vera e fuorviante. Il caso [6] di
 * tests/engine-prefillgemm-nmin.test.ts pre-registra quest'ordine.
 *
 * MA LE DUE CAUSE SONO INDIPENDENTI, e dal 2026-08-17 il rifiuto del cablaggio
 * PORTA CON SE' anche quella sulla shape quando c'e'. La ragione non e'
 * estetica: un sito q2_K a N=32 che dicesse solo «formato non cablato»
 * suggerirebbe che cablare il formato lo manderebbe sulla via veloce — che e'
 * FALSO, ed e' esattamente la trappola che il predicato esiste per disinnescare.
 * Chi legge quella riga in telemetria deve vedere entrambe le cause: la
 * dominante per prima, la concorrente subito dopo.
 *
 * Tutte e tre stanno in questo file per la ragione di sempre: un secondo
 * predicato di ammissibilita' fuori di qui e' esattamente il difetto che i gate
 * strutturali sorvegliano ([w4] sul cablaggio, [s4] sulla shape).
 */
function kernelVerdict(o: {
  kind: PrefillQuantKind; K: number; N: number; M: number;
}): { splits: number; kind: PrefillGemmKind }
  | { rejected: string; from: "kernel" | "wiring" | "shape" } {
  // LA SHAPE SI CALCOLA PRIMA DI TUTTO ma NON PARLA PER PRIMA. E' una funzione
  // pura e senza effetti, quindi calcolarla in anticipo non costa niente e non
  // cambia l'ordine della colpa: serve qui sopra al ramo del cablaggio, che la
  // allega come causa concorrente, e qui sotto come causa dominante quando il
  // cablaggio non ha niente da dire.
  //
  // IL CAST, come quello piu' sotto verso il kernel: il predicato dichiara
  // `PrefillGemmKind` perche' la sua ragione nomina un formato, ma va
  // interrogato anche sui kind che un kernel non ce l'hanno — la geometria di
  // uscita non dipende dalla famiglia, e chiederglielo solo per i kind noti
  // lascerebbe il controllo IRRAGGIUNGIBILE sui formati non ancora in elenco,
  // cioe' esattamente dove la trappola aspetta.
  const shape = prefillGemmShapeOk({ kind: o.kind as PrefillGemmKind, K: o.K, N: o.N });
  // IL CABLAGGIO PRIMA DELLA GEOMETRIA, e nello stesso ordine con cui il kernel
  // mette il formato prima del K: su un formato non cablato con K storto la
  // ragione strutturale e' il cablaggio — il K sarebbe una risposta vera e
  // inutile, perche' anche con K buono quel sito resterebbe legacy.
  if (isPrefillGemmKind(o.kind)) {
    const w = prefillGemmWiring(o.kind);
    if (!w.wired) {
      return {
        from: "wiring",
        rejected: `il formato "${o.kind}" ha il suo moltiplicatore multi-riga in produzione, ma `
          + `NON e' cablato: il piano instrada solo i kind con \`wired: true\` in `
          + `PREFILL_GEMM_SPEC (kernels/wgsl.ts), e questo non lo e' — «${w.why}»`
          // LA CAUSA CONCORRENTE, quando c'e'. Senza questa coda, cablare il
          // formato sembrerebbe bastare — e su questi siti non basta.
          + (shape.ok ? "" : ` — E NON BASTEREBBE CABLARLO: questo sito ha anche una SHAPE che il `
            + `piano non instrada su NESSUN formato, ${shape.why}`),
      };
    }
  }
  // LA SHAPE SUBITO DOPO IL CABLAGGIO, e prima di ogni domanda al kernel.
  //
  // Il predicato NON vive piu' qui in linea: ha un nome, una firma e una sede
  // sola — `prefillGemmShapeOk`, sopra — e questo e' il suo UNICO lettore. La
  // ragione dell'estrazione non e' estetica: un controllo in linea non si puo'
  // interrogare da un test senza far girare tutto il piano, e non si puo'
  // sorvegliare con un gate strutturale che pretenda che di predicati sulla
  // shape ce ne sia uno solo (tests/engine-prefill-q2k-q3k.test.ts, [s4]).
  if (!shape.ok) return { from: "shape", rejected: shape.why };
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
    return { from: "kernel", rejected: e instanceof Error ? e.message : String(e) };
  }
  // Il kernel NON ha rifiutato ⇒ il kind e' uno dei suoi. Il type guard non e'
  // cerimonia: e' cio' che permette di passare il kind REALE a
  // `prefillGemmSplitsFor` senza un secondo cast, e interroga l'elenco
  // ESPORTATO dal kernel, non una copia locale.
  if (!isPrefillGemmKind(o.kind)) {
    return {
      from: "kernel",
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
  /**
   * IL REGIME CHE CHIEDE LA ROTTA. Default `"prefill"`, cioe' esattamente il
   * comportamento di prima: chi non lo passa non vede alcuna differenza.
   *
   * ESISTE PER UNA RAGIONE SOLA, e conviene scriverla per esteso perche' tocca
   * una clausola che appartiene a un altro goal.
   *
   * `PREFILL_M1_LEGACY` rende legacy OGNI kind a M=1. Il suo stesso commento
   * dichiara che la clausola e' **conservativa e non derivata**: al banco, a
   * M=1, l'unica cella piu' lenta della legacy e' `q5_K` sulla via f32 (0,56x),
   * mentre `q5_K/idot` fa 3,47x e `q4_0/idot` 8,64x PIU' VELOCI. E aggiunge:
   * «restringerla alla cella misurata e' una decisione del PI, non
   * dell'implementatore».
   *
   * **Questo parametro NON la restringe, e non tocca il prefill di una virgola:
   * la SCOPA al regime che l'ha scritta.** La clausola sta nel done-when della
   * riga 2 di `engine-ttft`, che parla di chunk di prefill; il decode e' un
   * altro regime, con un'altra domanda e un'altra misura. Con
   * `regime: "prefill"` (il default, e quello che tutti i chiamanti di prima
   * usano) il comportamento e' identico byte per byte.
   *
   * LA MISURA CHE AUTORIZZA IL DECODE A M=1 — pre-registrata prima di guardare
   * i numeri, e contro il kernel che il decode emette DAVVERO (non contro la
   * forma del prefill, che era l'errore di it.13):
   *
   *     q8_0 K2048 N=8192   gemv 0,1324 ms (23,4% del picco) -> 0,0340   3,89x
   *     q8_0 K2048 N=4096   gemv 0,1616 ms ( 9,6% del picco) -> 0,0489   3,30x
   *
   * (`results/microbench/velocita-decode-2d-4090-linux-2026-08-16T02-56-25-413Z.json`,
   * graduatoria in `docs/deep-dive/velocita-decode-2d-prereg-2026-08-16.md`.)
   *
   * E la RAGIONE STRUTTURALE della clausola non si applica qui. Quella dice: «a
   * M=1 non c'e' riuso dei pesi da ammortizzare, che e' l'intero punto della
   * forma multi-riga». Vero — ma la forma split-K a M=1 non vince per riuso dei
   * pesi: vince per BANDA. Il GEMV a un workgroup per riga sta al 9,6-23,4% del
   * picco, lo split-K arriva al 91,0%. Spezzare il K da' a ogni workgroup un
   * accesso contiguo piu' lungo, e quello paga anche con una riga sola.
   *
   * Quello che NON e' cambiato: il predicato sulla SHAPE (`prefillGemmShapeOk`,
   * letto da `kernelVerdict`) vale in entrambi i regimi. `ssm_alpha`/`ssm_beta`
   * a N=32 restano legacy nel decode esattamente come nel prefill, su ogni
   * famiglia.
   */
  regime?: "prefill" | "decode";
}): PrefillGemmRoute {
  checkGeom(o, "planPrefillGemm");

  // M=1 prima di qualunque domanda al kernel: non e' la shape a non andare, e'
  // l'M. La costante porta con se' i numeri veri e il perche' resta com'e'.
  // Vale nel regime che l'ha scritta — il prefill — e non nel decode: v. il
  // parametro `regime` qui sopra, che porta la misura che lo autorizza.
  if (o.M === PREFILL_M1_LEGACY.M && (o.regime ?? "prefill") !== "decode") {
    return {
      via: "legacy",
      reason: PREFILL_M1_LEGACY.reasonFor(o),
      splits: 0, partialFloats: 0, wgStorageBytes: 0, xqU32: 0, xscF32: 0,
    };
  }

  // Il pavimento del DECODE, e vale SOLO li': v. `DEC_SPLITK_MIN_N`. Sta qui e
  // non in `kernelVerdict` perche' non e' una proprieta' del kernel — il kernel
  // su N=512 e' corretto — ma di quale regime ci guadagna.
  if ((o.regime ?? "prefill") === "decode" && o.N < DEC_SPLITK_MIN_N) {
    return {
      via: "legacy",
      reason: `N=${o.N} sotto il pavimento del DECODE (${DEC_SPLITK_MIN_N}): a M=1 questa forma `
        + `e' stata MISURATA perdente (q8_0 K=2048 N=512 da' 0,92x contro il gemv del decode), `
        + `mentre sopra il pavimento vince da 2,45x a 3,80x. Il PREFILL non ha questo limite: `
        + `a M righe c'e' riuso dei pesi da ammortizzare e la forma vince comunque`,
      splits: 0, partialFloats: 0, wgStorageBytes: 0, xqU32: 0, xscF32: 0,
    };
  }

  const verdict = kernelVerdict(o);
  if ("rejected" in verdict) {
    // TRE RIFIUTI DIVERSI, tre ragioni diverse. Quello del KERNEL dice «questa
    // shape non si moltiplica cosi'»; quello della SHAPE dice «si
    // moltiplicherebbe correttamente, ma non conviene e non e' mai stata
    // misurata li'»; quello del CABLAGGIO dice «questa forma si
    // moltiplicherebbe benissimo, ma in produzione non ci si passa». Dare a due
    // di loro la stessa frase manderebbe in telemetria una diagnosi falsa: si
    // leggerebbe un problema di geometria dove il kernel non c'entra niente.
    //
    // IL TERZO RAMO E' NUOVO, e toglie una bugia che c'era prima: il rifiuto su
    // N usciva marcato `kernel` e la frase mandava a leggere `prefillGemmCheck`
    // — un file che su N non decide niente e che nel suo commento dice proprio
    // di NON deciderlo. Chi fosse andato a cercarci la soglia non l'avrebbe
    // trovata. Ora la frase indica la sede vera: `prefillGemmShapeOk`, qui.
    return {
      via: "legacy",
      reason: verdict.from === "wiring"
        ? `la via veloce di prefill NON e' cablata su questo formato ⇒ si resta su legacy `
          + `(M gemv replicate, riuso pesi zero), che e' cio' che il motore emette oggi. `
          + `Il rifiuto arriva dal FLAG DI CABLAGGIO, non dal contorno del kernel: la forma `
          + `esiste, semplicemente nessun sito ci passa — «${verdict.rejected}»`
        : verdict.from === "shape"
          ? `la SHAPE di questo sito non e' instradabile ⇒ si resta su legacy (M gemv replicate, `
            + `riuso pesi zero). Il rifiuto NON arriva dal kernel — a questa shape il kernel e' `
            + `CORRETTO — ma dal PIANO, che e' l'unico posto dove si decide se una forma CONVIENE: `
            + `prefillgemmplan.ts, prefillGemmShapeOk — «${verdict.rejected}»`
          : `il moltiplicatore multi-riga di prefill NON accetta questa shape ⇒ si resta su legacy `
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
