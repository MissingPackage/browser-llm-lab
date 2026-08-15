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

import {
  attnDecodeWorkgroupStorageBytes, prefillGemmWorkgroupStorageBytes,
  qwenFusedChunkWorkgroupStorageBytes, qwenGemvResidualWorkgroupStorageBytes,
  PREFILL_GEMM_KINDS,
  type PrefillGemmOpts, type PrefillGemmKind,
} from "./kernels/wgsl";
import { mlaPartialsLen } from "./mlasplit";
import { PREFILL_M, PREFILL_M_DENSE05B } from "./prefillplan";
import { GLM47_FLASH, QWEN25_05B } from "./shape";

// ---------------------------------------------------------------------------
// Costanti derivate dall'inventario del parco kernel (C3a it.6).
// Ognuna è ri-verificata dal test scansionando src/engine/kernels/wgsl.ts.
// ---------------------------------------------------------------------------

/** Il `workgroup_size` più grande del parco: rmsnormWgsl e argmaxStage1/2. */
export const MAX_WORKGROUP_SIZE = 256;

/**
 * Storage binding nel bind group più affollato fra quelli che il WGSL dichiara
 * LETTERALMENTE: rmsPairGemvSiluFast (Qwen). Si chiamava
 * `MAX_STORAGE_BINDINGS_PER_STAGE`: il nome nuovo dice che è il massimo dei
 * binding STATICI, perché dalla fase 4 esistono kernel il cui numero di binding
 * è generato (l'arena expert: `expertArenaBindings`) e quel termine non si
 * scansiona dal sorgente.
 */
export const MAX_STATIC_STORAGE_BINDINGS = 7;

/**
 * Tetto di progetto sui buffer d'arena di UNA classe (C3a fase 4, strato 1).
 * A finestra 2 GiB servono 6+1 binding a budget 12 GiB e 7+1 a parco completo
 * (fase 4c): 8 è il margine, non una costante difensiva. Oltre, `ld4` avrebbe
 * più archi di quanti binding il device conceda.
 */
export const ARENA_BUFFERS_MAX = 8;

/**
 * Storage binding di una pipeline expert in modo arena: gli nBuf buffer
 * d'arena + x + out + selBuf. L'uniform `MoeIdx` NON entra nel conto:
 * `maxStorageBuffersPerShaderStage` conta i soli storage.
 */
export const expertArenaBindings = (nBuf: number): number => nBuf + 3;

/**
 * Workgroup storage del path Qwen2.5-0.5B: il MASSIMO fra i suoi consumatori,
 * CALCOLATO dalle formule che vivono accanto ai kernel.
 *
 * ERA UN 30_848 SCRITTO A MANO, e il numero era giusto per COINCIDENZA (it.24).
 * Il commento diceva «il consumatore massimo è questo», al singolare; misurando
 * il WGSL generato di ogni pipeline che `gpuforward.ts` crea, i consumatori
 * sopra i 16.384 B garantiti da WebGPU sono QUATTRO:
 *
 *   30.848  rmsPairGemmSiluChunkFast  K=896  mMax=8   (l'unico che era dichiarato)
 *   30.720  rmsGemmQkvChunkFast       K=896  mMax=8
 *   30.720  gemmResidChunkFast        K=896  mMax=8
 *   19.712  gemvResidualFast          K=4864 M=1      — path di DECODE
 *
 * I 30.848 erano semplicemente il massimo dei quattro. Il pericolo non era il
 * valore di oggi: era che bastasse portare `pairSilu` alla forma multi-riga
 * perché la costante scendesse mentre `rmsQkv` continuava a chiederne 30.720.
 * Il motore avrebbe **sotto-chiesto** `maxComputeWorkgroupStorageSize` — che
 * `limitsFor` usa come `requiredLimits`, cioè come TETTO del device — e
 * `createComputePipeline` sarebbe fallito in validazione su ogni device, 4090
 * compreso. Un massimo calcolato non può sbagliarsi così.
 *
 * L'ultimo termine non è raggiungibile dalla forma multi-riga: è il down-proj
 * del DECODE, gira a M=1 e non è un GEMM di prefill. Finché c'è lui, nessun
 * porting del prefill può portare questo path sotto la garanzia WebGPU.
 */
export const QWEN_WORKGROUP_STORAGE_BYTES = Math.max(
  // i tre kernel di chunk (K = dModel) e il down-proj del decode (K = dFfn):
  // le shape arrivano da `QWEN25_05B`, non da due numeri ricopiati qui
  qwenFusedChunkWorkgroupStorageBytes({ K: QWEN25_05B.dModel, mMax: PREFILL_M_DENSE05B }),
  qwenGemvResidualWorkgroupStorageBytes(QWEN25_05B.dFfn),
);

/**
 * Shape con cui si INTERROGA `prefillGemmWorkgroupStorageBytes` — quella che la
 * riga 1 del goal engine-ttft ha misurato (K2560xN9216, 4 fette da BK=2).
 *
 * K, N e `splits` NON entrano nel fabbisogno: il moltiplicatore tiene in memoria
 * di gruppo le sole attivazioni della tile, quindi il termine dipende dal solo
 * M. Sono qui perché la formula VALIDA i suoi argomenti (q4_0, K multiplo di 64,
 * blocchi divisibili per le fette) e rifiuta una shape inventata — ed è giusto
 * così: un fabbisogno si chiede a un kernel che potrebbe esistere. Il test
 * asserisce l'indipendenza da K/N/splits, così se un domani la tilatura cambia
 * e la shape inizia a contare, se ne accorge qualcuno.
 */
const prefillGemmShape = (M: number): PrefillGemmOpts =>
  ({ kind: "q4_0", K: 2560, N: 9216, M, splits: 4 });

/**
 * Shape del GEMM di prefill Q5_K — e qui, al contrario di quella q4_0, la shape
 * NON è una qualsiasi valida: è quella VERA di `ssm_out`, il tensore che la
 * riga 5 del goal engine-ttft ha misurato come collo del prefill (un Q5_K
 * rimasto sul percorso vecchio, K4096xN2560, 4 fette).
 *
 * Come per il gemello q4_0, K/N/`splits` non entrano nel fabbisogno — la
 * memoria di gruppo tiene le sole attivazioni della tile, quindi il termine
 * dipende dal solo M — e sono qui perché la formula VALIDA i suoi argomenti e
 * rifiuta una shape inventata. Scriverci la shape vera invece di una qualunque
 * valida costa nulla e dice a chi legge QUALE kernel si sta dichiarando.
 *
 * ATTENZIONE, il q5_K non è il q4_0 con un bit in più: le due vie si INVERTONO.
 * Sul q4_0 la via f32 è la più cara (256·M contro 72·M); qui è la via INTERA
 * (320·M contro 128·M), perché il blocco Q5_K porta scala E minimo per ogni
 * sotto-blocco da 32 e la via intera deve tenerne i parziali. Chi negozia i
 * limiti prima di sapere quale via girerà deve percio' chiedere 320·M — ed è
 * quello che il default (`prefillGemmIdot` omesso) fa, perché la formula
 * risponde il peggiore delle due. Il numero non si ricopia mai: si chiede a
 * `prefillGemmWorkgroupStorageBytes`, che vive accanto al kernel.
 */
const prefillGemmQ5KShape = (M: number): PrefillGemmOpts =>
  ({ kind: "q5_K", K: 4096, N: 2560, M, splits: 4 });

/**
 * UNA SHAPE REALE PER OGNI KIND, e il `Record` e' il punto (goal engine-kquant
 * riga 7). Fino al 2026-08-15 qui c'erano DUE termini scritti a mano — q4_0 e
 * q5_K — mentre i kind del prefill erano diventati sei: il `q4_1` **cablato e
 * in produzione** non aveva un termine proprio nel `Math.max`, e non l'avevano
 * i tre formati portati per il 35B.
 *
 * Perche' non bastava aggiungerne uno a mano: il termine q5_K sfonda il tetto a
 * M=97 ed e' pinnato nei test, ma il q6_K (352·M) lo sfonderebbe a **M=88** —
 * PRIMA — e nessun test se ne sarebbe accorto. Oggi e' inerte (5.632 B contro i
 * 30.848 del path fuso 0.5B, che domina) e lo e' finche' quei kind restano
 * `wired: false`; il giorno del cablaggio non lo sarebbe piu'.
 *
 * `Record<PrefillGemmKind, …>` e' la stessa garanzia di `PREFILL_GEMM_SPEC`:
 * chi allunga `PREFILL_GEMM_KINDS` senza scrivere la shape qui sotto non
 * ottiene un tetto sbagliato in silenzio, ottiene un errore di compilazione.
 *
 * Le shape sono QUELLE VERE, non una qualunque valida: K/N/`splits` non entrano
 * nel fabbisogno (la memoria di gruppo tiene le sole attivazioni della tile,
 * quindi il termine dipende dal solo M) ma la formula VALIDA i suoi argomenti,
 * e scrivere la shape vera dice a chi legge QUALE kernel si sta dichiarando.
 * Sono le stesse shape dei casi ktest, cioe' le sole misurate.
 */
const PREFILL_GEMM_SHAPES: Record<PrefillGemmKind, (M: number) => PrefillGemmOpts> = {
  q4_0: prefillGemmShape,
  q5_K: prefillGemmQ5KShape,
  // `blk.0-3.ffn_down` del 4B: 288 blocchi da 32, 4 fette a BK=2.
  q4_1: (M) => ({ kind: "q4_1", K: 9216, N: 2560, M, splits: 4 }),
  // expert `ffn_gate`/`ffn_up` del 35B: 8 superblocchi da 256, 4 fette.
  q4_K: (M) => ({ kind: "q4_K", K: 2048, N: 512, M, splits: 4 }),
  // expert `ffn_down` del 35B: 2 superblocchi, 2 fette — la fetta piu' corta
  // che il piano possa produrre, e il termine piu' RIPIDO dei sei (352·M).
  q6_K: (M) => ({ kind: "q6_K", K: 512, N: 2048, M, splits: 2 }),
  // `attn_*` del 35B: 64 blocchi da 32, 4 fette, PER pari (il ciclo avanza a 2).
  q8_0: (M) => ({ kind: "q8_0", K: 2048, N: 4096, M, splits: 4 }),
};

/**
 * mlaAttnDecode (kernel MLA MONOLITICO): scores[ctxMax] + red[64].
 * CONSUMATORE, dal 2026-08-03: `glmforward` (path per-layer, usato da glmroute)
 * e i ktest. NON più `glmmodel`: il forward di produzione è passato
 * all'attention split (fase 4b), il cui fabbisogno è costante in ctxMax
 * (`mlaSplitWorkgroupStorageBytes` in mlasplit.ts). Il termine resta qui perché
 * il monolitico è ancora eseguito da quei due consumatori: toglierlo o
 * condizionarlo è una decisione da docket, non un ritocco.
 */
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
   * L'attention MLA (solo GLM) tiene `scores[ctxMax]` in workgroup memory.
   * `false` toglie QUELLA voce — non toglie il fabbisogno dell'attenzione di
   * Qwen, che e' contata SEMPRE (goal engine-kernel-decode, docket item 2: il
   * commento di prima diceva "un consumatore che quel modello non ha", e non
   * era vero — il path q35 otteneva il limite giusto solo perche' nessuno
   * passava `false`).
   * Dalla fase 1 di quel goal la voce di Qwen NON cresce piu' col contesto: il
   * decode fa softmax in streaming e il suo workgroup storage e' costante
   * (`attnDecodeWorkgroupStorageBytes`, 1.536 B a headDim 256). Resta contata
   * comunque — un requisito si dichiara perche' esiste, non perche' e' il
   * massimo: QUELLA voce la batte `QWEN_WORKGROUP_STORAGE_BYTES`, mentre il
   * massimo complessivo puo' benissimo venire dall'MLA quando non e' spenta.
   */
  mlaAttention?: boolean;
  /**
   * Righe per chunk del GEMM multi-riga del prefill (`prefillGemmQ4SplitK*`).
   * Default: `PREFILL_M` del piano — qui non si ricopia quel numero, si importa.
   * Il fabbisogno di memoria di gruppo di quel kernel è LINEARE in M e costante
   * in ctxMax: è l'unica cosa di questo file che cambia se cambia la tilatura
   * del prefill.
   * ATTENZIONE al default: `PREFILL_M` è l'M NOMINALE del piano, e oggi nessun
   * percorso di prodotto lo legge — il prefill denso gira a
   * `PREFILL_M_DENSE05B` (8) e quello GLM a `GLM_PREFILL_M` (16). Vale 16 come
   * il maggiore dei due, quindi il default dichiara abbastanza per entrambi, e
   * un test lo sorveglia; l'M vero lo fisserà l'assemblatore della riga 2, che
   * dovrà passarlo esplicitamente.
   */
  prefillM?: number;
  /**
   * Quale via del GEMM di prefill girerà: `true` = intera
   * (`packed_4x8_integer_dot_product`, 1.152 B a M=16), `false` = f32
   * (4.096 B). Lo decide la language feature a RUNTIME.
   * OMETTERLO (il default) dichiara il PEGGIORE delle due — che è il numero
   * giusto per chi negozia i limiti prima di sapere quale via avrà: chiedere
   * 1.152 e ritrovarsi sulla via f32 significa pipeline invalida su un device
   * che concede esattamente il richiesto.
   */
  prefillGemmIdot?: boolean;
  /** Byte della KV cache di UN layer, se bindata intera. Default: formula GLM. */
  kvBytesPerLayer?: number;
  /**
   * Binding che non vengono dal modello di produzione ma dall'harness che lo
   * ospita (es. ktest binda i pesi densi veri di blk.0). Ognuno dichiara il
   * proprio consumatore, come tutti gli altri requisiti.
   */
  extraBindings?: Array<{ bytes: number; consumer: string }>;
  /**
   * Buffer d'arena della classe expert più frammentata (C3a fase 4, strato 1).
   * Ogni pipeline expert li binda TUTTI: è il termine che alza
   * `maxStorageBuffersPerShaderStage` sopra i 7 del path Qwen.
   * Il valore lo calcola `arenaNeeds` (residency.ts), che conosce il riparto
   * degli slot — qui non si ricopia quell'aritmetica.
   */
  arenaBuffers?: number;
  /**
   * Byte del buffer d'arena più grande. È IL requisito che alza il binding size
   * oltre i 250 MiB della testa Q6_K: con l'arena il binding non è più il
   * sotto-range di uno slab (~1,5 MB) ma il buffer intero.
   */
  arenaWindowBytes?: number;
}

/** I requisiti del motore, ciascuno col suo consumatore. */
export function engineNeeds(o: EngineNeedsOpts): LimitNeed[] {
  // Il GEMM multi-riga del prefill (goal engine-ttft riga 2): formula IMPORTATA
  // dal file del kernel, mai ricopiata — regola già vigente in questo file.
  const prefillM = o.prefillM ?? PREFILL_M;
  const prefillGemmVia = o.prefillGemmIdot === undefined
    ? undefined : o.prefillGemmIdot ? "idot" as const : "f32" as const;
  const prefillGemmViaLabel = prefillGemmVia ? `via ${prefillGemmVia}` : "peggiore fra idot e f32";
  // UN TERMINE PER FORMATO — e non coincidono fra loro, che è la ragione per cui
  // sono termini distinti e non uno solo: sul q4_0 la via cara è la f32 (256·M
  // contro 72·M), sul q5_K e sul q4_K si INVERTE (320·M sulla via intera, che
  // deve tenere i parziali di scala E minimo per ogni sotto-blocco da 32), e il
  // q6_K è il più ripido di tutti (352·M, il `xh` in più). Confonderli o
  // sommarli dichiarerebbe un kernel che non esiste: due GEMM di prefill non
  // girano mai insieme nello stesso workgroup, quindi il tetto lo negozia il
  // PIÙ ALTO. I valori che seguono sono gli stessi che entrano nel `Math.max` e
  // gli stessi che compaiono nel `consumer`: si calcolano una volta.
  // TUTTI i kind, non i due scritti a mano: un termine per formato, dalla stessa
  // formula e con la stessa via. Il piu' alto e' quello che negozia il tetto —
  // sommarli dichiarerebbe un kernel che non esiste (due GEMM di prefill non
  // girano mai insieme nello stesso workgroup).
  const prefillGemmPerKind = PREFILL_GEMM_KINDS.map((k) => ({
    kind: k,
    bytes: prefillGemmWorkgroupStorageBytes(PREFILL_GEMM_SHAPES[k](prefillM), prefillGemmVia),
  }));
  const prefillGemmMaxBytes = Math.max(...prefillGemmPerKind.map((x) => x.bytes));
  // I DUE rami di `attnDecodeWgsl` — decode e batch (prefill a chunk) — hanno
  // la STESSA formula dal task T1-kernel-batch-streaming. Un solo termine, letto
  // una volta sola dal file del kernel: due chiamate identiche nel `Math.max`
  // sarebbero una la copia dell'altra, e il giorno in cui una delle due forme si
  // scostasse la copia non se ne accorgerebbe comunque. Il ramo batch si vede
  // nel `consumer`, che e' dove un consumatore si DICHIARA; il `Math.max` e'
  // dove si CONTA, e contarlo due volte non cambia un byte.
  const attnDecodeBytes = attnDecodeWorkgroupStorageBytes(o.ctxMax);
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
      limit: "maxStorageBuffersPerShaderStage", value: MAX_STATIC_STORAGE_BINDINGS, hard: true,
      consumer: "rmsPairGemvSiluFastWgsl / rmsPairGemmSiluChunkFastWgsl (7 storage)",
    },
    {
      limit: "maxComputeWorkgroupStorageSize",
      // QUATTRO consumatori CONTATI. Il terzo era `attnDecodeWgsl`,
      // l'attenzione di decode di Qwen: dalla fase 1 (goal
      // engine-kernel-decode, docket item 2) il suo fabbisogno e' COSTANTE in
      // ctxMax — softmax in streaming — ma si conta lo stesso. Il quarto e'
      // il GEMM multi-riga del prefill (`prefillGemmQ4SplitK*`, goal
      // engine-ttft riga 2), che tiene le attivazioni della tile in memoria di
      // gruppo: 72·M B sulla via intera, 256·M B su quella f32, costante in
      // ctxMax. Entrambe le formule arrivano dal file del kernel, non sono
      // ricopiate qui.
      // Il GEMM del prefill NON alza il totale (4.096 B a M=16 contro i 30.848
      // del path fuso): si dichiara perche' ESISTE, non perche' vince — un
      // binding o un fabbisogno che non e' nella lista e' un requisito che
      // nessuno sta guardando. Entra comunque nel `Math.max`, ed e' li' che il
      // test lo verifica: `limitsFor` legge `value`, non `consumer`.
      //
      // QUINTO consumatore, DICHIARATO DA OGGI: il ramo `batch` di
      // `attnDecodeWgsl` — l'attenzione a chunk del prefill di q35gpumodel.
      // Chiedeva 4·ctxMax + 256 B (`scores[ctxMax]` in memoria di gruppo) e per
      // questo NON si dichiarava: a ctxMax 12.224 quel termine toccava i 49.152
      // B che il device di riferimento concede, e dichiararlo avrebbe fatto
      // fallire `limitsFor` dove passava. Il task T1-kernel-batch-streaming ha
      // tolto quel ramo dal legacy: softmax in streaming, stessa
      // `attnDecodeWorkgroupStorageBytes` del decode, COSTANTE in ctxMax. Ora
      // si conta — e contarlo non alza di un byte il requisito HARD, che e'
      // esattamente il punto: la garanzia sta nel describe "garanzia: il ramo
      // batch dell'attenzione e' contato e non alza il requisito" di
      // tests/gpulimits.test.ts, dove prima stava il sensore del debito.
      //
      // DEBITO CHE RESTA, esplicito e con la sua aritmetica: un `scores[ctxMax]`
      // in workgroup memory esiste ancora, in produzione e fuori dall'MLA, e NON
      // e' contato qui. Lo dichiara `attnPrefillChunkWgsl` (qh[headDim] + quel
      // `scores` + red[64] = 4·ctxMax + 1.280 B a headDim 256), che gpuforward.ts
      // istanzia dallo STESSO path che qui passa `mlaAttention: false` — il path
      // di conformita' 0.5B. Il pareggio col termine fuso e' a ctxMax 7.392
      // (4·7.392 + 1.280 = 30.848). Lo tiene fuori scope la clausola e2a della
      // riga 4; il suo pareggio resta asserito nei test.
      //
      // SESTO consumatore (task limits-q5k): il GEMM di prefill Q5_K, cioè
      // `ssm_out` — il tensore che la riga 5 ha misurato come collo del
      // prefill, rimasto sul percorso vecchio proprio perché la via veloce era
      // q4_0-only. Vale 320·M B sulla via intera e 128·M sulla f32 (le due vie
      // si invertono rispetto al q4_0: v. `prefillGemmQ5KShape`). A M=16 sono
      // 5.120 B contro i 30.848 del path fuso: come il gemello q4_0, si
      // dichiara perché ESISTE, non perché vince — e come lui entra nel
      // `Math.max`, perché `limitsFor` legge `value` e non `consumer`.
      // La soglia in cui alzerebbe il tetto è M=97 (320·97 = 31.040 > 30.848),
      // più bassa dei 121 del q4_0 perché il termine è più ripido: è pinnata
      // nei test, così un domani che alzasse la tilatura del prefill se ne
      // accorge chi la alza e non il device in validazione di pipeline.
      value: Math.max(
        QWEN_WORKGROUP_STORAGE_BYTES,
        prefillGemmMaxBytes, // TUTTI i kind del prefill, non i due scritti a mano
        attnDecodeBytes, // decode E batch: una formula sola (vedi sopra)
        o.mlaAttention === false ? 0 : mlaWorkgroupStorageBytes(o.ctxMax),
      ),
      hard: true,
      consumer: `max(rmsPairGemmSiluChunkFast ${QWEN_WORKGROUP_STORAGE_BYTES} B, `
        + prefillGemmPerKind.map((x) =>
          `prefillGemm ${x.kind} splitK M=${prefillM} (${prefillGemmViaLabel}) = ${x.bytes} B`).join(", ")
        + `, attnDecode (streaming, costante in ctxMax) = ${attnDecodeBytes} B, attnDecode batch (prefill a chunk, streaming, costante in ctxMax) = ${attnDecodeBytes} B${o.mlaAttention === false ? "" : `, mlaAttnDecode ${mlaWorkgroupStorageBytes(o.ctxMax)} B`} a ctxMax ${o.ctxMax})`,
    },
  ];
  if (o.arenaBuffers !== undefined) {
    if (o.arenaBuffers > ARENA_BUFFERS_MAX) {
      throw new Error(
        `gpulimits: ${o.arenaBuffers} buffer d'arena > ARENA_BUFFERS_MAX ${ARENA_BUFFERS_MAX} ` +
        "(ridurre la classe o allargare la finestra)");
    }
    needs.push({
      limit: "maxStorageBuffersPerShaderStage", value: expertArenaBindings(o.arenaBuffers), hard: true,
      consumer: `catena expert arena, ${o.arenaBuffers} binding d'arena + x + out + selBuf`,
    });
  }
  // Il binding singolo più grande: testa Q6_K, la KV di un layer, il buffer
  // partials dell'attention split, o un binding dichiarato dall'harness.
  const kvBytes = o.kvBytesPerLayer ?? glmKvBytesPerLayer(o.ctxMax);
  const candidates: Array<{ bytes: number; consumer: string }> = [
    { bytes: kvBytes, consumer: `KV cache di un layer a ctxMax ${o.ctxMax}` },
    ...(o.head ? [{ bytes: q6kHeadBytes(o.head.vocab, o.head.dModel), consumer: "output.weight Q6_K repacked" }] : []),
    ...(o.mlaAttention === false ? [] : [{
      // formula IMPORTATA da mlasplit.ts, non ricopiata: è lo stesso sizing con
      // cui glmmodel alloca il buffer. Oggi non vince mai (la testa Q6_K è 250
      // MiB contro ~1,3 MiB a ctxMax 525), ma la regola del file è che ogni
      // binding del motore compaia col suo consumatore: un binding che non è
      // nella lista è un requisito che nessuno sta guardando.
      bytes: mlaPartialsLen(GLM47_FLASH.nHead, GLM47_FLASH.kvLora, o.ctxMax) * 4,
      consumer: `attnPartials dello split MLA a ctxMax ${o.ctxMax} (glmmodel)`,
    }]),
    ...(o.arenaWindowBytes ? [{
      bytes: o.arenaWindowBytes,
      consumer: `finestra d'arena ExpertCache: un buffer di classe bindato intero (${(o.arenaWindowBytes / 2 ** 20).toFixed(0)} MiB)`,
    }] : []),
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
