// Conformance logits full-model Qwen3.5-4B (q1 fase 4, chiusura): replay
// teacher-forced dei golden full-corpus (prompt + 128 greedy dell'oracolo)
// nell'orchestratore GPU; per ogni posizione golden: top-1 del motore vs
// argmax dell'oracolo. Il rate misurato FISSA la soglia ratchet del 4B
// (spec §5 punto 3: mai import del PIN GLM).
//
// Prefill = step sequenziali SENZA readback (read=false: si accodano);
// readback solo alle posizioni golden. Pattern glmconf; modello via fetch
// Range per-tensore (it.8).
import { createEngineDevice } from "../gpudevice";
import { parseGguf, type GgufTensorInfo } from "../gguf";
import { createQ35GpuModel, q35TensorBytes } from "../q35gpumodel";
import { q35MoeConfig } from "../q35expertstore";
import { arenaNeeds } from "../residency";
import { validateQwen35 } from "../q35shape";

interface GoldenPos { argmax: number; top: Array<[number, number]> }
interface GoldenPrompt { id: string; file: string; promptTokens: number[]; generated: number[]; positions: GoldenPos[] }
interface Golden { modelSha256: string; oracle: { commit: string }; corpusHash: string; prompts: GoldenPrompt[] }
interface Cfg {
  prompts?: number[];
  maxGen?: number;
  /** modalità BENCH (it.10): riferimenti decode/prefill/TTFT full-resident. */
  bench?: { promptIdx: number; nDecode: number };
  /** modello (it.11): default 4b. */
  model?: "4b" | "9b" | "35b";
  /** budget arena expert in GiB (solo MoE; default 12) */
  arenaGiB?: number;
  /** DEBUG (it.17): dump hidden dopo il layer N sul PRIMO token, poi stop. */
  debugTap?: number;
  /** fase-D 3b fetta 3b: router GPU in OMBRA, con report di fedelta'. */
  routerShadow?: boolean;
  /** fase-D 3b, prima della 3c: profilo dei MISS per token (2 passate). */
  missTrace?: boolean;
  /**
   * fase-D 3b fetta 3c: GATE del path a submit unico. Due passate sullo STESSO
   * prompt e sulla STESSA cache — la prima col path sync di oggi (che scalda),
   * la seconda con quello ottimistico — e i numeri riportati SEPARATI per
   * passata. Mediarli nasconderebbe il fenomeno misurato in it.16 (a freddo
   * ogni token è sporco, a caldo nessuno).
   */
  optTrace?: boolean;
  /**
   * fase-D 3b fetta 3c-bis: la passata FREDDA la fa il path OTTIMISTICO invece
   * di quello sync. È il test del repair+replay nel suo regime peggiore — a
   * cache vuota ogni token è sporco (it.16: 39/39) — e va in un run a parte
   * perché la cache fredda esiste una volta sola per processo.
   */
  optCold?: boolean;
  /**
   * fase 4 (it.19): decomposizione del tempo GPU del token per categoria
   * (statico / router / expert / coda). Perturba la misura (spezza i pass) e
   * il report lo dichiara confrontando il ms/token con e senza sonda.
   */
  gpuTime?: boolean;
  /** SOLO MISURA (fase 4-ter): senza snapshot dello stato ricorrente. */
  noStateSnapshot?: boolean;
  /**
   * fase 4: GATE del prefill a chunk. Confronta i logits di `prefillChunk` con
   * quelli del `step()` sequenziale sullo stesso prompt — l'atteso e' la
   * BIT-IDENTITA', perche' ogni kernel batched e' ktestato bit-identico per
   * riga contro il suo per-riga (it.25-26-30-31).
   */
  prefillM?: number;
  /**
   * it.21: la CONFORMANCE golden prefilla il prompt a chunk invece che con
   * `step()`. Flag SEPARATO da `prefillM` di proposito: quello attiva il gate
   * dei due bracci ed esce, questo lascia girare il replay golden e cambia solo
   * COME si riempie la KV cache. Serve a rispondere alla domanda pratica —
   * quanto costa la via intera in token sbagliati contro l'oracolo — che il
   * confronto fra bracci non puo' rispondere.
   */
  confPrefillM?: number;
  /** fase 5 (docket item 11): tetto VRAM, da cui il budget expert si DERIVA. */
  vramGiB?: number;
  /**
   * fase 5 (it.36): i DUE path misurati entrambi a FREDDO nello stesso
   * processo, con la cache svuotata fra i bracci. E' il confronto che la
   * policy d'ingresso esige e che finora stava in due run diversi (it.18).
   */
  coldBoth?: boolean;
  /** fase 5: policy di residenza ("lru" default, "tier" = LRU + AUTOPIN). */
  expertPolicy?: "lru" | "tier";
}

/** riga di report di UNA passata del gate 3c: i per-token accanto ai totali. */
const pass2json = (
  r: {
    submits: number; readbacks: number; hits: number; misses: number; ms: number;
    dirtyTokens: number; replays: number; replayLayers: number; repairMs: number;
    encodeMs: number; embedMs: number; argmaxMs: number; tailCpuMs: number;
    readbackMs: number; tokenMs: number;
  },
  n: number,
) => ({
  submits: r.submits, submitsPerToken: r.submits / n,
  readbacks: r.readbacks, readbacksPerToken: r.readbacks / n,
  hits: r.hits, misses: r.misses,
  msPerToken: r.ms / n,
  dirtyTokens: r.dirtyTokens, replays: r.replays, replayLayers: r.replayLayers,
  repairMs: r.repairMs,
  // FASE 4-TER: il token fuori dai pass GPU, per voce e per token
  cpu: {
    encodeMs: r.encodeMs / n, embedMs: r.embedMs / n, argmaxMs: r.argmaxMs / n,
    tailCpuMs: r.tailCpuMs / n, readbackWaitMs: r.readbackMs / n, tokenMs: r.tokenMs / n,
  },
});

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const progress = (msg: string) => post({ type: "progress", msg });

// SHA = Q35_SHA256 (q35shape, pinnate in spec §1)
const MODELS = {
  "4b": { url: "/models/Qwen3.5-4B-Q4_0.gguf", file: "Qwen3.5-4B-Q4_0.gguf", sha: "298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb" },
  "9b": { url: "/models/Qwen3.5-9B-Q4_0.gguf", file: "Qwen3.5-9B-Q4_0.gguf", sha: "17670346b4260ddcb0173965145155885024f3c9a4a24389a3370751edbcde24" },
  "35b": { url: "/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", file: "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", sha: "a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c" },
} as const;
let URL_GGUF: string = MODELS["4b"].url;

async function range(off: number, len: number): Promise<Uint8Array> {
  const rr = await fetch(URL_GGUF, { headers: { Range: `bytes=${off}-${off + len - 1}` } });
  if (rr.status !== 206) throw new Error(`q35conf: Range non onorato (${rr.status})`);
  const ab = await rr.arrayBuffer();
  if (ab.byteLength !== len) throw new Error(`q35conf: Range corto ${ab.byteLength}/${len}`);
  return new Uint8Array(ab);
}

async function main(cfg: Cfg): Promise<void> {
  const t0 = performance.now();
  const M = MODELS[cfg.model ?? "4b"];
  URL_GGUF = M.url;
  const golden = (await (await fetch("/models/q35/golden-full.json")).json()) as Golden;
  if (golden.modelSha256 !== M.sha) throw new Error(`q35conf: SHA GGUF del golden (${golden.modelSha256.slice(0, 8)}) diverso dal pinnato per ${cfg.model ?? "4b"}`);

  const prompts = golden.prompts.filter((_, i) => !cfg.prompts || cfg.prompts.includes(i));
  const maxGen = cfg.maxGen ?? Infinity;
  const ctxMax = Math.max(...prompts.map((p) => p.promptTokens.length + Math.min(p.generated.length, maxGen))) + 8;

  progress(`golden: ${prompts.length} prompt, ctxMax ${ctxMax}`);
  const header = await range(0, 64 * 1024 * 1024);
  const f = parseGguf(header.buffer.slice(header.byteOffset, header.byteOffset + header.byteLength) as ArrayBuffer);
  const { shape, byName } = validateQwen35(f);

  const isMoe = shape.arch === "qwen35moe";
  const info = (name: string): GgufTensorInfo => {
    const t = byName.get(name);
    if (!t) throw new Error(`q35conf: tensore ${name} assente`);
    return t;
  };
  const arenaBudgetBytes = Math.floor((cfg.arenaGiB ?? 12) * (1 << 30));
  // La config di residenza si deduce dal GGUF (q35expertstore) e serve QUI,
  // prima del device: `arenaNeeds` deve sapere quante classi ci sono e quanto
  // pesa uno slab per dire quanti buffer d'arena servono. È la stessa config
  // che il modello passerà alla cache — una sola verità, non due stime.
  const moeCfg = isMoe ? q35MoeConfig(shape, info) : null;
  const { device } = await createEngineDevice({
    label: "q35conf",
    // `timestamp-query` si chiede solo se l'adapter la espone; il modello
    // controlla `device.features.has` e degrada dichiarando (gpuTimeStats null).
    // `subgroups` idem, e con la stessa logica di degrado: senza di lei il
    // gemv resta sul kernel a riduzione in shared memory. Va CHIESTA qui
    // perche' `createEngineDevice` concede solo cio' che il chiamante elenca —
    // un adapter che la espone non basta. Il cast: il tipo `GPUFeatureName` di
    // @webgpu/types non la elenca ancora (stessa forma di kdRunner.ts).
    optionalFeatures: ["timestamp-query", "subgroups" as GPUFeatureName],
    needs: (adapter) => ({
      ctxMax,
      head: { vocab: shape.vocab, dModel: shape.dModel },
      // MoE (it.17): i chunk dell'arena expert sono buffer da 2 GiB
      ...(isMoe ? { slabClassBytes: 2 * (1 << 30) } : {}),
      // Regime d'arena (fase-D fase 3b, fetta 3a): i buffer di classe si
      // bindano INTERI ⇒ due requisiti nuovi, e nessuno dei due è inventato
      // qui — li calcola `arenaNeeds` con l'aritmetica della cache.
      ...(moeCfg ? arenaNeeds({
        budgetBytes: arenaBudgetBytes,
        maxBufferBytes: adapter.limits.maxBufferSize,
        maxBindingBytes: adapter.limits.maxStorageBufferBindingSize,
        cfg: moeCfg,
      }) : {}),
    }),
  });
  const model = await createQ35GpuModel(device, {
    shape,
    info,
    read: (name) => range(f.dataOffset + info(name).offset, q35TensorBytes(info(name))),
    readRange: (name, off, len) => range(f.dataOffset + info(name).offset + off, len),
  }, ctxMax, arenaBudgetBytes, {
    routerShadow: cfg.routerShadow === true,
    select: cfg.optTrace === true ? "optimistic" : "cpu",
    telemetryGpu: cfg.gpuTime === true,
    debugNoStateSnapshot: cfg.noStateSnapshot === true,
    prefillM: cfg.prefillM,
    vramCeilingBytes: cfg.vramGiB !== undefined ? Math.floor(cfg.vramGiB * (1 << 30)) : undefined,
    expertPolicy: cfg.expertPolicy,
  });
  const loadMs = performance.now() - t0;
  progress(`modello su GPU in ${(loadMs / 1000).toFixed(1)} s (${model.dispatchesPerToken} dispatch/token)`);
  const vp = model.vramPlan();
  if (vp) {
    progress(`VRAM: tetto ${vp.ceilingBytes === null ? "n/d" : (vp.ceilingBytes / 2 ** 30).toFixed(2) + " GiB"} — ` +
      `allocati ${(vp.allocatedBytes / 2 ** 30).toFixed(2)} — riserva ${(vp.reserveBytes / 2 ** 30).toFixed(2)} ` +
      `⇒ budget expert ${(vp.expertBudgetBytes / 2 ** 30).toFixed(2)} GiB`);
  }

  if (cfg.debugTap !== undefined) {
    const p0 = golden.prompts[0];
    model.resetState();
    await model.readTap(cfg.debugTap); // arma il tap
    await model.step(p0.promptTokens[0], 0, false);
    const tap = await model.readTap(-2); // leggi e disarma
    post({ type: "done", report: { kind: "q35-debug-tap", layer: cfg.debugTap, token: p0.promptTokens[0], hidden: Array.from(tap), moe: model.moeStats ? model.moeStats() : null } });
    return;
  }

  if (cfg.bench) {
    // Riferimenti full-resident (fase 4, it.10): prefill sequenziale
    // read=false (sync con onSubmittedWorkDone), poi nDecode GREEDY con
    // readback (l'argmax dello step alimenta il successivo). DICHIARATO:
    // orchestratore correttezza-prima, frame di partenza pre-ottimizzazioni
    // (562 dispatch/token, nessuna fusione/batch: i moltiplicatori sono
    // materia delle fasi 6+/D, non di questo numero).
    const p = golden.prompts[cfg.bench.promptIdx];
    model.resetState();
    const P = p.promptTokens.length;
    // BRACCIO DEL PREFILL (goal engine-ttft, riga 0). Con `prefillm` il prompt
    // passa da `prefillChunk` — M righe in un submit — invece che da `step` una
    // posizione alla volta. Il chunk deve essere PIENO (il piano gemello ha le
    // taglie cotte dentro), quindi la CODA `nPre % M` la fa `step`, che e'
    // esattamente cio' che il contratto di `prefillChunk` prescrive.
    //
    // Il braccio si DICHIARA nel report (`prefillPath`). Non e' decorazione:
    // in `engine-kernel-decode` it.9 due JSON con numeri incompatibili sono
    // stati riconciliati confrontando questo campo e non i numeri — un prefill
    // a chunk e uno sequenziale differiscono di ~3x e senza il campo sembrano
    // una regressione.
    // `chunkM`, non `M`: in questo scope `M` e' gia' il descrittore del modello.
    const chunkM = model.prefillChunk !== null && cfg.prefillM ? cfg.prefillM : 0;
    const nPre = P - 1;
    const nChunk = chunkM ? Math.floor(nPre / chunkM) : 0;
    const prefillPath = chunkM
      ? `chunked M=${chunkM} (${nChunk} chunk + coda ${nPre - nChunk * chunkM} via step)`
      : "sequenziale (step per posizione)";
    const tPre = performance.now();
    // await OBBLIGATORIO: sul MoE step() contiene i readback del router
    // (fire-and-forget = mapAsync concorrenti sullo stesso staging, it.19);
    // sul denso con read=false ritorna subito: semantica di misura invariata.
    for (let c = 0; c < nChunk; c++) {
      await model.prefillChunk!(p.promptTokens.slice(c * chunkM, (c + 1) * chunkM), c * chunkM);
      if (c % 32 === 0) post({ type: "tick", msg: `prefill chunk ${c}/${nChunk}` });
    }
    for (let t = nChunk * chunkM; t < nPre; t++) await model.step(p.promptTokens[t], t, false);
    await device.queue.onSubmittedWorkDone();
    const prefillMs = performance.now() - tPre;
    const tFirst = performance.now();
    let tok = await model.step(p.promptTokens[P - 1], P - 1, true);
    const firstMs = performance.now() - tFirst;
    const stepMs: number[] = [];
    for (let i = 0; i < cfg.bench.nDecode; i++) {
      const ts = performance.now();
      tok = await model.step(tok, P + i, true);
      stepMs.push(performance.now() - ts);
      if (i % 16 === 0) post({ type: "tick", msg: `decode ${i}/${cfg.bench.nDecode}` });
    }
    const sorted = stepMs.slice().sort((a, b) => a - b);
    const p50 = sorted[Math.floor(sorted.length / 2)];
    const report = {
      schemaVersion: 1,
      kind: `q35-bench-${cfg.model ?? "4b"}-fullresident`,
      date: new Date().toISOString().slice(0, 10),
      model: M.file,
      modelSha256: golden.modelSha256,
      declared: "orchestratore correttezza-prima; sui densi decode BATCHED (fase D fase 3: argmax su GPU, K=32 token/submit, un readback ogni K); sul MoE ancora un readback del router per layer",
      prompt: { idx: cfg.bench.promptIdx, file: p.file, tokens: P },
      loadMs: Math.round(loadMs),
      prefillPath,
      prefill: { tokens: P - 1, ms: Math.round(prefillMs), tokS: (P - 1) / (prefillMs / 1000) },
      decode: { n: cfg.bench.nDecode, msPerTokenP50: p50, tokS: 1000 / p50, firstMs: Math.round(firstMs) },
      ttftMs: Math.round(loadMs + prefillMs + firstMs),
      dispatchesPerToken: model.dispatchesPerToken,
      moe: model.moeStats ? model.moeStats() : null,
    };
    post({ type: "done", report });
    return;
  }

  if (cfg.missTrace) {
    // MISURA PRIMA DI SCRIVERE (fase-D 3b, prima della fetta 3c): quanti MISS
    // ha un token? È il numero da cui dipende se il decode ottimistico paga.
    // Il repair+replay rigioca dal PRIMO layer sporco in giù: se ogni token è
    // sporco a un layer basso, il replay ricalcola quasi tutto il token e i 40
    // submit che toglie li ripaga in lavoro GPU. GLM esige per questo una
    // residenza >= 80%, e sul 35B a questa VRAM non ci siamo.
    // Due passate SULLO STESSO prompt: la prima a cache fredda (peggior caso),
    // la seconda con `resetState` — lo stato ricorrente riparte, la cache
    // expert NO — cioè il caso perfettamente caldo (miglior caso). L'uso vero
    // sta in mezzo, e questi due numeri lo delimitano.
    const p = golden.prompts[0];
    const tokens = [...p.promptTokens, ...p.generated];
    const passes: Array<{ misses: number[]; hits: number[]; firstDirtyLayerUnknown: true }> = [];
    for (let pass = 0; pass < 2; pass++) {
      model.resetState();
      const misses: number[] = [], hits: number[] = [];
      let prevM = model.moeStats!().misses, prevH = model.moeStats!().hits;
      for (let t = 0; t < tokens.length; t++) {
        await model.step(tokens[t], t, false);
        const st = model.moeStats!();
        misses.push(st.misses - prevM); hits.push(st.hits - prevH);
        prevM = st.misses; prevH = st.hits;
      }
      passes.push({ misses, hits, firstDirtyLayerUnknown: true });
      post({ type: "tick", msg: `miss-trace pass ${pass}: ${misses.reduce((a, b) => a + b, 0)} miss` });
    }
    const sum = (a: number[]): number => a.reduce((x, y) => x + y, 0);
    const report = {
      schemaVersion: 1,
      kind: `q35-misstrace-${cfg.model ?? "4b"}`,
      date: new Date().toISOString().slice(0, 10),
      model: M.file, arenaGiB: cfg.arenaGiB ?? 12,
      selectionsPerToken: (shape.nLayer as number) * (shape.nExpertUsed as number),
      tokens: tokens.length,
      passes: passes.map((pp, i) => ({
        pass: i, missTotal: sum(pp.misses), hitTotal: sum(pp.hits),
        missPerToken: pp.misses,
        dirtyTokens: pp.misses.filter((m) => m > 0).length,
        missPerTokenMedian: [...pp.misses].sort((a, b) => a - b)[Math.floor(pp.misses.length / 2)],
      })),
      moe: model.moeStats!(),
    };
    post({ type: "done", report });
    return;
  }

  if (cfg.prefillM) {
    // GATE DELLA FASE 4. Due bracci sullo STESSO prompt e sullo stesso modello:
    // (a) `step()` sequenziale, che e' il path di oggi; (b) `prefillChunk` a M
    // righe. Si confrontano i LOGITS dell'ultima posizione del chunk, che sono
    // gli unici che il prefill produce — e l'atteso e' bit per bit, non "vicino".
    const M = cfg.prefillM;
    // IL PROMPT NON E' PIU' CABLATO (it.20). Era `golden.prompts[0]`, quindi
    // `--prompts N` non cambiava cio' che il gate misurava e chi lo passava
    // credeva di aver variato qualcosa. Serve perche' il criterio argmax del
    // ruling ha dato 63/64 su UN prompt, e un conteggio su un campione solo non
    // distingue «tipico» da «fortunato» — e' la landmine del campione da 22
    // posizioni. Qui si prende il primo dei prompt SELEZIONATI: `prompts` e'
    // gia' la lista filtrata da `cfg.prompts`, quindi il default resta il
    // prompt 0 e il comportamento storico non cambia.
    const p = prompts[0];
    if (!p) throw new Error("q35 prefillChunk gate: nessun prompt selezionato");
    const all = [...p.promptTokens, ...p.generated];
    // TETTO IN TOKEN, non in chunk (it.33): il guadagno del prefill a chunk
    // CRESCE col contesto — misurato 1,17x su 47 token e 2,02x su 6456 — quindi
    // due M confrontati a numero di CHUNK uguale girerebbero su contesti diversi
    // e il confronto direbbe una cosa per un'altra. A parita' di token il
    // contesto e' lo stesso e la M e' l'unica variabile.
    const GATE_TOKENS = 1024;
    const nChunk = Math.min(Math.floor(GATE_TOKENS / M), Math.floor(all.length / M));
    if (nChunk < 3) throw new Error(`q35 prefillChunk gate: ${nChunk} chunk a M=${M} — servono almeno 3 (il primo si scarta)`);
    const toks = all.slice(0, nChunk * M);
    // SCALDATA prima dei due bracci (it.34). Senza, sul MoE il braccio
    // sequenziale gira a cache expert VUOTA e quello a chunk la trova calda: il
    // primo giro dava 30,8x, che e' la cache che si riempie, non il batch.
    // Stessa trappola di it.17, e la stessa cura: si misura a parita'.
    model.resetState();
    for (let i = 0; i < toks.length; i++) await model.step(toks[i], i, false);
    const seqLogits: Float32Array[] = [];
    const seqMs: number[] = [];
    model.resetState();
    for (let c = 0; c < nChunk; c++) {
      const t0c = performance.now();
      for (let i = 0; i < M - 1; i++) await model.step(toks[c * M + i], c * M + i, false);
      await model.step(toks[c * M + M - 1], c * M + M - 1, true);
      seqMs.push(performance.now() - t0c);
      seqLogits.push(model.lastLogits()!);
    }
    model.resetState();
    const chunkLogits: Float32Array[] = [];
    const chunkMs: number[] = [];
    for (let c = 0; c < nChunk; c++) {
      const t0c = performance.now();
      const lg = await model.prefillChunk!(toks.slice(c * M, (c + 1) * M), c * M);
      chunkMs.push(performance.now() - t0c);
      chunkLogits.push(lg);
    }
    let bitEqual = 0, maxAbs = 0, maxRel = 0, cmp = 0;
    // L'ARGMAX E' IL CRITERIO DI TRANSIZIONE (ruling del PI, 2026-08-14, docket
    // item 22). La bit-identita' resta MISURATA — si sospende come criterio, non
    // come informazione: `bitIdentical` continua a comparire nel report, e il
    // giorno in cui torna vera lo si vede senza rifare niente.
    let argmaxSame = 0;
    // I DISACCORDI SI GUARDANO COL MARGINE, NON COL CONTEGGIO (landmine di
    // HANDOFF: «un campione da 22 posizioni non distingue niente — guarda rango
    // e log-prob, stessa informazione a varianza molto piu' bassa»). Un argmax
    // diverso su un TESTA A TESTA fra due token separati da 1e-3 e' rumore che
    // cade da una parte; lo stesso disaccordo con un margine largo e' un
    // difetto. Senza questo dato il gate direbbe solo «63/64» e non si saprebbe
    // quale dei due.
    const argmaxDiffs: { chunk: number; seqTok: number; chunkTok: number;
      seqTop2Gap: number; chunkTop2Gap: number; swapMargin: number }[] = [];
    for (let c = 0; c < nChunk; c++) {
      let aBest = -Infinity, aArg = -1, aSecond = -Infinity;
      let bBest = -Infinity, bArg = -1, bSecond = -Infinity;
      for (let i = 0; i < shape.vocab; i++) {
        const a = seqLogits[c][i], b = chunkLogits[c][i];
        cmp++;
        if (Object.is(a, b)) bitEqual++;
        else {
          const d = Math.abs(a - b);
          maxAbs = Math.max(maxAbs, d);
          const den = Math.max(Math.abs(a), Math.abs(b));
          if (den > 0) maxRel = Math.max(maxRel, d / den);
        }
        if (a > aBest) { aSecond = aBest; aBest = a; aArg = i; } else if (a > aSecond) aSecond = a;
        if (b > bBest) { bSecond = bBest; bBest = b; bArg = i; } else if (b > bSecond) bSecond = b;
      }
      if (aArg === bArg) argmaxSame++;
      else {
        argmaxDiffs.push({
          chunk: c, seqTok: aArg, chunkTok: bArg,
          // quanto il vincitore batte il secondo, su ciascun braccio: se
          // entrambi i margini sono minuscoli, i due token erano appaiati e il
          // disaccordo e' una moneta che cade
          seqTop2Gap: aBest - aSecond, chunkTop2Gap: bBest - bSecond,
          // di quanto il sequenziale preferiva il PROPRIO vincitore a quello
          // scelto dal chunk: e' la distanza che la quantizzazione ha dovuto
          // colmare per ribaltare la scelta
          swapMargin: aBest - seqLogits[c][bArg],
        });
      }
    }
    // Il PRIMO chunk si scarta (docket item 10: il primo passaggio dopo il load
    // paga compilazione e prime allocazioni). Con due soli campioni la mediana
    // sarebbe quello.
    const med = (v: number[]): number => {
      const w = v.slice(1).sort((a, b) => a - b);
      return w[Math.floor(w.length / 2)];
    };
    post({
      type: "done",
      report: {
        schemaVersion: 1,
        kind: `q35-prefillchunk-${cfg.model ?? "4b"}`,
        date: new Date().toISOString().slice(0, 10),
        model: MODELS[cfg.model ?? "4b"].file, chunkM: M, chunks: nChunk, tokens: nChunk * M,
        declared: "bracci sullo STESSO modello e sullo stesso prompt; si confrontano i logits " +
          "dell'ultima posizione di ogni chunk, che sono gli unici che il prefill produce. " +
          "CRITERIO DI TRANSIZIONE (ruling PI 2026-08-14, docket item 22): l'atteso e' " +
          "l'ARGMAX IDENTICO su ogni chunk, col numero accanto. La BIT-IDENTITA' — che era " +
          "il criterio fino a it.14 — e' caduta in it.15, quando la via intera ha iniziato a " +
          "quantizzare le attivazioni a int8 nel solo ramo a chunk: i due bracci fanno " +
          "aritmetica diversa, quindi divergono per COSTRUZIONE e non per un difetto. " +
          "E' una SOSPENSIONE, non un'abolizione: quando il percorso sequenziale sara' " +
          "anch'esso su intero, i due bracci torneranno a fare la stessa aritmetica e la " +
          "bit-identita' TORNA A ESSERE IL CRITERIO. Resta misurata qui sotto proprio " +
          "perche' quel giorno si veda senza rifare niente.",
        gate: {
          // IL CRITERIO che decide oggi
          argmaxSame, chunks: nChunk, argmaxIdentical: argmaxSame === nChunk,
          // LA MISURA che il criterio di domani riprendera' in mano
          bitEqual, compared: cmp, bitIdentical: bitEqual === cmp, maxAbs, maxRel,
          argmaxDiffs,
          criterion: "argmax",
          bitIdentitySuspendedBy: "it.15 — prefillQuantXQ8Wgsl, attivazioni int8 sul solo ramo a chunk",
          bitIdentityReturnsWhen: "il percorso sequenziale (`step`) passa anch'esso sulla via intera",
        },
        msPerChunk: {
          sequential: med(seqMs), chunked: med(chunkMs), speedup: med(seqMs) / med(chunkMs),
          discardedFirst: true, samples: nChunk - 1,
          seqAll: seqMs.map((v) => Math.round(v * 100) / 100), chunkAll: chunkMs.map((v) => Math.round(v * 100) / 100),
        },
        msPerToken: { sequential: med(seqMs) / M, chunked: med(chunkMs) / M },
      },
    });
    return;
  }

  if (cfg.optTrace) {
    // GATE della fetta 3c. Le due passate girano sulla STESSA cache: la prima
    // col path sync (che è anche ciò che porta la residenza a regime), la
    // seconda col path a submit unico. `resetState` azzera lo stato ricorrente
    // ma NON la cache expert — è lo stesso strumento di it.16.
    const p = golden.prompts[0];
    const tokens = [...p.promptTokens, ...p.generated];
    const runPass = async (optimistic: boolean): Promise<{
      argmax: number[]; submits: number; readbacks: number; hits: number; misses: number;
      routing: Record<string, number>; ms: number; error: string | null;
      dirtyTokens: number; replays: number; replayLayers: number; repairMs: number;
      encodeMs: number; embedMs: number; argmaxMs: number; tailCpuMs: number;
      readbackMs: number; tokenMs: number;
    }> => {
      model.resetState();
      model.setOptimistic(optimistic);
      const p0 = model.perf(), m0 = model.moeStats!();
      const t = performance.now();
      const argmax: number[] = [];
      // Un token SPORCO fa alzare `step` (degrado definito, I2: non si campiona
      // mai). Qui si CATTURA invece di far morire il run: la passata si ferma
      // dove si è rotta e il report esce lo stesso col perché — un gate che
      // muore non lascia numeri, e i numeri delle passate precedenti sono già
      // stati pagati in minuti di GPU.
      let error: string | null = null;
      try {
        for (let i = 0; i < tokens.length; i++) {
          argmax.push(await model.step(tokens[i], i, true));
          if (i % 8 === 0) post({ type: "tick", msg: `${optimistic ? "optimistic" : "sync"} ${i}/${tokens.length}` });
        }
      } catch (e) {
        error = e instanceof Error ? e.message.slice(0, 400) : String(e);
        post({ type: "tick", msg: `pass ${optimistic ? "optimistic" : "sync"} interrotta a ${argmax.length}: ${error}` });
      }
      const ms = performance.now() - t;
      const p1 = model.perf(), m1 = model.moeStats!();
      // routing della PASSATA = cumulativo dopo meno cumulativo prima: la
      // struttura è la stessa del gate di it.14, dove il numero che porta il
      // peso è l'istogramma chiave per chiave e non il top-1.
      const routing: Record<string, number> = {};
      for (const [k, v] of Object.entries(m1.routing)) {
        const dlt = v - (m0.routing[k] ?? 0);
        if (dlt > 0) routing[k] = dlt;
      }
      return {
        argmax, submits: p1.submits - p0.submits, readbacks: p1.readbacks - p0.readbacks,
        hits: m1.hits - m0.hits, misses: m1.misses - m0.misses, routing, ms, error,
        dirtyTokens: p1.dirtyTokens - p0.dirtyTokens, replays: p1.replays - p0.replays,
        replayLayers: p1.replayLayers - p0.replayLayers, repairMs: p1.repairMs - p0.repairMs,
        encodeMs: p1.encodeMs - p0.encodeMs, embedMs: p1.embedMs - p0.embedMs,
        argmaxMs: p1.argmaxMs - p0.argmaxMs, tailCpuMs: p1.tailCpuMs - p0.tailCpuMs,
        readbackMs: p1.readbackMs - p0.readbackMs, tokenMs: p1.tokenMs - p0.tokenMs,
      };
    };
    // TRE passate, non due. La prima è FREDDA per forza (la cache parte vuota)
    // e il suo ms/token è dominato dall'I/O dei 3 341 miss: confrontarlo con
    // quello del path ottimistico — che gira per costruzione a cache calda —
    // sarebbe un confronto freddo-contro-caldo travestito da speedup. La
    // seconda passata è il path di OGGI sulla cache già calda, ed è quella
    // contro cui il numero va letto.
    // La passata FREDDA: col path sync (il "prima" storico) o col path
    // ottimistico (`--opt-cold`), che è il test vero del repair+replay perché a
    // cache vuota ogni token è sporco. Non possono stare nello stesso run: la
    // cache fredda esiste una volta sola per processo.
    const coldOpt = cfg.optCold === true;
    const syncCold = await runPass(coldOpt);
    // I DUE PATH A FREDDO, nello stesso processo: si svuota la cache expert e
    // si rifa' la passata con l'altro path. Finora i due bracci stavano in due
    // run (it.18) e il confronto non era una misura.
    let coldOther: Awaited<ReturnType<typeof runPass>> | null = null;
    if (cfg.coldBoth) {
      model.debugEvictAll!();
      coldOther = await runPass(!coldOpt);
    }
    // E il ms/token si misura come il docket item 10 impone, non con un
    // campione per braccio: bracci INTERLEAVATI, prima ripetizione SCARTATA
    // (la prima passata dopo il load paga compilazione e prime allocazioni),
    // mediana e dispersione riportate. I submit e i readback invece sono
    // CONTATORI esatti: non hanno dispersione e una ripetizione basterebbe.
    const REPS = 4;
    const syncRuns: Awaited<ReturnType<typeof runPass>>[] = [];
    const optRuns: Awaited<ReturnType<typeof runPass>>[] = [];
    for (let r = 0; r < REPS; r++) {
      syncRuns.push(await runPass(false));
      optRuns.push(await runPass(true));
      post({ type: "tick", msg: `rep ${r + 1}/${REPS}` });
    }
    const msOf = (rs: typeof syncRuns): number[] =>
      rs.slice(1).map((r) => r.ms / Math.max(1, r.argmax.length)).sort((a, b) => a - b);
    const disp = (v: number[]) => ({
      median: v[Math.floor(v.length / 2)], min: v[0], max: v[v.length - 1], samples: v.length,
    });
    const msSync = disp(msOf(syncRuns)), msOpt = disp(msOf(optRuns));
    const syncWarm = syncRuns[syncRuns.length - 1];
    const optim = optRuns[optRuns.length - 1];
    const n = tokens.length;
    const nCmp = Math.min(syncWarm.argmax.length, optim.argmax.length);
    const argmaxEqual = syncWarm.argmax.slice(0, nCmp).filter((v, i) => v === optim.argmax[i]).length;
    const sync = syncWarm;
    const keys = new Set([...Object.keys(sync.routing), ...Object.keys(optim.routing)]);
    let routingDiff = 0;
    for (const k of keys) if ((sync.routing[k] ?? 0) !== (optim.routing[k] ?? 0)) routingDiff++;
    const report = {
      schemaVersion: 1,
      kind: `q35-optimistic-${cfg.model ?? "4b"}`,
      date: new Date().toISOString().slice(0, 10),
      model: M.file, arenaGiB: cfg.arenaGiB ?? 12,
      tokens: n,
      selectionsPerToken: (shape.nLayer as number) * (shape.nExpertUsed as number),
      declared: "pass sync-cold = il path di oggi a cache FREDDA (è la passata che scalda: il suo " +
        "ms/token è dominato dall'I/O dei miss e NON è il termine di paragone); poi REPS coppie " +
        "interleavate sync-warm / optimistic-warm sulla stessa cache, prima coppia scartata " +
        "(docket item 10). Freddo e caldo restano separati: a freddo il path ottimistico non è " +
        "utilizzabile senza repair+replay, ed è la misura di it.16 a dirlo.",
      reps: REPS,
      passes: [
        { pass: coldOpt ? "optimistic-cold" : "sync-cold", tokensDone: syncCold.argmax.length, error: syncCold.error, ...pass2json(syncCold, Math.max(1, syncCold.argmax.length)) },
        ...(coldOther ? [{
          pass: coldOpt ? "sync-cold(evicted)" : "optimistic-cold(evicted)",
          tokensDone: coldOther.argmax.length, error: coldOther.error,
          ...pass2json(coldOther, Math.max(1, coldOther.argmax.length)),
        }] : []),
        { pass: "sync-warm", tokensDone: syncWarm.argmax.length, error: syncWarm.error, ...pass2json(syncWarm, Math.max(1, syncWarm.argmax.length)), msPerTokenDisp: msSync },
        { pass: "optimistic-warm", tokensDone: optim.argmax.length, error: optim.error, ...pass2json(optim, Math.max(1, optim.argmax.length)), msPerTokenDisp: msOpt },
      ],
      gate: {
        // il confronto che qualifica la fetta è A PARITÀ di residenza: pass 1
        // (sync, caldo) contro pass 2 (ottimistico, caldo)
        comparedPasses: "sync-warm (1) vs optimistic-warm (2)",
        argmaxEqual, argmaxCompared: nCmp, argmaxTotal: n, argmaxIdentical: nCmp === n && argmaxEqual === n,
        // Col path ottimistico a freddo questo confronto E' il gate del
        // repair+replay: il token riparato deve dare lo STESSO argmax del path
        // sync a caldo. Se il replay sbagliasse — stato ricorrente applicato
        // due volte, rientro dall'hidden sbagliato — divergerebbe qui.
        argmaxEqualColdVsSyncWarm: syncCold.argmax.slice(0, Math.min(syncCold.argmax.length, syncWarm.argmax.length))
          .filter((v, i) => v === syncWarm.argmax[i]).length,
        coldPath: coldOpt ? "optimistic" : "sync",
        routingKeys: keys.size, routingDiff, routingIdentical: routingDiff === 0,
        missesSyncWarm: syncWarm.misses, missesOptimistic: optim.misses,
        msPerTokenSyncWarm: msSync.median, msPerTokenOptimistic: msOpt.median,
        msPerTokenDelta: msOpt.median - msSync.median,
        submitsPerTokenOptimistic: optim.submits / Math.max(1, optim.argmax.length),
        readbacksPerTokenOptimistic: optim.readbacks / Math.max(1, optim.argmax.length),
      },
      gpuTime: model.gpuTimeStats ? model.gpuTimeStats() : null,
      moe: model.moeStats!(),
    };
    post({ type: "done", report });
    return;
  }

  let okTot = 0, posTot = 0;
  const perPrompt: { id: string; positions: number; top1: number; engineArgmax: number[]; promptS: number }[] = [];
  for (let pi = 0; pi < prompts.length; pi++) {
    const p = prompts[pi];
    const gen = Math.min(p.generated.length, maxGen);
    const tokens = [...p.promptTokens, ...p.generated.slice(0, gen - 1)];
    const P = p.promptTokens.length;
    model.resetState();
    const tp = performance.now();
    let ok = 0;
    const engineArgmax: number[] = [];
    // BATCH (fase D fase 3) sui modelli senza MoE, e SOLO sullo span che
    // serve. Lezione di it.10: il prefill qui gira gia' con `read=false`,
    // cioe' senza readback e senza attesa — batcharlo non toglie niente e
    // AGGIUNGE l'argmax su GPU su token che non lo useranno (misurato: 1,75 →
    // 1,82 s sullo smoke, che ha 33 token di prefill su 39). Il guadagno sta
    // dove ogni token vuole il proprio argmax: le posizioni generate.
    // Sul MoE `decodeBatch` e' null (routing su CPU per layer, docket item 8).
    const primaGen = P - 1; // la prima posizione di cui si legge l'argmax
    // IL PROMPT SI PUO' PREFILLARE A CHUNK (it.21), e prima non si poteva.
    //
    // PERCHE' SERVE. Il gate `q35-prefillchunk-4b` confronta i due bracci fra
    // LORO e dice 250/256 argmax uguali — ma non dice quanto conti in pratica,
    // perche' i logit intermedi dei chunk in produzione NON LI LEGGE NESSUNO:
    // il prefill serve a riempire la KV cache, e l'unica cosa che esce e' il
    // primo token. La domanda vera e' un'altra: la cache costruita dalla via
    // INTERA porta il motore a sbagliare piu' token contro l'oracolo?
    //
    // Qui la si puo' finalmente misurare nella metrica di casa — il top-1
    // contro il golden llama.cpp, ratchet 4B 1012/1024 = 98,828% — con la
    // generazione che resta su `step()`, cioe' esattamente lo scenario di
    // prodotto: prompt a chunk, poi decode.
    //
    // NOTA CHE CAMBIA IL SEGNO DELLA DOMANDA: l'oracolo E' llama.cpp, e
    // llama.cpp quantizza le attivazioni a Q8_0 per OGNI matmul q4_0
    // (`ggml-cpu.c`: `[GGML_TYPE_Q4_0].vec_dot = ggml_vec_dot_q4_0_q8_0`,
    // `vec_dot_type = GGML_TYPE_Q8_0`). La nostra via intera fa la stessa cosa
    // del riferimento; e' il nostro percorso sequenziale, in virgola mobile, a
    // essere piu' preciso dell'oracolo contro cui ci misuriamo.
    const confChunkM = model.prefillChunk !== null && cfg.confPrefillM ? cfg.confPrefillM : 0;
    const confChunks = confChunkM ? Math.floor(primaGen / confChunkM) : 0;
    for (let c = 0; c < confChunks; c++) {
      await model.prefillChunk!(tokens.slice(c * confChunkM, (c + 1) * confChunkM), c * confChunkM);
    }
    for (let t = confChunks * confChunkM; t < primaGen; t++) await model.step(tokens[t], t, false);
    const K = model.decodeBatch ? 32 : 1;
    for (let t = primaGen; t < tokens.length; t += K) {
      const n = Math.min(K, tokens.length - t);
      const ids = model.decodeBatch
        ? await model.decodeBatch(tokens.slice(t, t + n), t)
        : [await model.step(tokens[t], t, true)];
      for (let j = 0; j < ids.length; j++) {
        const gi = t + j - primaGen;
        if (gi >= 0 && gi < gen) {
          engineArgmax.push(ids[j]);
          if (ids[j] === p.positions[gi].argmax) ok++;
          posTot++;
        }
      }
      if ((t - primaGen) % 512 < K) post({ type: "tick", msg: `p${pi} ${t}/${tokens.length} (top1 ${ok}/${Math.max(1, engineArgmax.length)})` });
    }
    okTot += ok;
    const promptS = (performance.now() - tp) / 1000;
    perPrompt.push({ id: p.id, positions: gen, top1: ok, engineArgmax, promptS });
    progress(`p${pi} (${p.file.split("/").pop()}): top1 ${ok}/${gen} in ${promptS.toFixed(0)} s`);
  }

  const report = {
    schemaVersion: 1,
    kind: `q35-conf-${cfg.model ?? "4b"}`,
    date: new Date().toISOString().slice(0, 10),
    model: M.file,
    modelSha256: golden.modelSha256,
    oracleCommit: golden.oracle.commit,
    corpusHash: golden.corpusHash,
    engine: {
      orchestrator: "q35gpumodel correttezza-prima",
      dispatchesPerToken: model.dispatchesPerToken,
      dispatchBreakdown: model.dispatchBreakdown,
      vramPlan: model.vramPlan(),
      loadMs: Math.round(loadMs), perToken: model.perf(),
    },
    moe: model.moeStats ? model.moeStats() : null,
    routerShadow: model.routerShadowStats ? model.routerShadowStats() : null,
    top1: { ok: okTot, positions: posTot, rate: okTot / posTot },
    perPrompt: perPrompt.map(({ engineArgmax, ...r }) => r),
    engineArgmaxByPrompt: Object.fromEntries(perPrompt.map((r, i) => [String(i), r.engineArgmax])),
    totalMs: Math.round(performance.now() - t0),
  };
  post({ type: "done", report });
}

self.onmessage = (e: MessageEvent) => {
  void main(e.data as Cfg).catch((err: unknown) => {
    post({ type: "error", message: err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err) });
  });
};
