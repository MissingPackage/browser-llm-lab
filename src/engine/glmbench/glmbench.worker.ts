// Bench GLM full-model (goal C2 fase 6 slice 2, spec §8): protocollo B2 —
// warmup scartato + N repliche sullo stesso prompt, finestre dichiarate, ctx
// del corpus golden. Misura prefill (posizioni del prompt) e decode (greedy
// dal logit del motore, nessun teacher forcing: è il percorso di produzione)
// con i GATE hard del contratto: decode ≥13.43 tok/s, prefill ≥56.58 tok/s
// (floor = oracolo CPU C1, llama-bench 5f55650).
// Telemetria per fase (input C3): dispatch/token, sync router/token, hit-rate
// residenza, stallo ms/token scomposto (read OPFS / pack CPU / upload), slot
// occupati, eviction.
import { GLM47_FLASH as G, GLM47_FLASH_SHA256 } from "../shape";
import { dequantQ4_0Row } from "../quant";
import { createGlmModel } from "../glmmodel";
import { GLM_PREFILL_M } from "../moeprefillplan";
import { GlmOpfsSource } from "../glmsource";
import { grantedLimits, slabBufferCap } from "../gpulimits";
import { createEngineDevice } from "../gpudevice";
import type { GlmTelemetry } from "../glmmodel";
import {
  arenaNeeds, modelExpertPark, slabBudgetCtxAware, PARK_Q4_0, PARK_Q4_1,
  readBandwidth, type ReadRegime, type ExpertCacheStats,
} from "../residency";
import { SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1 } from "../moe";

interface GoldenPrompt { id: string; promptTokens: number[]; generated: number[] }
interface Golden { modelSha256: string; prompts: GoldenPrompt[] }
interface Cfg {
  prompt: number; nGen: number; replicates: number; budgetGiB: number;
  attribReplicates?: number; prefillBatch?: boolean; select?: "cpu" | "optimistic";
  /** C3c fase 3: budget CALCOLATO dalla formula ctx-aware (spec §2) invece del
   *  budgetGiB fisso. Il valore è il tetto allocabile MISURATO dal runner
   *  (nvidia-smi al lancio) — la sessione host sta in hostState del report. */
  allocCeilingBytes?: number;
  /** C3c fase 3: alloca KV/partials per QUESTO ctx (≥ nPrompt+nGen), per le
   *  run che provano il regime a contesto lungo senza un prompt da 6k token. */
  ctxMaxOverride?: number;
  /** C3c fase 6: bench ai budget stretti sul path sync coi meccanismi C3c
   *  (l'ottimistico rifiuta sotto l'80% di residenza per precondizione). */
  prefetch?: "inforward";
  policy?: "tier";
  parkFrac?: number;
}

// Allocazione slot del modo ottimistico (C3b it.4, docket item 6b): q4_1
// INTERA per prima (256 slot ≈ 1.3 GiB: azzera il miss surface dei 4 layer
// MoE q4_1), il resto del budget a q4_0. Il riparto proporzionale del
// default renderebbe q4_1 il vincolo artificiale della precondizione.
const optimisticSlots = (budgetBytes: number): { q4_0: number; q4_1: number } => {
  const park = modelExpertPark(G.nLayer);
  const q41 = Math.min(park.q4_1, Math.floor(budgetBytes / SLAB_DOWN_Q4_1.bytes));
  const rest = budgetBytes - q41 * SLAB_DOWN_Q4_1.bytes;
  const q40 = Math.min(park.q4_0, Math.floor(rest / SLAB_DOWN_Q4_0.bytes));
  return { q4_0: q40, q4_1: q41 };
};

// Funzione obiettivo (direction §2, ruling PI 2026-08-01): NON sono gate — il
// report li stampa perché il floor CPU non venga scambiato per l'obiettivo.
const UX_DECODE_TPS = 30;          // soglia normale
const UX_DECODE_TPS_THINKING = 60; // regime thinking (token di ragionamento = latenza pura)
const UX_TTFT_MS = 4000;

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const progress = (msg: string) => post({ type: "progress", msg });

// Delta dei contatori cumulativi della cache (niente reset: i contatori sono
// del componente, il bench ne prende le differenze per fase).
interface PhaseResidency {
  hits: number; misses: number; evictions: number; hitRate: number | null;
  // fonte dell'hit + retention SULLA FINESTRA (fase 4d, lezione kimi-k3-in-c:
  // mai un hit rate da solo — solo la retention concorda coi byte letti)
  hitsResident: number; hitsPrefetch: number; retention: number | null;
  readMs: number; packMs: number; uploadMs: number; stallMs: number;
  bytesRead: number; bytesUploaded: number;
  /**
   * IL REGIME DI LETTURA, DICHIARATO (it.20 di engine-velocita-decode). Senza
   * questi due campi il riferimento del 2026-08-15 diceva 15,330 tok/s e quello
   * di oggi 11,35 con lo stesso codice e gli stessi `bytesRead`: la differenza
   * era che le repliche del primo leggevano dalla page cache a 9,55 GiB/s —
   * 2,9x la banda che la SUA STESSA passata di warm-up aveva misurato sullo
   * stesso file. Il numero era una fotografia della RAM libera, e l'artefatto
   * non lo diceva. Ora lo dice: `readRegime: "os-cache"` e' l'avviso che quel
   * decode NON e' confrontabile con uno preso a cache fredda.
   */
  readGiBs: number | null; readRegime: ReadRegime;
}
const residencyDelta = (a: ExpertCacheStats, b: ExpertCacheStats): PhaseResidency => {
  const hits = b.hits - a.hits, misses = b.misses - a.misses;
  const evictions = b.evictions - a.evictions;
  const readMs = b.readMs - a.readMs, packMs = b.packMs - a.packMs, uploadMs = b.uploadMs - a.uploadMs;
  const bytesRead = b.bytesRead - a.bytesRead;
  const band = readBandwidth(bytesRead, readMs);
  return {
    readGiBs: band.gibs, readRegime: band.regime,
    hits, misses, evictions,
    hitRate: hits + misses > 0 ? hits / (hits + misses) : null,
    hitsResident: b.hitsResident - a.hitsResident,
    hitsPrefetch: b.hitsPrefetch - a.hitsPrefetch,
    retention: hits + misses > 0 ? 1 - evictions / (hits + misses) : null,
    readMs, packMs, uploadMs, stallMs: readMs + packMs + uploadMs,
    bytesRead, bytesUploaded: b.bytesUploaded - a.bytesUploaded,
  };
};

// Per-token del decode: wall, stallo residenza e miss del SINGOLO token —
// permette l'attribuzione senza strumentare il motore (i token a zero miss
// danno il costo puro GPU+sync, il resto la pendenza per miss).
interface TokenRec { ms: number; stallMs: number; misses: number }

interface PhaseResult {
  tokens: number; ms: number; toksPerSec: number; msPerToken: number;
  msPerTokenP50: number; residency: PhaseResidency;
}
// ttftMs: dal PRIMO forward di prefill al momento in cui il primo token
// generato è disponibile (argmax dei logits dell'ultima posizione di prompt).
// Modello GIÀ residente: il TTFT a freddo (che include load e popolamento
// della slab) è metrica di C3b — instant-on.
interface TelemDelta {
  forwards: number; encodeCpuMs: number; ensureMs: number; routerWaitMs: number;
  tailWaitMs: number; routerSyncs: number; submits: number;
  dirtyTokens: number; replays: number; replayLayers: number; repairMs: number;
  gpuBusyMs: number | null; gpuPasses: number; gpuPassOverflow: number;
  dispatches: number;
  gpuByCatMs: Record<string, number> | null;
  gpuByCatPasses: Record<string, number> | null;
}
const telemDelta = (a: GlmTelemetry, b: GlmTelemetry): TelemDelta => ({
  forwards: b.forwards - a.forwards,
  encodeCpuMs: b.encodeCpuMs - a.encodeCpuMs,
  ensureMs: b.ensureMs - a.ensureMs,
  routerWaitMs: b.routerWaitMs - a.routerWaitMs,
  tailWaitMs: b.tailWaitMs - a.tailWaitMs,
  routerSyncs: b.routerSyncs - a.routerSyncs,
  submits: b.submits - a.submits,
  dirtyTokens: b.dirtyTokens - a.dirtyTokens,
  replays: b.replays - a.replays,
  replayLayers: b.replayLayers - a.replayLayers,
  repairMs: b.repairMs - a.repairMs,
  gpuBusyMs: a.gpuBusyMs === null || b.gpuBusyMs === null ? null : b.gpuBusyMs - a.gpuBusyMs,
  gpuPasses: b.gpuPasses - a.gpuPasses,
  gpuPassOverflow: b.gpuPassOverflow - a.gpuPassOverflow,
  dispatches: b.dispatches - a.dispatches,
  gpuByCatPasses: b.gpuByCatPasses === null ? null : Object.fromEntries(
    Object.entries(b.gpuByCatPasses).map(([k, v]) => [k, v - (a.gpuByCatPasses?.[k] ?? 0)]),
  ),
  gpuByCatMs: b.gpuByCatMs === null ? null : Object.fromEntries(
    Object.entries(b.gpuByCatMs).map(([k, v]) => [k, v - (a.gpuByCatMs?.[k] ?? 0)]),
  ),
});

interface RunResult {
  prefill: PhaseResult; decode: PhaseResult; decodeTokens: TokenRec[]; generated: number[];
  ttftMs: number; telem?: { prefill: TelemDelta; decode: TelemDelta };
}

const median = (v: number[]): number => {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const stats = (v: number[]) => {
  const mean = v.reduce((a, b) => a + b, 0) / Math.max(1, v.length);
  const varr = v.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, v.length - 1);
  return { median: median(v), mean, stdev: v.length > 1 ? Math.sqrt(varr) : 0, reps: v };
};

async function main(cfg: Cfg): Promise<void> {
  const t0 = performance.now();
  const source = await GlmOpfsSource.open("/models/GLM-4.7-Flash-Q4_0.gguf", progress);
  const golden = (await (await fetch("/models/glm-conf-golden.json")).json()) as Golden;
  if (golden.modelSha256 !== GLM47_FLASH_SHA256) throw new Error("golden: SHA GGUF diverso dal canonico");
  const pr = golden.prompts[cfg.prompt];
  if (!pr) throw new Error(`prompt ${cfg.prompt} assente nel corpus golden`);

  const promptTokens = pr.promptTokens;
  const nPrompt = promptTokens.length;
  if (cfg.ctxMaxOverride && cfg.ctxMaxOverride < nPrompt + cfg.nGen) {
    throw new Error(`ctxMaxOverride ${cfg.ctxMaxOverride} < nPrompt+nGen ${nPrompt + cfg.nGen}`);
  }
  const ctxMax = Math.max(nPrompt + cfg.nGen, cfg.ctxMaxOverride ?? 0);

  // Il device si crea DOPO aver saputo ctxMax e budget: i limiti sono DERIVATI
  // dai consumatori (C3a it.6), non chiesti al massimo dell'adapter.
  // C3c fase 3: con allocCeilingBytes il budget è CALCOLATO dalla formula
  // ctx-aware (spec §2) — il breakdown finisce nel report.
  const slabBudget = cfg.allocCeilingBytes
    ? slabBudgetCtxAware({ allocCeilingBytes: cfg.allocCeilingBytes, ctxMax })
    : null;
  const budgetBytes = slabBudget ? slabBudget.budgetBytes : Math.floor(cfg.budgetGiB * (1 << 30));
  const optimistic = cfg.select === "optimistic";
  if (optimistic && cfg.parkFrac !== undefined) {
    throw new Error("glmbench: parkFrac (budget stretto) e' il regime sync — l'ottimistico rifiuta sotto l'80%");
  }
  // In ottimistico gli slot sono ESPLICITI (q4_1-first): la STESSA override va
  // sia ad arenaNeeds (sizing device) sia alla cache — una sola verita'.
  // C3c fase 6: coi budget stretti (parkFrac) l'override e' proporzionale al
  // parco, come nell'harness routing (una sola convenzione per i 1472/736).
  const slotsOverride = cfg.parkFrac !== undefined
    ? { q4_0: Math.floor(cfg.parkFrac * PARK_Q4_0), q4_1: Math.floor(cfg.parkFrac * PARK_Q4_1) }
    : optimistic ? optimisticSlots(budgetBytes) : undefined;
  const { adapter, device, has } = await createEngineDevice({
    label: "glmbench",
    // timestamp-query: livello 2 dell'attribuzione (gpuBusy). Se l'adapter non
    // la espone il bench gira lo stesso e il report lo dichiara (gpuBusyMs null).
    optionalFeatures: ["timestamp-query"],
    needs: (adapter) => ({
      ctxMax, head: { vocab: G.vocab, dModel: G.dModel },
      slabClassBytes: budgetBytes, // packing ExpertCache: soft, si tronca al disponibile
      // arena expert (C3a fase 4 strato 1): quanti buffer di classe e quanto
      // grandi. L'aritmetica e' quella della cache (residency), non ricopiata.
      ...arenaNeeds({
        budgetBytes, slotsOverride,
        maxBufferBytes: adapter.limits.maxBufferSize,
        maxBindingBytes: adapter.limits.maxStorageBufferBindingSize,
      }),
    }),
  });
  const hasTsq = has("timestamp-query");
  const limits = grantedLimits(device);
  const { maxBindingBytes: maxBind, maxBufferBytes: maxBuf } = slabBufferCap(device);

  progress("carico token_embd…");
  const embd = source.nonExpert("token_embd.weight");
  const xRow = new Float32Array(G.dModel);
  const embed = (tok: number): Float32Array => {
    dequantQ4_0Row(embd, 0, G.dModel, tok, xRow);
    return xRow;
  };

  progress(`modello 47 layer + head, ctxMax ${ctxMax}, slab ${(budgetBytes / (1 << 30)).toFixed(2)} GiB` +
    (slabBudget ? ` [ctx-aware: ceiling ${(slabBudget.allocCeilingBytes / (1 << 30)).toFixed(2)} − nonExp − kv ${(slabBudget.kvBytes / 2 ** 20).toFixed(0)} MiB]` : "") +
    (optimistic ? ` — OPTIMISTIC, slot ${JSON.stringify(slotsOverride)} (preload…)` : "") + "…");
  const tBuild0 = performance.now();
  const model = createGlmModel(device, source, {
    ctxMax, head: true,
    select: optimistic ? "optimistic" : undefined,
    // C3c fase 6: meccanismi del regime sync ai budget stretti
    prefetch: cfg.prefetch,
    // capacità di telemetria allocata, REGISTRAZIONE spenta: le repliche
    // headline (quelle dei gate) girano a overhead nullo
    telemetry: false, telemetryGpu: hasTsq,
    cache: { budgetBytes, slotsOverride, policy: cfg.policy, maxBindingBytes: maxBind, maxBufferBytes: maxBuf, timing: true },
  });
  const buildMs = performance.now() - tBuild0;

  const argmax = (lg: Float32Array): number => {
    let a = 0;
    for (let t = 1; t < lg.length; t++) if (lg[t] > lg[a]) a = t;
    return a;
  };

  // Una replica completa: prefill (posizioni 0..nPrompt-1, logits solo
  // sull'ultima) + decode greedy di nGen token. La KV riparte da pos 0 a ogni
  // replica: l'attention legge 0..pos, le entry oltre pos sono irrilevanti.
  const runOnce = async (label: string, withTelem = false): Promise<RunResult> => {
    const tel0 = withTelem ? await model.telemetry() : null;
    const rPre0 = model.cacheStats();
    const tPre0 = performance.now();
    let last = 0;
    let ttftMs = 0;
    const preTok: number[] = [];
    if (cfg.prefillBatch) {
      // fase 5: prefill batched (prefillChunk M=16, identita' bit-provata —
      // prefill-identity-p{4,6}-2026-08-06.json). Il per-token di preTok e' il
      // costo medio del chunk: l'attribuzione fine per-riga non esiste qui.
      for (let base = 0; base < nPrompt; base += GLM_PREFILL_M) {
        const m = Math.min(GLM_PREFILL_M, nPrompt - base);
        const rows = new Float32Array(m * G.dModel);
        for (let r = 0; r < m; r++) rows.set(embed(promptTokens[base + r]), r * G.dModel);
        const isLast = base + m >= nPrompt;
        const tTok = performance.now();
        const res = await model.prefillChunk(rows, base, isLast);
        const ms = performance.now() - tTok;
        for (let r = 0; r < m; r++) preTok.push(ms / m);
        if (isLast) { last = argmax(res.logits!); ttftMs = performance.now() - tPre0; }
        post({ type: "tick", msg: `${label}: prefill ${Math.min(base + m, nPrompt)}/${nPrompt} (${(1000 * Math.min(base + m, nPrompt) / (performance.now() - tPre0)).toFixed(2)} tok/s)` });
      }
    } else {
    for (let i = 0; i < nPrompt; i++) {
      const tTok = performance.now();
      const wantLogits = i === nPrompt - 1;
      const r = await model.forward(embed(promptTokens[i]), i, wantLogits);
      preTok.push(performance.now() - tTok);
      if (wantLogits) { last = argmax(r.logits!); ttftMs = performance.now() - tPre0; }
      if (i % 32 === 0) {
        post({ type: "tick", msg: `${label}: prefill ${i}/${nPrompt} (${(1000 * (i + 1) / (performance.now() - tPre0)).toFixed(2)} tok/s)` });
      }
    }
    }
    const preMs = performance.now() - tPre0;
    const rPre1 = model.cacheStats();
    const tel1 = withTelem ? await model.telemetry() : null;

    const tDec0 = performance.now();
    const decTok: number[] = [];
    const decodeTokens: TokenRec[] = [];
    const generated: number[] = [];
    for (let k = 0; k < cfg.nGen; k++) {
      generated.push(last);
      const sBefore = model.cacheStats();
      const tTok = performance.now();
      const r = await model.forward(embed(last), nPrompt + k, true);
      const ms = performance.now() - tTok;
      const d = residencyDelta(sBefore, model.cacheStats());
      decTok.push(ms);
      decodeTokens.push({ ms, stallMs: d.stallMs, misses: d.misses });
      last = argmax(r.logits!);
      if (k % 8 === 0) {
        post({ type: "tick", msg: `${label}: decode ${k}/${cfg.nGen} (${(1000 * (k + 1) / (performance.now() - tDec0)).toFixed(2)} tok/s)` });
      }
    }
    const decMs = performance.now() - tDec0;
    const rDec1 = model.cacheStats();
    const tel2 = withTelem ? await model.telemetry() : null;

    const phase = (tokens: number, ms: number, per: number[], res: PhaseResidency): PhaseResult => ({
      tokens, ms, toksPerSec: (tokens / ms) * 1000, msPerToken: ms / tokens,
      msPerTokenP50: median(per), residency: res,
    });
    const out: RunResult = {
      prefill: phase(nPrompt, preMs, preTok, residencyDelta(rPre0, rPre1)),
      decode: phase(cfg.nGen, decMs, decTok, residencyDelta(rPre1, rDec1)),
      decodeTokens, generated, ttftMs,
      telem: tel0 && tel1 && tel2 ? { prefill: telemDelta(tel0, tel1), decode: telemDelta(tel1, tel2) } : undefined,
    };
    post({
      type: "progress",
      msg: `${label}: prefill ${out.prefill.toksPerSec.toFixed(2)} tok/s — decode ${out.decode.toksPerSec.toFixed(2)} tok/s ` +
        `(hit ${(100 * (out.decode.residency.hitRate ?? 0)).toFixed(1)}%, stallo ${(out.decode.residency.stallMs / cfg.nGen).toFixed(1)} ms/token)`,
    });
    return out;
  };

  const warmup = await runOnce("warmup (scartato)");
  const reps: RunResult[] = [];
  for (let r = 0; r < cfg.replicates; r++) reps.push(await runOnce(`replica ${r + 1}/${cfg.replicates}`));

  // ---- repliche DEDICATE di attribuzione (C3a fase 1) ----
  // L'headline resta dalle repliche a telemetria SPENTA (pattern Qwen: mai
  // confrontare wall di finestre strumentate e non). Qui si accende tutto e si
  // rimisura: il wall di queste repliche è più alto e NON va nei gate.
  const attribReps: RunResult[] = [];
  const nAttrib = cfg.attribReplicates ?? 1;
  if (nAttrib > 0) {
    model.setTelemetry(true, true);
    for (let r = 0; r < nAttrib; r++) attribReps.push(await runOnce(`attribuzione ${r + 1}/${nAttrib}`, true));
    model.setTelemetry(false);
  }

  // ---- replica DEDICATA all'attribuzione per CATEGORIA (C3a fase 4b) ----
  // Separata dalle precedenti perche' spezza i compute pass ai confini di
  // categoria: il suo gpuBusy totale contiene i confini in piu' e non e'
  // confrontabile alla lettera con l'headline. Le QUOTE lo sono, e il
  // confronto fra i due totali dichiara quanto costano i confini.
  let catRep: RunResult | null = null;
  if (nAttrib > 0) {
    progress("attribuzione per categoria…");
    model.setTelemetry(true, true, true);
    catRep = await runOnce("attribuzione categorie", true);
    model.setTelemetry(false);
  }

  // ---- probe del floor di sync: mapAsync round-trip a GPU ~vuota ----
  // Cross-check indipendente: 46 readback router/token non possono costare meno
  // di 46 × questo numero, qualunque sia il resto.
  progress("probe floor di sync…");
  const probeSrc = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const probeDst = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const probeSamples: number[] = [];
  for (let i = 0; i < 64; i++) {
    const p0 = performance.now();
    const pe = device.createCommandEncoder();
    pe.copyBufferToBuffer(probeSrc, 0, probeDst, 0, 16);
    device.queue.submit([pe.finish()]);
    await probeDst.mapAsync(GPUMapMode.READ);
    probeDst.unmap();
    probeSamples.push(performance.now() - p0);
  }
  probeSrc.destroy(); probeDst.destroy();

  const decodeTps = stats(reps.map((r) => r.decode.toksPerSec));
  const prefillTps = stats(reps.map((r) => r.prefill.toksPerSec));
  const GATE_DECODE = 13.43, GATE_PREFILL = 56.58;
  const st = model.cacheStats();
  const nMoe = G.nLayer - G.denseLead;

  // aggregati di telemetria sulle repliche (mediane), input C3
  const decRes = reps.map((r) => r.decode.residency);
  const tele = {
    // NOMI ESPLICITI sulla provenienza (C3a it.3): i campi `*Planned`/`*Expected`
    // vengono dal piano statico, quelli `*Measured` da contatori a runtime.
    // Prima di questa iterazione `dispatchesPerToken` e `syncsPerToken` erano
    // valori derivati con nomi che li facevano sembrare misure.
    dispatchesPerTokenPlanned: model.dispatchesPerTokenPlanned,
    // sync path: 46 readback router + 1 di coda. Ottimistico: 1 readback di
    // coda per GIRO ⇒ atteso 1+P(dirty) <= 2 — e' il gate strutturale C3b.
    syncsPerTokenExpected: optimistic ? 2 : nMoe + 1,
    syncsNote: optimistic
      ? "optimistic: 1 readback di coda per giro (pulito 1, sporco 2 col replay); il valore e' il TETTO del gate strutturale, l'atteso vero e' 1+P(dirty). Confronto C3a sync path: 47/token."
      : "sync path: 46 readback router + 1 di coda per token",
    decode: {
      // ATTENZIONE (optimistic): gli ensure del decode avvengono SOLO nel
      // repair, sui soli miss ⇒ hitRate/retention 0% PER COSTRUZIONE — il
      // contatore cambia semantica, non e' una regressione vs i bench sync
      // (hit ~99%). La residenza effettiva e' optimistic.residencyRatio.
      hitRateNote: optimistic ? "semantica diversa in optimistic: solo i miss passano da ensure — v. optimistic.residencyRatio" : undefined,
      hitRate: median(decRes.map((r) => r.hitRate ?? 0)),
      // retention accanto all'hitRate, mai da sola ne' lui da solo (4d,
      // lezione kimi-k3-in-c: e' l'unica delle tre definizioni di "hit rate"
      // che concorda coi byte letti quando esiste un prefetch)
      retention: median(decRes.map((r) => r.retention ?? 0)),
      missesPerToken: median(decRes.map((r) => r.misses / cfg.nGen)),
      stallMsPerToken: median(decRes.map((r) => r.stallMs / cfg.nGen)),
      readMsPerToken: median(decRes.map((r) => r.readMs / cfg.nGen)),
      packMsPerToken: median(decRes.map((r) => r.packMs / cfg.nGen)),
      uploadMsPerToken: median(decRes.map((r) => r.uploadMs / cfg.nGen)),
      msPerToken: median(reps.map((r) => r.decode.msPerToken)),
      // quota di wall non spiegata dallo stallo residenza = GPU + sync + CPU
      residuoMsPerToken: median(reps.map((r, i) => r.decode.msPerToken - decRes[i].stallMs / cfg.nGen)),
      evictionsPerToken: median(decRes.map((r) => r.evictions / cfg.nGen)),
      // IL REGIME DI LETTURA DELLE REPLICHE, nella headline e non solo nelle
      // fasi: e' il numero che i gate confrontano, quindi e' li' che deve stare
      // l'avviso. `os-cache` significa che questo decode NON e' confrontabile
      // con uno preso a cache fredda — v. OPFS_DEVICE_CEILING_GIBS (it.20).
      ...(() => {
        const tot = (xs: number[]): number => xs.reduce((a, x) => a + x, 0);
        const b = readBandwidth(tot(decRes.map((r) => r.bytesRead)), tot(decRes.map((r) => r.readMs)));
        return { readGiBs: b.gibs, readRegime: b.regime };
      })(),
    },
    occupancy: { occupied: st.occupied, slots: st.slots },
    // attribuzione senza strumentare il motore: i token a ZERO miss danno il
    // costo puro (1.816 dispatch + 47 sync CPU↔GPU); la pendenza per miss dà
    // il costo marginale della residenza (retta ai minimi quadrati su tutti i
    // token di decode delle repliche).
    attribution: (() => {
      const all = reps.flatMap((r) => r.decodeTokens);
      const zero = all.filter((t) => t.misses === 0).map((t) => t.ms);
      const n = all.length;
      const mx = all.reduce((a, t) => a + t.misses, 0) / n;
      const my = all.reduce((a, t) => a + t.ms, 0) / n;
      const sxy = all.reduce((a, t) => a + (t.misses - mx) * (t.ms - my), 0);
      const sxx = all.reduce((a, t) => a + (t.misses - mx) ** 2, 0);
      const slope = sxx > 0 ? sxy / sxx : null;
      const withMiss = all.filter((t) => t.misses > 0);
      return {
        tokens: n, zeroMissTokens: zero.length,
        msPerTokenZeroMiss: zero.length ? median(zero) : null,
        msPerTokenNetStall: median(all.map((t) => t.ms - t.stallMs)),
        fitInterceptMs: slope === null ? null : my - slope * mx,
        fitSlopeMsPerMiss: slope,
        missCostMsMedian: withMiss.length ? median(withMiss.map((t) => t.stallMs / t.misses)) : null,
      };
    })(),
  };

  // ---- gap dalla funzione obiettivo (riportato, MAI gate) ----
  const ttft = stats(reps.map((r) => r.ttftMs));
  const gapTps = (target: number) => ({
    targetToksPerSec: target, measuredToksPerSec: decodeTps.median,
    factor: decodeTps.median > 0 ? target / decodeTps.median : null,
    deficitMsPerToken: 1000 / decodeTps.median - 1000 / target,
  });
  const objective = {
    note: "funzione obiettivo direction §2 (ruling PI 2026-08-01): il floor CPU 13.43/56.58 è gate d'ingresso INTERMEDIO, non obiettivo. Questi numeri non sono gate; la loro assenza dal report è però un FAIL di checklist (GOAL C3a).",
    gapDecode30: gapTps(UX_DECODE_TPS),
    gapDecode60: gapTps(UX_DECODE_TPS_THINKING),
    gapTtft4s: {
      budgetMs: UX_TTFT_MS, measuredMs: ttft.median,
      factor: ttft.median / UX_TTFT_MS,
      definition: "dal primo forward di prefill al primo token generato disponibile, modello GIÀ residente (il TTFT a freddo è metrica C3b/instant-on)",
      impliedPrefillToksPerSec: (nPrompt / UX_TTFT_MS) * 1000,
    },
  };

  // ---- attribuzione: quanto del wall è GPU e quanto è sync/CPU ----
  // gpuBusy = somma delle durate dei compute pass (NON include le bolle fra
  // submit). routerWait ⊇ parte di gpuBusy (mentre si aspetta il readback la
  // GPU sta lavorando): i due NON si sommano — la scomposizione ortogonale è
  // wall = gpuBusy + stallo residenza + (sync/CPU non-GPU).
  const attribution2 = (() => {
    const withT = attribReps.filter((r) => r.telem);
    if (!withT.length) return null;
    const per = (f: (t: TelemDelta) => number) => median(withT.map((r) => f(r.telem!.decode) / cfg.nGen));
    const wall = median(withT.map((r) => r.decode.msPerToken));
    const stall = median(withT.map((r) => r.decode.residency.stallMs / cfg.nGen));
    const gpu = withT[0].telem!.decode.gpuBusyMs === null ? null : per((t) => t.gpuBusyMs!);
    const encode = per((t) => t.encodeCpuMs);
    const routerWait = per((t) => t.routerWaitMs);
    const tailWait = per((t) => t.tailWaitMs);
    const ensure = per((t) => t.ensureMs);
    const syncCpu = gpu === null ? null : wall - gpu - stall;
    // proiezione: se i readback router fossero batchati a profondità K, il
    // termine sync scala come 1/K (stesso modello del decode-attrib Qwen)
    const project = (K: number) => {
      if (syncCpu === null) return null;
      const ms = gpu! + stall + syncCpu / K;
      return { K, msPerToken: ms, toksPerSec: 1000 / ms };
    };
    return {
      windowNote: "SOLO repliche di attribuzione (telemetria accesa): il wall qui è più alto dell'headline e non va nei gate",
      replicates: withT.length,
      wallMsPerToken: wall,
      gpuBusyMsPerToken: gpu,
      stallResidenzaMsPerToken: stall,
      syncCpuMsPerToken: syncCpu,
      quotaFuoriGpu: gpu === null ? null : (wall - gpu) / wall,
      misure: { encodeCpuMsPerToken: encode, routerWaitMsPerToken: routerWait, tailWaitMsPerToken: tailWait, ensureMsPerToken: ensure },
      // check di identità: i quattro segmenti sono DISGIUNTI e coprono il wall.
      // Se `unattributed` non è ~0 la strumentazione ha un buco o un doppio
      // conteggio — è il controllo che ha scoperto il double-count di ensure.
      identity: {
        sumSegmentiMsPerToken: encode + ensure + routerWait + tailWait,
        wallMsPerToken: wall,
        unattributedMsPerToken: wall - (encode + ensure + routerWait + tailWait),
        note: "encode/ensure/routerWait/tailWait sono disgiunti per costruzione; gpuBusy invece si SOVRAPPONE a routerWait (mentre si aspetta il readback la GPU esegue) e non va sommato",
      },
      routerSyncsPerToken: median(withT.map((r) => r.telem!.decode.routerSyncs / cfg.nGen)),
      submitsPerToken: median(withT.map((r) => r.telem!.decode.submits / cfg.nGen)),
      dispatchesPerTokenMeasured: median(withT.map((r) => r.telem!.decode.dispatches / cfg.nGen)),
      gpuPassesPerToken: median(withT.map((r) => r.telem!.decode.gpuPasses / cfg.nGen)),
      gpuPassOverflow: withT.reduce((a, r) => a + r.telem!.decode.gpuPassOverflow, 0),
      // il batching dei router sync non esiste in optimistic (routerSyncs 0):
      // la proiezione sarebbe rumore accanto al misurato — si sopprime.
      projectionByK: optimistic ? null : [2, 4, 8, 46].map(project),
      caveats: [
        "Chrome quantizza i timestamp GPU (~100 µs): con ~140 pass/token l'errore per pass si media, ma gpuBusy va letto come stima, non come misura esatta.",
        "routerWait include il tempo in cui la GPU esegue: NON è tutto overhead recuperabile. Il limite inferiore recuperabile è syncFloorProbe × routerSyncsPerToken.",
        "syncCpu è un RESIDUO (wall − gpuBusy − stallo): assorbe encode CPU, latenza submit→start, event loop e l'argmax del bench.",
      ],
    };
  })();

  // Attribuzione di gpuBusy per categoria di kernel: quale famiglia possiede il
  // tempo GPU. E' il primo task che spec §4 prescrive alla fase 4b, e la
  // domanda a cui risponde e' "dove conviene portare le fusioni".
  const attributionByCategory = (() => {
    const t = catRep?.telem?.decode;
    if (!t?.gpuByCatMs) return null;
    const perTok = Object.fromEntries(
      Object.entries(t.gpuByCatMs).map(([k, v]) => [k, v / cfg.nGen]),
    );
    const tot = Object.values(perTok).reduce((a, b) => a + b, 0);
    const quota = Object.fromEntries(Object.entries(perTok).map(([k, v]) => [k, v / tot]));
    return {
      note: "replica dedicata: i compute pass sono spezzati ai confini di categoria, "
        + "quindi il TOTALE contiene confini che l'headline non ha. Confrontare le quote, non i totali.",
      wallMsPerToken: catRep!.decode.msPerToken,
      gpuBusyMsPerToken: tot,
      gpuBusyMsPerTokenSenzaSplit: attribution2?.gpuBusyMsPerToken ?? null,
      costoDeiConfiniMsPerToken: attribution2?.gpuBusyMsPerToken == null ? null : tot - attribution2.gpuBusyMsPerToken,
      msPerToken: perTok,
      quota,
      passesPerToken: Object.fromEntries(
        Object.entries(catRep!.telem!.decode.gpuByCatPasses ?? {}).map(([k, v]) => [k, v / cfg.nGen]),
      ),
    };
  })();


  const probe = stats(probeSamples);
  const syncFloor = {
    mapRoundTripMs: probe,
    routerSyncsPerToken: attribution2?.routerSyncsPerToken ?? null,
    floorMsPerToken: (attribution2?.routerSyncsPerToken ?? 0) * probe.median,
    note: "floor teorico dei soli readback router a GPU scarica: nessun meccanismo che li mantenga può costare meno",
  };

  model.destroy();
  source.close();

  post({
    type: "done",
    report: {
      kind: "glm-bench", schemaVersion: 3, date: new Date().toISOString(),
      ggufSha256: GLM47_FLASH_SHA256,
      config: {
        promptIdx: cfg.prompt, promptId: pr.id, promptTokens: nPrompt, nGen: cfg.nGen,
        replicates: cfg.replicates,
        // con la formula ctx-aware il budget riportato è quello CALCOLATO
        budgetGiB: slabBudget ? +(budgetBytes / (1 << 30)).toFixed(3) : cfg.budgetGiB,
        budgetMode: slabBudget ? "ctx-aware (spec c3c §2)" : "fisso (config)",
        // C3c fase 6: regime sync ai budget stretti
        prefetch: cfg.prefetch ?? null, policy: cfg.policy ?? "lru",
        parkFrac: cfg.parkFrac ?? null, slotsOverride: slotsOverride ?? null,
        // spec §2: il report mostra il budget calcolato con gli addendi
        slabBudget, ctxMax,
        prefillPath: cfg.prefillBatch ? `chunked M=${GLM_PREFILL_M} (prefillChunk, fase 5)` : "decode-only (forward per posizione)",
        decodePath: optimistic
          ? 'single-step greedy su select:"optimistic" (1 submit + repair/replay, C3b)'
          : "single-step greedy (argmax dai logits del motore)",
        selectMode: optimistic ? "optimistic" : "cpu",
        rateWindow: "intera fase (prefill: nPrompt posizioni; decode: nGen token)",
        protocol: "B2: warmup scartato + N repliche, mediana come headline",
      },
      gates: {
        decodeGateToksPerSec: GATE_DECODE, decodeMedian: decodeTps.median,
        decodePass: decodeTps.median >= GATE_DECODE,
        prefillGateToksPerSec: GATE_PREFILL, prefillMedian: prefillTps.median,
        prefillPass: prefillTps.median >= GATE_PREFILL,
        // fase 4d — Planned vs Measured: il piano statico non contiene la testa
        // (rms + lm_head = 2 dispatch/token, eseguita a ogni token del decode
        // con readLogits — nota it.8): l'atteso e' planned+2 ESATTO, contatori
        // interi su piano statico. Uno scarto = drift piano/path. null (attrib
        // spenta) = gate non valutato, neutro per l'exit (page.ts).
        // In OTTIMISTICO il gate d'uguaglianza non e' valutabile: il replay
        // aggiunge dispatch LEGITTIMI (frazione di layer rigiocata) — si
        // riporta la banda attesa invece dell'uguaglianza.
        dispatchPlan: attribution2 && !optimistic ? {
          planned: model.dispatchesPerTokenPlanned,
          expectedWithHead: model.dispatchesPerTokenPlanned + 2,
          measured: attribution2.dispatchesPerTokenMeasured,
          pass: attribution2.dispatchesPerTokenMeasured === model.dispatchesPerTokenPlanned + 2,
        } : null,
        // gate STRUTTURALE C3b (contratto v2): sync/token <= 2 e submits/token
        // <= 2 in media steady-state sul ramo decode di PRODUZIONE. In
        // ottimistico ogni giro ha UN readback di coda ⇒ syncs == submits.
        structural: optimistic && attribution2 ? {
          submitsPerToken: attribution2.submitsPerToken,
          syncsPerToken: attribution2.submitsPerToken + attribution2.routerSyncsPerToken,
          routerSyncsPerToken: attribution2.routerSyncsPerToken,
          gate: 2,
          pass: attribution2.submitsPerToken <= 2
            && attribution2.submitsPerToken + attribution2.routerSyncsPerToken <= 2,
          vsC3aSyncPath: { syncsPerToken: nMoe + 1, note: "il termine che il decode ottimistico elimina (46 readback router + 1 coda)" },
        } : null,
        floorSource: "results/engine/moe-oracle/llama-bench-glm47flash-q4_0-2026-07-30.json (llama.cpp 5f55650, CPU i9-14900HX 16 thread, n_prompt 512 / n_gen 64)",
      },
      decodeToksPerSec: decodeTps, prefillToksPerSec: prefillTps,
      ttftMs: ttft,
      objective,
      // ---- statistiche del regime ottimistico (input della fase 5: confronto
      // con wp0-replay-sim ALLO STESSO budget slot) ----
      optimistic: optimistic ? (() => {
        const withT = attribReps.filter((r) => r.telem);
        const d = (f: (t: TelemDelta) => number) => withT.length ? median(withT.map((r) => f(r.telem!.decode))) : null;
        const dirtyTok = d((t) => t.dirtyTokens);
        const replays = d((t) => t.replays);
        const replayLayers = d((t) => t.replayLayers);
        return {
          slots: slotsOverride, park: modelExpertPark(G.nLayer),
          slotsPolicy: "q4_1-first (docket c3b item 6b): q4_1 intera, resto a q4_0",
          residencyRatio: slotsOverride
            ? (slotsOverride.q4_0 + slotsOverride.q4_1) / (modelExpertPark(G.nLayer).q4_0 + modelExpertPark(G.nLayer).q4_1)
            : null,
          // dalla finestra di attribuzione (contatori accesi):
          pDirty: dirtyTok === null ? null : dirtyTok / cfg.nGen,
          replaysPerToken: replays === null ? null : replays / cfg.nGen,
          replayFracPerDirty: replays && replayLayers ? replayLayers / (replays * G.nLayer) : null,
          repairMsPerToken: d((t) => t.repairMs / cfg.nGen),
          // dalla headline (sempre misurate): miss e stallo per token stanno
          // gia' in telemetry.decode (missesPerToken, stallMsPerToken)
          note: "pDirty/replayFrac/repair dalla finestra di attribuzione; miss/stallo per token in telemetry.decode (headline); la tassa si confronta in fase 5 con wp0-replay-sim-2026-08-06.json allo stesso budget slot",
        };
      })() : null,
      attribution2, attributionByCategory, syncFloorProbe: syncFloor,
      timestampQuery: { available: hasTsq, used: attribution2?.gpuBusyMsPerToken != null },
      // I limiti CONCESSI, non quelli sperati: una prestazione misurata su
      // limiti diversi non e' confrontabile (lezione B2).
      deviceLimits: limits,
      deviceFeatures: [...device.features].sort(), // del DEVICE: quelle dell'adapter sono solo annunciate
      adapterFeatures: [...adapter.features].sort(),
      telemetry: tele,
      warmup: { prefill: warmup.prefill, decode: warmup.decode },
      reps,
      buildMs, wallMs: performance.now() - t0,
      importMs: source.importMs,
    },
  });
}

self.onmessage = (ev: MessageEvent) => {
  void main(ev.data as Cfg).catch((e) => post({ type: "error", message: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) }));
};
