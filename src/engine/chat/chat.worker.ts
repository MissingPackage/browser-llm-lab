// Chat interattiva sul motore q35 (banco di prova, non un gate).
//
// PERCHÉ ESISTE. Tutti gli harness di questo repo sono TEACHER-FORCED: il
// golden fissa i token e il motore li replica. Non c'era un solo posto dove
// battere un prompt e leggere la risposta — cioè l'unico regime in cui il
// motore fa quello per cui esiste. Questa pagina è quel posto, e serve anche
// a esportare la conversazione insieme ai PARAMETRI che l'hanno prodotta:
// un tok/s senza il tetto VRAM, la policy e il path di select accanto non è
// un numero, è un aneddoto.
//
// NON TOCCA UNA RIGA DEL MOTORE. Tutto ciò che serve era già pubblico
// (`createQ35GpuModel`, `q35TokenizerFromMetadata`, `createEngineDevice`,
// `arenaNeeds`, `validateQwen35`): questo modulo è un CONSUMATORE. È una
// proprietà voluta, non una coincidenza — il CHECKPOINT A pretende l'albero
// congelato, e un banco di prova che modifica ciò che misura non è un banco
// di prova.
//
// AMBITO: famiglia Qwen 3.5/3.6 (4B, 9B, 35B-A3B). GLM-4.7-Flash NON è
// pilotabile da qui e la ragione è strutturale, non una svista: il suo path
// consuma id PRE-TOKENIZZATI dall'oracolo e in-engine non esiste un
// tokenizer GLM (q35tokenizer è il primo e unico, ed è BPE Qwen). Aggiungerlo
// è lavoro, non configurazione.
import { createEngineDevice } from "../gpudevice";
import { parseGguf, type GgufTensorInfo } from "../gguf";
import { createQ35GpuModel, q35TensorBytes, type Q35GpuModel } from "../q35gpumodel";
import { q35MoeConfig } from "../q35expertstore";
import { arenaNeeds } from "../residency";
import { validateQwen35, Q35_SHA256, type Q35Shape } from "../q35shape";
import { q35TokenizerFromMetadata, type Q35Tokenizer } from "../q35tokenizer";
import { ggufRangeReader } from "../ggufrange";

// I GGUF canonici: le SHA sono quelle PINNATE in q35shape, non ricopiate a
// mano — se il pin cambia, questa tabella cambia con lui.
const MODELS = {
  "4b": { url: "/models/Qwen3.5-4B-Q4_0.gguf", file: "Qwen3.5-4B-Q4_0.gguf", sha: Q35_SHA256["Qwen3.5-4B"] },
  "9b": { url: "/models/Qwen3.5-9B-Q4_0.gguf", file: "Qwen3.5-9B-Q4_0.gguf", sha: Q35_SHA256["Qwen3.5-9B"] },
  "35b": { url: "/models/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", file: "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", sha: Q35_SHA256["Qwen3.6-35B-A3B"] },
} as const;
export type ModelKey = keyof typeof MODELS;

export interface LoadCfg {
  model: ModelKey;
  /** tetto VRAM in GiB: il budget expert si DERIVA da qui (fase 5, it.35). */
  vramGiB: number | null;
  ctxMax: number;
  select: "cpu" | "optimistic";
  expertPolicy: "lru" | "tier";
}

export interface SamplingCfg {
  /** 0 = greedy: si usa l'argmax che `step` calcola, senza toccare i logits. */
  temperature: number;
  topK: number;
  topP: number;
  seed: number;
  maxNew: number;
}

const post = (m: unknown): void => (self as unknown as Worker).postMessage(m);
const progress = (msg: string): void => post({ type: "progress", msg });

let URL_GGUF: string = MODELS["4b"].url;
// il lettore condiviso (riga 2b): e' qui che vivranno la finestra di
// concorrenza e il coalescing, e questo worker li eredita senza cambiare riga
const range = ggufRangeReader(() => URL_GGUF, "chat");

// ——— stato del worker (un modello alla volta: la VRAM è una) ———
let model: Q35GpuModel | null = null;
let device: GPUDevice | null = null;
let tokenizer: Q35Tokenizer | null = null;
let shape: Q35Shape | null = null;
let loadCfg: LoadCfg | null = null;
let eosIds: number[] = [];
let chatTemplateRaw: string | null = null;
/** posizione del PROSSIMO token nel contesto: la KV si riusa fra i turni. */
let pos = 0;
/** il turno precedente si è chiuso da solo con <|im_end|>? */
let assistantOpen = false;
let stopRequested = false;

/** mulberry32: campionamento RIPRODUCIBILE dal seed esportato nel JSON. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * temperature → top-k → top-p, nell'ordine di llama.cpp. Gira sui logits che
 * `step(..., read=true)` ha già portato in CPU: non costa un readback in più.
 */
function sampleFrom(logits: Float32Array, s: SamplingCfg, rand: () => number): number {
  const k = s.topK > 0 ? Math.min(s.topK, logits.length) : logits.length;
  // top-k per selezione parziale: l'ordinamento pieno di 248k float per token
  // sarebbe il costo dominante del decode sul 4B.
  const idx = Array.from(logits.keys());
  idx.sort((a, b) => logits[b] - logits[a]);
  const top = idx.slice(0, k);
  const maxLogit = logits[top[0]];
  const probs = top.map((i) => Math.exp((logits[i] - maxLogit) / s.temperature));
  const sum = probs.reduce((a, b) => a + b, 0);
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  // top-p sul cumulato (i probs sono già decrescenti)
  let cut = probs.length;
  if (s.topP > 0 && s.topP < 1) {
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (acc >= s.topP) { cut = i + 1; break; }
    }
  }
  let norm = 0;
  for (let i = 0; i < cut; i++) norm += probs[i];
  let r = rand() * norm;
  for (let i = 0; i < cut; i++) {
    r -= probs[i];
    if (r <= 0) return top[i];
  }
  return top[cut - 1];
}

async function load(cfg: LoadCfg): Promise<void> {
  if (model) { model.destroy(); model = null; }
  if (device) { device.destroy(); device = null; }
  const t0 = performance.now();
  const M = MODELS[cfg.model];
  URL_GGUF = M.url;
  loadCfg = cfg;

  progress(`header di ${M.file}…`);
  const header = await range(0, 64 * 1024 * 1024);
  const f = parseGguf(header.buffer.slice(header.byteOffset, header.byteOffset + header.byteLength) as ArrayBuffer);
  const v = validateQwen35(f);
  shape = v.shape;
  const byName = v.byName;
  const info = (name: string): GgufTensorInfo => {
    const t = byName.get(name);
    if (!t) throw new Error(`chat: tensore ${name} assente`);
    return t;
  };

  tokenizer = q35TokenizerFromMetadata(f.metadata);
  // EOS dai METADATA, non da una costante: il 4B e il 35B oggi condividono
  // l'id 248046 (<|im_end|>) ma è una proprietà del file, non del codice.
  const eos = f.metadata["tokenizer.ggml.eos_token_id"];
  if (typeof eos !== "number") throw new Error("chat: tokenizer.ggml.eos_token_id assente");
  const eot = tokenizer.encode("<|endoftext|>", true);
  eosIds = [eos, ...(eot.length === 1 ? eot : [])];
  const ct = f.metadata["tokenizer.chat_template"];
  chatTemplateRaw = typeof ct === "string" ? ct : null;

  const isMoe = shape.arch === "qwen35moe";
  const ceilingBytes = cfg.vramGiB !== null ? Math.floor(cfg.vramGiB * (1 << 30)) : null;
  const arenaBudgetBytes = ceilingBytes ?? 12 * (1 << 30);
  const moeCfg = isMoe ? q35MoeConfig(shape, info) : null;
  progress(`device (ctxMax ${cfg.ctxMax}${isMoe ? `, arena ${(arenaBudgetBytes / 2 ** 30).toFixed(1)} GiB` : ""})…`);
  const dev = await createEngineDevice({
    label: "chat",
    // Feature chieste solo se l'adapter le espone (`createEngineDevice` filtra):
    // `subgroups` accende il gemv a riduzione per subgroup, `timestamp-query` è
    // quella che il `ready` qui sotto già DICHIARA con `dev.has(...)` — finché
    // non compariva in questa lista quel flag era falso per costruzione.
    // Il cast: `GPUFeatureName` di @webgpu/types non elenca subgroups.
    optionalFeatures: ["timestamp-query", "subgroups" as GPUFeatureName],
    needs: (adapter) => ({
      ctxMax: cfg.ctxMax,
      head: { vocab: shape!.vocab, dModel: shape!.dModel },
      ...(isMoe ? { slabClassBytes: 2 * (1 << 30) } : {}),
      ...(moeCfg ? arenaNeeds({
        budgetBytes: arenaBudgetBytes,
        maxBufferBytes: adapter.limits.maxBufferSize,
        maxBindingBytes: adapter.limits.maxStorageBufferBindingSize,
        cfg: moeCfg,
      }) : {}),
    }),
  });
  device = dev.device;

  progress("pesi sulla GPU…");
  model = await createQ35GpuModel(dev.device, {
    shape,
    info,
    read: (name) => range(f.dataOffset + info(name).offset, q35TensorBytes(info(name))),
    readRange: (name, off, len) => range(f.dataOffset + info(name).offset + off, len),
  }, cfg.ctxMax, arenaBudgetBytes, {
    // `optimistic` si può chiedere solo sul MoE: sui densi non c'è arena
    // expert e il path non esiste.
    select: isMoe ? cfg.select : "cpu",
    vramCeilingBytes: ceilingBytes ?? undefined,
    expertPolicy: cfg.expertPolicy,
  });
  pos = 0;
  assistantOpen = false;
  const loadMs = performance.now() - t0;

  post({
    type: "loaded",
    info: {
      model: cfg.model,
      file: M.file,
      sha256: M.sha,
      arch: shape.arch,
      loadMs: Math.round(loadMs),
      ctxMax: cfg.ctxMax,
      vocab: shape.vocab,
      nLayer: shape.nLayer,
      nExpert: shape.nExpert,
      nExpertUsed: shape.nExpertUsed,
      select: isMoe ? cfg.select : "cpu",
      expertPolicy: cfg.expertPolicy,
      dispatchBreakdown: model.dispatchBreakdown,
      vramPlan: model.vramPlan(),
      hasChatTemplate: chatTemplateRaw !== null,
      eosIds,
      adapter: dev.adapter.info ? { vendor: dev.adapter.info.vendor, architecture: dev.adapter.info.architecture, device: dev.adapter.info.device } : null,
      timestampQuery: dev.has("timestamp-query"),
    },
  });
}

/**
 * ChatML della famiglia Qwen, reso QUI e non dal template Jinja del GGUF.
 * DICHIARATO: il `tokenizer.chat_template` del file è un template Jinja con
 * vision/tool-call/think, e questo motore non ha un interprete Jinja. Si emette
 * la forma canonica `<|im_start|>ruolo\n…<|im_end|>\n` e si ESPORTA sia la
 * stringa resa sia gli id: chi confronta con llama.cpp vede esattamente cosa è
 * stato dato in pasto al modello invece di doverlo indovinare.
 */
function renderDelta(user: string, system: string | null, firstTurn: boolean): string {
  let s = "";
  if (assistantOpen) s += "<|im_end|>\n";
  if (firstTurn && system && system.trim().length > 0) s += `<|im_start|>system\n${system}<|im_end|>\n`;
  s += `<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n`;
  return s;
}

async function chat(turn: number, user: string, system: string | null, s: SamplingCfg): Promise<void> {
  if (!model || !tokenizer || !loadCfg) throw new Error("chat: nessun modello caricato");
  stopRequested = false;
  const m = model;
  const tk = tokenizer;

  const delta = renderDelta(user, system, pos === 0);
  const ids = tk.encode(delta, true);
  if (ids.length === 0) throw new Error("chat: delta vuoto");
  if (pos + ids.length + 1 >= loadCfg.ctxMax) {
    throw new Error(`chat: il prompt non entra nel contesto (${pos + ids.length} su ctxMax ${loadCfg.ctxMax})`);
  }

  const perf0 = m.perf();
  const moe0 = m.moeStats ? m.moeStats() : null;
  const posStart = pos;

  // PREFILL: tutti i token del delta tranne l'ultimo senza readback (si
  // accodano), come fa la conformance. L'`await` è obbligatorio sul MoE —
  // `step` contiene i readback del router e il fire-and-forget mette
  // mapAsync concorrenti sullo stesso staging (landmine di it.19).
  const tPre = performance.now();
  for (let i = 0; i < ids.length - 1; i++) await m.step(ids[i], pos++, false);
  const tFirst = performance.now();
  const prefillMs = tFirst - tPre;

  const rand = rng(s.seed);
  const greedy = s.temperature <= 0;
  let next = await m.step(ids[ids.length - 1], pos++, true);
  if (!greedy) next = sampleFrom(m.lastLogits()!, s, rand);
  const firstMs = performance.now() - tFirst;

  const gen: number[] = [];
  const stepMs: number[] = [];
  let emitted = 0;
  let stopReason: "eos" | "maxNew" | "ctx" | "user" = "maxNew";
  for (let i = 0; i < s.maxNew; i++) {
    if (eosIds.includes(next)) { stopReason = "eos"; break; }
    if (stopRequested) { stopReason = "user"; break; }
    if (pos >= loadCfg.ctxMax - 1) { stopReason = "ctx"; break; }
    gen.push(next);
    // Detokenizzazione INCREMENTALE sull'intera coda, non token per token: un
    // token può essere metà di una sequenza UTF-8 e decodificarlo da solo
    // produce U+FFFD (accenti e emoji spezzati a metà).
    const full = tk.decode(gen);
    if (full.length > emitted) {
      post({ type: "token", turn, text: full.slice(emitted) });
      emitted = full.length;
    }
    const t = performance.now();
    const argmax = await m.step(next, pos++, true);
    stepMs.push(performance.now() - t);
    next = greedy ? argmax : sampleFrom(m.lastLogits()!, s, rand);
  }
  assistantOpen = true;

  const perf1 = m.perf();
  const moe1 = m.moeStats ? m.moeStats() : null;
  const sorted = stepMs.slice().sort((a, b) => a - b);
  const p50 = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : null;
  post({
    type: "turnDone",
    turn,
    text: tk.decode(gen),
    stats: {
      renderedPrompt: delta,
      // STATO DELLA CACHE, e non e' decorazione: sul 35B il primo turno dopo il
      // load paga la lettura degli expert da disco e puo' costare 2-3x il
      // regime. Senza questo campo due turni identici sembrano lo stesso
      // esperimento — e' esattamente l'equivoco che ha fatto sembrare "non piu'
      // veloce" un modello misurato a freddo (2026-08-13).
      turnSinceLoad: turn,
      cacheState: turn === 0 ? "cold (primo turno dopo il load)" : "warm",
      promptTokens: ids.length,
      promptIds: ids,
      genTokens: gen.length,
      genIds: gen,
      posStart,
      posEnd: pos,
      stopReason,
      prefillMs,
      prefillTokS: ids.length > 1 ? (ids.length - 1) / (prefillMs / 1000) : null,
      ttftMs: prefillMs + firstMs,
      firstTokenMs: firstMs,
      decodeMsPerTokenP50: p50,
      decodeTokS: p50 !== null ? 1000 / p50 : null,
      decodeMsPerToken: stepMs,
      // i contatori del motore sono CUMULATIVI: qui vanno i delta del turno,
      // altrimenti il secondo turno erediterebbe i miss del primo
      engine: {
        submits: perf1.submits - perf0.submits,
        readbacks: perf1.readbacks - perf0.readbacks,
        dirtyTokens: perf1.dirtyTokens - perf0.dirtyTokens,
        replays: perf1.replays - perf0.replays,
        replayLayers: perf1.replayLayers - perf0.replayLayers,
        repairMs: perf1.repairMs - perf0.repairMs,
        // scomposizione del repair: `repairMs` = fetch + pack + upload + flush
        // + eviction, e `tailCpuMs` = repair + replay + contabilità. Senza
        // questi cinque il 43% del turno resta anonimo (goal 35b-residency).
        fetchRepairMs: perf1.fetchRepairMs - perf0.fetchRepairMs,
        fetchRepairCalls: perf1.fetchRepairCalls - perf0.fetchRepairCalls,
        fetchRepairBytes: perf1.fetchRepairBytes - perf0.fetchRepairBytes,
        fetchPrepMs: perf1.fetchPrepMs - perf0.fetchPrepMs,
        fetchPrepCalls: perf1.fetchPrepCalls - perf0.fetchPrepCalls,
        fetchPrepBytes: perf1.fetchPrepBytes - perf0.fetchPrepBytes,
        replayPassMs: perf1.replayPassMs - perf0.replayPassMs,
        flushMs: perf1.flushMs - perf0.flushMs,
        encodeMs: perf1.encodeMs - perf0.encodeMs,
        readbackMs: perf1.readbackMs - perf0.readbackMs,
        argmaxMs: perf1.argmaxMs - perf0.argmaxMs,
        embedMs: perf1.embedMs - perf0.embedMs,
        tailCpuMs: perf1.tailCpuMs - perf0.tailCpuMs,
        tokenMs: perf1.tokenMs - perf0.tokenMs,
      },
      moe: moe0 && moe1 ? {
        hits: moe1.hits - moe0.hits,
        misses: moe1.misses - moe0.misses,
        uploadedBytes: moe1.uploadedBytes - moe0.uploadedBytes,
        readMs: moe1.readMs - moe0.readMs,
        packMs: moe1.packMs - moe0.packMs,
        uploadMs: moe1.uploadMs - moe0.uploadMs,
        policy: moe1.policy,
      } : null,
    },
  });
}

self.onmessage = (e: MessageEvent) => {
  const m = e.data as { type: string; [k: string]: unknown };
  if (m.type === "stop") { stopRequested = true; return; }
  const run = async (): Promise<void> => {
    if (m.type === "load") await load(m.cfg as LoadCfg);
    else if (m.type === "chat") {
      await chat(m.turn as number, m.user as string, (m.system as string | null) ?? null, m.sampling as SamplingCfg);
    } else if (m.type === "reset") {
      if (model) model.resetState();
      pos = 0;
      assistantOpen = false;
      post({ type: "resetDone" });
    } else if (m.type === "snapshot") {
      post({
        type: "snapshot",
        data: model ? {
          vramPlan: model.vramPlan(),
          dispatchBreakdown: model.dispatchBreakdown,
          moe: model.moeStats ? model.moeStats() : null,
          perf: model.perf(),
          chatTemplateRaw,
        } : null,
      });
    }
  };
  run().catch((err: unknown) => post({ type: "error", message: err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err) }));
};
