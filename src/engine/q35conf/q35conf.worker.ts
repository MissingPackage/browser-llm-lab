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
import { parseGguf } from "../gguf";
import { createQ35GpuModel, q35TensorBytes } from "../q35gpumodel";
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
}

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
  const { device } = await createEngineDevice({
    label: "q35conf",
    needs: {
      ctxMax,
      head: { vocab: shape.vocab, dModel: shape.dModel },
      // MoE (it.17): i chunk dell'arena expert sono buffer da 2 GiB
      ...(isMoe ? { slabClassBytes: 2 * (1 << 30) } : {}),
    },
  });
  const model = await createQ35GpuModel(device, {
    shape,
    info: (name) => {
      const t = byName.get(name);
      if (!t) throw new Error(`q35conf: tensore ${name} assente`);
      return t;
    },
    read: (name) => {
      const t = byName.get(name);
      if (!t) throw new Error(`q35conf: tensore ${name} assente`);
      return range(f.dataOffset + t.offset, q35TensorBytes(t));
    },
    readRange: (name, off, len) => {
      const t = byName.get(name);
      if (!t) throw new Error(`q35conf: tensore ${name} assente`);
      return range(f.dataOffset + t.offset + off, len);
    },
  }, ctxMax, Math.floor((cfg.arenaGiB ?? 12) * (1 << 30)));
  const loadMs = performance.now() - t0;
  progress(`modello su GPU in ${(loadMs / 1000).toFixed(1)} s (${model.dispatchesPerToken} dispatch/token)`);

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
    const tPre = performance.now();
    // await OBBLIGATORIO: sul MoE step() contiene i readback del router
    // (fire-and-forget = mapAsync concorrenti sullo stesso staging, it.19);
    // sul denso con read=false ritorna subito: semantica di misura invariata.
    for (let t = 0; t < P - 1; t++) await model.step(p.promptTokens[t], t, false);
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
      declared: "orchestratore correttezza-prima (562 dispatch/token, zero fusioni/batch, readback logits per token nel decode): FRAME DI PARTENZA pre-ottimizzazioni, non un numero competitivo",
      prompt: { idx: cfg.bench.promptIdx, file: p.file, tokens: P },
      loadMs: Math.round(loadMs),
      prefill: { tokens: P - 1, ms: Math.round(prefillMs), tokS: (P - 1) / (prefillMs / 1000) },
      decode: { n: cfg.bench.nDecode, msPerTokenP50: p50, tokS: 1000 / p50, firstMs: Math.round(firstMs) },
      ttftMs: Math.round(loadMs + prefillMs + firstMs),
      dispatchesPerToken: model.dispatchesPerToken,
      moe: model.moeStats ? model.moeStats() : null,
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
    for (let t = 0; t < tokens.length; t++) {
      const gi = t - (P - 1);
      const need = gi >= 0 && gi < gen;
      const am = await model.step(tokens[t], t, need);
      if (need) {
        engineArgmax.push(am);
        if (am === p.positions[gi].argmax) ok++;
        posTot++;
      }
      if (t % 512 === 0) post({ type: "tick", msg: `p${pi} ${t}/${tokens.length} (top1 ${ok}/${Math.max(0, t - P + 2)})` });
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
    engine: { orchestrator: "q35gpumodel correttezza-prima", dispatchesPerToken: model.dispatchesPerToken, loadMs: Math.round(loadMs) },
    moe: model.moeStats ? model.moeStats() : null,
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
