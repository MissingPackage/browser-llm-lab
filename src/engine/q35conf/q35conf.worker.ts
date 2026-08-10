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
interface Cfg { prompts?: number[]; maxGen?: number }

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const progress = (msg: string) => post({ type: "progress", msg });

const URL_GGUF = "/models/Qwen3.5-4B-Q4_0.gguf";
const Q35_4B_SHA = "298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb";

async function range(off: number, len: number): Promise<Uint8Array> {
  const rr = await fetch(URL_GGUF, { headers: { Range: `bytes=${off}-${off + len - 1}` } });
  if (rr.status !== 206) throw new Error(`q35conf: Range non onorato (${rr.status})`);
  const ab = await rr.arrayBuffer();
  if (ab.byteLength !== len) throw new Error(`q35conf: Range corto ${ab.byteLength}/${len}`);
  return new Uint8Array(ab);
}

async function main(cfg: Cfg): Promise<void> {
  const t0 = performance.now();
  const golden = (await (await fetch("/models/q35/golden-full.json")).json()) as Golden;
  if (golden.modelSha256 !== Q35_4B_SHA) throw new Error("q35conf: SHA GGUF del golden diverso dal pinnato");

  const prompts = golden.prompts.filter((_, i) => !cfg.prompts || cfg.prompts.includes(i));
  const maxGen = cfg.maxGen ?? Infinity;
  const ctxMax = Math.max(...prompts.map((p) => p.promptTokens.length + Math.min(p.generated.length, maxGen))) + 8;

  progress(`golden: ${prompts.length} prompt, ctxMax ${ctxMax}`);
  const header = await range(0, 64 * 1024 * 1024);
  const f = parseGguf(header.buffer.slice(header.byteOffset, header.byteOffset + header.byteLength) as ArrayBuffer);
  const { shape, byName } = validateQwen35(f);

  const { device } = await createEngineDevice({
    label: "q35conf",
    needs: { ctxMax, head: { vocab: shape.vocab, dModel: shape.dModel } },
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
  }, ctxMax);
  const loadMs = performance.now() - t0;
  progress(`modello su GPU in ${(loadMs / 1000).toFixed(1)} s (${model.dispatchesPerToken} dispatch/token)`);

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
    kind: "q35-conf-4b",
    date: new Date().toISOString().slice(0, 10),
    model: "Qwen3.5-4B-Q4_0.gguf",
    modelSha256: golden.modelSha256,
    oracleCommit: golden.oracle.commit,
    corpusHash: golden.corpusHash,
    engine: { orchestrator: "q35gpumodel correttezza-prima", dispatchesPerToken: model.dispatchesPerToken, loadMs: Math.round(loadMs) },
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
