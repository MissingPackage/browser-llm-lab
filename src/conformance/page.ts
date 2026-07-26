// Entry point per lo script Playwright on-demand (scripts/conformance.mjs). NON fa parte
// della build di produzione: conformance.html non è referenziato da nessuna build multi-entry
// e Vite (default: solo index.html come entry) non lo include in `vite build` — vedi
// il report per la verifica `npm run build` fatta a valle.
//
// Esegue il contratto di conformance condiviso (src/conformance/contract.ts) contro
// ENTRAMBI gli adapter reali (motore reale, non un fake iniettato — è proprio questo che
// chiude il gap lasciato dai test unitari), e pubblica il risultato sia su
// `window.__conformanceResults` (per il driver Playwright) sia nel DOM (per un umano che
// apre la pagina).
import { WebLLMAdapter } from "../adapters/webllm";
import { TransformersJsAdapter } from "../adapters/transformersjs";
import { WllamaAdapter } from "../adapters/wllama";
import { runConformanceContract, type ContractResult } from "./contract";
import type { DataType, DeviceType } from "@huggingface/transformers";

const TOKEN_BUDGET = 16; // piccolo apposta: il check cardine (timestamp-per-token) non ha bisogno di altro

// Fixture ONNX minuscola (~4MB, pesi random) di Xenova — ha un chat template, quindi il
// percorso "messages array" dell'adapter è davvero esercitato, non solo un prompt grezzo.
const TRANSFORMERSJS_MODEL_ID = "Xenova/tiny-random-Phi3ForCausalLM";
// Non esiste una fixture MLC minuscola equivalente: il prebuilt realistico più piccolo è
// questo 0.5B, già in cache dai run precedenti (profilo persistente Playwright) quindi
// dovrebbe caricare "warm" e velocemente. "Tiny model" non è letteralmente raggiungibile
// per questo stack — lo segnaliamo qui e nel report, non lo nascondiamo.
const WEBLLM_MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f32_1-MLC";
// Stessa situazione di WebLLM: i GGUF davvero minuscoli (es. stories260K) non hanno chat
// template, quindi non esercitano il percorso `messages` che l'adapter usa davvero. Si usa
// perciò lo stesso 0.5B del bench — dichiarato, non nascosto. Il profilo Playwright è
// persistente, quindi il download da 491 MB avviene una sola volta.
const WLLAMA_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf";

interface ConformanceRun {
  status: "running" | "done" | "error";
  results: ContractResult[];
  error?: string;
}

declare global {
  interface Window {
    __conformanceResults?: ConformanceRun;
  }
}

const $ = (id: string) => document.getElementById(id)!;

function render(run: ConformanceRun): void {
  window.__conformanceResults = run;
  $("status").textContent = run.status;
  if (run.status === "error") {
    $("results").innerHTML = `<pre>${run.error}</pre>`;
    return;
  }
  $("results").innerHTML = run.results
    .map((r) => {
      const passCount = r.checks.filter((c) => c.pass).length;
      const items = r.checks
        .map((c) => `<li style="color:${c.pass ? "green" : "crimson"}">[${c.pass ? "PASS" : "FAIL"}] ${c.name} — ${escapeHtml(c.detail)}</li>`)
        .join("");
      return `<section><h2>${r.adapterId} (${r.modelId}) — ${passCount}/${r.checks.length} passed</h2><ul>${items}</ul></section>`;
    })
    .join("");
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c] ?? c);
}

// Trova la prima combinazione dtype/device che carica DAVVERO in questo browser. La lista
// di dtype supportati dipende dai file .onnx pubblicati per il modello (vedi
// DEFAULT_DTYPE_SUFFIX_MAPPING in @huggingface/transformers) e dal supporto WebGPU per le
// operazioni quantizzate, che varia per build del browser — non è deducibile staticamente,
// va provato dal vivo. q8/webgpu è il tentativo preferito (più vicino al bench reale);
// fp32/wasm è il fallback che funziona ovunque perché non richiede kernel WebGPU quantizzati.
async function pickWorkingConfig(
  modelId: string,
  candidates: Array<{ dtype: DataType; device: DeviceType }>,
): Promise<{ dtype: DataType; device: DeviceType; note: string }> {
  const notes: string[] = [];
  for (const c of candidates) {
    const probe = new TransformersJsAdapter({ dtype: c.dtype, device: c.device });
    try {
      await probe.load(modelId, () => {});
      await probe.dispose();
      return { dtype: c.dtype, device: c.device, note: notes.concat(`OK: dtype=${c.dtype} device=${c.device}`).join(" | ") };
    } catch (e) {
      notes.push(`rejected dtype=${c.dtype} device=${c.device}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  throw new Error(`no working transformers.js config among candidates: ${notes.join(" | ")}`);
}

async function main(): Promise<void> {
  const run: ConformanceRun = { status: "running", results: [] };
  render(run);

  try {
    const { dtype, device, note } = await pickWorkingConfig(TRANSFORMERSJS_MODEL_ID, [
      { dtype: "q8", device: "webgpu" },
      { dtype: "fp32", device: "wasm" },
    ]);
    console.log(`[conformance] transformers.js config: dtype=${dtype} device=${device} (${note})`);
    const tjsAdapter = new TransformersJsAdapter({ dtype, device });
    const r = await runConformanceContract(tjsAdapter, TRANSFORMERSJS_MODEL_ID, TOKEN_BUDGET);
    run.results.push(r);
  } catch (e) {
    run.results.push({
      adapterId: "transformersjs",
      modelId: TRANSFORMERSJS_MODEL_ID,
      checks: [{ name: "config selection / contract execution", pass: false, detail: e instanceof Error ? e.message : String(e) }],
    });
  }
  render(run);

  try {
    const webllmAdapter = new WebLLMAdapter();
    const r = await runConformanceContract(webllmAdapter, WEBLLM_MODEL_ID, TOKEN_BUDGET);
    run.results.push(r);
  } catch (e) {
    run.results.push({
      adapterId: "webllm",
      modelId: WEBLLM_MODEL_ID,
      checks: [{ name: "contract execution", pass: false, detail: e instanceof Error ? e.message : String(e) }],
    });
  }

  render(run);

  try {
    const wllamaAdapter = new WllamaAdapter();
    const r = await runConformanceContract(wllamaAdapter, WLLAMA_MODEL_ID, TOKEN_BUDGET);
    run.results.push(r);
  } catch (e) {
    run.results.push({
      adapterId: "wllama",
      modelId: WLLAMA_MODEL_ID,
      checks: [{ name: "contract execution", pass: false, detail: e instanceof Error ? e.message : String(e) }],
    });
  }

  run.status = "done";
  render(run);
}

main().catch((e) => {
  const run: ConformanceRun = { status: "error", results: [], error: e instanceof Error ? e.message : String(e) };
  render(run);
});
