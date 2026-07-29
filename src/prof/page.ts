import { normalizeDeviceLabel } from "../schema";
import type { DeviceProbe, BenchCell } from "../schema";
import type { MainToWorker } from "../protocol";
import type { ProfWorkerOut } from "./prof.worker";
import type { ProfRunFile } from "./profSchema";

// Stessa chiave delle altre pagine: la label del device è una sola per origine.
const DEVICE_LABEL_KEY = "blab:deviceLabel";

const $ = (id: string) => document.getElementById(id)!;
const statusEl = $("status");
const labelInput = $("device-label") as HTMLInputElement;
const runBtn = $("run") as HTMLButtonElement;
const exportBtn = $("export") as HTMLButtonElement;

const deviceLabel = (): string => normalizeDeviceLabel(labelInput.value);

try {
  labelInput.value = localStorage.getItem(DEVICE_LABEL_KEY) ?? "";
} catch { /* storage negato: si parte vuoti */ }
labelInput.addEventListener("input", () => {
  try { localStorage.setItem(DEVICE_LABEL_KEY, labelInput.value.trim()); } catch { /* idem */ }
});

const worker = new Worker(new URL("./prof.worker.ts", import.meta.url), { type: "module" });
const send = (m: MainToWorker | { type: "prof:dump" }) => worker.postMessage(m);

let probe: DeviceProbe | null = null;
let cell: BenchCell | null = null;
let ts = "";
let lastFile: ProfRunFile | null = null;

worker.onmessage = (e: MessageEvent<ProfWorkerOut>) => {
  const m = e.data;
  if (m.type === "probe:result") {
    probe = m.probe;
    $("probe-box").innerHTML = `<pre>${JSON.stringify(m.probe, null, 2)}</pre>`;
  } else if (m.type === "progress") {
    statusEl.textContent = `${Math.round(m.progress * 100)}% — ${m.text}`;
  } else if (m.type === "bench:result") {
    cell = m.cell;
    statusEl.textContent = "raccolgo i contatori…";
    send({ type: "prof:dump" });
  } else if (m.type === "prof:data") {
    lastFile = {
      schemaVersion: 1,
      kind: "dispatch-profile",
      source: "prof-page",
      deviceLabel: deviceLabel(),
      ts,
      probe: probe!,
      status: "done",
      sampleMs: m.sampleMs,
      clockMinDeltaMs: m.clockMinDeltaMs,
      missingApis: m.missingApis,
      totals: m.totals,
      samples: m.samples,
      cell,
    };
    statusEl.textContent = "done";
    renderSummary(lastFile);
    runBtn.disabled = false;
    exportBtn.disabled = false;
  } else if (m.type === "error") {
    statusEl.textContent = `ERROR: ${m.message}`;
    runBtn.disabled = false;
  } else {
    const _exhaustive: never = m;
    throw new Error(`unhandled ProfWorkerOut variant: ${JSON.stringify(_exhaustive)}`);
  }
};

runBtn.addEventListener("click", () => {
  const sel = $("model") as HTMLSelectElement;
  runBtn.disabled = true;
  exportBtn.disabled = true;
  cell = null;
  ts = new Date().toISOString();
  statusEl.textContent = "starting…";
  send({
    type: "bench",
    stack: "webllm",
    modelId: sel.value,
    quant: sel.selectedOptions[0].dataset.quant ?? "unknown",
  });
});

exportBtn.addEventListener("click", () => {
  if (!lastFile) return;
  // Riletta all'export, come nella pagina bench: una label corretta a run già
  // partito finisce comunque nel file.
  const out: ProfRunFile = { ...lastFile, deviceLabel: deviceLabel() };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `dispatch-profile-${out.deviceLabel}-${out.ts.replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

// Sintesi a schermo: serve al run manuale come sanity check immediato (sul 4090 il
// riferimento è ~270 dispatch/token con 7 submit/token → ~38 dispatch/submit).
// L'analisi vera resta offline sui campioni esportati.
function renderSummary(f: ProfRunFile): void {
  const t = f.totals;
  const gen = f.samples.filter((s) => s.phase.startsWith("generating"));
  let decodeRow = "<tr><td>dispatch/submit (finestra decode)</td><td>n/d — meno di 2 campioni</td></tr>";
  if (gen.length >= 2) {
    const dDispatch = gen[gen.length - 1].dispatch - gen[0].dispatch;
    const dSubmit = gen[gen.length - 1].submit - gen[0].submit;
    const ratio = dSubmit > 0 ? (dDispatch / dSubmit).toFixed(1) : "n/d";
    decodeRow = `<tr><td>dispatch/submit (finestra decode, stima grezza)</td><td>${ratio}</td></tr>`;
  }
  const tEncode = t.tDispatch + t.tBindGroup + t.tComputePass + t.tWriteBuffer +
    t.tSubmit + t.tCommandEncoder + t.tPassEnd;
  const usPerDispatch = t.dispatch > 0 ? ((tEncode / t.dispatch) * 1000).toFixed(1) : "n/d";
  $("results").innerHTML =
    `<table border="1" cellpadding="4">` +
    `<tr><th colspan="2">totali (load+warmup+bench)</th></tr>` +
    `<tr><td>dispatch</td><td>${t.dispatch}</td></tr>` +
    `<tr><td>submit</td><td>${t.submit}</td></tr>` +
    `<tr><td>createBindGroup</td><td>${t.bindGroup}</td></tr>` +
    `<tr><td>writeBuffer</td><td>${t.writeBuffer}</td></tr>` +
    decodeRow +
    `<tr><td>encode CPU µs/dispatch (aggregato)</td><td>${usPerDispatch}</td></tr>` +
    `<tr><td>campioni</td><td>${f.samples.length} @ ${f.sampleMs} ms</td></tr>` +
    `<tr><td>quanto clock (ms)</td><td>${f.clockMinDeltaMs ?? "n/d"}</td></tr>` +
    (f.missingApis.length ? `<tr><td>API mancanti</td><td>${f.missingApis.join(", ")}</td></tr>` : "") +
    `</table>`;
}

send({ type: "probe" });
