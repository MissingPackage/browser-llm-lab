import type { WorkerToMain, MainToWorker } from "./protocol";
import type { StackId } from "./schema";
import { newRunFile, addCell, normalizeDeviceLabel, type RunFile } from "./schema";
import { renderResultsTable } from "./render";

const worker = new Worker(new URL("./bench.worker.ts", import.meta.url), { type: "module" });
const $ = (id: string) => document.getElementById(id)!;

// Persistita per-origine: durante uno sweep manuale la si scrive una volta per device,
// non a ogni ricarica (sul telefono ricaricare è la norma, e ridigitarla è proprio
// l'occasione di sbagliarla).
const DEVICE_LABEL_KEY = "blab:deviceLabel";

function readStoredLabel(): string {
  try {
    return localStorage.getItem(DEVICE_LABEL_KEY) ?? "";
  } catch {
    return ""; // storage negato (private mode, policy): la label resta per-sessione
  }
}

function storeLabel(value: string): void {
  try {
    localStorage.setItem(DEVICE_LABEL_KEY, value);
  } catch {
    /* vedi sopra: non è un errore che valga la pena mostrare */
  }
}

const deviceLabel = (): string => normalizeDeviceLabel(($("device-label") as HTMLInputElement).value);

let run: RunFile | null = null;
const send = (m: MainToWorker) => worker.postMessage(m);

function applyStackFilter(): void {
  const stack = ($("stack") as HTMLSelectElement).value;
  const modelSel = $("model") as HTMLSelectElement;
  for (const opt of Array.from(modelSel.options)) {
    const compatible = opt.dataset.stack === stack;
    opt.hidden = !compatible;
    opt.disabled = !compatible;
  }
  const selected = modelSel.selectedOptions[0];
  if (!selected || selected.hidden) {
    const firstVisible = Array.from(modelSel.options).find((o) => !o.hidden);
    if (firstVisible) modelSel.value = firstVisible.value;
  }
}

worker.onmessage = (e: MessageEvent<WorkerToMain>) => {
  const m = e.data;
  if (m.type === "probe:result") {
    run = newRunFile(deviceLabel(), m.probe, new Date().toISOString());
    $("probe-box").innerHTML = `<pre>${JSON.stringify(m.probe, null, 2)}</pre>`;
  } else if (m.type === "progress") {
    $("status").textContent = `${Math.round(m.progress * 100)}% — ${m.text}`;
  } else if (m.type === "bench:result") {
    if (run) run = addCell(run, m.cell);
    $("status").textContent = "done";
    $("results").innerHTML = run ? renderResultsTable(run) : "";
    ($("export") as HTMLButtonElement).disabled = false;
    ($("run") as HTMLButtonElement).disabled = false;
  } else if (m.type === "error") {
    $("status").textContent = `ERROR: ${m.message}`;
    ($("run") as HTMLButtonElement).disabled = false;
    ($("export") as HTMLButtonElement).disabled = run === null || run.cells.length === 0;
  } else {
    const _exhaustive: never = m;
    throw new Error(`unhandled WorkerToMain variant: ${JSON.stringify(_exhaustive)}`);
  }
};

const labelInput = $("device-label") as HTMLInputElement;
labelInput.value = readStoredLabel();
// "input", non solo "change": `change` scatta al blur, e su Android una ricarica della tab in
// background (routine) prima del blur riporterebbe in campo la label del device *precedente*.
labelInput.addEventListener("input", () => storeLabel(labelInput.value.trim()));

$("stack").addEventListener("change", applyStackFilter);
applyStackFilter();

$("run").addEventListener("click", () => {
  const sel = $("model") as HTMLSelectElement;
  const stackSel = $("stack") as HTMLSelectElement;
  const quant = sel.selectedOptions[0].dataset.quant ?? "unknown";
  ($("run") as HTMLButtonElement).disabled = true;
  ($("export") as HTMLButtonElement).disabled = true;
  send({ type: "bench", stack: stackSel.value as StackId, modelId: sel.value, quant });
});

$("export").addEventListener("click", () => {
  if (!run) return;
  // Riletta qui, non solo alla creazione del run: così una label corretta dopo il bench
  // (o digitata a run già partito) finisce comunque nel file esportato.
  const out: RunFile = { ...run, deviceLabel: deviceLabel() };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${out.deviceLabel}-${out.ts.replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

send({ type: "probe" });
