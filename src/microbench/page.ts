import { normalizeDeviceLabel } from "../schema";
import type { MicrobenchWorkerOut } from "./microbench.worker";
import type { MicrobenchRunFile } from "./mbSchema";

// Stessa chiave della pagina bench: la label del device è una sola per origine.
const DEVICE_LABEL_KEY = "blab:deviceLabel";

const $ = (id: string) => document.getElementById(id)!;
const statusEl = $("status");
const labelInput = $("device-label") as HTMLInputElement;
const runBtn = $("run") as HTMLButtonElement;
const exportBtn = $("export") as HTMLButtonElement;

let lastRun: MicrobenchRunFile | null = null;

try {
  labelInput.value = localStorage.getItem(DEVICE_LABEL_KEY) ?? "";
} catch { /* storage negato: si parte vuoti */ }
labelInput.addEventListener("input", () => {
  try { localStorage.setItem(DEVICE_LABEL_KEY, labelInput.value); } catch { /* idem */ }
});

const worker = new Worker(new URL("./microbench.worker.ts", import.meta.url), { type: "module" });

worker.onmessage = (ev: MessageEvent<MicrobenchWorkerOut>) => {
  const m = ev.data;
  if (m.type === "progress") statusEl.textContent = m.message;
  if (m.type === "error") {
    statusEl.textContent = `ERROR: ${m.message}`;
    runBtn.disabled = false;
  }
  if (m.type === "done") {
    lastRun = m.runFile;
    statusEl.textContent = "done";
    renderTable(m.runFile);
    runBtn.disabled = false;
    exportBtn.disabled = false;
  }
};

runBtn.addEventListener("click", () => {
  runBtn.disabled = true;
  exportBtn.disabled = true;
  statusEl.textContent = "starting…";
  worker.postMessage({ type: "run", deviceLabel: normalizeDeviceLabel(labelInput.value) });
});

exportBtn.addEventListener("click", () => {
  if (!lastRun) return;
  const blob = new Blob([JSON.stringify(lastRun, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `microbench-${lastRun.deviceLabel}-${lastRun.ts.replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

function renderTable(run: MicrobenchRunFile): void {
  const rows = run.cells
    .map(
      (c) =>
        `<tr><td>${c.kernel}</td><td>${c.rowsN}×${c.colsK}</td><td>${c.timingSource}</td>` +
        `<td>${(c.gpuMs ?? c.cpuMs).mean.toFixed(3)}</td><td>${(c.gpuMs ?? c.cpuMs).stdev.toFixed(3)}</td>` +
        `<td>${c.effectiveGBps.toFixed(1)}</td></tr>`,
    )
    .join("");
  $("results").innerHTML =
    `<table border="1" cellpadding="4"><tr><th>kernel</th><th>N×K</th><th>timing</th>` +
    `<th>ms (mean)</th><th>stdev</th><th>GB/s eff.</th></tr>${rows}</table>`;
}
