import type { WorkerToMain, MainToWorker } from "./protocol";
import { newRunFile, addCell, type RunFile } from "./schema";
import { renderResultsTable } from "./render";

const worker = new Worker(new URL("./bench.worker.ts", import.meta.url), { type: "module" });
const $ = (id: string) => document.getElementById(id)!;

let run: RunFile | null = null;
const send = (m: MainToWorker) => worker.postMessage(m);

worker.onmessage = (e: MessageEvent<WorkerToMain>) => {
  const m = e.data;
  if (m.type === "probe:result") {
    run = newRunFile("4090-linux", m.probe, new Date().toISOString());
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
    ($("export") as HTMLButtonElement).disabled = run === null;
  } else {
    const _exhaustive: never = m;
    throw new Error(`unhandled WorkerToMain variant: ${JSON.stringify(_exhaustive)}`);
  }
};

$("run").addEventListener("click", () => {
  const sel = $("model") as HTMLSelectElement;
  const quant = sel.selectedOptions[0].dataset.quant ?? "unknown";
  ($("run") as HTMLButtonElement).disabled = true;
  ($("export") as HTMLButtonElement).disabled = true;
  send({ type: "bench", modelId: sel.value, quant });
});

$("export").addEventListener("click", () => {
  if (!run) return;
  const blob = new Blob([JSON.stringify(run, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${run.deviceLabel}-${run.ts.replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

send({ type: "probe" });
