// Pagina harness bench GLM (dev-only). Query: ?prompt=6&ngen=64&reps=3&budget=11
const $ = (id: string) => document.getElementById(id)!;
const q = new URLSearchParams(location.search);
const cfg = {
  prompt: q.has("prompt") ? Number(q.get("prompt")) : 6,
  nGen: q.has("ngen") ? Number(q.get("ngen")) : 64,
  replicates: q.has("reps") ? Number(q.get("reps")) : 3,
  budgetGiB: q.has("budget") ? Number(q.get("budget")) : 11,
};
const worker = new Worker(new URL("./glmbench.worker.ts", import.meta.url), { type: "module" });
const log = (line: string): void => {
  const el = document.createElement("div");
  el.textContent = `${new Date().toISOString().slice(11, 19)} ${line}`;
  $("log").prepend(el);
};
worker.onmessage = (e: MessageEvent) => {
  const m = e.data as {
    type: string; msg?: string; message?: string;
    report?: { gates?: { decodePass?: boolean; prefillPass?: boolean } };
  };
  if (m.type === "progress" || m.type === "tick") {
    $("live").textContent = m.msg ?? "";
    if (m.type === "progress") log(m.msg ?? "");
  } else if (m.type === "done") {
    (window as unknown as { __report?: unknown }).__report = m.report;
    const g = m.report?.gates;
    $("status").textContent = g?.decodePass && g?.prefillPass ? "done" : "done-gate-fail";
  } else if (m.type === "error") {
    $("status").textContent = `ERROR: ${m.message?.slice(0, 400)}`;
  }
};
worker.postMessage(cfg);
log(`start prompt=${cfg.prompt} nGen=${cfg.nGen} reps=${cfg.replicates} budget=${cfg.budgetGiB}GiB`);
