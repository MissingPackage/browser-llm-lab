// Pagina harness conformance logits GLM (dev-only). Query: ?prompts=7&maxgen=32&budget=12
const $ = (id: string) => document.getElementById(id)!;
const q = new URLSearchParams(location.search);
const cfg = {
  prompts: q.has("prompts") ? q.get("prompts")!.split(",").map(Number) : undefined,
  maxGen: q.has("maxgen") ? Number(q.get("maxgen")) : undefined,
  budgetGiB: q.has("budget") ? Number(q.get("budget")) : 12,
};
const worker = new Worker(new URL("./glmconf.worker.ts", import.meta.url), { type: "module" });
const log = (line: string): void => {
  const el = document.createElement("div");
  el.textContent = `${new Date().toISOString().slice(11, 19)} ${line}`;
  $("log").prepend(el);
};
worker.onmessage = (e: MessageEvent) => {
  const m = e.data as {
    type: string; msg?: string; message?: string;
    report?: { gateGolden?: { pass?: boolean }; gateCpuref?: { pass?: boolean | null } | null };
  };
  if (m.type === "progress" || m.type === "tick") {
    $("live").textContent = m.msg ?? "";
    if (m.type === "progress") log(m.msg ?? "");
  } else if (m.type === "done") {
    (window as unknown as { __report?: unknown }).__report = m.report;
    // gateCpuref: null/non valutato = neutro; false = divergenza dal cpuref-f64 (4d)
    const cpurefOk = m.report?.gateCpuref?.pass !== false;
    $("status").textContent = m.report?.gateGolden?.pass && cpurefOk ? "done" : "done-gate-fail";
  } else if (m.type === "error") {
    $("status").textContent = `ERROR: ${m.message?.slice(0, 400)}`;
  }
};
worker.postMessage(cfg);
log(`start prompts=${cfg.prompts ?? "tutti"} maxGen=${cfg.maxGen ?? "128"} budget=${cfg.budgetGiB}GiB`);
