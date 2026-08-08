// Pagina harness conformance routing GLM (dev-only). Config via query string:
// ?cap=200&prompts=0,1&budget=12 — il runner scripts/glm-route-run.mjs la
// legge via #status/__report come le altre pagine harness.
const $ = (id: string) => document.getElementById(id)!;
const q = new URLSearchParams(location.search);
const cfg = {
  cap: q.has("cap") ? Number(q.get("cap")) : undefined,
  prompts: q.has("prompts") ? q.get("prompts")!.split(",").map(Number) : undefined,
  budgetGiB: q.has("budget") ? Number(q.get("budget")) : 12,
  // C3c fase 4: ?prefetch=inforward — tap + prefetch nel path sync, recall
  // in-engine nel report. La run di FIRMA (14b) resta quella senza flag.
  prefetch: q.get("prefetch") === "inforward" ? ("inforward" as const) : undefined,
};

const worker = new Worker(new URL("./glmroute.worker.ts", import.meta.url), { type: "module" });
const log = (line: string): void => {
  const el = document.createElement("div");
  el.textContent = `${new Date().toISOString().slice(11, 19)} ${line}`;
  $("log").prepend(el);
};

worker.onmessage = (e: MessageEvent) => {
  const m = e.data as { type: string; msg?: string; message?: string; report?: unknown };
  if (m.type === "progress" || m.type === "tick") {
    $("live").textContent = m.msg ?? "";
    if (m.type === "progress") log(m.msg ?? "");
  } else if (m.type === "done") {
    (window as unknown as { __report?: unknown }).__report = m.report;
    // DUE gate: quello storico verso l'oracolo (setMatch decode) e quello dello
    // slice B (accordo del router GPU in ombra). Lo status ne riflette la
    // congiunzione — un run che passa l'uno e non l'altro non e' un run buono.
    const r = m.report as { gate?: { pass?: boolean }; gpuRouterAgreement?: { gate?: { pass?: boolean }; pct?: number } };
    const ok = r.gate?.pass === true && r.gpuRouterAgreement?.gate?.pass === true;
    $("status").textContent = ok ? "done" : "done-gate-fail";
    log(`done — oracolo ${r.gate?.pass} / router GPU ${r.gpuRouterAgreement?.gate?.pass} (${r.gpuRouterAgreement?.pct?.toFixed(4) ?? "—"}%)`);
  } else if (m.type === "error") {
    $("status").textContent = `ERROR: ${m.message?.slice(0, 400)}`;
  }
};
worker.postMessage(cfg);
log(`start cap=${cfg.cap ?? "∞"} prompts=${cfg.prompts ?? "tutti"} budget=${cfg.budgetGiB}GiB`);
