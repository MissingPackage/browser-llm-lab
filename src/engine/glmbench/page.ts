// Pagina harness bench GLM (dev-only).
// Query: ?prompt=6&ngen=64&reps=3&budget=11&attrib=1
const $ = (id: string) => document.getElementById(id)!;
const q = new URLSearchParams(location.search);
const cfg = {
  prompt: q.has("prompt") ? Number(q.get("prompt")) : 6,
  nGen: q.has("ngen") ? Number(q.get("ngen")) : 64,
  replicates: q.has("reps") ? Number(q.get("reps")) : 3,
  budgetGiB: q.has("budget") ? Number(q.get("budget")) : 11,
  // repliche dedicate di attribuzione (telemetria liv.1+2): 0 le disabilita
  attribReplicates: q.has("attrib") ? Number(q.get("attrib")) : 1,
  prefillBatch: q.get("prefillbatch") === "1", // fase 5: prefillChunk M=16
  // C3b fase 4: ?select=optimistic — decode a 1 submit + repair/replay
  select: q.get("select") === "optimistic" ? ("optimistic" as const) : ("cpu" as const),
  // C3c fase 6: bench ai budget stretti sul path sync coi meccanismi C3c
  prefetch: q.get("prefetch") === "inforward" ? ("inforward" as const) : undefined,
  policy: q.get("policy") === "tier" ? ("tier" as const) : undefined,
  parkFrac: q.has("parkfrac") ? Number(q.get("parkfrac")) : undefined,
  // C3c fase 3: ?ceiling=<bytes> ⇒ budget slab CALCOLATO dalla formula
  // ctx-aware (il runner misura il tetto con nvidia-smi); ?ctxmax=<n> alloca
  // KV/partials per un contesto lungo anche col prompt corto del corpus.
  allocCeilingBytes: q.has("ceiling") ? Number(q.get("ceiling")) : undefined,
  ctxMaxOverride: q.has("ctxmax") ? Number(q.get("ctxmax")) : undefined,
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
    report?: { gates?: { decodePass?: boolean; prefillPass?: boolean; dispatchPlan?: { pass: boolean } | null } };
  };
  if (m.type === "progress" || m.type === "tick") {
    $("live").textContent = m.msg ?? "";
    if (m.type === "progress") log(m.msg ?? "");
  } else if (m.type === "done") {
    (window as unknown as { __report?: unknown }).__report = m.report;
    const g = m.report?.gates;
    // dispatchPlan null (attrib spenta) = non valutato, neutro; false = drift piano/path
    const planOk = g?.dispatchPlan == null || g.dispatchPlan.pass === true;
    $("status").textContent = g?.decodePass && g?.prefillPass && planOk ? "done" : "done-gate-fail";
  } else if (m.type === "error") {
    $("status").textContent = `ERROR: ${m.message?.slice(0, 400)}`;
  }
};
worker.postMessage(cfg);
log(`start prompt=${cfg.prompt} nGen=${cfg.nGen} reps=${cfg.replicates} budget=${cfg.budgetGiB}GiB`);
