// Pagina harness conformance q35 4B (dev-only). Query: ?prompts=0,1&maxgen=32
const $ = (id: string) => document.getElementById(id)!;
const q = new URLSearchParams(location.search);
const cfg = {
  prompts: q.has("prompts") ? q.get("prompts")!.split(",").map(Number) : undefined,
  maxGen: q.has("maxgen") ? Number(q.get("maxgen")) : undefined,
  // ?bench=4,64 → riferimenti full-resident: prompt idx 4, 64 decode greedy
  model: (["9b", "35b"].includes(q.get("model") ?? "") ? q.get("model") : undefined) as "9b" | "35b" | undefined,
  debugTap: q.has("tap") ? Number(q.get("tap")) : undefined,
  arenaGiB: q.has("arena") ? Number(q.get("arena")) : undefined,
  // ?shadow=1 → router+resolve su GPU in OMBRA accanto alla selezione CPU
  routerShadow: q.get("shadow") === "1",
  bench: q.has("bench")
    ? { promptIdx: Number(q.get("bench")!.split(",")[0]), nDecode: Number(q.get("bench")!.split(",")[1] ?? 64) }
    : undefined,
};
const worker = new Worker(new URL("./q35conf.worker.ts", import.meta.url), { type: "module" });
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
    $("status").textContent = "done";
  } else if (m.type === "error") {
    $("status").textContent = `ERROR: ${m.message?.slice(0, 400)}`;
  }
};
worker.postMessage(cfg);
log(`start prompts=${cfg.prompts?.join(",") ?? "tutti"} maxGen=${cfg.maxGen ?? "tutti"}`);
