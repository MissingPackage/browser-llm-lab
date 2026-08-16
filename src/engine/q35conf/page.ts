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
  // ?misstrace=1 → profilo dei miss per token (2 passate sullo stesso prompt)
  missTrace: q.get("misstrace") === "1",
  // ?optimistic=1 → gate della fetta 3c: passata sync (fredda) + passata a
  // submit unico (calda), numeri riportati separati
  optTrace: q.get("optimistic") === "1",
  // ?optcold=1 → la passata fredda la fa il path ottimistico (repair+replay)
  optCold: q.get("optcold") === "1",
  // ?kfan=1 → A/B del collasso dei k nel decode ottimistico (riga 2c)
  kfan: q.get("kfan") === "1",
  // ?splitk=1 → A/B della rotta split-K nel decode (riga 2d). Il braccio e'
  // kfan+rotta contro kfan: isola UNA leva.
  splitk: q.get("splitk") === "1",
  // ?gputime=1 → decomposizione del tempo GPU del token per categoria
  gpuTime: q.get("gputime") === "1",
  // ?nosnap=1 → misura: senza snapshot dello stato ricorrente (fase 4-ter)
  noStateSnapshot: q.get("nosnap") === "1",
  // ?prefillm=8 → gate della fase 4: prefill a chunk contro step sequenziale
  prefillM: q.has("prefillm") ? Number(q.get("prefillm")) : undefined,
  // ?confprefillm=16 → la conformance golden prefilla a chunk (it.21)
  confPrefillM: q.has("confprefillm") ? Number(q.get("confprefillm")) : undefined,
  // ?vram=10.5 → tetto VRAM: il budget expert si deriva (docket item 11)
  vramGiB: q.has("vram") ? Number(q.get("vram")) : undefined,
  // ?coldboth=1 → i due path misurati entrambi a freddo, stesso processo
  coldBoth: q.get("coldboth") === "1",
  // ?policy=tier → LRU + AUTOPIN sulla cache expert (fase 5)
  expertPolicy: (q.get("policy") === "tier" ? "tier" : undefined) as "tier" | undefined,
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
