// Pagina del motore (fase A: solo modalità conformance, auto-run con ?conformance=1).
const $ = (id: string) => document.getElementById(id)!;

const worker = new Worker(new URL("./engine.worker.ts", import.meta.url), { type: "module" });

declare global {
  interface Window { __report?: unknown }
}

worker.onmessage = (e: MessageEvent) => {
  const m = e.data as {
    type: string; text?: string; frac?: number; message?: string;
    dispatchesPerToken?: number;
    report?: {
      top1Pct: number; agree: number; total: number; maxDlogitSampled: number;
      dispatchesPerToken: number; wallMs: number;
      perPrompt: { id: string; agree: number; total: number; mismatches: { pos: number; got: number; gold: number }[] }[];
    };
  };
  if (m.type === "progress") {
    $("status").textContent = `${Math.round((m.frac ?? 0) * 100)}% — ${m.text}`;
  } else if (m.type === "meta") {
    $("probe-box").textContent = JSON.stringify({ webgpu: true, dispatchesPerToken: m.dispatchesPerToken });
  } else if (m.type === "done" && m.report) {
    window.__report = m.report;
    const r = m.report;
    const rows = r.perPrompt
      .map((p) =>
        `<tr><td>${p.id}</td><td>${p.agree}/${p.total}</td>` +
        `<td>${p.mismatches.map((x) => `pos${x.pos}:${x.got}≠${x.gold}`).join(" ") || "—"}</td></tr>`)
      .join("");
    $("results").innerHTML =
      `<p>top-1: <b>${r.top1Pct.toFixed(2)}%</b> (${r.agree}/${r.total}) · ` +
      `max|Δlogit| campionato: ${r.maxDlogitSampled.toFixed(4)} · ` +
      `${r.dispatchesPerToken} dispatch/token · ${(r.wallMs / 1000).toFixed(1)}s</p>` +
      `<table border="1" cellpadding="4"><tr><th>prompt</th><th>agree</th><th>mismatch</th></tr>${rows}</table>`;
    $("status").textContent = "done";
  } else if (m.type === "error") {
    $("status").textContent = `ERROR: ${m.message}`;
  }
};

const params = new URLSearchParams(location.search);
if (params.get("conformance") === "1") {
  worker.postMessage({
    type: "conformance",
    modelUrl: "/models/qwen2.5-0.5b-instruct-q4_0.gguf",
    goldenUrl: "/results/engine/golden/golden-qwen25-05b-q4_0.json",
  });
} else {
  $("status").textContent = "aggiungi ?conformance=1 per il run";
}
