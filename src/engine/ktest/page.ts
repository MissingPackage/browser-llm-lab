// Pagina kernel-test del motore (dev-only, come conformance.html).
const $ = (id: string) => document.getElementById(id)!;

interface KResult { kernel: string; pass: boolean; maxAbs: number; maxRel: number; note?: string }

const worker = new Worker(new URL("./ktest.worker.ts", import.meta.url), { type: "module" });

function render(results: KResult[]): void {
  const rows = results
    .map((r) =>
      `<tr><td>${r.kernel}</td><td>${r.pass ? "PASS" : "FAIL"}</td>` +
      `<td>${r.maxAbs.toExponential(2)}</td><td>${r.maxRel.toExponential(2)}</td>` +
      `<td>${r.note ?? ""}</td></tr>`)
    .join("");
  $("results").innerHTML =
    `<table border="1" cellpadding="4"><tr><th>kernel</th><th>esito</th>` +
    `<th>max|Δ|</th><th>maxRel</th><th>note</th></tr>${rows}</table>`;
}

worker.onmessage = (e: MessageEvent) => {
  const m = e.data as { type: string; desc?: string; message?: string; results?: KResult[] };
  if (m.type === "adapter") {
    $("probe-box").textContent = JSON.stringify({ webgpu: true, adapter: m.desc });
  } else if (m.type === "done") {
    render(m.results ?? []);
    const allPass = (m.results ?? []).every((r) => r.pass);
    $("status").textContent = allPass ? "done" : "ERROR: kernel FAIL";
  } else if (m.type === "error") {
    if (m.results) render(m.results);
    $("status").textContent = `ERROR: ${m.message}`;
  }
};
