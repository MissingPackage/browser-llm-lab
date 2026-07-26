import type { RunFile } from "./schema";

const fmt = (n: number | null, digits = 1) => (n === null ? "—" : n.toFixed(digits));

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function renderResultsTable(run: RunFile): string {
  if (run.cells.length === 0) return "<p>Nessun risultato ancora.</p>";
  const rows = run.cells
    .map(
      (c) => `<tr>
        <td>${escapeHtml(c.stack)}</td><td>${escapeHtml(c.modelId)}</td><td>${escapeHtml(c.quant)}</td>
        <td>${fmt(c.load.loadMs, 0)}</td><td>${c.load.cacheState}</td>
        <td>${fmt(c.gen.ttftMs, 0)}</td><td>${fmt(c.gen.decodeToksPerSec)}</td>
        <td>${c.gen.promptTokens ?? "—"}/${c.gen.completionTokens ?? "—"}</td>
      </tr>`,
    )
    .join("");
  return `<table>
    <thead><tr><th>stack</th><th>model</th><th>quant</th><th>load ms</th><th>cache</th><th>TTFT ms</th><th>tok/s</th><th>tok in/out</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}
