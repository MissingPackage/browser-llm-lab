// Pagina della chat q35 (banco di prova). Tre cose e basta: scegliere il
// modello e i parametri, parlarci, esportare CONVERSAZIONE e PARAMETRI.
//
// L'export è il motivo per cui la pagina esiste in un repo di misura: il JSON
// porta i parametri di carico (tetto VRAM, ctx, path di select, policy), quelli
// di campionamento (temperatura/top-k/top-p/SEED), il prompt RESO in ChatML con
// gli id, e le metriche per turno prese dai contatori del motore. Un tok/s
// senza quel contorno non è confrontabile con niente.
import type { LoadCfg, ModelKey, SamplingCfg } from "./chat.worker";

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const worker = new Worker(new URL("./chat.worker.ts", import.meta.url), { type: "module" });

interface TurnStats { [k: string]: unknown }
interface Turn { role: "user" | "assistant"; content: string; stats?: TurnStats }

const session = {
  startedAt: new Date().toISOString(),
  loaded: null as Record<string, unknown> | null,
  loadCfg: null as LoadCfg | null,
  turns: [] as Turn[],
};

let turnCounter = 0;
let busy = false;
let liveEl: HTMLElement | null = null;
let liveText = "";

const log = (line: string): void => {
  const el = document.createElement("div");
  el.textContent = `${new Date().toISOString().slice(11, 19)} ${line}`;
  $("log").prepend(el);
};

const setBusy = (b: boolean): void => {
  busy = b;
  ($("send") as HTMLButtonElement).disabled = b || !session.loaded;
  ($("load") as HTMLButtonElement).disabled = b;
  ($("stop") as HTMLButtonElement).disabled = !b;
};

function bubble(role: "user" | "assistant", text: string): HTMLElement {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  wrap.append(who, body);
  $("transcript").append(wrap);
  wrap.scrollIntoView({ block: "end" });
  return body;
}

const readLoadCfg = (): LoadCfg => ({
  model: ($("model") as HTMLSelectElement).value as ModelKey,
  vramGiB: ($("vram") as HTMLInputElement).value.trim() === "" ? null : Number(($("vram") as HTMLInputElement).value),
  ctxMax: Number(($("ctx") as HTMLInputElement).value),
  select: ($("select") as HTMLSelectElement).value as "cpu" | "optimistic",
  expertPolicy: ($("policy") as HTMLSelectElement).value as "lru" | "tier",
});

const readSampling = (): SamplingCfg => ({
  temperature: Number(($("temp") as HTMLInputElement).value),
  topK: Number(($("topk") as HTMLInputElement).value),
  topP: Number(($("topp") as HTMLInputElement).value),
  seed: Number(($("seed") as HTMLInputElement).value),
  maxNew: Number(($("maxnew") as HTMLInputElement).value),
});

worker.onmessage = (e: MessageEvent) => {
  const m = e.data as { type: string; [k: string]: unknown };
  if (m.type === "progress") {
    $("status").textContent = m.msg as string;
    log(m.msg as string);
  } else if (m.type === "loaded") {
    session.loaded = m.info as Record<string, unknown>;
    session.loadCfg = readLoadCfg();
    const i = session.loaded;
    $("status").textContent = `pronto — ${i.file as string} in ${((i.loadMs as number) / 1000).toFixed(1)} s`;
    const vp = i.vramPlan as { expertBudgetBytes: number; allocatedBytes: number } | null;
    log(`caricato: ${i.arch as string}, ${i.nLayer as number} layer, vocab ${i.vocab as number}, ` +
      `dispatch/token ${(i.dispatchBreakdown as { total: number }).total}` +
      (vp ? ` — allocati ${(vp.allocatedBytes / 2 ** 30).toFixed(2)} GiB, budget expert ${(vp.expertBudgetBytes / 2 ** 30).toFixed(2)} GiB` : ""));
    setBusy(false);
  } else if (m.type === "token") {
    liveText += m.text as string;
    if (liveEl) { liveEl.textContent = liveText; liveEl.scrollIntoView({ block: "end" }); }
  } else if (m.type === "turnDone") {
    const st = m.stats as Record<string, unknown>;
    const text = m.text as string;
    if (liveEl) liveEl.textContent = text;
    session.turns.push({ role: "assistant", content: text, stats: st });
    const tps = st.decodeTokS as number | null;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent =
      `${st.genTokens as number} token · ${tps !== null ? tps.toFixed(2) : "n/d"} tok/s · ` +
      `TTFT ${((st.ttftMs as number) / 1000).toFixed(2)} s · prompt ${st.promptTokens as number} tok · stop: ${st.stopReason as string}` +
      (st.moe ? ` · miss ${(st.moe as { misses: number }).misses}` : "");
    liveEl?.parentElement?.append(meta);
    liveEl = null;
    liveText = "";
    $("status").textContent = "pronto";
    setBusy(false);
  } else if (m.type === "resetDone") {
    $("transcript").replaceChildren();
    session.turns = [];
    turnCounter = 0;
    log("contesto azzerato (KV e stato ricorrente)");
    setBusy(false);
  } else if (m.type === "snapshot") {
    pendingSnapshot?.(m.data as Record<string, unknown> | null);
    pendingSnapshot = null;
  } else if (m.type === "error") {
    $("status").textContent = "ERRORE";
    log(`ERRORE: ${(m.message as string).slice(0, 600)}`);
    setBusy(false);
  }
};

let pendingSnapshot: ((d: Record<string, unknown> | null) => void) | null = null;
const snapshot = (): Promise<Record<string, unknown> | null> =>
  new Promise((res) => { pendingSnapshot = res; worker.postMessage({ type: "snapshot" }); });

$("load").addEventListener("click", () => {
  setBusy(true);
  session.loaded = null;
  $("status").textContent = "carico…";
  worker.postMessage({ type: "load", cfg: readLoadCfg() });
});

$("send").addEventListener("click", () => {
  const input = $("input") as HTMLTextAreaElement;
  const user = input.value.trim();
  if (user === "" || busy) return;
  input.value = "";
  bubble("user", user);
  session.turns.push({ role: "user", content: user });
  liveEl = bubble("assistant", "");
  liveText = "";
  setBusy(true);
  $("status").textContent = "genero…";
  worker.postMessage({
    type: "chat",
    turn: turnCounter++,
    user,
    system: ($("system") as HTMLTextAreaElement).value,
    sampling: readSampling(),
  });
});

$("input").addEventListener("keydown", (e) => {
  const ev = e as KeyboardEvent;
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); $("send").click(); }
});

$("stop").addEventListener("click", () => worker.postMessage({ type: "stop" }));
$("reset").addEventListener("click", () => { setBusy(true); worker.postMessage({ type: "reset" }); });

function download(name: string, text: string, mime: string): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

const stamp = (): string => new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);

$("export-json").addEventListener("click", () => {
  void snapshot().then((snap) => {
    const report = {
      schemaVersion: 1,
      kind: "q35-chat-session",
      date: new Date().toISOString(),
      startedAt: session.startedAt,
      // DICHIARATO: `hostState` lo scrivono i runner node con nvidia-smi
      // (temperatura, clock, VRAM usata prima e dopo). Dal browser non è
      // leggibile, e un campo inventato sarebbe peggio di un campo assente:
      // questo JSON non è confrontabile con i riferimenti host-gated.
      hostState: null,
      declared: "chat interattiva (banco di prova). NON è un riferimento: nessun warm-up scartato, nessuna replica, prompt scelti a mano, host non dichiarato. I riferimenti stanno in results/engine e li producono i runner node.",
      model: session.loaded,
      params: {
        load: session.loadCfg,
        sampling: readSampling(),
        systemPrompt: ($("system") as HTMLTextAreaElement).value,
        chatTemplate: "ChatML canonico reso in-page (<|im_start|>ruolo\\n…<|im_end|>\\n); il template Jinja del GGUF NON è eseguito",
        chatTemplateRaw: snap?.chatTemplateRaw ?? null,
      },
      runtime: { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency },
      // SOMMARIO: le mediane sui turni CALDI, separate dal primo turno a
      // freddo. Senza questa separazione un JSON di chat si legge come una
      // misura sola, e sul 35B il primo turno paga la lettura degli expert da
      // disco (visto il 2026-08-13: 9,58 tok/s a freddo contro 22,6 di
      // riferimento a caldo). Resta un banco di prova, non un riferimento:
      // `hostState` e' null per costruzione e `declared` lo dice.
      summary: (() => {
        const st = session.turns.flatMap((t) => (t.stats ? [t.stats as Record<string, unknown>] : []));
        const warm = st.filter((x) => x.cacheState === "warm");
        const med = (xs: number[]): number | null =>
          xs.length > 0 ? xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null;
        const num = (xs: Record<string, unknown>[], k: string): number[] =>
          xs.map((x) => x[k]).filter((v): v is number => typeof v === "number");
        return {
          assistantTurns: st.length,
          coldTurn: st.length > 0 ? { decodeTokS: st[0].decodeTokS ?? null, ttftMs: st[0].ttftMs ?? null } : null,
          warmMedian: {
            turns: warm.length,
            decodeTokS: med(num(warm, "decodeTokS")),
            decodeMsPerTokenP50: med(num(warm, "decodeMsPerTokenP50")),
            ttftMs: med(num(warm, "ttftMs")),
          },
        };
      })(),
      turns: session.turns,
      final: snap,
    };
    download(`chat-${session.loadCfg?.model ?? "q35"}-${stamp()}.json`, JSON.stringify(report, null, 2), "application/json");
    log("export JSON scritto");
  });
});

$("export-md").addEventListener("click", () => {
  const i = session.loaded;
  const lines = [
    `# Conversazione — ${i ? (i.file as string) : "nessun modello"}`,
    "",
    `- data: ${new Date().toISOString()}`,
    ...(session.loadCfg ? [
      `- tetto VRAM: ${session.loadCfg.vramGiB ?? "n/d"} GiB · ctxMax ${session.loadCfg.ctxMax}`,
      `- select: ${session.loadCfg.select} · policy expert: ${session.loadCfg.expertPolicy}`,
    ] : []),
    `- campionamento: ${JSON.stringify(readSampling())}`,
    "",
  ];
  for (const t of session.turns) {
    lines.push(`## ${t.role}`, "", t.content, "");
    if (t.stats) {
      const s = t.stats as Record<string, number | null>;
      lines.push(`> ${s.genTokens} token · ${s.decodeTokS !== null && s.decodeTokS !== undefined ? (s.decodeTokS as number).toFixed(2) : "n/d"} tok/s · TTFT ${(((s.ttftMs as number) ?? 0) / 1000).toFixed(2)} s`, "");
    }
  }
  download(`chat-${session.loadCfg?.model ?? "q35"}-${stamp()}.md`, lines.join("\n"), "text/markdown");
  log("export Markdown scritto");
});

setBusy(false);
($("send") as HTMLButtonElement).disabled = true;
log("pronta — scegli modello e parametri, poi «Carica»");
