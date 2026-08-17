// Il TRASCRITTO di una sessione di chat in testo piatto, con le marche del
// template. Serve al corpus di valutazione dei quant: le altre sezioni sono
// prosa e codice, e nessuna misura il regime in cui il modello vive davvero —
// turni, contesto che cresce, token di fine turno.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1] : d; };
const IN = arg("in", null), OUT = arg("out", null);
if (!IN || !OUT) { console.error("uso: node scripts/chat-transcript.mjs --in artefatto.json --out testo.txt"); process.exit(2); }
const d = JSON.parse(readFileSync(IN, "utf8"));
const a = d.turns.filter((t) => t.role === "assistant");
const dom = d.kind === "llamacpp-chat-session"
  ? a.map((t) => t.question)
  : d.turns.filter((t) => t.role === "user").map((t) => t.content);
let s = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
for (const [i, t] of a.entries()) {
  s += `<|im_start|>user\n${dom[i]}<|im_end|>\n<|im_start|>assistant\n${t.content}<|im_end|>\n`;
}
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, s);
console.log(`[trascritto] ${a.length} turni · ${s.length} caratteri → ${OUT}`);
