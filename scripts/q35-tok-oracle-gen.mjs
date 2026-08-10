// Genera il riferimento oracolo del tokenizer q35 (fase 2, spec §5 punto 1).
// Per ogni file del corpus (01-08 GLM + 09-12 edge) esegue llama-tokenize
// b10333 con protocollo FISSO v2: --ids --no-bos --no-parse-special
// --no-escape. Il --no-escape è OBBLIGATORIO: senza, llama-tokenize processa
// gli escape (\\ → \, \n → newline) e il riferimento non è più il testo raw
// che il motore tokenizza (scoperto in it.3 su corpus 11, "\\n letterale").
// Cross-check: gli id devono essere IDENTICI fra 4B (denso) e 35B (MoE) —
// vocab di famiglia; divergenza = errore secco, il fixture non si scrive.
// Output: tests/fixtures/q35-tok-oracle.json
import { execFileSync } from "node:child_process";
import { globSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";

const BIN = join(homedir(), ".cache/llamacpp-vulkan/llama-b10333/llama-tokenize");
const M4 = join(homedir(), ".cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf");
const M35 = join(homedir(), ".cache/blab-models/q35/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf");
const PROTOCOL = ["--ids", "--no-bos", "--no-parse-special", "--no-escape"];

function tokenize(model, file) {
  const out = execFileSync(BIN, ["-m", model, "-f", file, ...PROTOCOL], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    maxBuffer: 1 << 24,
  });
  // la riga utile è la lista Python-style [1, 2, ...] — l'ultima non vuota che inizia con [
  const line = out.split("\n").filter((l) => l.trim().startsWith("[")).pop();
  if (!line) throw new Error(`nessuna lista id nell'output per ${file} (${model})`);
  return JSON.parse(line);
}

const corpus = [
  ...globSync("tools/oracle-moe/corpus/*.txt"),
  ...globSync("tools/oracle-moe/corpus-tok/*.txt"),
].sort();
if (corpus.length < 12) throw new Error(`corpus incompleto: ${corpus.length} file`);

const files = {};
let tot = 0;
for (const f of corpus) {
  const ids4 = tokenize(M4, f);
  const ids35 = tokenize(M35, f);
  if (JSON.stringify(ids4) !== JSON.stringify(ids35))
    throw new Error(`DIVERGENZA vocab 4B vs 35B su ${f}: il fixture non si scrive`);
  if (ids4.length === 0) throw new Error(`0 token per ${f}`);
  files[basename(f)] = ids4;
  tot += ids4.length;
  console.log(`${basename(f)}: ${ids4.length} token (bytes ${readFileSync(f).length})`);
}

writeFileSync(
  "tests/fixtures/q35-tok-oracle.json",
  JSON.stringify(
    {
      note: "Riferimento oracolo tokenizer q1 fase 2. llama-tokenize b10333 (build 08659901c). Cross-check 4B==35B eseguito alla generazione.",
      protocol: PROTOCOL.join(" "),
      files,
    },
    null,
    1,
  ),
);
console.log(`OK: ${corpus.length} file, ${tot} token totali, cross-check 4B==35B PASS`);
