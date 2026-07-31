# Genera i file corpus/*.txt (spec engine-fase-c1 §Corpus): 8 prompt eterogenei.
# Gli estratti dai file del repo vengono CONGELATI nei .txt committati — questo
# script è provenienza, non dipendenza: si esegue una volta e si committa l'output.
# Uso: python3 tools/oracle-moe/gen-corpus.py   (dalla root del repo)
from pathlib import Path

OUT = Path("tools/oracle-moe/corpus")
OUT.mkdir(exist_ok=True)


def excerpt(path, a, b):
    lines = Path(path).read_text().splitlines()
    return "\n".join(lines[a - 1 : b])


P = {}

P["01-code-ts-review.txt"] = (
    "Review the following TypeScript module from a WebGPU LLM inference engine. "
    "Identify potential bugs, questionable design choices and performance risks, "
    "and propose concrete improvements. Be thorough and specific.\n\n```ts\n"
    + excerpt("src/engine/gpuforward.ts", 1, 330) + "\n```\n"
)

P["02-code-wgsl-review.txt"] = (
    "The following WGSL kernels implement quantized matmul and attention for an "
    "LLM inference engine. Explain what each kernel does, then review them for "
    "correctness hazards (race conditions, uninitialized accumulators, workgroup "
    "sizing issues) and suggest optimizations.\n\n```wgsl\n"
    + excerpt("src/engine/kernels/wgsl.ts", 1, 300) + "\n```\n"
)

P["03-prosa-it-riassunto.txt"] = (
    "Il documento che segue analizza un motore di inferenza C per modelli MoE. "
    "Riassumilo per un collega ingegnere che non l'ha letto: le 5 idee chiave, i "
    "numeri misurati più importanti, e cosa sia trasferibile a un motore browser. "
    "Scrivi in italiano, in prosa.\n\n---\n"
    + excerpt("docs/engine/study/colibri.md", 1, 210) + "\n"
)

P["04-prosa-it-analisi.txt"] = (
    "Leggi questo documento di direzione tecnica e scrivi un'analisi critica: "
    "quali assunzioni sono più fragili, quali rischi sono sottovalutati, e che "
    "esperimenti proporresti per ridurli. Rispondi in italiano.\n\n---\n"
    + excerpt("docs/engine/direction.md", 1, 170) + "\n"
)

P["05-math-en.txt"] = """Solve each problem step by step, showing your reasoning in full before giving
the final answer.

1. A GPU kernel processes tokens in batches of K. Per batch it pays a fixed cost
   of 1.6 ms plus 0.9 ms per token. A second design pays 0.3 ms fixed but 1.15 ms
   per token. For which batch sizes K (integer, 1..64) is the first design
   strictly faster? Give the smallest such K.

2. A cache holds S slots. Requests follow a Zipf-like popularity where item i has
   probability proportional to 1/i, over N = 64 items. With S = 16 and an ideal
   cache that always holds the 16 most popular items, compute the hit rate.
   (Use H_64 ~ 4.7439 and H_16 ~ 3.3807.)

3. An expert-parallel model routes each token to 4 of 64 experts per layer, over
   46 layers. Assuming routing were uniform and independent, what is the expected
   number of DISTINCT experts touched in one layer over a window of 32 tokens?
   Show the derivation.

4. A file of 17.2 GB is read at 2.1 MB/s. A second mirror serves the same file at
   9.3 MB/s but only after a 45-minute queue. Which finishes first, and by how
   many minutes?

5. Prove or refute: for any sequence of cache requests, LRU with 2S slots incurs
   at most as many misses as OPT (Belady) with S slots. State the classical
   result you are using.
"""

P["06-multilingua-it-en.txt"] = (
    "The following technical study is written in Italian. First translate its "
    "key findings into English (faithfully, not word by word), then answer in "
    "English: which of its design decisions would NOT transfer to a browser "
    "environment, and why?\n\n---\n"
    + excerpt("docs/engine/study/ds4.md", 1, 90) + "\n"
)

P["07-json-extract.txt"] = """Extract every benchmark result from the log below into a JSON array. Each
element must have: engine (string), device (string), metric (string), value
(number), unit (string), date (ISO). Output ONLY the JSON array, no commentary.

--- log ---
[2026-07-29 18:02] webllm baseline on rtx4090: decode 122.4 tok/s (ctx 570)
[2026-07-29 18:41] our engine, first-light gate: decode 141.0 tok/s on rtx4090
[2026-07-29 19:15] prefill measured at 697.8 ms for 2048 tokens (rtx4090, our engine)
[2026-07-30 08:13] profiler: 148 dispatches per forward, 0 bind group creations (rtx4090)
[2026-07-30 19:25] multi-step K=8 decode: 287.5 tok/s mean over 3 replicas (rtx4090)
[2026-07-30 19:25] baseline same-day K=1: 238.3 tok/s (rtx4090)
[2026-07-30 21:10] opfs read bandwidth warm: 11.7 GB/s via SyncAccessHandle (nvme box)
[2026-07-30 21:11] opfs write bandwidth: 2.2 GB/s sustained (nvme box)
[2026-07-31 00:02] oracle cpu pp512: 56.58 tok/s on 32-core box (16 threads)
[2026-07-31 00:03] oracle cpu tg64: 13.43 tok/s on 32-core box (16 threads)
[2026-07-31 00:40] m4 hero demo target: 30 tok/s minimum sustained (macbook m4)
[2026-07-31 00:41] s22 encode floor measured: 18 ms/token cpu-side (galaxy s22)
--- end log ---
"""

P["08-prosa-en-essay.txt"] = """Write a detailed technical essay (several paragraphs) on the following thesis:
"For LLM inference in the browser, the memory system - not raw kernel speed -
is the differentiating layer." Use these notes as raw material, structure them
into an argument, and add counterarguments and your own verdict.

Notes:
- WebGPU engines exist (WebLLM, wllama); kernel-level parity is achievable.
- Browser memory is capped well below native: device limits, buffer caps,
  tab kills; a "browser tax" on residency.
- MoE models make residency >> touch: 30B params, 3B active per token.
- OPFS gives GB/s-class storage: warm reads near free, cold reads disk-bound.
- Expert paging with predictive prefetch could run models ~2x host memory.
- KV cache checkpoints on disk enable instant session resume (no re-prefill).
- Speculative decoding hides expert-load latency behind verified drafts.
- Telemetry must be zero-overhead when off, or measurements lie on mobile.
- Counterpoint: native apps do all this better; why the browser at all?
- Counterpoint: predictive prefetch recall is a model property - it may not hold.
"""

for name, text in P.items():
    (OUT / name).write_text(text)
    print(f"{name}: {len(text)} chars")
