# /// script
# requires-python = ">=3.11"
# dependencies = ["llama-cpp-python==0.3.16"]
# ///
# Golden logits dall'oracolo llama.cpp (spec engine fase A, §Soglie di conformance).
#
# Produce:
#   tests/fixtures/engine-corpus.json      — 4 prompt chat-templated, TOKEN ID (il
#                                            motore consuma id: il tokenizer non è in
#                                            fase A)
#   results/engine/golden/golden-qwen25-05b-q4_0.json — per posizione generata (greedy):
#                                            argmax id + top-32 (id, logit f32)
#
# Determinismo: greedy; ogni elemento di output ggml è ridotto da un solo thread, ma
# n_threads è comunque pinnato e registrato. Il GGUF è identificato da SHA-256.
#
# Uso: uv run scripts/gen-golden.py
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from llama_cpp import Llama

MODEL = Path.home() / ".cache/blab-models/qwen2.5-0.5b-instruct-q4_0.gguf"
REPO = Path(__file__).resolve().parent.parent
CORPUS_OUT = REPO / "tests/fixtures/engine-corpus.json"
GOLDEN_OUT = REPO / "results/engine/golden/golden-qwen25-05b-q4_0.json"
GEN_TOKENS = 128
TOP_K = 32
N_THREADS = 4

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Scrivi una funzione Python che inverte una stringa.",
    "List three prime numbers greater than 100 and briefly say why they are prime.",
    "Qual è la capitale della Francia? Rispondi con una sola parola.",
]
TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
)

sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
print(f"[golden] modello {MODEL.name} sha256={sha[:16]}…")

llm = Llama(
    model_path=str(MODEL), n_ctx=1024, n_threads=N_THREADS, n_batch=256,
    logits_all=True, verbose=False,
)

corpus = []
golden_prompts = []
for i, q in enumerate(PROMPTS):
    text = TEMPLATE.format(q=q)
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    llm.reset()
    llm.eval(tokens)
    positions = []
    generated = []
    for _ in range(GEN_TOKENS):
        logits = np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float32)
        top = np.argsort(-logits)[:TOP_K]
        argmax = int(top[0])
        positions.append({
            "argmax": argmax,
            "top": [[int(t), round(float(logits[t]), 4)] for t in top],
        })
        generated.append(argmax)
        llm.eval([argmax])
    corpus.append({"id": f"p{i}", "text": text, "template": "qwen-chatml", "tokens": tokens})
    golden_prompts.append({"id": f"p{i}", "promptTokens": tokens, "generated": generated,
                           "positions": positions})
    print(f"[golden] {i}: prompt {len(tokens)} tok → {GEN_TOKENS} generati")

CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
CORPUS_OUT.write_text(json.dumps({
    "schemaVersion": 1, "model": MODEL.name, "modelSha256": sha, "prompts": corpus,
}, indent=1))
GOLDEN_OUT.parent.mkdir(parents=True, exist_ok=True)
GOLDEN_OUT.write_text(json.dumps({
    "schemaVersion": 1, "kind": "engine-golden", "model": MODEL.name, "modelSha256": sha,
    "oracle": {"impl": "llama-cpp-python", "version": "0.3.16", "nThreads": N_THREADS,
               "nCtx": 1024, "sampling": "greedy"},
    "genTokens": GEN_TOKENS, "topK": TOP_K, "prompts": golden_prompts,
}, indent=1))
print(f"[golden] scritti {CORPUS_OUT.relative_to(REPO)} e {GOLDEN_OUT.relative_to(REPO)}")
