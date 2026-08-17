# /// script
# requires-python = ">=3.11"
# dependencies = ["llama-cpp-python==0.3.34", "numpy"]
# ///
# LA MISURA DI QUALITA' DI UN GGUF, teacher-forced e deterministica.
#
# A COSA SERVE. Decidere se una quantizzazione piu' aggressiva (che farebbe
# entrare il parco expert nell'arena, e quindi vale ~3,5x di velocita' sul 35B)
# costa troppa intelligenza. La decisione e' del PI; questo script produce la
# meta' misurabile.
#
# PERCHE' TEACHER-FORCED E NON GENERAZIONE. Generare introduce campionamento,
# quindi varianza, quindi la necessita' di ripetere; e a 8-11 tok/s su CPU
# costerebbe ore per un numero peggiore. Qui il modello non genera mai: si
# misura la sorpresa (NLL) su token FISSATI. Due esecuzioni danno lo stesso
# risultato bit per bit.
#
# LE DUE MISURE, e sono per domande diverse:
#   1. bits/token per SEZIONE di prosa (it tecnico, codice, wikitext-2 en).
#      Il danno di una quantizzazione non e' uniforme fra domini, e un numero
#      unico medierebbe proprio cio' che decide se il modello resta usabile.
#   2. log-prob e RANGO della risposta giusta su compiti verificabili (GSM8K,
#      MMLU). NON si contano le risposte corrette: e' la landmine gia' pagata da
#      questo repo — «un campione da 22 posizioni non distingue niente, ±1 colpo
#      vale ±4,5 punti; guarda rango e log-prob del bersaglio, stessa
#      informazione, varianza molto piu' bassa».
#
# COSA SCRIVE, e perche' scrive anche i vettori per token: il confronto fra due
# modelli e' APPAIATO (stesso token, stessa posizione), e l'appaiamento richiede
# i valori per posizione, non le loro medie. `quant-quality-compare.mjs` li usa
# e RIFIUTA di confrontare artefatti i cui token non coincidono.
#
# LA VERSIONE DELL'ORACOLO NON E' QUELLA DEI GOLDEN, ed e' deliberato.
# `gen-golden.py` pinna llama-cpp-python **0.3.16** e non va toccato: riprodurre
# quei golden e' il suo mestiere. Ma la 0.3.16 **non carica** le architetture
# Qwen3.5/3.6 («Failed to load model from file», verificato sul 4B): porta una
# llama.cpp anteriore a questi modelli. Qui serve la **0.3.34**, e la versione
# finisce nell'artefatto perche' due bracci misurati con oracoli diversi non
# sono confrontabili — `quant-quality-compare.mjs` lo rifiuta.
#
# MEMORIA: con logits_all=True llama.cpp alloca (n_ctx x vocab) float32. Sul 35B
# il vocabolario e' 248.320: a n_ctx 4096 sarebbero 4 GiB, a 1024 sono 1 GiB.
# Le finestre sono quindi da 1024 e la riduzione avviene a fette di righe.
#
# Uso:
#   uv run scripts/quant-quality.py --model PATH.gguf --corpus results/eval/quant-corpus.json \
#       --out results/eval/quant-quality-<tag>.json [--n-ctx 1024] [--n-threads 30]
#       [--max-windows N] [--max-tasks N] [--no-sha]
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama, llama_cpp

REPO = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", default=str(REPO / "results/eval/quant-corpus.json"))
ap.add_argument("--out", required=True)
ap.add_argument("--n-ctx", type=int, default=1024)
ap.add_argument("--n-threads", type=int, default=max(1, (os.cpu_count() or 8) - 2))
ap.add_argument("--n-batch", type=int, default=512)
ap.add_argument("--max-windows", type=int, default=0, help="0 = tutte")
ap.add_argument("--max-tasks", type=int, default=0, help="0 = tutti")
ap.add_argument("--no-sha", action="store_true", help="salta lo SHA del GGUF (piu' veloce, artefatto meno identificabile)")
args = ap.parse_args()

MODEL = Path(args.model)
CORPUS = json.loads(Path(args.corpus).read_text())

# Il template di chat: i compiti si misurano nel regime in cui il modello viene
# davvero usato. E' lo stesso di `gen-golden.py`, e la polarita' del thinking e'
# quella del file (`enable_thinking` non renderizzato qui: il blocco <think>
# viene CHIUSO subito, cosi' il bersaglio e' il primo token di risposta e non
# l'inizio di un ragionamento).
TEMPLATE = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


print(f"[qual] modello {MODEL.name} ({MODEL.stat().st_size/2**30:.2f} GiB)")
sha = "" if args.no_sha else sha256_file(MODEL)
if sha:
    print(f"[qual] sha256 {sha[:16]}…")

t_load = time.time()
llm = Llama(model_path=str(MODEL), n_ctx=args.n_ctx, n_threads=args.n_threads,
            n_batch=args.n_batch, logits_all=True, verbose=False)
load_s = time.time() - t_load
n_vocab = llm.n_vocab()
print(f"[qual] caricato in {load_s:.1f} s · vocab {n_vocab} · n_ctx {args.n_ctx} · {args.n_threads} thread")


def nll_of_window(tokens):
    """NLL per posizione di una finestra: nll[i] e' la sorpresa del token i+1.

    La softmax si calcola a fette di righe: la matrice intera dei logit e' 1 GiB
    e i suoi temporanei la raddoppierebbero.
    """
    llm.reset()
    llm.eval(tokens)
    scores = np.asarray(llm.scores[: len(tokens)], dtype=np.float32)
    tgt = np.asarray(tokens[1:], dtype=np.int64)
    n = len(tgt)
    nll = np.empty(n, dtype=np.float32)
    top1 = np.empty(n, dtype=np.int32)
    STEP = 128
    for a in range(0, n, STEP):
        b = min(a + STEP, n)
        blk = scores[a:b]
        m = blk.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(blk - m).sum(axis=1))
        rows = np.arange(b - a)
        nll[a:b] = lse - blk[rows, tgt[a:b]]
        top1[a:b] = blk.argmax(axis=1)
    return nll, top1


# ---- 1. bits/token sulle sezioni di prosa ----------------------------------
prose_out = []
for sec in CORPUS["prose"]:
    # add_bos=True e' la convenzione di llama-perplexity: la prima posizione di
    # ogni finestra non ha un predecessore, e senza BOS il modello pagherebbe
    # una sorpresa che non c'entra col quant.
    toks = llm.tokenize(sec["text"].encode("utf-8"), add_bos=True, special=False)
    W = args.n_ctx
    n_win = len(toks) // W
    if args.max_windows:
        n_win = min(n_win, args.max_windows)
    if n_win == 0:
        raise SystemExit(f"[qual] sezione {sec['id']}: {len(toks)} token < finestra {W}")
    all_nll, all_top1, all_tok = [], [], []
    t0 = time.time()
    for w in range(n_win):
        win = toks[w * W:(w + 1) * W]
        nll, top1 = nll_of_window(win)
        all_nll.append(nll)
        all_top1.append(top1)
        all_tok.append(np.asarray(win[1:], dtype=np.int32))
        print(f"\r[qual] {sec['id']}: finestra {w+1}/{n_win} · {(time.time()-t0):.0f} s   ", end="")
    nll = np.concatenate(all_nll)
    prose_out.append({
        "id": sec["id"], "domain": sec["domain"], "source": sec["source"],
        "sourceSha256": sec["sourceSha256"],
        "windows": n_win, "windowTokens": W, "addBos": True,
        "nTokensScored": int(nll.size),
        "bitsPerToken": float(nll.mean() / np.log(2)),
        "seconds": round(time.time() - t0, 1),
        # i vettori per posizione: sono cio' che rende APPAIABILE il confronto
        "tokens": np.concatenate(all_tok).tolist(),
        "nll": [round(float(x), 5) for x in nll],
        "top1": np.concatenate(all_top1).tolist(),
    })
    print(f"\r[qual] {sec['id']}: {n_win} finestre · {nll.size} token · "
          f"{prose_out[-1]['bitsPerToken']:.4f} bit/token · {prose_out[-1]['seconds']:.0f} s")

# ---- 2. log-prob e rango del bersaglio sui compiti -------------------------
tasks = CORPUS["tasks"][: args.max_tasks] if args.max_tasks else CORPUS["tasks"]
tasks_out = []
t0 = time.time()
for i, t in enumerate(tasks):
    p_toks = llm.tokenize(TEMPLATE.format(q=t["prompt"]).encode("utf-8"), add_bos=True, special=True)
    a_toks = llm.tokenize(t["answer"].encode("utf-8"), add_bos=False, special=False)
    seq = p_toks + a_toks
    if len(seq) > args.n_ctx:
        seq = seq[-args.n_ctx:]
        p_len = len(seq) - len(a_toks)
    else:
        p_len = len(p_toks)
    llm.reset()
    llm.eval(seq)
    scores = np.asarray(llm.scores[: len(seq)], dtype=np.float32)
    lp, ranks = [], []
    for k, tok in enumerate(a_toks):
        row = scores[p_len - 1 + k]
        m = row.max()
        lse = m + np.log(np.exp(row - m).sum())
        lp.append(float(row[tok] - lse))
        # rango del bersaglio: quanti token lo precedono in logit, +1
        ranks.append(int((row > row[tok]).sum()) + 1)
    tasks_out.append({
        "id": t["id"], "domain": t["domain"], "answer": t["answer"],
        "answerTokens": a_toks, "promptTokens": p_len,
        "logprob": [round(x, 5) for x in lp],
        "logprobSum": round(float(sum(lp)), 5),
        "rank": ranks,
        "rank1": ranks[0],
        "top1": int(scores[p_len - 1].argmax()),
    })
    if (i + 1) % 10 == 0:
        print(f"\r[qual] compiti {i+1}/{len(tasks)} · {(time.time()-t0):.0f} s   ", end="")
print(f"\r[qual] compiti {len(tasks_out)} · {(time.time()-t0):.0f} s                 ")

out = {
    "schemaVersion": 1,
    "kind": "quant-quality",
    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model": {"file": MODEL.name, "bytes": MODEL.stat().st_size, "sha256": sha},
    "corpus": {"path": str(Path(args.corpus).name), "sha256": CORPUS["corpusSha256"]},
    "oracle": {
        "impl": "llama-cpp-python", "version": "0.3.34",
        "llamaCpp": getattr(llama_cpp, "__version__", "n/d"),
        "nThreads": args.n_threads, "nCtx": args.n_ctx, "nBatch": args.n_batch,
        "nGpuLayers": 0, "mode": "teacher-forced (nessun campionamento)",
        "loadSeconds": round(load_s, 1),
    },
    "prose": prose_out,
    "tasks": tasks_out,
}
outp = Path(args.out)
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(out))
print(f"[qual] scritto {outp}")
for p in prose_out:
    print(f"[qual]   {p['id']:<11} {p['bitsPerToken']:.4f} bit/token  ({p['nTokensScored']} token)")
