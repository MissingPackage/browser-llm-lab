# /// script
# requires-python = ">=3.11"
# dependencies = ["llama-cpp-python==0.3.34", "numpy"]
# ///
# QUANTO E' SORPRENDENTE, PER IL RIFERIMENTO, CIO' CHE HA SCRITTO IL NOSTRO MOTORE.
#
# LA DOMANDA A CUI RISPONDE, e non e' quella di `quant-quality.py`. Quello
# confronta due FILE dentro llama.cpp; questo confronta il NOSTRO MOTORE con
# llama.cpp, che e' la preoccupazione vera: i due hanno prodotto conversazioni
# molto diverse, e la differenza puo' essere (a) la divergenza greedy attesa —
# accordo top-1 98,83%, quindi due traiettorie si separano dopo qualche decina di
# token e da li' in poi sono due conversazioni diverse — oppure (b) un difetto
# del motore che peggiora col contesto. Le due si distinguono, e cosi'.
#
# COME. Si prende il TRASCRITTO che il motore ha generato e lo si fa rileggere al
# riferimento in modo teacher-forced: per ogni token che il motore ha emesso, si
# guarda che **rango** e che **log-prob** gli dava llama.cpp in quella posizione.
#
#   - rango 1        il riferimento avrebbe scritto lo stesso token
#   - rango basso    scelta legittima, il riferimento la considerava
#   - rango enorme   il motore ha scritto qualcosa che il riferimento non
#                    contemplava: e' li' che si guarda
#
# E LA DOMANDA SPECIFICA CHE HA FATTO NASCERE QUESTO SCRIPT: il motore chiude i
# turni tardivi con `<|im_end|>` dopo 64-135 token mentre llama.cpp arriva a 400.
# Quel token e' nel trascritto: se il riferimento gli dava rango 1, la chiusura
# anticipata e' una proprieta' della traiettoria e non un difetto; se gli dava
# rango 500, il motore ha smesso di parlare per un motivo che il modello non
# aveva.
#
# COSA QUESTO SCRIPT NON PUO' DIRE. Non prova che il motore sia corretto: due
# implementazioni possono divergere su un token e riconvergere. Localizza DOVE
# divergono e QUANTO era improbabile cio' che abbiamo scritto — che e' l'unica
# cosa che si puo' misurare senza far girare i due motori sullo stesso hardware.
#
# Uso:
#   uv run scripts/transcript-audit.py --model FILE.gguf --chat results/chat/<artefatto>.json \
#       --out results/eval/audit-<tag>.json [--n-ctx 8192]
import argparse
import json
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--chat", required=True, help="artefatto di chat prodotto dal MOTORE")
ap.add_argument("--out", required=True)
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--n-threads", type=int, default=20)
args = ap.parse_args()

SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
USER = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

chat = json.loads(Path(args.chat).read_text())
ass = [t for t in chat["turns"] if t["role"] == "assistant"]
if chat.get("kind") == "llamacpp-chat-session":
    dom = [t["question"] for t in ass]
else:
    dom = [t["content"] for t in chat["turns"] if t["role"] == "user"]

llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_threads=args.n_threads,
            n_batch=512, logits_all=True, verbose=False)
V = llm.n_vocab()
print(f"[audit] {Path(args.model).name} · vocab {V} · trascritto {Path(args.chat).name} · {len(ass)} turni")

IM_END = llm.tokenize("<|im_end|>".encode(), add_bos=False, special=True)
IM_END = IM_END[0] if len(IM_END) == 1 else llm.token_eos()


def righe(n0: int, n1: int) -> np.ndarray:
    """Le righe di logit da n0 a n1 (escluso), come le ha scritte l'ultima eval."""
    return np.asarray(llm.scores[n0:n1], dtype=np.float32)


turni_out = []
first = True
for i, t in enumerate(ass):
    seg = (SYS if first else "<|im_end|>\n") + USER.format(q=dom[i])
    p_toks = llm.tokenize(seg.encode("utf-8"), add_bos=first, special=True)
    a_toks = llm.tokenize(t["content"].encode("utf-8"), add_bos=False, special=True)
    # il motore ha CHIUSO il turno con <|im_end|> quando lo stop e' eos: quel
    # token va auditato, ed e' il piu' interessante di tutti
    chiuso = t["stats"].get("stopReason") == "eos"
    seq = p_toks + a_toks + ([IM_END] if chiuso else [])
    base = llm.n_tokens
    if base + len(seq) > args.n_ctx:
        print(f"[audit] turno {i+1}: contesto esaurito ({base + len(seq)} > {args.n_ctx}), mi fermo")
        break
    llm.eval(seq)
    sc = righe(base, base + len(seq))

    # per ogni token GENERATO dal motore: rango e log-prob secondo il riferimento
    ranghi, lp = [], []
    off = len(p_toks) - 1                 # la riga che predice il 1o token generato
    gen = a_toks + ([IM_END] if chiuso else [])
    for k, tok in enumerate(gen):
        row = sc[off + k]
        m = row.max()
        lse = m + np.log(np.exp(row - m).sum())
        lp.append(float(row[tok] - lse))
        ranghi.append(int((row > row[tok]).sum()) + 1)
    r = np.array(ranghi)
    rec = {
        "turno": i + 1, "genTokens": len(a_toks), "chiusoConImEnd": chiuso,
        "rango1Frazione": float((r == 1).mean()),
        "rangoMediano": int(np.median(r)), "rangoMedio": float(r.mean()),
        "rangoMax": int(r.max()), "logprobMedia": float(np.mean(lp)),
        "primaDivergenza": int(np.argmax(r > 1)) if (r > 1).any() else None,
        # il token di chiusura, se c'e': e' la domanda che ha fatto nascere lo script
        "imEnd": ({"rango": int(r[-1]), "logprob": round(float(lp[-1]), 4)} if chiuso else None),
        "ranghi": ranghi,
    }
    turni_out.append(rec)
    fine = f" · <|im_end|> rango {rec['imEnd']['rango']} logprob {rec['imEnd']['logprob']:.3f}" if chiuso else ""
    print(f"[audit] turno {i+1:2} · {len(a_toks):4} token · top-1 {100*rec['rango1Frazione']:5.1f}%"
          f" · rango mediano {rec['rangoMediano']} · max {rec['rangoMax']}"
          f" · 1a divergenza al token {rec['primaDivergenza']}{fine}")
    first = False

tutti = np.concatenate([np.array(t["ranghi"]) for t in turni_out])
print(f"\n[audit] TOTALE {len(tutti)} token · top-1 {100*(tutti == 1).mean():.2f}% "
      f"· rango mediano {int(np.median(tutti))} · p99 {int(np.percentile(tutti, 99))}")

out = {
    "schemaVersion": 1, "kind": "transcript-audit",
    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "domanda": ("per ogni token che il MOTORE ha generato, che rango e che log-prob gli dava "
                "llama.cpp nella stessa posizione (teacher-forced)"),
    "riferimento": {"impl": "llama-cpp-python 0.3.34", "model": Path(args.model).name,
                    "nCtx": args.n_ctx, "nThreads": args.n_threads},
    "trascritto": {"file": Path(args.chat).name, "model": chat.get("model", {}).get("file"),
                   "sha256": chat.get("model", {}).get("sha256")},
    "top1Totale": float((tutti == 1).mean()),
    "turni": turni_out,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
print(f"[audit] scritto {args.out}")
