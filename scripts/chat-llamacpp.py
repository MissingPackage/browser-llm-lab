# /// script
# requires-python = ">=3.11"
# dependencies = ["llama-cpp-python==0.3.34", "numpy"]
# ///
# LA STESSA CONVERSAZIONE, SU LLAMA.CPP — il braccio di riferimento.
#
# A COSA SERVE. I due bracci del nostro motore (Q4_K_S e Q2_K) si confrontano
# fra loro, ma entrambi girano sul NOSTRO codice: se sbagliassimo qualcosa,
# sbaglieremmo in tutti e due allo stesso modo e il confronto non se ne
# accorgerebbe. llama.cpp e' l'implementazione canonica dello stesso file: le
# sue risposte sono il termine di paragone che non condivide i nostri difetti.
#
# PERCHE' IL Q4_K_S E NON IL Q2_K. La domanda a cui questo braccio risponde e'
# «quanto ci allontaniamo da una risposta di riferimento», e il riferimento e' il
# quant piu' fedele fra quelli che giriamo. Un quarto braccio (llama.cpp sul
# Q2_K) isolerebbe motore-contro-riferimento a parita' di quant: e' un'altra
# domanda, e si fa con lo stesso script cambiando --model.
#
# IL RENDERING E' COPIATO DAL NOSTRO, byte per byte. Non e' pedanteria: il
# template Jinja del GGUF NON viene eseguito dal nostro banco (goal
# velocita-decode, §thinking), quindi usare `create_chat_completion` di
# llama-cpp-python — che il template lo applica — confronterebbe due PROMPT
# diversi e attribuirebbe al modello una differenza che e' del rendering. La
# forma qui sotto e' quella che l'artefatto della chat riporta in
# `stats.renderedPrompt`.
#
# VELOCITA': si misura, ma NON e' confrontabile con quella del motore in
# browser. Questo gira su CPU, quello su GPU. Il numero e' qui perche' un
# riferimento senza il suo costo non dice se e' un riferimento praticabile.
#
# Uso:
#   uv run scripts/chat-llamacpp.py --model ~/.cache/.../file.gguf \
#       --out results/chat/chat-llamacpp-<tag>.json [--max-new 400] [--n-ctx 8192]
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama

REPO = Path(__file__).resolve().parent.parent

# LA STESSA CONVERSAZIONE di `scripts/chat-smoke.mjs`. Duplicata qui e non
# importata perche' quello e' un modulo JS: il gate e' che i due elenchi
# coincidano, e il test `tests/chat-conversazione.test.ts` lo verifica.
CONVERSAZIONE = [
    "Che relazione c'e' tra entropia dell'informazione e compressione?",
    "Puoi farmi un esempio numerico concreto con un alfabeto di quattro simboli?",
    "E se le probabilita' fossero tutte uguali, cosa cambierebbe in quell'esempio?",
    "Come si collega alla codifica di Huffman? Mostrami l'albero per l'esempio di prima.",
    "Perche' la codifica aritmetica riesce a fare meglio di Huffman?",
    "Nei modelli linguistici si parla di bit per token: e' la stessa entropia di cui parlavamo?",
    "Quindi una perplessita' di 6 quanti bit per token sono, e come la interpreto?",
    "Se quantizzo i pesi di un modello, cosa succede a quei bit per token?",
    "Riassumi in cinque punti quello che ci siamo detti finora.",
    "Dove cadrebbe questo ragionamento se i dati non fossero stazionari?",
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-new", type=int, default=400)
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--n-threads", type=int, default=max(1, (os.cpu_count() or 8) - 2))
ap.add_argument("--turns", type=int, default=10)
ap.add_argument("--no-sha", action="store_true")
args = ap.parse_args()

MODEL = Path(args.model)
SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
USER = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


sha = "" if args.no_sha else sha256_file(MODEL)
print(f"[llama] {MODEL.name} ({MODEL.stat().st_size / 2**30:.2f} GiB) sha {sha[:16]}…")

t0 = time.time()
# logits_all=False: qui si GENERA, e serve solo l'ultima riga di logit. Con True
# llama.cpp allocherebbe n_ctx x vocab float32 = 7,6 GiB a n_ctx 8192.
llm = Llama(model_path=str(MODEL), n_ctx=args.n_ctx, n_threads=args.n_threads,
            n_batch=512, logits_all=False, verbose=False)
load_s = time.time() - t0
print(f"[llama] caricato in {load_s:.1f} s · vocab {llm.n_vocab()} · {args.n_threads} thread\n")


def last_logits() -> np.ndarray:
    """I logit dell'ULTIMO token valutato, letti dal contesto.

    NON si usa `llm.scores`: con `logits_all=False` la `eval()` di
    llama-cpp-python 0.3.34 **non ci scrive affatto** (il blocco che lo faceva e'
    commentato nel pacchetto, «logits are only needed for logprobs»). Leggerlo
    comunque dava un buffer mai scritto — cioe' un argmax su zeri, che produce
    testo degenere con metriche di velocita' PERFETTAMENTE PLAUSIBILI. E' la
    classe di difetto che questo progetto teme di piu', e qui l'ha prodotta un
    dettaglio di una libreria: la guardia sul testo, sotto, esiste per questo.
    """
    return np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),))

EOS = {llm.token_eos()}
for t in ("<|im_end|>", "<|endoftext|>"):
    ids = llm.tokenize(t.encode(), add_bos=False, special=True)
    if len(ids) == 1:
        EOS.add(ids[0])

turns = []
first = True
for i, q in enumerate(CONVERSAZIONE[: args.turns]):
    # il segmento NUOVO: al primo turno c'e' anche il system; dai successivi si
    # chiude la risposta precedente con <|im_end|>, esattamente come fa la pagina
    seg = (SYS if first else "<|im_end|>\n") + USER.format(q=q)
    toks = llm.tokenize(seg.encode("utf-8"), add_bos=first, special=True)
    pos_start = llm.n_tokens

    t_pre = time.time()
    llm.eval(toks)                      # prefill del solo segmento nuovo: la KV si riusa
    nxt = int(last_logits().argmax())   # greedy, come il motore a temperature 0
    ttft_ms = (time.time() - t_pre) * 1000

    out_ids = []
    t_dec = time.time()
    stop = "maxNew"
    for _ in range(args.max_new):
        if nxt in EOS:
            stop = "eos"
            break
        out_ids.append(nxt)
        llm.eval([nxt])
        nxt = int(last_logits().argmax())
    dec_s = time.time() - t_dec
    text = llm.detokenize(out_ids).decode("utf-8", errors="replace")
    # il conteggio esclude il primo token, che e' gia' nel TTFT: stessa
    # convenzione dell'artefatto di chat
    tok_s = (len(out_ids) - 1) / dec_s if dec_s > 0 and len(out_ids) > 1 else 0.0
    turns.append({
        "role": "assistant", "question": q, "content": text,
        "stats": {
            "genTokens": len(out_ids), "decodeTokS": round(tok_s, 3),
            "ttftMs": round(ttft_ms, 1), "promptTokens": len(toks),
            "posStart": pos_start, "posEnd": llm.n_tokens, "stopReason": stop,
        },
    })
    # GUARDIA CONTRO IL TESTO DEGENERE. Un buffer di logit sbagliato non fa
    # fallire niente: produce token ripetuti e un tok/s credibile. Si controlla
    # che la risposta abbia varieta' lessicale minima, subito, invece di
    # scoprirlo leggendo l'artefatto.
    if len(set(out_ids)) < 10 or len(text.strip()) < 40:
        raise SystemExit(
            f"[llama] turno {i+1}: risposta DEGENERE ({len(out_ids)} token, "
            f"{len(set(out_ids))} distinti, {len(text.strip())} caratteri) — "
            "i logit letti non sono quelli del modello")
    print(f"[llama] turno {i+1:2}/{args.turns} · {len(out_ids):4} token · {tok_s:6.2f} tok/s "
          f"· TTFT {ttft_ms/1000:5.2f} s · ctx {llm.n_tokens} · {stop}")
    first = False

coda = turns[len(turns) // 2:]
regime = sum(t["stats"]["decodeTokS"] for t in coda) / max(1, len(coda))
print(f"\n[llama] regime (media degli ultimi {len(coda)} turni): {regime:.2f} tok/s")

out = {
    "schemaVersion": 1,
    "kind": "llamacpp-chat-session",
    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "declared": ("riferimento di QUALITA', non di velocita': gira su CPU e i suoi tok/s non "
                 "sono confrontabili con quelli del motore in browser, che gira su GPU"),
    "model": {"file": MODEL.name, "bytes": MODEL.stat().st_size, "sha256": sha},
    "engine": {"impl": "llama-cpp-python", "version": "0.3.34", "nThreads": args.n_threads,
               "nCtx": args.n_ctx, "nGpuLayers": 0, "loadSeconds": round(load_s, 1)},
    "params": {"sampling": "greedy (temperature 0)", "maxNew": args.max_new,
               "rendering": "copiato da chat.worker: system + im_start/im_end, template Jinja NON eseguito"},
    "regimeTokS": round(regime, 3),
    "turns": turns,
}
outp = Path(args.out)
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"[llama] scritto {outp}")
