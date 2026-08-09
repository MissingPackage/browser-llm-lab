# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf"]
# ///
# Enumerazione header GGUF famiglia q35 (fase 2, spec §3: i [VERIFY] si
# chiudono dal header = la fonte più vicina ai byte che eseguiamo; spec §2:
# inventario type del file UD, il reader deve FALLIRE RUMOROSO su type ignoto).
# Uso: uv run tools/oracle-moe/q35-header-dump.py <file.gguf> [...]
# Output: JSON su stdout (metadata rilevanti + istogramma type per classe tensore).
import json
import re
import sys

from gguf import GGUFReader

KEYS = (
    "general.architecture",
    "general.name",
    ".block_count",
    ".embedding_length",
    ".feed_forward_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".attention.layer_norm_rms_epsilon",
    ".context_length",
    ".expert_count",
    ".expert_used_count",
    ".expert_feed_forward_length",
    ".expert_shared_count",
    ".rope.freq_base",
    ".rope.dimension_count",
    ".rope.scaling",
    ".rope.mrope_section",
    ".ssm.",
    ".linear_",
    ".full_attention_interval",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",  # solo il conteggio
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    ".layer_types",
)


def classify(name: str) -> str:
    if re.search(r"_exps\.", name):
        return "expert"
    if re.match(r"blk\.\d+\.(linear_attn|ssm)", name):
        return "linear_attn"
    if re.match(r"blk\.\d+\.attn", name):
        return "attn"
    if re.match(r"blk\.\d+\.(ffn|shexp)", name) or "shexp" in name:
        return "ffn/shexp"
    if name.startswith(("token_embd", "output", "per_layer")):
        return "embd/head"
    return "other"


out = []
for path in sys.argv[1:]:
    r = GGUFReader(path)
    meta = {}
    for field in r.fields.values():
        if not any(k in field.name for k in KEYS):
            continue
        if field.name == "tokenizer.ggml.tokens":
            meta["tokenizer.ggml.tokens#count"] = len(field.data)
            continue
        try:
            v = field.contents()
        except Exception:
            v = f"<len {len(field.data)}>"
        if isinstance(v, list) and len(v) > 64:
            v = f"<list len {len(v)}: head {v[:8]}>"
        meta[field.name] = v
    types = {}
    for t in r.tensors:
        cls = classify(t.name)
        key = f"{cls}:{t.tensor_type.name}"
        types.setdefault(key, {"n": 0, "bytes": 0})
        types[key]["n"] += 1
        types[key]["bytes"] += int(t.n_bytes)
    out.append({"file": path, "nTensors": len(r.tensors), "meta": meta, "typeHistogram": types})

print(json.dumps(out, indent=1, default=str))
