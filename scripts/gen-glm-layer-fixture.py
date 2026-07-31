# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf==0.17.1", "numpy"]
# ///
# Fixture per la conformance layer-level GLM (goal engine-fase-c2, fase 4):
# estrae i byte GREZZI dei tensori di blk.0 (layer denso) dal GGUF reale +
# 16 righe di token_embd per i primi 16 token del corpus golden (prompt p0).
# Il ktest ricostruisce il layer su GPU (MLA absorbed) e lo confronta col
# cpuref f64 naive sugli STESSI byte. Output in public/models/ (gitignored,
# servito da vite alla root): ~52 MB, rigenerabile da questo script.
# Uso: uv run scripts/gen-glm-layer-fixture.py
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as T

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path.home() / ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf"
GOLDEN = ROOT / "results/engine/golden/glm47flash/golden-glm47flash-q4_0-2026-07-31.json"
OUT_DIR = ROOT / "public/models/glm-layer0"
N_TOKENS = 16
EMBD_ROW_BYTES = 2048 // 32 * 18  # riga Q4_0 di token_embd: 64 blocchi × 18 B

# (nome corto per il ktest, tensore GGUF, tipo atteso)
PICKS = [
    ("attnNorm", "blk.0.attn_norm.weight", T.F32),
    ("wQA", "blk.0.attn_q_a.weight", T.Q4_0),
    ("qANorm", "blk.0.attn_q_a_norm.weight", T.F32),
    ("wQB", "blk.0.attn_q_b.weight", T.Q4_0),
    ("wKvA", "blk.0.attn_kv_a_mqa.weight", T.Q8_0),
    ("kvANorm", "blk.0.attn_kv_a_norm.weight", T.F32),
    ("wKB", "blk.0.attn_k_b.weight", T.Q8_0),
    ("wVB", "blk.0.attn_v_b.weight", T.Q8_0),
    ("wO", "blk.0.attn_output.weight", T.Q4_0),
    ("ffnNorm", "blk.0.ffn_norm.weight", T.F32),
    ("wGate", "blk.0.ffn_gate.weight", T.Q4_0),
    ("wUp", "blk.0.ffn_up.weight", T.Q4_0),
    ("wDown", "blk.0.ffn_down.weight", T.Q4_1),
]

golden = json.loads(GOLDEN.read_text())
tokens = golden["prompts"][0]["promptTokens"][:N_TOKENS]
assert len(tokens) == N_TOKENS

reader = GGUFReader(str(MODEL))
by_name = {t.name: t for t in reader.tensors}

blobs: list[bytes] = []
tensors: dict[str, dict] = {}
offset = 0


def add(name: str, raw: bytes, ggml_type: int) -> None:
    global offset
    pad = (-len(raw)) % 4  # offset sempre 4-allineati (Float32Array view nel worker)
    tensors[name] = {"offset": offset, "bytes": len(raw), "ggmlType": ggml_type}
    blobs.append(raw + b"\x00" * pad)
    offset += len(raw) + pad


for short, name, qtype in PICKS:
    t = by_name[name]
    assert t.tensor_type == qtype, f"{name}: tipo {t.tensor_type} != {qtype}"
    if qtype == T.F32:
        raw = np.asarray(t.data, dtype=np.float32).reshape(-1).tobytes()
    else:
        raw = np.asarray(t.data).reshape(-1).astype(np.uint8).tobytes()
    add(short, raw, int(qtype))
    print(f"{short:9s} {name}: {qtype.name} {len(raw)} B")

embd = by_name["token_embd.weight"]
assert embd.tensor_type == T.Q4_0
embd_raw = np.asarray(embd.data).reshape(-1)
rows = b"".join(
    embd_raw[tok * EMBD_ROW_BYTES : (tok + 1) * EMBD_ROW_BYTES].astype(np.uint8).tobytes()
    for tok in tokens
)
add("embdRows", rows, int(T.Q4_0))
print(f"embdRows: {N_TOKENS} token del corpus golden p0, {len(rows)} B")

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "fixture.bin").write_bytes(b"".join(blobs))
(OUT_DIR / "meta.json").write_text(json.dumps({
    "schemaVersion": 1,
    "kind": "glm-layer0-fixture",
    "modelSha256": "d0bbdfcde6e323ebf90a8b9e95da57100e972be1ec6f0bfa0fad0feaa426557e",
    "goldenCorpus": GOLDEN.name,
    "tokens": tokens,
    "embdRowBytes": EMBD_ROW_BYTES,
    "tensors": tensors,
}, indent=1))
print(f"scritto {OUT_DIR} ({offset} B)")
