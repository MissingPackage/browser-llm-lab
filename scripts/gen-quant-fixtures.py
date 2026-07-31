# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf==0.17.1", "numpy"]
# ///
# Fixture per i test di dequant TS (goal engine-fase-c2, fase 3): estrae i primi
# 2 blocchi di tensori REALI del GGUF GLM-4.7-Flash per ciascun tipo quant nuovo
# (Q4_1, Q5_K, Q6_K) + Q4_0/Q8_0 di controllo, e li dequantizza con gguf-py
# (implementazione INDIPENDENTE dal motore e da llama.cpp C). Il test TS deve
# riprodurre questi float32 esattamente.
# Uso: uv run scripts/gen-quant-fixtures.py
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize

MODEL = Path.home() / ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf"
OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/glm-quant-blocks.json"
N_BLOCKS = 2

PICKS = [
    ("blk.1.ffn_down_exps.weight", T.Q4_1, 20, 32),
    ("blk.1.ffn_gate_shexp.weight", T.Q5_K, 176, 256),
    ("blk.1.ffn_down_shexp.weight", T.Q6_K, 210, 256),
    ("blk.1.ffn_gate_exps.weight", T.Q4_0, 18, 32),
    ("blk.1.attn_kv_a_mqa.weight", T.Q8_0, 34, 32),
]

reader = GGUFReader(str(MODEL))
by_name = {t.name: t for t in reader.tensors}
cases = []
for name, qtype, block_bytes, block_weights in PICKS:
    t = by_name[name]
    assert t.tensor_type == qtype, f"{name}: tipo {t.tensor_type} != {qtype}"
    raw = np.asarray(t.data).reshape(-1)[: N_BLOCKS * block_bytes].astype(np.uint8)
    vals = dequantize(raw.reshape(N_BLOCKS, block_bytes), qtype).astype(np.float32).reshape(-1)
    assert vals.size == N_BLOCKS * block_weights
    cases.append({
        "tensor": name,
        "ggmlType": int(qtype),
        "blockBytes": block_bytes,
        "blockWeights": block_weights,
        "nBlocks": N_BLOCKS,
        "bytesB64": base64.b64encode(raw.tobytes()).decode(),
        "expected": [float(v) for v in vals],
    })
    print(f"{name}: {qtype.name} {N_BLOCKS} blocchi, range [{vals.min():.4g}, {vals.max():.4g}]")

OUT.write_text(json.dumps({
    "schemaVersion": 1,
    "kind": "glm-quant-fixture",
    "modelSha256": "d0bbdfcde6e323ebf90a8b9e95da57100e972be1ec6f0bfa0fad0feaa426557e",
    "oracle": "gguf-py 0.17.1 dequantize (implementazione indipendente)",
    "cases": cases,
}, indent=1))
print(f"scritto {OUT}")
