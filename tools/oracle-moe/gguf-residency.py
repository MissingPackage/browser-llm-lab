# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf"]
# ///
from gguf import GGUFReader
import re, json
r = GGUFReader("/home/neuromancer/.cache/blab-models/GLM-4.7-Flash-Q4_0.gguf")
exp = 0; nonexp = 0; per_expert = {}
for t in r.tensors:
    nb = int(t.n_bytes)
    if re.search(r"_exps\.", t.name):
        exp += nb
        m = re.match(r"blk\.(\d+)\.", t.name)
        per_expert.setdefault(m.group(1), 0)
        per_expert[m.group(1)] += nb
    else:
        nonexp += nb
layers = len(per_expert)
one_layer = exp / layers
print(json.dumps({
  "routedExpertBytes": exp, "nonExpertBytes": nonexp,
  "moeLayers": layers, "bytesPerExpert": round(one_layer/64),
  "routedGB": round(exp/1e9,2), "nonExpertGB": round(nonexp/1e9,2)}, indent=1))
