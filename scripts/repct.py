"""Recompute rel_norm_pct / kl_pct in an evidence file as percentiles WITHIN each document (across its
layers and positions). Useful for prompt-mode runs where we want to know which layers matter most."""
import json, sys
from collections import defaultdict
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
evs = [json.loads(l) for l in open(src)]
by = defaultdict(list)
for e in evs: by[(e["doc_id"], e["t"])].append(e)
for key, es in by.items():
    rel = np.array([e["magnitude"]["rel_norm"] for e in es]); kl = np.array([e["effect"]["all"]["kl"] for e in es])
    for e in es:
        e["magnitude"]["rel_norm_pct"] = round(100.0 * float((rel < e["magnitude"]["rel_norm"]).mean()), 1)
        e["magnitude"]["kl_pct"] = round(100.0 * float((kl < e["effect"]["all"]["kl"]).mean()), 1)
with open(dst, "w") as f:
    for e in evs: f.write(json.dumps(e, ensure_ascii=False) + "\n")
print("rewrote", len(evs), "records with within-doc percentiles ->", dst)
