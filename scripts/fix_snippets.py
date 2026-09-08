"""Recompute attention-source snippets in evidence.jsonl so they never extend past the current token."""
import json, sys
from transformers import AutoTokenizer
from delta_nla.build_evidence import snippet
src, dst = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
docs = {}
for l in open("data/raw/docs.jsonl"):
    d = json.loads(l); docs[d["doc_id"]] = d["ids"]
n = 0; changed = 0
with open(dst, "w") as f:
    for l in open(src):
        e = json.loads(l); ids = docs[e["doc_id"]]; t = e["t"]
        for s in e["sources"]["top"]:
            new = snippet(tok, ids, t - s["offset"], t)
            changed += new != s["snippet"]; s["snippet"] = new
        f.write(json.dumps(e, ensure_ascii=False) + "\n"); n += 1
print(f"{n} records, {changed} snippets clipped -> {dst}")
