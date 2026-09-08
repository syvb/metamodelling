"""Rewrite the `context` field of evidence.jsonl with a longer text window decoded from docs.jsonl."""
import json, sys
from transformers import AutoTokenizer
src, dst, chars = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
docs = {}
for l in open("data/raw/docs.jsonl"):
    d = json.loads(l); docs[d["doc_id"]] = d["ids"]
n = 0
with open(dst, "w") as f:
    for l in open(src):
        e = json.loads(l)
        ids = docs[e["doc_id"]]
        e["context"] = tok.decode(ids[: e["t"] + 1])[-chars:]
        f.write(json.dumps(e, ensure_ascii=False) + "\n"); n += 1
print("rewrote", n, "contexts to", chars, "chars ->", dst)
