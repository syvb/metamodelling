"""Assemble the warm-start SFT dataset.

Inputs: evidence.jsonl (+ descriptions.jsonl from the LLM rewrite). Output: one jsonl with, per record,
the description variants and pointers to the stored vectors, plus a train/val split by document.

  verbalizer SFT:    (X, d)  -> description
  reconstructor SFT: description -> d   (and d_attn / d_mlp if you want sublayer heads)

Description sources per record (all kept; the trainer chooses the mix):
  llm        grounded LLM rewrite (primary)
  llm_short  its one-clause version
  template   templated no-LLM description (control / augmentation)
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from pathlib import Path

from .templates import describe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="data/evidence.jsonl")
    ap.add_argument("--descriptions", default="data/descriptions.jsonl")
    ap.add_argument("--raw", default="data/raw", help="dir with vec_*.npz (recorded as pointer only)")
    ap.add_argument("--out", default="data/warmstart.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default="")
    args = ap.parse_args()

    evs = [json.loads(l) for l in open(args.evidence)]
    descs = {}
    if Path(args.descriptions).exists():
        for l in open(args.descriptions):
            d = json.loads(l)
            if d.get("description"):
                descs[d["id"]] = d
    rng = random.Random(args.seed)
    docs = sorted({e["doc_id"] for e in evs})
    rng.shuffle(docs)
    val_docs = set(docs[: int(len(docs) * args.val_frac)])

    n_llm = 0; per_layer = Counter(); words = []
    with open(args.out, "w") as f:
        for e in evs:
            d = descs.get(e["id"])
            rec = {
                "id": e["id"], "doc_id": e["doc_id"], "layer": e["layer"], "t": e["t"],
                "split": "val" if e["doc_id"] in val_docs else "train",
                "vectors": {"dir": args.raw, "id": e["id"]},
                "template": describe(e, random.Random(__import__("zlib").crc32(e["id"].encode()))),
                "llm": d["description"] if d else None,
                "llm_short": d["short"] if d else None,
                "magnitude": e["magnitude"],
            }
            if d:
                n_llm += 1; words.append(len(d["description"].split()))
            per_layer[e["layer"]] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats = {"records": len(evs), "with_llm": n_llm, "val_docs": len(val_docs), "per_layer": dict(per_layer),
             "llm_words_mean": sum(words) / max(1, len(words))}
    print(json.dumps(stats, indent=1))
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, job_type="dataset", config=vars(args))
        wb.log(stats)
        tbl = wandb.Table(columns=["id", "layer", "context_tail", "llm", "llm_short", "template"])
        evmap = {e["id"]: e for e in evs}
        for rid in rng.sample(list(descs), min(200, len(descs))):
            e = evmap[rid]; d = descs[rid]
            tbl.add_data(rid, e["layer"], e["context"][-120:], d["description"], d["short"], describe(e))
        wb.log({"samples": tbl})
        art = wandb.Artifact("warmstart-dataset", type="dataset"); art.add_file(args.out); wb.log_artifact(art)
        wb.finish()


if __name__ == "__main__":
    main()
