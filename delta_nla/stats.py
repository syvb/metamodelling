"""Per-layer summary statistics of evidence.jsonl (for sanity checks and the dataset report)."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="data/evidence.jsonl")
    ap.add_argument("--wandb", default="")
    args = ap.parse_args()
    by = defaultdict(list)
    for l in open(args.evidence):
        e = json.loads(l); by[(e["layer"], tuple(e.get("span") or ()))].append(e)
    rows = []
    for key in sorted(by):
        es = by[key]; n = key[0]
        def med(f): return float(np.median([f(e) for e in es]))
        def frac(f): return float(np.mean([f(e) for e in es]))
        has_split = "attn" in es[0]["effect"]
        row = {
            "layer": (f"{es[0]['span'][0]}-{es[0]['span'][1]}" if es[0].get("span") else n), "n": len(es),
            "rel_norm_med": med(lambda e: e["magnitude"]["rel_norm"]),
            "kl_all_med": med(lambda e: e["effect"]["all"]["kl"]),
            "kl_p90": float(np.percentile([e["effect"]["all"]["kl"] for e in es], 90)),
            "kl_attn_med": med(lambda e: e["effect"]["attn"]["kl"]) if has_split else float("nan"),
            "kl_mlp_med": med(lambda e: e["effect"]["mlp"]["kl"]) if has_split else float("nan"),
            "later_kl_per_pos_med": med(lambda e: e["effect"]["all"].get("later_kl_sum", 0.0) / max(1, e["effect"]["all"].get("later_positions", 1))),
            "top1_flip_frac": frac(lambda e: e["effect"]["all"]["top1_changes"]),
            "attn_share_med": med(lambda e: e["magnitude"]["attn_share"]),
            "sink_frac_med": med(lambda e: e["sources"]["sink_frac"]),
            "has_source_frac": frac(lambda e: len(e["sources"]["top"]) > 0),
            "strong_source_frac": frac(lambda e: any(s["frac"] >= 0.15 for s in e["sources"]["top"])),
            "cos_Y_X_med": med(lambda e: e["magnitude"]["cos_Y_X"]),
        }
        rows.append(row)
    keys = list(rows[0].keys())
    print("\t".join(keys))
    for r in rows:
        print("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, job_type="stats")
        wb.log({"layer_stats": wandb.Table(columns=keys, data=[[r[k] for k in keys] for r in rows])}); wb.finish()


if __name__ == "__main__":
    main()
