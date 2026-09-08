"""Turn raw collection output into human-readable evidence records (evidence.jsonl).

Each output record contains everything a description writer (template or LLM) needs,
with token ids decoded and per-layer magnitude percentiles attached.
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .lens import Lens, load_vectors


def load_docs(raw: Path) -> dict[str, list[int]]:
    docs = {}
    with open(raw / "docs.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs[d["doc_id"]] = d["ids"]
    return docs


def snippet(tok, ids: list[int], s: int, t: int, before: int = 6, after: int = 3) -> str:
    """Short text window around token s, with the token itself bracketed. Never shows anything past
    the current position t (that would leak the true continuation to the description writer)."""
    lo, hi = max(0, s - before), min(len(ids), s + after + 1, t + 1)
    left = tok.decode(ids[lo:s])
    mid = tok.decode(ids[s:s + 1])
    right = tok.decode(ids[s + 1:hi])
    return f"{left}[[{mid}]]{right}".replace("\n", "\\n")


def sig(x: float, n: int = 3) -> float:
    """Round to n significant figures (keeps small probabilities visible)."""
    if x == 0:
        return 0.0
    return float(f"{x:.{n}g}")


def pct_rank(values: np.ndarray):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return 100.0 * ranks / max(1, len(values) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/evidence.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="tokenizer to use")
    ap.add_argument("--context-chars", type=int, default=700)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-p", type=float, default=1e-3, help="drop effect tokens with max(p, p_without) below this")
    ap.add_argument("--ref-records", default="", help="records.jsonl of a large run; percentiles are computed against it (nearest layer)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    raw = Path(args.raw)
    lens = Lens(raw / "unembed.pt", tok, device=args.device)
    docs = load_docs(raw)
    vecs = load_vectors(raw)
    recs = [json.loads(l) for l in open(raw / "records.jsonl")]
    recs = [r for r in recs if r["id"] in vecs]
    if args.limit:
        recs = recs[: args.limit]
    print(f"{len(recs)} records, {len(docs)} docs, {len(vecs)} vectors")

    # per-layer percentiles of relative update size and of causal effect size
    by_layer = defaultdict(list)
    for i, r in enumerate(recs):
        by_layer[r["layer"]].append(i)
    rel_pct = np.zeros(len(recs)); kl_pct = np.zeros(len(recs))
    if args.ref_records:
        # percentiles relative to a large reference collection (e.g. the fineweb run), nearest layer
        ref = defaultdict(lambda: ([], []))
        for l in open(args.ref_records):
            r = json.loads(l); ref[r["layer"]][0].append(r["norm_d"] / r["norm_X"]); ref[r["layer"]][1].append(r["effect"]["all"]["kl"])
        ref = {n: (np.sort(a), np.sort(b)) for n, (a, b) in ref.items()}
        ref_layers = np.array(sorted(ref))
        for n, idxs in by_layer.items():
            rn = int(ref_layers[np.abs(ref_layers - n).argmin()])
            ra, rb = ref[rn]
            for i in idxs:
                rel_pct[i] = 100.0 * np.searchsorted(ra, recs[i]["norm_d"] / recs[i]["norm_X"]) / len(ra)
                kl_pct[i] = 100.0 * np.searchsorted(rb, recs[i]["effect"]["all"]["kl"]) / len(rb)
        print(f"percentiles vs reference {args.ref_records} (layers {list(ref)})")
    else:
        for n, idxs in by_layer.items():
            rel = np.array([recs[i]["norm_d"] / recs[i]["norm_X"] for i in idxs])
            kl = np.array([recs[i]["effect"]["all"]["kl"] for i in idxs])
            rel_pct[idxs] = pct_rank(rel); kl_pct[idxs] = pct_rank(kl)

    def tokstr(i):
        return tok.decode([i])

    with open(args.out, "w") as f:
        for i, r in enumerate(recs):
            ids = docs[r["doc_id"]]
            t = r["t"]; n = r["layer"]
            v = vecs[r["id"]]
            lz = lens.evidence(v["X"], v["d"], v["d_attn"], v["d_mlp"], k=args.k)
            ctx = tok.decode(ids[: t + 1])
            ctx = ctx[-args.context_chars:]
            cur = tokstr(ids[t])
            eff = {}
            for name, e in r["effect"].items():
                def fmt(lst):
                    out = []
                    for x in lst:
                        s = lens.display(x["id"])
                        if s is None:
                            continue
                        # drop tokens whose probability is negligible both with and without the update:
                        # log-prob ratios there are noise, not a change the model's prediction reflects
                        p_hi = max(x["p"], x.get("p_abl", 0.0))
                        if p_hi < args.min_p:
                            continue
                        out.append({"tok": s, "dlp": round(x["dlp"], 3), "p": sig(x["p"]), **({"p_abl": sig(x["p_abl"])} if "p_abl" in x else {})})
                    return out[:args.k]
                eff[name] = {
                    "up": fmt(e["up"]), "down": fmt(e["down"]), "kl": round(e["kl"], 4),
                    "top1_true": tokstr(e["top1_true"]), "top1_without": tokstr(e["top1_abl"]),
                    "p_top1_true": sig(e["p_top1_true"]), "p_top1_without": sig(e["p_top1_true_under_abl"]),
                    "top1_changes": e["top1_true"] != e["top1_abl"],
                }
            src = r["attn_sources"]
            sources = []
            for s in src["top"]:
                if s["s"] == 0 or s["s"] == t:
                    continue
                if s["frac"] < 0.04:
                    continue
                st = tokstr(ids[s["s"]])
                if not any(ch.isalnum() for ch in st):
                    continue  # whitespace / punctuation source tokens are not informative to name
                sources.append({"offset": t - s["s"], "tok": st, "snippet": snippet(tok, ids, s["s"], t),
                                "frac": round(s["frac"], 3), "mass": round(s["mass"], 3)})
            na, nm = r["norm_d_attn"], r["norm_d_mlp"]
            out = {
                "id": r["id"], "doc_id": r["doc_id"], "t": t, "layer": n, "n_layers": r["n_layers"],
                "depth_frac": round(n / r["n_layers"], 2),
                "context": ctx, "current_token": cur,
                "magnitude": {
                    "rel_norm": round(r["norm_d"] / r["norm_X"], 4), "rel_norm_pct": round(float(rel_pct[i]), 1),
                    "kl_pct": round(float(kl_pct[i]), 1),
                    "attn_share": round(na / (na + nm + 1e-9), 3), "cos_attn_mlp": round(r["cos_attn_mlp"], 3),
                    "cos_d_X": round(r["cos_d_X"], 3), "cos_Y_X": round(r["cos_Y_X"], 4),
                },
                "lens": {k: ([{kk: (sig(vv) if isinstance(vv, float) else vv) for kk, vv in x.items() if kk in ("tok", "p", "p_X", "p_Y")} for x in val] if isinstance(val, list) else round(val, 3)) for k, val in lz.items()},
                "effect": eff,
                "sources": {"top": sources[:5], "sink_frac": round(src["sink_frac"], 3), "self_frac": round(src["self_frac"], 3)},
                "model_top": [{"tok": tokstr(x["id"]), "p": sig(x["p"])} for x in r["true_top"][:5]],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print(f"  {i}/{len(recs)}", flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
