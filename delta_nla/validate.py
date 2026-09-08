"""Groundedness check for LLM descriptions: every quoted token must appear in the evidence.

Reports per-record ungrounded quotes and aggregate stats; optionally writes a filtered file.
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter

QUOTE_RE = re.compile(r"'([^']{1,40})'")
BANNED = re.compile(r"\b(lens|measurement|evidence|signal|data shows|layer \d+)\b", re.I)


def evidence_tokens(ev: dict) -> set[str]:
    toks = set()
    def add(t):
        t = t.strip()
        if t:
            toks.add(t.lower())
    for name in ("all", "attn", "mlp"):
        e = ev["effect"][name]
        for x in e["up"] + e["down"]:
            add(x["tok"])
        add(e["top1_true"]); add(e["top1_without"])
    for k, v in ev["lens"].items():
        if isinstance(v, list):
            for x in v:
                add(x["tok"])
    for s in ev["sources"]["top"]:
        add(s["tok"])
    for x in ev["model_top"]:
        add(x["tok"])
    add(ev["current_token"])
    return toks


def check(ev: dict, desc: str) -> dict:
    toks = evidence_tokens(ev)
    quotes = [q.strip() for q in QUOTE_RE.findall(desc)]
    bad = [q for q in quotes if q.lower() not in toks]
    return {"n_quotes": len(quotes), "ungrounded": bad, "banned": bool(BANNED.search(desc)), "words": len(desc.split())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="data/evidence.jsonl")
    ap.add_argument("--descriptions", default="data/descriptions.jsonl")
    ap.add_argument("--out", default="", help="write only clean descriptions here")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()
    evs = {json.loads(l)["id"]: json.loads(l) for l in open(args.evidence)}
    descs = [json.loads(l) for l in open(args.descriptions)]
    agg = Counter(); shown = 0; clean = []
    for d in descs:
        ev = evs.get(d["id"])
        if ev is None:
            continue
        r = check(ev, d["description"])
        agg["n"] += 1; agg["quotes"] += r["n_quotes"]; agg["ungrounded_quotes"] += len(r["ungrounded"])
        agg["records_with_ungrounded"] += bool(r["ungrounded"]); agg["banned_words"] += r["banned"]
        agg["no_quotes"] += r["n_quotes"] == 0; agg["too_long"] += r["words"] > 60; agg["words"] += r["words"]
        ok = not r["ungrounded"] and not r["banned"] and r["n_quotes"] > 0 and r["words"] <= 70
        if ok:
            clean.append(d)
        elif shown < args.show:
            shown += 1
            print(f"[{d['id']}] ungrounded={r['ungrounded']} banned={r['banned']} words={r['words']}\n   {d['description']}")
    n = max(1, agg["n"])
    print(json.dumps({"records": agg["n"], "quotes_per_record": agg["quotes"] / n, "frac_records_with_ungrounded_quote": agg["records_with_ungrounded"] / n,
                      "frac_ungrounded_quotes": agg["ungrounded_quotes"] / max(1, agg["quotes"]), "frac_banned_words": agg["banned_words"] / n,
                      "frac_no_quotes": agg["no_quotes"] / n, "frac_too_long": agg["too_long"] / n, "mean_words": agg["words"] / n, "clean": len(clean)}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            for d in clean:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print("wrote", args.out, len(clean))


if __name__ == "__main__":
    main()
