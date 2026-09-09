"""Run the v2/v3 prompt over every layer for the math prompts and print per-prompt layer trajectories."""
import argparse, json, subprocess, sys
from collections import defaultdict
ap = argparse.ArgumentParser()
ap.add_argument("--evidence", default="data/math/evidence.jsonl")
ap.add_argument("--out", default="data/math/descriptions.jsonl")
ap.add_argument("--position", default="last", choices=["last", "prev", "both"])
ap.add_argument("--layers", default="", help="comma list; default every layer present")
ap.add_argument("--prompts", default="", help="comma list of prompt ids; default all")
ap.add_argument("--reasoning", default="medium")
ap.add_argument("--run", action="store_true", help="actually call the LLM (otherwise just print what exists)")
a = ap.parse_args()
evs = [json.loads(l) for l in open(a.evidence)]
by_doc = defaultdict(list)
for e in evs: by_doc[e["doc_id"]].append(e)
ids = []
for doc, es in by_doc.items():
    if a.prompts and doc not in a.prompts.split(","): continue
    tmax = max(e["t"] for e in es)
    for e in es:
        if a.position == "last" and e["t"] != tmax: continue
        if a.position == "prev" and e["t"] != tmax - 1: continue
        if a.layers and str(e["layer"]) not in a.layers.split(","): continue
        ids.append(e["id"])
if a.run:
    subprocess.run([sys.executable, "-m", "delta_nla.rewrite", "--prompt", "v2", "--reasoning", a.reasoning, "--evidence", a.evidence,
                    "--out", a.out, "--ids", ",".join(ids), "--concurrency", "16"], check=True)
descs = {}
try:
    for l in open(a.out): d = json.loads(l); descs[d["id"]] = d
except FileNotFoundError: pass
evmap = {e["id"]: e for e in evs}
for doc in by_doc:
    rows = [(evmap[i]["layer"], i) for i in ids if evmap[i]["doc_id"] == doc]
    if not rows: continue
    e0 = evmap[rows[0][1]]
    print("=" * 110); print(f"{doc}: {e0['context']!r}   model top: {[(x['tok'], x['p']) for x in e0['model_top'][:3]]}")
    for layer, i in sorted(rows):
        e = evmap[i]; eff = e["effect"]["all"]; d = descs.get(i)
        print(f"--- L{layer:>2} kl_pct={e['magnitude']['kl_pct']:>5} rel_pct={e['magnitude']['rel_norm_pct']:>5} attn={e['magnitude']['attn_share']:.2f} | gains {[x['tok'].strip() for x in eff['up'][:4]]} loses {[x['tok'].strip() for x in eff['down'][:3]]} | toward {[x['tok'].strip() for x in e['lens']['shift_up'][:3]]} | src {[(s['tok'].strip(), s['offset']) for s in e['sources']['top'][:2]]}")
        if d: print(f"    {d['description']}")
