"""How much better is a multi-block span update as a Delta-NLA target?  Uses the stored per-layer X vectors:
d_span(a->b) = X_b - X_a at the same (doc, position).  Reports, per span:
  |d|/|X_a| median, share of d energy along the mean direction, linear-from-X_a ridge FVE (honest baseline),
  lens legibility (top prob-shift token p >= 0.05 after), and the oracle token-list probe (MiniLM+ridge -> d_span).
"""
from __future__ import annotations
import argparse, glob, json, random
from collections import defaultdict
import numpy as np, torch
from .lens import Lens
from .ridge_probe import ridge_fit_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--evidence", default="data/raw/evidence.jsonl")
    ap.add_argument("--spans", default="12-18,18-24,6-12,12-24"); ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n", type=int, default=4500); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    spans = [tuple(int(x) for x in s.split("-")) for s in args.spans.split(",")]
    need_layers = sorted({l for s in spans for l in s})
    doc_of = {}
    for l in open(args.evidence):
        e = json.loads(l); doc_of[e["id"]] = e["doc_id"]
    X = defaultdict(dict)
    for f in sorted(glob.glob(f"{args.raw}/vec_*.npz")):
        with np.load(f) as z:
            ids = z["ids"]; Xa = z["X"]
            for i, rid in enumerate(ids):
                rid = str(rid); pos, L = rid.rsplit("_L", 1); L = int(L)
                if L in need_layers: X[L][pos] = Xa[i].astype(np.float32)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model); lens = Lens(f"{args.raw}/unembed.pt", tok)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    rng = random.Random(args.seed)
    print(f"{'span':>7} {'n':>5} {'|d|/|Xa|':>8} {'meanDir%':>8} {'linX->d':>8} {'lens_leg%':>9} {'oracleFVE':>9} {'oracleCos':>9} {'shufFVE':>8}")
    for a, b in spans:
        poss = sorted(set(X[a]) & set(X[b])); rng.shuffle(poss); poss = poss[: args.n]
        Xa = np.stack([X[a][p] for p in poss]); Xb = np.stack([X[b][p] for p in poss]); D = Xb - Xa
        rel = np.median(np.linalg.norm(D, axis=1) / np.linalg.norm(Xa, axis=1))
        docs = sorted({doc_of[f"{p}_L{a}"] for p in poss}); rng.shuffle(docs)
        nd = len(docs); va_docs = set(docs[: nd // 8]); te_docs = set(docs[nd // 8: nd // 4])
        split = {"tr": [], "va": [], "te": []}
        for k, p in enumerate(poss):
            d = doc_of[f"{p}_L{a}"]; split["te" if d in te_docs else "va" if d in va_docs else "tr"].append(k)
        tr, va, te = (np.array(split[k]) for k in ("tr", "va", "te"))
        mu = D[tr].mean(0); mean_share = (len(te) * (mu ** 2).sum()) / (D[te] ** 2).sum()
        # linear baseline from X_a (512 PCs)
        fm = Xa[tr].mean(0); _, _, V = np.linalg.svd(Xa[tr] - fm, full_matrices=False); P = V[:512]
        Z = (Xa - fm) @ P.T
        lin, _, _ = ridge_fit_eval(Z.astype(np.float32), D, tr, va, te, lams=(1e2, 1e3, 1e4, 1e5))
        # lens: prob shift X_a -> X_b, computed in batched chunks (one big matmul per chunk); oracle text from it
        texts = []; leg = 0
        W = lens.W_U; g = lens.g
        for c0 in range(0, len(poss), 256):
            xa = torch.from_numpy(Xa[c0:c0 + 256]); xb = torch.from_numpy(Xb[c0:c0 + 256])
            la = (g * xa / torch.sqrt((xa ** 2).mean(-1, keepdim=True) + lens.eps)) @ W.T
            lb = (g * xb / torch.sqrt((xb ** 2).mean(-1, keepdim=True) + lens.eps)) @ W.T
            pa = torch.softmax(la, -1); pb = torch.softmax(lb, -1); diff = pb - pa
            top_up = torch.topk(diff, 40, dim=-1); top_dn = torch.topk(-diff, 40, dim=-1)
            for r in range(diff.shape[0]):
                ups = [i for i in top_up.indices[r].tolist() if lens.display(i) is not None][:6]
                dns = [i for i in top_dn.indices[r].tolist() if lens.display(i) is not None][:6]
                leg += bool(ups) and float(pb[r, ups[0]]) >= 0.05
                texts.append("toward: " + " ".join(lens.display(i).strip() for i in ups) + ". away: " + " ".join(lens.display(i).strip() for i in dns))
            print(f"  span {a}->{b}: lens {min(c0 + 256, len(poss))}/{len(poss)}", flush=True)
        E = st.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        fve, cos, _ = ridge_fit_eval(E, D, tr, va, te)
        perm = np.random.default_rng(0).permutation(len(poss)); fs, _, _ = ridge_fit_eval(E[perm], D, tr, va, te)
        print(f"{a:>3}->{b:<3} {len(poss):>5} {rel:>8.3f} {100*mean_share:>8.1f} {lin:>8.3f} {100*leg/len(poss):>9.1f} {fve:>9.3f} {cos:>9.3f} {fs:>8.3f}", flush=True)
    print("\nReference (single block, oracle token lists): L12 0.025, L18 0.021, L24 0.053; linear X->d: 0.146/0.189/0.189")


if __name__ == "__main__":
    main()
