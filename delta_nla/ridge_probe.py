"""Stage-0 feasibility probe: is Delta linearly decodable from a sentence embedding of the description?

For each layer: embed descriptions (MiniLM), ridge-regress onto mean-centred Delta (and onto X as a
state-leak control), report held-out fraction of variance explained. Shuffled descriptions give the
null. Splits are by document.
"""
from __future__ import annotations
import argparse, glob, json, random
from collections import defaultdict
import numpy as np


def load_vecs(raw, ids_needed):
    X, D = {}, {}
    for f in sorted(glob.glob(f"{raw}/vec_*.npz")):
        with np.load(f) as z:
            ids = z["ids"]; Xa = z["X"]; Da = z["d"]
            for i, rid in enumerate(ids):
                rid = str(rid)
                if rid in ids_needed:
                    X[rid] = Xa[i].astype(np.float32); D[rid] = Da[i].astype(np.float32)
    return X, D


def ridge_fit_eval(E, Y, tr, va, te, lams=(1e-2, 1e-1, 1, 10, 100)):
    """Pick lambda on val, return test FVE (1 - MSE/var about train mean) and cosine."""
    em = E[tr].mean(0); ym = Y[tr].mean(0)
    Etr = E[tr] - em; Ytr = Y[tr] - ym
    G = Etr.T @ Etr; B = Etr.T @ Ytr
    best = None
    for lam in lams:
        W = np.linalg.solve(G + lam * np.eye(E.shape[1], dtype=np.float32), B)
        pv = (E[va] - em) @ W + ym
        fve = 1 - ((Y[va] - pv) ** 2).sum() / ((Y[va] - ym) ** 2).sum()
        if best is None or fve > best[0]:
            best = (fve, lam, W)
    _, lam, W = best
    pt = (E[te] - em) @ W + ym
    fve = 1 - ((Y[te] - pt) ** 2).sum() / ((Y[te] - ym) ** 2).sum()
    cos = float(np.mean(np.sum((pt - ym) * (Y[te] - ym), 1) / (np.linalg.norm(pt - ym, axis=1) * np.linalg.norm(Y[te] - ym, axis=1) + 1e-8)))
    return fve, cos, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="data/raw/evidence.jsonl")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--descriptions", required=True, help="jsonl with id + description (or 'template' to generate)")
    ap.add_argument("--field", default="description")
    ap.add_argument("--layers", default="")
    ap.add_argument("--spans", default="", help="comma list like 12-18,18-24 to restrict span records")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-per-layer", type=int, default=200)
    ap.add_argument("--features", default="minilm", choices=["minilm", "tfidf", "both", "scalars"], help="text featurisation; 'scalars' = 4 evidence scalars only (no text)")
    ap.add_argument("--residualize-x", action="store_true", help="target = d minus its ridge prediction from X (512 PCs), fit on train")
    ap.add_argument("--tfidf-dim", type=int, default=2048)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ev = {}
    for l in open(args.evidence):
        e = json.loads(l)
        key = f"{e['span'][0]}-{e['span'][1]}" if e.get("span") else e["layer"]
        if args.spans and str(key) not in args.spans.split(","):
            continue
        ev[e["id"]] = {"layer": key, "doc": e["doc_id"], "ev": e}
    if args.descriptions == "template":
        from .templates import describe
        texts = {i: describe(v["ev"], random.Random(hash(i) & 0xFFFF)) for i, v in ev.items()}
    elif args.descriptions == "context":   # calibration: the raw text window itself
        texts = {i: v["ev"]["context"][-300:] for i, v in ev.items()}
    elif args.descriptions == "oracle":    # ceiling for a token-list description: the evidence lists themselves
        def lists(e):
            eff = e["effect"]["all"]; lz = e["lens"]
            return ("gains: " + " ".join(x["tok"].strip() for x in eff["up"]) + ". loses: " + " ".join(x["tok"].strip() for x in eff["down"])
                    + ". toward: " + " ".join(x["tok"].strip() for x in lz["shift_up"]) + ". away: " + " ".join(x["tok"].strip() for x in lz["shift_down"])
                    + ". sources: " + " ".join(s["tok"].strip() for s in e["sources"]["top"]))
        texts = {i: lists(v["ev"]) for i, v in ev.items()}
    else:
        texts = {}
        for l in open(args.descriptions):
            d = json.loads(l)
            if d.get(args.field) and d["id"] in ev:
                texts[d["id"]] = d[args.field]
    keep_layers = {int(x) for x in args.layers.split(",")} if args.layers else None
    by_layer = defaultdict(list)
    for i in texts:
        if keep_layers is None or ev[i]["layer"] in keep_layers:
            by_layer[ev[i]["layer"]].append(i)
    by_layer = {n: ids for n, ids in by_layer.items() if len(ids) >= args.min_per_layer}
    all_ids = [i for ids in by_layer.values() for i in ids]
    print(f"{len(all_ids)} descriptions over layers {sorted(by_layer)}")
    X, D = load_vecs(args.raw, set(all_ids))

    feats = []
    if args.features == "scalars":
        Z = np.array([[ev[i]["ev"]["magnitude"]["rel_norm_pct"] / 100, ev[i]["ev"]["magnitude"]["kl_pct"] / 100, ev[i]["ev"]["magnitude"]["attn_share"],
                       float(ev[i]["ev"]["effect"]["all"]["top1_changes"])] for i in all_ids], dtype=np.float32)
        feats.append(np.concatenate([Z, (Z[:, :2] * 4).astype(int) / 4.0], 1))  # raw + bucketed
    if args.features in ("minilm", "both"):
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(args.model, device="cpu")
        feats.append(st.encode([texts[i] for i in all_ids], batch_size=64, normalize_embeddings=True, show_progress_bar=False))
    if args.features in ("tfidf", "both"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        M = tf.fit_transform([texts[i] for i in all_ids])
        k = min(args.tfidf_dim, M.shape[1] - 1)
        Z = TruncatedSVD(k, random_state=0).fit_transform(M).astype(np.float32)
        Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        feats.append(Z)
    emb = np.concatenate(feats, 1).astype(np.float32)
    print(f"feature dim {emb.shape[1]} ({args.features})")
    E_all = {i: emb[k] for k, i in enumerate(all_ids)}

    print(f"{'layer':>5} {'n':>5} | {'FVE_d':>7} {'cos_d':>6} | {'FVE_d_shuf':>10} | {'FVE_X':>7} {'cos_X':>6} | lam")
    for n in sorted(by_layer, key=str):
        ids = by_layer[n]
        docs = sorted({ev[i]["doc"] for i in ids}); rng.shuffle(docs)
        nd = len(docs); val_docs = set(docs[: nd // 8]); test_docs = set(docs[nd // 8: nd // 4])
        idx = {"tr": [], "va": [], "te": []}
        for k, i in enumerate(ids):
            d = ev[i]["doc"]; idx["te" if d in test_docs else "va" if d in val_docs else "tr"].append(k)
        tr, va, te = (np.array(idx[k]) for k in ("tr", "va", "te"))
        E = np.stack([E_all[i] for i in ids]).astype(np.float32)
        Dm = np.stack([D[i] for i in ids]); Xm = np.stack([X[i] for i in ids])
        Xm = Xm / np.linalg.norm(Xm, axis=1, keepdims=True)
        if args.residualize_x:
            # remove the part of d linearly predictable from X (fit on train docs only)
            fm = Xm[tr].mean(0); Xc = Xm[tr] - fm; mu = Dm[tr].mean(0)
            _, _, Vx = np.linalg.svd(Xc, full_matrices=False); Pz = Vx[:512]
            Ztr = Xc @ Pz.T; W = np.linalg.solve(Ztr.T @ Ztr + 1e2 * np.eye(512, dtype=np.float32), Ztr.T @ (Dm[tr] - mu))
            Dm = Dm - ((Xm - fm) @ Pz.T) @ W
        fve_d, cos_d, lam = ridge_fit_eval(E, Dm, tr, va, te)
        perm = np.random.default_rng(args.seed).permutation(len(ids))
        fve_s, _, _ = ridge_fit_eval(E[perm], Dm, tr, va, te)
        fve_x, cos_x, _ = ridge_fit_eval(E, Xm, tr, va, te)
        print(f"{str(n):>7} {len(ids):>5} | {fve_d:>7.3f} {cos_d:>6.3f} | {fve_s:>10.3f} | {fve_x:>7.3f} {cos_x:>6.3f} | {lam:g}")
    print("FVE = held-out fraction of variance explained (about the train mean); _shuf = descriptions shuffled across records; FVE_X = predicting the (unit-norm) incoming state instead of Delta")


if __name__ == "__main__":
    main()
