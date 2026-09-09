"""Diagnostic: are the reconstructor's *frozen* features informative about d?

Extract last-token (post-norm) and mean-pooled features from the truncated base model for each description,
then fit ridge -> normalised d on train, report held-out FVE. If this is clearly > 0 while the trained AR is ~0,
the AR optimisation (not the data) is the problem.
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from .data import load_pairs, split_by_doc, TargetNorm, fve
from .train_ar import build_prompt, truncate


def ridge(Etr, Ytr, Eva, Yva, Ete, Yte, lams=(1e-2, 1e-1, 1, 10, 100, 1000)):
    em, ym = Etr.mean(0), Ytr.mean(0); A = (Etr - em).T @ (Etr - em); B = (Etr - em).T @ (Ytr - ym)
    best = None
    for lam in lams:
        W = np.linalg.solve(A + lam * np.eye(A.shape[0], dtype=A.dtype), B)
        f = 1 - (((Yva - ym) - (Eva - em) @ W) ** 2).sum() / ((Yva - ym) ** 2).sum()
        if best is None or f > best[0]: best = (f, lam, W)
    _, lam, W = best
    return 1 - (((Yte - ym) - (Ete - em) @ W) ** 2).sum() / ((Yte - ym) ** 2).sum(), lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--ar-layers", type=int, default=24)
    ap.add_argument("--evidence", default="data/raw/evidence.jsonl"); ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--descriptions", default="data/descriptions_v34.jsonl"); ap.add_argument("--templates", action="store_true")
    ap.add_argument("--layers", default="12,18,24"); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "right"
    layers = [int(x) for x in args.layers.split(",")]
    pairs = load_pairs(args.evidence, args.descriptions, args.raw, layers, use_templates=args.templates, limit=args.limit)
    train, val = split_by_doc(pairs); norm = TargetNorm(train)
    n_layers_total = json.loads(open(args.evidence).readline())["n_layers"]
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = truncate(AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype), args.ar_layers).to(args.device).eval()
    feats = {}
    allp = train + val
    with torch.no_grad():
        for i in range(0, len(allp), args.bs):
            ps = allp[i:i + args.bs]
            enc = tok([build_prompt(p.layer, n_layers_total, p.text) for p in ps], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(args.device)
            out = model(**enc, output_hidden_states=True)
            am = enc.attention_mask
            hs = out.hidden_states
            last_idx = am.sum(1) - 1
            for name, h in (("last_postnorm", hs[-1]), ("last_preNorm", hs[-2]), ("mid", hs[len(hs) // 2])):
                lastv = h[torch.arange(h.shape[0], device=h.device), last_idx].float()
                meanv = (h.float() * am.unsqueeze(-1)).sum(1) / am.sum(1, keepdim=True)
                for p, a, b in zip(ps, lastv.cpu().numpy(), meanv.cpu().numpy()):
                    feats.setdefault(name + "/last", {})[p.id] = a; feats.setdefault(name + "/mean", {})[p.id] = b
            if i % (args.bs * 20) == 0: print(f"features {i}/{len(allp)}", flush=True)
    # per-layer ridge with a val split carved from train docs
    import random
    tr_docs = sorted({p.doc for p in train}); random.Random(1).shuffle(tr_docs); va_docs = set(tr_docs[: len(tr_docs) // 8])
    for name, F in feats.items():
        res = []
        for n in layers:
            trn = [p for p in train if p.layer == n and p.doc not in va_docs]; va = [p for p in train if p.layer == n and p.doc in va_docs]; te = [p for p in val if p.layer == n]
            E = lambda ps: np.stack([F[p.id] for p in ps]).astype(np.float64); Y = lambda ps: np.stack([norm.encode(p.d, p.layer) for p in ps]).astype(np.float64)
            f, lam = ridge(E(trn), Y(trn), E(va), Y(va), E(te), Y(te))
            res.append(f"L{n}: {f:.4f} (lam {lam:g})")
        print(f"{name:>20}: " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()
