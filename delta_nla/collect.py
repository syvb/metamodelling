"""Collect Delta-NLA warm-start evidence from a HF causal LM.

For each (document, position t, layer n) we record
  X      = residual stream entering block n           (hidden_states[n][t])
  Y      = residual stream leaving block n            (hidden_states[n+1][t])
  d      = Y - X, split into d_attn (self_attn output) and d_mlp (mlp output)
plus model-derived evidence about what the update *did*:
  * attention sources: which earlier tokens block n's attention pulled from at t,
    scored by attention weight x norm of the value's OV contribution
  * causal (total) effect: final next-token log-probs at t with the update removed
    (all / attention-only / mlp-only), compared to the true log-probs
  * norms of X, d, d_attn, d_mlp and their cosines
Vectors are stored in fp16 .npy shards; metadata goes to jsonl. Lens evidence is
computed offline (see lens.py) from the saved unembedding.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@dataclass
class Capture:
    attn_out: dict = field(default_factory=dict)   # layer -> [S, d]
    mlp_out: dict = field(default_factory=dict)    # layer -> [S, d]
    attn_w: dict = field(default_factory=dict)     # layer -> [H, S, S]
    v: dict = field(default_factory=dict)          # layer -> [S, KV*dh]
    next_kwargs: dict = field(default_factory=dict)  # layer -> kwargs seen by layer n+1


def install_hooks(model, layers: list[int], cap: Capture):
    handles = []
    blocks = model.model.layers
    for n in layers:
        blk = blocks[n]
        def attn_hook(mod, args, kwargs, out, n=n):
            cap.attn_out[n] = out[0][0].detach()
            if out[1] is not None:
                cap.attn_w[n] = out[1][0].detach()
        def mlp_hook(mod, args, out, n=n):
            cap.mlp_out[n] = out[0].detach()
        def v_hook(mod, args, out, n=n):
            cap.v[n] = out[0].detach()
        handles.append(blk.self_attn.register_forward_hook(attn_hook, with_kwargs=True))
        handles.append(blk.mlp.register_forward_hook(mlp_hook))
        handles.append(blk.self_attn.v_proj.register_forward_hook(v_hook))
        if n + 1 < len(blocks):
            def pre_hook(mod, args, kwargs, n=n):
                cap.next_kwargs[n] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
                if args:
                    cap.next_kwargs[n]["_args"] = args[1:]
            handles.append(blocks[n + 1].register_forward_pre_hook(pre_hook, with_kwargs=True))
    return handles


@torch.no_grad()
def attention_sources(model, n: int, t: int, attn_w: torch.Tensor, v: torch.Tensor, topk: int = 8):
    """Score source positions s<=t by sum_h a[h,t,s] * || v_kv(h)[s] W_O^h ||."""
    attn = model.model.layers[n].self_attn
    H = model.config.num_attention_heads
    KV = model.config.num_key_value_heads
    dh = attn.head_dim
    W_O = attn.o_proj.weight  # [d_model, H*dh]
    vv = v[: t + 1].view(t + 1, KV, dh).float()  # [S, KV, dh]
    a = attn_w[:, t, : t + 1].float()  # [H, S]
    M = torch.empty(H, t + 1, device=v.device)
    for h in range(H):
        kv = h // (H // KV)
        contrib = vv[:, kv, :] @ W_O[:, h * dh:(h + 1) * dh].float().T  # [S, d_model]
        M[h] = contrib.norm(dim=-1)
    score = (a * M).sum(0)  # [S]
    mass = a.mean(0)        # mean over heads of attention prob  [S]
    total = score.sum().item() + 1e-9
    k = min(topk, t + 1)
    top = torch.topk(score, k)
    return {
        "top": [{"s": int(s), "frac": float(sc / total), "mass": float(mass[s])} for sc, s in zip(top.values.tolist(), top.indices.tolist())],
        "sink_frac": float(score[0] / total),
        "self_frac": float(score[t] / total),
        "sink_mass": float(mass[0]),
        "self_mass": float(mass[t]),
    }


@torch.no_grad()
def run_tail(model, n: int, hidden: torch.Tensor, kwargs: dict, positions: list[int]):
    """Run layers n+1.. on `hidden` [B, S, d] (the residual leaving block n), return final logprobs at positions [B, P, V]."""
    blocks = model.model.layers
    kw = dict(kwargs)
    args = kw.pop("_args", ())
    kw.pop("past_key_values", None); kw["use_cache"] = False
    h = hidden
    for blk in blocks[n + 1:]:
        h = blk(h, *args, **kw)
        if isinstance(h, tuple):
            h = h[0]
    h = model.model.norm(h[:, positions, :])
    logits = model.lm_head(h).float()
    return F.log_softmax(logits, dim=-1)


def effect_summary(true_lp: torch.Tensor, abl_lp: torch.Tensor, k: int = 12):
    """true_lp, abl_lp: [V]. Returns tokens whose final log-prob the update raised / lowered most, plus KL."""
    diff = true_lp - abl_lp  # >0: update promoted this token in the final output
    p_true = true_lp.exp()
    # weight by sqrt(prob) so we surface tokens that matter, not 1e-9 -> 1e-8 noise
    w = diff * (p_true + abl_lp.exp()).sqrt()
    top_up = torch.topk(w, k)
    top_dn = torch.topk(-w, k)
    kl = float((p_true * (true_lp - abl_lp)).sum())
    return {
        "up": [{"id": int(i), "dlp": float(diff[i]), "p": float(p_true[i])} for i in top_up.indices.tolist()],
        "down": [{"id": int(i), "dlp": float(diff[i]), "p": float(p_true[i]), "p_abl": float(abl_lp[i].exp())} for i in top_dn.indices.tolist()],
        "kl": kl,
        "top1_true": int(true_lp.argmax()), "top1_abl": int(abl_lp.argmax()),
        "p_top1_true": float(p_true.max()), "p_top1_true_under_abl": float(abl_lp[true_lp.argmax()].exp()),
    }


def iter_docs(dataset: str, config: str | None, split: str, seed: int, num_shards: int = 64, skip: int = 0):
    """Stream documents from ONE shard of the dataset, in file order.

    Do NOT use datasets' streaming .shuffle() here: with a buffer it prefetches many multi-GB
    parquet files concurrently and used >8GB RAM on fineweb (it hung this VM once). Consecutive
    fineweb documents are already an unordered web sample, so file order is fine for our purposes.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True)
    rng = random.Random(seed)
    num_shards = min(num_shards, ds.num_shards)
    ds = ds.shard(num_shards=num_shards, index=rng.randrange(num_shards))
    if skip:
        ds = ds.skip(skip)
    for i, ex in enumerate(ds):
        yield i, ex["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layers", default="8,14,20,26,32")
    ap.add_argument("--n-docs", type=int, default=100)
    ap.add_argument("--positions-per-doc", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-len", type=int, default=64)
    ap.add_argument("--min-pos", type=int, default=16)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--dataset-config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--skip-docs", type=int, default=0, help="skip this many docs at the start of the shard")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--wandb", default="")
    ap.add_argument("--save-unembed", action="store_true", help="save lm_head + final norm weights for offline lens")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(args.seed); torch.manual_seed(args.seed)
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    log("loading", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="eager").to(args.device).eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    assert max(layers) < L
    log(f"layers={L} d={cfg.hidden_size} H={cfg.num_attention_heads} KV={cfg.num_key_value_heads}")

    if args.save_unembed:
        torch.save({"W_U": model.lm_head.weight.detach().to(torch.float16).cpu(),
                    "norm_w": model.model.norm.weight.detach().float().cpu(),
                    "eps": cfg.rms_norm_eps, "model": args.model}, out / "unembed.pt")
        log("saved unembed.pt")

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, job_type="collect", config=vars(args))

    cap = Capture()
    handles = install_hooks(model, layers, cap)

    docs_f = open(out / "docs.jsonl", "a")
    meta_f = open(out / "records.jsonl", "a")
    shard: dict[str, list] = {"X": [], "d": [], "d_attn": [], "d_mlp": []}
    shard_ids: list[str] = []
    shard_idx = len(list(out.glob("vec_*.npz")))

    def flush():
        nonlocal shard_idx, shard, shard_ids
        if not shard_ids:
            return
        np.savez(out / f"vec_{shard_idx:04d}.npz", ids=np.array(shard_ids),
                 **{k: np.stack(v).astype(np.float16) for k, v in shard.items()})
        log(f"wrote shard {shard_idx} ({len(shard_ids)} records)")
        shard_idx += 1
        shard = {k: [] for k in shard}; shard_ids = []

    n_rec = 0; n_doc = 0; t0 = time.time()
    sum_d = {n: torch.zeros(cfg.hidden_size, dtype=torch.float64) for n in layers}
    for doc_i, text in iter_docs(args.dataset, args.dataset_config, args.split, args.seed, skip=args.skip_docs):
        if n_doc >= args.n_docs:
            break
        ids_full = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if len(ids_full) < args.min_len:
            continue
        # random truncation window like the NLA paper
        if len(ids_full) > args.max_len:
            start = random.randint(0, len(ids_full) - args.max_len)
            ids = ids_full[start:start + args.max_len]
        else:
            start = 0; ids = ids_full
        S = len(ids)
        ids_d = ids.unsqueeze(0).to(args.device)
        with torch.no_grad():
            outp = model(ids_d, output_hidden_states=True, use_cache=False)
        hs = outp.hidden_states  # L+1 x [1,S,d]; hs[n] is the input to block n; hs[L] is post-final-norm? (see check below)
        true_lp_all = F.log_softmax(outp.logits[0].float(), dim=-1)  # [S, V]
        positions = sorted(random.sample(range(args.min_pos, S), min(args.positions_per_doc, S - args.min_pos)))
        doc_id = f"d{doc_i}"
        docs_f.write(json.dumps({"doc_id": doc_id, "ids": ids.tolist(), "start": start, "src_index": doc_i}) + "\n")

        for n in layers:
            X_all = hs[n][0].float()
            d_attn_all = cap.attn_out[n].float(); d_mlp_all = cap.mlp_out[n].float()
            # residual leaving block n, computed from hooks (hs[n+1] may equal it, but for the last layer HF applies the final norm)
            Y_all = X_all + d_attn_all + d_mlp_all
            if n + 1 < L:
                err = (hs[n + 1][0].float() - Y_all).abs().max().item()
                if err > 1e-2 * Y_all.abs().max().item():
                    log(f"WARNING layer {n}: hs[n+1] != X+d_attn+d_mlp (max abs err {err:.3g})")
            # ablation batch: for each position, 3 variants (remove all / remove attn / remove mlp)
            P = len(positions)
            hid = Y_all.to(dtype).unsqueeze(0).repeat(3 * P, 1, 1)
            for j, t in enumerate(positions):
                hid[3 * j + 0, t] = X_all[t].to(dtype)
                hid[3 * j + 1, t] = (X_all[t] + d_mlp_all[t]).to(dtype)
                hid[3 * j + 2, t] = (X_all[t] + d_attn_all[t]).to(dtype)
            abl_lp = run_tail(model, n, hid, cap.next_kwargs.get(n, {}), positions)  # [3P, P, V]
            for j, t in enumerate(positions):
                X = X_all[t]; da = d_attn_all[t]; dm = d_mlp_all[t]; d = da + dm
                rec_id = f"{doc_id}_t{t}_L{n}"
                true_lp = true_lp_all[t]
                eff = {name: effect_summary(true_lp, abl_lp[3 * j + v, j]) for v, name in enumerate(["all", "attn", "mlp"])}
                src = attention_sources(model, n, t, cap.attn_w[n], cap.v[n])
                nX, nd, na, nm = X.norm().item(), d.norm().item(), da.norm().item(), dm.norm().item()
                rec = {
                    "id": rec_id, "doc_id": doc_id, "t": t, "layer": n, "n_layers": L,
                    "norm_X": nX, "norm_d": nd, "norm_d_attn": na, "norm_d_mlp": nm,
                    "cos_d_X": float(F.cosine_similarity(d, X, dim=0)),
                    "cos_attn_mlp": float(F.cosine_similarity(da, dm, dim=0)),
                    "cos_Y_X": float(F.cosine_similarity(X + d, X, dim=0)),
                    "true_top": [{"id": int(i), "p": float(true_lp[i].exp())} for i in torch.topk(true_lp, 8).indices.tolist()],
                    "effect": eff, "attn_sources": src,
                }
                meta_f.write(json.dumps(rec) + "\n")
                shard["X"].append(X.cpu().numpy()); shard["d"].append(d.cpu().numpy())
                shard["d_attn"].append(da.cpu().numpy()); shard["d_mlp"].append(dm.cpu().numpy())
                shard_ids.append(rec_id)
                sum_d[n] += d.double().cpu()
                n_rec += 1
            if len(shard_ids) >= args.shard_size:
                flush()
        n_doc += 1
        if n_doc % 10 == 0:
            el = time.time() - t0
            log(f"docs={n_doc} records={n_rec} {el/n_doc:.2f}s/doc")
            if wb:
                wb.log({"docs": n_doc, "records": n_rec, "sec_per_doc": el / n_doc})
        meta_f.flush(); docs_f.flush()
    flush()
    torch.save({n: (sum_d[n] / max(1, n_rec // len(layers))).float() for n in layers}, out / "mean_d.pt")
    log(f"done: docs={n_doc} records={n_rec} in {time.time()-t0:.0f}s")
    if wb:
        art = __import__("wandb").Artifact(f"raw-{args.model.split('/')[-1]}", type="raw-evidence")
        art.add_dir(str(out))
        wb.log_artifact(art); wb.finish()
    meta_f.close(); docs_f.close()
    os._exit(0)  # skip interpreter finalisation: datasets' streaming threads crash on shutdown


if __name__ == "__main__":
    main()
