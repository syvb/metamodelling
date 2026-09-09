"""Reconstructor (AR) SFT: description -> d.  Truncated Qwen3 + LoRA + linear head on the final-token residual.

  python -m delta_nla.train_ar --model Qwen/Qwen3-8B --ar-layers 24 --descriptions data/descriptions_v34.jsonl --layers 12,18,24
Controls: --shuffle-texts trains on mismatched (text, d) pairs; FVE should then be ~0.
"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from .data import load_pairs, split_by_doc, TargetNorm, fve_norm, cosines


def build_prompt(layer: int, n_layers: int, text: str) -> str:
    return f"Change at layer {layer} of {n_layers} of a language model, described in words:\n{text}\nThe update vector is:"


class AR(nn.Module):
    def __init__(self, base, d_target: int, lora_r: int, head: str = "linear", pool: str = "last"):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.lm = get_peft_model(base, cfg) if lora_r > 0 else base
        d_model = base.config.hidden_size; self.pool = pool
        self.feat_norm = nn.LayerNorm(d_model, elementwise_affine=False)  # scale-free features (post-norm states carry large gains)
        if head == "linear":
            self.head = nn.Linear(d_model, d_target)
        else:
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_target))
        last = self.head if isinstance(self.head, nn.Linear) else self.head[-1]
        nn.init.zeros_(last.bias); nn.init.zeros_(last.weight)  # start at the mean predictor (FVE 0), learn upward
    def features(self, input_ids, attention_mask):
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, logits_to_keep=1)
        h = out.hidden_states[-1]  # after final norm of the (truncated) model
        if self.pool == "mean":
            am = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * am).sum(1) / am.sum(1)
        last = attention_mask.sum(1) - 1
        return h[torch.arange(h.shape[0], device=h.device), last]
    def forward(self, input_ids, attention_mask):
        pooled = self.features(input_ids, attention_mask)
        with torch.autocast(device_type="cuda", enabled=False):
            return self.head(self.feat_norm(pooled.float()))


def truncate(model, k: int):
    model.model.layers = model.model.layers[:k]
    model.config.num_hidden_layers = k
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--ar-layers", type=int, default=24, help="keep the first k blocks")
    ap.add_argument("--evidence", default="data/raw/evidence.jsonl"); ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--descriptions", default="data/descriptions_v34.jsonl"); ap.add_argument("--templates", action="store_true")
    ap.add_argument("--layers", default="12,18,24"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/ar"); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head-lr", type=float, default=1e-3); ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"]); ap.add_argument("--pool", default="last", choices=["last", "mean"])
    ap.add_argument("--target", default="unit", choices=["unit", "rms"], help="unit: direction of centred d (paper convention); rms: keeps relative norm")
    ap.add_argument("--cos-weight", type=float, default=1.0, help="loss = mse + cos_weight * (1 - cos)"); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=192); ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--shuffle-texts", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--wandb", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "right"
    layers = [int(x) for x in args.layers.split(",")]
    pairs = load_pairs(args.evidence, args.descriptions, args.raw, layers, use_templates=args.templates, limit=args.limit)
    train, val = split_by_doc(pairs)
    if args.shuffle_texts:
        texts = [p.text for p in train]; random.shuffle(texts)
        for p, t in zip(train, texts): p.text = t
    norm = TargetNorm(train, mode=args.target)
    n_layers_total = json.loads(open(args.evidence).readline())["n_layers"]
    print(f"train {len(train)} val {len(val)} layers {layers} d_target {train[0].d.shape[0]}", flush=True)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    base = truncate(base, args.ar_layers).to(args.device)
    model = AR(base, train[0].d.shape[0], args.lora_r, head=args.head, pool=args.pool).to(args.device)
    model.head.to(torch.float32)
    params = [{"params": [p for n, p in model.lm.named_parameters() if p.requires_grad], "lr": args.lr},
              {"params": model.head.parameters(), "lr": args.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=args.wd)
    steps_total = int(len(train) / args.bs * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / max(1, steps_total)))
    wb = None
    if args.wandb:
        import wandb; wb = wandb.init(project=args.wandb, job_type="train_ar", config=vars(args))

    def batchify(ps):
        enc = tok([build_prompt(p.layer, n_layers_total, p.text) for p in ps], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        y = torch.from_numpy(np.stack([norm.encode(p.d, p.layer) for p in ps]))
        return enc.input_ids.to(args.device), enc.attention_mask.to(args.device), y.to(args.device)

    @torch.no_grad()
    def evaluate():
        model.eval(); preds = {}
        for i in range(0, len(val), args.bs):
            ps = val[i:i + args.bs]; ids, am, _ = batchify(ps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                yh = model(ids, am).float().cpu().numpy()
            for p, y in zip(ps, yh): preds[p.id] = y
        res = {}
        for n in layers:
            vs = [p for p in val if p.layer == n]
            if not vs: continue
            P = np.stack([preds[p.id] for p in vs]); T = np.stack([norm.target(p.d, p.layer) for p in vs])
            res[f"fve_L{n}"] = fve_norm(P, T); res[f"cos_L{n}"] = cosines(P, T)
            res[f"pred_std_L{n}"] = float(P.std(0).mean() / (T.std(0).mean() + 1e-8))  # ~0 means collapsed to a constant
        res["fve_all"] = float(np.mean([v for k, v in res.items() if k.startswith("fve_")]))
        model.train(); return res

    step = 0; t0 = time.time(); model.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ep = 0
    while step < steps_total:
        random.shuffle(train); ep += 1
        for i in range(0, len(train) - args.bs + 1, args.bs):
            ids, am, y = batchify(train[i:i + args.bs])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                yh = model(ids, am)
            yf, y = yh.float(), y.float()
            mse = ((yf - y) ** 2).sum(-1).mean()
            cos = torch.nn.functional.cosine_similarity(yf, y, dim=-1).mean()
            loss = mse + args.cos_weight * (1 - cos)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                print(f"step {step}/{steps_total} loss {loss.item():.4f} mse {mse.item():.4f} cos {cos.item():.4f} {time.time()-t0:.0f}s", flush=True)
                if wb: wb.log({"loss": loss.item(), "mse": mse.item(), "cos": cos.item(), "step": step})
            if step % args.eval_every == 0 or step == steps_total:
                res = evaluate(); print("eval", json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
                if wb: wb.log({**res, "step": step})
            if step >= steps_total: break
    res = evaluate(); print("final", json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
    if args.lora_r > 0: model.lm.save_pretrained(args.out)
    torch.save({"head": model.head.state_dict(), "norm": norm.state_dict(), "args": vars(args), "final": res, "val_ids": [p.id for p in val]}, f"{args.out}/ar_extra.pt")
    if wb: wb.log({**{"final_" + k: v for k, v in res.items()}}); wb.finish()


if __name__ == "__main__":
    main()
