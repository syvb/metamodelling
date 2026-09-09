"""Reconstructor (AR) SFT: description -> d.  Truncated Qwen3 + LoRA + linear head on the final-token residual.

  python -m delta_nla.train_ar --model Qwen/Qwen3-8B --ar-layers 24 --descriptions data/descriptions_v34.jsonl --layers 12,18,24
Controls: --shuffle-texts trains on mismatched (text, d) pairs; FVE should then be ~0.
"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from .data import load_pairs, split_by_doc, TargetNorm, fve


def build_prompt(layer: int, n_layers: int, text: str) -> str:
    return f"Change at layer {layer} of {n_layers} of a language model, described in words:\n{text}\nThe update vector is:"


class AR(nn.Module):
    def __init__(self, base, d_target: int, lora_r: int):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.lm = get_peft_model(base, cfg)
        d_model = base.config.hidden_size
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_target))
        nn.init.zeros_(self.head[-1].bias)
    def forward(self, input_ids, attention_mask):
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1]  # after final norm of the (truncated) model
        last = attention_mask.sum(1) - 1
        pooled = h[torch.arange(h.shape[0], device=h.device), last]
        return self.head(pooled.float())


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
    norm = TargetNorm(train)
    n_layers_total = json.loads(open(args.evidence).readline())["n_layers"]
    print(f"train {len(train)} val {len(val)} layers {layers} d_target {train[0].d.shape[0]}", flush=True)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    base = truncate(base, args.ar_layers).to(args.device)
    model = AR(base, train[0].d.shape[0], args.lora_r).to(args.device)
    model.head.to(torch.float32)
    params = [{"params": [p for n, p in model.lm.named_parameters() if p.requires_grad], "lr": args.lr},
              {"params": model.head.parameters(), "lr": args.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
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
        model.eval(); preds = {}; 
        for i in range(0, len(val), args.bs):
            ps = val[i:i + args.bs]; ids, am, _ = batchify(ps)
            yh = model(ids, am).float().cpu().numpy()
            for p, y in zip(ps, yh): preds[p.id] = norm.decode(y, p.layer)
        res = {}
        for n in layers:
            vs = [p for p in val if p.layer == n]
            if not vs: continue
            P = np.stack([preds[p.id] for p in vs]); T = np.stack([p.d for p in vs])
            res[f"fve_L{n}"] = fve(P, T, norm.mean[n])
            res[f"cos_L{n}"] = float(np.mean(np.sum((P - norm.mean[n]) * (T - norm.mean[n]), 1) / (np.linalg.norm(P - norm.mean[n], axis=1) * np.linalg.norm(T - norm.mean[n], axis=1) + 1e-8)))
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
            loss = ((yh.float() - y.float()) ** 2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                print(f"step {step}/{steps_total} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
                if wb: wb.log({"loss": loss.item(), "step": step})
            if step % args.eval_every == 0 or step == steps_total:
                res = evaluate(); print("eval", json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
                if wb: wb.log({**res, "step": step})
            if step >= steps_total: break
    res = evaluate(); print("final", json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
    model.lm.save_pretrained(args.out); torch.save({"head": model.head.state_dict(), "norm": norm.state_dict(), "args": vars(args), "final": res}, f"{args.out}/ar_extra.pt")
    if wb: wb.log({**{"final_" + k: v for k, v in res.items()}}); wb.finish()


if __name__ == "__main__":
    main()
