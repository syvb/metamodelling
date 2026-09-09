"""Verbalizer (AV) SFT: (X, d) injected as two token embeddings -> description.  Qwen3 + LoRA + two affine injection maps.

Eval: validation loss with the true activations vs. with activations shuffled within the batch (the gap = information read from the injection).
"""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from .data import load_pairs, split_by_doc, TargetNorm, act_scale

MARK_X, MARK_D = "<|box_start|>", "<|box_end|>"   # existing rare Qwen tokens re-used as injection slots


def build_prompt(layer: int, n_layers: int) -> str:
    return (f"Layer {layer} of {n_layers} of a language model. State entering the layer: {MARK_X}. "
            f"Update made by the layer: {MARK_D}.\nDescribe what changed:\n")


class AV(nn.Module):
    def __init__(self, base, d_act: int, lora_r: int):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.lm = get_peft_model(base, cfg)
        d_model = base.config.hidden_size
        self.proj_x = nn.Linear(d_act, d_model); self.proj_d = nn.Linear(d_act, d_model)
        for lin in (self.proj_x, self.proj_d):   # identity init when shapes match, else small random
            if d_act == d_model: nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)
    def embed(self, input_ids, x_vec, d_vec, mark_x_id, mark_d_id):
        emb = self.lm.get_input_embeddings()(input_ids)          # [B, T, d_model]
        ex = self.proj_x(x_vec.float()).to(emb.dtype); ed = self.proj_d(d_vec.float()).to(emb.dtype)
        emb = emb.clone()
        bx = (input_ids == mark_x_id); bd = (input_ids == mark_d_id)
        emb[bx] = ex[bx.any(1)] if bx.any() else emb[bx]
        emb[bd] = ed[bd.any(1)] if bd.any() else emb[bd]
        return emb
    def forward(self, input_ids, attention_mask, labels, x_vec, d_vec, mark_x_id, mark_d_id):
        emb = self.embed(input_ids, x_vec, d_vec, mark_x_id, mark_d_id)
        return self.lm(inputs_embeds=emb, attention_mask=attention_mask, labels=labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--evidence", default="data/raw/evidence.jsonl"); ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--descriptions", default="data/descriptions_v34.jsonl"); ap.add_argument("--templates", action="store_true")
    ap.add_argument("--layers", default="12,18,24"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/av"); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--proj-lr", type=float, default=1e-3); ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--alpha-q", type=float, default=0.75, help="injection scale = this quantile of activation norms at the layer")
    ap.add_argument("--n-gen", type=int, default=24, help="validation samples to generate at the end")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--wandb", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "right"
    mark_x_id = tok.convert_tokens_to_ids(MARK_X); mark_d_id = tok.convert_tokens_to_ids(MARK_D)
    assert mark_x_id != tok.unk_token_id and mark_d_id != tok.unk_token_id
    layers = [int(x) for x in args.layers.split(",")]
    pairs = load_pairs(args.evidence, args.descriptions, args.raw, layers, use_templates=args.templates, limit=args.limit)
    train, val = split_by_doc(pairs)
    norm = TargetNorm(train); alpha = act_scale(train, args.alpha_q)
    n_layers_total = json.loads(open(args.evidence).readline())["n_layers"]
    print(f"train {len(train)} val {len(val)} layers {layers} alpha {alpha}", flush=True)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    model = AV(base, train[0].X.shape[0], args.lora_r).to(args.device)
    model.proj_x.float(); model.proj_d.float()
    opt = torch.optim.AdamW([{"params": [p for n, p in model.lm.named_parameters() if p.requires_grad], "lr": args.lr},
                             {"params": list(model.proj_x.parameters()) + list(model.proj_d.parameters()), "lr": args.proj_lr}], weight_decay=0.0)
    steps_total = int(len(train) / args.bs * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / max(1, steps_total)))
    wb = None
    if args.wandb:
        import wandb; wb = wandb.init(project=args.wandb, job_type="train_av", config=vars(args))

    def vecs(ps, shuffle=False):
        xs = np.stack([p.X / (np.linalg.norm(p.X) + 1e-6) * alpha[p.layer] for p in ps])
        dn = np.stack([norm.encode(p.d, p.layer) for p in ps]); dn = dn / (np.linalg.norm(dn, axis=1, keepdims=True) + 1e-6) * np.array([alpha[p.layer] for p in ps])[:, None]
        if shuffle:
            perm = np.roll(np.arange(len(ps)), 1); xs, dn = xs[perm], dn[perm]
        return torch.from_numpy(xs).to(args.device), torch.from_numpy(dn).to(args.device)

    def batchify(ps, shuffle=False):
        prompts = [build_prompt(p.layer, n_layers_total) for p in ps]
        full = [pr + p.text + tok.eos_token for pr, p in zip(prompts, ps)]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        labels = enc.input_ids.clone()
        for k, pr in enumerate(prompts):
            n_pr = len(tok(pr).input_ids); labels[k, :n_pr] = -100
        labels[enc.attention_mask == 0] = -100
        x, d = vecs(ps, shuffle)
        return enc.input_ids.to(args.device), enc.attention_mask.to(args.device), labels.to(args.device), x, d

    @torch.no_grad()
    def evaluate():
        model.eval(); tot = {"true": 0.0, "shuf": 0.0}; n = 0
        for i in range(0, len(val), args.bs):
            ps = val[i:i + args.bs]
            if len(ps) < 2: continue
            for key, sh in (("true", False), ("shuf", True)):
                ids, am, lab, x, d = batchify(ps, sh)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    out = model(ids, am, lab, x, d, mark_x_id, mark_d_id)
                tot[key] += out.loss.item() * len(ps)
            n += len(ps)
        model.train()
        return {"val_loss_true": tot["true"] / n, "val_loss_shuf": tot["shuf"] / n, "gap": (tot["shuf"] - tot["true"]) / n}

    @torch.no_grad()
    def generate(ps, max_new=120, temperature=1.0):
        model.eval(); tok.padding_side = "left"
        prompts = [build_prompt(p.layer, n_layers_total) for p in ps]
        enc = tok(prompts, return_tensors="pt", padding=True).to(args.device)
        x, d = vecs(ps)
        emb = model.embed(enc.input_ids, x, d, mark_x_id, mark_d_id)
        out = model.lm.generate(inputs_embeds=emb, attention_mask=enc.attention_mask, max_new_tokens=max_new, do_sample=temperature > 0, temperature=temperature if temperature > 0 else None, pad_token_id=tok.pad_token_id)
        tok.padding_side = "right"; model.train()
        return [tok.decode(o, skip_special_tokens=True).strip() for o in out]

    step = 0; t0 = time.time(); model.train(); Path(args.out).mkdir(parents=True, exist_ok=True)
    while step < steps_total:
        random.shuffle(train)
        for i in range(0, len(train) - args.bs + 1, args.bs):
            ids, am, lab, x, d = batchify(train[i:i + args.bs])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                out = model(ids, am, lab, x, d, mark_x_id, mark_d_id)
            out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                print(f"step {step}/{steps_total} loss {out.loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
                if wb: wb.log({"loss": out.loss.item(), "step": step})
            if step % args.eval_every == 0 or step == steps_total:
                res = evaluate(); print("eval", json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
                if wb: wb.log({**res, "step": step})
            if step >= steps_total: break
    res = evaluate(); print("final", json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
    model.lm.save_pretrained(args.out)
    torch.save({"proj_x": model.proj_x.state_dict(), "proj_d": model.proj_d.state_dict(), "norm": norm.state_dict(), "alpha": alpha, "args": vars(args), "final": res}, f"{args.out}/av_extra.pt")
    samples = []
    vs = val[: args.n_gen]
    for i in range(0, len(vs), args.bs):
        ps = vs[i:i + args.bs]
        for p, g in zip(ps, generate(ps)):
            samples.append({"id": p.id, "layer": p.layer, "target": p.text, "generated": g})
    with open(f"{args.out}/samples.jsonl", "w") as f:
        for s in samples: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    for s in samples[:4]:
        print("---", s["id"]); print("TARGET   :", s["target"][:300]); print("GENERATED:", s["generated"][:300])
    if wb:
        import wandb; wb.log({"samples": wandb.Table(columns=["id", "layer", "target", "generated"], data=[[s["id"], s["layer"], s["target"], s["generated"]] for s in samples])}); wb.finish()


if __name__ == "__main__":
    main()
