"""End-to-end warm-start evaluation: AV generates descriptions for held-out (X, d); AR reconstructs d from them.

Reports FVE per layer for (a) AV-generated descriptions, (b) the reference LLM descriptions, (c) shuffled text control.
"""
from __future__ import annotations
import argparse, json, random
import numpy as np, torch
from .data import load_pairs, split_by_doc, TargetNorm, fve_norm, cosines
from .train_ar import AR, build_prompt as ar_prompt, truncate
from .train_av import AV, build_prompt as av_prompt, MARK_X, MARK_D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--ar-dir", default="runs/ar"); ap.add_argument("--av-dir", default="runs/av")
    ap.add_argument("--evidence", default="data/raw/evidence.jsonl"); ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--descriptions", default="data/descriptions_v34.jsonl"); ap.add_argument("--templates", action="store_true")
    ap.add_argument("--layers", default="12,18,24"); ap.add_argument("--n-val", type=int, default=0, help="0 = all val records"); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--out", default="runs/e2e.jsonl"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--wandb", default="")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(args.model)
    layers = [int(x) for x in args.layers.split(",")]
    pairs = load_pairs(args.evidence, args.descriptions, args.raw, layers, use_templates=args.templates)
    train, val = split_by_doc(pairs); random.Random(0).shuffle(val)
    if args.n_val: val = val[: args.n_val]
    n_layers_total = json.loads(open(args.evidence).readline())["n_layers"]
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    # --- AV
    av_extra = torch.load(f"{args.av_dir}/av_extra.pt", weights_only=False)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    av = AV.__new__(AV); torch.nn.Module.__init__(av)
    av.lm = PeftModel.from_pretrained(base, args.av_dir).to(args.device)
    d_act = val[0].X.shape[0]; d_model = base.config.hidden_size
    av.proj_x = torch.nn.Linear(d_act, d_model).to(args.device); av.proj_d = torch.nn.Linear(d_act, d_model).to(args.device)
    av.proj_x.load_state_dict(av_extra["proj_x"]); av.proj_d.load_state_dict(av_extra["proj_d"])
    av_norm = TargetNorm.from_state_dict(av_extra["norm"]); alpha = {int(k): v for k, v in av_extra["alpha"].items()}
    mx, md = tok.convert_tokens_to_ids(MARK_X), tok.convert_tokens_to_ids(MARK_D)
    gen = {}
    tok.padding_side = "left"
    with torch.no_grad():
        for i in range(0, len(val), args.bs):
            ps = val[i:i + args.bs]
            xs = np.stack([p.X / (np.linalg.norm(p.X) + 1e-6) * alpha[p.layer] for p in ps])
            dn = np.stack([av_norm.encode(p.d, p.layer) * alpha[p.layer] for p in ps]).astype(np.float32)
            enc = tok([av_prompt(p.layer, n_layers_total) for p in ps], return_tensors="pt", padding=True).to(args.device)
            emb = av.embed(enc.input_ids, torch.from_numpy(xs).to(args.device), torch.from_numpy(dn).to(args.device), mx, md)
            out = av.lm.generate(inputs_embeds=emb, attention_mask=enc.attention_mask, max_new_tokens=args.max_new, do_sample=args.temperature > 0,
                                 temperature=args.temperature if args.temperature > 0 else None, top_k=0, top_p=1.0, pad_token_id=tok.pad_token_id)
            for p, o in zip(ps, out): gen[p.id] = tok.decode(o, skip_special_tokens=True).strip()
            print(f"generated {min(i + args.bs, len(val))}/{len(val)}", flush=True)
    del av, base; torch.cuda.empty_cache() if args.device == "cuda" else None
    # --- AR
    tok.padding_side = "right"
    ar_extra = torch.load(f"{args.ar_dir}/ar_extra.pt", weights_only=False)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype); base = truncate(base, ar_extra["args"]["ar_layers"]).to(args.device)
    ar = AR.__new__(AR); torch.nn.Module.__init__(ar)
    ar_args = ar_extra["args"]; ar.pool = ar_args.get("pool", "last"); ar.feat_norm = torch.nn.LayerNorm(d_model, elementwise_affine=False).to(args.device)
    ar.lm = PeftModel.from_pretrained(base, args.ar_dir).to(args.device) if ar_args.get("lora_r", 32) > 0 else base
    if ar_args.get("head", "mlp") == "linear":
        ar.head = torch.nn.Linear(d_model, d_act).to(args.device)
    else:
        ar.head = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.GELU(), torch.nn.Linear(d_model, d_act)).to(args.device)
    ar.head.load_state_dict(ar_extra["head"]); ar_norm = TargetNorm.from_state_dict(ar_extra["norm"])
    if "val_ids" in ar_extra:
        assert set(ar_extra["val_ids"]) >= {p.id for p in val}, "e2e val set is not a subset of the AR's val set (split mismatch)"
    def reconstruct(texts):
        preds = {}
        with torch.no_grad():
            for i in range(0, len(val), args.bs):
                ps = val[i:i + args.bs]
                enc = tok([ar_prompt(p.layer, n_layers_total, texts[p.id]) for p in ps], return_tensors="pt", padding=True, truncation=True, max_length=ar_extra["args"]["max_len"]).to(args.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    yh = ar(enc.input_ids, enc.attention_mask).float().cpu().numpy()
                for p, y in zip(ps, yh): preds[p.id] = y
        return preds
    ref = {p.id: p.text for p in val}
    shuf_ids = [p.id for p in val]; random.Random(1).shuffle(shuf_ids); shuf = {p.id: ref[s] for p, s in zip(val, shuf_ids)}
    results = {}
    for name, texts in (("av_generated", gen), ("reference_llm", ref), ("shuffled", shuf)):
        preds = reconstruct(texts)
        for n in layers:
            vs = [p for p in val if p.layer == n]
            if not vs: continue
            P = np.stack([preds[p.id] for p in vs]); T = np.stack([ar_norm.target(p.d, p.layer) for p in vs])
            results[f"{name}/fve_L{n}"] = fve_norm(P, T); results[f"{name}/cos_L{n}"] = cosines(P, T)
        results[f"{name}/fve_all"] = float(np.mean([v for k, v in results.items() if k.startswith(f"{name}/fve_L")]))
        results[f"{name}/cos_all"] = float(np.mean([v for k, v in results.items() if k.startswith(f"{name}/cos_L")]))
    print(json.dumps({k: round(v, 4) for k, v in results.items()}, indent=1))
    with open(args.out, "w") as f:
        for p in val: f.write(json.dumps({"id": p.id, "layer": p.layer, "reference": ref[p.id], "generated": gen[p.id]}, ensure_ascii=False) + "\n")
    if args.wandb:
        import wandb; wb = wandb.init(project=args.wandb, job_type="eval_e2e", config=vars(args)); wb.log(results)
        wb.log({"samples": wandb.Table(columns=["id", "layer", "reference", "generated"], data=[[p.id, p.layer, ref[p.id], gen[p.id]] for p in val[:200]])}); wb.finish()


if __name__ == "__main__":
    main()
