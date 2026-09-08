"""Download the pod's wandb artifacts (raw vectors, unembedding, evidence) into a local directory."""
import argparse, os, shutil
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="octahedral-systems/delta-nla")
ap.add_argument("--tag", default="Qwen3-8B")
ap.add_argument("--out", default="data/raw")
ap.add_argument("--what", default="raw,evidence", help="comma list of raw,evidence")
a = ap.parse_args()
api = wandb.Api()
os.makedirs(a.out, exist_ok=True)
for what in a.what.split(","):
    name = {"raw": f"raw-{a.tag}", "evidence": f"evidence-{a.tag}"}[what]
    art = api.artifact(f"{a.project}/{name}:latest")
    print(what, art.name, f"{art.size/1e9:.2f} GB", flush=True)
    d = art.download(root=a.out + "/_dl_" + what)
    for root, _, files in os.walk(d):
        for f in files:
            src = os.path.join(root, f); dst = os.path.join(a.out, f)
            shutil.move(src, dst); print("  ->", dst, f"{os.path.getsize(dst)/1e6:.1f} MB")
    shutil.rmtree(d, ignore_errors=True)
print("done; local copy in", a.out)
