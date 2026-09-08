"""Run on the pod after collection: upload evidence.jsonl as a wandb artifact, then terminate the pod."""
import os, sys, subprocess, time
import wandb
raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "delta-nla"), job_type="evidence")
art = wandb.Artifact("evidence-" + os.environ.get("MODEL_TAG", "model"), type="evidence")
import glob
files = ["evidence.jsonl", "records.jsonl", "docs.jsonl", "mean_d.pt"]
if os.environ.get("UPLOAD_VECTORS") == "1":
    files += [os.path.basename(x) for x in glob.glob(os.path.join(raw, "vec_*.npz"))]
for f in files:
    p = os.path.join(raw, f)
    if os.path.exists(p):
        art.add_file(p)
run.log_artifact(art)
run.finish()
print("uploaded evidence artifact", flush=True)
pod_id = os.environ.get("RUNPOD_POD_ID")
if pod_id and os.environ.get("RUNPOD_API_KEY") and os.environ.get("KEEP_POD") != "1":
    import runpod
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    time.sleep(5)
    runpod.terminate_pod(pod_id)
    print("terminated pod", pod_id)
