"""Launch / inspect / terminate the RunPod collection job.

  python scripts/runpod_launch.py launch [--gpu "NVIDIA A40"] [--n-docs 1500] ...
  python scripts/runpod_launch.py status
  python scripts/runpod_launch.py terminate <pod_id>
"""
import argparse, json, os, sys, time
import runpod

runpod.api_key = open(os.path.expanduser("~/.runpod_key")).read().strip()
WANDB_KEY = open(os.path.expanduser("~/.wandb_key")).read().strip()
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
REPO = "https://github.com/syvb/metamodelling.git"
FALLBACK_GPUS = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 4090", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40S"]


def launch(a):
    model_tag = a.model.split("/")[-1]
    cmd = (
        "set -x; cd /workspace && rm -rf metamodelling && git clone -q %s && cd metamodelling && "
        "pip install -q -r requirements-pod.txt 2>&1 | tail -2 && nvidia-smi --query-gpu=name,memory.total --format=csv && "
        "timeout %dh python -m delta_nla.collect --model %s --layers %s --n-docs %d --positions-per-doc %d --max-len %d "
        "--out data/raw --save-unembed --seed %d --wandb delta-nla --shard-size 2500 && "
        "python -m delta_nla.build_evidence --raw data/raw --out data/raw/evidence.jsonl --model %s --device cuda && "
        "python scripts/pod_finish.py data/raw; echo FINISHED; sleep infinity"
    ) % (REPO, a.max_hours, a.model, a.layers, a.n_docs, a.positions_per_doc, a.max_len, a.seed, a.model)
    attempts = [(a.gpu, a.cloud)] + [(g, c) for g in FALLBACK_GPUS for c in ("COMMUNITY", "SECURE") if (g, c) != (a.gpu, a.cloud)]
    pod = None
    for gpu, cloud in attempts:
        try:
            pod = runpod.create_pod(
                name=f"delta-nla-{model_tag}", image_name=IMAGE, gpu_type_id=gpu, cloud_type=cloud, gpu_count=1,
                container_disk_in_gb=80, volume_in_gb=0, min_memory_in_gb=24, min_vcpu_count=4, ports="22/tcp",
                docker_args=f"bash -lc '{cmd}'",
                env={"WANDB_API_KEY": WANDB_KEY, "WANDB_PROJECT": "delta-nla", "RUNPOD_API_KEY": runpod.api_key,
                     "MODEL_TAG": model_tag, "HF_HUB_ENABLE_HF_TRANSFER": "0", "PYTHONUNBUFFERED": "1", "KEEP_POD": "1" if a.keep else "0"},
            )
            print("launched on", gpu, cloud); break
        except runpod.error.QueryError as e:
            print("unavailable:", gpu, cloud, "-", str(e)[:80])
    if pod is None:
        sys.exit("no GPU available")
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "machineId")}, indent=1))
    print("POD_ID", pod["id"])


def status(a):
    for p in runpod.get_pods():
        print(p["id"], p["name"], p["desiredStatus"], p.get("machine", {}).get("gpuDisplayName"), f"${p.get('costPerHr')}/h", p.get("runtime") and p["runtime"].get("uptimeInSeconds"))


def terminate(a):
    runpod.terminate_pod(a.pod_id); print("terminated", a.pod_id)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("launch")
    l.add_argument("--gpu", default="NVIDIA A40"); l.add_argument("--cloud", default="COMMUNITY")
    l.add_argument("--model", default="Qwen/Qwen3-8B"); l.add_argument("--layers", default="6,12,18,24,30")
    l.add_argument("--n-docs", type=int, default=1500); l.add_argument("--positions-per-doc", type=int, default=3)
    l.add_argument("--max-len", type=int, default=512); l.add_argument("--seed", type=int, default=1)
    l.add_argument("--max-hours", type=int, default=4); l.add_argument("--keep", action="store_true")
    sub.add_parser("status")
    t = sub.add_parser("terminate"); t.add_argument("pod_id")
    a = ap.parse_args()
    {"launch": launch, "status": status, "terminate": terminate}[a.cmd](a)
