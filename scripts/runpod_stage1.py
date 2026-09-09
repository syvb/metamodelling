"""Launch the stage-1 warm-start feasibility run on a RunPod GPU.
  python scripts/runpod_stage1.py [--gpu "NVIDIA A100 80GB PCIe"] [--desc descriptions_v34.jsonl]
The pod clones the repo, pulls data from the HF dataset, runs scripts/stage1_pod.sh, keeps running for inspection (terminate with runpod_launch.py terminate <id>).
"""
import argparse, json, os, sys
import runpod
sys.path.insert(0, os.path.dirname(__file__))
from runpod_launch import IMAGE, REPO, WANDB_KEY  # noqa: E402
runpod.api_key = open(os.path.expanduser("~/.runpod_key")).read().strip()
HF_TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="NVIDIA A100 80GB PCIe"); ap.add_argument("--cloud", default="SECURE")
ap.add_argument("--desc", default="descriptions_v34.jsonl"); ap.add_argument("--layers", default="12,18,24"); ap.add_argument("--epochs", default="2")
a = ap.parse_args()
cmd = (
    "/start.sh >/dev/null 2>&1 & mkdir -p /workspace && exec > >(tee -a /workspace/boot.log) 2>&1; set -x; cd /workspace && rm -rf metamodelling && "
    f"git clone -q {REPO} && cd metamodelling && pip uninstall -y -q torchvision torchaudio; pip install -q -r requirements-pod.txt peft 2>&1 | tail -2 && "
    "python scripts/pull_hf_data.py && "
    f"LAYERS={a.layers} EPOCHS={a.epochs} bash scripts/stage1_pod.sh data/raw/{a.desc}; echo FINISHED; sleep infinity"
)
fallbacks = [(a.gpu, a.cloud), ("NVIDIA A100 80GB PCIe", "COMMUNITY"), ("NVIDIA H100 PCIe", "SECURE"), ("NVIDIA H100 PCIe", "COMMUNITY"), ("NVIDIA A100-SXM4-80GB", "SECURE")]
pod = None
for gpu, cloud in fallbacks:
    try:
        pod = runpod.create_pod(name="delta-nla-stage1", image_name=IMAGE, gpu_type_id=gpu, cloud_type=cloud, gpu_count=1,
                                container_disk_in_gb=100, volume_in_gb=0, min_memory_in_gb=48, min_vcpu_count=8, ports="22/tcp",
                                docker_args=f"bash -lc '{cmd}'",
                                env={"WANDB_API_KEY": WANDB_KEY, "WANDB_PROJECT": "delta-nla", "HF_TOKEN": HF_TOKEN, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        print("launched on", gpu, cloud); break
    except runpod.error.QueryError as e:
        if "no longer any instances" not in str(e) and "not available" not in str(e).lower():
            sys.exit(f"launch error: {str(e)[:300]}")
        print("unavailable:", gpu, cloud)
if pod is None:
    sys.exit("no GPU available")
print("POD_ID", pod["id"])
