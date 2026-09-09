#!/usr/bin/env bash
# Stage-1 warm-start feasibility run on one GPU pod. Assumes the repo is checked out, data/ populated from the HF
# dataset, and WANDB_API_KEY / HF_TOKEN in the environment.  Usage: bash scripts/stage1_pod.sh [descriptions.jsonl]
set -euo pipefail
DESC=${1:-data/descriptions_v34.jsonl}
LAYERS=${LAYERS:-12,18,24}
MODEL=${MODEL:-Qwen/Qwen3-8B}
EP=${EPOCHS:-2}
echo "== AR (true pairs)"
python -m delta_nla.train_ar --model $MODEL --ar-layers 24 --descriptions $DESC --layers $LAYERS --out runs/ar --epochs $EP --bs 16 --wandb delta-nla
echo "== AR (shuffled-text control, 1 epoch)"
python -m delta_nla.train_ar --model $MODEL --ar-layers 24 --descriptions $DESC --layers $LAYERS --out runs/ar_shuf --epochs 1 --bs 16 --shuffle-texts --wandb delta-nla
echo "== AV"
python -m delta_nla.train_av --model $MODEL --descriptions $DESC --layers $LAYERS --out runs/av --epochs $EP --bs 8 --wandb delta-nla
echo "== end-to-end"
python -m delta_nla.eval_e2e --model $MODEL --ar-dir runs/ar --av-dir runs/av --descriptions $DESC --layers $LAYERS --out runs/e2e.jsonl --wandb delta-nla
echo "== upload adapters to HF"
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for d, name in (("runs/ar", "syvb/delta-nla-qwen3-8b-ar-warmstart"), ("runs/av", "syvb/delta-nla-qwen3-8b-av-warmstart")):
    api.create_repo(name, exist_ok=True); api.upload_folder(folder_path=d, repo_id=name, commit_message="stage-1 warm-start adapter")
    print("uploaded", name)
PY
echo STAGE1_DONE
