#!/usr/bin/env bash
# Follow-up to stage1_pod.sh: retrain the reconstructor with the fixed trainer (unit-norm targets, cosine term,
# zero-init linear head), run its shuffled-text control, then re-run the end-to-end eval against the existing AV.
set -euo pipefail
DESC=${1:-data/raw/descriptions_v34.jsonl}
LAYERS=${LAYERS:-12,18,24}; MODEL=${MODEL:-Qwen/Qwen3-8B}
cd /workspace/metamodelling && git pull -q && git log --oneline | head -1
echo "== AR v2 (linear head, unit targets, 3 epochs)"
python -m delta_nla.train_ar --model $MODEL --ar-layers 24 --descriptions $DESC --layers $LAYERS --out runs/ar_v2 --epochs 3 --bs 16 --head linear --target unit --cos-weight 1.0 --lr 2e-4 --head-lr 1e-3 --wandb delta-nla
echo "== AR v2 shuffled-text control (1 epoch)"
python -m delta_nla.train_ar --model $MODEL --ar-layers 24 --descriptions $DESC --layers $LAYERS --out runs/ar_v2_shuf --epochs 1 --bs 16 --head linear --target unit --cos-weight 1.0 --lr 2e-4 --head-lr 1e-3 --shuffle-texts --wandb delta-nla
echo "== AR v2 mean-pooled variant (2 epochs)"
python -m delta_nla.train_ar --model $MODEL --ar-layers 24 --descriptions $DESC --layers $LAYERS --out runs/ar_v2_mean --epochs 2 --bs 16 --head linear --pool mean --target unit --cos-weight 1.0 --lr 2e-4 --head-lr 1e-3 --wandb delta-nla
echo "== end-to-end with AR v2"
python -m delta_nla.eval_e2e --model $MODEL --ar-dir runs/ar_v2 --av-dir runs/av --descriptions $DESC --layers $LAYERS --out runs/e2e_v2.jsonl --wandb delta-nla
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("syvb/delta-nla-qwen3-8b-ar-warmstart", exist_ok=True)
api.upload_folder(folder_path="runs/ar_v2", repo_id="syvb/delta-nla-qwen3-8b-ar-warmstart", commit_message="AR v2: linear head, unit-norm targets")
print("uploaded ar_v2")
PY
echo STAGE1B_DONE
