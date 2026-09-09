"""Download the fineweb part of the HF dataset into data/hf and link it as data/raw (used on pods)."""
import os
from huggingface_hub import snapshot_download
snapshot_download("syvb/delta-nla-qwen3-8b-warmstart", repo_type="dataset", local_dir="data/hf", allow_patterns=["fineweb/*"], token=os.environ.get("HF_TOKEN"))
os.makedirs("data", exist_ok=True)
if os.path.islink("data/raw") or not os.path.exists("data/raw"):
    if os.path.islink("data/raw"): os.unlink("data/raw")
    os.symlink(os.path.abspath("data/hf/fineweb"), "data/raw")
print(sorted(os.listdir("data/raw")))
