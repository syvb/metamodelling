"""Download one subdirectory of the HF dataset into a local dir (flattening the subdir)."""
import os, shutil, sys
from huggingface_hub import snapshot_download
sub, out = sys.argv[1], sys.argv[2]
tok = open(os.path.expanduser("~/.hf_token")).read().strip()
d = snapshot_download("syvb/delta-nla-qwen3-8b-warmstart", repo_type="dataset", local_dir="data/hf", allow_patterns=[f"{sub}/*"], token=tok)
os.makedirs(out, exist_ok=True)
for f in os.listdir(os.path.join(d, sub)):
    shutil.copy(os.path.join(d, sub, f), os.path.join(out, f))
print(sorted(os.listdir(out)))
