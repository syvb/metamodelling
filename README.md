# metamodelling — Delta-NLA warm-start

Delta-NLA is a variant of the Natural Language Autoencoder in which the verbalizer sees the residual
stream before (X) and after (Y) one transformer block and the reconstructor must recover the update
Δ = Y − X from the description alone. This repo currently builds the *warm-start* dataset: short,
grounded, change-language descriptions of what each block did at a token, for Qwen3-8B.

## Pipeline

| step | file | what it does |
|---|---|---|
| 1 | `delta_nla/collect.py` | GPU. For sampled (doc, position, layer): stores X, d, d_attn, d_mlp (fp16 `.npz`); attention sources (attention weight × OV-contribution norm); causal effect of removing the update (all / attention-only / MLP-only) on the final next-token log-probs. |
| 2 | `delta_nla/build_evidence.py` | Decodes everything to text, adds fixed-scale logit-lens evidence and lens probability shift, per-layer percentiles → `evidence.jsonl`. |
| 3 | `delta_nla/rewrite.py` (+ `prompts.py`) | Grounded LLM rewrite via OpenRouter: the writer sees only measurements and must quote tokens from them. Resumable, cost-tracked. |
| 3b | `delta_nla/templates.py` | Templated no-LLM descriptions (control / mix-in). |
| 4 | `delta_nla/validate.py` | Rejects descriptions whose quoted tokens do not appear in the evidence. |
| 5 | `delta_nla/build_dataset.py` | Assembles `warmstart.jsonl` with `llm`, `llm_short`, `template` variants and a doc-level split. |

`scripts/runpod_launch.py` runs steps 1–2 on a RunPod GPU and ships results through wandb artifacts
(project `delta-nla`); `scripts/memrun.sh` runs anything locally under a cgroup memory cap.

## Design notes

* The reconstruction target is the update, so the description must be about *change*. The most
  reliable evidence for that is causal: recompute the final logits with the update removed at the
  position. The raw logit lens of Δ is mostly noise at middle layers and is only shown to the writer
  as a low-priority signal.
* Lens differences are computed at a fixed scale (`W_U (g ⊙ Δ / rms(X))`) to avoid the artefact in
  `W_U Norm(Y) − W_U Norm(X)`.
* The writer never sees the true next token, so it cannot leak the answer into the description.
