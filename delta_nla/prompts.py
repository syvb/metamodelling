"""Prompt for the grounded LLM rewrite of Delta-NLA evidence into change descriptions."""
from __future__ import annotations
import json

SYSTEM = """You write short, plain-English descriptions of what ONE transformer layer changed in a language model's internal state at ONE token position. These descriptions are training targets for a model that must recover the layer's activation update from your text alone, so every description must be specific, literal, and grounded in the measurements you are given.

You are given measurements, not explanations. Describe the CHANGE the layer made, never the layer's mechanism, never a summary of the text, and never anything the measurements do not support.

Definitions of the measurements (all computed at the marked token):
- CAUSAL EFFECT ON FINAL PREDICTION: the model's final next-token probabilities were recomputed with this layer's update removed at this position. "up" tokens are next-token candidates whose final probability this update raised (dlp = change in log-prob; p = actual final probability). "down" tokens are candidates it lowered (p_abl = probability without the update). This is the most reliable signal: it says what the update DOES to the model's eventual prediction. It is given for the whole update and separately for its attention part and MLP part. kl_pct is where this update's overall causal effect ranks among updates by the same layer (0 = weakest, 100 = strongest).
- LENS SHIFT: reading the residual stream directly through the unembedding before and after the layer, which next-token candidates gained (shift_up) or lost (shift_down) probability. Reliable only for tokens with non-trivial probability; ignore entries with tiny p_X and p_Y.
- RAW UPDATE LENS: the update vector itself projected through the unembedding. Often noise at middle layers; use only when its tokens form a clear theme that agrees with the other signals.
- ATTENTION SOURCES: earlier tokens the layer's attention pulled information from at this position (offset = tokens back; frac = share of the attention output attributable to that source; snippet shows it in context with [[ ]]). sink_frac is the share going to the first token, which carries no information. When sink_frac is high and no source is listed, attention moved little information here.
- MAGNITUDE: rel_norm is |update| / |incoming residual|; rel_norm_pct ranks it among this layer's updates (100 = largest). attn_share is the fraction of the update coming from attention vs the MLP.
- STATE BEFORE/AFTER: the lens reading of the residual stream before (X) and after (Y) the layer, top candidates only. Background context; do not restate it.

Writing rules:
1. One to three sentences, at most 60 words. Change language: "raises", "lowers", "shifts toward/away from", "pulls in", "suppresses", "commits to", "makes less likely", "moves information from".
2. Lead with the strongest, best-supported change. Prefer the causal effect, then the lens shift. Name the raw update lens only if it clearly agrees.
3. Quote 2 to 5 literal tokens in single quotes, chosen from the measurements, e.g. 'or', 'Period'. Keep leading spaces out of quotes. When several quoted tokens share an obvious category, you may name the category as well ("sentence-initial connectives such as 'If' and 'Then'"), but always include the literal tokens.
4. If an attention source with frac >= 0.15 exists, say what was pulled in and from where, e.g. "pulling in 'Period' from 9 tokens back". Do not invent sources.
5. Express size once, using the percentiles: rel_norm_pct or kl_pct below 20 -> "slightly"/"a small update"; 20-60 -> "moderately"; 60-85 -> "substantially"; above 85 -> "strongly". If kl_pct < 15 and the top-1 does not change, say the layer makes only a minor adjustment and name what it nudges.
6. Mention attention vs MLP only if attn_share is above 0.7 or below 0.3, or if their causal effects clearly differ.
7. Do not describe the text's topic, genre, or grammar except as needed to make a quoted token intelligible (e.g. "the unit name 'Period'"). Do not mention what the text is about. Do not use the words "measurement", "lens", "evidence", "signal", "data shows", or "layer N".
8. Never claim anything about tokens that appear in none of the measurement lists.

Output JSON only: {"description": "...", "short": "..."} where "short" is a one-clause version of at most 15 words that keeps the two most important quoted tokens."""


def _toklist(items, keys=("tok",), n=6):
    out = []
    for x in items[:n]:
        parts = [repr(x["tok"].strip() or x["tok"]) if k == "tok" else f"{k}={x[k]}" for k in keys if k in x]
        out.append(" ".join(parts))
    return "; ".join(out) if out else "(none)"


def render_evidence(ev: dict) -> str:
    m = ev["magnitude"]; lens = ev["lens"]; eff = ev["effect"]; src = ev["sources"]
    lines = []
    lines.append(f"LAYER: {ev['layer']} of {ev['n_layers']} (depth {ev['depth_frac']:.0%})")
    lines.append(f"TEXT UP TO THE MARKED TOKEN (the marked token is the last one, shown as [[...]]):")
    ctx = ev["context"]
    cur = ev["current_token"]
    if ctx.endswith(cur):
        ctx = ctx[: len(ctx) - len(cur)] + f"[[{cur}]]"
    lines.append(ctx)
    lines.append("")
    lines.append(f"MAGNITUDE: rel_norm={m['rel_norm']} rel_norm_pct={m['rel_norm_pct']} kl_pct={m['kl_pct']} attn_share={m['attn_share']}")
    lines.append("")
    for name, label in (("all", "WHOLE UPDATE"), ("attn", "ATTENTION PART"), ("mlp", "MLP PART")):
        e = eff[name]
        lines.append(f"CAUSAL EFFECT ON FINAL PREDICTION ({label}): kl={e['kl']}; top-1 with update={e['top1_true']!r} (p={e['p_top1_true']}), without={e['top1_without']!r} (p of true top-1 without={e['p_top1_without']})")
        lines.append(f"  up:   {_toklist(e['up'], ('tok', 'dlp', 'p'))}")
        lines.append(f"  down: {_toklist(e['down'], ('tok', 'dlp', 'p', 'p_abl'))}")
    lines.append("")
    lines.append(f"LENS SHIFT up:   {_toklist(lens['shift_up'], ('tok', 'p_X', 'p_Y'))}")
    lines.append(f"LENS SHIFT down: {_toklist(lens['shift_down'], ('tok', 'p_X', 'p_Y'))}")
    lines.append(f"STATE BEFORE (X) top: {_toklist(lens['state_X'], ('tok', 'p'))}")
    lines.append(f"STATE AFTER (Y) top:  {_toklist(lens['state_Y'], ('tok', 'p'))}")
    lines.append(f"RAW UPDATE LENS up: {_toklist(lens['lens_d_up'])} | down: {_toklist(lens['lens_d_down'])}")
    lines.append("")
    if src["top"]:
        srcs = "; ".join(f"{s['tok'].strip()!r} offset={s['offset']} frac={s['frac']} snippet=\"{s['snippet']}\"" for s in src["top"])
    else:
        srcs = "(no informative source)"
    lines.append(f"ATTENTION SOURCES: {srcs}; sink_frac={src['sink_frac']} self_frac={src['self_frac']}")
    lines.append("")
    lines.append(f"MODEL'S ACTUAL FINAL TOP PREDICTIONS: {_toklist(ev['model_top'], ('tok', 'p'))}")
    return "\n".join(lines)


def build_messages(ev: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": render_evidence(ev) + "\n\nWrite the JSON now."}]
