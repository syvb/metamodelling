"""Templated (no-LLM) change descriptions from evidence records.

These are deliberately plain and varied. They exist (a) as a cheap control / mix-in for the
warm-start and (b) as a fallback when the LLM rewrite is unavailable.
"""
from __future__ import annotations
import random


def q(tok: str) -> str:
    return "'" + tok.strip() + "'"


def join(toks: list[str]) -> str:
    toks = [q(t) for t in toks]
    if len(toks) <= 1:
        return "".join(toks)
    return ", ".join(toks[:-1]) + " and " + toks[-1]


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def pl(toks: list[str], sing: str, plur: str) -> str:
    return sing if len(toks) == 1 else plur


def back(offset: int) -> str:
    return "the previous token" if offset == 1 else f"{offset} tokens back"


def magnitude_word(rel_pct: float, rng: random.Random) -> str:
    if rel_pct < 15:
        return rng.choice(["barely", "only slightly", "marginally"])
    if rel_pct < 50:
        return rng.choice(["modestly", "somewhat", "moderately"])
    if rel_pct < 85:
        return rng.choice(["noticeably", "substantially", "clearly"])
    return rng.choice(["strongly", "sharply", "heavily"])


def describe(ev: dict, rng: random.Random | None = None, max_items: int = 3) -> str:
    rng = rng or random.Random(hash(ev["id"]) & 0xFFFF)
    m = ev["magnitude"]; eff = ev["effect"]["all"]; lens = ev["lens"]
    mag = magnitude_word(m["rel_norm_pct"], rng)
    def dedup(a, b):
        seen = {t.strip().lower() for t in a}
        return a, [t for t in b if t.strip().lower() not in seen]
    up, down = dedup([x["tok"] for x in eff["up"][:max_items]], [x["tok"] for x in eff["down"][:max_items]])
    lup, ldown = dedup([x["tok"] for x in lens["shift_up"][:max_items] if x["p_Y"] > 0.01],
                       [x["tok"] for x in lens["shift_down"][:max_items] if x["p_X"] > 0.01])
    srcs = ev["sources"]["top"][:2]
    parts = []

    # 1. main clause: effect on the model's eventual prediction (total effect) or direct lens
    # prefer the causal effect whenever it is non-trivial; the lens shift is only legible when
    # the shifted tokens carry real probability, otherwise fall back to a "small diffuse change" sentence
    use_effect = up and (m["kl_pct"] >= 20 or rng.random() < 0.5)
    lup = [t for t, x in zip(lup, lens["shift_up"]) if x["p_Y"] >= 0.02]
    ldown = [t for t, x in zip(ldown, lens["shift_down"]) if x["p_X"] >= 0.02]
    if use_effect:
        tmpl = rng.choice([
            "The update {mag} pushes the final prediction toward {up}{down_clause}.",
            "This layer {mag} raises the model's support for {up}{down_clause}.",
            "Evidence shifts {mag} toward {up}{down_clause}.",
            "{Up} {become} more likely as next tokens{down_clause2}.",
        ])
        down_clause = f" and away from {join(down)}" if down else ""
        down_clause2 = f", while {join(down)} {pl(down, 'becomes', 'become')} less likely" if down else ""
        parts.append(tmpl.format(mag=mag, up=join(up), Up=cap(join(up)), become=pl(up, 'becomes', 'become'), down_clause=down_clause, down_clause2=down_clause2))
    elif lup:
        tmpl = rng.choice([
            "The representation {mag} shifts toward {up}{down_clause}.",
            "The layer {mag} strengthens {up}{down_clause3}.",
            "{Up} {become} more strongly represented{down_clause2}.",
        ])
        down_clause = f" and away from {join(ldown)}" if ldown else ""
        down_clause2 = f", while {join(ldown)} {pl(ldown, 'fades', 'fade')}" if ldown else ""
        down_clause3 = f" and weakens {join(ldown)}" if ldown else ""
        parts.append(tmpl.format(mag=mag, up=join(lup), Up=cap(join(lup)), become=pl(lup, 'becomes', 'become'), down_clause=down_clause, down_clause2=down_clause2, down_clause3=down_clause3))
    else:
        parts.append(rng.choice(["The layer changes the representation very little here.",
                                 "Only a small, diffuse adjustment happens at this token."]))

    # 2. optional: where the information came from / which sublayer did it
    if srcs and rng.random() < 0.6:
        s = srcs[0]
        parts.append(rng.choice([
            f"Attention draws mainly on {q(s['tok'])} ({back(s['offset'])}).",
            f"The change is fed by attention to the earlier token {q(s['tok'])}.",
            f"Information is pulled in from {q(s['tok'])} earlier in the text.",
        ]))
    elif rng.random() < 0.4:
        share = m["attn_share"]
        if share > 0.65:
            parts.append("Most of the change comes from attention rather than the MLP.")
        elif share < 0.35:
            parts.append("Most of the change comes from the MLP rather than attention.")

    # 3. optional: top-1 flip
    if eff["top1_changes"] and rng.random() < 0.7:
        parts.append(f"Without it the top prediction would be {q(eff['top1_without'])} instead of {q(eff['top1_true'])}.")
    return " ".join(parts)
