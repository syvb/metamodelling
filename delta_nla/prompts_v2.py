"""Prompt v2: describe the change in the model's *thinking* between two layers, without quoting tokens
or referring to completions. Based on the user's draft; the measurement legend is ours."""
from __future__ import annotations
from .prompts import render_evidence

SYSTEM = """You are an interpretability tool. You are shown text that was provided to a large language model (with the last token in double square brackets), and information about the changes in the model's internal state between two adjacent layers at that last token. Keep in mind that LLM activations can be very noisy, and it might not be clear how the provided information relates to the context. Keep in mind that the logit lens often tends to indicate the model's internal thoughts, and not what the model is actually planning to output at the current token.

How to read the information you are given:
- CAUSAL EFFECT ON FINAL PREDICTION: how the eventual output distribution differs with and without this layer's update ("up" rose because of the update, "down" fell). This is the most trustworthy indication of what the update contributes. Treat its tokens as clues to a topic, entity, relation or judgement being worked on; never report them as words. kl_pct ranks how consequential this update is compared with others by the same layer (0 = negligible, 100 = among the most consequential).
- LENS SHIFT and STATE BEFORE/AFTER: what the internal state most resembles when read directly, before and after the layer. Shifts here often reflect thoughts, associations and what is being held in mind rather than output plans. Ignore entries with tiny probabilities.
- RAW UPDATE LENS: a direct reading of the update vector; noisy. Use only when it agrees with the rest.
- ATTENTION SOURCES: earlier positions whose information the update drew on (offset = how many tokens back; frac = share). A high sink_frac with no sources means little was gathered from earlier text.
- MAGNITUDE: rel_norm_pct ranks the size of this update among updates by the same layer; attn_share is the fraction that came from gathering information from earlier text versus recall and transformation in place.

Start your answer immediately with a tag "<explanation>", then write a natural-language explanation of what the **change** between the two layers might be, and close with "</explanation>". (Do NOT describe what the *current* state is; describe the *change* between states.)

Rules for the explanation:
1. Write 3-4 sentences (2-3 when the change is small). Active voice; never refer to the model in the third person ("Shifting from..." not "The model shifts from"). Begin with the most specific content of the change, not with a generic phrase such as "Shifting from a diffuse continuation".
2. Do not quote individual tokens, and do not talk about words as words. Describe the referents and ideas: the person, place, quantity, event, relation or judgement being thought about, and how the balance between them changes (what gains weight, what recedes, what gets settled, what gets drawn in from earlier in the text).
3. Stay grounded. Every entity, concept or relation you mention must appear in the provided text or be a plain paraphrase of something in the information lists. Never introduce specifics from your own world knowledge (no new names, places, objects or facts), even if they seem likely.
4. You must NEVER refer to potential completions, nor to the type, form or structure of upcoming text. This forbids any mention of grammar, punctuation, plurals, subjects, connectives, phrases, spelling, numerals, digits or which specific value or name "comes next"; describe the thought behind such a choice instead (e.g. "the count of remaining tablespoons of oil" rather than "the numeral 2"; "the identity of the co-star" rather than "the surname").
5. Do not hedge or express uncertainty: if unsure, give your best guess confidently. You may describe the size of an effect ("a small shift", "a strong commitment"), and you must calibrate it: when kl_pct and rel_norm_pct are both low, describe a faint, minor adjustment and do not narrate a major settling.
6. Do not use LLM or measurement terminology in the response: no "model", "logit", "completion", "token", "layer", "probability", "update", "representation", "state", "vector", "noisy", "evidence", "signal", and no talk of attention heads or sources ("attention" in its everyday sense is fine). Do not refer to how you know anything; describe only the thinking process.
7. Do not repeat the text; only refer to its content as needed to say what the thought is about."""


def build_messages(ev: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": render_evidence(ev) + "\n\nWrite your explanation now."}]


# ---------------------------------------------------------------------------------------------
# "Thought view" of the evidence: drop things that invite completion-talk (function words, subword
# fragments, digits, punctuation, top-1 lines), keep content-bearing clues.
import re as _re

STOP = set("""a an the and or but of to in on at by for from with as into onto over under than then that this these those
it its it's is are was were be been being do does did don doesn didn aren isn wasn weren have has had having not no nor so
if when while where which who whom whose what why how all any each every some more most much many few both either neither
one two three up down out off about above below between through during before after again further once here there very
can could will would shall should may might must also just only own same such too s t d ll re ve m""".split())

def content_token(tok: str) -> bool:
    s = tok.strip()
    if not s or s.lower() in STOP:
        return False
    if any("一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" or "가" <= ch <= "힯" for ch in s):
        return True  # CJK: usually a whole word
    if not _re.search(r"[A-Za-zÀ-ɏЀ-ӿ]", s):
        return False  # digits / punctuation
    if len(s) < 3:
        return False
    if not tok.startswith(" ") and not s[0].isupper():
        return False  # mid-word fragment like 'ier', 'si'
    return True


def _tl(items, n=6):
    out = [x["tok"].strip() for x in items if content_token(x["tok"])]
    return ", ".join(out[:n]) if out else "(nothing distinctive)"


def render_thought_view(ev: dict) -> str:
    m = ev["magnitude"]; lens = ev["lens"]; eff = ev["effect"]; src = ev["sources"]
    ctx = ev["context"]; cur = ev["current_token"]
    if ctx.endswith(cur):
        ctx = ctx[: len(ctx) - len(cur)] + f"[[{cur}]]"
    L = []
    L.append(f"DEPTH: layer {ev['layer']} of {ev['n_layers']} ({ev['depth_frac']:.0%} through the network)")
    L.append("TEXT (last token in double brackets):")
    L.append(ctx)
    L.append("")
    L.append(f"MAGNITUDE: rel_norm_pct={m['rel_norm_pct']} (size rank among this layer's changes), kl_pct={m['kl_pct']} (consequence rank), attn_share={m['attn_share']} (fraction gathered from earlier text vs recalled in place)")
    L.append("")
    e = eff["all"]
    L.append(f"CAUSAL EFFECT ON THE EVENTUAL OUTPUT (whole change; kl={e['kl']}):")
    L.append(f"  gains: {_tl(e['up'])}")
    L.append(f"  loses: {_tl(e['down'])}")
    ea, em = eff["attn"], eff["mlp"]
    L.append(f"  from gathering earlier text -> gains: {_tl(ea['up'], 4)} | loses: {_tl(ea['down'], 4)}")
    L.append(f"  from in-place recall        -> gains: {_tl(em['up'], 4)} | loses: {_tl(em['down'], 4)}")
    L.append("")
    L.append(f"WHAT THE INTERNAL READING MOVES TOWARD: {_tl(lens['shift_up'])}")
    L.append(f"WHAT IT MOVES AWAY FROM: {_tl(lens['shift_down'])}")
    L.append(f"RAW CHANGE READING (noisy) toward: {_tl(lens['lens_d_up'])} | away: {_tl(lens['lens_d_down'])}")
    L.append("")
    if src["top"]:
        srcs = "; ".join(f"{s['tok'].strip()!r} ({s['offset']} back, share {s['frac']}) in \"{s['snippet']}\"" for s in src["top"][:4])
    else:
        srcs = "(no informative source)"
    L.append(f"EARLIER TEXT DRAWN ON: {srcs}; sink_frac={src['sink_frac']}")
    return "\n".join(L)


def build_messages(ev: dict) -> list[dict]:  # noqa: F811  (overrides the version above)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": render_thought_view(ev) + "\n\nWrite your explanation now."}]


BANNED_RE = _re.compile(r"""(?ix)\b(model|models|logit|logits|completion|completions|continuation|continuations|token|tokens|layer|layers|
    attention\s+(head|heads|weight|weights|source|sources|pattern|patterns)|probabilit(y|ies)|update|representation|vector|noisy|evidence|signal|grammar|grammatical|punctuation|plural|singular|
    connective|phrase|phrasing|numeral|digit|digits|spelling|surname|suffix|prefix|subword|the\s+next\s+word|comes\s+next|predict\w*|output|outputs)\b""")
QUOTE_RE = _re.compile(r"""["“”'‘’][^"“”'‘’]{1,30}["“”'‘’]""")


def violations(text: str) -> list[str]:
    v = []
    m = BANNED_RE.findall(text)
    if m:
        words = sorted({(x[0] if isinstance(x, tuple) else x).lower() for x in m})
        v.append("uses forbidden terminology: " + ", ".join(words))
    q = [x for x in QUOTE_RE.findall(text) if not x.endswith("’s") and "’" not in x[1:-1]]
    if q:
        v.append("quotes words: " + ", ".join(q[:4]))
    n_sent = len([s for s in _re.split(r"[.!?]+\s", text.strip()) if s.strip()])
    if n_sent > 5:
        v.append(f"too long ({n_sent} sentences)")
    if _re.search(r"\b(might|may|possibly|perhaps|likely|seems?|appears?|potential(ly)?|uncertain)\b", text, _re.I):
        v.append("hedges")
    return v


REPAIR = "Your explanation broke these rules: {v}. Rewrite it so that it obeys every rule, keeping the same content otherwise. Start again with <explanation>."
