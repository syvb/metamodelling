"""Prompt v2: describe the change in the model's *thinking* between two layers, without quoting tokens
or referring to completions. Based on the user's draft; the measurement legend is ours."""
from __future__ import annotations
from .prompts import render_evidence

SYSTEM = """You are an interpretability tool. You are shown text that was provided to a large language model, with the last token in double square brackets, and information about how the model's internal state changed between two adjacent layers at that last token. The bracketed token is the current position: the one whose transition you are describing; nothing after it exists yet. Keep in mind that LLM activations can be very noisy, and it might not be clear how the provided information relates to the context. Keep in mind that the logit lens often tends to indicate the model's internal thoughts, and not what the model is actually planning to output at the current token.

How to read the information, in order of trust:
1. CAUSAL EFFECT ON THE EVENTUAL OUTPUT: how the eventual output differs with and without this change ("gains" rose because of it, "loses" fell). This is the primary basis for your explanation. Its labels are clues to a topic, entity, relation or judgement being worked on; paraphrase a label into a concept only when its referent is clear from the text, otherwise omit it. Never fuse separate labels into a relation that neither the text nor the lists state (if the labels are "care", "players" and "priority", do not assert that care for the players was the priority; say what gains and what recedes). Translate a non-English label when its meaning is unambiguous, without embellishing it. It is split into the part that came from gathering earlier text and the part recalled in place.
2. EARLIER TEXT DRAWN ON: earlier positions whose content fed the change (offset = how many tokens back; share = fraction). Mention an earlier passage only when it explains what gained or lost weight. Let attn_share govern the wording: below 0.35 the change came mainly from internal recall (say so, and do not imply retrieval); 0.35-0.65 it came partly from earlier text and partly from recall; above 0.65 it came mainly from earlier text. A high sink_frac likewise points to recall rather than retrieval.
3. WHAT THE INTERNAL READING MOVES TOWARD / AWAY FROM: what the state most resembles when read directly, before versus after. This often reflects associations being held in mind rather than plans. Use it to refine the causal picture when it agrees; when it conflicts, keep only the shared theme. A strong shift here cannot outweigh an empty causal effect.
4. RAW CHANGE READING: noisy; use only as corroboration.
MAGNITUDE gives two separate things. rel_norm_pct is the size of the internal movement (rank among this layer's changes); kl_pct is its consequence for the eventual output. Describe them separately when they disagree: a large movement with low consequence is "a substantial internal shift that barely alters the overall line of thought"; a small movement with high consequence is "a small but decisive adjustment". Thresholds: below 20 = faint/minor; 20-60 = moderate; 60-85 = substantial; above 85 = strong. attn_share is the fraction of the change that came from gathering earlier text rather than recall in place.

Start your answer immediately with a tag "<explanation>", then write a natural-language explanation of what the **change** between the two layers might be, and close with "</explanation>". Describe the change, not the resulting state: every sentence should convey a contrast, what becomes more salient versus what loses salience, or what gets settled versus what gets dropped.

Structure, 3-4 sentences (2-3 when the change is faint):
- Sentence 1: the primary shift, stated as specifically as the information allows, opening directly with its content.
- Sentence 2: what recedes or is displaced.
- Sentence 3: what is drawn in from earlier text, or consolidated from internal recall (say which, following attn_share).
- Sentence 4 (optional): calibrated strength of the movement and of its consequence.
Low-information fallback: if the causal gains and losses hold nothing distinctive and kl_pct is below 15, do not claim anything was settled, selected or substantially reinforced; write two sentences saying the change is faint and only lightly favours the most relevant nearby theme over its alternatives.

Rules:
- Use impersonal active phrasing ("Emphasis moves from... to...", "Focus narrows to...", "The idea of ... gains weight over ..."); vary the opening; never refer to the model, the network or "it" as an agent thinking in the third person.
- Do not quote individual tokens and do not talk about words as words. Describe referents: the person, place, quantity, event, relation or judgement. Obvious aliases of one entity may be merged when the text supports it, but never extend a partial name using your own knowledge unless the text confirms it.
- You must NEVER refer to potential completions, nor to the type, form or structure of upcoming text: no grammar, punctuation, plurals, subjects, connectives, phrases, spelling, numerals, or which value or name "comes next". Describe the thought behind such a choice instead (the count of remaining tablespoons of oil, not the numeral; the identity of the co-star, not the surname).
- Quantities that are themselves the subject of the thinking (a count, an intermediate result, a product, a total) may be named in words as quantities, e.g. "the count of legs settles on four" or "the product of seven and twelve comes into view", but never framed as what will be written, and ONLY when that quantity appears in the internal-reading lists or the text. Never state the result of a calculation from your own arithmetic or knowledge: if the lists say "four" and "sixteen", those are the quantities in play even if you believe the true answer differs.
- When the only legible information is that a numeric answer is in flux (a note to that effect in the causal lists, and nothing distinctive in the internal reading), say plainly in two sentences that the answer is being worked out with no specific quantity or operation yet legible, and name only the faint theme the text supplies. Do not invent a reasoning step.
- Stay grounded: every entity, concept or relation you mention must appear in the provided text (including the quoted earlier passages) or be a plain paraphrase of a listed label whose referent is clear. Never add specifics from world knowledge. Translate a non-English label only when its meaning is unambiguous and relevant; otherwise ignore it. Ignore corrupted fragments and unrelated stray labels.
- Do not hedge or express uncertainty: if unsure, give your best guess confidently. Describe size only through the calibrated terms above.
- Do not discuss the instrumentation or the measurement process, and do not use LLM terminology ("model", "logit", "token", "layer", "completion", "probability", "update", "vector", "representation"). "Attention" in its everyday sense is fine.
- Do not restate the text; refer to its content only as needed to say what the thought is about.
Before answering, silently check: every noun phrase is supported by the input; every sentence describes a change rather than a topic; nothing refers to output form; size words match the thresholds."""


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
    if _re.fullmatch(r"\d+", s):
        return True  # digits carry quantity thoughts (kept; the prompt says how to talk about them)
    if not _re.search(r"[A-Za-zÀ-ɏЀ-ӿ]", s):
        return False  # punctuation
    if len(s) < 3:
        return False
    if not tok.startswith(" ") and not s[0].isupper():
        return False  # mid-word fragment like 'ier', 'si'
    return True


def _tl(items, n=6):
    """Render a token list for the writer. Digit tokens are only the leading digit of a numeric answer, so
    when they dominate a list they are collapsed into a plain statement instead of being shown."""
    toks = [x["tok"].strip() for x in items if content_token(x["tok"])]
    digits = [t for t in toks if _re.fullmatch(r"\d+", t)]
    words = [t for t in toks if not _re.fullmatch(r"\d+", t)]
    parts = []
    if words:
        parts.append(", ".join(words[:n]))
    if digits:
        parts.append("[leading digit of a numeric answer in flux; not informative about quantities]")
    return "; ".join(parts) if parts else "(nothing distinctive)"


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
