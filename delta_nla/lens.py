"""Offline logit-lens evidence for Delta-NLA records.

Fixed-scale lens: for an update vector v at a position whose incoming residual is X,
    lens(v) = W_U ( g * v / rms(X) )
i.e. we project the *change* through the final norm gain and unembedding at the scale
the incoming residual would have been normalised by. This avoids the rescaling artefact
in  W_U Norm(Y) - W_U Norm(X)  (Norm is nonlinear; that difference contains a copy of
X's own lens scaled by 1/rms(Y) - 1/rms(X)).

For the state lenses (what X / Y themselves "predict") we use the ordinary logit lens.
"""
from __future__ import annotations
import json, re, unicodedata
from pathlib import Path

import numpy as np
import torch


class Lens:
    def __init__(self, unembed_path: str | Path, tokenizer, device: str = "cpu"):
        d = torch.load(unembed_path, map_location=device)
        self.device = device
        self.W_U = d["W_U"].to(device).float()          # [V, d]
        self.g = d["norm_w"].to(device).float()          # [d]
        self.eps = float(d["eps"])
        self.tok = tokenizer
        self.V = self.W_U.shape[0]
        self._special = set(tokenizer.all_special_ids)
        self._display_cache: dict[int, str | None] = {}

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + self.eps)

    def state_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Ordinary logit lens of a residual vector x: W_U (g * x / rms(x))."""
        return (self.g * x.float() / self.rms(x)) @ self.W_U.T

    def delta_logits(self, v: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Fixed-scale lens of an update v, normalised by rms(X)."""
        return (self.g * v.float() / self.rms(X)) @ self.W_U.T

    # ---- token display / filtering ----
    def display(self, tid: int) -> str | None:
        """Human-readable form of a token, or None if it should be filtered."""
        if tid in self._display_cache:
            return self._display_cache[tid]
        out = self._display_token(tid)
        self._display_cache[tid] = out
        return out

    def _display_token(self, tid: int) -> str | None:
        if tid in self._special or tid >= len(self.tok):
            return None
        s = self.tok.decode([tid])
        if "�" in s:            # partial UTF-8 byte token
            return None
        stripped = s.strip()
        if not stripped:
            return None
        if not any(ch.isalnum() for ch in stripped):
            return None              # pure punctuation / symbols
        if any(unicodedata.category(ch).startswith("C") for ch in stripped):
            return None
        return s

    def top_tokens(self, logits: torch.Tensor, k: int = 5, sign: int = +1, pool: int = 60) -> list[dict]:
        """Top-k promoted (sign=+1) or suppressed (sign=-1) tokens after filtering and de-duplication."""
        vals, idx = torch.topk(sign * logits, pool)
        seen: set[str] = set()
        out = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            s = self.display(i)
            if s is None:
                continue
            key = s.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": i, "tok": s, "score": float(sign * v)})
            if len(out) >= k:
                break
        return out

    def state_top(self, x: torch.Tensor, k: int = 5) -> list[dict]:
        lg = self.state_logits(x)
        p = torch.softmax(lg, -1)
        out = self.top_tokens(lg, k)
        for o in out:
            o["p"] = float(p[o["id"]])
        return out

    def prob_shift(self, X: torch.Tensor, Y: torch.Tensor, k: int = 6) -> tuple[list[dict], list[dict]]:
        """Tokens whose ordinary logit-lens probability rose / fell most from X to Y.

        Unlike the raw lens of d (which is dominated by directions W_U barely reads at middle
        layers), this only surfaces tokens that are already legible in the lens at X or Y.
        """
        pX = torch.softmax(self.state_logits(X), -1)
        pY = torch.softmax(self.state_logits(Y), -1)
        diff = pY - pX
        up = self.top_tokens(diff, k, +1)
        down = self.top_tokens(diff, k, -1)
        for o in up + down:
            o["p_X"] = float(pX[o["id"]]); o["p_Y"] = float(pY[o["id"]])
        return up, down

    def evidence(self, X: np.ndarray, d: np.ndarray, d_attn: np.ndarray, d_mlp: np.ndarray, k: int = 6) -> dict:
        X_t, d_t, da_t, dm_t = (torch.from_numpy(a.astype(np.float32)).to(self.device) for a in (X, d, d_attn, d_mlp))
        Y_t = X_t + d_t
        shift_up, shift_down = self.prob_shift(X_t, Y_t, k)
        ev = {
            "state_X": self.state_top(X_t, k),
            "state_Y": self.state_top(Y_t, k),
            "shift_up": shift_up, "shift_down": shift_down,
        }
        for name, v in (("d", d_t), ("d_attn", da_t), ("d_mlp", dm_t)):
            lg = self.delta_logits(v, X_t)
            ev[f"lens_{name}_up"] = self.top_tokens(lg, k, +1)
            ev[f"lens_{name}_down"] = self.top_tokens(lg, k, -1)
            ev[f"lens_{name}_std"] = float(lg.std())
        return ev


def load_vectors(raw_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Return {rec_id: {X, d, d_attn, d_mlp}} for all shards in raw_dir (fp16 -> kept as fp16)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    keys = ("X", "d", "d_attn", "d_mlp")
    for f in sorted(Path(raw_dir).glob("vec_*.npz")):
        with np.load(f) as z:
            ids = z["ids"]
            arrs = {k: z[k] for k in keys}  # read each array once (NpzFile re-reads on every access)
        for j, rid in enumerate(ids):
            out[str(rid)] = {k: arrs[k][j] for k in keys}
    return out
