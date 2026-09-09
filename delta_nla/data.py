"""Warm-start pairs for Delta-NLA SFT: (layer, X, d, description) with doc-level splits and per-layer target normalisation."""
from __future__ import annotations
import glob, json, random, zlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


@dataclass
class Pair:
    id: str; doc: str; layer: int; text: str
    X: np.ndarray; d: np.ndarray  # fp32


def load_pairs(evidence: str, descriptions: str, raw: str, layers: list[int] | None = None, field: str = "description",
               use_templates: bool = False, limit: int = 0) -> list[Pair]:
    ev = {}
    for l in open(evidence):
        e = json.loads(l)
        if layers is None or e["layer"] in layers:
            ev[e["id"]] = e
    texts = {}
    if use_templates:
        from .templates import describe
        texts = {i: describe(e, random.Random(zlib.crc32(i.encode()))) for i, e in ev.items()}
    else:
        for l in open(descriptions):
            d = json.loads(l)
            if d.get(field) and d["id"] in ev:
                texts[d["id"]] = d[field]
    need = set(texts)
    pairs = []
    for f in sorted(glob.glob(f"{raw}/vec_*.npz")):
        with np.load(f) as z:
            ids = z["ids"]; Xa = z["X"]; Da = z["d"]
            for i, rid in enumerate(ids):
                rid = str(rid)
                if rid in need:
                    e = ev[rid]
                    pairs.append(Pair(rid, e["doc_id"], e["layer"], texts[rid], Xa[i].astype(np.float32), Da[i].astype(np.float32)))
    if limit:
        pairs = pairs[:limit]
    return pairs


def split_by_doc(pairs: list[Pair], val_frac: float = 0.1, seed: int = 0):
    docs = sorted({p.doc for p in pairs}); random.Random(seed).shuffle(docs)
    val = set(docs[: max(1, int(len(docs) * val_frac))])
    return [p for p in pairs if p.doc not in val], [p for p in pairs if p.doc in val]


class TargetNorm:
    """Per-layer target normalisation.
    mode="rms":  y = (d - mean) / scalar_rms          (keeps relative norms)
    mode="unit": y = (d - mean) / ||d - mean||         (direction only; the paper's convention, and the AV never sees ||d||)
    FVE is always computed about the per-layer mean, in the normalised space used for training."""
    def __init__(self, train: list[Pair], mode: str = "unit"):
        self.mode = mode; self.mean, self.scale = {}, {}
        by = {}
        for p in train:
            by.setdefault(p.layer, []).append(p.d)
        for n, ds in by.items():
            D = np.stack(ds); mu = D.mean(0)
            self.mean[n] = mu; self.scale[n] = float(np.sqrt(((D - mu) ** 2).mean()))
    def encode(self, d: np.ndarray, layer: int) -> np.ndarray:
        c = d - self.mean[layer]
        if self.mode == "unit":
            return c / (np.linalg.norm(c) + 1e-6)
        return c / self.scale[layer]
    def target(self, d: np.ndarray, layer: int) -> np.ndarray:
        """The vector FVE is computed against (normalised space, mean removed)."""
        return self.encode(d, layer)
    def state_dict(self):
        return {"mode": self.mode, "mean": {n: torch.from_numpy(m) for n, m in self.mean.items()}, "scale": self.scale}
    @classmethod
    def from_state_dict(cls, sd):
        o = cls.__new__(cls); o.mode = sd.get("mode", "rms"); o.mean = {int(n): m.numpy() for n, m in sd["mean"].items()}; o.scale = {int(n): s for n, s in sd["scale"].items()}; return o


def fve_norm(pred: np.ndarray, true: np.ndarray) -> float:
    """FVE in normalised target space: 1 - sum||t - p||^2 / sum||t - mean_t||^2 (mean over the eval set)."""
    return float(1 - ((true - pred) ** 2).sum() / ((true - true.mean(0)) ** 2).sum())


def cosines(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.sum(pred * true, 1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-8)))


def fve(pred: np.ndarray, true: np.ndarray, mean: np.ndarray) -> float:
    """Fraction of variance explained about `mean` (the per-layer mean update)."""
    return float(1 - ((true - pred) ** 2).sum() / ((true - mean) ** 2).sum())


def act_scale(train: list[Pair], q: float = 0.75) -> dict[int, float]:
    """Paper heuristic for the injection scale: the q-quantile of activation norms at the layer."""
    by = {}
    for p in train:
        by.setdefault(p.layer, []).append(float(np.linalg.norm(p.X)))
    return {n: float(np.quantile(v, q)) for n, v in by.items()}
