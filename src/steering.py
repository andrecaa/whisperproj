"""Activation steering (Phase D, AMBITION; PLAN §4D).

Steering vector = mean pooled activation difference between two language
groups at the most causal block from Phase C, computed on clips held out
from evaluation. At inference, alpha * v is added to that block's output at
every time frame (standard activation addition). Evaluation sweeps alpha:
flip rate of Whisper's language token vs. WER degradation of the transcript.
"""

import re

import numpy as np
import torch


def steering_vector(
    pooled: torch.Tensor,          # [n_layers+1, n_clips, d] from load_cached
    metadata,                      # matching DataFrame with "language"
    layer: int,
    lang_a: str,
    lang_b: str,
    clip_slice: slice,
) -> torch.Tensor:
    """v = mean(lang_b clips) - mean(lang_a clips) at `layer` (float32, [d]).

    clip_slice indexes WITHIN each language group (e.g. slice(200, 400) uses
    each group's second half, keeping the first half unseen for evaluation).
    """
    X = pooled[layer]
    idx_a = metadata.index[metadata["language"] == lang_a][clip_slice]
    idx_b = metadata.index[metadata["language"] == lang_b][clip_slice]
    v = X[list(idx_b)].mean(dim=0) - X[list(idx_a)].mean(dim=0)
    return v


_norm_re = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(s: str) -> str:
    return _norm_re.sub("", s.lower()).strip()


def wer(ref: str, hyp: str) -> float:
    """Word error rate via Levenshtein distance (capped at 1.0 for reporting)."""
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    # standard two-row dynamic program: prev/cur are rows of the edit
    # distance table; each cell takes the cheapest of deletion, insertion,
    # or substitution (free when the words match)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (rw != hw))
        prev = cur
    return min(1.0, prev[-1] / len(r))
