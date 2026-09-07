"""Probing (Phase B, BASELINE; PLAN §4B).

Trains simple supervised probes on cached pooled activations, one per
(layer, property) cell:

  * classification (language, gender, speaker-ID): standardize + logistic
    regression, metric = accuracy
  * regression (median F0): standardize + ridge (alpha chosen by internal CV
    on the train folds), metric = R^2

All probes use 5-fold cross-validation. When `groups` (speaker IDs) are
provided the folds are speaker-disjoint (GroupKFold), the anti-leakage
control from PLAN §4B; otherwise stratified/plain KFold with a fixed seed.
The shuffled-label control reuses the same machinery on permuted labels.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
N_SPLITS = 5


def _splitter(task: str, groups):
    if groups is not None:
        return GroupKFold(n_splits=N_SPLITS)
    if task == "classification":
        return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def _pipeline(task: str):
    if task == "classification":
        clf = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED)
    else:
        clf = RidgeCV(alphas=np.logspace(-1, 4, 11))
    return make_pipeline(StandardScaler(), clf)


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
    groups: np.ndarray | None = None,
) -> dict:
    """One probe: 5-fold CV accuracy (classification) or R^2 (regression)."""
    scores = cross_val_score(
        _pipeline(task),
        X,
        y,
        cv=_splitter(task, groups),
        groups=groups,
        scoring="accuracy" if task == "classification" else "r2",
    )
    return {"mean": float(scores.mean()), "std": float(scores.std()),
            "folds": [float(s) for s in scores]}


def probe_sweep(
    pooled: torch.Tensor,        # [n_layers+1, n_clips, d_model]
    pooled_logmel: torch.Tensor,  # [n_clips, n_mels]
    y: np.ndarray,
    task: str,
    groups: np.ndarray | None = None,
    shuffle_control: bool = False,
) -> dict:
    """Probe every layer plus the raw log-mel input (layer-0 control).

    Returns {"logmel": result, "layers": [result_per_layer 0..L],
             and, if requested, the same under "shuffled_*"}.
    """
    rng = np.random.default_rng(SEED)
    out = {
        "logmel": run_probe(pooled_logmel.numpy(), y, task, groups),
        "layers": [
            run_probe(pooled[k].numpy(), y, task, groups)
            for k in range(pooled.shape[0])
        ],
    }
    if shuffle_control:
        y_shuf = rng.permutation(y)
        out["shuffled_logmel"] = run_probe(pooled_logmel.numpy(), y_shuf, task, groups)
        out["shuffled_layers"] = [
            run_probe(pooled[k].numpy(), y_shuf, task, groups)
            for k in range(pooled.shape[0])
        ]
    return out


def run_low_data_probe(
    X: np.ndarray, y: np.ndarray, n_per_class: int, n_repeats: int = 10
) -> dict:
    """Sample-efficiency probe: train on only n_per_class examples per class,
    evaluate on all the rest; repeat with different random draws.

    Full-data linear probes saturate at 100% for easy properties and layer
    differences vanish; in the low-data regime only layers where the property
    is strongly and robustly linearly encoded reach high accuracy."""
    rng = np.random.default_rng(SEED)
    classes = np.unique(y)
    accs = []
    # each repeat draws a fresh tiny train set (n_per_class indices per
    # class, without replacement) and evaluates on ALL remaining clips;
    # repeating averages out the luck of a single draw
    for _ in range(n_repeats):
        train_idx = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == c),
                           size=min(n_per_class, (y == c).sum()), replace=False)
                for c in classes
            ]
        )
        mask = np.zeros(len(y), dtype=bool)   # True = train, False = eval
        mask[train_idx] = True
        pipe = _pipeline("classification")
        pipe.fit(X[mask], y[mask])
        accs.append(pipe.score(X[~mask], y[~mask]))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "folds": [float(a) for a in accs]}


def low_data_sweep(
    pooled: torch.Tensor,
    pooled_logmel: torch.Tensor,
    y: np.ndarray,
    n_per_class: int,
    n_repeats: int = 10,
) -> dict:
    """run_low_data_probe over every layer + the raw log-mel input."""
    return {
        "logmel": run_low_data_probe(pooled_logmel.numpy(), y, n_per_class, n_repeats),
        "layers": [
            run_low_data_probe(pooled[k].numpy(), y, n_per_class, n_repeats)
            for k in range(pooled.shape[0])
        ],
    }


def chance_level(y: np.ndarray, task: str) -> float:
    """Majority-class rate (classification) or 0.0 (regression R^2)."""
    if task != "classification":
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())
