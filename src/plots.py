"""
Figures

Figure 1: property decodability vs. encoder layer: one curve per property,
with chance levels (dashed), shuffled-label controls (dotted), and the raw
log-mel input probe as the leftmost point ("mel")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "language": "tab:blue",
    "gender": "tab:orange",
    "speaker": "tab:green",
    "f0": "tab:red",
}


def figure_probe_curves(tasks: dict, out_path: str | Path, model_name: str):
    """tasks: {name: {"label", "color_key", "chance", "sweep": probe_sweep dict}}"""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, t in tasks.items():
        sweep = t["sweep"]
        color = COLORS[t["color_key"]]
        n_layers = len(sweep["layers"])
        # x = -1 for the raw log-mel input, 0..L for embedding + blocks
        xs = list(range(-1, n_layers))
        means = [sweep["logmel"]["mean"]] + [r["mean"] for r in sweep["layers"]]
        stds = [sweep["logmel"]["std"]] + [r["std"] for r in sweep["layers"]]
        ax.errorbar(xs, means, yerr=stds, marker="o", ms=3.5, lw=1.5,
                    color=color, label=t["label"], capsize=2)
        ax.axhline(t["chance"], color=color, ls="--", lw=0.8, alpha=0.5)
        if "shuffled_layers" in sweep:
            shuf = [sweep["shuffled_logmel"]["mean"]] + [
                r["mean"] for r in sweep["shuffled_layers"]
            ]
            ax.plot(xs, shuf, ls=":", lw=1.0, color=color, alpha=0.6)

    ax.set_xlabel("encoder layer (mel = raw log-mel input, 0 = post-conv embedding)")
    ax.set_ylabel("probe accuracy / $R^2$ (5-fold CV)")
    ax.set_title(f"Property decodability across {model_name} encoder layers")
    n_layers = max(len(t["sweep"]["layers"]) for t in tasks.values())
    ax.set_xticks(list(range(-1, n_layers)))
    ax.set_xticklabels(["mel"] + [str(k) for k in range(n_layers)])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def figure_steering(directions: dict, out_path: str | Path, model_name: str,
                    layer: int):
    """Figure 3. directions: {"en->it": {"alphas": [...], "flip_rate": [...],
    "wer": [...], "wer_alpha0": float}, ...}; the control-vs-damage curve."""
    n_dir = len(directions)
    fig, axes = plt.subplots(1, n_dir, figsize=(5.5 * n_dir, 4), squeeze=False)
    for j, (name, d) in enumerate(directions.items()):
        ax = axes[0][j]
        ax.plot(d["alphas"], d["flip_rate"], "o-", color="tab:blue",
                label="language flip rate")
        ax.plot(d["alphas"], d["wer"], "s--", color="tab:red",
                label="WER vs reference")
        ax.axhline(d["wer_alpha0"], color="tab:red", ls=":", lw=1, alpha=0.6,
                   label="baseline WER")
        ax.set_xlabel(r"steering strength $\alpha$")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Steering at block {layer}, {model_name}")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def figure_patching(directions: dict, out_path: str | Path, model_name: str):
    """Figure 2. directions: {"en->it": {"delta": [n_pairs, n_blocks],
    "flip": [n_pairs, n_blocks], "blocks": [1..L]}, ...}
    Top row: per-pair delta-logit heatmap; bottom: mean delta + flip rate."""
    n_dir = len(directions)
    fig, axes = plt.subplots(2, n_dir, figsize=(6 * n_dir, 7),
                             gridspec_kw={"height_ratios": [2, 1]}, squeeze=False)
    vmax = max(np.abs(d["delta"]).max() for d in directions.values())

    for j, (name, d) in enumerate(directions.items()):
        delta, flip, blocks = (np.asarray(d["delta"]), np.asarray(d["flip"]),
                               d["blocks"])
        ax = axes[0][j]
        im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax,
                       vmax=vmax, extent=(blocks[0] - 0.5, blocks[-1] + 0.5,
                                          delta.shape[0], 0))
        ax.set_title(f"{name}: Δ language logit per patched block")
        ax.set_ylabel("pair")
        fig.colorbar(im, ax=ax, fraction=0.03)

        ax2 = axes[1][j]
        ax2.plot(blocks, delta.mean(axis=0), "o-", color="tab:red",
                 label="mean Δ logit")
        ax2.set_xlabel("patched encoder block")
        ax2.set_ylabel("mean Δ logit", color="tab:red")
        ax2.grid(alpha=0.25)
        ax3 = ax2.twinx()
        ax3.plot(blocks, flip.mean(axis=0), "s--", color="tab:blue",
                 label="flip rate")
        ax3.set_ylabel("flip rate", color="tab:blue")
        ax3.set_ylim(-0.02, 1.02)

    fig.suptitle(f"Block-update patching, {model_name} encoder")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path
