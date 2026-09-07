"""
E8: sparse autoencoder on the most causal block's activations

Collects frame-level block-8 activations across all five FLEURS languages,
trains a top-k SAE, then asks three questions:

1. Are there language-selective features? (mean activation per language,
selectivity = top language's share; top features tabulated with their max activating clips)
2. Does any single feature align with the E6 steering vector? 
3. E9-lite: what does steering DO in feature space? 

Outputs: results/e8_sae.json + results/figures/fig7_sae.png

Run:  uv run python experiments/e8_sae.py --config configs/e8.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_clips
from src.extract import load_cached, n_valid_frames
from src.patching import WHISPER_LANG, PatchedWhisper
from src.sae import train_sae
from src.steering import steering_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = cfg.sae
    layer = s["layer"]
    rng = np.random.default_rng(0)

    print(f"[E8] model={cfg.model_name} device={cfg.device} block={layer}")
    clips = load_clips(cfg.dataset)
    pw = PatchedWhisper(cfg)

    # ---- collect frame-level activations -----------------------------------
    t0 = time.time()
    # collection loop: run each clip through the encoder once and keep a
    # random sample of its speech frames (never padding frames), remembering
    # each frame's language and source clip for the analyses below
    frames, langs, clip_ids = [], [], []
    for c in clips:
        feats = pw.features(c["audio"])
        cap = pw.capture_layers(feats)
        valid = n_valid_frames(len(c["audio"]))[1]
        idx = rng.choice(valid, size=min(s["frames_per_clip"], valid),
                         replace=False)
        frames.append(cap[layer][0, idx].float().cpu())
        langs += [WHISPER_LANG[c["language"]]] * len(idx)
        clip_ids += [c["id"]] * len(idx)
    X = torch.cat(frames)
    langs = np.array(langs)
    clip_ids = np.array(clip_ids)
    print(f"[E8] collected {len(X)} frames x {X.shape[1]} dims "
          f"from {len(clips)} clips in {time.time() - t0:.0f}s")

    # ---- train -------------------------------------------------------------
    sae, stats = train_sae(X, s["n_features"], s["topk"], cfg.device,
                           s["epochs"], s["batch_size"], s["lr"])

    # feature activations over the dataset (batched)
    with torch.no_grad():
        acts = torch.cat([sae.encode(X[i:i + 8192].to(cfg.device)).cpu()
                          for i in range(0, len(X), 8192)])

    # ---- 1. language selectivity -------------------------------------------
    # selectivity: for each feature, its mean activation per language; a
    # feature whose top language holds ~100% of the total fires for that
    # language only
    lang_names = sorted(set(langs))
    mean_by_lang = np.stack([acts[langs == l].mean(dim=0).numpy()
                             for l in lang_names])          # [L, F]
    totals = mean_by_lang.sum(axis=0) + 1e-9
    selectivity = mean_by_lang.max(axis=0) / totals
    sel_lang = np.array(lang_names)[mean_by_lang.argmax(axis=0)]
    active = totals > 1e-6

    tables = {}
    for l in lang_names:
        feat_ids = np.flatnonzero((sel_lang == l) & active)
        order = feat_ids[np.argsort(-selectivity[feat_ids])][: s["top_features"]]
        rows = []
        for f in order:
            top_frames = np.argsort(-acts[:, f].numpy())[:200]
            top_clips = [str(c) for c in
                         np.unique(clip_ids[top_frames])[:5]]
            rows.append({"feature": int(f),
                         "selectivity": float(selectivity[f]),
                         "top_clips": top_clips})
        tables[l] = rows
    n_selective = int(((selectivity > 0.8) & active).sum())
    print(f"[E8] features with >80% single-language activation share: "
          f"{n_selective}/{int(active.sum())} active")

    # ---- 2. alignment with the steering vector -----------------------------
    cache = load_cached(s["steering_cache"])
    v = steering_vector(cache["pooled"], cache["metadata"], layer,
                        "en_us", "it_it", slice(*s["vector_slice"]))
    v_unit = (v / v.norm()).to(sae.W_dec.device)
    with torch.no_grad():
        cos = (sae.W_dec @ v_unit).cpu().numpy()   # decoder rows are unit-norm
    top_cos = np.argsort(-np.abs(cos))[:10]
    print("[E8] |cos(v, feature)| top-5: "
          + " ".join(f"f{f}:{cos[f]:+.2f}" for f in top_cos[:5]))

    # ---- 3. E9-lite: steering in feature space -----------------------------
    en_mask = langs == "en"
    Xen = X[torch.from_numpy(en_mask.nonzero()[0][:20000])]
    with torch.no_grad():
        a0 = torch.cat([sae.encode(Xen[i:i + 8192].to(cfg.device)).cpu()
                        for i in range(0, len(Xen), 8192)])
        Xs = Xen + s["steer_alpha"] * v
        a1 = torch.cat([sae.encode(Xs[i:i + 8192].to(cfg.device)).cpu()
                        for i in range(0, len(Xs), 8192)])
    dmean = (a1.mean(dim=0) - a0.mean(dim=0)).numpy()
    top_up = np.argsort(-dmean)[:20]
    it_selective = set(np.flatnonzero((sel_lang == "it") & active))
    overlap = sum(int(f) in it_selective for f in top_up)
    print(f"[E8] steering(+{s['steer_alpha']}v) top-20 upweighted features: "
          f"{overlap}/20 are Italian-selective")

    results = {
        "model": cfg.model_name, "layer": layer,
        "n_frames": int(len(X)), "n_features": s["n_features"],
        "topk": s["topk"], **stats,
        "n_selective_gt80": n_selective, "n_active": int(active.sum()),
        "top_features_by_language": tables,
        "steering_cos_top10": [{"feature": int(f), "cos": float(cos[f])}
                               for f in top_cos],
        "steer_top20_upweighted": [int(f) for f in top_up],
        "steer_top20_it_selective_overlap": overlap,
    }
    out = Path("results")
    (out / "e8_sae.json").write_text(json.dumps(results, indent=2))
    torch.save(sae.state_dict(), out / "e8_sae_weights.pt")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(selectivity[active], bins=40, color="tab:blue")
    axes[0].axvline(1 / len(lang_names), color="k", ls="--", lw=1,
                    label="uniform")
    axes[0].set_xlabel("language selectivity of feature")
    axes[0].set_ylabel("features")
    axes[0].legend()
    axes[1].hist(cos, bins=60, color="tab:purple")
    axes[1].set_xlabel("cos(steering vector, feature direction)")
    order = np.argsort(-np.abs(dmean))[:15]
    colors = ["tab:green" if int(f) in it_selective else "tab:gray"
              for f in order]
    axes[2].bar(range(len(order)), dmean[order], color=colors)
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels([f"f{f}" for f in order], rotation=60, fontsize=7)
    axes[2].set_ylabel(r"Δ mean activation at $+\alpha v$")
    axes[2].set_title("green = Italian-selective feature")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"SAE on block {layer} frames, {cfg.model_name} "
                 f"(FVU {stats['fvu']:.2f})")
    fig.tight_layout()
    fp = out / "figures" / "fig7_sae.png"
    fig.savefig(fp, dpi=200)
    fig.savefig(fp.with_suffix(".pdf"))
    print(f"[E8] wrote results/e8_sae.json and {fp}")


if __name__ == "__main__":
    main()
