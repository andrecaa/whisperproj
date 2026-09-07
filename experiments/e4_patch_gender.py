"""
E4: gender patching

Same-language (English) pairs differing in speaker gender, block-update
patching as in E3. Whisper exposes no gender in its output, so two metrics:
1. probe flip: a logistic gender probe (trained on held-out clips' pooled
FINAL-layer activations) read out on the patched final state: does the
target clip now read as the source's gender?
2. disruption: WER between baseline and patched transcriptions per block:
does moving gender through block k change what Whisper actually says?

Outputs: results/e4_patch_gender.json + results/figures/fig4_gender_patching.png

Run:  uv run python experiments/e4_patch_gender.py --config configs/e4.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_clips
from src.extract import load_cached, n_valid_frames
from src.patching import PatchedWhisper, block_update_patch, make_pairs
from src.steering import wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg.patching

    print(f"[E4] model={cfg.model_name} device={cfg.device} dtype={cfg.dtype}")
    clips = load_clips(cfg.dataset)
    males = [c for c in clips if c["gender"] == "male"]
    females = [c for c in clips if c["gender"] == "female"]

    # gender probe on held-out clips' pooled final-layer activations
    cache = load_cached(p["probe_cache"])
    md = cache["metadata"]
    lo, hi = p["probe_train_slice"]
    idx = md.index[md["language"] == "en_us"][lo:hi]
    X = cache["pooled"][-1][list(idx)].numpy()
    y = md.loc[idx, "gender"].to_numpy()
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, random_state=0))
    probe.fit(X, y)
    print(f"[E4] gender probe trained on {len(y)} held-out clips "
          f"(train acc {probe.score(X, y):.2f})")

    # mixed-direction pairs: female targets with male sources and vice versa
    half = p["n_pairs"] // 2
    pairs = (make_pairs(females[:half * 2], males, half, p["duration_tol"],
                        match_gender=False)
             + make_pairs(males[:half * 2], females, half, p["duration_tol"],
                          match_gender=False))
    print(f"[E4] {len(pairs)} pairs")

    pw = PatchedWhisper(cfg)
    blocks = list(range(1, pw.n_layers + 1))
    flip = np.zeros((len(pairs), len(blocks)), dtype=bool)
    baseline_ok = []
    disruption = np.zeros((min(p["n_transcribe"], len(pairs)), len(blocks)))
    t0 = time.time()

    def final_pooled(tgt_clip):
        """Pooled final-layer state of the CURRENT captured pass."""
        valid = n_valid_frames(len(tgt_clip["audio"]))[1]
        return (pw._captured[pw.n_layers][0, :valid].float().mean(dim=0)
                .cpu().numpy()[None, :])

    for i, (tgt, src) in enumerate(pairs):
        feats_t = pw.features(tgt["audio"])
        cap_t = pw.capture_layers(feats_t)
        baseline_ok.append(probe.predict(final_pooled(tgt))[0] == tgt["gender"])
        cap_s = pw.capture_layers(pw.features(src["audio"]))
        base_text = pw.transcribe(feats_t) if i < disruption.shape[0] else None
        for j, k in enumerate(blocks):
            patch = (k, block_update_patch(cap_t, cap_s, k))
            pw.encode(feats_t, patch=patch)
            flip[i, j] = probe.predict(final_pooled(tgt))[0] == src["gender"]
            if base_text is not None:
                disruption[i, j] = wer(base_text,
                                       pw.transcribe(feats_t, patch=patch))

    print(f"[E4] swept in {time.time() - t0:.0f}s; baseline probe acc on "
          f"targets {np.mean(baseline_ok):.2f}")
    print(f"[E4] probe-flip rate per block: "
          + " ".join(f"{x:.2f}" for x in flip.mean(axis=0)))
    print(f"[E4] mean WER disruption per block: "
          + " ".join(f"{x:.2f}" for x in disruption.mean(axis=0)))

    results = {
        "model": cfg.model_name,
        "blocks": blocks,
        "n_pairs": len(pairs),
        "baseline_probe_acc": float(np.mean(baseline_ok)),
        "probe_flip_rate": flip.mean(axis=0).tolist(),
        "wer_disruption": disruption.mean(axis=0).tolist(),
        "flip": flip.tolist(),
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "e4_patch_gender.json").write_text(json.dumps(results, indent=2))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(blocks, flip.mean(axis=0), "o-", color="tab:orange",
            label="gender probe flip rate")
    ax.plot(blocks, disruption.mean(axis=0), "s--", color="tab:gray",
            label="WER disruption (baseline vs patched)")
    ax.set_xlabel("patched encoder block")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(f"Gender block-update patching, {cfg.model_name}")
    fig.tight_layout()
    fp = out / "figures" / "fig4_gender_patching.png"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=200)
    fig.savefig(fp.with_suffix(".pdf"))
    print(f"[E4] wrote results/e4_patch_gender.json and {fp}")


if __name__ == "__main__":
    main()
