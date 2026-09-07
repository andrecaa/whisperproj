"""
E3: language patching sweep

For every en<->it pair and every encoder block k, replaces block k's residual
contribution in the target's forward pass with the source's and records: 
1. delta = [logit(src_lang) - logit(tgt_lang)]_patched - [same]_baseline
2. flip  = does the top language token become the source language?

Outputs: results/e3_patch_lang.json + results/figures/fig2_patching.png

Run:  uv run python experiments/e3_patch_lang.py --config configs/e3.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_clips
from src.patching import (WHISPER_LANG, PatchedWhisper, block_update_patch,
                          make_pairs, top_lang)
from src.plots import figure_patching


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg.patching

    print(f"[E3] model={cfg.model_name} device={cfg.device} dtype={cfg.dtype}")
    clips = load_clips(cfg.dataset)
    by_lang = {}
    for c in clips:
        by_lang.setdefault(c["language"], []).append(c)

    pw = PatchedWhisper(cfg)
    blocks = list(range(1, pw.n_layers + 1))
    results = {"model": cfg.model_name, "blocks": blocks, "directions": {}}
    fig_dirs = {}

    for tgt_lang, src_lang in p["directions"]:
        tl, sl = WHISPER_LANG[tgt_lang], WHISPER_LANG[src_lang]
        name = f"{tl}->{sl}"
        pairs = make_pairs(by_lang[tgt_lang], by_lang[src_lang], p["n_pairs"],
                           p["duration_tol"], p.get("match_gender", True))
        print(f"[E3] {name}: {len(pairs)} pairs")

        # result matrices: one row per pair, one column per patched block
        delta = np.zeros((len(pairs), len(blocks)))
        flip = np.zeros_like(delta, dtype=bool)
        full_flip = np.zeros(len(pairs), dtype=bool)
        base_top = []
        t0 = time.time()
        # per pair: capture both clips' layer activations once, get the
        # unpatched baseline, then re-run the target once per patched block
        for i, (tgt, src) in enumerate(pairs):
            feats_t, feats_s = pw.features(tgt["audio"]), pw.features(src["audio"])
            cap_t = pw.capture_layers(feats_t)
            cap_s = pw.capture_layers(feats_s)
            base = pw.lang_logits(feats_t)
            base_top.append(top_lang(base))
            d_base = base[sl] - base[tl]
            # sanity upper bound: full-state replacement == running the source
            full = pw.lang_logits(feats_t, patch=(pw.n_layers,
                                                  cap_s[pw.n_layers]))
            full_flip[i] = top_lang(full) == sl
            for j, k in enumerate(blocks):
                pl = pw.lang_logits(
                    feats_t, patch=(k, block_update_patch(cap_t, cap_s, k)))
                delta[i, j] = (pl[sl] - pl[tl]) - d_base
                flip[i, j] = top_lang(pl) == sl
        print(f"[E3] {name}: swept {len(pairs)} pairs x {len(blocks)} blocks "
              f"in {time.time() - t0:.0f}s; "
              f"full-replacement flip rate {full_flip.mean():.2f}; "
              f"baseline top-lang correct "
              f"{np.mean([b == tl for b in base_top]):.2f}")

        best = int(np.argmax(delta.mean(axis=0)))
        print(f"[E3] {name}: most causal block {blocks[best]} "
              f"(mean delta {delta.mean(axis=0)[best]:+.2f}, "
              f"flip rate {flip.mean(axis=0)[best]:.2f})")

        # sample transcriptions at the most causal block
        samples = []
        for tgt, src in pairs[: p.get("n_transcribe", 0)]:
            feats_t = pw.features(tgt["audio"])
            cap_t = pw.capture_layers(feats_t)
            cap_s = pw.capture_layers(pw.features(src["audio"]))
            k = blocks[best]
            samples.append({
                "target_text": tgt["text"][:120],
                "source_text": src["text"][:120],
                "baseline": pw.transcribe(feats_t),
                "patched": pw.transcribe(
                    feats_t, patch=(k, block_update_patch(cap_t, cap_s, k))),
            })

        results["directions"][name] = {
            "n_pairs": len(pairs),
            "mean_delta": delta.mean(axis=0).tolist(),
            "flip_rate": flip.mean(axis=0).tolist(),
            "full_replacement_flip_rate": float(full_flip.mean()),
            "most_causal_block": blocks[best],
            "delta": delta.tolist(),
            "flip": flip.tolist(),
            "transcription_samples": samples,
        }
        fig_dirs[name] = {"delta": delta, "flip": flip, "blocks": blocks}

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "e3_patch_lang.json").write_text(json.dumps(results, indent=2))
    fig = figure_patching(fig_dirs, out / "figures" / "fig2_patching.png",
                          cfg.model_name)
    print(f"[E3] wrote results/e3_patch_lang.json and {fig}")


if __name__ == "__main__":
    main()
