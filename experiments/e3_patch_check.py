"""
E3 gate: hand-verify ONE patched forward pass before any sweep

Checks on a single en/it pair: self-patch is a no-op, cross-patch changes it,
and the block-update patch's effect varies by layer

Run:  uv run python experiments/e3_patch_check.py --config configs/e3_debug.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_clips
from src.patching import (WHISPER_LANG, PatchedWhisper, block_update_patch,
                          make_pairs, top_lang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg.patching

    print(f"[E3c] model={cfg.model_name} device={cfg.device} dtype={cfg.dtype}")
    clips = load_clips(cfg.dataset)
    (tgt_lang, src_lang) = p["directions"][0]
    targets = [c for c in clips if c["language"] == tgt_lang]
    sources = [c for c in clips if c["language"] == src_lang]
    pair = make_pairs(targets, sources, 1, p["duration_tol"], p["match_gender"])[0]
    tgt, src = pair
    tl, sl = WHISPER_LANG[tgt_lang], WHISPER_LANG[src_lang]
    print(f"[E3c] pair: target {tgt['id']} ({tgt['duration']:.1f}s {tl}) <- "
          f"source {src['id']} ({src['duration']:.1f}s {sl})")

    pw = PatchedWhisper(cfg)
    feats_t, feats_s = pw.features(tgt["audio"]), pw.features(src["audio"])
    own = pw.capture_layers(feats_t)
    other = pw.capture_layers(feats_s)
    base = pw.lang_logits(feats_t)
    d_base = base[sl] - base[tl]

    # 1 + 2: self-patch no-op, cross-patch changes at EVERY layer
    for k in range(pw.n_layers + 1):
        self_p = pw.lang_logits(feats_t, patch=(k, own[k]))
        assert all(self_p[c] == base[c] for c in base), (
            f"layer {k}: self-patch changed logits {self_p} vs {base}"
        )
        cross = pw.lang_logits(feats_t, patch=(k, other[k]))
        assert any(cross[c] != base[c] for c in base), (
            f"layer {k}: cross-patch had NO effect"
        )
    print(f"[E3c] 1. self-patch bit-identical at all {pw.n_layers + 1} layers")
    print(f"[E3c] 2. cross-patch changes logits at all layers")

    # eyeball the per-layer effect and the transcriptions
    print(f"[E3c] 3. baseline top-lang={top_lang(base)}  "
          f"d = logit({sl})-logit({tl}) = {d_base:.2f}")
    for k in range(pw.n_layers + 1):
        cross = pw.lang_logits(feats_t, patch=(k, other[k]))
        d = cross[sl] - cross[tl]
        print(f"       layer {k:2d}: delta={d - d_base:+7.2f}  "
              f"top={top_lang(cross)}")
    # block-update mode (the sweep's real patch); self version must be a
    # near-no-op (float rounding only), cross version must vary BY LAYER
    for k in range(1, pw.n_layers + 1):
        self_u = pw.lang_logits(
            feats_t, patch=(k, block_update_patch(own, own, k)))
        assert top_lang(self_u) == top_lang(base), f"self-update flipped @{k}"
        assert max(abs(self_u[c] - base[c]) for c in base) < 0.05, (
            f"layer {k}: self block-update moved logits too much: "
            f"{ {c: self_u[c] - base[c] for c in base} }"
        )
    print(f"[E3c] 4. self block-update is a near-no-op at all blocks; "
          f"cross block-update per layer:")
    for k in range(1, pw.n_layers + 1):
        cross_u = pw.lang_logits(
            feats_t, patch=(k, block_update_patch(own, other, k)))
        d = cross_u[sl] - cross_u[tl]
        print(f"       block {k:2d}: delta={d - d_base:+7.2f}  "
              f"top={top_lang(cross_u)}")

    k_last = pw.n_layers // 2
    print(f"[E3c] transcription baseline : {pw.transcribe(feats_t)!r}")
    print(f"[E3c] transcription patched@{k_last}: "
          f"{pw.transcribe(feats_t, patch=(k_last, other[k_last]))!r}")
    print(f"[E3c] source reference       : {src['text'][:80]!r}")
    print("[E3c] PASSED")


if __name__ == "__main__":
    main()
