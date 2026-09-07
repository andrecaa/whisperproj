"""
E0: activation extraction sanity

Checks shapes (cached tensors must have the same dimensions the model dictates), 
determinism (extracting twice gives identical files), and cache==live (reloading from disk matches a fresh forward pass exactly)
for a small set of clips

Run: uv run python experiments/e0_sanity.py --config configs/e0_debug.yaml
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_clips
from src.extract import ActivationExtractor, extract_clips, load_cached, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    print(f"[E0] model={cfg.model_name} device={cfg.device} dtype={cfg.dtype}")
    clips = load_clips(cfg.dataset)
    print(f"[E0] loaded {len(clips)} clips "
          f"(durations {min(c['duration'] for c in clips):.1f}-"
          f"{max(c['duration'] for c in clips):.1f}s)")

    # -- extraction, twice ---------------------------------------------------
    info = extract_clips(cfg, clips)
    cfg2 = dataclasses.replace(cfg, out_dir=Path(str(cfg.out_dir) + "_rerun"))
    extract_clips(cfg2, clips)

    # -- 1. shapes -----------------------------------------------------------
    cache = load_cached(cfg.out_dir)
    L, D, T = info["n_layers"], info["d_model"], info["seq_len"]
    n = info["n_clips"]
    assert cache["pooled"].shape == (L + 1, n, D), cache["pooled"].shape
    assert cache["pooled_logmel"].shape == (n, info["n_mels"])
    assert cache["full"].shape == (L + 1, n, T, D), cache["full"].shape
    assert len(cache["metadata"]) == n
    print(f"[E0] 1. shapes OK: pooled {tuple(cache['pooled'].shape)}, "
          f"full {tuple(cache['full'].shape)}")

    # -- 2. determinism ------------------------------------------------------
    rerun = load_cached(cfg2.out_dir)
    for key in ("pooled", "pooled_logmel", "full"):
        assert torch.equal(cache[key], rerun[key]), f"non-deterministic: {key}"
    print("[E0] 2. determinism OK: two extraction runs are bit-identical")

    # -- 3. cache matches live forward pass ----------------------------------
    extractor = ActivationExtractor(cfg)
    max_diff = 0.0
    for i, clip in enumerate(clips):
        acts = extractor.encode(clip["audio"])
        live_pooled = pool(acts["layers"], acts["valid_enc_frames"])
        diff = (live_pooled - cache["pooled"][:, i]).abs().max().item()
        max_diff = max(max_diff, diff)
        assert torch.equal(live_pooled, cache["pooled"][:, i]), (
            f"clip {i}: cached pooled != live (max abs diff {diff:.3e})"
        )
        assert torch.equal(acts["layers"].half(), cache["full"][:, i]), (
            f"clip {i}: cached full != live"
        )
    print(f"[E0] 3. cache==live OK (max abs diff {max_diff:.1e})")

    # -- format summary ------------------------------------------------------
    md = cache["metadata"]
    print("\n[E0] PASSED. Cached-activation format:")
    print(f"  {cfg.out_dir}/")
    print(f"    pooled_layer00..{L:02d}.pt  [{n}, {D}] float32  "
          f"(00 = post-conv embedding; k = output of encoder block k)")
    print(f"    pooled_logmel.pt          [{n}, {info['n_mels']}] float32  "
          f"(layer-0 control probe input)")
    print(f"    full_layer00..{L:02d}.pt    [{n}, {T}, {D}] float16")
    print(f"    metadata.parquet          columns: {list(md.columns)}")
    print("  pooling = mean over valid (non-padding) encoder frames only")
    print(md.drop(columns="text").to_string(index=False))


if __name__ == "__main__":
    main()
