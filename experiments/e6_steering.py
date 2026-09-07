"""E6: steering vectors: flip rate & WER vs alpha (Phase D, PLAN §4D).

v = mean(lang_b) - mean(lang_a) pooled activations at the most causal block
(E3: block 8), computed on held-out clips. alpha * v is added to that block's
output on evaluation clips of lang_a; for each alpha we record:
  * flip rate  : fraction of clips whose predicted language token becomes b
  * WER        : the steered transcription vs. the reference text
    (the control-vs-damage tradeoff curve is itself the result)
  * mean delta logit(b) - logit(a) relative to alpha=0 (compare to patching)

Outputs: results/e6_steering.json + results/figures/fig3_steering.png

Run:  uv run python experiments/e6_steering.py --config configs/e6.yaml
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
from src.extract import load_cached
from src.patching import WHISPER_LANG, PatchedWhisper, top_lang
from src.plots import figure_steering
from src.steering import steering_vector, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = cfg.steering
    layer, alphas = s["layer"], s["alphas"]

    print(f"[E6] model={cfg.model_name} device={cfg.device} dtype={cfg.dtype} "
          f"block={layer}")
    cache = load_cached(s["cache_dir"])
    md = cache["metadata"]
    clips = load_clips(cfg.dataset)
    by_lang = {}
    for c in clips:
        by_lang.setdefault(c["language"], []).append(c)

    pw = PatchedWhisper(cfg)
    results = {"model": cfg.model_name, "layer": layer, "alphas": alphas,
               "directions": {}}
    fig_dirs = {}

    for lang_a, lang_b in s["directions"]:
        a, b = WHISPER_LANG[lang_a], WHISPER_LANG[lang_b]
        name = f"{a}->{b}"
        vec = steering_vector(cache["pooled"], md, layer, lang_a, lang_b,
                              slice(*s["vector_slice"]))
        print(f"[E6] {name}: |v| = {vec.norm():.2f} "
              f"(from clips {s['vector_slice']} per language)")
        v_dev = vec.to(cfg.device, cfg.torch_dtype)

        eval_clips = by_lang[lang_a][: s["eval_n"]]
        feats = [pw.features(c["audio"]) for c in eval_clips]

        flip_rate, mean_delta, wers = [], [], []
        samples = {}
        t0 = time.time()
        base_d = None   # logit gap at alpha=0, the reference for mean_delta
        # outer loop: one steering strength per iteration (alpha=0 is the
        # unsteered baseline); inner loop: language logits for every eval clip
        for alpha in alphas:
            steer = (layer, alpha * v_dev) if alpha else None
            flips, deltas = [], []
            for f in feats:
                ll = pw.lang_logits(f, steer=steer)
                flips.append(top_lang(ll) == b)
                deltas.append(ll[b] - ll[a])
            texts = [pw.transcribe(f, steer=steer)
                     for f in feats[: s["transcribe_n"]]]
            ws = [wer(c["text"], t)
                  for c, t in zip(eval_clips[: s["transcribe_n"]], texts)]
            if base_d is None:
                base_d = float(np.mean(deltas))
            flip_rate.append(float(np.mean(flips)))
            mean_delta.append(float(np.mean(deltas) - base_d))
            wers.append(float(np.mean(ws)))
            samples[alpha] = texts[0]
            print(f"[E6] {name} alpha={alpha:5.1f}: flip={flip_rate[-1]:.2f} "
                  f"delta={mean_delta[-1]:+6.2f} wer={wers[-1]:.2f}")
        print(f"[E6] {name}: done in {time.time() - t0:.0f}s")

        results["directions"][name] = {
            "vector_norm": float(vec.norm()),
            "n_eval": len(eval_clips),
            "n_transcribe": s["transcribe_n"],
            "flip_rate": flip_rate,
            "mean_delta": mean_delta,
            "wer": wers,
            "sample_transcriptions": samples,
        }
        fig_dirs[name] = {"alphas": alphas, "flip_rate": flip_rate,
                          "wer": wers, "wer_alpha0": wers[0]}

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "e6_steering.json").write_text(json.dumps(results, indent=2))
    fig = figure_steering(fig_dirs, out / "figures" / "fig3_steering.png",
                          cfg.model_name, layer)
    print(f"[E6] wrote results/e6_steering.json and {fig}")


if __name__ == "__main__":
    main()
