"""E5: patching refinements at the most causal block (PLAN §4C refinements).

All conditions swap parts of block k's contribution (it->en pairs, k=8 from
E3) and measure delta language logit + flip, as in E3:

  * time segments: the update patch applied only on quarter-windows of the
    speech region, plus a padding-only control: WHERE in time is the
    language evidence?
  * sublayers: swap only the self-attention contribution, or only the MLP
    contribution, of block k.
  * heads: swap one attention head's contribution at a time (out_proj is
    linear, so head h's contribution is its channel slice pushed through the
    corresponding rows of the projection).

Outputs: results/e5_patch_refine.json + results/figures/fig5_refine.png

Run:  uv run python experiments/e5_patch_refine.py --config configs/e5.yaml
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
from src.extract import n_valid_frames
from src.patching import WHISPER_LANG, PatchedWhisper, make_pairs, top_lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg.patching
    k = p["block"]

    print(f"[E5] model={cfg.model_name} device={cfg.device} block={k}")
    clips = load_clips(cfg.dataset)
    by_lang = {}
    for c in clips:
        by_lang.setdefault(c["language"], []).append(c)
    tgt_lang, src_lang = p["direction"]
    tl, sl = WHISPER_LANG[tgt_lang], WHISPER_LANG[src_lang]
    pairs = make_pairs(by_lang[tgt_lang], by_lang[src_lang], p["n_pairs"],
                       p["duration_tol"], p.get("match_gender", True))
    print(f"[E5] {tl}->{sl}: {len(pairs)} pairs")

    pw = PatchedWhisper(cfg)
    layer = pw.encoder.layers[k - 1]
    n_heads = pw.model.config.encoder_attention_heads
    head_dim = pw.model.config.d_model // n_heads

    # extra captures at block k: attn contribution, mlp contribution,
    # out_proj input (concatenated head outputs)
    extra = {}
    layer.self_attn.register_forward_hook(
        lambda m, a, o: extra.__setitem__("attn", (o[0] if isinstance(o, tuple)
                                                   else o).detach()))
    layer.fc2.register_forward_hook(
        lambda m, a, o: extra.__setitem__("mlp", o.detach()))
    layer.self_attn.out_proj.register_forward_pre_hook(
        lambda m, a: extra.__setitem__("heads_in", a[0].detach()))

    def capture_all(feats):
        caps = pw.capture_layers(feats)
        return caps, {n: extra[n].clone() for n in ("attn", "mlp", "heads_in")}

    W = pw.model.model.encoder.layers[k - 1].self_attn.out_proj.weight  # [d, d]

    conditions = ([f"Q{q + 1}" for q in range(p["n_segments"])]
                  + ["padding", "full", "attn", "mlp"]
                  + [f"head{h}" for h in range(n_heads)])
    delta = np.zeros((len(pairs), len(conditions)))
    flip = np.zeros_like(delta, dtype=bool)
    t0 = time.time()

    for i, (tgt, src) in enumerate(pairs):
        feats_t, feats_s = pw.features(tgt["audio"]), pw.features(src["audio"])
        cap_t, ex_t = capture_all(feats_t)
        cap_s, ex_s = capture_all(feats_s)
        base = pw.lang_logits(feats_t)
        d_base = base[sl] - base[tl]
        valid = min(n_valid_frames(len(tgt["audio"]))[1],
                    n_valid_frames(len(src["audio"]))[1])
        full_update = cap_t[k - 1] + (cap_s[k] - cap_s[k - 1])

        repl = {}
        # time segments: update only inside the window, own output elsewhere
        bounds = np.linspace(0, valid, p["n_segments"] + 1).astype(int)
        for q in range(p["n_segments"]):
            r = cap_t[k].clone()
            r[:, bounds[q]:bounds[q + 1]] = full_update[:, bounds[q]:bounds[q + 1]]
            repl[f"Q{q + 1}"] = r
        pad0 = max(n_valid_frames(len(tgt["audio"]))[1],
                   n_valid_frames(len(src["audio"]))[1])
        r = cap_t[k].clone()
        r[:, pad0:] = full_update[:, pad0:]
        repl["padding"] = r
        repl["full"] = full_update
        # sublayer swaps
        repl["attn"] = cap_t[k] + (ex_s["attn"] - ex_t["attn"])
        repl["mlp"] = cap_t[k] + (ex_s["mlp"] - ex_t["mlp"])
        # head swaps: delta of head h's channels pushed through out_proj rows
        dx = ex_s["heads_in"] - ex_t["heads_in"]
        for h in range(n_heads):
            dh = torch.zeros_like(dx)
            sl_ = slice(h * head_dim, (h + 1) * head_dim)
            dh[..., sl_] = dx[..., sl_]
            repl[f"head{h}"] = cap_t[k] + dh @ W.T

        for j, cond in enumerate(conditions):
            pl = pw.lang_logits(feats_t, patch=(k, repl[cond]))
            delta[i, j] = (pl[sl] - pl[tl]) - d_base
            flip[i, j] = top_lang(pl) == sl

    print(f"[E5] swept {len(pairs)} pairs x {len(conditions)} conditions "
          f"in {time.time() - t0:.0f}s")
    for j, cond in enumerate(conditions):
        print(f"[E5] {cond:8s} delta={delta[:, j].mean():+6.2f} "
              f"flip={flip[:, j].mean():.2f}")

    results = {
        "model": cfg.model_name, "block": k, "direction": f"{tl}->{sl}",
        "n_pairs": len(pairs), "conditions": conditions,
        "mean_delta": delta.mean(axis=0).tolist(),
        "flip_rate": flip.mean(axis=0).tolist(),
        "delta": delta.tolist(),
    }
    out = Path("results")
    (out / "e5_patch_refine.json").write_text(json.dumps(results, indent=2))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(conditions))
    ax.bar(x - 0.2, delta.mean(axis=0), 0.4, color="tab:red",
           label="mean Δ logit")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, flip.mean(axis=0), 0.4, color="tab:blue",
            label="flip rate")
    ax2.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("mean Δ logit", color="tab:red")
    ax2.set_ylabel("flip rate", color="tab:blue")
    ax.set_title(f"Partial patches of block {k} ({tl}->{sl}), {cfg.model_name}")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fp = out / "figures" / "fig5_refine.png"
    fig.savefig(fp, dpi=200)
    fig.savefig(fp.with_suffix(".pdf"))
    print(f"[E5] wrote results/e5_patch_refine.json and {fp}")


if __name__ == "__main__":
    main()
