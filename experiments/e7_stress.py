"""E7: steering stress test (PLAN §4D natural stress test).

Part A, code-switched audio: synthetic clips made by concatenating 5 s of
English and 5 s of Italian (both orders). Measures Whisper's baseline choice
on ambiguous input and the alpha needed to steer it either way, compared
with the pure-clip thresholds from E6.

Part B, related languages: the it-en steering vector applied to French and
Spanish clips. Does +alpha*v (toward Italian) capture Romance neighbors more
easily than -alpha*v (toward English)? Is the direction language-specific?

Outputs: results/e7_stress.json + results/figures/fig6_stress.png

Run:  uv run python experiments/e7_stress.py --config configs/e7.yaml
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
from src.extract import load_cached
from src.patching import WHISPER_LANG, PatchedWhisper, make_pairs, top_lang
from src.steering import steering_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = cfg.steering
    layer, alphas = s["layer"], s["alphas"]

    print(f"[E7] model={cfg.model_name} device={cfg.device} block={layer}")
    cache = load_cached(s["cache_dir"])
    clips = load_clips(cfg.dataset)
    by_lang = {}
    for c in clips:
        by_lang.setdefault(c["language"], []).append(c)

    pw = PatchedWhisper(cfg)
    v = steering_vector(cache["pooled"], cache["metadata"], layer,
                        "en_us", "it_it", slice(*s["vector_slice"]))
    v_dev = v.to(cfg.device, cfg.torch_dtype)
    results = {"model": cfg.model_name, "layer": layer, "alphas": alphas}

    # ---- Part A: code-switched clips --------------------------------------
    n_half = int(s["concat_seconds"] * 16_000)
    pairs = make_pairs(by_lang["en_us"], by_lang["it_it"], s["n_concat"],
                       duration_tol=10.0, match_gender=True)
    results["codeswitched"] = {}
    for order in ("en+it", "it+en"):
        feats = []
        for en, it in pairs:
            a, b = (en, it) if order == "en+it" else (it, en)
            audio = np.concatenate([a["audio"][:n_half], b["audio"][:n_half]])
            feats.append(pw.features(audio))
        base_top = [top_lang(pw.lang_logits(f)) for f in feats]
        counts = {l: base_top.count(l) / len(base_top)
                  for l in set(base_top)}
        row = {"baseline_top": counts, "to_it": [], "to_en": []}
        for alpha in alphas:
            for key, sign in (("to_it", +1), ("to_en", -1)):
                steer = (layer, sign * alpha * v_dev) if alpha else None
                tops = [top_lang(pw.lang_logits(f, steer=steer)) for f in feats]
                goal = "it" if sign > 0 else "en"
                row[key].append(sum(t == goal for t in tops) / len(tops))
        results["codeswitched"][order] = row
        print(f"[E7] concat {order}: baseline {counts}")
        print(f"       to_it {row['to_it']}")
        print(f"       to_en {row['to_en']}")

    # ---- Part B: related languages ----------------------------------------
    results["related"] = {}
    for lang in ("fr_fr", "es_419"):
        code = WHISPER_LANG[lang]
        feats = [pw.features(c["audio"])
                 for c in by_lang[lang][: s["n_related"]]]
        row = {"native_kept": [], "to_it": [], "to_en": []}
        for alpha in alphas:
            for key, sign in (("to_it", +1), ("to_en", -1)):
                steer = (layer, sign * alpha * v_dev) if alpha else None
                tops = [top_lang(pw.lang_logits(f, steer=steer)) for f in feats]
                goal = "it" if sign > 0 else "en"
                row[key].append(sum(t == goal for t in tops) / len(tops))
            row["native_kept"].append(
                sum(top_lang(pw.lang_logits(f)) == code for f in feats)
                / len(feats) if alpha == alphas[0] else None)
        row["native_kept"] = row["native_kept"][0]
        results["related"][code] = row
        print(f"[E7] {code}: native baseline {row['native_kept']:.2f}  "
              f"to_it {row['to_it']}  to_en {row['to_en']}")

    out = Path("results")
    (out / "e7_stress.json").write_text(json.dumps(results, indent=2))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for order, ls in (("en+it", "-"), ("it+en", "--")):
        d = results["codeswitched"][order]
        axes[0].plot(alphas, d["to_it"], "o" + ls, color="tab:green",
                     label=f"{order}: flip to it")
        axes[0].plot(alphas, d["to_en"], "s" + ls, color="tab:purple",
                     label=f"{order}: flip to en")
    axes[0].set_title("code-switched clips")
    for code, color in (("fr", "tab:blue"), ("es", "tab:orange")):
        d = results["related"][code]
        axes[1].plot(alphas, d["to_it"], "o-", color=color,
                     label=f"{code}: captured by it")
        axes[1].plot(alphas, d["to_en"], "s--", color=color,
                     label=f"{code}: captured by en")
    axes[1].set_title("related languages under the it-en vector")
    for ax in axes:
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("rate")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Steering stress test at block {layer}, {cfg.model_name}")
    fig.tight_layout()
    fp = out / "figures" / "fig6_stress.png"
    fig.savefig(fp, dpi=200)
    fig.savefig(fp.with_suffix(".pdf"))
    print(f"[E7] wrote results/e7_stress.json and {fp}")


if __name__ == "__main__":
    main()
