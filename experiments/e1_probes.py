"""E1 + E2: probe sweep with controls (run as one unit, PLAN §8.2).

Probes every cached layer for: language & gender (FLEURS), speaker ID &
median F0 (LibriSpeech). Controls baked in (E2): raw log-mel input probe,
shuffled-label probe, and for F0 both speaker-disjoint and random splits.
NOTE: FLEURS publishes no speaker identities, so language/gender probes use
stratified splits; recorded in the output and to be stated in the report.

Outputs: results/e1_probes.json + results/figures/fig1_probe_curves.png

Run:  uv run python experiments/e1_probes.py \
        [--fleurs activations/e1_fleurs] [--libri activations/e1_libri]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extract import load_cached
from src.plots import figure_probe_curves
from src.probes import chance_level, low_data_sweep, probe_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs", default="activations/e1_fleurs")
    ap.add_argument("--libri", default="activations/e1_libri")
    ap.add_argument("--model-name", default="whisper-small")
    ap.add_argument("--low-n", type=int, default=10,
                    help="train examples per class for the low-data probes")
    args = ap.parse_args()

    fleurs = load_cached(args.fleurs)
    libri = load_cached(args.libri)
    results = {"model": args.model_name, "tasks": {}}
    figure_tasks = {}

    def run(name, cache, y, task, groups=None, split_note="", color_key=None,
            mask=None, shuffle_control=True):
        t0 = time.time()
        pooled, logmel = cache["pooled"], cache["pooled_logmel"]
        if mask is not None:
            pooled, logmel, y = pooled[:, mask], logmel[mask], y[mask]
            groups = groups[mask] if groups is not None else None
        sweep = probe_sweep(pooled, logmel, y, task, groups, shuffle_control)
        results["tasks"][name] = {
            "task": task,
            "n_clips": int(len(y)),
            "split": split_note,
            "chance": chance_level(y, task),
            "sweep": sweep,
        }
        best = int(np.argmax([r["mean"] for r in sweep["layers"]]))
        print(f"[E1] {name:22s} n={len(y):4d}  best layer {best:2d} "
              f"({sweep['layers'][best]['mean']:.3f})  "
              f"input probe {sweep['logmel']['mean']:.3f}  "
              f"[{time.time() - t0:.0f}s]")
        if color_key:
            figure_tasks[name] = {
                "label": name.replace("_", " "),
                "color_key": color_key,
                "chance": results["tasks"][name]["chance"],
                "sweep": sweep,
            }

    fm, lm = fleurs["metadata"], libri["metadata"]

    run("language", fleurs, fm["language"].to_numpy(), "classification",
        split_note="stratified (FLEURS has no speaker ids)", color_key="language")
    run("gender", fleurs, fm["gender"].to_numpy(), "classification",
        split_note="stratified (FLEURS has no speaker ids)", color_key="gender")
    run("speaker_id", libri, lm["speaker_id"].to_numpy(), "classification",
        split_note="stratified (disjoint impossible for speaker ID)",
        color_key="speaker")

    f0 = lm["median_f0"].to_numpy(dtype=float)
    ok = ~np.isnan(f0)
    run("median_f0", libri, f0, "regression",
        groups=lm["speaker_id"].to_numpy(), mask=ok,
        split_note="speaker-disjoint (GroupKFold)", color_key="f0")
    run("median_f0_random_split", libri, f0, "regression", mask=ok,
        split_note="random KFold (leakage comparison, E2)",
        shuffle_control=False)

    # E1 refinement: sample-efficiency probes. Full-data language/gender
    # probes saturate at 100% on every layer (kept above for the appendix);
    # in the low-data regime the layer differences become visible, so the
    # figure shows these curves for language and gender.
    for name, cache, y in [
        ("language", fleurs, fm["language"].to_numpy()),
        ("gender", fleurs, fm["gender"].to_numpy()),
    ]:
        t0 = time.time()
        sweep = low_data_sweep(cache["pooled"], cache["pooled_logmel"], y,
                               args.low_n)
        results["tasks"][f"{name}_lowdata_n{args.low_n}"] = {
            "task": "classification",
            "n_clips": int(len(y)),
            "split": f"{args.low_n} train clips/class, eval on rest, 10 repeats",
            "chance": chance_level(y, "classification"),
            "sweep": sweep,
        }
        figure_tasks[name]["label"] = f"{name} ({args.low_n}/class)"
        figure_tasks[name]["sweep"] = sweep
        best = int(np.argmax([r["mean"] for r in sweep["layers"]]))
        print(f"[E1] {name + '_lowdata':22s} n={args.low_n}/class  "
              f"best layer {best:2d} ({sweep['layers'][best]['mean']:.3f})  "
              f"input probe {sweep['logmel']['mean']:.3f}  "
              f"[{time.time() - t0:.0f}s]")

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "e1_probes.json").write_text(json.dumps(results, indent=2))
    fig = figure_probe_curves(
        figure_tasks, out / "figures" / "fig1_probe_curves.png", args.model_name
    )
    print(f"[E1] wrote results/e1_probes.json and {fig}")


if __name__ == "__main__":
    main()
