"""
Run once per dataset config:
  uv run python experiments/e1_extract.py --config configs/e1_fleurs.yaml
  uv run python experiments/e1_extract.py --config configs/e1_libri.yaml
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_clips
from src.extract import extract_clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    print(f"[E1x] model={cfg.model_name} device={cfg.device} dtype={cfg.dtype}")
    t0 = time.time()
    clips = load_clips(cfg.dataset)
    print(f"[E1x] loaded {len(clips)} clips in {time.time() - t0:.0f}s")

    t0 = time.time()
    info = extract_clips(cfg, clips)
    print(f"[E1x] extracted {info['n_clips']} clips x "
          f"{info['n_layers'] + 1} layers x {info['d_model']} dims "
          f"in {time.time() - t0:.0f}s -> {info['out_dir']}")


if __name__ == "__main__":
    main()
