"""
Experiment configuration: one YAML per experiment (configs/*.yaml).
Config class defined and load_config function takes a path and returns the config object.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass
class Config:
    model_name: str
    device: str
    dtype: str
    out_dir: Path
    store_full: bool  # also cache full-sequence activations (patching needs them), not just pooled
    dataset: dict = field(default_factory=dict)
    patching: dict = field(default_factory=dict)
    steering: dict = field(default_factory=dict)
    sae: dict = field(default_factory=dict)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16}[self.dtype]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    return Config(
        model_name=raw["model_name"],
        device=raw["device"],
        dtype=raw["dtype"],
        out_dir=Path(raw["out_dir"]),
        store_full=raw.get("store_full", False),
        dataset=raw.get("dataset", {}),
        patching=raw.get("patching", {}),
        steering=raw.get("steering", {}),
        sae=raw.get("sae", {}),
    )
