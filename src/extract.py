"""
Activation extraction: forward hooks + on disk caching

For each clip we run the frozen Whisper encoder and capture the block's output hidden states. 

From those we store the pooled layer, the pooled logmel input, the full layer (optional), and the metadata table

Everything runs under no_grad on the frozen model so no training
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import Config

HOP = 160          # Whisper STFT hop at 16 kHz
CONV_STRIDE = 2    # encoder conv2 stride: 3000 mel frames -> 1500 encoder frames

# takes a clip of n samples and returns (valid mel frames, valid encoder frames)
def n_valid_frames(n_samples: int) -> tuple[int, int]:
    mel = math.ceil(n_samples / HOP)
    return mel, math.ceil(mel / CONV_STRIDE)

# class to run the frozen Whisper encoder and capture activations at each layer
class ActivationExtractor:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.processor = WhisperProcessor.from_pretrained(cfg.model_name)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(
                cfg.model_name, torch_dtype=cfg.torch_dtype
            )
            .to(cfg.device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.encoder = self.model.model.encoder
        self.n_layers = len(self.encoder.layers)
        self.d_model = self.model.config.d_model
        self._captured: dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        # layer 0 = input to block 1 (post conv + positional embedding)
        def pre_hook(module, args):
            self._captured[0] = args[0].detach()

        self.encoder.layers[0].register_forward_pre_hook(pre_hook)

        for k, layer in enumerate(self.encoder.layers, start=1):
            def hook(module, args, output, k=k):
                # layer forward returns a tuple in older transformers versions
                # returns a bare [B, T, D] tensor in newer ones
                h = output[0] if isinstance(output, tuple) else output
                self._captured[k] = h.detach()

            layer.register_forward_hook(hook)

    # run one clip through the encoder and return the activations per layer
    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> dict:
        feats = self.processor(
            audio, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(self.cfg.device, self.cfg.torch_dtype)

        self._captured.clear()
        self.encoder(feats)
        layers = torch.stack(
            [self._captured[k][0] for k in range(self.n_layers + 1)]
        )
        _, valid_enc = n_valid_frames(len(audio))
        return {
            "logmel": feats[0].float().cpu(),
            "layers": layers.float().cpu(),
            "valid_enc_frames": valid_enc,
        }

# mean pool over the first valid time frames of a tensor [..., T, D] -> [..., D]
def pool(seq: torch.Tensor, valid: int) -> torch.Tensor:
    return seq[..., :valid, :].mean(dim=-2)

# extract and cache activations for a list of clips under cfg.out_dir
def extract_clips(cfg: Config, clips: list[dict]) -> dict:
    extractor = ActivationExtractor(cfg)
    n = len(clips)
    T = extractor.encoder.max_source_positions  # 1500
    n_mels = extractor.model.config.num_mel_bins

    # preallocate the output tensors: one row per clip, one leading index per
    # layer (n_layers blocks + the layer-0 embedding). full is only
    # allocated when the config asks for full sequences
    pooled = torch.zeros(extractor.n_layers + 1, n, extractor.d_model)
    pooled_logmel = torch.zeros(n, n_mels)
    full = (
        torch.zeros(extractor.n_layers + 1, n, T, extractor.d_model,
                    dtype=torch.float16)
        if cfg.store_full
        else None
    )

    # main loop: one forward pass per clip, then pool over the
    # clip's real (so no padding) frames and collect its metadata row
    rows = []
    for i, clip in enumerate(clips):
        acts = extractor.encode(clip["audio"])
        valid = acts["valid_enc_frames"]
        pooled[:, i] = pool(acts["layers"], valid)
        valid_mel, _ = n_valid_frames(len(clip["audio"]))
        pooled_logmel[i] = acts["logmel"][:, :valid_mel].mean(dim=-1)
        if full is not None:
            full[:, i] = acts["layers"].half()
        labels = {
            k: v for k, v in clip.items()
            if k not in ("audio", "sr", "id", "speaker_id", "duration", "text")
        }
        rows.append(
            {
                "row": i,
                "clip_id": clip["id"],
                "speaker_id": clip["speaker_id"],
                "duration": clip["duration"],
                "valid_enc_frames": valid,
                "text": clip["text"],
                **labels,
            }
        )

    # write one file per layer plus the metadata table whose row order matches dim 0 of every tensor
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(extractor.n_layers + 1):
        torch.save(pooled[k].clone(), out / f"pooled_layer{k:02d}.pt")
        if full is not None:
            torch.save(full[k].clone(), out / f"full_layer{k:02d}.pt")
    torch.save(pooled_logmel, out / "pooled_logmel.pt")
    pd.DataFrame(rows).to_parquet(out / "metadata.parquet", index=False)

    return {
        "n_clips": n,
        "n_layers": extractor.n_layers,
        "d_model": extractor.d_model,
        "n_mels": n_mels,
        "seq_len": T,
        "out_dir": str(out),
    }

# reload a cache directory into memory (pooled, full if present, metadata)
def load_cached(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    pooled_files = sorted(out.glob("pooled_layer*.pt"))
    cache = {
        "pooled": torch.stack([torch.load(f) for f in pooled_files]),
        "pooled_logmel": torch.load(out / "pooled_logmel.pt"),
        "metadata": pd.read_parquet(out / "metadata.parquet"),
    }
    full_files = sorted(out.glob("full_layer*.pt"))
    if full_files:
        cache["full"] = torch.stack([torch.load(f) for f in full_files])
    return cache
