"""
Activation patching

Transplants encoder activations from a source clip into a target clip's
forward pass at one layer, and measures the effect on Whisper's OWN language
decision: the logits of the language tokens (<|en|>, <|it|>, ...) that the
decoder predicts as its first token after <|startoftranscript|>.
"""

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import Config

# FLEURS config name -> Whisper language token code
WHISPER_LANG = {"en_us": "en", "it_it": "it", "ja_jp": "ja",
                "fr_fr": "fr", "es_419": "es"}


class PatchedWhisper:
    # Frozen Whisper whose encoder can capture and/or overwrite any layer

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

        tok = self.processor.tokenizer
        self.lang_token_id = {}
        for code in WHISPER_LANG.values():
            tid = tok.convert_tokens_to_ids(f"<|{code}|>")
            assert tid != tok.unk_token_id, f"no language token for {code}"
            self.lang_token_id[code] = tid

        # intervention state, read by the hooks on every forward pass:
        # _patch REPLACES one layer's output, _steer ADDS a vector to it,
        # _captured records every layer's (possibly modified) output
        self._patch: tuple[int, torch.Tensor] | None = None
        self._steer: tuple[int, torch.Tensor] | None = None  # (layer, vec[d])
        self._captured: dict[int, torch.Tensor] = {}
        # one pre hook gives access to "layer 0" (the input of block 1),
        # a forward hook on each block covers layers 1..n_layers
        self.encoder.layers[0].register_forward_pre_hook(self._pre_hook)
        for k, layer in enumerate(self.encoder.layers, start=1):
            layer.register_forward_hook(self._make_hook(k))

    # -- hooks ---------------------------------------------------------------
    def _pre_hook(self, module, args):
        h = args[0]
        if self._patch is not None and self._patch[0] == 0:
            h = self._patch[1]
        if self._steer is not None and self._steer[0] == 0:
            h = h + self._steer[1]
        self._captured[0] = h.detach()
        if h is not args[0]:
            return (h,) + args[1:]

    def _make_hook(self, k):
        def hook(module, args, output, k=k):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            changed = False
            if self._patch is not None and self._patch[0] == k:
                h = self._patch[1]
                changed = True
            if self._steer is not None and self._steer[0] == k:
                h = h + self._steer[1]  # broadcast [d] over [1, T, d]
                changed = True
            self._captured[k] = h.detach()
            if changed:
                return (output.__class__((h,) + output[1:]) if is_tuple else h)
        return hook

    # -- forward passes ------------------------------------------------------
    def features(self, audio: np.ndarray) -> torch.Tensor:
        return self.processor(
            audio, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(self.cfg.device, self.cfg.torch_dtype)

    @torch.no_grad()
    def encode(self, feats: torch.Tensor, patch=None, steer=None):
        self._patch, self._steer = patch, steer
        self._captured.clear()
        try:
            return self.encoder(feats)
        finally:
            self._patch = self._steer = None

    @torch.no_grad()
    def capture_layers(self, feats: torch.Tensor) -> dict[int, torch.Tensor]:
        """All layer activations of an unpatched pass ({k: [1, 1500, d]})."""
        self.encode(feats)
        return {k: v.clone() for k, v in self._captured.items()}

    @torch.no_grad()
    def lang_logits(self, feats: torch.Tensor, patch=None, steer=None) -> dict[str, float]:
        """First-decoder-token logit of every language token."""
        enc = self.encode(feats, patch, steer)
        # feed the decoder only <|startoftranscript|>: its first prediction
        # is the language token, so these logits ARE Whisper's language call
        sot = torch.tensor([[self.model.config.decoder_start_token_id]],
                           device=self.cfg.device)
        logits = self.model(encoder_outputs=enc, decoder_input_ids=sot).logits[0, -1]
        return {c: logits[i].item() for c, i in self.lang_token_id.items()}

    @torch.no_grad()
    def transcribe(self, feats: torch.Tensor, patch=None, steer=None) -> str:
        self._patch, self._steer = patch, steer
        try:
            ids = self.model.generate(input_features=feats, max_new_tokens=100,
                                      do_sample=False)
        finally:
            self._patch = self._steer = None
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def top_lang(logits: dict[str, float]) -> str:
    return max(logits, key=logits.get)


def block_update_patch(
    cap_t: dict[int, torch.Tensor], cap_s: dict[int, torch.Tensor], k: int
) -> torch.Tensor:
    """
    Replacing the full hidden state at layer k is degenerate: the entire
    target state is overwritten, so every layer yields the same result (the
    source's). Instead we keep the target's input to block k and swap in the
    source's residual update:  h_t[k-1] + (h_s[k] - h_s[k-1])
    """
    assert k >= 1
    return cap_t[k - 1] + (cap_s[k] - cap_s[k - 1])


def make_pairs(
    targets: list[dict],
    sources: list[dict],
    n_pairs: int,
    duration_tol: float,
    match_gender: bool = True,
) -> list[tuple[dict, dict]]:
    """
    Greedily pair each target clip with an unused source clip of the same
    gender (optionally) and similar duration, so language is as close as
    possible to the only property differing within a pair
    """
    used = set()   # source indices already assigned, so no source repeats
    pairs = []
    # outer loop: walk the target clips in order; inner loop: scan all
    # sources for the unused one closest in duration (within tolerance)
    for t in targets:
        best, best_d = None, duration_tol
        for j, s in enumerate(sources):
            if j in used:
                continue
            if match_gender and s.get("gender") != t.get("gender"):
                continue
            d = abs(s["duration"] - t["duration"])
            if d <= best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append((t, sources[best]))
        if len(pairs) >= n_pairs:
            break
    return pairs
