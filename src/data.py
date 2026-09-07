"""Dataset download / subsetting / preprocessing (Phase A/B input side).

Every loader returns the same uniform format: a list of dicts with at least
{id, audio (float32 mono numpy array @ 16 kHz), sr, duration, speaker_id,
text} plus any property labels (language, gender, median_f0, ...) that
extraction copies into metadata.parquet.

Loaders (selected by the config's `dataset.loader` key):
  * hf_generic   : any HF dataset with an `audio` column (E0 uses the tiny
                   LibriSpeech dummy mirror)
  * fleurs       : google/fleurs, N languages x n_per_lang clips, streamed
                   (no full-archive downloads); labels: language, gender
  * librispeech  : openslr/librispeech_asr test-clean, speaker-balanced
                   subset; labels: speaker_id (and median F0 if requested)

Median F0 is computed directly from audio with librosa's pyin (median over
voiced frames); no external labels needed (PLAN §3).
"""

import numpy as np
from datasets import Audio, load_dataset

WHISPER_SR = 16_000

FLEURS_GENDER = {0: "male", 1: "female"}


def _load_parquet(pattern: str):
    """Load only the parquet files matching an hf:// glob (avoids pulling
    every split of a dataset) and make sure the audio column decodes."""
    ds = load_dataset("parquet", data_files=pattern, split="train")
    if not isinstance(ds.features["audio"], Audio):
        ds = ds.cast_column("audio", Audio(sampling_rate=WHISPER_SR))
    return ds


def _clip(audio_arr, sr: int, max_seconds: float, **labels) -> dict:
    assert sr == WHISPER_SR, f"expected {WHISPER_SR} Hz, got {sr}"
    arr = np.asarray(audio_arr, dtype=np.float32)[: int(max_seconds * WHISPER_SR)]
    return {"audio": arr, "sr": sr, "duration": len(arr) / WHISPER_SR, **labels}


def median_f0(audio: np.ndarray, sr: int = WHISPER_SR) -> float:
    """Median fundamental frequency over voiced frames (Hz); NaN if unvoiced."""
    import librosa

    f0, _, _ = librosa.pyin(audio, fmin=65.0, fmax=400.0, sr=sr)
    voiced = f0[~np.isnan(f0)]
    return float(np.median(voiced)) if len(voiced) else float("nan")


def _load_hf_generic(cfg: dict) -> list[dict]:
    ds = load_dataset(cfg["name"], cfg.get("config"), split=cfg["split"])
    clips = []
    for row in ds.select(range(min(cfg.get("n_clips", len(ds)), len(ds)))):
        a = row["audio"]
        clips.append(
            _clip(
                a["array"], a["sampling_rate"], cfg.get("max_seconds", 10.0),
                id=str(row.get("id", len(clips))),
                speaker_id=str(row.get("speaker_id", "")),
                text=row.get("text", ""),
            )
        )
    return clips


def _load_fleurs(cfg: dict) -> list[dict]:
    clips = []
    split = cfg.get("split", "test")
    for lang in cfg["langs"]:
        # HF's parquet conversion: fast, resumable file downloads (the native
        # repo only offers slow sequential tar streaming)
        ds = _load_parquet(
            f"hf://datasets/google/fleurs@refs/convert/parquet/{lang}/{split}/*.parquet"
        )
        taken = 0
        for row in ds:
            a = row["audio"]
            clips.append(
                _clip(
                    a["array"], a["sampling_rate"], cfg.get("max_seconds", 10.0),
                    id=f"{lang}_{row['id']}",
                    speaker_id="",  # FLEURS has no speaker identities
                    text=row["transcription"],
                    language=lang,
                    gender=FLEURS_GENDER.get(row["gender"], str(row["gender"])),
                )
            )
            taken += 1
            if taken >= cfg["n_per_lang"]:
                break
        print(f"[data] fleurs {lang}: {taken} clips")
    return clips


def _load_librispeech(cfg: dict) -> list[dict]:
    # only the requested split's parquet files; `load_dataset(..., "clean")`
    # would download all clean splits incl. ~30 GB of training data
    ds = _load_parquet(
        f"hf://datasets/openslr/librispeech_asr/clean/{cfg.get('split', 'test')}/*.parquet"
    )
    speakers = np.array(ds["speaker_id"])
    uniq = np.unique(speakers)
    per_speaker = int(np.ceil(cfg["n_clips"] / len(uniq)))
    idx = np.concatenate(
        [np.flatnonzero(speakers == s)[:per_speaker] for s in uniq]
    )[: cfg["n_clips"]]

    compute_f0 = cfg.get("compute_f0", False)
    clips = []
    for row in ds.select([int(i) for i in idx]):
        a = row["audio"]
        c = _clip(
            a["array"], a["sampling_rate"], cfg.get("max_seconds", 10.0),
            id=str(row["id"]),
            speaker_id=str(row["speaker_id"]),
            text=row["text"],
        )
        if compute_f0:
            c["median_f0"] = median_f0(c["audio"])
        clips.append(c)
    print(f"[data] librispeech: {len(clips)} clips, {len(uniq)} speakers")
    return clips


LOADERS = {
    "hf_generic": _load_hf_generic,
    "fleurs": _load_fleurs,
    "librispeech": _load_librispeech,
}


def load_clips(dataset_cfg: dict) -> list[dict]:
    return LOADERS[dataset_cfg.get("loader", "hf_generic")](dataset_cfg)
