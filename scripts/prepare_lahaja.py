"""Prepare the lahaja-eval dataset for codec evaluation.

Downloads the ``tiny-aya-translate/lahaja-eval`` dataset from HuggingFace,
resamples audio to 24 kHz, and writes a test split with ``manifest.json``
in the format expected by :mod:`eval.run_all`.

Audio decoding & resampling
---------------------------
The HuggingFace ``datasets`` library's :class:`~datasets.Audio` feature is
used with ``sampling_rate=24000`` and ``decode=True``.  Under the hood this
uses **torchcodec** as the decoding backend (the default for ``datasets``
≥ 3.x).  The decoded numpy arrays are then written to WAV via
:func:`torchaudio.save`.

Output layout
-------------
::

    data/lahaja/
    └── test/
        ├── manifest.json     # [{id, audio_path, text, speaker, duration}, …]
        ├── utt_00000.wav
        ├── utt_00001.wav
        └── …

Usage::

    uv run --extra all python scripts/prepare_lahaja.py

License: MIT
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Target sample rate for output WAV files (must match codec expectations).
TARGET_SR = 24_000
# Root output directory; a ``test/`` subdirectory will be created inside.
OUTPUT_DIR = Path("data/lahaja")
# Duration filters: skip utterances outside [MIN, MAX] seconds.
MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0


def main() -> None:
    """Download, resample, and write the lahaja-eval test split.

    Steps:
        1. Load the ``tiny-aya-translate/lahaja-eval`` dataset from
           HuggingFace (test split only).
        2. Cast the ``audio_filepath`` column to the ``Audio`` feature
           with target sample rate, triggering torchcodec decoding.
        3. Iterate over rows, skip utterances outside the duration
           window, and write each valid utterance as a 24 kHz mono WAV.
        4. Write a ``manifest.json`` with per-utterance metadata.
        5. Print a summary of the prepared dataset.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading tiny-aya-translate/lahaja-eval from HuggingFace...")
    ds = load_dataset("tiny-aya-translate/lahaja-eval", split="test")

    # Cast the audio column so that `datasets` decodes and resamples
    # audio on-the-fly via torchcodec when rows are accessed.
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=TARGET_SR, decode=True))

    test_dir = OUTPUT_DIR / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped = 0

    for idx in tqdm(range(len(ds)), desc="Preparing lahaja"):
        row = ds[idx]
        audio_data = row["audio_filepath"]

        # The Audio feature returns {"array": np.ndarray, "sampling_rate": int}.
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = int(audio_data["sampling_rate"])

        # Down-mix to mono if multi-channel.
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Apply duration-based filtering.
        duration = len(audio_array) / sr
        if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
            skipped += 1
            continue

        utt_id = f"utt_{idx:05d}"
        wav_relpath = f"test/{utt_id}.wav"
        wav_path = OUTPUT_DIR / wav_relpath

        # Convert to a torch tensor and write via torchaudio.
        waveform = torch.from_numpy(audio_array).unsqueeze(0)
        torchaudio.save(str(wav_path), waveform, sr)

        manifest.append({
            "id": utt_id,
            "audio_path": wav_relpath,
            "text": row.get("text", ""),
            "speaker": str(row.get("sp_id", f"spk_{idx:05d}")),
            "duration": round(len(audio_array) / sr, 4),
        })

    # Write the manifest consumed by eval/run_all.py and eval/reconstruct.py.
    manifest_path = test_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # Summary statistics.
    total_hours = sum(e["duration"] for e in manifest) / 3600.0
    n_speakers = len({e["speaker"] for e in manifest})

    print(f"\n{'=' * 60}")
    print(f"Dataset: lahaja-eval")
    print(f"Utterances: {len(manifest)} (skipped {skipped})")
    print(f"Speakers: {n_speakers}")
    print(f"Total hours: {total_hours:.2f}h")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
