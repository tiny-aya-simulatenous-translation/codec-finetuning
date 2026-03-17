"""Prepare audio datasets for codec fine-tuning.

Downloads or loads datasets from HuggingFace or local directories, resamples
audio to 24 kHz, creates speaker-disjoint (or random) train/val/test splits,
and writes WAV files alongside manifest JSONs consumed by the training loop.

Usage::

    uv run python prepare_data.py --config configs/datasets/turkish_sample.yaml
    uv run python prepare_data.py --config configs/datasets/hindi.yaml

Dependencies:
    - soundfile, librosa, numpy, tqdm, pyyaml (core)
    - datasets (optional, for HuggingFace loading)
    Install all via ``uv pip install -e '.[train]'``

License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from train.config_loader import _resolve_bases

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_dataset_config(config_path: str) -> Dict[str, Any]:
    """Load a dataset config with ``_bases_`` resolution but no experiment validation.

    Dataset configs (e.g. ``configs/datasets/turkish_sample.yaml``) lack
    experiment-level fields like ``optimizer`` or ``codec``.  This helper
    resolves base references without requiring those fields.

    Args:
        config_path: Path to a dataset YAML config file.

    Returns:
        The merged config dictionary.
    """
    config_path_obj = Path(config_path).resolve()
    logger.info("Loading config: %s", config_path_obj)

    with open(config_path_obj, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    config = _resolve_bases(raw, config_path_obj.parent)

    if "dataset" not in config:
        raise ValueError(
            f"Config {config_path} is missing a 'dataset' section. "
            "Check that you are passing a dataset config file."
        )
    return config


def prepare(config_path: str) -> None:
    """Run the full data-preparation pipeline from a dataset config.

    Steps:
        1. Load dataset config via :func:`train.config_loader.load_config`.
        2. Create the output directory.
        3. Load raw examples from HuggingFace or a local directory.
        4. Apply duration and clipping filters.
        5. Create speaker-disjoint (or random) splits.
        6. Write resampled WAV files and per-split manifest JSONs.
        7. Validate the output and print a summary.

    Args:
        config_path: Path to a dataset YAML config file, e.g.
            ``configs/datasets/turkish_sample.yaml``.
    """
    config = _load_dataset_config(config_path)
    ds_cfg: Dict[str, Any] = config["dataset"]

    output_dir = Path(ds_cfg["local_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw examples ────────────────────────────────────────────────
    source = ds_cfg["source"]
    if source == "huggingface":
        examples = _load_from_huggingface(config)
    elif source == "local":
        examples = _load_from_local(config)
    else:
        raise ValueError(f"Unknown dataset source: {source!r}. Use 'huggingface' or 'local'.")

    total_raw = len(examples)
    logger.info("Loaded %d raw examples.", total_raw)

    # ── Filter ───────────────────────────────────────────────────────────
    examples = _apply_filters(examples, config)

    # ── Check disk space ─────────────────────────────────────────────────
    estimated_gb = ds_cfg.get("estimated_hours", 1.0) * 0.5  # ~0.5 GB/h at 24 kHz mono 16-bit
    _check_disk_space(output_dir, estimated_gb)

    # ── Split ────────────────────────────────────────────────────────────
    splits = _create_splits(examples, config)

    # ── Write WAVs + manifests ───────────────────────────────────────────
    target_sr: int = ds_cfg.get("target_sr", 24_000)
    manifests: Dict[str, Path] = {}
    for split_name, split_examples in splits.items():
        manifests[split_name] = _write_split(split_examples, split_name, output_dir, target_sr)

    # ── Validate ─────────────────────────────────────────────────────────
    split_method = ds_cfg.get("splits", {}).get("method", "random")
    _validate_output(output_dir, check_speaker_disjoint=(split_method == "speaker_disjoint"))

    # ── Summary ──────────────────────────────────────────────────────────
    disjoint_str = "✓" if split_method == "speaker_disjoint" else "✗"

    print("\n══════════════════════════════════════════════════════════════")
    print(f"Dataset: {ds_cfg['name']}")
    print(f"Total examples: {total_raw}")
    print(f"After filtering: {sum(len(v) for v in splits.values())}")
    print("Splits:")
    for sname in ("train", "val", "test"):
        exs = splits.get(sname, [])
        speakers = {e["speaker"] for e in exs}
        hours = sum(e["duration"] for e in exs) / 3600.0
        print(f"  {sname + ':':6s} {len(exs)} utterances, {len(speakers)} speakers, {hours:.2f}h")
    print(f"Speaker-disjoint: {disjoint_str}")
    print(f"Output: {output_dir}/")
    print("══════════════════════════════════════════════════════════════\n")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_from_huggingface(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load a dataset from a HuggingFace repository.

    Uses the ``datasets`` library to stream/download the dataset specified
    in the config. Requires ``huggingface-cli login`` for private repos.

    Args:
        config: Full merged config dict. Reads keys under ``config["dataset"]``:
            ``hf_repo``, ``hf_split``, ``audio_column``, ``text_column``,
            ``speaker_column``, ``duration_column``.

    Returns:
        A list of example dicts, each with keys ``audio`` (:class:`numpy.ndarray`),
        ``sr`` (int), ``text`` (str), ``speaker`` (str), ``duration`` (float),
        and ``id`` (str).

    Raises:
        ImportError: If the ``datasets`` library is not installed.
        RuntimeError: If HuggingFace authentication fails for a private repo.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for HuggingFace loading. "
            "Install it with: uv pip install -e '.[train]'"
        ) from exc

    ds_cfg = config["dataset"]
    hf_repo: str = ds_cfg["hf_repo"]
    hf_split: str = ds_cfg.get("hf_split", "train")
    audio_col: str = ds_cfg.get("audio_column", "audio")
    text_col: str = ds_cfg.get("text_column", "text")
    speaker_col: Optional[str] = ds_cfg.get("speaker_column")
    duration_col: Optional[str] = ds_cfg.get("duration_column")

    logger.info("Loading HuggingFace dataset: %s (split=%s)", hf_repo, hf_split)
    try:
        hf_ds = load_dataset(hf_repo, split=hf_split)
    except Exception as exc:
        if "authentication" in str(exc).lower() or "401" in str(exc):
            raise RuntimeError(
                f"Authentication failed for '{hf_repo}'. "
                "Run `huggingface-cli login` with a valid token and retry."
            ) from exc
        raise

    examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(tqdm(hf_ds, desc="Loading HF examples")):
        audio_data = row[audio_col]
        # HF audio feature returns {"array": np.ndarray, "sampling_rate": int}
        if isinstance(audio_data, dict):
            audio_array = np.array(audio_data["array"], dtype=np.float32)
            sr = int(audio_data["sampling_rate"])
        else:
            audio_array = np.array(audio_data, dtype=np.float32)
            sr = int(ds_cfg.get("original_sr") or 24_000)

        text = str(row.get(text_col, ""))
        speaker = str(row[speaker_col]) if speaker_col and speaker_col in row else f"spk_{idx:05d}"

        if duration_col and duration_col in row:
            duration = float(row[duration_col])
        else:
            duration = len(audio_array) / sr

        examples.append({
            "audio": audio_array,
            "sr": sr,
            "text": text,
            "speaker": speaker,
            "duration": duration,
            "id": f"utt_{idx:05d}",
        })

    return examples


def _load_from_local(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load audio files from a local directory tree.

    Walks ``config["dataset"]["local_raw_dir"]`` recursively for ``.wav``,
    ``.flac``, and ``.mp3`` files. Speaker ID is inferred from the immediate
    parent directory name.

    Args:
        config: Full merged config dict. Reads ``config["dataset"]["local_raw_dir"]``.

    Returns:
        A list of example dicts with the same schema as
        :func:`_load_from_huggingface`.

    Raises:
        FileNotFoundError: If ``local_raw_dir`` does not exist.
    """
    ds_cfg = config["dataset"]
    raw_dir = Path(ds_cfg["local_raw_dir"])

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Local raw directory not found: {raw_dir}. "
            "Download or extract the dataset first."
        )

    audio_extensions = {".wav", ".flac", ".mp3"}
    audio_files = sorted(
        p for p in raw_dir.rglob("*") if p.suffix.lower() in audio_extensions
    )
    logger.info("Found %d audio files in %s", len(audio_files), raw_dir)

    examples: List[Dict[str, Any]] = []
    for idx, fpath in enumerate(tqdm(audio_files, desc="Loading local files")):
        try:
            audio_array, sr = sf.read(fpath, dtype="float32")
        except Exception:
            logger.warning("Failed to read %s, skipping.", fpath)
            continue

        # Force mono by averaging channels.
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        speaker = fpath.parent.name

        # Look for a matching transcript file (.txt) alongside the audio.
        txt_path = fpath.with_suffix(".txt")
        text = ""
        if txt_path.exists():
            try:
                text = txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                logger.warning("Failed to read transcript %s.", txt_path)

        duration = len(audio_array) / sr

        examples.append({
            "audio": audio_array,
            "sr": sr,
            "text": text,
            "speaker": speaker,
            "duration": duration,
            "id": f"utt_{idx:05d}",
        })

    return examples


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _apply_filters(examples: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter examples by duration and clipping ratio.

    Args:
        examples: List of example dicts (with ``audio``, ``duration`` keys).
        config: Full merged config dict. Reads ``config["dataset"]["filters"]``
            for ``min_duration_s``, ``max_duration_s``, and ``max_clip_ratio``.

    Returns:
        A new list containing only the examples that pass all filters.
    """
    filters = config["dataset"].get("filters", {})
    min_dur: float = filters.get("min_duration_s", 0.0)
    max_dur: float = filters.get("max_duration_s", float("inf"))
    max_clip: float = filters.get("max_clip_ratio", 1.0)

    kept: List[Dict[str, Any]] = []
    skipped_short = 0
    skipped_clip = 0
    truncated = 0

    for ex in examples:
        # Skip too-short utterances.
        if ex["duration"] < min_dur:
            skipped_short += 1
            continue

        # Clipping detection: ratio of samples at ±1.0.
        audio = ex["audio"]
        clip_ratio = np.mean(np.abs(audio) >= 1.0)
        if clip_ratio > max_clip:
            skipped_clip += 1
            continue

        # Truncate overly long utterances.
        if ex["duration"] > max_dur:
            max_samples = int(max_dur * ex["sr"])
            ex["audio"] = audio[:max_samples]
            ex["duration"] = max_dur
            truncated += 1

        kept.append(ex)

    logger.info(
        "Filtering: kept=%d, skipped_short=%d, skipped_clipped=%d, truncated=%d",
        len(kept),
        skipped_short,
        skipped_clip,
        truncated,
    )
    return kept


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _create_splits(
    examples: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Partition examples into train / val / test splits.

    Supports two methods:

    * ``"speaker_disjoint"``: groups by speaker, removes speakers with too
      few utterances, then assigns whole speakers to splits so that no
      speaker appears in more than one split.
    * ``"random"``: shuffles all examples and splits by ratio.

    Args:
        examples: Filtered list of example dicts.
        config: Full merged config dict. Reads ``config["dataset"]["splits"]``
            for ``method``, ``train_ratio``, ``val_ratio``, ``test_ratio``,
            and ``min_speaker_utterances``.

    Returns:
        A dict mapping split names (``"train"``, ``"val"``, ``"test"``) to
        lists of example dicts.

    Raises:
        AssertionError: If speaker-disjoint validation fails (a speaker
            appears in multiple splits).
    """
    split_cfg = config["dataset"].get("splits", {})
    method: str = split_cfg.get("method", "random")
    train_ratio: float = split_cfg.get("train_ratio", 0.8)
    val_ratio: float = split_cfg.get("val_ratio", 0.1)

    if method == "speaker_disjoint":
        return _speaker_disjoint_split(examples, split_cfg, train_ratio, val_ratio)

    # Default: random split.
    rng = random.Random(42)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _speaker_disjoint_split(
    examples: List[Dict[str, Any]],
    split_cfg: Dict[str, Any],
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create speaker-disjoint splits.

    Groups examples by speaker, filters speakers with fewer than
    ``min_speaker_utterances``, shuffles speakers deterministically,
    and greedily assigns speakers to splits so that the cumulative
    utterance counts approximate the target ratios.

    Args:
        examples: Filtered list of example dicts.
        split_cfg: The ``config["dataset"]["splits"]`` sub-dict.
        train_ratio: Target fraction for the training split.
        val_ratio: Target fraction for the validation split.

    Returns:
        A dict mapping ``"train"``, ``"val"``, ``"test"`` to example lists.

    Raises:
        AssertionError: If any speaker appears in more than one split.
    """
    min_utt: int = split_cfg.get("min_speaker_utterances", 1)

    # Group by speaker.
    by_speaker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_speaker[ex["speaker"]].append(ex)

    # Filter speakers with too few utterances.
    filtered_speakers = {
        spk: utts for spk, utts in by_speaker.items() if len(utts) >= min_utt
    }
    if len(filtered_speakers) < len(by_speaker):
        dropped = len(by_speaker) - len(filtered_speakers)
        logger.info("Dropped %d speakers with fewer than %d utterances.", dropped, min_utt)

    # Shuffle speakers deterministically.
    rng = random.Random(42)
    speaker_ids = sorted(filtered_speakers.keys())
    rng.shuffle(speaker_ids)

    total_utt = sum(len(filtered_speakers[s]) for s in speaker_ids)
    target_train = int(total_utt * train_ratio)
    target_val = int(total_utt * val_ratio)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for spk in speaker_ids:
        utts = filtered_speakers[spk]
        if counts["train"] < target_train:
            splits["train"].extend(utts)
            counts["train"] += len(utts)
        elif counts["val"] < target_val:
            splits["val"].extend(utts)
            counts["val"] += len(utts)
        else:
            splits["test"].extend(utts)
            counts["test"] += len(utts)

    # Assert no speaker overlap.
    train_spk = {e["speaker"] for e in splits["train"]}
    val_spk = {e["speaker"] for e in splits["val"]}
    test_spk = {e["speaker"] for e in splits["test"]}
    assert train_spk.isdisjoint(val_spk), "Speaker overlap between train and val!"
    assert train_spk.isdisjoint(test_spk), "Speaker overlap between train and test!"
    assert val_spk.isdisjoint(test_spk), "Speaker overlap between val and test!"

    return splits


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample an audio waveform to the target sample rate.

    Uses :func:`librosa.resample` when ``orig_sr != target_sr``. Returns
    the input unchanged otherwise.

    Args:
        audio: 1-D float32 waveform.
        orig_sr: Original sample rate in Hz.
        target_sr: Desired sample rate in Hz (typically 24 000).

    Returns:
        Resampled waveform as a 1-D :class:`numpy.ndarray`.
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_split(
    examples: List[Dict[str, Any]],
    split_name: str,
    output_dir: Path,
    target_sr: int,
) -> Path:
    """Write WAV files and a manifest JSON for a single split.

    Creates ``output_dir/split_name/`` with one WAV per example and a
    ``manifest.json`` listing all entries.

    Args:
        examples: Example dicts for this split.
        split_name: One of ``"train"``, ``"val"``, ``"test"``.
        output_dir: Root output directory (e.g. ``data/turkish_sample``).
        target_sr: Target sample rate for output WAVs.

    Returns:
        Path to the written ``manifest.json``.
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    for ex in tqdm(examples, desc=f"Writing {split_name}"):
        audio = _resample(ex["audio"], ex["sr"], target_sr)
        wav_relpath = f"{split_name}/{ex['id']}.wav"
        wav_path = output_dir / wav_relpath

        sf.write(str(wav_path), audio, target_sr)

        manifest.append({
            "id": ex["id"],
            "audio_path": wav_relpath,
            "text": ex["text"],
            "speaker": ex["speaker"],
            "duration": round(len(audio) / target_sr, 4),
        })

    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    logger.info("Wrote %d files to %s", len(manifest), split_dir)
    return manifest_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_output(output_dir: Path, check_speaker_disjoint: bool = True) -> None:
    """Validate that the prepared dataset is consistent and complete.

    Checks:
        - All expected splits (train, val, test) have directories and manifests.
        - Every WAV referenced in a manifest exists on disk.
        - No speaker appears in more than one split (only when
          *check_speaker_disjoint* is ``True``).

    Args:
        output_dir: Root output directory containing split sub-directories.
        check_speaker_disjoint: Whether to verify that no speaker appears in
            more than one split.  Set to ``False`` for datasets without
            speaker IDs (e.g. MediaSpeech) that use random splits.

    Raises:
        FileNotFoundError: If a manifest or WAV file is missing.
        AssertionError: If speaker overlap is detected across splits.
    """
    speakers_by_split: Dict[str, set] = {}

    for split_name in ("train", "val", "test"):
        manifest_path = output_dir / split_name / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        split_speakers: set = set()
        for entry in manifest:
            wav_path = output_dir / entry["audio_path"]
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing WAV referenced in manifest: {wav_path}")
            split_speakers.add(entry["speaker"])

        speakers_by_split[split_name] = split_speakers

    # Cross-split speaker overlap check (only for speaker-disjoint splits).
    if check_speaker_disjoint:
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = speakers_by_split[a] & speakers_by_split[b]
            assert not overlap, (
                f"Speaker overlap between {a} and {b}: {overlap}"
            )

    logger.info("Validation passed for %s", output_dir)


def _check_disk_space(output_dir: Path, estimated_gb: float) -> None:
    """Warn if available disk space is insufficient for the output.

    Args:
        output_dir: Directory where output will be written.
        estimated_gb: Estimated total output size in gigabytes.
    """
    try:
        stat = shutil.disk_usage(output_dir)
        free_gb = stat.free / (1024**3)
        required_gb = estimated_gb * 1.5
        if free_gb < required_gb:
            logger.warning(
                "Low disk space: %.1f GB free, but %.1f GB recommended "
                "(estimated output: %.1f GB). Proceed with caution.",
                free_gb,
                required_gb,
                estimated_gb,
            )
        else:
            logger.info("Disk space OK: %.1f GB free (need ~%.1f GB).", free_gb, required_gb)
    except OSError:
        logger.warning("Could not check disk space for %s.", output_dir)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the preparation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Prepare audio datasets for codec fine-tuning.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a dataset YAML config (e.g. configs/datasets/turkish_sample.yaml).",
    )
    args = parser.parse_args()

    prepare(args.config)


if __name__ == "__main__":
    main()
