"""Codec registry for the evaluation pipeline.

Provides a single extension point for adding new codecs to the benchmark.
Each codec registers a :class:`CodecEvalHooks` dataclass that tells the
evaluation pipeline how to load, encode, decode, and (optionally) run
codec-specific metrics.

Architecture
------------
The registry follows a **lazy-initialisation** pattern: built-in codecs
(Mimi, DualCodec, Kanade) are registered on first access via
:func:`_ensure_builtins`, avoiding import-time side effects from heavy
dependencies like ``transformers`` or ``dualcodec``.

Third-party codecs can be added at any time by calling
:func:`register_codec` with a :class:`CodecEvalHooks` instance.

Usage::

    from eval.codec_registry import get_codec_hooks

    hooks = get_codec_hooks("mimi")
    model = hooks.load(config, checkpoint, use_ema, device)
    reconstructed = hooks.encode_decode(model, waveform)

License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Type aliases for the three hook callables.  These are used in the
# CodecEvalHooks dataclass to enforce a consistent function signature
# across all registered codecs.

LoadFn = Callable[
    [Dict[str, Any], Optional[str], bool, torch.device],
    Any,
]
"""Signature: ``(config, checkpoint_path, use_ema, device) -> model``."""

EncodeDecodeFn = Callable[
    [Any, torch.Tensor],
    torch.Tensor,
]
"""Signature: ``(model, waveform) -> reconstructed_waveform``."""

CodecMetricFn = Callable[
    [Dict[str, Any], str, Any],
    Dict[str, Any],
]
"""Signature: ``(config, experiment_name, results_dir) -> metrics_dict``."""


@dataclass
class CodecEvalHooks:
    """Everything the eval pipeline needs to know about a codec.

    Attributes:
        name: Lowercase codec identifier (must match ``codec.name`` in config).
        load: Callable ``(config, checkpoint, use_ema, device) -> model``.
        encode_decode: Callable ``(model, waveform) -> reconstructed``.
        extra_metrics: Optional list of ``(stage_name, metric_fn)`` tuples.
            Each *metric_fn* is called as
            ``metric_fn(config, experiment, results_dir)``
            and should return a JSON-serialisable dict of results.
            These run as additional stages after the standard pipeline
            stages and before WandB publish.
    """

    name: str
    load: LoadFn
    encode_decode: EncodeDecodeFn
    extra_metrics: List[Tuple[str, CodecMetricFn]] = field(default_factory=list)


# Central registry mapping lowercase codec names to their hook objects.
_REGISTRY: Dict[str, CodecEvalHooks] = {}


def register_codec(hooks: CodecEvalHooks) -> None:
    """Register a codec's eval hooks.

    Args:
        hooks: A :class:`CodecEvalHooks` instance.

    Raises:
        ValueError: If a codec with the same name is already registered.
    """
    name = hooks.name.lower()
    if name in _REGISTRY:
        raise ValueError(
            f"Codec '{name}' is already registered. "
            "Use a unique name or call unregister_codec() first."
        )
    _REGISTRY[name] = hooks
    logger.debug("Registered codec: %s", name)


def unregister_codec(name: str) -> None:
    """Remove a codec from the registry."""
    _REGISTRY.pop(name.lower(), None)


def get_codec_hooks(name: str) -> CodecEvalHooks:
    """Look up eval hooks for *name*.

    Args:
        name: Codec name (case-insensitive).

    Returns:
        The registered :class:`CodecEvalHooks`.

    Raises:
        ValueError: If no codec with *name* is registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        _ensure_builtins()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown codec '{name}'. "
            f"Registered codecs: {', '.join(sorted(_REGISTRY))}. "
            "Register yours with eval.codec_registry.register_codec()."
        )
    return _REGISTRY[key]


def registered_codecs() -> List[str]:
    """Return the names of all registered codecs."""
    _ensure_builtins()
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in codec implementations
# ---------------------------------------------------------------------------

# Guard flag so that _ensure_builtins() runs at most once per process.
_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    """Lazily register the three built-in codecs on first access.

    Called automatically by :func:`get_codec_hooks` and
    :func:`registered_codecs`.  Uses a module-level guard flag to ensure
    the built-in hooks are constructed at most once.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    for hooks in _builtin_codecs():
        # Skip if a user already registered a custom version of this codec.
        if hooks.name not in _REGISTRY:
            _REGISTRY[hooks.name] = hooks


def _strip_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the ``_orig_mod.`` prefix added by ``torch.compile``.

    When a model is saved after ``torch.compile``, all parameter keys are
    prefixed with ``_orig_mod.``.  This must be removed before calling
    ``model.load_state_dict()`` on the un-compiled model instance.

    Args:
        state_dict: Raw state dict loaded from a checkpoint file.

    Returns:
        A new dict with the ``_orig_mod.`` prefix removed from all keys.
    """
    prefix = "_orig_mod."
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def _load_checkpoint_state(
    checkpoint: str,
    use_ema: bool,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Load and prefix-strip a state dict from a checkpoint file.

    Checkpoint files produced by the training loop contain both a regular
    ``model_state_dict`` and (optionally) an ``ema_state_dict``.  When
    *use_ema* is ``True`` and the EMA key exists, it takes precedence.

    Args:
        checkpoint: Path to a ``.pt`` checkpoint file.
        use_ema: Prefer EMA weights if available.
        device: Device to map tensors to during loading.

    Returns:
        The prefix-stripped state dict, or ``None`` if the checkpoint
        contains neither expected key.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if use_ema and "ema_state_dict" in ckpt:
        logger.info("Loading EMA weights from checkpoint.")
        return _strip_compile_prefix(ckpt["ema_state_dict"])
    if "model_state_dict" in ckpt:
        logger.info("Loading model weights from checkpoint.")
        return _strip_compile_prefix(ckpt["model_state_dict"])
    return None


# -- Mimi ------------------------------------------------------------------

def _load_mimi(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    device: torch.device,
) -> Any:
    """Load a Mimi codec model from HuggingFace ``transformers``.

    Args:
        config: Experiment config; must contain ``config["codec"]["pretrained"]``.
        checkpoint: Optional fine-tuned checkpoint path.
        use_ema: Whether to prefer EMA weights from the checkpoint.
        device: Target device (CPU or CUDA).

    Returns:
        The loaded :class:`transformers.MimiModel` in eval mode.
    """
    from transformers import MimiModel

    model = MimiModel.from_pretrained(config["codec"]["pretrained"])
    if checkpoint is not None:
        sd = _load_checkpoint_state(checkpoint, use_ema, device)
        if sd is not None:
            model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _encode_decode_mimi(model: Any, waveform: torch.Tensor) -> torch.Tensor:
    """Encode *waveform* to Mimi tokens and decode back to audio.

    Args:
        model: A loaded :class:`transformers.MimiModel`.
        waveform: Input tensor of shape ``(batch, 1, samples)``.

    Returns:
        Reconstructed audio tensor.
    """
    tokens = model.encode(waveform)
    return model.decode(tokens.audio_codes).audio_values


# -- DualCodec -------------------------------------------------------------

def _load_dualcodec(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    device: torch.device,
) -> Any:
    """Load a DualCodec model.

    Args:
        config: Experiment config; must contain ``config["codec"]["pretrained"]``.
        checkpoint: Optional fine-tuned checkpoint path.
        use_ema: Whether to prefer EMA weights from the checkpoint.
        device: Target device (CPU or CUDA).

    Returns:
        The loaded DualCodec model in eval mode.
    """
    import dualcodec

    model = dualcodec.get_model(config["codec"]["pretrained"])
    if checkpoint is not None:
        sd = _load_checkpoint_state(checkpoint, use_ema, device)
        if sd is not None:
            model.load_state_dict(sd, strict=False)
    return dualcodec.Inference(model, device=str(device))


def _encode_decode_dualcodec(model: Any, waveform: torch.Tensor) -> torch.Tensor:
    """Encode *waveform* to DualCodec tokens and decode back to audio.

    Args:
        model: A loaded DualCodec model.
        waveform: Input tensor of shape ``(batch, 1, samples)``.

    Returns:
        Reconstructed audio tensor.
    """
    # Disable outer autocast -- DualCodec Inference manages its own
    # mixed-precision internally (float16).  The bfloat16 autocast from
    # reconstruct.py causes "unsupported ScalarType BFloat16" in
    # torchaudio resampling inside the w2v-bert-2.0 feature path.
    with torch.amp.autocast("cuda", enabled=False):
        waveform = waveform.float()
        semantic_codes, acoustic_codes = model.encode(waveform)
        return model.decode(semantic_codes, acoustic_codes)


# -- Kanade ----------------------------------------------------------------

def _load_kanade(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    device: torch.device,
) -> Any:
    """Load a Kanade tokenizer model.

    Args:
        config: Experiment config; must contain ``config["codec"]["pretrained"]``.
        checkpoint: Optional fine-tuned checkpoint path.
        use_ema: Whether to prefer EMA weights from the checkpoint.
        device: Target device (CPU or CUDA).

    Returns:
        The loaded Kanade model in eval mode.
    """
    import kanade_tokenizer

    model = kanade_tokenizer.load_model(config["codec"]["pretrained"])
    if checkpoint is not None:
        sd = _load_checkpoint_state(checkpoint, use_ema, device)
        if sd is not None:
            model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _encode_decode_kanade(model: Any, waveform: torch.Tensor) -> torch.Tensor:
    """Encode *waveform* to Kanade tokens and decode back to audio.

    Args:
        model: A loaded Kanade tokenizer model.
        waveform: Input tensor of shape ``(batch, 1, samples)``.

    Returns:
        Reconstructed audio tensor.
    """
    tokens = model.encode(waveform)
    return model.decode(tokens)


# -- Registry ---------------------------------------------------------------

def _builtin_codecs() -> List[CodecEvalHooks]:
    """Return hooks for the three built-in codecs."""
    return [
        CodecEvalHooks(
            name="mimi",
            load=_load_mimi,
            encode_decode=_encode_decode_mimi,
        ),
        CodecEvalHooks(
            name="dualcodec",
            load=_load_dualcodec,
            encode_decode=_encode_decode_dualcodec,
        ),
        CodecEvalHooks(
            name="kanade",
            load=_load_kanade,
            encode_decode=_encode_decode_kanade,
        ),
    ]
