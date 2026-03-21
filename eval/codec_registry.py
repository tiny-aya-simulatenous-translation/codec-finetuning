"""Codec registry for the evaluation pipeline.

Provides a single extension point for adding new codecs to the benchmark.
Each codec registers a :class:`CodecEvalHooks` dataclass that tells the
evaluation pipeline how to load, encode, decode, and (optionally) run
codec-specific metrics.

To add a new codec, define its hooks and call :func:`register_codec` at
module level, or add an entry to ``_BUILTIN_CODECS`` at the bottom of
this file.

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

LoadFn = Callable[
    [Dict[str, Any], Optional[str], bool, torch.device],
    Any,
]

EncodeDecodeFn = Callable[
    [Any, torch.Tensor],
    torch.Tensor,
]

CodecMetricFn = Callable[
    [Dict[str, Any], str, Any],
    Dict[str, Any],
]


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

_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    """Lazily register the three built-in codecs on first access."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    for hooks in _builtin_codecs():
        if hooks.name not in _REGISTRY:
            _REGISTRY[hooks.name] = hooks


def _strip_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the ``_orig_mod.`` prefix added by ``torch.compile``."""
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
    """Load and prefix-strip a state dict from a checkpoint file."""
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
    from transformers import MimiModel

    model = MimiModel.from_pretrained(config["codec"]["pretrained"])
    if checkpoint is not None:
        sd = _load_checkpoint_state(checkpoint, use_ema, device)
        if sd is not None:
            model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _encode_decode_mimi(model: Any, waveform: torch.Tensor) -> torch.Tensor:
    tokens = model.encode(waveform)
    return model.decode(tokens.audio_codes).audio_values


# -- DualCodec -------------------------------------------------------------

def _load_dualcodec(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    device: torch.device,
) -> Any:
    import dualcodec

    model = dualcodec.load_model(config["codec"]["pretrained"])
    if checkpoint is not None:
        sd = _load_checkpoint_state(checkpoint, use_ema, device)
        if sd is not None:
            model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _encode_decode_dualcodec(model: Any, waveform: torch.Tensor) -> torch.Tensor:
    tokens = model.encode(waveform)
    return model.decode(tokens)


# -- Kanade ----------------------------------------------------------------

def _load_kanade(
    config: Dict[str, Any],
    checkpoint: Optional[str],
    use_ema: bool,
    device: torch.device,
) -> Any:
    import kanade_tokenizer

    model = kanade_tokenizer.load_model(config["codec"]["pretrained"])
    if checkpoint is not None:
        sd = _load_checkpoint_state(checkpoint, use_ema, device)
        if sd is not None:
            model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _encode_decode_kanade(model: Any, waveform: torch.Tensor) -> torch.Tensor:
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
