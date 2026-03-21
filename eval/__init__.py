"""Evaluation modules for codec-finetuning benchmark.

The codec registry (:mod:`eval.codec_registry`) is the single extension
point for adding new codecs to the evaluation pipeline.

License: MIT
"""

from eval.codec_registry import (  # noqa: F401
    CodecEvalHooks,
    register_codec,
    get_codec_hooks,
    registered_codecs,
)
