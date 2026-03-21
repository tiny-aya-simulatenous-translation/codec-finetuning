"""Evaluation package for the codec-finetuning benchmark.

This package implements a multi-stage evaluation pipeline for neural audio
codecs.  It measures reconstruction quality, latency, and signal-to-noise
characteristics, then aggregates results with bootstrap confidence intervals
and publishes them to Weights & Biases.

Modules
-------
- :mod:`eval.codec_registry` -- Extension point for registering new codecs.
- :mod:`eval.reconstruct` -- Stage 1: encode/decode test utterances (batched GPU).
- :mod:`eval.measure_ssnr` -- Stage 2: segmental signal-to-noise ratio.
- :mod:`eval.measure_ttfat` -- Stage 3: time-to-first-audio-token latency.
- :mod:`eval.bootstrap_eval` -- Stage 4: PESQ, STOI, DNSMOS, MCD with bootstrap CIs.
- :mod:`eval.log_to_wandb` -- Stage 6: publish all metrics & audio to WandB.
- :mod:`eval.run_all` -- Orchestrator that runs stages 1-6 in sequence.
- :mod:`eval._protobuf_compat` -- Internal shim for protobuf ≥5 compatibility.

Public API re-exports
---------------------
The most commonly used symbols from :mod:`eval.codec_registry` are
re-exported here for convenience::

    from eval import register_codec, get_codec_hooks, registered_codecs

License: MIT
"""

from eval.codec_registry import (  # noqa: F401
    CodecEvalHooks,
    register_codec,
    get_codec_hooks,
    registered_codecs,
)
