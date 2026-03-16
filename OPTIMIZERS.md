# Optimizer Reference for codec-finetuning

This document catalogs the 8 optimizers used in our hyperparameter sweeps,
explaining why each was selected, what it brings to codec fine-tuning,
and practical considerations for the sweep.

## Quick Reference Table

| # | Optimizer | Year | Key Idea | Used By | Pros | Cons | Availability | Verdict |
|---|-----------|------|----------|---------|------|------|-------------|---------|
| 1 | **AdamW** | 2017 | Decoupled weight decay from Adam's gradient-adaptive LR update | EnCodec, SoundStream, Moshi, DualCodec, Kanade -- every codec paper | Universal default. Robust. Extensive tuning recipes exist. | 2 state tensors per param (highest memory). LR sensitivity requires careful sweep. | `torch.optim.AdamW` (built-in) | **MUST INCLUDE** -- the baseline everything is measured against. |
| 2 | **RAdam** | 2019 | Rectified Adam -- dynamically adjusts adaptive LR variance during early training, reducing warmup dependence | Some audio/speech models | More stable early training without warmup. Drop-in AdamW replacement. Same memory. | Marginal benefit if warmup is already tuned. Rarely outperforms tuned AdamW+cosine. | `torch.optim.RAdam` (built-in) | **INCLUDE** -- cheap variant. May shine if sweep finds warmup=0 is optimal. |
| 3 | **Lion** | 2023 (Google, ICLR 2024) | Discovered via AutoML evolution. Uses only the **sign** of momentum (not magnitude). Update is always ±lr per param. | Google internal vision+language models. Growing community adoption. | **2x less optimizer memory** than AdamW (1 state tensor vs 2). Simpler update rule may stabilize GAN training. | Requires 3-10x larger weight decay than AdamW. LR tuning range shifts downward. Less community experience on audio. | `pip install lion-pytorch` | **ADD** -- memory savings are real when training generator + discriminator + WavLM simultaneously. |
| 4 | **Prodigy** | 2023 (ICML 2024) | **Learning-rate-free**. Automatically estimates distance-to-solution D and adapts LR proportionally. Built on D-Adaptation theory. | Diffusion model fine-tuning community (Stable Diffusion LoRA). | **Eliminates LR from the sweep entirely** -- the single most impactful hyperparameter. Particularly strong on fine-tuning (our use case). | Slower initial convergence (needs time to estimate D). Can overshoot on very small data. | `pip install prodigyopt` | **ADD** -- if it works, it removes our most expensive sweep dimension. Fine-tuning is its designed use case. |
| 5 | **Schedule-Free AdamW** | 2024 (Meta, NeurIPS 2024 Best Paper) | Eliminates LR schedules by integrating Polyak-Ruppert averaging directly into the optimizer. No warmup or cosine/linear schedule needed. | Meta internal. Won NeurIPS 2024 best paper. Growing academic adoption. | **Removes scheduler + warmup_steps from sweep** (2 fewer hyperparams). Matches or beats tuned AdamW+cosine. | Requires calling `optimizer.eval()` before validation. Relatively new. Less tested on GAN training. | `pip install schedulefree` | **ADD** -- removes 2 hyperparams from sweep. NeurIPS best paper is strong signal. |
| 6 | **SOAP** | 2024 (Shampoo + Adam hybrid) | Combines Shampoo's **second-order preconditioning** (gradient covariance) with Adam's simplicity. ~20% faster convergence than AdamW on language models. | NVIDIA NeMo emerging-optimizers. Academic LLM training. | Second-order gradient info especially valuable on **small datasets** (3.1h Turkish) where first-order signal is noisy. Better conditioned updates. | Extra memory for preconditioning matrices (~1.5-2x). Extra `shampoo_beta` hyperparameter. Higher per-step compute. | NVIDIA `emerging-optimizers` or manual impl | **ADD** -- second-order info is exactly what small-data fine-tuning needs. |
| 7 | **Adan** | 2022 (NeurIPS 2022, TPAMI 2024) | Adaptive **Nesterov triple-momentum**. Three momentum terms instead of Adam's two. Explicitly designed to work across CNNs, transformers, GANs, and diffusion models. | Tested on StyleGAN2, DiT, ViT, BERT, GPT-2. | **The only optimizer explicitly validated on CNN+transformer+GAN hybrids** -- exactly our codec architecture. Strong GAN and diffusion results. Nesterov look-ahead may escape sharp minima. | Three beta params (beta1, beta2, beta3) instead of two -- larger sweep space. Less mainstream adoption. | `pip install adan-pytorch` | **ADD** -- designed for exactly our architecture mix (CNN encoder + transformer + GAN discriminator). |
| 8 | **Muon** | 2024 (Keller Jordan et al.) | **Orthogonalizes** SGD-momentum updates via Newton-Schulz iteration. Replaces each update with nearest semi-orthogonal matrix, boosting rare gradient directions. | NanoGPT speedrun record (35% faster than AdamW, 12 consecutive records by 7 researchers). CIFAR-10 speedrun record. Kimi K2 (1T params). | **Less memory than AdamW** (same as SGD: 1 state tensor). <1% FLOP overhead. Works for CNNs (via flattening) and transformers. Strongest empirical evidence of any new optimizer. | Must use as **hybrid**: Muon for hidden 2D weights, AdamW for embeddings/norms/biases. Author's open question: "Will it work for fine-tuning?" -- **our sweep directly answers this**. Not tested on GAN training. | `torch.optim.Muon` (PyTorch 2.10+) or ~50-line vendored impl | **ADD** -- our sweep produces novel results: first Muon test on codec fine-tuning, GAN adversarial training, and audio domain. |

## Optimizer-Specific Sweep Considerations

When running the hyperparameter sweep, not all parameters are relevant for every
optimizer. The sweep agent (`train/sweep_agent.py`) automatically handles these
interactions.

| Optimizer | LR Swept? | Scheduler Swept? | Warmup Swept? | Extra Params to Sweep | Weight Decay Adjustment |
|-----------|:---------:|:----------------:|:-------------:|----------------------|------------------------|
| AdamW | YES `[1e-5, 1e-3]` | YES | YES | `beta2`: `[0.95, 0.99, 0.999]` | Standard `[0, 0.1]` |
| RAdam | YES `[1e-5, 1e-3]` | YES | Reduced importance | `beta2`: `[0.95, 0.99, 0.999]` | Standard `[0, 0.1]` |
| Lion | YES `[1e-6, 3e-4]` (lower) | YES | YES | -- | **3-10x higher**: `[0.03, 0.3]` |
| Prodigy | **NO** (auto-estimated) | **NO** (internal) | **NO** (internal) | `d_coef`: `[0.5, 1.0, 2.0]` | Standard `[0, 0.1]` |
| Schedule-Free | YES `[1e-5, 1e-3]` | **NO** (eliminated) | **NO** (eliminated) | -- | Standard `[0, 0.1]` |
| SOAP | YES `[1e-5, 1e-3]` | YES | YES | `shampoo_beta`: `[0.8, 0.9, 0.95]` | Standard `[0, 0.1]` |
| Adan | YES `[1e-5, 1e-3]` | YES | YES | `beta1, beta2, beta3` | Standard `[0, 0.1]` |
| Muon | YES `[0.01, 0.1]` (SGD-scale) | YES (AdamW part) | YES (AdamW part) | `ns_steps`: `[3, 5, 7]` | AdamW part only |

## Memory Footprint Comparison

When training a codec, we simultaneously hold the generator model, discriminator,
and a frozen SSL teacher (WavLM or w2v-bert-2.0) in GPU memory. Optimizer memory
matters.

| Optimizer | State Tensors per Param | Relative Memory | Notes |
|-----------|:-----------------------:|:---------------:|-------|
| AdamW | 2 (mean + variance) | 1.0x (baseline) | The standard |
| RAdam | 2 (mean + variance) | 1.0x | Same as AdamW |
| Lion | 1 (momentum) | **0.5x** | Half the optimizer memory of AdamW |
| Prodigy | 2 + D estimate | ~1.1x | Slight overhead for distance estimation |
| Schedule-Free | 2 + z-buffer | ~1.3x | Extra buffer for Polyak averaging |
| SOAP | 2 + preconditioners | ~1.5-2.0x | Preconditioning matrices add memory |
| Adan | 3 (triple momentum) | 1.5x | Three momentum terms |
| Muon (hidden) + AdamW (rest) | 1 (hidden) + 2 (rest) | **~0.7x** blended | Lowest overall memory |

## What Would Be Novel Findings

Our sweep explores territory that has not been tested before. Any of these
outcomes would be a noteworthy contribution:

1. **Muon on fine-tuning** -- The author explicitly lists this as an open question.
   Our sweep is the first test.
2. **Muon on GAN/adversarial training** -- Never tested. Novel territory.
3. **Muon on audio/codec domain** -- First application outside language and vision.
4. **Prodigy on codec fine-tuning** -- LR-free codec training would simplify the
   field significantly.
5. **Any non-AdamW optimizer beating AdamW on codecs** -- Would challenge the
   default assumption in every codec paper.
6. **Adan on real-world CNN+transformer+GAN** -- Validates the paper's claims on a
   practical architecture.
7. **Schedule-Free on adversarial training** -- Removes schedule tuning from GAN
   training.

## Implementation: optimizer_factory.py

All 8 optimizers are created through a single factory function in
`train/optimizer_factory.py`. Key behaviors:

- **Muon (hybrid):** Automatically separates model parameters into 2D hidden-layer
  weights (routed to Muon) and everything else -- embeddings, LayerNorm, biases,
  input/output layers (routed to AdamW). Uses parameter shape detection.
- **Lion:** Adjusts weight decay range (3-10x higher than AdamW per the paper).
- **Prodigy:** Disables external LR scheduling. Sets `lr=1.0` as the paper
  recommends (Prodigy internally adapts).
- **Schedule-Free:** Disables external LR scheduling. Calls `optimizer.eval()`
  before validation and `optimizer.train()` before training.
- **SOAP:** Adds `shampoo_beta` parameter for preconditioning momentum.
- **Adan:** Exposes three beta parameters (beta1, beta2, beta3).

## References

- **AdamW:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
- **RAdam:** Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond", ICLR 2020
- **Lion:** Chen et al., "Symbolic Discovery of Optimization Algorithms", ICLR 2024
- **Prodigy:** Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter-Free Learner", ICML 2024
- **Schedule-Free:** Defazio et al., "The Road Less Scheduled", NeurIPS 2024 (Best Paper)
- **SOAP:** Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam", NeurIPS 2024
- **Adan:** Xie et al., "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models", TPAMI 2024
- **Muon:** Jordan et al., "Muon: An optimizer for hidden layers in neural networks", 2024
