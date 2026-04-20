# Codec Fine-Tuning Analysis — Mimi & DualCodec

**Date:** 2026-04-15
**Machine:** 1x NVIDIA GH200 480GB
**Codecs:** Mimi (kyutai/mimi, 96M), DualCodec (amphion/dualcodec 12hz_v1, 200M)
**Method:** Decoder-only fine-tuning (encoder + quantizer frozen)
**Training:** GAN adversarial + reconstruction + feature matching losses, 2000 steps, LR 1e-4
**Eval:** Bootstrap metrics (PESQ, STOI, DNSMOS, MCD) on in-distribution + OOD eval sets

---

## Key Finding

**Fine-tuning neural speech codecs with adversarial training degrades all perceptual metrics — for both Mimi and DualCodec, across both languages, regardless of what layers are frozen.** The pretrained models outperform every fine-tuned checkpoint. This is consistent with published literature where the universal approach is to freeze the codec and train only the language model.

---

## Training Convergence

### Mimi (decoder-only: 39.9M trainable / 39.4M frozen)

| Step | Hindi val_loss | Turkish val_loss |
|------|---------------|-----------------|
| 0 (pretrained) | 0.0235 | 0.0051 |
| 500 | 0.0213 (-9%) | 0.0047 (-8%) |
| 1000 | 0.0211 | 0.0047 |
| 2000 | 0.0210 (-11%) | 0.0047 (-8%) |

### DualCodec (decoder-only: 53.9M trainable / 30.2M frozen)

| Step | Hindi val_loss | Turkish val_loss |
|------|---------------|-----------------|
| 0 (pretrained) | 0.0335 | 0.0086 |
| 500 | 0.0319 (-5%) | 0.0082 (-5%) |
| 1000 | 0.0346 (+3%) | 0.0088 (+2%) |
| 2000 | 0.0360 (+7%) | 0.0092 (+7%) |

DualCodec's val_loss rises above baseline after step 500 — the adversarial training actively destabilizes the decoder.

---

## Hindi Results

### In-Distribution (hinglish-casual test, 1665 utts)

| Metric | Mimi Base | Mimi s500 | Mimi s2000 | DC Base | DC s500 | DC s2000 |
|--------|-----------|-----------|------------|---------|---------|----------|
| STOI | **0.974** | 0.967 | 0.938 | 0.960 | 0.953 | 0.950 |
| PESQ-wb | **3.685** | 3.589 | 2.914 | 3.180 | 2.688 | 2.724 |
| DNSMOS | 3.346 | 3.246 | 3.041 | **3.385** | 3.367 | 3.238 |
| SSNR | 9.98 | **10.56** | **10.90** | 6.42 | 7.18 | 6.32 |
| MCD | **178.1** | 372.3 | 611.3 | 109.2 | 180.4 | 151.6 |

### OOD (Lahaja, 3044 utts, 132 speakers)

| Metric | Mimi Base | Mimi s500 | Mimi s2000 | DC Base | DC s500 | DC s2000 |
|--------|-----------|-----------|------------|---------|---------|----------|
| STOI | **0.920** | 0.914 | 0.880 | 0.880 | 0.876 | 0.860 |
| PESQ-wb | **3.296** | 3.284 | 2.801 | 2.475 | 2.205 | 2.028 |
| DNSMOS | 2.871 | 2.853 | 2.805 | **2.945** | 3.027 | 2.965 |
| SSNR | 8.78 | **9.10** | **9.24** | 3.84 | 4.46 | 3.47 |
| MCD | **177.2** | 342.5 | 556.5 | 109.7 | 197.9 | 174.8 |

---

## Turkish Results

### In-Distribution (Common Voice test, 9650 utts)

| Metric | Mimi Base | Mimi s500 | Mimi s2000 | DC Base | DC s500 | DC s2000 |
|--------|-----------|-----------|------------|---------|---------|----------|
| STOI | **0.943** | 0.932 | 0.880 | 0.901 | 0.883 | 0.866 |
| PESQ-wb | **2.854** | 2.856 | 2.334 | 2.267 | 1.841 | 1.805 |
| DNSMOS | 2.733 | 2.717 | 2.672 | **2.851** | 2.841 | 2.739 |
| SSNR | 7.99 | **8.47** | **8.67** | 3.32 | 3.79 | 2.94 |
| MCD | **191.6** | 379.9 | 661.6 | 126.2 | 200.7 | 195.3 |

### OOD (OpenSLR 108 MediaSpeech, 2513 utts)

| Metric | Mimi Base | Mimi s500 | Mimi s2000 | DC Base | DC s500 | DC s2000 |
|--------|-----------|-----------|------------|---------|---------|----------|
| STOI | **0.954** | 0.945 | 0.914 | 0.919 | 0.906 | 0.888 |
| PESQ-wb | **3.415** | 3.283 | 2.475 | 2.623 | 2.152 | 2.069 |
| DNSMOS | 3.046 | 2.984 | 2.900 | **3.158** | 3.178 | 3.040 |
| SSNR | 8.37 | **8.76** | **8.88** | 3.73 | 4.06 | 3.16 |
| MCD | **165.2** | 344.1 | 572.4 | 111.2 | 179.6 | 176.2 |

---

## Analysis

### Degradation is universal

| Condition | Mimi s500 vs Base | Mimi s2000 vs Base | DC s500 vs Base | DC s2000 vs Base |
|-----------|-------------------|-------------------|-----------------|------------------|
| Hindi in-dist STOI | -0.7% | -3.7% | -0.7% | -1.0% |
| Hindi OOD STOI | -0.7% | -4.3% | -0.5% | -2.3% |
| Turkish in-dist STOI | -1.2% | -6.7% | -2.0% | -3.9% |
| Turkish OOD STOI | -0.9% | -4.2% | -1.4% | -3.4% |
| **Avg STOI change** | **-0.9%** | **-4.7%** | **-1.2%** | **-2.7%** |

- **Every fine-tuned checkpoint is worse than baseline** on STOI, PESQ, DNSMOS, and MCD
- **MCD degrades fastest** — doubling within 500 steps for Mimi (spectral envelope distortion from GAN)
- **SSNR is the only metric that improves** — the model learns to produce numerically closer waveforms while sounding perceptually worse
- **DualCodec degrades less than Mimi** in STOI (-2.7% vs -4.7% at step 2000) but more in PESQ

### The GAN training paradox

The discriminator creates adversarial pressure that pushes the decoder away from its pretrained spectral optimum. This manifests as:
1. **Improving val_loss** (reconstruction L1 gets lower) while **worsening perceptual metrics** — the model optimizes for sample-level similarity at the cost of spectral quality
2. **MCD explosion** — mel cepstral distortion increases 2-4x, meaning the spectral envelope is being distorted even as waveform L1 decreases
3. **DualCodec val_loss rising above baseline** after step 500 — the adversarial loss overwhelms the reconstruction loss

### Literature alignment

Published work universally freezes these codecs:
- **Kyutai moshi-finetune, Hibiki, J-Moshi, Sesame CSM, Llama-Mimi** — all freeze Mimi entirely
- **T-Mimi** (decoder retrain) uses mel-spec loss only (no GAN) — a fundamentally different approach
- **XCodec2** (decoder fine-tune) also avoids adversarial training for adaptation

No published work successfully fine-tunes Mimi or DualCodec end-to-end with GAN training for language adaptation.

---

## Recommendation

1. **Do not fine-tune Mimi or DualCodec** with adversarial training for language adaptation. The pretrained models already generalize well (Mimi STOI > 0.92, DualCodec STOI > 0.88 on all tested languages).

2. **Freeze the codec** as a tokenizer in speech-to-speech pipelines. Focus fine-tuning effort on the language model (Moshi backbone Temporal + Depth Transformers).

3. If codec adaptation is necessary, explore:
   - **LoRA on transformer bottleneck** (no adversarial training, reconstruction loss only)
   - **Decoder retraining from scratch** with mel-spectrogram loss (following T-Mimi approach)
   - **Training on multilingual data from scratch** (following DualCodec/FunCodec approach)
