# Codec Baseline Benchmark Results

**Date:** 2026-04-14
**Machine:** 1x NVIDIA GH200 480GB (aarch64, 64 Neoverse-V2 cores, 525 GB RAM)
**Codecs:** Base Mimi (kyutai/mimi, 96M params), Base DualCodec (amphion/dualcodec 12hz_v1, 200M params)
**Pipeline:** `eval/run_all.py` — Reconstruction (batched GPU) + SSNR + Bootstrap (PESQ/STOI/DNSMOS/MCD, 16 workers, 20 bootstrap resamples, 95% CI)
**Alignment:** `latency_ms: 0` (batch encode-decode produces aligned output; streaming latency trim is not needed)

---

## Datasets

| Dataset | Language | Split | Utterances | Hours | Speakers | Role | Source |
|---------|----------|-------|-----------|-------|----------|------|--------|
| hinglish-casual | Hindi | test | 1,665 | 5.1h | 7 (TTS) | In-distribution eval | `tiny-aya-translate/hinglish-casual` (private) |
| Lahaja | Hindi | test | 3,044 | 6.2h | 132 | OOD eval | `tiny-aya-translate/lahaja-eval` (public) |
| Common Voice Turkish | Turkish | test | 9,650 | 7.1h | ~1,800 | In-distribution eval | `ysdede/commonvoice_17_tr_fixed` |
| OpenSLR 108 MediaSpeech | Turkish | test | 2,513 | 10.0h | — | OOD eval (broadcast) | `ymoslem/MediaSpeech` |

---

## Hindi Results

### In-Distribution (hinglish-casual test, 1665 utterances)

| Metric | Base Mimi | Base DualCodec | Best |
|--------|-----------|----------------|------|
| STOI | **0.974** [0.974, 0.975] | 0.960 | Mimi |
| PESQ (wb) | **3.685** [3.679, 3.691] | 3.180 [3.173, 3.188] | Mimi |
| PESQ (nb) | **3.963** [3.958, 3.967] | 3.666 [3.660, 3.673] | Mimi |
| SSNR (dB) | **9.98** | 6.42 | Mimi |
| DNSMOS-OVRL | 3.346 | **3.385** | DualCodec |
| DNSMOS-SIG | 3.566 | **3.600** | DualCodec |
| DNSMOS-BAK | 4.161 | **4.166** | DualCodec |
| MCD | 178.1 | **109.2** | DualCodec |

### Out-of-Distribution (Lahaja, 3044 utterances, 132 natural speakers)

| Metric | Base Mimi | Base DualCodec | Best |
|--------|-----------|----------------|------|
| STOI | **0.920** | 0.880 | Mimi |
| PESQ (wb) | **3.296** | 2.475 | Mimi |
| PESQ (nb) | **3.795** | 3.184 | Mimi |
| SSNR (dB) | **8.78** | 3.84 | Mimi |
| DNSMOS-OVRL | 2.871 | **2.945** | DualCodec |
| DNSMOS-SIG | 3.396 | **3.466** | DualCodec |
| DNSMOS-BAK | **3.536** | **3.581** | DualCodec |
| MCD | 177.2 | **109.7** | DualCodec |

---

## Turkish Results

### In-Distribution (Common Voice test, 9650 utterances)

| Metric | Base Mimi | Base DualCodec | Best |
|--------|-----------|----------------|------|
| STOI | **0.943** | 0.901 | Mimi |
| PESQ (wb) | **2.854** | 2.267 | Mimi |
| PESQ (nb) | **3.559** | 2.901 | Mimi |
| SSNR (dB) | **7.99** | 3.32 | Mimi |
| DNSMOS-OVRL | 2.733 | **2.851** | DualCodec |
| DNSMOS-SIG | 3.190 | **3.329** | DualCodec |
| DNSMOS-BAK | 3.625 | **3.644** | DualCodec |
| MCD | 191.6 | **126.2** | DualCodec |

### Out-of-Distribution (OpenSLR 108 MediaSpeech, 2513 utterances, broadcast)

| Metric | Base Mimi | Base DualCodec | Best |
|--------|-----------|----------------|------|
| STOI | **0.954** | 0.919 | Mimi |
| PESQ (wb) | **3.415** | 2.623 | Mimi |
| PESQ (nb) | **3.860** | 3.235 | Mimi |
| SSNR (dB) | **8.37** | 3.73 | Mimi |
| DNSMOS-OVRL | 3.046 | **3.158** | DualCodec |
| DNSMOS-SIG | 3.495 | **3.572** | DualCodec |
| DNSMOS-BAK | 3.720 | **3.809** | DualCodec |
| MCD | 165.2 | **111.2** | DualCodec |

---

## Summary

### Cross-Dataset Comparison (all eval sets)

| Metric | Mimi wins | DualCodec wins | Pattern |
|--------|-----------|----------------|---------|
| STOI | 4/4 | 0/4 | Mimi has higher sample-level intelligibility |
| PESQ (wb) | 4/4 | 0/4 | Mimi has better reference-dependent perceptual quality |
| SSNR | 4/4 | 0/4 | Mimi reconstructs waveforms more faithfully |
| DNSMOS-OVRL | 0/4 | 4/4 | DualCodec produces more natural-sounding audio |
| DNSMOS-SIG | 0/4 | 4/4 | DualCodec preserves speech signal quality better |
| MCD | 0/4 | 4/4 | DualCodec has more stable spectral envelope preservation |

### Key Findings

1. **Two distinct quality profiles:** Base Mimi dominates reference-dependent metrics (STOI, PESQ, SSNR) measuring how close the reconstruction is to the original. DualCodec dominates no-reference metrics (DNSMOS) and spectral stability (MCD), indicating its output sounds more natural even when differing from the original at the sample level.

2. **DualCodec's MCD is remarkably stable** across datasets (109-126) while Mimi's varies more (165-192). DualCodec's w2v-bert-2.0 semantic pathway provides robust spectral envelope preservation regardless of domain.

3. **Turkish CV test has lower PESQ than Turkish OpenSLR** for both codecs (Mimi: 2.85 vs 3.42). Common Voice's crowd-sourced browser recordings are noisier than OpenSLR's broadcast audio — codecs reconstruct cleaner audio more faithfully. This means the "OOD" eval actually has *higher* quality than the "in-distribution" eval for Turkish.

4. **OOD generalization gap:** Hindi shows a larger OOD gap (STOI 0.974→0.920 for Mimi) than Turkish (0.943→0.954). The Lahaja dataset's 132 diverse natural speakers vs hinglish-casual's 7 TTS speakers create a genuine domain shift that doesn't exist in Turkish's CV-to-broadcast comparison.

---

## Metric Definitions

| Metric | Full Name | Scale | Higher/Lower = Better | Reference |
|--------|-----------|-------|----------------------|-----------|
| STOI | Short-Time Objective Intelligibility | 0-1 | Higher | Ref-dependent |
| PESQ (wb) | Perceptual Evaluation of Speech Quality (wideband) | 1-4.5 | Higher | Ref-dependent |
| PESQ (nb) | PESQ (narrowband) | 1-4.5 | Higher | Ref-dependent |
| SSNR | Segmental Signal-to-Noise Ratio | dB | Higher | Ref-dependent |
| DNSMOS-OVRL | Deep Noise Suppression MOS (overall) | 1-5 | Higher | No-reference |
| DNSMOS-SIG | DNSMOS (signal quality) | 1-5 | Higher | No-reference |
| DNSMOS-BAK | DNSMOS (background quality) | 1-5 | Higher | No-reference |
| MCD | Mel Cepstral Distortion | - | Lower | Ref-dependent |
