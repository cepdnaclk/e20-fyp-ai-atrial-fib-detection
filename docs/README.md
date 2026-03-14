---
layout: home
permalink: index.html

repository-name: e20-fyp-ai-atrial-fib-detection
title: Diffusion-Based Counterfactual ECG Generation for Atrial Fibrillation Data Augmentation
---

<style>
/* ── Custom Styling for Research Project Page ── */

/* Hero section */
.hero-banner {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: #fff;
  padding: 2.5rem 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
  text-align: center;
}
.hero-banner h1 {
  font-size: 1.8rem;
  margin-bottom: 0.5rem;
  color: #fff;
  line-height: 1.3;
}
.hero-banner .subtitle {
  font-size: 1rem;
  opacity: 0.85;
  margin-bottom: 1rem;
}
.hero-banner .badge-row {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

/* Highlight box for key results */
.highlight-box {
  background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
  border-left: 5px solid #2e7d32;
  padding: 1.2rem 1.5rem;
  border-radius: 0 8px 8px 0;
  margin: 1.5rem 0;
}
.highlight-box.blue {
  background: linear-gradient(135deg, #e3f2fd, #e8eaf6);
  border-left-color: #1565c0;
}
.highlight-box.amber {
  background: linear-gradient(135deg, #fff8e1, #fff3e0);
  border-left-color: #f57f17;
}

/* Section styling */
h2 {
  border-bottom: 2px solid #1565c0;
  padding-bottom: 0.4rem;
  margin-top: 2.5rem;
}

/* Figure containers */
.figure-container {
  text-align: center;
  margin: 1.5rem auto;
}
.figure-container img {
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.figure-container .caption {
  font-size: 0.9rem;
  color: #555;
  font-style: italic;
  margin-top: 0.5rem;
  padding: 0 1rem;
  line-height: 1.4;
}

/* Two-column figure layout */
.figure-row {
  display: flex;
  gap: 1rem;
  margin: 1.5rem 0;
  align-items: flex-start;
}
.figure-row .figure-col {
  flex: 1;
  text-align: center;
}
.figure-row .figure-col img {
  width: 100%;
  max-width: 100%;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.figure-row .figure-col .caption {
  font-size: 0.85rem;
  color: #555;
  font-style: italic;
  margin-top: 0.4rem;
}

/* Results metric cards */
.metric-cards {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin: 1.5rem 0;
}
.metric-card {
  flex: 1;
  min-width: 180px;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
}
.metric-card .value {
  font-size: 1.6rem;
  font-weight: 700;
  color: #1565c0;
}
.metric-card .label {
  font-size: 0.85rem;
  color: #666;
  margin-top: 0.25rem;
}

/* Team cards */
.team-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin: 1rem 0;
}
.team-member {
  background: #f8f9fa;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 1rem 1.2rem;
  flex: 1;
  min-width: 250px;
}
.team-member .name {
  font-weight: 600;
  font-size: 1rem;
  color: #1a1a2e;
}
.team-member .role {
  font-size: 0.85rem;
  color: #666;
}
.team-member .links {
  margin-top: 0.4rem;
  font-size: 0.85rem;
}
.team-member .links a {
  margin-right: 0.6rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .figure-row { flex-direction: column; }
  .metric-cards { flex-direction: column; }
  .team-grid { flex-direction: column; }
  .hero-banner h1 { font-size: 1.4rem; }
}
</style>

<!-- Hero Banner -->
<div class="hero-banner">
  <h1>Diffusion-Based Counterfactual ECG Generation<br>for Atrial Fibrillation Data Augmentation</h1>
  <div class="subtitle">Final Year Research Project — Group 29 · Department of Computer Engineering · University of Peradeniya</div>
  <div class="badge-row">
    <a href="https://github.com/cepdnaclk/e20-fyp-ai-atrial-fib-detection"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub"></a>
    <a href="https://huggingface.co/TharakaDil2001/diffusion-ecg-augmentation"><img src="https://img.shields.io/badge/🤗_Model-Hub-yellow" alt="Model Hub"></a>
    <a href="https://huggingface.co/spaces/TharakaDil2001/ecg-augmentation-demo"><img src="https://img.shields.io/badge/🤗_Demo-Space-orange" alt="Demo"></a>
    <a href="https://www.search-project.eu/"><img src="https://img.shields.io/badge/SEARCH-EU_Horizon-blue" alt="SEARCH"></a>
  </div>
</div>

<!-- Key Result Highlight -->
<div class="highlight-box">
  <strong>🔬 Key Result:</strong> Replacing 33% of real training data with diffusion-generated synthetic ECGs yields <strong>statistically equivalent</strong> classifier performance (TOST p &lt; 0.001, Δ = ±2%), with <strong>zero near-copies</strong> of training data among generated samples — enabling privacy-preserving dataset sharing and class balancing.
</div>

---

#### Team

<div class="team-grid">
  <div class="team-member">
    <div class="name">Tharaka Dilshan</div>
    <div class="role">E/20/069 · Student Researcher</div>
    <div class="links">
      <a href="mailto:e20069@eng.pdn.ac.lk">📧 Email</a>
      <a href="https://www.linkedin.com/in/tharaka-dilshan-237a8b345/">LinkedIn</a>
      <a href="https://orcid.org/0009-0006-1672-2317">ORCID</a>
    </div>
  </div>
  <div class="team-member">
    <div class="name">Nethmini Karunarathne</div>
    <div class="role">E/20/189 · Student Researcher</div>
    <div class="links">
      <a href="mailto:e20189@eng.pdn.ac.lk">📧 Email</a>
      <a href="https://www.linkedin.com/in/nethmini-karunarathne-b20460206/">LinkedIn</a>
      <a href="https://orcid.org/0009-0009-5459-7846">ORCID</a>
    </div>
  </div>
</div>

#### Supervisors

<div class="team-grid">
  <div class="team-member">
    <div class="name">Dr. Vajira Thambawita</div>
    <div class="role">SimulaMet, Oslo, Norway</div>
    <div class="links"><a href="https://vajira.info/">🌐 Website</a> <a href="https://orcid.org/0000-0001-6026-0929">ORCID</a></div>
  </div>
  <div class="team-member">
    <div class="name">Prof. Mary M. Maleckar</div>
    <div class="role">Tulane University / Simula Research Laboratory</div>
    <div class="links"><a href="https://orcid.org/0000-0002-7012-3853">ORCID</a></div>
  </div>
  <div class="team-member">
    <div class="name">Dr. Roshan Ragel</div>
    <div class="role">University of Peradeniya</div>
    <div class="links"><a href="https://orcid.org/0000-0002-4511-2335">ORCID</a></div>
  </div>
  <div class="team-member">
    <div class="name">Dr. Isuru Nawinne</div>
    <div class="role">University of Peradeniya</div>
    <div class="links"><a href="https://orcid.org/0009-0001-4760-3533">ORCID</a></div>
  </div>
</div>

#### Collaborators

<div class="team-grid">
  <div class="team-member">
    <div class="name">Isuri Devindi</div>
    <div class="role">University of Maryland, College Park, USA</div>
    <div class="links"><a href="https://orcid.org/0009-0005-6615-7937">ORCID</a></div>
  </div>
  <div class="team-member">
    <div class="name">Prof. Jørgen K. Kanters</div>
    <div class="role">University of Copenhagen, Denmark</div>
    <div class="links"><a href="https://orcid.org/0000-0002-3267-4910">ORCID</a></div>
  </div>
</div>

#### Table of Content

1. [Abstract](#abstract)
2. [Related Works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract

Automated detection of atrial fibrillation (AFib) using deep learning requires large, balanced ECG datasets, yet clinical data remains scarce, imbalanced, and constrained by privacy regulations. We present a **diffusion-based data augmentation pipeline** that generates synthetic ECG segments by transforming existing recordings into counterfactual waveforms of the opposing class. The pipeline uses a **partial-noise conditional denoising process** with **classifier-free guidance**, operating on single-lead ECG signals. A **content-style disentangled UNet architecture** separates class-invariant morphology from class-discriminative rhythm features. A **multi-stage plausibility post-validator** enforces morphological and physiological constraints and verifies rhythm consistency criteria, retaining only waveforms that satisfy quality thresholds.

We evaluate the generated counterfactual data through a **three-regime protocol** using a ResNet-BiLSTM classifier: training on originals only, counterfactuals only, and an augmented mixture. The augmented mixture achieves **95.05% accuracy** and **98.60% AUROC**, statistically equivalent to original-only training (95.63% accuracy, TOST p = 0.007, Δ = ±2%). Furthermore, **none of the accepted counterfactuals are near-copies** of training data (maximum correlation: 0.30), indicating the generated signals are novel and privacy-preserving.

<div class="highlight-box blue">
  This research is conducted as part of the EU-funded <strong><a href="https://www.search-project.eu/">SEARCH Initiative</a></strong> through an international collaboration between the University of Peradeniya (Sri Lanka), SimulaMet (Norway), Tulane University (USA), and the University of Copenhagen (Denmark).
</div>

---

## Related Works

Generative models for ECG synthesis have been explored using GANs, VAEs, and more recently, diffusion models. The table below compares representative methods:

| Method | Model | Counterfactual | Key Limitations |
|---|---|---|---|
| PGAN-ECG (Golany et al.) | GAN | No | Limited rhythm diversity |
| WaveGAN-ECG (Donahue et al.) | GAN | No | Mode collapse on long signals |
| ECG-VAE (Biswal et al.) | VAE | No | Blurred morphology |
| CoFE-GAN (Jang et al., 2025) | GAN | Yes | Requires iterative latent inversion |
| GCX-ECG (Alcaraz-Segura et al.) | GAN | Yes | Weak morphology preservation |
| **Ours** | **Diffusion** | **Yes** | Single-lead; automated validation only |

<div class="highlight-box">
  <strong>Key advantages of our approach:</strong>
  <ul>
    <li>Operates directly in waveform space — no latent inversion required</li>
    <li>Content-style disentanglement preserves morphology while modifying rhythm</li>
    <li>Multi-stage plausibility validation ensures clinical realism</li>
    <li>Formal statistical equivalence testing quantifies augmentation safety</li>
  </ul>
</div>

---

## Methodology

### Pipeline Overview

Our pipeline consists of five stages: data preparation, classifier training, two-stage diffusion model training, counterfactual generation with filtering, and three-regime augmentation evaluation. The figure below shows the complete training and inference pipeline:

<div class="figure-container">
  <img src="./images/pipeline_overview.png" alt="Pipeline Overview" style="width: 100%; max-width: 960px;">
  <div class="caption"><strong>Figure 1.</strong> Overview of the proposed pipeline. <strong>(A) Training</strong> proceeds in two stages: Stage 1 (epochs 1–50) trains the UNet for noise reconstruction conditioned on label y; Stage 2 (epochs 51–100) fine-tunes for counterfactual generation conditioned on y' = 1−y, adding L_flip and L_morph losses. The classifier is frozen. <strong>(B) Inference:</strong> partial-noise initialization, DDIM denoising with CFG, and two quality gates filter the output.</div>
</div>

### Two-Stage Training

<div class="highlight-box blue">
  <strong>Stage 1 (Epochs 1–50): Reconstruction</strong> — The content encoder extracts class-invariant morphological features (256-d VAE latent), and the style encoder captures class-discriminative rhythm features (128-d embedding). The conditional 1D UNet learns to denoise ECG signals with classifier-free guidance (10% label dropout).<br><br>
  <strong>Stage 2 (Epochs 51–100): Counterfactual Fine-tuning</strong> — The UNet is conditioned on the <em>target</em> class label (y' = 1 − y), while a frozen AFib classifier provides supervision through a flip loss, and a morphology preservation loss constrains deviation from the source signal.
</div>

### Inference Pipeline

1. **Partial-noise initialization** — Corrupt source ECG to 60% noise level
2. **DDIM denoising** — 50 steps with classifier-free guidance (scale w = 3)
3. **Post-processing** — Savitzky-Golay smoothing
4. **Gate 1: Flip verification** — Frozen classifier must predict target class
5. **Gate 2: Plausibility check** — Score P = 0.3·M + 0.3·Φ + 0.4·C ≥ 0.7
6. **Accept** — Output filtered counterfactual with verified label

### Content-Style Disentanglement

The architecture separates ECG signals into two representations:

| Component | Architecture | Output | Purpose |
|---|---|---|---|
| Content Encoder | 5 Conv layers + BatchNorm + VAE | c ∈ ℝ²⁵⁶ | Class-invariant morphology (beat shape, amplitude) |
| Style Encoder | 4 Conv layers + InstanceNorm | s ∈ ℝ¹²⁸ | Class-discriminative rhythm (RR regularity, P-waves) |

### Plausibility Validator

The plausibility score **P = 0.3·M + 0.3·Φ + 0.4·C** combines three components:

| Component | Criteria |
|---|---|
| **Morphology (M)** | Amplitude within ±3 norm. units, ≥3 R-peaks, QRS integrity, low spike rate |
| **Physiology (Φ)** | RR intervals within 0.3–2.0s (30–200 bpm), RR coefficient of variation < 0.6 |
| **Clinical Directionality (C)** | Normal→AFib: RR variability increase ≥30%; AFib→Normal: decrease ≥20% |

---

## Experiment Setup and Implementation

### Dataset

We use the **MIMIC-IV ECG Diagnostic Electrocardiogram Matched Subset** (PhysioNet), processing Lead II recordings:

| Property | Value |
|---|---|
| Source | MIMIC-IV ECG (PhysioNet) |
| Lead | Lead II |
| Sampling rate | 250 Hz (resampled from 500 Hz) |
| Segment length | 10 seconds (2,500 samples) |
| Filtering | Bandpass 0.5–40 Hz (NeuroKit2) |
| Normalization | Global min-max to [−1.5, 1.5] |
| Split | Patient-level 70/15/15 |

| Partition | Segments | Normal / AFib |
|---|---|---|
| Training | 104,855 | 52,447 / 52,408 |
| Validation | 22,469 | 11,239 / 11,230 |
| Test | 22,469 | 11,239 / 11,230 |
| **Total** | **149,793** | **74,925 / 74,868** |

<div class="figure-container">
  <img src="./images/final_distribution.png" alt="Dataset Distribution" style="width: 70%; max-width: 550px;">
  <div class="caption"><strong>Figure 2.</strong> Class distribution in the MIMIC-IV ECG dataset after preprocessing.</div>
</div>

### AFib Classifier

A **ResNet-BiLSTM** classifier serves dual purposes: (1) style encoder guidance during diffusion training, and (2) downstream evaluation of generated ECGs.

- **Architecture:** Multi-scale CNN front-end → ResNet-34 → BiLSTM → Self-attention
- **Training:** Focal loss (α = 0.65, γ = 2.0), Adam optimizer (lr = 10⁻³), early stopping on validation AUROC
- **After training:** Frozen for all subsequent experiments

### Three-Regime Evaluation Protocol

To assess augmentation safety, we train identical classifiers under three data regimes:

| | Dataset A (Original) | Dataset B (CF Only) | Dataset C (Augmented) |
|---|---|---|---|
| Original ECGs | 18,681 | 0 | 12,454 |
| CF ECGs | 0 | 6,227 × 3 | 6,227 |
| CF fraction | 0% | 100% | 33.33% |
| **Training total** | **18,681** | **18,681** | **18,681** |

All classifiers evaluated on the same held-out **test set of 22,469 original ECGs**.

### Hardware & Training

<div class="metric-cards">
  <div class="metric-card">
    <div class="value">48 GB</div>
    <div class="label">NVIDIA RTX 6000 Ada</div>
  </div>
  <div class="metric-card">
    <div class="value">21.6 hrs</div>
    <div class="label">Training Time</div>
  </div>
  <div class="metric-card">
    <div class="value">19.1M</div>
    <div class="label">Parameters</div>
  </div>
  <div class="metric-card">
    <div class="value">~4 hrs</div>
    <div class="label">Generation (22K samples)</div>
  </div>
</div>

---

## Results and Analysis

### Generated Counterfactual ECG Examples

Of 22,469 raw counterfactuals generated, **7,784 passed all quality gates** (34.6% acceptance rate), with equal counts from each conversion direction (3,892 Normal→AFib, 3,892 AFib→Normal).

<div class="figure-container">
  <img src="./images/ecg_comparison.png" alt="ECG Comparison" style="width: 90%; max-width: 750px;">
  <div class="caption"><strong>Figure 3.</strong> Counterfactual ECG transformations. Normal→AFib (top rows) and AFib→Normal (bottom rows). Left: original signal; center: generated counterfactual; right: overlay with Pearson correlation. Green badges show classifier predictions confirming successful class conversion.</div>
</div>

<div class="figure-row">
  <div class="figure-col">
    <img src="./images/viz_AFib_to_Normal_001000.png" alt="AFib to Normal 1">
    <div class="caption">AFib → Normal (#1)</div>
  </div>
  <div class="figure-col">
    <img src="./images/viz_Normal_to_AFib_001000.png" alt="Normal to AFib 1">
    <div class="caption">Normal → AFib (#1)</div>
  </div>
</div>

<div class="figure-row">
  <div class="figure-col">
    <img src="./images/viz_AFib_to_Normal_003000.png" alt="AFib to Normal 2">
    <div class="caption">AFib → Normal (#2)</div>
  </div>
  <div class="figure-col">
    <img src="./images/viz_Normal_to_AFib_003000.png" alt="Normal to AFib 2">
    <div class="caption">Normal → AFib (#2)</div>
  </div>
</div>

### Training Progress

<div class="figure-row">
  <div class="figure-col">
    <img src="./images/epoch_050_reconstruction.png" alt="Stage 1 Epoch 50">
    <div class="caption"><strong>Stage 1:</strong> Reconstruction (Epoch 50)</div>
  </div>
  <div class="figure-col">
    <img src="./images/epoch_100_counterfactual.png" alt="Stage 2 Epoch 100">
    <div class="caption"><strong>Stage 2:</strong> Counterfactual (Epoch 100)</div>
  </div>
</div>

### Signal Quality and Plausibility

<div class="metric-cards">
  <div class="metric-card">
    <div class="value">0.76</div>
    <div class="label">Mean Plausibility Score</div>
  </div>
  <div class="metric-card">
    <div class="value">0.30</div>
    <div class="label">Max Nearest-Neighbor Corr.</div>
  </div>
  <div class="metric-card">
    <div class="value" style="color: #2e7d32;">0</div>
    <div class="label">Near-copies Detected</div>
  </div>
  <div class="metric-card">
    <div class="value">12.58 dB</div>
    <div class="label">PSNR</div>
  </div>
</div>

<div class="figure-row">
  <div class="figure-col">
    <img src="./images/fig2_signal_quality_distributions.png" alt="Signal Quality">
    <div class="caption"><strong>Figure 4.</strong> Signal quality distributions</div>
  </div>
  <div class="figure-col">
    <img src="./images/fig3_clinical_features.png" alt="Clinical Features">
    <div class="caption"><strong>Figure 5.</strong> Clinical feature analysis</div>
  </div>
</div>

### Augmentation Evaluation (5-Fold Cross-Validation)

| Training Regime | Accuracy | F1 Score | AUROC |
|---|---|---|---|
| **A** — Original only | 95.63 ± 0.33% | 95.65 ± 0.35% | 98.90 ± 0.16% |
| **B** — CF only | 85.94 ± 1.32% | 86.70 ± 1.24% | 93.24 ± 1.47% |
| **C** — Augmented (67% + 33%) | **95.05 ± 0.50%** | **95.09 ± 0.46%** | **98.60 ± 0.17%** |

<div class="figure-container">
  <img src="./images/performance_comparison_5fold.png" alt="Performance Comparison" style="width: 80%; max-width: 650px;">
  <div class="caption"><strong>Figure 6.</strong> Classifier performance across three training datasets. Error bars show standard deviation over 5 folds.</div>
</div>

<div class="figure-row">
  <div class="figure-col">
    <img src="./images/roc_curves_5fold.png" alt="ROC Curves">
    <div class="caption"><strong>Figure 7.</strong> ROC curves (5-fold)</div>
  </div>
  <div class="figure-col">
    <img src="./images/confusion_matrices_5fold.png" alt="Confusion Matrices">
    <div class="caption"><strong>Figure 8.</strong> Confusion matrices (5-fold)</div>
  </div>
</div>

### Statistical Equivalence Testing

Formal statistical tests confirm that **Dataset C (augmented) is equivalent to Dataset A (original only)**:

| Test | N | Result | p-value |
|---|---|---|---|
| McNemar's test | 22,469 | Δ = 0.54% | < 0.001 |
| **TOST equivalence (±2%)** | 22,469 | **Equivalent** | **< 0.001** |
| **Non-inferiority (2%)** | 22,469 | **Non-inferior** | **< 0.001** |
| Dunnett's (A vs C) | 5 | No sig. diff. | 0.340 |
| Dunnett's (A vs B) | 5 | Sig. diff. | < 0.001 |

<div class="figure-row">
  <div class="figure-col">
    <img src="./images/augmentation_viability_summary.png" alt="Viability Summary">
    <div class="caption"><strong>Figure 9.</strong> Augmentation viability analysis</div>
  </div>
  <div class="figure-col">
    <img src="./images/augmentation_pairwise_differences.png" alt="Pairwise Differences">
    <div class="caption"><strong>Figure 10.</strong> Pairwise performance differences</div>
  </div>
</div>

### Key Findings

<div class="highlight-box">
<ol>
  <li><strong>Augmentation is safe:</strong> 33% synthetic content yields statistically equivalent classifier performance (TOST p &lt; 0.001, Δ = ±2%)</li>
  <li><strong>Synthetic-only training retains 89.9%</strong> of original accuracy, confirming the model captures genuine rhythm-discriminative features</li>
  <li><strong>Zero privacy risk:</strong> No generated sample exceeds 0.80 correlation with any training example (mean: 0.30)</li>
  <li><strong>Clinical validity:</strong> Normal→AFib counterfactuals show increased RR variability; AFib→Normal show regularized rhythm</li>
</ol>
</div>

---

## Conclusion

We presented a diffusion-based counterfactual ECG generation pipeline that transforms single-lead ECG recordings into counterfactuals of the opposing class using a content-style disentangled architecture with partial-noise initialization and classifier-free guidance.

The goal of this work is not to improve classifier accuracy over original data, but to establish that **synthetic counterfactual data can safely substitute for or supplement real patient recordings** without degrading diagnostic performance. This is valuable in two scenarios:

<div class="highlight-box amber">
<ol>
  <li><strong>Privacy-preserving data sharing</strong> — When original data cannot be shared across institutions due to patient privacy and consent restrictions</li>
  <li><strong>Data augmentation</strong> — When labeled data is scarce and augmentation is needed to improve model training</li>
</ol>
</div>

Evaluation on the MIMIC-IV ECG database (149,793 segments) confirms this goal:
- **TOST equivalence testing** establishes that augmented training yields equivalent performance within ±2% (p < 0.001)
- **Non-inferiority testing** confirms the augmented model performs within 0.54 percentage points of the baseline
- **Uniqueness analysis** confirms zero near-copies, validating the privacy properties

---

## Publications

1. Tharaka Dilshan, Nethmini Karunarathne, Isuri Devindi, Mary M. Maleckar, Jørgen K. Kanters, Roshan Ragel, Isuru Nawinne, Vajira Thambawita. "Diffusion-Based Counterfactual ECG Generation for Atrial Fibrillation Data Augmentation" (2025). *Under Review.*

## Links

- [Project Repository](https://github.com/cepdnaclk/e20-fyp-ai-atrial-fib-detection)
- [Project Page](https://cepdnaclk.github.io/e20-fyp-ai-atrial-fib-detection)
- [🤗 Model Hub](https://huggingface.co/TharakaDil2001/diffusion-ecg-augmentation)
- [🤗 Live Demo](https://huggingface.co/spaces/TharakaDil2001/ecg-augmentation-demo)
- [Source Code (PERA_AF_Detection)](https://github.com/vlbthambawita/PERA_AF_Detection)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)
- [SEARCH Project (EU Horizon)](https://www.search-project.eu/)

---

*This work is part of the European project SEARCH, supported by the Innovative Health Initiative Joint Undertaking (IHI JU) under grant agreement No. 101172997.*
