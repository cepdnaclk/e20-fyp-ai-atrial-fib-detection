# Phase 3: Counterfactual ECG Generation for Explainable AI

## Overview

This phase implements a Content-Style Disentangled Diffusion Model for generating counterfactual ECGs. The goal is to flip the classifier prediction (Normal ↔ AFib) while preserving realistic ECG morphology.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ECG Signal Input                         │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    Content Encoder      │     │     Style Encoder       │
│  (Class-Invariant)      │     │  (Class-Discriminative) │
│                         │     │                         │
│  Captures:              │     │  Captures:              │
│  - Heart rate           │     │  - P-wave presence      │
│  - Basic rhythm         │     │  - RR regularity        │
│  - Signal morphology    │     │  - Fibrillatory waves   │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │   Conditional Diffusion UNet  │
              │   (Conditioned on Content +   │
              │    Style + Timestep)          │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Generated ECG Output      │
              └───────────────────────────────┘
```

## Key Concepts

### 1. Content Encoder
- Extracts features that are **NOT** related to the Normal/AFib classification
- Should capture: heart rate, QRS morphology, signal amplitude
- Uses VAE-style encoding with reparameterization trick

### 2. Style Encoder  
- Extracts features that **ARE** related to the Normal/AFib classification
- Should capture: P-wave presence/absence, RR interval regularity, fibrillatory waves
- Trained with classification loss to ensure style is class-discriminative

### 3. Counterfactual Generation
- **Reconstruction**: Content(X) + Style(X) → X (should match original)
- **Counterfactual**: Content(X_normal) + Style(X_afib) → Counterfactual that looks like AFib

## Files

| File | Description |
|------|-------------|
| `01_train_content_style_diffusion.py` | Main training script for the disentangled model |
| `02_evaluate_counterfactuals.py` | Evaluation and validation of counterfactuals |
| `run_training.sh` | Bash script to start training in tmux |

## Usage

### Step 1: Train the Model

```bash
cd /scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-gpu
python 01_train_content_style_diffusion.py
```

Or use tmux for persistent training:
```bash
./run_training.sh
```

### Step 2: Evaluate Counterfactuals

After training completes:
```bash
python 02_evaluate_counterfactuals.py
```

## Expected Outcomes

### 1. Reconstruction Test
- Content(X) + Style(X) should reconstruct X with high fidelity
- Target: MSE < 0.01, Correlation > 0.95

### 2. Class Flip Test
- Content(Normal) + Style(AFib) → Classifier predicts AFib
- Content(AFib) + Style(Normal) → Classifier predicts Normal
- Target: Flip rate > 80%

### 3. Clinical Validity
- Counterfactuals should show clinically meaningful changes:
  - Normal → AFib: Missing P-waves, irregular RR intervals
  - AFib → Normal: Regular P-waves, consistent RR intervals

## Model Outputs

```
models/phase3_counterfactual/
├── checkpoint_epoch_*.pth    # Training checkpoints
├── final_model.pth           # Final trained model
├── ecg_classifier.pth        # ECG classifier for validation
└── results/
    ├── epoch_*_reconstruction.png
    ├── epoch_*_counterfactual.png
    └── counterfactual_results/
        ├── counterfactual_overlay.png
        └── counterfactual_difference.png
```

## Validation Metrics

1. **Flip Rate**: Percentage of counterfactuals that flip the classifier prediction
2. **Reconstruction Error**: MSE between original and reconstructed ECG
3. **RR Irregularity Change**: Measures change in RR interval variability
4. **Visual Inspection**: Overlay of original and counterfactual for clinician review

## References

- Phase 2 (Unconditional Diffusion): `../phase_2_diffusion/`
- Pre-trained diffusion model: `models/phase2_diffusion/diffusion_v2/best.pth`
