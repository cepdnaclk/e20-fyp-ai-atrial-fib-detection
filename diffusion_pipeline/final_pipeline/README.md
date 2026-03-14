# Diffusion-Based Counterfactual ECG Generation for AFib Detection

A pipeline that generates counterfactual ECG signals to augment training data for Atrial Fibrillation detection. Given a Normal ECG, the model generates a realistic AFib version (and vice versa) while preserving the original morphology.

## Project Structure

```
final_pipeline/
├── src/                          # Source code (ordered by pipeline stage)
│   ├── data_preparation.py       # Stage 1: Raw MIMIC-IV → normalized 10s segments
│   ├── classifier/
│   │   ├── model_architecture.py # ResNet-BiLSTM classifier architecture
│   │   └── train_classifier.py   # Classifier training script  
│   ├── diffusion/
│   │   ├── train_diffusion.py    # Stage 3: Diffusion model training (both stages)
│   │   └── diffusion_models.py   # Shared diffusion model components
│   ├── generation/
│   │   ├── generate_counterfactuals.py   # Stage 4: Generate counterfactuals
│   │   └── plausibility_validator.py     # Clinical plausibility filtering
│   ├── evaluation/
│   │   ├── three_way_evaluation.py       # Stage 5: 3-condition 5-fold evaluation
│   │   ├── comprehensive_metrics.py      # Signal quality metrics
│   │   └── statistical_analysis.py       # TOST equivalence testing
│   └── utils/
│       └── shared_models.py      # Shared model definitions
├── data/                         # Dataset (symlinks to processed data)
├── models/                       # Trained weights
│   ├── classifier/               # ResNet-BiLSTM weights
│   └── diffusion/                # Diffusion model weights + checkpoints
└── results/
    ├── counterfactuals/          # Generated counterfactual ECGs
    ├── metrics/                  # Evaluation result JSONs
    ├── figures/
    │   ├── dataset_overview/     # Data distribution, train/val/test split, ECG samples
    │   ├── training/             # Reconstruction + counterfactual plots per epoch
    │   ├── counterfactual_examples/  # 40 source-vs-counterfactual comparison plots
    │   ├── evaluation/           # 5-fold confusion matrices, ROC, accuracy comparison
    │   └── metrics/              # Signal quality distributions, flip rate, clinical features
    └── logs/                     # Training and generation logs
```

## Pipeline Execution Order

Run these scripts in sequence. Each stage depends on the output of the previous one.

### Stage 1: Data Preparation
```bash
python src/data_preparation.py
```
Loads raw MIMIC-IV ECG recordings, extracts Lead II, applies bandpass filtering and NeuroKit2 cleaning, performs global normalization to [-1.5, 1.5], and splits into train/val/test sets.

**Input:** Raw MIMIC-IV waveform files  
**Output:** `data/train_data.npz`, `data/val_data.npz`, `data/test_data.npz`

### Stage 2: Classifier Training
```bash
python src/classifier/train_classifier.py
```
Trains a ResNet-BiLSTM classifier on the original ECG data. This classifier serves two roles: (1) it provides the flip loss signal during diffusion Stage 2 training, and (2) it is the architecture used in the three-way evaluation.

**Input:** `data/train_data.npz`  
**Output:** `models/classifier/afib_reslstm_final.pth`

### Stage 3: Diffusion Model Training
```bash
python src/diffusion/train_diffusion.py
```
Trains the diffusion model in two stages:
- **Stage 1 (Epochs 1–50):** Reconstruction training. Losses: noise prediction MSE, style classification cross-entropy, KL divergence.
- **Stage 2 (Epochs 51–100):** Counterfactual fine-tuning. Adds flip loss and similarity loss to monitor counterfactual quality.

**Input:** `data/train_data.npz`, `models/classifier/afib_reslstm_final.pth`  
**Output:** `models/diffusion/final_model.pth`, checkpoints every 10 epochs

### Stage 4: Counterfactual Generation
```bash
python src/generation/generate_counterfactuals.py
```
Generates counterfactuals for each training ECG using SDEdit (60% noise strength, 50 DDIM steps, CFG scale 3.0). Each output is validated by `plausibility_validator.py` using morphological, physiological, and clinical criteria. Signals scoring below 0.7 are rejected (with up to 3 retries).

**Input:** `data/train_data.npz`, `models/diffusion/final_model.pth`  
**Output:** `results/counterfactuals/counterfactual_full_data.npz`

### Stage 5: Evaluation
```bash
# Three-condition 5-fold cross-validation
python src/evaluation/three_way_evaluation.py

# Signal quality metrics
python src/evaluation/comprehensive_metrics.py

# Statistical equivalence testing (TOST)
python src/evaluation/statistical_analysis.py
```

Evaluates the counterfactuals across three training conditions:
- **Condition A:** Train classifier on original data only
- **Condition B:** Train classifier on counterfactuals only
- **Condition C:** Train classifier on 67% originals + 33% counterfactuals

**Output:** `results/metrics/*.json`, `results/figures/evaluation/*.png`

## Key Results

| Condition | Accuracy | F1 | AUROC |
|-----------|----------|-----|-------|
| A (Original only) | 95.63 ± 0.33% | 95.65 ± 0.40% | 98.90 ± 0.20% |
| B (CF only) | 85.94 ± 1.32% | 86.70 ± 1.20% | 93.24 ± 1.50% |
| C (Augmented) | 95.05 ± 0.50% | 95.09 ± 0.50% | 98.60 ± 0.20% |

Statistical equivalence between Condition A and C confirmed by TOST (p = 0.007, margin = ±2%).

## Model Architecture

- **Content Encoder:** VAE with BatchNorm, produces 256-d class-invariant representation (morphology)
- **Style Encoder:** CNN with InstanceNorm + classifier head, produces 128-d class-discriminative representation (rhythm)
- **Conditional UNet:** Denoising network with FiLM conditioning on content, style, timestep, and class label
- **DDIM Scheduler:** Deterministic sampler with SDEdit support for partial denoising

## Dataset

- **Source:** MIMIC-IV ECG Diagnostic Database
- **Signals:** ~150,000 Lead II segments (10 seconds at 250 Hz = 2,500 samples each)
- **Classes:** Normal sinus rhythm vs Atrial Fibrillation (balanced)
- **Normalization:** Global min-max scaling to [-1.5, 1.5]

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, pandas, matplotlib, seaborn
- neurokit2, wfdb
- tqdm
- CUDA-capable GPU (trained on NVIDIA RTX 6000 Ada, 48 GB VRAM)
