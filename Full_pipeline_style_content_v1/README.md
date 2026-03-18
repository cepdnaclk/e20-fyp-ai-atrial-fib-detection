# Counterfactual ECG Generation for Clinical Teaching

## 🎯 Project Goal

Generate **counterfactual ECGs** for clinical teaching. Given an ECG classified as "Normal", we generate a minimally modified version that would be classified as "AFib" (and vice versa). These counterfactuals help clinicians understand:

> "What would this Normal patient's ECG look like if they had AFib?"

### Key Requirements:
1. **High Flip Rate**: The counterfactual should flip the classifier's prediction (Normal → AFib or AFib → Normal)
2. **High Correlation**: The counterfactual should be nearly identical to the original (>90% correlation)
3. **Minimal Edits**: Only the features relevant to AFib/Normal distinction should change
4. **Clinically Meaningful**: Changes should be interpretable (not adversarial noise)

---

## 📊 Current Status (Updated: January 26, 2026)

### ✅ V2 Model Results - Significant Improvement!

**V2 Model Training Completed Successfully!**

Training was run with proper 70/15/15 train/val/test stratified splits, early stopping, and dropout regularization.

#### Verified Results on Held-Out Test Set (200 samples):

| Metric | Normal→AFib | AFib→Normal | Overall |
|--------|-------------|-------------|---------|
| **Flip Rate** | 65.2% | 98.5% | **76.0%** |
| **Correlation** | 95.1% | 86.2% | **92.2%** |

#### Comparison: V1 vs V2

| Metric | V1 (Overfitted) | V2 (Proper Splits) | Improvement |
|--------|-----------------|-------------------|-------------|
| Test Flip Rate | 12% | **76%** | +64% ⬆️ |
| Test Correlation | N/A | **92.2%** | ✓ |
| Generalization |  Failed |  Works | ⭐ |

#### Key Findings:

1. **AFib→Normal flips are easier** (98.5%) than Normal→AFib (65.2%)
   - This makes clinical sense: removing irregular features is simpler than adding them

2. **High correlation maintained** (92.2% overall)
   - Counterfactuals are visually similar to originals

3. **V2 generalizes!** 
   - Unlike V1 which only worked on training data, V2 works on unseen test data

### Model Files:
- Best model: `./minimal_edit_v2/best_model.pth` (saved at epoch 4)
- Test results: `./minimal_edit_v2/test_results.json`
- Overlays: `./v2_counterfactual_overlays_*/`

---

### Previous Issue: V1 Model Overfitting (RESOLVED)

**V1 Model Analysis Results:**

| Metric | Training Set | Test Set (Unseen) |
|--------|-------------|-------------------|
| Flip Rate | 99.9% | **12%** ← Severe drop |
| Correlation | 99.27% | N/A |
| Edit Size | Small | 62% of signal (too large!) |

**Root cause**: No train/val/test split → model overfitted to memorize specific samples.
**Solution**: V2 with proper data splits ✅

---

## 🔬 Technical Concepts Explained

### 1. What is a Counterfactual?

A **counterfactual** answers "what if?" questions. In our case:
- Original ECG: Normal rhythm
- Counterfactual: "What would this ECG look like if the patient had AFib?"

The key insight is that the counterfactual should be **minimally different** from the original - only the features that distinguish AFib from Normal should change.

### 2. Why Not Just Generate New ECGs?

We tried a **diffusion model** that generates entirely new ECGs. Problem:
- Flip rate: 100% ✓
- Correlation with original: **0.34** ✗

The generated ECGs looked nothing like the originals - useless for teaching "what changed."

### 3. Residual Edit Approach

Instead of generating a new ECG, we predict the **edit (residual)** to apply:

```
counterfactual = original + strength × edit_network(original, target_class)
```

Where:
- `edit_network`: Neural network that predicts what to change
- `target_class`: The class we want to flip to (AFib or Normal)
- `strength`: Learnable parameter controlling edit magnitude (0-1)

### 4. Loss Functions

The model is trained with multiple objectives:

```python
L_total = λ_flip × L_flip + λ_similarity × L_similarity + λ_sparsity × L_sparsity + λ_identity × L_identity
```

| Loss | Purpose | Formula |
|------|---------|---------|
| `L_flip` | Counterfactual should be classified as target class | `CrossEntropy(classifier(cf), target)` |
| `L_similarity` | Counterfactual should be close to original | `MSE(cf, original)` |
| `L_sparsity` | Edits should be sparse/focused | `mean(\|edit\|)` |
| `L_identity` | When target=original class, output should equal input | `MSE(cf, original)` when same class |

### 5. The Overfitting Problem

**Training on all data without validation** leads to:
- Model learns "for sample #1234, add this specific pattern"
- This works perfectly on training samples (99.9% flip)
- But fails on new samples (12% flip) because patterns are sample-specific

**Solution**: Proper train/val/test splits + early stopping

### 6. Adversarial vs Meaningful Edits

- **Adversarial edits**: Imperceptible noise that fools the classifier (SNR > 40dB)
- **Meaningful edits**: Visible changes to clinically relevant features (SNR < 25dB)

Our analysis shows the V1 model makes **visible edits** (SNR ~11dB), but they're not targeting the right features - they're random noise that happened to work on training samples.

---

## 📁 Project Structure

```
Full_pipeline_style_content/
├── 📂 Data & Models
│   ├── ecg_afib_data/                  # ECG dataset (93,066 samples)
│   │   ├── X_combined.npy              # ECG signals (93066, 2500)
│   │   └── y_combined.npy              # Labels ('A' or 'N')
│   ├── best_model/                     # Pre-trained classifier
│   │   └── best_model.pth              # AFibResLSTM weights (~97% accuracy)
│   ├── minimal_edit_training/          # V1 model (OVERFITTED - don't use)
│   │   └── best_model.pth
│   └── minimal_edit_v2/                # V2 model (with proper splits)
│       ├── best_model.pth              # Best model by validation
│       ├── config.json                 # Training configuration
│       ├── test_results.json           # Test set evaluation
│       ├── split_indices.json          # Train/val/test indices
│       └── training_history.png        # Training curves
│
├── 📂 Training Scripts
│   ├── train_minimal_edit_model.py     # V1: NO TRAIN/VAL SPLIT (deprecated)
│   ├── train_minimal_edit_v2.py        # V2: With proper splits ⭐ CURRENT
│   └── train_gradient_guided_counterfactual.py  # V3: Gradient-guided (FUTURE)
│
├── 📂 Analysis & Visualization
│   ├── analyze_counterfactual_changes.py   # Diagnose edit quality
│   ├── generate_v2_overlays.py             # Generate V2 counterfactual overlays ⭐ MAIN
│   ├── generate_overlays_with_minimal_edit.py  # Generate overlay images (V1)
│   └── generate_successful_overlays.py     # Generate best overlays
│
├── 📂 Output Directories
│   ├── v2_counterfactual_overlays_*/       # V2 overlay visualizations ⭐ LATEST
│   ├── counterfactual_analysis_*/          # Diagnostic analysis results
│   ├── counterfactual_overlays_*/          # V1 overlay images
│   └── successful_counterfactuals_*/       # High-quality overlays
│
└── 📄 Documentation
    ├── README.md                       # This file
    └── PROJECT_STATUS_DOCUMENTATION.md # Legacy docs
```

---

## 🚀 How to Resume This Work

### Quick Start: Generate Counterfactuals

If you just want to generate counterfactuals using the trained V2 model:

```bash
cd /scratch1/e20-fyp-ai-atrial-fib-det/old_vision/PERA_AF_Detection/Pipeline_Implementation/Full_pipeline_style_content
conda activate atfib

# Generate counterfactual overlays
python generate_v2_overlays.py
```

This will:
1. Load the trained V2 model from `./minimal_edit_v2/best_model.pth`
2. Verify performance on test samples
3. Generate overlay visualizations in `./v2_counterfactual_overlays_*/`

### Step-by-Step Guide

#### Step 1: Check Current Status

```bash
cd /scratch1/e20-fyp-ai-atrial-fib-det/old_vision/PERA_AF_Detection/Pipeline_Implementation/Full_pipeline_style_content
conda activate atfib

# Check V2 model exists
ls -la ./minimal_edit_v2/

# View test results
cat ./minimal_edit_v2/test_results.json

# View latest overlays
ls -la ./v2_counterfactual_overlays_*/
```

#### Step 2: Re-train V2 (if needed)

Only do this if you want to experiment with different hyperparameters:

```bash
python train_minimal_edit_v2.py
```

This takes ~1-2 hours on RTX 6000.

#### Step 3: View Results

Open overlay images in `./v2_counterfactual_overlays_*/` to see:
- Original ECG (blue)
- Counterfactual ECG (red)  
- Overlay with highlighted differences

### Interpretation Guide

- **Flip Rate**: Percentage of counterfactuals that successfully flip the classifier
  - Target: >60% is good, >80% is excellent
  
- **Correlation**: Similarity between original and counterfactual
  - Target: >85% means minimal edits, >90% is excellent
  
- **Normal→AFib**: Adding AFib features (harder)
- **AFib→Normal**: Removing AFib features (easier)

---

## 🔧 Training Configuration

### V2 Training Script (`train_minimal_edit_v2.py`)

```python
# Key hyperparameters
num_epochs = 100          # Maximum epochs
batch_size = 32           # Batch size
lr = 1e-4                 # Learning rate
weight_decay = 1e-4       # L2 regularization
lambda_identity = 1.0     # Weight for identity loss
lambda_flip = 1.0         # Weight for flip loss  
lambda_similarity = 2.0   # Weight for similarity loss (higher = preserve more)
lambda_sparsity = 0.1     # Weight for sparse edits
patience = 15             # Early stopping patience
dropout = 0.1             # Dropout rate

# Data split
train_ratio = 0.70        # 70% for training
val_ratio = 0.15          # 15% for validation
test_ratio = 0.15         # 15% for final test
```

### Model Architecture: ResidualEditModelV2

```
Input: ECG (1, 2500) + Target Class (0 or 1)
   ↓
Encoder: 4-layer CNN with BatchNorm + Dropout
   ↓
Class Embedding: Learned 256-dim embedding per class
   ↓
Spatial Attention: Focus edits on important regions
   ↓
Decoder: Transposed convolutions → Edit residual
   ↓
Output: original + sigmoid(edit_strength) × edit
```

---

## 📈 Results Log

### V1 Model (train_minimal_edit_model.py) - DEPRECATED ❌

- **Training completed**: Yes
- **Training metrics**: 99.9% flip, 99.27% correlation
- **Test metrics**: ~12% flip (FAILED - overfit)
- **Status**: ❌ OVERFITTED - Do not use
- **Model path**: `./minimal_edit_training/best_model.pth`

### V2 Model (train_minimal_edit_v2.py) - COMPLETED ✅

- **Training started**: January 26, 2026 @ 03:44
- **Training completed**: January 26, 2026 @ 04:49 (early stopped at epoch 19, best at epoch 4)
- **Test metrics (verified on 200 samples)**:
  - **Overall Flip Rate**: 76.0%
  - **Overall Correlation**: 92.2%
  - Normal→AFib: 65.2% flip, 95.1% correlation
  - AFib→Normal: 98.5% flip, 86.2% correlation
- **Status**: ✅ SUCCESSFUL - Use this model
- **Model path**: `./minimal_edit_v2/best_model.pth`
- **Overlays**: `./v2_counterfactual_overlays_*/`

**To generate more counterfactuals:**
```bash
python generate_v2_overlays.py
```

### V3 Gradient-Guided (OPTIONAL FUTURE WORK) ⏳

- **Status**: Not started (may not be needed since V2 works)
- **When to use**: If you want to improve Normal→AFib flip rate beyond 65%
- **Script**: `train_gradient_guided_counterfactual.py` (to be created)

---

## 🔮 Future Work: Gradient-Guided Approach

**Use this approach if V2 still fails to generalize.**

### The Core Problem

The current model can freely edit ANY part of the ECG. It learns sample-specific edits rather than class-specific edits. We need to **constrain WHERE the model can edit**.

### Concept: Saliency-Guided Editing

The classifier "knows" what makes an ECG AFib vs Normal - its gradients reveal this!

```python
# For a given ECG x with label y:
saliency = |∂L(classifier(x), y) / ∂x|

# High saliency regions = important for classification
# We should focus edits ONLY on these regions
```

### How It Works

1. **Compute saliency map** for each ECG using the frozen classifier
2. **Create attention mask** from saliency (top 10-20% most important regions)
3. **Mask the edit**: `edit_masked = edit × attention_mask`
4. **Apply masked edit**: `counterfactual = original + edit_masked`

### Why This Should Work

- Edits are **forced** to target classifier-relevant features
- No more random noise that works on training samples only
- Edits will be interpretable (focused on P-waves, RR intervals, etc.)

### Implementation Outline

```python
def compute_saliency(classifier, ecg, label):
    """Compute gradient-based saliency map"""
    ecg.requires_grad = True
    logits, _ = classifier(ecg)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    saliency = torch.abs(ecg.grad)
    return saliency

def create_attention_mask(saliency, top_percent=0.2):
    """Keep top 20% most salient regions"""
    threshold = torch.quantile(saliency, 1 - top_percent)
    mask = (saliency >= threshold).float()
    # Smooth the mask to avoid sharp edges
    mask = F.avg_pool1d(mask.unsqueeze(1), kernel_size=21, stride=1, padding=10).squeeze(1)
    return mask

class GradientGuidedEditModel(nn.Module):
    def forward(self, x, target_class, saliency_mask):
        # Generate raw edit
        edit = self.edit_network(x, target_class)
        # Mask to saliency regions only
        edit_masked = edit * saliency_mask
        # Apply
        counterfactual = x + self.strength * edit_masked
        return counterfactual, edit_masked
```

### Expected Benefits

| Aspect | Current Approach | Gradient-Guided |
|--------|-----------------|-----------------|
| Edit location | Anywhere | Only classifier-relevant regions |
| Generalization | Poor (overfits) | Better (constrained) |
| Interpretability | Low | High (edits match saliency) |

---

## 🏗️ Architecture Details

### Classifier: AFibResLSTM

The frozen classifier used to determine if counterfactuals flip:

```
Multi-Scale Conv1D (3 parallel branches with kernels 5, 11, 21)
    ↓
Feature Fusion (concatenate + 1x1 conv)
    ↓
ResNet-34 Backbone (1D convolutions)
    ↓
Bidirectional LSTM (128 hidden units)
    ↓
Self-Attention Layer
    ↓
Fully Connected → 2 classes (AFib, Normal)
```

**Loading the classifier:**
```python
from counterfactual_training import AFibResLSTM, ModelConfig
config = ModelConfig()
classifier = AFibResLSTM(config).to(device)
ckpt = torch.load('./best_model/best_model.pth', map_location=device)
classifier.load_state_dict(ckpt['model_state_dict'])
classifier.eval()
```

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total samples | 93,066 |
| AFib samples | 35,356 (38%) |
| Normal samples | 57,710 (62%) |
| Sampling rate | 250 Hz |
| Duration | 10 seconds |
| Samples per ECG | 2,500 |

---

## ⚠️ Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Symptom**: `torch.OutOfMemoryError: CUDA out of memory`

**Solution**: 
```bash
# Check what's using GPU
nvidia-smi

# Kill other processes or use smaller batch size
# In training script, change batch_size=32 to batch_size=16
```

### Issue 2: Model Loading Errors

**Symptom**: `TypeError: AFibResLSTM.__init__() got unexpected keyword argument`

**Solution**: 
```python
# Always use ModelConfig:
from counterfactual_training import AFibResLSTM, ModelConfig
config = ModelConfig()
classifier = AFibResLSTM(config).to(device)
```

### Issue 3: Low Test Flip Rate

**Symptom**: Training flip rate high, test flip rate low

**Diagnosis**: Model is overfitting

**Solutions**:
1. Use V2 training with proper splits ✓
2. Increase `lambda_similarity` to constrain edits more
3. Add more dropout
4. If still failing → use gradient-guided approach

### Issue 4: Counterfactuals Look Identical to Originals

**Symptom**: Can't see any difference between original and counterfactual

**Possible causes**:
- Edit strength too low → Check `sigmoid(model.edit_strength)` value
- Edits are adversarial (imperceptible) → Run `analyze_counterfactual_changes.py`

---

## 📝 Change Log

### January 26, 2026

1. **Identified overfitting problem** in V1 model
   - Training: 99.9% flip, 99.27% correlation
   - Test: 12% flip (model memorized training samples)

2. **Created V2 training script** (`train_minimal_edit_v2.py`)
   - Added 70/15/15 train/val/test splits
   - Added early stopping with patience=15
   - Added dropout and batch normalization
   - Added spatial attention for focused edits

3. **Created analysis script** (`analyze_counterfactual_changes.py`)
   - Measures SNR (signal-to-noise ratio of edits)
   - Determines if edits are adversarial or meaningful
   - Generates detailed visualization

4. **Documented future work** (gradient-guided approach)
   - Explained saliency-based editing concept
   - Provided implementation outline

5. **Updated documentation** (this README)
   - Explained all concepts (counterfactuals, losses, overfitting)
   - Added resumption guide
   - Added troubleshooting section

---

## 📚 References

1. Wachter et al. "Counterfactual Explanations without Opening the Black Box" (2017)
2. Mothilal et al. "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations" (DiCE, 2020)
3. Jia et al. "ResNet-LSTM for ECG Classification" (2020)
4. PhysioNet ECG Databases

---

## 🆘 Quick Reference for Resuming

```bash
# 1. Navigate to project
cd /scratch1/e20-fyp-ai-atrial-fib-det/old_vision/PERA_AF_Detection/Pipeline_Implementation/Full_pipeline_style_content
conda activate atfib

# 2. Check V2 status
cat ./minimal_edit_v2/test_results.json 2>/dev/null || echo "V2 not trained yet"

# 3. If not trained, run:
python train_minimal_edit_v2.py

# 4. If V2 failed (flip < 50%), implement gradient-guided approach

# 5. Generate visualizations:
python generate_successful_overlays.py
```

---

*Last updated: January 26, 2026 - Added overfitting diagnosis and V2 training approach*
