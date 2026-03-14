"""
Three-Way Classifier Evaluation
================================

Compare classifier performance across three data conditions:
A. Original data only (baseline)
B. Counterfactual data only (generalization test)
C. Mixed data (50% original + 50% counterfactual) (augmentation test)

Evaluation:
- Train AFibResLSTM classifier on each condition
- Test on original test set
- Compare accuracy, precision, recall, F1, AUROC
- Statistical significance testing
- Training time comparison

Author: Phase 3 Counterfactual Generation
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import model architecture
sys.path.insert(0, str(Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')))
from model_architecture import AFibResLSTM, ModelConfig

# ============================================================================
# Configuration
# ============================================================================

class Config:
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    CF_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/generated_counterfactuals'
    OUTPUT_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/three_way_evaluation'
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 40
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    EARLY_STOPPING_PATIENCE = 10
    
    # Focal Loss params (from original model)
    FOCAL_ALPHA = 0.65
    FOCAL_GAMMA = 2.0

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
print(f"Device: {Config.DEVICE}")

# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=0.65, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma  * bce_loss
        return focal_loss.mean()

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "="*60)
print("Loading Data")
print("="*60)

# Original data
train_data = np.load(Config.DATA_DIR / 'train_data.npz')
original_train_X = torch.tensor(train_data['X'], dtype=torch.float32)
original_train_y = torch.tensor(train_data['y'], dtype=torch.long)

val_data = np.load(Config.DATA_DIR / 'val_data.npz')
original_val_X = torch.tensor(val_data['X'], dtype=torch.float32)
original_val_y = torch.tensor(val_data['y'], dtype=torch.long)

test_data = np.load(Config.DATA_DIR / 'test_data.npz')
test_X = torch.tensor(test_data['X'], dtype=torch.float32)
test_y = torch.tensor(test_data['y'], dtype=torch.long)

# Counterfactual data — load from assembled full data (has richer keys)
CF_FULL_PATH = Config.PROJECT_ROOT / 'notebooks/phase_3_counterfactual/results/counterfactual_full_data.npz'
CF_SIMPLE_PATH = Config.CF_DIR / 'counterfactual_test_data.npz'

if CF_FULL_PATH.exists():
    print(f"Loading CF data from: {CF_FULL_PATH}")
    cf_data = np.load(CF_FULL_PATH)
    cf_X = torch.tensor(cf_data['counterfactuals'], dtype=torch.float32)
    cf_y = torch.tensor(cf_data['target_labels'], dtype=torch.long)
elif CF_SIMPLE_PATH.exists():
    print(f"Loading CF data from: {CF_SIMPLE_PATH}")
    cf_data = np.load(CF_SIMPLE_PATH)
    cf_X = torch.tensor(cf_data['X'], dtype=torch.float32)
    cf_y = torch.tensor(cf_data['y'], dtype=torch.long)
else:
    raise FileNotFoundError("No counterfactual data found!")

# Ensure proper shape (N, 1, 2500)
if original_train_X.dim() == 2:
    original_train_X = original_train_X.unsqueeze(1)
if original_val_X.dim() == 2:
    original_val_X = original_val_X.unsqueeze(1)
if test_X.dim() == 2:
    test_X = test_X.unsqueeze(1)
if cf_X.dim() == 2:
    cf_X = cf_X.unsqueeze(1)

print(f"Original train: {len(original_train_X)} (Normal: {(original_train_y==0).sum()}, AFib: {(original_train_y==1).sum()})")
print(f"Original val:   {len(original_val_X)} (Normal: {(original_val_y==0).sum()}, AFib: {(original_val_y==1).sum()})")
print(f"Test set:       {len(test_X)} (Normal: {(test_y==0).sum()}, AFib: {(test_y==1).sum()})")
print(f"Counterfactual: {len(cf_X)} (Normal: {(cf_y==0).sum()}, AFib: {(cf_y==1).sum()})")

# ============================================================================
# Prepare Three Datasets
# ============================================================================

print("\n" + "="*60)
print("Preparing Three-Way Datasets")
print("="*60)

train_size = len(original_train_X)  # 104,855
val_size = len(original_val_X)      # 22,469

# --- Condition A: Original only (baseline) ---
train_A_X = original_train_X
train_A_y = original_train_y
val_A_X = original_val_X
val_A_y = original_val_y

print(f"Condition A (Original): Train={len(train_A_X)}, Val={len(val_A_X)}")

# --- Condition B: Counterfactual only ---
# We have 22K CFs. Split 80/20 → train base ~17.9K, val base ~4.5K
# Then duplicate with slight noise to match original sizes (104K train, 22K val)
perm = torch.randperm(len(cf_X))
cf_train_count = int(0.8 * len(cf_X))

train_B_base_X = cf_X[perm[:cf_train_count]]
train_B_base_y = cf_y[perm[:cf_train_count]]
val_B_base_X = cf_X[perm[cf_train_count:]]
val_B_base_y = cf_y[perm[cf_train_count:]]

print(f"  CF base split: train={len(train_B_base_X)}, val={len(val_B_base_X)}")

# Duplicate train CFs to match original train size
def duplicate_with_noise(X, y, target_size, noise_std=0.01):
    """Duplicate data with small Gaussian noise to reach target_size."""
    if len(X) >= target_size:
        return X[:target_size], y[:target_size]
    
    result_X = [X]
    result_y = [y]
    remaining = target_size - len(X)
    
    while remaining > 0:
        batch = min(remaining, len(X))
        indices = np.random.choice(len(X), batch, replace=True)
        noisy_X = X[indices] + torch.randn(batch, *X.shape[1:]) * noise_std
        result_X.append(noisy_X)
        result_y.append(y[indices])
        remaining -= batch
    
    return torch.cat(result_X, dim=0), torch.cat(result_y, dim=0)

train_B_X, train_B_y = duplicate_with_noise(train_B_base_X, train_B_base_y, train_size)
val_B_X, val_B_y = duplicate_with_noise(val_B_base_X, val_B_base_y, val_size)

print(f"Condition B (Counterfactual): Train={len(train_B_X)} (from {len(train_B_base_X)} CFs), "
      f"Val={len(val_B_X)} (from {len(val_B_base_X)} CFs)")
print(f"  Train class balance: Normal={int((train_B_y==0).sum())}, AFib={int((train_B_y==1).sum())}")
print(f"  Val class balance:   Normal={int((val_B_y==0).sum())}, AFib={int((val_B_y==1).sum())}")

# --- Condition C: Mixed (50% original + 50% counterfactual) ---
half_train = train_size // 2
half_val = val_size // 2

# Get CF portions large enough
cf_train_for_mix, cf_train_y_mix = duplicate_with_noise(train_B_base_X, train_B_base_y, half_train)
cf_val_for_mix, cf_val_y_mix = duplicate_with_noise(val_B_base_X, val_B_base_y, half_val)

train_C_X = torch.cat([original_train_X[:half_train], cf_train_for_mix[:half_train]], dim=0)
train_C_y = torch.cat([original_train_y[:half_train], cf_train_y_mix[:half_train]], dim=0)
val_C_X = torch.cat([original_val_X[:half_val], cf_val_for_mix[:half_val]], dim=0)
val_C_y = torch.cat([original_val_y[:half_val], cf_val_y_mix[:half_val]], dim=0)

# Shuffle mixed
perm_train = torch.randperm(len(train_C_X))
train_C_X = train_C_X[perm_train]
train_C_y = train_C_y[perm_train]

perm_val = torch.randperm(len(val_C_X))
val_C_X = val_C_X[perm_val]
val_C_y = val_C_y[perm_val]

print(f"Condition C (Mixed 50/50): Train={len(train_C_X)}, Val={len(val_C_X)}")
print(f"  Train class balance: Normal={int((train_C_y==0).sum())}, AFib={int((train_C_y==1).sum())}")
print(f"  Val class balance:   Normal={int((val_C_y==0).sum())}, AFib={int((val_C_y==1).sum())}")

# ============================================================================
# Training Function
# ============================================================================

def train_classifier(train_X, train_y, val_X, val_y, condition_name):
    """Train AFibResLSTM classifier and return results."""
    print(f"\n{'='*60}")
    print(f"Training Classifier - Condition: {condition_name}")
    print(f"{'='*60}")
    
    # Create dataloaders
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model_config = ModelConfig()
    model = AFibResLSTM(model_config).to(Config.DEVICE)
    
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, 
                                 weight_decay=Config.WEIGHT_DECAY)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            
            # Normalize
            mean = X_batch.mean(dim=2, keepdim=True)
            std = X_batch.std(dim=2, keepdim=True) + 1e-8
            X_batch = (X_batch - mean) / std
            
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
                
                mean = X_batch.mean(dim=2, keepdim=True)
                std = X_batch.std(dim=2, keepdim=True) + 1e-8
                X_batch = (X_batch - mean) / std
                
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, Config.OUTPUT_DIR / f'best_model_{condition_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    training_time = time.time() - start_time
    
    # Load best model
    checkpoint = torch.load(Config.OUTPUT_DIR / f'best_model_{condition_name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test on original test set
    model.eval()
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    test_preds = []
    test_probs = []
    test_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(Config.DEVICE)
            
            mean = X_batch.mean(dim=2, keepdim=True)
            std = X_batch.std(dim=2, keepdim=True) + 1e-8
            X_batch = (X_batch - mean) / std
            
            logits, _ = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())  # Probability of AFib
            test_labels.extend(y_batch.numpy())
    
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall = recall_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    auroc = roc_auc_score(test_labels, test_probs)
    cm = confusion_matrix(test_labels, test_preds)
    
    results = {
        'condition': condition_name,
        'training_time_seconds': training_time,
        'training_time_hours': training_time / 3600,
        'best_epoch': best_epoch,
        'total_epochs_run': epoch,
        'test_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(recall),
            'specificity': float(cm[0, 0] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
            'f1_score': float(f1),
            'auroc': float(auroc),
            'confusion_matrix': {
                'TN': int(cm[0, 0]),
                'FP': int(cm[0, 1]),
                'FN': int(cm[1, 0]),
                'TP': int(cm[1, 1]),
            }
        },
        'test_probs': test_probs.tolist(),
        'test_preds': test_preds.tolist(),
        'test_labels': test_labels.tolist(),
        'history': history,
    }
    
    print(f"\n{condition_name} Results:")
    print(f"  Training Time: {training_time/3600:.2f} hours")
    print(f"  Best Epoch: {best_epoch}/{epoch}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test Precision: {precision:.4f}")
    print(f"  Test Recall: {recall:.4f}")
    print(f"  Test F1: {f1:.4f}")
    print(f"  Test AUROC: {auroc:.4f}")
    print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    return results, model

# ============================================================================
# Train All Three Conditions
# ============================================================================

results_all = {}

# Condition A: Original
results_A, model_A = train_classifier(train_A_X, train_A_y, val_A_X, val_A_y, "original")
results_all['original'] = results_A

# Condition B: Counterfactual
results_B, model_B = train_classifier(train_B_X, train_B_y, val_B_X, val_B_y, "counterfactual")
results_all['counterfactual'] = results_B

# Condition C: Mixed
results_C, model_C = train_classifier(train_C_X, train_C_y, val_C_X, val_C_y, "mixed")
results_all['mixed'] = results_C

# ============================================================================
# Statistical Comparison
# ============================================================================

print("\n" + "="*60)
print("Statistical Comparison")
print("="*60)

# Compare accuracies(using bootstrap for confidence intervals)
def bootstrap_ci(data, n_iterations=1000, ci=95):
    """Compute bootstrap confidence interval."""
    means = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper

# Extract predictions
preds_A = np.array(results_A['test_preds'])
preds_B = np.array(results_B['test_preds'])
preds_C = np.array(results_C['test_preds'])
labels = np.array(results_A['test_labels'])

# Per-sample accuracy
acc_A = (preds_A == labels).astype(float)
acc_B = (preds_B == labels).astype(float)
acc_C = (preds_C == labels).astype(float)

# Statistical tests
t_ab, p_ab = stats.ttest_rel(acc_A, acc_B)
t_ac, p_ac = stats.ttest_rel(acc_A, acc_C)
t_bc, p_bc = stats.ttest_rel(acc_B, acc_C)

# Effect sizes (Cohen's d)
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x)**2 + np.std(y)**2) / 2)

d_ab = cohens_d(acc_A, acc_B)
d_ac = cohens_d(acc_A, acc_C)
d_bc = cohens_d(acc_B, acc_C)

statistical_comparison = {
    'original_vs_counterfactual': {
        't_statistic': float(t_ab),
        'p_value': float(p_ab),
        'cohens_d': float(d_ab),
        'significant': p_ab < 0.05,
    },
    'original_vs_mixed': {
        't_statistic': float(t_ac),
        'p_value': float(p_ac),
        'cohens_d': float(d_ac),
        'significant': p_ac < 0.05,
    },
    'counterfactual_vs_mixed': {
        't_statistic': float(t_bc),
        'p_value': float(p_bc),
        'cohens_d': float(d_bc),
        'significant': p_bc < 0.05,
    }
}

results_all['statistical_comparison'] = statistical_comparison

print(f"Original vs Counterfactual: t={t_ab:.3f}, p={p_ab:.4f}, d={d_ab:.3f} {'✓' if p_ab < 0.05 else ''}")
print(f"Original vs Mixed: t={t_ac:.3f}, p={p_ac:.4f}, d={d_ac:.3f} {'✓' if p_ac < 0.05 else ''}")
print(f"Counterfactual vs Mixed: t={t_bc:.3f}, p={p_bc:.4f}, d={d_bc:.3f} {'✓' if p_bc < 0.05 else ''}")

# Save all results
with open(Config.OUTPUT_DIR / 'three_way_results.json', 'w') as f:
    # Don't save predictions/probs (too large)
    results_save = {}
    for key in results_all:
        if key == 'statistical_comparison':
            results_save[key] = results_all[key]
        else:
            results_save[key] = {k: v for k, v in results_all[key].items() 
                                if k not in ['test_probs', 'test_preds', 'test_labels']}
    json.dump(results_save, f, indent=2)

print(f"\n✓ Results saved to {Config.OUTPUT_DIR}")

# ============================================================================
# Visualizations
# ============================================================================

print("\n" + "="*60)
print("Creating Visualizations")
print("="*60)

# 1. Performance comparison bar chart
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']
conditions = ['Original', 'Counterfactual', 'Mixed']
values = {
    'Original': [results_A['test_metrics'][m] for m in metrics],
    'Counterfactual': [results_B['test_metrics'][m] for m in metrics],
    'Mixed': [results_C['test_metrics'][m] for m in metrics],
}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.25

for i, cond in enumerate(conditions):
    ax.bar(x + i*width, values[cond], width, label=cond)

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Three-Way Classifier Performance Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.9, 1.0])

plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / 'performance_comparison.png', dpi=200)
plt.close()
print("✓ Performance comparison saved")

# 2. ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, results in [('Original', results_A), ('Counterfactual', results_B), ('Mixed', results_C)]:
    fpr, tpr, _ = roc_curve(results['test_labels'], results['test_probs'])
    auroc = results['test_metrics']['auroc']
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUROC={auroc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Three-Way Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / 'roc_curves.png', dpi=200)
plt.close()
print("✓ ROC curves saved")

# 3. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, results) in zip(axes, [('Original', results_A), ('Counterfactual', results_B), ('Mixed', results_C)]):
    cm = np.array([[results['test_metrics']['confusion_matrix']['TN'], 
                    results['test_metrics']['confusion_matrix']['FP']],
                   [results['test_metrics']['confusion_matrix']['FN'],
                    results['test_metrics']['confusion_matrix']['TP']]])
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'AFib'])
    ax.set_yticklabels(['Normal', 'AFib'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{name}\nAcc={results["test_metrics"]["accuracy"]:.4f}')
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}',
                         ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                         fontsize=14, fontweight='bold')

plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / 'confusion_matrices.png', dpi=200)
plt.close()
print("✓ Confusion matrices saved")

# 4. Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, results, color in [('Original', results_A, 'blue'), 
                             ('Counterfactual', results_B, 'red'), 
                             ('Mixed', results_C, 'green')]:
    epochs = range(1, len(results['history']['train_loss']) + 1)
    axes[0].plot(epochs, results['history']['train_loss'], label=name, color=color, linewidth=2)
    axes[1].plot(epochs, results['history']['val_acc'], label=name, color=color, linewidth=2)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss Curves')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Validation Accuracy Curves')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Training Progress Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / 'training_curves.png', dpi=200)
plt.close()
print("✓ Training curves saved")

print("\n" + "="*60)
print("Three-Way Evaluation Complete!")
print("="*60)
print(f"Results saved to: {Config.OUTPUT_DIR}")
print("\nSummary:")
for cond in ['Original', 'Counterfactual', 'Mixed']:
    key = cond.lower()
    print(f"  {cond}: Accuracy={results_all[key]['test_metrics']['accuracy']:.4f}, "
          f"AUROC={results_all[key]['test_metrics']['auroc']:.4f}, "
          f"Time={results_all[key]['training_time_hours']:.2f}h")
print("="*60)
