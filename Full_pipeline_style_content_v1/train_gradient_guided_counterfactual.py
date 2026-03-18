"""
GRADIENT-GUIDED COUNTERFACTUAL ECG GENERATION
==============================================

Key Insight:
------------
The classifier already knows what distinguishes AFib from Normal. We use its
GRADIENTS to identify WHERE changes are needed, then train a model to make
MINIMAL, LOCALIZED edits only in those regions.

This approach:
1. Uses classifier gradients to create an "importance mask" 
2. Applies edits ONLY where the mask indicates changes are needed
3. Preserves everything else (QRS complexes, T-waves, baseline)

For AFib, the classifier typically focuses on:
- P-wave regions (absence → fibrillatory waves)
- RR intervals (regular → irregular)
- Baseline between beats (smooth → f-waves)

The model learns these patterns from data - no hardcoded medical knowledge!

Architecture:
------------
1. SaliencyMaskGenerator: Uses classifier gradients to find important regions
2. LocalizedEditNetwork: Predicts small edits for those regions only
3. Output = Original * (1 - mask) + Edited * mask

This ensures:
- High correlation with original (most of the signal unchanged)
- Flip success (classifier sees the important changes)
- Physiological plausibility (changes are learned from real data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json

# Suppress wandb
os.environ["WANDB_MODE"] = "disabled"

# Import base models
from counterfactual_training import (
    AFibResLSTM,
    ModelConfig
)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# SALIENCY-BASED IMPORTANCE MASK
# ============================================================================

def compute_saliency_mask(classifier, ecg, target_class, smooth_window=25):
    """
    Compute a saliency mask showing which parts of the ECG are important
    for classification. The mask will be higher in regions that need to
    change to flip the class.
    
    Args:
        classifier: The AFibResLSTM classifier
        ecg: [batch, 1, 2500] ECG signal
        target_class: [batch] target class to flip towards
        smooth_window: Smoothing window for the mask
        
    Returns:
        mask: [batch, 1, 2500] importance mask (0-1)
    """
    ecg = ecg.clone().requires_grad_(True)
    
    # Forward pass
    logits, _ = classifier(ecg)
    
    # We want gradients w.r.t. the TARGET class (not current prediction)
    # This shows what needs to change to become the target class
    target_score = logits.gather(1, target_class.unsqueeze(1))
    
    # Backward pass
    classifier.zero_grad()
    target_score.sum().backward()
    
    # Get gradient magnitude
    saliency = ecg.grad.abs()  # [batch, 1, 2500]
    
    # Smooth the saliency to get broader regions
    if smooth_window > 1:
        kernel = torch.ones(1, 1, smooth_window, device=ecg.device) / smooth_window
        saliency = F.conv1d(saliency, kernel, padding=smooth_window//2)
    
    # Normalize to [0, 1] per sample
    saliency_min = saliency.view(saliency.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
    saliency_max = saliency.view(saliency.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    mask = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
    
    return mask.detach()


# ============================================================================
# LOCALIZED EDIT NETWORK  
# ============================================================================

class LocalizedEditNetwork(nn.Module):
    """
    Network that predicts localized edits based on:
    1. The original ECG
    2. The target class
    3. A saliency mask showing where changes matter
    
    The output is a RESIDUAL that gets added to the original ECG,
    but only in regions indicated by the mask.
    """
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # Encoder - processes the ECG to understand its structure
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),  # Capture wider context
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Class embedding - tells the network which class to target
        self.class_embed = nn.Embedding(2, hidden_dim)
        
        # Mask processor - incorporates saliency information
        self.mask_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        # Feature fusion with FiLM (Feature-wise Linear Modulation)
        # The class embedding modulates the features
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)
        
        # Decoder - generates the edit residual
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, kernel_size=15, padding=7),
            nn.Tanh(),  # Output bounded to [-1, 1]
        )
        
        # Learnable scale for the edit (starts small)
        self.edit_scale = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, ecg, target_class, saliency_mask):
        """
        Args:
            ecg: [batch, 1, 2500] Original ECG
            target_class: [batch] Target class (0 or 1)
            saliency_mask: [batch, 1, 2500] Importance mask from classifier
            
        Returns:
            counterfactual: [batch, 1, 2500] Edited ECG
            edit: [batch, 1, 2500] The raw edit (before masking)
            masked_edit: [batch, 1, 2500] The edit after applying mask
        """
        batch_size = ecg.shape[0]
        
        # Encode ECG
        ecg_features = self.encoder(ecg)  # [batch, hidden, 2500]
        
        # Encode saliency mask
        mask_features = self.mask_encoder(saliency_mask)  # [batch, hidden, 2500]
        
        # Get class embedding and apply FiLM conditioning
        class_emb = self.class_embed(target_class)  # [batch, hidden]
        gamma = self.film_gamma(class_emb).unsqueeze(-1)  # [batch, hidden, 1]
        beta = self.film_beta(class_emb).unsqueeze(-1)  # [batch, hidden, 1]
        
        # Modulate ECG features based on target class
        modulated = gamma * ecg_features + beta
        
        # Concatenate with mask features
        combined = torch.cat([modulated, mask_features], dim=1)  # [batch, hidden*2, 2500]
        
        # Decode to get raw edit
        raw_edit = self.decoder(combined)  # [batch, 1, 2500]
        
        # Scale the edit
        scale = torch.sigmoid(self.edit_scale) * 0.5  # Max edit scale of 0.5
        scaled_edit = raw_edit * scale
        
        # Apply saliency mask - edit more in important regions
        # Use soft masking: edit = edit * (0.1 + 0.9 * mask)
        # This ensures a small baseline edit everywhere but focuses on important regions
        soft_mask = 0.1 + 0.9 * saliency_mask
        masked_edit = scaled_edit * soft_mask
        
        # Apply edit to original
        counterfactual = ecg + masked_edit
        
        return counterfactual, scaled_edit, masked_edit


# ============================================================================
# DATASET
# ============================================================================

class ECGDataset(Dataset):
    def __init__(self, signals, labels, means, stds):
        self.signals = signals
        self.labels = labels
        self.means = means
        self.stds = stds
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        mean = self.means[idx]
        std = self.stds[idx]
        
        # Normalize
        normalized = (signal - mean) / (std + 1e-6)
        
        return {
            'signal': torch.tensor(normalized, dtype=torch.float32).unsqueeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'mean': torch.tensor(mean, dtype=torch.float32),
            'std': torch.tensor(std, dtype=torch.float32)
        }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def waveform_preservation_loss(original, counterfactual, mask):
    """
    Ensure that regions with LOW saliency (not important for classification)
    remain almost identical to the original.
    
    Uses inverse mask: where mask is low, enforce high similarity
    """
    inverse_mask = 1.0 - mask
    
    # L1 loss weighted by inverse mask
    diff = torch.abs(original - counterfactual)
    weighted_diff = diff * inverse_mask
    
    return weighted_diff.mean()


def edit_sparsity_loss(edit):
    """
    Encourage sparse edits - most of the signal should be unchanged.
    Uses L1 norm to encourage many zeros.
    """
    return edit.abs().mean()


def edit_smoothness_loss(edit):
    """
    Encourage smooth edits - no sudden spikes that look unnatural.
    Penalizes high-frequency components in the edit.
    """
    # First derivative
    diff = edit[:, :, 1:] - edit[:, :, :-1]
    return diff.abs().mean()


def correlation_loss(original, counterfactual):
    """
    Maximize correlation between original and counterfactual.
    This ensures overall waveform shape is preserved.
    """
    # Flatten and compute correlation
    orig_flat = original.view(original.size(0), -1)
    cf_flat = counterfactual.view(counterfactual.size(0), -1)
    
    # Normalize
    orig_norm = (orig_flat - orig_flat.mean(dim=1, keepdim=True)) / (orig_flat.std(dim=1, keepdim=True) + 1e-8)
    cf_norm = (cf_flat - cf_flat.mean(dim=1, keepdim=True)) / (cf_flat.std(dim=1, keepdim=True) + 1e-8)
    
    # Correlation
    corr = (orig_norm * cf_norm).mean(dim=1)
    
    # We want to maximize correlation, so minimize (1 - corr)
    return (1 - corr).mean()


# ============================================================================
# TRAINING
# ============================================================================

def train_gradient_guided_model(
    data_path='./ecg_afib_data/X_combined.npy',
    labels_path='./ecg_afib_data/y_combined.npy',
    norm_params_path='./enhanced_counterfactual_training/norm_params.npy',
    classifier_path='./best_model/best_model.pth',
    output_dir='./gradient_guided_training',
    num_epochs=30,
    batch_size=32,
    lr=1e-4,
    # Loss weights
    lambda_flip=1.0,          # Classification flip
    lambda_correlation=5.0,   # Strong correlation enforcement
    lambda_preservation=2.0,  # Preserve non-important regions
    lambda_sparsity=0.5,      # Sparse edits
    lambda_smoothness=0.3,    # Smooth edits
    lambda_identity=1.0,      # Same class → same ECG
):
    """
    Train the gradient-guided counterfactual model
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/visualizations', exist_ok=True)
    
    print("="*70)
    print("GRADIENT-GUIDED COUNTERFACTUAL TRAINING")
    print("="*70)
    print("""
    This approach uses classifier gradients to identify WHERE in the ECG
    changes are needed, then applies MINIMAL, LOCALIZED edits only there.
    
    Goal: High flip rate + High correlation (>0.9)
    """)
    
    # ========================================
    # 1. Load Data
    # ========================================
    print("\n📂 Loading data...")
    signals = np.load(data_path)
    labels = np.load(labels_path)
    norm_params = np.load(norm_params_path, allow_pickle=True).item()
    
    if labels.dtype.kind in ['U', 'S', 'O']:
        labels = np.array([1 if l == 'A' else 0 for l in labels])
    
    means = norm_params['means']
    stds = norm_params['stds']
    
    print(f"   Dataset size: {len(signals):,}")
    print(f"   AFib: {np.sum(labels == 1):,}, Normal: {np.sum(labels == 0):,}")
    
    # Create dataset
    dataset = ECGDataset(signals, labels, means, stds)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # ========================================
    # 2. Initialize Models
    # ========================================
    print("\n🔧 Initializing models...")
    
    # Edit network (trainable)
    edit_net = LocalizedEditNetwork(hidden_dim=128).to(DEVICE)
    print(f"   Edit network parameters: {sum(p.numel() for p in edit_net.parameters()):,}")
    
    # Classifier (for gradient computation and flip verification)
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    clf_ckpt = torch.load(classifier_path, map_location=DEVICE, weights_only=False)
    classifier.load_state_dict(clf_ckpt['model_state_dict'])
    print("   ✅ Classifier loaded")
    
    # ========================================
    # 3. Optimizer
    # ========================================
    optimizer = torch.optim.AdamW(edit_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # ========================================
    # 4. Training Loop
    # ========================================
    print("\n🚀 Starting training...")
    
    history = {
        'flip_loss': [],
        'correlation_loss': [],
        'preservation_loss': [],
        'sparsity_loss': [],
        'identity_loss': [],
        'total_loss': [],
        'flip_rate': [],
        'mean_correlation': [],
    }
    
    best_score = 0
    
    for epoch in range(num_epochs):
        edit_net.train()
        
        epoch_metrics = {k: 0.0 for k in history.keys()}
        epoch_flips = 0
        epoch_total = 0
        correlations = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            signals_batch = batch['signal'].to(DEVICE)
            labels_batch = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # ========================================
            # Compute Saliency Mask (for flip direction)
            # ========================================
            target_class = 1 - labels_batch  # Flip direction
            classifier.eval()  # For clean gradients
            saliency_mask = compute_saliency_mask(classifier, signals_batch, target_class)
            classifier.train()  # Back to train for loss computation
            
            # ========================================
            # Forward pass - Generate Counterfactual
            # ========================================
            cf, raw_edit, masked_edit = edit_net(signals_batch, target_class, saliency_mask)
            
            # ========================================
            # Loss 1: Flip Loss (classifier should predict target class)
            # ========================================
            logits, _ = classifier(cf)
            flip_loss = F.cross_entropy(logits, target_class)
            
            # ========================================
            # Loss 2: Correlation Loss (preserve overall shape)
            # ========================================
            corr_loss = correlation_loss(signals_batch, cf)
            
            # ========================================
            # Loss 3: Preservation Loss (keep non-important regions unchanged)
            # ========================================
            pres_loss = waveform_preservation_loss(signals_batch, cf, saliency_mask)
            
            # ========================================
            # Loss 4: Sparsity Loss (minimize edit magnitude)
            # ========================================
            sparse_loss = edit_sparsity_loss(masked_edit)
            
            # ========================================
            # Loss 5: Smoothness Loss (smooth transitions in edits)
            # ========================================
            smooth_loss = edit_smoothness_loss(masked_edit)
            
            # ========================================
            # Loss 6: Identity Loss (same class → no change)
            # ========================================
            cf_identity, _, _ = edit_net(signals_batch, labels_batch, saliency_mask * 0)  # Zero mask for identity
            identity_loss = F.mse_loss(cf_identity, signals_batch)
            
            # ========================================
            # Total Loss
            # ========================================
            total_loss = (
                lambda_flip * flip_loss +
                lambda_correlation * corr_loss +
                lambda_preservation * pres_loss +
                lambda_sparsity * sparse_loss +
                lambda_smoothness * smooth_loss +
                lambda_identity * identity_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(edit_net.parameters(), 1.0)
            optimizer.step()
            
            # ========================================
            # Metrics
            # ========================================
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                flips = (pred == target_class).sum().item()
                epoch_flips += flips
                epoch_total += signals_batch.shape[0]
                
                # Compute correlations
                for i in range(signals_batch.shape[0]):
                    orig = signals_batch[i].cpu().numpy().flatten()
                    cf_np = cf[i].cpu().numpy().flatten()
                    corr = np.corrcoef(orig, cf_np)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            epoch_metrics['flip_loss'] += flip_loss.item()
            epoch_metrics['correlation_loss'] += corr_loss.item()
            epoch_metrics['preservation_loss'] += pres_loss.item()
            epoch_metrics['sparsity_loss'] += sparse_loss.item()
            epoch_metrics['identity_loss'] += identity_loss.item()
            epoch_metrics['total_loss'] += total_loss.item()
            
            pbar.set_postfix({
                'flip': f'{flip_loss.item():.3f}',
                'corr': f'{1-corr_loss.item():.3f}',
                'fr': f'{epoch_flips/max(1,epoch_total)*100:.1f}%',
            })
        
        scheduler.step()
        
        # Epoch averages
        num_batches = len(dataloader)
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
            history[k].append(epoch_metrics[k])
        
        flip_rate = epoch_flips / epoch_total
        mean_corr = np.mean(correlations) if correlations else 0
        history['flip_rate'].append(flip_rate)
        history['mean_correlation'].append(mean_corr)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"   Flip Rate: {flip_rate*100:.1f}%")
        print(f"   Mean Correlation: {mean_corr:.4f}")
        print(f"   Flip Loss: {epoch_metrics['flip_loss']:.4f}")
        print(f"   Corr Loss: {epoch_metrics['correlation_loss']:.4f}")
        print(f"   Edit Scale: {torch.sigmoid(edit_net.edit_scale).item()*0.5:.4f}")
        
        # ========================================
        # Visualize some examples
        # ========================================
        if (epoch + 1) % 5 == 0 or epoch == 0:
            visualize_counterfactuals(
                edit_net, classifier, dataset, 
                f'{output_dir}/visualizations/epoch_{epoch+1}.png',
                device=DEVICE
            )
        
        # ========================================
        # Save best model (maximize flip_rate * correlation)
        # ========================================
        score = flip_rate * mean_corr
        if score > best_score:
            best_score = score
            torch.save({
                'epoch': epoch,
                'model_state_dict': edit_net.state_dict(),
                'flip_rate': flip_rate,
                'mean_correlation': mean_corr,
                'score': score,
            }, f'{output_dir}/best_model.pth')
            print(f"   💾 Best model saved! Score: {score:.4f} (FR: {flip_rate:.2f} × Corr: {mean_corr:.2f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': edit_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, f'{output_dir}/checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model and history
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': edit_net.state_dict(),
        'history': history,
    }, f'{output_dir}/final_model.pth')
    
    # Plot training history
    plot_training_history(history, f'{output_dir}/training_history.png')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best score: {best_score:.4f}")
    print(f"Output directory: {output_dir}/")
    
    return history


def visualize_counterfactuals(edit_net, classifier, dataset, save_path, device, num_samples=4):
    """Visualize original vs counterfactual ECGs with overlay"""
    edit_net.eval()
    classifier.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4*num_samples))
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        ecg = sample['signal'].unsqueeze(0).to(device)
        label = sample['label'].unsqueeze(0).to(device)
        target = 1 - label
        
        with torch.no_grad():
            # Compute saliency
            classifier.eval()
            ecg_grad = ecg.clone().requires_grad_(True)
            logits, _ = classifier(ecg_grad)
            logits.gather(1, target.unsqueeze(1)).sum().backward()
            saliency = ecg_grad.grad.abs()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            # Generate counterfactual
            cf, raw_edit, masked_edit = edit_net(ecg, target, saliency)
            
            # Classify both
            orig_pred, _ = classifier(ecg)
            cf_pred, _ = classifier(cf)
            orig_class = orig_pred.argmax().item()
            cf_class = cf_pred.argmax().item()
            
            # Compute correlation
            orig_np = ecg.cpu().numpy().flatten()
            cf_np = cf.cpu().numpy().flatten()
            corr = np.corrcoef(orig_np, cf_np)[0, 1]
        
        class_names = ['Normal', 'AFib']
        t = np.arange(2500) / 250  # Time in seconds
        
        # Plot 1: Original ECG
        axes[i, 0].plot(t, orig_np, 'b-', linewidth=0.8, label='Original')
        axes[i, 0].set_title(f'Original: {class_names[label.item()]} (pred: {class_names[orig_class]})')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot 2: Counterfactual ECG
        axes[i, 1].plot(t, cf_np, 'r-', linewidth=0.8, label='Counterfactual')
        axes[i, 1].set_title(f'Counterfactual → {class_names[target.item()]} (pred: {class_names[cf_class]})')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Plot 3: OVERLAY (the key visualization for clinicians!)
        axes[i, 2].plot(t, orig_np, 'b-', linewidth=0.8, alpha=0.7, label='Original')
        axes[i, 2].plot(t, cf_np, 'r-', linewidth=0.8, alpha=0.7, label='Counterfactual')
        # Highlight the edit regions
        edit_np = masked_edit.cpu().numpy().flatten()
        significant = np.abs(edit_np) > 0.01
        axes[i, 2].fill_between(t, orig_np.min(), orig_np.max(), 
                                where=significant, alpha=0.2, color='yellow', label='Edit regions')
        axes[i, 2].set_title(f'OVERLAY - Correlation: {corr:.3f} | Flip: {"✓" if cf_class == target.item() else "✗"}')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].set_ylabel('Amplitude')
        axes[i, 2].legend(loc='upper right')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    edit_net.train()


def plot_training_history(history, save_path):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Losses
    axes[0, 0].plot(history['flip_loss'], label='Flip', color='red')
    axes[0, 0].plot(history['correlation_loss'], label='Correlation', color='blue')
    axes[0, 0].plot(history['preservation_loss'], label='Preservation', color='green')
    axes[0, 0].set_title('Primary Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['sparsity_loss'], label='Sparsity', color='orange')
    axes[0, 1].plot(history['identity_loss'], label='Identity', color='purple')
    axes[0, 1].set_title('Regularization Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(history['total_loss'], 'k-', linewidth=2)
    axes[0, 2].set_title('Total Loss')
    axes[0, 2].grid(True)
    
    # Metrics
    axes[1, 0].plot(history['flip_rate'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=0.8, color='r', linestyle='--', label='Target 80%')
    axes[1, 0].set_title('Flip Rate')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['mean_correlation'], 'b-', linewidth=2)
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='Target 0.9')
    axes[1, 1].set_title('Mean Correlation')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Combined score
    combined = [f * c for f, c in zip(history['flip_rate'], history['mean_correlation'])]
    axes[1, 2].plot(combined, 'purple', linewidth=2)
    axes[1, 2].set_title('Combined Score (FR × Corr)')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train_gradient_guided_model(
        num_epochs=30,
        batch_size=32,
        lr=1e-4,
        lambda_flip=1.0,
        lambda_correlation=5.0,      # Strong emphasis on preserving shape
        lambda_preservation=2.0,
        lambda_sparsity=0.5,
        lambda_smoothness=0.3,
        lambda_identity=1.0,
    )
