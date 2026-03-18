"""
PROPER COUNTERFACTUAL TRAINING - MINIMAL EDIT APPROACH
=======================================================

The problem with the current model:
- It was trained to GENERATE ECGs from noise
- It doesn't learn to PRESERVE the original and make MINIMAL changes

What we need:
- Train the model to RECONSTRUCT the original ECG when given the same class
- Train to make MINIMAL changes when flipping to opposite class
- Enforce similarity between original and counterfactual

New Training Objectives:
========================

1. IDENTITY LOSS (same class):
   - Input: ECG of class A, condition on class A
   - Output: Should be EXACTLY the same as input
   - Loss: MSE(output, input)

2. FLIP LOSS (opposite class):
   - Input: ECG of class A, condition on class B
   - Output: Should flip the classifier's prediction
   - Loss: CrossEntropy(classifier(output), class_B)

3. MINIMAL EDIT LOSS:
   - The counterfactual should be CLOSE to the original
   - Loss: MSE(counterfactual, original) * lambda
   - But not too close that it doesn't flip

4. PSD PRESERVATION LOSS:
   - The frequency content should be similar
   - Loss: MSE(PSD(counterfactual), PSD(original))

5. BEAT PRESERVATION LOSS:
   - Individual heartbeats should look similar
   - Only RR intervals and P-waves should change

Key Innovation:
- Instead of starting from noise, we start from the ORIGINAL ECG
- We train to predict what EDITS are needed, not the full ECG
- The model learns: output = original + edits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import DDIMScheduler
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
    UNet1DConditional,
    StyleEncoderWrapper,
    PretrainedContentEncoder,
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
# RESIDUAL EDIT MODEL
# ============================================================================

class ResidualEditModel(nn.Module):
    """
    Instead of generating a full ECG, this model predicts the RESIDUAL
    (the edit) that should be applied to the original ECG.
    
    output = original + alpha * edit_network(original, target_class)
    
    where alpha is learned and controls the edit strength
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # Encoder: compress ECG to features
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 1250
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # 625
            nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=2),  # 313
            nn.ReLU(),
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(2, hidden_dim)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder: predict the edit residual
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),  # 626
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 1252
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # 2504
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),  # 2504
        )
        
        # Adaptive pooling to handle size mismatch
        self.output_adapt = nn.AdaptiveAvgPool1d(2500)
        
        # Learnable edit strength (start small)
        self.edit_strength = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x, target_class):
        """
        Args:
            x: [batch, 1, 2500] - Original ECG
            target_class: [batch] - Target class (0 or 1)
            
        Returns:
            counterfactual: [batch, 1, 2500] - Edited ECG
            edit: [batch, 1, 2500] - The edit that was applied
        """
        batch_size = x.shape[0]
        
        # Encode ECG
        features = self.encoder(x)  # [batch, hidden, seq]
        
        # Get class embedding
        class_emb = self.class_embed(target_class)  # [batch, hidden]
        
        # Global average pooling on features
        features_global = features.mean(dim=-1)  # [batch, hidden]
        
        # Fuse features with class
        fused = self.fusion(torch.cat([features_global, class_emb], dim=-1))  # [batch, hidden]
        
        # Broadcast to spatial dimensions
        fused = fused.unsqueeze(-1).expand(-1, -1, features.shape[-1])  # [batch, hidden, seq]
        
        # Add fused features to encoder output
        combined = features + fused
        
        # Decode to get edit
        edit = self.decoder(combined)  # [batch, 1, ~2504]
        
        # Adapt to correct size
        edit = self.output_adapt(edit)  # [batch, 1, 2500]
        
        # Apply edit with learnable strength
        strength = torch.sigmoid(self.edit_strength)  # Bounded 0-1
        counterfactual = x + strength * edit
        
        return counterfactual, edit


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
# TRAINING
# ============================================================================

def train_minimal_edit_model(
    data_path='./ecg_afib_data/X_combined.npy',
    labels_path='./ecg_afib_data/y_combined.npy',
    norm_params_path='./enhanced_counterfactual_training/norm_params.npy',
    classifier_path='./best_model/best_model.pth',
    output_dir='./minimal_edit_training',
    num_epochs=50,
    batch_size=32,
    lr=1e-4,
    lambda_identity=1.0,
    lambda_flip=1.0,
    lambda_similarity=2.0,  # Strong similarity enforcement
):
    """
    Train the minimal edit model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("MINIMAL EDIT COUNTERFACTUAL TRAINING")
    print("="*70)
    
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # ========================================
    # 2. Initialize Models
    # ========================================
    print("\n🔧 Initializing models...")
    
    # Edit model (trainable)
    edit_model = ResidualEditModel().to(DEVICE)
    print(f"   Edit model parameters: {sum(p.numel() for p in edit_model.parameters()):,}")
    
    # Classifier (frozen, for guidance)
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    clf_ckpt = torch.load(classifier_path, map_location=DEVICE, weights_only=False)
    classifier.load_state_dict(clf_ckpt['model_state_dict'])
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print("   ✅ Classifier loaded (frozen)")
    
    # ========================================
    # 3. Optimizer
    # ========================================
    optimizer = torch.optim.AdamW(edit_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # ========================================
    # 4. Training Loop
    # ========================================
    print("\n🚀 Starting training...")
    
    history = {
        'identity_loss': [],
        'flip_loss': [],
        'similarity_loss': [],
        'total_loss': [],
        'flip_rate': [],
        'mean_correlation': []
    }
    
    best_combined_score = 0
    
    for epoch in range(num_epochs):
        edit_model.train()
        classifier.train()  # Need gradients for flip loss
        
        epoch_identity_loss = 0
        epoch_flip_loss = 0
        epoch_similarity_loss = 0
        epoch_total_loss = 0
        epoch_flips = 0
        epoch_total = 0
        correlations = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            signals_batch = batch['signal'].to(DEVICE)
            labels_batch = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # ========================================
            # IDENTITY LOSS: Same class → Same ECG
            # ========================================
            cf_same, edit_same = edit_model(signals_batch, labels_batch)
            identity_loss = F.mse_loss(cf_same, signals_batch)
            
            # Edit for same class should be minimal
            identity_loss += 0.1 * edit_same.abs().mean()
            
            # ========================================
            # FLIP LOSS: Opposite class → Flip prediction
            # ========================================
            target_class = 1 - labels_batch  # Flip class
            cf_flip, edit_flip = edit_model(signals_batch, target_class)
            
            # Classify counterfactual
            logits, _ = classifier(cf_flip)
            flip_loss = F.cross_entropy(logits, target_class)
            
            # ========================================
            # SIMILARITY LOSS: CF should be close to original
            # ========================================
            similarity_loss = F.mse_loss(cf_flip, signals_batch)
            
            # ========================================
            # TOTAL LOSS
            # ========================================
            total_loss = (
                lambda_identity * identity_loss +
                lambda_flip * flip_loss +
                lambda_similarity * similarity_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(edit_model.parameters(), 1.0)
            optimizer.step()
            
            # ========================================
            # Metrics
            # ========================================
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                flips = (pred == target_class).sum().item()
                epoch_flips += flips
                epoch_total += signals_batch.shape[0]
                
                # Correlation
                for i in range(signals_batch.shape[0]):
                    orig = signals_batch[i].cpu().numpy().flatten()
                    cf = cf_flip[i].cpu().numpy().flatten()
                    corr = np.corrcoef(orig, cf)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            epoch_identity_loss += identity_loss.item()
            epoch_flip_loss += flip_loss.item()
            epoch_similarity_loss += similarity_loss.item()
            epoch_total_loss += total_loss.item()
            
            pbar.set_postfix({
                'id': f'{identity_loss.item():.4f}',
                'flip': f'{flip_loss.item():.4f}',
                'sim': f'{similarity_loss.item():.4f}',
                'fr': f'{epoch_flips/epoch_total*100:.1f}%'
            })
        
        scheduler.step()
        
        # Epoch averages
        num_batches = len(dataloader)
        avg_identity = epoch_identity_loss / num_batches
        avg_flip = epoch_flip_loss / num_batches
        avg_similarity = epoch_similarity_loss / num_batches
        avg_total = epoch_total_loss / num_batches
        flip_rate = epoch_flips / epoch_total
        mean_corr = np.mean(correlations) if correlations else 0
        
        history['identity_loss'].append(avg_identity)
        history['flip_loss'].append(avg_flip)
        history['similarity_loss'].append(avg_similarity)
        history['total_loss'].append(avg_total)
        history['flip_rate'].append(flip_rate)
        history['mean_correlation'].append(mean_corr)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"   Identity Loss: {avg_identity:.4f}")
        print(f"   Flip Loss: {avg_flip:.4f}")
        print(f"   Similarity Loss: {avg_similarity:.4f}")
        print(f"   Flip Rate: {flip_rate*100:.1f}%")
        print(f"   Mean Correlation: {mean_corr:.4f}")
        print(f"   Edit Strength: {torch.sigmoid(edit_model.edit_strength).item():.4f}")
        
        # Combined score: we want high flip rate AND high correlation
        combined_score = flip_rate * mean_corr
        
        # Save best model
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': edit_model.state_dict(),
                'flip_rate': flip_rate,
                'mean_correlation': mean_corr,
                'combined_score': combined_score
            }, f'{output_dir}/best_model.pth')
            print(f"   💾 Best model saved! Score: {combined_score:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': edit_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f'{output_dir}/checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': edit_model.state_dict(),
        'history': history
    }, f'{output_dir}/final_model.pth')
    
    # Plot history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['identity_loss'], label='Identity')
    axes[0, 0].plot(history['flip_loss'], label='Flip')
    axes[0, 0].plot(history['similarity_loss'], label='Similarity')
    axes[0, 0].set_title('Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['flip_rate'], 'g-', label='Flip Rate')
    axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='Target (80%)')
    axes[0, 1].set_title('Flip Rate')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['mean_correlation'], 'b-', label='Mean Correlation')
    axes[1, 0].axhline(y=0.9, color='r', linestyle='--', label='Target (0.9)')
    axes[1, 0].set_title('Morphology Preservation (Correlation)')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined score
    combined = [f * c for f, c in zip(history['flip_rate'], history['mean_correlation'])]
    axes[1, 1].plot(combined, 'purple', label='Flip Rate × Correlation')
    axes[1, 1].set_title('Combined Score (Higher = Better)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=150)
    plt.close()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best combined score: {best_combined_score:.4f}")
    print(f"Output directory: {output_dir}/")

if __name__ == "__main__":
    train_minimal_edit_model(
        num_epochs=50,
        batch_size=32,
        lr=1e-4,
        lambda_identity=1.0,
        lambda_flip=1.0,
        lambda_similarity=2.0
    )
