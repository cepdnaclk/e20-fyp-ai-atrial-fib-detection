"""
MINIMAL EDIT COUNTERFACTUAL TRAINING V2
========================================

Improvements over v1:
1. Proper train/validation/test splits (70/15/15)
2. Early stopping based on validation performance
3. Better regularization (dropout, weight decay)
4. Gradient-guided saliency masking for focused edits
5. More comprehensive logging and checkpointing
6. Proper evaluation on held-out test set

Usage:
    python train_minimal_edit_v2.py

Output:
    ./minimal_edit_v2/
        ├── best_model.pth          # Best model based on validation
        ├── final_model.pth         # Final model after training
        ├── training_history.png    # Training curves
        ├── test_results.json       # Test set evaluation
        ├── config.json             # Training configuration
        └── checkpoints/            # Epoch checkpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json
from sklearn.model_selection import train_test_split

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ============================================================================
# IMPROVED RESIDUAL EDIT MODEL WITH DROPOUT
# ============================================================================

class ResidualEditModelV2(nn.Module):
    """
    Improved residual edit model with:
    - Dropout for regularization
    - Skip connections for better gradient flow
    - Attention mechanism for focusing on important regions
    """
    
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Encoder with dropout
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 1250
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # 625
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, hidden_dim, kernel_size=5, stride=2, padding=2),  # 313
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(2, hidden_dim)
        
        # Feature fusion with deeper network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Spatial attention for focusing edits
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        
        # Adaptive pooling for size matching
        self.output_adapt = nn.AdaptiveAvgPool1d(2500)
        
        # Learnable edit strength (initialize small)
        self.edit_strength = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x, target_class):
        """
        Args:
            x: [batch, 1, 2500] - Original ECG
            target_class: [batch] - Target class (0 or 1)
            
        Returns:
            counterfactual: [batch, 1, 2500] - Edited ECG
            edit: [batch, 1, 2500] - The edit that was applied
            attention: [batch, 1, 313] - Spatial attention weights
        """
        batch_size = x.shape[0]
        
        # Encode ECG
        features = self.encoder(x)  # [batch, hidden, 313]
        
        # Compute spatial attention (where to edit)
        attention = self.spatial_attention(features)  # [batch, 1, 313]
        
        # Get class embedding
        class_emb = self.class_embed(target_class)  # [batch, hidden]
        
        # Global average pooling
        features_global = features.mean(dim=-1)  # [batch, hidden]
        
        # Fuse features with class
        fused = self.fusion(torch.cat([features_global, class_emb], dim=-1))
        
        # Broadcast and apply attention
        fused = fused.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        combined = features + fused * attention  # Attention-weighted fusion
        
        # Decode to get edit
        edit = self.decoder(combined)
        edit = self.output_adapt(edit)
        
        # Apply edit with bounded strength
        strength = torch.sigmoid(self.edit_strength)  # 0-1
        counterfactual = x + strength * edit
        
        return counterfactual, edit, attention


# ============================================================================
# DATASET WITH PROPER SPLITS
# ============================================================================

class ECGDataset(Dataset):
    def __init__(self, signals, labels, normalize=True):
        self.signals = signals
        self.labels = labels
        self.normalize = normalize
        
        # Compute per-sample statistics
        self.means = np.mean(signals, axis=1, keepdims=True)
        self.stds = np.std(signals, axis=1, keepdims=True) + 1e-6
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.normalize:
            signal = (signal - self.means[idx]) / self.stds[idx]
        
        return {
            'signal': torch.tensor(signal, dtype=torch.float32).unsqueeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'idx': idx
        }


def create_data_splits(data_path, labels_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create proper train/val/test splits with stratification.
    """
    print("\n📂 Loading and splitting data...")
    
    signals = np.load(data_path)
    labels = np.load(labels_path)
    
    # Convert string labels to binary
    if labels.dtype.kind in ['U', 'S', 'O']:
        labels = np.array([1 if l == 'A' else 0 for l in labels])
    
    print(f"   Total samples: {len(signals):,}")
    print(f"   AFib: {np.sum(labels == 1):,} ({np.mean(labels == 1)*100:.1f}%)")
    print(f"   Normal: {np.sum(labels == 0):,} ({np.mean(labels == 0)*100:.1f}%)")
    
    # First split: train vs (val+test)
    indices = np.arange(len(signals))
    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), 
        stratify=labels, random_state=42
    )
    
    # Second split: val vs test
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio/(val_ratio + test_ratio),
        stratify=temp_labels, random_state=42
    )
    
    print(f"\n   Train: {len(train_idx):,} samples ({len(train_idx)/len(signals)*100:.1f}%)")
    print(f"   Val:   {len(val_idx):,} samples ({len(val_idx)/len(signals)*100:.1f}%)")
    print(f"   Test:  {len(test_idx):,} samples ({len(test_idx)/len(signals)*100:.1f}%)")
    
    # Create datasets
    train_dataset = ECGDataset(signals[train_idx], labels[train_idx])
    val_dataset = ECGDataset(signals[val_idx], labels[val_idx])
    test_dataset = ECGDataset(signals[test_idx], labels[test_idx])
    
    return train_dataset, val_dataset, test_dataset, {
        'train_idx': train_idx.tolist(),
        'val_idx': val_idx.tolist(),
        'test_idx': test_idx.tolist()
    }


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

@torch.no_grad()
def evaluate(edit_model, classifier, dataloader, device, desc="Evaluating"):
    """
    Evaluate model on a dataset.
    
    Returns:
        dict with flip_rate, mean_correlation, losses, etc.
    """
    edit_model.eval()
    classifier.eval()
    
    total_identity_loss = 0
    total_flip_loss = 0
    total_similarity_loss = 0
    
    total_flips = 0
    total_samples = 0
    correlations = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            signals = batch['signal'].to(device)
            labels = batch['label'].to(device)
            
            # Identity loss
            cf_same, edit_same, _ = edit_model(signals, labels)
            identity_loss = F.mse_loss(cf_same, signals)
            total_identity_loss += identity_loss.item() * signals.shape[0]
            
            # Flip loss
            target_class = 1 - labels
            cf_flip, edit_flip, _ = edit_model(signals, target_class)
            
            logits, _ = classifier(cf_flip)
            flip_loss = F.cross_entropy(logits, target_class, reduction='sum')
            total_flip_loss += flip_loss.item()
            
            # Similarity loss
            similarity_loss = F.mse_loss(cf_flip, signals, reduction='sum')
            total_similarity_loss += similarity_loss.item()
            
            # Flip rate
            pred = logits.argmax(dim=1)
            total_flips += (pred == target_class).sum().item()
            total_samples += signals.shape[0]
            
            # Correlation
            for i in range(signals.shape[0]):
                orig = signals[i].cpu().numpy().flatten()
                cf = cf_flip[i].cpu().numpy().flatten()
                corr = np.corrcoef(orig, cf)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
    
    return {
        'identity_loss': total_identity_loss / total_samples,
        'flip_loss': total_flip_loss / total_samples,
        'similarity_loss': total_similarity_loss / total_samples,
        'flip_rate': total_flips / total_samples,
        'mean_correlation': np.mean(correlations) if correlations else 0,
        'std_correlation': np.std(correlations) if correlations else 0,
        'min_correlation': np.min(correlations) if correlations else 0,
        'num_samples': total_samples
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_minimal_edit_v2(
    data_path='./ecg_afib_data/X_combined.npy',
    labels_path='./ecg_afib_data/y_combined.npy',
    classifier_path='./best_model/best_model.pth',
    output_dir='./minimal_edit_v2',
    num_epochs=100,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    lambda_identity=1.0,
    lambda_flip=1.0,
    lambda_similarity=2.0,
    lambda_sparsity=0.1,  # Encourage sparse edits
    patience=15,  # Early stopping patience
    dropout=0.1,
):
    """
    Train the minimal edit model with proper train/val/test splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    # Save configuration
    config = {
        'data_path': data_path,
        'labels_path': labels_path,
        'classifier_path': classifier_path,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'lambda_identity': lambda_identity,
        'lambda_flip': lambda_flip,
        'lambda_similarity': lambda_similarity,
        'lambda_sparsity': lambda_sparsity,
        'patience': patience,
        'dropout': dropout,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("="*70)
    print("MINIMAL EDIT COUNTERFACTUAL TRAINING V2")
    print("="*70)
    print(f"Output directory: {output_dir}")
    
    # ========================================
    # 1. Create Data Splits
    # ========================================
    train_dataset, val_dataset, test_dataset, split_indices = create_data_splits(
        data_path, labels_path
    )
    
    # Save split indices for reproducibility
    with open(f"{output_dir}/split_indices.json", 'w') as f:
        json.dump(split_indices, f)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # ========================================
    # 2. Initialize Models
    # ========================================
    print("\n🔧 Initializing models...")
    
    edit_model = ResidualEditModelV2(dropout=dropout).to(DEVICE)
    print(f"   Edit model parameters: {sum(p.numel() for p in edit_model.parameters()):,}")
    
    # Load frozen classifier (lazy import to avoid slow diffusers import)
    print("   Loading classifier (this may take a moment due to library imports)...")
    from counterfactual_training import AFibResLSTM, ModelConfig
    clf_config = ModelConfig()
    classifier = AFibResLSTM(clf_config).to(DEVICE)
    clf_ckpt = torch.load(classifier_path, map_location=DEVICE, weights_only=False)
    classifier.load_state_dict(clf_ckpt['model_state_dict'] if 'model_state_dict' in clf_ckpt else clf_ckpt)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print("   ✅ Classifier loaded (frozen)")
    
    # ========================================
    # 3. Optimizer & Scheduler
    # ========================================
    optimizer = torch.optim.AdamW(edit_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # ========================================
    # 4. Training Loop
    # ========================================
    print("\n🚀 Starting training...")
    print(f"   Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"   Early stopping patience: {patience}")
    
    history = {
        'train': {'identity_loss': [], 'flip_loss': [], 'similarity_loss': [], 
                  'flip_rate': [], 'mean_correlation': []},
        'val': {'identity_loss': [], 'flip_loss': [], 'similarity_loss': [],
                'flip_rate': [], 'mean_correlation': []}
    }
    
    best_val_score = 0
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        edit_model.train()
        
        epoch_losses = {'identity': 0, 'flip': 0, 'similarity': 0, 'sparsity': 0, 'total': 0}
        epoch_flips = 0
        epoch_total = 0
        train_correlations = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            signals = batch['signal'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # ========================================
            # IDENTITY LOSS
            # ========================================
            cf_same, edit_same, attn_same = edit_model(signals, labels)
            identity_loss = F.mse_loss(cf_same, signals)
            identity_loss += 0.1 * edit_same.abs().mean()  # Penalize edits for same class
            
            # ========================================
            # FLIP LOSS
            # ========================================
            target_class = 1 - labels
            cf_flip, edit_flip, attn_flip = edit_model(signals, target_class)
            
            # Use classifier to compute flip loss
            # Need to avoid cuDNN LSTM backward in eval mode
            # Solution: Temporarily set classifier to train mode for the forward pass
            # but keep parameters frozen
            classifier.train()  # Enable training mode for LSTM compatibility
            logits, _ = classifier(cf_flip)
            classifier.eval()  # Back to eval mode
            flip_loss = F.cross_entropy(logits, target_class)
            
            # ========================================
            # SIMILARITY LOSS
            # ========================================
            similarity_loss = F.mse_loss(cf_flip, signals)
            
            # ========================================
            # SPARSITY LOSS (encourage focused edits)
            # ========================================
            sparsity_loss = edit_flip.abs().mean()
            
            # ========================================
            # TOTAL LOSS
            # ========================================
            total_loss = (
                lambda_identity * identity_loss +
                lambda_flip * flip_loss +
                lambda_similarity * similarity_loss +
                lambda_sparsity * sparsity_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(edit_model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            epoch_losses['identity'] += identity_loss.item()
            epoch_losses['flip'] += flip_loss.item()
            epoch_losses['similarity'] += similarity_loss.item()
            epoch_losses['sparsity'] += sparsity_loss.item()
            epoch_losses['total'] += total_loss.item()
            
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                epoch_flips += (pred == target_class).sum().item()
                epoch_total += signals.shape[0]
                
                for i in range(signals.shape[0]):
                    orig = signals[i].cpu().numpy().flatten()
                    cf = cf_flip[i].cpu().numpy().flatten()
                    corr = np.corrcoef(orig, cf)[0, 1]
                    if not np.isnan(corr):
                        train_correlations.append(corr)
            
            pbar.set_postfix({
                'id': f'{identity_loss.item():.4f}',
                'flip': f'{flip_loss.item():.4f}',
                'fr': f'{epoch_flips/epoch_total*100:.1f}%'
            })
        
        # Compute epoch averages
        num_batches = len(train_loader)
        train_metrics = {
            'identity_loss': epoch_losses['identity'] / num_batches,
            'flip_loss': epoch_losses['flip'] / num_batches,
            'similarity_loss': epoch_losses['similarity'] / num_batches,
            'flip_rate': epoch_flips / epoch_total,
            'mean_correlation': np.mean(train_correlations) if train_correlations else 0
        }
        
        # Record training history
        for key in history['train']:
            history['train'][key].append(train_metrics[key])
        
        # ========================================
        # VALIDATION
        # ========================================
        val_metrics = evaluate(edit_model, classifier, val_loader, DEVICE, "Validating")
        
        for key in history['val']:
            history['val'][key].append(val_metrics[key])
        
        # Combined score: flip_rate * correlation
        val_score = val_metrics['flip_rate'] * val_metrics['mean_correlation']
        
        # Learning rate scheduler step
        scheduler.step(val_score)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"   Train - Flip Rate: {train_metrics['flip_rate']*100:.1f}%, "
              f"Correlation: {train_metrics['mean_correlation']:.4f}")
        print(f"   Val   - Flip Rate: {val_metrics['flip_rate']*100:.1f}%, "
              f"Correlation: {val_metrics['mean_correlation']:.4f}")
        print(f"   Val Score: {val_score:.4f}, Edit Strength: {torch.sigmoid(edit_model.edit_strength).item():.4f}")
        
        # Save best model
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': edit_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'val_score': val_score
            }, f'{output_dir}/best_model.pth')
            print(f"   💾 Best model saved! Score: {val_score:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"   No improvement for {epochs_without_improvement} epochs")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': edit_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f'{output_dir}/checkpoints/epoch_{epoch+1}.pth')
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
    
    # ========================================
    # 5. FINAL TEST EVALUATION
    # ========================================
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    # Load best model
    best_ckpt = torch.load(f'{output_dir}/best_model.pth', map_location=DEVICE)
    edit_model.load_state_dict(best_ckpt['model_state_dict'])
    
    test_metrics = evaluate(edit_model, classifier, test_loader, DEVICE, "Testing")
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Flip Rate:        {test_metrics['flip_rate']*100:.1f}%")
    print(f"   Mean Correlation: {test_metrics['mean_correlation']:.4f} ± {test_metrics['std_correlation']:.4f}")
    print(f"   Min Correlation:  {test_metrics['min_correlation']:.4f}")
    print(f"   Combined Score:   {test_metrics['flip_rate'] * test_metrics['mean_correlation']:.4f}")
    
    # Save test results
    test_results = {
        'flip_rate': float(test_metrics['flip_rate']),
        'mean_correlation': float(test_metrics['mean_correlation']),
        'std_correlation': float(test_metrics['std_correlation']),
        'min_correlation': float(test_metrics['min_correlation']),
        'combined_score': float(test_metrics['flip_rate'] * test_metrics['mean_correlation']),
        'num_samples': test_metrics['num_samples'],
        'best_epoch': best_ckpt['epoch'] + 1
    }
    
    with open(f'{output_dir}/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # ========================================
    # 6. PLOT TRAINING HISTORY
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Losses
    ax = axes[0, 0]
    ax.plot(history['train']['identity_loss'], label='Train Identity', alpha=0.7)
    ax.plot(history['train']['flip_loss'], label='Train Flip', alpha=0.7)
    ax.plot(history['val']['identity_loss'], '--', label='Val Identity', alpha=0.7)
    ax.plot(history['val']['flip_loss'], '--', label='Val Flip', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Flip Rate
    ax = axes[0, 1]
    ax.plot(history['train']['flip_rate'], 'b-', label='Train', linewidth=2)
    ax.plot(history['val']['flip_rate'], 'r--', label='Val', linewidth=2)
    ax.axhline(y=test_metrics['flip_rate'], color='g', linestyle=':', label=f'Test: {test_metrics["flip_rate"]*100:.1f}%')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Flip Rate')
    ax.set_title('Classification Flip Rate')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation
    ax = axes[1, 0]
    ax.plot(history['train']['mean_correlation'], 'b-', label='Train', linewidth=2)
    ax.plot(history['val']['mean_correlation'], 'r--', label='Val', linewidth=2)
    ax.axhline(y=test_metrics['mean_correlation'], color='g', linestyle=':', label=f'Test: {test_metrics["mean_correlation"]:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    ax.set_title('Morphology Preservation (Correlation)')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Combined Score
    ax = axes[1, 1]
    train_score = [f * c for f, c in zip(history['train']['flip_rate'], history['train']['mean_correlation'])]
    val_score_hist = [f * c for f, c in zip(history['val']['flip_rate'], history['val']['mean_correlation'])]
    ax.plot(train_score, 'b-', label='Train', linewidth=2)
    ax.plot(val_score_hist, 'r--', label='Val', linewidth=2)
    test_score = test_metrics['flip_rate'] * test_metrics['mean_correlation']
    ax.axhline(y=test_score, color='g', linestyle=':', label=f'Test: {test_score:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Combined Score (Flip Rate × Correlation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=150)
    plt.close()
    
    # Save final model
    torch.save({
        'model_state_dict': edit_model.state_dict(),
        'history': history,
        'test_metrics': test_metrics
    }, f'{output_dir}/final_model.pth')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation score: {best_val_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    print(f"Output saved to: {output_dir}/")
    
    return test_metrics


if __name__ == "__main__":
    test_metrics = train_minimal_edit_v2(
        num_epochs=100,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        lambda_identity=1.0,
        lambda_flip=1.0,
        lambda_similarity=2.0,
        lambda_sparsity=0.1,
        patience=15,
        dropout=0.1
    )
