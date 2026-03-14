"""
Phase 3: Content-Style Counterfactual Generation
=================================================

Proper implementation with Content and Style Encoders:
- Content Encoder: Extracts class-invariant features (ECG morphology, timing)
- Style Encoder: Extracts class-discriminative features (AFib vs Normal patterns)
- Decoder: Reconstructs ECG from Content + Style

Key Requirements:
1. Content(Original) + Style(Original) → Perfect Reconstruction
2. Content(Original) + Style(Other Class) → Counterfactual that flips classifier

This uses a VAE-style architecture (not diffusion) for more reliable reconstruction.
"""

import os
import sys
import subprocess

# Auto-select GPU
def get_free_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = [(int(l.split(',')[0]), int(l.split(',')[1])) for l in lines if l.strip()]
        if gpu_info:
            best_gpu = max(gpu_info, key=lambda x: x[1])
            return str(best_gpu[0])
    except:
        pass
    return '0'

os.environ['CUDA_VISIBLE_DEVICES'] = get_free_gpu()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import signal as scipy_signal
from tqdm import tqdm

# Add models path
sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/content_style_v2'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Data normalization constants
REAL_MEAN = -0.00396
REAL_STD = 0.14716

# ============================================================================
# Content Encoder - Extracts class-invariant features
# ============================================================================

class ContentEncoder(nn.Module):
    """
    Extracts class-invariant content from ECG.
    Uses convolutional layers to capture morphological features.
    """
    def __init__(self, in_channels=1, content_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Input: (B, 1, 2500)
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(16),  # Fixed size output
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 16, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, content_dim),
        )
        
        self.content_dim = content_dim
        
    def forward(self, x):
        # x: (B, 1, 2500)
        features = self.encoder(x)  # (B, 512, 16)
        features = features.view(features.size(0), -1)  # (B, 512*16)
        content = self.fc(features)  # (B, content_dim)
        return content


# ============================================================================
# Style Encoder - Extracts class-discriminative features
# ============================================================================

class StyleEncoder(nn.Module):
    """
    Extracts class-discriminative style from ECG.
    Includes a classifier head to ensure style captures class information.
    """
    def __init__(self, in_channels=1, style_dim=64, num_classes=2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(8),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, style_dim),
        )
        
        # Classifier head for style supervision
        self.classifier = nn.Linear(style_dim, num_classes)
        
        self.style_dim = style_dim
        
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        style = self.fc(features)
        class_logits = self.classifier(style)
        return style, class_logits


# ============================================================================
# Decoder - Reconstructs ECG from Content + Style
# ============================================================================

class Decoder(nn.Module):
    """
    Reconstructs ECG from content and style embeddings.
    Uses transposed convolutions for upsampling.
    """
    def __init__(self, content_dim=256, style_dim=64, out_channels=1, seq_len=2500):
        super().__init__()
        
        self.seq_len = seq_len
        combined_dim = content_dim + style_dim
        
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512 * 16),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder = nn.Sequential(
            # Input: (B, 512, 16)
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),  # -> 32
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 64
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 128
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 256
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 512
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
        )
        
        # Final upsampling to target length
        self.final = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, content, style):
        # Concatenate content and style
        combined = torch.cat([content, style], dim=1)  # (B, content_dim + style_dim)
        
        # Project to initial feature map
        x = self.fc(combined)  # (B, 512*16)
        x = x.view(x.size(0), 512, 16)  # (B, 512, 16)
        
        # Decode
        x = self.decoder(x)  # (B, 16, 512)
        
        # Upsample to target length
        x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        x = self.final(x)  # (B, 1, 2500)
        
        return x


# ============================================================================
# Complete Content-Style Model
# ============================================================================

class ContentStyleModel(nn.Module):
    """
    Complete model for counterfactual generation.
    """
    def __init__(self, content_dim=256, style_dim=64, num_classes=2, seq_len=2500):
        super().__init__()
        
        self.content_encoder = ContentEncoder(content_dim=content_dim)
        self.style_encoder = StyleEncoder(style_dim=style_dim, num_classes=num_classes)
        self.decoder = Decoder(content_dim=content_dim, style_dim=style_dim, seq_len=seq_len)
        
    def forward(self, x):
        """Forward pass for reconstruction."""
        content = self.content_encoder(x)
        style, class_logits = self.style_encoder(x)
        reconstructed = self.decoder(content, style)
        return reconstructed, class_logits
    
    def encode(self, x):
        """Encode ECG to content and style."""
        content = self.content_encoder(x)
        style, class_logits = self.style_encoder(x)
        return content, style, class_logits
    
    def decode(self, content, style):
        """Decode from content and style."""
        return self.decoder(content, style)
    
    def generate_counterfactual(self, x_original, x_style_source):
        """
        Generate counterfactual by combining:
        - Content from original ECG
        - Style from style_source ECG (different class)
        """
        content = self.content_encoder(x_original)
        style, _ = self.style_encoder(x_style_source)
        counterfactual = self.decoder(content, style)
        return counterfactual


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-4):
    """Train the content-style model."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_recon_loss = 0
        total_style_loss = 0
        total_loss = 0
        
        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, class_logits = model(signals)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, signals)
            
            # Style classification loss
            style_loss = F.cross_entropy(class_logits, labels)
            
            # Combined loss - balance reconstruction and style classification
            # Higher style weight to make style more class-discriminative
            loss = recon_loss + 1.0 * style_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_style_loss += style_loss.item()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_recon_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(DEVICE)
                labels = labels.to(DEVICE)
                
                reconstructed, class_logits = model(signals)
                val_recon_loss += F.mse_loss(reconstructed, signals).item()
                
                preds = class_logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_recon_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Save best model
        if val_recon_loss < best_val_loss:
            best_val_loss = val_recon_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_recon_loss,
            }, RESULTS_DIR / 'best_model.pth')
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Recon={total_recon_loss/len(train_loader):.6f}, "
                  f"Style={total_style_loss/len(train_loader):.4f}, "
                  f"Val_Recon={val_recon_loss:.6f}, Val_Acc={val_acc:.1f}%")
    
    # Load best model
    checkpoint = torch.load(RESULTS_DIR / 'best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    return model


# ============================================================================
# Classifier for Validation
# ============================================================================

def load_classifier():
    """Load the pre-trained AFibResLSTM classifier."""
    classifier_path = PROJECT_ROOT / 'models/afib_reslstm_final.pth'
    
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    return classifier


class ClassifierWrapper(nn.Module):
    """Wrapper with per-sample normalization."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Per-sample z-score normalization
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_normalized = (x - mean) / std
        logits, _ = self.model(x_normalized)
        return logits


# ============================================================================
# Visualization
# ============================================================================

def to_millivolts(normalized_signal):
    return normalized_signal * REAL_STD + REAL_MEAN


def analyze_rr_intervals(ecg_signal, fs=500):
    ecg = ecg_signal.flatten()
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    peaks, _ = scipy_signal.find_peaks(ecg_norm, height=0.5, distance=int(0.3 * fs))
    
    if len(peaks) < 2:
        return {'rr_irregularity': 0, 'num_beats': 0}
    
    rr_intervals = np.diff(peaks) / fs * 1000
    return {
        'rr_irregularity': np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0,
        'num_beats': len(peaks)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CONTENT-STYLE COUNTERFACTUAL GENERATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_data = np.load(DATA_DIR / 'train_data.npz')
    val_data = np.load(DATA_DIR / 'val_data.npz')
    
    train_signals = torch.tensor(train_data['X'], dtype=torch.float32)
    train_labels = torch.tensor(train_data['y'], dtype=torch.long)
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    
    if train_signals.dim() == 2:
        train_signals = train_signals.unsqueeze(1)
        val_signals = val_signals.unsqueeze(1)
    
    print(f"  Train: {train_signals.shape}")
    print(f"  Val: {val_signals.shape}")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_signals, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_signals, val_labels)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create or load model
    model = ContentStyleModel(content_dim=256, style_dim=64, num_classes=2, seq_len=2500).to(DEVICE)
    
    model_path = RESULTS_DIR / 'best_model.pth'
    if model_path.exists():
        print("\nLoading existing model...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint['epoch']+1}")
    else:
        print("\nTraining model...")
        model = train_model(model, train_loader, val_loader, epochs=100, lr=1e-4)
    
    # Load classifier
    print("\nLoading classifier...")
    raw_classifier = load_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    classifier.eval()
    
    # Validate classifier
    with torch.no_grad():
        test_batch = val_signals[:200].to(DEVICE)
        test_labels_batch = val_labels[:200].to(DEVICE)
        preds = classifier(test_batch).argmax(dim=1)
        acc = (preds == test_labels_batch).float().mean().item() * 100
        print(f"  Classifier accuracy: {acc:.1f}%")
    
    # Separate by class
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    print(f"\nNormal samples: {len(normal_idx)}")
    print(f"AFib samples: {len(afib_idx)}")
    
    time_axis = np.arange(val_signals.shape[-1]) / 500
    
    # ========================================================================
    # TEST 1: Perfect Reconstruction
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: PERFECT RECONSTRUCTION")
    print("Content(Original) + Style(Original) → Should match Original")
    print("="*70)
    
    model.eval()
    num_test = 10
    
    with torch.no_grad():
        test_idx = torch.cat([normal_idx[:5], afib_idx[:5]])
        test_signals = val_signals[test_idx].to(DEVICE)
        test_labels_actual = val_labels[test_idx]
        
        # Reconstruct using same content and style
        reconstructed, _ = model(test_signals)
        
        # Metrics
        mse = F.mse_loss(reconstructed, test_signals).item()
        
        correlations = []
        for i in range(num_test):
            orig = test_signals[i, 0].cpu().numpy()
            recon = reconstructed[i, 0].cpu().numpy()
            corr, _ = pearsonr(orig, recon)
            correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        
        print(f"\nMSE: {mse:.6f}")
        print(f"Mean Correlation: {mean_corr:.4f}")
        print(f"Correlations: {[f'{c:.3f}' for c in correlations]}")
        
        recon_pass = mean_corr > 0.9
        print(f"\n{'PASS' if recon_pass else 'FAIL'}: Correlation > 0.9")
    
    # Visualization
    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    fig.suptitle(f'TEST 1: Reconstruction - Mean Corr={mean_corr:.3f}', fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(test_signals[i, 0].cpu().numpy())
        recon_mv = to_millivolts(reconstructed[i, 0].cpu().numpy())
        
        axes[i, 0].plot(time_axis, orig_mv, 'b-', linewidth=1.2, alpha=0.8, label='Original')
        axes[i, 0].plot(time_axis, recon_mv, 'g-', linewidth=1.2, alpha=0.7, label='Reconstructed')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'Sample {i+1} (Class={test_labels_actual[i].item()}): Corr={correlations[i]:.3f}')
        
        # Zoomed
        axes[i, 1].plot(time_axis[:1000], orig_mv[:1000], 'b-', linewidth=1.5, alpha=0.8, label='Original')
        axes[i, 1].plot(time_axis[:1000], recon_mv[:1000], 'g-', linewidth=1.5, alpha=0.7, label='Reconstructed')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title('Zoomed (0-2s)')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test1_reconstruction.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test1_reconstruction.png'}")
    
    # ========================================================================
    # TEST 2: Counterfactual Normal → AFib
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: COUNTERFACTUAL (Normal → AFib)")
    print("Content(Normal) + Style(AFib sample) → Should flip to AFib")
    print("="*70)
    
    num_cf = 20
    
    with torch.no_grad():
        # Get normal ECGs
        normal_ecgs = val_signals[normal_idx[:num_cf]].to(DEVICE)
        
        # Get AFib ECGs to use as style source
        afib_style_source = val_signals[afib_idx[:num_cf]].to(DEVICE)
        
        # Generate counterfactuals
        counterfactuals = model.generate_counterfactual(normal_ecgs, afib_style_source)
        
        # Classify
        orig_logits = classifier(normal_ecgs)
        cf_logits = classifier(counterfactuals)
        
        orig_preds = orig_logits.argmax(dim=1)
        cf_preds = cf_logits.argmax(dim=1)
        
        orig_probs = F.softmax(orig_logits, dim=1)
        cf_probs = F.softmax(cf_logits, dim=1)
    
    originally_normal = (orig_preds == 0).sum().item()
    flipped_to_afib = ((orig_preds == 0) & (cf_preds == 1)).sum().item()
    
    print(f"\nOriginal predictions: {orig_preds.cpu().numpy()}")
    print(f"Counterfactual predictions: {cf_preds.cpu().numpy()}")
    print(f"\nOriginally Normal: {originally_normal}/{num_cf}")
    print(f"Flipped to AFib: {flipped_to_afib}/{max(1,originally_normal)} ({100*flipped_to_afib/max(1,originally_normal):.1f}%)")
    print(f"\nMean AFib prob: Original={orig_probs[:,1].mean():.3f}, CF={cf_probs[:,1].mean():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle('TEST 2: Normal→AFib Counterfactuals - mV', fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(normal_ecgs[i, 0].cpu().numpy())
        cf_mv = to_millivolts(counterfactuals[i, 0].cpu().numpy())
        
        axes[i, 0].plot(time_axis, orig_mv, 'b-', linewidth=1)
        axes[i, 0].set_title(f'Original Normal (P(AFib)={orig_probs[i,1]:.2f})')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_axis, cf_mv, 'r-', linewidth=1)
        axes[i, 1].set_title(f'Counterfactual (P(AFib)={cf_probs[i,1]:.2f})')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(time_axis, orig_mv, 'b-', linewidth=1, alpha=0.7, label='Original')
        axes[i, 2].plot(time_axis, cf_mv, 'r-', linewidth=1, alpha=0.7, label='Counterfactual')
        axes[i, 2].fill_between(time_axis, orig_mv, cf_mv, alpha=0.3, color='purple')
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_title(f'Overlay ({orig_preds[i].item()}→{cf_preds[i].item()})')
        axes[i, 2].set_ylabel('mV')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test2_normal_to_afib.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test2_normal_to_afib.png'}")
    
    # ========================================================================
    # TEST 3: Counterfactual AFib → Normal
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: COUNTERFACTUAL (AFib → Normal)")
    print("Content(AFib) + Style(Normal sample) → Should flip to Normal")
    print("="*70)
    
    with torch.no_grad():
        # Get AFib ECGs
        afib_ecgs = val_signals[afib_idx[:num_cf]].to(DEVICE)
        
        # Get Normal ECGs to use as style source
        normal_style_source = val_signals[normal_idx[:num_cf]].to(DEVICE)
        
        # Generate counterfactuals
        counterfactuals = model.generate_counterfactual(afib_ecgs, normal_style_source)
        
        # Classify
        orig_logits = classifier(afib_ecgs)
        cf_logits = classifier(counterfactuals)
        
        orig_preds = orig_logits.argmax(dim=1)
        cf_preds = cf_logits.argmax(dim=1)
        
        orig_probs = F.softmax(orig_logits, dim=1)
        cf_probs = F.softmax(cf_logits, dim=1)
    
    originally_afib = (orig_preds == 1).sum().item()
    flipped_to_normal = ((orig_preds == 1) & (cf_preds == 0)).sum().item()
    
    print(f"\nOriginal predictions: {orig_preds.cpu().numpy()}")
    print(f"Counterfactual predictions: {cf_preds.cpu().numpy()}")
    print(f"\nOriginally AFib: {originally_afib}/{num_cf}")
    print(f"Flipped to Normal: {flipped_to_normal}/{max(1,originally_afib)} ({100*flipped_to_normal/max(1,originally_afib):.1f}%)")
    print(f"\nMean Normal prob: Original={orig_probs[:,0].mean():.3f}, CF={cf_probs[:,0].mean():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle('TEST 3: AFib→Normal Counterfactuals - mV', fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(afib_ecgs[i, 0].cpu().numpy())
        cf_mv = to_millivolts(counterfactuals[i, 0].cpu().numpy())
        
        axes[i, 0].plot(time_axis, orig_mv, 'r-', linewidth=1)
        axes[i, 0].set_title(f'Original AFib (P(Normal)={orig_probs[i,0]:.2f})')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_axis, cf_mv, 'b-', linewidth=1)
        axes[i, 1].set_title(f'Counterfactual (P(Normal)={cf_probs[i,0]:.2f})')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(time_axis, orig_mv, 'r-', linewidth=1, alpha=0.7, label='Original AFib')
        axes[i, 2].plot(time_axis, cf_mv, 'b-', linewidth=1, alpha=0.7, label='Counterfactual')
        axes[i, 2].fill_between(time_axis, orig_mv, cf_mv, alpha=0.3, color='purple')
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_title(f'Overlay ({orig_preds[i].item()}→{cf_preds[i].item()})')
        axes[i, 2].set_ylabel('mV')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test3_afib_to_normal.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test3_afib_to_normal.png'}")
    
    # ========================================================================
    # Clinical Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("CLINICAL ANALYSIS: RR Intervals")
    print("="*70)
    
    print("\nNormal → AFib (expect: increased RR irregularity):")
    normal_ecgs_cpu = val_signals[normal_idx[:5]].to(DEVICE)
    afib_style_cpu = val_signals[afib_idx[:5]].to(DEVICE)
    
    with torch.no_grad():
        cf_samples = model.generate_counterfactual(normal_ecgs_cpu, afib_style_cpu)
    
    for i in range(5):
        orig_rr = analyze_rr_intervals(normal_ecgs_cpu[i, 0].cpu().numpy())
        cf_rr = analyze_rr_intervals(cf_samples[i, 0].cpu().numpy())
        change = cf_rr['rr_irregularity'] - orig_rr['rr_irregularity']
        print(f"  Sample {i+1}: {orig_rr['rr_irregularity']:.3f} → {cf_rr['rr_irregularity']:.3f} ({'+' if change > 0 else ''}{change:.3f})")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nReconstruction: Mean Corr = {mean_corr:.4f} ({'PASS' if recon_pass else 'FAIL'})")
    print(f"Normal→AFib Flip Rate: {100*flipped_to_afib/max(1,originally_normal):.1f}%")
    print(f"AFib→Normal Flip Rate: {100*flipped_to_normal/max(1,originally_afib):.1f}%")
    print(f"\nResults: {RESULTS_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
