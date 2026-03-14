"""
Phase 3: Disentangled Counterfactual Generation
================================================

Key insight: The content encoder must NOT capture class information.
We use adversarial training to ensure content is class-invariant.

Architecture:
- Content Encoder: Extracts class-INVARIANT features
  - Trained with adversarial loss to REMOVE class information
- Style Encoder: Extracts class-DISCRIMINATIVE features
  - Trained with classification loss to CAPTURE class information
- Decoder: Reconstructs ECG from Content + Style

This ensures that swapping style actually changes the class.
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

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/disentangled_v1'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REAL_MEAN = -0.00396
REAL_STD = 0.14716

# ============================================================================
# Gradient Reversal Layer
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ============================================================================
# Encoders
# ============================================================================

class ContentEncoder(nn.Module):
    """
    Extracts class-INVARIANT content.
    Uses adversarial training to remove class information.
    """
    def __init__(self, in_channels=1, content_dim=256):
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
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(16),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 16, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, content_dim),
        )
        
        # Adversarial classifier (to remove class info)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.adversarial_classifier = nn.Sequential(
            nn.Linear(content_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x, return_adversarial=False):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        content = self.fc(features)
        
        if return_adversarial:
            # Apply gradient reversal for adversarial training
            reversed_content = self.grl(content)
            adv_logits = self.adversarial_classifier(reversed_content)
            return content, adv_logits
        
        return content


class StyleEncoder(nn.Module):
    """
    Extracts class-DISCRIMINATIVE style.
    Trained with classification loss.
    """
    def __init__(self, in_channels=1, style_dim=128, num_classes=2):
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
        
        # Strong classifier for style
        self.classifier = nn.Sequential(
            nn.Linear(style_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        style = self.fc(features)
        class_logits = self.classifier(style)
        return style, class_logits


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    def __init__(self, content_dim=256, style_dim=128, out_channels=1, seq_len=2500):
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
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
        )
        
        self.final = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, content, style):
        combined = torch.cat([content, style], dim=1)
        x = self.fc(combined)
        x = x.view(x.size(0), 512, 16)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        x = self.final(x)
        return x


# ============================================================================
# Complete Model
# ============================================================================

class DisentangledModel(nn.Module):
    def __init__(self, content_dim=256, style_dim=128, num_classes=2, seq_len=2500):
        super().__init__()
        
        self.content_encoder = ContentEncoder(content_dim=content_dim)
        self.style_encoder = StyleEncoder(style_dim=style_dim, num_classes=num_classes)
        self.decoder = Decoder(content_dim=content_dim, style_dim=style_dim, seq_len=seq_len)
        
    def forward(self, x, return_all=False):
        content, adv_logits = self.content_encoder(x, return_adversarial=True)
        style, class_logits = self.style_encoder(x)
        reconstructed = self.decoder(content, style)
        
        if return_all:
            return reconstructed, class_logits, adv_logits
        return reconstructed, class_logits
    
    def generate_counterfactual(self, x_original, x_style_source):
        content = self.content_encoder(x_original, return_adversarial=False)
        style, _ = self.style_encoder(x_style_source)
        counterfactual = self.decoder(content, style)
        return counterfactual


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=150, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_recon = 0
        total_style = 0
        total_adv = 0
        
        for signals, labels in train_loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            reconstructed, class_logits, adv_logits = model(signals, return_all=True)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, signals)
            
            # Style classification loss (should predict class)
            style_loss = F.cross_entropy(class_logits, labels)
            
            # Adversarial loss for content (should NOT predict class)
            # GRL already reverses gradients, so we maximize this
            adv_loss = F.cross_entropy(adv_logits, labels)
            
            # Combined loss
            # recon: want low, style: want low (good classification), adv: want HIGH (can't classify from content)
            loss = recon_loss + 1.0 * style_loss + 0.5 * adv_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_style += style_loss.item()
            total_adv += adv_loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_recon = 0
        style_correct = 0
        adv_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(DEVICE)
                labels = labels.to(DEVICE)
                
                reconstructed, class_logits, adv_logits = model(signals, return_all=True)
                val_recon += F.mse_loss(reconstructed, signals).item()
                
                style_correct += (class_logits.argmax(dim=1) == labels).sum().item()
                adv_correct += (adv_logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
        
        val_recon /= len(val_loader)
        style_acc = 100 * style_correct / val_total
        adv_acc = 100 * adv_correct / val_total  # Want this to be ~50% (random)
        
        if val_recon < best_val_loss:
            best_val_loss = val_recon
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_recon,
            }, RESULTS_DIR / 'best_model.pth')
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Recon={total_recon/len(train_loader):.6f}, "
                  f"Style={total_style/len(train_loader):.4f}, "
                  f"Adv={total_adv/len(train_loader):.4f}, "
                  f"Val_Recon={val_recon:.6f}, "
                  f"Style_Acc={style_acc:.1f}%, Adv_Acc={adv_acc:.1f}%")
    
    checkpoint = torch.load(RESULTS_DIR / 'best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    return model


# ============================================================================
# Classifier
# ============================================================================

def load_classifier():
    classifier_path = PROJECT_ROOT / 'models/afib_reslstm_final.pth'
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    return classifier

class ClassifierWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_normalized = (x - mean) / std
        logits, _ = self.model(x_normalized)
        return logits


# ============================================================================
# Visualization
# ============================================================================

def to_millivolts(signal):
    return signal * REAL_STD + REAL_MEAN

def analyze_rr_intervals(ecg_signal, fs=500):
    ecg = ecg_signal.flatten()
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    peaks, _ = scipy_signal.find_peaks(ecg_norm, height=0.5, distance=int(0.3 * fs))
    if len(peaks) < 2:
        return {'rr_irregularity': 0}
    rr_intervals = np.diff(peaks) / fs * 1000
    return {'rr_irregularity': np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0}


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("DISENTANGLED COUNTERFACTUAL GENERATION")
    print("(With Adversarial Training for Class-Invariant Content)")
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
    
    train_dataset = torch.utils.data.TensorDataset(train_signals, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_signals, val_labels)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create or load model
    model = DisentangledModel(content_dim=256, style_dim=128, num_classes=2, seq_len=2500).to(DEVICE)
    
    model_path = RESULTS_DIR / 'best_model.pth'
    if model_path.exists():
        print("\nLoading existing model...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint['epoch']+1}")
    else:
        print("\nTraining model with adversarial disentanglement...")
        model = train_model(model, train_loader, val_loader, epochs=150, lr=1e-4)
    
    # Load classifier
    print("\nLoading classifier...")
    raw_classifier = load_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    classifier.eval()
    
    with torch.no_grad():
        test_batch = val_signals[:200].to(DEVICE)
        test_labels_batch = val_labels[:200].to(DEVICE)
        preds = classifier(test_batch).argmax(dim=1)
        acc = (preds == test_labels_batch).float().mean().item() * 100
        print(f"  Classifier accuracy: {acc:.1f}%")
    
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    print(f"\nNormal samples: {len(normal_idx)}")
    print(f"AFib samples: {len(afib_idx)}")
    
    time_axis = np.arange(val_signals.shape[-1]) / 500
    
    # TEST 1: Reconstruction
    print("\n" + "="*70)
    print("TEST 1: PERFECT RECONSTRUCTION")
    print("="*70)
    
    model.eval()
    num_test = 10
    
    with torch.no_grad():
        test_idx = torch.cat([normal_idx[:5], afib_idx[:5]])
        test_signals = val_signals[test_idx].to(DEVICE)
        test_labels_actual = val_labels[test_idx]
        
        reconstructed, _ = model(test_signals)
        
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
        print(f"\n{'PASS' if mean_corr > 0.9 else 'FAIL'}: Correlation > 0.9")
    
    # Save reconstruction plot
    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    fig.suptitle(f'TEST 1: Reconstruction - Mean Corr={mean_corr:.3f}', fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(test_signals[i, 0].cpu().numpy())
        recon_mv = to_millivolts(reconstructed[i, 0].cpu().numpy())
        
        axes[i, 0].plot(time_axis, orig_mv, 'b-', linewidth=1.2, label='Original')
        axes[i, 0].plot(time_axis, recon_mv, 'g-', linewidth=1.2, alpha=0.7, label='Reconstructed')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'Sample {i+1} (Class={test_labels_actual[i].item()}): Corr={correlations[i]:.3f}')
        axes[i, 0].set_ylabel('mV')
        
        axes[i, 1].plot(time_axis[:1000], orig_mv[:1000], 'b-', linewidth=1.5, label='Original')
        axes[i, 1].plot(time_axis[:1000], recon_mv[:1000], 'g-', linewidth=1.5, alpha=0.7, label='Reconstructed')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title('Zoomed (0-2s)')
        axes[i, 1].set_ylabel('mV')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test1_reconstruction.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test1_reconstruction.png'}")
    
    # TEST 2: Normal → AFib
    print("\n" + "="*70)
    print("TEST 2: COUNTERFACTUAL (Normal → AFib)")
    print("="*70)
    
    num_cf = 20
    
    with torch.no_grad():
        normal_ecgs = val_signals[normal_idx[:num_cf]].to(DEVICE)
        afib_style_source = val_signals[afib_idx[:num_cf]].to(DEVICE)
        
        counterfactuals = model.generate_counterfactual(normal_ecgs, afib_style_source)
        
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
    
    # Save plot
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle('TEST 2: Normal→AFib Counterfactuals', fontsize=16, fontweight='bold')
    
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
    
    # TEST 3: AFib → Normal
    print("\n" + "="*70)
    print("TEST 3: COUNTERFACTUAL (AFib → Normal)")
    print("="*70)
    
    with torch.no_grad():
        afib_ecgs = val_signals[afib_idx[:num_cf]].to(DEVICE)
        normal_style_source = val_signals[normal_idx[:num_cf]].to(DEVICE)
        
        counterfactuals = model.generate_counterfactual(afib_ecgs, normal_style_source)
        
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
    
    # Save plot
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle('TEST 3: AFib→Normal Counterfactuals', fontsize=16, fontweight='bold')
    
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
    
    # Clinical Analysis
    print("\n" + "="*70)
    print("CLINICAL ANALYSIS: RR Intervals")
    print("="*70)
    
    print("\nNormal → AFib (expect: increased RR irregularity):")
    with torch.no_grad():
        normal_ecgs_cpu = val_signals[normal_idx[:5]].to(DEVICE)
        afib_style_cpu = val_signals[afib_idx[:5]].to(DEVICE)
        cf_samples = model.generate_counterfactual(normal_ecgs_cpu, afib_style_cpu)
    
    for i in range(5):
        orig_rr = analyze_rr_intervals(normal_ecgs_cpu[i, 0].cpu().numpy())
        cf_rr = analyze_rr_intervals(cf_samples[i, 0].cpu().numpy())
        change = cf_rr['rr_irregularity'] - orig_rr['rr_irregularity']
        print(f"  Sample {i+1}: {orig_rr['rr_irregularity']:.3f} → {cf_rr['rr_irregularity']:.3f} ({'+' if change > 0 else ''}{change:.3f})")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nReconstruction: Mean Corr = {mean_corr:.4f} ({'PASS' if mean_corr > 0.9 else 'FAIL'})")
    print(f"Normal→AFib Flip Rate: {100*flipped_to_afib/max(1,originally_normal):.1f}%")
    print(f"AFib→Normal Flip Rate: {100*flipped_to_normal/max(1,originally_afib):.1f}%")
    print(f"\nResults: {RESULTS_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
