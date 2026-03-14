"""
Phase 3: Classifier-Guided Minimal Perturbation Counterfactuals
================================================================

Hybrid approach:
1. High-quality VAE for perfect reconstruction
2. Classifier-guided latent perturbation to flip class
3. Clinical constraints (R-R intervals, P-waves)
4. Realism discriminator
"""

import os
import sys
import subprocess

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
import json
from datetime import datetime

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

# Configuration
PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/guided_v2'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REAL_MEAN = -0.00396
REAL_STD = 0.14716


# ============================================================================
# High-Quality VAE Encoder
# ============================================================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 15, 2, 7), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 11, 2, 5), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 7, 2, 3), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 5, 2, 2), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 3, 2, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc_mu = nn.Linear(512 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16, latent_dim)
        
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# ============================================================================
# High-Quality Decoder
# ============================================================================
class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 16), nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 4, 2, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, 4, 2, 1), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
        )
        self.final = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 3, 1, 1),
        )
        
    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 16)
        h = self.deconv(h)
        h = F.interpolate(h, size=2500, mode='linear', align_corners=False)
        return self.final(h)


# ============================================================================
# Discriminator for Realism
# ============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 15, 2, 7), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 11, 2, 5), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 7, 2, 3), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 5, 2, 2), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)


# ============================================================================
# Style Modifier Network
# ============================================================================
class StyleModifier(nn.Module):
    """Learns to modify latent code to flip class while minimizing change."""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim),
        )
        # Initialize to near-identity
        self.net[-1].weight.data *= 0.01
        self.net[-1].bias.data.zero_()
        
    def forward(self, z, target_class):
        # target_class: (B, 1) - 0 for normal, 1 for afib
        inp = torch.cat([z, target_class.float()], dim=1)
        delta = self.net(inp)
        return z + delta  # Residual modification


# ============================================================================
# Complete Model
# ============================================================================
class CounterfactualVAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.style_modifier = StyleModifier(latent_dim)
        self.latent_dim = latent_dim
        
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def generate_counterfactual(self, x, target_class):
        z, _, _ = self.encode(x)
        z_modified = self.style_modifier(z, target_class)
        return self.decode(z_modified), z, z_modified


# ============================================================================
# Clinical Feature Extraction
# ============================================================================
def detect_r_peaks(signal, fs=500):
    """Detect R-peaks in ECG signal."""
    signal = signal.flatten()
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    peaks, _ = scipy_signal.find_peaks(signal_norm, height=0.5, distance=int(0.3 * fs))
    return peaks

def compute_rr_features(signal, fs=500):
    """Compute R-R interval features."""
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return {'rr_mean': 0, 'rr_std': 0, 'rr_irregularity': 0}
    rr = np.diff(peaks) / fs * 1000  # ms
    return {
        'rr_mean': np.mean(rr),
        'rr_std': np.std(rr),
        'rr_irregularity': np.std(rr) / (np.mean(rr) + 1e-8)
    }

def check_signal_validity(signal, fs=500):
    """Check if signal is physiologically valid."""
    signal = signal.flatten()
    # Amplitude check
    amp_range = np.max(signal) - np.min(signal)
    if amp_range < 0.01 or amp_range > 10:
        return False, "Invalid amplitude"
    # R-peak detection
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return False, "Cannot detect R-peaks"
    # Heart rate check
    rr_mean = np.mean(np.diff(peaks)) / fs
    hr = 60 / rr_mean if rr_mean > 0 else 0
    if hr < 30 or hr > 200:
        return False, f"Invalid HR: {hr:.0f}"
    return True, "Valid"


# ============================================================================
# Classifier Wrapper
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
    """Wrapper that handles LSTM backward pass requirements."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Disable dropout for inference-like behavior during training
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        
    def forward(self, x):
        # Must be in training mode for LSTM backward pass
        self.model.train()
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        logits, _ = self.model(x_norm)
        return logits


# ============================================================================
# Training Function
# ============================================================================
def train_stage1(model, train_loader, val_loader, epochs=50):
    """Stage 1: Train VAE for perfect reconstruction."""
    print("\n" + "="*70)
    print("STAGE 1: Training VAE for Perfect Reconstruction")
    print("="*70)
    
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for signals, _ in train_loader:
            signals = signals.to(DEVICE)
            optimizer.zero_grad()
            
            recon, mu, logvar, _ = model(signals)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, signals)
            # KL divergence (small weight for better reconstruction)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.0001 * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        correlations = []
        
        with torch.no_grad():
            for signals, _ in val_loader:
                signals = signals.to(DEVICE)
                recon, mu, logvar, _ = model(signals)
                val_loss += F.mse_loss(recon, signals).item()
                
                for i in range(min(10, signals.size(0))):
                    corr, _ = pearsonr(
                        signals[i, 0].cpu().numpy(),
                        recon[i, 0].cpu().numpy()
                    )
                    correlations.append(corr)
        
        val_loss /= len(val_loader)
        mean_corr = np.mean(correlations)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_corr'].append(mean_corr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), RESULTS_DIR / 'stage1_best.pth')
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.6f}, "
                  f"Val={val_loss:.6f}, Corr={mean_corr:.4f}")
    
    model.load_state_dict(torch.load(RESULTS_DIR / 'stage1_best.pth'))
    print(f"\nStage 1 Complete. Best Correlation: {max(history['val_corr']):.4f}")
    return history


def train_stage2(model, discriminator, classifier, train_loader, val_loader, epochs=100):
    """Stage 2: Train style modifier with classifier guidance."""
    print("\n" + "="*70)
    print("STAGE 2: Classifier-Guided Style Modification")
    print("="*70)
    
    # Freeze encoder/decoder, train style modifier
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    
    optimizer_g = torch.optim.AdamW(model.style_modifier.parameters(), lr=5e-5)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
    
    history = {'flip_rate': [], 'similarity': [], 'd_loss': []}
    best_score = 0
    
    for epoch in range(epochs):
        model.train()
        discriminator.train()
        
        total_flip = 0
        total_sim = 0
        total_d_loss = 0
        n_batches = 0
        
        for signals, labels in train_loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)
            target_labels = 1 - labels  # Flip target
            
            # ---- Train Discriminator ----
            optimizer_d.zero_grad()
            
            with torch.no_grad():
                cf, _, _ = model.generate_counterfactual(
                    signals, target_labels.unsqueeze(1)
                )
            
            real_score = discriminator(signals)
            fake_score = discriminator(cf.detach())
            
            d_loss = F.binary_cross_entropy_with_logits(
                real_score, torch.ones_like(real_score)
            ) + F.binary_cross_entropy_with_logits(
                fake_score, torch.zeros_like(fake_score)
            )
            d_loss.backward()
            optimizer_d.step()
            
            # ---- Train Style Modifier ----
            optimizer_g.zero_grad()
            
            cf, z_orig, z_mod = model.generate_counterfactual(
                signals, target_labels.unsqueeze(1)
            )
            
            # Classifier flip loss
            cf_logits = classifier(cf)
            flip_loss = F.cross_entropy(cf_logits, target_labels)
            
            # Minimal change loss
            sim_loss = F.mse_loss(cf, signals)
            
            # Latent similarity
            latent_loss = F.mse_loss(z_mod, z_orig)
            
            # Realism loss
            fake_score = discriminator(cf)
            real_loss = F.binary_cross_entropy_with_logits(
                fake_score, torch.ones_like(fake_score)
            )
            
            # Combined loss
            loss = 2.0 * flip_loss + 1.0 * sim_loss + 0.1 * latent_loss + 0.5 * real_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.style_modifier.parameters(), 1.0)
            optimizer_g.step()
            
            # Metrics
            cf_preds = cf_logits.argmax(dim=1)
            flip_rate = (cf_preds == target_labels).float().mean().item()
            similarity = 1 - sim_loss.item()
            
            total_flip += flip_rate
            total_sim += similarity
            total_d_loss += d_loss.item()
            n_batches += 1
        
        avg_flip = total_flip / n_batches
        avg_sim = total_sim / n_batches
        
        history['flip_rate'].append(avg_flip)
        history['similarity'].append(avg_sim)
        history['d_loss'].append(total_d_loss / n_batches)
        
        # Combined score
        score = avg_flip * 0.6 + avg_sim * 0.4
        if score > best_score:
            best_score = score
            torch.save({
                'model': model.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, RESULTS_DIR / 'stage2_best.pth')
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: Flip={avg_flip*100:.1f}%, "
                  f"Sim={avg_sim:.4f}, D_loss={total_d_loss/n_batches:.4f}")
    
    checkpoint = torch.load(RESULTS_DIR / 'stage2_best.pth')
    model.load_state_dict(checkpoint['model'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    
    print(f"\nStage 2 Complete. Best Flip Rate: {max(history['flip_rate'])*100:.1f}%")
    return history


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_counterfactuals(model, classifier, val_signals, val_labels):
    """Comprehensive evaluation of counterfactual quality."""
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    model.eval()
    results = {
        'reconstruction': {},
        'normal_to_afib': {},
        'afib_to_normal': {},
    }
    
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    # Test reconstruction
    print("\n--- Reconstruction Test ---")
    with torch.no_grad():
        test_signals = val_signals[:100].to(DEVICE)
        recon, _, _, _ = model(test_signals)
        
        mse = F.mse_loss(recon, test_signals).item()
        correlations = []
        for i in range(100):
            corr, _ = pearsonr(
                test_signals[i, 0].cpu().numpy(),
                recon[i, 0].cpu().numpy()
            )
            correlations.append(corr)
        
        results['reconstruction']['mse'] = float(mse)
        results['reconstruction']['mean_corr'] = float(np.mean(correlations))
        print(f"MSE: {mse:.6f}, Mean Corr: {np.mean(correlations):.4f}")
    
    # Test Normal -> AFib
    print("\n--- Normal → AFib ---")
    with torch.no_grad():
        normal_ecgs = val_signals[normal_idx[:50]].to(DEVICE)
        target = torch.ones(50, 1).to(DEVICE)
        
        cf, _, _ = model.generate_counterfactual(normal_ecgs, target)
        
        orig_preds = classifier(normal_ecgs).argmax(dim=1)
        cf_preds = classifier(cf).argmax(dim=1)
        
        originally_normal = (orig_preds == 0).sum().item()
        flipped = ((orig_preds == 0) & (cf_preds == 1)).sum().item()
        
        # Similarity
        similarity = []
        for i in range(50):
            corr, _ = pearsonr(
                normal_ecgs[i, 0].cpu().numpy(),
                cf[i, 0].cpu().numpy()
            )
            similarity.append(corr)
        
        results['normal_to_afib']['flip_rate'] = float(flipped / max(1, originally_normal))
        results['normal_to_afib']['mean_similarity'] = float(np.mean(similarity))
        print(f"Flip Rate: {flipped}/{originally_normal} ({100*flipped/max(1,originally_normal):.1f}%)")
        print(f"Mean Similarity: {np.mean(similarity):.4f}")
    
    # Test AFib -> Normal
    print("\n--- AFib → Normal ---")
    with torch.no_grad():
        afib_ecgs = val_signals[afib_idx[:50]].to(DEVICE)
        target = torch.zeros(50, 1).to(DEVICE)
        
        cf, _, _ = model.generate_counterfactual(afib_ecgs, target)
        
        orig_preds = classifier(afib_ecgs).argmax(dim=1)
        cf_preds = classifier(cf).argmax(dim=1)
        
        originally_afib = (orig_preds == 1).sum().item()
        flipped = ((orig_preds == 1) & (cf_preds == 0)).sum().item()
        
        similarity = []
        for i in range(50):
            corr, _ = pearsonr(
                afib_ecgs[i, 0].cpu().numpy(),
                cf[i, 0].cpu().numpy()
            )
            similarity.append(corr)
        
        results['afib_to_normal']['flip_rate'] = float(flipped / max(1, originally_afib))
        results['afib_to_normal']['mean_similarity'] = float(np.mean(similarity))
        print(f"Flip Rate: {flipped}/{originally_afib} ({100*flipped/max(1,originally_afib):.1f}%)")
        print(f"Mean Similarity: {np.mean(similarity):.4f}")
    
    return results


def create_visualizations(model, classifier, val_signals, val_labels):
    """Create visualization plots."""
    print("\n--- Creating Visualizations ---")
    
    model.eval()
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    time_axis = np.arange(2500) / 500
    
    # Normal -> AFib visualization
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle('Normal → AFib Counterfactuals', fontsize=16, fontweight='bold')
    
    with torch.no_grad():
        normal_ecgs = val_signals[normal_idx[:5]].to(DEVICE)
        target = torch.ones(5, 1).to(DEVICE)
        cf, _, _ = model.generate_counterfactual(normal_ecgs, target)
        
        orig_probs = F.softmax(classifier(normal_ecgs), dim=1)
        cf_probs = F.softmax(classifier(cf), dim=1)
    
    for i in range(5):
        orig = normal_ecgs[i, 0].cpu().numpy() * REAL_STD + REAL_MEAN
        counterfactual = cf[i, 0].cpu().numpy() * REAL_STD + REAL_MEAN
        diff = counterfactual - orig
        
        axes[i, 0].plot(time_axis, orig, 'b-', lw=1)
        axes[i, 0].set_title(f'Original Normal (P(AFib)={orig_probs[i,1]:.2f})')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_axis, counterfactual, 'r-', lw=1)
        axes[i, 1].set_title(f'Counterfactual (P(AFib)={cf_probs[i,1]:.2f})')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(time_axis, orig, 'b-', lw=1, alpha=0.7, label='Original')
        axes[i, 2].plot(time_axis, counterfactual, 'r-', lw=1, alpha=0.7, label='CF')
        axes[i, 2].fill_between(time_axis, orig, counterfactual, alpha=0.3)
        axes[i, 2].legend()
        axes[i, 2].set_title('Overlay + Difference')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'normal_to_afib_cf.png', dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'normal_to_afib_cf.png'}")
    
    # AFib -> Normal visualization
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle('AFib → Normal Counterfactuals', fontsize=16, fontweight='bold')
    
    with torch.no_grad():
        afib_ecgs = val_signals[afib_idx[:5]].to(DEVICE)
        target = torch.zeros(5, 1).to(DEVICE)
        cf, _, _ = model.generate_counterfactual(afib_ecgs, target)
        
        orig_probs = F.softmax(classifier(afib_ecgs), dim=1)
        cf_probs = F.softmax(classifier(cf), dim=1)
    
    for i in range(5):
        orig = afib_ecgs[i, 0].cpu().numpy() * REAL_STD + REAL_MEAN
        counterfactual = cf[i, 0].cpu().numpy() * REAL_STD + REAL_MEAN
        
        axes[i, 0].plot(time_axis, orig, 'r-', lw=1)
        axes[i, 0].set_title(f'Original AFib (P(Normal)={orig_probs[i,0]:.2f})')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_axis, counterfactual, 'b-', lw=1)
        axes[i, 1].set_title(f'Counterfactual (P(Normal)={cf_probs[i,0]:.2f})')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(time_axis, orig, 'r-', lw=1, alpha=0.7, label='Original')
        axes[i, 2].plot(time_axis, counterfactual, 'b-', lw=1, alpha=0.7, label='CF')
        axes[i, 2].fill_between(time_axis, orig, counterfactual, alpha=0.3)
        axes[i, 2].legend()
        axes[i, 2].set_title('Overlay + Difference')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'afib_to_normal_cf.png', dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'afib_to_normal_cf.png'}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("\n" + "="*70)
    print("CLASSIFIER-GUIDED COUNTERFACTUAL GENERATION V2")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    # Create models
    model = CounterfactualVAE(latent_dim=512).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Load classifier
    print("\nLoading classifier...")
    raw_classifier = load_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    classifier.eval()
    
    # Check for existing checkpoints
    stage2_path = RESULTS_DIR / 'stage2_best.pth'
    stage1_path = RESULTS_DIR / 'stage1_best.pth'
    
    if stage2_path.exists():
        print("\nLoading Stage 2 checkpoint...")
        checkpoint = torch.load(stage2_path)
        model.load_state_dict(checkpoint['model'])
        discriminator.load_state_dict(checkpoint['discriminator'])
    elif stage1_path.exists():
        print("\nLoading Stage 1 checkpoint, continuing to Stage 2...")
        model.load_state_dict(torch.load(stage1_path))
        train_stage2(model, discriminator, classifier, train_loader, val_loader, epochs=100)
    else:
        # Full training
        train_stage1(model, train_loader, val_loader, epochs=50)
        train_stage2(model, discriminator, classifier, train_loader, val_loader, epochs=100)
    
    # Evaluation
    results = evaluate_counterfactuals(model, classifier, val_signals, val_labels)
    
    # Save results
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualizations
    create_visualizations(model, classifier, val_signals, val_labels)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Reconstruction Correlation: {results['reconstruction']['mean_corr']:.4f}")
    print(f"Normal→AFib Flip Rate: {results['normal_to_afib']['flip_rate']*100:.1f}%")
    print(f"AFib→Normal Flip Rate: {results['afib_to_normal']['flip_rate']*100:.1f}%")
    print(f"Normal→AFib Similarity: {results['normal_to_afib']['mean_similarity']:.4f}")
    print(f"AFib→Normal Similarity: {results['afib_to_normal']['mean_similarity']:.4f}")
    print("="*70)
    
    # Check if goals met
    recon_ok = results['reconstruction']['mean_corr'] > 0.9
    flip_ok = (results['normal_to_afib']['flip_rate'] > 0.5 and 
               results['afib_to_normal']['flip_rate'] > 0.5)
    sim_ok = (results['normal_to_afib']['mean_similarity'] > 0.7 and
              results['afib_to_normal']['mean_similarity'] > 0.7)
    
    print(f"\nGoals Met:")
    print(f"  Reconstruction > 0.9: {'✓' if recon_ok else '✗'}")
    print(f"  Flip Rate > 50%: {'✓' if flip_ok else '✗'}")
    print(f"  Similarity > 0.7: {'✓' if sim_ok else '✗'}")
    
    if not (recon_ok and flip_ok and sim_ok):
        print("\n⚠️  Some goals not met. Consider:")
        print("  - Adjusting loss weights")
        print("  - Training for more epochs")
        print("  - Trying different architectures")


if __name__ == '__main__':
    main()
