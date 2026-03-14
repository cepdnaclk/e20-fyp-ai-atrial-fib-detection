"""
Phase 3: Clinical Evaluation with Millivolt Visualization
==========================================================

This script:
1. Tests PERFECT reconstruction (Content + Style same ECG)
2. Tests counterfactual with CLASSIFIER FLIP verification
3. Visualizes in CLINICAL RANGE (millivolts, not normalized)
4. Analyzes clinical features (P-wave, RR intervals)
"""

import os
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
from scipy import signal
from scipy.stats import pearsonr
import math

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual'
RESULTS_DIR = MODEL_DIR / 'clinical_evaluation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Clinical conversion constants (from Phase 2)
REAL_MEAN = -0.00396
REAL_STD = 0.14716
ECG_SCALE = 1.0  # ADC to mV scale factor (adjust based on your data)

# ============================================================================
# Conversion Functions
# ============================================================================

def to_millivolts(normalized_signal):
    """Convert normalized signal to millivolts."""
    # Denormalize: signal * std + mean
    denorm = normalized_signal * REAL_STD + REAL_MEAN
    # Scale to millivolts (typical ECG range: -1 to +2 mV)
    mv = denorm * ECG_SCALE
    return mv

def denormalize(signal):
    """Denormalize signal."""
    return signal * REAL_STD + REAL_MEAN

# ============================================================================
# Model Definitions
# ============================================================================

def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, content_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 8, hidden_dim * 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8),
        )
        self.flat_size = hidden_dim * 8 * 8
        self.fc_mu = nn.Linear(self.flat_size, content_dim)
        self.fc_logvar = nn.Linear(self.flat_size, content_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        return z, mu, logvar

class StyleEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, style_dim=128, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, stride=2, padding=3),
            nn.InstanceNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_style = nn.Linear(hidden_dim * 4, style_dim)
        self.classifier = nn.Linear(style_dim, num_classes)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        style = self.fc_style(h)
        class_logits = self.classifier(style)
        return style, class_logits

class ConditionalResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)

class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhcl,bhck->bhlk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhlk,bhck->bhcl', attn, v).reshape(B, C, L)
        return x + self.proj(out)

class ConditionalUNet(nn.Module):
    def __init__(self, in_ch=1, model_ch=64, content_dim=256, style_dim=128):
        super().__init__()
        time_dim = model_ch * 4
        cond_dim = time_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.content_proj = nn.Linear(content_dim, time_dim)
        self.style_proj = nn.Linear(style_dim, time_dim)
        
        self.input_conv = nn.Conv1d(in_ch, model_ch, 3, padding=1)
        
        self.down1 = ConditionalResBlock(model_ch, model_ch, cond_dim)
        self.down2 = ConditionalResBlock(model_ch, model_ch * 2, cond_dim)
        self.down3 = ConditionalResBlock(model_ch * 2, model_ch * 4, cond_dim)
        self.down4 = ConditionalResBlock(model_ch * 4, model_ch * 8, cond_dim)
        
        self.downsample1 = nn.Conv1d(model_ch, model_ch, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv1d(model_ch * 2, model_ch * 2, 3, stride=2, padding=1)
        self.downsample3 = nn.Conv1d(model_ch * 4, model_ch * 4, 3, stride=2, padding=1)
        self.downsample4 = nn.Conv1d(model_ch * 8, model_ch * 8, 3, stride=2, padding=1)
        
        self.mid1 = ConditionalResBlock(model_ch * 8, model_ch * 8, cond_dim)
        self.mid_attn = SelfAttention1D(model_ch * 8)
        self.mid2 = ConditionalResBlock(model_ch * 8, model_ch * 8, cond_dim)
        
        self.up4 = ConditionalResBlock(model_ch * 16, model_ch * 8, cond_dim)
        self.up3 = ConditionalResBlock(model_ch * 8, model_ch * 4, cond_dim)   # 4+4=8 -> 4
        self.up2 = ConditionalResBlock(model_ch * 4, model_ch * 2, cond_dim)   # 2+2=4 -> 2
        self.up1 = ConditionalResBlock(model_ch * 2, model_ch, cond_dim)       # 1+1=2 -> 1
        
        # Upsample layers take output from previous up block
        self.upsample4 = nn.ConvTranspose1d(model_ch * 8, model_ch * 8, 4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose1d(model_ch * 8, model_ch * 4, 4, stride=2, padding=1)  # up4 outputs 8*ch
        self.upsample2 = nn.ConvTranspose1d(model_ch * 4, model_ch * 2, 4, stride=2, padding=1)  # up3 outputs 4*ch
        self.upsample1 = nn.ConvTranspose1d(model_ch * 2, model_ch, 4, stride=2, padding=1)      # up2 outputs 2*ch
        
        self.out_norm = nn.GroupNorm(32, model_ch)
        self.out_conv = nn.Conv1d(model_ch, in_ch, 3, padding=1)
        
    def _match_size(self, x, target):
        if x.size(-1) != target.size(-1):
            diff = target.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[:, :, :target.size(-1)]
        return x
        
    def forward(self, x, t, content, style):
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        c_emb = self.content_proj(content)
        s_emb = self.style_proj(style)
        cond = t_emb + c_emb + s_emb
        
        h = self.input_conv(x)
        h1 = self.down1(h, cond)
        h = self.downsample1(h1)
        h2 = self.down2(h, cond)
        h = self.downsample2(h2)
        h3 = self.down3(h, cond)
        h = self.downsample3(h3)
        h4 = self.down4(h, cond)
        h = self.downsample4(h4)
        
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        
        h = self.upsample4(h)
        h = self._match_size(h, h4)
        h = self.up4(torch.cat([h, h4], dim=1), cond)  # 8+8=16 -> 8
        
        h = self.upsample3(h)  # 8 -> 4
        h = self._match_size(h, h3)
        h = self.up3(torch.cat([h, h3], dim=1), cond)  # 4+4=8 -> 4
        
        h = self.upsample2(h)  # 4 -> 2
        h = self._match_size(h, h2)
        h = self.up2(torch.cat([h, h2], dim=1), cond)  # 2+2=4 -> 2
        
        h = self.upsample1(h)  # 2 -> 1
        h = self._match_size(h, h1)
        h = self.up1(torch.cat([h, h1], dim=1), cond)  # 1+1=2 -> 1
        
        h = F.silu(self.out_norm(h))
        h = self._match_size(h, x)
        return self.out_conv(h)

class DDIMScheduler:
    def __init__(self, num_timesteps=1000, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, device=device) / num_timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clamp(betas, 0.0001, 0.9999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    @torch.no_grad()
    def sample(self, model, content, style, shape, num_steps=100):
        """Sample with more steps for better quality."""
        device = content.device
        x = torch.randn(shape, device=device)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch, content, style)
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = pred_x0
        return x

# ============================================================================
# BiLSTM Classifier
# ============================================================================

class ECGClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def load_or_train_classifier():
    """Load existing or train new classifier."""
    classifier = ECGClassifier().to(DEVICE)
    
    # Check for existing trained classifier - delete old one that didn't work
    classifier_path = MODEL_DIR / 'ecg_classifier_v2.pth'
    if classifier_path.exists():
        print(f"Loading classifier from: {classifier_path}")
        try:
            classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
            # Verify it works
            classifier.eval()
            return classifier
        except:
            print("Failed to load, will retrain...")
    
    print("Training ECG classifier (this may take a few minutes)...")
    train_data = np.load(DATA_DIR / 'train_data.npz')
    train_signals = torch.tensor(train_data['X'], dtype=torch.float32)
    train_labels = torch.tensor(train_data['y'], dtype=torch.long)
    
    if train_signals.dim() == 2:
        train_signals = train_signals.unsqueeze(1)
    
    # Check class balance
    unique, counts = np.unique(train_labels.numpy(), return_counts=True)
    print(f"  Class distribution: {dict(zip(unique, counts))}")
    
    # Use a better optimizer setup
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    dataset = torch.utils.data.TensorDataset(train_signals, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
    best_acc = 0
    classifier.train()
    for epoch in range(30):  # More epochs
        total_loss = 0
        correct = 0
        total = 0
        for signals, labels in loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = classifier(signals)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        acc = 100*correct/total
        scheduler.step()
        
        if epoch % 5 == 0 or acc > best_acc:
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={acc:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), classifier_path)
    
    print(f"Best accuracy: {best_acc:.1f}%")
    print(f"Classifier saved to: {classifier_path}")
    
    # Reload best model
    classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
    return classifier

# ============================================================================
# Clinical Analysis Functions
# ============================================================================

def analyze_rr_intervals(ecg_signal, fs=500):
    """Detect R-peaks and compute RR interval statistics."""
    ecg = ecg_signal.flatten()
    
    # Normalize for peak detection
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    
    # Find peaks
    height = 0.5
    distance = int(0.3 * fs)
    peaks, _ = signal.find_peaks(ecg_norm, height=height, distance=distance)
    
    if len(peaks) < 2:
        return {'rr_mean': 0, 'rr_std': 0, 'rr_irregularity': 0, 'num_beats': 0}
    
    rr_intervals = np.diff(peaks) / fs * 1000  # ms
    
    return {
        'rr_mean': np.mean(rr_intervals),
        'rr_std': np.std(rr_intervals),
        'rr_irregularity': np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0,
        'num_beats': len(peaks)
    }

# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PHASE 3: CLINICAL EVALUATION")
    print("="*70)
    
    # Load classifier
    classifier = load_or_train_classifier()
    classifier.eval()
    
    # Test classifier on validation data
    print("\n--- Classifier Validation ---")
    val_data = np.load(DATA_DIR / 'val_data.npz')
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    
    if val_signals.dim() == 2:
        val_signals = val_signals.unsqueeze(1)
    
    with torch.no_grad():
        test_batch = val_signals[:100].to(DEVICE)
        test_labels = val_labels[:100].to(DEVICE)
        preds = classifier(test_batch).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
        print(f"Classifier validation accuracy: {acc:.1f}%")
    
    # Load Phase 3 model
    model_path = MODEL_DIR / 'final_model.pth'
    if not model_path.exists():
        print(f"\nModel not found: {model_path}")
        return
    
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    config = checkpoint['config']
    content_encoder = ContentEncoder(content_dim=config['content_dim']).to(DEVICE)
    style_encoder = StyleEncoder(style_dim=config['style_dim']).to(DEVICE)
    unet = ConditionalUNet(model_ch=config['model_channels'],
                          content_dim=config['content_dim'],
                          style_dim=config['style_dim']).to(DEVICE)
    
    content_encoder.load_state_dict(checkpoint['content_encoder'])
    style_encoder.load_state_dict(checkpoint['style_encoder'])
    unet.load_state_dict(checkpoint['unet'])
    
    content_encoder.eval()
    style_encoder.eval()
    unet.eval()
    
    scheduler = DDIMScheduler(1000, DEVICE)
    
    # Separate classes
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    print(f"\nNormal samples: {len(normal_idx)}")
    print(f"AFib samples: {len(afib_idx)}")
    
    # ========================================================================
    # TEST 1: Perfect Reconstruction
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: PERFECT RECONSTRUCTION")
    print("Content(Original) + Style(Original) → Should exactly match Original")
    print("="*70)
    
    num_test = 10
    with torch.no_grad():
        test_signals = val_signals[:num_test].to(DEVICE)
        
        # Encode with same content and style
        content, _, _ = content_encoder(test_signals)
        style, _ = style_encoder(test_signals)
        
        # Generate with 100 steps for better quality
        reconstructed = scheduler.sample(unet, content, style, test_signals.shape, num_steps=100)
        
        # Metrics
        mse = F.mse_loss(reconstructed, test_signals).item()
        
        correlations = []
        for i in range(num_test):
            orig = test_signals[i, 0].cpu().numpy()
            recon = reconstructed[i, 0].cpu().numpy()
            corr, _ = pearsonr(orig, recon)
            correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        
        print(f"\nReconstruction MSE: {mse:.6f}")
        print(f"Mean Correlation: {mean_corr:.4f}")
        print(f"Individual Correlations: {[f'{c:.3f}' for c in correlations]}")
        
        # Pass/Fail
        recon_pass = mse < 0.01 and mean_corr > 0.9
        print(f"\n{'PASS' if recon_pass else 'FAIL'}: MSE < 0.01 and Correlation > 0.9")
    
    # Visualization in millivolts
    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    fig.suptitle('TEST 1: Perfect Reconstruction (Clinical Range - mV)', fontsize=16, fontweight='bold')
    
    time_axis = np.arange(val_signals.shape[-1]) / 500  # seconds
    
    for i in range(5):
        orig_mv = to_millivolts(test_signals[i, 0].cpu().numpy())
        recon_mv = to_millivolts(reconstructed[i, 0].cpu().numpy())
        
        # Overlay
        axes[i, 0].plot(time_axis, orig_mv, 'b-', linewidth=1.2, alpha=0.8, label='Original')
        axes[i, 0].plot(time_axis, recon_mv, 'g-', linewidth=1.2, alpha=0.7, label='Reconstructed')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude (mV)')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'Sample {i+1}: Overlay (Corr={correlations[i]:.3f})')
        
        # Zoomed (first 2 seconds)
        axes[i, 1].plot(time_axis[:1000], orig_mv[:1000], 'b-', linewidth=1.5, alpha=0.8, label='Original')
        axes[i, 1].plot(time_axis[:1000], recon_mv[:1000], 'g-', linewidth=1.5, alpha=0.7, label='Reconstructed')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('Amplitude (mV)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title(f'Sample {i+1}: Zoomed (0-2s)')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test1_reconstruction_clinical.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test1_reconstruction_clinical.png'}")
    
    # ========================================================================
    # TEST 2: Counterfactual Normal → AFib
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: COUNTERFACTUAL (Normal → AFib)")
    print("Content(Normal) + Style(AFib) → Should flip classifier to AFib")
    print("="*70)
    
    num_cf = min(20, len(normal_idx), len(afib_idx))
    
    with torch.no_grad():
        normal_ecgs = val_signals[normal_idx[:num_cf]].to(DEVICE)
        afib_ecgs = val_signals[afib_idx[:num_cf]].to(DEVICE)
        
        # Encode
        normal_content, _, _ = content_encoder(normal_ecgs)
        afib_style, _ = style_encoder(afib_ecgs)
        
        # Generate counterfactual
        cf_normal_to_afib = scheduler.sample(unet, normal_content, afib_style, 
                                              normal_ecgs.shape, num_steps=100)
        
        # Classify
        orig_logits = classifier(normal_ecgs)
        cf_logits = classifier(cf_normal_to_afib)
        
        orig_preds = orig_logits.argmax(dim=1)
        cf_preds = cf_logits.argmax(dim=1)
        
        # Probability scores
        orig_probs = F.softmax(orig_logits, dim=1)
        cf_probs = F.softmax(cf_logits, dim=1)
        
        flip_to_afib = ((orig_preds == 0) & (cf_preds == 1)).sum().item()
        orig_normal = (orig_preds == 0).sum().item()
        
        print(f"\nOriginal predictions: {orig_preds.cpu().numpy()}")
        print(f"Counterfactual predictions: {cf_preds.cpu().numpy()}")
        print(f"\nOriginal correctly classified as Normal: {orig_normal}/{num_cf}")
        print(f"Flipped to AFib: {flip_to_afib}/{orig_normal} ({100*flip_to_afib/max(1,orig_normal):.1f}%)")
        
        # Average probability change
        orig_afib_prob = orig_probs[:, 1].mean().item()
        cf_afib_prob = cf_probs[:, 1].mean().item()
        print(f"\nMean AFib probability - Original: {orig_afib_prob:.3f}, Counterfactual: {cf_afib_prob:.3f}")
        print(f"Probability shift: +{cf_afib_prob - orig_afib_prob:.3f}")
    
    # Visualization in millivolts
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle('TEST 2: Counterfactual Normal→AFib (Clinical Range - mV)', fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(normal_ecgs[i, 0].cpu().numpy())
        cf_mv = to_millivolts(cf_normal_to_afib[i, 0].cpu().numpy())
        diff = cf_mv - orig_mv
        
        # Original
        axes[i, 0].plot(time_axis, orig_mv, 'b-', linewidth=1)
        axes[i, 0].set_title(f'Original Normal (P(AFib)={orig_probs[i,1]:.2f})')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Counterfactual
        axes[i, 1].plot(time_axis, cf_mv, 'r-', linewidth=1)
        axes[i, 1].set_title(f'Counterfactual (P(AFib)={cf_probs[i,1]:.2f})')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Overlay with difference
        axes[i, 2].plot(time_axis, orig_mv, 'b-', linewidth=1, alpha=0.7, label='Original')
        axes[i, 2].plot(time_axis, cf_mv, 'r-', linewidth=1, alpha=0.7, label='Counterfactual')
        axes[i, 2].fill_between(time_axis, orig_mv, cf_mv, alpha=0.3, color='purple', label='Difference')
        axes[i, 2].legend(loc='upper right', fontsize=8)
        axes[i, 2].set_title(f'Overlay (Pred: {orig_preds[i].item()}→{cf_preds[i].item()})')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].set_ylabel('mV')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test2_counterfactual_normal_to_afib.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test2_counterfactual_normal_to_afib.png'}")
    
    # ========================================================================
    # TEST 3: Counterfactual AFib → Normal
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: COUNTERFACTUAL (AFib → Normal)")
    print("Content(AFib) + Style(Normal) → Should flip classifier to Normal")
    print("="*70)
    
    with torch.no_grad():
        afib_ecgs = val_signals[afib_idx[:num_cf]].to(DEVICE)
        normal_ecgs = val_signals[normal_idx[:num_cf]].to(DEVICE)
        
        # Encode
        afib_content, _, _ = content_encoder(afib_ecgs)
        normal_style, _ = style_encoder(normal_ecgs)
        
        # Generate counterfactual
        cf_afib_to_normal = scheduler.sample(unet, afib_content, normal_style,
                                              afib_ecgs.shape, num_steps=100)
        
        # Classify
        orig_logits = classifier(afib_ecgs)
        cf_logits = classifier(cf_afib_to_normal)
        
        orig_preds = orig_logits.argmax(dim=1)
        cf_preds = cf_logits.argmax(dim=1)
        
        orig_probs = F.softmax(orig_logits, dim=1)
        cf_probs = F.softmax(cf_logits, dim=1)
        
        flip_to_normal = ((orig_preds == 1) & (cf_preds == 0)).sum().item()
        orig_afib = (orig_preds == 1).sum().item()
        
        print(f"\nOriginal predictions: {orig_preds.cpu().numpy()}")
        print(f"Counterfactual predictions: {cf_preds.cpu().numpy()}")
        print(f"\nOriginal correctly classified as AFib: {orig_afib}/{num_cf}")
        print(f"Flipped to Normal: {flip_to_normal}/{orig_afib} ({100*flip_to_normal/max(1,orig_afib):.1f}%)")
        
        orig_normal_prob = orig_probs[:, 0].mean().item()
        cf_normal_prob = cf_probs[:, 0].mean().item()
        print(f"\nMean Normal probability - Original: {orig_normal_prob:.3f}, Counterfactual: {cf_normal_prob:.3f}")
        print(f"Probability shift: +{cf_normal_prob - orig_normal_prob:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle('TEST 3: Counterfactual AFib→Normal (Clinical Range - mV)', fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(afib_ecgs[i, 0].cpu().numpy())
        cf_mv = to_millivolts(cf_afib_to_normal[i, 0].cpu().numpy())
        
        axes[i, 0].plot(time_axis, orig_mv, 'r-', linewidth=1)
        axes[i, 0].set_title(f'Original AFib (P(Normal)={orig_probs[i,0]:.2f})')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_axis, cf_mv, 'b-', linewidth=1)
        axes[i, 1].set_title(f'Counterfactual (P(Normal)={cf_probs[i,0]:.2f})')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(time_axis, orig_mv, 'r-', linewidth=1, alpha=0.7, label='Original AFib')
        axes[i, 2].plot(time_axis, cf_mv, 'b-', linewidth=1, alpha=0.7, label='Counterfactual')
        axes[i, 2].fill_between(time_axis, orig_mv, cf_mv, alpha=0.3, color='purple')
        axes[i, 2].legend(loc='upper right', fontsize=8)
        axes[i, 2].set_title(f'Overlay (Pred: {orig_preds[i].item()}→{cf_preds[i].item()})')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].set_ylabel('mV')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test3_counterfactual_afib_to_normal.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test3_counterfactual_afib_to_normal.png'}")
    
    # ========================================================================
    # Clinical Feature Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("CLINICAL FEATURE ANALYSIS: RR Intervals")
    print("="*70)
    
    print("\nNormal → AFib Counterfactuals:")
    normal_ecgs_np = val_signals[normal_idx[:5]].numpy()
    cf_np = cf_normal_to_afib[:5].cpu().numpy()
    
    for i in range(5):
        orig_rr = analyze_rr_intervals(normal_ecgs_np[i, 0, :])
        cf_rr = analyze_rr_intervals(cf_np[i, 0, :])
        
        print(f"  Sample {i+1}:")
        print(f"    Original RR irregularity: {orig_rr['rr_irregularity']:.4f} ({orig_rr['num_beats']} beats)")
        print(f"    Counterfactual RR irregularity: {cf_rr['rr_irregularity']:.4f} ({cf_rr['num_beats']} beats)")
        change = cf_rr['rr_irregularity'] - orig_rr['rr_irregularity']
        print(f"    Change: {'+' if change > 0 else ''}{change:.4f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nTest 1 - Reconstruction: MSE={mse:.6f}, Corr={mean_corr:.4f}")
    print(f"Test 2 - Normal→AFib flip rate: {100*flip_to_afib/max(1,orig_normal):.1f}%")
    print(f"Test 3 - AFib→Normal flip rate: {100*flip_to_normal/max(1,orig_afib):.1f}%")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*70)

if __name__ == '__main__':
    main()
