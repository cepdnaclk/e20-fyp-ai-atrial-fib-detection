"""
Phase 3: Counterfactual Evaluation and Validation
==================================================

This script:
1. Loads the trained Content-Style Diffusion model
2. Generates counterfactuals for test ECGs
3. Validates that counterfactuals flip the classifier prediction
4. Visualizes the changes between original and counterfactual
5. Identifies clinically meaningful changes (P-waves, RR intervals, etc.)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy import signal
from scipy.stats import pearsonr

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual'
RESULTS_DIR = MODEL_DIR / 'counterfactual_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Classifier path (AFib-BiLSTM)
CLASSIFIER_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# Load ECG Classifier (for validation)
# ============================================================================

class ECGClassifier(nn.Module):
    """
    Simple BiLSTM classifier for AFib detection.
    If a pre-trained model exists, we'll load it. Otherwise, train a quick one.
    """
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
        # x: (B, 1, L) -> (B, L, 1)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out

def load_or_train_classifier():
    """Load existing classifier or train a new one."""
    classifier = ECGClassifier().to(DEVICE)
    
    # Check for existing classifier
    classifier_paths = [
        CLASSIFIER_DIR / 'models/bilstm_afib_classifier.pth',
        MODEL_DIR / 'ecg_classifier.pth',
    ]
    
    for path in classifier_paths:
        if path.exists():
            print(f"Loading classifier from: {path}")
            classifier.load_state_dict(torch.load(path, map_location=DEVICE))
            return classifier
    
    # Train a quick classifier
    print("Training ECG classifier...")
    
    train_data = np.load(DATA_DIR / 'train_data.npz')
    train_signals = torch.tensor(train_data['X'], dtype=torch.float32)
    train_labels = torch.tensor(train_data['y'], dtype=torch.long)
    
    if train_signals.dim() == 2:
        train_signals = train_signals.unsqueeze(1)
    
    # Quick training
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(train_signals, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    classifier.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        for signals, labels in loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = classifier(signals)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={100*correct/total:.1f}%")
    
    # Save classifier
    torch.save(classifier.state_dict(), MODEL_DIR / 'ecg_classifier.pth')
    print(f"Classifier saved to: {MODEL_DIR / 'ecg_classifier.pth'}")
    
    return classifier

# ============================================================================
# Load Content-Style Models (from training script)
# ============================================================================

# Import model architectures
import sys
sys.path.insert(0, str(Path(__file__).parent))

import math

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
        )
        self.flat_size = hidden_dim * 8 * (2500 // 32)
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
        self.up3 = ConditionalResBlock(model_ch * 12, model_ch * 4, cond_dim)
        self.up2 = ConditionalResBlock(model_ch * 6, model_ch * 2, cond_dim)
        self.up1 = ConditionalResBlock(model_ch * 3, model_ch, cond_dim)
        
        self.upsample4 = nn.ConvTranspose1d(model_ch * 8, model_ch * 8, 4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose1d(model_ch * 4, model_ch * 4, 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose1d(model_ch * 2, model_ch * 2, 4, stride=2, padding=1)
        self.upsample1 = nn.ConvTranspose1d(model_ch, model_ch, 4, stride=2, padding=1)
        
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
        h = self.up4(torch.cat([h, h4], dim=1), cond)
        h = self.upsample3(h)
        h = self._match_size(h, h3)
        h = self.up3(torch.cat([h, h3], dim=1), cond)
        h = self.upsample2(h)
        h = self._match_size(h, h2)
        h = self.up2(torch.cat([h, h2], dim=1), cond)
        h = self.upsample1(h)
        h = self._match_size(h, h1)
        h = self.up1(torch.cat([h, h1], dim=1), cond)
        
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
    def sample(self, model, content, style, shape, num_steps=50):
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
# Clinical Feature Analysis
# ============================================================================

def analyze_rr_intervals(ecg_signal, fs=500):
    """Detect R-peaks and compute RR interval statistics."""
    # Simple R-peak detection using scipy
    ecg = ecg_signal.flatten()
    
    # Find peaks
    height = np.mean(ecg) + 0.5 * np.std(ecg)
    distance = int(0.3 * fs)  # Minimum 300ms between beats
    peaks, _ = signal.find_peaks(ecg, height=height, distance=distance)
    
    if len(peaks) < 2:
        return {'rr_mean': 0, 'rr_std': 0, 'rr_irregularity': 0, 'num_beats': 0}
    
    rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
    
    return {
        'rr_mean': np.mean(rr_intervals),
        'rr_std': np.std(rr_intervals),
        'rr_irregularity': np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0,
        'num_beats': len(peaks)
    }

def compute_difference_map(original, counterfactual):
    """Compute pointwise difference between original and counterfactual."""
    diff = counterfactual - original
    return diff

def highlight_changes(original, counterfactual, threshold=0.1):
    """Identify regions where significant changes occurred."""
    diff = np.abs(counterfactual - original)
    significant = diff > threshold
    return significant

# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("\n" + "="*70)
    print("COUNTERFACTUAL EVALUATION AND VALIDATION")
    print("="*70)
    
    # Load classifier
    classifier = load_or_train_classifier()
    classifier.eval()
    
    # Load data
    print("\nLoading test data...")
    val_data = np.load(DATA_DIR / 'val_data.npz')
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    
    if val_signals.dim() == 2:
        val_signals = val_signals.unsqueeze(1)
    
    # Check for trained model
    model_path = MODEL_DIR / 'final_model.pth'
    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("Please run 01_train_content_style_diffusion.py first!")
        print("\nTo train, run:")
        print(f"  cd {Path(__file__).parent}")
        print(f"  python 01_train_content_style_diffusion.py")
        return
    
    # Load models
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
    # TEST 1: Reconstruction (Content + Style from same ECG)
    # ========================================================================
    print("\n" + "-"*50)
    print("TEST 1: Reconstruction Quality")
    print("-"*50)
    
    with torch.no_grad():
        test_signals = val_signals[:20].to(DEVICE)
        test_labels = val_labels[:20]
        
        content, _, _ = content_encoder(test_signals)
        style, _ = style_encoder(test_signals)
        
        shape = test_signals.shape
        reconstructed = scheduler.sample(unet, content, style, shape, num_steps=50)
        
        # Reconstruction error
        recon_error = F.mse_loss(reconstructed, test_signals).item()
        print(f"Reconstruction MSE: {recon_error:.6f}")
        
        # Correlation
        correlations = []
        for i in range(len(test_signals)):
            orig = test_signals[i, 0].cpu().numpy()
            recon = reconstructed[i, 0].cpu().numpy()
            corr, _ = pearsonr(orig, recon)
            correlations.append(corr)
        print(f"Mean Correlation: {np.mean(correlations):.4f}")
    
    # ========================================================================
    # TEST 2: Counterfactual Generation (Normal -> AFib)
    # ========================================================================
    print("\n" + "-"*50)
    print("TEST 2: Counterfactual Generation (Normal -> AFib)")
    print("-"*50)
    
    num_test = min(10, len(normal_idx), len(afib_idx))
    
    with torch.no_grad():
        # Get normal ECGs
        normal_ecgs = val_signals[normal_idx[:num_test]].to(DEVICE)
        
        # Get style from AFib ECGs
        afib_ecgs = val_signals[afib_idx[:num_test]].to(DEVICE)
        
        # Encode
        normal_content, _, _ = content_encoder(normal_ecgs)
        afib_style, _ = style_encoder(afib_ecgs)
        
        # Generate counterfactuals: Normal content + AFib style
        cf_normal_to_afib = scheduler.sample(unet, normal_content, afib_style, 
                                              normal_ecgs.shape, num_steps=50)
        
        # Classify original and counterfactual
        orig_logits = classifier(normal_ecgs)
        cf_logits = classifier(cf_normal_to_afib)
        
        orig_preds = orig_logits.argmax(dim=1)
        cf_preds = cf_logits.argmax(dim=1)
        
        flip_count = (orig_preds != cf_preds).sum().item()
        flip_rate = flip_count / num_test * 100
        
        print(f"Original predictions: {orig_preds.cpu().numpy()}")
        print(f"Counterfactual predictions: {cf_preds.cpu().numpy()}")
        print(f"Flip rate (Normal -> AFib): {flip_rate:.1f}% ({flip_count}/{num_test})")
    
    # ========================================================================
    # TEST 3: Counterfactual Generation (AFib -> Normal)
    # ========================================================================
    print("\n" + "-"*50)
    print("TEST 3: Counterfactual Generation (AFib -> Normal)")
    print("-"*50)
    
    with torch.no_grad():
        # Get AFib ECGs
        afib_ecgs = val_signals[afib_idx[:num_test]].to(DEVICE)
        
        # Get style from Normal ECGs
        normal_ecgs = val_signals[normal_idx[:num_test]].to(DEVICE)
        
        # Encode
        afib_content, _, _ = content_encoder(afib_ecgs)
        normal_style, _ = style_encoder(normal_ecgs)
        
        # Generate counterfactuals: AFib content + Normal style
        cf_afib_to_normal = scheduler.sample(unet, afib_content, normal_style,
                                              afib_ecgs.shape, num_steps=50)
        
        # Classify
        orig_logits = classifier(afib_ecgs)
        cf_logits = classifier(cf_afib_to_normal)
        
        orig_preds = orig_logits.argmax(dim=1)
        cf_preds = cf_logits.argmax(dim=1)
        
        flip_count = (orig_preds != cf_preds).sum().item()
        flip_rate = flip_count / num_test * 100
        
        print(f"Original predictions: {orig_preds.cpu().numpy()}")
        print(f"Counterfactual predictions: {cf_preds.cpu().numpy()}")
        print(f"Flip rate (AFib -> Normal): {flip_rate:.1f}% ({flip_count}/{num_test})")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print("\n" + "-"*50)
    print("Creating Visualizations")
    print("-"*50)
    
    # Move data to CPU for visualization
    normal_ecgs_np = val_signals[normal_idx[:5]].numpy()
    cf_normal_to_afib_np = cf_normal_to_afib[:5].cpu().numpy()
    
    # Time axis
    time_axis = np.arange(normal_ecgs_np.shape[-1]) / 500  # Assuming 500 Hz
    
    # 1. Overlay comparison
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    fig.suptitle('Counterfactual Overlay: Original Normal (Blue) vs Counterfactual AFib (Red)', 
                 fontsize=14, fontweight='bold')
    
    for i in range(5):
        ax = axes[i]
        ax.plot(time_axis, normal_ecgs_np[i, 0, :], color='blue', linewidth=1, 
                alpha=0.8, label='Original (Normal)')
        ax.plot(time_axis, cf_normal_to_afib_np[i, 0, :], color='red', linewidth=1, 
                alpha=0.7, label='Counterfactual (AFib style)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (mV)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'counterfactual_overlay.png', dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'counterfactual_overlay.png'}")
    
    # 2. Difference highlighting
    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    fig.suptitle('Original vs Counterfactual with Difference Map', fontsize=14, fontweight='bold')
    
    for i in range(5):
        orig = normal_ecgs_np[i, 0, :]
        cf = cf_normal_to_afib_np[i, 0, :]
        diff = cf - orig
        
        # Left: Overlay
        axes[i, 0].plot(time_axis, orig, color='blue', linewidth=0.8, alpha=0.8, label='Original')
        axes[i, 0].plot(time_axis, cf, color='red', linewidth=0.8, alpha=0.7, label='Counterfactual')
        axes[i, 0].set_title(f'Sample {i+1}: Overlay')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Right: Difference
        axes[i, 1].fill_between(time_axis, diff, 0, where=diff > 0, color='green', alpha=0.5, label='Added')
        axes[i, 1].fill_between(time_axis, diff, 0, where=diff < 0, color='purple', alpha=0.5, label='Removed')
        axes[i, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes[i, 1].set_title(f'Sample {i+1}: Difference (CF - Original)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'counterfactual_difference.png', dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'counterfactual_difference.png'}")
    
    # 3. RR interval analysis
    print("\n" + "-"*50)
    print("Clinical Feature Analysis: RR Intervals")
    print("-"*50)
    
    for i in range(min(3, len(normal_ecgs_np))):
        orig_rr = analyze_rr_intervals(normal_ecgs_np[i, 0, :])
        cf_rr = analyze_rr_intervals(cf_normal_to_afib_np[i, 0, :])
        
        print(f"\nSample {i+1}:")
        print(f"  Original RR irregularity: {orig_rr['rr_irregularity']:.4f}")
        print(f"  Counterfactual RR irregularity: {cf_rr['rr_irregularity']:.4f}")
        print(f"  Change: {(cf_rr['rr_irregularity'] - orig_rr['rr_irregularity']):.4f}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70)

if __name__ == '__main__':
    main()
