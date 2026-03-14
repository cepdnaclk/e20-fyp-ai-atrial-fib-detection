"""
Phase 3: WGAN-GP Based Counterfactual ECG Generator V9
=======================================================

Key Requirements:
1. Realistic output - no abnormal signals
2. Preserve most of original signal
3. Only modify class-discriminative features: R-R intervals and P-waves

Approach: WGAN-GP with Focused Modification

Architecture:
1. GENERATOR: Takes original signal + target class
   - Outputs a MODIFICATION MASK (what to change) + REPLACEMENT (what to change it to)
   - Mask focuses on P-wave regions and timing shifts
   
2. DISCRIMINATOR: WGAN-GP critic
   - Ensures output looks like real ECG
   - Class-conditional to ensure class-specific realism

3. LOSSES:
   - Wasserstein loss (realism)
   - Classifier flip loss (change class)
   - Reconstruction loss (preserve similarity)
   - Sparsity loss on mask (only change what's necessary)
"""

import os
import sys
import subprocess
import math

def get_free_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = [(int(l.split(',')[0]), int(l.split(',')[1])) for l in lines if l.strip()]
        if gpu_info:
            return str(max(gpu_info, key=lambda x: x[1])[0])
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
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scipy import signal as scipy_signal
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/wgan_v9'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

FS = 500
SIGNAL_LENGTH = 2500


# ============================================================================
# Beat Detection Utilities
# ============================================================================

def detect_r_peaks_batch(signals, fs=500):
    """Detect R-peaks for a batch of signals."""
    batch_peaks = []
    for sig in signals:
        sig_np = sig.cpu().numpy().flatten()
        nyq = fs / 2
        b, a = scipy_signal.butter(2, [5/nyq, 15/nyq], btype='band')
        try:
            filtered = scipy_signal.filtfilt(b, a, sig_np)
            diff = np.diff(filtered)
            squared = diff ** 2
            window = int(0.08 * fs)
            ma = np.convolve(squared, np.ones(window)/window, mode='same')
            peaks, _ = scipy_signal.find_peaks(ma, distance=int(0.3*fs), height=np.max(ma)*0.1)
            batch_peaks.append(peaks)
        except:
            batch_peaks.append(np.array([]))
    return batch_peaks


def compute_rr_cv(signal, fs=500):
    """Compute R-R interval coefficient of variation."""
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    signal = signal.flatten()
    
    nyq = fs / 2
    b, a = scipy_signal.butter(2, [5/nyq, 15/nyq], btype='band')
    try:
        filtered = scipy_signal.filtfilt(b, a, signal)
        diff_sig = np.diff(filtered)
        squared = diff_sig ** 2
        window = int(0.08 * fs)
        ma = np.convolve(squared, np.ones(window)/window, mode='same')
        peaks, _ = scipy_signal.find_peaks(ma, distance=int(0.3*fs), height=np.max(ma)*0.1)
        
        if len(peaks) < 2:
            return 0.0
        
        rr_intervals = np.diff(peaks) / fs * 1000
        cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8)
        return cv
    except:
        return 0.0


# ============================================================================
# WGAN Architecture
# ============================================================================

class ResBlock1D(nn.Module):
    """Residual block for 1D signals."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm1d(channels)
        self.norm2 = nn.InstanceNorm1d(channels)
    
    def forward(self, x):
        h = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        h = self.norm2(self.conv2(h))
        return x + h


class CounterfactualGenerator(nn.Module):
    """
    Generator that produces counterfactual ECG.
    
    Instead of generating from scratch, it:
    1. Takes original signal + target class
    2. Produces a modification (delta) to apply
    3. Delta is sparse - only modifies necessary regions
    """
    def __init__(self, signal_length=2500, hidden_dim=128, num_classes=2):
        super().__init__()
        
        self.signal_length = signal_length
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, 64)
        
        # Encoder: understand the input signal
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),  # /2
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),  # /4
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 5, stride=2, padding=2),  # /8
            nn.LeakyReLU(0.2),
        )
        
        # Class conditioning injection
        self.class_proj = nn.Linear(64, 256)
        
        # Transformer for global context
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=2
        )
        
        # Decoder: generate modification
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),  # *2
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),  # *4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),  # *8
            nn.LeakyReLU(0.2),
        )
        
        # Output heads
        # Modification head: what changes to make
        self.mod_head = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, 3, padding=1),
            nn.Tanh()  # Modifications in [-1, 1] range
        )
        
        # Mask head: where to apply changes (sparse)
        self.mask_head = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, 3, padding=1),
            nn.Sigmoid()  # Soft mask [0, 1]
        )
    
    def forward(self, x, target_class):
        """
        Args:
            x: Original signal (B, 1, L)
            target_class: Target class (B,)
        
        Returns:
            counterfactual: Modified signal
            mask: Where modifications were applied
            delta: What modifications were made
        """
        B = x.shape[0]
        
        # Encode signal
        h = self.encoder(x)  # (B, 256, L/8)
        
        # Add class conditioning
        class_emb = self.class_embed(target_class)  # (B, 64)
        class_cond = self.class_proj(class_emb)  # (B, 256)
        h = h + class_cond.unsqueeze(-1)  # Broadcast across time
        
        # Global attention
        h = h.permute(0, 2, 1)  # (B, L/8, 256)
        h = self.attention(h)
        h = h.permute(0, 2, 1)  # (B, 256, L/8)
        
        # Decode
        h = self.decoder(h)  # (B, 32, L)
        
        # Adjust length if needed
        if h.shape[2] != x.shape[2]:
            h = F.interpolate(h, size=x.shape[2], mode='linear', align_corners=False)
        
        # Get modification and mask
        delta = self.mod_head(h)  # (B, 1, L)
        mask = self.mask_head(h)  # (B, 1, L)
        
        # Scale delta by signal amplitude
        signal_std = x.std(dim=2, keepdim=True) + 1e-6
        delta = delta * signal_std * 0.5  # Limit modification magnitude
        
        # Apply modification: counterfactual = original + mask * delta
        counterfactual = x + mask * delta
        
        # Ensure physiologically valid range
        counterfactual = torch.clamp(counterfactual, -3, 3)
        
        return counterfactual, mask, delta


class WGANCritic(nn.Module):
    """
    WGAN-GP Critic (Discriminator).
    Class-conditional to ensure class-specific realism.
    """
    def __init__(self, signal_length=2500, num_classes=2):
        super().__init__()
        
        self.class_embed = nn.Embedding(num_classes, 64)
        
        self.main = nn.Sequential(
            nn.Conv1d(1 + 64, 64, 7, stride=2, padding=3),  # 1250
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),  # 625
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 5, stride=2, padding=2),  # 313
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 5, stride=2, padding=2),  # 157
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 5, stride=2, padding=2),  # 79
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 79, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x, class_label):
        """
        Args:
            x: Signal (B, 1, L)
            class_label: Class (B,)
        """
        B = x.shape[0]
        
        # Class conditioning
        class_emb = self.class_embed(class_label)  # (B, 64)
        class_cond = class_emb.unsqueeze(-1).expand(-1, -1, x.shape[2])  # (B, 64, L)
        
        # Concatenate signal with class embedding
        h = torch.cat([x, class_cond], dim=1)  # (B, 1+64, L)
        
        h = self.main(h)
        h = h.view(B, -1)
        
        return self.fc(h)


# ============================================================================
# Classifier Wrapper
# ============================================================================

def load_classifier():
    classifier_path = PROJECT_ROOT / 'models/afib_reslstm_final.pth'
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    return classifier


class ClassifierWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    
    def forward(self, x):
        self.model.train()  # LSTM requires train mode for backward
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        logits, _ = self.model(x_norm)
        return logits


# ============================================================================
# Training Losses
# ============================================================================

def gradient_penalty(critic, real, fake, class_label, device):
    """Compute gradient penalty for WGAN-GP."""
    B = real.shape[0]
    alpha = torch.rand(B, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    critic_interp = critic(interpolated, class_label)
    
    gradients = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(B, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty


def sparsity_loss(mask):
    """Encourage sparse modifications."""
    # L1 loss encourages sparsity
    return mask.abs().mean()


def smoothness_loss(mask):
    """Encourage smooth mask (no sudden changes)."""
    diff = mask[:, :, 1:] - mask[:, :, :-1]
    return diff.abs().mean()


def rr_consistency_loss(original, counterfactual, target_class, device):
    """
    Encourage RR variability to change in the expected direction.
    Normal (0): Low CV
    AFib (1): High CV
    """
    loss = 0
    batch_size = original.shape[0]
    
    for i in range(min(batch_size, 4)):  # Sample a few for efficiency
        orig_cv = compute_rr_cv(original[i])
        cf_cv = compute_rr_cv(counterfactual[i])
        
        if target_class[i].item() == 1:  # Target is AFib - want higher CV
            if cf_cv < orig_cv:
                loss += (orig_cv - cf_cv) * 2
        else:  # Target is Normal - want lower CV
            if cf_cv > orig_cv:
                loss += (cf_cv - orig_cv) * 2
    
    return torch.tensor(loss / batch_size, device=device)


# ============================================================================
# Training
# ============================================================================

def train_wgan(train_loader, val_signals, val_labels, classifier, 
               epochs=100, lr_g=1e-4, lr_c=1e-4, n_critic=5):
    """Train WGAN-GP for counterfactual generation."""
    
    generator = CounterfactualGenerator().to(DEVICE)
    critic = WGANCritic().to(DEVICE)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr_c, betas=(0.5, 0.9))
    
    # Loss weights
    lambda_gp = 10.0      # Gradient penalty
    lambda_flip = 2.0     # Classifier flip
    lambda_sim = 5.0      # Similarity to original
    lambda_sparse = 0.5   # Sparsity of modifications
    lambda_smooth = 0.2   # Smoothness of mask
    lambda_rr = 1.0       # RR direction consistency
    
    history = {'g_loss': [], 'c_loss': [], 'flip_rate': [], 'similarity': []}
    best_score = 0
    
    for epoch in range(epochs):
        generator.train()
        critic.train()
        
        epoch_g_loss = 0
        epoch_c_loss = 0
        n_batches = 0
        
        for batch_idx, (real_signals, labels) in enumerate(train_loader):
            real_signals = real_signals.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Target class is opposite of original
            target_labels = 1 - labels
            
            B = real_signals.shape[0]
            
            # =====================
            # Train Critic
            # =====================
            for _ in range(n_critic):
                opt_c.zero_grad()
                
                # Generate counterfactuals
                with torch.no_grad():
                    fake_signals, _, _ = generator(real_signals, target_labels)
                
                # Real signals should score high
                real_score = critic(real_signals, labels)
                
                # Fake signals should score low
                fake_score = critic(fake_signals.detach(), target_labels)
                
                # Wasserstein loss
                c_loss = fake_score.mean() - real_score.mean()
                
                # Gradient penalty
                gp = gradient_penalty(critic, real_signals, fake_signals.detach(), 
                                       target_labels, DEVICE)
                c_loss = c_loss + lambda_gp * gp
                
                c_loss.backward()
                opt_c.step()
            
            # =====================
            # Train Generator
            # =====================
            opt_g.zero_grad()
            
            # Generate counterfactuals
            fake_signals, mask, delta = generator(real_signals, target_labels)
            
            # Adversarial loss (fool critic)
            g_adv = -critic(fake_signals, target_labels).mean()
            
            # Classifier flip loss
            classifier_logits = classifier(fake_signals)
            g_flip = F.cross_entropy(classifier_logits, target_labels)
            
            # Similarity loss (preserve original)
            g_sim = F.mse_loss(fake_signals, real_signals)
            
            # Sparsity loss (only modify what's necessary)
            g_sparse = sparsity_loss(mask)
            
            # Smoothness loss (smooth modifications)
            g_smooth = smoothness_loss(mask)
            
            # RR consistency loss
            g_rr = rr_consistency_loss(real_signals, fake_signals, target_labels, DEVICE)
            
            # Total generator loss
            g_loss = (g_adv + 
                      lambda_flip * g_flip + 
                      lambda_sim * g_sim + 
                      lambda_sparse * g_sparse + 
                      lambda_smooth * g_smooth +
                      lambda_rr * g_rr)
            
            g_loss.backward()
            opt_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_c_loss += c_loss.item()
            n_batches += 1
        
        # Validation
        generator.eval()
        with torch.no_grad():
            val_signals_gpu = val_signals[:100].to(DEVICE)
            val_labels_gpu = val_labels[:100].to(DEVICE)
            target = 1 - val_labels_gpu
            
            cf_signals, _, _ = generator(val_signals_gpu, target)
            
            # Flip rate
            cf_logits = classifier(cf_signals)
            cf_preds = cf_logits.argmax(dim=1)
            flip_rate = (cf_preds == target).float().mean().item()
            
            # Similarity
            similarities = []
            for i in range(min(50, len(cf_signals))):
                orig = val_signals_gpu[i, 0].cpu().numpy()
                cf = cf_signals[i, 0].cpu().numpy()
                corr, _ = pearsonr(orig, cf)
                similarities.append(corr)
            mean_sim = np.mean(similarities)
        
        epoch_g_loss /= n_batches
        epoch_c_loss /= n_batches
        
        history['g_loss'].append(epoch_g_loss)
        history['c_loss'].append(epoch_c_loss)
        history['flip_rate'].append(flip_rate)
        history['similarity'].append(mean_sim)
        
        # Save best model
        score = flip_rate * 0.5 + mean_sim * 0.5
        if score > best_score:
            best_score = score
            torch.save({
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'epoch': epoch
            }, RESULTS_DIR / 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | G: {epoch_g_loss:.4f} | C: {epoch_c_loss:.4f} | "
                  f"Flip: {flip_rate*100:.1f}% | Sim: {mean_sim:.3f}")
    
    return generator, history


# ============================================================================
# Evaluation and Visualization
# ============================================================================

def check_validity(signal, fs=500):
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    signal = signal.flatten()
    
    amp = np.max(signal) - np.min(signal)
    if amp < 0.01 or amp > 10:
        return False
    
    nyq = fs / 2
    b, a = scipy_signal.butter(2, [5/nyq, 15/nyq], btype='band')
    try:
        filtered = scipy_signal.filtfilt(b, a, signal)
        peaks, _ = scipy_signal.find_peaks(np.diff(filtered)**2, distance=int(0.3*fs))
        if len(peaks) < 2:
            return False
        
        rr = np.diff(peaks) / fs
        hr = 60 / np.mean(rr)
        if hr < 30 or hr > 200:
            return False
    except:
        return False
    
    return True


def create_visualization(orig, cf, mask, delta, orig_class, target_class, 
                          orig_prob, cf_prob, idx, save_path):
    """Create detailed visualization."""
    orig_np = orig.cpu().numpy().flatten()
    cf_np = cf.cpu().numpy().flatten()
    mask_np = mask.cpu().numpy().flatten()
    delta_np = delta.cpu().numpy().flatten()
    
    time = np.arange(len(orig_np)) / FS
    
    orig_cv = compute_rr_cv(orig_np, FS)
    cf_cv = compute_rr_cv(cf_np, FS)
    corr, _ = pearsonr(orig_np, cf_np)
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1.5, 1, 1, 1, 0.6])
    
    class_names = ['Normal', 'AFib']
    
    # Row 1: Original and Counterfactual
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time, orig_np, 'b-', lw=1)
    ax1.set_title(f'ORIGINAL ({class_names[orig_class]}) | RR CV: {orig_cv:.3f}', 
                  fontweight='bold', color='blue', fontsize=12)
    ax1.set_xlabel('Time (s)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(time, cf_np, 'r-', lw=1)
    ax2.set_title(f'COUNTERFACTUAL (→{class_names[target_class]}) | RR CV: {cf_cv:.3f}', 
                  fontweight='bold', color='red', fontsize=12)
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    
    # Row 2: Zoomed segments
    for col, (s, e) in enumerate([(0, 500), (1000, 1500)]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(time[s:e], orig_np[s:e], 'b-', lw=1.5)
        ax.set_title(f'Orig {s/FS:.1f}-{e/FS:.1f}s')
        ax.grid(True, alpha=0.3)
        
        ax = fig.add_subplot(gs[1, col + 2])
        ax.plot(time[s:e], cf_np[s:e], 'r-', lw=1.5)
        ax.set_title(f'CF {s/FS:.1f}-{e/FS:.1f}s')
        ax.grid(True, alpha=0.3)
    
    # Row 3: Modification mask and delta
    ax_mask = fig.add_subplot(gs[2, :2])
    ax_mask.fill_between(time, 0, mask_np, alpha=0.5, color='purple')
    ax_mask.set_title('Modification Mask (where changes applied)', fontsize=11)
    ax_mask.set_xlabel('Time (s)')
    ax_mask.set_ylabel('Mask value')
    ax_mask.set_ylim([0, 1])
    ax_mask.grid(True, alpha=0.3)
    
    ax_delta = fig.add_subplot(gs[2, 2:])
    ax_delta.plot(time, delta_np, 'g-', lw=0.8)
    ax_delta.fill_between(time, 0, delta_np, alpha=0.3, color='green')
    ax_delta.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_delta.set_title('Modification Delta (what was changed)', fontsize=11)
    ax_delta.set_xlabel('Time (s)')
    ax_delta.set_ylabel('Delta')
    ax_delta.grid(True, alpha=0.3)
    
    # Row 4: RR comparison
    orig_peaks = detect_r_peaks_batch([torch.tensor(orig_np)])[0]
    cf_peaks = detect_r_peaks_batch([torch.tensor(cf_np)])[0]
    
    ax_rr1 = fig.add_subplot(gs[3, :2])
    if len(orig_peaks) > 1:
        rr = np.diff(orig_peaks) / FS * 1000
        ax_rr1.bar(range(len(rr)), rr, color='blue', alpha=0.7)
        ax_rr1.axhline(np.mean(rr), color='darkblue', linestyle='--', lw=2)
    ax_rr1.set_xlabel('Beat #')
    ax_rr1.set_ylabel('RR (ms)')
    ax_rr1.set_title(f'Original RR Intervals (CV={orig_cv:.3f})')
    ax_rr1.grid(True, alpha=0.3)
    
    ax_rr2 = fig.add_subplot(gs[3, 2:])
    if len(cf_peaks) > 1:
        rr = np.diff(cf_peaks) / FS * 1000
        ax_rr2.bar(range(len(rr)), rr, color='red', alpha=0.7)
        ax_rr2.axhline(np.mean(rr), color='darkred', linestyle='--', lw=2)
    ax_rr2.set_xlabel('Beat #')
    ax_rr2.set_ylabel('RR (ms)')
    ax_rr2.set_title(f'Counterfactual RR Intervals (CV={cf_cv:.3f})')
    ax_rr2.grid(True, alpha=0.3)
    
    # Row 5: Summary
    ax_sum = fig.add_subplot(gs[4, :])
    ax_sum.axis('off')
    
    cv_change = cf_cv - orig_cv
    expected = 'INCREASE' if target_class == 1 else 'DECREASE'
    actual = 'INCREASED' if cv_change > 0 else 'DECREASED'
    correct_rr = (target_class == 1 and cv_change > 0) or (target_class == 0 and cv_change < 0)
    flipped = (cf_prob > 0.5) if target_class == 1 else (cf_prob < 0.5)
    valid = check_validity(cf_np, FS)
    mask_sparsity = (mask_np < 0.5).mean() * 100
    
    summary = f"""
    SAMPLE {idx} | {class_names[orig_class]} → {class_names[target_class]}
    ════════════════════════════════════════════════════════════════════════════════════════════════════
    CORRELATION: {corr:.4f}  |  RR CV: {orig_cv:.3f} → {cf_cv:.3f} ({'+' if cv_change > 0 else ''}{cv_change:.3f})  |  Expected: {expected}  Actual: {actual}  {'✓' if correct_rr else '✗'}
    CLASSIFIER: P({class_names[target_class]}) = {orig_prob:.3f} → {cf_prob:.3f}  {'✓ FLIPPED' if flipped else '✗ NOT FLIPPED'}  |  VALID: {'✓' if valid else '✗'}  |  MASK SPARSITY: {mask_sparsity:.1f}%
    """
    ax_sum.text(0.02, 0.5, summary, fontfamily='monospace', fontsize=11, va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'corr': corr, 'flipped': flipped, 'valid': valid,
        'orig_cv': orig_cv, 'cf_cv': cf_cv, 'correct_rr': correct_rr,
        'mask_sparsity': mask_sparsity
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("WGAN-GP COUNTERFACTUAL ECG GENERATOR V9")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load classifier
    print("\nLoading classifier...")
    raw_classifier = load_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    
    # Load data
    print("Loading data...")
    train_data = np.load(DATA_DIR / 'train_data.npz')
    val_data = np.load(DATA_DIR / 'val_data.npz')
    
    train_signals = torch.tensor(train_data['X'], dtype=torch.float32)
    train_labels = torch.tensor(train_data['y'], dtype=torch.long)
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    
    if train_signals.dim() == 2:
        train_signals = train_signals.unsqueeze(1)
    if val_signals.dim() == 2:
        val_signals = val_signals.unsqueeze(1)
    
    print(f"Train: {len(train_signals)}, Val: {len(val_signals)}")
    
    # Create data loader
    train_dataset = TensorDataset(train_signals, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # Train WGAN
    print("\n" + "="*70)
    print("TRAINING WGAN-GP")
    print("="*70)
    
    generator, history = train_wgan(
        train_loader, val_signals, val_labels, classifier,
        epochs=50, lr_g=1e-4, lr_c=1e-4, n_critic=5
    )
    
    # Load best model
    checkpoint = torch.load(RESULTS_DIR / 'best_model.pth')
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # ========================================================================
    # Generate and visualize counterfactuals
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING COUNTERFACTUALS")
    print("="*70)
    
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    num_samples = 10
    results = {'n2a': [], 'a2n': []}
    
    print("\n--- Normal → AFib ---")
    for i in tqdm(range(num_samples), desc="N→AF"):
        x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
        target = torch.tensor([1], device=DEVICE)
        
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 1].item()
            cf, mask, delta = generator(x, target)
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 1].item()
        
        save_path = RESULTS_DIR / f'n2a_{i+1:02d}.png'
        result = create_visualization(
            x[0, 0], cf[0, 0], mask[0, 0], delta[0, 0],
            0, 1, orig_prob, cf_prob, i+1, save_path
        )
        results['n2a'].append(result)
        print(f"  #{i+1}: P(AF) {orig_prob:.3f}→{cf_prob:.3f} | Corr: {result['corr']:.3f} | CV: {result['orig_cv']:.3f}→{result['cf_cv']:.3f}")
    
    print("\n--- AFib → Normal ---")
    for i in tqdm(range(num_samples), desc="AF→N"):
        x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
        target = torch.tensor([0], device=DEVICE)
        
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 0].item()
            cf, mask, delta = generator(x, target)
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 0].item()
        
        save_path = RESULTS_DIR / f'a2n_{i+1:02d}.png'
        result = create_visualization(
            x[0, 0], cf[0, 0], mask[0, 0], delta[0, 0],
            1, 0, orig_prob, cf_prob, i+1, save_path
        )
        results['a2n'].append(result)
        print(f"  #{i+1}: P(N) {orig_prob:.3f}→{cf_prob:.3f} | Corr: {result['corr']:.3f} | CV: {result['orig_cv']:.3f}→{result['cf_cv']:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    n2a = results['n2a']
    a2n = results['a2n']
    
    n2a_valid = sum(r['valid'] for r in n2a)
    n2a_flip = sum(r['flipped'] for r in n2a)
    n2a_rr = sum(r['correct_rr'] for r in n2a)
    n2a_corr = np.mean([r['corr'] for r in n2a])
    
    a2n_valid = sum(r['valid'] for r in a2n)
    a2n_flip = sum(r['flipped'] for r in a2n)
    a2n_rr = sum(r['correct_rr'] for r in a2n)
    a2n_corr = np.mean([r['corr'] for r in a2n])
    
    print(f"\nNormal → AFib:")
    print(f"  Valid: {n2a_valid}/{num_samples} ({100*n2a_valid/num_samples:.0f}%)")
    print(f"  Flipped: {n2a_flip}/{num_samples} ({100*n2a_flip/num_samples:.0f}%)")
    print(f"  Correct RR: {n2a_rr}/{num_samples} ({100*n2a_rr/num_samples:.0f}%)")
    print(f"  Correlation: {n2a_corr:.4f}")
    
    print(f"\nAFib → Normal:")
    print(f"  Valid: {a2n_valid}/{num_samples} ({100*a2n_valid/num_samples:.0f}%)")
    print(f"  Flipped: {a2n_flip}/{num_samples} ({100*a2n_flip/num_samples:.0f}%)")
    print(f"  Correct RR: {a2n_rr}/{num_samples} ({100*a2n_rr/num_samples:.0f}%)")
    print(f"  Correlation: {a2n_corr:.4f}")
    
    total = 2 * num_samples
    print(f"\nOVERALL:")
    print(f"  Valid: {n2a_valid+a2n_valid}/{total} ({100*(n2a_valid+a2n_valid)/total:.0f}%)")
    print(f"  Flipped: {n2a_flip+a2n_flip}/{total} ({100*(n2a_flip+a2n_flip)/total:.0f}%)")
    print(f"  Correct RR: {n2a_rr+a2n_rr}/{total} ({100*(n2a_rr+a2n_rr)/total:.0f}%)")
    print(f"  Correlation: {(n2a_corr+a2n_corr)/2:.4f}")
    
    # Save results
    summary = {
        'n2a': {'valid': n2a_valid/num_samples, 'flip': n2a_flip/num_samples, 
                'rr': n2a_rr/num_samples, 'corr': float(n2a_corr)},
        'a2n': {'valid': a2n_valid/num_samples, 'flip': a2n_flip/num_samples,
                'rr': a2n_rr/num_samples, 'corr': float(a2n_corr)}
    }
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save training history
    with open(RESULTS_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
