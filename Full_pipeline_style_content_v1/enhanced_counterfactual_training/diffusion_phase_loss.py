# ============================================================================
# SINGLE-FOLD ECG DIFFUSION WITH FULL DIAGNOSTICS
# ============================================================================
"""
Clean training for 30-50 epochs with:
✅ Single train/val split (no 5-fold)
✅ PSD analysis
✅ NeuroKit2 validation
✅ Phase shift measurement
✅ Smoothness loss properly unclamped
✅ Clean logging (not overwhelming)
✅ Your existing architecture (working!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import DDIMScheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy import signal as scipy_signal

# Import your existing models
from diffusion import (
    UNet1DConditional, 
    StyleEncoderWrapper, 
    PretrainedContentEncoder,
    EMA,
    compute_frequency_loss,
    ModelConfig
)

# ============================================================================
# IMPROVED SMOOTHNESS LOSS (UNCLAMPED)
# ============================================================================

def compute_temporal_smoothness_loss_unclamped(pred_x0):
    """
    Smoothness loss WITHOUT hard clamp (as recommended by research)
    """
    # First derivative (velocity)
    diff1 = pred_x0[:, :, 1:] - pred_x0[:, :, :-1]
    
    # Second derivative (acceleration)
    diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
    
    # L2 smoothness (NO clamp - let gradients flow!)
    smoothness_loss = torch.mean(diff2 ** 2)
    
    return smoothness_loss

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def diagnose_sample(original, generated, epoch, save_dir, prefix=""):
    """
    Complete diagnostic analysis of generated ECG
    
    Returns:
        dict with all metrics
    """
    print(f"\n{'='*70}")
    print(f"{prefix}DIAGNOSTIC ANALYSIS - EPOCH {epoch}")
    print(f"{'='*70}")
    
    # Basic statistics
    print(f"\n📊 Signal Statistics:")
    print(f"  Original  → Mean: {original.mean():7.2f}, Std: {original.std():7.2f}, Range: [{original.min():7.2f}, {original.max():7.2f}]")
    print(f"  Generated → Mean: {generated.mean():7.2f}, Std: {generated.std():7.2f}, Range: [{generated.min():7.2f}, {generated.max():7.2f}]")
    
    fs = 250  # Sampling frequency
    
    # ============================================
    # CHECK 1: PSD Analysis
    # ============================================
    f_orig, psd_orig = scipy_signal.welch(original, fs=fs, nperseg=512)
    f_gen, psd_gen = scipy_signal.welch(generated, fs=fs, nperseg=512)
    
    idx_50hz = np.argmin(np.abs(f_orig - 50))
    power_above_50 = np.sum(psd_gen[idx_50hz:])
    total_power = np.sum(psd_gen)
    noise_ratio = power_above_50 / (total_power + 1e-10)
    
    print(f"\n🔍 CHECK 1: Power Spectral Density")
    print(f"  Noise ratio (>50Hz): {noise_ratio*100:5.2f}%  ", end="")
    if noise_ratio < 0.15:
        print("✅ ECG-like!")
    elif noise_ratio < 0.30:
        print("⚠️  Moderate noise")
    else:
        print("❌ High noise")
    
    # ============================================
    # CHECK 2: NeuroKit2 R-peak Detection
    # ============================================
    try:
        import neurokit2 as nk
        
        print(f"\n🔍 CHECK 2: NeuroKit2 R-peak Detection")
        
        # Process original
        try:
            _, info_orig = nk.ecg_process(original, sampling_rate=fs)
            r_peaks_orig = info_orig['ECG_R_Peaks']
            hr_orig = len(r_peaks_orig) / 10 * 60
            print(f"  Original  → {len(r_peaks_orig):2d} peaks ({hr_orig:5.1f} BPM)")
        except:
            print(f"  Original  → Failed to detect")
            r_peaks_orig = []
        
        # Process generated
        try:
            _, info_gen = nk.ecg_process(generated, sampling_rate=fs)
            r_peaks_gen = info_gen['ECG_R_Peaks']
            hr_gen = len(r_peaks_gen) / 10 * 60
            print(f"  Generated → {len(r_peaks_gen):2d} peaks ({hr_gen:5.1f} BPM)  ", end="")
            
            if len(r_peaks_orig) > 0:
                ratio = len(r_peaks_gen) / len(r_peaks_orig)
                if 0.5 < ratio < 1.5:
                    print("✅ Good match!")
                else:
                    print(f"⚠️  Ratio: {ratio:.2f}x")
            else:
                print("✅ Structure detected!")
        except Exception as e:
            print(f"  Generated → ❌ Failed: {str(e)[:40]}")
    
    except ImportError:
        print(f"\n⚠️  NeuroKit2 not installed (pip install neurokit2)")
    
    # ============================================
    # CHECK 3: Phase Shift Analysis
    # ============================================
    print(f"\n🔍 CHECK 3: Phase Shift & Correlation")
    
    # Cross-correlation
    correlation_func = scipy_signal.correlate(generated, original, mode='full')
    lags = scipy_signal.correlation_lags(len(generated), len(original), mode='full')
    max_idx = np.argmax(correlation_func)
    lag_samples = lags[max_idx]
    lag_ms = lag_samples / fs * 1000
    
    # Pearson correlation
    pearson_raw = np.corrcoef(original, generated)[0, 1]
    
    # Align and recompute
    if lag_samples > 0:
        aligned = np.pad(generated[lag_samples:], (0, lag_samples), mode='edge')
    elif lag_samples < 0:
        lag_samples_abs = abs(lag_samples)
        aligned = np.pad(generated, (lag_samples_abs, 0), mode='edge')[:len(generated)]
    else:
        aligned = generated
    
    pearson_aligned = np.corrcoef(original, aligned)[0, 1]
    
    print(f"  Raw Pearson:     {pearson_raw:6.4f}")
    print(f"  Aligned Pearson: {pearson_aligned:6.4f}  (shift: {lag_ms:+6.1f} ms)  ", end="")
    
    if pearson_aligned > 0.7:
        print("✅ Excellent!")
    elif pearson_aligned > 0.5:
        print("✅ Good")
    elif pearson_aligned > 0.3:
        print("⚠️  Moderate")
    else:
        print("❌ Poor")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    create_diagnostic_plot(original, generated, aligned, epoch, save_dir,
                          psd_orig=(f_orig, psd_orig),
                          psd_gen=(f_gen, psd_gen),
                          noise_ratio=noise_ratio,
                          pearson_raw=pearson_raw,
                          pearson_aligned=pearson_aligned,
                          lag_ms=lag_ms)
    
    print(f"  📊 Plot saved: {save_dir}/plots/epoch_{epoch:03d}_diagnostics.png")
    print(f"{'='*70}\n")
    
    return {
        'noise_ratio': noise_ratio,
        'pearson_raw': pearson_raw,
        'pearson_aligned': pearson_aligned,
        'lag_ms': lag_ms
    }

def create_diagnostic_plot(original, generated, aligned, epoch, save_dir,
                          psd_orig, psd_gen, noise_ratio,
                          pearson_raw, pearson_aligned, lag_ms):
    """Enhanced visualization with all diagnostics"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Full signal overlay
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(original, label='Original', color='black', alpha=0.7, linewidth=1)
    ax1.plot(generated, label='Generated', color='#ff7f0e', alpha=0.8, linewidth=1)
    ax1.set_title(f'Epoch {epoch} - Full Signal (Pearson: {pearson_raw:.4f})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Amplitude')
    
    # Plot 2: Zoom (first 2 seconds)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(original[:500], label='Original', color='black', alpha=0.7, linewidth=1.5)
    ax2.plot(generated[:500], label='Generated', color='#ff7f0e', alpha=0.8, linewidth=1.5)
    ax2.set_title('Zoom: First 2 Seconds', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Amplitude')
    
    # Plot 3: After alignment
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(original[:500], label='Original', color='black', alpha=0.7, linewidth=1.5)
    ax3.plot(aligned[:500], label='Aligned', color='green', alpha=0.8, linewidth=1.5)
    ax3.set_title(f'After Alignment (shift: {lag_ms:+.1f}ms, r={pearson_aligned:.4f})', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Amplitude')
    
    # Plot 4: PSD
    ax4 = fig.add_subplot(gs[2, :])
    f_orig, psd_o = psd_orig
    f_gen, psd_g = psd_gen
    ax4.semilogy(f_orig, psd_o, label='Original PSD', color='black', alpha=0.7, linewidth=2)
    ax4.semilogy(f_gen, psd_g, label='Generated PSD', color='#ff7f0e', alpha=0.8, linewidth=2)
    ax4.axvline(50, color='red', linestyle='--', linewidth=2, label='50 Hz cutoff')
    ax4.set_xlim([0, 125])
    ax4.set_title(f'Power Spectral Density (Noise ratio: {noise_ratio*100:.1f}%)', fontsize=11)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Histogram
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.hist(original, bins=50, alpha=0.5, label='Original', color='black', density=True)
    ax5.hist(generated, bins=50, alpha=0.5, label='Generated', color='orange', density=True)
    ax5.set_title('Amplitude Distribution', fontsize=11)
    ax5.set_xlabel('Amplitude')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary metrics
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    
    summary_text = f"""
    SUMMARY METRICS
    ═══════════════════════════════
    
    Noise Ratio:      {noise_ratio*100:5.2f}%
    
    Pearson (raw):    {pearson_raw:6.4f}
    Pearson (aligned): {pearson_aligned:6.4f}
    
    Phase shift:      {lag_ms:+7.2f} ms
    
    Status:
    {'  ✅ ECG-like structure' if noise_ratio < 0.2 else '  ❌ High noise'}
    {'  ✅ Good correlation' if pearson_aligned > 0.6 else '  ⚠️  Moderate correlation' if pearson_aligned > 0.3 else '  ❌ Poor correlation'}
    {'  ✅ Good alignment' if abs(lag_ms) < 50 else '  ⚠️  Phase shift present'}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(f"{save_dir}/plots/epoch_{epoch:03d}_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN TRAINING FUNCTION (SINGLE FOLD)
# ============================================================================

def train_single_fold(
    dataset_path="./ecg_afib_data/X_combined.npy",
    weights_path="./best_model/best_model.pth",
    save_dir="./single_fold_debug",
    epochs=30,
    batch_size=16,
    val_split=0.2,
    use_ema=True,
    use_cfg=True,
    guidance_scale=3.0,
    learning_rate=5e-5
):
    """
    Clean single-fold training with diagnostics
    """
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║        SINGLE-FOLD ECG DIFFUSION WITH DIAGNOSTICS               ║
╚══════════════════════════════════════════════════════════════════╝

Configuration:
  Device:          {DEVICE}
  Epochs:          {epochs}
  Batch Size:      {batch_size}
  Learning Rate:   {learning_rate}
  EMA:             {'✅ Enabled' if use_ema else '❌ Disabled'}
  CFG:             {'✅ Enabled' if use_cfg else '❌ Disabled'}
  Guidance Scale:  {guidance_scale}
  
""")
    
    # ========================================
    # DATA LOADING
    # ========================================
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    signals_raw = np.load(dataset_path)
    
    # Z-score normalization (per-sample)
    signals_normalized = (signals_raw - signals_raw.mean(1, keepdims=True)) / \
                        (signals_raw.std(1, keepdims=True) + 1e-6)
    
    print(f"Dataset size: {len(signals_normalized):,}")
    print(f"Signal shape: {signals_normalized.shape}")
    
    # Save normalization params
    norm_params = {
        'means': signals_raw.mean(1),
        'stds': signals_raw.std(1)
    }
    np.save(f"{save_dir}/norm_params.npy", norm_params)
    
    # Train/Val split
    n_samples = len(signals_normalized)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = signals_normalized[train_indices]
    val_data = signals_normalized[val_indices]
    
    # Create datasets
    class ECGDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
    
    train_dataset = ECGDataset(train_data)
    val_dataset = ECGDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_data):,}")
    print(f"Val samples:   {len(val_data):,}")
    
    # ========================================
    # MODEL INITIALIZATION
    # ========================================
    print("\n" + "="*70)
    print("INITIALIZING MODELS")
    print("="*70)
    
    unet = UNet1DConditional().to(DEVICE)
    style_net = StyleEncoderWrapper(weights_path, DEVICE)
    content_net = PretrainedContentEncoder().to(DEVICE)
    content_net.eval()
    
    style_proj = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512)
    ).to(DEVICE)
    
    for layer in style_proj:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    
    # EMA
    if use_ema:
        model_container = nn.ModuleList([unet, style_proj])
        ema = EMA(model_container, decay=0.9999)
        ema.register()
        print("✅ EMA initialized")
    else:
        ema = None
        print("⚠️  EMA disabled")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(style_proj.parameters()),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # Scheduler
    diffusion_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False
    )
    
    # Validation sample
    val_sample_idx = val_indices[0]
    val_sample_raw = signals_raw[val_sample_idx]  # Save original for denorm
    val_sample_norm = signals_normalized[val_sample_idx]
    val_sample_tensor = torch.tensor(val_sample_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    print("✅ Models initialized")
    
    # ========================================
    # TRAINING LOOP
    # ========================================
    print("\n" + "="*70)
    print(f"STARTING TRAINING ({epochs} EPOCHS)")
    print("="*70)
    
    best_pearson = -1.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # ====================================
        # TRAINING
        # ====================================
        unet.train()
        style_proj.train()
        train_losses = {'total': [], 'mse': [], 'freq': [], 'smooth': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs}", leave=False)
        for batch in pbar:
            x = batch.to(DEVICE)
            
            with torch.no_grad():
                style = style_net(x)
                content = content_net(x)
            
            style_emb = style_proj(style).unsqueeze(1)
            conditioning = torch.cat([content, style_emb], dim=1)
            
            # CFG dropout
            if use_cfg and np.random.random() < 0.1:
                conditioning = torch.zeros_like(conditioning)
            
            noise = torch.randn_like(x)
            t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
            noisy_x = diffusion_scheduler.add_noise(x, noise, t)
            
            pred_noise = unet(noisy_x, t, conditioning)
            
            # Losses
            mse_loss = F.mse_loss(pred_noise, noise)
            freq_loss = compute_frequency_loss(pred_noise, noise)
            
            # Predict clean signal for smoothness
            alpha_prod_t = diffusion_scheduler.alphas_cumprod[t][:, None, None]
            beta_prod_t = 1 - alpha_prod_t
            pred_clean = (noisy_x - beta_prod_t.sqrt() * pred_noise) / alpha_prod_t.sqrt()
            smooth_loss = compute_temporal_smoothness_loss_unclamped(pred_clean)
            
            # Total loss
            loss = mse_loss + 0.3 * freq_loss + 0.01 * smooth_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            
            if use_ema:
                ema.update()
            
            train_losses['total'].append(loss.item())
            train_losses['mse'].append(mse_loss.item())
            train_losses['freq'].append(freq_loss.item())
            train_losses['smooth'].append(smooth_loss.item())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_losses = {k: np.mean(v) for k, v in train_losses.items()}
        
        # ====================================
        # VALIDATION
        # ====================================
        if use_ema:
            ema.apply_shadow()
        
        unet.eval()
        style_proj.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(DEVICE)
                
                style = style_net(x)
                content = content_net(x)
                style_emb = style_proj(style).unsqueeze(1)
                conditioning = torch.cat([content, style_emb], dim=1)
                
                noise = torch.randn_like(x)
                t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
                noisy_x = diffusion_scheduler.add_noise(x, noise, t)
                
                pred_noise = unet(noisy_x, t, conditioning)
                loss = F.mse_loss(pred_noise, noise)
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # ====================================
        # SAMPLE & DIAGNOSE
        # ====================================
        with torch.no_grad():
            # Generate sample
            style = style_net(val_sample_tensor)
            content = content_net(val_sample_tensor)
            style_emb = style_proj(style).unsqueeze(1)
            
            if use_cfg:
                cond_input = torch.cat([content, style_emb], dim=1)
                uncond_input = torch.zeros_like(cond_input)
            else:
                cond_input = torch.cat([content, style_emb], dim=1)
            
            # DDIM sampling
            test_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
                clip_sample=False
            )
            test_scheduler.set_timesteps(50)
            
            latents = torch.randn_like(val_sample_tensor)
            
            for t in test_scheduler.timesteps:
                if use_cfg:
                    latent_input = torch.cat([latents] * 2)
                    t_input = torch.cat([t.unsqueeze(0).to(DEVICE)] * 2)
                    cond = torch.cat([cond_input, uncond_input])
                    
                    noise_pred = unet(latent_input, t_input, cond)
                    noise_cond, noise_uncond = noise_pred.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = unet(latents, t.unsqueeze(0).to(DEVICE), cond_input)
                
                latents = test_scheduler.step(noise_pred, t, latents).prev_sample
            
            generated_norm = latents.squeeze().cpu().numpy()
        
        # Denormalize both
        original_denorm = val_sample_raw
        generated_denorm = generated_norm * norm_params['stds'][val_sample_idx] + norm_params['means'][val_sample_idx]
        
        # Diagnose
        metrics = diagnose_sample(original_denorm, generated_denorm, epoch, save_dir)
        
        # ====================================
        # LOGGING
        # ====================================
        lr_scheduler.step()
        
        print(f"\n📊 Epoch {epoch:2d}: "
              f"Train={avg_losses['total']:.4f} "
              f"(MSE:{avg_losses['mse']:.4f}, Freq:{avg_losses['freq']:.4f}, Smooth:{avg_losses['smooth']:.4f}) | "
              f"Val={avg_val_loss:.4f} | "
              f"Pearson(aligned)={metrics['pearson_aligned']:.4f}")
        
        # Save best model
        if metrics['pearson_aligned'] > best_pearson:
            best_pearson = metrics['pearson_aligned']
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'style_proj_state_dict': style_proj.state_dict(),
                'metrics': metrics,
                'val_loss': avg_val_loss
            }, f"{save_dir}/best_model.pth")
            
            print(f"   ✅ New best! Saved (Pearson: {best_pearson:.4f})")
        
        if use_ema:
            ema.restore()
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Pearson (aligned): {best_pearson:.4f}")
    print(f"\nResults saved in: {save_dir}/")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    train_single_fold(
        dataset_path="./ecg_afib_data/X_combined.npy",
        weights_path="./best_model/best_model.pth",
        save_dir="./single_fold_debug",
        epochs=30,
        batch_size=16,
        use_ema=True,
        use_cfg=True,
        guidance_scale=3.0,
        learning_rate=5e-5
    )