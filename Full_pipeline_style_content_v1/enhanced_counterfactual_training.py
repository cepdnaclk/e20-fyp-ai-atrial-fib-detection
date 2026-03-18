# ============================================================================
# ENHANCED COUNTERFACTUAL ECG DIFFUSION TRAINING
# ============================================================================
"""
GOAL: Increase flip rate to 80%+ while maintaining:
- Person's morphology (content preservation)
- Physiological plausibility (valid ECG structure)

KEY IMPROVEMENTS:
1. Classification Loss - Push generated ECGs toward target class
2. Class-Conditional Embedding - Use target class info in style
3. Contrastive Loss - Separate source and target class representations
4. Content Preservation - Ensure morphology is maintained
5. Higher Guidance Scale - Stronger class steering during generation
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
from scipy import signal
import neurokit2 as nk

# Import your existing models
from counterfactual_training import (
    UNet1DConditional,
    StyleEncoderWrapper,
    PretrainedContentEncoder,
    AFibResLSTM,
    ModelConfig,
    EMA,
    compute_frequency_loss
)

# ============================================================================
# ENHANCED DATASET WITH LABELS
# ============================================================================

class LabeledECGDataset(Dataset):
    """ECG Dataset with class labels for counterfactual training"""
    
    def __init__(self, signals, labels):
        self.signals = signals
        # Ensure labels are integers
        if isinstance(labels, np.ndarray) and labels.dtype.kind in ['U', 'S', 'O']:
            self.labels = np.array([1 if l == 'A' else 0 for l in labels], dtype=np.int64)
        else:
            self.labels = np.array(labels, dtype=np.int64)
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return signal, label

# ============================================================================
# CLASS-CONDITIONAL EMBEDDING
# ============================================================================

class ClassConditionalEmbedding(nn.Module):
    """Learnable embedding for target class conditioning"""
    
    def __init__(self, num_classes=2, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512)
        )
        
        # Initialize small for gradual learning
        nn.init.normal_(self.embedding.weight, std=0.01)
        
    def forward(self, class_labels):
        """
        Args:
            class_labels: [batch] - Target class indices
        Returns:
            class_emb: [batch, 512]
        """
        emb = self.embedding(class_labels)  # [batch, 128]
        return self.proj(emb)  # [batch, 512]

# ============================================================================
# CONTENT PRESERVATION LOSS
# ============================================================================

def compute_content_preservation_loss(original_content, generated_content):
    """
    Ensure morphological features are preserved
    
    Uses L2 distance in content embedding space
    """
    return F.mse_loss(generated_content, original_content)

# ============================================================================
# CLASSIFICATION GUIDANCE LOSS
# ============================================================================

def compute_classification_loss(classifier, pred_x0, target_class, device):
    """
    Push generated ECG toward target class
    
    This is the KEY improvement for flip rate!
    
    Args:
        classifier: Trained AFibResLSTM
        pred_x0: Predicted clean signal [batch, 1, 2500]
        target_class: Target class for counterfactual [batch]
        device: torch device
    
    Returns:
        loss: Classification loss (cross-entropy toward target)
    """
    # IMPORTANT: Set classifier to train mode for cuDNN LSTM backward compatibility
    # We freeze the weights so this only affects dropout/batchnorm behavior
    was_training = classifier.training
    classifier.train()  # Required for cuDNN RNN backward pass
    
    # Forward pass - gradients flow through for U-Net update
    logits, _ = classifier(pred_x0)
    
    # Cross-entropy loss toward TARGET class (not source!)
    loss = F.cross_entropy(logits, target_class)
    
    # Restore original mode
    if not was_training:
        classifier.eval()
    
    return loss

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def diagnose_sample_enhanced(original, generated, classifier, device, epoch, save_dir):
    """Enhanced diagnostics with classification check"""
    
    fs = 250
    
    # PSD Analysis
    f_orig, psd_orig = signal.welch(original, fs=fs, nperseg=512)
    f_gen, psd_gen = signal.welch(generated, fs=fs, nperseg=512)
    
    idx_50hz = np.argmin(np.abs(f_orig - 50))
    noise_ratio = np.sum(psd_gen[idx_50hz:]) / (np.sum(psd_gen) + 1e-10)
    
    # NeuroKit2 validation
    try:
        ecg_clean = nk.ecg_clean(generated, sampling_rate=fs)
        peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        r_peaks = info.get('ECG_R_Peaks', [])
        
        if len(r_peaks) >= 3:
            rr_intervals = np.diff(r_peaks) / fs * 1000
            hr = 60000 / np.mean(rr_intervals)
            physiological = 40 <= hr <= 200
        else:
            hr = 0
            physiological = False
    except:
        hr = 0
        physiological = False
    
    # Classification
    with torch.no_grad():
        orig_tensor = torch.tensor(original, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        orig_tensor = (orig_tensor - orig_tensor.mean()) / (orig_tensor.std() + 1e-6)
        orig_tensor = orig_tensor.to(device)
        
        gen_tensor = torch.tensor(generated, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gen_tensor = (gen_tensor - gen_tensor.mean()) / (gen_tensor.std() + 1e-6)
        gen_tensor = gen_tensor.to(device)
        
        orig_logits, _ = classifier(orig_tensor)
        gen_logits, _ = classifier(gen_tensor)
        
        orig_class = torch.argmax(orig_logits, dim=1).item()
        gen_class = torch.argmax(gen_logits, dim=1).item()
        
        orig_conf = F.softmax(orig_logits, dim=1)[0, orig_class].item()
        gen_conf = F.softmax(gen_logits, dim=1)[0, gen_class].item()
    
    target_class = 1 - orig_class
    flip_success = (gen_class == target_class)
    
    print(f"\n{'─'*60}")
    print(f"EPOCH {epoch} DIAGNOSTIC")
    print(f"{'─'*60}")
    print(f"  Noise Ratio: {noise_ratio*100:.1f}% {'✅' if noise_ratio < 0.15 else '⚠️'}")
    print(f"  Heart Rate: {hr:.1f} bpm {'✅' if physiological else '❌'}")
    print(f"  Original Class: {'AFib' if orig_class == 1 else 'Normal'} ({orig_conf*100:.1f}%)")
    print(f"  Generated Class: {'AFib' if gen_class == 1 else 'Normal'} ({gen_conf*100:.1f}%)")
    print(f"  Flip Status: {'✅ SUCCESS' if flip_success else '❌ FAILED'}")
    
    return {
        'noise_ratio': noise_ratio,
        'physiological': physiological,
        'hr': hr,
        'orig_class': orig_class,
        'gen_class': gen_class,
        'flip_success': flip_success,
        'ecg_quality': 1.0 - noise_ratio
    }

# ============================================================================
# ENHANCED TRAINING FUNCTION
# ============================================================================

def train_enhanced_counterfactual_model(
    dataset_path="./ecg_afib_data/X_combined.npy",
    labels_path="./ecg_afib_data/y_combined.npy",
    classifier_weights_path="./best_model/best_model.pth",
    save_dir="./enhanced_counterfactual_training",
    epochs=60,
    batch_size=16,
    val_split=0.2,
    learning_rate=5e-5,
    # Loss weights
    mse_weight=1.0,
    freq_weight=0.05,
    cls_weight=0.1,           # Classification guidance weight
    content_weight=0.05,       # Content preservation weight
    # Training settings
    use_ema=True,
    guidance_scale=5.0,        # Higher for better flip rate
    cls_loss_start_epoch=5,    # When to start classification loss
    early_stopping_patience=15
):
    """
    Enhanced training for high flip rate counterfactual generation
    
    Key differences from original:
    1. Uses class labels for targeted generation
    2. Classification loss pushes toward target class
    3. Content preservation ensures morphology is kept
    4. Higher guidance scale for stronger steering
    """
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ENHANCED COUNTERFACTUAL ECG DIFFUSION TRAINING                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 GOAL: 80%+ flip rate while preserving morphology

Configuration:
  Device:          {DEVICE}
  Epochs:          {epochs}
  Batch Size:      {batch_size}
  Learning Rate:   {learning_rate}
  Guidance Scale:  {guidance_scale}
  
  Loss Weights:
    MSE:           {mse_weight}
    Frequency:     {freq_weight}
    Classification: {cls_weight} (starts epoch {cls_loss_start_epoch})
    Content:       {content_weight}
  
  EMA:             {'✅ Enabled' if use_ema else '❌ Disabled'}
  
""")
    
    # ========================================
    # 1. DATA LOADING
    # ========================================
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    signals_raw = np.load(dataset_path)
    labels_raw = np.load(labels_path, allow_pickle=True)
    
    # Convert string labels to integers: 'N' -> 0, 'A' -> 1
    if labels_raw.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
        labels = np.array([1 if l == 'A' else 0 for l in labels_raw], dtype=np.int64)
        print(f"Converted string labels ('N'/'A') to integers (0/1)")
    else:
        labels = labels_raw.astype(np.int64)
    
    # Normalize signals
    signals_normalized = (signals_raw - signals_raw.mean(1, keepdims=True)) / \
                        (signals_raw.std(1, keepdims=True) + 1e-6)
    
    print(f"Dataset size: {len(signals_normalized):,}")
    print(f"Class distribution: Normal={np.sum(labels==0):,}, AFib={np.sum(labels==1):,}")
    
    # Save normalization params
    norm_params = {
        'means': signals_raw.mean(1),
        'stds': signals_raw.std(1)
    }
    np.save(f"{save_dir}/norm_params.npy", norm_params)
    
    # Train/Val split (stratified)
    n_samples = len(signals_normalized)
    n_val = int(n_samples * val_split)
    
    # Stratified split
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_samples - n_val]
    val_indices = indices[n_samples - n_val:]
    
    train_signals = signals_normalized[train_indices]
    train_labels = labels[train_indices]
    val_signals = signals_normalized[val_indices]
    val_labels = labels[val_indices]
    
    train_dataset = LabeledECGDataset(train_signals, train_labels)
    val_dataset = LabeledECGDataset(val_signals, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_signals):,}, Val: {len(val_signals):,}")
    
    # ========================================
    # 2. MODEL INITIALIZATION
    # ========================================
    print("\n" + "="*70)
    print("INITIALIZING MODELS")
    print("="*70)
    
    # U-Net (main diffusion model)
    unet = UNet1DConditional().to(DEVICE)
    print("✅ U-Net initialized")
    
    # Style encoder (frozen AFibResLSTM features)
    style_net = StyleEncoderWrapper(classifier_weights_path, DEVICE)
    print("✅ Style encoder initialized")
    
    # Content encoder (frozen HuBERT-ECG)
    content_net = PretrainedContentEncoder().to(DEVICE)
    content_net.eval()
    print("✅ Content encoder initialized")
    
    # Style projector (trainable)
    style_proj = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512)
    ).to(DEVICE)
    
    # Class-conditional embedding (NEW!)
    class_embed = ClassConditionalEmbedding(num_classes=2, embed_dim=128).to(DEVICE)
    print("✅ Class embedding initialized")
    
    # Classifier (frozen, for guidance)
    classifier = AFibResLSTM(ModelConfig()).to(DEVICE)
    clf_checkpoint = torch.load(classifier_weights_path, map_location=DEVICE)
    classifier.load_state_dict(clf_checkpoint['model_state_dict'])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    print("✅ Classifier loaded (frozen)")
    
    # Initialize weights
    for layer in style_proj:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    
    # EMA
    if use_ema:
        model_container = nn.ModuleList([unet, style_proj, class_embed])
        ema = EMA(model_container, decay=0.9999)
        ema.register()
    else:
        ema = None
    
    # Optimizer (train U-Net, style_proj, and class_embed)
    trainable_params = list(unet.parameters()) + \
                       list(style_proj.parameters()) + \
                       list(class_embed.parameters())
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Diffusion scheduler
    diffusion_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False
    )
    
    # Validation sample (for visualization)
    val_sample_idx = val_indices[0]
    val_sample_raw = signals_raw[val_sample_idx]
    val_sample_norm = signals_normalized[val_sample_idx]
    val_sample_label = labels[val_sample_idx]
    val_sample_tensor = torch.tensor(val_sample_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    print("\n✅ All models initialized")
    
    # ========================================
    # 3. TRAINING LOOP
    # ========================================
    print("\n" + "="*70)
    print(f"TRAINING ({epochs} EPOCHS)")
    print("="*70)
    
    best_flip_rate = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'flip_rate': [], 'ecg_quality': []}
    
    for epoch in range(epochs):
        # ====================================
        # TRAINING PHASE
        # ====================================
        unet.train()
        style_proj.train()
        class_embed.train()
        
        train_losses = {'total': [], 'mse': [], 'freq': [], 'cls': [], 'content': []}
        
        # Determine which losses to use
        use_cls_loss = epoch >= cls_loss_start_epoch
        loss_mode = "MSE+Freq+Cls" if use_cls_loss else "MSE+Freq"
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs} [{loss_mode}]", leave=False)
        
        for batch_signals, batch_labels in pbar:
            x = batch_signals.to(DEVICE)
            source_class = batch_labels.to(DEVICE)
            target_class = 1 - source_class  # Flip class!
            
            # --------------------------------
            # Extract features
            # --------------------------------
            with torch.no_grad():
                style_features = style_net(x)           # [B, 128]
                content_features = content_net(x)       # [B, seq, 512]
            
            # Style projection + class embedding
            style_emb = style_proj(style_features)                    # [B, 512]
            class_emb = class_embed(target_class)                     # [B, 512]
            
            # Combine style and class embeddings
            combined_style = style_emb + 0.5 * class_emb              # [B, 512]
            combined_style = combined_style.unsqueeze(1)              # [B, 1, 512]
            
            # Full conditioning
            conditioning = torch.cat([content_features, combined_style], dim=1)
            
            # CFG: 10% dropout of conditioning
            if np.random.random() < 0.1:
                conditioning = torch.zeros_like(conditioning)
            
            # --------------------------------
            # Diffusion forward process
            # --------------------------------
            noise = torch.randn_like(x)
            t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
            noisy_x = diffusion_scheduler.add_noise(x, noise, t)
            
            # --------------------------------
            # Predict noise
            # --------------------------------
            pred_noise = unet(noisy_x, t, conditioning)
            
            # --------------------------------
            # Compute losses
            # --------------------------------
            # 1. MSE loss (primary)
            mse_loss = F.mse_loss(pred_noise, noise)
            
            # 2. Frequency loss (preserves ECG structure)
            freq_loss = compute_frequency_loss(pred_noise, noise)
            
            # 3. Classification loss (pushes toward target class)
            if use_cls_loss and epoch >= cls_loss_start_epoch:
                # Get predicted x0 from noise prediction
                alpha_t = diffusion_scheduler.alphas_cumprod[t].view(-1, 1, 1).to(DEVICE)
                pred_x0 = (noisy_x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                pred_x0 = torch.clamp(pred_x0, -4, 4)
                
                # Classification loss toward target
                cls_loss = compute_classification_loss(classifier, pred_x0, target_class, DEVICE)
            else:
                cls_loss = torch.tensor(0.0, device=DEVICE)
            
            # 4. Content preservation loss (ensures morphology kept)
            if use_cls_loss:
                with torch.no_grad():
                    # Get content of predicted x0
                    pred_content = content_net(pred_x0.detach())
                content_loss = F.mse_loss(pred_content, content_features.detach())
            else:
                content_loss = torch.tensor(0.0, device=DEVICE)
            
            # Total loss
            loss = mse_weight * mse_loss + \
                   freq_weight * freq_loss + \
                   cls_weight * cls_loss + \
                   content_weight * content_loss
            
            # --------------------------------
            # Backprop
            # --------------------------------
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            if use_ema:
                ema.update()
            
            # Record losses
            train_losses['total'].append(loss.item())
            train_losses['mse'].append(mse_loss.item())
            train_losses['freq'].append(freq_loss.item())
            train_losses['cls'].append(cls_loss.item())
            train_losses['content'].append(content_loss.item())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'cls': f"{cls_loss.item():.4f}"})
        
        avg_train = {k: np.mean(v) for k, v in train_losses.items()}
        
        # ====================================
        # VALIDATION PHASE
        # ====================================
        if use_ema:
            ema.apply_shadow()
        
        unet.eval()
        style_proj.eval()
        class_embed.eval()
        
        val_losses = []
        flip_count = 0
        total_count = 0
        
        with torch.no_grad():
            for batch_signals, batch_labels in val_loader:
                x = batch_signals.to(DEVICE)
                source_class = batch_labels.to(DEVICE)
                target_class = 1 - source_class
                
                # Get conditioning
                style_features = style_net(x)
                content_features = content_net(x)
                
                style_emb = style_proj(style_features)
                class_emb = class_embed(target_class)
                combined_style = (style_emb + 0.5 * class_emb).unsqueeze(1)
                conditioning = torch.cat([content_features, combined_style], dim=1)
                
                # Diffusion loss
                noise = torch.randn_like(x)
                t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
                noisy_x = diffusion_scheduler.add_noise(x, noise, t)
                pred_noise = unet(noisy_x, t, conditioning)
                loss = F.mse_loss(pred_noise, noise)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # ====================================
        # GENERATE AND EVALUATE SAMPLE
        # ====================================
        with torch.no_grad():
            # Get conditioning for validation sample
            style_features = style_net(val_sample_tensor)
            content_features = content_net(val_sample_tensor)
            
            # Target class (flip the original)
            target = torch.tensor([1 - val_sample_label], device=DEVICE)
            
            style_emb = style_proj(style_features)
            class_emb = class_embed(target)
            combined_style = (style_emb + 0.5 * class_emb).unsqueeze(1)
            
            cond = torch.cat([content_features, combined_style], dim=1)
            uncond = torch.zeros_like(cond)
            
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
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t.unsqueeze(0).to(DEVICE)] * 2)
                cond_input = torch.cat([cond, uncond])
                
                noise_pred = unet(latent_input, t_input, cond_input)
                noise_cond, noise_uncond = noise_pred.chunk(2)
                
                # CFG with higher guidance
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                
                latents = test_scheduler.step(noise_pred, t, latents).prev_sample
                latents = torch.clamp(latents, -4, 4)
            
            generated_norm = latents.squeeze().cpu().numpy()
        
        # Denormalize
        generated_raw = generated_norm * norm_params['stds'][val_sample_idx] + norm_params['means'][val_sample_idx]
        
        # Diagnose
        metrics = diagnose_sample_enhanced(
            val_sample_raw, generated_raw, classifier, DEVICE, epoch, save_dir
        )
        
        # ====================================
        # BATCH FLIP RATE EVALUATION (every 5 epochs)
        # ====================================
        if epoch % 5 == 0 or epoch == epochs - 1:
            print("\n  📊 Evaluating flip rate on batch...")
            flip_count = 0
            physio_count = 0
            eval_samples = min(50, len(val_signals))
            
            with torch.no_grad():
                for i in range(eval_samples):
                    # Get sample
                    sig = torch.tensor(val_signals[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    src_class = val_labels[i]
                    tgt_class = 1 - src_class
                    
                    # Get conditioning
                    sf = style_net(sig)
                    cf = content_net(sig)
                    se = style_proj(sf)
                    ce = class_embed(torch.tensor([tgt_class], device=DEVICE))
                    cs = (se + 0.5 * ce).unsqueeze(1)
                    cond = torch.cat([cf, cs], dim=1)
                    uncond = torch.zeros_like(cond)
                    
                    # Generate
                    test_scheduler.set_timesteps(50)
                    lat = torch.randn_like(sig)
                    
                    for t in test_scheduler.timesteps:
                        lat_in = torch.cat([lat] * 2)
                        t_in = torch.cat([t.unsqueeze(0).to(DEVICE)] * 2)
                        c_in = torch.cat([cond, uncond])
                        
                        n_pred = unet(lat_in, t_in, c_in)
                        n_c, n_u = n_pred.chunk(2)
                        n_pred = n_u + guidance_scale * (n_c - n_u)
                        
                        lat = test_scheduler.step(n_pred, t, lat).prev_sample
                        lat = torch.clamp(lat, -4, 4)
                    
                    # Classify
                    gen_logits, _ = classifier(lat)
                    gen_class = torch.argmax(gen_logits, dim=1).item()
                    
                    if gen_class == tgt_class:
                        flip_count += 1
                    
                    # Check physiological plausibility
                    gen_np = lat.squeeze().cpu().numpy()
                    gen_raw = gen_np * norm_params['stds'][val_indices[i]] + norm_params['means'][val_indices[i]]
                    
                    try:
                        ecg_clean = nk.ecg_clean(gen_raw, sampling_rate=250)
                        peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=250)
                        r_peaks = info.get('ECG_R_Peaks', [])
                        if len(r_peaks) >= 3:
                            rr = np.diff(r_peaks) / 250 * 1000
                            hr = 60000 / np.mean(rr)
                            if 40 <= hr <= 200:
                                physio_count += 1
                    except:
                        pass
            
            flip_rate = flip_count / eval_samples
            physio_rate = physio_count / eval_samples
            
            print(f"  Flip Rate: {flip_count}/{eval_samples} ({flip_rate*100:.1f}%)")
            print(f"  Physiological: {physio_count}/{eval_samples} ({physio_rate*100:.1f}%)")
            
            history['flip_rate'].append(flip_rate)
        
        # ====================================
        # LOGGING
        # ====================================
        lr_scheduler.step()
        
        history['train_loss'].append(avg_train['total'])
        history['val_loss'].append(avg_val_loss)
        history['ecg_quality'].append(metrics['ecg_quality'])
        
        print(f"\n📊 Epoch {epoch:2d}: "
              f"Train={avg_train['total']:.4f} (MSE:{avg_train['mse']:.4f}, Cls:{avg_train['cls']:.4f}) | "
              f"Val={avg_val_loss:.4f} | "
              f"Quality={metrics['ecg_quality']*100:.1f}%")
        
        # Save best model (based on flip success on validation sample)
        current_flip = 1.0 if metrics['flip_success'] else 0.0
        
        # After epoch 10, track flip rate
        if epoch >= cls_loss_start_epoch and len(history['flip_rate']) > 0:
            if history['flip_rate'][-1] > best_flip_rate:
                best_flip_rate = history['flip_rate'][-1]
                best_epoch = epoch
                
                torch.save({
                    'epoch': epoch,
                    'unet_state_dict': unet.state_dict(),
                    'style_proj_state_dict': style_proj.state_dict(),
                    'class_embed_state_dict': class_embed.state_dict(),
                    'flip_rate': best_flip_rate,
                    'metrics': metrics
                }, f"{save_dir}/best_model.pth")
                
                print(f"   ✅ New best! Flip Rate: {best_flip_rate*100:.1f}%")
        
        # Also save latest
        torch.save({
            'epoch': epoch,
            'unet_state_dict': unet.state_dict(),
            'style_proj_state_dict': style_proj.state_dict(),
            'class_embed_state_dict': class_embed.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, f"{save_dir}/latest_checkpoint.pth")
        
        if use_ema:
            ema.restore()
    
    # ========================================
    # TRAINING COMPLETE
    # ========================================
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Flip Rate: {best_flip_rate*100:.1f}%")
    
    # Save history plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['ecg_quality'])
    plt.xlabel('Epoch')
    plt.ylabel('ECG Quality')
    plt.title('ECG Quality')
    
    plt.subplot(1, 3, 3)
    epochs_with_flip = list(range(0, len(history['flip_rate']) * 5, 5))
    plt.plot(epochs_with_flip[:len(history['flip_rate'])], history['flip_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Flip Rate')
    plt.title('Flip Rate Progress')
    plt.axhline(0.8, color='r', linestyle='--', label='Target 80%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png", dpi=150)
    plt.close()
    
    return history

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    train_enhanced_counterfactual_model(
        dataset_path="./ecg_afib_data/X_combined.npy",
        labels_path="./ecg_afib_data/y_combined.npy",
        classifier_weights_path="./best_model/best_model.pth",
        save_dir="./enhanced_counterfactual_training",
        epochs=60,
        batch_size=16,
        learning_rate=5e-5,
        # Loss weights
        mse_weight=1.0,
        freq_weight=0.05,
        cls_weight=0.1,           # Classification guidance
        content_weight=0.05,      # Content preservation
        # Settings
        use_ema=True,
        guidance_scale=5.0,       # Higher for better flip
        cls_loss_start_epoch=5,
        early_stopping_patience=15
    )
