"""
MINIMAL INTERVENTION COUNTERFACTUAL ECG GENERATION
===================================================

This approach uses DDIM INVERSION to:
1. Encode the ORIGINAL ECG into the noise space (deterministic)
2. Decode with TARGET CLASS conditioning
3. Result: Minimal changes that flip the class

This is fundamentally different from the previous approach which
started from random noise and generated a completely new ECG.

Key Concept:
- DDIM is deterministic (same noise → same output for same conditioning)
- We can "invert" the process: find the noise that would generate the original ECG
- Then decode that noise with different class conditioning
- The result will be the "closest" ECG in the target class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from tqdm import tqdm
import os
import json
from datetime import datetime
from scipy.stats import pearsonr
from scipy import signal as scipy_signal

# Import models
from counterfactual_training import (
    UNet1DConditional,
    StyleEncoderWrapper,
    PretrainedContentEncoder,
    AFibResLSTM,
    ModelConfig
)
from enhanced_counterfactual_training import ClassConditionalEmbedding

# ============================================================================
# DDIM INVERSION
# ============================================================================

def ddim_inversion(
    unet,
    scheduler,
    latents,  # The original ECG
    conditioning,
    num_inference_steps=50,
    guidance_scale=1.0  # No guidance during inversion
):
    """
    Invert the DDIM process to find the noise that would generate this ECG.
    
    This is the key to preserving morphology:
    - Instead of starting from random noise
    - We find the noise representation of the ORIGINAL ECG
    - Then we can decode with different conditioning
    
    Args:
        unet: The diffusion model
        scheduler: DDIM scheduler
        latents: Original ECG [batch, channels, length]
        conditioning: Content + style conditioning
        num_inference_steps: Number of inversion steps
        
    Returns:
        inverted_noise: The noise that would generate this ECG
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Reverse the timesteps for inversion
    timesteps = scheduler.timesteps.flip(0)
    
    # Start from the original signal
    inverted = latents.clone()
    
    for i, t in enumerate(tqdm(timesteps, desc="DDIM Inversion", leave=False)):
        # Predict noise at this timestep
        with torch.no_grad():
            t_tensor = t.unsqueeze(0).to(latents.device)
            noise_pred = unet(inverted, t_tensor, conditioning)
        
        # Get alpha values (ensure on correct device)
        alpha_prod_t = scheduler.alphas_cumprod[t].to(latents.device)
        
        if i < len(timesteps) - 1:
            alpha_prod_t_next = scheduler.alphas_cumprod[timesteps[i + 1]].to(latents.device)
        else:
            alpha_prod_t_next = torch.tensor(1.0, device=latents.device)
        
        # DDIM inversion step (reverse of sampling)
        # x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        # We want to go from x_t to x_{t+1} (more noisy)
        
        # Predicted x_0
        pred_x0 = (inverted - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prod_t_next) * noise_pred
        
        # x_{t+1}
        inverted = torch.sqrt(alpha_prod_t_next) * pred_x0 + dir_xt
    
    return inverted

def ddim_sample_from_noise(
    unet,
    scheduler,
    noise,  # Starting noise (from inversion)
    conditioning,
    uncond_conditioning,
    num_inference_steps=50,
    guidance_scale=3.0
):
    """
    Sample from inverted noise with new conditioning.
    
    Args:
        noise: Inverted noise representation of original ECG
        conditioning: New conditioning (with target class)
        uncond_conditioning: Unconditional embedding for CFG
        
    Returns:
        counterfactual: Generated ECG
    """
    scheduler.set_timesteps(num_inference_steps)
    
    latents = noise.clone()
    
    for t in tqdm(scheduler.timesteps, desc="DDIM Sampling", leave=False):
        # CFG: predict both conditional and unconditional
        latent_input = torch.cat([latents] * 2)
        t_input = torch.cat([t.unsqueeze(0)] * 2).to(latents.device)
        cond_input = torch.cat([conditioning, uncond_conditioning])
        
        with torch.no_grad():
            noise_pred = unet(latent_input, t_input, cond_input)
        
        # Apply CFG
        noise_cond, noise_uncond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # DDIM step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents

# ============================================================================
# MINIMAL INTERVENTION GENERATOR
# ============================================================================

class MinimalInterventionGenerator:
    """
    Generate counterfactuals using DDIM inversion for minimal changes.
    """
    
    def __init__(
        self,
        checkpoint_path,
        classifier_path,
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Initializing Minimal Intervention Generator on {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # U-Net
        self.unet = UNet1DConditional().to(self.device)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.unet.eval()
        print("✅ U-Net loaded")
        
        # Style projector
        self.style_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        ).to(self.device)
        self.style_proj.load_state_dict(checkpoint['style_proj_state_dict'])
        self.style_proj.eval()
        
        # Class embedding
        self.class_embed = ClassConditionalEmbedding(num_classes=2, embed_dim=128).to(self.device)
        self.class_embed.load_state_dict(checkpoint['class_embed_state_dict'])
        self.class_embed.eval()
        print("✅ Class embedding loaded")
        
        # Style encoder
        self.style_net = StyleEncoderWrapper(classifier_path, self.device)
        self.style_net.eval()
        
        # Content encoder
        self.content_net = PretrainedContentEncoder().to(self.device)
        self.content_net.eval()
        print("✅ Content encoder loaded")
        
        # Classifier
        config = ModelConfig()
        self.classifier = AFibResLSTM(config).to(self.device)
        clf_checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(clf_checkpoint['model_state_dict'])
        self.classifier.eval()
        print("✅ Classifier loaded")
        
        # Scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False
        )
        
        print("✅ All models loaded!\n")
    
    def classify(self, ecg_tensor):
        """Classify ECG and return class, confidence, probabilities"""
        with torch.no_grad():
            logits, _ = self.classifier(ecg_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
        return pred_class, confidence, probs
    
    def get_conditioning(self, ecg_tensor, target_class):
        """Get conditioning for generation"""
        with torch.no_grad():
            # Content (morphology)
            content = self.content_net(ecg_tensor)  # [1, 38, 512]
            
            # Style (rhythm)
            style = self.style_net(ecg_tensor)  # [1, 128]
            style_emb = self.style_proj(style).unsqueeze(1)  # [1, 1, 512]
            
            # Class embedding for TARGET class
            target_tensor = torch.tensor([target_class], device=self.device)
            class_emb = self.class_embed(target_tensor).unsqueeze(1)  # [1, 1, 512]
            
            # Combine
            conditioning = torch.cat([content, style_emb, class_emb], dim=1)
            uncond = torch.zeros_like(conditioning)
            
        return conditioning, uncond
    
    def generate_counterfactual_minimal(
        self,
        ecg_normalized,  # [1, 2500] normalized ECG
        target_class,
        num_inference_steps=50,
        guidance_scale=2.0,  # Lower guidance = closer to original
        edit_strength=0.7    # How much to "edit" (1.0 = full, 0.0 = no change)
    ):
        """
        Generate counterfactual with minimal intervention.
        
        Key difference from before:
        - Uses DDIM inversion to start from original ECG
        - edit_strength controls how much we deviate from original
        """
        with torch.no_grad():
            # Prepare input
            ecg_tensor = torch.tensor(ecg_normalized, dtype=torch.float32)
            ecg_tensor = ecg_tensor.unsqueeze(0).to(self.device)  # [1, 1, 2500]
            
            # Get conditioning for ORIGINAL class (for inversion)
            orig_class = 1 - target_class
            orig_conditioning, _ = self.get_conditioning(ecg_tensor, orig_class)
            
            # Get conditioning for TARGET class (for generation)
            target_conditioning, uncond = self.get_conditioning(ecg_tensor, target_class)
            
            # Step 1: DDIM Inversion - encode original ECG to noise space
            print(f"   Step 1: Inverting original ECG to noise space...")
            inverted_noise = ddim_inversion(
                self.unet,
                self.scheduler,
                ecg_tensor,
                orig_conditioning,
                num_inference_steps=num_inference_steps,
                guidance_scale=1.0  # No guidance during inversion
            )
            
            # Step 2: Interpolate between original and inverted noise
            # This controls edit strength
            if edit_strength < 1.0:
                # Partial edit: blend original with inverted
                random_noise = torch.randn_like(ecg_tensor)
                starting_noise = edit_strength * inverted_noise + (1 - edit_strength) * inverted_noise
            else:
                starting_noise = inverted_noise
            
            # Step 3: DDIM Sample with TARGET class conditioning
            print(f"   Step 2: Sampling with target class conditioning...")
            counterfactual = ddim_sample_from_noise(
                self.unet,
                self.scheduler,
                starting_noise,
                target_conditioning,
                uncond,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            cf_numpy = counterfactual.squeeze().cpu().numpy()
            
        return cf_numpy
    
    def full_pipeline(
        self,
        ecg_raw,
        mean,
        std,
        guidance_scale=2.0,
        edit_strength=0.8
    ):
        """Run full pipeline with minimal intervention"""
        print("\n" + "="*70)
        print("MINIMAL INTERVENTION COUNTERFACTUAL")
        print("="*70)
        
        # Normalize
        ecg_norm = (ecg_raw - mean) / (std + 1e-6)
        ecg_norm = ecg_norm.reshape(1, -1)
        
        # Classify original
        ecg_tensor = torch.tensor(ecg_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        orig_class, orig_conf, orig_probs = self.classify(ecg_tensor)
        
        print(f"\n📊 Original: {'AFib' if orig_class == 1 else 'Normal'} ({orig_conf*100:.1f}%)")
        
        # Target class
        target_class = 1 - orig_class
        print(f"🎯 Target: {'AFib' if target_class == 1 else 'Normal'}")
        print(f"📏 Edit strength: {edit_strength}, Guidance: {guidance_scale}")
        
        # Generate counterfactual
        print(f"\n🔄 Generating minimal intervention counterfactual...")
        cf_norm = self.generate_counterfactual_minimal(
            ecg_norm,
            target_class,
            guidance_scale=guidance_scale,
            edit_strength=edit_strength
        )
        
        # Denormalize
        cf_raw = cf_norm * std + mean
        
        # Classify counterfactual
        cf_tensor = torch.tensor(cf_norm.reshape(1, 1, -1), dtype=torch.float32).to(self.device)
        cf_class, cf_conf, cf_probs = self.classify(cf_tensor)
        
        print(f"\n📊 Counterfactual: {'AFib' if cf_class == 1 else 'Normal'} ({cf_conf*100:.1f}%)")
        
        flip_success = (cf_class == target_class)
        
        # Compute morphology preservation
        correlation = np.corrcoef(ecg_raw.flatten(), cf_raw.flatten())[0, 1]
        mse = np.mean((ecg_raw - cf_raw) ** 2)
        
        print(f"\n📈 Morphology Preservation:")
        print(f"   Pearson Correlation: {correlation:.4f}")
        print(f"   MSE: {mse:.6f}")
        
        status = "✅ FLIP SUCCESS" if flip_success else "❌ FLIP FAILED"
        print(f"\n{status} | Correlation: {correlation:.4f}")
        
        return {
            'original_raw': ecg_raw,
            'counterfactual_raw': cf_raw,
            'original_normalized': ecg_norm.flatten(),
            'counterfactual_normalized': cf_norm,
            'original_class': orig_class,
            'original_confidence': orig_conf,
            'original_probs': orig_probs,
            'counterfactual_class': cf_class,
            'counterfactual_confidence': cf_conf,
            'counterfactual_probs': cf_probs,
            'flip_successful': flip_success,
            'target_class': target_class,
            'correlation': correlation,
            'mse': mse
        }

def plot_minimal_intervention_result(result, save_path=None):
    """Visualize minimal intervention result"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    original = result['original_raw']
    counterfactual = result['counterfactual_raw']
    t = np.arange(len(original)) / 250
    
    # Colors
    orig_color = '#2563eb'
    cf_color = '#16a34a' if result['flip_successful'] else '#dc2626'
    
    # Plot 1: Overlay
    ax1 = axes[0]
    ax1.plot(t, original, color=orig_color, alpha=0.8, linewidth=1, label='Original')
    ax1.plot(t, counterfactual, color=cf_color, alpha=0.8, linewidth=1, label='Counterfactual')
    ax1.set_title(
        f"ECG Overlay | Original: {'AFib' if result['original_class']==1 else 'Normal'} "
        f"→ Counterfactual: {'AFib' if result['counterfactual_class']==1 else 'Normal'} "
        f"| Correlation: {result['correlation']:.4f}",
        fontsize=12, fontweight='bold'
    )
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed overlay (first 3 seconds)
    ax2 = axes[1]
    zoom = int(3 * 250)
    ax2.plot(t[:zoom], original[:zoom], color=orig_color, alpha=0.8, linewidth=1.5, label='Original')
    ax2.plot(t[:zoom], counterfactual[:zoom], color=cf_color, alpha=0.8, linewidth=1.5, label='Counterfactual')
    ax2.set_title('First 3 Seconds (Zoomed)', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference
    ax3 = axes[2]
    diff = counterfactual - original
    ax3.plot(t, diff, color='purple', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.fill_between(t, diff, 0, alpha=0.3, color='purple')
    ax3.set_title(f'Difference (CF - Original) | Mean: {diff.mean():.4f}, Std: {diff.std():.4f}', fontsize=11)
    ax3.set_ylabel('Difference')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Probabilities
    ax4 = axes[3]
    x = np.arange(2)
    width = 0.35
    ax4.bar(x - width/2, result['original_probs'], width, label='Original', color=orig_color, alpha=0.7)
    ax4.bar(x + width/2, result['counterfactual_probs'], width, label='Counterfactual', color=cf_color, alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Normal', 'AFib'])
    ax4.set_ylabel('Probability')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.set_title('Classification Probabilities', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (op, cp) in enumerate(zip(result['original_probs'], result['counterfactual_probs'])):
        ax4.text(i - width/2, op + 0.02, f'{op*100:.1f}%', ha='center', fontsize=9)
        ax4.text(i + width/2, cp + 0.02, f'{cp*100:.1f}%', ha='center', fontsize=9)
    
    ax4.set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("MINIMAL INTERVENTION COUNTERFACTUAL GENERATION")
    print("="*70)
    print("\nThis approach uses DDIM inversion to preserve morphology.\n")
    
    # Paths
    checkpoint_path = './enhanced_counterfactual_training/best_model.pth'
    classifier_path = './best_model/best_model.pth'
    data_path = './ecg_afib_data/X_combined.npy'
    labels_path = './ecg_afib_data/y_combined.npy'
    norm_path = './enhanced_counterfactual_training/norm_params.npy'
    output_dir = './counterfactual_minimal_intervention'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = MinimalInterventionGenerator(checkpoint_path, classifier_path)
    
    # Load data
    signals = np.load(data_path)
    labels = np.load(labels_path)
    norm_params = np.load(norm_path, allow_pickle=True).item()
    
    if labels.dtype.kind in ['U', 'S', 'O']:
        labels = np.array([1 if l == 'A' else 0 for l in labels])
    
    means = norm_params['means']
    stds = norm_params['stds']
    
    print(f"\n📂 Dataset: {len(signals):,} samples")
    
    # Test on a few samples
    num_samples = 10
    
    # Select balanced samples
    afib_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]
    
    test_indices = np.concatenate([
        np.random.choice(afib_idx, num_samples // 2, replace=False),
        np.random.choice(normal_idx, num_samples // 2, replace=False)
    ])
    np.random.shuffle(test_indices)
    
    results = []
    
    for i, idx in enumerate(test_indices):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}/{num_samples} (Index: {idx}, True: {'AFib' if labels[idx]==1 else 'Normal'})")
        print("="*70)
        
        try:
            result = generator.full_pipeline(
                signals[idx],
                means[idx],
                stds[idx],
                guidance_scale=2.0,
                edit_strength=0.8
            )
            
            result['sample_index'] = int(idx)
            result['true_label'] = int(labels[idx])
            results.append(result)
            
            # Save plot
            plot_path = f"{output_dir}/sample_{i+1:03d}_idx{idx}.png"
            plot_minimal_intervention_result(result, save_path=plot_path)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r['flip_successful'])
    correlations = [r['correlation'] for r in results]
    
    print(f"Total samples: {len(results)}")
    print(f"Flip success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"Mean correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"Min correlation: {np.min(correlations):.4f}")
    print(f"Max correlation: {np.max(correlations):.4f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(results),
        'flip_success_rate': successful / len(results) if results else 0,
        'mean_correlation': float(np.mean(correlations)),
        'std_correlation': float(np.std(correlations)),
        'min_correlation': float(np.min(correlations)),
        'max_correlation': float(np.max(correlations))
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/")

if __name__ == "__main__":
    main()
