"""
COMPLETE COUNTERFACTUAL ECG GENERATION PIPELINE
================================================
Using the enhanced model trained with classification loss

Process:
1. Load trained models (U-Net, encoders, classifier)
2. Select input ECG
3. Extract content (morphology) from input
4. Extract style (rhythm) from target class
5. Generate counterfactual
6. Classify both original and counterfactual
7. Visualize results
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

# Import models directly from counterfactual_training
from counterfactual_training import (
    UNet1DConditional,
    StyleEncoderWrapper,
    PretrainedContentEncoder,
    AFibResLSTM,
    ModelConfig
)

# Import ClassConditionalEmbedding from enhanced training
from enhanced_counterfactual_training import ClassConditionalEmbedding

# ============================================================================
# COUNTERFACTUAL GENERATOR CLASS
# ============================================================================

class CounterfactualECGGenerator:
    """
    Complete pipeline for generating counterfactual ECGs
    """
    
    def __init__(
        self,
        unet_checkpoint_path,
        style_encoder_weights_path,
        classifier_weights_path,
        device='cuda'
    ):
        """
        Initialize all models
        
        Args:
            unet_checkpoint_path: Path to trained U-Net weights (enhanced model)
            style_encoder_weights_path: Path to AFibResLSTM weights (for style)
            classifier_weights_path: Path to AFibResLSTM weights (for classification)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Initializing Counterfactual Generator on {self.device}")
        
        # ========================================
        # 1. Load Diffusion Model (U-Net)
        # ========================================
        print("\n1️⃣ Loading Diffusion Model (U-Net)...")
        self.unet = UNet1DConditional().to(self.device)
        
        checkpoint = torch.load(unet_checkpoint_path, map_location=self.device)
        if 'unet_state_dict' in checkpoint:
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
            print(f"   Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"   Best flip rate: {checkpoint.get('flip_rate', 'unknown')}")
        else:
            self.unet.load_state_dict(checkpoint)
        
        self.unet.eval()
        print("   ✅ U-Net loaded")
        
        # ========================================
        # 2. Load Style Encoder & Projector
        # ========================================
        print("\n2️⃣ Loading Style Encoder...")
        self.style_net = StyleEncoderWrapper(style_encoder_weights_path, self.device)
        self.style_net.eval()
        
        # Match the architecture from checkpoint (2-layer: 128→256→512)
        self.style_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        ).to(self.device)
        
        if 'style_proj_state_dict' in checkpoint:
            self.style_proj.load_state_dict(checkpoint['style_proj_state_dict'])
        
        self.style_proj.eval()
        print("   ✅ Style encoder loaded")
        
        # ========================================
        # 3. Load Class Conditional Embedding
        # ========================================
        print("\n3️⃣ Loading Class Conditional Embedding...")
        self.class_embed = ClassConditionalEmbedding(
            num_classes=2, 
            embed_dim=128
        ).to(self.device)
        
        if 'class_embed_state_dict' in checkpoint:
            self.class_embed.load_state_dict(checkpoint['class_embed_state_dict'])
        
        self.class_embed.eval()
        print("   ✅ Class embedding loaded")
        
        # ========================================
        # 4. Load Content Encoder
        # ========================================
        print("\n4️⃣ Loading Content Encoder...")
        self.content_net = PretrainedContentEncoder().to(self.device)
        self.content_net.eval()
        print("   ✅ Content encoder loaded")
        
        # ========================================
        # 5. Load Classifier
        # ========================================
        print("\n5️⃣ Loading Classifier (AFibResLSTM)...")
        config = ModelConfig()
        self.classifier = AFibResLSTM(config).to(self.device)
        
        classifier_checkpoint = torch.load(classifier_weights_path, map_location=self.device)
        if 'model_state_dict' in classifier_checkpoint:
            self.classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
        else:
            self.classifier.load_state_dict(classifier_checkpoint)
        
        self.classifier.eval()
        print("   ✅ Classifier loaded")
        
        # ========================================
        # 6. Load Diffusion Scheduler
        # ========================================
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False
        )
        
        print("\n✅ All models loaded successfully!\n")
    
    def classify_ecg(self, ecg_signal):
        """
        Classify an ECG signal
        
        Args:
            ecg_signal: [1, 2500] numpy array (normalized)
        
        Returns:
            predicted_class: 0 (Normal) or 1 (AFib)
            confidence: Probability of predicted class
            probabilities: [prob_normal, prob_afib]
        """
        with torch.no_grad():
            x = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits, _ = self.classifier(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
            
        return predicted_class, confidence, probs
    
    def generate_counterfactual(
        self,
        input_ecg_normalized,
        target_class,
        guidance_scale=3.0,
        num_inference_steps=50,
        style_source_ecg=None
    ):
        """
        Generate counterfactual ECG
        
        Args:
            input_ecg_normalized: [1, 2500] - Input ECG (z-score normalized)
            target_class: 0 (Normal) or 1 (AFib) - Target rhythm class
            guidance_scale: CFG strength
            num_inference_steps: Number of DDIM steps
            style_source_ecg: Optional specific ECG to extract style from
        
        Returns:
            counterfactual_ecg: [1, 2500] - Generated counterfactual
        """
        print(f"\n🔄 Generating counterfactual (target: {'AFib' if target_class == 1 else 'Normal'})...")
        
        with torch.no_grad():
            # ========================================
            # 1. Extract Content (from input ECG)
            # ========================================
            input_tensor = torch.tensor(
                input_ecg_normalized, 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)  # [1, 1, 2500]
            
            content = self.content_net(input_tensor)  # Patient morphology
            print(f"   ✅ Content extracted: {content.shape}")
            
            # ========================================
            # 2. Extract Style (from target class or source ECG)
            # ========================================
            if style_source_ecg is not None:
                style_tensor = torch.tensor(
                    style_source_ecg,
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
            else:
                style_tensor = input_tensor
            
            style = self.style_net(style_tensor)  # [1, 128]
            style_emb = self.style_proj(style).unsqueeze(1)  # [1, 1, 512]
            
            print(f"   ✅ Style extracted: {style.shape}")
            
            # ========================================
            # 3. Get Class Embedding for Target Class
            # ========================================
            target_class_tensor = torch.tensor([target_class], device=self.device)
            class_emb = self.class_embed(target_class_tensor).unsqueeze(1)  # [1, 512] -> [1, 1, 512]
            
            print(f"   ✅ Class embedding for target: {'AFib' if target_class == 1 else 'Normal'}")
            
            # ========================================
            # 4. Combine Content + Style + Class
            # ========================================
            conditioning = torch.cat([content, style_emb, class_emb], dim=1)  # [1, seq+2, 512]
            uncond_input = torch.zeros_like(conditioning)
            
            # ========================================
            # 5. DDIM Sampling (Reverse Diffusion)
            # ========================================
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Start from random noise
            latents = torch.randn_like(input_tensor)
            
            print(f"   🔄 Running {num_inference_steps} DDIM steps...")
            for t in tqdm(self.scheduler.timesteps, desc="Sampling", leave=False):
                # CFG: Run conditional and unconditional in parallel
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t.unsqueeze(0).to(self.device)] * 2)
                cond_input = torch.cat([conditioning, uncond_input])
                
                # Predict noise
                noise_pred = self.unet(latent_input, t_input, cond_input)
                
                # Split and apply CFG
                noise_cond, noise_uncond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                
                # Denoise step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            counterfactual_normalized = latents.squeeze().cpu().numpy()
            
            print(f"   ✅ Counterfactual generated!")
            
        return counterfactual_normalized
    
    def full_pipeline(
        self,
        input_ecg_raw,
        input_mean,
        input_std,
        guidance_scale=3.0,
        style_source_ecg_raw=None,
        style_source_mean=None,
        style_source_std=None
    ):
        """
        Complete pipeline: Classify → Generate → Classify
        """
        print("\n" + "="*70)
        print("COUNTERFACTUAL GENERATION PIPELINE")
        print("="*70)
        
        # ========================================
        # 1. Normalize Input
        # ========================================
        input_ecg_normalized = (input_ecg_raw - input_mean) / (input_std + 1e-6)
        input_ecg_normalized = input_ecg_normalized.reshape(1, -1)
        
        # ========================================
        # 2. Classify Original
        # ========================================
        print("\n📊 STEP 1: Classify Original ECG")
        orig_class, orig_conf, orig_probs = self.classify_ecg(input_ecg_normalized)
        
        print(f"   Original Class: {'AFib' if orig_class == 1 else 'Normal'}")
        print(f"   Confidence: {orig_conf*100:.2f}%")
        print(f"   Probabilities: Normal={orig_probs[0]*100:.1f}%, AFib={orig_probs[1]*100:.1f}%")
        
        # ========================================
        # 3. Determine Target Class
        # ========================================
        target_class = 1 - orig_class  # Flip: 0→1, 1→0
        print(f"\n🎯 STEP 2: Target Class = {'AFib' if target_class == 1 else 'Normal'}")
        
        # ========================================
        # 4. Prepare Style Source (if provided)
        # ========================================
        style_source_normalized = None
        if style_source_ecg_raw is not None:
            style_source_normalized = (style_source_ecg_raw - style_source_mean) / (style_source_std + 1e-6)
            style_source_normalized = style_source_normalized.reshape(1, -1)
            print(f"   Using provided style source from target class")
        else:
            print(f"   Using input ECG for style (relying on model to change rhythm)")
        
        # ========================================
        # 5. Generate Counterfactual
        # ========================================
        print(f"\n🔮 STEP 3: Generate Counterfactual")
        
        counterfactual_normalized = self.generate_counterfactual(
            input_ecg_normalized,
            target_class,
            guidance_scale=guidance_scale,
            style_source_ecg=style_source_normalized
        )
        
        # Denormalize
        counterfactual_raw = counterfactual_normalized * input_std + input_mean
        
        # ========================================
        # 6. Classify Counterfactual
        # ========================================
        print(f"\n📊 STEP 4: Classify Counterfactual ECG")
        cf_class, cf_conf, cf_probs = self.classify_ecg(
            counterfactual_normalized.reshape(1, -1)
        )
        
        print(f"   Counterfactual Class: {'AFib' if cf_class == 1 else 'Normal'}")
        print(f"   Confidence: {cf_conf*100:.2f}%")
        print(f"   Probabilities: Normal={cf_probs[0]*100:.1f}%, AFib={cf_probs[1]*100:.1f}%")
        
        # ========================================
        # 7. Check if Flip Successful
        # ========================================
        flip_successful = (cf_class == target_class)
        
        print(f"\n{'='*70}")
        print(f"RESULT: {'✅ FLIP SUCCESSFUL!' if flip_successful else '❌ FLIP FAILED'}")
        print(f"{'='*70}")
        
        if flip_successful:
            print(f"Original: {'AFib' if orig_class == 1 else 'Normal'} ({orig_conf*100:.1f}%) → "
                  f"Counterfactual: {'AFib' if cf_class == 1 else 'Normal'} ({cf_conf*100:.1f}%)")
        else:
            print(f"⚠️  Counterfactual still classified as {'AFib' if cf_class == 1 else 'Normal'}")
        
        return {
            'input_ecg_raw': input_ecg_raw,
            'counterfactual_raw': counterfactual_raw,
            'counterfactual_normalized': counterfactual_normalized,
            'original_class': orig_class,
            'original_confidence': orig_conf,
            'original_probs': orig_probs,
            'counterfactual_class': cf_class,
            'counterfactual_confidence': cf_conf,
            'counterfactual_probs': cf_probs,
            'flip_successful': flip_successful,
            'target_class': target_class
        }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_counterfactual_results(results, save_path=None):
    """Create comprehensive visualization of counterfactual generation"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    original = results['input_ecg_raw']
    counterfactual = results['counterfactual_raw']
    
    # Plot 1: Original ECG (Full)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(original, color='#2563eb', linewidth=1, alpha=0.8)
    ax1.set_title(
        f"Original ECG - Class: {'AFib' if results['original_class'] == 1 else 'Normal'} "
        f"(Confidence: {results['original_confidence']*100:.1f}%)",
        fontsize=14, fontweight='bold', color='#2563eb'
    )
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(original)])
    
    # Plot 2: Counterfactual ECG (Full)
    ax2 = fig.add_subplot(gs[1, :])
    color = '#16a34a' if results['flip_successful'] else '#dc2626'
    ax2.plot(counterfactual, color=color, linewidth=1, alpha=0.8)
    ax2.set_title(
        f"Counterfactual ECG - Class: {'AFib' if results['counterfactual_class'] == 1 else 'Normal'} "
        f"(Confidence: {results['counterfactual_confidence']*100:.1f}%) "
        f"{'✅ FLIP SUCCESS' if results['flip_successful'] else '❌ FLIP FAILED'}",
        fontsize=14, fontweight='bold', color=color
    )
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_xlabel('Time (samples)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, len(counterfactual)])
    
    # Plot 3: Overlay (Zoom - 2 seconds)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(original[:500], color='#2563eb', linewidth=1.5, alpha=0.7, label='Original')
    ax3.plot(counterfactual[:500], color=color, linewidth=1.5, alpha=0.7, label='Counterfactual')
    ax3.set_title('Overlay - First 2 Seconds', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Amplitude')
    ax3.set_xlabel('Time (samples)')
    
    # Plot 4: Overlay (Middle 2 seconds)
    ax4 = fig.add_subplot(gs[2, 1])
    mid = len(original) // 2
    ax4.plot(original[mid:mid+500], color='#2563eb', linewidth=1.5, alpha=0.7, label='Original')
    ax4.plot(counterfactual[mid:mid+500], color=color, linewidth=1.5, alpha=0.7, label='Counterfactual')
    ax4.set_title('Overlay - Middle 2 Seconds', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylabel('Amplitude')
    ax4.set_xlabel('Time (samples)')
    
    # Plot 5: Classification Probabilities
    ax5 = fig.add_subplot(gs[3, 0])
    classes = ['Normal', 'AFib']
    x_pos = np.arange(len(classes))
    width = 0.35
    
    ax5.bar(x_pos - width/2, results['original_probs'], width, 
           label='Original', color='#2563eb', alpha=0.7)
    ax5.bar(x_pos + width/2, results['counterfactual_probs'], width,
           label='Counterfactual', color=color, alpha=0.7)
    
    ax5.set_ylabel('Probability', fontsize=11)
    ax5.set_title('Classification Probabilities', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(classes)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 1])
    
    for i, (orig_p, cf_p) in enumerate(zip(results['original_probs'], results['counterfactual_probs'])):
        ax5.text(i - width/2, orig_p + 0.02, f'{orig_p*100:.1f}%', ha='center', fontsize=9)
        ax5.text(i + width/2, cf_p + 0.02, f'{cf_p*100:.1f}%', ha='center', fontsize=9)
    
    # Plot 6: Summary
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    
    summary = f"""
    COUNTERFACTUAL GENERATION SUMMARY
    ═════════════════════════════════════════
    
    Input Class:          {'AFib' if results['original_class'] == 1 else 'Normal'}
    Input Confidence:     {results['original_confidence']*100:.2f}%
    
    Target Class:         {'AFib' if results['target_class'] == 1 else 'Normal'}
    
    Generated Class:      {'AFib' if results['counterfactual_class'] == 1 else 'Normal'}
    Generated Confidence: {results['counterfactual_confidence']*100:.2f}%
    
    Status:               {'✅ FLIP SUCCESSFUL' if results['flip_successful'] else '❌ FLIP FAILED'}
    """
    
    ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_counterfactual_generation(
    unet_checkpoint="./enhanced_counterfactual_training/best_model.pth",
    classifier_checkpoint="./best_model/best_model.pth",
    dataset_path="./ecg_afib_data/X_combined.npy",
    labels_path="./ecg_afib_data/y_combined.npy",
    norm_params_path="./enhanced_counterfactual_training/norm_params.npy",
    output_dir="./counterfactual_results_enhanced",
    num_samples=30
):
    """Run counterfactual generation on multiple samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================
    # 1. Initialize Generator
    # ========================================
    generator = CounterfactualECGGenerator(
        unet_checkpoint_path=unet_checkpoint,
        style_encoder_weights_path=classifier_checkpoint,
        classifier_weights_path=classifier_checkpoint,
        device='cuda'
    )
    
    # ========================================
    # 2. Load Dataset
    # ========================================
    print(f"\n📂 Loading dataset from {dataset_path}")
    signals_raw = np.load(dataset_path)
    labels = np.load(labels_path)
    
    # Convert labels if string
    if labels.dtype.kind in ['U', 'S', 'O']:
        labels = np.array([1 if l == 'A' else 0 for l in labels])
    
    # Load normalization params
    norm_params = np.load(norm_params_path, allow_pickle=True).item()
    means = norm_params['means']
    stds = norm_params['stds']
    
    print(f"   Dataset size: {len(signals_raw):,}")
    print(f"   AFib samples: {np.sum(labels == 1):,}")
    print(f"   Normal samples: {np.sum(labels == 0):,}")
    
    # ========================================
    # 3. Generate Counterfactuals
    # ========================================
    results_list = []
    flip_success_count = 0
    afib_to_normal_success = 0
    normal_to_afib_success = 0
    afib_to_normal_total = 0
    normal_to_afib_total = 0
    
    # Sample balanced
    afib_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    
    n_each = num_samples // 2
    selected_afib = np.random.choice(afib_indices, min(n_each, len(afib_indices)), replace=False)
    selected_normal = np.random.choice(normal_indices, min(n_each, len(normal_indices)), replace=False)
    sample_indices = np.concatenate([selected_afib, selected_normal])
    np.random.shuffle(sample_indices)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}/{len(sample_indices)} (Index: {idx})")
        print(f"True label: {'AFib' if labels[idx] == 1 else 'Normal'}")
        print(f"{'='*70}")
        
        try:
            results = generator.full_pipeline(
                input_ecg_raw=signals_raw[idx],
                input_mean=means[idx],
                input_std=stds[idx],
                guidance_scale=3.0
            )
            
            # Save plot
            plot_path = f"{output_dir}/counterfactual_sample_{i+1:03d}_idx{idx}.png"
            plot_counterfactual_results(results, save_path=plot_path)
            
            results['sample_index'] = int(idx)
            results['true_label'] = int(labels[idx])
            results_list.append(results)
            
            if results['flip_successful']:
                flip_success_count += 1
            
            if results['original_class'] == 1:  # AFib to Normal
                afib_to_normal_total += 1
                if results['flip_successful']:
                    afib_to_normal_success += 1
            else:  # Normal to AFib
                normal_to_afib_total += 1
                if results['flip_successful']:
                    normal_to_afib_success += 1
                    
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {e}")
            continue
    
    # ========================================
    # 4. Summary Statistics
    # ========================================
    total_processed = len(results_list)
    
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total samples processed: {total_processed}")
    print(f"Successful flips: {flip_success_count} ({flip_success_count/total_processed*100:.1f}%)")
    print(f"Failed flips: {total_processed - flip_success_count}")
    print(f"\nBreakdown:")
    if afib_to_normal_total > 0:
        print(f"  AFib → Normal: {afib_to_normal_success}/{afib_to_normal_total} ({afib_to_normal_success/afib_to_normal_total*100:.1f}%)")
    if normal_to_afib_total > 0:
        print(f"  Normal → AFib: {normal_to_afib_success}/{normal_to_afib_total} ({normal_to_afib_success/normal_to_afib_total*100:.1f}%)")
    print(f"\nResults saved in: {output_dir}/")
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': total_processed,
        'flip_success_rate': flip_success_count / total_processed if total_processed > 0 else 0,
        'afib_to_normal': {
            'total': afib_to_normal_total,
            'success': afib_to_normal_success,
            'rate': afib_to_normal_success / afib_to_normal_total if afib_to_normal_total > 0 else 0
        },
        'normal_to_afib': {
            'total': normal_to_afib_total,
            'success': normal_to_afib_success,
            'rate': normal_to_afib_success / normal_to_afib_total if normal_to_afib_total > 0 else 0
        }
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results_list

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    results = run_counterfactual_generation(
        unet_checkpoint="./enhanced_counterfactual_training/best_model.pth",
        classifier_checkpoint="./best_model/best_model.pth",
        dataset_path="./ecg_afib_data/X_combined.npy",
        labels_path="./ecg_afib_data/y_combined.npy",
        norm_params_path="./enhanced_counterfactual_training/norm_params.npy",
        output_dir="./counterfactual_results_enhanced",
        num_samples=30
    )
