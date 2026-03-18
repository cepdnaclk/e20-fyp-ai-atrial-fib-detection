"""
Generate Counterfactuals and Reconstructions using Trained Img2Img Model
=========================================================================

This script:
1. Loads the trained Image-to-Image diffusion model
2. Generates counterfactuals for test samples
3. Generates reconstructions (same-class denoising)
4. Creates overlay visualizations
5. Computes evaluation metrics

Usage:
    python generate.py --model_path ./checkpoints/best_model.pth --num_samples 100
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import os
import sys

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))
from counterfactual_training import AFibResLSTM, ModelConfig

# Local imports
from model import ECGImg2ImgDiffusion, Img2ImgNoiseScheduler


class Img2ImgGenerator:
    """Generate counterfactuals and reconstructions from trained model"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.load_model(model_path)
        
        # Load classifier
        self.load_classifier()
        
    def load_model(self, model_path):
        """Load trained diffusion model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Build model config
        model_config = {
            'base_channels': self.config.get('base_channels', 64),
            'channel_mults': tuple(self.config.get('channel_mults', [1, 2, 4, 8])),
            'time_embed_dim': self.config.get('time_embed_dim', 256),
            'class_embed_dim': self.config.get('class_embed_dim', 256),
            'dropout': self.config.get('dropout', 0.1),
            'use_content_conditioning': self.config.get('use_content_conditioning', True),
            'content_dim': self.config.get('content_dim', 512),
        }
        
        self.model = ECGImg2ImgDiffusion(model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize scheduler
        self.scheduler = Img2ImgNoiseScheduler(
            num_train_timesteps=self.config.get('num_train_timesteps', 1000),
            beta_start=self.config.get('beta_start', 0.0001),
            beta_end=self.config.get('beta_end', 0.02),
            beta_schedule=self.config.get('beta_schedule', 'linear')
        )
        
        print(f"Loaded model from: {model_path}")
        print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        
    def load_classifier(self):
        """Load frozen classifier for evaluation"""
        classifier_path = '../best_model/best_model.pth'
        config = ModelConfig()
        self.classifier = AFibResLSTM(config).to(self.device)
        ckpt = torch.load(classifier_path, map_location=self.device)
        self.classifier.load_state_dict(ckpt['model_state_dict'])
        self.classifier.eval()
        print("Loaded classifier")
        
    @torch.no_grad()
    def generate_counterfactual(self, x, target_class, noise_strength=0.5, num_steps=50):
        """
        Generate counterfactual using SDEdit-style denoising.
        
        Args:
            x: [batch, 1, 2500] - Original ECG
            target_class: [batch] or int - Target class
            noise_strength: How much noise to add (0-1)
            num_steps: Number of denoising steps
            
        Returns:
            counterfactual: [batch, 1, 2500]
        """
        batch_size = x.shape[0]
        
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class] * batch_size).to(self.device)
        
        # Add noise at controlled strength
        noise = torch.randn_like(x)
        num_train_steps = self.config.get('num_train_timesteps', 1000)
        timestep = int(noise_strength * (num_train_steps - 1))
        
        noisy_x = self.scheduler.add_noise(x, noise, torch.tensor([timestep]).to(self.device))
        
        # Set up inference
        self.scheduler.set_timesteps(num_steps)
        
        sample = noisy_x
        
        # Find starting point
        for i, t in enumerate(self.scheduler.timesteps):
            if t <= timestep:
                start_idx = i
                break
        else:
            start_idx = len(self.scheduler.timesteps) - 1
        
        # Denoise
        for t in self.scheduler.timesteps[start_idx:]:
            noise_pred = self.model(sample, t.expand(batch_size).to(self.device), target_class)
            sample = self.scheduler.step(noise_pred, t, sample)
        
        return sample
    
    @torch.no_grad()
    def generate_reconstruction(self, x, noise_strength=0.5, num_steps=50):
        """
        Reconstruct by denoising with SAME class.
        Tests if model preserves signal when no class change is needed.
        
        Args:
            x: [batch, 1, 2500] - Original ECG
            noise_strength: How much noise to add
            num_steps: Denoising steps
            
        Returns:
            reconstruction: [batch, 1, 2500]
        """
        # First classify to get original class
        logits, _ = self.classifier(x)
        original_class = logits.argmax(dim=1)
        
        # Reconstruct with same class
        return self.generate_counterfactual(x, original_class, noise_strength, num_steps)
    
    @torch.no_grad()
    def classify(self, x):
        """Classify ECG and return class + confidence"""
        logits, _ = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1)
        confidence = probs.gather(1, pred_class.unsqueeze(1)).squeeze(1)
        return pred_class, confidence
    
    def visualize_overlay(self, original, counterfactual, reconstruction,
                          orig_class, cf_class, orig_conf, cf_conf, save_path):
        """Create overlay visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Time axis
        t = np.arange(original.shape[0]) / 250  # 250 Hz
        
        # 1. Original ECG
        axes[0, 0].plot(t, original, 'b-', linewidth=0.8, label='Original')
        axes[0, 0].set_title(f"Original ECG - {'AFib' if orig_class == 1 else 'Normal'} ({orig_conf*100:.1f}%)",
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Counterfactual
        axes[0, 1].plot(t, counterfactual, 'r-', linewidth=0.8, label='Counterfactual')
        flip_status = "✓ FLIPPED" if cf_class != orig_class else "✗ NOT FLIPPED"
        axes[0, 1].set_title(f"Counterfactual - {'AFib' if cf_class == 1 else 'Normal'} ({cf_conf*100:.1f}%) {flip_status}",
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Overlay
        axes[1, 0].plot(t, original, 'b-', linewidth=0.8, alpha=0.7, label='Original')
        axes[1, 0].plot(t, counterfactual, 'r-', linewidth=0.8, alpha=0.7, label='Counterfactual')
        
        # Highlight differences
        diff = np.abs(counterfactual - original)
        diff_threshold = np.percentile(diff, 90)
        diff_regions = diff > diff_threshold
        for i in range(len(t)):
            if diff_regions[i]:
                axes[1, 0].axvspan(t[max(0, i-2)], t[min(len(t)-1, i+2)], alpha=0.1, color='yellow')
        
        corr = np.corrcoef(original, counterfactual)[0, 1]
        axes[1, 0].set_title(f"Overlay - Correlation: {corr:.4f}", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. Difference
        axes[1, 1].plot(t, original, 'b-', linewidth=0.6, alpha=0.5, label='Original')
        axes[1, 1].plot(t, reconstruction, 'g-', linewidth=0.6, alpha=0.8, label='Reconstruction')
        recon_corr = np.corrcoef(original, reconstruction)[0, 1]
        axes[1, 1].set_title(f"Reconstruction (same-class) - Correlation: {recon_corr:.4f}",
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def run_generation(self, X, y, num_samples=100, noise_strength=0.5, num_steps=50,
                       output_dir='./outputs'):
        """
        Generate counterfactuals and reconstructions for samples.
        
        Args:
            X: [N, 2500] - ECG signals
            y: [N] - Labels ('A' or 'N')
            num_samples: Number of samples to process
            noise_strength: SDEdit noise level
            num_steps: Denoising steps
            output_dir: Output directory
        """
        # Create output directories
        cf_dir = Path(output_dir) / 'counterfactuals'
        recon_dir = Path(output_dir) / 'reconstructions'
        overlay_dir = Path(output_dir) / 'overlays'
        
        cf_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert labels
        y_int = np.array([1 if l == 'A' else 0 for l in y])
        
        # Select balanced samples
        afib_idx = np.where(y_int == 1)[0]
        normal_idx = np.where(y_int == 0)[0]
        
        n_each = min(num_samples // 2, len(afib_idx), len(normal_idx))
        np.random.seed(42)
        selected_afib = np.random.choice(afib_idx, n_each, replace=False)
        selected_normal = np.random.choice(normal_idx, n_each, replace=False)
        selected_idx = np.concatenate([selected_afib, selected_normal])
        
        print(f"\nGenerating counterfactuals for {len(selected_idx)} samples...")
        print(f"  Noise strength: {noise_strength}")
        print(f"  Denoising steps: {num_steps}")
        
        results = {
            'samples': [],
            'normal_to_afib': {'flip': 0, 'total': 0, 'correlations': []},
            'afib_to_normal': {'flip': 0, 'total': 0, 'correlations': []},
            'reconstruction_correlations': []
        }
        
        for idx in tqdm(selected_idx):
            # Get sample
            signal = X[idx]
            label = y_int[idx]
            
            # Normalize
            signal_norm = (signal - signal.mean()) / (signal.std() + 1e-6)
            x = torch.tensor(signal_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Classify original
            orig_class, orig_conf = self.classify(x)
            orig_class = orig_class.item()
            orig_conf = orig_conf.item()
            
            # Generate counterfactual
            target_class = 1 - orig_class
            cf = self.generate_counterfactual(x, target_class, noise_strength, num_steps)
            
            # Classify counterfactual
            cf_class, cf_conf = self.classify(cf)
            cf_class = cf_class.item()
            cf_conf = cf_conf.item()
            
            # Generate reconstruction
            recon = self.generate_reconstruction(x, noise_strength, num_steps)
            
            # Convert to numpy
            cf_np = cf[0, 0].cpu().numpy()
            recon_np = recon[0, 0].cpu().numpy()
            
            # Compute metrics
            cf_corr = np.corrcoef(signal_norm, cf_np)[0, 1]
            recon_corr = np.corrcoef(signal_norm, recon_np)[0, 1]
            flip_success = (cf_class == target_class)
            
            # Store results
            key = 'normal_to_afib' if orig_class == 0 else 'afib_to_normal'
            results[key]['total'] += 1
            if flip_success:
                results[key]['flip'] += 1
            results[key]['correlations'].append(cf_corr)
            results['reconstruction_correlations'].append(recon_corr)
            
            results['samples'].append({
                'idx': int(idx),
                'orig_class': orig_class,
                'cf_class': cf_class,
                'orig_conf': orig_conf,
                'cf_conf': cf_conf,
                'cf_correlation': cf_corr,
                'recon_correlation': recon_corr,
                'flip_success': flip_success
            })
            
            # Save arrays
            np.save(cf_dir / f'cf_{idx}.npy', cf_np)
            np.save(recon_dir / f'recon_{idx}.npy', recon_np)
            
            # Create visualization
            self.visualize_overlay(
                signal_norm, cf_np, recon_np,
                orig_class, cf_class, orig_conf, cf_conf,
                overlay_dir / f'overlay_{idx}.png'
            )
        
        # Compute summary
        summary = {}
        for key in ['normal_to_afib', 'afib_to_normal']:
            if results[key]['total'] > 0:
                summary[key] = {
                    'flip_rate': results[key]['flip'] / results[key]['total'],
                    'correlation': np.mean(results[key]['correlations']),
                    'total': results[key]['total']
                }
            else:
                summary[key] = {'flip_rate': 0, 'correlation': 0, 'total': 0}
        
        total = sum(results[k]['total'] for k in ['normal_to_afib', 'afib_to_normal'])
        total_flip = sum(results[k]['flip'] for k in ['normal_to_afib', 'afib_to_normal'])
        all_corrs = results['normal_to_afib']['correlations'] + results['afib_to_normal']['correlations']
        
        summary['overall'] = {
            'flip_rate': total_flip / total if total > 0 else 0,
            'correlation': np.mean(all_corrs) if all_corrs else 0,
            'reconstruction_correlation': np.mean(results['reconstruction_correlations']),
            'total': total
        }
        
        # Print results
        print("\n" + "="*70)
        print("GENERATION RESULTS")
        print("="*70)
        print(f"\n| Direction      | Flip Rate | CF Correlation | Samples |")
        print("-"*70)
        print(f"| Normal→AFib    | {summary['normal_to_afib']['flip_rate']*100:8.1f}% | {summary['normal_to_afib']['correlation']:14.4f} | {summary['normal_to_afib']['total']:7d} |")
        print(f"| AFib→Normal    | {summary['afib_to_normal']['flip_rate']*100:8.1f}% | {summary['afib_to_normal']['correlation']:14.4f} | {summary['afib_to_normal']['total']:7d} |")
        print("-"*70)
        print(f"| Overall        | {summary['overall']['flip_rate']*100:8.1f}% | {summary['overall']['correlation']:14.4f} | {summary['overall']['total']:7d} |")
        print("="*70)
        print(f"\nReconstruction Correlation: {summary['overall']['reconstruction_correlation']:.4f}")
        print(f"  (Higher = better morphology preservation when no change needed)")
        
        # Save results
        results['summary'] = summary
        with open(Path(output_dir) / 'generation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"\nResults saved to: {output_dir}")
        
        return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--data_path', type=str, default='../ecg_afib_data')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--noise_strength', type=float, default=0.5)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    X = np.load(os.path.join(args.data_path, 'X_combined.npy'))
    y = np.load(os.path.join(args.data_path, 'y_combined.npy'))
    print(f"Loaded {len(X)} samples")
    
    # Initialize generator
    generator = Img2ImgGenerator(args.model_path)
    
    # Run generation
    generator.run_generation(
        X, y,
        num_samples=args.num_samples,
        noise_strength=args.noise_strength,
        num_steps=args.num_steps,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
