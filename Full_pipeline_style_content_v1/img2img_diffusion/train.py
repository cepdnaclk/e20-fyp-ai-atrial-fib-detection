"""
Image-to-Image Diffusion Training Script for ECG Counterfactuals
=================================================================

This script trains a diffusion model that can transform ECGs from one class to another
while preserving the patient's unique morphological features.

Key Training Innovations:
1. SDEdit-style training: Model learns to denoise partially noised ECGs
2. Class-conditional denoising: Target class guides the denoising process
3. Content preservation loss: Ensures morphology remains similar
4. Classification guidance: Generated ECGs should classify as target class
5. Reconstruction loss: When target=source, output should match input

Usage:
    python train.py --epochs 100 --batch_size 32 --noise_strength_range 0.3 0.8

Output:
    ./checkpoints/best_model.pth
    ./outputs/training_logs/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import json
import os
import sys
import argparse
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))
from counterfactual_training import AFibResLSTM, ModelConfig

# Local imports
from model import ECGImg2ImgDiffusion, Img2ImgNoiseScheduler, get_default_config

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainConfig:
    def __init__(self):
        # Data
        self.data_path = '../ecg_afib_data'
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Model
        self.base_channels = 64
        self.channel_mults = (1, 2, 4, 8)
        self.time_embed_dim = 256
        self.class_embed_dim = 256
        self.dropout = 0.1
        self.use_content_conditioning = True
        self.content_dim = 512
        
        # Diffusion
        self.num_train_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.beta_schedule = 'linear'
        
        # SDEdit-specific
        self.noise_strength_min = 0.3  # Minimum noise strength
        self.noise_strength_max = 0.8  # Maximum noise strength
        
        # Training
        self.epochs = 100
        self.batch_size = 32
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        
        # Loss weights
        self.lambda_noise = 1.0           # Noise prediction loss
        self.lambda_content = 0.5         # Content preservation loss
        self.lambda_classification = 0.1  # Classification guidance
        self.lambda_reconstruction = 0.5  # Reconstruction loss (same-class)
        
        # Early stopping
        self.patience = 15
        self.min_delta = 0.001
        
        # Checkpointing
        self.save_every = 5
        self.output_dir = './checkpoints'
        self.log_dir = './outputs/training_logs'
        
        # Classifier path
        self.classifier_path = '../best_model/best_model.pth'
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def get_model_config(self):
        return {
            'base_channels': self.base_channels,
            'channel_mults': self.channel_mults,
            'time_embed_dim': self.time_embed_dim,
            'class_embed_dim': self.class_embed_dim,
            'dropout': self.dropout,
            'use_content_conditioning': self.use_content_conditioning,
            'content_dim': self.content_dim,
            'num_train_timesteps': self.num_train_timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'beta_schedule': self.beta_schedule,
        }


# ============================================================================
# DATASET
# ============================================================================

class ECGDataset(Dataset):
    """ECG Dataset with labels and normalization"""
    
    def __init__(self, signals, labels, normalize=True):
        self.signals = signals.astype(np.float32)
        
        # Convert labels to int
        if isinstance(labels[0], str):
            self.labels = np.array([1 if l == 'A' else 0 for l in labels], dtype=np.int64)
        else:
            self.labels = np.array(labels, dtype=np.int64)
            
        self.normalize = normalize
        
        # Compute per-sample statistics
        if normalize:
            self.means = np.mean(self.signals, axis=1, keepdims=True)
            self.stds = np.std(self.signals, axis=1, keepdims=True) + 1e-6
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.normalize:
            signal = (signal - self.means[idx]) / self.stds[idx]
        
        # [1, 2500]
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return signal, label


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_noise_loss(pred_noise, true_noise):
    """Standard MSE loss for noise prediction"""
    return F.mse_loss(pred_noise, true_noise)


def compute_content_loss(original_content, generated_content):
    """Preserve morphological features"""
    return F.mse_loss(generated_content, original_content)


def compute_classification_loss(classifier, pred_x0, target_class, device):
    """
    Guide generated ECG toward target class.
    Uses cross-entropy loss with the frozen classifier.
    """
    classifier.train()  # Required for cuDNN RNN backward
    
    logits, _ = classifier(pred_x0)
    loss = F.cross_entropy(logits, target_class)
    
    return loss


def compute_reconstruction_loss(pred_x0, original, same_class_mask):
    """
    When target_class == original_class, output should match input.
    This teaches the model to preserve the signal when no change is needed.
    """
    if same_class_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_x0.device)
    
    # Only compute for same-class samples
    loss = F.mse_loss(pred_x0[same_class_mask], original[same_class_mask])
    return loss


def compute_frequency_loss(pred, target):
    """Preserve frequency content (ECG structure)"""
    pred_fft = torch.fft.rfft(pred.float(), dim=-1)
    target_fft = torch.fft.rfft(target.float(), dim=-1)
    
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    return F.l1_loss(pred_mag, target_mag)


# ============================================================================
# TRAINING LOOP
# ============================================================================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize models
        self._init_models()
        self._init_data()
        self._init_training()
        
    def _init_models(self):
        """Initialize diffusion model and classifier"""
        # Diffusion model
        model_config = self.config.get_model_config()
        self.model = ECGImg2ImgDiffusion(model_config).to(self.device)
        
        # Noise scheduler
        self.scheduler = Img2ImgNoiseScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule=self.config.beta_schedule
        )
        
        # Load classifier (frozen)
        classifier_config = ModelConfig()
        self.classifier = AFibResLSTM(classifier_config).to(self.device)
        ckpt = torch.load(self.config.classifier_path, map_location=self.device)
        self.classifier.load_state_dict(ckpt['model_state_dict'])
        for p in self.classifier.parameters():
            p.requires_grad = False
        print("Loaded frozen classifier")
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")
        
    def _init_data(self):
        """Load and split data"""
        # Load data
        X = np.load(os.path.join(self.config.data_path, 'X_combined.npy'))
        y = np.load(os.path.join(self.config.data_path, 'y_combined.npy'))
        print(f"Loaded data: {X.shape}, labels: {len(y)}")
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        # Convert labels for stratification
        y_int = np.array([1 if l == 'A' else 0 for l in y])
        
        # Train/temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_int, test_size=(1 - self.config.train_ratio),
            stratify=y_int, random_state=42
        )
        
        # Val/test split
        val_ratio_adjusted = self.config.val_ratio / (self.config.val_ratio + self.config.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio_adjusted),
            stratify=y_temp, random_state=42
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create datasets
        self.train_dataset = ECGDataset(X_train, y_train)
        self.val_dataset = ECGDataset(X_val, y_val)
        self.test_dataset = ECGDataset(X_test, y_test)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Save split indices
        split_info = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_afib': int(y_train.sum()),
            'val_afib': int(y_val.sum()),
            'test_afib': int(y_test.sum())
        }
        with open(os.path.join(self.config.output_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
            
        # Store test data for later
        self.X_test = X_test
        self.y_test = y_test
        
    def _init_training(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs
        )
        
        self.scaler = GradScaler()
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_flip_rate': [], 'val_flip_rate': [],
            'train_correlation': [], 'val_correlation': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_step(self, batch):
        """Single training step"""
        x, labels = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        batch_size = x.shape[0]
        
        # Sample target class (counterfactual or same)
        # 70% counterfactual (flip), 30% reconstruction (same class)
        flip_mask = torch.rand(batch_size, device=self.device) > 0.3
        target_class = torch.where(flip_mask, 1 - labels, labels)
        same_class_mask = ~flip_mask
        
        # Sample noise strength (SDEdit-style)
        strength = torch.FloatTensor(batch_size).uniform_(
            self.config.noise_strength_min,
            self.config.noise_strength_max
        ).to(self.device)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise at sampled timesteps
        timesteps = (strength * (self.config.num_train_timesteps - 1)).long()
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        
        # Predict noise with target class conditioning
        with autocast():
            noise_pred = self.model(noisy_x, timesteps, target_class)
            
            # Noise prediction loss
            loss_noise = compute_noise_loss(noise_pred, noise)
            
            # Estimate x0 for additional losses
            alpha_t = self.scheduler.scheduler.alphas_cumprod[timesteps]
            alpha_t = alpha_t.view(-1, 1, 1).to(self.device)
            pred_x0 = (noisy_x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -5, 5)  # Clamp for stability
            
            # Content preservation loss
            if self.model.content_encoder is not None:
                with torch.no_grad():
                    original_content = self.model.get_content_embedding(x)
                pred_content = self.model.get_content_embedding(pred_x0)
                loss_content = compute_content_loss(original_content, pred_content)
            else:
                loss_content = torch.tensor(0.0, device=self.device)
            
            # Classification loss (for counterfactuals only)
            if flip_mask.sum() > 0:
                loss_cls = compute_classification_loss(
                    self.classifier, pred_x0[flip_mask], target_class[flip_mask], self.device
                )
            else:
                loss_cls = torch.tensor(0.0, device=self.device)
            
            # Reconstruction loss (for same-class)
            loss_recon = compute_reconstruction_loss(pred_x0, x, same_class_mask)
            
            # Total loss
            total_loss = (
                self.config.lambda_noise * loss_noise +
                self.config.lambda_content * loss_content +
                self.config.lambda_classification * loss_cls +
                self.config.lambda_reconstruction * loss_recon
            )
        
        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'total_loss': total_loss.item(),
            'loss_noise': loss_noise.item(),
            'loss_content': loss_content.item() if isinstance(loss_content, torch.Tensor) else loss_content,
            'loss_cls': loss_cls.item() if isinstance(loss_cls, torch.Tensor) else loss_cls,
            'loss_recon': loss_recon.item()
        }
    
    @torch.no_grad()
    def generate_counterfactual(self, x, target_class, num_steps=50, noise_strength=0.5):
        """
        Generate counterfactual using SDEdit-style denoising.
        
        Args:
            x: [batch, 1, 2500] - Original ECG
            target_class: [batch] - Target class
            num_steps: Number of denoising steps
            noise_strength: How much noise to add (0-1)
        """
        self.model.eval()
        
        # Add noise at controlled strength
        noise = torch.randn_like(x)
        timestep = int(noise_strength * (self.config.num_train_timesteps - 1))
        noisy_x = self.scheduler.add_noise(x, noise, torch.tensor([timestep]).to(x.device))
        
        # Set up inference
        self.scheduler.set_timesteps(num_steps)
        
        # Start from the noised version
        sample = noisy_x
        
        # Only denoise from our starting timestep
        start_idx = 0
        for i, t in enumerate(self.scheduler.timesteps):
            if t <= timestep:
                start_idx = i
                break
        
        # Denoise
        for t in self.scheduler.timesteps[start_idx:]:
            noise_pred = self.model(sample, t.expand(x.shape[0]).to(self.device), target_class)
            sample = self.scheduler.step(noise_pred, t, sample)
        
        self.model.train()
        return sample
    
    @torch.no_grad()
    def evaluate(self, loader, num_samples=100):
        """Evaluate on validation set"""
        self.model.eval()
        
        all_losses = []
        flip_successes = []
        correlations = []
        
        samples_evaluated = 0
        
        for batch in loader:
            if samples_evaluated >= num_samples:
                break
                
            x, labels = batch
            x = x.to(self.device)
            labels = labels.to(self.device)
            batch_size = x.shape[0]
            
            # Target = opposite class
            target_class = 1 - labels
            
            # Generate counterfactuals
            counterfactuals = self.generate_counterfactual(x, target_class)
            
            # Evaluate flip rate
            logits, _ = self.classifier(counterfactuals)
            preds = logits.argmax(dim=1)
            flip_successes.extend((preds == target_class).cpu().numpy())
            
            # Evaluate correlation
            for i in range(batch_size):
                orig = x[i, 0].cpu().numpy()
                cf = counterfactuals[i, 0].cpu().numpy()
                corr = np.corrcoef(orig, cf)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            samples_evaluated += batch_size
        
        self.model.train()
        
        flip_rate = np.mean(flip_successes) if flip_successes else 0
        mean_corr = np.mean(correlations) if correlations else 0
        
        return {
            'flip_rate': flip_rate,
            'correlation': mean_corr
        }
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("Starting Image-to-Image Diffusion Training")
        print(f"{'='*60}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Noise strength range: [{self.config.noise_strength_min}, {self.config.noise_strength_max}]")
        print(f"{'='*60}\n")
        
        # Save config
        with open(os.path.join(self.config.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        start_time = datetime.now()
        
        for epoch in range(self.config.epochs):
            epoch_start = datetime.now()
            
            # Training
            self.model.train()
            train_losses = []
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
            for batch in pbar:
                losses = self.train_step(batch)
                train_losses.append(losses['total_loss'])
                pbar.set_postfix({'loss': f"{losses['total_loss']:.4f}"})
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            val_metrics = self.evaluate(self.val_loader, num_samples=200)
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_flip_rate'].append(val_metrics['flip_rate'])
            self.history['val_correlation'].append(val_metrics['correlation'])
            
            # LR step
            self.lr_scheduler.step()
            
            # Logging
            epoch_time = (datetime.now() - epoch_start).seconds
            print(f"\nEpoch {epoch+1} Summary (took {epoch_time}s):")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Flip Rate: {val_metrics['flip_rate']*100:.1f}%")
            print(f"  Val Correlation: {val_metrics['correlation']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check (based on validation flip rate + correlation combined)
            val_score = val_metrics['flip_rate'] + val_metrics['correlation']
            if val_score > -self.best_val_loss:  # Maximize combined score
                self.best_val_loss = -val_score
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  → New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Final save
        total_time = (datetime.now() - start_time).seconds / 60
        print(f"\nTraining complete in {total_time:.1f} minutes")
        
        # Save training curves
        self.plot_history()
        
        # Final evaluation on test set
        print("\n" + "="*60)
        print("Final Test Set Evaluation")
        print("="*60)
        self.final_evaluation()
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.config.output_dir, 'best_model.pth')
        else:
            path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, path)
        
    def plot_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Flip Rate
        axes[1].plot(self.history['val_flip_rate'], label='Val Flip Rate', color='green')
        axes[1].axhline(y=0.76, color='r', linestyle='--', label='V2 Baseline (76%)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Flip Rate')
        axes[1].set_title('Validation Flip Rate')
        axes[1].legend()
        axes[1].grid(True)
        
        # Correlation
        axes[2].plot(self.history['val_correlation'], label='Val Correlation', color='purple')
        axes[2].axhline(y=0.922, color='r', linestyle='--', label='V2 Baseline (92.2%)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Correlation')
        axes[2].set_title('Validation Correlation')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'training_history.png'), dpi=150)
        plt.close()
        
    def final_evaluation(self):
        """Comprehensive evaluation on test set"""
        test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False
        )
        
        # Load best model
        best_ckpt = torch.load(os.path.join(self.config.output_dir, 'best_model.pth'))
        self.model.load_state_dict(best_ckpt['model_state_dict'])
        self.model.eval()
        
        results = {
            'normal_to_afib': {'flip': 0, 'total': 0, 'correlations': []},
            'afib_to_normal': {'flip': 0, 'total': 0, 'correlations': []}
        }
        
        with torch.no_grad():
            for x, labels in tqdm(test_loader, desc='Testing'):
                x = x.to(self.device)
                labels = labels.to(self.device)
                target_class = 1 - labels
                
                # Generate counterfactuals
                cfs = self.generate_counterfactual(x, target_class)
                
                # Classify
                logits, _ = self.classifier(cfs)
                preds = logits.argmax(dim=1)
                
                for i in range(x.shape[0]):
                    orig_label = labels[i].item()
                    key = 'normal_to_afib' if orig_label == 0 else 'afib_to_normal'
                    
                    results[key]['total'] += 1
                    if preds[i] == target_class[i]:
                        results[key]['flip'] += 1
                    
                    corr = np.corrcoef(
                        x[i, 0].cpu().numpy(),
                        cfs[i, 0].cpu().numpy()
                    )[0, 1]
                    if not np.isnan(corr):
                        results[key]['correlations'].append(corr)
        
        # Compute summary
        summary = {}
        for key in ['normal_to_afib', 'afib_to_normal']:
            flip_rate = results[key]['flip'] / results[key]['total'] if results[key]['total'] > 0 else 0
            mean_corr = np.mean(results[key]['correlations']) if results[key]['correlations'] else 0
            summary[key] = {
                'flip_rate': flip_rate,
                'correlation': mean_corr,
                'total': results[key]['total']
            }
        
        # Overall
        total = sum(r['total'] for r in results.values())
        total_flip = sum(r['flip'] for r in results.values())
        all_corrs = results['normal_to_afib']['correlations'] + results['afib_to_normal']['correlations']
        
        summary['overall'] = {
            'flip_rate': total_flip / total if total > 0 else 0,
            'correlation': np.mean(all_corrs) if all_corrs else 0,
            'total': total
        }
        
        # Print results
        print("\n" + "-"*60)
        print("| Direction      | Flip Rate | Correlation | Samples |")
        print("-"*60)
        print(f"| Normal→AFib    | {summary['normal_to_afib']['flip_rate']*100:8.1f}% | {summary['normal_to_afib']['correlation']:11.4f} | {summary['normal_to_afib']['total']:7d} |")
        print(f"| AFib→Normal    | {summary['afib_to_normal']['flip_rate']*100:8.1f}% | {summary['afib_to_normal']['correlation']:11.4f} | {summary['afib_to_normal']['total']:7d} |")
        print("-"*60)
        print(f"| Overall        | {summary['overall']['flip_rate']*100:8.1f}% | {summary['overall']['correlation']:11.4f} | {summary['overall']['total']:7d} |")
        print("-"*60)
        
        # Comparison with V2
        print("\nComparison with V2 Residual Method:")
        print(f"  V2 Flip Rate: 76.0% → Img2Img: {summary['overall']['flip_rate']*100:.1f}%")
        print(f"  V2 Correlation: 92.2% → Img2Img: {summary['overall']['correlation']*100:.1f}%")
        
        # Save results
        with open(os.path.join(self.config.output_dir, 'test_results.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Image-to-Image ECG Diffusion')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--noise_strength_min', type=float, default=0.3)
    parser.add_argument('--noise_strength_max', type=float, default=0.8)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = TrainConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.noise_strength_min = args.noise_strength_min
    config.noise_strength_max = args.noise_strength_max
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
