"""
Generate Overlays for SUCCESSFUL Counterfactuals Only
======================================================

This script finds samples where the counterfactual successfully flips the class
and generates visualization overlays for those successful cases.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import json

from train_minimal_edit_model import ResidualEditModel
from counterfactual_training import AFibResLSTM, ModelConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("Loading models...")
    
    # Load classifier
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    checkpoint = torch.load('./best_model/best_model.pth', map_location=DEVICE, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    classifier.eval()
    
    # Load edit model
    edit_model = ResidualEditModel(hidden_dim=256).to(DEVICE)
    checkpoint = torch.load('./minimal_edit_training/best_model.pth', map_location=DEVICE, weights_only=False)
    edit_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    edit_model.eval()
    
    # Load data
    X = np.load('./ecg_afib_data/X_combined.npy')
    y = np.load('./ecg_afib_data/y_combined.npy')
    labels = (y == 'A').astype(np.int64)
    
    print(f"Loaded {len(X)} samples")
    
    # Create output directory
    output_dir = f"./successful_counterfactuals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find successful flips with high correlation
    np.random.seed(123)
    test_indices = np.random.choice(len(X), size=1000, replace=False)
    
    successful_n2a = []  # Normal to AFib
    successful_a2n = []  # AFib to Normal
    
    print("Searching for successful high-quality counterfactuals...")
    
    with torch.no_grad():
        for idx in test_indices:
            ecg = torch.tensor(X[idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            original_label = labels[idx]
            target_label = 1 - original_label
            
            target_tensor = torch.tensor([target_label], dtype=torch.long).to(DEVICE)
            cf, edit = edit_model(ecg, target_tensor)
            
            logits, _ = classifier(cf)
            pred = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, pred].item()
            
            corr = np.corrcoef(ecg.squeeze().cpu().numpy(), cf.squeeze().cpu().numpy())[0, 1]
            
            # Successful flip with good correlation
            if pred == target_label and corr > 0.9:
                result = {
                    'idx': int(idx),
                    'original_label': int(original_label),
                    'target_label': int(target_label),
                    'pred': int(pred),
                    'confidence': float(confidence),
                    'correlation': float(corr),
                    'ecg': ecg.squeeze().cpu().numpy(),
                    'cf': cf.squeeze().cpu().numpy(),
                    'edit': edit.squeeze().cpu().numpy()
                }
                
                if original_label == 0:  # Normal to AFib
                    successful_n2a.append(result)
                else:  # AFib to Normal
                    successful_a2n.append(result)
    
    print(f"Found {len(successful_n2a)} high-quality Normal→AFib counterfactuals")
    print(f"Found {len(successful_a2n)} high-quality AFib→Normal counterfactuals")
    
    # Generate overlays for the best examples
    class_names = ['Normal', 'AFib']
    
    def plot_overlay(result, output_path):
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        time = np.arange(len(result['ecg'])) / 250
        
        orig_class = class_names[result['original_label']]
        cf_class = class_names[result['target_label']]
        
        # Panel 1: Original
        ax1 = axes[0]
        ax1.plot(time, result['ecg'], 'b-', linewidth=0.8, label=f'Original ({orig_class})')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Original ECG - {orig_class}', fontsize=12, fontweight='bold', color='blue')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, time[-1]])
        
        # Panel 2: Counterfactual
        ax2 = axes[1]
        ax2.plot(time, result['cf'], 'r-', linewidth=0.8, label=f'Counterfactual ({cf_class})')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Counterfactual ECG - {cf_class} (confidence: {result["confidence"]:.1%})', 
                      fontsize=12, fontweight='bold', color='red')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, time[-1]])
        
        # Panel 3: OVERLAY - This is the key visualization!
        ax3 = axes[2]
        ax3.plot(time, result['ecg'], 'b-', linewidth=1.0, alpha=0.7, label=f'Original ({orig_class})')
        ax3.plot(time, result['cf'], 'r-', linewidth=1.0, alpha=0.7, label=f'Counterfactual ({cf_class})')
        
        # Highlight significant changes
        edit_mag = np.abs(result['edit'])
        threshold = np.percentile(edit_mag, 90)
        significant = edit_mag > threshold
        
        y_min, y_max = ax3.get_ylim()
        ax3.fill_between(time, y_min, y_max, where=significant, 
                         alpha=0.2, color='yellow', label='Key changes')
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title(f'OVERLAY: What minimal changes convert {orig_class} → {cf_class}? (Correlation: {result["correlation"]:.4f})',
                      fontsize=12, fontweight='bold', color='purple')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, time[-1]])
        
        fig.suptitle(f'Sample {result["idx"]}: Successful Counterfactual | {orig_class} → {cf_class}',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return result['correlation']
    
    # Sort by correlation and take best 5 from each direction
    successful_n2a.sort(key=lambda x: x['correlation'], reverse=True)
    successful_a2n.sort(key=lambda x: x['correlation'], reverse=True)
    
    print("\n" + "="*60)
    print("Generating overlays for best Normal→AFib counterfactuals:")
    print("="*60)
    for i, result in enumerate(successful_n2a[:5]):
        path = f"{output_dir}/N2A_sample_{result['idx']}_corr{result['correlation']:.3f}.png"
        corr = plot_overlay(result, path)
        print(f"  {i+1}. Sample {result['idx']}: Correlation = {corr:.4f}")
    
    print("\n" + "="*60)
    print("Generating overlays for best AFib→Normal counterfactuals:")
    print("="*60)
    for i, result in enumerate(successful_a2n[:5]):
        path = f"{output_dir}/A2N_sample_{result['idx']}_corr{result['correlation']:.3f}.png"
        corr = plot_overlay(result, path)
        print(f"  {i+1}. Sample {result['idx']}: Correlation = {corr:.4f}")
    
    # Create a zoomed comparison for the best example from each direction
    print("\n" + "="*60)
    print("Creating detailed zoomed views...")
    print("="*60)
    
    def plot_zoomed(result, output_path, direction):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        time = np.arange(len(result['ecg'])) / 250
        
        # Find region with most edits
        edit_mag = np.abs(result['edit'])
        window_size = 500  # 2 seconds
        best_start = 0
        best_score = 0
        for i in range(0, len(result['edit']) - window_size, 50):
            score = np.sum(edit_mag[i:i+window_size])
            if score > best_score:
                best_score = score
                best_start = i
        
        start = best_start
        end = start + window_size
        
        # Full signal overlay
        ax1 = axes[0, 0]
        ax1.plot(time, result['ecg'], 'b-', linewidth=0.8, alpha=0.7, label='Original')
        ax1.plot(time, result['cf'], 'r-', linewidth=0.8, alpha=0.7, label='Counterfactual')
        ax1.axvspan(time[start], time[end], alpha=0.2, color='green', label='Zoomed region')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Full ECG Overlay', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Zoomed overlay
        ax2 = axes[0, 1]
        tw = time[start:end]
        ax2.plot(tw, result['ecg'][start:end], 'b-', linewidth=1.5, label='Original', marker='o', markersize=1)
        ax2.plot(tw, result['cf'][start:end], 'r-', linewidth=1.5, label='Counterfactual', marker='s', markersize=1)
        ax2.fill_between(tw, result['ecg'][start:end], result['cf'][start:end], alpha=0.3, color='yellow')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('ZOOMED: Region with Key Changes', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.5)
        
        # Difference signal
        ax3 = axes[1, 0]
        diff = result['cf'] - result['ecg']
        ax3.fill_between(time, 0, diff, where=diff >= 0, color='green', alpha=0.5, label='Added')
        ax3.fill_between(time, 0, diff, where=diff < 0, color='red', alpha=0.5, label='Removed')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Δ Amplitude')
        ax3.set_title('Difference: Counterfactual - Original', fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Edit magnitude histogram
        ax4 = axes[1, 1]
        ax4.hist(edit_mag, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(np.percentile(edit_mag, 90), color='red', linestyle='--', label='90th percentile')
        ax4.axvline(np.mean(edit_mag), color='green', linestyle='--', label='Mean')
        ax4.set_xlabel('Edit Magnitude')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Edit Magnitudes', fontweight='bold')
        ax4.legend(loc='upper right')
        
        class_names = ['Normal', 'AFib']
        orig = class_names[result['original_label']]
        tgt = class_names[result['target_label']]
        
        fig.suptitle(f'{direction}: {orig} → {tgt} | Correlation: {result["correlation"]:.4f} | Sample {result["idx"]}',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    if successful_n2a:
        plot_zoomed(successful_n2a[0], f"{output_dir}/ZOOMED_best_N2A.png", "Normal → AFib")
        print(f"  Created detailed zoomed view for best Normal→AFib")
    
    if successful_a2n:
        plot_zoomed(successful_a2n[0], f"{output_dir}/ZOOMED_best_A2N.png", "AFib → Normal")
        print(f"  Created detailed zoomed view for best AFib→Normal")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"   Total images generated: {min(5, len(successful_n2a)) + min(5, len(successful_a2n)) + 2}")


if __name__ == "__main__":
    main()
