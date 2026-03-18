"""
Generate Counterfactual Overlays Using Trained Minimal Edit Model
==================================================================

This script loads your trained minimal edit model that achieves:
- Flip Rate: 99.9%
- Mean Correlation: 0.9927 (99.27%)

And generates clinical overlay visualizations for teaching purposes.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import json

# Import the model architecture from training script
from train_minimal_edit_model import ResidualEditModel
from counterfactual_training import AFibResLSTM, ModelConfig

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def load_classifier(checkpoint_path):
    """Load the pre-trained AFib classifier."""
    config = ModelConfig()
    model = AFibResLSTM(config).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Classifier loaded from {checkpoint_path}")
    return model


def load_minimal_edit_model(checkpoint_path):
    """Load the trained minimal edit model."""
    model = ResidualEditModel(hidden_dim=256).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Minimal edit model loaded from {checkpoint_path}")
    
    # Print the learned edit strength
    with torch.no_grad():
        edit_strength = torch.sigmoid(model.edit_strength).item()
        print(f"   Learned edit strength: {edit_strength:.4f}")
    
    return model


def load_data(data_dir="./ecg_afib_data"):
    """Load ECG data."""
    X = np.load(f"{data_dir}/X_combined.npy")
    y = np.load(f"{data_dir}/y_combined.npy")
    
    # Convert labels to binary
    labels = (y == 'A').astype(np.int64)  # 1 for AFib, 0 for Normal
    
    print(f"✅ Loaded {len(X)} ECG samples")
    print(f"   AFib: {np.sum(labels == 1)}, Normal: {np.sum(labels == 0)}")
    
    return X, labels


def generate_counterfactual(model, ecg, target_class):
    """Generate counterfactual ECG using the minimal edit model."""
    with torch.no_grad():
        ecg_tensor = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        target_tensor = torch.tensor([target_class], dtype=torch.long).to(DEVICE)
        
        counterfactual, edit = model(ecg_tensor, target_tensor)
        
        return counterfactual.squeeze().cpu().numpy(), edit.squeeze().cpu().numpy()


def get_classification(classifier, ecg):
    """Get classifier prediction and confidence."""
    with torch.no_grad():
        ecg_tensor = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        output = classifier(ecg_tensor)
        # Handle if model returns tuple (logits, other_outputs)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
        return pred, confidence, probs[0].cpu().numpy()


def plot_clinical_overlay(original, counterfactual, edit, 
                          orig_pred, cf_pred, orig_probs, cf_probs,
                          sample_idx, output_dir):
    """Create clinical overlay visualization for teaching."""
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    time = np.arange(len(original)) / 250  # 250 Hz sampling rate
    
    class_names = ['Normal', 'AFib']
    orig_class = class_names[orig_pred]
    cf_class = class_names[cf_pred]
    
    # 1. Original ECG
    ax1 = axes[0]
    ax1.plot(time, original, 'b-', linewidth=0.8, label='Original ECG')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Original ECG - Classified as: {orig_class} (confidence: {orig_probs[orig_pred]:.1%})', 
                  fontsize=12, fontweight='bold', color='blue')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time[-1]])
    
    # 2. Counterfactual ECG
    ax2 = axes[1]
    ax2.plot(time, counterfactual, 'r-', linewidth=0.8, label='Counterfactual ECG')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Counterfactual ECG - Classified as: {cf_class} (confidence: {cf_probs[cf_pred]:.1%})', 
                  fontsize=12, fontweight='bold', color='red')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, time[-1]])
    
    # 3. Overlay (the key visualization for clinicians!)
    ax3 = axes[2]
    ax3.plot(time, original, 'b-', linewidth=0.8, alpha=0.7, label=f'Original ({orig_class})')
    ax3.plot(time, counterfactual, 'r-', linewidth=0.8, alpha=0.7, label=f'Counterfactual ({cf_class})')
    
    # Highlight regions where changes were made
    edit_magnitude = np.abs(edit)
    threshold = np.percentile(edit_magnitude, 90)  # Top 10% of edits
    significant_edit_mask = edit_magnitude > threshold
    
    # Create shaded regions for significant edits
    ax3.fill_between(time, ax3.get_ylim()[0], ax3.get_ylim()[1],
                     where=significant_edit_mask, alpha=0.2, color='yellow',
                     label='Significant changes')
    
    ax3.set_ylabel('Amplitude')
    ax3.set_title('OVERLAY: Original (Blue) vs Counterfactual (Red) - Yellow regions show key changes', 
                  fontsize=12, fontweight='bold', color='purple')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, time[-1]])
    
    # 4. Edit magnitude (shows where changes were made)
    ax4 = axes[3]
    ax4.fill_between(time, 0, np.abs(edit), color='green', alpha=0.5, label='Edit magnitude')
    ax4.axhline(y=threshold, color='red', linestyle='--', label=f'Significance threshold')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('|Edit|')
    ax4.set_title('Edit Magnitude - Shows WHERE changes were applied to flip classification', 
                  fontsize=12, fontweight='bold', color='green')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, time[-1]])
    
    # Calculate and display metrics
    correlation = np.corrcoef(original, counterfactual)[0, 1]
    total_edit = np.sum(np.abs(edit))
    max_edit = np.max(np.abs(edit))
    
    fig.suptitle(f'Sample {sample_idx}: {orig_class} → {cf_class} | Correlation: {correlation:.4f} | Edit Sum: {total_edit:.2f}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overlay_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'sample_idx': int(sample_idx),
        'original_class': orig_class,
        'counterfactual_class': cf_class,
        'flip_success': bool(orig_pred != cf_pred),
        'correlation': float(correlation),
        'total_edit': float(total_edit),
        'max_edit': float(max_edit),
        'original_confidence': float(orig_probs[orig_pred]),
        'counterfactual_confidence': float(cf_probs[cf_pred])
    }


def plot_zoomed_comparison(original, counterfactual, edit, sample_idx, output_dir):
    """Create zoomed views of specific time windows for detailed analysis."""
    
    time = np.arange(len(original)) / 250
    
    # Find the region with the most significant edits
    edit_magnitude = np.abs(edit)
    window_size = 500  # 2 seconds at 250 Hz
    
    # Sliding window to find the region with most edits
    best_window_start = 0
    best_window_score = 0
    for i in range(0, len(edit) - window_size, 50):
        window_score = np.sum(edit_magnitude[i:i+window_size])
        if window_score > best_window_score:
            best_window_score = window_score
            best_window_start = i
    
    # Create zoomed plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    start = best_window_start
    end = start + window_size
    time_window = time[start:end]
    
    # Zoomed overlay
    ax1 = axes[0]
    ax1.plot(time_window, original[start:end], 'b-', linewidth=1.5, label='Original', marker='o', markersize=2)
    ax1.plot(time_window, counterfactual[start:end], 'r-', linewidth=1.5, label='Counterfactual', marker='s', markersize=2)
    ax1.fill_between(time_window, original[start:end], counterfactual[start:end], alpha=0.3, color='yellow')
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'ZOOMED VIEW (2 seconds) - Region with Most Significant Changes', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.5)
    
    # Edit in this window
    ax2 = axes[1]
    ax2.bar(time_window, np.abs(edit[start:end]), width=0.004, color='green', alpha=0.7)
    ax2.axhline(y=np.percentile(edit_magnitude, 90), color='red', linestyle='--', label='90th percentile')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('|Edit|', fontsize=12)
    ax2.set_title('Edit Magnitude in Zoomed Window', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zoomed_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Generate counterfactual overlays using the trained minimal edit model."""
    
    # Create output directory
    output_dir = f"./counterfactual_overlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load models
    classifier = load_classifier("./best_model/best_model.pth")
    edit_model = load_minimal_edit_model("./minimal_edit_training/best_model.pth")
    
    # Load data
    X, labels = load_data()
    
    # Select samples for visualization (mix of Normal and AFib)
    np.random.seed(42)  # For reproducibility
    
    normal_indices = np.where(labels == 0)[0]
    afib_indices = np.where(labels == 1)[0]
    
    # Select 5 from each class
    selected_normal = np.random.choice(normal_indices, size=min(5, len(normal_indices)), replace=False)
    selected_afib = np.random.choice(afib_indices, size=min(5, len(afib_indices)), replace=False)
    selected_indices = np.concatenate([selected_normal, selected_afib])
    
    print(f"\n📊 Generating overlays for {len(selected_indices)} samples...")
    
    results = []
    
    for i, idx in enumerate(selected_indices):
        ecg = X[idx]
        original_label = labels[idx]
        target_label = 1 - original_label  # Flip to opposite class
        
        # Get original classification
        orig_pred, orig_conf, orig_probs = get_classification(classifier, ecg)
        
        # Generate counterfactual
        counterfactual, edit = generate_counterfactual(edit_model, ecg, target_label)
        
        # Get counterfactual classification
        cf_pred, cf_conf, cf_probs = get_classification(classifier, counterfactual)
        
        # Create visualizations
        result = plot_clinical_overlay(
            ecg, counterfactual, edit,
            orig_pred, cf_pred, orig_probs, cf_probs,
            idx, output_dir
        )
        
        plot_zoomed_comparison(ecg, counterfactual, edit, idx, output_dir)
        
        results.append(result)
        
        print(f"   Sample {i+1}/{len(selected_indices)}: "
              f"{'Normal' if original_label == 0 else 'AFib'} → "
              f"{'Normal' if cf_pred == 0 else 'AFib'} | "
              f"Corr: {result['correlation']:.4f} | "
              f"Flip: {'✅' if result['flip_success'] else '❌'}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    flip_success_rate = np.mean([r['flip_success'] for r in results])
    mean_correlation = np.mean([r['correlation'] for r in results])
    mean_edit = np.mean([r['total_edit'] for r in results])
    
    print(f"  Flip Success Rate: {flip_success_rate:.1%}")
    print(f"  Mean Correlation:  {mean_correlation:.4f}")
    print(f"  Mean Total Edit:   {mean_edit:.2f}")
    
    # Save results
    summary = {
        'flip_success_rate': float(flip_success_rate),
        'mean_correlation': float(mean_correlation),
        'mean_total_edit': float(mean_edit),
        'individual_results': results
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}")
    print(f"   - {len(results)} overlay images generated")
    print(f"   - {len(results)} zoomed comparison images generated")
    print(f"   - Summary saved to summary.json")
    
    return output_dir, results


if __name__ == "__main__":
    output_dir, results = main()
