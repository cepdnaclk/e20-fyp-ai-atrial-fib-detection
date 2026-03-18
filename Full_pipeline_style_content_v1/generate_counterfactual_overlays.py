"""
GENERATE COUNTERFACTUALS WITH OVERLAY VISUALIZATION
====================================================

This script generates counterfactual ECGs and creates overlay visualizations
that show exactly what changes when flipping from Normal → AFib or vice versa.

Output:
- Overlay plots showing original (blue) vs counterfactual (red)
- Highlighted regions where changes were made
- Metrics: correlation, flip success, edit magnitude

This is designed as a teaching tool for clinicians to understand:
"What would this Normal patient's ECG look like if they had AFib?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import os

# Import models
from counterfactual_training import AFibResLSTM, ModelConfig
from train_gradient_guided_counterfactual import (
    LocalizedEditNetwork,
    compute_saliency_mask
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def load_models(
    edit_model_path='./gradient_guided_training/best_model.pth',
    classifier_path='./best_model/best_model.pth'
):
    """Load the trained edit model and classifier"""
    
    # Load classifier
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    clf_ckpt = torch.load(classifier_path, map_location=DEVICE, weights_only=False)
    classifier.load_state_dict(clf_ckpt['model_state_dict'])
    classifier.eval()
    print("✅ Classifier loaded")
    
    # Load edit model
    edit_net = LocalizedEditNetwork(hidden_dim=128).to(DEVICE)
    edit_ckpt = torch.load(edit_model_path, map_location=DEVICE, weights_only=False)
    edit_net.load_state_dict(edit_ckpt['model_state_dict'])
    edit_net.eval()
    print(f"✅ Edit model loaded (epoch {edit_ckpt.get('epoch', 'unknown')})")
    print(f"   Flip rate: {edit_ckpt.get('flip_rate', 0)*100:.1f}%")
    print(f"   Correlation: {edit_ckpt.get('mean_correlation', 0):.4f}")
    
    return edit_net, classifier


def generate_counterfactual(edit_net, classifier, ecg, target_class):
    """
    Generate a counterfactual for a single ECG
    
    Args:
        edit_net: LocalizedEditNetwork model
        classifier: AFibResLSTM classifier
        ecg: [1, 1, 2500] ECG tensor
        target_class: int (0=Normal, 1=AFib)
        
    Returns:
        counterfactual: [1, 1, 2500] tensor
        info: dict with metrics
    """
    edit_net.eval()
    classifier.eval()
    
    target = torch.tensor([target_class], device=DEVICE)
    
    with torch.no_grad():
        # Get original prediction
        orig_logits, _ = classifier(ecg)
        orig_pred = orig_logits.argmax(dim=1).item()
        orig_probs = F.softmax(orig_logits, dim=1)
    
    # Compute saliency mask
    ecg_grad = ecg.clone().requires_grad_(True)
    logits, _ = classifier(ecg_grad)
    logits.gather(1, target.unsqueeze(1)).sum().backward()
    saliency = ecg_grad.grad.abs()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency = saliency.detach()
    
    with torch.no_grad():
        # Generate counterfactual
        cf, raw_edit, masked_edit = edit_net(ecg, target, saliency)
        
        # Classify counterfactual
        cf_logits, _ = classifier(cf)
        cf_pred = cf_logits.argmax(dim=1).item()
        cf_probs = F.softmax(cf_logits, dim=1)
        
        # Compute metrics
        orig_np = ecg.cpu().numpy().flatten()
        cf_np = cf.cpu().numpy().flatten()
        correlation = np.corrcoef(orig_np, cf_np)[0, 1]
        
        edit_np = masked_edit.cpu().numpy().flatten()
        edit_magnitude = np.abs(edit_np).mean()
        edit_max = np.abs(edit_np).max()
        
        flip_success = (cf_pred == target_class)
    
    info = {
        'original_class': orig_pred,
        'target_class': target_class,
        'counterfactual_class': cf_pred,
        'flip_success': flip_success,
        'correlation': float(correlation),
        'edit_magnitude_mean': float(edit_magnitude),
        'edit_magnitude_max': float(edit_max),
        'original_probs': orig_probs.cpu().numpy().tolist()[0],
        'counterfactual_probs': cf_probs.cpu().numpy().tolist()[0],
    }
    
    return cf, masked_edit, saliency, info


def create_overlay_visualization(
    ecg_original, ecg_counterfactual, edit, saliency, info,
    save_path=None, show=False, sampling_rate=250
):
    """
    Create a detailed overlay visualization for clinical teaching
    
    Shows:
    1. Full ECG overlay (original blue, counterfactual red)
    2. Edit magnitude (yellow highlight where changes occurred)
    3. Zoomed view of interesting regions
    4. Metrics and statistics
    """
    class_names = ['Normal', 'AFib']
    
    orig_np = ecg_original.cpu().numpy().flatten()
    cf_np = ecg_counterfactual.cpu().numpy().flatten()
    edit_np = edit.cpu().numpy().flatten()
    saliency_np = saliency.cpu().numpy().flatten()
    
    t = np.arange(len(orig_np)) / sampling_rate  # Time in seconds
    
    fig = plt.figure(figsize=(20, 14))
    
    # Create grid
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    # ========================================
    # Main overlay plot (spans full width)
    # ========================================
    ax_main = fig.add_subplot(gs[0:2, :])
    
    # Plot both ECGs
    ax_main.plot(t, orig_np, 'b-', linewidth=1.0, alpha=0.8, label=f'Original ({class_names[info["original_class"]]})')
    ax_main.plot(t, cf_np, 'r-', linewidth=1.0, alpha=0.8, label=f'Counterfactual ({class_names[info["counterfactual_class"]]})')
    
    # Highlight edit regions
    significant_edit = np.abs(edit_np) > 0.02 * np.abs(edit_np).max()
    if np.any(significant_edit):
        ymin, ymax = ax_main.get_ylim()
        ax_main.fill_between(t, ymin, ymax, where=significant_edit, 
                            alpha=0.2, color='yellow', label='Edit regions')
    
    # Title with key info
    flip_status = "✓ FLIPPED" if info['flip_success'] else "✗ NOT FLIPPED"
    ax_main.set_title(
        f"ECG Counterfactual: {class_names[info['original_class']]} → {class_names[info['target_class']]}  |  "
        f"{flip_status}  |  Correlation: {info['correlation']:.3f}",
        fontsize=14, fontweight='bold'
    )
    ax_main.set_xlabel('Time (seconds)', fontsize=12)
    ax_main.set_ylabel('Amplitude (normalized)', fontsize=12)
    ax_main.legend(loc='upper right', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim([0, t[-1]])
    
    # ========================================
    # Zoomed views (3 windows of ~2s each)
    # ========================================
    zoom_windows = [
        (0, 2.5),      # First heartbeats
        (3.5, 6.0),    # Middle
        (7.0, 9.5),    # End
    ]
    
    for i, (start, end) in enumerate(zoom_windows):
        ax = fig.add_subplot(gs[2, i])
        mask = (t >= start) & (t <= end)
        
        ax.plot(t[mask], orig_np[mask], 'b-', linewidth=1.2, alpha=0.8, label='Original')
        ax.plot(t[mask], cf_np[mask], 'r-', linewidth=1.2, alpha=0.8, label='Counterfactual')
        
        # Highlight edits
        if np.any(significant_edit[mask]):
            ax.fill_between(t[mask], orig_np[mask].min(), orig_np[mask].max(),
                           where=significant_edit[mask], alpha=0.3, color='yellow')
        
        ax.set_title(f'Zoom: {start:.1f}s - {end:.1f}s', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('Amplitude')
    
    # ========================================
    # Difference plot
    # ========================================
    ax_diff = fig.add_subplot(gs[2, 3])
    diff = cf_np - orig_np
    ax_diff.plot(t, diff, 'g-', linewidth=0.5)
    ax_diff.fill_between(t, 0, diff, alpha=0.3, color='green')
    ax_diff.set_title('Difference (CF - Original)', fontsize=10)
    ax_diff.set_xlabel('Time (s)')
    ax_diff.set_ylabel('Δ Amplitude')
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_xlim([0, t[-1]])
    
    # ========================================
    # Saliency map (what classifier focuses on)
    # ========================================
    ax_sal = fig.add_subplot(gs[3, 0:2])
    ax_sal.fill_between(t, 0, saliency_np, alpha=0.7, color='purple')
    ax_sal.set_title('Classifier Attention (Saliency Map)', fontsize=10)
    ax_sal.set_xlabel('Time (s)')
    ax_sal.set_ylabel('Importance')
    ax_sal.set_xlim([0, t[-1]])
    ax_sal.set_ylim([0, 1])
    ax_sal.grid(True, alpha=0.3)
    
    # ========================================
    # Metrics panel
    # ========================================
    ax_metrics = fig.add_subplot(gs[3, 2:4])
    ax_metrics.axis('off')
    
    metrics_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  COUNTERFACTUAL GENERATION METRICS                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Original Class:        {class_names[info['original_class']]:>10}                        ║
    ║  Target Class:          {class_names[info['target_class']]:>10}                        ║
    ║  Counterfactual Class:  {class_names[info['counterfactual_class']]:>10}                        ║
    ║                                                              ║
    ║  Flip Success:          {'Yes ✓' if info['flip_success'] else 'No ✗':>10}                        ║
    ║                                                              ║
    ║  ── Similarity Metrics ──────────────────────────────────    ║
    ║  Correlation:           {info['correlation']:>10.4f}                        ║
    ║  Mean Edit Magnitude:   {info['edit_magnitude_mean']:>10.4f}                        ║
    ║  Max Edit Magnitude:    {info['edit_magnitude_max']:>10.4f}                        ║
    ║                                                              ║
    ║  ── Classifier Confidence ───────────────────────────────    ║
    ║  Original - Normal:     {info['original_probs'][0]*100:>10.1f}%                       ║
    ║  Original - AFib:       {info['original_probs'][1]*100:>10.1f}%                       ║
    ║  CF - Normal:           {info['counterfactual_probs'][0]*100:>10.1f}%                       ║
    ║  CF - AFib:             {info['counterfactual_probs'][1]*100:>10.1f}%                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax_metrics.text(0.0, 0.5, metrics_text, fontfamily='monospace', fontsize=9,
                   verticalalignment='center', transform=ax_metrics.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def generate_batch_counterfactuals(
    edit_net, classifier,
    data_path='./ecg_afib_data/X_combined.npy',
    labels_path='./ecg_afib_data/y_combined.npy',
    norm_params_path='./enhanced_counterfactual_training/norm_params.npy',
    output_dir='./counterfactual_overlays',
    num_samples=50,
    visualize_count=20
):
    """
    Generate counterfactuals for multiple samples and create visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING COUNTERFACTUAL ECGs")
    print("="*70)
    
    # Load data
    signals = np.load(data_path)
    labels = np.load(labels_path)
    norm_params = np.load(norm_params_path, allow_pickle=True).item()
    
    if labels.dtype.kind in ['U', 'S', 'O']:
        labels = np.array([1 if l == 'A' else 0 for l in labels])
    
    means = norm_params['means']
    stds = norm_params['stds']
    
    # Sample indices (balanced between classes)
    normal_idx = np.where(labels == 0)[0]
    afib_idx = np.where(labels == 1)[0]
    
    n_per_class = num_samples // 2
    selected_normal = np.random.choice(normal_idx, min(n_per_class, len(normal_idx)), replace=False)
    selected_afib = np.random.choice(afib_idx, min(n_per_class, len(afib_idx)), replace=False)
    selected = np.concatenate([selected_normal, selected_afib])
    np.random.shuffle(selected)
    
    print(f"\nGenerating counterfactuals for {len(selected)} samples...")
    print(f"   {len(selected_normal)} Normal → AFib")
    print(f"   {len(selected_afib)} AFib → Normal")
    
    results = []
    
    for i, idx in enumerate(selected):
        signal = signals[idx]
        label = labels[idx]
        mean, std = means[idx], stds[idx]
        
        # Normalize
        normalized = (signal - mean) / (std + 1e-6)
        ecg = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Target is opposite class
        target_class = 1 - label
        
        # Generate counterfactual
        cf, edit, saliency, info = generate_counterfactual(edit_net, classifier, ecg, target_class)
        
        info['sample_idx'] = int(idx)
        results.append(info)
        
        # Create visualization for first N samples
        if i < visualize_count:
            save_path = f'{output_dir}/sample_{i+1:03d}_idx{idx}.png'
            create_overlay_visualization(ecg, cf, edit, saliency, info, save_path=save_path)
        
        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{len(selected)} samples")
    
    # Compute summary statistics
    flip_rate = np.mean([r['flip_success'] for r in results])
    mean_correlation = np.mean([r['correlation'] for r in results])
    std_correlation = np.std([r['correlation'] for r in results])
    mean_edit_mag = np.mean([r['edit_magnitude_mean'] for r in results])
    
    # Separate by direction
    normal_to_afib = [r for r in results if r['original_class'] == 0]
    afib_to_normal = [r for r in results if r['original_class'] == 1]
    
    summary = {
        'total_samples': len(results),
        'overall': {
            'flip_rate': float(flip_rate),
            'mean_correlation': float(mean_correlation),
            'std_correlation': float(std_correlation),
            'mean_edit_magnitude': float(mean_edit_mag),
        },
        'normal_to_afib': {
            'count': len(normal_to_afib),
            'flip_rate': float(np.mean([r['flip_success'] for r in normal_to_afib])) if normal_to_afib else 0,
            'mean_correlation': float(np.mean([r['correlation'] for r in normal_to_afib])) if normal_to_afib else 0,
        },
        'afib_to_normal': {
            'count': len(afib_to_normal),
            'flip_rate': float(np.mean([r['flip_success'] for r in afib_to_normal])) if afib_to_normal else 0,
            'mean_correlation': float(np.mean([r['correlation'] for r in afib_to_normal])) if afib_to_normal else 0,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save results
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump({'summary': summary, 'samples': results}, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"""
    Total Samples: {summary['total_samples']}
    
    Overall:
      - Flip Rate:        {summary['overall']['flip_rate']*100:.1f}%
      - Mean Correlation: {summary['overall']['mean_correlation']:.4f} ± {summary['overall']['std_correlation']:.4f}
      - Mean Edit Size:   {summary['overall']['mean_edit_magnitude']:.4f}
    
    Normal → AFib ({summary['normal_to_afib']['count']} samples):
      - Flip Rate:        {summary['normal_to_afib']['flip_rate']*100:.1f}%
      - Mean Correlation: {summary['normal_to_afib']['mean_correlation']:.4f}
    
    AFib → Normal ({summary['afib_to_normal']['count']} samples):
      - Flip Rate:        {summary['afib_to_normal']['flip_rate']*100:.1f}%
      - Mean Correlation: {summary['afib_to_normal']['mean_correlation']:.4f}
    
    Output saved to: {output_dir}/
    """)
    
    # Create summary plot
    create_summary_plot(results, f'{output_dir}/summary_plot.png')
    
    return summary, results


def create_summary_plot(results, save_path):
    """Create a summary plot of all results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    correlations = [r['correlation'] for r in results]
    flip_success = [r['flip_success'] for r in results]
    edit_mags = [r['edit_magnitude_mean'] for r in results]
    
    # Correlation histogram
    axes[0, 0].hist(correlations, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(correlations):.3f}')
    axes[0, 0].set_title('Distribution of Correlations')
    axes[0, 0].set_xlabel('Correlation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Correlation vs Edit magnitude
    colors = ['green' if s else 'red' for s in flip_success]
    axes[0, 1].scatter(edit_mags, correlations, c=colors, alpha=0.6)
    axes[0, 1].set_xlabel('Edit Magnitude (mean)')
    axes[0, 1].set_ylabel('Correlation')
    axes[0, 1].set_title('Correlation vs Edit Magnitude\n(Green=Flipped, Red=Not Flipped)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Success rate by original class
    normal_to_afib = [r for r in results if r['original_class'] == 0]
    afib_to_normal = [r for r in results if r['original_class'] == 1]
    
    flip_rates = [
        np.mean([r['flip_success'] for r in normal_to_afib]) if normal_to_afib else 0,
        np.mean([r['flip_success'] for r in afib_to_normal]) if afib_to_normal else 0,
    ]
    bars = axes[1, 0].bar(['Normal → AFib', 'AFib → Normal'], [fr * 100 for fr in flip_rates],
                          color=['steelblue', 'coral'], edgecolor='black')
    axes[1, 0].set_ylabel('Flip Rate (%)')
    axes[1, 0].set_title('Flip Success Rate by Direction')
    axes[1, 0].set_ylim([0, 100])
    for bar, rate in zip(bars, flip_rates):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{rate*100:.1f}%', ha='center', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Correlation by direction
    corr_by_dir = [
        [r['correlation'] for r in normal_to_afib],
        [r['correlation'] for r in afib_to_normal],
    ]
    axes[1, 1].boxplot(corr_by_dir, labels=['Normal → AFib', 'AFib → Normal'])
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_title('Correlation Distribution by Direction')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate counterfactual ECGs')
    parser.add_argument('--model', type=str, default='./gradient_guided_training/best_model.pth',
                       help='Path to trained edit model')
    parser.add_argument('--classifier', type=str, default='./best_model/best_model.pth',
                       help='Path to classifier')
    parser.add_argument('--output', type=str, default='./counterfactual_overlays',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to process')
    parser.add_argument('--visualize', type=int, default=20,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Load models
    edit_net, classifier = load_models(args.model, args.classifier)
    
    # Generate counterfactuals
    summary, results = generate_batch_counterfactuals(
        edit_net, classifier,
        output_dir=args.output,
        num_samples=args.num_samples,
        visualize_count=args.visualize
    )
