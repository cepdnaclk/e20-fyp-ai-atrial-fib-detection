"""
Generate counterfactual overlays using the trained V2 model.
This script evaluates the model on held-out test data and creates visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from tqdm import tqdm

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def load_models():
    """Load classifier and counterfactual model"""
    # Load classifier
    print("Loading classifier...")
    from counterfactual_training import AFibResLSTM, ModelConfig
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    clf_ckpt = torch.load('./best_model/best_model.pth', map_location=DEVICE, weights_only=False)
    classifier.load_state_dict(clf_ckpt['model_state_dict'])
    classifier.eval()
    
    # Load V2 counterfactual model
    print("Loading V2 counterfactual model...")
    from train_minimal_edit_v2 import ResidualEditModelV2
    edit_model = ResidualEditModelV2(dropout=0.1).to(DEVICE)
    edit_ckpt = torch.load('./minimal_edit_v2/best_model.pth', map_location=DEVICE, weights_only=False)
    edit_model.load_state_dict(edit_ckpt['model_state_dict'])
    edit_model.eval()
    
    print(f"   Edit strength: {torch.sigmoid(edit_model.edit_strength).item():.4f}")
    
    return classifier, edit_model

def verify_on_test_samples(classifier, edit_model, X, y, test_indices, n_samples=100):
    """Verify model performance on test samples"""
    print(f"\nVerifying on {n_samples} test samples...")
    
    np.random.seed(42)
    sample_indices = np.random.choice(test_indices, min(n_samples, len(test_indices)), replace=False)
    
    results = {
        'normal_to_afib': {'flipped': 0, 'total': 0, 'correlations': []},
        'afib_to_normal': {'flipped': 0, 'total': 0, 'correlations': []}
    }
    
    for idx in tqdm(sample_indices, desc="Verifying"):
        ecg = X[idx]
        label = 0 if y[idx] == 'A' else 1  # 0=AFib, 1=Normal
        target = 1 - label
        
        ecg_tensor = torch.FloatTensor(ecg).unsqueeze(0).unsqueeze(0).to(DEVICE)
        target_tensor = torch.LongTensor([target]).to(DEVICE)
        
        with torch.no_grad():
            # Get original prediction
            orig_logits, _ = classifier(ecg_tensor)
            orig_pred = orig_logits.argmax(dim=1).item()
            
            # Generate counterfactual
            cf, edit, _ = edit_model(ecg_tensor, target_tensor)
            
            # Get counterfactual prediction
            cf_logits, _ = classifier(cf)
            cf_pred = cf_logits.argmax(dim=1).item()
            
            # Compute correlation
            ecg_np = ecg.flatten()
            cf_np = cf.squeeze().cpu().numpy().flatten()
            corr = np.corrcoef(ecg_np, cf_np)[0, 1]
        
        # Track results
        if label == 1:  # Normal -> AFib
            results['normal_to_afib']['total'] += 1
            if cf_pred == 0:  # Flipped to AFib
                results['normal_to_afib']['flipped'] += 1
            results['normal_to_afib']['correlations'].append(corr)
        else:  # AFib -> Normal
            results['afib_to_normal']['total'] += 1
            if cf_pred == 1:  # Flipped to Normal
                results['afib_to_normal']['flipped'] += 1
            results['afib_to_normal']['correlations'].append(corr)
    
    # Print results
    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)
    
    for direction, data in results.items():
        if data['total'] > 0:
            flip_rate = data['flipped'] / data['total'] * 100
            mean_corr = np.mean(data['correlations'])
            print(f"\n{direction.replace('_', ' ').title()}:")
            print(f"   Flip Rate: {data['flipped']}/{data['total']} ({flip_rate:.1f}%)")
            print(f"   Mean Correlation: {mean_corr:.4f}")
    
    # Overall
    total = sum(d['total'] for d in results.values())
    flipped = sum(d['flipped'] for d in results.values())
    all_corrs = results['normal_to_afib']['correlations'] + results['afib_to_normal']['correlations']
    
    print(f"\nOVERALL:")
    print(f"   Flip Rate: {flipped}/{total} ({flipped/total*100:.1f}%)")
    print(f"   Mean Correlation: {np.mean(all_corrs):.4f}")
    
    return results

def generate_overlay_plot(ecg_orig, ecg_cf, orig_label, cf_pred, correlation, sample_idx, save_dir):
    """Generate a clinical overlay visualization"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    t = np.arange(len(ecg_orig)) / 250  # Time in seconds
    
    # Panel 1: Original
    axes[0].plot(t, ecg_orig, 'b-', linewidth=0.8, label='Original ECG')
    axes[0].set_title(f'Original ECG - Classified as: {"AFib" if orig_label == 0 else "Normal"}', fontsize=14)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Counterfactual
    axes[1].plot(t, ecg_cf, 'r-', linewidth=0.8, label='Counterfactual ECG')
    axes[1].set_title(f'Counterfactual ECG - Classified as: {"AFib" if cf_pred == 0 else "Normal"}', fontsize=14)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Overlay
    axes[2].plot(t, ecg_orig, 'b-', linewidth=1.0, alpha=0.7, label='Original')
    axes[2].plot(t, ecg_cf, 'r-', linewidth=1.0, alpha=0.7, label='Counterfactual')
    
    # Highlight differences
    diff = np.abs(ecg_cf - ecg_orig)
    threshold = np.percentile(diff, 90)
    highlight_mask = diff > threshold
    for i in range(len(t)-1):
        if highlight_mask[i]:
            axes[2].axvspan(t[i], t[i+1], alpha=0.3, color='yellow')
    
    axes[2].set_title(f'Overlay (Correlation: {correlation:.4f})', fontsize=14)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Add info
    direction = f"{'AFib' if orig_label == 0 else 'Normal'} → {'AFib' if cf_pred == 0 else 'Normal'}"
    flipped = orig_label != cf_pred
    fig.suptitle(f'Sample {sample_idx}: {direction} | Flipped: {"✓" if flipped else "✗"} | Correlation: {correlation:.4f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'overlay_{sample_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./v2_counterfactual_overlays_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X = np.load('./ecg_afib_data/X_combined.npy')
    y = np.load('./ecg_afib_data/y_combined.npy')
    
    # Load split indices
    with open('./minimal_edit_v2/split_indices.json', 'r') as f:
        split_indices = json.load(f)
    
    test_indices = split_indices['test_idx']
    print(f"Test set size: {len(test_indices)}")
    
    # Load models
    classifier, edit_model = load_models()
    
    # Verify on test samples
    results = verify_on_test_samples(classifier, edit_model, X, y, test_indices, n_samples=200)
    
    # Generate overlay plots for random test samples
    print("\nGenerating overlay visualizations...")
    np.random.seed(123)
    plot_indices = np.random.choice(test_indices, 20, replace=False)
    
    for i, idx in enumerate(tqdm(plot_indices, desc="Generating overlays")):
        ecg = X[idx]
        label = 0 if y[idx] == 'A' else 1
        target = 1 - label
        
        ecg_tensor = torch.FloatTensor(ecg).unsqueeze(0).unsqueeze(0).to(DEVICE)
        target_tensor = torch.LongTensor([target]).to(DEVICE)
        
        with torch.no_grad():
            cf, edit, _ = edit_model(ecg_tensor, target_tensor)
            cf_logits, _ = classifier(cf)
            cf_pred = cf_logits.argmax(dim=1).item()
        
        ecg_np = ecg.flatten()
        cf_np = cf.squeeze().cpu().numpy().flatten()
        corr = np.corrcoef(ecg_np, cf_np)[0, 1]
        
        generate_overlay_plot(ecg_np, cf_np, label, cf_pred, corr, idx, save_dir)
    
    print(f"\nOverlays saved to: {save_dir}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'num_test_samples': len(test_indices),
        'num_verified': 200,
        'num_visualized': 20,
        'results': {
            'normal_to_afib': {
                'flip_rate': results['normal_to_afib']['flipped'] / max(1, results['normal_to_afib']['total']),
                'mean_correlation': float(np.mean(results['normal_to_afib']['correlations'])) if results['normal_to_afib']['correlations'] else 0
            },
            'afib_to_normal': {
                'flip_rate': results['afib_to_normal']['flipped'] / max(1, results['afib_to_normal']['total']),
                'mean_correlation': float(np.mean(results['afib_to_normal']['correlations'])) if results['afib_to_normal']['correlations'] else 0
            }
        }
    }
    
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
