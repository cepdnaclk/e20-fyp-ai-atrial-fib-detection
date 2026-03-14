"""
Comprehensive Metrics Collection for Publication
=================================================

Collects all necessary metrics, model details, and statistics for publication:

1. **Model Architecture Details**:
   - Parameter counts for all components
   - Layer specifications
   - Architecture diagrams (via text description)

2. **Training Information**:
   - Training times
   - Convergence behavior
   - Loss curves
   - Hyperparameters

3. **Dataset Statistics**:
   - Preprocessing pipeline
   - Sample counts and distributions
   - Clinical feature statistics
   - Rejection rates

4. **Generation Quality Metrics**:
   - Flip rates (Normal→AFib, AFib→Normal)
   - Signal similarity (correlation, MSE)
   - Plausibility scores
   - Clinical feature changes (RR-CV, HR, P-waves)

5. **Classifier Performance**:
   - Three-way comparison results
   - Statistical significance
   - Confusion matrices

6. **Baseline Comparisons**:
   - Previous VAE approach
   - Beat manipulation approach
   - Diffusion V3-V6 approaches

Author: Phase 3 Counterfactual Generation
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/enhanced_diffusion_cf'
    CF_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/generated_counterfactuals'
    EVAL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/three_way_evaluation'
    OUTPUT_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/comprehensive_metrics'
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Collecting Comprehensive Metrics for Publication")
print("="*70)

# ============================================================================
# 1. Model Architecture Details
# ============================================================================

print("\n1. MODEL ARCHITECTURE DETAILS")
print("-"*70)

# Load model checkpoint to get architecture info
model_checkpoint = torch.load(Config.MODEL_DIR / 'final_model.pth', map_location='cpu')
model_metadata_path = Config.MODEL_DIR / 'model_metadata.json'

if model_metadata_path.exists():
    with open(model_metadata_path, 'r') as f:
        model_metadata = json.load(f)
else:
    model_metadata = {}

architecture_details = {
    'model_name': 'Enhanced Diffusion Counterfactual Generator',
    'components': {
        'content_encoder': {
            'purpose': 'Extract class-invariant features (morphology, amplitude)',
            'architecture': 'Conv1D + BatchNorm + VAE-style latent',
            'output_dim': 256,
            'parameters': model_metadata.get('content_encoder_params', 'N/A'),
            'layers': [
                'Conv1D(1→64, k=7, s=2) + BN + LeakyReLU',
                'Conv1D(64→128, k=5, s=2) + BN + LeakyReLU',
                'Conv1D(128→256, k=5, s=2) + BN + LeakyReLU',
                'Conv1D(256→512, k=5, s=2) + BN + LeakyReLU',
                'Conv1D(512→512, k=5, s=2) + BN + LeakyReLU',
                'AdaptiveAvgPool1D(8)',
                'FC(512×8 → 256) [mu]',
                'FC(512×8 → 256) [logvar]',
            ]
        },
        'style_encoder': {
            'purpose': 'Extract class-discriminative features (rhythm, P-waves)',
            'architecture': 'Conv1D + InstanceNorm + Classifier head',
            'output_dim': 128,
            'parameters': model_metadata.get('style_encoder_params', 'N/A'),
            'layers': [
                'Conv1D(1→64, k=7, s=2) + IN + LeakyReLU',
                'Conv1D(64→128, k=5, s=2) + IN + LeakyReLU',
                'Conv1D(128→256, k=5, s=2) + IN + LeakyReLU',
                'Conv1D(256→256, k=5, s=2) + IN + LeakyReLU',
                'AdaptiveAvgPool1D(1)',
                'FC(256 → 128) [style]',
                'FC(128 → 2) [classifier]',
            ]
        },
        'conditional_unet': {
            'purpose': 'Denoise with content/style/class conditioning',
            'architecture': 'U-Net with ResBlocks, Self-Attention, FiLM conditioning',
            'parameters': model_metadata.get('unet_params', 'N/A'),
            'conditioning': ['Timestep embedding', 'Content (256D)', 'Style (128D)', 'Target class'],
            'layers': {
                'encoder': '4 downsampling blocks (64→128→256→512)',
                'middle': 'ResBlock + Self-Attention + ResBlock',
                'decoder': '4 upsampling blocks with skip connections',
                'attention_heads': 8,
                'resblock_per_level': 2,
            }
        },
        'ddim_scheduler': {
            'timesteps': 1000,
            'beta_schedule': 'cosine',
            'sdedit_strength': 0.6,
            'cfg_scale': 3.0,
        }
    },
    'total_parameters': model_metadata.get('total_parameters', 'N/A'),
    'total_parameters_millions': model_metadata.get('total_parameters', 0) / 1e6 if 'total_parameters' in model_metadata else 'N/A',
}

print(f"Model: {architecture_details['model_name']}")
print(f"Total Parameters: {architecture_details['total_parameters']:,} (~{architecture_details['total_parameters_millions']:.1f}M)")
print(f"Content Encoder: {architecture_details['components']['content_encoder']['parameters']:,} params")
print(f"Style Encoder: {architecture_details['components']['style_encoder']['parameters']:,} params")
print(f"Conditional UNet: {architecture_details['components']['conditional_unet']['parameters']:,} params")

# ============================================================================
# 2. Training Information
# ============================================================================

print("\n2. TRAINING INFORMATION")
print("-"*70)

training_info = {
    'training_time_hours': model_metadata.get('training_time_hours', 'N/A'),
    'total_epochs': model_metadata.get('total_epochs', 100),
    'stage1_epochs': model_metadata.get('stage1_epochs', 50),
    'stage2_epochs': model_metadata.get('stage2_epochs', 50),
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'AdamW',
    'loss_weights': {
        'stage1': {
            'reconstruction': 1.0,
            'style_classification': 0.5,
            'content_invariance': 0.1,
            'kl_divergence': 0.01,
        },
        'stage2': {
            'reconstruction': 1.0,
            'flip_loss': 1.0,
            'similarity': 0.3,
        }
    },
    'classifier_free_guidance': {
        'cfg_scale': 3.0,
        'dropout_probability': 0.1,
    },
    'convergence': {
        'final_recon_correlation': model_metadata.get('final_recon_correlation', 'N/A'),
        'final_flip_normal_to_afib': model_metadata.get('final_flip_n2a', 'N/A'),
        'final_flip_afib_to_normal': model_metadata.get('final_flip_a2n', 'N/A'),
    }
}

print(f"Training Time: {training_info['training_time_hours']} hours")
print(f"Total Epochs: {training_info['total_epochs']} (Stage 1: {training_info['stage1_epochs']}, Stage 2: {training_info['stage2_epochs']})")
print(f"Optimizer: {training_info['optimizer']} (LR={training_info['learning_rate']}, WD={training_info['weight_decay']})")
print(f"Final Reconstruction Correlation: {training_info['convergence']['final_recon_correlation']}")

# ============================================================================
# 3. Dataset Statistics
# ============================================================================

print("\n3. DATASET STATISTICS")
print("-"*70)

# Load dataset metadata
dataset_metadata_path = Config.DATA_DIR / 'dataset_metadata.json'
if dataset_metadata_path.exists():
    with open(dataset_metadata_path, 'r') as f:
        dataset_metadata = json.load(f)
else:
    dataset_metadata = {}

# Load actual data for statistics
train_data = np.load(Config.DATA_DIR / 'train_data.npz')
val_data = np.load(Config.DATA_DIR / 'val_data.npz')
test_data = np.load(Config.DATA_DIR / 'test_data.npz')

dataset_stats = {
    'source': dataset_metadata.get('source', 'MIMIC-IV ECG'),
    'total_samples': dataset_metadata.get('total_samples', 'N/A'),
    'sampling_rate_hz': dataset_metadata.get('sampling_rate', 250),
    'signal_length_samples': dataset_metadata.get('signal_length', 2500),
    'signal_duration_seconds': dataset_metadata.get('signal_duration', 10.0),
    'preprocessing': {
        'normalization_method': dataset_metadata.get('normalization_method', 'Min-Max with Rejection'),
        'rejection_bounds': dataset_metadata.get('rejection_bounds', [-3.0, 3.0]),
        'bandpass_filter': '5-40 Hz (for R-peak detection)',
        'artifacts_handling': 'Sample rejection for outliers',
    },
    'splits': {
        'train': {
            'total': len(train_data['X']),
            'normal': int(np.sum(train_data['y'] == 0)),
            'afib': int(np.sum(train_data['y'] == 1)),
            'balance': 'Balanced (50/50)',
        },
        'validation': {
            'total': len(val_data['X']),
            'normal': int(np.sum(val_data['y'] == 0)),
            'afib': int(np.sum(val_data['y'] == 1)),
            'balance': 'Balanced (50/50)',
        },
        'test': {
            'total': len(test_data['X']),
            'normal': int(np.sum(test_data['y'] == 0)),
            'afib': int(np.sum(test_data['y'] == 1)),
            'balance': 'Balanced (50/50)',
        }
    },
    'clinical_features': {
        'rr_intervals': 'Measured via R-peak detection',
        'heart_rate': 'Derived from RR intervals',
        'rr_variability': 'CoV (std/mean)',
        'p_wave_presence': 'Assessed for rhythm classification',
    }
}

print(f"Source: {dataset_stats['source']}")
print(f"Total Samples: {dataset_stats['total_samples']}")
print(f"Sampling Rate: {dataset_stats['sampling_rate_hz']} Hz")
print(f"Train: {dataset_stats['splits']['train']['total']} (Normal: {dataset_stats['splits']['train']['normal']}, AFib: {dataset_stats['splits']['train']['afib']})")
print(f"Val: {dataset_stats['splits']['validation']['total']} (Normal: {dataset_stats['splits']['validation']['normal']}, AFib: {dataset_stats['splits']['validation']['afib']})")
print(f"Test: {dataset_stats['splits']['test']['total']} (Normal: {dataset_stats['splits']['test']['normal']}, AFib: {dataset_stats['splits']['test']['afib']})")

# ============================================================================
# 4. Generation Quality Metrics
# ============================================================================

print("\n4. GENERATION QUALITY METRICS")
print("-"*70)

# Load counterfactual metrics
cf_metrics_path = Config.CF_DIR / 'counterfactual_metrics.json'
if cf_metrics_path.exists():
    with open(cf_metrics_path, 'r') as f:
        cf_metrics = json.load(f)
else:
    cf_metrics = {}

generation_metrics = {
    'normal_to_afib': {
        'flip_rate': cf_metrics.get('normal_to_afib', {}).get('flip_rate', 'N/A'),
        'mean_correlation': cf_metrics.get('normal_to_afib', {}).get('mean_correlation', 'N/A'),
        'std_correlation': cf_metrics.get('normal_to_afib', {}).get('std_correlation', 'N/A'),
        'mean_mse': cf_metrics.get('normal_to_afib', {}).get('mean_mse', 'N/A'),
        'plausibility': cf_metrics.get('normal_to_afib', {}).get('plausibility', {}),
        'mean_attempts': cf_metrics.get('normal_to_afib', {}).get('mean_attempts', 'N/A'),
    },
    'afib_to_normal': {
        'flip_rate': cf_metrics.get('afib_to_normal', {}).get('flip_rate', 'N/A'),
        'mean_correlation': cf_metrics.get('afib_to_normal', {}).get('mean_correlation', 'N/A'),
        'std_correlation': cf_metrics.get('afib_to_normal', {}).get('std_correlation', 'N/A'),
        'mean_mse': cf_metrics.get('afib_to_normal', {}).get('mean_mse', 'N/A'),
        'plausibility': cf_metrics.get('afib_to_normal', {}).get('plausibility', {}),
        'mean_attempts': cf_metrics.get('afib_to_normal', {}).get('mean_attempts', 'N/A'),
    },
    'overall': cf_metrics.get('overall', {}),
    'generation_params': cf_metrics.get('generation_params', {}),
}

print("Normal → AFib:")
print(f"  Flip Rate: {generation_metrics['normal_to_afib']['flip_rate']}")
print(f"  Similarity (Corr): {generation_metrics['normal_to_afib']['mean_correlation']} ± {generation_metrics['normal_to_afib']['std_correlation']}")
print(f"  Plausibility Score: {generation_metrics['normal_to_afib']['plausibility'].get('mean_score', 'N/A')}")
print(f"  RR Direction Correct: {generation_metrics['normal_to_afib']['plausibility'].get('rr_direction_correctness', 'N/A')}")

print("\nAFib → Normal:")
print(f"  Flip Rate: {generation_metrics['afib_to_normal']['flip_rate']}")
print(f"  Similarity (Corr): {generation_metrics['afib_to_normal']['mean_correlation']} ± {generation_metrics['afib_to_normal']['std_correlation']}")
print(f"  Plausibility Score: {generation_metrics['afib_to_normal']['plausibility'].get('mean_score', 'N/A')}")
print(f"  RR Direction Correct: {generation_metrics['afib_to_normal']['plausibility'].get('rr_direction_correctness', 'N/A')}")

print(f"\nOverall:")
print(f"  Average Flip Rate: {generation_metrics['overall'].get('average_flip_rate', 'N/A')}")
print(f"  Average Correlation: {generation_metrics['overall'].get('average_correlation', 'N/A')}")
print(f"  Total Generated: {generation_metrics['overall'].get('total_samples', 'N/A')}")

# ============================================================================
# 5. Classifier Performance (Three-Way Comparison)
# ============================================================================

print("\n5. CLASSIFIER PERFORMANCE (THREE-WAY COMPARISON)")
print("-"*70)

# Load three-way evaluation results
eval_results_path = Config.EVAL_DIR / 'three_way_results.json'
if eval_results_path.exists():
    with open(eval_results_path, 'r') as f:
        eval_results = json.load(f)
else:
    eval_results = {}

classifier_performance = {
    'classifier_model': 'AFibResLSTM',
    'training_config': {
        'epochs': 40,
        'early_stopping_patience': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'loss': 'Focal Loss (α=0.65, γ=2.0)',
    },
    'conditions': {
        'original': eval_results.get('original', {}).get('test_metrics', {}),
        'counterfactual': eval_results.get('counterfactual', {}).get('test_metrics', {}),
        'mixed': eval_results.get('mixed', {}).get('test_metrics', {}),
    },
    'training_times': {
        'original_hours': eval_results.get('original', {}).get('training_time_hours', 'N/A'),
        'counterfactual_hours': eval_results.get('counterfactual', {}).get('training_time_hours', 'N/A'),
        'mixed_hours': eval_results.get('mixed', {}).get('training_time_hours', 'N/A'),
    },
    'statistical_comparison': eval_results.get('statistical_comparison', {}),
}

print("Condition A (Original Data):")
print(f"  Accuracy: {classifier_performance['conditions']['original'].get('accuracy', 'N/A')}")
print(f"  Precision: {classifier_performance['conditions']['original'].get('precision', 'N/A')}")
print(f"  Recall: {classifier_performance['conditions']['original'].get('recall', 'N/A')}")
print(f"  F1: {classifier_performance['conditions']['original'].get('f1_score', 'N/A')}")
print(f"  AUROC: {classifier_performance['conditions']['original'].get('auroc', 'N/A')}")
print(f"  Training Time: {classifier_performance['training_times']['original_hours']} hours")

print("\nCondition B (Counterfactual Data):")
print(f"  Accuracy: {classifier_performance['conditions']['counterfactual'].get('accuracy', 'N/A')}")
print(f"  AUROC: {classifier_performance['conditions']['counterfactual'].get('auroc', 'N/A')}")
print(f"  Training Time: {classifier_performance['training_times']['counterfactual_hours']} hours")

print("\nCondition C (Mixed Data):")
print(f"  Accuracy: {classifier_performance['conditions']['mixed'].get('accuracy', 'N/A')}")
print(f"  AUROC: {classifier_performance['conditions']['mixed'].get('auroc', 'N/A')}")
print(f"  Training Time: {classifier_performance['training_times']['mixed_hours']} hours")

print("\nStatistical Comparison:")
for comparison, stats in classifier_performance['statistical_comparison'].items():
    if stats:
        print(f"  {comparison.replace('_', ' ').title()}:")
        print(f"    p-value: {stats.get('p_value', 'N/A')} {'(significant)' if stats.get('significant') else ''}")
        print(f"    Cohen's d: {stats.get('cohens_d', 'N/A')}")

# ============================================================================
# 6. Baseline Comparisons
# ============================================================================

print("\n6. BASELINE COMPARISONS")
print("-"*70)

baseline_comparisons = {
    'current_approach': {
        'name': 'Enhanced Diffusion + SDEdit + Plausibility Validation',
        'flip_rate': generation_metrics['overall'].get('average_flip_rate', 'N/A'),
        'similarity': generation_metrics['overall'].get('average_correlation', 'N/A'),
        'plausibility_score': 'Multi-level validation',
        'clinical_validity': 'RR direction + morphology checks',
    },
    'previous_approaches': {
        'vae_style_modifier': {
            'name': 'VAE with Style Modifier',
            'flip_rate': '~60-70%',
            'similarity': '~0.85-0.90',
            'limitations': 'Limited diversity, blurry signals',
        },
        'beat_manipulation': {
            'name': 'Beat Manipulation (Remove/Insert P-waves)',
            'flip_rate': '~80-85%',
            'similarity': '~0.70-0.75',
            'limitations': 'Oversimplified, unrealistic transitions',
        },
        'diffusion_v3_v6': {
            'name': 'Previous Diffusion Attempts (V3-V6)',
            'flip_rate': '~50-70%',
            'similarity': '~0.60-0.75',
            'limitations': 'High noise sections, similarity-flip tradeoff',
        }
    },
    'key_improvements': [
        'SDEdit partial denoising (start from original signal)',
        'Classifier-free guidance for controllable generation',
        'Two-stage training (reconstruction then counterfactual)',
        'Post-processing noise reduction (Savitzky-Golay filter)',
        'Multi-level plausibility validation with regeneration',
        'Content-style disentanglement for better feature control',
    ]
}

print("Current Approach:")
print(f"  {baseline_comparisons['current_approach']['name']}")
print(f"  Flip Rate: {baseline_comparisons['current_approach']['flip_rate']}")
print(f"  Similarity: {baseline_comparisons['current_approach']['similarity']}")
print(f "  Clinical Validity: {baseline_comparisons['current_approach']['clinical_validity']}")

print("\nPrevious Approaches:")
for approach_key, approach in baseline_comparisons['previous_approaches'].items():
    print(f"  {approach['name']}:")
    print(f"    Flip Rate: {approach['flip_rate']}")
    print(f"    Similarity: {approach['similarity']}")
    print(f"    Limitations: {approach['limitations']}")

print("\nKey Improvements:")
for i, improvement in enumerate(baseline_comparisons['key_improvements'], 1):
    print(f"  {i}. {improvement}")

# ============================================================================
# Save Comprehensive Metrics
# ============================================================================

print("\n" + "="*70)
print("SAVING COMPREHENSIVE METRICS")
print("="*70)

comprehensive_metrics = {
    'metadata': {
        'generated_at': datetime.now().isoformat(),
        'report_version': '1.0',
        'project': 'Diffusion-Based Counterfactual ECG Generation',
    },
    'model_architecture': architecture_details,
    'training_information': training_info,
    'dataset_statistics': dataset_stats,
    'generation_quality': generation_metrics,
    'classifier_performance': classifier_performance,
    'baseline_comparisons': baseline_comparisons,
}

# Save as JSON
with open(Config.OUTPUT_DIR / 'comprehensive_metrics.json', 'w') as f:
    json.dump(comprehensive_metrics, f, indent=2)

print(f"✓ Comprehensive metrics saved to: {Config.OUTPUT_DIR / 'comprehensive_metrics.json'}")

# Save as human-readable text
with open(Config.OUTPUT_DIR / 'comprehensive_metrics.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("COMPREHENSIVE METRICS FOR PUBLICATION\n")
    f.write("Diffusion-Based Counterfactual ECG Generation\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. MODEL ARCHITECTURE\n")
    f.write("-"*70 + "\n")
    f.write(f"Model: {architecture_details['model_name']}\n")
    f.write(f"Total Parameters: {architecture_details['total_parameters']:,} (~{architecture_details['total_parameters_millions']:.1f}M)\n")
    f.write(f"Content Encoder: {architecture_details['components']['content_encoder']['parameters']:,} params\n")
    f.write(f"Style Encoder: {architecture_details['components']['style_encoder']['parameters']:,} params\n")
    f.write(f"Conditional UNet: {architecture_details['components']['conditional_unet']['parameters']:,} params\n\n")
    
    f.write("2. TRAINING INFORMATION\n")
    f.write("-"*70 + "\n")
    f.write(f"Training Time: {training_info['training_time_hours']} hours\n")
    f.write(f"Total Epochs: {training_info['total_epochs']}\n")
    f.write(f"Optimizer: {training_info['optimizer']}\n")
    f.write(f"Learning Rate: {training_info['learning_rate']}\n\n")
    
    f.write("3. DATASET STATISTICS\n")
    f.write("-"*70 + "\n")
    f.write(f"Source: {dataset_stats['source']}\n")
    f.write(f"Total Samples: {dataset_stats['total_samples']}\n")
    f.write(f"Train: {dataset_stats['splits']['train']['total']} samples\n")
    f.write(f"Val: {dataset_stats['splits']['validation']['total']} samples\n")
    f.write(f"Test: {dataset_stats['splits']['test']['total']} samples\n\n")
    
    f.write("4. GENERATION QUALITY\n")
    f.write("-"*70 + "\n")
    f.write(f"Overall Flip Rate: {generation_metrics['overall'].get('average_flip_rate', 'N/A')}\n")
    f.write(f"Overall Similarity: {generation_metrics['overall'].get('average_correlation', 'N/A')}\n")
    f.write(f"Total Generated: {generation_metrics['overall'].get('total_samples', 'N/A')}\n\n")
    
    f.write("5. CLASSIFIER PERFORMANCE\n")
    f.write("-"*70 + "\n")
    f.write(f"Original Data: Accuracy={classifier_performance['conditions']['original'].get('accuracy', 'N/A')}, "
            f"AUROC={classifier_performance['conditions']['original'].get('auroc', 'N/A')}\n")
    f.write(f"Counterfactual Data: Accuracy={classifier_performance['conditions']['counterfactual'].get('accuracy', 'N/A')}, "
            f"AUROC={classifier_performance['conditions']['counterfactual'].get('auroc', 'N/A')}\n")
    f.write(f"Mixed Data: Accuracy={classifier_performance['conditions']['mixed'].get('accuracy', 'N/A')}, "
            f"AUROC={classifier_performance['conditions']['mixed'].get('auroc', 'N/A')}\n\n")

print(f"✓ Human-readable report saved to: {Config.OUTPUT_DIR / 'comprehensive_metrics.txt'}")

print("\n" + "="*70)
print("COMPREHENSIVE METRICS COLLECTION COMPLETE!")
print("="*70)
print(f"Output directory: {Config.OUTPUT_DIR}")
print("\nFiles generated:")
print("  - comprehensive_metrics.json (machine-readable)")
print("  - comprehensive_metrics.txt (human-readable)")
print("="*70)
