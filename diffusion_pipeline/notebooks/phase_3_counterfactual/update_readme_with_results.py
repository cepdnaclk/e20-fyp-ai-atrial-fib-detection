#!/usr/bin/env python3
"""
Update README_RESULTS.md with three-way evaluation results.
Run this after 18_three_way_evaluation.py completes.

Usage:
    python update_readme_with_results.py
"""

import json
import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
THREE_WAY_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/three_way_evaluation'
RESULTS_FILE = THREE_WAY_DIR / 'three_way_results.json'
README_FILE = PROJECT_ROOT / 'notebooks/phase_3_counterfactual/README_RESULTS.md'

def main():
    if not RESULTS_FILE.exists():
        print(f"❌ Three-way results not found: {RESULTS_FILE}")
        print("   Run 18_three_way_evaluation.py first.")
        return
    
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    
    print("✓ Loaded three-way results")
    
    # Extract metrics
    conditions = ['original', 'counterfactual', 'mixed']
    labels = ['Original (A)', 'Counterfactual (B)', 'Mixed (C)']
    
    # Build results table
    metrics_table = "| Metric | Original (A) | Counterfactual (B) | Mixed (C) |\n"
    metrics_table += "|--------|-------------|-------------------|----------|\n"
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']:
        row = f"| **{metric.replace('_', ' ').title()}** |"
        for cond in conditions:
            val = results[cond]['test_metrics'][metric]
            row += f" {val:.4f} |"
        metrics_table += row + "\n"
    
    # Add training time and best epoch
    for metric_key, label in [('training_time_hours', 'Training Time (h)'), ('best_epoch', 'Best Epoch')]:
        row = f"| **{label}** |"
        for cond in conditions:
            val = results[cond][metric_key]
            if metric_key == 'training_time_hours':
                row += f" {val:.2f} |"
            else:
                row += f" {val} |"
        metrics_table += row + "\n"
    
    # Build statistical comparison table
    stat_table = "| Comparison | t-statistic | p-value | Cohen's d | Significant? |\n"
    stat_table += "|-----------|------------|---------|-----------|-------------|\n"
    
    stat_keys = [
        ('original_vs_counterfactual', 'Original vs CF'),
        ('original_vs_mixed', 'Original vs Mixed'),
        ('counterfactual_vs_mixed', 'CF vs Mixed'),
    ]
    
    for key, label in stat_keys:
        data = results['statistical_comparison'][key]
        sig = "✓" if data['significant'] else "✗"
        stat_table += f"| {label} | {data['t_statistic']:.3f} | {data['p_value']:.4e} | {data['cohens_d']:.3f} | {sig} |\n"
    
    # Read README
    readme_text = README_FILE.read_text()
    
    # Replace the pending results table
    old_results = """| Metric | Original (A) | Counterfactual (B) | Mixed (C) |
|--------|-------------|-------------------|-----------|
| Accuracy | *pending* | *pending* | *pending* |
| Precision | *pending* | *pending* | *pending* |
| Recall | *pending* | *pending* | *pending* |
| F1-Score | *pending* | *pending* | *pending* |
| AUROC | *pending* | *pending* | *pending* |
| Training Time | *pending* | *pending* | *pending* |
| Best Epoch | *pending* | *pending* | *pending* |"""
    
    readme_text = readme_text.replace(old_results, metrics_table.strip())
    
    # Replace pendign statistical comparison
    old_stats = """| Comparison | t-statistic | p-value | Cohen's d | Significant? |
|-----------|------------|---------|-----------|-------------|
| Original vs CF | — | — | — | — |
| Original vs Mixed | — | — | — | — |
| CF vs Mixed | — | — | — | — |"""
    
    readme_text = readme_text.replace(old_stats, stat_table.strip())
    
    # Replace the pending note
    readme_text = readme_text.replace(
        "> **Note**: Three-way evaluation results will be populated when training completes (~3-4 hours GPU).\n> Check: `models/phase3_counterfactual/three_way_evaluation/three_way_results.json`\n\n",
        ""
    )
    
    # Replace the pending statistical note
    readme_text = readme_text.replace(
        "*Pending three-way evaluation completion.*\n\n",
        ""
    )
    
    # Write updated README
    README_FILE.write_text(readme_text)
    
    # Print summary
    print("\n" + "="*60)
    print("Three-Way Evaluation Results Summary")
    print("="*60)
    for cond, label in zip(conditions, labels):
        m = results[cond]['test_metrics']
        t = results[cond]['training_time_hours']
        print(f"\n{label}:")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1-Score:  {m['f1_score']:.4f}")
        print(f"  AUROC:     {m['auroc']:.4f}")
        print(f"  Time:      {t:.2f}h")
    
    print("\n" + "="*60)
    print("Statistical Comparison")
    print("="*60)
    for key, label in stat_keys:
        d = results['statistical_comparison'][key]
        sig = "✓" if d['significant'] else "✗"
        print(f"  {label}: t={d['t_statistic']:.3f}, p={d['p_value']:.4e}, d={d['cohens_d']:.3f} {sig}")
    
    print(f"\n✓ README_RESULTS.md updated at: {README_FILE}")

if __name__ == '__main__':
    main()
