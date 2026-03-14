"""
Enhanced Statistical Analysis for Three-Regime Classifier Evaluation
=====================================================================

Addresses the N=5 limitation of fold-level tests by using per-sample
predictions (N=22,469) from all 15 trained models.

Tests performed:
1. Per-sample McNemar's test (A vs B, A vs C, B vs C)
2. Wilcoxon signed-rank on predicted probabilities
3. Per-sample TOST equivalence
4. Dunnett's test on fold-level metrics
5. Fold-level paired t-test, TOST, non-inferiority (supplementary)

Usage:
    python enhanced_statistical_analysis.py
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/three_way_evaluation'
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
OUTPUT_DIR = MODEL_DIR  # save results alongside existing outputs

sys.path.insert(0, str(PROJECT_ROOT / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks/phase_3_counterfactual'))
from model_architecture import AFibResLSTM, ModelConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS = 5
CONDITIONS = ['A', 'B', 'C']
COND_LABELS = {'A': 'Dataset A (Original)', 'B': 'Dataset B (CF-only)', 'C': 'Dataset C (Augmented)'}
SEED = 42
DELTA = 0.02  # Equivalence margin (2 percentage points)

print("=" * 70)
print("  ENHANCED STATISTICAL ANALYSIS")
print(f"  Device: {DEVICE}")
print("=" * 70)

# ============================================================================
# Step 1: Load test data
# ============================================================================

print("\n[Step 1] Loading test data...")
test_data = np.load(DATA_DIR / 'test_data.npz')
test_X = torch.tensor(test_data['X'], dtype=torch.float32)
test_y = torch.tensor(test_data['y'], dtype=torch.long)
if test_X.dim() == 2:
    test_X = test_X.unsqueeze(1)

N_TEST = len(test_X)
print(f"  Test set: {N_TEST} samples (Normal: {(test_y==0).sum()}, AFib: {(test_y==1).sum()})")

# ============================================================================
# Step 2: Load all 15 models and run inference
# ============================================================================

print("\n[Step 2] Loading models and running inference...")

all_preds = {}   # {cond: [fold0_preds, fold1_preds, ...]}
all_probs = {}   # {cond: [fold0_probs, fold1_probs, ...]}

for cond in CONDITIONS:
    all_preds[cond] = []
    all_probs[cond] = []
    
    for fold in range(N_FOLDS):
        model_path = MODEL_DIR / f'model_{cond}_fold{fold}.pth'
        if not model_path.exists():
            print(f"  WARNING: {model_path} not found, skipping")
            continue
        
        # Load model
        model_config = ModelConfig()
        model = AFibResLSTM(model_config).to(DEVICE)
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        # Run inference
        preds_list, probs_list = [], []
        with torch.no_grad():
            for i in range(0, N_TEST, 64):
                batch = test_X[i:i+64].to(DEVICE)
                mean = batch.mean(dim=2, keepdim=True)
                std = batch.std(dim=2, keepdim=True) + 1e-8
                logits, _ = model((batch - mean) / std)
                probs = torch.softmax(logits, dim=1)
                preds_list.append(torch.argmax(logits, dim=1).cpu().numpy())
                probs_list.append(probs[:, 1].cpu().numpy())  # P(AFib)
        
        fold_preds = np.concatenate(preds_list)
        fold_probs = np.concatenate(probs_list)
        all_preds[cond].append(fold_preds)
        all_probs[cond].append(fold_probs)
        
        acc = (fold_preds == test_y.numpy()).mean()
        print(f"  {cond}_fold{fold}: acc={acc:.4f}")
        
        del model
        torch.cuda.empty_cache()

# ============================================================================
# Step 3: Compute ensemble predictions (mean probability across 5 folds)
# ============================================================================

print("\n[Step 3] Computing ensemble predictions (mean probability across folds)...")

ensemble_probs = {}   # {cond: mean_probs}
ensemble_preds = {}   # {cond: preds from mean_probs}
labels = test_y.numpy()

for cond in CONDITIONS:
    prob_stack = np.array(all_probs[cond])  # (5, N_TEST)
    ensemble_probs[cond] = prob_stack.mean(axis=0)  # (N_TEST,)
    ensemble_preds[cond] = (ensemble_probs[cond] >= 0.5).astype(int)
    
    acc = (ensemble_preds[cond] == labels).mean()
    print(f"  {COND_LABELS[cond]}: ensemble acc = {acc:.4f}")

# ============================================================================
# Step 4: Per-sample McNemar's test (N=22,469)
# ============================================================================

print("\n" + "=" * 70)
print("  4. McNEMAR'S TEST (per-sample, N={})".format(N_TEST))
print("=" * 70)

def mcnemars_test(preds_1, preds_2, true_labels):
    """McNemar's test with continuity correction."""
    correct_1 = (preds_1 == true_labels)
    correct_2 = (preds_2 == true_labels)
    b = int(np.sum(correct_1 & ~correct_2))  # only model 1 correct
    c = int(np.sum(~correct_1 & correct_2))  # only model 2 correct
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p_value), b, c

mcnemar_results = {}
comparisons = [('A_vs_B', 'A', 'B'), ('A_vs_C', 'A', 'C'), ('B_vs_C', 'B', 'C')]

for comp_name, c1, c2 in comparisons:
    chi2, p_val, b, c = mcnemars_test(ensemble_preds[c1], ensemble_preds[c2], labels)
    mcnemar_results[comp_name] = {
        'chi2': chi2, 'p_value': p_val,
        'only_first_correct': b, 'only_second_correct': c,
        'N': N_TEST
    }
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"\n  {comp_name}: chi2={chi2:.3f}, p={p_val:.6f} {sig}")
    print(f"    Only {COND_LABELS[c1].split('(')[0].strip()} correct: {b}")
    print(f"    Only {COND_LABELS[c2].split('(')[0].strip()} correct: {c}")

# ============================================================================
# Step 5: Wilcoxon signed-rank test on predicted probabilities (N=22,469)
# ============================================================================

print("\n" + "=" * 70)
print("  5. WILCOXON SIGNED-RANK TEST (per-sample probabilities, N={})".format(N_TEST))
print("=" * 70)

wilcoxon_results = {}
for comp_name, c1, c2 in comparisons:
    probs_1 = ensemble_probs[c1]
    probs_2 = ensemble_probs[c2]
    diff = probs_1 - probs_2
    
    # Wilcoxon signed-rank (two-sided)
    stat, p_val = stats.wilcoxon(diff, alternative='two-sided')
    
    wilcoxon_results[comp_name] = {
        'statistic': float(stat),
        'p_value': float(p_val),
        'mean_diff': float(diff.mean()),
        'median_diff': float(np.median(diff)),
        'N': N_TEST
    }
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    print(f"\n  {comp_name}: W={stat:.1f}, p={p_val:.6f} {sig}")
    print(f"    Mean prob diff: {diff.mean():+.4f}")
    print(f"    Median prob diff: {np.median(diff):+.4f}")

# ============================================================================
# Step 6: Per-sample TOST equivalence test (N=22,469)
# ============================================================================

print("\n" + "=" * 70)
print(f"  6. PER-SAMPLE TOST EQUIVALENCE (margin = +/-{DELTA})")
print("=" * 70)

def per_sample_tost(correct_1, correct_2, delta=0.02):
    """
    TOST on per-sample accuracy difference.
    correct_1, correct_2: boolean arrays (N,) indicating correct classification.
    Tests if |p1 - p2| < delta where p1, p2 are accuracy rates.
    """
    n = len(correct_1)
    p1 = correct_1.mean()
    p2 = correct_2.mean()
    diff = p1 - p2
    
    # Standard error for difference of proportions (paired, McNemar-style)
    b = int(np.sum(correct_1 & ~correct_2))
    c = int(np.sum(~correct_1 & correct_2))
    se = np.sqrt((b + c) / n**2) if (b + c) > 0 else 1e-10
    
    # TOST: two one-sided z-tests
    z_upper = (diff - delta) / se   # H0: diff >= delta
    z_lower = (diff + delta) / se   # H0: diff <= -delta (i.e., -diff >= delta)
    
    p_upper = stats.norm.cdf(z_upper)  # want this small (diff < delta)
    p_lower = 1 - stats.norm.cdf(z_lower)  # want this small (diff > -delta)
    
    p_tost = max(p_upper, p_lower)
    equivalent = p_tost < 0.05
    
    return {
        'diff': float(diff),
        'se': float(se),
        'z_upper': float(z_upper),
        'z_lower': float(z_lower),
        'p_upper': float(p_upper),
        'p_lower': float(p_lower),
        'p_tost': float(p_tost),
        'equivalent': equivalent,
        'N': n
    }

tost_results = {}
for comp_name, c1, c2 in comparisons:
    correct_1 = (ensemble_preds[c1] == labels)
    correct_2 = (ensemble_preds[c2] == labels)
    result = per_sample_tost(correct_1, correct_2, delta=DELTA)
    tost_results[comp_name] = result
    
    equiv_str = "EQUIVALENT" if result['equivalent'] else "NOT EQUIVALENT"
    print(f"\n  {comp_name}: {equiv_str} (p_TOST={result['p_tost']:.6f})")
    print(f"    Accuracy diff: {result['diff']:+.4f}")
    print(f"    SE: {result['se']:.6f}")
    print(f"    p_upper (diff < +{DELTA}): {result['p_upper']:.6f}")
    print(f"    p_lower (diff > -{DELTA}): {result['p_lower']:.6f}")

# ============================================================================
# Step 7: Non-inferiority test (per-sample, N=22,469)
# ============================================================================

print("\n" + "=" * 70)
print(f"  7. PER-SAMPLE NON-INFERIORITY (margin = {DELTA})")
print("=" * 70)

noninf_results = {}
for comp_name, c1, c2 in [('A_vs_C', 'A', 'C'), ('A_vs_B', 'A', 'B')]:
    correct_1 = (ensemble_preds[c1] == labels)
    correct_2 = (ensemble_preds[c2] == labels)
    
    n = len(correct_1)
    p1 = correct_1.mean()
    p2 = correct_2.mean()
    diff = p1 - p2  # positive means c1 better
    
    b = int(np.sum(correct_1 & ~correct_2))
    c = int(np.sum(~correct_1 & correct_2))
    se = np.sqrt((b + c) / n**2) if (b + c) > 0 else 1e-10
    
    # H0: p1 - p2 >= delta (c2 is inferior by >= delta)
    # H1: p1 - p2 < delta (c2 is non-inferior)
    z = (diff - DELTA) / se
    p_val = stats.norm.cdf(z)
    non_inferior = p_val < 0.05
    
    noninf_results[comp_name] = {
        'diff': float(diff),
        'se': float(se),
        'z': float(z),
        'p_value': float(p_val),
        'non_inferior': non_inferior,
        'N': n
    }
    
    ni_str = "NON-INFERIOR" if non_inferior else "INCONCLUSIVE"
    print(f"\n  {comp_name}: {ni_str} (p={p_val:.6f})")
    print(f"    Accuracy diff: {diff:+.4f} (margin={DELTA})")

# ============================================================================
# Step 8: Dunnett's test on fold-level metrics
# ============================================================================

print("\n" + "=" * 70)
print("  8. DUNNETT'S TEST (fold-level, N=5)")
print("=" * 70)

# Load fold-level results
with open(MODEL_DIR / 'three_way_results_5fold.json') as f:
    fold_data = json.load(f)

agg = fold_data['aggregated_results']

def dunnetts_test(control_vals, treatment_vals_list, treatment_names):
    """
    Dunnett's test: compare multiple treatments against a single control.
    Uses paired design (same folds).
    
    Approximation: use Bonferroni-corrected paired t-tests as a conservative
    implementation of Dunnett's procedure.
    """
    n_treatments = len(treatment_vals_list)
    results = {}
    
    for i, (treat_vals, name) in enumerate(zip(treatment_vals_list, treatment_names)):
        control = np.array(control_vals)
        treat = np.array(treat_vals)
        diff = control - treat
        
        t_stat, p_raw = stats.ttest_rel(control, treat)
        
        # Bonferroni correction (conservative approximation of Dunnett's)
        p_corrected = min(p_raw * n_treatments, 1.0)
        
        results[name] = {
            't_statistic': float(t_stat),
            'p_raw': float(p_raw),
            'p_corrected': float(p_corrected),
            'mean_diff': float(diff.mean()),
            'std_diff': float(diff.std(ddof=1)),
            'significant': p_corrected < 0.05,
            'N': len(control)
        }
    
    return results

dunnett_results = {}
for metric in ['accuracy', 'f1_score', 'auroc']:
    control_vals = agg['A_original'][metric]['values']
    treat_B = agg['B_counterfactual'][metric]['values']
    treat_C = agg['C_augmented'][metric]['values']
    
    result = dunnetts_test(
        control_vals,
        [treat_B, treat_C],
        ['A_vs_B', 'A_vs_C']
    )
    dunnett_results[metric] = result
    
    print(f"\n  {metric}:")
    for comp in ['A_vs_B', 'A_vs_C']:
        r = result[comp]
        sig = "*" if r['significant'] else "n.s."
        print(f"    {comp}: diff={r['mean_diff']:+.4f}, t={r['t_statistic']:.3f}, "
              f"p_raw={r['p_raw']:.4f}, p_corrected={r['p_corrected']:.4f} {sig}")

# ============================================================================
# Step 9: Fold-level supplementary tests (for completeness)
# ============================================================================

print("\n" + "=" * 70)
print("  9. FOLD-LEVEL SUPPLEMENTARY TESTS (N=5)")
print("=" * 70)

fold_level_results = {}

for metric in ['accuracy', 'f1_score', 'auroc']:
    vals_a = np.array(agg['A_original'][metric]['values'])
    vals_c = np.array(agg['C_augmented'][metric]['values'])
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(vals_a, vals_c)
    diff = vals_a - vals_c
    
    # TOST
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t_upper = (diff.mean() - DELTA) / se
    t_lower = (diff.mean() + DELTA) / se
    p_upper = stats.t.cdf(t_upper, df=len(diff)-1)
    p_lower = 1 - stats.t.cdf(t_lower, df=len(diff)-1)
    p_tost = max(p_upper, p_lower)
    
    # Non-inferiority
    t_ni = (diff.mean() - DELTA) / se
    p_ni = stats.t.cdf(t_ni, df=len(diff)-1)
    
    # Cohen's d
    d = diff.mean() / (diff.std(ddof=1) + 1e-10)
    
    fold_level_results[metric] = {
        'paired_t': {'t': float(t_stat), 'p': float(p_val), 'diff': float(diff.mean())},
        'tost': {'p_tost': float(p_tost), 'equivalent': p_tost < 0.05},
        'non_inferiority': {'p': float(p_ni), 'non_inferior': p_ni < 0.05},
        'cohens_d': float(d)
    }
    
    print(f"\n  {metric} (A vs C):")
    print(f"    Paired t-test: diff={diff.mean():+.4f}, p={p_val:.4f}")
    print(f"    TOST (margin={DELTA}): p={p_tost:.4f} {'EQUIV' if p_tost < 0.05 else 'NOT EQUIV'}")
    print(f"    Non-inferiority: p={p_ni:.4f} {'NI' if p_ni < 0.05 else 'INCONCLUSIVE'}")
    print(f"    Cohen's d: {d:.4f}")

# ============================================================================
# Step 10: Summary table
# ============================================================================

print("\n" + "=" * 70)
print("  SUMMARY: A vs C (Primary Comparison)")
print("=" * 70)

ac_mcnemar = mcnemar_results['A_vs_C']
ac_wilcoxon = wilcoxon_results['A_vs_C']
ac_tost = tost_results['A_vs_C']
ac_noninf = noninf_results['A_vs_C']
ac_dunnett = dunnett_results['accuracy']['A_vs_C']
ac_fold = fold_level_results['accuracy']

print(f"""
  {'Test':<35} {'N':<8} {'Result':<20} {'p-value':<12}
  {'-'*75}
  McNemar's (per-sample)              {N_TEST:<8} {'Sig. diff' if ac_mcnemar['p_value'] < 0.05 else 'No sig. diff':<20} {ac_mcnemar['p_value']:<12.6f}
  Wilcoxon signed-rank (per-sample)   {N_TEST:<8} {'Sig. diff' if ac_wilcoxon['p_value'] < 0.05 else 'No sig. diff':<20} {ac_wilcoxon['p_value']:<12.6f}
  Per-sample TOST (margin={DELTA})      {N_TEST:<8} {'Equivalent' if ac_tost['equivalent'] else 'Not equiv.':<20} {ac_tost['p_tost']:<12.6f}
  Per-sample non-inferiority          {N_TEST:<8} {'Non-inferior' if ac_noninf['non_inferior'] else 'Inconclusive':<20} {ac_noninf['p_value']:<12.6f}
  Dunnett's (Bonf. corrected)         {5:<8} {'Sig. diff' if ac_dunnett['significant'] else 'No sig. diff':<20} {ac_dunnett['p_corrected']:<12.6f}
  Fold-level paired t-test            {5:<8} {'Sig. diff' if ac_fold['paired_t']['p'] < 0.05 else 'No sig. diff':<20} {ac_fold['paired_t']['p']:<12.6f}
  Fold-level TOST (margin={DELTA})      {5:<8} {'Equivalent' if ac_fold['tost']['equivalent'] else 'Not equiv.':<20} {ac_fold['tost']['p_tost']:<12.6f}
""")

# ============================================================================
# Step 11: Save results
# ============================================================================

print("\n[Step 11] Saving results...")

save_data = {
    'description': 'Enhanced statistical analysis with per-sample tests (N=22,469)',
    'ensemble_accuracy': {
        cond: float((ensemble_preds[cond] == labels).mean()) for cond in CONDITIONS
    },
    'mcnemar_tests': mcnemar_results,
    'wilcoxon_tests': wilcoxon_results,
    'per_sample_tost': tost_results,
    'per_sample_non_inferiority': noninf_results,
    'dunnett_tests': dunnett_results,
    'fold_level_supplementary': fold_level_results,
    'config': {
        'n_test': N_TEST,
        'n_folds': N_FOLDS,
        'delta': DELTA,
        'ensemble_method': 'mean_probability_across_folds',
        'seed': SEED
    }
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

output_path = OUTPUT_DIR / 'enhanced_statistical_results.json'
with open(output_path, 'w') as f:
    json.dump(save_data, f, indent=2, cls=NumpyEncoder)

print(f"  Results saved: {output_path}")
print("\nDone.")
