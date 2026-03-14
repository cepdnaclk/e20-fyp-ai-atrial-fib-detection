#!/usr/bin/env python3
"""Quick diagnostic to estimate acceptance rate"""
import sys
sys.path.append('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual')
import torch
import numpy as np
from pathlib import Path
from shared_models import load_classifier, ClassifierWrapper
from plausibility_validator import PlausibilityValidator
from scipy.stats import pearsonr

# Load test data
DATA_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/data/processed/diffusion')
train_data = np.load(DATA_DIR / 'train_data.npz')
X_train = torch.FloatTensor(train_data['X'])
y_train = torch.LongTensor(train_data['y'])
if X_train.dim() == 2:
    X_train = X_train.unsqueeze(1)

# Load classifier
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = load_classifier(DEVICE)
wrapper = ClassifierWrapper(classifier).to(DEVICE)
validator = PlausibilityValidator()

# Test on existing counterfactuals (if any)
CF_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual/results')
batch_files = list(CF_DIR.glob('batch_Normal_to_AFib_*.npz'))

if batch_files:
    print(f"Testing {len(batch_files)} existing Normal→AFib batches...")
    total = 0
    flip_pass = 0
    plaus_pass = 0
    sim_pass = 0
    
    for bf in batch_files[:5]:  # Test first 5 batches
        data = np.load(bf)
        cfs = torch.FloatTensor(data['counterfactuals'])
        originals = data['originals']
        
        for i in range(min(50, len(cfs))):
            total += 1
            cf = cfs[i:i+1].unsqueeze(1).to(DEVICE)
            orig = originals[i]
            
            # Gate 1: Classifier flip
            mean = cf.mean(dim=2, keepdim=True)
            std = cf.std(dim=2, keepdim=True) + 1e-8
            cf_norm = (cf - mean) / std
            logits, _ = wrapper.model(cf_norm)
            pred = logits.argmax(dim=1).item()
            if pred == 1:  # Target is AFib
                flip_pass += 1
                
                # Gate 2: Plausibility
                val_result = validator.validate(
                    cfs[i].numpy(), original_ecg=orig,
                    target_class=1, original_class=0
                )
                if val_result['valid'] and val_result['score'] >= 0.7:
                    plaus_pass += 1
                    
                    # Gate 3: Similarity
                    try:
                        corr, _ = pearsonr(orig, cfs[i].numpy())
                        if corr > 0.3:
                            sim_pass += 1
                    except:
                        pass
    
    print(f"\nResults from {total} existing CFs:")
    print(f"  Gate 1 (flip) pass: {flip_pass}/{total} = {100*flip_pass/total:.1f}%")
    print(f"  Gate 2 (plausibility) pass: {plaus_pass}/{flip_pass if flip_pass > 0 else 1} = {100*plaus_pass/max(1,flip_pass):.1f}%")
    print(f"  Gate 3 (similarity) pass: {sim_pass}/{plaus_pass if plaus_pass > 0 else 1} = {100*sim_pass/max(1,plaus_pass):.1f}%")
    print(f"  Overall acceptance: {sim_pass}/{total} = {100*sim_pass/total:.1f}%")
    print(f"\n  Estimated: Need ~{int(10000 / (sim_pass/total))} attempts for 10K CFs")
    print(f"  At 1 attempt/sec: ~{int(10000 / (sim_pass/total) / 3600)} hours")
else:
    print("No existing batches found - can't estimate acceptance rate yet")
    print("The script is likely still generating the first samples")
