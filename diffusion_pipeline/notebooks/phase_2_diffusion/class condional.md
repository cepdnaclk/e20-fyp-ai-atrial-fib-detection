# class condional

# LOADING BEST MODEL FOR EVALUATION

Loading: vae_class_cond_best.pth
Epoch: 58
Val loss: 0.318923

Getting validation samples...
Loaded 64 validation samples
ECG shape: torch.Size([64, 1, 2500])
Labels shape: torch.Size([64, 1])
Normal samples: 30
AFib samples: 34

================================================================================
TEST 1: RECONSTRUCTION QUALITY
================================================================================

Overall Metrics:
MSE: 0.017094 (1.71%)
MAE: 0.099290
RMSE: 0.128562

Per-Sample Statistics:
MSE - Mean: 0.017094, Std: 0.006795
MAE - Mean: 0.099290, Std: 0.019800
Worst MSE: 0.040086
Best MSE: 0.008149

Normal ECGs:
MSE: 0.016827
MAE: 0.098649

AFib ECGs:
MSE: 0.017330
MAE: 0.099855

Clinical Context:
ECG range: [-1.5, 1.5] mV = 3.0 mV
RMSE as % of range: 4.29%
MAE as % of range: 3.31%

================================================================================
TEST 2: LATENT SPACE HEALTH (POSTERIOR COLLAPSE CHECK)
================================================================================

Latent Space Statistics:
Mean std: 0.991013
Median std: 0.999986
Min std: 0.067471
Max std: 1.002225

Posterior Collapse Analysis:
Total latent dimensions: 12,779,520
Collapsed dimensions (std < 0.01): 0
Collapse percentage: 0.00%
✅ HEALTHY LATENT SPACE
Latent dimensions are being used effectively.

================================================================================
TEST 3: CLASS-CONDITIONAL GENERATION
================================================================================

Generated AFib samples (n=5):
Mean: -0.231479
Std: 0.387103
Range: [-1.449412, 1.495082]

Generated Normal samples (n=5):
Mean: -0.736119
Std: 0.389753
Range: [-1.446401, 1.484540]

Quality Assessment:
✅ AFib generation: GOOD (reasonable statistics)
⚠️ Normal generation: POOR (unusual statistics)

================================================================================
CREATING VISUALIZATIONS
================================================================================

1. Creating reconstruction comparison...
   Saved: reconstruction_comparison.png
2. Creating reconstruction overlays...
   Saved: reconstruction_overlay.png
3. Creating class-conditional generations...
   Saved: class_conditional_generations.png

================================================================================
EVALUATION SUMMARY
================================================================================

📊 RECONSTRUCTION QUALITY:
MSE: 0.017094 (⚠️ FAIL - Target: <0.01)
RMSE: 0.128562
MAE: 0.099290
Error as % of signal range: 4.29%

🧬 LATENT SPACE HEALTH:
Mean std: 0.991013 (✅ HEALTHY)
Collapsed dimensions: 0.00%
Status: ✅ Healthy

🎨 CLASS-CONDITIONAL GENERATION:
AFib quality: ✅ Good
Normal quality: ⚠️ Poor

💡 ASSESSMENT:
❌ ISSUES DETECTED - Review training before proceeding

📈 NEXT STEP:
Run Cell 9 to test classifier agreement
This is the CRITICAL metric for class-conditional improvement!

📁 All visualizations saved to:
D:\research\codes\models\phase2_diffusion\results\vae_class_conditional

================================================================================
EVALUATION COMPLETE
================================================================================

================================================================================
PART 1: CLASSIFIER ACCURACY (Predictions vs Ground Truth)
================================================================================

📊 OVERALL ACCURACY:
Original ECGs: 20,855 / 22,484 = 92.75%
Reconstructed ECGs: 19,595 / 22,484 = 87.15%
Accuracy change: -5.60% ⚠️

📈 PER-CLASS ACCURACY:
Normal (Class 0):
Original: 10,634 / 11,243 = 94.58%
Reconstructed: 8,800 / 11,243 = 78.27%
Change: -16.31%
AFib (Class 1):
Original: 10,221 / 11,241 = 90.93%
Reconstructed: 10,795 / 11,241 = 96.03%
Change: +5.11%

================================================================================
PART 2: CLASSIFIER AGREEMENT (Original vs Reconstructed Predictions)
================================================================================

📊 OVERALL AGREEMENT:
Total samples: 22,484
Agreement: 19,560 / 22,484 = 87.00%

📈 PER-CLASS AGREEMENT:
Normal→Normal: 9,183 / 11,243 = 81.68%
AFib→AFib: 10,377 / 11,241 = 92.31%

================================================================================
PART 3: CONFUSION MATRICES
================================================================================

Original ECG Confusion Matrix:
Pred: Normal Pred: AFib
True: Normal 10634 609
True: AFib 1020 10221

Reconstructed ECG Confusion Matrix:
Pred: Normal Pred: AFib
True: Normal 8800 2443
True: AFib 446 10795

Prediction Agreement Matrix (Orig vs Recon):
Recon: Normal Recon: AFib
Orig: Normal 8988 2666
Orig: AFib 258 10572

================================================================================
PART 4: COMPARISON TO BASELINE
================================================================================

🎯 AGREEMENT COMPARISON:
Overall:
Baseline (no class cond.): 86.00%
Current (class cond.): 87.00%
Improvement: +1.00% ✅
Normal→Normal:
Baseline: 80.00%
Current: 81.68%
Improvement: +1.68% ✅
AFib→AFib:
Baseline: 92.00%
Current: 92.31%
Improvement: +0.31% ✅

================================================================================
PART 5: SUCCESS EVALUATION
================================================================================

✅ SUCCESS CRITERIA:

1. Overall agreement ≥88%: 87.00% ❌ FAIL
2. Normal agreement ≥85%: 81.68% ❌ FAIL
3. Improvement over baseline: +1.00% ✅ PASS

✅ PARTIAL SUCCESS
Improvement achieved, acceptable for proof-of-concept!

================================================================================
CREATING COMPREHENSIVE VISUALIZATIONS
================================================================================

1. Creating accuracy comparison plot...
   Saved: accuracy_and_agreement_comparison.png
2. Creating confusion matrices plot...
   Saved: confusion_matrices.png
3. Creating ECG comparison examples (agreement cases)...
   Saved: ecg_examples_agreement.png
4. Creating ECG comparison examples (disagreement cases)...
   Saved: ecg_examples_disagreement.png
5. Creating side-by-side comparison grid...
   Saved: ecg_side_by_side_comparison.png

================================================================================
FINAL SUMMARY
================================================================================

📊 CLASSIFIER ACCURACY:
Original ECG: 92.75%
Reconstructed ECG: 87.15%
Accuracy drop: -5.60%

📊 CLASSIFIER AGREEMENT:
Overall: 87.00% (baseline: 86.00%, +1.00%)
Normal→Normal: 81.68% (baseline: 80.00%, +1.68%)
AFib→AFib: 92.31% (baseline: 92.00%, +0.31%)

🎯 CONCLUSION:
✅ CLASS-CONDITIONAL VAE SHOWS IMPROVEMENT!
Beats baseline, demonstrating proof-of-concept.

📁 All visualizations saved to: D:\research\codes\models\phase2_diffusion\results\vae_class_conditional

================================================================================
COMPREHENSIVE ANALYSIS COMPLETE
================================================================================

================================================================================
CLINICAL VALIDATION: CLASSIFIER AGREEMENT TEST (Class-Conditional VAE)
================================================================================

Step 1: Loading Phase 1 AFib Classifier...
✅ Classifier loaded successfully
Device: cuda
Checkpoint: best_model.pth

Step 2: Setting up preprocessing...
✅ Preprocessing function defined

Step 3: Getting test samples...
Loaded validation data: (22484, 2500)

Test set created:
Total samples: 200
AFib samples: 100
Normal samples: 100
Test tensor shape: torch.Size([200, 1, 2500])
Labels tensor shape: torch.Size([200, 1])

================================================================================
STEP 4: CLASSIFIER ON ORIGINAL ECGS
================================================================================

Classifier Performance on ORIGINAL ECGs:
Accuracy: 93.00%
Mean confidence: 0.9016
AFib accuracy: 95.00%
Normal accuracy: 91.00%

================================================================================
STEP 5: RECONSTRUCT ECGS THROUGH CLASS-CONDITIONAL VAE
================================================================================
VAE Reconstruction (with class conditioning):
Input shape: torch.Size([200, 1, 2500])
Labels shape: torch.Size([200, 1])
Output shape: torch.Size([200, 1, 2500])
Reconstruction MSE: 0.016144
Input range: [-1.500, 1.500]
Output range: [-1.496, 1.483]

================================================================================
STEP 6: CLASSIFIER ON RECONSTRUCTED ECGS
================================================================================

Classifier Performance on RECONSTRUCTED ECGs:
Accuracy: 82.50%
Mean confidence: 0.8304
AFib accuracy: 94.00%
Normal accuracy: 71.00%

================================================================================
STEP 7: PREDICTION AGREEMENT ANALYSIS
================================================================================

🎯 CRITICAL METRIC:
Prediction Agreement: 85.50%

Agreement by class:

- AFib samples: 95.00%
- Normal samples: 76.00%

Detailed breakdown:

- Both correct: 161 (80.50%)
- Both wrong: 10 (5.00%)
- Disagree: 29 (14.50%)

Confidence comparison:

- Original mean conf: 0.9016
- Reconstructed mean conf: 0.8304
- Confidence drop: 0.1094

================================================================================
STEP 8: VISUALIZING EXAMPLES
================================================================================

Saved: classifier_agreement_examples.png
