# NORMAL

# LOADING BEST MODEL FOR EVALUATION

Loading: vae_60ep_best.pth
Epoch: 54
Val loss: 0.322888

Getting validation samples...
Loaded 64 validation samples

================================================================================
TEST 1: RECONSTRUCTION QUALITY
================================================================================

Overall Metrics:
MSE: 0.025694 (2.57%)
MAE: 0.111584
RMSE: 0.158766

Per-Sample Statistics:
MSE - Mean: 0.025694, Std: 0.007176
MAE - Mean: 0.111584, Std: 0.017272
Worst MSE: 0.044753
Best MSE: 0.010152

Clinical Context:
ECG range: [-1.5, 1.5] mV = 3.0 mV
RMSE as % of range: 5.29%
MAE as % of range: 3.72%

================================================================================
TEST 2: LATENT SPACE HEALTH (POSTERIOR COLLAPSE CHECK)
================================================================================

Latent Space Statistics:
Mean std: 0.991172
Median std: 1.000048
Min std: 0.060559
Max std: 1.003908

Posterior Collapse Analysis:
Total latent dimensions: 12,779,520
Collapsed dimensions (std < 0.01): 0
Collapse percentage: 0.00%
✅ HEALTHY LATENT SPACE
Latent dimensions are being used effectively.

================================================================================
TEST 3: RANDOM GENERATION FROM LATENT SPACE
================================================================================

Random Generation Statistics:
Mean: -0.563820
Std: 0.389341
Range: [-1.468188, 1.389070]
Expected range: [-1.5, 1.5] (Clinical range)
⚠️ POOR QUALITY GENERATION
Random samples may not be realistic ECGs.

================================================================================
CREATING VISUALIZATIONS
================================================================================
Saved: error_analysis.png

================================================================================
EVALUATION SUMMARY
================================================================================

📊 RECONSTRUCTION QUALITY:
MSE: 0.025694 (FAIL - Target: <0.01)
RMSE: 0.158766
MAE: 0.111584
Error as % of signal range: 5.29%

🧬 LATENT SPACE HEALTH:
Mean std: 0.991172 (HEALTHY)
Collapsed dimensions: 0.00%
Status: ✅ Healthy

🎲 RANDOM GENERATION:
Mean: -0.563820
Std: 0.389341
Quality: ⚠️ Poor

💡 RECOMMENDATIONS:

1. Current MSE is 0.025694, target is 0.01
2. This is EXPECTED with perceptual loss (prioritizes features over pixel-perfect)
3. Check classifier agreement in Cell 9 - that's more important!

📁 All visualizations saved to:
D:\research\codes\models\phase2_diffusion\results\vae_perceptual_fixed_60ep

================================================================================
EVALUATION COMPLETE
================================================================================
================================================================================
================================================================================
================================================================================
================================================================================
CLINICAL VALIDATION: CLASSIFIER AGREEMENT TEST
================================================================================

Step 1: Loading Phase 1 AFib Classifier...
✅ Classifier loaded successfully
Device: cuda
Checkpoint: best_model.pth

Step 2: Setting up preprocessing...
✅ Preprocessing function defined

Step 3: Getting test samples...
Loaded validation data: (22484, 2500)
Labels: (22484,)

Test set created:
Total samples: 200
AFib samples: 100
Normal samples: 100
Test tensor shape: torch.Size([200, 1, 2500])
Test tensor range: [-1.500, 1.500]

================================================================================
STEP 4: CLASSIFIER ON ORIGINAL ECGS
================================================================================
Original ECGs normalization check:
Per-sample means: min=-0.000000, max=0.000000, avg=0.000000
Per-sample stds: min=1.000200, max=1.000200, avg=1.000200
Expected: means ≈ 0, stds ≈ 1

Classifier Performance on ORIGINAL ECGs:
Accuracy: 93.00%
Mean confidence: 0.9016
Predictions: AFib=104, Normal=96
AFib accuracy: 95.00%
Normal accuracy: 91.00%

================================================================================
STEP 5: RECONSTRUCT ECGS THROUGH VAE
================================================================================
VAE Reconstruction:
Input shape: torch.Size([200, 1, 2500])
Output shape: torch.Size([200, 1, 2500])
Reconstruction MSE: 0.024329
Input range: [-1.500, 1.500]
Output range: [-1.479, 1.460]

================================================================================
STEP 6: CLASSIFIER ON RECONSTRUCTED ECGS
================================================================================
Reconstructed ECGs normalization check:
Per-sample means: min=-0.000000, max=0.000000, avg=-0.000000
Per-sample stds: min=1.000200, max=1.000200, avg=1.000200
Expected: means ≈ 0, stds ≈ 1

Classifier Performance on RECONSTRUCTED ECGs:
Accuracy: 84.50%
Mean confidence: 0.8160
Predictions: AFib=123, Normal=77
AFib accuracy: 96.00%
Normal accuracy: 73.00%

================================================================================
STEP 7: PREDICTION AGREEMENT ANALYSIS
================================================================================

🎯 CRITICAL METRIC:
Prediction Agreement: 85.50%
(How often original and reconstructed get same prediction)

Agreement by class:

- AFib samples: 97.00%
- Normal samples: 74.00%

Detailed breakdown:

- Both correct: 163 (81.50%)
- Both wrong: 8 (4.00%)
- Disagree: 29 (14.50%)

Disagreements breakdown:

- Original correct, Recon wrong: 23
- Original wrong, Recon correct: 6

Confidence comparison:

- Original mean conf: 0.9016
- Reconstructed mean conf: 0.8160
- Confidence drop: 0.1134

================================================================================
STEP 8: VISUALIZING EXAMPLES
================================================================================

Saved: classifier_agreement_examples.png
