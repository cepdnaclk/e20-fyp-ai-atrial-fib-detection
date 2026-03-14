# Comparison Table: Normal VAE vs Class-Conditional VAE

| Metric                               | Normal VAE       | Class-Conditional VAE          | Difference |
| ------------------------------------ | ---------------- | ------------------------------ | ---------- |
| **Model Info**                       |                  |                                |            |
| Best Epoch                           | 54               | 58                             | +4         |
| Val Loss                             | 0.322888         | 0.318923                       | -0.004 ✅  |
| **Reconstruction Quality**           |                  |                                |            |
| MSE                                  | 0.025694 (2.57%) | 0.017094 (1.71%)               | -0.0086 ✅ |
| MAE                                  | 0.111584         | 0.099290                       | -0.012 ✅  |
| RMSE                                 | 0.158766         | 0.128562                       | -0.030 ✅  |
| RMSE as % of range                   | 5.29%            | 4.29%                          | -1.00% ✅  |
| Worst MSE                            | 0.044753         | 0.040086                       | -0.005 ✅  |
| Best MSE                             | 0.010152         | 0.008149                       | -0.002 ✅  |
| **Latent Space Health**              |                  |                                |            |
| Mean Std                             | 0.991172         | 0.991013                       | ~Same      |
| Collapsed Dimensions                 | 0.00%            | 0.00%                          | Same ✅    |
| Status                               | Healthy ✅       | Healthy ✅                     | Same       |
| **Classifier on Original ECGs**      |                  |                                |            |
| Overall Accuracy                     | 93.00%           | 93.00%                         | Same       |
| AFib Accuracy                        | 95.00%           | 95.00%                         | Same       |
| Normal Accuracy                      | 91.00%           | 91.00%                         | Same       |
| Mean Confidence                      | 0.9016           | 0.9016                         | Same       |
| **Classifier on Reconstructed ECGs** |                  |                                |            |
| Overall Accuracy                     | 84.50%           | 82.50%                         | -2.00% ❌  |
| AFib Accuracy                        | 96.00%           | 94.00%                         | -2.00% ❌  |
| Normal Accuracy                      | 73.00%           | 71.00%                         | -2.00% ❌  |
| Mean Confidence                      | 0.8160           | 0.8304                         | +0.014 ✅  |
| **Prediction Agreement**             |                  |                                |            |
| Overall Agreement                    | 85.50%           | 85.50%                         | Same       |
| AFib Agreement                       | 97.00%           | 95.00%                         | -2.00% ❌  |
| Normal Agreement                     | 74.00%           | 76.00%                         | +2.00% ✅  |
| **Detailed Breakdown**               |                  |                                |            |
| Both Correct                         | 81.50% (163)     | 80.50% (161)                   | -1.00%     |
| Both Wrong                           | 4.00% (8)        | 5.00% (10)                     | +1.00%     |
| Disagree                             | 14.50% (29)      | 14.50% (29)                    | Same       |
| **Confidence**                       |                  |                                |            |
| Confidence Drop                      | 0.1134           | 0.1094                         | -0.004 ✅  |
| **Random/Conditional Generation**    |                  |                                |            |
| Generation Quality                   | ⚠️ Poor          | AFib: ✅ Good, Normal: ⚠️ Poor | Partial ✅ |

---

## Summary

| Category                | Winner                   | Notes                      |
| ----------------------- | ------------------------ | -------------------------- |
| Reconstruction Quality  | **Class-Conditional** ✅ | Lower MSE (0.017 vs 0.026) |
| Latent Space            | **Tie**                  | Both healthy               |
| Classifier Agreement    | **Tie**                  | Same overall (85.5%)       |
| Normal Agreement        | **Class-Conditional** ✅ | 76% vs 74% (+2%)           |
| AFib Agreement          | **Normal VAE** ✅        | 97% vs 95%                 |
| Confidence Preservation | **Class-Conditional** ✅ | Less drop (0.109 vs 0.113) |
| Generation Quality      | **Class-Conditional** ✅ | AFib generation works      |
