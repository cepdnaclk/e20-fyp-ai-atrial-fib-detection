# Explainable AI for Counterfactual Simulation of Cardiac Arrhythmias

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**An Interactive XAI Framework for "What-If" ECG Simulation**

*Final Year Research Project - Group 29*  
*Department of Computer Engineering, University of Peradeniya, Sri Lanka*  
*2025-2026*

---

## Institutional Collaboration

This research is conducted through an international collaboration between the Department of Computer Engineering at University of Peradeniya (Sri Lanka), SimulaMet (Simula Metropolitan Center for Digital Engineering) in Oslo, Norway, and is part of the EU-funded SEARCH Initiative for cardiovascular AI research.

**Research Team:**

Students:
- Dilshan D.M.T. (E/20/069)
- Karunarathne K.N.P. (E/20/189)

Supervisors:
- **Dr. Vajira Thambawita** - Senior Researcher, SimulaMet, Oslo, Norway ([https://vajira.info/](https://vajira.info/))
- **Dr. Molly Maleckar** - Research Professor, SimulaMet, Oslo, Norway
- Dr. Isuru Nawinne - Senior Lecturer, University of Peradeniya, Sri Lanka
- Dr. Roshan Ragel - Professor, University of Peradeniya, Sri Lanka

---

## Overview

Deep learning models have demonstrated remarkable performance in detecting Atrial Fibrillation (AFib) from ECG signals, achieving accuracies exceeding 97%. However, these models operate as "black boxes," limiting their adoption in clinical practice due to lack of interpretability. Physicians cannot verify the reasoning behind AI predictions, creating a fundamental trust gap between AI systems and clinical decision-making.

This research addresses this critical limitation by developing an interactive Explainable AI (XAI) system that generates realistic counterfactual ECG signals. Our approach enables clinicians to explore "what-if" scenarios through an adjustable "Counterfactual Effort Score" parameter, providing transparent insights into AI decision-making processes.

### Core Functionality

The system operates in two modes:

- **Explanation Mode**: Given an AFib-classified ECG, generate a counterfactual showing what the patient's ECG would look like if classified as Normal
- **Prognosis Mode**: Given a Normal ECG, generate a counterfactual showing what changes would lead to an AFib classification

This approach moves beyond traditional XAI methods like Grad-CAM heatmaps, which only indicate where the model focused attention. Our system reveals what specific changes would be required to alter the diagnostic outcome, providing actionable clinical insights.

---

## Research Documentation

This section provides access to key project documentation, presentations, and comprehensive reviews developed throughout the research process.

### Project Presentations

**Research Proposal Presentation** (December 2024)  
Comprehensive presentation outlining the research objectives, methodology, and expected contributions.  
[View Presentation](https://www.canva.com/design/DAG4qXiPg9M/5yH3pT2uZwcUHr7vq-2UZA/edit)

**Project Workflow Documentation**  
Detailed visualization of Phase 1 and Phase 2 implementation pipeline, including data flow, model architecture, and evaluation metrics.  
[View Workflow](https://www.canva.com/design/DAGxWNCz_50/m1i5Yjk3yRWb9SeOnVByYw/edit)

### Literature Review

**Comprehensive Literature Review** - Counterfactual Explanations in Medical AI and ECG Analysis  
Systematic review covering foundational work in counterfactual generation, XAI for ECG, and AFib detection benchmarks. Includes critical evaluation of three counterfactual generation approaches (Heuristic/Prototype Methods, Latent Space Methods, and Diffusion/Style Transfer).  
[Download Literature Review (PDF)](https://drive.google.com/file/d/1XpHOSr7Cyw3i7yFoR6qp6hdMFHNk54KU/view?usp=sharing)

### Additional Resources

For access to additional technical documentation, dataset preparation notebooks, or research materials, please contact the research team directly.

---

## Current Progress and Achievements

### Phase 1: High-Accuracy AFib Classifier (Completed)

We implemented and validated a ResNet-BiLSTM classifier with attention mechanism for AFib detection. The model was trained on a combined dataset from PhysioNet 2017 Challenge and MIT-BIH AFDB.

**Performance Metrics:**
- Accuracy: 98.11%
- F1-Score: 0.9664
- AUROC: 0.9972
- Sensitivity: 96.34%
- Specificity: 98.59%

**Dataset Characteristics:**
- Combined PhysioNet 2017 + MIT-BIH AFDB
- Single-lead ECG recordings
- Sampling rate: 300 Hz
- Total recordings: 8,528 (PhysioNet) + 48 subjects (MIT-BIH)

### Phase 2: CoFE/VAE Counterfactual Engine (Validated)

We successfully implemented and validated a Conditional Variational Autoencoder (CoFE framework) for generating counterfactual ECG signals.

**Validation Results:**
- Classification flip success rate: 90% (Normal to AFib and vice versa)
- Mean Squared Error: 0.145 ± 0.051
- Signal distortion: <0.5% from original signal
- Physiological plausibility: Maintained through gradient-based latent optimization

The CoFE prototype demonstrates proof-of-concept for counterfactual ECG generation while maintaining temporal coherence and physiological realism.

### Phase 3: DiffStyleTS Diffusion Model (In Progress)

We are currently advancing the generative architecture to a state-of-the-art disentangled diffusion model to address limitations observed in the VAE approach. Traditional VAEs suffer from signal blurriness, posterior collapse, and entangled latent representations. The DiffStyleTS (Diffusion-based Style Transfer for Time Series) architecture addresses these issues through:

- Explicit content-style disentanglement via separate encoders
- Diffusion-based generation for high-fidelity signal quality
- Conditional denoising process for controlled counterfactual generation
- Provable interpretability through enforced architectural separation

The target architecture introduces the "Counterfactual Effort Score" - a continuous, interpretable metric that allows clinicians to control the strength of counterfactual transformation by adjusting content-style weighting parameters (W1, W2).

---

## Technical Architecture

```
Input: Patient ECG Signal
         |
         v
+------------------------------------------+
|  Phase 1: AFib Classifier (CNN-BiLSTM)  |
|  Output: Classification (AFib/Normal)    |
+------------------------------------------+
         |
         v
+------------------------------------------+
|  Phase 2/3: Counterfactual Generator     |
|                                          |
|  Content Encoder (C)                     |
|    - Extracts temporal patterns          |
|    - Preserves signal morphology         |
|                                          |
|  Style Encoder (S)                       |
|    - Captures rhythm characteristics     |
|    - Encodes class-specific features     |
|                                          |
|  Diffusion Denoiser                      |
|    - Iterative refinement process        |
|    - Condition: W1·C + W2·S              |
|                                          |
|  Effort Score = (W1, W2)                 |
|    - Clinician-adjustable parameters     |
|    - Controls transformation strength    |
+------------------------------------------+
         |
         v
Output: Counterfactual ECG + Explanation
        "Minimal changes required for 
         diagnostic flip"
```

---

## Datasets

The research utilizes three primary datasets across different phases:

**Classifier Training and Evaluation:**

| Dataset | Description | Size | Sampling Rate |
|---------|-------------|------|---------------|
| PhysioNet 2017 | AF Classification Challenge | 8,528 recordings | 300 Hz |
| MIT-BIH AFDB | Atrial Fibrillation Database | 48 subjects | 250 Hz |

**Counterfactual Engine Training:**

| Dataset | Description | Size | Sampling Rate |
|---------|-------------|------|---------------|
| MIMIC-IV ECG | Large-scale clinical database | 100,000+ subjects | Variable |

### Preprocessing Pipeline

All ECG data undergoes standardized preprocessing:

1. Resampling to uniform 250 Hz sampling rate
2. Multi-stage bandpass filtering (0.5-40 Hz)
3. Segmentation into 10-second windows
4. Z-score normalization
5. Binary labeling (AFib vs. Normal Sinus Rhythm)

---

## Methodology

The research follows a six-phase development pipeline:

**Phase 1: Data Preparation and Preprocessing** (Completed)
- Dataset integration and standardization
- Signal quality assessment and filtering
- Window segmentation and labeling

**Phase 2: Classifier Training** (Completed)
- Architecture selection and implementation
- Hyperparameter optimization
- Performance benchmarking against state-of-the-art

**Phase 3: Counterfactual Engine Training** (In Progress)
- DiffStyleTS architecture implementation
- Content and style encoder training
- Diffusion transformer denoiser training

**Phase 4: Counterfactual Generation Backend** (Planned)
- Effort score optimization algorithm
- Real-time inference pipeline
- Batch processing capabilities

**Phase 5: Evaluation and Validation** (Planned)
- Quantitative metrics: flip rate, MSE, signal quality
- Clinical validation with cardiologist review
- Comparison against commercial AI systems (MUSE 12SL)

**Phase 6: Interactive Clinician Interface** (Planned)
- Web-based dashboard development
- Real-time counterfactual generation
- Visualization and interpretation tools

---

## Research Contributions

### 1. Counterfactual Effort Score

A novel, interpretable metric that quantifies the "strength" of counterfactual transformation. Clinicians can adjust W1 (content preservation) and W2 (style transfer) to explore different scenarios:

- Low effort (W1≈1.0, W2≈0.0): Minimal modifications, subtle risk indicators
- High effort (W1≈0.0, W2≈1.0): Maximum transformation, clear diagnostic boundary

This provides a continuous risk assessment beyond binary classification.

### 2. Disentangled Diffusion Architecture

Our DiffStyleTS implementation addresses fundamental limitations of prior counterfactual generation methods:

**Problem with Path 1 (Heuristic/Prototype Methods):**
- Cut-and-paste stitching creates temporal misalignment
- Produces artifacts at transition points
- Not truly generative - cannot learn data distribution

**Problem with Path 2 (VAE-based Methods):**
- Posterior collapse during training
- Blurry signal reconstruction
- Entangled latent space representations
- Limited sample quality

**Our Solution (Path 3 - Diffusion Models):**
- High-fidelity generation through iterative denoising
- Enforced disentanglement via architectural design
- Provable interpretability through content-style separation
- State-of-the-art sample quality

### 3. Clinical Trust Bridge

Unlike standard XAI techniques that show attribution (where the model looked), our system provides:

- Concrete examples of what would change for different diagnoses
- Interactive exploration of diagnostic boundaries
- Quantifiable risk assessment through effort scores
- Physiologically plausible signal modifications

This addresses the fundamental clinical need to understand not just model predictions, but the reasoning behind them.

---

## Evaluation Framework

### Quantitative Metrics

**Counterfactual Quality:**
- Classification flip rate: Percentage of successful diagnostic reversals
- Mean Squared Error (MSE): Signal fidelity measurement
- Temporal Dynamic Warping (DTW): Shape preservation
- Frechet Inception Distance (FID): Distributional similarity

**Clinical Validation:**
- Expert cardiologist review of counterfactual realism
- Comparison against commercial AI (MUSE 12SL from GE Healthcare)
- Ablation studies on effort score impact
- User study with clinicians on interpretability

### Validation Plan

We will augment our Phase 1 training dataset with synthetically generated counterfactuals and demonstrate classifier performance improvement, validating both the quality of generated signals and their utility for data augmentation.

---

## Technical Implementation

**Core Frameworks:**
- PyTorch 2.0+ for deep learning
- NeuroKit2, SciPy, NumPy for signal processing
- Pandas for data management
- WFDB for ECG format handling
- Weights & Biases for experiment tracking

**Model Architectures:**
- Classifier: CNN-BiLSTM with multi-head attention
- Generator (Current): CoFE/VAE with gradient-based optimization
- Generator (Target): DiffStyleTS disentangled diffusion model

**Deployment Stack (Planned):**
- Backend: FastAPI for REST API
- Frontend: Streamlit for rapid prototyping
- Containerization: Docker
- Demonstration: Hugging Face Spaces

---

## Research Context

### Problem Statement

Atrial Fibrillation affects over 60 million people worldwide and contributes to approximately 454,000 deaths annually. While AI-based detection systems achieve high accuracy, their clinical adoption remains limited due to interpretability concerns. Physicians require transparent, verifiable explanations before trusting AI recommendations for patient care decisions.

### Research Gaps Addressed

We identified three critical limitations in existing AFib AI systems:

1. **Binary vs. Continuous Risk**: Current systems output discrete classifications without quantifying diagnostic certainty or progression risk
2. **Static vs. Dynamic Explanation**: No mechanism for interactive exploration of diagnostic boundaries
3. **Unrealistic Generation**: Prior counterfactual methods produce physiologically implausible signals through stitching or suffer from VAE blurriness

Our research directly addresses all three gaps through the DiffStyleTS architecture and interactive effort score mechanism.

---

## Literature Review Summary

### Foundational References

**Counterfactual Generation:**
- Kapsecker et al. (2025): "CoFE: A Framework Generating Counterfactual ECG for Explainable Cardiac AI-Diagnostics" - arXiv:2508.16033
- Nagda et al. (2025): "DiffStyleTS: Diffusion Model for Style Transfer in Time Series"
- Jang et al. (2025): "A novel XAI framework using generative counterfactual XAI (GCX)" - arXiv

**AFib Detection Benchmarks:**
- Jia et al. (2020): ResNet-BiLSTM-Attention achieving 98.11% accuracy
- Petmezas et al. (2021): CNN-LSTM with Focal Loss achieving 99.29% specificity
- Andersen et al. (2019): Hybrid CNN-LSTM achieving 97.8% accuracy

**XAI for ECG:**
- Hicks et al. (2021): "Explaining deep neural networks for knowledge discovery in electrocardiogram analysis" - Nature Communications
- Tanyel et al. (2023): "VCCE - Counterfactual ECG via Prototypes"
- Thambawita et al. (2021): Deep learning for ECG interpretation

### Critical Evaluation

We performed a systematic review of counterfactual generation approaches, categorizing them into three paths:

1. **Heuristic/Prototype Methods**: Simple but non-generative, creating temporal artifacts
2. **Latent Space Methods (VAE)**: Generative but suffer from quality issues
3. **Diffusion/Style Transfer**: State-of-the-art quality with provable interpretability

Our work builds on Path 3, implementing the most recent advances in diffusion-based time series generation.

---

## Timeline

```
2025 Q1  [Completed]  Phase 1: Data Pipeline + Classifier Training
2025 Q2  [Completed]  Phase 2: CoFE/VAE Prototype Validation
2025 Q3  [Current]    Phase 3: DiffStyleTS Implementation
2025 Q4  [Planned]    Phase 4: Backend Optimization
2026 Q1  [Planned]    Phase 5: Clinical Validation
2026 Q2  [Planned]    Phase 6: Interactive Dashboard + Thesis Defense
```

---

## Current Development Status

As of November 2025, we are actively working on Phase 3 - implementing the DiffStyleTS architecture. Specific tasks include:

- Training content and style encoders on MIMIC-IV ECG dataset
- Implementing diffusion transformer denoiser
- Developing effort score interpolation mechanism
- Establishing evaluation pipeline for counterfactual quality assessment

The CoFE/VAE prototype has successfully validated our approach, achieving 90% classification flip rate with minimal signal distortion. We are now advancing to the diffusion-based architecture to improve generation quality and provide stronger interpretability guarantees.

---

## Collaboration and Contact

This research is part of the EU-funded SEARCH Initiative (www.search-project.eu), which develops AI-powered solutions for cardiovascular disease detection and monitoring.

**Primary Supervisors:**  

Dr. Vajira Thambawita  
Senior Researcher, SimulaMet, Oslo, Norway  
Website: https://vajira.info/  
Email: vajira@simula.no  
Twitter: @vlbthambawita

Dr. Molly Maleckar  
Research Professor, SimulaMet, Oslo, Norway  
Website: https://www.simulamet.no/  
Email: molly@simula.no

**Student Researchers:**  
Dilshan D.M.T. - e20069@eng.pdn.ac.lk  
Karunarathne K.N.P. - e20189@eng.pdn.ac.lk

---

## Acknowledgments

We acknowledge the support of:
- SimulaMet (Oslo, Norway) for research guidance and computational resources
- EU SEARCH Initiative for funding and consortium collaboration
- University of Peradeniya for academic infrastructure
- PhysioNet, MIT-BIH, and MIMIC-IV teams for open-access datasets

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{pera_af_detection_2025,
  title={Explainable AI for Counterfactual Simulation of Cardiac Arrhythmias},
  author={Dilshan, D.M.T. and Karunarathne, K.N.P.},
  year={2025},
  institution={University of Peradeniya, Sri Lanka},
  collaboration={SimulaMet, Oslo, Norway},
  note={EU-funded SEARCH Initiative}
}
```

---

## Related Work

Dr. Vajira Thambawita's related projects:
- [deepfake-ecg](https://github.com/vlbthambawita/deepfake-ecg) - Unlimited realistic ECG generator
- [Pulse2Pulse](https://github.com/vlbthambawita/Pulse2Pulse) - DeepFake ECG generation framework

External initiatives:
- [SEARCH Project](https://www.search-project.eu/) - EU cardiovascular AI initiative

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Last Updated: November 2025*  
*Research Status: Active Development (Phase 3)*  
*Expected Completion: June 2026*
