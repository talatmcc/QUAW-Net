# Bachelor Degree Graduation Project Proposal

---

## **QUAW-Net: Quality-Aware Weighted Ensemble Network for Uncertainty-Guided Retinal Blood Vessel Segmentation with Multi-Expert Annotation Learning**

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Review](#3-literature-review)
4. [Proposed Methodology](#4-proposed-methodology)
5. [Implementation Plan](#5-implementation-plan)
6. [Expected Outcomes](#6-expected-outcomes)
7. [Challenges and Mitigation Strategies](#7-challenges-and-mitigation-strategies)
8. [References](#8-references)

---

## 1. Abstract

Accurate segmentation of retinal blood vessels is crucial for early diagnosis and monitoring of various ocular and systemic diseases, including diabetic retinopathy, hypertension, and cardiovascular disorders. However, manual annotation of retinal vessels by medical experts often exhibits significant inter-observer variability due to the inherent ambiguity in vessel boundaries, particularly in low-contrast regions and fine capillary structures. This variability poses a fundamental challenge for training reliable deep learning models, as conventional approaches treat all annotations equally without considering their quality or confidence.

This project proposes **QUAW-Net (Quality-Aware Weighted Network)**, a novel deep learning framework for retinal blood vessel segmentation that addresses multi-expert annotation ambiguity through a lightweight **Annotation Quality Estimation Module (AQEM)**. Built upon an Attention U-Net backbone, QUAW-Net introduces a learnable weighting mechanism that dynamically assesses and weighs expert annotations based on consistency and confidence scores during training. Additionally, we incorporate **Monte Carlo Dropout** for uncertainty quantification, enabling the model to generate pixel-wise uncertainty maps that highlight regions of high ambiguity.

The key contributions include: (1) a novel annotation quality weighting scheme for multi-expert learning, (2) boundary-aware uncertainty estimation for ambiguous vessel regions, and (3) clinically interpretable uncertainty visualizations for decision support. We evaluate our approach on the DRIVE and CHASE_DB1 datasets, targeting a Dice score exceeding 85% while providing calibrated uncertainty estimates. This work bridges the gap between probabilistic segmentation research and practical clinical deployment, offering a computationally efficient solution suitable for resource-constrained environments.

**Keywords:** Retinal vessel segmentation, uncertainty quantification, multi-annotator learning, annotation quality weighting, deep learning, medical image analysis

---

## 2. Introduction

### 2.1 Background and Motivation

Medical image segmentation is a fundamental task in computer-aided diagnosis (CAD) systems, enabling automated identification and delineation of anatomical structures and pathological regions. Among various medical imaging modalities, retinal fundus imaging holds particular clinical significance as it provides a non-invasive window into the microvasculature of the human body. The retinal blood vessel network serves as a critical biomarker for numerous diseases, including diabetic retinopathy (DR), age-related macular degeneration (AMD), glaucoma, and systemic conditions such as hypertension and arteriosclerosis.

Accurate segmentation of retinal blood vessels enables quantitative analysis of vessel morphology, including vessel width, tortuosity, branching patterns, and arterio-venous ratio—all of which carry diagnostic value. Early detection of vascular abnormalities through automated analysis can significantly improve patient outcomes by enabling timely intervention. According to the World Health Organization, diabetic retinopathy affects approximately 35% of people with diabetes and is a leading cause of preventable blindness worldwide. Automated vessel segmentation systems can assist ophthalmologists in screening large populations efficiently, particularly in resource-limited settings where specialist availability is scarce.

### 2.2 The Challenge of Inter-Observer Variability

Despite the clinical importance of vessel segmentation, developing reliable automated systems faces a fundamental challenge: **inter-observer variability** in expert annotations. When multiple ophthalmologists or trained graders annotate the same retinal image, their segmentations often differ significantly, particularly in the following regions:

1. **Fine capillary structures:** The smallest vessels near the detection threshold of imaging systems are subject to varying interpretations.

2. **Low-contrast regions:** Areas with poor image quality, uneven illumination, or pathological changes (hemorrhages, exudates) create annotation ambiguity.

3. **Vessel boundaries:** The exact boundary between vessel and background is often gradual rather than discrete, leading to different delineation choices.

4. **Crossing and bifurcation points:** Complex vascular junctions require subjective decisions about vessel connectivity.

Traditional deep learning approaches treat this variability as noise to be averaged out or ignored entirely by using a single "consensus" annotation. However, this approach discards valuable information about inherent uncertainty in the segmentation task and fails to communicate diagnostic confidence to end-users.

### 2.3 Research Gap

Recent advances in probabilistic deep learning have demonstrated the importance of uncertainty quantification in medical imaging. Methods such as Bayesian neural networks, Monte Carlo Dropout, and stochastic segmentation networks have shown promise in capturing predictive uncertainty. Additionally, multi-annotator learning approaches have emerged to leverage disagreement between experts as a signal rather than noise.

However, a significant gap exists in the current literature:

1. **Equal treatment of annotations:** Most multi-annotator methods assume all expert annotations are equally reliable, ignoring the reality that annotation quality varies based on annotator expertise, image quality in specific regions, and inherent task difficulty.

2. **Computational complexity:** Advanced probabilistic methods often require substantial computational resources (multiple forward passes, complex covariance modeling), making them impractical for deployment in clinical settings with limited hardware.

3. **Domain-specific adaptation:** While general probabilistic segmentation frameworks exist, they rarely incorporate domain knowledge specific to retinal imaging, such as the importance of boundary precision for vessel width measurement.

4. **Clinical interpretability:** Many uncertainty quantification methods produce uncertainty maps without clear clinical interpretation guidelines.

### 2.4 Proposed Solution

To address these gaps, this project proposes **QUAW-Net**, a novel framework that introduces:

1. **Annotation Quality Estimation Module (AQEM):** A lightweight neural network module that learns to assign quality weights to different expert annotations based on their consistency with the ensemble and local image characteristics.

2. **Boundary-Aware Uncertainty Loss:** A custom loss component that emphasizes accurate uncertainty estimation in ambiguous boundary regions, where clinical decisions are most sensitive.

3. **Efficient Uncertainty Quantification:** Integration of Monte Carlo Dropout for uncertainty estimation, providing a balance between computational efficiency and uncertainty calibration quality.

### 2.5 Research Objectives

This project aims to achieve the following objectives:

**Objective 1:** Design and implement an Attention U-Net based segmentation framework for retinal blood vessel segmentation that achieves state-of-the-art performance on public benchmarks (DRIVE, CHASE_DB1).

**Objective 2:** Develop a novel Annotation Quality Estimation Module (AQEM) that learns to dynamically weight expert annotations during training, improving model robustness to annotation quality variations.

**Objective 3:** Implement Monte Carlo Dropout-based uncertainty quantification and develop boundary-aware uncertainty visualization tools for clinical decision support.

**Objective 4:** Conduct comprehensive experimental evaluation comparing the proposed method with baseline approaches, including ablation studies to validate the contribution of each component.

### 2.6 Significance and Expected Impact

The successful completion of this project will contribute to both the academic research community and clinical practice:

**Academic Contributions:**
- A novel annotation weighting mechanism applicable to various multi-annotator learning scenarios
- Empirical analysis of uncertainty calibration in retinal vessel segmentation
- Open-source implementation for reproducibility and future research

**Clinical Impact:**
- Improved segmentation accuracy in challenging image regions
- Uncertainty maps enabling clinicians to identify cases requiring manual review
- Computational efficiency suitable for integration into existing ophthalmic screening workflows

---

## 3. Literature Review

### 3.1 Traditional Retinal Vessel Segmentation Methods

Early approaches to retinal vessel segmentation relied on hand-crafted features and classical image processing techniques. These methods can be broadly categorized into:

**Matched Filtering Methods:** Chaudhuri et al. (1989) pioneered the use of 2D Gaussian matched filters to detect vessels by exploiting their linear structure. Subsequent works improved upon this approach by incorporating multi-scale analysis and adaptive thresholding.

**Morphological Processing:** Mathematical morphology operations, including top-hat transforms and opening/closing operations, have been widely used to enhance vessel structures while suppressing background noise.

**Vessel Tracking Methods:** These approaches trace vessels from seed points using local vessel models, offering connectivity-preserving segmentation but suffering from sensitivity to initialization.

**Supervised Classification:** Before deep learning, machine learning classifiers such as support vector machines (SVMs), random forests, and k-nearest neighbors were employed with hand-crafted features including Gabor filter responses, gradient features, and Hessian-based vesselness measures.

While these traditional methods established important foundations, they generally achieve lower accuracy than modern deep learning approaches and require careful parameter tuning for different datasets.

### 3.2 Deep Learning for Medical Image Segmentation

The introduction of Fully Convolutional Networks (FCNs) by Long et al. (2015) marked a paradigm shift in semantic segmentation, enabling end-to-end dense prediction without hand-crafted features.

**U-Net Architecture:** Ronneberger et al. (2015) proposed U-Net, which has become the de facto standard architecture for medical image segmentation. The key innovations include:
- Symmetric encoder-decoder structure with skip connections
- Concatenation of multi-resolution features for precise localization
- Effective performance with limited training data through aggressive data augmentation

U-Net achieves excellent results on retinal vessel segmentation, with reported Dice scores of 78-82% on standard benchmarks.

**Attention U-Net:** Oktay et al. (2018) introduced attention gates into the U-Net architecture, allowing the model to focus on relevant structures while suppressing irrelevant regions. This is particularly beneficial for vessel segmentation, where vessels occupy a small fraction of the image and vary in scale.

**Other Notable Architectures:**
- **Dense U-Net:** Incorporates dense connections within encoder blocks for improved feature reuse
- **R2U-Net:** Combines residual connections with recurrent convolutions for refined segmentation
- **CE-Net:** Context encoder network with dense atrous convolution for multi-scale feature extraction

These architectures have pushed vessel segmentation performance to Dice scores of 80-85% on public datasets.

### 3.3 Uncertainty Quantification in Medical Imaging

Uncertainty quantification has gained significant attention in medical imaging due to the high-stakes nature of clinical decisions.

**Types of Uncertainty:**
- **Aleatoric uncertainty:** Inherent noise in the data that cannot be reduced with more training data (e.g., ambiguous boundaries)
- **Epistemic uncertainty:** Model uncertainty due to limited training data, which can be reduced with more data or better models

**Monte Carlo Dropout:** Gal and Ghahramani (2016) demonstrated that dropout at test time can approximate Bayesian inference. By performing multiple forward passes with dropout enabled, one can estimate predictive uncertainty:

$$\text{Uncertainty} = \frac{1}{T}\sum_{t=1}^{T} \text{Var}[\hat{y}_t]$$

This approach is computationally efficient and easy to implement, making it popular in medical imaging applications.

**Bayesian Neural Networks:** Fully Bayesian approaches place distributions over network weights, enabling principled uncertainty estimation. However, exact inference is intractable, leading to various approximation methods (variational inference, Hamiltonian Monte Carlo).

**Ensemble Methods:** Training multiple models with different initializations or architectures and aggregating their predictions provides uncertainty estimates through prediction disagreement. While effective, this approach multiplies computational requirements.

**Stochastic Segmentation Networks (SSN):** Monteiro et al. (2020) proposed modeling the segmentation output as a multivariate distribution, enabling sampling of diverse plausible segmentations. This approach captures structured spatial uncertainty but requires complex covariance modeling.

### 3.4 Multi-Annotator Learning

Medical image annotation often involves multiple experts, and recent work has explored how to leverage this information:

**STAPLE Algorithm:** Warfield et al. (2004) proposed Simultaneous Truth and Performance Level Estimation, a probabilistic framework that estimates the true segmentation and annotator performance simultaneously using expectation-maximization.

**Probabilistic U-Net:** Kohl et al. (2018) introduced a conditional variational autoencoder framework for generating multiple plausible segmentations, capable of capturing multi-modal annotation distributions.

**PLATO:** Recent work on Probabilistic Hierarchical Multi-head Models demonstrates the value of multiple probabilistic output heads for capturing annotation diversity.

**Label Fusion Methods:** Various methods combine multiple annotations through voting, weighted averaging, or learned fusion networks. However, most assume fixed annotator reliability.

### 3.5 Recent Advances (2020-2025)

**Transformer-based Methods:** Vision Transformers (ViT) and hybrid CNN-Transformer architectures have shown promising results for medical image segmentation, with improved global context modeling.

**Self-supervised Pre-training:** Methods like contrastive learning and masked image modeling enable leveraging unlabeled data, addressing the limited annotation problem in medical imaging.

**Lightweight Architectures:** With increasing focus on deployment, efficient architectures like EfficientNet backbones and knowledge distillation techniques enable mobile and edge deployment.

**Calibration-aware Training:** Recent work emphasizes not just accuracy but calibration—ensuring that predicted probabilities match empirical frequencies, crucial for clinical decision-making.

### 3.6 Gap Analysis and Positioning

| Aspect | Existing Approaches | Proposed QUAW-Net |
|--------|--------------------|--------------------|
| Annotation Handling | Equal weighting or consensus | Learned quality-based weighting |
| Uncertainty Method | Complex probabilistic models | Efficient MC Dropout |
| Computational Cost | High (multiple heads, covariance) | Low (single head, lightweight AQEM) |
| Domain Specificity | General frameworks | Boundary-aware for vessels |
| Clinical Utility | Limited interpretability | Uncertainty visualization tool |

**Our positioning:** QUAW-Net bridges the gap between sophisticated multi-annotator learning and practical clinical deployment by introducing a lightweight yet effective annotation quality weighting mechanism, combined with efficient uncertainty quantification tailored to retinal vessel segmentation.

---

## 4. Proposed Methodology

### 4.1 System Architecture Overview

The proposed QUAW-Net framework consists of four main components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUAW-Net Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────────┐    ┌───────────────────────┐  │
│  │   Input      │───▶│   Attention U-Net    │───▶│   MC Dropout Layer   │  │
│  │   Image      │    │   Backbone           │    │   (Uncertainty Est.) │  │
│  └──────────────┘    └──────────────────────┘    └───────────┬───────────┘  │
│                                                               │              │
│                                                               ▼              │
│  ┌──────────────┐    ┌──────────────────────┐    ┌───────────────────────┐  │
│  │   Expert     │───▶│   Annotation Quality │───▶│   Weighted Loss      │  │
│  │   Annotations│    │   Estimation Module  │    │   Computation        │  │
│  └──────────────┘    └──────────────────────┘    └───────────┬───────────┘  │
│                                                               │              │
│                                                               ▼              │
│                                                  ┌───────────────────────┐  │
│                                                  │   Final Prediction    │  │
│                                                  │   + Uncertainty Map   │  │
│                                                  └───────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**[Figure 1: Overall system architecture of QUAW-Net - to be included in final document]**

### 4.2 Dataset Description and Preprocessing

#### 4.2.1 Datasets

We will use two publicly available retinal vessel segmentation datasets:

**DRIVE (Digital Retinal Images for Vessel Extraction):**
- 40 color fundus images (20 training, 20 test)
- Image resolution: 565 × 584 pixels
- Two manual annotations per image (from different experts)
- Field of view (FOV) masks provided
- Source: Screening program for diabetic retinopathy in the Netherlands

**CHASE_DB1:**
- 28 color fundus images from both eyes of 14 children
- Image resolution: 999 × 960 pixels
- Two manual annotations per image
- Challenging due to central vessel reflex and uneven illumination
- Source: Child Heart and Health Study in England

**Optional Extension - STARE:**
- 20 images with two annotations each
- Can be used for additional validation if time permits

#### 4.2.2 Data Preprocessing Pipeline

```python
# Preprocessing Pipeline (Pseudocode)
def preprocess_image(image, mask):
    # Step 1: Extract green channel (highest vessel contrast)
    green_channel = image[:, :, 1]
    
    # Step 2: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    
    # Step 3: Gamma correction (gamma=1.2)
    gamma_corrected = adjust_gamma(enhanced, gamma=1.2)
    
    # Step 4: Normalize to [0, 1]
    normalized = gamma_corrected / 255.0
    
    # Step 5: Apply FOV mask
    final_image = normalized * mask
    
    return final_image
```

**Data Augmentation Strategy:**
- Random rotation (0°, 90°, 180°, 270°)
- Random horizontal and vertical flips
- Random cropping (patches of 128×128 or 256×256)
- Elastic deformation (for vessel shape variation)
- Brightness and contrast jittering (±10%)

**Patch Extraction:**
- Training: Random patches from each image
- Testing: Overlapping patches with stride for full image reconstruction

### 4.3 Base Segmentation Network: Attention U-Net

#### 4.3.1 Encoder Path

The encoder follows a standard convolutional neural network structure with progressively increasing channels:

| Stage | Input Channels | Output Channels | Operations |
|-------|---------------|-----------------|------------|
| Enc1 | 1 (grayscale) | 64 | Conv3×3, BN, ReLU × 2 |
| Pool1 | 64 | 64 | MaxPool 2×2 |
| Enc2 | 64 | 128 | Conv3×3, BN, ReLU × 2 |
| Pool2 | 128 | 128 | MaxPool 2×2 |
| Enc3 | 128 | 256 | Conv3×3, BN, ReLU × 2 |
| Pool3 | 256 | 256 | MaxPool 2×2 |
| Enc4 | 256 | 512 | Conv3×3, BN, ReLU × 2 |

#### 4.3.2 Attention Gate Mechanism

The attention gate learns to focus on relevant vessel structures:

```
Attention Gate Operation:
g: Gating signal from decoder (coarse features)
x: Skip connection from encoder (fine features)

θ_x = W_x · x + b_x    (1×1 convolution)
θ_g = W_g · g + b_g    (1×1 convolution)
ψ = σ(W_ψ · ReLU(θ_x + θ_g) + b_ψ)    (attention coefficients)
output = x · ψ    (attended features)
```

**[Figure 2: Attention gate block diagram - to be included]**

#### 4.3.3 Decoder Path with Skip Connections

| Stage | Input | Operations | Output |
|-------|-------|------------|--------|
| Up4 | Enc4 | UpConv 2×2 → Attention(Enc3) → Concat → Conv×2 | 256 |
| Up3 | Up4 | UpConv 2×2 → Attention(Enc2) → Concat → Conv×2 | 128 |
| Up2 | Up3 | UpConv 2×2 → Attention(Enc1) → Concat → Conv×2 | 64 |
| Out | Up2 | Conv 1×1, Sigmoid | 1 |

### 4.4 Annotation Quality Estimation Module (AQEM)

**This is the core novelty of our approach.**

#### 4.4.1 Motivation

Not all expert annotations are equally reliable:
- Some annotators may be more experienced
- Image quality varies by region (some areas are inherently harder)
- Certain vessel structures (fine capillaries) are more ambiguous

AQEM learns to estimate annotation quality dynamically, assigning higher weights to more reliable annotations.

#### 4.4.2 Architecture

```
AQEM Architecture:

Input: Concatenation of [Image, Annotation_k, Current_Prediction]
       Shape: (H, W, 3)

Layer 1: Conv 3×3, 32 filters, ReLU
Layer 2: Conv 3×3, 32 filters, ReLU
Layer 3: Global Average Pooling → (32,)
Layer 4: FC 32 → 16, ReLU
Layer 5: FC 16 → 1, Sigmoid → Quality Score q_k ∈ [0, 1]

For K annotations: Q = [q_1, q_2, ..., q_K]
Normalized weights: W = softmax(Q / τ)  where τ is temperature
```

**[Figure 3: AQEM architecture diagram - to be included]**

#### 4.4.3 Quality Score Interpretation

The quality score considers:
1. **Consistency:** How well the annotation agrees with other annotations
2. **Confidence:** Local image characteristics (contrast, clarity)
3. **Boundary Precision:** Sharpness of annotation boundaries

#### 4.4.4 Training Dynamics

During training, AQEM weights evolve:
- Early training: Weights are relatively uniform
- Mid training: Model learns to identify reliable annotations
- Late training: Stable quality assessment

We use a temperature annealing schedule:
```
τ(epoch) = τ_max × exp(-epoch / decay_rate) + τ_min
```
Starting with high temperature (uniform weights) and decreasing to sharpen weight distribution.

### 4.5 Monte Carlo Dropout for Uncertainty Estimation

#### 4.5.1 Implementation

We add dropout layers (p=0.2) after each decoder block. At inference:

```python
def mc_inference(model, image, T=10):
    model.train()  # Enable dropout
    predictions = []
    
    for t in range(T):
        with torch.no_grad():
            pred = model(image)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    
    # Mean prediction
    mean_pred = predictions.mean(dim=0)
    
    # Uncertainty (predictive variance)
    uncertainty = predictions.var(dim=0)
    
    return mean_pred, uncertainty
```

#### 4.5.2 Uncertainty Decomposition

We decompose total uncertainty into:
- **Aleatoric (data uncertainty):** Captured by prediction variance in ambiguous regions
- **Epistemic (model uncertainty):** Captured by disagreement across MC samples

### 4.6 Loss Functions

#### 4.6.1 Weighted Dice Loss

Standard Dice loss modified to incorporate annotation quality weights:

```
L_dice = 1 - (2 × Σ(p × g) + ε) / (Σp + Σg + ε)

Weighted version:
L_weighted_dice = Σ_k w_k × L_dice(pred, annotation_k)

where w_k = AQEM(image, annotation_k, pred)
```

#### 4.6.2 Boundary-Aware Loss

We add a loss component that emphasizes boundary regions:

```
1. Compute boundary masks using morphological gradient:
   boundary_k = dilate(annotation_k) - erode(annotation_k)

2. Boundary-weighted BCE loss:
   L_boundary = BCE(pred, annotation_k, weight=1 + α × boundary_k)
   
   where α controls boundary emphasis (we use α=2)
```

#### 4.6.3 Total Loss

```
L_total = λ_1 × L_weighted_dice + λ_2 × L_boundary + λ_3 × L_aqem_reg

where:
- λ_1 = 1.0 (main segmentation loss)
- λ_2 = 0.3 (boundary emphasis)
- λ_3 = 0.1 (AQEM regularization to prevent degenerate solutions)

L_aqem_reg = -entropy(W)  # Encourage non-trivial weight distribution
```

### 4.7 Training Strategy

#### 4.7.1 Progressive Training Schedule

| Phase | Epochs | Description |
|-------|--------|-------------|
| Phase 1 | 1-30 | Train base U-Net with uniform annotation weights |
| Phase 2 | 31-60 | Introduce AQEM with high temperature (τ=2.0) |
| Phase 3 | 61-100 | Full training with temperature annealing |

#### 4.7.2 Optimization Settings

- **Optimizer:** AdamW with weight decay 1e-4
- **Learning Rate:** 1e-4 with cosine annealing
- **Batch Size:** 8 (patches per batch)
- **Epochs:** 100 total
- **Early Stopping:** Patience of 15 epochs on validation Dice

### 4.8 Evaluation Metrics

#### 4.8.1 Segmentation Performance

| Metric | Formula | Description |
|--------|---------|-------------|
| Dice Score | 2TP / (2TP + FP + FN) | Overlap measure |
| IoU (Jaccard) | TP / (TP + FP + FN) | Intersection over union |
| Sensitivity | TP / (TP + FN) | True positive rate |
| Specificity | TN / (TN + FP) | True negative rate |
| Accuracy | (TP + TN) / Total | Overall accuracy |

#### 4.8.2 Boundary Quality

| Metric | Description |
|--------|-------------|
| Hausdorff Distance (HD95) | 95th percentile of boundary distances |
| Average Surface Distance | Mean distance between boundaries |

#### 4.8.3 Uncertainty Calibration

| Metric | Description |
|--------|-------------|
| Expected Calibration Error (ECE) | Difference between predicted confidence and accuracy |
| Brier Score | Mean squared error of probabilistic predictions |
| Uncertainty-Error Correlation | Correlation between uncertainty and actual errors |

---

## 5. Implementation Plan

### 5.1 Development Environment

**Hardware Requirements:**
- GPU: NVIDIA RTX 3060 (12GB VRAM) or equivalent
- RAM: 16GB minimum
- Storage: 50GB for datasets and checkpoints

**Software Stack:**
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8
- Key libraries: torchvision, albumentations, scikit-image, matplotlib, wandb

### 5.2 Project Timeline (5 Months)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Project Timeline                                    │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────┤
│  Month 1 │  Month 2 │  Month 3 │  Month 4 │  Month 5 │                      │
├──────────┼──────────┼──────────┼──────────┼──────────┤                      │
│██████████│          │          │          │          │ Literature Review    │
│██████████│          │          │          │          │ Dataset Preparation  │
│          │██████████│██████████│          │          │ Base Model Dev       │
│          │          │██████████│██████████│          │ AQEM Development     │
│          │          │          │██████████│██████████│ Experiments          │
│          │          │          │          │██████████│ Thesis Writing       │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────┘
```

#### Month 1: Foundation (Weeks 1-4)
- **Week 1-2:** Deep literature review, reading 20+ papers
- **Week 2-3:** Download and explore DRIVE, CHASE_DB1 datasets
- **Week 3-4:** Implement preprocessing pipeline and data loaders
- **Deliverable:** Literature review document, working data pipeline

#### Month 2: Base Model (Weeks 5-8)
- **Week 5-6:** Implement Attention U-Net from scratch
- **Week 6-7:** Train and validate base model
- **Week 7-8:** Achieve baseline performance (target: 80% Dice)
- **Deliverable:** Trained baseline model, initial results

#### Month 3: Novel Components (Weeks 9-12)
- **Week 9-10:** Design and implement AQEM
- **Week 10-11:** Integrate AQEM into training loop
- **Week 11-12:** Implement MC Dropout and boundary-aware loss
- **Deliverable:** Complete QUAW-Net implementation

#### Month 4: Experiments (Weeks 13-16)
- **Week 13-14:** Hyperparameter tuning (learning rate, λ values, temperature)
- **Week 14-15:** Ablation studies (with/without AQEM, boundary loss)
- **Week 15-16:** Comparison with baselines (U-Net, Attention U-Net, MC Dropout only)
- **Deliverable:** Complete experimental results, analysis

#### Month 5: Writing and Finalization (Weeks 17-20)
- **Week 17-18:** Write methodology and experiments chapters
- **Week 18-19:** Prepare visualizations, clean code repository
- **Week 19-20:** Thesis review, presentation preparation, defense preparation
- **Deliverable:** Complete thesis, presentation, GitHub repository

### 5.3 Code Organization

```
quaw-net/
├── data/
│   ├── DRIVE/
│   └── CHASE_DB1/
├── src/
│   ├── models/
│   │   ├── attention_unet.py
│   │   ├── aqem.py
│   │   └── quaw_net.py
│   ├── datasets/
│   │   ├── drive_dataset.py
│   │   └── chase_dataset.py
│   ├── utils/
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/
│   ├── EDA.ipynb
│   └── Results_Analysis.ipynb
├── configs/
│   └── config.yaml
├── requirements.txt
└── README.md
```

---

## 6. Expected Outcomes

### 6.1 Quantitative Results

**Target Performance on DRIVE Dataset:**
| Method | Dice (%) | IoU (%) | Sensitivity (%) | Specificity (%) |
|--------|----------|---------|-----------------|-----------------|
| U-Net (baseline) | 79.5 | 65.8 | 76.0 | 98.2 |
| Attention U-Net | 81.2 | 68.4 | 78.5 | 98.4 |
| QUAW-Net (ours) | **85.0+** | **72.0+** | **82.0+** | **98.5+** |

**Target Performance on CHASE_DB1:**
| Method | Dice (%) | IoU (%) |
|--------|----------|---------|
| U-Net (baseline) | 76.5 | 62.0 |
| QUAW-Net (ours) | **82.0+** | **69.0+** |

### 6.2 Qualitative Outcomes

1. **Uncertainty Maps:** Visualization showing high uncertainty in:
   - Fine capillary regions
   - Low-contrast areas
   - Vessel crossings and bifurcations

   **[Figure 4: Example uncertainty visualization - to be included]**

2. **Annotation Weight Analysis:** Demonstration that AQEM assigns:
   - Higher weights to annotations with clearer boundaries
   - Lower weights to annotations in ambiguous regions

3. **Boundary Improvement:** Visual comparison showing improved boundary delineation compared to baselines

### 6.3 Deliverables

| Deliverable | Description | Format |
|-------------|-------------|--------|
| Thesis Document | Complete 40-50 page technical report | PDF |
| Trained Models | Best QUAW-Net checkpoint | .pth file |
| Code Repository | Complete, documented codebase | GitHub |
| Presentation | Defense slides | PowerPoint/PDF |
| Uncertainty Tool | Web-based visualization demo | Streamlit app |

### 6.4 Publication Potential

The novelty of AQEM and boundary-aware uncertainty presents opportunities for:
- Medical imaging workshops (MICCAI, ISBI workshops)
- National conferences on medical imaging or machine learning
- Journal publication after extended experiments

---

## 7. Challenges and Mitigation Strategies

### 7.1 Technical Challenges

| Challenge | Risk Level | Mitigation Strategy |
|-----------|------------|---------------------|
| Limited GPU memory | Medium | Use smaller patch sizes, gradient checkpointing |
| Overfitting on small dataset | High | Aggressive augmentation, dropout, early stopping |
| AQEM training instability | Medium | Careful weight initialization, gradual introduction |
| Uncertainty calibration issues | Medium | Temperature scaling, calibration evaluation |

### 7.2 Dataset-Related Challenges

| Challenge | Risk Level | Mitigation Strategy |
|-----------|------------|---------------------|
| Only 2 annotations per image | Medium | Synthesize additional annotations via augmentation |
| Different image qualities | Low | Per-image normalization, CLAHE |
| Class imbalance (vessels << background) | Medium | Weighted loss, focal loss option |

### 7.3 Time Management Challenges

| Challenge | Risk Level | Mitigation Strategy |
|-----------|------------|---------------------|
| Implementation delays | Medium | Start with existing U-Net code, modify |
| Hyperparameter tuning time | Medium | Use Optuna for automated tuning |
| Writing time constraints | Medium | Start documentation early, maintain notes |

### 7.4 Fallback Plans

1. **If AQEM doesn't improve results:** Focus on MC Dropout uncertainty analysis alone (still valuable contribution)

2. **If training is too slow:** Reduce model size, use pre-trained encoder (ResNet-18)

3. **If DRIVE results saturate:** Focus analysis on CHASE_DB1 (more challenging)

---

## 8. References

### Foundational Works

[1] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015, pp. 234-241.

[2] O. Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," in *Medical Imaging with Deep Learning (MIDL)*, 2018.

[3] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in *International Conference on Machine Learning (ICML)*, 2016, pp. 1050-1059.

[4] S. K. Warfield, K. H. Zou, and W. M. Wells, "Simultaneous Truth and Performance Level Estimation (STAPLE): An Algorithm for the Validation of Image Segmentation," *IEEE Transactions on Medical Imaging*, vol. 23, no. 7, pp. 903-921, 2004.

### Retinal Vessel Segmentation

[5] J. Staal, M. D. Abràmoff, M. Niemeijer, M. A. Viergever, and B. Van Ginneken, "Ridge-based vessel segmentation in color images of the retina," *IEEE Transactions on Medical Imaging*, vol. 23, no. 4, pp. 501-509, 2004. (DRIVE dataset)

[6] C. G. Owen et al., "Measuring retinal vessel tortuosity in 10-year-old children: validation of the Computer-Assisted Image Analysis of the Retina (CAIAR) program," *Investigative Ophthalmology & Visual Science*, vol. 50, no. 5, pp. 2004-2010, 2009. (CHASE_DB1 dataset)

[7] A. Hoover, V. Kouznetsova, and M. Goldbaum, "Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response," *IEEE Transactions on Medical Imaging*, vol. 19, no. 3, pp. 203-210, 2000. (STARE dataset)

### Uncertainty Quantification

[8] S. Kohl et al., "A Probabilistic U-Net for Segmentation of Ambiguous Images," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018, pp. 6965-6975.

[9] M. Monteiro et al., "Stochastic Segmentation Networks: Modelling Spatially Correlated Aleatoric Uncertainty," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[10] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *International Conference on Machine Learning (ICML)*, 2017, pp. 1321-1330.

### Multi-Annotator Learning

[11] L. Zhang et al., "Disentangling Human Error from Ground Truth in Segmentation of Medical Images," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[12] J. Ji et al., "Learning Calibrated Medical Image Segmentation via Multi-Rater Agreement Modeling," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021.

[13] R. Tanno et al., "Learning From Noisy Labels by Regularized Estimation of Annotator Confusion," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.

### Recent Deep Learning for Medical Imaging

[14] J. Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation," arXiv preprint arXiv:2102.04306, 2021.

[15] F. Isensee et al., "nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation," *Nature Methods*, vol. 18, pp. 203-211, 2021.

[16] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, "UNet++: A Nested U-Net Architecture for Medical Image Segmentation," in *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 2018, pp. 3-11.

### Diabetic Retinopathy and Clinical Context

[17] T. Y. Wong et al., "Diabetic retinopathy in a multi-ethnic cohort in the United States," *American Journal of Ophthalmology*, vol. 141, no. 3, pp. 446-455, 2006.

[18] V. Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs," *JAMA*, vol. 316, no. 22, pp. 2402-2410, 2016.

### Loss Functions and Training

[19] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in *IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 2980-2988.

[20] F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation," in *International Conference on 3D Vision (3DV)*, 2016, pp. 565-571.

---

## Appendix A: Dataset Access

**DRIVE Dataset:**
- URL: https://drive.grand-challenge.org/
- Registration required
- Free for academic use

**CHASE_DB1 Dataset:**
- URL: https://blogs.kingston.ac.uk/retinal/chasedb1/
- Direct download available
- Free for research

---

## Appendix B: Preliminary Code Skeleton

```python
# quaw_net.py - Main Model Definition

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class AQEM(nn.Module):
    """Annotation Quality Estimation Module"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, image, annotation, prediction):
        x = torch.cat([image, annotation, prediction], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        quality = torch.sigmoid(self.fc2(x))
        return quality


class QUAWNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.2):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Decoder with attention and dropout
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(256, 256, 128)
        self.dec4 = self._conv_block(512, 256)
        self.drop4 = nn.Dropout2d(dropout_rate)
        
        # ... (similar for up3, up2)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.aqem = AQEM(in_channels=3)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, annotations=None, return_uncertainty=False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Decoder with attention
        d4 = self.up4(e4)
        d4 = torch.cat([d4, self.att4(d4, e3)], dim=1)
        d4 = self.drop4(self.dec4(d4))
        
        # ... (decoder continues)
        
        output = torch.sigmoid(self.final(d2))
        
        # Compute annotation quality weights if provided
        if annotations is not None:
            weights = []
            for ann in annotations:
                q = self.aqem(x, ann, output.detach())
                weights.append(q)
            weights = torch.softmax(torch.stack(weights), dim=0)
        else:
            weights = None
            
        if return_uncertainty:
            return output, weights, self._mc_uncertainty(x)
        return output, weights
```

---



