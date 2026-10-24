# Project Report — Pneumonia Detection from Chest X-rays

## 1. Introduction
Chest radiography is the most common way to diagnose pneumonia, but reading X-rays takes trained radiologists.
This project builds and evaluates a deep-learning classifier that labels a chest X-ray as Normal or Pneumonia.
It shows the core medical image-analysis workflow: careful pre-processing, transfer learning, class-imbalance
handling and evaluation with clinical metrics.

## 2. Data and analysis
PneumoniaMNIST contains 5,856 paediatric chest X-rays at 128 × 128 resolution, with official train/val/test splits.
- **Imbalance**: 74 % of training images are Pneumonia. A naive model that always says "Pneumonia" would score
  62.5 % accuracy on the test set. This is why we report sensitivity, specificity and AUC rather than accuracy alone.
- **Appearance**: pneumonia shows as hazy opacities in the lung fields. Normal lungs look dark with clearly visible
  rib and vessel structure (see `results/dataset_samples.png`).

## 3. Method summary
| Component | Choice | Reason |
|---|---|---|
| Backbone | ResNet-18, ImageNet-pretrained | Small dataset → transfer learning converges faster and generalises better than training from scratch |
| Input adaptation | Sum RGB conv1 filters → 1 channel | Keeps the pretrained edge/texture filters for grayscale input |
| Augmentation | small affine + brightness/contrast | Mimics patient positioning and exposure variation; no flips (anatomy is asymmetric) |
| Loss | Class-weighted cross-entropy | Corrects the 1:3 imbalance |
| Model selection | Best validation AUC | Threshold-independent, and avoids tuning on the test set |

## 4. Results
| Metric | Test value |
|---|---|
| Accuracy | 0.9567 |
| ROC-AUC | 0.9935 |
| Sensitivity | 0.9923 |
| Specificity | 0.8974 |
| Precision | 0.9416 |
| F1 | 0.9663 |

Validation AUC peaked at 0.9985. The gap to test AUC (0.9935) is small, so the model is not badly overfitted.
Most errors are Normal scans predicted as Pneumonia (24 FP vs 3 FN). Several of these have probabilities just
above 0.5 (0.54, 0.58), so moving the decision threshold could trade some sensitivity for specificity.
Reading the ROC curve: a threshold of 0.83 gives 93 % specificity at 99.0 % sensitivity, and a threshold of 0.96
gives 95 % specificity at 96.7 % sensitivity.

## 5. Limitations and challenges
1. **Low resolution** — 128 × 128 images lose fine texture. Subtle or early pneumonia may not be visible.
2. **Single-centre, paediatric data** — all images come from one hospital and patients aged 1–5. Performance on
   adult X-rays or images from other scanners (domain shift) is unknown and likely lower.
3. **Label granularity** — bacterial and viral pneumonia are merged. Clinically they are treated differently.
4. **Possible shortcuts** — CNNs can learn hospital-specific cues (text markers, positioning) rather than
   pathology. An explainability check (e.g. Grad-CAM, see Project 3) should confirm that the model looks at the lungs.
5. **Threshold choice** — 0.5 is arbitrary. A deployed system would pick the threshold from the clinical cost of
   FN vs FP and would need calibration.
6. **Not a medical device** — no prospective validation was done. Results are for study only.

## 6. Future work
Full-resolution images, adult datasets (e.g. RSNA, CheXpert) for external validation, Grad-CAM heat-maps for
clinician trust, probability calibration, and splitting viral vs bacterial pneumonia.
