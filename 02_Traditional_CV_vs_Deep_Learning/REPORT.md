# Project Report — Traditional Computer Vision vs Deep Learning

## 1. Introduction
The aim is a fair, controlled comparison between the classical recognition pipeline (feature engineering +
shallow classifier) and an end-to-end CNN. Both use the same data and the same test set. Comparing them by
accuracy alone is incomplete, so cost, interpretability and robustness are measured as well.

## 2. Data and analysis
CIFAR-10 is small (32 × 32), balanced and visually varied. Objects appear at different scales and poses against
cluttered backgrounds, and inter-class similarity is high (cat/dog, automobile/truck, deer/horse). These properties
are hard for fixed descriptors: HOG with 8-px cells gives only a 4×4 cell grid per image.

## 3. Key design decisions
| Decision | Reason |
|---|---|
| HOG + HSV colour histogram | HOG alone ignores colour, which matters for frog, ship, airplane (sky/sea backgrounds) |
| Standardisation before SVM | HOG and histogram features have very different scales |
| Linear SVM on all data, RBF SVM on a 10k subset | Kernel SVM memory/time grows roughly with n² |
| No pretrained weights for the CNN | Keeps the comparison fair: both models only see CIFAR-10 |
| Corruptions applied only at test time | Measures out-of-distribution robustness, not what can be learned |

## 4. Results summary
- Accuracy: Linear SVM 56.5 %, RBF SVM 55.9 % (with 5× less data), CNN 92.7 %.
- The RBF SVM matches the linear SVM using only 20 % of the data, so the non-linear kernel extracts more from each sample.
  The bottleneck is the features, not the classifier.
- Under Gaussian noise all models drop to about 35–38 %. Under blur the CNN drops to 29 %, below both SVMs (about 45 %).
- Under brightness/contrast shifts the CNN keeps 81–89 % while the SVMs lose 5–11 points. The fixed HSV
  histogram bins are sensitive to brightness.

## 5. Discussion
Hand-crafted features encode prior knowledge (edges matter, colour matters) but cannot adapt. Cat and dog differ
in fur texture and face shape at a scale finer than an 8 × 8 HOG cell, so the SVM confuses them. The CNN builds a
hierarchy of features from data and captures these details, but it relies on high-frequency information. That
reliance is exactly what blur removes. This trade-off between texture dependence and robustness is a well-known
property of CNNs (Geirhos et al., 2019; Hendrycks & Dietterich, 2019).

## 6. Limitations and challenges
1. **Hardware asymmetry** — the SVMs ran on CPU and the CNN on GPU, so absolute times are not directly comparable.
   Linear-SVM training time (1,140 s) also included contention from other jobs running on the machine at the same time.
2. **Limited hyper-parameter search** — SVM C, the HOG cell size and CNN depth were chosen once from common
   defaults, not tuned by cross-validation. A stronger traditional pipeline (e.g. dense SIFT + Fisher vectors or bag-of-visual-words) would score
   higher, but published results for such methods remain well below modern CNNs.
3. **Low resolution** — 32 × 32 images favour learned features. On larger images with rigid objects (e.g.
   pedestrians, Project 5) HOG does relatively better.
4. **Robustness test scope** — one severity level per corruption. A full CIFAR-10-C style evaluation would use 15 corruptions × 5 severities.
5. **Feature extraction cost** — HOG is computed in Python per image (scikit-image) and the linear SVM is solved on 50k × 836 features. Together they make the traditional
   pipeline slower to train than the GPU CNN. This could be optimised (OpenCV HOG, multiprocessing, SGD-based linear SVM).
6. **Memory** — RBF SVM on all 50k samples was not feasible in reasonable time, so it was trained on a subset.

## 7. Future work
Train the CNN with noise/blur augmentation and re-test robustness, use a pretrained backbone (transfer learning)
as a third family, and combine HOG with CNN features.
