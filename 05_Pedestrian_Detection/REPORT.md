# Project Report — Pedestrian Detection for Autonomous Robots

## 1. Introduction
Pedestrian detection is a core perception task for any robot that shares space with people. It is an
**object-detection** problem: the system must say *where* each person is (a bounding box), not only *whether* a
person is present. This project compares a classical detector with a modern deep detector, used both off-the-shelf
and fine-tuned, using standard COCO metrics.

## 2. Data and analysis
Penn-Fudan has 170 campus street images with 423 annotated pedestrians. Most are large and upright (median box
height 285 px), and many are partly occluded by other people. Two properties matter for the evaluation:
- **Incomplete labels**: only the main pedestrians are annotated. Small background people, seated people and
  people cut off at the image border are often not labelled. Hand-inspection of the zero-shot detector's 103
  false positives showed that most are real people.
- **Tight boxes**: boxes are derived from segmentation masks, so they fit the body exactly. Detectors that output
  looser boxes are penalised at high IoU thresholds.

## 3. Methods
| | HOG + SVM | Faster R-CNN zero-shot | Faster R-CNN fine-tuned |
|---|---|---|---|
| Features | Hand-crafted gradient histograms | ResNet-50 + FPN (learned) | same |
| Proposal | Exhaustive multi-scale sliding window | Region Proposal Network | same |
| Training data | INRIA Person (pre-trained in OpenCV) | COCO (80 classes) | COCO → Penn-Fudan 120 images |
| Adaptation to this dataset | Box-shrink factor calibrated on train split | none | 12 epochs fine-tuning |

## 4. Results summary
| | AP@0.5 | AP@[.5:.95] | F1@0.5 | FPS |
|---|---|---|---|---|
| HOG + SVM | 0.470 | 0.176 | 0.516 | 27 (CPU) |
| Faster R-CNN zero-shot | 0.976 | 0.839 | 0.705 | 15 (GPU) |
| Faster R-CNN fine-tuned | 0.993 | 0.856 | 0.925 | 16 (GPU) |

Without box calibration HOG scored only AP@0.5 = 0.23. Its native boxes include the padding of the 64×128 window,
which shows how much detection metrics depend on box conventions.

## 5. Discussion
- Learned features and a proposal network handle occlusion, pose variation and crowded scenes far better than a single
  rigid HOG template (see `results/qualitative_comparison.png`: HOG merges or misses people in the crowded scene).
- Transfer learning is very data-efficient: 120 images and about 4 minutes of training were enough to adapt the
  detector to the annotation style. Most of the gain is in **precision**: the model learns to ignore background people that
  this dataset does not label. For a real robot this is *not* desirable; every person should be detected. This shows that the
  training labels must match the deployment goal.

## 6. Limitations and challenges
1. **Small test set** — 50 images / 123 people. One extra error changes AP by about 1 point. Cross-validation or a larger
   benchmark (Caltech Pedestrian, CityPersons, EuroCity Persons) is needed for firm conclusions.
2. **Incomplete annotations** — penalise detectors that find unlabelled people, so zero-shot precision is underestimated.
3. **Easy domain** — daytime, good weather, large upright pedestrians. Night, rain, small/distant people (the hardest and most
   safety-critical case for a moving robot) and unusual poses (children, wheelchairs) are not represented.
4. **Speed on embedded hardware** — 16 FPS was measured on a laptop RTX 5080. On a robot computer (e.g. Jetson) Faster R-CNN would
   be several times slower. A one-stage detector (YOLO, SSD) or TensorRT optimisation would be needed for real-time control.
5. **Single frames only** — no tracking or temporal smoothing, and no distance estimation. A robot also needs to know
   *how far* each person is (stereo/depth) and where they are moving.
6. **Software change** — OpenCV 5 removed `HOGDescriptor` from the main module, so the project pins OpenCV 4.x.

## 7. Future work
YOLO-family detector for real-time comparison, evaluation on CityPersons (including small/occluded subsets), tracking
(e.g. ByteTrack) for stable IDs, and depth fusion for distance-aware braking.
