# Vision-Assignment — 5 Computer Vision Projects

Five end-to-end computer-vision projects. Each one covers a **different CV concept** and a different kind of
model. Every project has its own folder containing source code, dataset information, `requirements.txt`, results,
performance evaluation, visualisations and a short report.

| # | Project | CV concept | Method | Dataset | Headline result |
|---|---|---|---|---|---|
| 1 | [Medical Image Classification and Analysis](01_Medical_Image_Classification/) | Image **classification**, transfer learning, class imbalance | ResNet-18 (ImageNet → X-ray) | PneumoniaMNIST (5,856 chest X-rays) | **95.7 %** acc, **0.994** AUC, 99.2 % sensitivity |
| 2 | [Traditional CV vs Deep Learning](02_Traditional_CV_vs_Deep_Learning/) | **Hand-crafted features** vs **learned features**, robustness | HOG + colour histogram → SVM vs CNN from scratch | CIFAR-10 (60,000 images) | SVM **56.5 %** vs CNN **92.7 %**; CNN less robust to blur/noise |
| 3 | [Explainable AI for Object Recognition](03_Explainable_AI_Object_Recognition/) | **Explainability**, attribution maps, faithfulness evaluation | ResNet-50 + Saliency, Grad-CAM, Grad-CAM++, Occlusion | Imagenette (3,925 val images) | 99.5 % acc; Grad-CAM most faithful (deletion AUC 0.108 vs random 0.145) |
| 4 | [Road Lane Detection and Departure Warning](04_Road_Lane_Detection/) | **Classical image processing**: colour spaces, edges, Hough transform, tracking | HLS masking + Canny + Hough + EMA tracking | Udacity lane-line images and videos | Offset error **1.3 cm**, warning precision/recall **98 %**, 50–90 FPS on CPU |
| 5 | [Pedestrian Detection for Autonomous Robots](05_Pedestrian_Detection/) | **Object detection**, sliding window vs region proposals, fine-tuning | HOG + SVM vs Faster R-CNN (zero-shot / fine-tuned) | Penn-Fudan (170 images, 423 people) | AP@0.5: HOG **0.47** → Faster R-CNN fine-tuned **0.99** |

## Repository structure
```
Vision-Assignment/
├── README.md                ← this file
├── requirements.txt         ← union of all project requirements
├── make_gif.py              ← helper: video → GIF preview (used in Project 4)
├── 01_Medical_Image_Classification/
│   ├── README.md            ← title, problem, objective, dataset, method, tools, results, conclusion
│   ├── REPORT.md            ← brief project report + limitations
│   ├── DATASET.md           ← dataset source / link / licence
│   ├── requirements.txt
│   ├── src/                 ← source code
│   ├── models/              ← trained weights (Projects 1, 2, 5)
│   └── results/             ← metrics.json, plots, sample input/output images
├── 02_Traditional_CV_vs_Deep_Learning/   (same layout)
├── 03_Explainable_AI_Object_Recognition/ (same layout)
├── 04_Road_Lane_Detection/               (same layout, + output videos/GIFs)
└── 05_Pedestrian_Detection/              (same layout)
```

## Quick start
```bash
git clone https://github.com/Joshus7/Vision-Assignment.git
cd Vision-Assignment
python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt                     # for a CUDA build of PyTorch see https://pytorch.org

cd 01_Medical_Image_Classification && python src/train.py
cd ../02_Traditional_CV_vs_Deep_Learning && python src/compare.py
cd ../03_Explainable_AI_Object_Recognition && python src/explain.py
cd ../04_Road_Lane_Detection && python src/lane_detection.py
cd ../05_Pedestrian_Detection && python src/detect.py
```
Every script **downloads its own dataset** into the project's `data/` folder (git-ignored) and writes all metrics and
figures into `results/`. Trained weights are saved in each project's `models/` folder (Projects 1, 2 and 5; Project 3 uses
ImageNet-pretrained weights as-is and Project 4 is classical, so neither has trained weights). The committed `results/` folders contain the outputs of the runs reported in the READMEs.

## Environment used for the reported results
- Windows 11, Python 3.10.11
- PyTorch 2.7.1 (CUDA 11.8), torchvision 0.22.1, NVIDIA GeForce RTX 5080 Laptop GPU (16 GB)
- OpenCV 4.14 (Projects 2, 4 and 5 also run with 4.8+; Project 5 needs < 5 for `HOGDescriptor`), scikit-learn 1.7, scikit-image 0.25, medmnist 3.0, torchmetrics 1.9

All projects also run on CPU only, more slowly (Projects 1, 2, 3 and 5 take longer; Project 4 is CPU-only anyway).

## Concepts covered across the five projects
- Pre-processing: normalisation, colour-space conversion (HLS/HSV), augmentation, class weighting
- Classical CV: HOG, colour histograms, Canny, Sobel, Hough transform, morphology, ROI masking, sliding windows, NMS
- Deep learning: CNN training from scratch, transfer learning, fine-tuning a two-stage detector
- Evaluation: accuracy/precision/recall/F1/specificity, ROC-AUC, confusion matrices, COCO AP, PR curves,
  deletion/insertion faithfulness, robustness to corruptions, FPS, controlled ground-truth experiments
- Critical analysis: limitations, failure cases and future work in every `REPORT.md`
