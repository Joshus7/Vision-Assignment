# Dataset: Penn-Fudan Database for Pedestrian Detection and Segmentation

- **Source**: Liming Wang, Jianbo Shi, Gang Song, I-fan Shen, "Object Detection Combining Recognition and Segmentation", ACCV 2007
- **Page**: https://www.cis.upenn.edu/~jshi/ped_html/
- **Download**: https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip (51 MB)
- **Content**: 170 PNG images (96 Penn, 74 Fudan) with instance masks (`PedMasks/*_mask.png`, one id per person); 423 pedestrian instances
- **Boxes**: derived from the instance masks (min/max x, y of each instance)
- **Split used**: random 120 train / 50 test with seed 0 — the exact file list is saved in `results/split.json`

`src/detect.py` downloads and extracts the zip into `data/` on first run.
