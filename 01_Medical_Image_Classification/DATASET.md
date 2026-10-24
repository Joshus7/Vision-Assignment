# Dataset: PneumoniaMNIST (MedMNIST v2)

- **Source**: MedMNIST — https://medmnist.com/ (Yang et al., *Scientific Data* 2023)
- **Direct download** (128×128 version): https://zenodo.org/records/10519652/files/pneumoniamnist_128.npz
- **Original data**: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", *Cell* 2018 — https://data.mendeley.com/datasets/rscbjbr9sj/2
- **Licence**: CC BY 4.0
- **Task**: binary classification, 0 = Normal, 1 = Pneumonia
- **Size**: 5,856 grayscale chest X-rays (train 4,708 / val 524 / test 624)

`src/train.py` downloads the file into `data/` the first time it runs. Nothing has to be fetched by hand.
The data folder is git-ignored to keep the repository small.
