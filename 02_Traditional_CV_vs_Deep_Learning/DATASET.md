# Dataset: CIFAR-10

- **Source**: Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009 — https://www.cs.toronto.edu/~kriz/cifar.html
- **Download**: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz (163 MB)
- **Content**: 60,000 RGB images, 32×32, 10 classes (6,000 each) — 50,000 train / 10,000 test
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

`src/compare.py` downloads the archive into `data/` through `torchvision.datasets.CIFAR10(download=True)`.
The download server can be slow (about 100 KB/s was observed). If it is, place the archive in `data/` by hand.
