"""Traditional computer vision (hand-crafted features + SVM) vs deep learning (CNN)
on CIFAR-10.

Pipelines compared on the same 10,000-image test set:
  A. HOG + colour histogram features -> Linear SVM (all 50k training images)
  B. HOG + colour histogram features -> RBF-kernel SVM (10k training subset)
  C. Small VGG-style CNN trained from scratch (all 50k training images)

Besides accuracy we measure training time, per-image inference time and robustness
to Gaussian noise, blur and brightness shifts. Results go to ../results/.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RES = ROOT / "results"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


# ---------------------------------------------------------------- traditional
def extract_features(imgs):
    """HOG on the grayscale image (shape/edges) + 8x8x8 HSV histogram (colour)."""
    feats = []
    for im in imgs:
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        h = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        ch = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).ravel()
        ch /= ch.sum() + 1e-6
        feats.append(np.concatenate([h, ch]))
    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------- deep learning
class SmallVGG(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.features(x))


def to_tensor(imgs):
    x = (imgs.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def augment(x):
    """Random crop (pad 4) + horizontal flip, on GPU."""
    n = x.size(0)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = np.random.randint(0, 9, 2)
    x = x[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(n, device=x.device) < 0.5
    x[flip] = x[flip].flip(3)
    return x


def train_cnn(xtr, ytr, epochs, dev):
    model = SmallVGG().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    bs = 128
    steps = (len(xtr) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, epochs=epochs, steps_per_epoch=steps)
    xtr, ytr = xtr.to(dev), torch.from_numpy(ytr).to(dev)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        tot, correct = 0.0, 0
        for k in range(steps):
            idx = perm[k * bs:(k + 1) * bs]
            x, y = augment(xtr[idx]), ytr[idx]
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx); correct += (out.argmax(1) == y).sum().item()
        hist.append((tot / len(xtr), correct / len(xtr)))
        print(f"  CNN epoch {ep+1:2d}: loss={hist[-1][0]:.4f} train_acc={hist[-1][1]:.4f}")
    return model, hist


@torch.no_grad()
def cnn_predict(model, x, dev):
    model.eval()
    return torch.cat([model(x[i:i + 1000].to(dev)).argmax(1).cpu() for i in range(0, len(x), 1000)]).numpy()


# ---------------------------------------------------------------- corruptions
def corrupt(imgs, kind, rng):
    if kind == "clean":
        return imgs
    out = imgs.astype(np.float32)
    if kind == "gaussian_noise":
        out = out + rng.normal(0, 20, out.shape)
    elif kind == "blur":
        out = np.stack([cv2.GaussianBlur(im, (5, 5), 1.2) for im in out])
    elif kind == "brightness":
        out = out * 1.0 + 60
    elif kind == "low_contrast":
        out = (out - out.mean(axis=(1, 2, 3), keepdims=True)) * 0.4 + out.mean(axis=(1, 2, 3), keepdims=True)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--rbf_subset", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(0); np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RES.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)

    tr, te = CIFAR10(DATA, train=True, download=True), CIFAR10(DATA, train=False, download=True)
    xtr, ytr = tr.data, np.array(tr.targets)
    xte, yte = te.data, np.array(te.targets)
    print(f"train {xtr.shape}  test {xte.shape}")

    results, preds = {}, {}

    # --- A/B: hand-crafted features
    t = time.time(); ftr = extract_features(xtr); fte = extract_features(xte)
    feat_time = time.time() - t
    print(f"feature dim={ftr.shape[1]}  extraction {feat_time:.1f}s")

    t = time.time()
    lin = make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=5000, dual=False)).fit(ftr, ytr)
    lin_train = time.time() - t
    t = time.time(); p = lin.predict(fte); lin_inf = (time.time() - t) / len(fte)
    preds["HOG+Color / Linear SVM"] = p
    results["HOG+Color / Linear SVM"] = dict(train_images=len(xtr), train_time_sec=lin_train + feat_time * len(xtr) / (len(xtr) + len(xte)),
                                              infer_ms_per_img=1000 * (lin_inf + feat_time / (len(xtr) + len(xte))))

    sub = rng.choice(len(xtr), args.rbf_subset, replace=False)
    t = time.time()
    rbf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(ftr[sub], ytr[sub])
    rbf_train = time.time() - t
    t = time.time(); p = rbf.predict(fte); rbf_inf = (time.time() - t) / len(fte)
    preds["HOG+Color / RBF SVM"] = p
    results["HOG+Color / RBF SVM"] = dict(train_images=int(args.rbf_subset), train_time_sec=rbf_train + feat_time * args.rbf_subset / (len(xtr) + len(xte)),
                                           infer_ms_per_img=1000 * (rbf_inf + feat_time / (len(xtr) + len(xte))))

    # --- C: CNN
    t = time.time(); cnn, hist = train_cnn(to_tensor(xtr), ytr, args.epochs, dev); cnn_train = time.time() - t
    xte_t = to_tensor(xte)
    cnn_predict(cnn, xte_t[:1000], dev)  # warm-up
    if dev == "cuda": torch.cuda.synchronize()
    t = time.time(); p = cnn_predict(cnn, xte_t, dev)
    if dev == "cuda": torch.cuda.synchronize()
    cnn_inf = (time.time() - t) / len(xte)
    preds["CNN (SmallVGG)"] = p
    results["CNN (SmallVGG)"] = dict(train_images=len(xtr), train_time_sec=cnn_train, infer_ms_per_img=1000 * cnn_inf,
                                     params=sum(q.numel() for q in cnn.parameters()), device=dev)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(lin, MODELS / "hog_color_linear_svm.joblib")
    joblib.dump(rbf, MODELS / "hog_color_rbf_svm.joblib")
    torch.save(cnn.state_dict(), MODELS / "small_vgg_cifar10.pt")

    for name, p in preds.items():
        results[name]["accuracy"] = accuracy_score(yte, p)
        results[name]["macro_f1"] = f1_score(yte, p, average="macro")
        results[name]["per_class_accuracy"] = {c: float((p[yte == i] == i).mean()) for i, c in enumerate(CLASSES)}

    # --- robustness
    kinds = ["clean", "gaussian_noise", "blur", "brightness", "low_contrast"]
    robust = {n: {} for n in preds}
    for kind in kinds:
        xc = corrupt(xte, kind, rng)
        fc = extract_features(xc)
        robust["HOG+Color / Linear SVM"][kind] = accuracy_score(yte, lin.predict(fc))
        robust["HOG+Color / RBF SVM"][kind] = accuracy_score(yte, rbf.predict(fc))
        robust["CNN (SmallVGG)"][kind] = accuracy_score(yte, cnn_predict(cnn, to_tensor(xc), dev))
        print(f"robustness {kind}: " + ", ".join(f"{n}={robust[n][kind]:.3f}" for n in robust))
        if kind != "clean":
            cv2.imwrite(str(RES / f"corruption_{kind}.png"),
                        cv2.cvtColor(np.hstack([cv2.resize(i, (96, 96), interpolation=cv2.INTER_NEAREST) for i in xc[:8]]), cv2.COLOR_RGB2BGR))
    for n in results:
        results[n]["robustness_accuracy"] = robust[n]

    def rnd(o):
        if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, (float, np.floating)): return round(float(o), 4)
        return o
    results = rnd(results)
    (RES / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({n: {k: v for k, v in r.items() if k not in ("per_class_accuracy",)} for n, r in results.items()}, indent=2))

    # ---------------------------------------------------- figures
    names = list(results)
    colors = ["#8c8c8c", "#4c72b0", "#dd8452"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, key, title in [(ax[0], "accuracy", "Test accuracy"), (ax[1], "train_time_sec", "Training time (s)"),
                          (ax[2], "infer_ms_per_img", "Inference time (ms / image)")]:
        vals = [results[n][key] for n in names]
        bars = a.bar(range(3), vals, color=colors); a.set_xticks(range(3), [n.replace(" / ", "\n") for n in names], fontsize=8)
        a.set_title(title)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "comparison_bars.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.27
    for k, n in enumerate(names):
        ax.bar(np.arange(10) + (k - 1) * w, [results[n]["per_class_accuracy"][c] for c in CLASSES], w, label=n, color=colors[k])
    ax.set_xticks(range(10), CLASSES, rotation=30); ax.set_ylabel("accuracy"); ax.set_title("Per-class test accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(RES / "per_class_accuracy.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, n in enumerate(names):
        ax.plot(kinds, [robust[n][c] for c in kinds], marker="o", label=n, color=colors[k])
    ax.set_ylabel("accuracy"); ax.set_title("Robustness to image corruptions"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "robustness.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for a, n in zip(axes, ["HOG+Color / Linear SVM", "CNN (SmallVGG)"]):
        cm = confusion_matrix(yte, preds[n], normalize="true")
        a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(10), CLASSES, rotation=60, fontsize=8); a.set_yticks(range(10), CLASSES, fontsize=8)
        for i in range(10):
            for j in range(10):
                a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if cm[i, j] > 0.5 else "black")
        a.set_title(n); a.set_xlabel("Predicted"); a.set_ylabel("True")
    fig.tight_layout(); fig.savefig(RES / "confusion_matrices.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h[0] for h in hist], label="train loss"); ax2 = ax.twinx()
    ax2.plot([h[1] for h in hist], color="#dd8452", label="train acc"); ax.set_xlabel("epoch")
    ax.set_ylabel("loss"); ax2.set_ylabel("accuracy"); ax.set_title("CNN training curve")
    fig.tight_layout(); fig.savefig(RES / "cnn_training_curve.png", dpi=120); plt.close(fig)

    # HOG visualisation: what the traditional pipeline "sees"
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for k in range(6):
        im = xte[k]
        _, vis = hog(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), orientations=9, pixels_per_cell=(4, 4),
                     cells_per_block=(2, 2), visualize=True)
        axes[0, k].imshow(im); axes[0, k].set_title(CLASSES[yte[k]], fontsize=9); axes[0, k].axis("off")
        axes[1, k].imshow(vis, cmap="gray"); axes[1, k].set_title("HOG", fontsize=9); axes[1, k].axis("off")
    fig.suptitle("Input images and their HOG descriptors"); fig.tight_layout()
    fig.savefig(RES / "hog_visualization.png", dpi=110); plt.close(fig)

    # Sample predictions from both families
    idx = rng.choice(len(xte), 12, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(15, 6.2))
    for a, i in zip(axes.ravel(), idx):
        a.imshow(xte[i]); a.axis("off")
        s, c = CLASSES[preds["HOG+Color / Linear SVM"][i]], CLASSES[preds["CNN (SmallVGG)"][i]]
        a.set_title(f"GT: {CLASSES[yte[i]]}\nSVM: {s}\nCNN: {c}", fontsize=8,
                    color="green" if c == CLASSES[yte[i]] else "red")
    fig.suptitle("Sample predictions (title green when CNN is correct)"); fig.tight_layout()
    fig.savefig(RES / "sample_predictions.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
