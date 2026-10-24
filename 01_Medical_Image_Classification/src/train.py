"""Pneumonia detection from chest X-rays (PneumoniaMNIST, 128x128).

Fine-tunes an ImageNet-pretrained ResNet-18 and evaluates it on the official
held-out test split. All figures and metrics are written to ../results/.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from medmnist import PneumoniaMNIST
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RES = ROOT / "results"
CLASSES = ["Normal", "Pneumonia"]


def get_loaders(size, batch):
    DATA.mkdir(exist_ok=True)
    mean, std = [0.5], [0.5]
    train_tf = T.Compose([
        T.ToTensor(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    kw = dict(size=size, root=str(DATA), download=True)
    tr = PneumoniaMNIST(split="train", transform=train_tf, **kw)
    va = PneumoniaMNIST(split="val", transform=eval_tf, **kw)
    te = PneumoniaMNIST(split="test", transform=eval_tf, **kw)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch, shuffle=sh, num_workers=0, pin_memory=True)
    return tr, mk(tr, True), mk(va, False), mk(te, False), te


def build_model():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Grayscale input: collapse the RGB filters of the first conv into one channel.
    w = m.conv1.weight.data.sum(dim=1, keepdim=True)
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(dev)), 1)[:, 1]
        probs.append(p.cpu()); ys.append(y.squeeze(1))
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RES.mkdir(exist_ok=True)

    tr_ds, tr, va, te, te_ds = get_loaders(args.size, args.batch)
    labels = tr_ds.labels.squeeze()
    counts = np.bincount(labels)
    print(f"train={len(tr_ds)} class counts={counts.tolist()} device={dev}")
    # Inverse-frequency class weights to counter the ~1:3 Normal:Pneumonia imbalance.
    cw = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=dev)

    model = build_model().to(dev)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs, steps_per_epoch=len(tr))

    hist = {"train_loss": [], "val_auc": [], "val_acc": []}
    best_auc, best_state = -1, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0
        for x, y in tr:
            x, y = x.to(dev), y.squeeze(1).long().to(dev)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(x)
        p, y = predict(model, va, dev)
        auc, acc = roc_auc_score(y, p), accuracy_score(y, p > 0.5)
        hist["train_loss"].append(tot / len(tr_ds)); hist["val_auc"].append(auc); hist["val_acc"].append(acc)
        print(f"epoch {ep+1:2d} loss={hist['train_loss'][-1]:.4f} val_auc={auc:.4f} val_acc={acc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    MODELS.mkdir(exist_ok=True); torch.save(best_state, MODELS / "resnet18_pneumonia.pt")

    p, y = predict(model, te, dev)
    pred = (p > 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "test_samples": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall_sensitivity": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "confusion_matrix": cm.tolist(),
        "best_val_auc": best_auc,
        "train_time_sec": train_time,
        "epochs": args.epochs, "image_size": args.size,
    }
    metrics = {k: (round(float(v), 4) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()}
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RES / "classification_report.txt").write_text(classification_report(y, pred, target_names=CLASSES, digits=4))
    print(json.dumps(metrics, indent=2))

    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(1, args.epochs + 1), hist["train_loss"], marker="o"); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(range(1, args.epochs + 1), hist["val_auc"], marker="o", label="val AUC")
    ax[1].plot(range(1, args.epochs + 1), hist["val_acc"], marker="s", label="val accuracy")
    ax[1].set_title("Validation metrics"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(RES / "training_curves.png", dpi=120); plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASSES); ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Test confusion matrix")
    fig.tight_layout(); fig.savefig(RES / "confusion_matrix.png", dpi=120); plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"ResNet-18 (AUC={metrics['roc_auc']:.3f})"); ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC - test set"); ax.legend()
    fig.tight_layout(); fig.savefig(RES / "roc_curve.png", dpi=120); plt.close(fig)

    # Sample input/output grid: 6 correct + 6 misclassified
    imgs = te_ds.imgs
    rng = np.random.default_rng(0)
    correct = rng.permutation(np.where(pred == y)[0])[:6]
    wrong = rng.permutation(np.where(pred != y)[0])[:6]
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.8))
    for row, idxs, tag in [(0, correct, "correct"), (1, wrong, "wrong")]:
        for k, ax in enumerate(axes[row]):
            ax.axis("off")
            if k >= len(idxs):
                continue
            i = idxs[k]
            ax.imshow(imgs[i], cmap="gray")
            ax.set_title(f"GT: {CLASSES[y[i]]}\nPred: {CLASSES[pred[i]]} (p={p[i]:.2f})",
                         fontsize=9, color="green" if tag == "correct" else "red")
    fig.suptitle("Sample test predictions (top: correct, bottom: misclassified)")
    fig.tight_layout(); fig.savefig(RES / "sample_predictions.png", dpi=110); plt.close(fig)

    # Dataset overview
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for c in range(2):
        idx = np.where(labels == c)[0][:5]
        for k, ax in enumerate(axes[c]):
            ax.imshow(tr_ds.imgs[idx[k]], cmap="gray"); ax.axis("off"); ax.set_title(CLASSES[c], fontsize=9)
    fig.suptitle("PneumoniaMNIST training samples"); fig.tight_layout()
    fig.savefig(RES / "dataset_samples.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
