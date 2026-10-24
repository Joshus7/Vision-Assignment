"""Pedestrian detection for autonomous robots on the Penn-Fudan dataset.

Three detectors are compared on the same held-out test images:
  A. HOG + linear SVM (Dalal & Triggs, OpenCV's built-in people detector)
  B. Faster R-CNN (ResNet-50 FPN v2) pretrained on COCO, used zero-shot ("person" class)
  C. The same Faster R-CNN fine-tuned on the Penn-Fudan training split
Metrics: COCO-style AP@0.5, AP@[0.5:0.95], AR@100, precision / recall / F1 at a fixed
score threshold, precision-recall curves and throughput (FPS).
"""
import argparse
import json
import random
import time
import zipfile
import urllib.request
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RES = ROOT / "results"
URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
SCORE_THR = 0.5


# ------------------------------------------------------------------ data
def download():
    root = DATA / "PennFudanPed"
    if root.exists():
        return root
    DATA.mkdir(exist_ok=True)
    z = DATA / "PennFudanPed.zip"
    print("downloading Penn-Fudan (~51 MB)...")
    urllib.request.urlretrieve(URL, z)
    with zipfile.ZipFile(z) as zf:
        dest = DATA.resolve()
        for n in zf.namelist():
            if not (dest / n).resolve().is_relative_to(dest):
                raise RuntimeError(f"unsafe archive member {n}")
        zf.extractall(DATA)
    z.unlink()
    return root


def load_item(root, name):
    """Boxes are derived from the per-instance segmentation masks (one id per person)."""
    img = Image.open(root / "PNGImages" / f"{name}.png").convert("RGB")
    mask = np.array(Image.open(root / "PedMasks" / f"{name}_mask.png"))
    boxes = []
    for i in np.unique(mask)[1:]:
        ys, xs = np.where(mask == i)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    return img, torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


class PennFudan(torch.utils.data.Dataset):
    def __init__(self, root, names, train):
        self.root, self.names, self.train = root, names, train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img, boxes = load_item(self.root, self.names[i])
        x = TF.to_tensor(img)
        if self.train:
            if random.random() < 0.5:  # horizontal flip
                x = x.flip(-1)
                w = x.shape[-1]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
            x = TF.adjust_brightness(x, random.uniform(0.75, 1.25))
            x = TF.adjust_contrast(x, random.uniform(0.75, 1.25))
        return x, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


# ------------------------------------------------------------------ detectors
class HOGDetector:
    def __init__(self, shrink=(0.0, 0.0)):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # The 64x128 detection window contains a margin around the person; shrink=(sx, sy)
        # trims that fraction of the box width/height from each side to match tight annotations.
        self.shrink = shrink

    def __call__(self, img):
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        rects, weights = self.hog.detectMultiScale(bgr, winStride=(4, 4), padding=(8, 8), scale=1.05,
                                                   hitThreshold=0.0)
        rects = np.asarray(rects).reshape(-1, 4).astype(np.float32)
        weights = np.asarray(weights, np.float32).reshape(-1)
        if len(rects) == 0:
            return torch.zeros(0, 4), torch.zeros(0)
        sx, sy = self.shrink
        d = rects[:, 2:] * np.array([sx, sy], np.float32)
        boxes = np.c_[rects[:, :2] + d, rects[:, :2] + rects[:, 2:] - d]
        keep = cv2.dnn.NMSBoxes(rects.tolist(), weights.tolist(), score_threshold=0.0, nms_threshold=0.45)
        keep = np.asarray(keep).reshape(-1)
        # Map SVM margins to (0,1) so they can be thresholded and ranked like probabilities.
        scores = 1 / (1 + np.exp(-2 * weights[keep]))
        return torch.from_numpy(boxes[keep]), torch.from_numpy(scores.astype(np.float32))


class TorchDetector:
    def __init__(self, model, person_label, dev):
        self.model, self.label, self.dev = model.eval(), person_label, dev

    @torch.no_grad()
    def __call__(self, img):
        out = self.model([TF.to_tensor(img).to(self.dev)])[0]
        k = out["labels"] == self.label
        return out["boxes"][k].cpu(), out["scores"][k].cpu()


def finetune(train_ds, epochs, dev):
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)  # background + pedestrian
    model.to(dev)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    hist = []
    for ep in range(epochs):
        model.train(); tot = 0
        for xs, ts in loader:
            xs = [x.to(dev) for x in xs]
            ts = [{k: v.to(dev) for k, v in t.items()} for t in ts]
            loss = sum(model(xs, ts).values())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        hist.append(tot / len(loader))
        print(f"  fine-tune epoch {ep+1:2d}: loss={hist[-1]:.4f}")
    return model, hist


# ------------------------------------------------------------------ evaluation
def evaluate(det, root, names):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    all_scores, all_tp, n_gt = [], [], 0
    tp_thr = fp_thr = 0
    outputs = {}
    t_total = 0.0
    for n in names:
        img, gt = load_item(root, n)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t = time.perf_counter()
        boxes, scores = det(img)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_total += time.perf_counter() - t
        outputs[n] = (boxes, scores)
        metric.update([{"boxes": boxes, "scores": scores, "labels": torch.ones(len(boxes), dtype=torch.int64)}],
                      [{"boxes": gt, "labels": torch.ones(len(gt), dtype=torch.int64)}])
        # Greedy matching at IoU 0.5 (highest score first) for PR curve and fixed-threshold metrics.
        n_gt += len(gt)
        order = scores.argsort(descending=True)
        matched = torch.zeros(len(gt), dtype=torch.bool)
        ious = box_iou(boxes[order], gt) if len(boxes) and len(gt) else torch.zeros(len(boxes), len(gt))
        for r, s in enumerate(scores[order]):
            hit = False
            if len(gt):
                iou = ious[r].clone(); iou[matched] = 0
                j = int(iou.argmax())
                if iou[j] >= 0.5:
                    matched[j] = True; hit = True
            all_scores.append(float(s)); all_tp.append(hit)
            if s >= SCORE_THR:
                tp_thr += hit; fp_thr += not hit
    m = {k: float(v) for k, v in metric.compute().items() if k in ("map", "map_50", "map_75", "mar_100", "map_small", "map_medium", "map_large")}
    o = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[o]
    prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    rec = np.cumsum(tp) / n_gt
    p_ = tp_thr / max(tp_thr + fp_thr, 1); r_ = tp_thr / n_gt
    m.update({"precision@0.5": p_, "recall@0.5": r_, "f1@0.5": 2 * p_ * r_ / max(p_ + r_, 1e-9),
              "fps": len(names) / t_total, "n_gt": n_gt})
    return m, (prec, rec), outputs


def draw(img, gt, boxes, scores, title):
    im = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for b in gt.int().tolist():
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 200, 0), 2)
    for b, s in zip(boxes.int().tolist(), scores.tolist()):
        if s < SCORE_THR:
            continue
        cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 2)
        cv2.putText(im, f"{s:.2f}", (b[0], max(b[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n_test", type=int, default=50)
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RES.mkdir(exist_ok=True)
    root = download()
    names = sorted(p.stem for p in (root / "PNGImages").glob("*.png"))
    rng = np.random.default_rng(0); rng.shuffle(names)
    test, train = sorted(names[:args.n_test]), sorted(names[args.n_test:])
    print(f"images: {len(names)}  train={len(train)}  test={len(test)}")
    (RES / "split.json").write_text(json.dumps({"train": train, "test": test}, indent=1))

    # dataset statistics
    heights, counts = [], []
    for n in names:
        _, b = load_item(root, n)
        counts.append(len(b)); heights += (b[:, 3] - b[:, 1]).tolist()
    stats = {"images": len(names), "pedestrians": int(sum(counts)), "mean_per_image": float(np.mean(counts)),
             "box_height_px": {"min": float(min(heights)), "median": float(np.median(heights)), "max": float(max(heights))}}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(heights, bins=30, color="#4c72b0"); ax[0].set_title("Pedestrian box height (px)")
    ax[1].hist(counts, bins=range(1, max(counts) + 2), color="#dd8452", align="left"); ax[1].set_title("Pedestrians per image")
    fig.tight_layout(); fig.savefig(RES / "dataset_stats.png", dpi=110); plt.close(fig)

    t = time.time()
    ft_model, hist = finetune(PennFudan(root, train, True), args.epochs, dev)
    ft_time = time.time() - t
    MODELS.mkdir(exist_ok=True); torch.save(ft_model.state_dict(), MODELS / "fasterrcnn_pennfudan.pt")

    coco = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).to(dev)
    # Calibrate the HOG box margin on the TRAINING split (the test split is never used for tuning).
    raw_hog, cache = HOGDetector(), {}
    for n in train:
        cache[n] = raw_hog(load_item(root, n)[0])
    best = (-1, (0.0, 0.0))
    for sx in (0.0, 0.05, 0.1, 0.15, 0.2):
        for sy in (0.0, 0.03, 0.06, 0.1):
            it = iter(train)
            def shrunk(img, sx=sx, sy=sy):
                b, s = cache[next(it)]
                wh = (b[:, 2:] - b[:, :2]) * torch.tensor([sx, sy])
                return torch.cat([b[:, :2] + wh, b[:, 2:] - wh], 1), s
            ap50 = evaluate(shrunk, root, train)[0]["map_50"]
            if ap50 > best[0]:
                best = (ap50, (sx, sy))
    print(f"HOG margin calibrated on train split: shrink={best[1]} (train AP50={best[0]:.3f})")

    detectors = {
        "HOG + SVM": HOGDetector(shrink=best[1]),
        "Faster R-CNN (COCO, zero-shot)": TorchDetector(coco, 1, dev),
        "Faster R-CNN (fine-tuned)": TorchDetector(ft_model, 1, dev),
    }
    results, curves, outs = {}, {}, {}
    for name, det in detectors.items():
        det(load_item(root, test[0])[0])  # warm-up
        m, c, o = evaluate(det, root, test)
        results[name], curves[name], outs[name] = m, c, o
        print(name, json.dumps({k: round(v, 4) for k, v in m.items()}))
    results["Faster R-CNN (fine-tuned)"]["finetune_time_sec"] = ft_time
    results["HOG + SVM"]["box_shrink_xy"] = list(best[1])

    def rnd(o):
        if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, float): return round(o, 4)
        return o
    (RES / "metrics.json").write_text(json.dumps(rnd({"dataset": stats, "detectors": results,
                                                      "score_threshold": SCORE_THR, "device": dev}), indent=2))

    # ---------------------------------------------- figures
    colors = ["#8c8c8c", "#4c72b0", "#dd8452"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for (n, (p, r)), c in zip(curves.items(), colors):
        ax.plot(r, p, label=f"{n} (AP50={results[n]['map_50']:.3f})", color=c)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall @ IoU 0.5 (test split)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "pr_curves.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for a, (key, title) in zip(axes, [("map_50", "AP@0.5"), ("map", "AP@[0.5:0.95]"), ("fps", "Throughput (FPS)")]):
        vals = [results[n][key] for n in results]
        bars = a.bar(range(3), vals, color=colors)
        a.set_xticks(range(3), [n.replace(" (", "\n(") for n in results], fontsize=8); a.set_title(title)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "metric_comparison.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(hist) + 1), hist, marker="o"); ax.set_xlabel("epoch"); ax.set_ylabel("total loss")
    ax.set_title("Faster R-CNN fine-tuning loss"); fig.tight_layout(); fig.savefig(RES / "training_loss.png", dpi=120); plt.close(fig)

    # qualitative side-by-side on 4 test images, chosen to include crowded scenes
    crowded = sorted(test, key=lambda n: -len(load_item(root, n)[1]))
    picks = crowded[:2] + [test[3], test[17]]
    fig, axes = plt.subplots(len(picks), 3, figsize=(15, 3.3 * len(picks)))
    for r, n in enumerate(picks):
        img, gt = load_item(root, n)
        for c, name in enumerate(detectors):
            b, s = outs[name][n]
            im, t = draw(img, gt, b, s, name)
            axes[r, c].imshow(im); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(t)
    fig.suptitle("Green = ground truth, red = detection (score >= 0.5)")
    fig.tight_layout(); fig.savefig(RES / "qualitative_comparison.png", dpi=80); plt.close(fig)
    for n in picks[:2]:
        img, gt = load_item(root, n)
        b, s = outs["Faster R-CNN (fine-tuned)"][n]
        Image.fromarray(draw(img, gt, b, s, "")[0]).save(RES / f"output_{n}.jpg", quality=90)


if __name__ == "__main__":
    main()
