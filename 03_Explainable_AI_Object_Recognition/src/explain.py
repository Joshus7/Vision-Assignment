"""Explainable AI for object recognition.

An ImageNet-pretrained ResNet-50 classifies Imagenette images. Four attribution
methods explain each prediction:
  * Vanilla gradient saliency
  * Grad-CAM
  * Grad-CAM++
  * Occlusion sensitivity (model-agnostic, perturbation based)
Explanations are scored quantitatively with the deletion / insertion metrics
(Petsiuk et al., RISE, 2018), a random-map baseline, a pointing-game-style
energy check, and the Adebayo et al. (2018) model-randomisation sanity check.
"""
import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.stats import spearmanr
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
# Imagenette WordNet id -> (ImageNet-1k index, readable name)
WNIDS = {
    "n01440764": (0, "tench"), "n02102040": (217, "English springer"), "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"), "n03028079": (497, "church"), "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"), "n03425413": (571, "gas pump"), "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def download():
    root = DATA / "imagenette2-160"
    if root.exists():
        return root
    DATA.mkdir(exist_ok=True)
    tgz = DATA / "imagenette2-160.tgz"
    print("downloading Imagenette (~94 MB)...")
    urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as tf:
        dest = DATA.resolve()
        for m in tf.getmembers():  # refuse path traversal / links
            if not (dest / m.name).resolve().is_relative_to(dest) or m.issym() or m.islnk():
                raise RuntimeError(f"unsafe archive member {m.name}")
        tf.extractall(DATA)
    tgz.unlink()
    return root


def load_split(root, split):
    items = []
    for wnid, (idx, name) in WNIDS.items():
        for p in sorted((root / split / wnid).glob("*.JPEG")):
            items.append((p, idx, name))
    return items


PRE = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])  # pixel space [0,1]


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class Explainer:
    def __init__(self, model):
        self.model = model
        self.acts = None
        model.layer4.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        self.acts = o
        if o.requires_grad:
            o.retain_grad()

    def gradients(self, x, cls):
        x = x.clone().requires_grad_(True)
        out = self.model(norm(x))
        self.model.zero_grad()
        out[0, cls].backward()
        return x.grad.detach(), self.acts.detach(), self.acts.grad.detach(), out.detach()

    def saliency(self, x, cls):
        g, *_ = self.gradients(x, cls)
        return g.abs().max(1)[0][0]

    def gradcam(self, x, cls, plus=False):
        _, a, g, out = self.gradients(x, cls)
        if plus:
            # Grad-CAM++ weights (Chattopadhay et al. 2018), with exp(score) gradient trick.
            g2, g3 = g.pow(2), g.pow(3)
            denom = 2 * g2 + (a * g3).sum((2, 3), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(g)).sum((2, 3), keepdim=True)
        else:
            w = g.mean((2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return cam

    @torch.no_grad()
    def occlusion(self, x, cls, patch=32, stride=16):
        H = x.shape[-1]
        base = torch.softmax(self.model(norm(x)), 1)[0, cls]
        pos = list(range(0, H - patch + 1, stride))
        batch, coords = [], []
        for i in pos:
            for j in pos:
                xo = x.clone()
                xo[..., i:i + patch, j:j + patch] = 0.5
                batch.append(xo); coords.append((i, j))
        probs = torch.cat([torch.softmax(self.model(norm(torch.cat(batch[k:k + 64]))), 1)[:, cls]
                           for k in range(0, len(batch), 64)])
        heat = torch.zeros(H, H, device=x.device); cnt = torch.zeros_like(heat)
        for (i, j), p in zip(coords, probs):
            heat[i:i + patch, j:j + patch] += base - p
            cnt[i:i + patch, j:j + patch] += 1
        return F.relu(heat / cnt.clamp(min=1))


def normalize_map(m):
    m = m - m.min()
    return m / (m.max() + 1e-8)


@torch.no_grad()
def deletion_insertion(model, x, cls, sal, steps=20):
    """Return (deletion AUC, insertion AUC) of class probability vs. fraction of pixels."""
    order = torch.argsort(sal.flatten(), descending=True)
    n = order.numel()
    blurred = T.functional.gaussian_blur(x, 31, 10.0)
    dels, inss = [], []
    for s in range(steps + 1):
        k = int(n * s / steps)
        mask = torch.zeros(n, device=x.device); mask[order[:k]] = 1
        mask = mask.view(1, 1, *sal.shape)
        dels.append(x * (1 - mask) + blurred * mask)   # remove most-important pixels first
        inss.append(blurred * (1 - mask) + x * mask)   # reveal most-important pixels first
    pd_ = torch.softmax(model(norm(torch.cat(dels))), 1)[:, cls].cpu().numpy()
    pi_ = torch.softmax(model(norm(torch.cat(inss))), 1)[:, cls].cpu().numpy()
    auc = lambda y: float(np.trapezoid(y, dx=1.0 / steps))
    return auc(pd_), auc(pi_), pd_, pi_


def overlay(img, heat, alpha=0.5):
    cmap = plt.get_cmap("jet")(normalize_map(heat))[..., :3]
    return np.clip((1 - alpha) * img + alpha * cmap, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=300, help="validation images for deletion/insertion")
    args = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RES.mkdir(exist_ok=True)
    root = download()
    val = load_split(root, "val")
    print(f"validation images: {len(val)}")

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ex = Explainer(model)
    imagenet_names = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

    # ---------------------------------------------------- 1. recognition accuracy
    sub_idx = torch.tensor([v[0] for v in WNIDS.values()], device=dev)
    top1, top5, top1_10 = 0, 0, 0
    preds = []
    with torch.no_grad():
        for k in range(0, len(val), 128):
            chunk = val[k:k + 128]
            x = torch.stack([PRE(Image.open(p).convert("RGB")) for p, _, _ in chunk]).to(dev)
            y = torch.tensor([c for _, c, _ in chunk], device=dev)
            out = model(norm(x))
            t5 = out.topk(5, 1).indices
            top1 += (t5[:, 0] == y).sum().item(); top5 += (t5 == y[:, None]).any(1).sum().item()
            p10 = sub_idx[out[:, sub_idx].argmax(1)]
            top1_10 += (p10 == y).sum().item()
            preds += t5[:, 0].tolist()
    acc = {"top1_imagenet1k": top1 / len(val), "top5_imagenet1k": top5 / len(val),
           "top1_restricted_to_10_classes": top1_10 / len(val), "n_images": len(val)}
    print(acc)

    # ---------------------------------------------------- 2. faithfulness evaluation
    methods = ["Saliency", "Grad-CAM", "Grad-CAM++", "Occlusion", "Random"]
    eval_idx = rng.choice(len(val), args.n_eval, replace=False)
    scores = {m: {"del": [], "ins": []} for m in methods}
    curves = {m: {"del": [], "ins": []} for m in methods}
    energy = {m: [] for m in methods}
    for n, i in enumerate(eval_idx):
        p, _, _ = val[i]
        x = PRE(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
        cls = preds[i]  # explain the model's own prediction
        maps = {
            "Saliency": ex.saliency(x, cls),
            "Grad-CAM": ex.gradcam(x, cls),
            "Grad-CAM++": ex.gradcam(x, cls, plus=True),
            "Occlusion": ex.occlusion(x, cls),
            "Random": torch.rand(224, 224, device=dev),
        }
        for m, s in maps.items():
            d, ins, dc, ic = deletion_insertion(model, x, cls, s)
            scores[m]["del"].append(d); scores[m]["ins"].append(ins)
            curves[m]["del"].append(dc); curves[m]["ins"].append(ic)
            # fraction of attribution mass in the central 50% area (objects are usually centred in Imagenette)
            s = normalize_map(s)
            energy[m].append(float(s[56:168, 56:168].sum() / (s.sum() + 1e-8)))
        if (n + 1) % 50 == 0:
            print(f"  evaluated {n+1}/{args.n_eval}: " + ", ".join(
                f"{m} del={np.mean(scores[m]['del']):.3f} ins={np.mean(scores[m]['ins']):.3f}" for m in methods))

    faith = {m: {"deletion_auc_mean": float(np.mean(scores[m]["del"])), "deletion_auc_std": float(np.std(scores[m]["del"])),
                 "insertion_auc_mean": float(np.mean(scores[m]["ins"])), "insertion_auc_std": float(np.std(scores[m]["ins"])),
                 "centre_energy": float(np.mean(energy[m]))} for m in methods}

    # ---------------------------------------------------- 3. sanity check (model randomisation)
    rand_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(dev).eval()
    torch.nn.init.normal_(rand_model.fc.weight, std=0.01)
    for blk in rand_model.layer4:  # cascade-randomise the top block
        for mod in blk.modules():
            if isinstance(mod, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(mod.weight)
    for q in rand_model.parameters():
        q.requires_grad_(False)
    rex = Explainer(rand_model)
    sanity = {"Grad-CAM": [], "Saliency": []}
    for i in eval_idx[:100]:
        x = PRE(Image.open(val[i][0]).convert("RGB")).unsqueeze(0).to(dev)
        cls = preds[i]
        sanity["Grad-CAM"].append(spearmanr(ex.gradcam(x, cls).flatten().cpu(), rex.gradcam(x, cls).flatten().cpu())[0])
        sanity["Saliency"].append(spearmanr(ex.saliency(x, cls).flatten().cpu(), rex.saliency(x, cls).flatten().cpu())[0])
    sanity = {k: float(np.nanmean(v)) for k, v in sanity.items()}

    metrics = {"recognition": acc, "faithfulness": faith,
               "sanity_check_spearman_trained_vs_randomised": sanity, "n_eval_images": args.n_eval}

    def rnd(o):
        if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, float): return round(o, 4)
        return o
    (RES / "metrics.json").write_text(json.dumps(rnd(metrics), indent=2))
    print(json.dumps(rnd(metrics), indent=2))

    # ---------------------------------------------------- figures
    colors = {"Saliency": "#8172b3", "Grad-CAM": "#dd8452", "Grad-CAM++": "#c44e52", "Occlusion": "#55a868", "Random": "#8c8c8c"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    xs = np.linspace(0, 1, 21)
    for m in methods:
        ax[0].plot(xs, np.mean(curves[m]["del"], 0), label=f"{m} (AUC {faith[m]['deletion_auc_mean']:.3f})", color=colors[m])
        ax[1].plot(xs, np.mean(curves[m]["ins"], 0), label=f"{m} (AUC {faith[m]['insertion_auc_mean']:.3f})", color=colors[m])
    ax[0].set_title("Deletion curve (lower AUC = more faithful)"); ax[1].set_title("Insertion curve (higher AUC = more faithful)")
    for a in ax:
        a.set_xlabel("fraction of pixels removed / inserted"); a.set_ylabel("class probability"); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "deletion_insertion_curves.png", dpi=120); plt.close(fig)

    def gallery(indices, fname, title):
        fig, axes = plt.subplots(len(indices), 5, figsize=(15, 3.1 * len(indices)))
        for r, i in enumerate(indices):
            p, gt, gtname = val[i]
            x = PRE(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            cls = preds[i]
            img = x[0].permute(1, 2, 0).cpu().numpy()
            maps = [ex.saliency(x, cls), ex.gradcam(x, cls), ex.gradcam(x, cls, plus=True), ex.occlusion(x, cls)]
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"GT: {gtname}\nPred: {imagenet_names[cls]}", fontsize=9,
                                 color="green" if cls == gt else "red")
            for c, (m, h) in enumerate(zip(["Saliency", "Grad-CAM", "Grad-CAM++", "Occlusion"], maps), 1):
                h = h.cpu().numpy()
                axes[r, c].imshow(normalize_map(h), cmap="hot") if m == "Saliency" else axes[r, c].imshow(overlay(img, h))
                if r == 0:
                    axes[r, c].set_title(m)
            for a in axes[r]:
                a.axis("off")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(RES / fname, dpi=100); plt.close(fig)

    labels = np.array([c for _, c, _ in val]); preds_np = np.array(preds)
    per_class = [rng.choice(np.where((labels == c) & (preds_np == c))[0]) for c, _ in list(WNIDS.values())[:6]]
    gallery(per_class, "explanations_correct.png", "Explanations for correct predictions")
    wrong = np.where(preds_np != labels)[0]
    gallery(rng.choice(wrong, min(4, len(wrong)), replace=False), "explanations_failures.png",
            "Explanations for misclassified images (failure analysis)")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(list(sanity), list(sanity.values()), color=["#dd8452", "#8172b3"])
    ax.set_ylabel("Spearman rank correlation"); ax.set_ylim(-0.1, 1)
    ax.set_title("Sanity check: trained vs. randomised layer4\n(low = explanation depends on learned weights)")
    fig.tight_layout(); fig.savefig(RES / "sanity_check.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
