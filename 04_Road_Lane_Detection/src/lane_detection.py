"""Road lane detection and lane-departure warning with classical computer vision.

Two pipelines are compared:
  * baseline  : grayscale -> Gaussian blur -> Canny -> ROI -> Hough -> line averaging
  * improved  : HLS colour mask (white + yellow) + Sobel-x gradient -> ROI -> Hough
                -> slope/position filtering -> robust fit -> temporal smoothing (EMA)
Departure warning uses the vehicle's lateral offset from the lane centre (camera
assumed at the image centre, lane width 3.7 m).
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
BASE = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/"
IMAGES = ["solidWhiteCurve.jpg", "solidWhiteRight.jpg", "solidYellowCurve.jpg",
          "solidYellowCurve2.jpg", "solidYellowLeft.jpg", "whiteCarLaneSwitch.jpg"]
VIDEOS = ["solidWhiteRight.mp4", "solidYellowLeft.mp4", "challenge.mp4"]
LANE_WIDTH_M = 3.7
WARN_M = 0.5  # warn when the car is more than 0.5 m from the lane centre


def download():
    for sub, names in [("test_images", IMAGES), ("test_videos", VIDEOS)]:
        (DATA / sub).mkdir(parents=True, exist_ok=True)
        for n in names:
            dst = DATA / sub / n
            if not dst.exists():
                print("downloading", n)
                urllib.request.urlretrieve(BASE + f"{sub}/{n}", dst)


def roi_mask(shape):
    h, w = shape[:2]
    poly = np.array([[(0, h), (0, 0.88 * h), (0.45 * w, 0.61 * h), (0.55 * w, 0.61 * h), (w, 0.88 * h), (w, h)]], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, poly, 255)
    return m, poly


# ------------------------------------------------------------------ edge maps
def edges_baseline(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.Canny(g, 50, 150)


def edges_improved(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L, S = hls[..., 1], hls[..., 2]
    # Adaptive white threshold: top of the lightness distribution inside the frame.
    white = cv2.inRange(hls, (0, max(190, int(np.percentile(L, 97))), 0), (180, 255, 255))
    yellow = cv2.inRange(hls, (15, 80, 90), (35, 220, 255))
    color = cv2.bitwise_or(white, yellow)
    sx = np.abs(cv2.Sobel(cv2.GaussianBlur(L, (5, 5), 0), cv2.CV_64F, 1, 0, ksize=3))
    sx = np.uint8(255 * sx / (sx.max() + 1e-6))
    grad = ((sx > 40) & (S > 60)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(color, grad)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.Canny(mask, 50, 150)


# ------------------------------------------------------------------ line fitting
def fit_lanes(edges, shape, strict):
    h, w = shape[:2]
    m, _ = roi_mask(shape)
    e = cv2.bitwise_and(edges, m)
    segs = cv2.HoughLinesP(e, 1, np.pi / 180, 20, minLineLength=15, maxLineGap=25)
    left, right = [], []
    if segs is not None:
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            if x2 == x1:
                continue
            s = (y2 - y1) / (x2 - x1)
            L = np.hypot(x2 - x1, y2 - y1)
            if strict:
                # Reject near-horizontal / near-vertical segments and those on the wrong side.
                if not 0.45 < abs(s) < 2.0:
                    continue
                if s < 0 and max(x1, x2) > 0.55 * w or s > 0 and min(x1, x2) < 0.45 * w:
                    continue
            elif abs(s) < 0.3:
                continue
            (left if s < 0 else right).append((x1, y1, x2, y2, L))
    y_bot, y_top = h, int(0.63 * h)
    lanes = []
    for group in (left, right):
        if not group:
            lanes.append(None); continue
        g = np.array(group, float)
        if strict:
            # Length-weighted least squares x = a*y + b over all segment endpoints.
            ys = np.r_[g[:, 1], g[:, 3]]; xs = np.r_[g[:, 0], g[:, 2]]; wts = np.r_[g[:, 4], g[:, 4]]
            a, b = np.polyfit(ys, xs, 1, w=wts)
        else:
            s = np.mean((g[:, 3] - g[:, 1]) / (g[:, 2] - g[:, 0]))
            icpt = np.mean(g[:, 1] - s * g[:, 0])
            a, b = 1 / s, -icpt / s
        lanes.append(np.array([a * y_bot + b, y_bot, a * y_top + b, y_top]))
    return lanes, e


class LaneTracker:
    """Exponential moving average over frames; holds the last estimate for a few missing frames."""

    def __init__(self, alpha=0.2, max_miss=10):
        self.alpha, self.max_miss = alpha, max_miss
        self.state, self.miss = [None, None], [0, 0]

    def update(self, lanes, w):
        for k, ln in enumerate(lanes):
            prev = self.state[k]
            # Reject jumps of more than 12% of image width at the bottom (outliers).
            if ln is not None and prev is not None and abs(ln[0] - prev[0]) > 0.12 * w:
                ln = None
            if ln is None:
                self.miss[k] += 1
                if self.miss[k] > self.max_miss:
                    self.state[k] = None
            else:
                self.miss[k] = 0
                self.state[k] = ln if prev is None else (1 - self.alpha) * prev + self.alpha * ln
        return list(self.state)


def departure(lanes, w):
    if lanes[0] is None or lanes[1] is None:
        return None, None, "NO LANE"
    xl, xr = lanes[0][0], lanes[1][0]
    lane_px = xr - xl
    if lane_px <= 0:
        return None, None, "NO LANE"
    offset_m = (w / 2 - (xl + xr) / 2) * LANE_WIDTH_M / lane_px  # + = car right of centre
    status = "OK" if abs(offset_m) < WARN_M else ("DEPARTING RIGHT" if offset_m > 0 else "DEPARTING LEFT")
    return offset_m, lane_px, status


def draw(img, lanes, status, offset):
    out = img.copy(); layer = np.zeros_like(img)
    if lanes[0] is not None and lanes[1] is not None:
        l, r = lanes
        pts = np.array([[l[0], l[1]], [l[2], l[3]], [r[2], r[3]], [r[0], r[1]]], np.int32)
        cv2.fillPoly(layer, [pts], (0, 180, 0) if status == "OK" else (0, 0, 220))
    for ln in lanes:
        if ln is not None:
            cv2.line(layer, (int(ln[0]), int(ln[1])), (int(ln[2]), int(ln[3])), (0, 0, 255), 10)
    out = cv2.addWeighted(out, 1, layer, 0.45, 0)
    color = (0, 200, 0) if status == "OK" else (0, 0, 255)
    cv2.rectangle(out, (10, 10), (420, 80), (0, 0, 0), -1)
    cv2.putText(out, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if offset is not None:
        cv2.putText(out, f"Offset: {offset:+.2f} m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def process_video(path, pipeline, writer_path=None, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    tracker = LaneTracker() if pipeline == "improved" else None
    writer, stats = None, {"frames": 0, "both": 0, "offsets": [], "bottom_x": [], "widths": [], "warn": 0, "t": 0.0}
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and stats["frames"] >= max_frames):
            break
        h, w = frame.shape[:2]
        t = time.perf_counter()
        if pipeline == "baseline":
            lanes, _ = fit_lanes(edges_baseline(frame), frame.shape, strict=False)
        else:
            lanes, _ = fit_lanes(edges_improved(frame), frame.shape, strict=True)
            lanes = tracker.update(lanes, w)
        off, width, status = departure(lanes, w)
        stats["t"] += time.perf_counter() - t
        stats["frames"] += 1
        if off is not None:
            stats["both"] += 1; stats["offsets"].append(off); stats["widths"].append(width / w)
            stats["warn"] += status != "OK"
        stats["bottom_x"].append([np.nan if l is None else l[0] / w for l in lanes])
        if writer_path:
            if writer is None:
                writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
            writer.write(draw(frame, lanes, status, off))
    cap.release()
    if writer:
        writer.release()
    bx = np.array(stats["bottom_x"])
    jitter = np.nanmean(np.abs(np.diff(bx, axis=0))) * 100  # % of image width per frame
    widths = np.array(stats["widths"])
    return {
        "frames": stats["frames"],
        "both_lanes_detected_rate": stats["both"] / stats["frames"],
        "mean_frame_jitter_pct_width": float(jitter),
        "lane_width_cv": float(widths.std() / widths.mean()) if len(widths) else None,
        "mean_abs_offset_m": float(np.mean(np.abs(stats["offsets"]))) if stats["offsets"] else None,
        "warning_frame_rate": stats["warn"] / max(stats["both"], 1),
        "fps": stats["frames"] / stats["t"],
    }, bx


def vanishing_row(lanes):
    """Image row where the left and right lane lines intersect."""
    (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = lanes
    al, ar = (lx2 - lx1) / (ly2 - ly1), (rx2 - rx1) / (ry2 - ry1)  # dx/dy
    return ly1 + (rx1 - lx1) / (al - ar)


def lateral_shift(img, shift_m, lane_px, y_vp):
    h, w = img.shape[:2]
    d = shift_m * lane_px / LANE_WIDTH_M      # pixel shift at the bottom row
    k = d / (h - y_vp)
    # x' = x - k*(y - y_vp): the scene moves left when the car moves right
    M = np.float32([[1, -k, k * y_vp], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def shift_experiment(frames, shifts_m):
    """Simulate lateral drift of the car by a known amount.

    For a flat road, moving the camera sideways by d metres moves every ground point
    horizontally by an amount proportional to its distance below the horizon, i.e. a
    horizontal shear anchored at the vanishing-point row. Applying that shear gives
    frames whose true offset is known exactly (offset of the original frame + d).
    """
    rows = []
    for f in frames:
        h, w = f.shape[:2]
        lanes0, _ = fit_lanes(edges_improved(f), f.shape, True)
        off0, lane_px, _ = departure(lanes0, w)
        if off0 is None:
            continue
        for sm in shifts_m:
            fs = lateral_shift(f, sm, lane_px, vanishing_row(lanes0))
            lanes, _ = fit_lanes(edges_improved(fs), fs.shape, True)
            off, _, status = departure(lanes, w)
            true = off0 + sm
            rows.append({"shift_m": sm, "true_offset": true, "pred_offset": off,
                         "true_warn": abs(true) >= WARN_M, "pred_warn": None if off is None else status != "OK"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video-output", action="store_true")
    args = ap.parse_args()
    download(); RES.mkdir(exist_ok=True)

    # ---------- 1. step-by-step visualisation on test images
    img = cv2.imread(str(DATA / "test_images" / "solidYellowCurve2.jpg"))
    m, poly = roi_mask(img.shape)
    eb, ei = edges_baseline(img), edges_improved(img)
    lanes_b, _ = fit_lanes(eb, img.shape, False)
    lanes_i, roi_e = fit_lanes(ei, img.shape, True)
    roi_vis = img.copy(); cv2.polylines(roi_vis, poly, True, (0, 255, 255), 3)
    steps = [("Input", img), ("Baseline: Canny edges", eb), ("Improved: colour+gradient edges", ei),
             ("Region of interest", roi_vis), ("Masked edges", roi_e),
             ("Output", draw(img, lanes_i, departure(lanes_i, img.shape[1])[2], departure(lanes_i, img.shape[1])[0]))]
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.5))
    for a, (t, im) in zip(axes.ravel(), steps):
        a.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap="gray"); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig(RES / "pipeline_steps.png", dpi=100); plt.close(fig)

    fig, axes = plt.subplots(len(IMAGES), 2, figsize=(12, 3.4 * len(IMAGES)))
    for r, n in enumerate(IMAGES):
        im = cv2.imread(str(DATA / "test_images" / n)); w = im.shape[1]
        for c, (name, fn, strict) in enumerate([("baseline", edges_baseline, False), ("improved", edges_improved, True)]):
            ln, _ = fit_lanes(fn(im), im.shape, strict)
            off, _, st = departure(ln, w)
            out = draw(im, ln, st, off)
            if c == 1:
                cv2.imwrite(str(RES / f"output_{n}"), out)
            axes[r, c].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); axes[r, c].set_title(f"{n} - {name}", fontsize=9); axes[r, c].axis("off")
    fig.tight_layout(); fig.savefig(RES / "test_images_comparison.png", dpi=80); plt.close(fig)

    # ---------- 2. video evaluation (proxy metrics, no lane ground truth available)
    video_metrics, traces = {}, {}
    for v in VIDEOS:
        for p in ["baseline", "improved"]:
            out = None if args.no_video_output or p == "baseline" else RES / f"output_{v}"
            met, bx = process_video(DATA / "test_videos" / v, p, out)
            video_metrics[f"{v}/{p}"] = met; traces[(v, p)] = bx
            print(v, p, json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in met.items()}))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for a, v in zip(axes, VIDEOS):
        for p, c in [("baseline", "#8c8c8c"), ("improved", "#dd8452")]:
            bx = traces[(v, p)]
            a.plot(bx[:, 0], color=c, label=f"{p} left"); a.plot(bx[:, 1], color=c, ls="--", label=f"{p} right")
        a.set_title(v); a.set_xlabel("frame"); a.set_ylabel("lane x at bottom (fraction of width)"); a.set_ylim(-0.1, 1.1)
    axes[0].legend(fontsize=8); fig.suptitle("Temporal stability of detected lane positions (smoother = better)")
    fig.tight_layout(); fig.savefig(RES / "temporal_stability.png", dpi=110); plt.close(fig)

    # ---------- 3. controlled lane-departure experiment
    frames = [cv2.imread(str(DATA / "test_images" / n)) for n in IMAGES]
    cap = cv2.VideoCapture(str(DATA / "test_videos" / "solidWhiteRight.mp4"))
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % 15 == 0:
            frames.append(f)
        k += 1
    cap.release()
    shifts = np.round(np.arange(-1.2, 1.21, 0.1), 2)
    rows = shift_experiment(frames, shifts)
    valid = [r for r in rows if r["pred_offset"] is not None]
    err = np.array([r["pred_offset"] - r["true_offset"] for r in valid])
    tp = sum(r["true_warn"] and r["pred_warn"] for r in valid); fp = sum(not r["true_warn"] and r["pred_warn"] for r in valid)
    fn = sum(r["true_warn"] and not r["pred_warn"] for r in valid); tn = sum(not r["true_warn"] and not r["pred_warn"] for r in valid)
    dep = {"trials": len(rows), "detected": len(valid), "offset_MAE_m": float(np.abs(err).mean()),
           "offset_RMSE_m": float(np.sqrt((err ** 2).mean())),
           "warning_precision": tp / max(tp + fp, 1), "warning_recall": tp / max(tp + fn, 1),
           "warning_accuracy": (tp + tn) / len(valid), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}
    print("departure:", dep)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([r["true_offset"] for r in valid], [r["pred_offset"] for r in valid], s=10, alpha=0.5)
    ax.plot([-1.5, 1.5], [-1.5, 1.5], "k--", lw=1)
    for s in (-WARN_M, WARN_M):
        ax.axvline(s, color="r", lw=0.8, ls=":"); ax.axhline(s, color="r", lw=0.8, ls=":")
    ax.set_xlabel("true lateral offset (m)"); ax.set_ylabel("estimated offset (m)")
    ax.set_title(f"Lane-departure offset estimation\nMAE={dep['offset_MAE_m']:.3f} m, warning acc={dep['warning_accuracy']:.3f}")
    fig.tight_layout(); fig.savefig(RES / "departure_offset_scatter.png", dpi=120); plt.close(fig)

    # Example of a simulated departure
    f = frames[0]; w = f.shape[1]
    ln0, _ = fit_lanes(edges_improved(f), f.shape, True); _, lane_px, _ = departure(ln0, w)
    panels = []
    for sm in (-0.6, 0.0, 0.9):
        fs = lateral_shift(f, sm, lane_px, vanishing_row(ln0))
        ln, _ = fit_lanes(edges_improved(fs), fs.shape, True); off, _, st = departure(ln, w)
        panels.append(cv2.resize(draw(fs, ln, st, off), (480, 270)))
    cv2.imwrite(str(RES / "departure_warning_examples.jpg"), np.hstack(panels))

    def rnd(o):
        if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, (float, np.floating)): return round(float(o), 4)
        if isinstance(o, np.integer): return int(o)
        return o
    (RES / "metrics.json").write_text(json.dumps(rnd({"video": video_metrics, "departure_experiment": dep}), indent=2))


if __name__ == "__main__":
    main()
