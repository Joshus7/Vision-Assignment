# Project Report — Road Lane Detection and Departure Warning

## 1. Introduction
Lane detection is one of the oldest computer-vision tasks in driver assistance. This project builds a complete
lane-departure warning system using only classical, interpretable image processing. It shows colour-space
analysis, edge detection, the Hough transform, geometric model fitting and temporal filtering. Every stage can be
inspected visually (`results/pipeline_steps.png`).

## 2. Analysis of the data
- The Udacity clips are from a forward-facing camera on Californian highways in daylight.
- Lane paint is **white** (dashed/solid) or **yellow** (solid). In grayscale, yellow paint on light concrete has
  almost no contrast, which is why the baseline is unstable on the challenge video.
- The challenge clip adds **tree shadows**, a **concrete bridge** (bright, low-contrast road), a **curve** and the
  **car hood** at the bottom of the frame.
- No lane annotations exist, so evaluation needed a different strategy (Section 4).

## 3. Design decisions
| Problem | Decision |
|---|---|
| Shadows change brightness but not hue | Segment paint in **HLS**: white = high L, yellow = hue band |
| Lighting differs between clips | White threshold adapts to the 97th percentile of L in each frame |
| Hough returns many noisy segments (trees, cars, guard rails) | Slope range and image-side filters, then **length-weighted** least squares |
| Single-frame estimates flicker | EMA tracker with outlier rejection and short-term memory |
| Departure moves the lane toward the image edge | ROI widened at the bottom (hexagon) |

## 4. Evaluation strategy
1. **Controlled departure experiment.** A lateral camera translation over a flat road moves each ground point
   sideways by an amount proportional to (y − y_vanishing). Shearing the image about the vanishing-point row
   therefore simulates the car drifting by an exact, known distance. This gives ground truth for offset error and
   warning precision/recall without manual labelling.
2. **Proxy metrics on real video.** Detection rate, frame-to-frame jitter, lane-width consistency and FPS.
   The physical lane width is constant, so variation in measured width is error.
3. **Visual inspection** of all test images and output videos.

## 5. Results summary
- Offset MAE **0.013 m**. Warning precision **0.984**, recall **0.979** (402 trials with detections).
- In 123 of 525 trials (119 of them at simulated shifts of 0.8 m or more, i.e. the car is close to or on the line) one lane boundary moves
  out of the region the detector accepts, and the system reports "NO LANE" instead of a warning.
- The improved pipeline cuts jitter by 4–14× and lane-width variation by 1.5–5× compared with the baseline.
  On the challenge clip the gain is largest (jitter 2.41 % → 0.17 %).
- Speed: 54–87 fps (improved) vs 217–346 fps (baseline) on CPU, both well above real time.

## 6. Limitations and challenges
1. **Straight-line model.** Lanes are fitted as straight lines, so sharp curves are cut short near the horizon.
   A second-order polynomial on a bird's-eye (perspective-warped) view would handle curves better.
2. **Fixed ROI and camera geometry.** The ROI and the 3.7 m lane width assume a known camera mounting and a highway.
   Hills, lane changes and urban roads break these assumptions.
3. **Simulated departures.** The shear model is exact only for points on a flat ground plane. Objects above the
   road (cars, trees) are distorted, though they are mostly outside the ROI. Real departure recordings would be
   a better test.
4. **Missed detections at large departures.** When the car is on top of a line, the side filter rejects it. A real system
   should treat "lost lane after drift" as a warning condition, or track the adjacent lane.
5. **Conditions not tested.** Night, rain, snow, worn or missing markings, and heavy traffic hiding the lines are
   not in the dataset. Classical thresholds would likely need re-tuning. A learned segmentation model
   (e.g. trained on TuSimple or CULane) would generalise better at a higher compute cost.
6. **No pixel-level ground truth**, so detection accuracy on real frames is judged visually and by proxy metrics only.
