# Project Report — Explainable AI for Object Recognition

## 1. Introduction
Explainable AI (XAI) tries to make a model's decisions understandable to people. For image classifiers the most
common tool is an **attribution map** that highlights the pixels responsible for a prediction. This project
implements four attribution methods for a ResNet-50 and, more importantly, **tests whether they are faithful** to
the model, using objective metrics from the XAI literature.

## 2. Data and analysis
Imagenette (10 easy-to-distinguish ImageNet classes, 3,925 validation images) was chosen because:
- the pretrained model already performs well on it, so explanations can be studied for both correct and incorrect predictions;
- objects are usually centred and large, which makes a simple localisation check (centre energy) meaningful;
- it is small (94 MB) and loads quickly.

## 3. Implementation notes
- A forward hook on `layer4` stores the activations and calls `retain_grad()`. One backward pass of the
  target logit then gives both input gradients (saliency) and feature-map gradients (Grad-CAM/Grad-CAM++).
- Grad-CAM++ uses the closed-form α weights of Chattopadhay et al. (2018) under the exponential-score assumption.
- Occlusion batches all 169 occluded copies of an image through the network (64 at a time).
- Deletion/insertion replace pixels with a heavily blurred copy rather than black. Black pixels are
  out-of-distribution and change predictions by themselves.
- The model explains its **own** predicted class, which is the right target when asking "why did the model say this?".

## 4. Results summary
| | Deletion ↓ | Insertion ↑ | Sanity corr. ↓ |
|---|---|---|---|
| Grad-CAM | 0.108 | 0.319 | 0.08 |
| Grad-CAM++ | 0.110 | 0.310 | – |
| Occlusion | 0.162 | 0.336 | – |
| Saliency | 0.140 | 0.179 | 0.33 |
| Random | 0.145 | 0.144 | – |

Recognition: 80.9 % top-1 (1000-way), 96.9 % top-5, 99.5 % top-1 among the 10 classes.

## 5. Discussion
- Class-activation methods (Grad-CAM, Grad-CAM++) give compact, object-centred maps, and the metrics confirm these regions matter to the model.
- Grad-CAM and Grad-CAM++ were almost identical here, because most Imagenette images have a single object.
  Grad-CAM++ is designed for multi-instance scenes.
- Occlusion is model-agnostic and directly measures "what happens if this part is missing", but it costs 169×
  more compute and its resolution depends on the patch size.
- The disagreement between deletion and insertion for occlusion shows that no single faithfulness metric is enough.

## 6. Limitations and challenges
1. **Low spatial resolution of Grad-CAM** — maps come from a 7×7 grid, so thin structures (a chain-saw blade, a golf-ball dimple) cannot be localised precisely.
2. **Metric artefacts** — deletion/insertion depend on the replacement value (blur here). Partly blurred images are still
   out-of-distribution, so the model may react to the blur itself and not only to the missing information.
3. **No ground-truth "true explanation"** — faithfulness metrics are proxies. Imagenette has no bounding boxes or masks,
   so pointing-game accuracy could only be approximated by the centre-energy statistic.
4. **Sanity check only at the top of the network** — only `layer4` and `fc` were randomised (one cascade step).
   A full cascading randomisation would be more thorough. Grad-CAM++ and Occlusion were not included, to save compute.
5. **Single model** — conclusions may differ for Vision Transformers, where Grad-CAM is less natural and attention rollout is often used instead.
6. **Compute** — occlusion with 169 passes per image is the slowest step (most of the ~30 min runtime).

## 7. Future work
Integrated Gradients and SmoothGrad, RISE, a dataset with object masks (e.g. PASCAL VOC) for a real pointing game,
and comparison with a Vision Transformer.
