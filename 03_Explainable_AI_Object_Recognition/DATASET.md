# Dataset: Imagenette (imagenette2-160)

- **Source**: fast.ai — https://github.com/fastai/imagenette (Jeremy Howard), a subset of ImageNet (ILSVRC-2012)
- **Download**: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz (94 MB)
- **Classes (WordNet id → ImageNet index)**: n01440764 tench (0), n02102040 English springer (217),
  n02979186 cassette player (482), n03000684 chain saw (491), n03028079 church (497), n03394916 French horn (566),
  n03417042 garbage truck (569), n03425413 gas pump (571), n03445777 golf ball (574), n03888257 parachute (701)
- **Size**: 9,469 train / 3,925 validation images, short side 160 px
- **Licence**: same terms as ImageNet (non-commercial research/education)

`src/explain.py` downloads and extracts the archive into `data/` on first run. It checks archive paths for
path-traversal safety. Only the validation split is used, because the model is pretrained and not retrained.
