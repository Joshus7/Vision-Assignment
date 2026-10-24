# Dataset: Udacity CarND-LaneLines-P1 test data

- **Repository**: https://github.com/udacity/CarND-LaneLines-P1 (MIT licence)
- **Images** (`test_images/`): solidWhiteCurve, solidWhiteRight, solidYellowCurve, solidYellowCurve2, solidYellowLeft, whiteCarLaneSwitch — 960×540 JPEG
- **Videos** (`test_videos/`): solidWhiteRight.mp4 (221 frames), solidYellowLeft.mp4 (681 frames), challenge.mp4 (251 frames, 1280×720)
- **Raw file URL pattern**: `https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_images/<name>`

`src/lane_detection.py` downloads every file into `data/` on first run (about 28 MB). The dataset has no lane
annotations. See the README for how accuracy is evaluated without them.
