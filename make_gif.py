"""Create a small animated GIF preview from a video (used for README demos).

usage: python make_gif.py <video> <out.gif> [start_frame] [n_frames] [step] [width]
"""
import sys

import cv2
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
start, n, step, width = (int(a) for a in (sys.argv[3:] + ["0", "60", "3", "480"][len(sys.argv) - 3:])[:4])
cap = cv2.VideoCapture(src)
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
frames, k = [], 0
while len(frames) < n:
    ok, f = cap.read()
    if not ok:
        break
    if k % step == 0:
        h = int(f.shape[0] * width / f.shape[1])
        frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(f, (width, h)), cv2.COLOR_BGR2RGB)))
    k += 1
frames[0].save(dst, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
print(dst, len(frames), "frames")
