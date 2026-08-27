import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mindicom import Slice

BASE = os.environ.get("DICOM_DATA_DIR", "./data")
OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
folder = os.path.join(BASE, "patient1_dicom_series")

def load_series(folder):
    files = []
    for root, dirs, fnames in os.walk(folder):
        for fn in fnames:
            if fn.lower().endswith(('.url', '.txt', '.ini', '.db')):
                continue
            files.append(os.path.join(root, fn))
    slices = []
    for f in files:
        try:
            s = Slice(f)
            if s.rows and s.cols:
                slices.append(s)
        except Exception:
            pass
    slices.sort(key=lambda s: s.z())
    return slices

slices = load_series(folder)
print("n slices:", len(slices))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = len(slices)
idxs = list(range(0, n, max(1, n // 40)))[:40]
ncols = 8
nrows = (len(idxs) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.2, nrows*2.2))
axes = axes.flatten()
for ax, i in zip(axes, idxs):
    hu = slices[i].hu
    ax.imshow(hu, cmap='gray', vmin=-200, vmax=1200)
    ax.set_title(f"i={i} z={slices[i].z():.0f}", fontsize=7)
    ax.axis('off')
for ax in axes[len(idxs):]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT, "craniofacial_slice_montage.png"), dpi=130)
print("wrote montage")
