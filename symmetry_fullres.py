"""Estimate the skull's sagittal symmetry axis (mirror column x0, in the
same full-resolution 512x512 pixel coordinate system as defect_localize2.py),
using only slices that are clear of the defect (and clear of the noisy
skull-base / vertex-tip regions)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mindicom import Slice
from voxelmesh import component_containing_seed

BASE = os.environ.get("DICOM_DATA_DIR", "./data")
OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
folder = os.path.join(BASE, "patient1_dicom_series")

# clear-of-defect, clear-of-noise slice ranges (from defect_localize2 output)
CLEAN_RANGES = [(15, 55), (181, 202)]


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
            if s.rows and s.cols and s.pixels_raw.size == s.rows * s.cols:
                slices.append(s)
        except Exception:
            pass
    slices.sort(key=lambda s: s.z())
    return slices


if __name__ == '__main__':
    t0 = time.time()
    slices = load_series(folder)
    idxs = [i for a, b in CLEAN_RANGES for i in range(a, b)]
    print("using", len(idxs), "clean slices")

    masks = []
    for i in idxs:
        m = slices[i].hu >= 250
        if m.sum() < 200:
            continue
        comp = component_containing_seed(m[None, :, :], verbose=False)[0]
        masks.append(comp)
    stack = np.stack(masks, axis=0)
    print("stack shape:", stack.shape, f"({time.time()-t0:.0f}s)")

    ny, nx = stack.shape[1], stack.shape[2]
    c_full = (nx - 1) / 2.0
    best_offset, best_score = 0, -1
    scores = []
    for offset in range(-nx // 3, nx // 3):
        flipped = np.roll(stack[:, :, ::-1], offset, axis=2)
        score = int((stack & flipped).sum())
        scores.append((offset, score))
        if score > best_score:
            best_score, best_offset = score, offset
    x0 = c_full + best_offset / 2.0
    print(f"symmetry axis: x0 = {x0:.2f}  (image width {nx}), best overlap = {best_score}")

    np.save(os.path.join(OUT, "symmetry_x0.npy"), np.array([x0]))
    print(f"done in {time.time()-t0:.0f}s")
