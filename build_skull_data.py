"""Build & cache the craniofacial skull volume + cleaned skull mask so the
implant-design steps (defect localization, symmetry plane, curve fitting,
Delaunay reconstruction) don't need to recompute this every time."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mindicom import Slice
from voxelmesh import component_containing_seed

BASE = os.environ.get("DICOM_DATA_DIR", "./data")
OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
SERIES_DIR = os.path.join(BASE, "patient1_dicom_series")
CACHE = os.path.join(OUT, "skull_data.npz")

BONE_HU_THRESHOLD = 250
IN_PLANE_STEP = 2


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


def build_volume(slices, in_plane_step=1):
    rows, cols = slices[0].rows, slices[0].cols
    dy, dx = slices[0].pixel_spacing
    zs = [s.z() for s in slices]
    diffs = np.diff(sorted(zs))
    diffs = diffs[diffs > 1e-6]
    dz = float(np.median(diffs)) if len(diffs) else (slices[0].slice_thickness or 1.0)

    vol = np.empty((len(slices), rows, cols), dtype=np.float32)
    for i, s in enumerate(slices):
        hu = s.hu
        raw = s.pixels_raw
        sentinel = raw == np.iinfo(raw.dtype).max
        if sentinel.any() and sentinel.mean() < 0.2:
            hu = hu.copy()
            hu[sentinel] = -1024.0
        vol[i] = hu

    origin = (min(zs), 0.0, 0.0)
    spacing = (dz, dy, dx)
    if in_plane_step > 1:
        vol = vol[:, ::in_plane_step, ::in_plane_step]
        spacing = (dz, dy * in_plane_step, dx * in_plane_step)
    return vol, spacing, origin


if __name__ == '__main__':
    t0 = time.time()
    slices = load_series(SERIES_DIR)
    print("loaded", len(slices), "slices")
    vol, spacing, origin = build_volume(slices, in_plane_step=IN_PLANE_STEP)
    print("volume shape:", vol.shape, "spacing:", spacing, "origin:", origin)

    raw_solid = vol >= BONE_HU_THRESHOLD
    print("growing skull component...")
    skull = component_containing_seed(raw_solid, verbose=True)
    print("skull voxels:", int(skull.sum()))

    np.savez_compressed(CACHE, vol=vol, skull=skull, spacing=spacing, origin=origin)
    print("saved cache to", CACHE, f"({os.path.getsize(CACHE)/1e6:.1f} MB)")
    print(f"done in {time.time()-t0:.1f}s")
