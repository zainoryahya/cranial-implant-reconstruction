"""
Full-resolution, per-slice defect localization (no 3D downsampling, so no
risk of erasing thin bone bridges). For each slice: bone threshold -> largest
2D connected component (removes stray hardware dots) -> morphological
boundary -> angular gap from centroid -> corner points.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mindicom import Slice
from voxelmesh import _dilate6, component_containing_seed

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
            if s.rows and s.cols and s.pixels_raw.size == s.rows * s.cols:
                slices.append(s)
        except Exception:
            pass
    slices.sort(key=lambda s: s.z())
    return slices


def erode2d(mask2d):
    m3 = mask2d[None, :, :]
    return (~_dilate6(~m3))[0]


def boundary2d(mask2d):
    return mask2d & ~erode2d(mask2d)


def largest_2d_component(mask2d):
    m3 = mask2d[None, :, :]
    comp = component_containing_seed(m3, seed=None, max_iter=2000, verbose=False)
    return comp[0]


def slice_gap(mask2d, cx, cy, min_gap_deg=8.0):
    b = boundary2d(mask2d)
    ys, xs = np.nonzero(b)
    if len(xs) < 10:
        return False, None, None
    ang = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(ang)
    ang_sorted = ang[order]
    xs_s, ys_s = xs[order], ys[order]
    gaps = np.diff(ang_sorted)
    wrap_gap = (ang_sorted[0] + 2*np.pi) - ang_sorted[-1]
    all_gaps = np.append(gaps, wrap_gap)
    gi = np.argmax(all_gaps)
    gap_size = all_gaps[gi]
    if np.degrees(gap_size) < min_gap_deg:
        return False, None, None
    if gi == len(all_gaps) - 1:
        p_before = (xs_s[-1], ys_s[-1])
        p_after = (xs_s[0], ys_s[0])
    else:
        p_before = (xs_s[gi], ys_s[gi])
        p_after = (xs_s[gi+1], ys_s[gi+1])
    return True, p_before, p_after


if __name__ == '__main__':
    t0 = time.time()
    slices = load_series(folder)
    print("n slices:", len(slices))
    dy, dx = slices[0].pixel_spacing
    zs = [s.z() for s in slices]
    dz = float(np.median(np.diff(sorted(zs))))

    rows = []
    for k, s in enumerate(slices):
        mask = s.hu >= 250
        if mask.sum() < 200:
            continue
        comp = largest_2d_component(mask)
        ys, xs = np.nonzero(comp)
        cy, cx = ys.mean(), xs.mean()
        found, p0, p1 = slice_gap(comp, cx, cy, min_gap_deg=8.0)
        if found:
            a0 = np.degrees(np.arctan2(p0[1]-cy, p0[0]-cx))
            a1 = np.degrees(np.arctan2(p1[1]-cy, p1[0]-cx))
            gap_deg = (a1 - a0) % 360
            rows.append((k, s.z(), gap_deg, p0, p1, cx, cy, int(comp.sum())))
        if k % 20 == 0:
            print(f"  ...processed slice {k}/{len(slices)}  ({time.time()-t0:.0f}s)")

    print(f"\nslices with gap>=8deg: {len(rows)}  total time {time.time()-t0:.0f}s")
    for k, z, gap_deg, p0, p1, cx, cy, npix in rows:
        print(f"  k={k:3d} z={z:7.1f} gap={gap_deg:6.1f}deg corner0={p0} corner1={p1} n={npix}")

    np.savez(os.path.join(OUT, "defect_gaps_fullres.npz"),
             ks=np.array([r[0] for r in rows]),
             zs=np.array([r[1] for r in rows]),
             gap_deg=np.array([r[2] for r in rows]),
             p0=np.array([r[3] for r in rows]),
             p1=np.array([r[4] for r in rows]),
             centroids=np.array([(r[5], r[6]) for r in rows]),
             pixel_spacing=np.array([dy, dx]), dz=dz)
    print("saved defect_gaps_fullres.npz")
