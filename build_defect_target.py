"""
Build the mirrored-reference implant target: for each defect slice, mirror
the intact contralateral bone ring across the sagittal symmetry axis x0, and
take (mirrored AND NOT actual, restricted to the angular gap sector) as the
patient-specific target patch for that slice (naturally inherits the
patient's own bone thickness from the mirrored side, slice by slice).

This mirrored target is the *reference* both reconstruction methods build
from:
  - Method A (BOA + curve fitting) smooths its jagged voxel boundary into
    clean parametric curves.
  - Method B (Delaunay) triangulates its raw boundary points directly.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mindicom import Slice
from voxelmesh import component_containing_seed, _dilate6

BASE = os.environ.get("DICOM_DATA_DIR", "./data")
OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
folder = os.path.join(BASE, "patient1_dicom_series")

K_START, K_END = 59, 179  # inclusive defect slice range found earlier


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


def dilate2d(mask2d, n=1):
    m = mask2d[None, :, :]
    for _ in range(n):
        m = _dilate6(m)
    return m[0]


if __name__ == '__main__':
    t0 = time.time()
    slices = load_series(folder)
    x0 = float(np.load(os.path.join(OUT, "symmetry_x0.npy"))[0])
    print("symmetry x0 =", x0)

    ny, nx = slices[0].rows, slices[0].cols
    c_full = (nx - 1) / 2.0
    offset = int(round(2 * (x0 - c_full)))

    n_k = K_END - K_START + 1
    actual_stack = np.zeros((n_k, ny, nx), dtype=bool)
    target_stack = np.zeros((n_k, ny, nx), dtype=bool)

    for idx, k in enumerate(range(K_START, K_END + 1)):
        m = slices[k].hu >= 250
        comp = component_containing_seed(m[None, :, :], verbose=False)[0]
        mirrored = np.roll(comp[:, ::-1], offset, axis=1)

        ys, xs = np.nonzero(comp)
        cy, cx = ys.mean(), xs.mean()

        # angular gap (corners) on the ACTUAL ring, to restrict the target
        # patch to the true gap sector (avoid unrelated left-right asymmetry
        # elsewhere, e.g. nasal septum) - reuse the same method as
        # defect_localize2.slice_gap
        from defect_localize2 import boundary2d, slice_gap
        found, p0, p1 = slice_gap(comp, cx, cy, min_gap_deg=8.0)

        if found:
            raw_target = mirrored & (~comp)
            a0 = np.arctan2(p0[1]-cy, p0[0]-cx)
            a1 = np.arctan2(p1[1]-cy, p1[0]-cx)
            yy, xx = np.nonzero(raw_target)
            ang = np.arctan2(yy - cy, xx - cx)
            # angle must lie within [a0,a1] going the "short way" that is the
            # actual gap (a1-a0 mod 2pi as computed in slice_gap)
            rel = (ang - a0) % (2*np.pi)
            gap_span = (a1 - a0) % (2*np.pi)
            keep = rel <= gap_span
            mask_keep = np.zeros_like(raw_target)
            mask_keep[yy[keep], xx[keep]] = True
            raw_target = raw_target & mask_keep
            # small dilation to close 1px seams from the roll/flip quantization
            raw_target = dilate2d(raw_target, 1) & mirrored
        else:
            # no real gap detected on this slice (ring already closed, or
            # below the min_gap_deg noise floor) - do NOT fall back to the
            # unrestricted mirror-difference, that picks up unrelated
            # natural asymmetry (orbits, sinuses, etc.) as a false patch.
            raw_target = np.zeros_like(comp)

        actual_stack[idx] = comp
        target_stack[idx] = raw_target

        if idx % 20 == 0:
            print(f"  slice {k} ({idx+1}/{n_k}) target pixels={int(raw_target.sum())}  ({time.time()-t0:.0f}s)")

    dy, dx_ = slices[0].pixel_spacing
    zs_all = [s.z() for s in slices]
    dz = float(np.median(np.diff(sorted(zs_all))))
    z0 = slices[K_START].z()

    np.savez_compressed(os.path.join(OUT, "defect_target.npz"),
                         actual=actual_stack, target=target_stack,
                         k_start=K_START, k_end=K_END,
                         spacing=np.array([dz, dy, dx_]), z0=z0, x0=x0)
    print(f"saved defect_target.npz  total target voxels={int(target_stack.sum())}")
    print(f"done in {time.time()-t0:.0f}s")
