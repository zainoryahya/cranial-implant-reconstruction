"""
Locate the skull defect (gap in the bone ring) per-slice, and estimate the
sagittal symmetry plane, using the cached skull_data.npz from
build_skull_data.py.

Boundary extraction follows the reference papers: beta(A) = A - erode(A),
using erode(A) = ~dilate(~A) (standard duality), 6-connectivity dilation
from voxelmesh._dilate6 (with a size-1 z-axis this reduces to 2D 4-connectivity
dilation per slice, which is fine for this purpose).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from voxelmesh import _dilate6

OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
CACHE = os.path.join(OUT, "skull_data.npz")


def erode2d(mask2d):
    m3 = mask2d[None, :, :]
    er = ~_dilate6(~m3)
    return er[0]


def boundary2d(mask2d):
    return mask2d & ~erode2d(mask2d)


def find_symmetry_x(skull, exclude_z=(130, 190)):
    nz, ny, nx = skull.shape
    keep = np.ones(nz, dtype=bool)
    keep[exclude_z[0]:exclude_z[1]] = False
    sub = skull[keep]
    c_full = (nx - 1) / 2.0
    best_offset, best_score = 0, -1
    for offset in range(-nx // 2, nx // 2):
        flipped = sub[:, :, ::-1]
        flipped = np.roll(flipped, offset, axis=2)
        score = int((sub & flipped).sum())
        if score > best_score:
            best_score = score
            best_offset = offset
    c0 = c_full + best_offset / 2.0
    return c0, best_score


def slice_gap(mask2d, cx, cy, min_gap_deg=8.0):
    """Return (gap_found, gap_start_angle, gap_end_angle, corner_pts) for the
    largest angular gap in the boundary of mask2d, as seen from (cx,cy)."""
    b = boundary2d(mask2d)
    ys, xs = np.nonzero(b)
    if len(xs) < 10:
        return False, None, None, None
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
        return False, None, None, None
    if gi == len(all_gaps) - 1:
        p_before = (xs_s[-1], ys_s[-1]); a_before = ang_sorted[-1]
        p_after = (xs_s[0], ys_s[0]); a_after = ang_sorted[0] + 2*np.pi
    else:
        p_before = (xs_s[gi], ys_s[gi]); a_before = ang_sorted[gi]
        p_after = (xs_s[gi+1], ys_s[gi+1]); a_after = ang_sorted[gi+1]
    return True, a_before, a_after, (p_before, p_after)


if __name__ == '__main__':
    d = np.load(CACHE)
    vol, skull, spacing, origin = d['vol'], d['skull'], d['spacing'], d['origin']
    print("volume:", vol.shape, "spacing:", spacing)

    c0, score = find_symmetry_x(skull)
    print(f"symmetry axis: column x={c0:.2f} (of {skull.shape[2]}), overlap score={score}")

    nz = skull.shape[0]
    results = []
    for k in range(nz):
        mask2d = skull[k]
        if mask2d.sum() < 50:
            continue
        ys, xs = np.nonzero(mask2d)
        cy, cx = ys.mean(), xs.mean()
        found, a0, a1, pts = slice_gap(mask2d, cx, cy, min_gap_deg=8.0)
        if found:
            gap_deg = np.degrees(a1 - a0)
            results.append((k, gap_deg, pts[0], pts[1], cx, cy))

    print(f"\nslices with a gap >=8deg: {len(results)}")
    for k, gap_deg, p0, p1, cx, cy in results:
        z = origin[0] + k * spacing[0]
        print(f"  k={k:3d} z={z:7.1f}  gap={gap_deg:6.1f}deg  corner0={p0} corner1={p1}  centroid=({cx:.0f},{cy:.0f})")

    np.savez(os.path.join(OUT, "defect_gaps.npz"),
             ks=np.array([r[0] for r in results]),
             gap_deg=np.array([r[1] for r in results]),
             p0=np.array([r[2] for r in results]),
             p1=np.array([r[3] for r in results]),
             centroids=np.array([(r[4], r[5]) for r in results]),
             symmetry_x=c0)
    print("\nsaved defect_gaps.npz")
