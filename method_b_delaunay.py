"""
Method B: direct Delaunay-triangulated surface reconstruction (contrasted
with Method A's per-slice independent BOA curve fitting).

Instead of fitting a curve slice-by-slice, this gathers boundary sample
points (theta, z, radius) from the mirrored target patch across ALL defect
slices at once, and builds ONE 2D Delaunay triangulation in the unrolled
(theta, z) parameter plane - separately for the outer bone-table surface and
the inner bone-table surface. matplotlib.tri provides a dependency-free
Delaunay implementation (no scipy needed).

The Delaunay-triangulated surfaces are then used as linear interpolants
r_outer(theta,z) / r_inner(theta,z) to rasterize a solid volume (so the
final STL export can reuse the same watertight cuberille mesher as Method
A - the comparison between methods is about how the surface is derived, not
the final meshing step).
"""
import os, sys, time
import numpy as np
import matplotlib.tri as mtri
sys.path.insert(0, os.path.dirname(__file__))
from method_a_boa import polar_profile
from voxelmesh import cuberille_mesh, mesh_signed_volume, write_binary_stl

OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")


def gather_points(actual, target, n_bins=60):
    """Return per-slice centroid/corner info AND flattened (theta,z_idx,r)
    point clouds for outer & inner surfaces."""
    n_k = actual.shape[0]
    theta_o, z_o, r_o = [], [], []
    theta_i, z_i_, r_i = [], [], []
    slice_info = []
    for i in range(n_k):
        am, tm = actual[i], target[i]
        if tm.sum() < 20:
            slice_info.append(None)
            continue
        ys, xs = np.nonzero(am)
        cy, cx = ys.mean(), xs.mean()
        ys_t, xs_t = np.nonzero(tm)
        ang_t = np.arctan2(ys_t - cy, xs_t - cx)
        a0, a1 = ang_t.min(), ang_t.max()
        prof = polar_profile(tm, cx, cy, a0, a1, n_bins=n_bins)
        slice_info.append((cx, cy, a0, a1))
        if prof is None:
            continue
        theta_c, r_in, r_out = prof
        valid_o = ~np.isnan(r_out)
        valid_i = ~np.isnan(r_in)
        theta_o.append(theta_c[valid_o]); z_o.append(np.full(valid_o.sum(), i)); r_o.append(r_out[valid_o])
        theta_i.append(theta_c[valid_i]); z_i_.append(np.full(valid_i.sum(), i)); r_i.append(r_in[valid_i])

    theta_o = np.concatenate(theta_o); z_o = np.concatenate(z_o); r_o = np.concatenate(r_o)
    theta_i = np.concatenate(theta_i); z_i_ = np.concatenate(z_i_); r_i = np.concatenate(r_i)
    return slice_info, (theta_o, z_o, r_o), (theta_i, z_i_, r_i)


def build_interp(theta, z, r):
    triang = mtri.Triangulation(theta, z)
    interp = mtri.LinearTriInterpolator(triang, r)
    return triang, interp


def query_with_fallback(interp, theta_q, z_q, theta_known, z_known, r_known):
    """Linear interpolation inside the hull; nearest-neighbour fallback
    outside it (LinearTriInterpolator returns masked values there)."""
    vals = interp(theta_q, z_q)
    out = np.asarray(np.ma.filled(vals, np.nan), dtype=np.float64)
    bad = np.isnan(out)
    if bad.any():
        # nearest neighbour fallback (small number of edge points expected)
        pts_known = np.column_stack([theta_known, z_known])
        for idx in np.nonzero(bad)[0]:
            d2 = (pts_known[:, 0]-theta_q[idx])**2 + (pts_known[:, 1]-z_q[idx])**2
            out[idx] = r_known[np.argmin(d2)]
    return out


def process_all(target_npz=os.path.join(OUT, "defect_target.npz")):
    d = np.load(target_npz)
    actual, target = d['actual'], d['target']
    spacing = d['spacing']
    z0 = float(d['z0'])
    n_k = actual.shape[0]

    t0 = time.time()
    slice_info, (theta_o, z_o, r_o), (theta_i, z_i_, r_i) = gather_points(actual, target)
    print(f"outer pts: {len(theta_o)}  inner pts: {len(theta_i)}  ({time.time()-t0:.0f}s)")

    _, interp_o = build_interp(theta_o, z_o, r_o)
    _, interp_i = build_interp(theta_i, z_i_, r_i)
    print(f"Delaunay triangulation built  ({time.time()-t0:.0f}s)")

    ny, nx = actual.shape[1], actual.shape[2]
    out_stack = np.zeros_like(target)
    for i in range(n_k):
        info = slice_info[i]
        if info is None:
            continue
        cx, cy, a0, a1 = info
        yy, xx = np.mgrid[0:ny, 0:nx]
        ang_full = np.arctan2(yy - cy, xx - cx)
        rad_full = np.hypot(xx - cx, yy - cy)
        span = (a1 - a0) % (2*np.pi)
        rel_full = (ang_full - a0) % (2*np.pi)
        in_sector = rel_full <= span
        if not in_sector.any():
            continue

        ys_sec, xs_sec = np.nonzero(in_sector)
        theta_q = ang_full[ys_sec, xs_sec]
        z_q = np.full(len(theta_q), i, dtype=np.float64)

        r_out_q = query_with_fallback(interp_o, theta_q, z_q, theta_o, z_o, r_o)
        r_in_q = query_with_fallback(interp_i, theta_q, z_q, theta_i, z_i_, r_i)

        r_pix = rad_full[ys_sec, xs_sec]
        keep = (r_pix <= r_out_q) & (r_pix >= r_in_q)
        sl = np.zeros((ny, nx), dtype=bool)
        sl[ys_sec[keep], xs_sec[keep]] = True
        out_stack[i] = sl

        if i % 20 == 0:
            print(f"  slice {i}/{n_k}  ({time.time()-t0:.0f}s)")

    print(f"done rasterizing in {time.time()-t0:.0f}s, total voxels {int(out_stack.sum())}")
    return out_stack, spacing, z0


if __name__ == '__main__':
    out_stack, spacing, z0 = process_all()
    np.savez_compressed(os.path.join(OUT, "method_b_smoothed.npz"),
                         smoothed=out_stack, spacing=spacing, z0=z0)

    dz, dy, dx = spacing
    origin = (z0, 0.0, 0.0)
    tris = cuberille_mesh(out_stack, (dz, dy, dx), origin)
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"Method B mesh: {tris.shape[0]:,} triangles, {vol_cm3:.1f} cm^3")
    out_path = os.path.join(OUT, "Implant_MethodB_Delaunay.stl")
    write_binary_stl(out_path, tris, name=b"Implant Method B Delaunay")
    print("wrote", out_path, f"{os.path.getsize(out_path)/1e6:.1f} MB")
