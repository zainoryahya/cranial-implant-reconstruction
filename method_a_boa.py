"""
Method A: per-slice cubic Bezier curve fitting (inner + outer bone-table
boundary of the implant patch) with free control points optimized by the
Butterfly Optimization Algorithm - directly analogous to how the reference
papers fit NURBS/rational-Ball curves through boundary points with GA/ABC,
just swapping in BOA as the optimization engine.

Per defect slice:
  1. Sample the mirrored target patch (from build_defect_target.py) in polar
     coordinates around the slice centroid -> discrete (theta, r_outer) and
     (theta, r_inner) boundary point sets.
  2. Fit a cubic Bezier curve in (theta, r) space to each, with endpoints
     anchored at the actual bone-ring corner readings (so the implant seams
     flush with the real bone) and two free interior control points
     optimized via BOA to minimize sum-of-squared radial error.
  3. Resample the fitted curves finely and rasterize the ring between them
     -> a smoothed 2D patch mask (replacing the jagged voxel boundary).
  4. Stack across slices -> volume -> cuberille mesh -> STL.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from boa import butterfly_optimize
from voxelmesh import cuberille_mesh, mesh_signed_volume, write_binary_stl

OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")


def bezier(P0, P1, P2, P3, t):
    t = t[:, None]
    return ((1-t)**3)*P0 + 3*((1-t)**2)*t*P1 + 3*(1-t)*(t**2)*P2 + (t**3)*P3


def polar_profile(mask2d, cx, cy, a0, a1, n_bins=48):
    """For each angle bin between a0->a1 (short way), return min/max radius
    of mask2d pixels falling in that bin. NaN where no pixels present."""
    ys, xs = np.nonzero(mask2d)
    if len(xs) == 0:
        return None
    ang = np.arctan2(ys - cy, xs - cx)
    span = (a1 - a0) % (2*np.pi)
    rel = (ang - a0) % (2*np.pi)
    keep = rel <= span
    rel, r = rel[keep], np.hypot(xs[keep]-cx, ys[keep]-cy)
    if len(r) == 0:
        return None
    bins = np.linspace(0, span, n_bins+1)
    idx = np.clip(np.digitize(rel, bins)-1, 0, n_bins-1)
    r_out = np.full(n_bins, np.nan)
    r_in = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            r_out[b] = r[sel].max()
            r_in[b] = r[sel].min()
    theta_centers = a0 + (bins[:-1] + bins[1:]) / 2.0
    return theta_centers, r_in, r_out


def fit_bezier_boa(theta_pts, r_pts, P0, P3, n_pop=16, n_iter=50, seed=0):
    """Fit cubic Bezier (theta,r) curve through (theta_pts,r_pts), endpoints
    fixed at P0,P3 (each (theta,r)); free params = P1,P2 (theta,r) each."""
    theta_pts = np.asarray(theta_pts)
    r_pts = np.asarray(r_pts)
    t_sample = np.linspace(0, 1, 40)

    th_lo, th_hi = min(P0[0], P3[0]), max(P0[0], P3[0])
    r_lo, r_hi = np.nanmin(r_pts) - 3, np.nanmax(r_pts) + 3
    bounds = np.array([[th_lo, th_hi], [r_lo, r_hi], [th_lo, th_hi], [r_lo, r_hi]])

    def fitness(pop):
        n = pop.shape[0]
        P1 = pop[:, 0:2]
        P2 = pop[:, 2:4]
        out = np.empty(n)
        for i in range(n):
            curve = bezier(np.array(P0), P1[i], P2[i], np.array(P3), t_sample)
            # nearest-curve-sample distance for each data point (small n, ok)
            d2 = ((curve[None, :, 1] - r_pts[:, None])**2
                  + ((curve[None, :, 0] - theta_pts[:, None]) * 50)**2)
            out[i] = np.nanmean(np.min(d2, axis=1))
        return out

    best_x, best_f, _ = butterfly_optimize(fitness, bounds, n_pop=n_pop,
                                            n_iter=n_iter, seed=seed)
    P1, P2 = best_x[0:2], best_x[2:4]
    return P1, P2, best_f


def process_all(target_npz=os.path.join(OUT, "defect_target.npz")):
    d = np.load(target_npz)
    actual, target = d['actual'], d['target']
    spacing = d['spacing']  # dz, dy, dx
    z0 = float(d['z0'])
    n_k = actual.shape[0]

    out_stack = np.zeros_like(target)
    t0 = time.time()
    n_fit = 0
    for i in range(n_k):
        am = actual[i]
        tm = target[i]
        if tm.sum() < 20:
            continue
        ys, xs = np.nonzero(am)
        cy, cx = ys.mean(), xs.mean()

        # corners: boundary between target and actual (approx via nonzero
        # extremes of target's angle span)
        ys_t, xs_t = np.nonzero(tm)
        ang_t = np.arctan2(ys_t - cy, xs_t - cx)
        a0 = ang_t.min()
        a1 = ang_t.max()
        # handle wraparound: use the polar_profile's own span logic instead
        prof = polar_profile(tm, cx, cy, a0, a1, n_bins=48)
        if prof is None:
            continue
        theta_c, r_in, r_out = prof
        valid = ~np.isnan(r_out)
        if valid.sum() < 5:
            continue

        # anchor endpoints using actual ring thickness just outside the gap
        def ring_radius_at(angle, width=0.05):
            ys2, xs2 = np.nonzero(am)
            ang2 = np.arctan2(ys2 - cy, xs2 - cx)
            d = np.abs(((ang2 - angle + np.pi) % (2*np.pi)) - np.pi)
            sel = d < width
            if sel.sum() < 2:
                width2 = width
                while sel.sum() < 2 and width2 < np.pi:
                    width2 += 0.05
                    d = np.abs(((ang2 - angle + np.pi) % (2*np.pi)) - np.pi)
                    sel = d < width2
            if sel.sum() < 2:
                return None  # no actual bone found near this angle at all
            r2 = np.hypot(xs2[sel]-cx, ys2[sel]-cy)
            return r2.min(), r2.max()

        rr0 = ring_radius_at(a0)
        rr1 = ring_radius_at(a1)
        if rr0 is None or rr1 is None:
            continue
        rin0, rout0 = rr0
        rin1, rout1 = rr1

        P0_out, P3_out = (a0, rout0), (a1, rout1)
        P0_in, P3_in = (a0, rin0), (a1, rin1)

        P1o, P2o, _ = fit_bezier_boa(theta_c[valid], r_out[valid], P0_out, P3_out, seed=i)
        P1i, P2i, _ = fit_bezier_boa(theta_c[valid], r_in[valid], P0_in, P3_in, seed=i+10000)

        t_fine = np.linspace(0, 1, 200)
        curve_out = bezier(np.array(P0_out), P1o, P2o, np.array(P3_out), t_fine)
        curve_in = bezier(np.array(P0_in), P1i, P2i, np.array(P3_in), t_fine)

        # rasterize smoothed ring segment
        ny, nx = am.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        ang_full = np.arctan2(yy - cy, xx - cx)
        rad_full = np.hypot(xx - cx, yy - cy)
        span = (a1 - a0) % (2*np.pi)
        rel_full = (ang_full - a0) % (2*np.pi)
        in_sector = rel_full <= span

        rel_curve = (curve_out[:, 0] - a0) % (2*np.pi)
        order = np.argsort(rel_curve)
        r_out_interp = np.interp(rel_full, rel_curve[order], curve_out[order, 1],
                                  left=curve_out[order, 1][0], right=curve_out[order, 1][-1])
        rel_curve_i = (curve_in[:, 0] - a0) % (2*np.pi)
        order_i = np.argsort(rel_curve_i)
        r_in_interp = np.interp(rel_full, rel_curve_i[order_i], curve_in[order_i, 1],
                                 left=curve_in[order_i, 1][0], right=curve_in[order_i, 1][-1])

        smoothed = in_sector & (rad_full <= r_out_interp) & (rad_full >= r_in_interp)
        out_stack[i] = smoothed
        n_fit += 1
        if i % 15 == 0:
            print(f"  slice {i}/{n_k} fitted (n_fit={n_fit})  {time.time()-t0:.0f}s")

    print(f"fitted {n_fit}/{n_k} slices in {time.time()-t0:.0f}s")
    return out_stack, spacing, z0


if __name__ == '__main__':
    out_stack, spacing, z0 = process_all()
    np.savez_compressed(os.path.join(OUT, "method_a_smoothed.npz"),
                         smoothed=out_stack, spacing=spacing, z0=z0)
    print("saved method_a_smoothed.npz, total voxels:", int(out_stack.sum()))

    dz, dy, dx = spacing
    origin = (z0, 0.0, 0.0)
    tris = cuberille_mesh(out_stack, (dz, dy, dx), origin)
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"Method A mesh: {tris.shape[0]:,} triangles, {vol_cm3:.1f} cm^3")
    out_path = os.path.join(OUT, "Implant_MethodA_BOA_Curve.stl")
    write_binary_stl(out_path, tris, name=b"Implant Method A BOA Curve")
    print("wrote", out_path, f"{os.path.getsize(out_path)/1e6:.1f} MB")
