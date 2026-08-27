"""
Full implant-reconstruction pipeline applied to the third CT dataset
(Patient 3, paediatric pre-operative CT, age 6), reusing every verified
component from the craniofacial-
case pipeline (mindicom.py, voxelmesh.py, boa.py, method_a_boa.py,
method_b_delaunay.py) unchanged.

Survey findings (see task #49 in the project conversation):
  - 512x512, 206 slices, in-plane spacing 0.429mm, slice spacing ~1.0mm.
  - Unlike two infant datasets surveyed alongside this one (excluded from
    the pipeline; unfused cranial sutures, not a real defect),
    component-connectivity analysis on Patient3's pre-op scan shows ONE
    dominant connected bone-ring component per slice (not several
    comparably-sized fragments), confirming this is genuine single-defect
    anatomy suitable for the mirror-symmetric pipeline, not an artefact of
    unfused cranial sutures.
  - A curved external cable/tube runs alongside the head in many slices;
    it is correctly excluded from the "largest connected component" by
    construction (it is not 6-connected to the bone ring), but IS present
    in the raw thresholded mask, so any "raw mask" pixel count is not
    representative of the ring by itself.
  - Following the D19 dataset's lesson (component_containing_seed(seed=None)
    is unreliable for slices where the nearest-to-centroid seed lands on a
    stray fragment), this script uses full connected_components() enumeration
    and always takes the largest component explicitly, both for defect
    localisation and for whole-skull 3D growth seeding.
  - A companion "post-op" scan of the same patient exists (~2 months later,
    this dataset). Initial low-resolution visual survey suggested a healed
    cranioplasty implant; a full-resolution follow-up scan instead showed a
    LARGER open bone defect (~130mm superior-inferior extent, vs ~18mm in
    the "pre-op" scan) at the same location, with no evidence of implant
    material. This is far more consistent with "post-op" meaning post-
    DECOMPRESSIVE-CRANIECTOMY (the surgeon enlarging the opening to relieve
    intracranial pressure) than post-cranioplasty (closing it). There is
    therefore no real surgical implant available in this patient's imaging
    to validate the reconstruction below against; this dataset was not
    processed further (see project conversation).
  - IMPORTANT LIMITATION, discovered after building the mirrored target:
    the defect's two corner points straddle the sagittal symmetry axis x0
    on most defect slices (e.g. k=157: corners at x=368 and x=136, x0=239.5),
    i.e. the defect is bifrontal / crosses (or nearly crosses) the midline.
    Pure mirror-symmetric reconstruction is fundamentally unable to fully
    restore a midline-crossing defect (the paper's own Discussion section
    lists this as a known limitation of both methods). A pre/post-fusion
    gap-closure check confirms this: only 3 of 14 defect slices close to
    within 3 degrees after adding the mirrored target; the rest retain a
    partial residual gap. The reconstruction below should therefore be read
    as a partial, best-effort fill (correct wherever the mirroring
    assumption holds) rather than a complete closure, and the two methods
    disagree in volume far more than in the other two case studies (Method
    A: 1.2 cm^3, Method B: 0.7 cm^3, vs 2.9% agreement in the original
    craniofacial case) -- itself a symptom of the same underlying issue,
    since both methods are extrapolating from the same sparse, partial
    mirrored data.

Run stages in order:
  1. build_skull_volume()      -> skull_data_patient3.npz
  2. localize_defect()          -> defect_gaps_fullres_patient3.npz
  3. estimate_symmetry()        -> symmetry_x0_patient3.npy
  4. build_defect_target()      -> defect_target_patient3.npz
  5. run_method_a() / run_method_b() -> method_[a|b]_smoothed_patient3.npz + STL
  6. combine_and_verify()       -> fused STLs + manifold/gap audit
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mindicom import Slice
from voxelmesh import (connected_components, _dilate6, cuberille_mesh,
                        mesh_signed_volume, write_binary_stl)

FOLDER = os.environ.get("DICOM_DATA_DIR", "./data/patient3_dicom_series")
OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
BONE_HU_THRESHOLD = 250
IN_PLANE_STEP = 2
SKULL_SEED_2D_SLICE = 128   # known-good slice (single dominant ring component, see survey)
MIN_GAP_DEG = 8.0


def load_series(folder=FOLDER):
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


def robust_largest_2d(mask2d, min_voxels=20):
    """Largest connected component via full enumeration (see module
    docstring: seed=None nearest-centroid heuristic is unreliable)."""
    comps = connected_components(mask2d[None, :, :], min_voxels=min_voxels,
                                  max_components=30, verbose=False)
    return comps[0][0] if comps else None


def erode2d(mask2d):
    return (~_dilate6((~mask2d)[None, :, :]))[0]


def boundary2d(mask2d):
    return mask2d & ~erode2d(mask2d)


def slice_gap(mask2d, cx, cy, min_gap_deg=MIN_GAP_DEG):
    b = boundary2d(mask2d)
    ys, xs = np.nonzero(b)
    if len(xs) < 10:
        return False, None, None
    ang = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(ang)
    ang_sorted = ang[order]
    xs_s, ys_s = xs[order], ys[order]
    gaps = np.diff(ang_sorted)
    wrap_gap = (ang_sorted[0] + 2 * np.pi) - ang_sorted[-1]
    all_gaps = np.append(gaps, wrap_gap)
    gi = np.argmax(all_gaps)
    gap_size = all_gaps[gi]
    if np.degrees(gap_size) < min_gap_deg:
        return False, None, None
    if gi == len(all_gaps) - 1:
        p_before, p_after = (xs_s[-1], ys_s[-1]), (xs_s[0], ys_s[0])
    else:
        p_before, p_after = (xs_s[gi], ys_s[gi]), (xs_s[gi + 1], ys_s[gi + 1])
    return True, p_before, p_after


def build_skull_volume():
    slices = load_series()
    rows, cols = slices[0].rows, slices[0].cols
    dy, dx = slices[0].pixel_spacing
    zs = [s.z() for s in slices]
    dz = float(np.median(np.diff(sorted(zs))))
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
    if IN_PLANE_STEP > 1:
        vol = vol[:, ::IN_PLANE_STEP, ::IN_PLANE_STEP]
        spacing = (dz, dy * IN_PLANE_STEP, dx * IN_PLANE_STEP)

    raw_solid = vol >= BONE_HU_THRESHOLD
    seed2d = robust_largest_2d(raw_solid[SKULL_SEED_2D_SLICE])
    ys, xs = np.nonzero(seed2d)
    seed = (SKULL_SEED_2D_SLICE, int(ys[0]), int(xs[0]))
    print(f"seed slice {SKULL_SEED_2D_SLICE}: component size {seed2d.sum()} px, seed voxel {seed}")

    # 3D growth from the verified seed (6-connected dilation intersected
    # with threshold mask, same primitive as Algorithm 1 in the paper)
    from voxelmesh import component_containing_seed
    skull = component_containing_seed(raw_solid, seed=seed, max_iter=1000, verbose=True)

    np.savez_compressed(os.path.join(OUT, "skull_data_patient3.npz"),
                         vol=vol, skull=skull, spacing=spacing, origin=origin)
    print(f"skull voxels: {int(skull.sum()):,}  shape {skull.shape}")
    return skull, spacing, origin


def localize_defect():
    """Full-resolution per-slice angular gap scan, using the ROBUST
    largest-component selection (not the seed=None heuristic)."""
    slices = load_series()
    dy, dx = slices[0].pixel_spacing
    zs = [s.z() for s in slices]
    dz = float(np.median(np.diff(sorted(zs))))

    rows = []
    t0 = time.time()
    for k, s in enumerate(slices):
        mask = s.hu >= BONE_HU_THRESHOLD
        if mask.sum() < 200:
            continue
        comp = robust_largest_2d(mask)
        if comp is None:
            continue
        ys, xs = np.nonzero(comp)
        if len(xs) < 50:
            continue
        cy, cx = ys.mean(), xs.mean()
        found, p0, p1 = slice_gap(comp, cx, cy)
        if found:
            a0 = np.degrees(np.arctan2(p0[1] - cy, p0[0] - cx))
            a1 = np.degrees(np.arctan2(p1[1] - cy, p1[0] - cx))
            gap_deg = (a1 - a0) % 360
            rows.append((k, s.z(), gap_deg, p0, p1, cx, cy, int(comp.sum())))
        if k % 40 == 0:
            print(f"  ...slice {k}/{len(slices)} ({time.time()-t0:.0f}s)")

    print(f"slices with gap>={MIN_GAP_DEG}deg: {len(rows)}  ({time.time()-t0:.0f}s total)")
    for r in rows:
        print(f"  k={r[0]:3d} z={r[1]:7.1f} gap={r[2]:6.1f}deg n={r[7]}")

    np.savez(os.path.join(OUT, "defect_gaps_fullres_patient3.npz"),
              ks=np.array([r[0] for r in rows]),
              zs=np.array([r[1] for r in rows]),
              gap_deg=np.array([r[2] for r in rows]),
              p0=np.array([r[3] for r in rows]),
              p1=np.array([r[4] for r in rows]),
              centroids=np.array([(r[5], r[6]) for r in rows]),
              pixel_spacing=np.array([dy, dx]), dz=dz)
    print("saved defect_gaps_fullres_patient3.npz")
    return rows


CLEAN_RANGES = [(55, 80), (105, 130), (175, 196)]  # defect-free vault slices
K_START, K_END = 150, 172  # candidate defect range (padded; refined visually)


def estimate_symmetry():
    slices = load_series()
    idxs = [i for a, b in CLEAN_RANGES for i in range(a, b)]
    print("using", len(idxs), "clean slices")

    masks = []
    for i in idxs:
        m = slices[i].hu >= BONE_HU_THRESHOLD
        if m.sum() < 200:
            continue
        comp = robust_largest_2d(m)
        if comp is not None:
            masks.append(comp)
    stack = np.stack(masks, axis=0)
    print("stack shape:", stack.shape)

    ny, nx = stack.shape[1], stack.shape[2]
    c_full = (nx - 1) / 2.0
    best_offset, best_score = 0, -1
    for offset in range(-nx // 3, nx // 3):
        flipped = np.roll(stack[:, :, ::-1], offset, axis=2)
        score = int((stack & flipped).sum())
        if score > best_score:
            best_score, best_offset = score, offset
    x0 = c_full + best_offset / 2.0
    print(f"symmetry axis: x0 = {x0:.2f}  (image width {nx}), best overlap = {best_score}")
    np.save(os.path.join(OUT, "symmetry_x0_patient3.npy"), np.array([x0]))
    return x0


def dilate2d(mask2d, n=1):
    m = mask2d[None, :, :]
    for _ in range(n):
        m = _dilate6(m)
    return m[0]


def build_defect_target(k_start=K_START, k_end=K_END):
    slices = load_series()
    x0 = float(np.load(os.path.join(OUT, "symmetry_x0_patient3.npy"))[0])
    print("symmetry x0 =", x0)

    ny, nx = slices[0].rows, slices[0].cols
    c_full = (nx - 1) / 2.0
    offset = int(round(2 * (x0 - c_full)))

    n_k = k_end - k_start + 1
    actual_stack = np.zeros((n_k, ny, nx), dtype=bool)
    target_stack = np.zeros((n_k, ny, nx), dtype=bool)
    gap_report = []

    for idx, k in enumerate(range(k_start, k_end + 1)):
        m = slices[k].hu >= BONE_HU_THRESHOLD
        comp = robust_largest_2d(m)
        if comp is None:
            continue
        mirrored = np.roll(comp[:, ::-1], offset, axis=1)

        ys, xs = np.nonzero(comp)
        cy, cx = ys.mean(), xs.mean()
        found, p0, p1 = slice_gap(comp, cx, cy)

        if found:
            raw_target = mirrored & (~comp)
            a0 = np.arctan2(p0[1] - cy, p0[0] - cx)
            a1 = np.arctan2(p1[1] - cy, p1[0] - cx)
            yy, xx = np.nonzero(raw_target)
            ang = np.arctan2(yy - cy, xx - cx)
            rel = (ang - a0) % (2 * np.pi)
            gap_span = (a1 - a0) % (2 * np.pi)
            keep = rel <= gap_span
            mask_keep = np.zeros_like(raw_target)
            mask_keep[yy[keep], xx[keep]] = True
            raw_target = raw_target & mask_keep
            raw_target = dilate2d(raw_target, 1) & mirrored
        else:
            raw_target = np.zeros_like(comp)

        actual_stack[idx] = comp
        target_stack[idx] = raw_target
        gap_report.append((k, found, int(raw_target.sum())))
        print(f"  slice {k} found={found} target_px={int(raw_target.sum())}")

    dy, dx_ = slices[0].pixel_spacing
    zs_all = [s.z() for s in slices]
    dz = float(np.median(np.diff(sorted(zs_all))))
    z0 = slices[k_start].z()

    np.savez_compressed(os.path.join(OUT, "defect_target_patient3.npz"),
                         actual=actual_stack, target=target_stack,
                         k_start=k_start, k_end=k_end,
                         spacing=np.array([dz, dy, dx_]), z0=z0, x0=x0)
    print(f"saved defect_target_patient3.npz  total target voxels={int(target_stack.sum())}")
    return actual_stack, target_stack


def run_method_a():
    from method_a_boa import process_all as process_all_a
    out_stack, spacing, z0 = process_all_a(target_npz=os.path.join(OUT, "defect_target_patient3.npz"))
    np.savez_compressed(os.path.join(OUT, "method_a_smoothed_patient3.npz"),
                         smoothed=out_stack, spacing=spacing, z0=z0)
    dz, dy, dx = spacing
    tris = cuberille_mesh(out_stack, (dz, dy, dx), (z0, 0.0, 0.0))
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"Method A mesh: {tris.shape[0]:,} triangles, {vol_cm3:.1f} cm^3")
    out_path = os.path.join(OUT, "Implant_MethodA_BOA_Patient3.stl")
    write_binary_stl(out_path, tris, name=b"Implant Method A BOA Patient3")
    print("wrote", out_path)
    return out_stack


def run_method_b():
    from method_b_delaunay import process_all as process_all_b
    out_stack, spacing, z0 = process_all_b(target_npz=os.path.join(OUT, "defect_target_patient3.npz"))
    np.savez_compressed(os.path.join(OUT, "method_b_smoothed_patient3.npz"),
                         smoothed=out_stack, spacing=spacing, z0=z0)
    dz, dy, dx = spacing
    tris = cuberille_mesh(out_stack, (dz, dy, dx), (z0, 0.0, 0.0))
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"Method B mesh: {tris.shape[0]:,} triangles, {vol_cm3:.1f} cm^3")
    out_path = os.path.join(OUT, "Implant_MethodB_Delaunay_Patient3.stl")
    write_binary_stl(out_path, tris, name=b"Implant Method B Delaunay Patient3")
    print("wrote", out_path)
    return out_stack


def combine_and_verify(implant_npz, out_stl, label):
    skull_d = np.load(os.path.join(OUT, "skull_data_patient3.npz"))
    skull = skull_d['skull']; spacing = skull_d['spacing']; origin = skull_d['origin']
    target_d = np.load(os.path.join(OUT, "defect_target_patient3.npz"))
    k_start = int(target_d['k_start'])
    imp_d = np.load(os.path.join(OUT, implant_npz))
    implant_full = imp_d['smoothed']
    # downsample to match half-res skull grid, one extra dilation to seal the seam
    ds = implant_full[:, ::2, ::2].astype(bool)
    implant_ds = _dilate6(ds[None])[0]
    combined = skull.copy()
    n_k = implant_ds.shape[0]
    seg = combined[k_start:k_start + n_k]
    seg |= implant_ds[:seg.shape[0]]
    tris = cuberille_mesh(combined, tuple(spacing), tuple(origin))
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"{label}: {tris.shape[0]:,} triangles, {vol_cm3:.1f} cm^3")
    write_binary_stl(os.path.join(OUT, out_stl), tris, name=label.encode('ascii', 'ignore'))
    return combined, tris


def export_skull_only():
    skull_d = np.load(os.path.join(OUT, "skull_data_patient3.npz"))
    skull = skull_d['skull']; spacing = skull_d['spacing']; origin = skull_d['origin']
    tris = cuberille_mesh(skull, tuple(spacing), tuple(origin))
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"Skull only: {tris.shape[0]:,} triangles, {vol_cm3:.1f} cm^3")
    write_binary_stl(os.path.join(OUT, "Skull_Only_Patient3.stl"), tris, name=b"Patient3 Skull Only")


if __name__ == '__main__':
    print(__doc__)
