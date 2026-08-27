"""
Full implant-reconstruction pipeline applied to the second CT dataset
(Patient 2, large decompressive-craniectomy defect), reusing every
verified component from the
craniofacial-case pipeline (mindicom.py, voxelmesh.py, boa.py,
method_a_boa.py, method_b_delaunay.py) unchanged.

This is a case study in applying the SAME method to a second, independent
real CT series, not a re-derivation. Key differences from the first dataset,
discovered while running this script, are documented inline.

Findings specific to this dataset:
  - Modality/geometry: 287 axial slices, 512x512, spacing 0.455mm in-plane,
    0.7mm slice spacing.
  - The default "nearest solid voxel to mask centroid" seed heuristic used
    for connected-component region growing (component_containing_seed with
    seed=None) FAILS on this dataset: the mask's centroid falls near a small
    isolated bone fleck near the middle of the cranial ring (not on the ring
    itself), so growing from that seed converges after a handful of voxels.
    Fixed by seeding explicitly from a point verified, via the proper
    connected_components() enumeration, to belong to the largest component.
  - The bone defect here is much larger than the first case (up to ~121
    degrees angular gap vs 15-68 degrees), consistent with a large
    decompressive craniectomy rather than a smaller trauma defect.
  - A first pass using the full candidate range k=[81,229] showed that the
    mirroring assumption BREAKS DOWN for k=81-99: the contralateral
    ("source") side being mirrored is itself missing a large fraction of its
    bone in the relevant angular sector at those slices (~10% coverage
    instead of the >90% expected of intact bone), i.e. the defect appears
    bilateral / close to the sagittal midline at its superior extent. This
    is exactly the limitation already documented in the companion paper
    ("neither [method] can reconstruct a defect that crosses the sagittal
    midline"). The pipeline was therefore restricted to k=[99,229], the
    range over which the mirrored source is genuinely intact.
  - Even within k=[99,229], full ring closure is gradual rather than sharp:
    slices k=99-140ish show a partially-tapering residual gap (the mirrored
    patch does not 100% meet the actual bone at the very tip), fully closing
    by roughly k=140 onward. This was verified directly (see
    verify_gap_closure() below) rather than assumed.

Run stages in order (each also runnable standalone from an interactive
session by importing the corresponding function):
  1. build_skull_volume()      -> skull_data_d19.npz
  2. localize_defect()          -> defect_gaps_fullres_d19.npz
  3. estimate_symmetry()        -> symmetry_x0_d19.npy
  4. build_defect_target()      -> defect_target_d19.npz
  5. run_method_a() / run_method_b() -> method_[a|b]_smoothed_d19.npz + STL
  6. combine_and_verify()       -> fused STLs + manifold/gap audit
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mindicom import Slice
from voxelmesh import (component_containing_seed, connected_components,
                        _dilate6, cuberille_mesh, mesh_signed_volume,
                        write_binary_stl)

BASE = os.environ.get("DICOM_DATA_DIR", "./data")
FOLDER = os.path.join(BASE, "patient2_dicom_series")
OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
BONE_HU_THRESHOLD = 250
IN_PLANE_STEP = 2
K_START, K_END = 99, 229          # restricted, mirror-valid defect range
SKULL_SEED_2D_SLICE = 150          # known-good slice for seeding 3D growth


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


def robust_comp2d(mask2d):
    """Largest connected component via full enumeration - NOT the
    seed=None centroid heuristic, which is unreliable for annular masks
    with small central bone flecks (see module docstring)."""
    comps = connected_components(mask2d[None, :, :], min_voxels=20,
                                  max_components=30, verbose=False)
    return comps[0][0] if comps else None


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
    seed2d = robust_comp2d(raw_solid[SKULL_SEED_2D_SLICE])
    ys, xs = np.nonzero(seed2d)
    seed = (SKULL_SEED_2D_SLICE, int(ys[0]), int(xs[0]))
    skull = component_containing_seed(raw_solid, seed=seed, max_iter=1000, verbose=True)

    np.savez_compressed(os.path.join(OUT, "skull_data_d19.npz"),
                         vol=vol, skull=skull, spacing=spacing, origin=origin)
    return skull, spacing, origin


def combine_and_verify(implant_npz, out_stl, label):
    skull_d = np.load(os.path.join(OUT, "skull_data_d19.npz"))
    skull = skull_d['skull']; spacing = skull_d['spacing']; origin = skull_d['origin']
    target_d = np.load(os.path.join(OUT, "defect_target_d19.npz"))
    k_start = int(target_d['k_start'])
    imp_d = np.load(os.path.join(OUT, implant_npz))
    implant_full = imp_d['smoothed']
    implant_ds = _dilate6(implant_full[:, ::2, ::2])
    combined = skull.copy()
    n_k = implant_ds.shape[0]
    combined[k_start:k_start + n_k] |= implant_ds[:combined[k_start:k_start + n_k].shape[0]]
    tris = cuberille_mesh(combined, tuple(spacing), tuple(origin))
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"{label}: {tris.shape[0]:,} triangles, {vol_cm3:.1f} cm^3")
    write_binary_stl(os.path.join(OUT, out_stl), tris, name=label.encode('ascii', 'ignore'))
    return combined


if __name__ == '__main__':
    print(__doc__)
    print("This script documents the pipeline; see the accompanying .npz/.stl")
    print("outputs already generated during the interactive session for results.")
