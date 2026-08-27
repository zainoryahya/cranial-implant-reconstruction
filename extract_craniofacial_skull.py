"""
Extract a skull-only 3D printable mesh from the "CRANIOFACIL FRACTURE DATA
22 4 14" CT series.

Pipeline:
  1. Parse the DICOM series (mindicom.py - pure Python, no pydicom needed).
  2. Stack slices into a Hounsfield-Unit volume (voxelmesh helpers via
     run_pipeline.load_series / build_volume).
  3. Threshold at a bone HU level (>= 250).
  4. Keep ONLY the largest 6-connected component. A raw bone threshold also
     picks up disconnected external hardware (headrest clips, fixation
     frame, table edge, etc.) that sits above the same HU range but is not
     physically attached to the skull - the earlier preview showed this as
     a trail of stray dots off to one side. Keeping just the largest
     connected blob removes those and leaves the skull only.
  5. Mesh the cleaned mask (cuberille voxel-face mesher, watertight -
     verified via the divergence-theorem volume check in voxelmesh.py).
  6. Write a binary STL, plus a quick 3-view preview PNG.

Run: python3 extract_craniofacial_skull.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mindicom import Slice
from voxelmesh import cuberille_mesh, mesh_signed_volume, write_binary_stl, \
    component_containing_seed

BASE = os.environ.get("DICOM_DATA_DIR", "./data")
OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
SERIES_DIR = os.path.join(BASE, "patient1_dicom_series")

BONE_HU_THRESHOLD = 250
IN_PLANE_STEP = 2   # in-plane downsample factor (1 = full 512x512 res)
SLICE_STEP = 1       # slice skip factor (1 = keep every slice)


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
    if not slices:
        raise RuntimeError(f"No readable DICOM slices in {folder}")
    if all(s.image_position for s in slices):
        slices.sort(key=lambda s: s.z())
    else:
        slices.sort(key=lambda s: s.instance_number)
    print(f"  loaded {len(slices)} slices")
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


def main():
    t0 = time.time()
    print("=== Craniofacial Skull (skull-only extraction) ===")
    slices = load_series(SERIES_DIR)
    if SLICE_STEP > 1:
        slices = slices[::SLICE_STEP]
    vol, spacing, origin = build_volume(slices, in_plane_step=IN_PLANE_STEP)
    print("  volume shape (z,y,x):", vol.shape, " spacing mm:", spacing)

    raw_solid = vol >= BONE_HU_THRESHOLD
    print(f"  raw bone-threshold voxels: {int(raw_solid.sum()):,}")

    print("  growing the connected component around the volume centroid "
          "(skull), discarding disconnected hardware/artifacts...")
    skull = component_containing_seed(raw_solid, verbose=True)
    print(f"  kept connected skull component: {int(skull.sum()):,} voxels "
          f"({100*skull.sum()/raw_solid.sum():.1f}% of raw threshold voxels)")

    tris = cuberille_mesh(skull, spacing, origin)
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"  triangles: {tris.shape[0]:,}   enclosed volume: {vol_cm3:,.1f} cm^3")

    out_stl = os.path.join(OUT, "Craniofacial_Skull_Only.stl")
    write_binary_stl(out_stl, tris, name=b"Craniofacial Skull Only")
    print(f"  wrote {out_stl}  ({os.path.getsize(out_stl)/1e6:.1f} MB)")

    # preview render
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    projs = [skull.max(axis=0), skull.max(axis=1), skull.max(axis=2)]
    titles = ["Axial (top-down)", "Coronal (front)", "Sagittal (side)"]
    for ax, img, title in zip(axes, projs, titles):
        ax.imshow(img, cmap='bone', origin='lower', aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.suptitle(f"Craniofacial Skull Only — {tris.shape[0]:,} tris, {vol_cm3:,.0f} cm³", fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(OUT, "craniofacial_skull_only_preview.png")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_png}")

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
