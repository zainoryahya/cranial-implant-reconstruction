import os, glob, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from mindicom import Slice
from voxelmesh import cuberille_mesh, mesh_signed_volume, write_binary_stl

BASE = os.environ.get("DICOM_DATA_DIR", "./data")
OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")

def load_series(folder):
    files = []
    for root, dirs, fnames in os.walk(folder):
        for fn in fnames:
            if fn.lower().endswith(('.url', '.txt', '.ini', '.db')):
                continue
            files.append(os.path.join(root, fn))
    slices = []
    errors = 0
    for f in files:
        try:
            s = Slice(f)
            if s.rows and s.cols and s.pixels_raw.size == s.rows * s.cols:
                slices.append(s)
        except Exception as e:
            errors += 1
    if not slices:
        raise RuntimeError(f"No readable DICOM slices in {folder}")
    # sort by z position if available, else instance number
    if all(s.image_position for s in slices):
        slices.sort(key=lambda s: s.z())
    else:
        slices.sort(key=lambda s: s.instance_number)
    print(f"  loaded {len(slices)} slices ({errors} skipped/unreadable)")
    return slices


def build_volume(slices, in_plane_step=1, clean_sentinel=True, sentinel_max_frac=0.2):
    rows, cols = slices[0].rows, slices[0].cols
    dy, dx = slices[0].pixel_spacing
    zs = [s.z() for s in slices]
    if len(zs) > 1:
        diffs = np.diff(sorted(zs))
        diffs = diffs[diffs > 1e-6]
        dz = float(np.median(diffs)) if len(diffs) else (slices[0].slice_thickness or 1.0)
    else:
        dz = slices[0].slice_thickness or 1.0

    vol = np.empty((len(slices), rows, cols), dtype=np.float32)
    for i, s in enumerate(slices):
        hu = s.hu
        if clean_sentinel:
            # some scanners pad out-of-FOV pixels with the max representable
            # raw value; strip that so it doesn't get meshed as bone.
            raw = s.pixels_raw
            sentinel = raw == np.iinfo(raw.dtype).max
            # only strip if it's a small minority of pixels (border padding),
            # not if the max value IS the real foreground signal (e.g. a
            # pre-segmented binary mask stored as 0/65535).
            if sentinel.any() and sentinel.mean() < sentinel_max_frac:
                hu = hu.copy()
                hu[sentinel] = -1024.0
        vol[i] = hu

    origin = (min(zs), 0.0, 0.0)  # z origin real, y/x relative (shape only, fine)
    spacing = (dz, dy, dx)

    if in_plane_step > 1:
        vol = vol[:, ::in_plane_step, ::in_plane_step]
        spacing = (dz, dy * in_plane_step, dx * in_plane_step)

    return vol, spacing, origin


def process(name, folder, threshold_fn, in_plane_step=1, slice_step=1, out_name=None,
            clean_sentinel=True):
    print(f"=== {name} ===")
    t0 = time.time()
    slices = load_series(folder)
    print("  modality:", slices[0].modality, "rows/cols:", slices[0].rows, slices[0].cols,
          "pixel_spacing:", slices[0].pixel_spacing)
    if slice_step > 1:
        slices = slices[::slice_step]
    vol, spacing, origin = build_volume(slices, in_plane_step=in_plane_step,
                                         clean_sentinel=clean_sentinel)
    print("  volume shape (z,y,x):", vol.shape, "spacing mm:", spacing)
    print("  value range:", float(vol.min()), float(vol.max()))

    solid = threshold_fn(vol)
    n_solid = int(solid.sum())
    print("  solid voxel count:", n_solid, f"({100*n_solid/solid.size:.1f}% of volume)")
    if n_solid == 0:
        print("  !! nothing above threshold, skipping")
        return None

    tris = cuberille_mesh(solid, spacing, origin)
    vol_check = mesh_signed_volume(tris) / 1000.0  # mm^3 -> cm^3
    print(f"  triangles: {tris.shape[0]:,}   enclosed volume: {vol_check:,.1f} cm^3")

    out_path = os.path.join(OUT, out_name or (name.replace(' ', '_') + '.stl'))
    write_binary_stl(out_path, tris, name=name.encode('ascii', 'ignore'))
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  wrote {out_path}  ({size_mb:.1f} MB)  [{time.time()-t0:.1f}s]")
    return dict(name=name, path=out_path, tris=tris.shape[0], size_mb=size_mb,
                volume_cm3=vol_check, solid=solid, spacing=spacing, origin=origin, vol=vol)


results = {}

# 1) Craniofacial fracture CT -> bone threshold
results['craniofacial'] = process(
    "Craniofacial Fracture Skull",
    os.path.join(BASE, "patient1_dicom_series"),
    threshold_fn=lambda v: v >= 250,
    in_plane_step=3,
    slice_step=2,
)

# 2) Patient 2 CT (large decompressive-craniectomy defect) -> bone threshold
results['data19'] = process(
    "DATA_19_2_1016 Skull",
    os.path.join(BASE, "patient2_dicom_series"),
    threshold_fn=lambda v: v >= 250,
    in_plane_step=3,
    slice_step=2,
)

# 3) missing data -> already a binary bone mask (0 / 65535); do NOT strip
#    65535 as "sentinel padding" here, it IS the foreground signal.
results['missing'] = process(
    "Missing_Data Bone Mask",
    os.path.join(BASE, "unused_dataset"),
    threshold_fn=lambda v: v >= 30000,
    in_plane_step=2,
    clean_sentinel=False,
)

# 4) Brain tumour MRI -> no HU, no real skull signal; use intensity
#    percentile to show overall head/brain shape (caveat: not bone).
def mri_threshold(v):
    nz = v[v > 0]
    thr = np.percentile(nz, 55) if nz.size else 1.0
    return v >= thr

results['mri'] = process(
    "Brain_Tumour_MRI Head Shape",
    os.path.join(BASE, "unused_mri_dataset"),
    threshold_fn=mri_threshold,
    in_plane_step=2,
    slice_step=2,
)

print("\nDONE")
for k, r in results.items():
    if r:
        print(k, '->', r['path'], f"{r['tris']:,} tris", f"{r['size_mb']:.1f}MB")

# --- quick shape preview renders (3 orthogonal max-projections per dataset) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for k, r in results.items():
    if not r:
        continue
    solid = r['solid']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    proj_axial = solid.max(axis=0)      # top-down (y,x)
    proj_coronal = solid.max(axis=1)    # front (z,x)
    proj_sagittal = solid.max(axis=2)   # side (z,y)
    for ax, img, title in zip(
        axes,
        [proj_axial, proj_coronal, proj_sagittal],
        ["Axial (top-down)", "Coronal (front)", "Sagittal (side)"]
    ):
        ax.imshow(img, cmap='bone', origin='lower', aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.suptitle(f"{r['name']}  —  {r['tris']:,} tris, {r['volume_cm3']:,.0f} cm³", fontsize=11)
    fig.tight_layout()
    png_path = os.path.join(OUT, k + "_preview.png")
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print("preview:", png_path)
