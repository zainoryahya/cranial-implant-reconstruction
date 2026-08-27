"""Union the (full-res) implant mask with the (256-res, in_plane_step=2)
cached skull mask into ONE watertight mesh, so the fit can be inspected in a
single STL. The implant is downsampled by the same factor (2) and placed at
its correct z-slice range before the union."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from voxelmesh import cuberille_mesh, mesh_signed_volume, write_binary_stl, _dilate6

OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")


def combine(implant_npz_key_file, out_name, label):
    skull_d = np.load(os.path.join(OUT, "skull_data.npz"))
    skull = skull_d['skull']            # (225,256,256) bool
    spacing = skull_d['spacing']        # dz, dy(*2), dx(*2)
    origin = skull_d['origin']          # (z0_full, 0, 0)

    target_d = np.load(os.path.join(OUT, "defect_target.npz"))
    k_start, k_end = int(target_d['k_start']), int(target_d['k_end'])

    imp_d = np.load(implant_npz_key_file)
    implant_full = imp_d['smoothed']    # (n_k, 512, 512) bool, full-res

    # downsample implant by 2 in-plane to match skull_data resolution
    implant_ds = implant_full[:, ::2, ::2]

    # The implant mask (mirrored, full-res) and the skull mask (independent
    # bone threshold, half-res) don't come from the same computation, so
    # their edges can miss each other by ~1 voxel after downsampling,
    # leaving a hairline crack at the seam. Dilate the implant by 1 voxel
    # before the union so it fuses cleanly into the surrounding bone
    # instead of just touching it.
    implant_ds = _dilate6(implant_ds)

    combined = skull.copy()
    n_k = implant_ds.shape[0]
    combined[k_start:k_start+n_k] |= implant_ds[:combined[k_start:k_start+n_k].shape[0]]

    tris = cuberille_mesh(combined, tuple(spacing), tuple(origin))
    vol_cm3 = mesh_signed_volume(tris) / 1000.0
    print(f"{label}: combined mesh {tris.shape[0]:,} triangles, {vol_cm3:.1f} cm^3")
    out_path = os.path.join(OUT, out_name)
    write_binary_stl(out_path, tris, name=label.encode('ascii', 'ignore'))
    print("wrote", out_path, f"{os.path.getsize(out_path)/1e6:.1f} MB")
    return combined, spacing, origin


if __name__ == '__main__':
    combine(os.path.join(OUT, "method_a_smoothed.npz"),
            "Skull_Plus_Implant_MethodA.stl", "Skull Plus Implant Method A")
    combine(os.path.join(OUT, "method_b_smoothed.npz"),
            "Skull_Plus_Implant_MethodB.stl", "Skull Plus Implant Method B")
