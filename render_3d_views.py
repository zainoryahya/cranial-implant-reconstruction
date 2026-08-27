"""
Render shaded 3D views of the skull (showing the defect opening) and of the
skull+implant assemblies (implant in a distinct colour), for the paper's
figures. Pure NumPy/Matplotlib, reusing the same cuberille mesher as the
rest of the pipeline (no new third-party dependencies).

Meshes are rendered at a coarser voxel resolution than the delivered STL
files purely for interactive/render-time reasons (matplotlib's Poly3DCollection
is not GPU-accelerated); this script is for FIGURE GENERATION only and does
not alter any of the delivered STL geometry.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, os.path.dirname(__file__))
from voxelmesh import cuberille_mesh

OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
XY_STEP, Z_STEP = 4, 3  # render-only downsample factors (see module docstring)

SKULL_RGB = (0.87, 0.82, 0.72)
IMPLANT_A_RGB = (0.80, 0.15, 0.15)   # crimson
IMPLANT_B_RGB = (0.10, 0.45, 0.75)   # blue


def block_any_downsample(mask, factors):
    """Downsample a boolean array by OR-pooling over non-overlapping blocks
    (preserves thin structures far better than naive strided slicing)."""
    fz, fy, fx = factors
    nz, ny, nx = mask.shape
    nz2, ny2, nx2 = (nz // fz) * fz, (ny // fy) * fy, (nx // fx) * fx
    m = mask[:nz2, :ny2, :nx2]
    m = m.reshape(nz2 // fz, fz, ny2 // fy, fy, nx2 // fx, fx)
    return m.any(axis=(1, 3, 5))


def mesh_to_xyz_verts(tris):
    return tris[:, :, [2, 1, 0]].astype(np.float64)


def shade(verts, base_rgb, light_dir=(0.4, -0.5, 0.75), ambient=0.38, diffuse=0.7, alpha=1.0):
    v0, v1, v2 = verts[:, 0, :], verts[:, 1, :], verts[:, 2, :]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1
    n = n / norm
    ld = np.array(light_dir, dtype=np.float64)
    ld = ld / np.linalg.norm(ld)
    intensity = np.clip(n @ ld, 0, 1)
    shade_val = np.clip(ambient + diffuse * intensity, 0, 1.15)
    base = np.array(base_rgb)
    colors = np.clip(base[None, :] * shade_val[:, None], 0, 1)
    a = np.full((len(colors), 1), alpha)
    return np.concatenate([colors, a], axis=1)


def setup_axes(ax, all_verts_list):
    allv = np.concatenate(all_verts_list, axis=0).reshape(-1, 3)
    xs, ys, zs = allv[:, 0], allv[:, 1], allv[:, 2]
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(zs.min(), zs.max())
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.set_axis_off()


def load_skull_ds():
    d = np.load(os.path.join(OUT, 'skull_data.npz'))
    skull = d['skull']  # (225,256,256) half-res bool
    spacing = d['spacing']  # dz, dy, dx at half-res
    skull_ds = block_any_downsample(skull, (Z_STEP, XY_STEP, XY_STEP))
    spacing_ds = (spacing[0] * Z_STEP, spacing[1] * XY_STEP, spacing[2] * XY_STEP)
    return skull_ds, spacing_ds, skull.shape


def load_implant_ds(npz_name, skull_full_shape):
    d = np.load(os.path.join(OUT, npz_name))
    implant_full = d['smoothed']  # (n_k, 512, 512) full-res bool
    target_d = np.load(os.path.join(OUT, 'defect_target.npz'))
    k_start = int(target_d['k_start'])
    implant_half = implant_full[:, ::2, ::2]  # match skull's 256-res grid
    n_k = implant_half.shape[0]
    embedded = np.zeros(skull_full_shape, dtype=bool)  # (225,256,256)
    embedded[k_start:k_start + n_k] = implant_half[:min(n_k, skull_full_shape[0] - k_start)]
    implant_ds = block_any_downsample(embedded, (Z_STEP, XY_STEP, XY_STEP))
    return implant_ds


def render_panel(ax, mesh_list, elev, azim):
    """mesh_list: list of (verts_xyz, rgb, alpha)"""
    all_verts = []
    for verts, rgb, alpha in mesh_list:
        colors = shade(verts, rgb, alpha=alpha)
        pc = Poly3DCollection(verts, facecolors=colors, edgecolor='none', linewidths=0)
        pc.set_zsort('average')
        ax.add_collection3d(pc)
        all_verts.append(verts)
    setup_axes(ax, all_verts)
    ax.view_init(elev=elev, azim=azim)


if __name__ == '__main__':
    t0 = time.time()
    skull_ds, spacing_ds, skull_full_shape = load_skull_ds()
    skull_tris = cuberille_mesh(skull_ds, spacing_ds, (0, 0, 0))
    skull_verts = mesh_to_xyz_verts(skull_tris)
    print(f"skull mesh: {skull_tris.shape[0]:,} triangles ({time.time()-t0:.1f}s)")

    implA_ds = load_implant_ds('method_a_smoothed.npz', skull_full_shape)
    implA_tris = cuberille_mesh(implA_ds, spacing_ds, (0, 0, 0))
    implA_verts = mesh_to_xyz_verts(implA_tris)
    print(f"implant A mesh: {implA_tris.shape[0]:,} triangles ({time.time()-t0:.1f}s)")

    implB_ds = load_implant_ds('method_b_smoothed.npz', skull_full_shape)
    implB_tris = cuberille_mesh(implB_ds, spacing_ds, (0, 0, 0))
    implB_verts = mesh_to_xyz_verts(implB_tris)
    print(f"implant B mesh: {implB_tris.shape[0]:,} triangles ({time.time()-t0:.1f}s)")

    np.savez(os.path.join(OUT, 'render_meshes_cache.npz'),
             skull_verts=skull_verts.astype(np.float32),
             implA_verts=implA_verts.astype(np.float32),
             implB_verts=implB_verts.astype(np.float32))
    print(f"cached render meshes ({time.time()-t0:.1f}s total)")
