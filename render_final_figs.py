import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, os.path.dirname(__file__))
from voxelmesh import cuberille_mesh

OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
DEFECT_AZIM = -17.0
SKULL_RGB = (0.87, 0.82, 0.72)
IMPLANT_A_RGB = (0.80, 0.15, 0.15)
IMPLANT_B_RGB = (0.10, 0.45, 0.75)

# ---- context (whole-skull, coarse) mesh ----
_d = np.load(os.path.join(OUT, 'skull_data.npz'))
_skull = _d['skull']; _spacing = _d['spacing']


def block_any_downsample(mask, factors):
    fz, fy, fx = factors
    nz, ny, nx = mask.shape
    nz2, ny2, nx2 = (nz // fz) * fz, (ny // fy) * fy, (nx // fx) * fx
    m = mask[:nz2, :ny2, :nx2]
    m = m.reshape(nz2 // fz, fz, ny2 // fy, fy, nx2 // fx, fx)
    return m.any(axis=(1, 3, 5))


def to_xyz(tris):
    return tris[:, :, [2, 1, 0]].astype(np.float64)


def shade(verts, base_rgb, light_dir=(0.4, -0.5, 0.75), ambient=0.38, diffuse=0.7):
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
    return np.concatenate([colors, np.ones((len(colors), 1))], axis=1)


def draw_panel(ax, mesh_rgb_list, elev, azim, title=None):
    all_v = []
    for verts, rgb in mesh_rgb_list:
        colors = shade(verts, rgb)
        pc = Poly3DCollection(verts, facecolors=colors, edgecolor='none', linewidths=0)
        pc.set_zsort('average')
        ax.add_collection3d(pc)
        all_v.append(verts)
    allv = np.concatenate(all_v, axis=0).reshape(-1, 3)
    xs, ys, zs = allv[:, 0], allv[:, 1], allv[:, 2]
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(zs.min(), zs.max())
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=11)


def context_skull_verts():
    skull_ds = block_any_downsample(_skull, (3, 4, 4))
    sp = (_spacing[0] * 3, _spacing[1] * 4, _spacing[2] * 4)
    tris = cuberille_mesh(skull_ds, sp, (0, 0, 0))
    return to_xyz(tris)


def context_implant_verts(npz_name):
    d = np.load(os.path.join(OUT, npz_name))
    implant_full = d['smoothed']
    target_d = np.load(os.path.join(OUT, 'defect_target.npz'))
    k_start = int(target_d['k_start'])
    implant_half = implant_full[:, ::2, ::2]
    n_k = implant_half.shape[0]
    embedded = np.zeros(_skull.shape, dtype=bool)
    embedded[k_start:k_start + n_k] = implant_half[:min(n_k, _skull.shape[0] - k_start)]
    implant_ds = block_any_downsample(embedded, (3, 4, 4))
    sp = (_spacing[0] * 3, _spacing[1] * 4, _spacing[2] * 4)
    tris = cuberille_mesh(implant_ds, sp, (0, 0, 0))
    return to_xyz(tris)


def crop_skull_verts(zr, yr, xr):
    crop = _skull[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]]
    tris = cuberille_mesh(crop, tuple(_spacing), (0, 0, 0))
    return to_xyz(tris)


def crop_implant_verts(npz_name, zr, yr, xr):
    d = np.load(os.path.join(OUT, npz_name))
    implant_full = d['smoothed']
    target_d = np.load(os.path.join(OUT, 'defect_target.npz'))
    k_start = int(target_d['k_start'])
    implant_half = implant_full[:, ::2, ::2]
    n_k = implant_half.shape[0]
    embedded = np.zeros(_skull.shape, dtype=bool)
    embedded[k_start:k_start + n_k] = implant_half[:min(n_k, _skull.shape[0] - k_start)]
    crop = embedded[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]]
    tris = cuberille_mesh(crop, tuple(_spacing), (0, 0, 0))
    return to_xyz(tris)


CROP_Z, CROP_Y, CROP_X = (85, 165), (0, 115), (95, 256)
