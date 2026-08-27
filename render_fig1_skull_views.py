import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from render_3d_views import render_panel, SKULL_RGB

OUT = os.environ.get("PIPELINE_OUT_DIR", "./output")
cache = np.load(os.path.join(OUT, 'render_meshes_cache.npz'))
skull_verts = cache['skull_verts'].astype(np.float64)

DEFECT_AZIM = -17.0  # mean gap-bisector angle computed from defect_gaps_fullres.npz

views = [
    ("A: defect-facing", 8, DEFECT_AZIM),
    ("B: opposite side", 8, DEFECT_AZIM + 180),
    ("C: superior (top-down)", 80, DEFECT_AZIM),
    ("D: rotated 90 deg", 8, DEFECT_AZIM + 90),
]

t0 = time.time()
fig = plt.figure(figsize=(11, 11))
for i, (title, elev, azim) in enumerate(views):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    render_panel(ax, [(skull_verts, SKULL_RGB, 1.0)], elev, azim)
    ax.set_title(title, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_skull_3d_views_TEST.png'), dpi=130, facecolor='white')
print(f"saved test figure ({time.time()-t0:.1f}s)")
