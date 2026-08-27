"""Quantitative manifold/watertightness check on the exported STL meshes.

For a closed, manifold, consistently-oriented triangulated surface, every
undirected edge must be shared by EXACTLY 2 triangles, and (for consistent
winding) the two half-edges along that shared edge must run in opposite
directions. This is a stronger, independent check than the divergence-
theorem volume sign test already used in the pipeline: the divergence
theorem can return a sensible positive volume even in some degenerate
configurations, whereas the edge-manifold check directly certifies the
topological closure and orientation consistency needed for slicing /
3D printing.
"""
import struct
import numpy as np
from collections import defaultdict

def read_binary_stl(path):
    with open(path, 'rb') as f:
        header = f.read(80)
        n_tri = struct.unpack('<I', f.read(4))[0]
        tris = np.empty((n_tri, 3, 3), dtype=np.float64)
        for i in range(n_tri):
            f.read(12)  # normal
            v = struct.unpack('<9f', f.read(36))
            f.read(2)   # attribute byte count
            tris[i] = np.array(v).reshape(3, 3)
    return tris

def manifold_check(tris, decimals=4):
    # quantize vertex coordinates to merge numerically-identical vertices
    verts = tris.reshape(-1, 3)
    q = np.round(verts, decimals)
    vmap = {}
    idx = np.empty(len(q), dtype=np.int64)
    for i, p in enumerate(map(tuple, q)):
        j = vmap.get(p)
        if j is None:
            j = len(vmap)
            vmap[p] = j
        idx[i] = j
    tri_idx = idx.reshape(-1, 3)

    edge_count = defaultdict(int)
    edge_dir_ok = defaultdict(lambda: [0, 0])  # count of each direction
    for a, b, c in tri_idx:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_count[key] += 1
            if u < v:
                edge_dir_ok[key][0] += 1
            else:
                edge_dir_ok[key][1] += 1

    n_edges = len(edge_count)
    non2 = sum(1 for c in edge_count.values() if c != 2)
    inconsistent_winding = sum(1 for k, (f, r) in edge_dir_ok.items()
                                if edge_count[k] == 2 and (f != 1 or r != 1))
    n_verts = len(vmap)
    n_tris = len(tri_idx)
    euler = n_verts - n_edges + n_tris  # should be 2 for a closed genus-0 surface (sphere-like)
    return dict(n_triangles=n_tris, n_vertices=n_verts, n_edges=n_edges,
                non_2_manifold_edges=non2, inconsistent_winding_edges=inconsistent_winding,
                euler_characteristic=euler)

files = {
    "Whole skull": "Craniofacial_Skull_Only.stl",
    "Implant Method A (BOA)": "Implant_MethodA_BOA_Curve.stl",
    "Implant Method B (Delaunay)": "Implant_MethodB_Delaunay.stl",
    "Fused skull+implant A": "Skull_Plus_Implant_MethodA.stl",
    "Fused skull+implant B": "Skull_Plus_Implant_MethodB.stl",
}

print(f"{'Mesh':32s} {'tris':>10s} {'verts':>10s} {'edges':>10s} {'non2':>6s} {'badwind':>8s} {'euler':>7s}")
results = {}
for label, fn in files.items():
    tris = read_binary_stl(fn)
    r = manifold_check(tris)
    results[label] = r
    print(f"{label:32s} {r['n_triangles']:10d} {r['n_vertices']:10d} {r['n_edges']:10d} "
          f"{r['non_2_manifold_edges']:6d} {r['inconsistent_winding_edges']:8d} {r['euler_characteristic']:7d}")

import json
with open('manifold_check_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nsaved manifold_check_results.json")
