"""
Cuberille (voxel-face) surface mesh extractor + binary STL writer.
No skimage/vtk/trimesh available (no network access to install), so this
implements a from-scratch, numerically-verified voxel surface mesher.

Volume axis convention: solid[z, y, x]  (0=slice axis, 1=row/y, 2=col/x)
spacing = (dz, dy, dx) in mm, origin = (oz, oy, ox) in mm (world coords of
voxel index (0,0,0) corner).
"""
import struct
import numpy as np

# For each of the 6 axis-aligned face directions, 4 corner offsets (in
# z,y,x unit-cube-local order) chosen so that the two triangles
# (c0,c1,c2) and (c0,c2,c3) wind counter-clockwise when viewed from
# OUTSIDE the cube (i.e. normal points away from the voxel interior).
# Verified numerically below via the divergence-theorem volume check.
FACES = {
    ('x', +1): [(0,0,1), (0,1,1), (1,1,1), (1,0,1)],
    ('x', -1): [(0,0,0), (1,0,0), (1,1,0), (0,1,0)],
    ('y', +1): [(0,1,0), (1,1,0), (1,1,1), (0,1,1)],
    ('y', -1): [(0,0,0), (0,0,1), (1,0,1), (1,0,0)],
    ('z', +1): [(1,0,0), (1,0,1), (1,1,1), (1,1,0)],
    ('z', -1): [(0,0,0), (0,1,0), (0,1,1), (0,0,1)],
}
AXIS_IDX = {'z': 0, 'y': 1, 'x': 2}


def _face_mask(solid, axis, sign):
    axis_i = AXIS_IDX[axis]
    shifted = np.zeros_like(solid)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if sign == +1:
        dst[axis_i] = slice(0, -1)
        src[axis_i] = slice(1, None)
    else:
        dst[axis_i] = slice(1, None)
        src[axis_i] = slice(0, -1)
    shifted[tuple(dst)] = solid[tuple(src)]
    return solid & (~shifted)


def _dilate6(mask):
    """3D 6-connectivity dilation (no wraparound), pure numpy."""
    out = mask.copy()
    for axis in range(3):
        dst = [slice(None)] * 3
        src = [slice(None)] * 3
        dst[axis] = slice(1, None)
        src[axis] = slice(0, -1)
        shifted_pos = np.zeros_like(mask)
        shifted_pos[tuple(dst)] = mask[tuple(src)]

        dst2 = [slice(None)] * 3
        src2 = [slice(None)] * 3
        dst2[axis] = slice(0, -1)
        src2[axis] = slice(1, None)
        shifted_neg = np.zeros_like(mask)
        shifted_neg[tuple(dst2)] = mask[tuple(src2)]

        out |= shifted_pos | shifted_neg
    return out


def connected_components(solid, min_voxels=1, max_components=None, verbose=False):
    """Label 6-connected components of a 3D boolean array via iterative
    flood-fill (region growing by repeated dilation). Pure numpy, no scipy.
    Returns a list of boolean masks, sorted largest-first, each with at
    least `min_voxels` voxels."""
    unlabeled = solid.copy()
    components = []
    guard = 0
    while unlabeled.any():
        guard += 1
        if max_components is not None and guard > max_components:
            break
        seed = tuple(np.argwhere(unlabeled)[0])
        comp = np.zeros_like(solid)
        comp[seed] = True
        while True:
            grown = _dilate6(comp) & unlabeled
            grown |= comp
            if grown.sum() == comp.sum():
                break
            comp = grown
        n = int(comp.sum())
        if n >= min_voxels:
            components.append(comp)
        unlabeled &= ~comp
        if verbose:
            print(f"    component #{guard}: {n:,} voxels")
    components.sort(key=lambda c: c.sum(), reverse=True)
    return components


def largest_component(solid, verbose=False):
    comps = connected_components(solid, verbose=verbose)
    if not comps:
        return solid
    return comps[0]


def component_containing_seed(solid, seed=None, max_iter=1000, verbose=False):
    """Grow (via 6-connected region growing / repeated dilation) just the
    ONE component that contains `seed` (a (z,y,x) voxel index). If seed is
    None, uses the solid voxel closest to the mask's centroid - a good
    default when the target structure (e.g. the skull) is the big blob
    roughly in the middle of the volume, and there may be many small
    disconnected specks elsewhere that we don't want to waste time
    labeling individually.
    """
    if seed is None:
        idx = np.argwhere(solid)
        centroid = idx.mean(axis=0)
        d2 = ((idx - centroid) ** 2).sum(axis=1)
        seed = tuple(idx[np.argmin(d2)])
        if verbose:
            print(f"    seed (nearest solid voxel to centroid): {seed}")
    comp = np.zeros_like(solid)
    comp[seed] = True
    it = 0
    while True:
        it += 1
        grown = _dilate6(comp) & solid
        grown |= comp
        n_new = int(grown.sum())
        if verbose and it % 20 == 0:
            print(f"    growing... iter {it}, {n_new:,} voxels")
        if n_new == int(comp.sum()):
            break
        comp = grown
        if it >= max_iter:
            if verbose:
                print(f"    !! hit max_iter={max_iter}, stopping (component may be incomplete)")
            break
    if verbose:
        print(f"    converged after {it} iterations, {int(comp.sum()):,} voxels")
    return comp


def cuberille_mesh(solid, spacing, origin=(0.0, 0.0, 0.0)):
    """solid: 3D bool array [z,y,x]. spacing: (dz,dy,dx). Returns (V, F)-free
    triangle soup as an (N,3,3) float32 array of triangle vertices (world mm)."""
    dz, dy, dx = spacing
    oz, oy, ox = origin
    scale = np.array([dz, dy, dx], dtype=np.float64)
    orig = np.array([oz, oy, ox], dtype=np.float64)

    tri_chunks = []
    for (axis, sign), corners in FACES.items():
        mask = _face_mask(solid, axis, sign)
        idx = np.argwhere(mask)  # (N,3) in z,y,x voxel-index space
        if idx.shape[0] == 0:
            continue
        corners_arr = np.array(corners, dtype=np.float64)  # (4,3)
        quad = idx[:, None, :].astype(np.float64) + corners_arr[None, :, :]  # (N,4,3)
        quad_world = quad * scale[None, None, :] + orig[None, None, :]
        t1 = quad_world[:, [0, 2, 1], :]
        t2 = quad_world[:, [0, 3, 2], :]
        tri_chunks.append(t1)
        tri_chunks.append(t2)

    if not tri_chunks:
        return np.zeros((0, 3, 3), dtype=np.float32)
    tris = np.concatenate(tri_chunks, axis=0).astype(np.float32)
    return tris


def mesh_signed_volume(tris):
    """Divergence-theorem signed volume check: should be positive and equal
    to the true enclosed volume for a correctly-wound closed mesh."""
    v0 = tris[:, 0, :].astype(np.float64)
    v1 = tris[:, 1, :].astype(np.float64)
    v2 = tris[:, 2, :].astype(np.float64)
    cross = np.cross(v1, v2)
    vol = np.einsum('ij,ij->i', v0, cross).sum() / 6.0
    return vol


def write_binary_stl(path, tris, name=b"mesh"):
    n = tris.shape[0]
    with open(path, 'wb') as f:
        header = (name + b' ' * 80)[:80]
        f.write(header)
        f.write(struct.pack('<I', n))
        if n == 0:
            return
        v0 = tris[:, 0, :]
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]
        # STL wants (x,y,z) vertex order; our arrays are (z,y,x) -> reorder
        def to_xyz(v):
            return np.stack([v[:, 2], v[:, 1], v[:, 0]], axis=1)
        v0x, v1x, v2x = to_xyz(v0), to_xyz(v1), to_xyz(v2)
        e1 = v1x - v0x
        e2 = v2x - v0x
        normals = np.cross(e1, e2)
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_len[norm_len == 0] = 1.0
        normals = (normals / norm_len).astype(np.float32)

        rec_dtype = np.dtype([
            ('normal', '<f4', 3),
            ('v0', '<f4', 3),
            ('v1', '<f4', 3),
            ('v2', '<f4', 3),
            ('attr', '<u2'),
        ])
        rec = np.zeros(n, dtype=rec_dtype)
        rec['normal'] = normals
        rec['v0'] = v0x
        rec['v1'] = v1x
        rec['v2'] = v2x
        rec['attr'] = 0
        f.write(rec.tobytes())


if __name__ == '__main__':
    # Self-test: single voxel cube, spacing (1,1,1) -> expected volume 1.0
    solid = np.zeros((3, 3, 3), dtype=bool)
    solid[1, 1, 1] = True
    tris = cuberille_mesh(solid, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    vol = mesh_signed_volume(tris)
    print("single voxel: n_tris =", tris.shape[0], "signed volume =", vol)
    assert tris.shape[0] == 12, "expected 12 triangles (6 faces x 2)"
    assert abs(vol - 1.0) < 1e-6, f"expected volume 1.0, got {vol}"

    # Bigger block test: 4x3x2 solid block, non-unit spacing
    solid2 = np.zeros((6, 6, 6), dtype=bool)
    solid2[1:5, 1:4, 1:3] = True  # dz=4, dy=3, dx=2
    tris2 = cuberille_mesh(solid2, spacing=(2.0, 1.5, 0.5), origin=(10, -5, 3))
    vol2 = mesh_signed_volume(tris2)
    expected = 4*2.0 * 3*1.5 * 2*0.5
    print("block: n_tris =", tris2.shape[0], "signed volume =", vol2, "expected", expected)
    assert abs(vol2 - expected) < 1e-3

    # Connected-components test: two separate blobs, different sizes
    solid3 = np.zeros((10, 10, 10), dtype=bool)
    solid3[1:3, 1:3, 1:3] = True      # small blob, 8 voxels
    solid3[5:9, 5:9, 5:9] = True      # big blob, 64 voxels
    comps = connected_components(solid3)
    print("n components:", len(comps), "sizes:", [int(c.sum()) for c in comps])
    assert len(comps) == 2
    assert int(comps[0].sum()) == 64 and int(comps[1].sum()) == 8
    biggest = largest_component(solid3)
    assert int(biggest.sum()) == 64
    print("connected-components self-test PASSED")

    print("ALL SELF-TESTS PASSED")
